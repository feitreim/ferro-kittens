/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A kittens port of cuda-oxide's canonical Blackwell `gemm_sol_final`.
//!
//! Both entries are expressed through ferro-kittens' typed tiles and pipeline
//! primitives. On B200 the M256xN256 entry serves 4K; M512xN256 serves 8K/16K. They
//! share CLC work stealing, a two-CTA cluster, four TMA/shared stages, two TMEM
//! accumulator halves, an unrolled K loop, and L2-aware output ordering.
//!
//! Data layout is the upstream contract: row-major FP16 A `[M, K]`, row-major
//! FP16 B `[N, K]` (therefore `A·Bᵀ`), and packed row-major BF16 C `[M, N]`.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::cluster;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{
    DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, thread, warp,
};

use std::error::Error;

use kittens::global::{GlobalLayout, GlobalRows, store_shared_rows};
use kittens::ldst::store_tile_x4;
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcCursor, ClcQueue};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, Element, F16, SharedCell, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};

/// The whole kernel. Both shipped entries pass this and nothing else does.
pub const WHOLE: u8 = 0;
/// TMA and every barrier, with the MMA and the drain removed: what the memory
/// pipeline alone can sustain.
pub const FEED_ONLY: u8 = 1;
/// The MMA and every barrier, with the loads removed — the producer arrives on
/// `load` instead of filling it, so the ring recycles on tensor-core issue and
/// no global traffic occurs at all.
pub const ISSUE_ONLY: u8 = 2;
/// The whole kernel except the drain: the accumulator fills and is released
/// without being read, so the difference against [`WHOLE`] is the epilogue's
/// cost *including* whatever of it fails to overlap.
pub const NO_DRAIN: u8 = 3;

const fn loads(ablate: u8) -> bool {
    ablate != ISSUE_ONLY
}
const fn multiplies(ablate: u8) -> bool {
    ablate != FEED_ONLY
}
const fn drains(ablate: u8) -> bool {
    ablate == WHOLE
}

const BLOCK_M: usize = 128;
const BLOCK_N: usize = 256;
/// Columns of `C` one cluster owns — the same for both variants, and `pub`
/// because the ablation arms size their own grids from it.
pub const N_TILE: usize = BLOCK_N;
/// Depth one stage carries, so `k / K_TILE` is the K loop's trip count.
pub const K_TILE: usize = BLOCK_K;
const HALF_N: usize = BLOCK_N / 2;
const BLOCK_K: usize = 64;
const CHUNKS: usize = BLOCK_K / 16;
const STAGES: usize = 4;
const ACCUM_COLUMNS: u32 = 512;
const EPILOGUE_WARPS: u32 = (BLOCK_M / 32) as u32;
const TMA_WARP: u32 = EPILOGUE_WARPS;
const MMA_WARP: u32 = TMA_WARP + 1;
pub const THREADS: u32 = (MMA_WARP + 1) * 32;
const RANKS: u32 = 2;
const LEADER: u32 = 0;
const PAIR: u16 = 0b11;
const BAND_N: usize = 64;
const SMALL_STAGE_N: usize = BLOCK_N;
const LARGE_STAGE_N: usize = HALF_N;

const _: () = {
    assert!(THREADS == 192);
    assert!(CHUNKS == 4);
    assert!(2 * BLOCK_N as u32 == ACCUM_COLUMNS);
};

/// The `A` tile a tensor map is built for — `pub` because `experiments/`'
/// ablation arms build the same two maps this file's own runner does.
pub type ATile = SharedTile<F16, BLOCK_M, BLOCK_K, Swizzle128B>;

/// The `B` panel a tensor map is built for, per [`ATile`].
pub type BPanel = SharedTile<F16, 64, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<F16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<F16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;
type StageBand = RegTile<32, BAND_N, BaseLdtm>;

#[repr(C)]
#[derive(Clone, Copy)]
struct TileInfo {
    row: u32,
    column: u32,
    has_work: u32,
}

const fn align_up(offset: usize, alignment: usize) -> usize {
    offset.next_multiple_of(alignment)
}

const A0_OFFSET: usize = 0;
const SMALL_B_OFFSET: usize = A0_OFFSET + ARing::BYTES;
const SMALL_RINGS_END: usize = SMALL_B_OFFSET + BRing::BYTES;
const LARGE_A1_OFFSET: usize = A0_OFFSET + ARing::BYTES;
const LARGE_B_OFFSET: usize = LARGE_A1_OFFSET + ARing::BYTES;
const LARGE_RINGS_END: usize = LARGE_B_OFFSET + BRing::BYTES;

const fn load_offset(rings_end: usize) -> usize {
    align_up(rings_end, 8)
}
const fn free_offset(rings_end: usize) -> usize {
    load_offset(rings_end) + STAGES * 8
}
const fn full_offset(rings_end: usize) -> usize {
    free_offset(rings_end) + STAGES * 8
}
const fn empty_offset(rings_end: usize) -> usize {
    full_offset(rings_end) + 2 * 8
}
const fn ready_offset(rings_end: usize) -> usize {
    empty_offset(rings_end) + 2 * 8
}
const fn tmem_offset(rings_end: usize) -> usize {
    align_up(ready_offset(rings_end) + 8, 4)
}
const fn info_offset(rings_end: usize) -> usize {
    align_up(
        tmem_offset(rings_end) + 4,
        SharedCell::<TileInfo>::ALIGNMENT,
    )
}
const fn queue_offset(rings_end: usize) -> usize {
    align_up(
        info_offset(rings_end) + SharedCell::<TileInfo>::BYTES,
        ClcQueue::ALIGNMENT,
    )
}
const fn stage_offset(rings_end: usize) -> usize {
    align_up(queue_offset(rings_end) + ClcQueue::BYTES, 128)
}
const fn shared_plan(rings_end: usize, stage_n: usize) -> usize {
    stage_offset(rings_end) + EPILOGUE_WARPS as usize * 32 * stage_n * Bf16::BYTES
}

pub const SMALL_SHARED_BYTES: usize = shared_plan(SMALL_RINGS_END, SMALL_STAGE_N);
pub const LARGE_SHARED_BYTES: usize = shared_plan(LARGE_RINGS_END, LARGE_STAGE_N);
const _: () = {
    assert!(SMALL_SHARED_BYTES == 196_864);
    assert!(LARGE_SHARED_BYTES == 229_632);
    assert!(LARGE_SHARED_BYTES <= 233_472);
};

#[derive(Clone, Copy)]
struct Common {
    b: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    full: SemaphoreRing<2>,
    empty: SemaphoreRing<2>,
    ready: Semaphore,
    info: SharedCell<TileInfo>,
    queue: ClcQueue,
    stage_base: *mut u8,
    c: GlobalRows<Bf16>,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl Common {
    #[inline(always)]
    unsafe fn attach(
        smem: *mut u8,
        rings_end: usize,
        b_offset: usize,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> Self {
        unsafe {
            Self {
                b: BRing::attach(smem.add(b_offset)),
                load: SemaphoreRing::attach(smem.add(load_offset(rings_end)).cast::<Barrier>()),
                free: SemaphoreRing::attach(smem.add(free_offset(rings_end)).cast::<Barrier>()),
                full: SemaphoreRing::attach(smem.add(full_offset(rings_end)).cast::<Barrier>()),
                empty: SemaphoreRing::attach(smem.add(empty_offset(rings_end)).cast::<Barrier>()),
                ready: Semaphore::attach(smem.add(ready_offset(rings_end)).cast::<Barrier>()),
                info: SharedCell::attach(smem.add(info_offset(rings_end))),
                queue: ClcQueue::attach(smem.add(queue_offset(rings_end))),
                stage_base: smem.add(stage_offset(rings_end)),
                c: GlobalRows::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
            }
        }
    }

    #[inline(always)]
    unsafe fn initialize(self, free_arrivals: u32) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                self.load.init_all(1);
                self.free.init_all(free_arrivals);
                self.full.init_all(1);
                self.empty.init_all(RANKS * EPILOGUE_WARPS * 32);
                self.ready.init(1);
                self.queue.cursor().arm();
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            cluster::cluster_sync();
        }
    }

    #[inline(always)]
    unsafe fn retire(self) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            if thread::threadIdx_x() == 0 {
                self.queue.cursor().disarm();
                self.load.inval_all();
                self.free.inval_all();
                self.full.inval_all();
                self.empty.inval_all();
                self.ready.inval();
            }
        }
    }

    #[inline(always)]
    fn accumulator(base: Accumulator, index: u32) -> Accumulator {
        if index.is_multiple_of(2) {
            base
        } else {
            base.columns_right(BLOCK_N as u32)
        }
    }

    #[inline(always)]
    fn swizzle_group(self) -> u32 {
        if self.tiles_m <= 16 { 2 } else { 8 }
    }

    #[inline(always)]
    unsafe fn publish(self, row: u32, column: u32, has_work: bool) {
        unsafe {
            self.info.write(TileInfo {
                row,
                column,
                has_work: has_work as u32,
            });
            self.ready.arrive();
        }
    }

    #[inline(always)]
    unsafe fn next_info(self, sequence: u32) -> TileInfo {
        unsafe {
            self.ready.wait(sequence & 1);
            self.info.read()
        }
    }

    #[inline(always)]
    unsafe fn release_accumulator(self, sequence: u32) {
        unsafe {
            let empty = self.empty.sem(sequence);
            if self.rank == LEADER {
                empty.arrive();
            } else {
                empty.at_rank(LEADER).arrive();
            }
        }
    }

    #[inline(always)]
    unsafe fn drain<const STAGE_N: usize>(
        self,
        accumulator: Accumulator,
        tile_row: u32,
        tile_column: u32,
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(BLOCK_N.is_multiple_of(STAGE_N));
            };
            let stage =
                SharedTile::<Bf16, 32, STAGE_N, Swizzle128B>::from_raw(self.stage_base.add(
                    self.warp_id as usize * SharedTile::<Bf16, 32, STAGE_N, Swizzle128B>::BYTES,
                ));
            let row = tile_row + self.rank * BLOCK_M as u32 + self.warp_id * 32;
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let band: StageBand =
                        accumulator.tile_x8(32 * self.warp_id, column + band_column);
                    store_tile_x4(stage.chunk_writer(), 0, band_column, self.lane, band);
                    band_column += BAND_N as u32;
                }
                warp::sync_mask(u32::MAX);
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    row,
                    tile_column + column,
                    self.lane,
                    stage,
                );
                warp::sync_mask(u32::MAX);
                column += STAGE_N as u32;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Small {
    common: Common,
    a: ARing,
    accumulator: Accumulator,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl Small {
    #[inline(always)]
    unsafe fn producer<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut cursor: ClcCursor = common.queue.cursor();
            let mut raw_item = cluster::cluster_idx();
            let mut sequence = 0u32;
            let wide_tiles_n = common.tiles_n / 2;
            let valid_items = common.tiles_m * wide_tiles_n;

            loop {
                if raw_item < valid_items {
                    let (tile_n, tile_m) = pipeline::grouped(
                        raw_item,
                        wide_tiles_n,
                        common.tiles_m,
                        common.swizzle_group(),
                    );
                    common.publish(tile_m, tile_n, true);
                    let a_row =
                        (tile_m * (2 * BLOCK_M) as u32 + common.rank * BLOCK_M as u32) as i32;
                    let b_row = (tile_n * BLOCK_N as u32 + common.rank * HALF_N as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        common.free.wait_recycled(global_k);
                        if loads(ABLATE) {
                            let load = common.load.sem(global_k).at_rank(LEADER);
                            let k_base = (k * BLOCK_K as u32) as i32;
                            let b = common.b.tile(global_k);
                            let bytes = self
                                .a
                                .tile(global_k)
                                .tma_load_2d_arriving_at(self.a_map, k_base, a_row, load)
                                + b.tma_load_2d_at_arriving_at::<64>(
                                    self.b_map, 0, k_base, b_row, load,
                                )
                                + b.tma_load_2d_at_arriving_at::<64>(
                                    self.b_map,
                                    64,
                                    k_base,
                                    b_row + 64,
                                    load,
                                );
                            if common.rank == LEADER {
                                common
                                    .load
                                    .sem(global_k)
                                    .expect_tx(bytes.across_ranks(RANKS));
                            }
                        } else if common.rank == LEADER {
                            common.load.sem(global_k).arrive();
                        }
                        k += 1;
                    }
                    sequence += 1;
                }

                let Some(next) = cursor.next() else {
                    common.publish(0, 0, false);
                    break;
                };
                raw_item = next;
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply_stage<const ABLATE: u8>(
        self,
        accumulator: Accumulator,
        sequence: u32,
        k: u32,
    ) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                common.load.wait(global_k);
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        accumulator.raw(),
                        self.a.tile(global_k).k_walk(),
                        common.b.tile(global_k).k_walk(),
                        MmaShape::M256_N256,
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                if common.next_info(sequence).has_work == 0 {
                    break;
                }
                let accumulator = Common::accumulator(self.accumulator, sequence);
                if common.rank == LEADER && sequence >= 2 {
                    common.empty.wait(sequence - 2);
                }

                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 1);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 2);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 3);
                    k += 4;
                }
                if common.rank == LEADER {
                    commit_multicast_cg2(common.full.sem(sequence), PAIR);
                }
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                let info = common.next_info(sequence);
                if info.has_work == 0 {
                    break;
                }
                common.full.wait(sequence);
                if drains(ABLATE) {
                    common.drain::<SMALL_STAGE_N>(
                        Common::accumulator(self.accumulator, sequence),
                        info.row * (2 * BLOCK_M) as u32,
                        info.column * BLOCK_N as u32,
                    );
                }
                common.release_accumulator(sequence);
                sequence += 1;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Large {
    common: Common,
    a0: ARing,
    a1: ARing,
    accumulator: Accumulator,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl Large {
    #[inline(always)]
    unsafe fn producer<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut cursor = common.queue.cursor();
            let mut raw_item = cluster::cluster_idx();
            let mut sequence = 0u32;
            let macro_tiles_m = common.tiles_m / 2;
            let wide_tiles_n = common.tiles_n / 2;
            let valid_items = macro_tiles_m * wide_tiles_n;

            loop {
                if raw_item < valid_items {
                    let (tile_n, macro_m) = pipeline::grouped(
                        raw_item,
                        wide_tiles_n,
                        macro_tiles_m,
                        common.swizzle_group(),
                    );
                    common.publish(macro_m, tile_n, true);
                    let a_row0 = (macro_m * 512 + common.rank * BLOCK_M as u32) as i32;
                    let a_row1 = a_row0 + 256;
                    let b_row = (tile_n * BLOCK_N as u32 + common.rank * HALF_N as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        common.free.wait_recycled(global_k);
                        if loads(ABLATE) {
                            let load = common.load.sem(global_k).at_rank(LEADER);
                            let k_base = (k * BLOCK_K as u32) as i32;
                            let b = common.b.tile(global_k);
                            let bytes = self
                                .a0
                                .tile(global_k)
                                .tma_load_2d_arriving_at(self.a_map, k_base, a_row0, load)
                                + self
                                    .a1
                                    .tile(global_k)
                                    .tma_load_2d_arriving_at(self.a_map, k_base, a_row1, load)
                                + b.tma_load_2d_at_arriving_at::<64>(
                                    self.b_map, 0, k_base, b_row, load,
                                )
                                + b.tma_load_2d_at_arriving_at::<64>(
                                    self.b_map,
                                    64,
                                    k_base,
                                    b_row + 64,
                                    load,
                                );
                            if common.rank == LEADER {
                                common
                                    .load
                                    .sem(global_k)
                                    .expect_tx(bytes.across_ranks(RANKS));
                            }
                        } else if common.rank == LEADER {
                            common.load.sem(global_k).arrive();
                        }
                        k += 1;
                    }
                    sequence += 1;
                }

                let Some(next) = cursor.next() else {
                    common.publish(0, 0, false);
                    break;
                };
                raw_item = next;
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply_stage<const ABLATE: u8>(self, sequence: u32, k: u32) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                common.load.wait(global_k);
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        self.accumulator.raw(),
                        self.a0.tile(global_k).k_walk(),
                        common.b.tile(global_k).k_walk(),
                        MmaShape::M256_N256,
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
                if k + 1 == common.k_blocks {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && k == 0 {
                    common.empty.sem(1).wait((sequence - 1) & 1);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        self.accumulator.columns_right(BLOCK_N as u32).raw(),
                        self.a1.tile(global_k).k_walk(),
                        common.b.tile(global_k).k_walk(),
                        MmaShape::M256_N256,
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
                if k + 1 == common.k_blocks {
                    commit_multicast_cg2(common.full.sem(1), PAIR);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                if common.next_info(sequence).has_work == 0 {
                    break;
                }
                if common.rank == LEADER && sequence > 0 {
                    common.empty.sem(0).wait((sequence - 1) & 1);
                }

                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE>(sequence, k);
                    self.multiply_stage::<ABLATE>(sequence, k + 1);
                    self.multiply_stage::<ABLATE>(sequence, k + 2);
                    self.multiply_stage::<ABLATE>(sequence, k + 3);
                    k += 4;
                }
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue<const ABLATE: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                let info = common.next_info(sequence);
                if info.has_work == 0 {
                    break;
                }
                let mut half = 0u32;
                while half < 2 {
                    common.full.sem(half).wait(sequence & 1);
                    if drains(ABLATE) {
                        common.drain::<LARGE_STAGE_N>(
                            self.accumulator.columns_right(half * BLOCK_N as u32),
                            info.row * 512 + half * 256,
                            info.column * BLOCK_N as u32,
                        );
                    }
                    let empty = common.empty.sem(half);
                    if common.rank == LEADER {
                        empty.arrive();
                    } else {
                        empty.at_rank(LEADER).arrive();
                    }
                    half += 1;
                }
                sequence += 1;
            }
        }
    }
}

/// The M256xN256 device body, with one dial on it.
///
/// The kernel below is this at [`WHOLE`] and nothing else; `experiments/`'
/// ablation arms are the same text at [`FEED_ONLY`], [`ISSUE_ONLY`] and
/// [`NO_DRAIN`], so no arm can drift from the kernel it decomposes.
///
/// # Safety
///
/// As [`kernels::gemm_sol_m256`].
#[inline(always)]
pub unsafe fn m256_body<const ABLATE: u8>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(
            smem,
            SMALL_RINGS_END,
            SMALL_B_OFFSET,
            tiles_m,
            tiles_n,
            k_blocks,
            ldc,
            c,
        );
        common.initialize(1);
        let state = Small {
            common,
            a: ARing::attach(smem.add(A0_OFFSET)),
            accumulator: Accumulator::from_raw(alloc_cluster(
                smem.add(tmem_offset(SMALL_RINGS_END)).cast(),
                ACCUM_COLUMNS,
            )),
            a_map,
            b_map,
        };

        if common.warp_id == TMA_WARP && common.lane == 0 {
            state.producer::<ABLATE>();
        } else if common.warp_id == MMA_WARP && common.lane == 0 {
            state.multiply::<ABLATE>();
        } else if common.warp_id < EPILOGUE_WARPS {
            state.epilogue::<ABLATE>();
        }

        common.retire();
        dealloc_cluster(state.accumulator.raw(), ACCUM_COLUMNS);
    }
}

/// The M512xN256 device body, per [`m256_body`].
///
/// # Safety
///
/// As [`kernels::gemm_sol_m512`].
#[inline(always)]
pub unsafe fn m512_body<const ABLATE: u8>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(
            smem,
            LARGE_RINGS_END,
            LARGE_B_OFFSET,
            tiles_m,
            tiles_n,
            k_blocks,
            ldc,
            c,
        );
        common.initialize(2);
        let state = Large {
            common,
            a0: ARing::attach(smem.add(A0_OFFSET)),
            a1: ARing::attach(smem.add(LARGE_A1_OFFSET)),
            accumulator: Accumulator::from_raw(alloc_cluster(
                smem.add(tmem_offset(LARGE_RINGS_END)).cast(),
                ACCUM_COLUMNS,
            )),
            a_map,
            b_map,
        };

        if common.warp_id == TMA_WARP && common.lane == 0 {
            state.producer::<ABLATE>();
        } else if common.warp_id == MMA_WARP && common.lane == 0 {
            state.multiply::<ABLATE>();
        } else if common.warp_id < EPILOGUE_WARPS {
            state.epilogue::<ABLATE>();
        }

        common.retire();
        dealloc_cluster(state.accumulator.raw(), ACCUM_COLUMNS);
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// # Safety
    ///
    /// The maps and output must cover the stated tile grid and K depth. The
    /// launch must be one two-CTA cluster per M256xN256 output tile.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            const { assert!(SMALL_SHARED_BYTES == 196_864) };
            m256_body::<WHOLE>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c);
        }
    }

    /// # Safety
    ///
    /// As [`gemm_sol_m256`], with an even M256 tile count.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            const { assert!(LARGE_SHARED_BYTES == 229_632) };
            m512_body::<WHOLE>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    M256xN256,
    M512xN256,
}

impl Variant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::M256xN256 => "M256xN256",
            Self::M512xN256 => "M512xN256",
        }
    }

    /// Rows of `C` one cluster owns.
    pub const fn m_tile(self) -> usize {
        match self {
            Self::M256xN256 => 256,
            Self::M512xN256 => 512,
        }
    }

    /// The dynamic shared plan this variant's kernel is launched with.
    pub const fn shared_bytes(self) -> usize {
        match self {
            Self::M256xN256 => SMALL_SHARED_BYTES,
            Self::M512xN256 => LARGE_SHARED_BYTES,
        }
    }
}

pub const fn select_variant(m: usize) -> Variant {
    if m >= 8_192 {
        Variant::M512xN256
    } else {
        Variant::M256xN256
    }
}

fn validate_shape(m: usize, n: usize, k: usize, variant: Variant) -> Result<(), Box<dyn Error>> {
    if m < variant.m_tile()
        || !m.is_multiple_of(variant.m_tile())
        || n < BLOCK_N
        || !n.is_multiple_of(BLOCK_N)
        || k < STAGES * BLOCK_K
        || !k.is_multiple_of(STAGES * BLOCK_K)
    {
        return Err(format!(
            "{m}x{n}x{k} violates {}'s M{}xN256, K%256 contract",
            variant.name(),
            variant.m_tile(),
        )
        .into());
    }
    Ok(())
}

fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

fn stage_f16(rows: usize, k: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u16> {
    let mut staged = Vec::with_capacity(rows * k);
    for row in 0..rows {
        for depth in 0..k {
            staged.push(half::f16::from_f32(value(row, depth)).to_bits());
        }
    }
    staged
}

fn check_output(observed: &[u16], m: usize, n: usize, k: usize) -> Result<f64, Box<dyn Error>> {
    let exact: Vec<f32> = (0..7 * 21)
        .map(|cell| {
            (0..k)
                .map(|depth| a_value(cell / 21, depth) * b_value(cell % 21, depth))
                .sum()
        })
        .collect();
    let reference: Vec<u16> = exact
        .iter()
        .map(|&value| half::bf16::from_f32(value).to_bits())
        .collect();

    let mut wrong = 0usize;
    let mut sample = Vec::new();
    let mut worst = 0.0f64;
    for row in 0..m {
        for column in 0..n {
            let value = observed[row * n + column];
            let cell = (row % 7) * 21 + column % 21;
            if value != reference[cell] {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!(
                        "C[{row}, {column}] = {}, want {}",
                        half::bf16::from_bits(value).to_f32(),
                        exact[cell],
                    ));
                }
            }
            let got = half::bf16::from_bits(value).to_f32() as f64;
            let want = exact[cell] as f64;
            worst = worst.max((got - want).abs() / want.abs().max(1.0));
        }
    }
    if wrong != 0 {
        return Err(format!(
            "{wrong}/{} BF16 outputs differ:\n  {}",
            m * n,
            sample.join("\n  "),
        )
        .into());
    }
    Ok(worst)
}

fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    variant: Variant,
    initialize: bool,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    validate_shape(m, n, k, variant)?;
    let stream = context.default_stream();
    let module = unsafe { kernels::load(context)? };
    let a = if initialize {
        DeviceBuffer::from_host(&stream, &stage_f16(m, k, a_value))?
    } else {
        DeviceBuffer::<u16>::zeroed(&stream, m * k)?
    };
    let b = if initialize {
        DeviceBuffer::from_host(&stream, &stage_f16(n, k, b_value))?
    } else {
        DeviceBuffer::<u16>::zeroed(&stream, n * k)?
    };
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let config = LaunchConfig1D::new(
        grid_for(crate::bench::Shape { m, n, k }, variant),
        THREADS,
        variant.shared_bytes() as u32,
    );
    let (stream_ref, module_ref) = (&stream, &module);
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    let k_blocks = (k / BLOCK_K) as u32;

    let launch_once: Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>> =
        match variant {
            Variant::M256xN256 => {
                let prepared = module_ref.prepare_gemm_sol_m256(config)?;
                Box::new(move |output| {
                    unsafe {
                        module_ref.gemm_sol_m256(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks,
                            n as u32, output,
                        )?
                    };
                    Ok(())
                })
            }
            Variant::M512xN256 => {
                let prepared = module_ref.prepare_gemm_sol_m512(config)?;
                Box::new(move |output| {
                    unsafe {
                        module_ref.gemm_sol_m512(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks,
                            n as u32, output,
                        )?
                    };
                    Ok(())
                })
            }
        };

    launch_once(&mut c)?;
    stream.synchronize()?;
    let label = if initialize {
        let worst = check_output(&c.to_host_vec(&stream)?, m, n, k)?;
        format!(
            "{} {m}x{n}x{k} exact over {} BF16 outputs, worst |rel| {worst:.2e}",
            variant.name(),
            m * n,
        )
    } else {
        format!("{} {m}x{n}x{k}", variant.name())
    };
    let mut launch = || launch_once(&mut c);
    let result = then(&stream, &mut launch)?;
    Ok((label, result))
}

fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for variant in [Variant::M256xN256, Variant::M512xN256] {
        notes.push(run(context, 1024, 1024, 512, variant, true, nothing_after)?.0);
    }
    Ok(notes.join("; "))
}

pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: crate::bench::Shape,
) -> Result<crate::bench::Timings, Box<dyn Error>> {
    let crate::bench::Shape { m, n, k } = shape;
    let variant = select_variant(m);
    run(context, 1024, 1024, 512, variant, true, nothing_after)?;
    Ok(run(context, m, n, k, variant, false, crate::bench::time)?.1)
}

/// One two-CTA cluster per output tile, in CTAs — `pub` for the ablation arms,
/// which must launch the geometry they are decomposing and not another.
pub fn grid_for(shape: crate::bench::Shape, variant: Variant) -> u32 {
    RANKS * (shape.m / variant.m_tile() * (shape.n / BLOCK_N)) as u32
}

pub fn grid(shape: crate::bench::Shape) -> u32 {
    grid_for(shape, select_variant(shape.m))
}
