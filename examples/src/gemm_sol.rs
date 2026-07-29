/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A kittens port of cuda-oxide's canonical Blackwell `gemm_sol_final`.
//!
//! Three entries, expressed through ferro-kittens' typed tiles and pipeline
//! primitives, differing only in the cluster tile they own: `[256, 128]`,
//! `[256, 256]` and `[512, 256]`. They share CLC work stealing, a two-CTA
//! cluster, four TMA/shared stages, two TMEM accumulator halves, an unrolled K
//! loop, and L2-aware output ordering, and [`select_variant`] picks between them
//! on wave arithmetic alone.
//!
//! On B200 that is `[256, 128]` at and below 37 wide output tiles — half a wave
//! of them — `[256, 256]` through 4K, and `[512, 256]` from 8K. The two narrow-tile branches are what
//! `bench sol` and `bench sol-small` added; the doc on [`select_variant`] is the
//! rule and the measurements that bound it.
//!
//! Data layout is the upstream contract: row-major FP16 A `[M, K]`, row-major
//! FP16 B `[N, K]` (therefore `A·Bᵀ`), and packed row-major BF16 C `[M, N]`.
//! `n` is a multiple of 256 for the two 256-wide entries and of 128 for
//! `[256, 128]`, which is the one place the shape contract widens.

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

const BLOCK_M: usize = 128;
const BLOCK_N: usize = 256;
const HALF_N: usize = BLOCK_N / 2;
const BLOCK_K: usize = 64;
const CHUNKS: usize = BLOCK_K / 16;
const STAGES: usize = 4;
const NARROW_N: usize = BLOCK_N / 2;
const HALF_NARROW_N: usize = NARROW_N / 2;
const EPILOGUE_WARPS: u32 = (BLOCK_M / 32) as u32;
const TMA_WARP: u32 = EPILOGUE_WARPS;
const MMA_WARP: u32 = TMA_WARP + 1;
pub const THREADS: u32 = (MMA_WARP + 1) * 32;
const RANKS: u32 = 2;
const LEADER: u32 = 0;
const PAIR: u16 = 0b11;
const BAND_N: usize = 64;
/// The TMA box `B` arrives in, and therefore the step the load loop takes
/// through the half-panel a rank owns.
const B_BOX: usize = 64;
const LARGE_STAGE_N: usize = HALF_N;

const _: () = {
    assert!(THREADS == 192);
    assert!(CHUNKS == 4);
    assert!(HALF_N.is_multiple_of(B_BOX));
    assert!(HALF_NARROW_N.is_multiple_of(B_BOX));
};

type ATile = SharedTile<F16, BLOCK_M, BLOCK_K, Swizzle128B>;

type BPanel = SharedTile<F16, 64, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<F16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<F16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type NarrowBRing = SharedTileRing<F16, HALF_NARROW_N, BLOCK_K, Swizzle128B, STAGES>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;
type StageBand = RegTile<32, BAND_N, BaseLdtm>;

/// The `tcgen05` shape a `[256, N]` cluster tile issues. It is a `const fn`
/// rather than a `Variant` method because the MMA warp reads it inside a
/// width-generic body, where the width is a const parameter and the shape has
/// to fold away with it.
const fn mma_shape(n: usize) -> MmaShape {
    match n {
        NARROW_N => MmaShape::M256_N128,
        BLOCK_N => MmaShape::M256_N256,
        _ => panic!("only the 128- and 256-wide cluster tiles are built"),
    }
}

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
const NARROW_RINGS_END: usize = SMALL_B_OFFSET + NarrowBRing::BYTES;
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

pub const SMALL_SHARED_BYTES: usize = shared_plan(SMALL_RINGS_END, BLOCK_N);
pub const NARROW_SHARED_BYTES: usize = shared_plan(NARROW_RINGS_END, NARROW_N);
pub const LARGE_SHARED_BYTES: usize = shared_plan(LARGE_RINGS_END, LARGE_STAGE_N);
const _: () = {
    assert!(SMALL_SHARED_BYTES == 196_864);
    assert!(NARROW_SHARED_BYTES == 131_328);
    assert!(LARGE_SHARED_BYTES == 229_632);
    assert!(LARGE_SHARED_BYTES <= 233_472);
};

#[derive(Clone, Copy)]
struct Common {
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
    group: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl Common {
    #[inline(always)]
    unsafe fn attach(
        smem: *mut u8,
        rings_end: usize,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        group: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> Self {
        unsafe {
            Self {
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
                group,
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
    fn accumulator<const N: usize>(base: TmemTile<BLOCK_M, N>, index: u32) -> TmemTile<BLOCK_M, N> {
        if index.is_multiple_of(2) {
            base
        } else {
            base.columns_right(N as u32)
        }
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
    unsafe fn drain<const N: usize, const STAGE_N: usize>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(N.is_multiple_of(STAGE_N));
            };
            let stage =
                SharedTile::<Bf16, 32, STAGE_N, Swizzle128B>::from_raw(self.stage_base.add(
                    self.warp_id as usize * SharedTile::<Bf16, 32, STAGE_N, Swizzle128B>::BYTES,
                ));
            let row = tile_row + self.rank * BLOCK_M as u32 + self.warp_id * 32;
            let mut column = 0u32;
            while column < N as u32 {
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

/// The one-accumulator entry, generic in the cluster tile's width.
///
/// `N` is the tile's columns of `C`, `HALF` the half-panel of `B` a rank
/// loads, and `STAGE` the columns of the shared epilogue staging buffer. Two
/// widths are instantiated — `[256, 256]` and `[256, 128]` — and they differ in
/// nothing else, which is what makes the pair a controlled comparison of tile
/// quantization against operand traffic.
#[derive(Clone, Copy)]
struct Small<const N: usize, const HALF: usize, const STAGE: usize> {
    common: Common,
    a: ARing,
    b: SharedTileRing<F16, HALF, BLOCK_K, Swizzle128B, STAGES>,
    accumulator: TmemTile<BLOCK_M, N>,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl<const N: usize, const HALF: usize, const STAGE: usize> Small<N, HALF, STAGE> {
    #[inline(always)]
    unsafe fn producer(self) {
        unsafe {
            const { assert!(N == 2 * HALF, "a rank loads exactly half the panel") };
            let common = self.common;
            let mut cursor: ClcCursor = common.queue.cursor();
            let mut raw_item = cluster::cluster_idx();
            let mut sequence = 0u32;
            let valid_items = common.tiles_m * common.tiles_n;

            loop {
                if raw_item < valid_items {
                    let (tile_n, tile_m) =
                        pipeline::grouped(raw_item, common.tiles_n, common.tiles_m, common.group);
                    common.publish(tile_m, tile_n, true);
                    let a_row =
                        (tile_m * (2 * BLOCK_M) as u32 + common.rank * BLOCK_M as u32) as i32;
                    let b_row = (tile_n * N as u32 + common.rank * HALF as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        common.free.wait_recycled(global_k);
                        let load = common.load.sem(global_k).at_rank(LEADER);
                        let k_base = (k * BLOCK_K as u32) as i32;
                        let mut bytes = self
                            .a
                            .tile(global_k)
                            .tma_load_2d_arriving_at(self.a_map, k_base, a_row, load);
                        let b = self.b.tile(global_k);
                        let mut box_row = 0usize;
                        while box_row < HALF {
                            bytes = bytes
                                + b.tma_load_2d_at_arriving_at::<B_BOX>(
                                    self.b_map,
                                    box_row,
                                    k_base,
                                    b_row + box_row as i32,
                                    load,
                                );
                            box_row += B_BOX;
                        }
                        if common.rank == LEADER {
                            common
                                .load
                                .sem(global_k)
                                .expect_tx(bytes.across_ranks(RANKS));
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
    unsafe fn multiply_stage(self, accumulator: TmemTile<BLOCK_M, N>, sequence: u32, k: u32) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                common.load.wait(global_k);
                mma_walk_cg2::<F16, CHUNKS>(
                    accumulator.raw(),
                    self.a.tile(global_k).k_walk(),
                    self.b.tile(global_k).k_walk(),
                    mma_shape(N),
                    k > 0,
                );
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply(self) {
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
                    self.multiply_stage(accumulator, sequence, k);
                    self.multiply_stage(accumulator, sequence, k + 1);
                    self.multiply_stage(accumulator, sequence, k + 2);
                    self.multiply_stage(accumulator, sequence, k + 3);
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
    unsafe fn epilogue(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                let info = common.next_info(sequence);
                if info.has_work == 0 {
                    break;
                }
                common.full.wait(sequence);
                common.drain::<N, STAGE>(
                    Common::accumulator(self.accumulator, sequence),
                    info.row * (2 * BLOCK_M) as u32,
                    info.column * N as u32,
                );
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
    b: BRing,
    accumulator: Accumulator,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl Large {
    #[inline(always)]
    unsafe fn producer(self) {
        unsafe {
            let common = self.common;
            let mut cursor = common.queue.cursor();
            let mut raw_item = cluster::cluster_idx();
            let mut sequence = 0u32;
            let macro_tiles_m = common.tiles_m / 2;
            let valid_items = macro_tiles_m * common.tiles_n;

            loop {
                if raw_item < valid_items {
                    let (tile_n, macro_m) =
                        pipeline::grouped(raw_item, common.tiles_n, macro_tiles_m, common.group);
                    common.publish(macro_m, tile_n, true);
                    let a_row0 = (macro_m * 512 + common.rank * BLOCK_M as u32) as i32;
                    let a_row1 = a_row0 + 256;
                    let b_row = (tile_n * BLOCK_N as u32 + common.rank * HALF_N as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        common.free.wait_recycled(global_k);
                        let load = common.load.sem(global_k).at_rank(LEADER);
                        let k_base = (k * BLOCK_K as u32) as i32;
                        let b = self.b.tile(global_k);
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
    unsafe fn multiply_stage(self, sequence: u32, k: u32) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                common.load.wait(global_k);
                mma_walk_cg2::<F16, CHUNKS>(
                    self.accumulator.raw(),
                    self.a0.tile(global_k).k_walk(),
                    self.b.tile(global_k).k_walk(),
                    MmaShape::M256_N256,
                    k > 0,
                );
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
                if k + 1 == common.k_blocks {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && k == 0 {
                    common.empty.sem(1).wait((sequence - 1) & 1);
                }
                mma_walk_cg2::<F16, CHUNKS>(
                    self.accumulator.columns_right(BLOCK_N as u32).raw(),
                    self.a1.tile(global_k).k_walk(),
                    self.b.tile(global_k).k_walk(),
                    MmaShape::M256_N256,
                    k > 0,
                );
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
                if k + 1 == common.k_blocks {
                    commit_multicast_cg2(common.full.sem(1), PAIR);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply(self) {
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
                    self.multiply_stage(sequence, k);
                    self.multiply_stage(sequence, k + 1);
                    self.multiply_stage(sequence, k + 2);
                    self.multiply_stage(sequence, k + 3);
                    k += 4;
                }
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue(self) {
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
                    common.drain::<BLOCK_N, LARGE_STAGE_N>(
                        self.accumulator.columns_right(half * BLOCK_N as u32),
                        info.row * 512 + half * 256,
                        info.column * BLOCK_N as u32,
                    );
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
        group: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            const { assert!(SMALL_SHARED_BYTES == 196_864) };
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let common = Common::attach(
                smem,
                SMALL_RINGS_END,
                tiles_m,
                tiles_n,
                k_blocks,
                group,
                ldc,
                &mut c,
            );
            common.initialize(1);
            let state = Small::<BLOCK_N, HALF_N, BLOCK_N> {
                common,
                a: ARing::attach(smem.add(A0_OFFSET)),
                b: BRing::attach(smem.add(SMALL_B_OFFSET)),
                accumulator: TmemTile::from_raw(alloc_cluster(
                    smem.add(tmem_offset(SMALL_RINGS_END)).cast(),
                    2 * BLOCK_N as u32,
                )),
                a_map,
                b_map,
            };

            if common.warp_id == TMA_WARP && common.lane == 0 {
                state.producer();
            } else if common.warp_id == MMA_WARP && common.lane == 0 {
                state.multiply();
            } else if common.warp_id < EPILOGUE_WARPS {
                state.epilogue();
            }

            common.retire();
            dealloc_cluster(state.accumulator.raw(), 2 * BLOCK_N as u32);
        }
    }

    /// The same entry at half the width: a `[256, 128]` cluster tile, which
    /// quadruples the tile count of a square problem against
    /// [`gemm_sol_m256`]'s and is what a shape too small to fill a wave of the
    /// wider one is for.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256`], with `n` a multiple of 128 rather than 256.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_328,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_n128(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        group: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            const { assert!(NARROW_SHARED_BYTES == 131_328) };
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let common = Common::attach(
                smem,
                NARROW_RINGS_END,
                tiles_m,
                tiles_n,
                k_blocks,
                group,
                ldc,
                &mut c,
            );
            common.initialize(1);
            let state = Small::<NARROW_N, HALF_NARROW_N, NARROW_N> {
                common,
                a: ARing::attach(smem.add(A0_OFFSET)),
                b: NarrowBRing::attach(smem.add(SMALL_B_OFFSET)),
                accumulator: TmemTile::from_raw(alloc_cluster(
                    smem.add(tmem_offset(NARROW_RINGS_END)).cast(),
                    2 * NARROW_N as u32,
                )),
                a_map,
                b_map,
            };

            if common.warp_id == TMA_WARP && common.lane == 0 {
                state.producer();
            } else if common.warp_id == MMA_WARP && common.lane == 0 {
                state.multiply();
            } else if common.warp_id < EPILOGUE_WARPS {
                state.epilogue();
            }

            common.retire();
            dealloc_cluster(state.accumulator.raw(), 2 * NARROW_N as u32);
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
        group: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            const { assert!(LARGE_SHARED_BYTES == 229_632) };
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let common = Common::attach(
                smem,
                LARGE_RINGS_END,
                tiles_m,
                tiles_n,
                k_blocks,
                group,
                ldc,
                &mut c,
            );
            common.initialize(2);
            let state = Large {
                common,
                a0: ARing::attach(smem.add(A0_OFFSET)),
                a1: ARing::attach(smem.add(LARGE_A1_OFFSET)),
                b: BRing::attach(smem.add(LARGE_B_OFFSET)),
                accumulator: Accumulator::from_raw(alloc_cluster(
                    smem.add(tmem_offset(LARGE_RINGS_END)).cast(),
                    2 * BLOCK_N as u32,
                )),
                a_map,
                b_map,
            };

            if common.warp_id == TMA_WARP && common.lane == 0 {
                state.producer();
            } else if common.warp_id == MMA_WARP && common.lane == 0 {
                state.multiply();
            } else if common.warp_id < EPILOGUE_WARPS {
                state.epilogue();
            }

            common.retire();
            dealloc_cluster(state.accumulator.raw(), 2 * BLOCK_N as u32);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    M256xN128,
    M256xN256,
    M512xN256,
}

impl Variant {
    pub const ALL: [Variant; 3] = [Variant::M256xN128, Variant::M256xN256, Variant::M512xN256];

    pub const fn name(self) -> &'static str {
        match self {
            Self::M256xN128 => "M256xN128",
            Self::M256xN256 => "M256xN256",
            Self::M512xN256 => "M512xN256",
        }
    }

    pub const fn m_tile(self) -> usize {
        match self {
            Self::M256xN128 | Self::M256xN256 => 256,
            Self::M512xN256 => 512,
        }
    }

    pub const fn n_tile(self) -> usize {
        match self {
            Self::M256xN128 => NARROW_N,
            Self::M256xN256 | Self::M512xN256 => BLOCK_N,
        }
    }

    pub const fn shared_bytes(self) -> usize {
        match self {
            Self::M256xN128 => NARROW_SHARED_BYTES,
            Self::M256xN256 => SMALL_SHARED_BYTES,
            Self::M512xN256 => LARGE_SHARED_BYTES,
        }
    }
}

/// Clusters this device holds at once: 148 SMs, one CTA per SM, two CTAs to a
/// cluster.
///
/// Both the one CTA and the two are facts rather than choices — every shared
/// plan here declares more than half of the 233472 B an SM divides, and a
/// `cta_group::2` MMA has to have its pair co-resident — so this is a device
/// property, not a tuning knob. It is written down rather than queried because
/// [`select_variant`] is a `const fn` that [`grid`] calls with a shape and
/// nothing else; `bench sol`'s table 0 prints the same number off
/// `cuDeviceGetAttribute` beside every row it divides, which is where a device
/// that disagrees with this would show up.
const RESIDENT_CLUSTERS: usize = 74;

/// The entry a shape gets. All three branches are wave arithmetic.
///
/// A launch is one cluster per output tile and takes `ceil(tiles / 74)` waves of
/// them, so `tiles / (waves * 74)` is the fraction of it that is not idling and
/// halving the tile's `N` doubles the tile count. That doubling raises the
/// fraction **only** while both counts fit the same number of waves — at or
/// below half a wave of wide tiles — and above it the two are equal and the
/// narrow tile is left paying its costs for nothing. `bench sol-small` measures
/// exactly that boundary: at 16 and 36 wide tiles the narrow entry is 1.27x,
/// and at 64 and 144 it is 0.93x and 0.81x, both reproduced twice.
///
/// `M512xN256` above 8192 is #138's crossover, which upstream takes at 16384;
/// `bench sol` has it 1.12x at 8192³ against a wave efficiency both entries
/// share, so what it wins there is operand traffic per flop and not tiles.
pub const fn select_variant(m: usize, n: usize) -> Variant {
    if !n.is_multiple_of(BLOCK_N) {
        // The only entry whose contract admits this `n` at all.
        Variant::M256xN128
    } else if m >= 8_192 && m.is_multiple_of(2 * 256) {
        Variant::M512xN256
    } else if 2 * (m / 256) * (n / BLOCK_N) <= RESIDENT_CLUSTERS {
        Variant::M256xN128
    } else {
        Variant::M256xN256
    }
}

/// The N-band width [`pipeline::grouped`] walks, which was a rule inside the
/// kernel until it became a launch parameter.
///
/// The rule is unchanged and the default is what it computed, because the sweep
/// that could have changed it found nothing to change it to: `bench sol`'s
/// table 3 takes `G` over `{1, 2, 4, 8, 16}` and gets 1880.3 to 1876.2 TFLOP/s
/// at 8192³ — flat to 0.2% — and a 1311 to 1367 range at 4096³ against rows
/// whose own launches spread 1.3 to 3.6%. Tuning on the second of those would be
/// tuning on noise.
pub const fn default_group(m: usize) -> u32 {
    if m / 256 <= 16 { 2 } else { 8 }
}

/// Everything the launch decides that is not the shape: which entry, and how
/// wide a band of `N` its traversal walks before it steps in `M`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub variant: Variant,
    pub group: u32,
}

impl Plan {
    pub const fn shipped(shape: crate::bench::Shape) -> Plan {
        Plan {
            variant: select_variant(shape.m, shape.n),
            group: default_group(shape.m),
        }
    }
}

fn validate_shape(m: usize, n: usize, k: usize, variant: Variant) -> Result<(), Box<dyn Error>> {
    if m < variant.m_tile()
        || !m.is_multiple_of(variant.m_tile())
        || n < variant.n_tile()
        || !n.is_multiple_of(variant.n_tile())
        || k < STAGES * BLOCK_K
        || !k.is_multiple_of(STAGES * BLOCK_K)
    {
        return Err(format!(
            "{m}x{n}x{k} violates {}'s M{}xN{}, K%256 contract",
            variant.name(),
            variant.m_tile(),
            variant.n_tile(),
        )
        .into());
    }
    Ok(())
}

/// `A`'s generator. Small integers, so every product and every partial sum is
/// exact in fp32 and [`check_output`] is `==` rather than a tolerance.
///
/// `pub` because `experiments/`' copy of the *unported* upstream kernel is
/// staged and checked by these four functions rather than by re-derivations of
/// them (#138). Two kernels compared on one clock have to read byte-identical
/// operands, and the way to guarantee that is to call the same code.
pub fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

/// `B`'s generator, per [`a_value`].
pub fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

/// One operand, packed f16 row-major with K contiguous — the layout both this
/// kernel and upstream's take.
pub fn stage_f16(rows: usize, k: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u16> {
    let mut staged = Vec::with_capacity(rows * k);
    for row in 0..rows {
        for depth in 0..k {
            staged.push(half::f16::from_f32(value(row, depth)).to_bits());
        }
    }
    staged
}

/// Every one of `m * n` BF16 outputs against the exact reference, per
/// [`a_value`]. Returns the worst relative error, which is bf16's own and not
/// a tolerance the comparison was given.
pub fn check_output(observed: &[u16], m: usize, n: usize, k: usize) -> Result<f64, Box<dyn Error>> {
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
    plan: Plan,
    initialize: bool,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    let Plan { variant, group } = plan;
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
    let tiles_n = (n / variant.n_tile()) as u32;
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
            Variant::M256xN128 => {
                let prepared = module_ref.prepare_gemm_sol_m256_n128(config)?;
                Box::new(move |output| {
                    unsafe {
                        module_ref.gemm_sol_m256_n128(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group,
                            n as u32, output,
                        )?
                    };
                    Ok(())
                })
            }
            Variant::M256xN256 => {
                let prepared = module_ref.prepare_gemm_sol_m256(config)?;
                Box::new(move |output| {
                    unsafe {
                        module_ref.gemm_sol_m256(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group,
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
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group,
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
    for variant in Variant::ALL {
        let plan = Plan {
            variant,
            group: default_group(1024),
        };
        notes.push(run(context, 1024, 1024, 512, plan, true, nothing_after)?.0);
    }
    Ok(notes.join("; "))
}

/// Check the plan at the gate size, then time it at `shape` — the order every
/// number in this file comes out of, and the entry point a sweep varying the
/// plan calls instead of [`bench`].
pub fn bench_plan(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: crate::bench::Shape,
    plan: Plan,
) -> Result<crate::bench::Timings, Box<dyn Error>> {
    let crate::bench::Shape { m, n, k } = shape;
    run(context, 1024, 1024, 512, plan, true, nothing_after)?;
    Ok(run(context, m, n, k, plan, false, crate::bench::time)?.1)
}

pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: crate::bench::Shape,
) -> Result<crate::bench::Timings, Box<dyn Error>> {
    bench_plan(context, shape, Plan::shipped(shape))
}

fn grid_for(shape: crate::bench::Shape, variant: Variant) -> u32 {
    RANKS * (shape.m / variant.m_tile() * (shape.n / variant.n_tile())) as u32
}

pub fn grid(shape: crate::bench::Shape) -> u32 {
    grid_for(shape, select_variant(shape.m, shape.n))
}

/// Clusters the launch asks for, which is one per output tile — the number the
/// wave arithmetic divides by residency.
pub fn clusters(shape: crate::bench::Shape, variant: Variant) -> u32 {
    grid_for(shape, variant) / RANKS
}
