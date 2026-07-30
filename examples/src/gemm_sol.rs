/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A kittens port of cuda-oxide's canonical Blackwell `gemm_sol_final`.
//!
//! Three entries, expressed through ferro-kittens' typed tiles and pipeline
//! primitives, differing only in the cluster tile they own: `[256, 128]`,
//! `[256, 256]` and `[512, 256]`. They share CLC work stealing, a two-CTA
//! cluster, four TMA/shared stages, two TMEM accumulator halves, a K loop
//! unrolled four ways over a **compile-time** ring stage, and L2-aware output
//! ordering, and [`select_variant`] picks between them on wave arithmetic alone.
//!
//! What the K loop is bound by is measured rather than assumed, and it is not the
//! same thing at the two 256-row entries. `bench sol-ablate` ladders every phase
//! in `K`: the tensor core can be issued at peak from one warp
//! (`issue only` is 100.0% of peak a K block), the barrier round trip is zero,
//! and the whole of `[256, 256]`'s 21% deficit is its **feed's duty cycle** — the
//! operand pipeline alone needs 95.5% of the time the tensor core is busy, where
//! `[512, 256]`'s needs 55.7%, because a `[256, 256]` cluster tile moves 1.33× the
//! operand bytes per flop. That is a property of the tile, so it is [`Variant`]'s
//! to answer and not the K loop's.
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

/// The whole kernel. Every shipped entry passes this and nothing else does.
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
/// [`ISSUE_ONLY`] with the MMA warp's `load.wait` removed as well, so the K loop
/// issues `tcgen05.mma` with nothing in front of it.
///
/// This is the arm #144 did not have and named as the thing it could not
/// separate. `issue only` keeps the whole `load`/`free` handshake, so the
/// distance it measures is barrier round-trip *fused with* tensor-core issue
/// rate; this arm drops one of the two and `issue only − mma only` is the round
/// trip alone. The producer keeps `free.wait_recycled`, so it is still throttled
/// and the `ready` publication still bounds the MMA warp one output tile at a
/// time — what runs unthrottled is exactly the K loop, which is what is being
/// priced.
pub const MMA_ONLY: u8 = 4;

const fn loads(ablate: u8) -> bool {
    ablate != ISSUE_ONLY && ablate != MMA_ONLY
}
const fn multiplies(ablate: u8) -> bool {
    ablate != FEED_ONLY
}
const fn drains(ablate: u8) -> bool {
    ablate == WHOLE
}
const fn waits_on_load(ablate: u8) -> bool {
    ablate != MMA_ONLY
}

const BLOCK_M: usize = 128;
/// Columns of `C` one wide cluster tile owns — `pub` because the ablation arms
/// instantiate the same device body at the same width.
pub const BLOCK_N: usize = 256;
/// The half-panel of `B` a rank of a wide cluster tile loads.
pub const HALF_N: usize = BLOCK_N / 2;
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
/// through the half-panel a rank owns — the shipped value, which every entry
/// carried unconditionally until it became [`Small`]'s and [`Large`]'s `BOX`
/// parameter.
///
/// It is 64 because `B`'s tensor map is built for a 64-row box and every entry
/// shared one map: the narrow entry's half-panel *is* 64 rows, so 64 is the only
/// height all three can name. Nothing about the wide entries asks for it — their
/// half-panel is `128 x 64`, the same shape as [`ATile`], which already arrives
/// in a single TMA — so they pay two instructions where one would do.
///
/// **And it costs nothing, which is why 64 still ships.** `bench sol-ablate`'s
/// wide-`B` arms are that map built per entry instead of once, at byte-for-byte
/// identical traffic, and they move the launch by 0.998 and 1.014 across two
/// passes, the K-block rate from 0.3503 to 0.3484 µs, and `feed only` — where the
/// feed is alone and has nothing to hide behind — by 1.008 and 0.981. The feed's
/// ceiling is bytes, not instructions.
pub const B_BOX: usize = 64;
/// The box the two 256-wide entries' half-panel would arrive in whole.
pub const WIDE_B_BOX: usize = HALF_N;
const LARGE_STAGE_N: usize = HALF_N;

const _: () = {
    assert!(THREADS == 192);
    assert!(CHUNKS == 4);
    assert!(HALF_N.is_multiple_of(B_BOX));
    assert!(HALF_NARROW_N.is_multiple_of(B_BOX));
    assert!(HALF_N.is_multiple_of(WIDE_B_BOX));
};

/// The `A` tile a tensor map is built for — `pub` because `experiments/`' arms
/// build the same maps this file's own runner does.
pub type ATile = SharedTile<F16, BLOCK_M, BLOCK_K, Swizzle128B>;

/// The `B` panel a tensor map is built for at [`B_BOX`], per [`ATile`].
pub type BPanel = SharedTile<F16, B_BOX, BLOCK_K, Swizzle128B>;
/// The `B` panel a tensor map is built for at [`WIDE_B_BOX`]: a rank's whole
/// half-panel in one box.
pub type WideBPanel = SharedTile<F16, WIDE_B_BOX, BLOCK_K, Swizzle128B>;
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
/// Where the wide one-accumulator entry's rings end, which is the offset every
/// barrier and the epilogue staging buffer are placed after.
pub const SMALL_RINGS_END: usize = SMALL_B_OFFSET + BRing::BYTES;
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
/// loads, `BOX` the TMA box that half-panel arrives in, and `STAGE` the columns
/// of the shared epilogue staging buffer. Two widths are instantiated —
/// `[256, 256]` and `[256, 128]` — and they differ in nothing else, which is
/// what makes the pair a controlled comparison of tile quantization against
/// operand traffic.
#[derive(Clone, Copy)]
struct Small<const N: usize, const HALF: usize, const BOX: usize, const STAGE: usize> {
    common: Common,
    a: ARing,
    b: SharedTileRing<F16, HALF, BLOCK_K, Swizzle128B, STAGES>,
    accumulator: TmemTile<BLOCK_M, N>,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl<const N: usize, const HALF: usize, const BOX: usize, const STAGE: usize>
    Small<N, HALF, BOX, STAGE>
{
    #[inline(always)]
    unsafe fn producer<const ABLATE: u8>(self) {
        unsafe {
            const {
                assert!(N == 2 * HALF, "a rank loads exactly half the panel");
                assert!(HALF.is_multiple_of(BOX), "the panel is whole boxes");
            };
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
                        if loads(ABLATE) {
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
                                    + b.tma_load_2d_at_arriving_at::<BOX>(
                                        self.b_map,
                                        box_row,
                                        k_base,
                                        b_row + box_row as i32,
                                        load,
                                    );
                                box_row += BOX;
                            }
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
        accumulator: TmemTile<BLOCK_M, N>,
        sequence: u32,
        k: u32,
    ) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                if waits_on_load(ABLATE) {
                    common.load.wait(global_k);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        accumulator.raw(),
                        self.a.tile(global_k).k_walk(),
                        self.b.tile(global_k).k_walk(),
                        mma_shape(N),
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
            }
        }
    }

    /// One K block at a **compile-time** stage.
    ///
    /// `k_blocks` is a multiple of `STAGES` — every entry's shape contract is
    /// `k % (STAGES * BLOCK_K) == 0` — so `global_k % STAGES == k % STAGES`, and
    /// the four positions of a four-way unroll are always stages 0, 1, 2, 3 in
    /// that order. Handing each position its stage as a const is what turns two
    /// operand descriptors and two barrier addresses from runtime arithmetic
    /// into folded offsets. The phase parity is the one thing that genuinely
    /// moves with `global_k`, and it moves once a turn instead of four times.
    ///
    /// Upstream's `gemm_sol_final` states the same fact — *"`k_iters % 4 == 0`,
    /// so the producer's global stage and this local stage agree at every tile
    /// boundary. Keeping this expression loop-local lets the unroll pass fold
    /// each stage match."* — and spells it as `#[unroll(4)]` over a loop-local
    /// `k_idx & 3` with a `match` on it. That attribute is rewritten only inside
    /// a `#[kernel]` or `#[device_function]` body, so a const parameter is the
    /// spelling reachable from a plain `impl` method; it also does not depend on
    /// an unroll pass firing.
    #[inline(always)]
    unsafe fn multiply_at<const ABLATE: u8, const SLOT: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        parity: u32,
        accumulate: bool,
    ) {
        unsafe {
            let common = self.common;
            if common.rank == LEADER {
                if waits_on_load(ABLATE) {
                    common.load.sem(SLOT).wait(parity);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        accumulator.raw(),
                        self.a.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
                        mma_shape(N),
                        accumulate,
                    );
                }
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
            }
        }
    }

    /// The K walk, `STAGES` blocks a turn.
    ///
    /// `FOLD` picks the spelling: [`Self::multiply_at`]'s const stage, or
    /// [`Self::multiply_stage`]'s `global_k`, which is what the port carried
    /// before the two were measured against each other. The two compute the same
    /// `C` by construction — the const is `global_k % STAGES` and nothing else —
    /// so the difference between them is entirely how much of the MMA warp's
    /// issue stream is scalar arithmetic.
    #[inline(always)]
    unsafe fn walk_k<const ABLATE: u8, const FOLD: bool>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        sequence: u32,
    ) {
        unsafe {
            let common = self.common;
            if FOLD {
                let turns = common.k_blocks / STAGES as u32;
                let cycle = sequence * turns;
                let mut turn = 0u32;
                while turn < turns {
                    let parity = (cycle + turn) & 1;
                    self.multiply_at::<ABLATE, 0>(accumulator, parity, turn > 0);
                    self.multiply_at::<ABLATE, 1>(accumulator, parity, true);
                    self.multiply_at::<ABLATE, 2>(accumulator, parity, true);
                    self.multiply_at::<ABLATE, 3>(accumulator, parity, true);
                    turn += 1;
                }
            } else {
                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 1);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 2);
                    self.multiply_stage::<ABLATE>(accumulator, sequence, k + 3);
                    k += 4;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8, const FOLD: bool>(self) {
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

                self.walk_k::<ABLATE, FOLD>(accumulator, sequence);
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
                    common.drain::<N, STAGE>(
                        Common::accumulator(self.accumulator, sequence),
                        info.row * (2 * BLOCK_M) as u32,
                        info.column * N as u32,
                    );
                }
                common.release_accumulator(sequence);
                sequence += 1;
            }
        }
    }
}

/// The two-accumulator entry: two `A` rings against one shared `B` half-panel,
/// so a K block is twice the MMA against the same one `load` round trip.
///
/// `BOX` is the TMA box that half-panel arrives in, exactly as on [`Small`].
#[derive(Clone, Copy)]
struct Large<const BOX: usize> {
    common: Common,
    a0: ARing,
    a1: ARing,
    b: BRing,
    accumulator: Accumulator,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
}

impl<const BOX: usize> Large<BOX> {
    #[inline(always)]
    unsafe fn producer<const ABLATE: u8>(self) {
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
                        if loads(ABLATE) {
                            let load = common.load.sem(global_k).at_rank(LEADER);
                            let k_base = (k * BLOCK_K as u32) as i32;
                            let b = self.b.tile(global_k);
                            let mut bytes = self
                                .a0
                                .tile(global_k)
                                .tma_load_2d_arriving_at(self.a_map, k_base, a_row0, load)
                                + self
                                    .a1
                                    .tile(global_k)
                                    .tma_load_2d_arriving_at(self.a_map, k_base, a_row1, load);
                            let mut box_row = 0usize;
                            while box_row < HALF_N {
                                bytes = bytes
                                    + b.tma_load_2d_at_arriving_at::<BOX>(
                                        self.b_map,
                                        box_row,
                                        k_base,
                                        b_row + box_row as i32,
                                        load,
                                    );
                                box_row += BOX;
                            }
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
                if waits_on_load(ABLATE) {
                    common.load.wait(global_k);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        self.accumulator.raw(),
                        self.a0.tile(global_k).k_walk(),
                        self.b.tile(global_k).k_walk(),
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
                        self.b.tile(global_k).k_walk(),
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

    /// One K block at a compile-time stage, per [`Small::multiply_at`].
    ///
    /// `first` and `last` are the two places this entry's K block is not
    /// interchangeable with its neighbours — the `empty` wait at `k == 0` and the
    /// `full` commit at `k + 1 == k_blocks` — and both are literals at three of
    /// the four positions, so they fold away with the stage.
    #[inline(always)]
    unsafe fn multiply_at<const ABLATE: u8, const SLOT: u32>(
        self,
        sequence: u32,
        parity: u32,
        accumulate: bool,
        first: bool,
        last: bool,
    ) {
        unsafe {
            let common = self.common;
            if common.rank == LEADER {
                if waits_on_load(ABLATE) {
                    common.load.sem(SLOT).wait(parity);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        self.accumulator.raw(),
                        self.a0.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
                        MmaShape::M256_N256,
                        accumulate,
                    );
                }
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
                if last {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && first {
                    common.empty.sem(1).wait((sequence - 1) & 1);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS>(
                        self.accumulator.columns_right(BLOCK_N as u32).raw(),
                        self.a1.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
                        MmaShape::M256_N256,
                        accumulate,
                    );
                }
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
                if last {
                    commit_multicast_cg2(common.full.sem(1), PAIR);
                }
            }
        }
    }

    /// The K walk, per [`Small::walk_k`].
    #[inline(always)]
    unsafe fn walk_k<const ABLATE: u8, const FOLD: bool>(self, sequence: u32) {
        unsafe {
            let common = self.common;
            if FOLD {
                let turns = common.k_blocks / STAGES as u32;
                let cycle = sequence * turns;
                let mut turn = 0u32;
                while turn < turns {
                    let parity = (cycle + turn) & 1;
                    let last = turn + 1 == turns;
                    self.multiply_at::<ABLATE, 0>(sequence, parity, turn > 0, turn == 0, false);
                    self.multiply_at::<ABLATE, 1>(sequence, parity, true, false, false);
                    self.multiply_at::<ABLATE, 2>(sequence, parity, true, false, false);
                    self.multiply_at::<ABLATE, 3>(sequence, parity, true, false, last);
                    turn += 1;
                }
            } else {
                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE>(sequence, k);
                    self.multiply_stage::<ABLATE>(sequence, k + 1);
                    self.multiply_stage::<ABLATE>(sequence, k + 2);
                    self.multiply_stage::<ABLATE>(sequence, k + 3);
                    k += 4;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8, const FOLD: bool>(self) {
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
                self.walk_k::<ABLATE, FOLD>(sequence);
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
                        common.drain::<BLOCK_N, LARGE_STAGE_N>(
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

/// The width-generic device body, with two dials on it.
///
/// The two `[256, N]` kernels below are this at [`WHOLE`] and [`B_BOX`] and
/// nothing else; `experiments/`' arms are the same text at another `ABLATE` or
/// another `BOX`, so no arm can drift from the kernel it decomposes and a
/// wide-box arm differs from the shipped kernel in exactly one const.
///
/// # Safety
///
/// As [`kernels::gemm_sol_m256`], at the entry's own width.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_body<
    const N: usize,
    const HALF: usize,
    const BOX: usize,
    const STAGE: usize,
    const RINGS_END: usize,
    const ABLATE: u8,
    const FOLD: bool,
>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    group: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(smem, RINGS_END, tiles_m, tiles_n, k_blocks, group, ldc, c);
        common.initialize(1);
        let state = Small::<N, HALF, BOX, STAGE> {
            common,
            a: ARing::attach(smem.add(A0_OFFSET)),
            b: SharedTileRing::attach(smem.add(SMALL_B_OFFSET)),
            accumulator: TmemTile::from_raw(alloc_cluster(
                smem.add(tmem_offset(RINGS_END)).cast(),
                2 * N as u32,
            )),
            a_map,
            b_map,
        };

        if common.warp_id == TMA_WARP && common.lane == 0 {
            state.producer::<ABLATE>();
        } else if common.warp_id == MMA_WARP && common.lane == 0 {
            state.multiply::<ABLATE, FOLD>();
        } else if common.warp_id < EPILOGUE_WARPS {
            state.epilogue::<ABLATE>();
        }

        common.retire();
        dealloc_cluster(state.accumulator.raw(), 2 * N as u32);
    }
}

/// The `[512, 256]` device body, per [`small_body`].
///
/// # Safety
///
/// As [`kernels::gemm_sol_m512`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn large_body<const BOX: usize, const ABLATE: u8, const FOLD: bool>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    group: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(
            smem,
            LARGE_RINGS_END,
            tiles_m,
            tiles_n,
            k_blocks,
            group,
            ldc,
            c,
        );
        common.initialize(2);
        let state = Large::<BOX> {
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
            state.producer::<ABLATE>();
        } else if common.warp_id == MMA_WARP && common.lane == 0 {
            state.multiply::<ABLATE, FOLD>();
        } else if common.warp_id < EPILOGUE_WARPS {
            state.epilogue::<ABLATE>();
        }

        common.retire();
        dealloc_cluster(state.accumulator.raw(), 2 * BLOCK_N as u32);
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// # Safety
    ///
    /// The maps and output must cover the stated tile grid and K depth. The
    /// launch must be one two-CTA cluster per M256xN256 output tile, and
    /// `b_map`'s box must be [`B_BOX`] rows tall.
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, WHOLE, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            );
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
            small_body::<NARROW_N, HALF_NARROW_N, B_BOX, NARROW_N, NARROW_RINGS_END, WHOLE, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            );
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
            large_body::<B_BOX, WHOLE, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            );
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
