/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A kittens port of cuda-oxide's canonical Blackwell `gemm_sol_final`:
//! `C = A·Bᵀ` for row-major FP16 `A [M, K]` and `B [N, K]` into packed
//! row-major BF16 `C [M, N]`, with `k` a multiple of 256.
//!
//! One launch is one two-CTA cluster per output tile and one CTA per SM, at 320
//! threads a CTA at the two 256-wide entries and 192 at the narrow one: eight
//! epilogue warps splitting the accumulator's columns (four at `[256, 128]`, see
//! [`NARROW_WARPGROUPS`]), one TMA warp, one MMA warp. The producer
//! walks `K` in 64-deep blocks through four TMA/shared stages, the MMA warp
//! issues `tcgen05.mma` at `cta_group::2` into two TMEM accumulator halves, and
//! the epilogue lifts 64-column bands out of TMEM, `stmatrix`es them into a
//! per-warp staging tile and stores whole rows of `C`. The K loop is unrolled
//! four ways over a compile-time ring stage, and CLC work stealing hands out the
//! next output tile in an L2-aware `grouped` order.
//!
//! Three entries differ only in the cluster tile they own — `[256, 128]`,
//! `[256, 256]` and `[512, 256]` — and [`select_variant`] picks between them on
//! wave arithmetic alone: the narrow tile at and below half a wave of wide ones,
//! `[256, 256]` through 4K, `[512, 256]` from 8K. `n` is a multiple of the
//! entry's own `N`, which is the one place the shape contract widens.
//!
//! **There are no dials here.** Every const this kernel was chosen by is baked
//! to what it ships at, and the arms it was chosen *against* — the phase ladder
//! (`whole`, `feed only`, `issue only`, `no drain`, `mma only`), the four drain
//! rungs, the doubling ladder, the runtime-stage control, the wide `B` box, the
//! warpgroup split and the watched build — live in `experiments/src/gemm_sol.rs`
//! and are run by `bench sol`, `bench sol-ablate` and `bench sol-watch`. That
//! crate holds the same device body with the dials still on it, so an arm stays
//! buildable and a verdict stays re-runnable; this file is what a reader should
//! be able to read top to bottom.
//!
//! [`check`] runs all three entries against an exact reference. Why the kernel
//! is shaped this way, and every measurement behind it:
//! `docs/kernels/gemm_sol.md`.

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
use kittens::mma::{commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcCursor, ClcQueue};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{
    Bf16, Element, F16, SharedCell, SharedCellRing, SharedTile, SharedTileRing, Swizzle128B,
};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};

const BLOCK_M: usize = 128;
pub const BLOCK_N: usize = 256;
pub const HALF_N: usize = BLOCK_N / 2;
const BLOCK_K: usize = 64;
const CHUNKS: usize = BLOCK_K / 16;
const STAGES: usize = 4;
/// Items the tile handoff holds at once: the depth of the `ready`
/// [`SemaphoreRing`] and of the `info` [`SharedCellRing`] behind it.
///
/// Four is the minimum the chain of back-pressure admits and not a margin. The
/// four-step derivation, and the exhaustive search in
/// [`kittens::sync::handoff::depth_needed`] that checks it, are in
/// `docs/kernels/gemm_sol.md`.
const ITEMS: usize = 4;
/// Output tiles [`Small`] keeps in flight, and step 3 of [`ITEMS`]' derivation.
///
/// It is also the distance the release is deferred by, which is the whole of
/// oxide-train #86's ping-pong: `Small::multiply` waits `empty(sequence − 2)`, so
/// the accumulator an item takes was drained *two* items ago and item `i − 1`'s
/// K walk has run since. [`Large`] cannot have that — its two slots are one
/// item's two M-halves and its allocation is already the SM's whole 512 columns
/// — and `docs/kernels/gemm_sol.md` has what follows from the asymmetry.
const ACCUMULATORS: usize = 2;
const _: () = assert!(ITEMS >= ACCUMULATORS + 2, "steps 3 and 4 above");
const NARROW_N: usize = BLOCK_N / 2;
const HALF_NARROW_N: usize = NARROW_N / 2;
/// Tensor-memory columns a body allocates: [`ACCUMULATORS`] stages of an
/// `[BLOCK_M, N]` tile, in one `alloc_cluster`.
///
/// [`small_body`] takes this as a parameter rather than computing it from its
/// own `N`, for the reason `HALF_N` is a parameter beside `BLOCK_N`: a const
/// argument cannot be arithmetic on a const parameter without
/// `generic_const_exprs`, and `alloc_cluster` takes its count as one since
/// #128. The assert inside that body is what keeps the two in step. The
/// `[512, 256]` entry is one configuration and reads this directly.
const ACCUM_COLUMNS: usize = ACCUMULATORS * BLOCK_N;
const NARROW_ACCUM_COLUMNS: usize = ACCUMULATORS * NARROW_N;
/// TMEM rows one warp owns, and therefore how many epilogue warps **one
/// warpgroup** can have: `tcgen05.ld` reaches only the 32 tensor-memory lanes of
/// the issuing warp's own sub-partition, indexed within its warpgroup, so four
/// warps tile a `[128, N]` accumulator's rows exactly and a fifth has none left.
const EPILOGUE_ROWS: u32 = (BLOCK_M / 32) as u32;
/// Warpgroups the two 256-wide entries run their epilogue across, and it changed
/// in #197. Warp `w` drains rows `32 · (w % EPILOGUE_ROWS)` of column half
/// `w / EPILOGUE_ROWS`, so a second warpgroup splits the accumulator's
/// **columns** and not its rows.
///
/// The split had been measured at one warpgroup through a `.local` frame the
/// library no longer emits: `store_tile_x4` homed each drained band to a depot
/// until the rolled walks were marked (#166, landed in #180/#181/#184), and the
/// band *is* the frame — a `RegTile<32, 64>` is 64 f32 is 256 B. Re-run against
/// the same `bench sol-ablate` table 6 that recorded the loss, the sign is the
/// other way at both entries and both passes, and the drain it was blamed on is
/// now the part that gets faster: 4.72 µs to 3.05 µs an item at `[512, 256]`.
///
/// The hardware argument that made the loss plausible is unchanged and was never
/// the whole story — warps 0 and 4 do own the same 32 tensor-memory lanes, so the
/// column split puts two requesters on each of four sub-partitions rather than
/// spreading them over eight. That ceiling is real; it just sat above the depot
/// rather than below it.
pub const WARPGROUPS: u32 = 2;
/// The narrow entry keeps one warpgroup, and the reason is measurement rather
/// than mechanism: `bench sol-ablate`'s warpgroup table has arms at `[256, 256]`
/// and `[512, 256]` only, so `[256, 128]` has no A/B of its own. The split is
/// legal there — the asserts below cover `NARROW_N` at both widths, and its
/// per-warp band would be exactly [`BAND_N`] — and shipping it would be shipping
/// a configuration nothing in this tree has timed.
pub const NARROW_WARPGROUPS: u32 = 1;

const fn epilogue_warps(groups: u32) -> u32 {
    EPILOGUE_ROWS * groups
}
const fn tma_warp(groups: u32) -> u32 {
    epilogue_warps(groups)
}
const fn mma_warp(groups: u32) -> u32 {
    tma_warp(groups) + 1
}
/// Threads a launch takes at `groups` epilogue warpgroups: 192 at one, 320 at two.
pub const fn threads(groups: u32) -> u32 {
    (mma_warp(groups) + 1) * 32
}
/// The two 256-wide entries' block. It is no longer every entry's, which is why
/// the launcher asks [`Variant::threads`] instead of reading this: the narrow
/// entry is at [`NARROW_WARPGROUPS`] and takes 192 where these take 320.
pub const THREADS: u32 = threads(WARPGROUPS);
const RANKS: u32 = 2;
const LEADER: u32 = 0;
const PAIR: u16 = 0b11;
/// Columns of the accumulator a warp lifts into registers in one pass, and two
/// `tcgen05.ld.16x256b.x8` issues: [`Common::drain`] puts both in flight before
/// the single `tcgen05.wait::ld`, which is #117's finding — the LDTM half of a
/// staged epilogue is its wait and not its issue.
pub const BAND_N: usize = 64;
/// `.x8` issues a [`BAND_N`] band takes: one per 16 rows per 64 columns, over a
/// 32-row band.
const BAND_ISSUES: usize = 2;
/// The TMA box `B` arrives in, and therefore the step the load loop takes through
/// the half-panel a rank owns.
///
/// It is 64 because all three entries share one tensor map and the narrow entry's
/// half-panel *is* 64 rows. The wide entries therefore pay two instructions where
/// one would do, which measures as nothing: the feed's ceiling is bytes, and the
/// `wide B` arm that asked is in `experiments/`.
pub const B_BOX: usize = 64;
/// A warp's columns of the `[512, 256]` entry's staging buffer, before the
/// warpgroup split halves them.
pub const LARGE_STAGE_N: usize = HALF_N;

const _: () = {
    assert!(THREADS == 320);
    assert!(threads(NARROW_WARPGROUPS) == 192);
    assert!(threads(WARPGROUPS) == 320);
    assert!(
        epilogue_warps(NARROW_WARPGROUPS) as usize * BLOCK_N
            == epilogue_warps(WARPGROUPS) as usize * (BLOCK_N / 2)
    );
    assert!(
        epilogue_warps(NARROW_WARPGROUPS) as usize * NARROW_N
            == epilogue_warps(WARPGROUPS) as usize * (NARROW_N / 2)
    );
    assert!(
        epilogue_warps(NARROW_WARPGROUPS) as usize * LARGE_STAGE_N
            == epilogue_warps(WARPGROUPS) as usize * (LARGE_STAGE_N / 2)
    );
    assert!(BLOCK_N.is_multiple_of(2 * BAND_N));
    assert!(NARROW_N.is_multiple_of(2 * BAND_N));
    assert!(LARGE_STAGE_N.is_multiple_of(2 * BAND_N));
    assert!(CHUNKS == 4);
    assert!(HALF_N.is_multiple_of(B_BOX));
    assert!(HALF_NARROW_N.is_multiple_of(B_BOX));
};

pub type ATile = SharedTile<F16, BLOCK_M, BLOCK_K, Swizzle128B>;

pub type BPanel = SharedTile<F16, B_BOX, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<F16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<F16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type NarrowBRing = SharedTileRing<F16, HALF_NARROW_N, BLOCK_K, Swizzle128B, STAGES>;
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
    align_up(ready_offset(rings_end) + ITEMS * 8, 4)
}
const fn info_offset(rings_end: usize) -> usize {
    align_up(
        tmem_offset(rings_end) + 4,
        SharedCellRing::<TileInfo, ITEMS>::ALIGNMENT,
    )
}
const fn queue_offset(rings_end: usize) -> usize {
    align_up(
        info_offset(rings_end) + SharedCellRing::<TileInfo, ITEMS>::BYTES,
        ClcQueue::ALIGNMENT,
    )
}
const fn stage_offset(rings_end: usize) -> usize {
    align_up(queue_offset(rings_end) + ClcQueue::BYTES, 128)
}
/// `stage_n` is the **entry's** staging width, not a warp's: at two epilogue
/// warpgroups twice as many warps stage half as many columns each, so this total
/// is the same either way.
const fn shared_plan(rings_end: usize, stage_n: usize) -> usize {
    stage_offset(rings_end) + EPILOGUE_ROWS as usize * 32 * stage_n * Bf16::BYTES
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
    full: SemaphoreRing<ACCUMULATORS>,
    empty: SemaphoreRing<ACCUMULATORS>,
    ready: SemaphoreRing<ITEMS>,
    info: SharedCellRing<TileInfo, ITEMS>,
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
                ready: SemaphoreRing::attach(smem.add(ready_offset(rings_end)).cast::<Barrier>()),
                info: SharedCellRing::attach(smem.add(info_offset(rings_end))),
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

    /// Arm every barrier, and the two counts a caller has to get right are the
    /// two it passes: `free_arrivals` is MMA commits a stage (one per `A` ring,
    /// so 1 at the `[256, N]` entries and 2 at `[512, 256]`), `empty_arrivals`
    /// is [`Common::release_accumulator`]'s signallers — one per epilogue warp
    /// per rank.
    #[inline(always)]
    unsafe fn initialize(self, free_arrivals: u32, empty_arrivals: u32) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                self.load.init_all(1);
                self.free.init_all(free_arrivals);
                self.full.init_all(1);
                self.empty.init_all(empty_arrivals);
                self.ready.init_all(1);
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
                self.ready.inval_all();
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
    fn handoff(self, index: u32) -> (Semaphore, u32, SharedCell<TileInfo>) {
        let depth = ITEMS as u32;
        (
            self.ready.sem(index % depth),
            (index / depth) & 1,
            self.info.cell(index % depth),
        )
    }

    #[inline(always)]
    unsafe fn publish(self, index: u32, row: u32, column: u32, has_work: bool) {
        unsafe {
            let (ready, _, info) = self.handoff(index);
            info.write(TileInfo {
                row,
                column,
                has_work: has_work as u32,
            });
            ready.arrive();
        }
    }

    #[inline(always)]
    unsafe fn next_info(self, sequence: u32) -> TileInfo {
        unsafe {
            let (ready, parity, info) = self.handoff(sequence);
            ready.wait(parity);
            info.read()
        }
    }

    /// Hand an accumulator slot back to the MMA warp: **one arrival a warp**,
    /// from its lane 0, on the leader rank's copy of the barrier.
    ///
    /// The count [`Common::initialize`] arms `empty` with is this call's
    /// arithmetic and nothing else — `RANKS * epilogue_warps(GROUPS)`, so 16 at
    /// the 256-wide entries and 8 at the narrow one. A lane guard that the count
    /// does not follow is a kernel that never launches a second item.
    ///
    /// What makes the guard legal is the `warp::sync_mask` [`Common::drain`]
    /// puts between the last band's load and this call: `tcgen05.wait::ld`
    /// retires every load the *warp* has outstanding, so by the time lane 0 is
    /// through that sync no lane of it still has the accumulator in flight.
    /// Nothing below that load reads tensor memory, which is why this is also
    /// where the call belongs.
    #[inline(always)]
    unsafe fn release_accumulator(self, slot: u32) {
        unsafe {
            if self.lane != 0 {
                return;
            }
            let empty = self.empty.sem(slot);
            if self.rank == LEADER {
                empty.arrive();
            } else {
                empty.at_rank(LEADER).arrive();
            }
        }
    }

    #[inline(always)]
    unsafe fn staging<const STAGE_N: usize>(self) -> SharedTile<Bf16, 32, STAGE_N, Swizzle128B> {
        unsafe {
            SharedTile::from_raw(
                self.stage_base.add(
                    self.warp_id as usize * SharedTile::<Bf16, 32, STAGE_N, Swizzle128B>::BYTES,
                ),
            )
        }
    }

    #[inline(always)]
    fn row_block(self) -> u32 {
        self.warp_id % EPILOGUE_ROWS
    }

    #[inline(always)]
    fn column_group(self) -> u32 {
        self.warp_id / EPILOGUE_ROWS
    }

    #[inline(always)]
    fn drain_row(self, tile_row: u32) -> u32 {
        tile_row + self.rank * BLOCK_M as u32 + self.row_block() * 32
    }

    /// The columns of the accumulator this warp owns: `GROUPS` warpgroups split
    /// the tile's `N` between them, and every warp of a group drains the same
    /// span at a different 32 rows.
    #[inline(always)]
    fn drain_columns<const N: usize, const GROUPS: u32>(self) -> (u32, u32) {
        let span = N as u32 / GROUPS;
        (self.column_group() * span, span)
    }

    /// TMEM → registers → shared → `C`, a [`BAND_N`] band at a time, with both of
    /// the band's `.x8` issues in flight before one `tcgen05.wait::ld`.
    ///
    /// `slot` is released **the instant the last band's `tcgen05.wait::ld`
    /// retires**, which is the instant the accumulator stops being read, and not
    /// at the end of the drain. Everything below that load — the last band's
    /// `stmatrix` pass and the `ld.shared`/`st.global` pass under it — leaves the
    /// next item's critical path, and at `[512, 256]` that path is the whole
    /// story: see `docs/kernels/gemm_sol.md`.
    #[inline(always)]
    unsafe fn drain<const N: usize, const STAGE_N: usize, const GROUPS: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
        slot: u32,
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(N.is_multiple_of(STAGE_N * GROUPS as usize));
            };
            let stage = self.staging::<STAGE_N>();
            let row = self.drain_row(tile_row);
            let (base, span) = self.drain_columns::<N, GROUPS>();
            let last_band = span - BAND_N as u32;
            let mut column = 0u32;
            while column < span {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let band: StageBand = accumulator.tile_x8_batched::<32, BAND_N, BAND_ISSUES>(
                        32 * self.row_block(),
                        base + column + band_column,
                    );
                    if column + band_column == last_band {
                        warp::sync_mask(u32::MAX);
                        self.release_accumulator(slot);
                    }
                    store_tile_x4(stage.chunk_writer(), 0, band_column, self.lane, band);
                    band_column += BAND_N as u32;
                }
                warp::sync_mask(u32::MAX);
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    row,
                    tile_column + base + column,
                    self.lane,
                    stage,
                );
                warp::sync_mask(u32::MAX);
                column += STAGE_N as u32;
            }
        }
    }
}

/// The one-accumulator entry, generic in the cluster tile's width: `N` columns of
/// `C`, `HALF` the half-panel of `B` a rank loads, and `STAGE` **a warp's**
/// columns of the shared epilogue staging buffer.
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
            const {
                assert!(N == 2 * HALF, "a rank loads exactly half the panel");
                assert!(HALF.is_multiple_of(B_BOX), "the panel is whole boxes");
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
                    common.publish(sequence, tile_m, tile_n, true);
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
                    common.publish(sequence, 0, 0, false);
                    break;
                };
                raw_item = next;
            }
        }
    }

    /// One K block at a **compile-time** stage. Every entry's contract is
    /// `k % (STAGES * BLOCK_K) == 0`, so `global_k % STAGES == k % STAGES` and the
    /// four positions of the unroll are always stages 0, 1, 2, 3 in that order;
    /// only the phase parity moves with `global_k`, once a turn instead of four
    /// times.
    ///
    /// It is worth +2.2% to +4.3% at 4096³ over indexing off `global_k`, and the
    /// arm that says so — `runtime stage`, which computes the same `C` — is in
    /// `experiments/`.
    #[inline(always)]
    unsafe fn multiply_at<const SLOT: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        parity: u32,
        accumulate: bool,
    ) {
        unsafe {
            let common = self.common;
            if common.rank == LEADER {
                common.load.sem(SLOT).wait(parity);
                mma_walk_cg2::<F16, CHUNKS, _, _>(
                    accumulator,
                    self.a.tile(SLOT).k_walk(),
                    self.b.tile(SLOT).k_walk(),
                    accumulate,
                );
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
            }
        }
    }

    /// The K walk, [`STAGES`] blocks a turn.
    #[inline(always)]
    unsafe fn walk_k(self, accumulator: TmemTile<BLOCK_M, N>, sequence: u32) {
        unsafe {
            let common = self.common;
            let turns = common.k_blocks / STAGES as u32;
            let cycle = sequence * turns;
            let mut turn = 0u32;
            while turn < turns {
                let parity = (cycle + turn) & 1;
                self.multiply_at::<0>(accumulator, parity, turn > 0);
                self.multiply_at::<1>(accumulator, parity, true);
                self.multiply_at::<2>(accumulator, parity, true);
                self.multiply_at::<3>(accumulator, parity, true);
                turn += 1;
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
                if common.rank == LEADER && sequence >= ACCUMULATORS as u32 {
                    common.empty.wait(sequence - ACCUMULATORS as u32);
                }

                self.walk_k(accumulator, sequence);
                if common.rank == LEADER {
                    commit_multicast_cg2(common.full.sem(sequence), PAIR);
                }
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue<const GROUPS: u32>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                let info = common.next_info(sequence);
                if info.has_work == 0 {
                    break;
                }
                let (row, column) = (info.row * (2 * BLOCK_M) as u32, info.column * N as u32);
                common.full.wait(sequence);
                common.drain::<N, STAGE, GROUPS>(
                    Common::accumulator(self.accumulator, sequence),
                    row,
                    column,
                    sequence,
                );
                sequence += 1;
            }
        }
    }
}

/// The two-accumulator entry: two `A` rings against one shared `B` half-panel, so
/// a K block is twice the MMA against the same one `load` round trip.
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
                    common.publish(sequence, macro_m, tile_n, true);
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
                    common.publish(sequence, 0, 0, false);
                    break;
                };
                raw_item = next;
            }
        }
    }

    /// Both accumulator halves at one compile-time stage, per
    /// [`Small::multiply_at`]. The second half's `empty` is awaited between the
    /// two MMA issues of the item's first stage, which is the one point in the
    /// walk where the first half is already issued and the second is not.
    #[inline(always)]
    unsafe fn multiply_at<const SLOT: u32>(
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
                common.load.sem(SLOT).wait(parity);
                mma_walk_cg2::<F16, CHUNKS, _, _>(
                    self.accumulator,
                    self.a0.tile(SLOT).k_walk(),
                    self.b.tile(SLOT).k_walk(),
                    accumulate,
                );
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
                if last {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && first {
                    common.empty.sem(1).wait((sequence - 1) & 1);
                }
                mma_walk_cg2::<F16, CHUNKS, _, _>(
                    self.accumulator.columns_right(BLOCK_N as u32),
                    self.a1.tile(SLOT).k_walk(),
                    self.b.tile(SLOT).k_walk(),
                    accumulate,
                );
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
                if last {
                    commit_multicast_cg2(common.full.sem(1), PAIR);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn walk_k(self, sequence: u32) {
        unsafe {
            let common = self.common;
            let turns = common.k_blocks / STAGES as u32;
            let cycle = sequence * turns;
            let mut turn = 0u32;
            while turn < turns {
                let parity = (cycle + turn) & 1;
                let last = turn + 1 == turns;
                self.multiply_at::<0>(sequence, parity, turn > 0, turn == 0, false);
                self.multiply_at::<1>(sequence, parity, true, false, false);
                self.multiply_at::<2>(sequence, parity, true, false, false);
                self.multiply_at::<3>(sequence, parity, true, false, last);
                turn += 1;
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
                self.walk_k(sequence);
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
                    common.drain::<BLOCK_N, { LARGE_STAGE_N / WARPGROUPS as usize }, WARPGROUPS>(
                        self.accumulator.columns_right(half * BLOCK_N as u32),
                        info.row * 512 + half * 256,
                        info.column * BLOCK_N as u32,
                        half,
                    );
                    half += 1;
                }
                sequence += 1;
            }
        }
    }
}

/// The width-generic device body both `[256, N]` entries instantiate.
///
/// `ACCUM_COLUMNS` is [`ACCUM_COLUMNS`] at the entry's own width, and a
/// parameter for the reason `HALF` is one — see that constant.
///
/// # Safety
///
/// As [`kernels::gemm_sol_m256`], at the entry's own width.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn small_body<
    const N: usize,
    const ACCUM_COLUMNS: usize,
    const HALF: usize,
    const STAGE: usize,
    const RINGS_END: usize,
    const GROUPS: u32,
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
    const { assert!(ACCUM_COLUMNS == ACCUMULATORS * N) };
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(smem, RINGS_END, tiles_m, tiles_n, k_blocks, group, ldc, c);
        common.initialize(1, RANKS * epilogue_warps(GROUPS));
        let state = Small::<N, HALF, STAGE> {
            common,
            a: ARing::attach(smem.add(A0_OFFSET)),
            b: SharedTileRing::attach(smem.add(SMALL_B_OFFSET)),
            accumulator: TmemTile::from_raw(alloc_cluster::<ACCUM_COLUMNS>(
                smem.add(tmem_offset(RINGS_END)).cast(),
            )),
            a_map,
            b_map,
        };

        if common.warp_id == tma_warp(GROUPS) && common.lane == 0 {
            state.producer();
        } else if common.warp_id == mma_warp(GROUPS) && common.lane == 0 {
            state.multiply();
        } else if common.warp_id < epilogue_warps(GROUPS) {
            state.epilogue::<GROUPS>();
        }

        common.retire();
        dealloc_cluster::<ACCUM_COLUMNS>(state.accumulator.raw());
    }
}

/// The `[512, 256]` device body, per [`small_body`].
///
/// # Safety
///
/// As [`kernels::gemm_sol_m512`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn large_body(
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
        common.initialize(2, RANKS * epilogue_warps(WARPGROUPS));
        let state = Large {
            common,
            a0: ARing::attach(smem.add(A0_OFFSET)),
            a1: ARing::attach(smem.add(LARGE_A1_OFFSET)),
            b: BRing::attach(smem.add(LARGE_B_OFFSET)),
            accumulator: Accumulator::from_raw(alloc_cluster::<ACCUM_COLUMNS>(
                smem.add(tmem_offset(LARGE_RINGS_END)).cast(),
            )),
            a_map,
            b_map,
        };

        if common.warp_id == tma_warp(WARPGROUPS) && common.lane == 0 {
            state.producer();
        } else if common.warp_id == mma_warp(WARPGROUPS) && common.lane == 0 {
            state.multiply();
        } else if common.warp_id < epilogue_warps(WARPGROUPS) {
            state.epilogue();
        }

        common.retire();
        dealloc_cluster::<ACCUM_COLUMNS>(state.accumulator.raw());
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
        block = (320, 1, 1),
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                { BLOCK_N / WARPGROUPS as usize },
                SMALL_RINGS_END,
                WARPGROUPS,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c);
        }
    }

    /// The same entry at half the width.
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
            small_body::<
                NARROW_N,
                NARROW_ACCUM_COLUMNS,
                HALF_NARROW_N,
                NARROW_N,
                NARROW_RINGS_END,
                NARROW_WARPGROUPS,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c);
        }
    }

    /// # Safety
    ///
    /// As [`gemm_sol_m256`], with an even M256 tile count.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (320, 1, 1),
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
            large_body(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c);
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

    /// The block this entry's `#[launch_contract]` states, which is no longer one
    /// number for all three: the 256-wide entries run [`WARPGROUPS`] epilogue
    /// warpgroups and the narrow one runs [`NARROW_WARPGROUPS`]. It sits beside
    /// [`Variant::shared_bytes`] because it is the same kind of fact — what the
    /// host must pass to launch this entry and not another.
    pub const fn threads(self) -> u32 {
        match self {
            Self::M256xN128 => threads(NARROW_WARPGROUPS),
            Self::M256xN256 | Self::M512xN256 => threads(WARPGROUPS),
        }
    }
}

/// CTAs of one entry an SM holds, and the residency `modal_app.py`'s occupancy
/// gate reads this file for. It is fixed by the shared plan rather than by
/// registers — the smallest of the three is 131 328 B of the 233 472 an SM
/// divides and the two that ship are 196 864 and 229 632 — so no register count
/// can move it *up*. It can be moved to zero: at [`THREADS`]' 320 a thread may
/// have 168 registers before an SM cannot place one CTA at all, which is a
/// launch that fails rather than a launch that is slow, and is what the gate
/// watches now that the two 256-wide entries run ten warps instead of six.
const CTAS_PER_SM: u32 = 1;

/// Clusters this device holds at once: 148 SMs, one CTA per SM because every
/// shared plan here declares more than half of the 233472 B an SM divides, and
/// two CTAs to a cluster because a `cta_group::2` MMA needs its pair resident.
const RESIDENT_CLUSTERS: usize = 74;
const _: () = assert!(RESIDENT_CLUSTERS == (148 * CTAS_PER_SM / RANKS) as usize);

/// The entry a shape gets; all three branches are wave arithmetic. A launch is one
/// cluster per output tile and takes `ceil(tiles / RESIDENT_CLUSTERS)` waves of
/// them, so `tiles / (waves * RESIDENT_CLUSTERS)` is the fraction that is not
/// idling — and halving the tile's `N` doubles the tile count, which raises that
/// fraction only while both counts fit the same number of waves.
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

/// The N band [`pipeline::grouped`] walks before it steps in `M`.
pub const fn default_group(m: usize) -> u32 {
    if m / 256 <= 16 { 2 } else { 8 }
}

/// Blocks a shape's launch asks for: one two-CTA cluster per output tile of the
/// entry that owns it.
fn grid(m: usize, n: usize, variant: Variant) -> u32 {
    RANKS * (m / variant.m_tile() * (n / variant.n_tile())) as u32
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

/// Stage the operands, launch `variant` once and compare every output with `==`.
fn run(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    variant: Variant,
) -> Result<String, Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    validate_shape(m, n, k, variant)?;
    let stream = context.default_stream();
    let module = unsafe { kernels::load(context)? };
    let a = DeviceBuffer::from_host(&stream, &stage_f16(m, k, a_value))?;
    let b = DeviceBuffer::from_host(&stream, &stage_f16(n, k, b_value))?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let config = LaunchConfig1D::new(
        grid(m, n, variant),
        variant.threads(),
        variant.shared_bytes() as u32,
    );
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / variant.n_tile()) as u32;
    let k_blocks = (k / BLOCK_K) as u32;
    let group = default_group(m);

    match variant {
        Variant::M256xN128 => {
            let prepared = module.prepare_gemm_sol_m256_n128(config)?;
            unsafe {
                module.gemm_sol_m256_n128(
                    &stream, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group, n as u32,
                    &mut c,
                )?
            };
        }
        Variant::M256xN256 => {
            let prepared = module.prepare_gemm_sol_m256(config)?;
            unsafe {
                module.gemm_sol_m256(
                    &stream, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group, n as u32,
                    &mut c,
                )?
            };
        }
        Variant::M512xN256 => {
            let prepared = module.prepare_gemm_sol_m512(config)?;
            unsafe {
                module.gemm_sol_m512(
                    &stream, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group, n as u32,
                    &mut c,
                )?
            };
        }
    }
    stream.synchronize()?;

    let worst = check_output(&c.to_host_vec(&stream)?, m, n, k)?;
    Ok(format!(
        "{} {m}x{n}x{k} exact over {} BF16 outputs, worst |rel| {worst:.2e}",
        variant.name(),
        m * n,
    ))
}

/// The shallowest `k` the contract admits, at a grid deep enough that a cluster
/// has to take a second item: each of these is above [`RESIDENT_CLUSTERS`]
/// clusters and at `k_blocks == STAGES`, which is the shape [`ITEMS`]'
/// derivation is tight at.
const SHALLOW_K_GATE: [(Variant, usize, usize); 3] = [
    (Variant::M256xN128, 2048, 2048),
    (Variant::M256xN256, 2560, 2048),
    (Variant::M512xN256, 5120, 2048),
];

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for variant in Variant::ALL {
        notes.push(run(context, 1024, 1024, 512, variant)?);
    }
    for (variant, m, n) in SHALLOW_K_GATE {
        notes.push(run(context, m, n, STAGES * BLOCK_K, variant)?);
    }
    Ok(notes.join("; "))
}
