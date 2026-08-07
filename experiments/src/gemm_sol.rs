/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The lab copy of `examples/src/gemm_sol.rs`, and the whole dial surface.
//!
//! `examples/` ships one kernel at one configuration — [`WHOLE`],
//! [`SHIPPED_DRAIN`], [`WATCH_OFF`], [`SHIPPED_GROUPS`] — with every const below
//! baked to that value and every other arm gone. This file is the same device
//! body with the dials still on it, so [`crate::sol_ablate`] and
//! [`crate::sol_watch`] can instantiate the arms the shipped configuration was
//! chosen *by*. It is the same relationship [`crate::gemm`] has to
//! `examples/src/gemm.rs`, for the same reason: an arm nobody can build is a
//! verdict nobody can re-run, and this tree re-runs them (#166, #197).
//!
//! **Both crates emit `gemm_sol_m256`, `gemm_sol_m256_n128` and
//! `gemm_sol_m512`**, and `regcount`'s opcode census carries a row for each copy.
//! Two identical rows are the claim that baking the dials in moved no
//! instruction; a difference between them is a finding.
//!
//! A kittens port of cuda-oxide's canonical Blackwell `gemm_sol_final`:
//! `C = A·Bᵀ` for row-major FP16 `A [M, K]` and `B [N, K]` into packed
//! row-major BF16 `C [M, N]`, with `k` a multiple of 256.
//!
//! One launch is one two-CTA cluster per output tile and one CTA per SM, at 320
//! threads a CTA at the two 256-wide entries and 192 at the narrow one: eight
//! epilogue warps splitting the accumulator's columns (four at `[256, 128]`, see
//! [`NARROW_GROUPS`]), one TMA warp, one MMA warp. The producer
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
//! [`check`] runs all three against an exact reference; `bench sol` and
//! `bench sol-ablate` time them and take them apart, through the `ABLATE`,
//! `DRAIN` and `WATCH` dials below.
//!
//! Why the kernel is shaped this way, and every measurement behind it:
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
use kittens::ldst::{store_packed_x4, store_tile_x4};
use kittens::mma::{commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcCursor, ClcQueue, Harvest};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{
    Bf16, Element, F16, SharedCell, SharedCellRing, SharedTile, SharedTileRing, Swizzle128B,
};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::watchdog::{self, ReadBack};

/// The whole kernel: the arm every shipped entry passes.
pub const WHOLE: u8 = 0;
/// TMA and every barrier, with the MMA and the drain removed.
pub const FEED_ONLY: u8 = 1;
/// The MMA and every barrier, with the loads removed — the producer arrives on
/// `load` instead of filling it, so no global traffic occurs at all.
pub const ISSUE_ONLY: u8 = 2;
/// The whole kernel except the drain: the accumulator fills and is released
/// without being read.
pub const NO_DRAIN: u8 = 3;
/// [`ISSUE_ONLY`] without the MMA warp's `load.wait`, so the K loop issues
/// `tcgen05.mma` with nothing in front of it.
pub const MMA_ONLY: u8 = 4;
/// The drain's **global half** run a second time per band, aimed at the
/// cluster's own first output tile so the extra bytes stay in L2. It computes a
/// wrong `C` and is on no correctness gate, as every doubling probe here is.
pub const TWICE_GLOBAL: u8 = 5;
/// [`TWICE_GLOBAL`] with the `cvt` + `stmatrix` pass doubled as well.
pub const TWICE_SHARED: u8 = 6;
/// [`TWICE_SHARED`] with the LDTM doubled too — the whole drain twice.
pub const TWICE_ALL: u8 = 7;

/// Every `ABLATE` value, with the name [`arm_name`] prints for it.
pub const ABLATE_ARMS: [(&str, u8); 8] = [
    ("whole", WHOLE),
    ("feed only", FEED_ONLY),
    ("issue only", ISSUE_ONLY),
    ("no drain", NO_DRAIN),
    ("mma only", MMA_ONLY),
    ("twice global", TWICE_GLOBAL),
    ("twice shared", TWICE_SHARED),
    ("twice all", TWICE_ALL),
];

/// Every `DRAIN` value, per [`ABLATE_ARMS`].
pub const DRAIN_RUNGS: [(&str, u8); 5] = [
    ("per issue", DRAIN_PER_ISSUE),
    ("paired", DRAIN_PAIRED),
    ("wide", DRAIN_WIDE),
    ("pack16", DRAIN_PACK16),
    ("nocvt", DRAIN_NOCVT),
];

/// Every `WATCH` value, per [`ABLATE_ARMS`].
pub const WATCH_RUNGS: [(&str, u8); 3] = [
    ("off", WATCH_OFF),
    ("watched", WATCH_DEEP),
    ("watched one deep", WATCH_ONE_DEEP),
];

const fn dial_is_a_permutation<const N: usize>(dial: [(&str, u8); N]) -> bool {
    let mut seen = [false; N];
    let mut arm = 0usize;
    while arm < N {
        let value = dial[arm].1 as usize;
        assert!(
            value < N,
            "a dial value is outside the dense range: the arm is missing from the list"
        );
        assert!(
            !seen[value],
            "two arms of one dial share a value, which silently remaps one onto the other"
        );
        seen[value] = true;
        arm += 1;
    }
    true
}

const _: () = {
    assert!(dial_is_a_permutation(ABLATE_ARMS));
    assert!(dial_is_a_permutation(DRAIN_RUNGS));
    assert!(dial_is_a_permutation(WATCH_RUNGS));
};

pub fn arm_name(dial: &[(&'static str, u8)], value: u8) -> &'static str {
    match dial.iter().find(|(_, listed)| *listed == value) {
        Some((name, _)) => name,
        None => "UNLISTED DIAL VALUE",
    }
}

const fn loads(ablate: u8) -> bool {
    ablate != ISSUE_ONLY && ablate != MMA_ONLY
}
const fn multiplies(ablate: u8) -> bool {
    ablate != FEED_ONLY
}
const fn drains(ablate: u8) -> bool {
    ablate == WHOLE || twice(ablate) > 0
}
const fn waits_on_load(ablate: u8) -> bool {
    ablate != MMA_ONLY
}
/// How much of the drain the doubling ladder repeats: 0 none, 1 the global
/// half, 2 the `cvt` + `stmatrix` half too, 3 the whole of it including LDTM.
const fn twice(ablate: u8) -> u8 {
    if ablate >= TWICE_GLOBAL && ablate <= TWICE_ALL {
        ablate - TWICE_GLOBAL + 1
    } else {
        0
    }
}

/// 64-column bands with a wait per LDTM issue.
pub const DRAIN_PER_ISSUE: u8 = 0;
/// A 64-column band with **both** of its LDTM issues in flight before one wait.
pub const DRAIN_PAIRED: u8 = 1;
/// A 128-column band, all four issues in flight before one wait, at 128 f32 a
/// thread live.
pub const DRAIN_WIDE: u8 = 2;
/// `tcgen05.ld.16x256b.x8.pack::16b` in place of the `.x8` load and **no `cvt`**.
///
/// **It faults on the device and nothing launches it** — `.pack::16b` reads the
/// segment as 16-bit-typed and does not address an fp32 allocation. Its SASS
/// census is the finding; [`DRAIN_NOCVT`] is the oracle that can be timed.
pub const DRAIN_PACK16: u8 = 3;
/// The shipped drain with the `cvt` pass removed and nothing else, so
/// `paired − nocvt` is what the convert costs. **It computes a wrong `C` by
/// construction** and is on no correctness gate.
pub const DRAIN_NOCVT: u8 = 4;
pub const SHIPPED_DRAIN: u8 = DRAIN_PAIRED;

/// No watching: the kernel is byte for byte the shipped one. Above this every
/// spin becomes [`kittens::sync::Semaphore::wait_before`] and every warp writes a
/// four-word mark past the end of `C`, so a launch that stops making progress
/// terminates carrying where all six of its warps were.
pub const WATCH_OFF: u8 = 0;
/// Deadlines and marks on the shipped [`ITEMS`]-deep item handoff.
pub const WATCH_DEEP: u8 = 1;
/// [`WATCH_DEEP`] on a **one-deep** item handoff, which reproduces under the
/// instrument the defect [`ITEMS`] fixes. It differs from [`WATCH_DEEP`] in the
/// handoff's depth and in nothing else.
pub const WATCH_ONE_DEEP: u8 = 2;

const fn watches(watch: u8) -> bool {
    watch != WATCH_OFF
}
const fn items(watch: u8) -> u32 {
    if watch == WATCH_ONE_DEEP {
        1
    } else {
        ITEMS as u32
    }
}

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
const ACCUMULATORS: usize = 2;
const _: () = assert!(ITEMS >= ACCUMULATORS + 2, "steps 3 and 4 above");
/// Ticks of the SM clock a watched spin gives a phase before it calls the
/// protocol stalled — about 0.6 s at 1.8 GHz, four orders of magnitude past the
/// ~26 µs an output tile takes at 4096³, so no arm times out on being slow.
const DEADLINE: u64 = 1 << 30;
const NARROW_N: usize = BLOCK_N / 2;
const HALF_NARROW_N: usize = NARROW_N / 2;
/// Tensor-memory columns a body allocates: [`ACCUMULATORS`] stages of an
/// `[BLOCK_M, N]` tile, in one `alloc_cluster`.
///
/// [`small_body`] takes this as a parameter rather than computing it from its
/// own `N`, for the reason `HALF_N` is a parameter beside `BLOCK_N`: a const
/// argument cannot be arithmetic on a const parameter without
/// `generic_const_exprs`, and `alloc_cluster` takes its count as one since
/// #128. The assert inside that body is what keeps the two in step.
pub const ACCUM_COLUMNS: usize = ACCUMULATORS * BLOCK_N;
const NARROW_ACCUM_COLUMNS: usize = ACCUMULATORS * NARROW_N;
/// TMEM rows one warp owns, and therefore how many epilogue warps **one
/// warpgroup** can have: `tcgen05.ld` reaches only the 32 tensor-memory lanes of
/// the issuing warp's own sub-partition, indexed within its warpgroup, so four
/// warps tile a `[128, N]` accumulator's rows exactly and a fifth has none left.
const EPILOGUE_ROWS: u32 = (BLOCK_M / 32) as u32;
/// Warpgroups the epilogue runs across. Warp `w` drains rows
/// `32 · (w % EPILOGUE_ROWS)` of column half `w / EPILOGUE_ROWS`, so a second
/// warpgroup splits the accumulator's **columns** and not its rows.
pub const ONE_WARPGROUP: u32 = 1;
/// [`ONE_WARPGROUP`]'s twin: eight epilogue warps splitting the columns. Kept as
/// a rung — it is exact, and it loses.
pub const TWO_WARPGROUPS: u32 = 2;
/// What the two 256-wide entries ship, and it changed in #197.
///
/// The split was measured at one CTA per SM through a `.local` frame the library
/// no longer emits: `store_tile_x4` homed each drained band to a depot until the
/// rolled walks were marked (#166, landed in #180/#181/#184), and the band *is*
/// the frame — a `RegTile<32, 64>` is 64 f32 is 256 B. Re-run against the same
/// `bench sol-ablate` table 6 that recorded the loss, the sign is the other way
/// at both entries and both passes, and the drain it was blamed on is now the
/// part that gets faster: 4.72 µs to 3.05 µs an item at `[512, 256]`.
///
/// The hardware argument that made the loss plausible is unchanged and was never
/// the whole story — warps 0 and 4 do own the same 32 tensor-memory lanes, so the
/// column split puts two requesters on each of four sub-partitions rather than
/// spreading them over eight. That ceiling is real; it just sat above the depot
/// rather than below it.
pub const SHIPPED_GROUPS: u32 = TWO_WARPGROUPS;
/// The narrow entry keeps one warpgroup, and the reason is measurement rather
/// than mechanism: `bench sol-ablate`'s warpgroup table has arms at `[256, 256]`
/// and `[512, 256]` only, so `[256, 128]` has no A/B of its own. The split is
/// legal there — the asserts below cover `NARROW_N` at both widths, and its
/// per-warp band would be exactly [`BAND_N`] — and shipping it would be shipping
/// a configuration nothing in this tree has timed.
pub const NARROW_GROUPS: u32 = ONE_WARPGROUP;

/// Every lane of every epilogue warp arrives on `empty` when an accumulator is
/// released: 256 arrivals a slot at one warpgroup and 512 at two.
///
/// It is the scope the port carried, and it is one arrival per *thread* for a
/// fact that is per *warp* — a warp is through its drain when its last
/// `tcgen05.wait::ld` has retired, and `tcgen05.wait::ld` retires the warp's
/// loads and not the lane's.
///
/// The dial is two independent bits, because the release has two questions and
/// #218 answered only the first: bit 0 is the arrival **scope**, bit 1 is the
/// arrival's **place in the drain**.
pub const ALL_LANES: u8 = 0;
/// [`ALL_LANES`]' twin: one arrival a warp, from its lane 0, which is 16 a slot
/// at two warpgroups and 8 at one. oxide-train's `gemm_sol_final` releases this
/// way.
pub const ONE_LANE: u8 = 1;
/// [`ALL_LANES`] released at the last band's `tcgen05.wait::ld` rather than at
/// the end of the drain — the timing arm, holding the scope.
pub const ALL_LANES_EARLY: u8 = 2;
/// Both: one arrival a warp, at the last band's load. What the entries ship.
pub const ONE_LANE_EARLY: u8 = 3;
/// Bit 0 of the release dial: one arrival a warp instead of 32.
pub const fn one_lane(release: u8) -> bool {
    release & 1 != 0
}
/// Bit 1 of the release dial: the arrival sits at the last band's
/// `tcgen05.wait::ld` — the instant the accumulator stops being read — instead
/// of at the end of the drain, so the `stmatrix` and `st.global` passes below
/// that load leave the next item's critical path.
pub const fn early(release: u8) -> bool {
    release & 2 != 0
}
/// What the three entries ship. The scope bit landed in #218 (a null on its own)
/// and the timing bit in #221; tables 6 and 7 have both A/Bs.
pub const SHIPPED_RELEASE: u8 = ONE_LANE_EARLY;

const fn epilogue_warps(groups: u32) -> u32 {
    EPILOGUE_ROWS * groups
}
/// Arrivals `empty` is armed with, which is [`Common::release_accumulator`]'s
/// signaller count and nothing else. Getting it out of step with the guard there
/// is a launch that never reaches its second item.
pub const fn empty_arrivals(groups: u32, release: u8) -> u32 {
    RANKS * epilogue_warps(groups) * if one_lane(release) { 1 } else { 32 }
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
/// entry is at [`NARROW_GROUPS`] and takes 192 where these take 320.
pub const THREADS: u32 = threads(SHIPPED_GROUPS);
const RANKS: u32 = 2;
const LEADER: u32 = 0;
const PAIR: u16 = 0b11;
/// Columns of the accumulator a warp lifts into registers in one pass.
pub const BAND_N: usize = 64;
/// The TMA box `B` arrives in, and therefore the step the load loop takes through
/// the half-panel a rank owns.
///
/// It is 64 because all three entries share one tensor map and the narrow entry's
/// half-panel *is* 64 rows. The wide entries therefore pay two instructions where
/// one would do, which measures as nothing: the feed's ceiling is bytes.
pub const B_BOX: usize = 64;
/// The box the two 256-wide entries' half-panel would arrive in whole.
pub const WIDE_B_BOX: usize = HALF_N;
pub const LARGE_STAGE_N: usize = HALF_N;

const _: () = {
    assert!(THREADS == 320);
    assert!(threads(ONE_WARPGROUP) == 192);
    assert!(threads(TWO_WARPGROUPS) == 320);
    assert!(
        epilogue_warps(ONE_WARPGROUP) as usize * BLOCK_N
            == epilogue_warps(TWO_WARPGROUPS) as usize * (BLOCK_N / 2)
    );
    assert!(
        epilogue_warps(ONE_WARPGROUP) as usize * NARROW_N
            == epilogue_warps(TWO_WARPGROUPS) as usize * (NARROW_N / 2)
    );
    assert!(
        epilogue_warps(ONE_WARPGROUP) as usize * LARGE_STAGE_N
            == epilogue_warps(TWO_WARPGROUPS) as usize * (LARGE_STAGE_N / 2)
    );
    assert!(BLOCK_N.is_multiple_of(2 * BAND_N));
    assert!(NARROW_N.is_multiple_of(2 * BAND_N));
    assert!(LARGE_STAGE_N.is_multiple_of(2 * BAND_N));
    assert!(CHUNKS == 4);
    assert!(HALF_N.is_multiple_of(B_BOX));
    assert!(HALF_NARROW_N.is_multiple_of(B_BOX));
    assert!(HALF_N.is_multiple_of(WIDE_B_BOX));
};

pub type ATile = SharedTile<F16, BLOCK_M, BLOCK_K, Swizzle128B>;

pub type BPanel = SharedTile<F16, B_BOX, BLOCK_K, Swizzle128B>;
pub type WideBPanel = SharedTile<F16, WIDE_B_BOX, BLOCK_K, Swizzle128B>;
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

/// A watched arm's marks land in [`REPORT_ROWS`] rows of `C` past the output:
/// [`REPORT_SLOTS`] warp slots a CTA — its six warps rounded up — of
/// [`REPORT_FIELDS`] words each, being site, `sequence`, ring index and parity.
pub const REPORT_FIELDS: u32 = 4;
pub const REPORT_SLOTS: u32 = 8;
/// Four rows hold the 512 CTAs of a 4096² grid at the `ldc >= 2048` these arms
/// are run at; the launcher asserts it rather than trusting it.
pub const REPORT_ROWS: usize = 4;

pub const SITE_FEED: u32 = 1;
pub const SITE_QUERY: u32 = 2;
pub const SITE_MULTIPLY: u32 = 3;
pub const SITE_DRAIN: u32 = 4;
pub const SITE_EXIT: u32 = 5;
pub const SITE_READY: u32 = 6;
pub const SITE_FREE: u32 = 7;
pub const SITE_CLC: u32 = 8;
pub const SITE_EMPTY: u32 = 9;
pub const SITE_FULL: u32 = 10;
pub const SITE_LOAD: u32 = 11;

pub const fn site_name(site: u32) -> &'static str {
    match site {
        0 => "-",
        SITE_FEED => "feeding item",
        SITE_QUERY => "querying clc",
        SITE_MULTIPLY => "multiplying item",
        SITE_DRAIN => "draining item",
        SITE_EXIT => "exited",
        SITE_READY => "STALL ready",
        SITE_FREE => "STALL free",
        SITE_CLC => "STALL clc",
        SITE_EMPTY => "STALL empty",
        SITE_FULL => "STALL full",
        SITE_LOAD => "STALL load",
        _ => "?",
    }
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
/// The watched arms' one-word "somebody has stalled" flag, which every role polls
/// at its loop head so a warp that gave up takes the other five out with it and
/// the launch reaches `cluster_sync` instead of hanging on it.
const fn stop_offset(rings_end: usize) -> usize {
    align_up(
        info_offset(rings_end) + SharedCellRing::<TileInfo, ITEMS>::BYTES,
        4,
    )
}
const fn queue_offset(rings_end: usize) -> usize {
    align_up(stop_offset(rings_end) + 4, ClcQueue::ALIGNMENT)
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
    stop: SharedCell<u32>,
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
                stop: SharedCell::attach(smem.add(stop_offset(rings_end))),
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
    unsafe fn initialize(self, free_arrivals: u32, empty_arrivals: u32) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                self.load.init_all(1);
                self.free.init_all(free_arrivals);
                self.full.init_all(1);
                self.empty.init_all(empty_arrivals);
                self.ready.init_all(1);
                self.stop.write(0);
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
    fn handoff<const WATCH: u8>(self, index: u32) -> (Semaphore, u32, SharedCell<TileInfo>) {
        let depth = items(WATCH);
        (
            self.ready.sem(index % depth),
            (index / depth) & 1,
            self.info.cell(index % depth),
        )
    }

    #[inline(always)]
    unsafe fn publish<const WATCH: u8>(self, index: u32, row: u32, column: u32, has_work: bool) {
        unsafe {
            let (ready, _, info) = self.handoff::<WATCH>(index);
            info.write(TileInfo {
                row,
                column,
                has_work: has_work as u32,
            });
            ready.arrive();
        }
    }

    #[inline(always)]
    unsafe fn next_info<const WATCH: u8>(self, sequence: u32) -> TileInfo {
        unsafe {
            let (ready, parity, info) = self.handoff::<WATCH>(sequence);
            if watches(WATCH) && !ready.wait_before(parity, DEADLINE) {
                self.stall(SITE_READY, sequence, sequence % items(WATCH), parity);
                return TileInfo {
                    row: 0,
                    column: 0,
                    has_work: 0,
                };
            }
            if !watches(WATCH) {
                ready.wait(parity);
            }
            info.read()
        }
    }

    #[inline(always)]
    unsafe fn wait_load<const WATCH: u8>(self, slot: u32, parity: u32) {
        unsafe {
            if watches(WATCH) {
                if !self.load.sem(slot).wait_before(parity, DEADLINE) {
                    self.stall(SITE_LOAD, 0, slot, parity);
                }
            } else {
                self.load.sem(slot).wait(parity);
            }
        }
    }

    #[inline(always)]
    unsafe fn wait_load_at<const WATCH: u8>(self, index: u32) {
        unsafe {
            self.wait_load::<WATCH>(index % STAGES as u32, (index / STAGES as u32) & 1);
        }
    }

    #[inline(always)]
    unsafe fn wait_empty_at<const WATCH: u8>(self, half: u32, parity: u32) {
        unsafe {
            if watches(WATCH) {
                if !self.empty.sem(half).wait_before(parity, DEADLINE) {
                    self.stall(SITE_EMPTY, 0, half, parity);
                }
            } else {
                self.empty.sem(half).wait(parity);
            }
        }
    }

    #[inline(always)]
    unsafe fn wait_full<const WATCH: u8>(self, sequence: u32) -> bool {
        unsafe {
            if watches(WATCH) {
                let ok = self.full.wait_before(sequence, DEADLINE);
                if !ok {
                    self.stall(
                        SITE_FULL,
                        sequence,
                        sequence % ACCUMULATORS as u32,
                        (sequence / ACCUMULATORS as u32) & 1,
                    );
                }
                ok
            } else {
                self.full.wait(sequence);
                true
            }
        }
    }

    #[inline(always)]
    unsafe fn stalled<const WATCH: u8>(self) -> bool {
        unsafe { watches(WATCH) && self.stop.read() != 0 }
    }

    #[inline(always)]
    unsafe fn stall(self, site: u32, sequence: u32, index: u32, parity: u32) {
        unsafe {
            self.write_mark(site, sequence, index, parity);
            self.stop.write(1);
        }
    }

    #[inline(always)]
    unsafe fn mark<const WATCH: u8>(self, site: u32, sequence: u32, index: u32, parity: u32) {
        if watches(WATCH) {
            unsafe { self.write_mark(site, sequence, index, parity) }
        }
    }

    #[inline(always)]
    unsafe fn write_mark(self, site: u32, sequence: u32, index: u32, parity: u32) {
        unsafe {
            if self.lane != 0 {
                return;
            }
            let slot = REPORT_FIELDS * (REPORT_SLOTS * thread::blockIdx_x() + self.warp_id);
            let row = self.tiles_m * (2 * BLOCK_M) as u32;
            let words = [site, sequence, index, parity];
            let mut field = 0usize;
            while field < words.len() {
                self.c
                    .at(row, slot + field as u32)
                    .cast::<u16>()
                    .write_volatile(words[field] as u16);
                field += 1;
            }
        }
    }

    /// Hand an accumulator slot back to the MMA warp, at the scope [`one_lane`]
    /// reads out of `RELEASE`. [`empty_arrivals`] is the same bit read from the
    /// other end and the two have to agree.
    ///
    /// What makes the guard legal is the `warp::sync_mask` that precedes every
    /// call: `tcgen05.wait::ld` retires every load the *warp* has outstanding, so
    /// past that sync no lane of it still has the accumulator in flight, and
    /// nothing below the last band's load reads tensor memory at all — which is
    /// also what makes [`early`] legal.
    #[inline(always)]
    unsafe fn release_accumulator<const RELEASE: u8>(self, slot: u32) {
        unsafe {
            if one_lane(RELEASE) && self.lane != 0 {
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

    #[inline(always)]
    fn drain_columns<const N: usize, const GROUPS: u32>(self) -> (u32, u32) {
        let span = N as u32 / GROUPS;
        (self.column_group() * span, span)
    }

    #[inline(always)]
    unsafe fn drain<const ABLATE: u8, const N: usize, const STAGE_N: usize, const GROUPS: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
        again: (u32, u32),
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(N.is_multiple_of(STAGE_N * GROUPS as usize));
            };
            let stage = self.staging::<STAGE_N>();
            let row = self.drain_row(tile_row);
            let (base, span) = self.drain_columns::<N, GROUPS>();
            let mut column = 0u32;
            while column < span {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let band: StageBand =
                        accumulator.tile_x8(32 * self.row_block(), base + column + band_column);
                    store_tile_x4(stage.chunk_writer(), 0, band_column, self.lane, band);
                    if twice(ABLATE) >= 2 {
                        let restaged: StageBand = if twice(ABLATE) >= 3 {
                            accumulator.tile_x8(32 * self.row_block(), base + column + band_column)
                        } else {
                            band
                        };
                        store_tile_x4(stage.chunk_writer(), 0, band_column, self.lane, restaged);
                    }
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
                if twice(ABLATE) >= 1 {
                    store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                        self.c,
                        self.drain_row(again.0),
                        again.1 + base + column,
                        self.lane,
                        stage,
                    );
                }
                warp::sync_mask(u32::MAX);
                column += STAGE_N as u32;
            }
        }
    }

    #[inline(always)]
    unsafe fn drain_batched<
        const N: usize,
        const STAGE_N: usize,
        const BAND: usize,
        const ISSUES: usize,
        const GROUPS: u32,
        const RELEASE: u8,
    >(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
        slot: u32,
    ) where
        BaseLdtm: kittens::reg::FragmentLayout<32, BAND>,
    {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND));
                assert!(N.is_multiple_of(STAGE_N * GROUPS as usize));
                assert!(ISSUES == BAND / 32);
                assert!(ISSUES <= kittens::tmem::ISSUE_LIMIT);
            };
            let stage = self.staging::<STAGE_N>();
            let row = self.drain_row(tile_row);
            let (base, span) = self.drain_columns::<N, GROUPS>();
            let last_band = span - BAND as u32;
            let mut column = 0u32;
            while column < span {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let band: RegTile<32, BAND, BaseLdtm> = accumulator
                        .tile_x8_batched::<32, BAND, ISSUES>(
                            32 * self.row_block(),
                            base + column + band_column,
                        );
                    if early(RELEASE) && column + band_column == last_band {
                        warp::sync_mask(u32::MAX);
                        self.release_accumulator::<RELEASE>(slot);
                    }
                    store_tile_x4(stage.chunk_writer(), 0, band_column, self.lane, band);
                    band_column += BAND as u32;
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

    #[inline(always)]
    unsafe fn drain_packed<const N: usize, const STAGE_N: usize, const GROUPS: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(N.is_multiple_of(STAGE_N * GROUPS as usize));
            };
            let stage = self.staging::<STAGE_N>();
            let row = self.drain_row(tile_row);
            let (base, span) = self.drain_columns::<N, GROUPS>();
            let mut column = 0u32;
            while column < span {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let packed = accumulator
                        .fragments_pack16_x8(32 * self.row_block(), base + column + band_column);
                    let mut block = 0usize;
                    while block < 8 {
                        store_packed_x4(
                            stage.chunk_writer(),
                            16 * (block as u32 / 4),
                            band_column + 16 * (block as u32 % 4),
                            self.lane,
                            [
                                packed[4 * block],
                                packed[4 * block + 1],
                                packed[4 * block + 2],
                                packed[4 * block + 3],
                            ],
                        );
                        block += 1;
                    }
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

    #[inline(always)]
    unsafe fn drain_nocvt<const N: usize, const STAGE_N: usize, const GROUPS: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
    ) {
        unsafe {
            const {
                assert!(STAGE_N.is_multiple_of(BAND_N));
                assert!(N.is_multiple_of(STAGE_N * GROUPS as usize));
            };
            let stage = self.staging::<STAGE_N>();
            let row = self.drain_row(tile_row);
            let (base, span) = self.drain_columns::<N, GROUPS>();
            let mut column = 0u32;
            while column < span {
                let mut band_column = 0u32;
                while band_column < STAGE_N as u32 {
                    let band: StageBand = accumulator.tile_x8_batched::<32, BAND_N, 2>(
                        32 * self.row_block(),
                        base + column + band_column,
                    );
                    let mut block = 0usize;
                    while block < 8 {
                        let slot = block / 4;
                        store_packed_x4(
                            stage.chunk_writer(),
                            16 * slot as u32,
                            band_column + 16 * (block as u32 % 4),
                            self.lane,
                            [
                                band.get(slot, 4 * (block % 4)).to_bits(),
                                band.get(slot, 4 * (block % 4) + 1).to_bits(),
                                band.get(slot, 4 * (block % 4) + 2).to_bits(),
                                band.get(slot, 4 * (block % 4) + 3).to_bits(),
                            ],
                        );
                        block += 1;
                    }
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

    /// Which drain a `DRAIN` value names. `WIDE_BAND` and `WIDE_ISSUES` are
    /// parameters because Rust monomorphizes both sides of a `match` on a const: a
    /// literal 128-column band here would break the two-warpgroup build, whose
    /// 64-wide staging tile cannot divide it, even though it never takes that arm.
    #[inline(always)]
    /// The drain rung `DRAIN` names, and **the accumulator's release with it**:
    /// only the batched rungs can place the arrival at the last band's load, so
    /// every other rung takes the late release here rather than carrying a dial
    /// it cannot honour.
    #[allow(clippy::too_many_arguments)]
    unsafe fn drain_dial<
        const ABLATE: u8,
        const DRAIN: u8,
        const N: usize,
        const STAGE_N: usize,
        const GROUPS: u32,
        const RELEASE: u8,
        const WIDE_BAND: usize,
        const WIDE_ISSUES: usize,
    >(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        tile_row: u32,
        tile_column: u32,
        again: (u32, u32),
        slot: u32,
    ) where
        BaseLdtm: kittens::reg::FragmentLayout<32, WIDE_BAND>,
    {
        unsafe {
            let batched = DRAIN == DRAIN_PAIRED || DRAIN == DRAIN_WIDE;
            match DRAIN {
                DRAIN_PAIRED => self.drain_batched::<N, STAGE_N, BAND_N, 2, GROUPS, RELEASE>(
                    accumulator,
                    tile_row,
                    tile_column,
                    slot,
                ),
                DRAIN_WIDE => self
                    .drain_batched::<N, STAGE_N, WIDE_BAND, WIDE_ISSUES, GROUPS, RELEASE>(
                        accumulator,
                        tile_row,
                        tile_column,
                        slot,
                    ),
                DRAIN_PACK16 => {
                    self.drain_packed::<N, STAGE_N, GROUPS>(accumulator, tile_row, tile_column)
                }
                DRAIN_NOCVT => {
                    self.drain_nocvt::<N, STAGE_N, GROUPS>(accumulator, tile_row, tile_column)
                }
                _ => self.drain::<ABLATE, N, STAGE_N, GROUPS>(
                    accumulator,
                    tile_row,
                    tile_column,
                    again,
                ),
            }
            if !(batched && early(RELEASE)) {
                self.release_accumulator::<RELEASE>(slot);
            }
        }
    }
}

/// The one-accumulator entry, generic in the cluster tile's width: `N` columns of
/// `C`, `HALF` the half-panel of `B` a rank loads, `BOX` the TMA box it arrives
/// in, and `STAGE` **a warp's** columns of the shared epilogue staging buffer.
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
    unsafe fn producer<const ABLATE: u8, const WATCH: u8>(self) {
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

            'items: loop {
                if raw_item < valid_items {
                    let (tile_n, tile_m) =
                        pipeline::grouped(raw_item, common.tiles_n, common.tiles_m, common.group);
                    common.publish::<WATCH>(sequence, tile_m, tile_n, true);
                    common.mark::<WATCH>(SITE_FEED, sequence, raw_item, 0);
                    let a_row =
                        (tile_m * (2 * BLOCK_M) as u32 + common.rank * BLOCK_M as u32) as i32;
                    let b_row = (tile_n * N as u32 + common.rank * HALF as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        if watches(WATCH) {
                            if !common.free.wait_recycled_before(global_k, DEADLINE) {
                                common.stall(SITE_FREE, sequence, global_k, 0);
                                break 'items;
                            }
                        } else {
                            common.free.wait_recycled(global_k);
                        }
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

                common.mark::<WATCH>(SITE_QUERY, sequence, raw_item, 0);
                let next = if watches(WATCH) {
                    match cursor.next_before(DEADLINE) {
                        Harvest::Item(item) => Some(item),
                        Harvest::Done => None,
                        Harvest::Stalled => {
                            common.stall(SITE_CLC, sequence, raw_item, 0);
                            break;
                        }
                    }
                } else {
                    cursor.next()
                };
                let Some(next) = next else {
                    common.publish::<WATCH>(sequence, 0, 0, false);
                    common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    break;
                };
                raw_item = next;
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply_stage<const ABLATE: u8, const WATCH: u8>(
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
                    common.wait_load_at::<WATCH>(global_k);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        accumulator,
                        self.a.tile(global_k).k_walk(),
                        self.b.tile(global_k).k_walk(),
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
            }
        }
    }

    /// One K block at a **compile-time** stage. Every entry's contract is
    /// `k % (STAGES * BLOCK_K) == 0`, so `global_k % STAGES == k % STAGES` and the
    /// four positions of the unroll are always stages 0, 1, 2, 3 in that order;
    /// only the phase parity moves with `global_k`, once a turn instead of four
    /// times.
    #[inline(always)]
    unsafe fn multiply_at<const ABLATE: u8, const WATCH: u8, const SLOT: u32>(
        self,
        accumulator: TmemTile<BLOCK_M, N>,
        parity: u32,
        accumulate: bool,
    ) {
        unsafe {
            let common = self.common;
            if common.rank == LEADER {
                if waits_on_load(ABLATE) {
                    common.wait_load::<WATCH>(SLOT, parity);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        accumulator,
                        self.a.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
                        accumulate,
                    );
                }
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
            }
        }
    }

    /// The K walk, `STAGES` blocks a turn. `FOLD` picks the spelling:
    /// [`Self::multiply_at`]'s const stage, or [`Self::multiply_stage`]'s
    /// `global_k`, which compute the same `C` by construction.
    #[inline(always)]
    unsafe fn walk_k<const ABLATE: u8, const FOLD: bool, const WATCH: u8>(
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
                    self.multiply_at::<ABLATE, WATCH, 0>(accumulator, parity, turn > 0);
                    self.multiply_at::<ABLATE, WATCH, 1>(accumulator, parity, true);
                    self.multiply_at::<ABLATE, WATCH, 2>(accumulator, parity, true);
                    self.multiply_at::<ABLATE, WATCH, 3>(accumulator, parity, true);
                    turn += 1;
                }
            } else {
                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE, WATCH>(accumulator, sequence, k);
                    self.multiply_stage::<ABLATE, WATCH>(accumulator, sequence, k + 1);
                    self.multiply_stage::<ABLATE, WATCH>(accumulator, sequence, k + 2);
                    self.multiply_stage::<ABLATE, WATCH>(accumulator, sequence, k + 3);
                    k += 4;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8, const FOLD: bool, const WATCH: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                if common.stalled::<WATCH>() {
                    break;
                }
                if common.next_info::<WATCH>(sequence).has_work == 0 {
                    if !common.stalled::<WATCH>() {
                        common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    }
                    break;
                }
                common.mark::<WATCH>(SITE_MULTIPLY, sequence, 0, 0);
                let accumulator = Common::accumulator(self.accumulator, sequence);
                if common.rank == LEADER && sequence >= ACCUMULATORS as u32 {
                    let ahead = sequence - ACCUMULATORS as u32;
                    if watches(WATCH) {
                        if !common.empty.wait_before(ahead, DEADLINE) {
                            common.stall(
                                SITE_EMPTY,
                                sequence,
                                ahead % ACCUMULATORS as u32,
                                (ahead / ACCUMULATORS as u32) & 1,
                            );
                            break;
                        }
                    } else {
                        common.empty.wait(ahead);
                    }
                }

                self.walk_k::<ABLATE, FOLD, WATCH>(accumulator, sequence);
                if common.rank == LEADER {
                    commit_multicast_cg2(common.full.sem(sequence), PAIR);
                }
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue<
        const ABLATE: u8,
        const DRAIN: u8,
        const WATCH: u8,
        const GROUPS: u32,
        const RELEASE: u8,
        const WIDE_BAND: usize,
        const WIDE_ISSUES: usize,
    >(
        self,
    ) where
        BaseLdtm: kittens::reg::FragmentLayout<32, WIDE_BAND>,
    {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            let mut again = (0u32, 0u32);
            loop {
                if common.stalled::<WATCH>() {
                    break;
                }
                let info = common.next_info::<WATCH>(sequence);
                if info.has_work == 0 {
                    if !common.stalled::<WATCH>() {
                        common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    }
                    break;
                }
                common.mark::<WATCH>(SITE_DRAIN, sequence, info.row, info.column);
                let (row, column) = (info.row * (2 * BLOCK_M) as u32, info.column * N as u32);
                if twice(ABLATE) > 0 && sequence == 0 {
                    again = (row, column);
                }
                if !common.wait_full::<WATCH>(sequence) {
                    break;
                }
                if drains(ABLATE) || DRAIN >= DRAIN_PACK16 {
                    common.drain_dial::<
                        ABLATE,
                        DRAIN,
                        N,
                        STAGE,
                        GROUPS,
                        RELEASE,
                        WIDE_BAND,
                        WIDE_ISSUES,
                    >(
                        Common::accumulator(self.accumulator, sequence),
                        row,
                        column,
                        again,
                        sequence,
                    );
                } else {
                    common.release_accumulator::<RELEASE>(sequence);
                }
                sequence += 1;
            }
        }
    }
}

/// The two-accumulator entry: two `A` rings against one shared `B` half-panel, so
/// a K block is twice the MMA against the same one `load` round trip. `BOX` is
/// the TMA box that half-panel arrives in, exactly as on [`Small`].
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
    unsafe fn producer<const ABLATE: u8, const WATCH: u8>(self) {
        unsafe {
            let common = self.common;
            let mut cursor = common.queue.cursor();
            let mut raw_item = cluster::cluster_idx();
            let mut sequence = 0u32;
            let macro_tiles_m = common.tiles_m / 2;
            let valid_items = macro_tiles_m * common.tiles_n;

            'items: loop {
                if raw_item < valid_items {
                    let (tile_n, macro_m) =
                        pipeline::grouped(raw_item, common.tiles_n, macro_tiles_m, common.group);
                    common.publish::<WATCH>(sequence, macro_m, tile_n, true);
                    common.mark::<WATCH>(SITE_FEED, sequence, raw_item, 0);
                    let a_row0 = (macro_m * 512 + common.rank * BLOCK_M as u32) as i32;
                    let a_row1 = a_row0 + 256;
                    let b_row = (tile_n * BLOCK_N as u32 + common.rank * HALF_N as u32) as i32;

                    let mut k = 0u32;
                    while k < common.k_blocks {
                        let global_k = sequence * common.k_blocks + k;
                        if watches(WATCH) {
                            if !common.free.wait_recycled_before(global_k, DEADLINE) {
                                common.stall(SITE_FREE, sequence, global_k, 0);
                                break 'items;
                            }
                        } else {
                            common.free.wait_recycled(global_k);
                        }
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

                common.mark::<WATCH>(SITE_QUERY, sequence, raw_item, 0);
                let next = if watches(WATCH) {
                    match cursor.next_before(DEADLINE) {
                        Harvest::Item(item) => Some(item),
                        Harvest::Done => None,
                        Harvest::Stalled => {
                            common.stall(SITE_CLC, sequence, raw_item, 0);
                            break;
                        }
                    }
                } else {
                    cursor.next()
                };
                let Some(next) = next else {
                    common.publish::<WATCH>(sequence, 0, 0, false);
                    common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    break;
                };
                raw_item = next;
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply_stage<const ABLATE: u8, const WATCH: u8>(self, sequence: u32, k: u32) {
        unsafe {
            let common = self.common;
            let global_k = sequence * common.k_blocks + k;
            if common.rank == LEADER {
                if waits_on_load(ABLATE) {
                    common.wait_load_at::<WATCH>(global_k);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        self.accumulator,
                        self.a0.tile(global_k).k_walk(),
                        self.b.tile(global_k).k_walk(),
                        k > 0,
                    );
                }
                commit_multicast_cg2(common.free.sem(global_k), PAIR);
                if k + 1 == common.k_blocks {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && k == 0 {
                    common.wait_empty_at::<WATCH>(1, (sequence - 1) & 1);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        self.accumulator.columns_right(BLOCK_N as u32),
                        self.a1.tile(global_k).k_walk(),
                        self.b.tile(global_k).k_walk(),
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
    unsafe fn multiply_at<const ABLATE: u8, const WATCH: u8, const SLOT: u32>(
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
                    common.wait_load::<WATCH>(SLOT, parity);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        self.accumulator,
                        self.a0.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
                        accumulate,
                    );
                }
                commit_multicast_cg2(common.free.sem(SLOT), PAIR);
                if last {
                    commit_multicast_cg2(common.full.sem(0), PAIR);
                }

                if sequence > 0 && first {
                    common.wait_empty_at::<WATCH>(1, (sequence - 1) & 1);
                }
                if multiplies(ABLATE) {
                    mma_walk_cg2::<F16, CHUNKS, _, _>(
                        self.accumulator.columns_right(BLOCK_N as u32),
                        self.a1.tile(SLOT).k_walk(),
                        self.b.tile(SLOT).k_walk(),
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

    #[inline(always)]
    unsafe fn walk_k<const ABLATE: u8, const FOLD: bool, const WATCH: u8>(self, sequence: u32) {
        unsafe {
            let common = self.common;
            if FOLD {
                let turns = common.k_blocks / STAGES as u32;
                let cycle = sequence * turns;
                let mut turn = 0u32;
                while turn < turns {
                    let parity = (cycle + turn) & 1;
                    let last = turn + 1 == turns;
                    self.multiply_at::<ABLATE, WATCH, 0>(
                        sequence,
                        parity,
                        turn > 0,
                        turn == 0,
                        false,
                    );
                    self.multiply_at::<ABLATE, WATCH, 1>(sequence, parity, true, false, false);
                    self.multiply_at::<ABLATE, WATCH, 2>(sequence, parity, true, false, false);
                    self.multiply_at::<ABLATE, WATCH, 3>(sequence, parity, true, false, last);
                    turn += 1;
                }
            } else {
                let mut k = 0u32;
                while k < common.k_blocks {
                    self.multiply_stage::<ABLATE, WATCH>(sequence, k);
                    self.multiply_stage::<ABLATE, WATCH>(sequence, k + 1);
                    self.multiply_stage::<ABLATE, WATCH>(sequence, k + 2);
                    self.multiply_stage::<ABLATE, WATCH>(sequence, k + 3);
                    k += 4;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn multiply<const ABLATE: u8, const FOLD: bool, const WATCH: u8>(self) {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            loop {
                if common.stalled::<WATCH>() {
                    break;
                }
                if common.next_info::<WATCH>(sequence).has_work == 0 {
                    if !common.stalled::<WATCH>() {
                        common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    }
                    break;
                }
                common.mark::<WATCH>(SITE_MULTIPLY, sequence, 0, 0);
                if common.rank == LEADER && sequence > 0 {
                    if watches(WATCH) {
                        if !common
                            .empty
                            .sem(0)
                            .wait_before((sequence - 1) & 1, DEADLINE)
                        {
                            common.stall(SITE_EMPTY, sequence, 0, (sequence - 1) & 1);
                            break;
                        }
                    } else {
                        common.empty.sem(0).wait((sequence - 1) & 1);
                    }
                }
                self.walk_k::<ABLATE, FOLD, WATCH>(sequence);
                sequence += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn epilogue<
        const ABLATE: u8,
        const DRAIN: u8,
        const WATCH: u8,
        const GROUPS: u32,
        const RELEASE: u8,
        const WARP_STAGE: usize,
        const WIDE_BAND: usize,
        const WIDE_ISSUES: usize,
    >(
        self,
    ) where
        BaseLdtm: kittens::reg::FragmentLayout<32, WIDE_BAND>,
    {
        unsafe {
            let common = self.common;
            let mut sequence = 0u32;
            let mut again = (0u32, 0u32);
            'items: loop {
                if common.stalled::<WATCH>() {
                    break;
                }
                let info = common.next_info::<WATCH>(sequence);
                if info.has_work == 0 {
                    if !common.stalled::<WATCH>() {
                        common.mark::<WATCH>(SITE_EXIT, sequence, 0, 0);
                    }
                    break;
                }
                common.mark::<WATCH>(SITE_DRAIN, sequence, info.row, info.column);
                if twice(ABLATE) > 0 && sequence == 0 {
                    again = (info.row * 512, info.column * BLOCK_N as u32);
                }
                let mut half = 0u32;
                while half < 2 {
                    if watches(WATCH) {
                        if !common.full.sem(half).wait_before(sequence & 1, DEADLINE) {
                            common.stall(SITE_FULL, sequence, half, sequence & 1);
                            break 'items;
                        }
                    } else {
                        common.full.sem(half).wait(sequence & 1);
                    }
                    if drains(ABLATE) || DRAIN >= DRAIN_PACK16 {
                        common.drain_dial::<ABLATE, DRAIN, BLOCK_N, WARP_STAGE, GROUPS, RELEASE, WIDE_BAND, WIDE_ISSUES>(
                            self.accumulator.columns_right(half * BLOCK_N as u32),
                            info.row * 512 + half * 256,
                            info.column * BLOCK_N as u32,
                            again,
                            half,
                        );
                    } else {
                        common.release_accumulator::<RELEASE>(half);
                    }
                    half += 1;
                }
                sequence += 1;
            }
        }
    }
}

/// The width-generic device body every `[256, N]` entry and every ablation arm
/// instantiates, so no arm can drift from the kernel it decomposes.
///
/// `ACCUM_COLUMNS` is [`ACCUM_COLUMNS`] at the entry's own width, and a
/// parameter for the reason `HALF` is one — see that constant.
///
/// # Safety
///
/// As [`kernels::gemm_sol_m256`], at the entry's own width.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_body<
    const N: usize,
    const ACCUM_COLUMNS: usize,
    const HALF: usize,
    const BOX: usize,
    const STAGE: usize,
    const RINGS_END: usize,
    const ABLATE: u8,
    const FOLD: bool,
    const DRAIN: u8,
    const WATCH: u8,
    const GROUPS: u32,
    const RELEASE: u8,
    const WIDE_BAND: usize,
    const WIDE_ISSUES: usize,
>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    group: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) where
    BaseLdtm: kittens::reg::FragmentLayout<32, WIDE_BAND>,
{
    const { assert!(ACCUM_COLUMNS == ACCUMULATORS * N) };
    unsafe {
        let smem = DynamicSharedArray::<u8, 128>::get_raw();
        let common = Common::attach(smem, RINGS_END, tiles_m, tiles_n, k_blocks, group, ldc, c);
        common.initialize(1, empty_arrivals(GROUPS, RELEASE));
        let state = Small::<N, HALF, BOX, STAGE> {
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
            state.producer::<ABLATE, WATCH>();
        } else if common.warp_id == mma_warp(GROUPS) && common.lane == 0 {
            state.multiply::<ABLATE, FOLD, WATCH>();
        } else if common.warp_id < epilogue_warps(GROUPS) {
            state.epilogue::<ABLATE, DRAIN, WATCH, GROUPS, RELEASE, WIDE_BAND, WIDE_ISSUES>();
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
pub unsafe fn large_body<
    const BOX: usize,
    const ABLATE: u8,
    const FOLD: bool,
    const DRAIN: u8,
    const WATCH: u8,
    const GROUPS: u32,
    const RELEASE: u8,
    const WARP_STAGE: usize,
    const WIDE_BAND: usize,
    const WIDE_ISSUES: usize,
>(
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    group: u32,
    ldc: u32,
    c: &mut DisjointSlice<u16>,
) where
    BaseLdtm: kittens::reg::FragmentLayout<32, WIDE_BAND>,
{
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
        common.initialize(2, empty_arrivals(GROUPS, RELEASE));
        let state = Large::<BOX> {
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

        if common.warp_id == tma_warp(GROUPS) && common.lane == 0 {
            state.producer::<ABLATE, WATCH>();
        } else if common.warp_id == mma_warp(GROUPS) && common.lane == 0 {
            state.multiply::<ABLATE, FOLD, WATCH>();
        } else if common.warp_id < epilogue_warps(GROUPS) {
            state.epilogue::<ABLATE, DRAIN, WATCH, GROUPS, RELEASE, WARP_STAGE, WIDE_BAND, WIDE_ISSUES>();
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
                B_BOX,
                { BLOCK_N / TWO_WARPGROUPS as usize },
                SMALL_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                SHIPPED_GROUPS,
                SHIPPED_RELEASE,
                BAND_N,
                2,
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
                B_BOX,
                NARROW_N,
                NARROW_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                NARROW_GROUPS,
                SHIPPED_RELEASE,
                128,
                4,
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                SHIPPED_GROUPS,
                SHIPPED_RELEASE,
                { LARGE_STAGE_N / TWO_WARPGROUPS as usize },
                BAND_N,
                2,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c);
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
    /// number for all three: the 256-wide entries run [`SHIPPED_GROUPS`] epilogue
    /// warpgroups and the narrow one runs [`NARROW_GROUPS`]. It sits beside
    /// [`Variant::shared_bytes`] because it is the same kind of fact — what the
    /// host must pass to launch this entry and not another.
    pub const fn threads(self) -> u32 {
        match self {
            Self::M256xN128 => threads(NARROW_GROUPS),
            Self::M256xN256 | Self::M512xN256 => threads(SHIPPED_GROUPS),
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
pub fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

pub fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

pub fn stage_f16(rows: usize, k: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u16> {
    let mut staged = Vec::with_capacity(rows * k);
    for row in 0..rows {
        for depth in 0..k {
            staged.push(half::f16::from_f32(value(row, depth)).to_bits());
        }
    }
    staged
}

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
        watchdog::stage(&stream, &stage_f16(m, k, a_value))?
    } else {
        watchdog::cleared::<u16>(&stream, m * k)?
    };
    let b = if initialize {
        watchdog::stage(&stream, &stage_f16(n, k, b_value))?
    } else {
        watchdog::cleared::<u16>(&stream, n * k)?
    };
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;

    let mut c = watchdog::cleared::<u16>(&stream, m * n)?;
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / variant.n_tile()) as u32;
    let config = LaunchConfig1D::new(
        grid_for(crate::bench::Shape { m, n, k }, variant),
        variant.threads(),
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

    // Nothing at all in a default build. With `--features wedge` and
    // `KITTENS_WEDGE_SECONDS` set, a launch that does not return, queued
    // immediately in front of this row's own on this row's own stream -- which
    // is the shape #146 had, and is the whole of what `kittens::watchdog` is a
    // guard against. See `crate::wedge` for why the injection is here and not
    // at the top of the process.
    #[cfg(feature = "wedge")]
    crate::wedge::inject(&stream)?;
    launch_once(&mut c)?;
    watchdog::wait(&stream)?;
    let label = if initialize {
        let worst = check_output(&c.read_back(&stream)?, m, n, k)?;
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
        let plan = Plan {
            variant,
            group: default_group(1024),
        };
        notes.push(run(context, 1024, 1024, 512, plan, true, nothing_after)?.0);
    }
    for (variant, m, n) in SHALLOW_K_GATE {
        let plan = Plan {
            variant,
            group: default_group(m),
        };
        notes.push(run(context, m, n, STAGES * BLOCK_K, plan, true, nothing_after)?.0);
    }
    Ok(notes.join("; "))
}

/// Check the plan at the gate size, then time it at `shape`.
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

pub fn clusters(shape: crate::bench::Shape, variant: Variant) -> u32 {
    grid_for(shape, variant) / RANKS
}
