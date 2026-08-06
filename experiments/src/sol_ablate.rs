//! Why the small entry's K loop is short of peak, and what each phase of the
//! launch costs — `bench sol-ablate`.
//!
//! #144 built the first arms and #146 settled the tiling, and between them they
//! left one number owned by nobody: **the `[256, 256]` entry's K loop ran at
//! 75.8% of tensor-core peak where the `[512, 256]` entry's ran at 99.4%, on the
//! same K-loop code.** This file is the instrument that explains it, and the
//! answer is one number per entry.
//!
//! # The answer: the feed's duty cycle
//!
//! `feed only` is the operand pipeline with the MMA and the drain removed, and
//! laddered in `K` it gives the time the feed needs for one K block with nothing
//! competing. Against the time the tensor core needs for the same K block at
//! peak:
//!
//! | entry | feed alone | MMA at peak | **the feed's duty** |
//! | --- | ---: | ---: | ---: |
//! | `[256, 256]` | 0.2636 µs | 0.2759 µs | **95.5%** |
//! | `[512, 256]` | 0.3075 µs | 0.5518 µs | **55.7%** |
//!
//! The small entry needs its feed running 95.5% of the time the tensor core is
//! busy; the large entry needs it 55.7%. That is the 1.33× in operand bytes per
//! flop between a `[256, 256]` cluster tile and a `[512, 256]` one — #146's
//! "0.75× the operand traffic per flop" seen from the other end — and at 95.5%
//! there is no slack for the TMA's writes and the tensor core's operand reads to
//! get out of each other's way. The cost of them not doing so is measured
//! directly: `no drain − issue only` is **+0.0675 µs a K block, 24.5%**, at
//! `[256, 256]` and **+0.0002 µs, 0.0%**, at `[512, 256]`.
//!
//! **Everything else in the K loop is free, and each of those is a measurement
//! rather than an argument.** Laddered in `K` at 4096² on `[256, 256]`, in µs a
//! K block against 0.2759 at peak:
//!
//! | arm | µs a K block | of peak |
//! | --- | ---: | ---: |
//! | `feed only` | 0.2636 | 104.6% |
//! | `mma only` | 0.2742 | **100.6%** |
//! | `issue only` | 0.2758 | **100.0%** |
//! | `no drain` | 0.3433 | 80.4% |
//! | `whole` | 0.3503 | 78.8% |
//! | `runtime stage` | 0.3615 | 76.3% |
//!
//! - **The tensor core can be issued at peak from one warp at this tile.**
//!   `issue only` is 100.0% and `mma only` 100.6%. There is no `tcgen05` issue-rate
//!   deficit, and the 11 points that *appeared* to be one in the un-laddered
//!   launch were the arm's own per-tile constant — which is why every arm here is
//!   laddered and none is quoted off a launch.
//! - **The barrier round trip is zero.** `issue only − mma only` differs by one
//!   const — the MMA warp's `load.wait`, dropped — and the two agree to 0.6%,
//!   with the arm that keeps the wait nominally *faster*. #144 listed this as
//!   unseparated; it is separated, and it is nothing.
//! - **The drain costs the K loop 2.0%** (`whole − no drain` on the rate) and
//!   2.9 µs a tile on the constant. That constant reproduces PR #147's
//!   independently measured 2.82 µs of serial drain per tile.
//! - **The producer's instruction count is not a term.** See below.
//!
//! # The arms
//!
//! Each arm is [`crate::gemm_sol`]'s own device body at a different `ABLATE`
//! const, so no arm can drift from the kernel it decomposes — the shipped
//! kernels are the same text at `WHOLE`.
//!
//! | arm | producer | MMA warp | epilogue warps |
//! | --- | --- | --- | --- |
//! | `whole` | TMA | wait, `tcgen05.mma`, commit | LDTM → `stmatrix` → `st.global` |
//! | `no drain` | TMA | wait, `tcgen05.mma`, commit | handshake only |
//! | `issue only` | arrives on `load`, no TMA | wait, `tcgen05.mma`, commit | handshake only |
//! | `feed only` | TMA | wait and commit, no MMA | handshake only |
//! | `mma only` | arrives on `load`, no TMA | `tcgen05.mma`, commit, **no wait** | handshake only |
//!
//! **Every barrier survives every arm but the last, which drops exactly one.**
//! In `mma only` the producer keeps `free.wait_recycled` so it is still
//! throttled, and `ready` still bounds the MMA warp one output tile at a time, so
//! what runs unthrottled is the K loop and only the K loop.
//!
//! **These compute a wrong `C` on purpose** and are on no correctness gate.
//! `regcount`'s opcode census is the gate instead: `feed only` must show zero in
//! the `mma` column and the whole kernel's `tma` count, `issue only` the reverse,
//! `no drain` zero across `ldtm`, `stmatrix` and the store columns while keeping
//! both of the others. An arm whose census does not read that way did not remove
//! what it names, and its number is void.
//!
//! # The epilogue, which is the other half of this file
//!
//! Nine more arms decompose the drain rather than the launch: a **doubling
//! ladder** (`twice global`/`twice shared`/`twice all`), four **drain rungs**
//! (`per issue`, `paired`, `wide`, and the `nocvt` oracle), the never-launched
//! `pack16`, and the **warpgroup split** (`two warpgroups` and its own control).
//!
//! Each ladder rung repeats one more pass of the drain per band, with the extra
//! global stores aimed at the cluster's own first output tile so they stay in L2.
//! An extra pass has nothing left to hide behind, so it is paid **serially by
//! construction**: `twice all − per issue` is the drain's own occupancy cost `D`,
//! and `paired − no drain` is what the launch pays. µs per tile per cluster:
//!
//! | | 4096³ `[256,256]` | 8192³ `[512,256]` |
//! | --- | ---: | ---: |
//! | `ld.shared` + `st.global` | 0.97 (36%) | 3.58 (41%) |
//! | `stmatrix`, and a doubled one's write-after-write | 1.59 (59%) | 6.01 (69%) |
//! | **`cvt`, `paired − nocvt`** | **−0.18 (0%)** | **−1.21 (0%)** |
//! | **LDTM and its wait** | **0.14 (5%)** | **−0.86 (0%)** |
//! | `D`, the whole chain serially | 2.70 | 8.73 |
//! | `E`, what the launch pays | **5.35 (198% of `D`)** | **9.45 (108% of `D`)** |
//!
//! **`E > D`, so there is no overlap to fail** — the first drain costs the launch
//! more than a whole extra one does, which is only possible if it slows the phase
//! beside it. At 4096³, where the accumulator *is* double-buffered across output
//! tiles, it is nearly twice as bad, so buffering is not sufficient.
//!
//! **And four of the five levers inside it measure zero.**
//!
//! - *LDTM waits.* Doubling `tcgen05.ld` and its waits costs nothing. #117 took
//!   eight issues and eight waits a band down to one and one for +23.6/+8.6/+3.6%;
//!   from there the rest are covered by the other warps. `paired` takes the last
//!   two to one and is worth +0.7% at 8192³ and nothing at 4096³.
//! - *The convert.* `nocvt` holds the LDTM count, the wait count, the eight
//!   `stmatrix` a band and the stores, and takes `cvt` to **zero** — and is 0.0 to
//!   1.4% *slower*. So the 69% is `stmatrix` and a doubled pass's own
//!   write-after-write, not the `cvt` beside it. This is also what closes
//!   `.pack::16b`: it folds the convert into the load (census `cvt` 0) and folding
//!   the convert away is worth nothing. It faults besides.
//! - *Store width.* Already maximal — `stmatrix.m8n8.x4` is the widest b16 form
//!   and `st.global.v4` is 16 bytes.
//! - *Warps.* The split doubles the epilogue's warps at the same total work, the
//!   same byte-identical shared plan, and a `no drain` control at its own 320
//!   threads that matches the 192-thread one to a tenth of a percent. It first
//!   measured the drain **−9% at 4096³ and +3% at 8192³** and was read against
//!   the lane argument in [`crate::gemm_sol::TWO_WARPGROUPS`] — warps 0 and 4 own
//!   the *same* 32 tensor-memory lanes, so a column split puts two requesters on
//!   each of four sub-partitions rather than spreading over more of them.
//!
//!   **That reading did not survive #166.** Every row of it was taken through a
//!   `.local` depot the library no longer emits, and the split pays a band's
//!   frame twice where one warpgroup pays it once. Re-run, the drain goes
//!   **3.35 → 3.10 µs at 4096³ and 4.72 → 3.05 µs at 8192³** and the launch is
//!   +0.8% and +2.1%, both reproduced in two passes. The lane ceiling is still
//!   there; it was simply never the term this table was reaching. Both 256-wide
//!   entries ship the split as of #197, and the arms in this file stay pinned to
//!   [`ONE_WARPGROUP`] so the ladder keeps comparing against what it always did.
//!
//! So the drain is bound by something other than the store path, and it is not
//! bytes: 262 144 B into shared per cluster per
//! tile in ~6 µs is 3.5 TB/s device-wide against an SM's ~230 GB/s of shared write
//! bandwidth, under 10% of it. It is not bank conflicts either — `Swizzle128B`
//! maps one `stmatrix.m8n8` matrix's eight rows onto eight chunks whose banks tile
//! all 32 exactly once, at every staging pitch here, because every row pitch is a
//! whole multiple of 128 B and the row term drops out of the bank index. What is
//! left is the `stmatrix` scatter's **transaction** rate, which nothing in this
//! file separates from its byte rate.
//!
//! # The two levers, one of which pays
//!
//! Neither is an ablation: both compute the same `C` as the shipped kernel and
//! differ from it in one const.
//!
//! **`runtime stage` pays, and it is the port defect.** The MMA warp used to
//! index both operand rings and both stage barriers off
//! `global_k = sequence * k_blocks + k`, a runtime value, so every offset and
//! both operand descriptors were runtime arithmetic inside its stream — and the
//! `mbarrier.try_wait.parity` spin loop re-derived the barrier's address on every
//! poll. But `k_blocks` is a multiple of `STAGES` (the shape contract is
//! `k % 256 == 0`), so `global_k % STAGES == k % STAGES` and the four positions
//! of the four-way unroll are always stages 0, 1, 2, 3. Handing each its stage as
//! a const folds the offsets to literals and the parity to one register a turn.
//! It is worth **+2.2% to +4.3% at 4096³** across two containers and four passes,
//! and it takes the K loop from 76.3% to 78.8% of peak. Upstream states the same
//! fact — *"`k_iters % 4 == 0` … keeping this expression loop-local lets the
//! unroll pass fold each stage match"* — and spells it `#[unroll(4)]` over a
//! loop-local `k_idx & 3`; that attribute is rewritten only inside a `#[kernel]`
//! or `#[device_function]` body, so a const parameter is the spelling reachable
//! from a plain `impl` method. Its gain is **not** in the MMA issue stream:
//! `mma only` against `runtime mma` is 1.018 / 1.001 / 1.005 / 0.998, null. Nor is
//! it in the `mbarrier.try_wait.parity` spin body: the PTX says that body went from
//! **7 instructions to 8** — folding the stage to a literal made LLVM
//! re-materialize `mov.b64` of the dynamic-shared symbol plus `add.s64` inside the
//! loop rather than keep a register live — and the kernel got faster anyway. What
//! is left is the scalar work **between one `load.wait` returning and the next
//! being entered**: pre-fold each stage computed its own parity (`add.s32`,
//! `bfe.u32`), its own ring offsets and its own two descriptors on that path, and
//! post-fold the offsets are literals, the descriptors are hoisted, the accumulate
//! predicate is constant at three of four positions, and the parity is computed
//! once a turn. That path exists only when the waits do, which is why `whole`
//! gains and `mma only` — which has no waits — does not.
//!
//! **`wide B` does not pay, and that is the result.** It is the same body at
//! `BOX = 128`: a rank's `B` half-panel arriving in **one** TMA instead of two, at
//! byte-for-byte identical traffic, which takes the producer from 3 instructions
//! a K block to 2 at `[256, 256]` and from 4 to 3 at `[512, 256]`. It moves
//! nothing — 0.998 and 1.014 on the launch across two passes, 0.3484 against
//! 0.3503 µs a K block on the rate, and 1.008 / 0.981 on `feed only` itself,
//! where the feed is alone and has nothing to hide behind. So the feed's ceiling
//! is bytes and not instructions, and **#144's conclusion that the feed is not
//! an instruction-rate problem survives** even though the comparison it drew
//! (ceiling against in-situ rate) could not have failed. The comparison that can
//! fail is ceiling against what peak MMA demands, and that is the duty-cycle
//! table at the top — which says the feed is not the problem at `[512, 256]` and
//! *is* the whole of it at `[256, 256]`.
//!
//! # The reference
//!
//! The last table is the same ladder on #145's byte-for-byte copy of upstream's
//! `gemm_sol_final` at the same tile: **0.3352 µs a K block, 82.3% of peak**. So
//! the deficit splits. 17.7 points are structural — upstream pays them too, and
//! the duty-cycle table says why — and the port's own share was 6.0 points before
//! the fold and is 3.5 after it. Our drain-free K loop (80.4%) is still 1.9 points
//! behind upstream's drain-inclusive one, so about two points of what is left is
//! in the feed's interaction rather than in the epilogue.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract};

use kittens::global::GlobalLayout;
use kittens::shared::F16;

use crate::bench::{Shape, Timings, time};
use crate::gemm_sol::{
    ACCUM_COLUMNS, ATile, B_BOX, BAND_N, BLOCK_N, BPanel, DRAIN_NOCVT, DRAIN_PACK16, DRAIN_PAIRED,
    DRAIN_PER_ISSUE, DRAIN_WIDE, FEED_ONLY, HALF_N, ISSUE_ONLY, LARGE_STAGE_N, MMA_ONLY, NO_DRAIN,
    ONE_WARPGROUP, SHIPPED_DRAIN, SMALL_RINGS_END, TWICE_ALL, TWICE_GLOBAL, TWICE_SHARED,
    TWO_WARPGROUPS, Variant, WATCH_OFF, WHOLE, WIDE_B_BOX, WideBPanel, clusters, default_group,
    large_body, small_body, threads,
};

/// SMs on the B200 this file's arithmetic is for, and CTAs one of them holds at
/// either entry's shared plan. A cluster is two CTAs, so [`WAVE`] clusters are
/// resident at once and everything below follows from that.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 1;
const RANKS: u32 = 2;
/// Clusters resident at once — 74.
pub const WAVE: u32 = SMS * CTAS_PER_SM / RANKS;

const _: () = assert!(WAVE == 74);

/// Dense FP16 tensor-core peak, TFLOP/s. Every `of peak` column divides by this.
const PEAK: f64 = 2250.0;

/// Depth one stage carries, so `k / K_TILE` is the K loop's trip count.
const K_TILE: usize = 64;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The whole kernel except the drain.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_nodrain(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                NO_DRAIN,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The MMA and every barrier, with no global traffic at all.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_issue(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                ISSUE_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// TMA and every barrier, with the MMA and the drain gone.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_feed(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                FEED_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_issue`] with the MMA warp's `load.wait` dropped too,
    /// which is the one difference that prices the barrier round trip.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_mma(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                MMA_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped kernel with `B`'s half-panel arriving in one TMA instead of
    /// two: two producer instructions a K block instead of three, at identical
    /// bytes.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wideb(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                WIDE_B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_feed`] at the wide box, which is the feed's own ceiling
    /// asked at two instruction counts and one byte count.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wideb_feed(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                WIDE_B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                FEED_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped kernel with the MMA warp's ring index computed from
    /// `global_k` at runtime, which is what the port carried before the stage
    /// became a const. Identical `C`, and the control for the fold.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_runtime(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                false,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_mma`] at the runtime ring index: the arm with nothing
    /// memory-shaped left in it, on both spellings of the same walk.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_runtime_mma(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                MMA_ONLY,
                false,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_nodrain`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_nodrain(
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
            large_body::<
                B_BOX,
                NO_DRAIN,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_issue`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_issue(
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
            large_body::<
                B_BOX,
                ISSUE_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_feed`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_feed(
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
            large_body::<
                B_BOX,
                FEED_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_mma`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_mma(
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
            large_body::<
                B_BOX,
                MMA_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wideb`] at the `[512, 256]` entry, where the same lever
    /// takes four producer instructions to three and should buy nothing,
    /// because that entry is already MMA-bound.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wideb(
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
            large_body::<
                WIDE_B_BOX,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wideb_feed`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wideb_feed(
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
            large_body::<
                WIDE_B_BOX,
                FEED_ONLY,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_runtime`] at the `[512, 256]` entry, whose K loop is
    /// already at 99.4% of peak and so has nothing for the fold to buy.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_runtime(
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
            large_body::<
                B_BOX,
                WHOLE,
                false,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_runtime_mma`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], with `b_map`'s box the height
    /// this arm's `BOX` names. Computes a wrong `C` unless its arm is `WHOLE`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_runtime_mma(
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
            large_body::<
                B_BOX,
                MMA_ONLY,
                false,
                SHIPPED_DRAIN,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The drain's `ld.shared` + `st.global` half run twice a band, the extra
    /// stores aimed at the cluster's own first output tile so they stay in L2.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_global(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                TWICE_GLOBAL,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The rung above with the `cvt` + `stmatrix` pass doubled as well.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_shared(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                TWICE_SHARED,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The whole drain twice, LDTM included — the drain priced serially.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_all(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                TWICE_ALL,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped drain, compiled **here**: two LDTM issues a band in flight
    /// before one wait. The same-crate baseline every drain rung is read against.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_paired(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                DRAIN_PAIRED,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The drain that shipped through #144: a wait per LDTM issue.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_per_issue(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// 128-column bands, four LDTM issues behind one wait. A published loser.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wide(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                DRAIN_WIDE,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped drain with the `cvt` pass removed and nothing else — the oracle
    /// that prices the convert. It computes a wrong `C` by construction.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_nocvt(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                DRAIN_NOCVT,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// `.pack::16b` in place of the `.x8` load and no `cvt` at all. **It faults on
    /// the device** and nothing launches it; it is here for its opcode census.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_pack16(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                DRAIN_PACK16,
                WATCH_OFF,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The drain's `ld.shared` + `st.global` half run twice a band, the extra
    /// stores aimed at the cluster's own first output tile so they stay in L2.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_global(
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
            large_body::<
                B_BOX,
                TWICE_GLOBAL,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The rung above with the `cvt` + `stmatrix` pass doubled as well.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_shared(
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
            large_body::<
                B_BOX,
                TWICE_SHARED,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The whole drain twice, LDTM included — the drain priced serially.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_all(
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
            large_body::<
                B_BOX,
                TWICE_ALL,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped drain, compiled **here**: two LDTM issues a band in flight
    /// before one wait. The same-crate baseline every drain rung is read against.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_paired(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                DRAIN_PAIRED,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The drain that shipped through #144: a wait per LDTM issue.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_per_issue(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                DRAIN_PER_ISSUE,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// 128-column bands, four LDTM issues behind one wait. A published loser.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wide(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                DRAIN_WIDE,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// The shipped drain with the `cvt` pass removed and nothing else — the oracle
    /// that prices the convert. It computes a wrong `C` by construction.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_nocvt(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                DRAIN_NOCVT,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// `.pack::16b` in place of the `.x8` load and no `cvt` at all. **It faults on
    /// the device** and nothing launches it; it is here for its opcode census.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes a wrong `C`
    /// unless its `ABLATE` is `WHOLE` and its `DRAIN` an exact rung.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_pack16(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                DRAIN_PACK16,
                WATCH_OFF,
                ONE_WARPGROUP,
                LARGE_STAGE_N,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// Eight epilogue warps splitting the accumulator's **columns** — two
    /// warpgroups, `EPILOGUE_ROWS` warps each, at a byte-identical shared plan.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], **at 320 threads**.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (320, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wg2(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                { BLOCK_N / 2 },
                SMALL_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                TWO_WARPGROUPS,
                BAND_N,
                2,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wg2`]'s own `no drain` control. It cannot share the
    /// 192-thread one: the launch geometry differs, so the subtraction would
    /// otherwise carry two warps of scheduling with it.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], **at 320 threads**.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (320, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wg2_nodrain(
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
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                { BLOCK_N / 2 },
                SMALL_RINGS_END,
                NO_DRAIN,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                TWO_WARPGROUPS,
                BAND_N,
                2,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wg2`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], **at 320 threads**.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (320, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wg2(
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
            large_body::<
                B_BOX,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                TWO_WARPGROUPS,
                { LARGE_STAGE_N / 2 },
                BAND_N,
                2,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wg2_nodrain`] at the `[512, 256]` entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`], **at 320 threads**.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (320, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wg2_nodrain(
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
            large_body::<
                B_BOX,
                NO_DRAIN,
                true,
                SHIPPED_DRAIN,
                WATCH_OFF,
                TWO_WARPGROUPS,
                { LARGE_STAGE_N / 2 },
                BAND_N,
                2,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }
}

/// One phase kept or removed, and the `B` box it is asked at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Whole,
    NoDrain,
    IssueOnly,
    FeedOnly,
    MmaOnly,
    WideWhole,
    WideFeed,
    RuntimeWhole,
    RuntimeMma,
    TwiceGlobal,
    TwiceShared,
    TwiceAll,
    /// The shipped drain compiled in this crate — the same-crate baseline the
    /// other drain rungs are read against.
    Paired,
    PerIssue,
    Wide,
    NoCvt,
    /// Eight epilogue warps splitting the accumulator's columns.
    Wg2,
    /// [`Arm::Wg2`]'s own control, at its own thread count.
    Wg2NoDrain,
    /// **Never launched.** `.pack::16b` faults on the device against an fp32
    /// accumulator — `Xid 13, Out Of Range Address` on every SM — so this arm
    /// exists for its opcode census and has no timed row. See
    /// [`crate::gemm_sol::DRAIN_PACK16`].
    Pack16,
}

impl Arm {
    /// The five that decompose the launch, in the order the budget subtracts them.
    const DECOMPOSE: [Arm; 5] = [
        Arm::Whole,
        Arm::NoDrain,
        Arm::IssueOnly,
        Arm::MmaOnly,
        Arm::FeedOnly,
    ];
    /// The wide-`B` comparison: two whole kernels and two feeds, one box apart.
    const BOXES: [Arm; 4] = [Arm::Whole, Arm::WideWhole, Arm::FeedOnly, Arm::WideFeed];
    /// The doubling ladder. Each rung adds one pass of the drain to the rung
    /// below, so the differences price the chain a third at a time — and the top
    /// rung against `per issue` prices a whole extra drain, serially, by
    /// construction.
    ///
    /// Not to be confused with [`Arm::LADDER`], which is the depth table's arm
    /// set: that one sweeps `K` with the kernel held, this one holds the shape and
    /// repeats a phase.
    const DOUBLING: [Arm; 4] = [
        Arm::PerIssue,
        Arm::TwiceGlobal,
        Arm::TwiceShared,
        Arm::TwiceAll,
    ];
    /// The drain rungs, all at [`crate::gemm_sol::WHOLE`], all on the same shared
    /// plan and all compiled in this crate, so one `no drain` control serves every
    /// one of them and no comparison crosses a crate boundary.
    const DRAINS: [Arm; 4] = [Arm::Paired, Arm::PerIssue, Arm::Wide, Arm::NoCvt];
    /// The drain rungs that compute a right `C` and are therefore on [`check`].
    /// `NoCvt` and `Pack16` are not among them and cannot be.
    const EXACT_DRAINS: [Arm; 4] = [Arm::Paired, Arm::PerIssue, Arm::Wide, Arm::Wg2];
    /// The warpgroup split and the pair of controls it is read against. Each arm
    /// subtracts the `no drain` at **its own** thread count, because the split
    /// changes the launch geometry and a shared control would carry that with it.
    const WARPGROUPS: [Arm; 4] = [Arm::Paired, Arm::NoDrain, Arm::Wg2, Arm::Wg2NoDrain];
    /// Every arm the depth table turns into a rate. The two levers under test
    /// come last, because they are not phases removed.
    const LADDER: [Arm; 7] = [
        Arm::Whole,
        Arm::NoDrain,
        Arm::IssueOnly,
        Arm::MmaOnly,
        Arm::FeedOnly,
        Arm::WideWhole,
        Arm::RuntimeWhole,
    ];
    /// The fold comparison: the shipped kernel and the runtime ring index, whole
    /// and stripped to the MMA warp. `mma only` is the pair that isolates the
    /// mechanism, because nothing memory-shaped is left in it.
    const FOLDS: [Arm; 4] = [Arm::RuntimeWhole, Arm::Whole, Arm::RuntimeMma, Arm::MmaOnly];

    fn name(self) -> &'static str {
        match self {
            Arm::Whole => "whole",
            Arm::NoDrain => "no drain",
            Arm::IssueOnly => "issue only",
            Arm::FeedOnly => "feed only",
            Arm::MmaOnly => "mma only",
            Arm::WideWhole => "wide B",
            Arm::WideFeed => "wide B feed",
            Arm::RuntimeWhole => "runtime stage",
            Arm::RuntimeMma => "runtime mma",
            Arm::TwiceGlobal => "twice global",
            Arm::TwiceShared => "twice shared",
            Arm::TwiceAll => "twice all",
            Arm::Paired => "paired",
            Arm::PerIssue => "per issue",
            Arm::Wide => "wide",
            Arm::NoCvt => "nocvt",
            Arm::Wg2 => "two warpgroups",
            Arm::Wg2NoDrain => "two wg, no drain",
            Arm::Pack16 => "pack16",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Arm::Whole => "the shipped kernel",
            Arm::NoDrain => "TMA + MMA + every barrier; the drain removed",
            Arm::IssueOnly => "MMA + every barrier; no global traffic at all",
            Arm::FeedOnly => "TMA + every barrier; no MMA, no drain",
            Arm::MmaOnly => "issue only, minus the MMA warp's load.wait",
            Arm::WideWhole => "the shipped kernel, B in one TMA not two",
            Arm::WideFeed => "feed only, B in one TMA not two",
            Arm::RuntimeWhole => "the shipped kernel, ring index off global_k",
            Arm::RuntimeMma => "mma only, ring index off global_k",
            Arm::TwiceGlobal => "+ a second ld.shared + st.global pass a band",
            Arm::TwiceShared => "+ a second cvt + stmatrix pass as well",
            Arm::TwiceAll => "+ a second LDTM: the whole drain twice",
            Arm::Paired => "the shipped drain, this crate: 2 LDTM, 1 wait",
            Arm::PerIssue => "the drain #144 shipped: a wait per LDTM issue",
            Arm::Wide => "128-column bands, 4 LDTM in flight, 1 wait",
            Arm::NoCvt => "the shipped drain minus the cvt; WRONG C",
            Arm::Wg2 => "8 epilogue warps, columns split, 320 threads",
            Arm::Wg2NoDrain => "the same launch with the drain removed",
            Arm::Pack16 => ".pack::16b; FAULTS ON THE DEVICE, never launched",
        }
    }

    /// Threads this arm's launch takes. The warpgroup-split arms are 320 and
    /// everything else is 192, and getting this wrong is a launch that either
    /// deadlocks on an arrival count or leaves warps out of the epilogue.
    fn threads(self) -> u32 {
        match self {
            Arm::Wg2 | Arm::Wg2NoDrain => threads(TWO_WARPGROUPS),
            _ => threads(ONE_WARPGROUP),
        }
    }

    /// The `B` tensor map's box height, which is the one thing outside the device
    /// body that a wide-box arm changes.
    fn box_rows(self) -> usize {
        match self {
            Arm::WideWhole | Arm::WideFeed => WIDE_B_BOX,
            _ => B_BOX,
        }
    }

    /// TMA instructions the producer issues per K block at this arm and entry —
    /// the count the wide-box prediction is about, derived here so the table can
    /// be read without the census beside it.
    fn tma_per_k_block(self, variant: Variant) -> usize {
        if matches!(self, Arm::IssueOnly | Arm::MmaOnly | Arm::RuntimeMma) {
            return 0;
        }
        let a_tiles = if variant == Variant::M512xN256 { 2 } else { 1 };
        a_tiles + HALF_N / self.box_rows()
    }
}

/// Tiles the shape launches, and the fraction of the waves they fill.
///
/// The launch is one cluster per output tile and the device holds [`WAVE`] of
/// them, so this is the whole of it: no residency model, no measurement.
pub fn quantization(shape: Shape, variant: Variant) -> (u32, u32, f64) {
    let tiles = clusters(shape, variant);
    let waves = tiles.div_ceil(WAVE);
    (tiles, waves, tiles as f64 / (waves * WAVE) as f64)
}

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// Time one arm at one shape.
///
/// Operands are zeroed rather than staged, exactly as `gemm_sol::bench` times the
/// shipped kernel: nothing here is checked, most of these arms cannot be, and
/// tensor cores do not take data-dependent time.
pub fn measure(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
    arm: Arm,
) -> Result<Timings, Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    if !m.is_multiple_of(variant.m_tile())
        || !n.is_multiple_of(variant.n_tile())
        || !k.is_multiple_of(STAGE_ALIGNED_K)
    {
        return Err(format!("{shape} violates {}'s tile contract", variant.name()).into());
    }
    let stream = context.default_stream();
    let shipped = unsafe { crate::gemm_sol::kernels::load(context)? };
    let ablated = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::<u16>::zeroed(&stream, m * k)?;
    let b = DeviceBuffer::<u16>::zeroed(&stream, n * k)?;
    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    // The one thing outside the device body that a wide-box arm changes: the
    // descriptor has to be the box the kernel asks for, or the transaction count
    // and the shape agreement are both wrong.
    let b_map = if arm.box_rows() == WIDE_B_BOX {
        b_layout.tensor_map::<WideBPanel>(&stream)?
    } else {
        b_layout.tensor_map::<BPanel>(&stream)?
    };

    let config = LaunchConfig1D::new(
        RANKS * (m / variant.m_tile() * (n / variant.n_tile())) as u32,
        arm.threads(),
        variant.shared_bytes() as u32,
    );
    let (stream_ref, a_ptr, b_ptr) = (&stream, a_map.as_ptr(), b_map.as_ptr());
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / variant.n_tile()) as u32;
    let k_blocks = (k / K_TILE) as u32;
    let group = default_group(m);
    let ldc = n as u32;

    // One spelling per arm, and the argument list written once: the arms differ
    // in the entry point they call and in nothing else, which is the claim every
    // table below rests on.
    macro_rules! launcher {
        ($module:expr, $prepare:ident, $call:ident) => {{
            let module = $module;
            let prepared = module.$prepare(config)?;
            let launch: Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>> =
                Box::new(move |output| {
                    unsafe {
                        module.$call(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group,
                            ldc, output,
                        )?
                    };
                    Ok(())
                });
            launch
        }};
    }

    let launch_once = match (variant, arm) {
        (Variant::M256xN256, Arm::Wg2) => {
            launcher!(&ablated, prepare_gemm_sol_m256_wg2, gemm_sol_m256_wg2)
        }
        (Variant::M256xN256, Arm::Wg2NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_wg2_nodrain,
                gemm_sol_m256_wg2_nodrain
            )
        }
        (Variant::M512xN256, Arm::Wg2) => {
            launcher!(&ablated, prepare_gemm_sol_m512_wg2, gemm_sol_m512_wg2)
        }
        (Variant::M512xN256, Arm::Wg2NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_wg2_nodrain,
                gemm_sol_m512_wg2_nodrain
            )
        }
        (Variant::M256xN256, Arm::TwiceGlobal) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_global,
                gemm_sol_m256_twice_global
            )
        }
        (Variant::M256xN256, Arm::TwiceShared) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_shared,
                gemm_sol_m256_twice_shared
            )
        }
        (Variant::M256xN256, Arm::TwiceAll) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_all,
                gemm_sol_m256_twice_all
            )
        }
        (Variant::M256xN256, Arm::Paired) => {
            launcher!(&ablated, prepare_gemm_sol_m256_paired, gemm_sol_m256_paired)
        }
        (Variant::M256xN256, Arm::PerIssue) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_per_issue,
                gemm_sol_m256_per_issue
            )
        }
        (Variant::M256xN256, Arm::Wide) => {
            launcher!(&ablated, prepare_gemm_sol_m256_wide, gemm_sol_m256_wide)
        }
        (Variant::M256xN256, Arm::NoCvt) => {
            launcher!(&ablated, prepare_gemm_sol_m256_nocvt, gemm_sol_m256_nocvt)
        }
        (Variant::M512xN256, Arm::TwiceGlobal) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_global,
                gemm_sol_m512_twice_global
            )
        }
        (Variant::M512xN256, Arm::TwiceShared) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_shared,
                gemm_sol_m512_twice_shared
            )
        }
        (Variant::M512xN256, Arm::TwiceAll) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_all,
                gemm_sol_m512_twice_all
            )
        }
        (Variant::M512xN256, Arm::Paired) => {
            launcher!(&ablated, prepare_gemm_sol_m512_paired, gemm_sol_m512_paired)
        }
        (Variant::M512xN256, Arm::PerIssue) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_per_issue,
                gemm_sol_m512_per_issue
            )
        }
        (Variant::M512xN256, Arm::Wide) => {
            launcher!(&ablated, prepare_gemm_sol_m512_wide, gemm_sol_m512_wide)
        }
        (Variant::M512xN256, Arm::NoCvt) => {
            launcher!(&ablated, prepare_gemm_sol_m512_nocvt, gemm_sol_m512_nocvt)
        }
        (_, Arm::Pack16) => {
            return Err(
                "pack16 faults on the device and is never launched; see gemm_sol::DRAIN_PACK16"
                    .into(),
            );
        }
        (Variant::M256xN256, Arm::Whole) => {
            launcher!(&shipped, prepare_gemm_sol_m256, gemm_sol_m256)
        }
        (Variant::M256xN256, Arm::NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_nodrain,
                gemm_sol_m256_nodrain
            )
        }
        (Variant::M256xN256, Arm::IssueOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m256_issue, gemm_sol_m256_issue)
        }
        (Variant::M256xN256, Arm::FeedOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m256_feed, gemm_sol_m256_feed)
        }
        (Variant::M256xN256, Arm::MmaOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m256_mma, gemm_sol_m256_mma)
        }
        (Variant::M256xN256, Arm::WideWhole) => {
            launcher!(&ablated, prepare_gemm_sol_m256_wideb, gemm_sol_m256_wideb)
        }
        (Variant::M256xN256, Arm::WideFeed) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_wideb_feed,
                gemm_sol_m256_wideb_feed
            )
        }
        (Variant::M256xN256, Arm::RuntimeWhole) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_runtime,
                gemm_sol_m256_runtime
            )
        }
        (Variant::M256xN256, Arm::RuntimeMma) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_runtime_mma,
                gemm_sol_m256_runtime_mma
            )
        }
        (Variant::M512xN256, Arm::Whole) => {
            launcher!(&shipped, prepare_gemm_sol_m512, gemm_sol_m512)
        }
        (Variant::M512xN256, Arm::NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_nodrain,
                gemm_sol_m512_nodrain
            )
        }
        (Variant::M512xN256, Arm::IssueOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m512_issue, gemm_sol_m512_issue)
        }
        (Variant::M512xN256, Arm::FeedOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m512_feed, gemm_sol_m512_feed)
        }
        (Variant::M512xN256, Arm::MmaOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m512_mma, gemm_sol_m512_mma)
        }
        (Variant::M512xN256, Arm::WideWhole) => {
            launcher!(&ablated, prepare_gemm_sol_m512_wideb, gemm_sol_m512_wideb)
        }
        (Variant::M512xN256, Arm::WideFeed) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_wideb_feed,
                gemm_sol_m512_wideb_feed
            )
        }
        (Variant::M512xN256, Arm::RuntimeWhole) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_runtime,
                gemm_sol_m512_runtime
            )
        }
        (Variant::M512xN256, Arm::RuntimeMma) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_runtime_mma,
                gemm_sol_m512_runtime_mma
            )
        }
        (Variant::M256xN128, _) => return Err("the narrow entry has no arms".into()),
    };

    launch_once(&mut c)?;
    stream.synchronize()?;
    let mut launch = || launch_once(&mut c);
    time(&stream, &mut launch)
}

/// `K` a launch must be a multiple of: `STAGES * BLOCK_K`, which every entry's
/// shape contract already states.
const STAGE_ALIGNED_K: usize = 256;

/// The two shapes the port has a gap at, with the entry each one ships.
const HEADLINE: [(Shape, Variant); 2] = [
    (
        Shape {
            m: 4096,
            n: 4096,
            k: 4096,
        },
        Variant::M256xN256,
    ),
    (
        Shape {
            m: 8192,
            n: 8192,
            k: 8192,
        },
        Variant::M512xN256,
    ),
];

/// `K` at fixed `M` and `N`: the tile grid, the waves, the `C` traffic and the
/// epilogue's total cost all hold, and only the arithmetic each tile amortizes
/// its fixed cost over moves.
const DEPTHS: [usize; 4] = [2048, 4096, 8192, 16384];

/// Flops one cluster's K block is, so a measured per-K-block time has a peak to
/// be a fraction of.
fn k_block_flops(variant: Variant) -> f64 {
    2.0 * variant.m_tile() as f64 * variant.n_tile() as f64 * K_TILE as f64
}

/// The per-tile constant and per-K-block rate a depth ladder decomposes into.
///
/// `ms = waves * (fixed + k_blocks * rate)`, taken through the shallowest and
/// deepest rungs, which is the whole of the model #144 established is a straight
/// line to 0.5% over an eightfold range in `K`.
fn ladder_fit(rungs: &[(usize, u32, f64)]) -> Option<(f64, f64)> {
    let (shallow, deep) = (rungs.first()?, rungs.last()?);
    let per_wave = |&(_, waves, ms): &(usize, u32, f64)| ms / waves as f64;
    let blocks = |&(k, _, _): &(usize, u32, f64)| (k / K_TILE) as f64;
    if blocks(deep) == blocks(shallow) {
        return None;
    }
    let rate = (per_wave(deep) - per_wave(shallow)) / (blocks(deep) - blocks(shallow));
    Some((per_wave(shallow) - blocks(shallow) * rate, rate))
}

fn depth_ladder(
    label: &str,
    base: Shape,
    variant: Variant,
    mut bench: impl FnMut(Shape) -> Result<Timings, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    println!("\n  {label} at m={}, n={}", base.m, base.n);
    println!(
        "  {:<18}{:>9}{:>8}{:>11}{:>12}{:>10}{:>14}",
        "shape", "k blocks", "waves", "min ms", "TFLOP/s", "of peak", "ms per tile"
    );
    let mut rungs = Vec::new();
    for k in DEPTHS {
        let shape = Shape { k, ..base };
        let (_, waves, _) = quantization(shape, variant);
        let timings = match bench(shape) {
            Ok(timings) => timings,
            Err(error) => {
                println!("  {:<18}FAIL  {error}", shape.to_string());
                continue;
            }
        };
        let rate = tflops(shape, timings.min());
        println!(
            "  {:<18}{:>9}{:>8}{:>11.4}{:>12.1}{:>9.1}%{:>14.5}",
            shape.to_string(),
            k / K_TILE,
            waves,
            timings.min(),
            rate,
            100.0 * rate / PEAK,
            timings.min() / waves as f64,
        );
        rungs.push((k, waves, timings.min()));
    }
    if let Some((fixed, rate)) = ladder_fit(&rungs) {
        let at_peak = k_block_flops(variant) * WAVE as f64 / PEAK / 1e9;
        println!(
            "  fit: {fixed:.4} ms fixed per tile + {:.4} us per k block, \
             against {:.4} us at peak — the K loop is {:.1}% of peak",
            1e3 * rate,
            1e3 * at_peak,
            100.0 * at_peak / rate,
        );
    }
    Ok(())
}

fn quantization_table() {
    println!(
        "\nwave quantization, derived and not measured — one cluster per output tile,\n\
         {WAVE} clusters resident (148 SMs, one CTA each at either shared plan, two\n\
         CTAs a cluster). `ceiling` is the most of peak the shape can reach however\n\
         perfect the cadence is."
    );
    println!(
        "  {:<18}{:>12}{:>10}{:>8}{:>12}{:>10}",
        "shape", "entry", "clusters", "waves", "last wave", "ceiling"
    );
    for (shape, variant) in HEADLINE {
        let (tiles, waves, efficiency) = quantization(shape, variant);
        let last = tiles - WAVE * (waves - 1);
        println!(
            "  {:<18}{:>12}{:>10}{:>8}{:>11.0}%{:>10.3}",
            shape.to_string(),
            variant.name(),
            tiles,
            waves,
            100.0 * last as f64 / WAVE as f64,
            efficiency,
        );
    }
}

fn denominator(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let Some(baseline) = crate::bench::CUBLASLT_F16 else {
        println!("\n0. no cuBLASLt column: this build has no `cublas` feature.");
        return Ok(());
    };
    println!(
        "\n0. the denominator, in this container — {} ({}). `of ceiling` is our ratio\n\
         divided by the derived wave-quantization ceiling: what would be left to explain\n\
         if the schedule were the only loss.",
        baseline.name,
        (baseline.about)()
    );
    println!(
        "  {:<18}{:>12}{:>13}{:>13}{:>13}",
        "shape", "ours TF/s", "theirs TF/s", "ours/theirs", "of ceiling"
    );
    for (shape, variant) in HEADLINE {
        let (_, _, ceiling) = quantization(shape, variant);
        let ours = measure(context, shape, variant, Arm::Whole)?;
        let (theirs, _) = (baseline.bench)(context, shape)?;
        let ratio = theirs.min() / ours.min();
        println!(
            "  {:<18}{:>12.1}{:>13.1}{:>13.3}{:>13.3}",
            shape.to_string(),
            tflops(shape, ours.min()),
            tflops(shape, theirs.min()),
            ratio,
            ratio / ceiling,
        );
    }
    Ok(())
}

fn ablation_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n1. the ablation arms — one phase of the item removed per row. `of whole` is this\n\
         arm's throughput over the whole kernel's, so a row near 1.00 is a phase that was\n\
         already free. `tma/k` is the producer's instruction count per K block, which is\n\
         what table 2 moves. `of peak` is only meaningful on `whole`: an arm that skipped\n\
         the arithmetic still gets credited with it. Every row but `mma only` keeps every\n\
         barrier, and that one drops the MMA warp's `load.wait` and nothing else, so\n\
         `issue only - mma only` is the barrier round trip alone."
    );
    for (shape, variant) in HEADLINE {
        let (tiles, waves, ceiling) = quantization(shape, variant);
        println!(
            "\n  {shape} — {}, {tiles} clusters over {waves} waves, ceiling {ceiling:.3}",
            variant.name()
        );
        println!(
            "  {:<12}{:>7}{:>11}{:>11}{:>12}{:>10}{:>10}  what it runs",
            "arm", "tma/k", "min ms", "median ms", "TFLOP/s", "of peak", "of whole"
        );
        let mut whole = f64::NAN;
        for arm in Arm::DECOMPOSE {
            let timings = measure(context, shape, variant, arm)?;
            if arm == Arm::Whole {
                whole = timings.min();
            }
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<12}{:>7}{:>11.4}{:>11.4}{:>12.1}{:>9.1}%{:>10.2}  {}",
                arm.name(),
                arm.tma_per_k_block(variant),
                timings.min(),
                timings.median(),
                rate,
                100.0 * rate / PEAK,
                whole / timings.min(),
                arm.about(),
            );
        }
    }
    Ok(())
}

/// Bytes of `C` one cluster writes at one output tile — the drain's whole job,
/// and what makes the two entries' drains comparable per byte.
fn output_bytes(variant: Variant) -> f64 {
    (variant.m_tile() * variant.n_tile() * 2) as f64
}

/// One block of arm rows, and the minima so a caller can subtract them.
fn arm_rows(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
    arms: &[Arm],
    against: Arm,
) -> Result<Vec<f64>, Box<dyn Error>> {
    println!(
        "  {:<14}{:>11}{:>11}{:>12}{:>10}{:>11}  what it runs",
        "arm", "min ms", "median ms", "TFLOP/s", "of peak", "of base"
    );
    let mut minima = Vec::with_capacity(arms.len());
    let mut base = f64::NAN;
    for &arm in arms {
        let timings = measure(context, shape, variant, arm)?;
        if arm == against {
            base = timings.min();
        }
        let rate = tflops(shape, timings.min());
        println!(
            "  {:<14}{:>11.4}{:>11.4}{:>12.1}{:>9.1}%{:>11.3}  {}",
            arm.name(),
            timings.min(),
            timings.median(),
            rate,
            100.0 * rate / PEAK,
            base / timings.min(),
            arm.about(),
        );
        minima.push(timings.min());
    }
    Ok(minima)
}

/// The doubling ladder, and the two terms `whole − no drain` cannot separate.
///
/// Each rung repeats one more pass of the drain per band, with the extra global
/// stores aimed at the cluster's own first output tile so they stay in L2 and the
/// probe prices instructions rather than a second HBM stream. An extra pass has
/// nothing left to hide behind, so it is paid **serially by construction**: that
/// makes `twice all − paired` the drain's own occupancy cost `D`, against which
/// `E = paired − no drain` is what the launch actually pays.
///
/// `E ≈ D` is a drain that fails to overlap at all; `E < D` is one partly hidden;
/// `E > D` is a drain that also slows the phase it runs beside.
fn chain_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n3. the doubling ladder — one more pass of the drain per rung, the extra stores\n\
         aimed at the cluster's own first tile so they stay in L2. `per tile` is the launch\n\
         divided by the tiles a critical-path cluster walks. every rung computes a wrong\n\
         `C` and is on no correctness gate; the census is what says each doubled exactly\n\
         the columns it names."
    );
    for (shape, variant) in HEADLINE {
        let (_, waves, _) = quantization(shape, variant);
        let per_tile = |milliseconds: f64| 1e3 * milliseconds / waves as f64;
        println!("\n  {shape} — {}", variant.name());
        let minima = arm_rows(context, shape, variant, &Arm::DOUBLING, Arm::PerIssue)?;
        let no_drain = measure(context, shape, variant, Arm::NoDrain)?.min();
        let nocvt = measure(context, shape, variant, Arm::NoCvt)?.min();
        let paired = measure(context, shape, variant, Arm::Paired)?.min();

        let (base, twice_all) = (minima[0], minima[3]);
        let exposed = per_tile(paired - no_drain);
        let serial = per_tile(twice_all - base);
        println!(
            "\n  the drain at {shape}, µs per tile per cluster ({:.0} B of `C`):",
            output_bytes(variant)
        );
        for (name, cost) in [
            ("ld.shared + st.global", per_tile(minima[1] - minima[0])),
            ("cvt + stmatrix", per_tile(minima[2] - minima[1])),
            ("tcgen05.ld and its wait", per_tile(minima[3] - minima[2])),
        ] {
            println!(
                "  {name:<26}{cost:>8.2}{:>9.0}%  of the serial drain",
                100.0 * cost / serial
            );
        }
        println!(
            "  {:<26}{serial:>8.2}           serial, `twice all - per issue`",
            "the whole chain"
        );
        println!(
            "  {:<26}{:>8.2}{:>9.0}%  of the serial drain, and with no write-after-write on\n\
             {:>36}its own staging tile, which the rung above pays",
            "cvt alone, `paired - nocvt`",
            per_tile(paired - nocvt),
            100.0 * per_tile(paired - nocvt) / serial,
            "",
        );
        println!(
            "  {:<26}{:>8.2}           `per issue - no drain`: the drain #144 shipped, so the\n\
             {:>36}wait this file removed is worth {:.2} in situ",
            "what it paid before",
            per_tile(base - no_drain),
            "",
            per_tile(base - paired),
        );
        println!(
            "  {:<26}{exposed:>8.2}{:>9.0}%  of the serial drain — `paired - no drain`",
            "what the launch pays",
            100.0 * exposed / serial
        );
        println!(
            "  {:<26}{:>8.2}{:>9.0}%  of the serial drain",
            "hidden behind the MMA",
            serial - exposed,
            100.0 * (serial - exposed) / serial
        );
    }
    Ok(())
}

/// The drain rungs against one another and against one `no drain` control.
///
/// Every rung declares the same shared plan and is compiled in this crate, so the
/// difference is the drain's own structure and nothing else. Two whole passes,
/// because a few percent between two launches is not quotable from one.
fn drain_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n4. the drains — the same epilogue at four issue structures, one `no drain`\n\
         control serving all of them. `epilogue` is `arm - no drain` in µs per tile per\n\
         cluster, which is the term this table exists to shrink. `nocvt` is an oracle and\n\
         not a candidate: it computes a wrong `C` and is here for what its time says about\n\
         the convert, which is that the convert is free."
    );
    let baseline = crate::bench::CUBLASLT_F16;
    for (shape, variant) in HEADLINE {
        let (_, waves, _) = quantization(shape, variant);
        let theirs = match baseline {
            Some(baseline) => Some((baseline.bench)(context, shape)?.0.min()),
            None => None,
        };
        for pass in 1..=2 {
            println!("\n  {shape} — {}, pass {pass}", variant.name());
            let minima = arm_rows(context, shape, variant, &Arm::DRAINS, Arm::Paired)?;
            let no_drain = measure(context, shape, variant, Arm::NoDrain)?.min();
            println!(
                "  {:<14}{:>11}{:>12}{:>13}{:>14}",
                "drain", "min ms", "epilogue", "of launch", "vs cuBLASLt"
            );
            for (&arm, &min) in Arm::DRAINS.iter().zip(minima.iter()) {
                println!(
                    "  {:<14}{min:>11.4}{:>10.2} µs{:>12.1}%{:>14}",
                    arm.name(),
                    1e3 * (min - no_drain) / waves as f64,
                    100.0 * (min - no_drain) / min,
                    match theirs {
                        Some(theirs) => format!("{:.3}", theirs / min),
                        None => "—".to_string(),
                    },
                );
            }
            println!(
                "  {:<14}{no_drain:>11.4}       the control every row above subtracts",
                "no drain"
            );
        }
    }
    Ok(())
}

/// The warpgroup split, each arm against the `no drain` at **its own** launch.
///
/// Four warps cannot fill the shared-memory pipeline and lane ownership forbids a
/// fifth *within* a warpgroup — but warp 4 is warpgroup 1's warp 0 and owns the
/// same rows, so eight warps split the accumulator's **columns**. Two whole passes
/// each, because the effect being looked for is a few percent and one pass of that
/// is not a number.
fn warpgroup_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n6. the epilogue's warpgroups — four warps against eight, splitting the\n\
         accumulator's columns rather than its rows, at a byte-identical shared plan.\n\
         each arm subtracts the `no drain` at its own thread count, because the split\n\
         changes the launch geometry. `epilogue` is `whole - no drain` within a thread\n\
         count, in µs per tile per cluster; `of one wg` above 1.00 is the split ahead."
    );
    for (shape, variant) in HEADLINE {
        let (_, waves, _) = quantization(shape, variant);
        for pass in 1..=2 {
            println!("\n  {shape} — {}, pass {pass}", variant.name());
            let minima = arm_rows(context, shape, variant, &Arm::WARPGROUPS, Arm::Paired)?;
            println!(
                "  {:<16}{:>9}{:>11}{:>13}{:>12}",
                "warpgroups", "threads", "min ms", "epilogue", "of one wg"
            );
            for (name, whole, control, count) in [
                ("one", minima[0], minima[1], threads(ONE_WARPGROUP)),
                ("two", minima[2], minima[3], threads(TWO_WARPGROUPS)),
            ] {
                println!(
                    "  {name:<16}{count:>9}{whole:>11.4}{:>10.2} µs{:>12.3}",
                    1e3 * (whole - control) / waves as f64,
                    minima[0] / whole,
                );
            }
        }
    }
    Ok(())
}

/// The drain rungs that compute a right `C`, against the reference
/// `gemm_sol::check` uses, at the size it uses.
///
/// [`Arm::Paired`] and [`Arm::Wide`] rebuild the drain, so "it is faster" is worth
/// nothing until they are known to write what the shipped drain writes — and a
/// wrong register order out of a batched LDTM is exactly how that breaks without
/// faulting. [`Arm::PerIssue`] rides along so a failure can be attributed to the
/// batch rather than to this harness.
///
/// [`Arm::Pack16`] is deliberately absent: it computes a wrong `C` by
/// construction, which is stated on [`crate::gemm_sol::DRAIN_PACK16`] and is the
/// whole reason it is an oracle rather than a candidate.
pub fn check(context: &Arc<CudaContext>) -> Result<String, Box<dyn Error>> {
    let (m, n, k) = (1024usize, 1024usize, 512usize);
    let stream = context.default_stream();
    let ablated = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::from_host(
        &stream,
        &crate::gemm_sol::stage_f16(m, k, crate::gemm_sol::a_value),
    )?;
    let b = DeviceBuffer::from_host(
        &stream,
        &crate::gemm_sol::stage_f16(n, k, crate::gemm_sol::b_value),
    )?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;
    let (tiles_m, k_blocks) = ((m / 256) as u32, (k / K_TILE) as u32);

    let mut notes = Vec::new();
    for variant in [Variant::M256xN256, Variant::M512xN256] {
        let tiles_n = (n / variant.n_tile()) as u32;
        let group = default_group(m);
        for arm in Arm::EXACT_DRAINS {
            let config = LaunchConfig1D::new(
                RANKS * (m / variant.m_tile() * (n / variant.n_tile())) as u32,
                arm.threads(),
                variant.shared_bytes() as u32,
            );
            let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
            macro_rules! launch {
                ($prepare:ident, $call:ident) => {{
                    let prepared = ablated.$prepare(config)?;
                    unsafe {
                        ablated.$call(
                            &stream,
                            &prepared,
                            a_map.as_ptr(),
                            b_map.as_ptr(),
                            tiles_m,
                            tiles_n,
                            k_blocks,
                            group,
                            n as u32,
                            &mut c,
                        )?
                    };
                }};
            }
            match (variant, arm) {
                (Variant::M256xN256, Arm::Paired) => {
                    launch!(prepare_gemm_sol_m256_paired, gemm_sol_m256_paired)
                }
                (Variant::M256xN256, Arm::PerIssue) => {
                    launch!(prepare_gemm_sol_m256_per_issue, gemm_sol_m256_per_issue)
                }
                (Variant::M256xN256, Arm::Wg2) => {
                    launch!(prepare_gemm_sol_m256_wg2, gemm_sol_m256_wg2)
                }
                (Variant::M256xN256, _) => {
                    launch!(prepare_gemm_sol_m256_wide, gemm_sol_m256_wide)
                }
                (Variant::M512xN256, Arm::Paired) => {
                    launch!(prepare_gemm_sol_m512_paired, gemm_sol_m512_paired)
                }
                (Variant::M512xN256, Arm::PerIssue) => {
                    launch!(prepare_gemm_sol_m512_per_issue, gemm_sol_m512_per_issue)
                }
                (Variant::M512xN256, Arm::Wg2) => {
                    launch!(prepare_gemm_sol_m512_wg2, gemm_sol_m512_wg2)
                }
                (Variant::M512xN256, _) => {
                    launch!(prepare_gemm_sol_m512_wide, gemm_sol_m512_wide)
                }
                // The narrow entry has no arms, as `measure` says: the drain rungs
                // are instantiated for it through the shipped kernel's own body
                // and `examples`' gate covers them there.
                (Variant::M256xN128, _) => {
                    return Err("the narrow entry has no drain rungs of its own".into());
                }
            }
            stream.synchronize()?;
            let worst = crate::gemm_sol::check_output(&c.to_host_vec(&stream)?, m, n, k)
                .map_err(|error| format!("{} {}: {error}", variant.name(), arm.name()))?;
            notes.push(format!(
                "{} {} exact over {} outputs, worst |rel| {worst:.2e}",
                variant.name(),
                arm.name(),
                m * n,
            ));
        }
    }
    Ok(notes.join("; "))
}

fn box_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n2. the B box: the producer's instruction count at constant bytes. a rank's\n\
         half-panel of B is 128x64 — the same shape as A, which already arrives in one TMA —\n\
         and it arrives in two 64-row boxes only because one 64-row tensor map served all\n\
         three entries. `wide B` is that map built per entry instead. the traffic is\n\
         byte-identical, so a row that moves is issue rate and a row that does not is\n\
         bandwidth. taken twice round-robin, because that is the only way a few percent at\n\
         these shapes is quotable."
    );
    for pass in 1..=2 {
        println!("\n  pass {pass}");
        println!(
            "  {:<12}{:<18}{:>7}{:>11}{:>12}{:>10}{:>12}{:>9}",
            "arm", "shape", "tma/k", "min ms", "TFLOP/s", "of peak", "vs narrow", "spread"
        );
        for (shape, variant) in HEADLINE {
            let mut narrow_whole = None;
            let mut narrow_feed = None;
            for arm in Arm::BOXES {
                let timings = measure(context, shape, variant, arm)?;
                let rate = tflops(shape, timings.min());
                let reference = match arm {
                    Arm::Whole => {
                        narrow_whole = Some(rate);
                        None
                    }
                    Arm::FeedOnly => {
                        narrow_feed = Some(rate);
                        None
                    }
                    Arm::WideWhole => narrow_whole,
                    Arm::WideFeed => narrow_feed,
                    _ => None,
                };
                println!(
                    "  {:<12}{:<18}{:>7}{:>11.4}{:>12.1}{:>9.1}%{:>12}{:>8.1}%",
                    arm.name(),
                    shape.to_string(),
                    arm.tma_per_k_block(variant),
                    timings.min(),
                    rate,
                    100.0 * rate / PEAK,
                    reference
                        .map(|reference| format!("{:.3}", rate / reference))
                        .unwrap_or_else(|| "-".to_string()),
                    100.0 * timings.spread(),
                );
            }
        }
    }
    Ok(())
}

fn fold_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n3. the ring index, folded against computed. the MMA warp indexes both operand\n\
         rings and both stage barriers off `global_k = sequence * k_blocks + k`, a runtime\n\
         value, so every offset and both operand descriptors are runtime arithmetic inside\n\
         its issue stream. but `k_blocks` is a multiple of `STAGES` — the shape contract is\n\
         `k % 256 == 0` — so `global_k % STAGES == k % STAGES`, and the four positions of the\n\
         four-way unroll are always stages 0, 1, 2, 3. `whole` hands each position its stage\n\
         as a const; `runtime stage` is the spelling the port carried. same `C`, same bytes,\n\
         same barriers — the only difference is how much of the warp's stream is scalar.\n\n\
         the `mma only` pair is the one that isolates it: no TMA, no drain, no barrier wait,\n\
         so a difference there is the issue stream and nothing else. upstream states the same\n\
         fact and folds it with `#[unroll(4)]` over a loop-local `k_idx & 3`. taken twice\n\
         round-robin."
    );
    for pass in 1..=2 {
        println!("\n  pass {pass}");
        println!(
            "  {:<14}{:<18}{:>11}{:>12}{:>10}{:>13}{:>9}",
            "arm", "shape", "min ms", "TFLOP/s", "of peak", "vs runtime", "spread"
        );
        for (shape, variant) in HEADLINE {
            let mut reference = None;
            for arm in Arm::FOLDS {
                let timings = measure(context, shape, variant, arm)?;
                let rate = tflops(shape, timings.min());
                let against = match arm {
                    Arm::RuntimeWhole | Arm::RuntimeMma => {
                        reference = Some(rate);
                        None
                    }
                    _ => reference.take(),
                };
                println!(
                    "  {:<14}{:<18}{:>11.4}{:>12.1}{:>9.1}%{:>13}{:>8.1}%",
                    arm.name(),
                    shape.to_string(),
                    timings.min(),
                    rate,
                    100.0 * rate / PEAK,
                    against
                        .map(|against| format!("{:.3}", rate / against))
                        .unwrap_or_else(|| "-".to_string()),
                    100.0 * timings.spread(),
                );
            }
        }
    }
    Ok(())
}

fn depth_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n4. the depth ladder — `K` alone, so the tile grid and the waves are fixed and only\n\
         the arithmetic each tile amortizes over moves. the fit line under each rung set is\n\
         the per-tile constant and the per-K-block rate the ladder decomposes into, and the\n\
         percent it quotes is the K loop's own — the number this file exists for.\n\n\
         **every arm is laddered, not only `whole`.** table 1's arms are launches, and a\n\
         launch cannot tell a per-K-block cadence from a per-tile constant the arm happens to\n\
         carry — an arm's TMEM allocation, CLC query and retire are a per-tile cost like any\n\
         other, and at 64 k blocks one microsecond of it is 5.7 points of `of peak`. laddering\n\
         each arm is what makes its number a rate. `wide B` is here for the same reason: it is\n\
         the claim that the producer's instruction count moves the rate, tested on the rate."
    );
    for (base, variant) in HEADLINE {
        for arm in Arm::LADDER {
            depth_ladder(
                &format!("{} {}", variant.name(), arm.name()),
                base,
                variant,
                |shape| measure(context, shape, variant, arm),
            )?;
        }
    }
    Ok(())
}

/// The same ladder on the kernel this one is a port of, at the same tile.
#[cfg(feature = "gemm-sol-upstream")]
fn upstream_depth_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n5. the same ladder on upstream's own `gemm_sol_final`, byte for byte, at the same\n\
         `[256, 256]` tile and on this clock. the fit's percent-of-peak is the reference the\n\
         port's own K loop is short of: a difference here is the port's, and a shortfall\n\
         both share is the algorithm's at this tile."
    );
    depth_ladder(
        "upstream M256xN256",
        HEADLINE[0].0,
        Variant::M256xN256,
        |shape| crate::gemm_sol_upstream::bench(context, shape),
    )
}

#[cfg(not(feature = "gemm-sol-upstream"))]
fn upstream_depth_table(_: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n5. no upstream ladder: this build has no `gemm-sol-upstream` feature.\n\
         `modal_app.py::upstream_bench` is the entry point that turns it on."
    );
    Ok(())
}

/// Everything above, in one container — `bench sol-ablate`.
pub fn decompose(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    quantization_table();
    denominator(context)?;
    ablation_table(context)?;
    box_table(context)?;
    fold_table(context)?;
    chain_table(context)?;
    drain_table(context)?;
    warpgroup_table(context)?;
    depth_table(context)?;
    upstream_depth_table(context)?;
    println!(
        "\nno arm here is checked against a CPU reference and most of them cannot be.\n\
         `modal_app.py::regcount`'s opcode census is what says an arm removed the phase it\n\
         names; a row whose census disagrees is void and not a finding. the shipped\n\
         `gemm_sol_m256` and `gemm_sol_m512` are the same device body at `WHOLE`, so that\n\
         census also says the dial moved no instruction in the kernels that ship."
    );
    Ok(())
}
