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
//! `mma only` against `runtime mma` is 1.018 / 1.001 / 1.005 / 0.998, null. It is
//! in the barrier poll loop, which is the one place the folded address is
//! re-materialized per iteration.
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
    ATile, B_BOX, BLOCK_N, BPanel, FEED_ONLY, HALF_N, ISSUE_ONLY, MMA_ONLY, NO_DRAIN,
    SMALL_RINGS_END, THREADS, Variant, WHOLE, WIDE_B_BOX, WideBPanel, clusters, default_group,
    large_body, small_body,
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, NO_DRAIN, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, ISSUE_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, FEED_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, MMA_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, WIDE_B_BOX, BLOCK_N, SMALL_RINGS_END, WHOLE, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, WIDE_B_BOX, BLOCK_N, SMALL_RINGS_END, FEED_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, WHOLE, false>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            small_body::<BLOCK_N, HALF_N, B_BOX, BLOCK_N, SMALL_RINGS_END, MMA_ONLY, false>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, NO_DRAIN, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, ISSUE_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, FEED_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, MMA_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<WIDE_B_BOX, WHOLE, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<WIDE_B_BOX, FEED_ONLY, true>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, WHOLE, false>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
            large_body::<B_BOX, MMA_ONLY, false>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c,
            )
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
        THREADS,
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
