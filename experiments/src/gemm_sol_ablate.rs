//! What `gemm_sol`'s time is made of — issue #138's three candidate costs, plus
//! the one that is arithmetic rather than a candidate.
//!
//! The port is 0.795 of cuBLASLt at 4096³, 0.873 at 8192³ and 0.946 at 16384³,
//! and the PR names three suspects: TMA feed cadence, MMA issue cadence, and
//! epilogue overlap. This file separates them, and it puts a fourth in front of
//! them because the fourth is free to derive and the other three are not.
//!
//! # The fourth: wave quantization
//!
//! The shared plan is 196 864 B for M256 and 229 632 B for M512, both above
//! half of a Blackwell SM's 233 472 B, so **one CTA per SM** — and a cluster is
//! two CTAs, so a wave on this 148-SM B200 is **74 clusters**. The launch is
//! exactly one cluster per output tile, so the ceiling any cadence is measured
//! against is
//!
//! ```text
//! clusters / (74 * ceil(clusters / 74))
//! ```
//!
//! At 4096³ with the M256 entry that is 256 clusters over four waves of 74 —
//! **0.865**, with the last wave 46% full. At 8192³ with M512 it is 512 over
//! seven — 0.988. The two shapes with a gap are therefore not the same problem,
//! and a number quoted against peak at 4096³ that does not divide by 0.865 is
//! attributing a scheduling loss to a cadence.
//!
//! [`sawtooth`] measures that rather than asserting it: `N` moves in tile
//! steps, the cluster count moves with it, and predicted efficiency runs
//! sawtooth against a work total that only rises. A shape with *more* work that
//! finishes in the same wall time is wave quantization, visible without a
//! counter.
//!
//! # The three: the ablation arms
//!
//! Each arm is [`crate::gemm_sol`]'s own device body at a different `ABLATE`
//! const, so no arm can drift from the kernel it decomposes — the shipped
//! kernels are the same text at `WHOLE`.
//!
//! | arm | producer | MMA warp | epilogue warps |
//! | --- | --- | --- | --- |
//! | `whole` | TMA | `tcgen05.mma` | LDTM → `stmatrix` → `st.global` |
//! | `no drain` | TMA | `tcgen05.mma` | handshake only |
//! | `issue only` | arrives on `load`, no TMA | `tcgen05.mma` | handshake only |
//! | `feed only` | TMA | waits and commits, no MMA | handshake only |
//!
//! **Every barrier survives every arm.** The `load`/`free` ring, the `full`/
//! `empty` accumulator handshake, the CLC cursor and the `ready` publication
//! are untouched, so an arm removes one *phase* and never the pipeline that
//! phase runs in. That is what makes `whole − no drain` the epilogue's cost
//! *including its overlap failure* rather than its cost in isolation.
//!
//! **These compute a wrong `C` on purpose** and are on no correctness gate,
//! like the third of this crate's entry points that already do. `regcount`'s
//! opcode census is the gate that matters here instead: `feed only` must show
//! zero in the `mma` column and the whole kernel's `tma` count, `issue only`
//! the reverse, and `no drain` zero across `ldtm`, `stmatrix` and the store
//! columns while keeping both of the others. An arm whose census does not read
//! that way did not remove what it names, and its number is void.
//!
//! The census says they do. `regcount` on this tree:
//!
//! ```text
//! kernel                     mma   ldtm    tma   stmatrix   st.g.v4   st.g.b32
//! gemm_sol_m256               16      2      3          2         1          4
//! gemm_sol_m256_nodrain       16      0      3          0         0          0
//! gemm_sol_m256_issue         16      0      0          0         0          0
//! gemm_sol_m256_feed           0      0      3          0         0          0
//! gemm_sol_m512               32      2      4          2         1          4
//! gemm_sol_m512_nodrain       32      0      4          0         0          0
//! gemm_sol_m512_issue         32      0      0          0         0          0
//! gemm_sol_m512_feed           0      0      4          0         0          0
//! ```
//!
//! `mbar.arrive` is 6 on every M256 arm and 8 on every M512 arm, which is the
//! claim that no arm dropped a barrier — and `gemm_sol_m256`/`gemm_sol_m512`
//! are opcode-identical between the two crates, so putting the dial on the
//! shipped body moved no instruction in the shipped kernel.
//!
//! Registers fall from 96 to 31-33 on the M256 arms. That cannot buy an
//! ablation occupancy it did not earn: at 192 threads even 104 registers admits
//! three CTAs, and both shared plans admit exactly one, so shared memory binds
//! residency in every arm and the register column is inert here.
//!
//! # What it found
//!
//! One B200 container, 5 warm-up and min of 30 per row. `whole` reproduced to
//! 1.5% across the four tables that measure it.
//!
//! The depth ladder is a straight line in `K` at both variants — the M512 one
//! to within 0.5% over an eightfold range — so the launch is exactly a per-tile
//! constant plus a per-`K`-block rate, and both fall out of it:
//!
//! | | per k block | at peak | of peak | per tile, fixed |
//! | --- | ---: | ---: | ---: | ---: |
//! | M256, m=n=4096 | 0.364 µs | 0.2759 µs | **75.8%** | **3.25 µs** |
//! | M512, m=n=8192 | 0.555 µs | 0.5518 µs | **99.4%** | **12.7 µs** |
//!
//! Fed back, the fit reproduces both launches to under 1%: `4 × (3.25 + 64 ×
//! 0.364) = 106.2 µs` against 105.5 measured, and `7 × (12.7 + 128 × 0.555) =
//! 586 µs` against 586.0.
//!
//! **The M512 K loop already runs at 99.4% of tensor-core peak.** Nothing in
//! its steady state is worth another look. Its whole deficit is the 12.7 µs it
//! pays per tile, and the ablation says what that is: the drain is 9.99 µs of
//! it. The M256 K loop is a different problem — it runs at 75.8%, and that is a
//! cadence rather than a constant.
//!
//! Subtracting the arms gives an additive budget for each launch, in
//! milliseconds and as a share of the whole:
//!
//! | term | 4096³ M256 | 8192³ M512 |
//! | --- | ---: | ---: |
//! | arithmetic at peak | 0.0611 (58.1%) | 0.4889 (83.4%) |
//! | wave quantization | 0.0095 (9.1%) | 0.0059 (1.0%) |
//! | epilogue, `whole − no drain` | **0.0207 (19.7%)** | **0.0699 (11.9%)** |
//! | TMA feed in situ, `no drain − issue only` | 0.0056 (5.3%) | 0.0082 (1.4%) |
//! | MMA issue cadence, the rest of `issue only` | 0.0083 (7.9%) | 0.0131 (2.2%) |
//! | measured | 0.1052 | 0.5860 |
//!
//! **The epilogue is the largest single loss at both shapes**, and at 8192³ it
//! is 11.9 of the 13.2 points between the launch and peak — the other two
//! suspects together are 3.6%. At 4096³ it is still the largest, but it is no
//! longer alone: quantization, MMA issue cadence and the feed are 9.1%, 7.9%
//! and 5.3%, and a fix that took only the epilogue would leave 22% on the
//! table there against 3.6% at 8192³.
//!
//! The drain costs the same *per byte of `C`* in both variants — 5.18 µs for
//! 131 072 B at M256 and 9.99 µs for 262 144 B at M512, which is 25.3 and
//! 26.2 GB/s per cluster. 74 clusters of that is 1.9 TB/s of stores against a
//! part that does 8, so the drain is not store-bandwidth-bound and not
//! variant-specific: it is the `tcgen05.ld` → `stmatrix` → `st.global` chain
//! itself, running at a fixed rate in four warps.
//!
//! **The feed is not the problem at either shape.** `feed only` moves the
//! launch's whole operand traffic — 1.07 GB at 4096³, 6.44 GB at 8192³ — in
//! 57% and 49% of the launch, which is 18.0 and 22.4 TB/s against 11.0 and 10.2
//! in situ. Those rates are far above the part's 8 TB/s of HBM, so most of it
//! is L2, which is what the `A` matrix fitting in L2 at these sizes predicts.
//!
//! # What this does not separate
//!
//! - **The epilogue's own cost from its failure to overlap.** `whole − no
//!   drain` is both. That it is *not* purely serial is visible anyway: the
//!   depth ladder's per-tile constant at M256 is 3.25 µs and the drain
//!   difference there is 5.18 µs per tile, so removing the drain also sped the
//!   K loop up — the epilogue is taking issue bandwidth from the MMA warp and
//!   not merely appending to it. Splitting the two needs a fifth arm this file
//!   does not have.
//! - **Barrier round-trip latency from tensor-core issue rate.** `issue only`
//!   keeps the whole `load`/`free` handshake, so the 24.2% the M256 K loop is
//!   short of peak is those two fused. The suggestive number is that M512
//!   issues two MMA chains per handshake and loses 0.6% where M256 issues one
//!   and loses 10.5% on the same arm — but that is one observation across two
//!   variants, not a separated term.
//! - **`feed only`'s ceiling from what the feed can do under contention.** That
//!   arm has no MMA reading the same shared memory, so 18-22 TB/s is an upper
//!   bound on the feed and not its in-situ rate. The in-situ number is the
//!   `no drain − issue only` row, which is the one quoted in the budget.
//! - **The 4096³ sawtooth's residual.** `corrected` is flat to 0.6% at M512 and
//!   drifts about 6% downward with `N` at M256, and that drift is not separated
//!   from the growing L2 footprint of `B`.
//! - **cuBLASLt's own schedule.** The `of ceiling` column divides our ratio by
//!   *our* wave quantization and the library has its own, unknown, so that
//!   column is not a residual. It reads 0.878 at both shapes, which the budget
//!   above says is a coincidence of two different budgets and not a constant.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract};

use kittens::global::GlobalLayout;
use kittens::shared::F16;

use crate::bench::{Shape, Timings, time};
use crate::gemm_sol::{
    ATile, BPanel, FEED_ONLY, ISSUE_ONLY, K_TILE, N_TILE, NO_DRAIN, THREADS, Variant, grid_for,
    m256_body, m512_body,
};

/// SMs on the B200 this file's arithmetic is for, and CTAs one of them holds at
/// either variant's shared plan. A cluster is two CTAs, so [`WAVE`] clusters
/// are resident at once and everything below follows from that.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 1;
const RANKS: u32 = 2;
/// Clusters resident at once — 74.
pub const WAVE: u32 = SMS * CTAS_PER_SM / RANKS;

const _: () = assert!(WAVE == 74);

#[cuda_module]
pub mod kernels {
    use super::*;

    /// TMA and every barrier, with the MMA and the drain gone.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`]. Computes no `C`.
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m256_body::<FEED_ONLY>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }

    /// The MMA and every barrier, with no global traffic at all.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_feed`].
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m256_body::<ISSUE_ONLY>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }

    /// The whole kernel except the drain.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_feed`].
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m256_body::<NO_DRAIN>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }

    /// [`gemm_sol_m256_feed`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m512`]. Computes no `C`.
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m512_body::<FEED_ONLY>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }

    /// [`gemm_sol_m256_issue`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_feed`].
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m512_body::<ISSUE_ONLY>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }

    /// [`gemm_sol_m256_nodrain`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_feed`].
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
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe { m512_body::<NO_DRAIN>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c) }
    }
}

/// One phase of the item, kept or removed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Whole,
    NoDrain,
    IssueOnly,
    FeedOnly,
}

impl Arm {
    const ALL: [Arm; 4] = [Arm::Whole, Arm::NoDrain, Arm::IssueOnly, Arm::FeedOnly];

    fn name(self) -> &'static str {
        match self {
            Arm::Whole => "whole",
            Arm::NoDrain => "no drain",
            Arm::IssueOnly => "issue only",
            Arm::FeedOnly => "feed only",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Arm::Whole => "the shipped kernel",
            Arm::NoDrain => "TMA + MMA + every barrier; the drain removed",
            Arm::IssueOnly => "MMA + every barrier; no global traffic at all",
            Arm::FeedOnly => "TMA + every barrier; no MMA, no drain",
        }
    }
}

/// Clusters the shape launches, and the fraction of the waves they fill.
///
/// The launch is one cluster per output tile and the device holds [`WAVE`] of
/// them, so this is the whole of it: no residency model, no measurement.
pub fn quantization(shape: Shape, variant: Variant) -> (u32, u32, f64) {
    let clusters = grid_for(shape, variant) / RANKS;
    let waves = clusters.div_ceil(WAVE);
    (clusters, waves, clusters as f64 / (waves * WAVE) as f64)
}

fn legal(shape: Shape, variant: Variant) -> Result<(), Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    if !m.is_multiple_of(variant.m_tile()) || !n.is_multiple_of(N_TILE) || !k.is_multiple_of(256) {
        return Err(format!("{shape} violates {}'s tile contract", variant.name()).into());
    }
    Ok(())
}

/// Time one arm at one shape.
///
/// Operands are zeroed rather than staged, exactly as [`crate::gemm_sol::bench`]
/// times the shipped kernel: nothing here is checked, three of the four arms
/// cannot be, and tensor cores do not take data-dependent time.
pub fn measure(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
    arm: Arm,
) -> Result<Timings, Box<dyn Error>> {
    legal(shape, variant)?;
    let Shape { m, n, k } = shape;
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
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;

    let config = LaunchConfig1D::new(
        grid_for(shape, variant),
        THREADS,
        variant.shared_bytes() as u32,
    );
    let (stream_ref, a_ptr, b_ptr) = (&stream, a_map.as_ptr(), b_map.as_ptr());
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let k_blocks = (k / K_TILE) as u32;
    let ldc = n as u32;

    // One spelling per arm, and the argument list written once: the arms differ
    // in the entry point they call and in nothing else, which is the claim the
    // table rests on.
    macro_rules! launcher {
        ($module:expr, $prepare:ident, $call:ident) => {{
            let module = $module;
            let prepared = module.$prepare(config)?;
            let launch: Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>> =
                Box::new(move |output| {
                    unsafe {
                        module.$call(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, ldc,
                            output,
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
    };

    launch_once(&mut c)?;
    stream.synchronize()?;
    let mut launch = || launch_once(&mut c);
    time(&stream, &mut launch)
}

const PEAK: f64 = 2250.0;

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// The two shapes the PR has a gap at, with the entry each one ships.
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

/// `N` in tile steps at fixed `M` and `K`, so the cluster count sweeps across
/// the wave boundary and predicted efficiency runs sawtooth.
///
/// The rows are chosen so consecutive ones straddle a boundary in both
/// directions: at M256 and `M = K = 4096` a tile of `N` is 16 clusters, so 74
/// does not divide the step and the fill walks 0.76 → 0.86 → 0.97 → 0.82 →
/// 1.00. If the measured rate follows that curve rather than the work, the
/// 4096³ deficit is the schedule and not a cadence — and the strongest single
/// row is the one where **more** work finishes in the same time.
const SAWTOOTH_M256: [usize; 6] = [3584, 4096, 4608, 4864, 5120, 9472];
const SAWTOOTH_M512: [usize; 5] = [7168, 8192, 8704, 9216, 9472];

/// `K` at fixed `M` and `N`: the tile grid, the waves, the `C` traffic and the
/// epilogue's total cost all hold, and only the amount of arithmetic each tile
/// amortizes its fixed cost over moves. A rate that climbs with `K` here is a
/// per-tile cost — prologue, drain, CLC query — and not a steady-state cadence.
const DEPTHS: [usize; 4] = [2048, 4096, 8192, 16384];

fn quantization_table() {
    println!(
        "\nwave quantization, derived and not measured — one cluster per output tile,\n\
         {WAVE} clusters resident (148 SMs, one CTA each at either shared plan, two\n\
         CTAs a cluster). `ceiling` is the most of peak the shape can reach however\n\
         perfect the cadence is."
    );
    println!(
        "  {:<18}{:>10}{:>10}{:>8}{:>12}{:>14}",
        "shape", "variant", "clusters", "waves", "last wave", "ceiling"
    );
    for (shape, variant) in HEADLINE {
        let (clusters, waves, efficiency) = quantization(shape, variant);
        let last = clusters - WAVE * (waves - 1);
        println!(
            "  {:<18}{:>10}{:>10}{:>8}{:>11.0}%{:>13.3}",
            shape.to_string(),
            variant.name(),
            clusters,
            waves,
            100.0 * last as f64 / WAVE as f64,
            efficiency,
        );
    }
}

fn ablation_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe ablation arms — one phase of the item removed per row, every barrier kept.\n\
         `of whole` is this arm's throughput over the whole kernel's, so a row near 1.00\n\
         is a phase that was already free and a row far above it is a phase that is not.\n\
         Three of the four compute no `C`; see this module's docs and the opcode census.\n\
         `of peak` is only meaningful on the `whole` row — an arm that skipped the\n\
         arithmetic still gets credited with it, which is why `feed only` reads above 100%."
    );
    for (shape, variant) in HEADLINE {
        let (clusters, waves, ceiling) = quantization(shape, variant);
        println!(
            "\n  {shape} — {}, {clusters} clusters over {waves} waves, ceiling {ceiling:.3}",
            variant.name()
        );
        println!(
            "  {:<12}{:>11}{:>11}{:>12}{:>11}{:>11}  {}",
            "arm", "min ms", "median ms", "TFLOP/s", "of peak", "of whole", "what it runs"
        );
        let mut whole = f64::NAN;
        for arm in Arm::ALL {
            let timings = measure(context, shape, variant, arm)?;
            if arm == Arm::Whole {
                whole = timings.min();
            }
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<12}{:>11.4}{:>11.4}{:>12.1}{:>10.1}%{:>11.2}  {}",
                arm.name(),
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

fn sawtooth(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe wave sawtooth — `N` in tile steps, `M` and `K` held. `predicted` is the\n\
         derived ceiling above; `corrected` is the measured rate divided by it, which is\n\
         what the kernel does per resident cluster and should be flat if quantization is\n\
         the whole of the shape dependence."
    );
    for (base, columns, variant) in [
        (
            Shape {
                m: 4096,
                n: 0,
                k: 4096,
            },
            &SAWTOOTH_M256[..],
            Variant::M256xN256,
        ),
        (
            Shape {
                m: 8192,
                n: 0,
                k: 8192,
            },
            &SAWTOOTH_M512[..],
            Variant::M512xN256,
        ),
    ] {
        println!("\n  {} at m={}, k={}", variant.name(), base.m, base.k);
        println!(
            "  {:<18}{:>10}{:>7}{:>12}{:>11}{:>12}{:>12}",
            "shape", "clusters", "waves", "predicted", "min ms", "TFLOP/s", "corrected"
        );
        for &n in columns {
            let shape = Shape { n, ..base };
            let (clusters, waves, predicted) = quantization(shape, variant);
            let timings = measure(context, shape, variant, Arm::Whole)?;
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<18}{:>10}{:>7}{:>12.3}{:>11.4}{:>12.1}{:>12.1}",
                shape.to_string(),
                clusters,
                waves,
                predicted,
                timings.min(),
                rate,
                rate / predicted,
            );
        }
    }
    Ok(())
}

fn depth(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe depth ladder — `K` alone, so the tile grid and the waves are fixed and only\n\
         the arithmetic each tile amortizes over moves. `per tile` is the launch's\n\
         milliseconds divided by the items a critical-path cluster walks, and a straight\n\
         line through it in `k blocks` reads off the per-tile constant directly."
    );
    for (base, variant) in HEADLINE {
        println!("\n  {} at m={}, n={}", variant.name(), base.m, base.n);
        println!(
            "  {:<18}{:>9}{:>10}{:>11}{:>12}{:>11}{:>13}",
            "shape", "k blocks", "waves", "min ms", "TFLOP/s", "of peak", "ms per tile"
        );
        for k in DEPTHS {
            let shape = Shape { k, ..base };
            let (_, waves, _) = quantization(shape, variant);
            let timings = measure(context, shape, variant, Arm::Whole)?;
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<18}{:>9}{:>10}{:>11.4}{:>12.1}{:>10.1}%{:>13.5}",
                shape.to_string(),
                k / K_TILE,
                waves,
                timings.min(),
                rate,
                100.0 * rate / PEAK,
                timings.min() / waves as f64,
            );
        }
    }
    Ok(())
}

fn denominator(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let Some(baseline) = crate::bench::CUBLASLT_F16 else {
        println!("\nno cuBLASLt column: this build has no `cublas` feature.");
        return Ok(());
    };
    println!(
        "\nthe denominator, in this container — {} ({}). `of ceiling` is our ratio\n\
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

/// Everything above, in one container — `bench sol`.
pub fn decompose(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    quantization_table();
    denominator(context)?;
    ablation_table(context)?;
    sawtooth(context)?;
    depth(context)?;
    println!(
        "\nno arm here is checked against a CPU reference and three of them cannot be.\n\
         `modal_app.py::regcount`'s opcode census is what says an arm removed the phase\n\
         it names; a row whose census disagrees is void and not a finding."
    );
    Ok(())
}
