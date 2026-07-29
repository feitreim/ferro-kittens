//! Timed runs of the examples that run, at sizes chosen to cross a regime.
//!
//! Three rules this file exists to enforce, in the order they matter:
//!
//! 1. **A number only ever comes out of a checked run.** An example's timing
//!    entry point is not reachable except through its own CPU reference — see
//!    [`gemm::run`](crate::gemm) — so there is no path here that reports
//!    throughput for a kernel that computed the wrong answer. A harness that
//!    would happily print TFLOP/s for garbage is a trap you fall into once, and
//!    the number outlives the run that produced it.
//! 2. **The metric follows what the kernel is bound by.** [`Bound`] is stated
//!    per example rather than assumed by the harness: a GEMM's `2·M·N·K` is
//!    real work and TFLOP/s describes it, while a normalization kernel does a
//!    couple of flops per element it reads and a FLOP/s figure for one is a
//!    large-looking number that describes nothing.
//! 3. **The clock is the device's.** CUDA events either side of the launch, not
//!    wall clock around the driver call.
//! 4. **A rate wants a denominator.** `1069 TFLOP/s` is a number nobody can
//!    act on without first re-deriving what it should have been; `0.52 of
//!    cuBLASLt at the same shape on the same device` is one they can. Where a
//!    tuned library implements the same problem, [`Baseline`] puts it in the
//!    table — **through this same [`time`], on the same stream, after the same
//!    warm-up, and having passed the same CPU reference** — so the difference
//!    between the two figures is the two kernels and not the two harnesses.
//!    Rule 1 has no exception for it: the baseline is checked before it is
//!    timed, exactly as ours is, because the way a baseline goes wrong is by
//!    computing a *different* GEMM quickly.
//!
//! Every example that is *not* in [`cases`] is in [`SKIPPED`] with the reason,
//! so a missing row never leaves a reader guessing whether a kernel is slow or
//! simply not written yet.

use std::error::Error;
use std::process::ExitCode;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};

use crate::{gemm, layernorm, softmax};

/// Launches discarded before timing begins. The first pays module load and the
/// first launch of a given shape pays the driver's own setup for it; neither is
/// representative of the next thousand.
const WARMUP: usize = 5;
/// Timed launches per size.
const ITERATIONS: usize = 30;

/// Dense bf16 tensor-core peak for one B200. NVIDIA's HGX B200 page lists
/// 36 PFLOPS FP16/BF16 for the 8-GPU board, footnoted "Sparse. Dense is ½
/// sparse spec shown" — 36/8/2 = 2.25 PFLOP/s dense per GPU.
/// <https://www.nvidia.com/en-us/data-center/hgx/>
const B200_BF16_DENSE_TFLOPS: f64 = 2250.0;

/// What a kernel is bound by, and therefore which number is worth printing.
#[derive(Clone, Copy)]
pub enum Bound {
    /// Work is flops; reported as TFLOP/s.
    Compute,
    /// Work is bytes read plus bytes written; reported as GB/s.
    Memory,
}

impl Bound {
    fn unit(self) -> &'static str {
        match self {
            Bound::Compute => "TFLOP/s",
            Bound::Memory => "GB/s",
        }
    }

    fn rate(self, work: f64, seconds: f64) -> f64 {
        match self {
            Bound::Compute => work / seconds / 1e12,
            Bound::Memory => work / seconds / 1e9,
        }
    }

    /// The denominator of a `% of peak` column, where one is defensible. An
    /// unsourced peak is worse than no ratio, so a memory-bound example prints
    /// no ratio until someone lands a cited HBM figure next to the one above.
    fn peak(self) -> Option<f64> {
        match self {
            Bound::Compute => Some(B200_BF16_DENSE_TFLOPS),
            Bound::Memory => None,
        }
    }
}

/// A problem size. A kernel with no reduction depth leaves `k` at 1 and prints
/// as `m x n`.
#[derive(Clone, Copy)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Shape { m, n, k } = *self;
        // `pad` rather than `write!`, so the table's column width reaches it.
        out.pad(&if k == 1 {
            format!("{m}x{n}")
        } else {
            format!("{m}x{n}x{k}")
        })
    }
}

/// Per-launch kernel times in milliseconds, sorted.
///
/// The headline is the minimum: it is the least noise-contaminated estimate of
/// what the kernel can do, since every source of error on a quiet device adds
/// time and none subtracts it. The median and maximum are printed beside it
/// because a gap between them is a finding — a device sharing work, a clock
/// dropping, a first-touch cost that never amortizes — and the table's job is
/// to surface that, not to hide it behind one number.
pub struct Timings(Vec<f64>);

impl Timings {
    /// The headline. `pub` because [`gemm::compare`](crate::gemm::compare)
    /// puts both schedulers' headlines in one row of its own table.
    pub fn min(&self) -> f64 {
        self.0[0]
    }

    fn median(&self) -> f64 {
        self.0[self.0.len() / 2]
    }

    fn max(&self) -> f64 {
        self.0[self.0.len() - 1]
    }
}

/// Time `launch` with CUDA events recorded either side of it on `stream`.
///
/// The events measure the kernel's own span on the device, which is the thing
/// under test; wall clock around the call would measure the driver's launch
/// path and the host's scheduling as well, and at the small end of the sweep
/// those are the same order as the kernel.
///
/// This is the argument [`gemm::bench`] passes to the checked-run entry point,
/// which is what makes "verified, then timed" the only order that can happen.
pub fn time(
    stream: &CudaStream,
    launch: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<Timings, Box<dyn Error>> {
    // Progress goes to stderr, not into the table: a sweep is minutes long and
    // a reader watching it should be able to tell a slow size from a stuck one.
    // Reaching this line at all is the check having passed.
    eprintln!("  checked; {WARMUP} warm-up then {ITERATIONS} timed launches");
    let timing_enabled = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let start = stream.context().new_event(timing_enabled)?;
    let stop = stream.context().new_event(timing_enabled)?;

    for _ in 0..WARMUP {
        launch()?;
    }
    stream.synchronize()?;

    let mut milliseconds = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        start.record(stream)?;
        launch()?;
        stop.record(stream)?;
        milliseconds.push(start.elapsed_ms(&stop)? as f64);
    }
    milliseconds.sort_by(f64::total_cmp);
    Ok(Timings(milliseconds))
}

/// An example's verify-then-time entry point: check at a size against that
/// kernel's own CPU reference, and only then return the clock.
///
/// Named rather than spelled out at each use because it is the harness'
/// central contract — the signature *is* rule 1 at the top of this file, and
/// the two places that hold one should visibly hold the same thing.
type Bench = fn(&Arc<CudaContext>, Shape) -> Result<Timings, Box<dyn Error>>;

/// The same contract, plus the identity of the algorithm the library chose —
/// which ours has no analogue of, since ours only has the one.
pub type BaselineBench = fn(&Arc<CudaContext>, Shape) -> Result<(Timings, String), Box<dyn Error>>;

/// One example this harness can run. Adding the next kernel is a `Case` and
/// nothing else: the sizes, the work each size does, the grid it launches, and
/// the verify-then-time entry point in the kernel's own module.
struct Case {
    name: &'static str,
    bound: Bound,
    sizes: &'static [Shape],
    /// Flops or bytes at a size, per [`Bound`].
    work: fn(Shape) -> f64,
    /// Blocks the launch asks for — the column that explains the small end of
    /// the sweep, where there are fewer of them than the device has SMs.
    blocks: fn(Shape) -> u32,
    bench: Bench,
    /// A tuned library measured at the same sizes, where the build has one —
    /// the denominator, per [`CUBLASLT`].
    baseline: Option<Baseline>,
}

/// A vendor implementation of the same problem, timed through this same
/// harness so the comparison is between two kernels rather than two harnesses.
///
/// The point of a baseline as a *column* rather than a one-off measurement is
/// that it does not decay. An absolute like "1069 TFLOP/s" has to be
/// re-contextualized by every reader who meets it later; a ratio against the
/// library on the same device, the same shape and the same day carries its own
/// context, and every future change to the kernel gets one for free.
#[derive(Clone, Copy)]
pub struct Baseline {
    pub name: &'static str,
    /// The library's version, printed above the table. A baseline whose
    /// version is not stated cannot be reproduced later.
    pub about: fn() -> String,
    /// The same contract [`Case::bench`] has — checked, then timed — plus the
    /// identity of the algorithm the library chose.
    pub bench: BaselineBench,
}

/// The GEMM's denominator, behind the off-by-default `cublas` feature (#92).
///
/// Off by default so tier 1 CI, which has no CUDA toolkit and therefore no
/// `libcublasLt.so`, is unaffected and this crate still builds for anyone
/// without one. `modal run modal_app.py::bench` turns it on; see
/// [`crate::cublaslt`] for the configuration and for what is and is not fair
/// about the comparison.
#[cfg(feature = "cublas")]
pub const CUBLASLT: Option<Baseline> = Some(Baseline {
    name: "cuBLASLt",
    about: crate::cublaslt::about,
    bench: crate::cublaslt::bench,
});
#[cfg(not(feature = "cublas"))]
pub const CUBLASLT: Option<Baseline> = None;

/// Sizes for `gemm`, picked to cross a regime rather than to be large.
///
/// A cluster owns a `256 x 128` tile of `C`, so the first row is the smallest
/// legal problem there is: one cluster, two CTAs, on a device with well over a
/// hundred SMs. That end measures launch and drain and nothing else. The second
/// is the size `modal_app.py::examples` checks, kept so the two runs share a
/// point. From there each row roughly quadruples the cluster count: 2048 is the
/// first size past a full wave of CTAs, and 4096 and 8192 are where the K
/// pipeline should dominate and the curve flatten at this kernel's ceiling.
///
/// **16384³ is here because it did not flatten** (#86). The persistent grid
/// holds 222 clusters and walks `ceil(tiles / 222)` waves of them, so the last
/// wave is ragged and the *fraction of the last wave that is full* is a pure
/// function of the shape: 92.3% at 8192³ against **99.7%** at 16384³. Dividing
/// the measured rate by that fraction gives 660 / 945 / 1171 TFLOP/s at
/// 2048³ / 4096³ / 8192³ — which is the arithmetic that says 16384³ should
/// reach ~1171 if ragged waves were the whole story, and the reason to run it
/// is that the arithmetic can be wrong.
///
/// It is not wrong, and it is also not the ceiling. 16384³ measures **1178.5**
/// and **1117.3 TFLOP/s** in two full runs of this tree — straddling the
/// prediction, on the least reproducible row in the file, where 8192³ repeats
/// to 0.02%. So the climb does stop where ragged waves say it stops. What that
/// does *not* settle is what the kernel could do, and [`GEMM_DEPTH_SIZES`]
/// below reaches 1369.9 TFLOP/s with no constant changed.
///
/// It costs 1.6 GiB of device memory — `A` and `B` are 512 MiB of bf16 each and
/// `C` is 512 MiB of bf16 since #108, where it was 1 GiB of fp32 — against the
/// 180 GiB a B200 carries, so the size is bounded by host staging time and not
/// by the device.
const GEMM_SIZES: &[Shape] = &[
    // The smallest shape that is **one cluster running exactly one item**, and
    // therefore §7's "the small-size floor is one item boundary" row. It was
    // `256x128x256` through #102; #87 widened the pair tile to `[256, 256]`, so
    // `n = 128` no longer divides the tiling and the one-item shape is now this
    // one. The row measures the same thing — a single item's cost, undivided —
    // and the absolute microseconds are not comparable across that change,
    // because the item it is one of has twice the columns.
    Shape {
        m: 256,
        n: 256,
        k: 256,
    },
    Shape {
        m: 512,
        n: 256,
        k: 256,
    },
    Shape {
        m: 1024,
        n: 1024,
        k: 1024,
    },
    Shape {
        m: 2048,
        n: 2048,
        k: 2048,
    },
    Shape {
        m: 4096,
        n: 4096,
        k: 4096,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
    Shape {
        m: 16384,
        n: 16384,
        k: 16384,
    },
];

/// Five shapes that do **exactly the same work** and differ only in how much of
/// it L2 can hold — #86's bandwidth question, asked without a profiler.
///
/// A cluster owns `256` rows of `A` and `128` columns of `B`, so a tile reads
/// `384 · K` elements and computes `2 · 256 · 128 · K` flops whatever the
/// problem is. **Arithmetic intensity is a property of the tile, not of the
/// shape** — 85.3 FLOP/byte, always — so every shape below requests the same
/// 12.9 GB of operands from the memory system and writes the same 134 MB of
/// `C` (268 MB before #108 made the output bf16; it is the same in every row
/// either way, which is all this table needs of it). Holding `M · N` and `K` fixed at 8192³'s values holds all of that
/// fixed: 2048 tiles, 10 waves of 222 clusters, 92.3% wave efficiency,
/// 1.1 TFLOP, one grid of 444 blocks. Only the aspect ratio moves.
///
/// What the aspect ratio moves is the *working set*. The item map is row-major,
/// so the 222 clusters of a wave sit on `ceil(222 / tiles_n)` rows of tiles and
/// span `min(222 · 128, N)` columns; they march through K together, and at one
/// K block the bytes they collectively touch are
/// `(rows_of_A + columns_of_B) · 64 · 2`:
///
/// | shape | tiles_n | distinct per K block | reuse | implied HBM read |
/// | --- | ---: | ---: | ---: | ---: |
/// | 32768 x 2048 | 16 | 0.72 MB | 15.1x | 0.85 GB |
/// | 16384 x 4096 | 32 | 0.75 MB | 14.5x | 0.89 GB |
/// | 8192 x 8192 | 64 | 1.18 MB | 9.2x | 1.39 GB |
/// | 4096 x 16384 | 128 | 2.16 MB | 5.0x | 2.55 GB |
/// | 2048 x 32768 | 256 | 3.67 MB | 3.0x | 4.34 GB |
///
/// So the traffic that has to come off HBM varies **5.1x across rows that are
/// otherwise the same run**, while the traffic that has to come off L2 does not
/// vary at all. If the kernel is bound by what HBM can deliver, this table is a
/// ramp. If it is bound by what L2 or the SMs can absorb, it is flat. Those are
/// the two answers #86 says want opposite fixes, and no third thing this
/// benchmark can vary distinguishes them so cleanly.
///
/// The last two rows are worth reading against each other on their own:
/// `32768 x 2048` and `2048 x 32768` are transposes, with identical *total*
/// footprints of 570 MB, and they differ only in which operand the row-major
/// walk re-reads. A gap between those two is not about how much data there is.
///
/// `4096 x 16384` is here for a second reason: its 2.16 MB per K block is the
/// same working set 16384³ has, at 8192³'s wave structure. If the flattening
/// above is a footprint effect, this row shows it with the quantization held
/// still.
pub const GEMM_FOOTPRINT_SIZES: &[Shape] = &[
    Shape {
        m: 32768,
        n: 2048,
        k: 8192,
    },
    Shape {
        m: 16384,
        n: 4096,
        k: 8192,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
    Shape {
        m: 4096,
        n: 16384,
        k: 8192,
    },
    Shape {
        m: 2048,
        n: 32768,
        k: 8192,
    },
];

/// One square output, four reduction depths — what an *item boundary* costs.
///
/// `M` and `N` are 8192 throughout, so every row has the same 2048 tiles, the
/// same 10 waves over 222 clusters, the same 92.3% wave efficiency, the same
/// grid of 444 blocks, and writes the same 134 MB of `C`. Every row also has
/// the same operand traffic *per flop* and the same L2 working set per K block,
/// because neither depends on `K`. What changes is how much arithmetic sits
/// between one boundary and the next: 8, 32, 128 and 512 K blocks.
///
/// The boundary is not free and #83 said so without pricing it. `lcf` folds the
/// epilogue into the item, so a cluster finishing a tile drains its accumulator
/// through LDTM, writes 64 KiB of bf16 per CTA to global memory, and only then
/// issues the next tile's first `STAGES` loads — the store and the refill
/// cannot cross, which is exactly what #15's `lcsf` is for. That cost is per
/// *tile* and constant, so as a fraction of the run it goes as `1/K`, and a
/// sweep over `K` at a fixed tile count is the only way to see it separately
/// from everything else this kernel does.
///
/// Read the rows as a ramp with an asymptote: the top is what the steady-state
/// pipeline can do, and the distance from a row to it is what that row's
/// boundaries cost. It is also the honest version of the "small size" question
/// — the smallest rows of [`GEMM_SIZES`] confound a short reduction with a grid
/// far under one wave, and this separates them by holding the grid full.
///
/// **One thing it does not hold fixed, and the fit has to answer for it.** `K`
/// sets the operand footprint: `A` and `B` are 8 MiB each in the first row and
/// **512 MiB each in the last**, so the sweep crosses L2 between the second row
/// and the third. That is visible in the numbers — the marginal cost of a K
/// block rises 0.378 → 0.503 → 0.569 µs across the three intervals, which is
/// the *opposite* of what a pipeline failing to fill would do, and it makes the
/// curve convex. So a straight line through all four rows overstates the
/// intercept, and `examples/README.md` §7 reports the fixed cost as a range
/// over point selections rather than as one number off one fit.
pub const GEMM_DEPTH_SIZES: &[Shape] = &[
    Shape {
        m: 8192,
        n: 8192,
        k: 512,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 2048,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 32768,
    },
];

/// Sizes for `softmax`, as `rows x columns x planes`. The columns are the
/// kernel's own 128 and the planes stay at the two the correctness run uses, so
/// `blockIdx.y` is under test at every size; the rows are what sweeps. One CTA
/// owns 128 rows, so the first row is two CTAs, and the last moves half a
/// gigabyte — far enough past the launch floor for the number to be bandwidth
/// and not scheduling.
const SOFTMAX_SIZES: &[Shape] = &[
    Shape {
        m: 128,
        n: 128,
        k: 2,
    },
    Shape {
        m: 512,
        n: 128,
        k: 2,
    },
    Shape {
        m: 4096,
        n: 128,
        k: 2,
    },
    Shape {
        m: 32768,
        n: 128,
        k: 2,
    },
    Shape {
        m: 262144,
        n: 128,
        k: 2,
    },
    Shape {
        m: 524288,
        n: 128,
        k: 2,
    },
];

/// Sizes for `layernorm`, as `rows x columns`. It has no plane coordinate — a
/// CTA takes its whole position from `blockIdx.x` — so the rows carry what
/// `softmax`'s rows and planes carry together, and they are deliberately twice
/// `SOFTMAX_SIZES`' so the two sweeps land on the *same block counts* and the
/// same bytes moved. A row of this table and the row above it are comparable
/// numbers, which is the point: the two kernels differ in their band and in
/// two parameter vectors, and nothing else.
const LAYERNORM_SIZES: &[Shape] = &[
    Shape {
        m: 256,
        n: 128,
        k: 1,
    },
    Shape {
        m: 1024,
        n: 128,
        k: 1,
    },
    Shape {
        m: 8192,
        n: 128,
        k: 1,
    },
    Shape {
        m: 65536,
        n: 128,
        k: 1,
    },
    Shape {
        m: 524288,
        n: 128,
        k: 1,
    },
    Shape {
        m: 1048576,
        n: 128,
        k: 1,
    },
];

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "gemm",
            bound: Bound::Compute,
            sizes: GEMM_SIZES,
            work: |shape| 2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64,
            blocks: |shape| gemm::grid(shape.m, shape.n),
            bench: gemm::bench,
            baseline: CUBLASLT,
        },
        Case {
            name: "gemm-footprint",
            bound: Bound::Compute,
            sizes: GEMM_FOOTPRINT_SIZES,
            work: |shape| 2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64,
            blocks: |shape| gemm::grid(shape.m, shape.n),
            bench: gemm::bench,
            // These two exist to compare the kernel against *itself* with one
            // variable moved, and a baseline is neither needed nor free: each
            // extra row re-stages its operands on the host, which at these
            // sizes is the long pole. `CUBLASLT` here is a one-word change for
            // whoever wants to know whether the library loses 23% to aspect
            // ratio too — an interesting question, and not this issue's.
            baseline: None,
        },
        Case {
            name: "gemm-depth",
            bound: Bound::Compute,
            sizes: GEMM_DEPTH_SIZES,
            work: |shape| 2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64,
            blocks: |shape| gemm::grid(shape.m, shape.n),
            bench: gemm::bench,
            baseline: None,
        },
        Case {
            name: "softmax",
            bound: Bound::Memory,
            // Every element is read once and written once, both as bf16, and
            // read twice more out of shared memory — the denominator counts
            // the global traffic, which is what an HBM figure would divide.
            // Bytes and not flops because the arithmetic is two `ex2.approx`
            // and a multiply per element; #76 is what that costs, and it is
            // not something a FLOP/s number would have described.
            work: |shape| 2.0 * 2.0 * (shape.m * shape.n * shape.k) as f64,
            sizes: SOFTMAX_SIZES,
            blocks: |shape| softmax::grid(shape.m, shape.k),
            bench: softmax::bench,
            baseline: None,
        },
        Case {
            name: "layernorm",
            bound: Bound::Memory,
            // Every element is read once and written once, both as bf16. The
            // two parameter vectors are `2 * COLUMNS` elements per CTA against
            // `ROWS * COLUMNS` of activations — under 2% and not counted, which
            // is the honest direction to round a denominator.
            work: |shape| 2.0 * 2.0 * (shape.m * shape.n) as f64,
            sizes: LAYERNORM_SIZES,
            blocks: |shape| layernorm::grid(shape.m),
            bench: layernorm::bench,
            baseline: None,
        },
    ]
}

/// Examples with no row above, and why. There is no path through this file
/// that times a launch it did not first check — so a missing reference is what
/// keeps one out, not its speed.
const SKIPPED: &[(&str, &str)] = &[("flash_forward", "no launcher yet — would report TFLOP/s")];

/// The ratio table, and the caveats that make it readable.
///
/// Both minimum *and* median are printed for both sides, and each gets its own
/// ratio column, because #86 measured 16384³ at 1117 and 1179 TFLOP/s in two
/// runs of the same tree — 5.5% apart, with a median-to-minimum gap of 11%
/// where every other size repeats to about 1%. A single minimum at that size
/// could flatter either party, and two ratios that disagree say so out loud.
///
/// The caveats are printed under the table rather than left to the module docs
/// on purpose: this output gets pasted into issues, and a ratio travelling
/// without them is a ratio that will be over-read.
fn compare(case: &Case, baseline: Baseline, rows: &[(Shape, Timings, Timings, String)]) {
    if rows.is_empty() {
        return;
    }
    let rate =
        |shape: Shape, milliseconds: f64| case.bound.rate((case.work)(shape), milliseconds / 1e3);

    println!(
        "\n{} against {} ({}) — same process, same stream, same clock, same \
         {WARMUP} warm-up and min of {ITERATIONS}, both checked first",
        case.name,
        baseline.name,
        (baseline.about)()
    );
    println!(
        "{:<16}{:>12}{:>12}{:>12}{:>12}{:>12}{:>13}{:>14}{:>12}",
        "shape",
        "ours min",
        "ours med",
        "theirs min",
        "theirs med",
        "ours TF/s",
        "theirs TF/s",
        "ours/theirs",
        "at median"
    );
    for (shape, ours, theirs, _) in rows {
        println!(
            "{:<16}{:>12.4}{:>12.4}{:>12.4}{:>12.4}{:>12.1}{:>13.1}{:>14.3}{:>12.3}",
            shape,
            ours.min(),
            ours.median(),
            theirs.min(),
            theirs.median(),
            rate(*shape, ours.min()),
            rate(*shape, theirs.min()),
            theirs.min() / ours.min(),
            theirs.median() / ours.median(),
        );
    }

    println!(
        "\nthe last two columns are our throughput divided by theirs — equivalently their\n\
         milliseconds divided by ours — so above 1.00 is us ahead and 0.50 is half their\n\
         rate. milliseconds are the kernel's own span between CUDA events on the same\n\
         stream, not wall clock around the driver call."
    );
    println!(
        "both sides read byte-identical operands — plain packed bf16 [m, k] and [n, k],\n\
         K contiguous. ours reads them through a TMA tensor map and {0} through its own\n\
         matrix layouts; both descriptors are built once, outside the clock, as is every\n\
         allocation on both sides. no operand is rearranged for either party.",
        baseline.name
    );
    println!(
        "{0} is handed a 32 MiB workspace, allocated before the warm-up; ours uses none.",
        baseline.name
    );
    println!(
        "ours computes only this form: both operands K-major, alpha 1, beta 0, no epilogue,\n\
         and m % 256 == n % 128 == k % 64 == 0. {0} takes any of that, so a like-for-like\n\
         rate against a library that is also general reads in our favour.",
        baseline.name
    );
    println!(
        "the algorithm {0}'s heuristic chose, so the baseline can be reproduced:",
        baseline.name
    );
    for (shape, _, _, algorithm) in rows {
        // 18 rather than 16: `16384x16384x16384` is seventeen characters, and
        // in the table the next column is right-aligned and absorbs it.
        println!("  {shape:<18}{algorithm}");
    }
}

fn report(context: &Arc<CudaContext>, case: &Case, sizes: &[Shape]) -> usize {
    let bound = case.bound;
    println!(
        "\n{} — {}, over {ITERATIONS} timed launches after {WARMUP} warm-up",
        case.name,
        match bound {
            Bound::Compute => "compute-bound, so the metric is flops per second",
            Bound::Memory => "memory-bound, so the metric is bytes moved per second",
        }
    );
    println!(
        "{:<16}{:>8}{:>11}{:>11}{:>11}{:>12}{}",
        "shape",
        "blocks",
        "min ms",
        "median ms",
        "max ms",
        bound.unit(),
        if bound.peak().is_some() {
            format!("{:>12}", "% of peak")
        } else {
            String::new()
        }
    );

    let mut failures = 0;
    let mut compared = Vec::new();
    for &shape in sizes {
        eprintln!("{shape}: staging and checking");
        let timings = match (case.bench)(context, shape) {
            Ok(timings) => timings,
            Err(error) => {
                println!("{shape:<16}FAIL  {error}");
                failures += 1;
                continue;
            }
        };
        let rate = bound.rate((case.work)(shape), timings.min() / 1e3);
        println!(
            "{:<16}{:>8}{:>11.4}{:>11.4}{:>11.4}{:>12.1}{}",
            shape,
            (case.blocks)(shape),
            timings.min(),
            timings.median(),
            timings.max(),
            rate,
            match bound.peak() {
                Some(peak) => format!("{:>11.1}%", 100.0 * rate / peak),
                None => String::new(),
            }
        );

        // The baseline runs here rather than in a sweep of its own so that the
        // two measurements of a size are adjacent in time. A device that
        // throttles or gets shared halfway through a sweep would otherwise
        // move one side of the ratio and not the other.
        let Some(baseline) = case.baseline else {
            continue;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        match (baseline.bench)(context, shape) {
            Ok((against, algorithm)) => compared.push((shape, timings, against, algorithm)),
            Err(error) => {
                println!("{:<16}{} FAIL  {error}", shape, baseline.name);
                failures += 1;
            }
        }
    }
    if let Some(baseline) = case.baseline {
        compare(case, baseline, &compared);
    }
    failures
}

/// What a `bench <name> [<m> <n> <k>]` argument list selects: one case, and
/// optionally one size of it instead of that case's whole sweep.
///
/// Both narrowings exist for the same reason, which is that the sweep is the
/// wrong thing to point an instrument at. A profiler serializes and replays
/// every launch it is asked about, and the interesting one is buried two
/// hundred launches deep behind five sizes that are not the question; a
/// diagnostic table wants its own rows without re-staging 16384³ beside them.
///
/// Nothing else changes — the same [`Case`], the same staging, the same CPU
/// reference, the same clock. A narrowed run is still a *checked* run, which is
/// the rule this file exists to enforce and the one a diagnostic has the most
/// reason to want to slip.
fn selection() -> Option<(String, Option<Shape>)> {
    let arguments: Vec<String> = std::env::args().skip(2).collect();
    let size = |text: &String| text.parse().ok();
    match arguments.as_slice() {
        [name] => Some((name.clone(), None)),
        [name, m, n, k] => Some((
            name.clone(),
            Some(Shape {
                m: size(m)?,
                n: size(n)?,
                k: size(k)?,
            }),
        )),
        _ => None,
    }
}

/// `cargo oxide run kittens-examples -- bench`. Unlike the status table, this
/// is meaningless without a device, so a missing one is a failure rather than
/// something to degrade past.
pub fn main() -> ExitCode {
    let selected = selection();
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            println!("bench needs a CUDA device: {error}");
            return ExitCode::FAILURE;
        }
    };
    match (
        context.device_name(),
        context.compute_capability(),
        context.multiprocessor_count(),
    ) {
        (Ok(name), Ok((major, minor)), Ok(sms)) => {
            println!("{name}, sm_{major}{minor}, {sms} SMs")
        }
        _ => println!("device 0 (attributes unavailable)"),
    }

    // `bench swizzle` is not a [`Case`] and does not pretend to be one: a
    // `Case` sweeps *sizes* with the kernel held fixed, and this sweeps the
    // kernel's item traversal with the size held fixed (#89). It rides this
    // argument rather than `main`'s — where `clc` sits, which is the closer
    // sibling — for one reason worth stating plainly: `modal_app.py::bench`
    // already passes its `--case` through and already turns on `--features
    // cublas`, so the sweep reaches a B200 with a cuBLASLt column and no new
    // Modal entry point. Two changes to that file were in flight.
    if let Some((name, _)) = &selected
        && name == "swizzle"
    {
        return match gemm::swizzle(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // `bench tile` is the same kind of thing one lever further down: #87's
    // pair tile and pipeline depth with the size held fixed, where `swizzle`
    // holds the tile and moves the traversal. It rides this argument for the
    // same reason.
    if let Some((name, _)) = &selected
        && name == "tile"
    {
        return match gemm::tile_sweep(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // `bench epilogue` is the third of these, and the one lever `swizzle` and
    // `tile` both hold fixed: #15's store placement, with the size sweeping and
    // everything else — grid, shared plan, tensor memory, tile, traversal,
    // schedule — identical between the two columns.
    if let Some((name, _)) = &selected
        && name == "epilogue"
    {
        return match gemm::epilogue_sweep(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // `bench staged` is the fifth: the epilogue's *shape* where `epilogue`
    // sweeps its placement — a register drain against one staged through
    // shared memory by `stmatrix`. It rides this argument for the reason the
    // four above do, and it is its own sweep rather than two more columns of
    // `epilogue` because it is the one thing on that axis that moves the
    // shared plan, so it carries its own envelope control.
    if let Some((name, _)) = &selected
        && name == "staged"
    {
        return match gemm::staged_sweep(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // `bench widths` is the sixth, and the only one whose four arms declare
    // the *same* shared plan: #117's instruction widths on the staged
    // epilogue's two halves — `.x8` LDTM and `stmatrix.x4` — separately and
    // composed, against `staged` as the control. It is its own sweep rather
    // than more columns of `staged` because the control is different: that
    // one's baseline is the register drain, this one's is #116's result.
    if let Some((name, _)) = &selected
        && name == "widths"
    {
        return match gemm::widths_sweep(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // `bench ablation` is the fourth, and the one that decomposes rather than
    // sweeps: the same size sweep `gemm-depth` runs, with one phase of the
    // item removed per rung and everything else held at the shipped kernel's.
    // Where the three above ask "which setting is fastest", this asks "what is
    // the time made of", which is the question a fitted intercept has been
    // standing in for since #90.
    if let Some((name, _)) = &selected
        && name == "ablation"
    {
        return match gemm::ablation_ladder(&context, CUBLASLT) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some((name, _)) = &selected
        && !cases().iter().any(|case| case.name == name)
    {
        println!("no case named {name}; the sweep runs all of them when given no arguments");
        return ExitCode::FAILURE;
    }

    let mut failures = 0;
    for case in cases() {
        let sizes = match &selected {
            Some((name, _)) if name != case.name => continue,
            Some((_, Some(shape))) => vec![*shape],
            _ => case.sizes.to_vec(),
        };
        failures += report(&context, &case, &sizes);
        match case.bound.peak() {
            Some(peak) => println!(
                "% of peak is against {peak} TFLOP/s, dense bf16 for one B200: NVIDIA's HGX\n\
                 B200 page lists 36 PFLOPS FP16/BF16 across 8 GPUs, footnoted \"Sparse. Dense\n\
                 is ½ sparse spec shown\" — https://www.nvidia.com/en-us/data-center/hgx/"
            ),
            // Deliberately absent rather than forgotten. The same page states
            // the B200's flops per GPU and does not state its HBM bandwidth
            // per GPU, and an unsourced denominator is worse than none.
            None => println!("no % of peak: no sourced per-GPU HBM bandwidth figure to divide by"),
        }
    }

    println!("\nnot run:");
    for (name, reason) in SKIPPED {
        println!("  {name:<16}{reason}");
    }

    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        println!("\n{failures} size(s) failed their check; no throughput reported for those");
        ExitCode::FAILURE
    }
}
