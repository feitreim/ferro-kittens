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
//!
//! Every example that is *not* in [`cases`] is in [`SKIPPED`] with the reason,
//! so a missing row never leaves a reader guessing whether a kernel is slow or
//! simply not written yet.

use std::error::Error;
use std::process::ExitCode;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};

use crate::gemm;

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
    fn min(&self) -> f64 {
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
    bench: fn(&Arc<CudaContext>, Shape) -> Result<Timings, Box<dyn Error>>,
}

/// Sizes for `gemm`, picked to cross a regime rather than to be large.
///
/// A cluster owns a `256 x 128` tile of `C`, so the first row is the smallest
/// legal problem there is: one cluster, two CTAs, on a device with well over a
/// hundred SMs. That end measures launch and drain and nothing else. The second
/// is the size `modal_app.py::examples` checks, kept so the two runs share a
/// point. From there each row roughly quadruples the cluster count: 2048 is the
/// first size past a full wave of CTAs, and 4096 and 8192 are where the K
/// pipeline should dominate and the curve flatten at this kernel's ceiling.
const GEMM_SIZES: &[Shape] = &[
    Shape {
        m: 256,
        n: 128,
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
];

fn cases() -> Vec<Case> {
    vec![Case {
        name: "gemm",
        bound: Bound::Compute,
        sizes: GEMM_SIZES,
        work: |shape| 2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64,
        blocks: |shape| gemm::grid(shape.m, shape.n),
        bench: gemm::bench,
    }]
}

/// Examples with no row above, and why. All three are aspirational and sit
/// behind their own cargo feature, so they are not even compiled into the
/// default build — the status table `main` prints is the same claim from the
/// other side.
const SKIPPED: &[(&str, &str)] = &[
    (
        "softmax",
        "aspirational (#9, the TMA store) — would report GB/s",
    ),
    (
        "layernorm",
        "aspirational (#3, #9, #13, #22) — would report GB/s",
    ),
    (
        "flash_forward",
        "aspirational (#7, #11, #22, #23, #31) — would report TFLOP/s",
    ),
];

fn report(context: &Arc<CudaContext>, case: &Case) -> usize {
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
    for &shape in case.sizes {
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
    }
    failures
}

/// `cargo oxide run kittens-examples -- bench`. Unlike the status table, this
/// is meaningless without a device, so a missing one is a failure rather than
/// something to degrade past.
pub fn main() -> ExitCode {
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

    let mut failures = 0;
    for case in cases() {
        failures += report(&context, &case);
        if let Some(peak) = case.bound.peak() {
            println!(
                "% of peak is against {peak} TFLOP/s, dense bf16 for one B200: NVIDIA's HGX\n\
                 B200 page lists 36 PFLOPS FP16/BF16 across 8 GPUs, footnoted \"Sparse. Dense\n\
                 is ½ sparse spec shown\" — https://www.nvidia.com/en-us/data-center/hgx/"
            );
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
