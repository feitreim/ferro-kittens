//! The clock, and the shape a timed run is taken at — nothing else.
//!
//! Three rules this file exists to enforce, in the order they matter:
//!
//! 1. **A number only ever comes out of a checked run.** [`time`] takes the
//!    launch as a closure, and every kernel here hands it one only from inside
//!    its own verify-then-time entry point — so there is no path that reports
//!    throughput for a kernel that computed the wrong answer. A harness that
//!    would happily print TFLOP/s for garbage is a trap you fall into once, and
//!    the number outlives the run that produced it.
//! 2. **The clock is the device's.** CUDA events either side of the launch, not
//!    wall clock around the driver call.
//! 3. **A single number is not a measurement.** [`Timings`] keeps the launches
//!    both sorted and in the order they were issued, because a minimum that
//!    will not repeat has two causes with opposite fixes and only the unsorted
//!    samples tell them apart.
//!
//! **The sweeps that use this live in `experiments/`**, and this file is
//! deliberately the part they do not own: `experiments/src/bench.rs` re-exports
//! these three items rather than carrying a second copy of them, so the
//! `softmax` and `layernorm` rows of that harness are timed by exactly the
//! clock their own [`bench`](crate::softmax::bench) entry points are written
//! against.

use std::error::Error;

use cuda_core::CudaStream;

/// Launches discarded before timing begins. The first pays module load and the
/// first launch of a given shape pays the driver's own setup for it; neither is
/// representative of the next thousand.
///
/// `pub` because a table that quotes a minimum has to say how many launches it
/// is the minimum *of*, and `experiments/`' sweeps print both figures in their
/// headers rather than writing them down a second time.
pub const WARMUP: usize = 5;
/// Timed launches per size, per [`WARMUP`].
pub const ITERATIONS: usize = 30;

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

/// Per-launch kernel times in milliseconds, kept both sorted and in the order
/// they were launched.
///
/// The headline is the minimum: it is the least noise-contaminated estimate of
/// what the kernel can do, since every source of error on a quiet device adds
/// time and none subtracts it. The median and maximum are printed beside it
/// because a gap between them is a finding — a device sharing work, a clock
/// dropping, a first-touch cost that never amortizes — and the table's job is
/// to surface that, not to hide it behind one number.
///
/// **The launch order is kept because sorting answers only half the question**
/// (#122). A row whose minimum will not repeat has two possible causes with
/// opposite fixes: the distribution is wide and `min` of [`ITERATIONS`] is
/// sampling its left tail, which more samples fix, or the device is slowing
/// down inside the call, which no number of samples fixes. [`Timings::spread`]
/// sees the first and [`Timings::drift`] sees the second, and neither is
/// visible once the samples are sorted.
pub struct Timings {
    sorted: Vec<f64>,
    launched: Vec<f64>,
}

impl Timings {
    fn new(launched: Vec<f64>) -> Timings {
        let mut sorted = launched.clone();
        sorted.sort_by(f64::total_cmp);
        Timings { sorted, launched }
    }

    /// The headline.
    pub fn min(&self) -> f64 {
        self.sorted[0]
    }

    /// Printed beside the headline, and quoted in its own right where a ratio
    /// is only as stable as its denominator.
    pub fn median(&self) -> f64 {
        self.sorted[self.sorted.len() / 2]
    }

    /// As [`Timings::median`].
    pub fn max(&self) -> f64 {
        self.sorted[self.sorted.len() - 1]
    }

    /// How wide this call's own distribution is, as `max/min - 1`.
    ///
    /// It bounds what `min` of [`ITERATIONS`] can be asked to do. A call whose
    /// launches all land within 1% of each other has a minimum that is the
    /// floor; a call spread over 15% has one that is wherever the luckiest of
    /// thirty draws fell, and two such calls will not agree.
    pub fn spread(&self) -> f64 {
        self.max() / self.min() - 1.0
    }

    /// The same call's second half against its first, **in launch order** — the
    /// separator between a wide distribution and a moving device.
    ///
    /// Noise that is stationary leaves this at zero however wide
    /// [`Timings::spread`] is. A clock stepping down under sustained load does
    /// not: it puts the fast launches at the start and the slow ones at the
    /// end, so the sign and the size of this are the thermal question asked
    /// directly rather than inferred from a spread that cannot tell the two
    /// apart.
    pub fn drift(&self) -> f64 {
        let half = self.launched.len() / 2;
        let mean = |over: &[f64]| over.iter().sum::<f64>() / over.len() as f64;
        mean(&self.launched[half..]) / mean(&self.launched[..half]) - 1.0
    }
}

/// Time `launch` with CUDA events recorded either side of it on `stream`.
///
/// The events measure the kernel's own span on the device, which is the thing
/// under test; wall clock around the call would measure the driver's launch
/// path and the host's scheduling as well, and at the small end of a sweep
/// those are the same order as the kernel.
///
/// This is the argument every kernel's `bench` passes to its own checked-run
/// entry point, which is what makes "verified, then timed" the only order that
/// can happen.
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
    Ok(Timings::new(milliseconds))
}
