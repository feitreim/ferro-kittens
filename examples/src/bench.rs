//! The clock, and the shape a timed run is taken at — nothing else.
//!
//! Four rules this file exists to enforce, in the order they matter:
//!
//! 1. **A number only ever comes out of a checked run.** [`time`] takes the
//!    launch as a closure, and every kernel here hands it one only from inside
//!    its own verify-then-time entry point, so there is no path that reports
//!    throughput for a kernel that computed the wrong answer.
//! 2. **The clock is the device's.** CUDA events either side of the launch,
//!    not wall clock around the driver call — at the small end of a sweep the
//!    driver's launch path is the same order as the kernel.
//! 3. **A wait on a device has a deadline.** Every wait here goes through
//!    [`kittens::watchdog`], which polls the event the clock already records
//!    instead of blocking on it, so a launch that stops making progress ends
//!    this process in seconds naming the row — where a bare
//!    `cuEventSynchronize` rides the container's whole `timeout=` (#146,
//!    `bench --case sol-k`, twenty minutes of a B200 saying nothing).
//! 4. **A single number is not a measurement.** [`Timings`] keeps the launches
//!    both sorted and in the order they were issued. The headline is the
//!    minimum, since every source of error on a quiet device adds time and
//!    none subtracts it; [`Timings::spread`] and [`Timings::drift`] are the two
//!    ways a minimum fails to repeat, and they want opposite fixes.
//!
//! `experiments/src/bench.rs` includes this file through `#[path]` and
//! re-exports it rather than carrying a second copy, so the sweeps and the
//! examples are timed by the same clock. That is why [`Timings::median`],
//! [`Timings::spread`] and [`Timings::drift`] have no caller in this crate:
//! they are columns only a sweep prints.
//!
//! Why the harness is shaped this way: `docs/kernels/harness.md`.

use std::error::Error;

use cuda_core::CudaStream;

use kittens::watchdog;

/// Launches discarded before timing begins. `pub` because a table quoting a
/// minimum has to say how many launches it is the minimum *of*.
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

    pub fn min(&self) -> f64 {
        self.sorted[0]
    }

    pub fn median(&self) -> f64 {
        self.sorted[self.sorted.len() / 2]
    }

    pub fn max(&self) -> f64 {
        self.sorted[self.sorted.len() - 1]
    }

    /// How wide this call's own distribution is: it bounds what `min` of
    /// [`ITERATIONS`] can be asked to do.
    pub fn spread(&self) -> f64 {
        self.max() / self.min() - 1.0
    }

    /// The same call's second half against its first, **in launch order**.
    /// Stationary noise leaves this at zero however wide [`Timings::spread`]
    /// is; a clock stepping down under load does not.
    pub fn drift(&self) -> f64 {
        let half = self.launched.len() / 2;
        let mean = |over: &[f64]| over.iter().sum::<f64>() / over.len() as f64;
        mean(&self.launched[half..]) / mean(&self.launched[..half]) - 1.0
    }
}

/// Time `launch` with CUDA events recorded either side of it on `stream`.
///
/// This is the argument every kernel's `bench` passes to its own checked-run
/// entry point, which is what makes "verified, then timed" the only order that
/// can happen.
pub fn time(
    stream: &CudaStream,
    launch: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<Timings, Box<dyn Error>> {
    // To stderr and not into the table: a sweep is minutes long and a reader
    // watching it should be able to tell a slow size from a stuck one.
    eprintln!("  checked; {WARMUP} warm-up then {ITERATIONS} timed launches");
    let timing_enabled = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let start = stream.context().new_event(timing_enabled)?;
    let stop = stream.context().new_event(timing_enabled)?;

    for _ in 0..WARMUP {
        launch()?;
    }
    watchdog::wait(stream)?;

    let mut milliseconds = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        start.record(stream)?;
        launch()?;
        stop.record(stream)?;
        // The wait `elapsed_ms` would do anyway, with a deadline on it, and on
        // the event it already has: nothing is recorded, allocated or launched
        // that was not before, and the pair bracketing the launch is untouched.
        // `elapsed_ms` then synchronizes two events that have already been seen
        // complete. See `kittens::watchdog` for why the poll is the same wait.
        watchdog::wait_event(&stop)?;
        milliseconds.push(start.elapsed_ms(&stop)? as f64);
    }
    Ok(Timings::new(milliseconds))
}
