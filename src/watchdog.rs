//! A deadline on the host's wait for a launch, so a kernel that does not return
//! costs one row instead of a container.
//!
//! Every wait on a device here is ultimately `cuEventSynchronize` or
//! `cuStreamSynchronize`, and neither has a timeout. A launch that stops making
//! progress therefore stops the *host* too: the process sits in the driver, the
//! sweep prints nothing more, and the only thing that ever ends it is Modal's
//! `timeout=` — at B200 rates, for the whole of a ceiling sized for a cold build
//! plus a sweep. That is how `bench --case sol-k` was found (#146: twenty
//! minutes of silence at `4096x4096x1024`), and it is why that case and
//! `profile` were kept out of `session --arms`: an arm that can wedge takes
//! every row after it.
//!
//! This is the wait with a deadline on it. [`wait`] records an event behind
//! whatever is queued on the stream and *polls* it rather than blocking, so the
//! host keeps its own clock: past [`budget`] it prints the row the sweep last
//! announced and ends the process. One row fails in seconds, and the rows after
//! it — in later processes of the same container — run.
//!
//! ```no_run
//! use kittens::watchdog::{ReadBack, watching};
//!
//! # fn row(
//! #     stream: &cuda_core::CudaStream,
//! #     out: &cuda_core::DeviceBuffer<u16>,
//! #     launch: impl Fn() -> Result<(), Box<dyn std::error::Error>>,
//! # ) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
//! watching("gemm-sol [256, 256] at 4096x4096x1024");
//! launch()?;
//! // Blocks as `to_host_vec` does, and ends the process instead of hanging.
//! Ok(out.read_back(stream)?)
//! # }
//! ```
//!
//! # What it does and does not cover
//!
//! It covers every call in this tree that waits on a stream, and the list is
//! longer than it looks because three of `cuda-core`'s conveniences synchronize
//! inside themselves: `DeviceBuffer::from_host`, `DeviceBuffer::zeroed` and
//! `DeviceBuffer::to_host_vec`. An unguarded readback taken after a launch *is*
//! an unguarded wait for that launch, and so is an unguarded staging call. So
//! [`wait`] replaces every `stream.synchronize()`, [`stage`] and [`cleared`]
//! replace the two constructors, and [`ReadBack::read_back`] replaces the
//! readback. Nothing in `src/`, `examples/`, `experiments/` or `device-tests/`
//! waits on a device any other way.
//!
//! It does not cover a wedge with no launch in it — a container that never
//! reaches Python, a `cargo` that hangs — which is `scripts/modal-run`'s startup
//! and silence budgets, one level out.
//!
//! # Why the process ends rather than the call failing
//!
//! Returning an error would be tidier and would be a lie: the launch is still
//! running, the stream still has work on it, and every buffer the row allocated
//! is still owned by a context the driver will not release. A caller that
//! recovered would be timing a device that is still busy with the launch it gave
//! up on. So the deadline is fatal by construction, and the isolation it needs
//! is a *process* boundary — which `modal_app.py` already has, one per arm.
//!
//! It is [`std::process::abort`] and not [`std::process::exit`] on purpose:
//! `exit` runs the C runtime's handlers, and the driver's own teardown handler
//! wants the context — which is the thing that is wedged. Aborting skips all of
//! it, and the kernel dies with the process when the driver reclaims it.
//!
//! Design notes: `docs/library/watchdog.md`.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError};

/// How long a single wait may take before it is a wedge, in milliseconds.
///
/// Three orders of magnitude above the work, and two below what a wedge used to
/// ride. The slowest single launch anywhere in this tree is the 16384³ GEMM at
/// about 8 ms (`experiments/src/bench.rs`, `GEMM_SIZES`' last row); a wait also
/// covers the staging copies queued ahead of it and the driver's lazy load of a
/// module on its first launch, which are seconds and not milliseconds. 30 s
/// clears all of that by a factor of thousands and still fails a wedged `bench`
/// arm inside a minute, against the 3600 s `SWEEPING` ceiling it used to hold a
/// B200 for.
///
/// A per-case budget is [`BUDGET_VARIABLE`], and `modal_app.py::wedge_demo`
/// is what sets it: a deadline nobody has watched fire is not a deadline.
pub const DEFAULT_BUDGET_MS: u64 = 30_000;

/// Overrides [`DEFAULT_BUDGET_MS`] for one process. Read once, at the first
/// wait, so every wait in a run carries the same budget.
pub const BUDGET_VARIABLE: &str = "KITTENS_LAUNCH_BUDGET_MS";

/// How long a wait spins before it starts sleeping between polls.
///
/// `cuEventSynchronize` spins under `CU_CTX_SCHED_AUTO`, which is what these
/// runs get, so spinning here is the wait it replaces rather than an addition to
/// it — and that is the property the clock in `examples/src/bench.rs` needs,
/// since it polls the same `stop` event it then reads a duration off. Past a
/// launch's own length there is nothing left to be prompt for, and sleeping
/// stops a wedge from burning a core for the rest of its budget.
const SPIN: Duration = Duration::from_millis(50);

/// Between polls once [`SPIN`] is past. 30 s of budget is 30 000 `cuEventQuery`
/// calls at this rate, which is free beside what it is waiting for.
const POLL: Duration = Duration::from_millis(1);

/// The row a sweep last announced, which is what a fired deadline names.
static WATCHING: Mutex<Option<String>> = Mutex::new(None);

static BUDGET: OnceLock<Duration> = OnceLock::new();

/// Name what the next waits are waiting for.
///
/// Call it where the sweep already announces a row — the announcement and this
/// are the same fact, and `sol::sweep_k`'s doc has been leaning on the first
/// half of it since #149: *"the last announced row is the one that did not
/// return"*. A sweep that names nothing is named by its own command line, which
/// still says which case is in flight.
pub fn watching(what: impl Into<String>) {
    *WATCHING.lock().unwrap_or_else(|held| held.into_inner()) = Some(what.into());
}

/// The deadline every wait in this process carries.
pub fn budget() -> Duration {
    *BUDGET.get_or_init(|| parse_budget(std::env::var(BUDGET_VARIABLE).ok().as_deref()))
}

/// [`BUDGET_VARIABLE`]'s value as a budget. Unset and unparseable are the same
/// answer on purpose: a typo in a budget must not silently remove the deadline,
/// and there is no reading of `KITTENS_LAUNCH_BUDGET_MS=forever` that a run
/// should honour.
fn parse_budget(setting: Option<&str>) -> Duration {
    Duration::from_millis(
        setting
            .and_then(|text| text.trim().parse().ok())
            .filter(|&milliseconds| milliseconds > 0)
            .unwrap_or(DEFAULT_BUDGET_MS),
    )
}

fn announced() -> String {
    WATCHING
        .lock()
        .unwrap_or_else(|held| held.into_inner())
        .clone()
        .unwrap_or_else(|| std::env::args().collect::<Vec<_>>().join(" "))
}

/// Print which launch did not come back, and end the process.
///
/// On both streams: `stdout` is where the table being written is, and `stderr`
/// is where the row announcements are, and a reader watching one should not have
/// to find the other. Flushed explicitly because [`std::process::abort`] runs no
/// handlers, which is the whole reason it is the one used.
fn expired(waited: Duration) -> ! {
    let report = format!(
        "\n== kittens: the launch watchdog fired ==\n  \
         {}\n  \
         did not complete within {:.1} s ({} = {} ms). the launch is still in flight,\n  \
         so this process ends here rather than reporting a device it gave up on.\n",
        announced(),
        waited.as_secs_f64(),
        BUDGET_VARIABLE,
        budget().as_millis(),
    );
    println!("{report}");
    let _ = std::io::stdout().flush();
    eprintln!("{report}");
    let _ = std::io::stderr().flush();
    std::process::abort()
}

/// Set to anything to have every wait say how long it took, on `stderr`.
///
/// Two uses, and the second is why it is checked in rather than deleted. It says
/// how much of a sweep is the host waiting on the device rather than staging or
/// checking — and it is the only thing that can distinguish *this* wait
/// returning promptly from *some other* driver call blocking, which is a
/// distinction a run that does not come back has no other way to make.
pub const TRACE_VARIABLE: &str = "KITTENS_WATCHDOG_TRACE";

fn traced() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os(TRACE_VARIABLE).is_some())
}

/// Wait for `event`, or end the process past [`budget`].
///
/// The one to reach for when the caller already has an event recorded behind the
/// work — the timed loop in `examples/src/bench.rs` does, and paying for a
/// second event there would put a driver call inside the clock's own loop.
pub fn wait_event(event: &CudaEvent) -> Result<(), DriverError> {
    let start = Instant::now();
    let budget = budget();
    loop {
        if event.query()? {
            if traced() {
                eprintln!("  watchdog: waited {:.3} s", start.elapsed().as_secs_f64());
            }
            return Ok(());
        }
        let waited = start.elapsed();
        if waited >= budget {
            expired(waited);
        }
        if waited < SPIN {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(POLL);
        }
    }
}

/// Wait for everything queued on `stream`, or end the process past [`budget`].
///
/// The replacement for `stream.synchronize()`: same wait, same point in the
/// stream, with a clock on it.
pub fn wait(stream: &CudaStream) -> Result<(), DriverError> {
    // `None` is `CU_EVENT_DISABLE_TIMING`: nothing here reads a duration off it,
    // and the untimed event is the cheaper one to record.
    let event = stream.context().new_event(None)?;
    event.record(stream)?;
    wait_event(&event)
}

/// `DeviceBuffer::from_host`, behind the same deadline.
///
/// The constructors synchronize too — `from_host` enqueues a host-to-device copy
/// and then waits for the stream, as [`cleared`]'s `memset` does — so a staging
/// call taken while a launch is in flight is a wait on that launch, with no
/// deadline on it. That is not a theoretical hole: it is what the first run of
/// `modal_app.py::wedge_demo` found, sitting in `DeviceBuffer::from_host` behind
/// a kernel that had ten minutes left to run.
///
/// So the rule is the one [`ReadBack::read_back`] follows and there are no
/// exceptions to it: **every call in this tree that waits on a stream drains it
/// under the deadline first.** What is left after that is arithmetic — an event
/// created, recorded and queried on a stream that is almost always already
/// empty, which is microseconds against a copy measured in milliseconds.
pub fn stage<T: DeviceCopy>(
    stream: &CudaStream,
    data: &[T],
) -> Result<DeviceBuffer<T>, DriverError> {
    wait(stream)?;
    DeviceBuffer::from_host(stream, data)
}

/// `DeviceBuffer::zeroed`, behind the same deadline. See [`stage`].
pub fn cleared<T: DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    wait(stream)?;
    DeviceBuffer::zeroed(stream, len)
}

/// Reading a device buffer back, behind the same deadline.
///
/// `DeviceBuffer::to_host_vec` synchronizes on the stream inside itself, so a
/// readback taken straight after a launch *is* the wait for that launch — and
/// every check in this tree is one. This is that call with [`wait`] in front of
/// it, which is why the sites that use it read exactly as they did.
pub trait ReadBack<T> {
    /// `to_host_vec`, having first waited for the stream under the deadline.
    fn read_back(&self, stream: &CudaStream) -> Result<Vec<T>, DriverError>;
}

impl<T: DeviceCopy> ReadBack<T> for DeviceBuffer<T> {
    fn read_back(&self, stream: &CudaStream) -> Result<Vec<T>, DriverError> {
        wait(stream)?;
        self.to_host_vec(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A budget that is not a number is the default and never the absence of
    /// one — the failure mode worth a test is a deadline switched off by a typo.
    #[test]
    fn a_budget_only_moves_for_a_positive_number() {
        let default = Duration::from_millis(DEFAULT_BUDGET_MS);
        assert_eq!(parse_budget(Some("1500")), Duration::from_millis(1500));
        assert_eq!(parse_budget(Some(" 1500 ")), Duration::from_millis(1500));
        assert_eq!(parse_budget(None), default);
        assert_eq!(parse_budget(Some("")), default);
        assert_eq!(parse_budget(Some("forever")), default);
        assert_eq!(parse_budget(Some("-1")), default);
        assert_eq!(parse_budget(Some("0")), default);
    }

    #[test]
    fn an_unnamed_sweep_is_named_by_its_command_line() {
        assert!(!announced().is_empty());
        watching("gemm-sol [256, 256] at 4096x4096x1024");
        assert_eq!(announced(), "gemm-sol [256, 256] at 4096x4096x1024");
    }
}
