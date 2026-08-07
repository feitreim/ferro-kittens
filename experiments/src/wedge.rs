//! A launch that does not come back, on purpose — the control for the launch
//! watchdog.
//!
//! `scripts/modal-run` has `modal_app.py::stall` for exactly this reason: *a
//! check nobody has watched fail is not a check*. [`kittens::watchdog`] is the
//! same kind of guard one level in, and this is its stall — a kernel that spins
//! for far longer than any budget, queued **immediately in front of a row's own
//! launch, on the row's own stream**, so the wait that row takes next is a wait
//! on a launch that is still running. That is #146's failure exactly
//! (`bench --case sol-k` at `4096x4096x1024`, twenty minutes of a B200 saying
//! nothing), reproducible on demand and at a cost of one row.
//!
//! It is behind the off-by-default `wedge` feature, so no shipping build carries
//! it: `regcount`'s register table, the occupancy-step gate and the local-memory
//! census all read the crate's default feature set and none of them ever sees
//! this kernel. `modal_app.py::wedge_demo` is the only thing that turns it on.
//!
//! # Where it is injected, and why it moved
//!
//! [`inject`] is called from [`crate::gemm_sol::run`], between that row's
//! staging and its checked launch. It was first called at the top of
//! [`crate::bench::main`] instead — before anything else the process did — on the
//! theory that wedging earlier can only be a stronger test. Two runs said
//! otherwise, and both are worth keeping written down:
//!
//! 1. The first sat in `DeviceBuffer::from_host` staging the gate size's
//!    operands, because the constructors synchronize inside themselves too and
//!    only the readbacks had been routed through the deadline.
//!    [`kittens::watchdog::stage`] and [`kittens::watchdog::cleared`] exist
//!    because of that run.
//! 2. The second, with those guarded, stopped even earlier — before the sweep's
//!    own header reached the log — with the only candidate ahead of the first
//!    guarded wait being `kernels::load`, the driver's JIT and module load. That
//!    is a wait this crate does not own and cannot put an event behind.
//!
//! Neither is the failure the watchdog is for, and wedging a process before it
//! has loaded its kernels is not a stronger test of "a launch that does not
//! return", it is a different one. So the injection sits where the real thing
//! happened: everything a row needs is already loaded and staged, and the next
//! thing that happens is a launch that will not finish.
//!
//! # The spin is bounded, and that is deliberate
//!
//! An unbounded `while true {}` would demonstrate the same thing and would leave
//! a B200 running a kernel nobody is waiting for if the host somehow failed to
//! die. [`WEDGE_SECONDS`] is two minutes in `wedge_demo` against a budget of five
//! seconds: indistinguishable from a true wedge for far longer than the
//! demonstration needs, and self-terminating if everything else about it goes
//! wrong. Getting a spin that is *both* bounded and real took four attempts and
//! the watchdog's own trace to diagnose — see [`kernels::wedge`].

use std::error::Error;
use std::sync::OnceLock;

use cuda_core::{CudaStream, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Set to a number of seconds to wedge the first row that launches; unset to do
/// nothing at all. The arm that gets wedged is chosen by the environment rather
/// than by which case has an injection point.
pub const WEDGE_SECONDS: &str = "KITTENS_WEDGE_SECONDS";

/// One wedge a process. A sweep is a ladder of rows and every one of them would
/// otherwise queue another spin, which would make the second row's failure a
/// test of the first row's leftovers.
static INJECTED: OnceLock<()> = OnceLock::new();

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Occupy the device for `ticks` of the SM clock, computing nothing.
    ///
    /// **This does not work yet, and the reason is written down rather than
    /// guessed at again.** Two spellings have been run on a B200 — this one and
    /// the same loop on `globaltimer` — and both launch and return immediately,
    /// wedging nothing. The launch watchdog is what says so, once asked to trace:
    /// the guarded wait taken straight after this launch reports
    ///
    /// ```text
    ///   wedge: queued; the next wait on this stream is a wait on it
    ///   watchdog: waited 0.000 s
    /// ```
    ///
    /// with a sixty-second spin supposedly in front of it, and the sweep then
    /// runs its whole ladder in the time that ladder costs with nothing in front
    /// of it. So the guard records, polls and reports correctly; the kernel is
    /// absent.
    ///
    /// The mechanism is LLVM's forward-progress rule: **a loop with no side
    /// effect may be assumed to terminate, and may therefore be deleted.** A loop
    /// whose only content is a special-register read has no side effect, and both
    /// `clock64` and `globaltimer` are special-register reads. That is why
    /// [`kittens::sync::Semaphore::wait_before`] is not a counter-example even
    /// though `sol_watch` demonstrably spins on it: its condition calls
    /// `mbarrier_try_wait_parity`, which touches memory.
    ///
    /// **So the next spelling to try is one whose condition touches memory** —
    /// spin on a volatile read of a device word the host leaves at zero, bounded
    /// by `clock64` so the kernel still gives the device back. A third attempt
    /// wrote each `globaltimer` reading to a `*mut u64` and also returned
    /// immediately, so a store in the *body* was not enough on its own and the
    /// condition is the part to move.
    ///
    /// The duration is approximate either way, and deliberately: `ticks` are SM
    /// clocks, so a B200 at about 1.9 GHz gives roughly `n / 2` seconds for
    /// `n * 1e9`. The demonstration only needs "much longer than the budget", and
    /// shorter-than-nominal is the safe direction for a kernel whose other job is
    /// to give the device back.
    ///
    /// One warp of one block is enough: what the host waits on is the stream, and
    /// the stream is not drained until this returns however small the launch is.
    #[kernel]
    pub unsafe fn wedge(ticks: u64) {
        let start = cuda_device::debug::clock64();
        while cuda_device::debug::clock64().wrapping_sub(start) < ticks {}
    }
}

/// Queue the wedge on `stream` if [`WEDGE_SECONDS`] asks for one, once.
pub fn inject(stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    let Some(seconds) = std::env::var(WEDGE_SECONDS)
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
    else {
        return Ok(());
    };
    if INJECTED.set(()).is_err() {
        return Ok(());
    }
    println!(
        "\n{WEDGE_SECONDS}={seconds}: queueing a {seconds} s spin in front of this row's own\n\
         launch, on this row's own stream. the wait that follows it is a wait on a launch\n\
         that is still running, which is what `kittens::watchdog` is for."
    );
    // Each step is announced, because the first three attempts at this
    // demonstration each stopped at a different one of them and the log could
    // not say which. They cost one line and they are the difference between a
    // control that reports and a control that has to be re-run to be read.
    eprintln!("  wedge: loading the module");
    let module = kernels::load(stream.context())?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    eprintln!("  wedge: launching");
    // SM clocks, not nanoseconds -- see the kernel. A B200 runs this at about
    // half the nominal seconds, which is far more than any budget needs.
    unsafe { module.wedge(stream, config, seconds * 1_000_000_000)? };
    eprintln!("  wedge: queued; the next wait on this stream is a wait on it");
    Ok(())
}
