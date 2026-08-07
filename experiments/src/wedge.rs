//! A launch that does not come back, on purpose — the control for the launch
//! watchdog.
//!
//! `scripts/modal-run` has `modal_app.py::stall` for exactly this reason: *a
//! check nobody has watched fail is not a check*. [`kittens::watchdog`] is the
//! same kind of guard one level in, and this is its stall — a kernel that spins
//! for far longer than any budget, launched ahead of a sweep's own work on the
//! same stream, so the wait the sweep does next is a wait on a launch that is
//! still running. That is a hang in the shape the real ones had (#146,
//! `bench --case sol-k` at `4096x4096x1024`), reproducible on demand and at a
//! cost of one row.
//!
//! It is behind the off-by-default `wedge` feature, so no shipping build carries
//! it: `regcount`'s register table, the occupancy-step gate and the local-memory
//! census all read the crate's default feature set and none of them ever sees
//! this kernel. `modal_app.py::wedge_demo` is the only thing that turns it on.
//!
//! # The spin is bounded, and that is deliberate
//!
//! An unbounded `while true {}` would demonstrate the same thing and would leave
//! a B200 running a kernel nobody is waiting for if the host somehow failed to
//! die. [`WEDGE_SECONDS`] defaults to ten minutes: two orders of magnitude past
//! [`kittens::watchdog::DEFAULT_BUDGET_MS`], indistinguishable from a true wedge
//! for as long as the demonstration takes, and self-terminating if everything
//! else about the demonstration goes wrong.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Set to a number of seconds to wedge the run's first wait; unset to do
/// nothing. Read at the top of [`crate::bench::main`], so any `bench <case>`
/// can be wedged and the arm that gets it is chosen by the environment rather
/// than by which case has an injection point.
pub const WEDGE_SECONDS: &str = "KITTENS_WEDGE_SECONDS";

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Occupy the device for `nanoseconds`, computing nothing.
    ///
    /// `globaltimer` is a wall clock in nanoseconds and is read through an
    /// intrinsic with side effects, so the loop survives optimization — the same
    /// property `Semaphore::wait_before` relies on for `clock64`. One warp of one
    /// block is enough: what the host waits on is the stream, and the stream is
    /// not drained until this returns however small it is.
    #[kernel]
    pub unsafe fn wedge(nanoseconds: u64) {
        let start = cuda_device::debug::globaltimer();
        while cuda_device::debug::globaltimer().wrapping_sub(start) < nanoseconds {}
    }
}

/// Queue the wedge on the default stream if [`WEDGE_SECONDS`] asks for one.
///
/// Ahead of everything the sweep then does, and on the stream the sweep uses, so
/// nothing about the sweep has to know this exists. The first wait it takes is a
/// wait on this — and *which* wait that is has already been a finding. The first
/// run of the demonstration sat in `DeviceBuffer::from_host`, staging the gate
/// size's operands, because the constructors synchronize inside themselves too
/// and only the readbacks had been routed through the deadline.
/// [`kittens::watchdog::stage`] and [`kittens::watchdog::cleared`] exist because
/// of that run: a control earns its keep the first time it fires.
pub fn inject(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let Some(seconds) = std::env::var(WEDGE_SECONDS)
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
    else {
        return Ok(());
    };
    println!(
        "\n{WEDGE_SECONDS}={seconds}: queueing a {seconds} s spin on the default stream ahead\n\
         of this run's own work. every wait after this one is a wait on a launch that is\n\
         still going, which is what `kittens::watchdog` is for."
    );
    let stream = context.default_stream();
    let module = kernels::load(context)?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.wedge(&stream, config, seconds * 1_000_000_000)? };
    Ok(())
}
