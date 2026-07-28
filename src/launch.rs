//! The launch envelope: what a kernel's shared plan has to be admitted for.
//!
//! A CUDA block gets 48 KiB of dynamic shared memory without asking. Every
//! plan past that is *opt-in* — `cuFuncSetAttribute` with
//! `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, per function, before the
//! launch — and a launch that skips it is not slow, it is inadmissible. This
//! library's whole point is tiles big enough to keep tcgen05 fed, so the
//! kernels here cross 48 KiB as a matter of course: `gemm` at 72 KiB,
//! `flash_forward` at 144 KiB.
//!
//! # Why this exists when cuda-oxide already has one
//!
//! `PreparedLaunch::__prepare` issues the opt-in, and `CudaFunction`'s setter
//! is `pub(crate)` so that it is the only thing that can. That is the right
//! default — it validates the plan against the device before mutating the
//! function — but it is reachable only from a `#[launch_contract]` kernel's
//! generated `prepare_*`, and a contract is a claim about the whole launch:
//! its domain, its block shape, and the index space each output slice is
//! partitioned by. A kernel that partitions its output some other way (warp
//! bands through a raw cursor, which is what [`crate::global::store_rows`] is
//! for) cannot state that claim honestly, and should not have to invent one to
//! get 144 KiB of shared memory.
//!
//! So this is the same opt-in with the same check in front of it, on the path
//! that does not go through a contract. Nothing here replaces `prepare_*`: a
//! kernel that *can* state a contract should, and gets this for free.
//!
//! # Reading a zero
//!
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` answers **0** both for a plan
//! the device cannot fit and for a plan nobody opted into, and those want
//! opposite fixes — shrink the tiles, or call this. #70 was an hour of the
//! wrong one. [`admit_shared_plan`] separates them: it returns
//! [`SharedPlanTooLarge`] with the device's own ceiling beside the ask, so a
//! plan that genuinely does not fit says so in bytes rather than in a zero.

use cuda_core::{CudaFunction, DriverError, IntoResult, sys};

/// A shared plan past what this device will admit even with the opt-in.
///
/// Distinct from a [`DriverError`] on purpose: the driver is working fine and
/// the answer is that the tiles are too big.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedPlanTooLarge {
    /// Dynamic shared memory the launch asked for.
    pub bytes: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    pub limit: u32,
}

impl core::fmt::Display for SharedPlanTooLarge {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "shared plan of {} B is past this device's {} B opt-in limit per block",
            self.bytes, self.limit
        )
    }
}

impl std::error::Error for SharedPlanTooLarge {}

/// Either reason a large plan is not admitted, kept apart so a caller can tell
/// them apart.
#[derive(Debug)]
pub enum AdmitError {
    /// The plan does not fit this device.
    TooLarge(SharedPlanTooLarge),
    /// The driver refused the query or the attribute write.
    Driver(DriverError),
}

impl core::fmt::Display for AdmitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooLarge(error) => error.fmt(formatter),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AdmitError {}

impl From<DriverError> for AdmitError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

/// Opt `function` into `bytes` of dynamic shared memory, so that a launch at
/// that size — and an occupancy query about one — is admissible.
///
/// Idempotent, and monotonic: a function already admitted for at least `bytes`
/// is left alone, so two callers preparing the same kernel cannot lower the
/// ceiling underneath each other. Under 48 KiB this is a no-op.
///
/// The check happens first and is the device's own: `bytes` past
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` is
/// [`AdmitError::TooLarge`] and the attribute is never written.
///
/// # Example
///
/// ```no_run
/// # fn occupancy(
/// #     function: &cuda_core::CudaFunction,
/// # ) -> Result<u32, Box<dyn std::error::Error>> {
/// kittens::launch::admit_shared_plan(function, 147_536)?;
/// Ok(function.max_active_blocks_per_multiprocessor(128, 147_536)?)
/// # }
/// ```
pub fn admit_shared_plan(function: &CudaFunction, bytes: u32) -> Result<(), AdmitError> {
    let context = function.context();
    let limit = context.max_opt_in_shared_memory_per_block()?;
    if bytes > limit {
        return Err(AdmitError::TooLarge(SharedPlanTooLarge { bytes, limit }));
    }
    if bytes <= function.max_dynamic_shared_memory_bytes()? {
        return Ok(());
    }
    context.bind_to_thread()?;
    // SAFETY: the handle is borrowed from `function`, which owns its module
    // for the duration of the call, and `bytes` is within the device's own
    // opt-in limit as just queried.
    unsafe {
        sys::cuFuncSetAttribute(
            function.cu_function(),
            sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    }
    .result()?;
    Ok(())
}
