//! The launch envelope: what a kernel's shared plan has to be admitted for.
//!
//! A CUDA block gets 48 KiB of dynamic shared memory without asking; anything
//! past that is opt-in, per function, before the launch. Kernels written
//! against this library cross 48 KiB as a matter of course, so a launch that
//! skips the opt-in is not slow, it is inadmissible.
//!
//! [`admit_shared_plan`] is that opt-in for kernels that do not go through a
//! `#[launch_contract]`'s generated `prepare_*` — which issues it already. It
//! also separates the two things a zero from
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` can mean: a plan nobody opted
//! into, and a plan the device cannot fit.
//!
//! Design notes: `docs/library/launch.md`.

use cuda_core::{CudaFunction, DriverError, IntoResult, sys};

/// A shared plan past what this device will admit even with the opt-in.
///
/// Not a [`DriverError`]: the driver is working fine and the answer is that the
/// tiles are too big.
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

/// Either reason a large plan is not admitted.
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
/// Idempotent and monotonic: a function already admitted for at least `bytes`
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
///
/// A plan the device will not take names both numbers, where an occupancy
/// query would only have answered zero:
///
/// ```no_run
/// use kittens::launch::{AdmitError, admit_shared_plan};
///
/// # fn admit(
/// #     function: &cuda_core::CudaFunction,
/// #     bytes: u32,
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// match admit_shared_plan(function, bytes) {
///     Ok(()) => Ok(()),
///     // Shrink the tiles: this device has no more to give.
///     Err(AdmitError::TooLarge(plan)) => {
///         Err(format!("{} B asked, {} B available", plan.bytes, plan.limit).into())
///     }
///     Err(driver) => Err(driver.into()),
/// }
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
