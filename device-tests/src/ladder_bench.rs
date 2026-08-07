//! Is a streamed local-memory band actually slower? — issue #63.
//!
//! `cargo oxide run device-tests -- bench-ladder`, or
//! `modal run modal_app.py::ladder_bench`. **This one needs a B200.**
//!
//! #60's ladder found the in-place spellings coming in 81 to 130 registers
//! *under* the fused one at three shapes — on a stack frame at least as large
//! as the form they beat. That is not in-place winning; it is `ptxas` deciding
//! the band does not fit, leaving it addressable in local memory and streaming
//! it through a handful of registers. `ptxas -v` reports what was *allocated*
//! and never what it *cost*, so the ladder could pose that question and not
//! answer it. This is the first thing in the repo to put a clock on a register
//! claim.
//!
//! Four shapes, not fourteen, because unlike the rest of #60 this costs GPU
//! time: `[32, 96]`, `[48, 64]` and `[64, 64]`, where the effect is largest,
//! and `[32, 128]` as a control where `fused` wins on both static counters
//! (168/0 against 252/0) and therefore ought to win on the clock too.
//!
//! Three properties this file exists to keep.
//!
//! 1. **Nothing prints a number for a launch it did not verify.** `bench.rs`'s
//!    rule, and the reason it is worth having is that a spelling which lost its
//!    accumulate would look *fast*. Every configuration that gets timed is
//!    launched once first and compared elementwise against an `f64` reference
//!    that walks [`BaseLdtm`]'s ownership map on the host — over every block of
//!    the grid, not a sample of it.
//! 2. **The work cannot be optimized away**, which is the trap #60 fell into
//!    once already: its first probe design let the compiler restructure the
//!    loop nest until the band never had to exist, and every rung collapsed to
//!    32 registers. Four things stop it here, and the fourth is *measured*
//!    rather than argued —
//!    - `steps` is a runtime kernel argument, so the trip count is not a
//!      constant and the recurrence cannot be closed or unrolled away;
//!    - each iteration reads a *different* `M * N` block of `scores`
//!      (`step * M * N + ..`), so the loads are not loop-invariant and cannot
//!      be hoisted out;
//!    - `out_acc` is a true serial dependency — step `s` needs step `s - 1` —
//!      and it is dumped to memory at the end, so no part of the chain is dead;
//!    - and the appendix times the same rung at `steps` and at `2 * steps` and
//!      prints the ratio. A hoisted load or an eliminated loop does not double.
//! 3. **The kernel timed is the kernel priced.** The rungs here are the ladder
//!    probe body at the same `(M, N, FORM)`, differing only in that each block
//!    dumps into its own band so a grid may run it at all. `regcount` prints
//!    the twins side by side and flags any rung where the two disagree, and the
//!    register and local-frame columns below are read off the *driver*
//!    (`CU_FUNC_ATTRIBUTE_NUM_REGS`, `..._LOCAL_SIZE_BYTES`) for the loaded
//!    function rather than copied out of a table.
//!
//! And two numbers that are not wall time, because wall time on its own would
//! answer the wrong question:
//!
//! - **Occupancy.** 255 registers a thread caps a B200 SM hard where 47 does
//!   not, so a form can be quicker per thread and worse in a real kernel. The
//!   `warps/SM` column is `cuOccupancyMaxActiveBlocksPerMultiprocessor` for the
//!   loaded function at this launch shape — one warp a block, so blocks per SM
//!   *is* warps per SM — and the device grid is deliberately larger than any
//!   spelling can hold resident, so a form that fits fewer pays for it in waves.
//! - **A noise floor.** Timing is noisier than register counting, and #60's
//!   sweep only got to assert a table because it ran a determinism control
//!   first. Every rung is measured [`ROUNDS`] times, each round taking the
//!   minimum over [`ITERATIONS`] launches, and the table prints the spread
//!   across those rounds. A difference smaller than the spread is not a
//!   difference.
//!
//! ## What it measured
//!
//! One run, NVIDIA B200, sm_100, 148 SMs, 64 warps/SM. `1 warp` is nanoseconds
//! of kernel time per band element per step for a single 32-thread block;
//! `device` is picoseconds for the same, over 9472 such blocks. `x` is against
//! the `fused` row of the same shape.
//!
//! ```text
//!                  regs  frame  warps/SM   occ    1 warp ns    x    device ps    x
//!  [32,  96] fused   128   1536      16    25%       4.703   1.00       8.6    1.00
//!            assign   47   1920      32    50%       4.654   0.99       8.5    0.99
//!            rebound 251   1536       8    12%       4.772   1.01      10.3    1.20
//!            open_c   47   1920      32    50%       4.654   0.99       8.4    0.98
//!            all_ip   32   1152      32    50%       4.400   0.94       6.4    0.75
//!  [48,  64] fused   128   1536      16    25%       4.756   1.00       8.8    1.00
//!            assign   40   1920      32    50%       4.538   0.95       9.3    1.05
//!            rebound 202   1536       8    12%       4.839   1.02      10.6    1.19
//!            open_c   40   1920      32    50%       4.537   0.95       9.3    1.05
//!            all_ip   32   1152      32    50%       4.349   0.91       6.6    0.74
//!  [64,  64] fused   162   2560      12    19%       5.117   1.00      12.4    1.00
//!            assign   32   2560      32    50%       4.816   0.94      10.2    0.82
//!            rebound 168   2560      12    19%       5.249   1.03      10.8    0.88
//!            open_c   32   2560      32    50%       4.812   0.94      10.2    0.82
//!            all_ip   32   1536      32    50%       4.614   0.90       7.2    0.58
//!  [32, 128] fused   168   2560      12    19%       4.798   1.00      10.6    1.00
//!            assign  252   2560       8    12%       4.526   0.94       8.9    0.85
//!            rebound 255   2624       8    12%       4.950   1.03      11.7    1.10
//!            open_c  252   2560       8    12%       4.526   0.94       8.9    0.84
//!            all_ip   32   1536      32    50%       4.355   0.91       6.5    0.62
//! ```
//!
//! Noise floor: the widest round-to-round spread over all twenty rungs and
//! both grids is **0.8%**, and most cells are 0.1–0.6%. Linearity: the
//! furthest any rung's `t(2S)/t(S)` sits from 2.00 is **0.010**, so no rung's
//! loop was hoisted or shortened. Every launch matched the `f64` reference to
//! 1.4e-4 against a 0.02 tolerance — the same figure at all twenty rungs,
//! which is what "the same arithmetic, five ways" looks like when it is true.
//!
//! **On this probe, a streamed band is not slower.**
//! [`LADDER_ALL_IN_PLACE`](crate::LADDER_ALL_IN_PLACE) — 32 registers at all
//! four shapes, the whole band left in local memory — is the **fastest
//! spelling at every shape on both grids**, by 6–10% per warp and 25–42%
//! across the device. Whatever the local traffic costs here, it is smaller
//! than what the freed registers buy.
//!
//! **On a real kernel it went the other way, and that is the actual result.**
//! #47 timed the same phenomenon on `softmax` — a `[32, 128]` band at 39
//! registers on a 1024-byte frame against a `[32, 32]` band at 66 registers on
//! 256 — and the streamed one cost **2.6x**. Both numbers are right. What
//! differs is what the freed registers were able to buy: `softmax` is
//! shared-memory capped at 6 blocks an SM at either register count, so
//! streaming bought it nothing and the local traffic showed undiluted, where
//! this probe uses no shared memory at all and streaming bought it 2–4x the
//! resident warps. A streamed band costs real time; whether it is worth paying
//! is a question about which resource is binding in that kernel. The price
//! when the payment falls short is visible in the table above: at `[48, 64]`
//! `assign` runs 32 warps an SM against `fused`'s 16 and is still 5% *slower*
//! across the device.
//!
//! **And the control comes out backwards.** `[32, 128]` was picked because it
//! is the ladder's cleanest "fused wins" cell — 168 registers and no spill
//! against `assign`'s 252. On the clock `assign` is **faster**: 0.94x per warp,
//! 0.85x across the device. So the shape where the counters say fused wins is
//! a shape where fused loses, and the three shapes where in-place "wins" by
//! 81–130 registers are three where the per-warp times are within 6%.
//!
//! **Nothing in `ptxas -v` orders this table.** Not registers — `assign` is 47
//! at `[32, 96]` and 252 at `[32, 128]` and is 0.99x and 0.85x fused
//! respectively, the *cheaper* one being the one that fails to win. Not the
//! frame — `assign` carries the largest frame at `[32, 96]` and ties the
//! fastest time. Not occupancy either, which is the one that had a real
//! mechanism behind it: at `[32, 128]` `assign` runs 8 warps/SM against
//! `fused`'s 12 and is 15% faster anyway, and at `[48, 64]` it runs 32 against
//! 16 and is 5% slower. Occupancy explains more of the device column than
//! anything else does and it does not explain all of it.
//!
//! Two things do reproduce at all four shapes. `assign` and `open_coded` agree
//! to 0.1% on the clock, as they already agreed to the byte on every static
//! counter (#60) — the strongest form of #31's claim yet. And `rebound` is the
//! slowest spelling per warp at every shape, 1.01–1.03x fused, small but three
//! to five times the noise floor: materialization between statements is real
//! and it is the *only* part of #31's model that survives a clock.
//!
//! **What this does not settle.** The probe reads `M * N / 32` fp32 per thread
//! per step and does about three flops on each, so it is a load-heavy loop
//! with a long serial dependence, and the per-warp column is nearly flat
//! (4.35–5.25 ns across every spelling and shape) because at one warp a B200
//! SM is issue- and latency-bound rather than short of registers. A
//! compute-dense inner loop might price a streamed band very differently. Nor
//! was local-memory *traffic* measured: that the band is in local memory is
//! read off the frame and the register count, and no profiler was run. What is
//! measured is the thing the register table was being used to predict, which
//! is time — and it does not predict it.

use std::error::Error;
use std::fmt::Write as _;
use std::process::ExitCode;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaFunction, CudaStream, DeviceBuffer};
use cuda_host::CudaKernel;

use kittens::reg::{BaseLdtm, FragmentLayout, RegTile};

use crate::kernels;
use kittens::watchdog::{self, ReadBack};

/// Iterations of the probe's accumulate loop per launch, as the runtime
/// argument the kernel takes. Large enough that a launch is tens of
/// microseconds at one warp and milliseconds at a full grid, so the CUDA event
/// pair is measuring the kernel and not the driver.
const STEPS: u32 = 192;
/// The scale both bands are multiplied by. A half rather than a one: the
/// recurrence `out = k * out + k * block` is then a contraction, so the `f64`
/// reference and the device's `f32` stay within one rounding error of each
/// other however many steps run — and a `k` of one would have made a dropped
/// scale invisible to the check.
const K: f32 = 0.5;
/// How far an element may sit from the reference. The staged values run to
/// ~4100 in magnitude and are one apart from each other, so the accumulated
/// `f32` rounding error is under 1e-3 while any misplaced element is out by at
/// least 1. Two orders of margin on either side.
const TOLERANCE: f64 = 0.02;

/// Launches discarded before a round starts timing.
const WARMUP: usize = 3;
/// Timed launches in a round. The round's figure is the minimum over them:
/// every source of error on a quiet device adds time and none subtracts it.
const ITERATIONS: usize = 20;
/// Rounds, each producing one such minimum. The spread across rounds is this
/// harness's noise floor, printed rather than assumed.
const ROUNDS: usize = 5;

/// Blocks the device grid asks for per SM. Blackwell caps *resident* blocks
/// per SM at 32, so 64 one-warp blocks each is at least two waves for any
/// spelling and eight for one that holds only 8 warps — which is the point.
/// Every spelling is handed identical work and runs it in as many waves as its
/// own register footprint allows, so the occupancy column has somewhere to
/// show up in the time.
const BLOCKS_PER_SM: u32 = 64;

/// The four shapes, in the order the tables print them: the three where
/// in-place appears to win, then the control.
const SHAPES: [(usize, usize); 4] = [(32, 96), (48, 64), (64, 64), (32, 128)];

/// The staged score at `(step, index)` — `scores[step * M * N + index]`.
///
/// Exactly representable, distinct per element, and cheap for the host
/// reference to reproduce. The `+ 1` keeps element zero from being zero, and
/// the step term keeps consecutive steps from staging identical bands, so a
/// kernel that read one step's block twice would not agree with the reference.
fn score(step: usize, index: usize) -> f32 {
    (index + 1 + step % 3) as f32
}

/// One spelling at one shape: the name of the spelling, and the loaded
/// function, which is both what gets launched and where the register, frame
/// and occupancy columns are read from.
struct Rung {
    spelling: &'static str,
    function: CudaFunction,
}

/// The host half of `timed_ladder!` — a `(spelling, PTX symbol)` pair per rung.
///
/// The symbol comes from the `CudaKernel` marker `#[kernel]` generates rather
/// than from a `format!`, so a rung renamed on the device side stops compiling
/// here instead of failing at load time on a GPU that costs money.
macro_rules! rungs {
    ($m:literal, $n:literal) => {
        [
            rung!($m, $n, fused),
            rung!($m, $n, assign),
            rung!($m, $n, rebound),
            rung!($m, $n, open_coded),
            rung!($m, $n, all_in_place),
        ]
    };
}

macro_rules! rung {
    ($m:literal, $n:literal, $spelling:ident) => {
        (
            stringify!($spelling),
            <kernels::${concat(__ladder_timed_, $m, x, $n, _, $spelling, _CudaKernel)}
                as CudaKernel>::PTX_NAME,
        )
    };
}

/// One launch, argument for argument as `#[cuda_module]`'s own generated
/// launchers do it.
///
/// There is no generated launcher to call. `#[cuda_module]` emits a typed
/// method per kernel it can *read*, and these rungs are written by a
/// `macro_rules!` inside the module — which the attribute macro cannot see
/// through, since the inner macro has not expanded when it runs. The PTX entry
/// is there all the same (`#[kernel]` expands afterwards), so the kernel is
/// reachable by symbol; what would be a guess is the argument encoding, and
/// `cuda_host`'s `push_kernel_*` helpers exist exactly so the generated
/// methods do not open-code that ABI. This calls the same helpers, in the same
/// order, for the same reason.
fn launch(
    function: &CudaFunction,
    stream: &CudaStream,
    blocks: u32,
    scores: &DeviceBuffer<f32>,
    steps: u32,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    let mut arguments: Vec<*mut std::ffi::c_void> = Vec::new();
    let (mut scores_ptr, mut scores_len) = cuda_host::read_only_device_buffer_arg(scores);
    cuda_host::push_kernel_device_slice(&mut arguments, &mut scores_ptr, &mut scores_len);
    let mut steps = steps;
    cuda_host::push_kernel_scalar(&mut arguments, &mut steps);
    let mut k = K;
    cuda_host::push_kernel_scalar(&mut arguments, &mut k);
    let (mut out_ptr, mut out_len) = cuda_host::writable_device_buffer_arg(out);
    cuda_host::push_kernel_device_slice(&mut arguments, &mut out_ptr, &mut out_len);
    // Safety: every `Rung`'s function comes from the module loaded on the
    // context that owns `stream`; the arguments are marshalled by the helpers
    // that define this kernel's ABI and outlive the call; and the grid is one
    // warp per block, which is the index space the probe assumes — each block
    // dumps into its own `M * N` band (`STRIDED`), so no two blocks write the
    // same element.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            function,
            (blocks, 1, 1),
            (32, 1, 1),
            0,
            stream,
            &mut arguments,
        )?
    };
    Ok(())
}

/// Kernel milliseconds for one rung in one configuration: one minimum per
/// round, kept in order so a drift across rounds stays visible.
#[derive(Default)]
struct Repeated(Vec<f64>);

impl Repeated {
    fn best(&self) -> f64 {
        self.0.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn worst(&self) -> f64 {
        self.0.iter().copied().fold(0.0, f64::max)
    }

    /// Round-to-round spread as a percentage of the best round. This is the
    /// number a claimed difference has to clear.
    fn spread(&self) -> f64 {
        100.0 * (self.worst() - self.best()) / self.best()
    }
}

/// What one rung measured, in the order the tables print it.
struct Measured {
    spelling: &'static str,
    registers: u32,
    local_bytes: u32,
    warps_per_sm: u32,
    warp: Repeated,
    device: Repeated,
    /// Milliseconds at `2 * STEPS` over milliseconds at `STEPS`, one warp.
    scaling: f64,
    /// Largest absolute deviation from the `f64` reference, over every element
    /// of every block of every checked launch.
    deviation: f64,
}

fn function_attribute(
    function: &CudaFunction,
    attribute: cuda_core::sys::CUfunction_attribute,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    let mut value: i32 = 0;
    let status = unsafe {
        cuda_core::sys::cuFuncGetAttribute(&mut value, attribute, function.cu_function())
    };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuFuncGetAttribute({name}) failed with driver status {status}").into(),
        );
    }
    Ok(value as u32)
}

fn device_attribute(
    context: &CudaContext,
    attribute: cuda_core::sys::CUdevice_attribute,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    let mut value: i32 = 0;
    let status =
        unsafe { cuda_core::sys::cuDeviceGetAttribute(&mut value, attribute, context.cu_device()) };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuDeviceGetAttribute({name}) failed with driver status {status}").into(),
        );
    }
    Ok(value as u32)
}

/// The accumulator the probe should arrive at, per `(lane, slot, value)` of a
/// warp's dump, in `f64`.
///
/// Independent of the kernel in everything but the ownership map: it walks
/// [`RegTile::coordinate`] on the host, indexes the staged buffer by hand and
/// runs the same recurrence. A spelling that dropped a scale, an accumulate or
/// a step disagrees here before anything is timed.
fn reference<const M: usize, const N: usize>(steps: u32) -> Vec<f64>
where
    BaseLdtm: FragmentLayout<M, N>,
{
    let slots = RegTile::<M, N, BaseLdtm>::SLOTS;
    let values = RegTile::<M, N, BaseLdtm>::VALUES;
    let mut expected = vec![0.0f64; M * N];
    for lane in 0..32u32 {
        for slot in 0..slots {
            for value in 0..values {
                let (row, column) = RegTile::<M, N, BaseLdtm>::coordinate(lane, slot, value);
                let index = row as usize * N + column as usize;
                let mut accumulated = 0.0f64;
                for step in 0..steps as usize {
                    accumulated = accumulated * K as f64 + score(step, index) as f64 * K as f64;
                }
                expected[lane as usize * slots * values + slot * values + value] = accumulated;
            }
        }
    }
    expected
}

/// Every element of every block against the reference band, which each block
/// computes identically. Returns the largest deviation seen, so the tables can
/// state how close the check came rather than only that it passed.
fn verify(observed: &[f32], expected: &[f64]) -> Result<f64, Box<dyn Error>> {
    let elements = expected.len();
    let mut worst = 0.0f64;
    let mut mismatches = 0usize;
    let mut report = String::new();
    for (index, &got) in observed.iter().enumerate() {
        let want = expected[index % elements];
        let deviation = (got as f64 - want).abs();
        worst = worst.max(deviation);
        if deviation <= TOLERANCE {
            continue;
        }
        mismatches += 1;
        if mismatches <= 4 {
            let _ = write!(
                report,
                "\n    block {} element {}: expected {want}, got {got}",
                index / elements,
                index % elements
            );
        }
    }
    if mismatches == 0 {
        return Ok(worst);
    }
    Err(format!(
        "{mismatches} of {} elements wrong, tolerance {TOLERANCE}{report}",
        observed.len()
    )
    .into())
}

/// [`ITERATIONS`] timed launches after [`WARMUP`] discarded ones, in kernel
/// milliseconds off a CUDA event pair; the minimum is the round's figure.
fn round(
    stream: &CudaStream,
    launch: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<f64, Box<dyn Error>> {
    let timing_enabled = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let start = stream.context().new_event(timing_enabled)?;
    let stop = stream.context().new_event(timing_enabled)?;

    for _ in 0..WARMUP {
        launch()?;
    }
    watchdog::wait(stream)?;

    let mut best = f64::INFINITY;
    for _ in 0..ITERATIONS {
        start.record(stream)?;
        launch()?;
        stop.record(stream)?;
        best = best.min(start.elapsed_ms(&stop)? as f64);
    }
    Ok(best)
}

/// One launch, checked. Every configuration that gets timed goes through this
/// first, at the same grid and the same step count it will be timed at.
#[allow(clippy::too_many_arguments)]
fn checked(
    stream: &CudaStream,
    rung: &Rung,
    blocks: u32,
    steps: u32,
    scores: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    expected: &[f64],
) -> Result<f64, Box<dyn Error>> {
    launch(&rung.function, stream, blocks, scores, steps, out)?;
    let observed = out.read_back(stream)?;
    verify(&observed, expected).map_err(|error| {
        format!(
            "{} at {blocks} block(s), {steps} steps: {error}",
            rung.spelling
        )
        .into()
    })
}

fn measure<const M: usize, const N: usize>(
    stream: &CudaStream,
    rungs: &[Rung],
    blocks: u32,
) -> Result<Vec<Measured>, Box<dyn Error>>
where
    BaseLdtm: FragmentLayout<M, N>,
{
    let elements = M * N;
    let staged: Vec<f32> = (0..2 * STEPS as usize)
        .flat_map(|step| (0..elements).map(move |index| score(step, index)))
        .collect();
    let scores = watchdog::stage(stream, &staged)?;
    let mut one = watchdog::cleared::<f32>(stream, elements)?;
    let mut many = watchdog::cleared::<f32>(stream, blocks as usize * elements)?;
    let short_reference = reference::<M, N>(STEPS);
    let long_reference = reference::<M, N>(2 * STEPS);

    // Verified first, timed second, with nothing in between that could change
    // what runs. All three timed configurations are covered.
    eprintln!("[{M}, {N}]: checking {} spellings", rungs.len());
    let mut deviations = Vec::with_capacity(rungs.len());
    for rung in rungs {
        let short = checked(stream, rung, 1, STEPS, &scores, &mut one, &short_reference)?;
        let long = checked(
            stream,
            rung,
            1,
            2 * STEPS,
            &scores,
            &mut one,
            &long_reference,
        )?;
        let grid = checked(
            stream,
            rung,
            blocks,
            STEPS,
            &scores,
            &mut many,
            &short_reference,
        )?;
        deviations.push(short.max(long).max(grid));
    }

    // Rounds outermost so a thermal or clock drift lands on every spelling
    // alike and shows up as spread rather than as an ordering.
    let mut warp: Vec<Repeated> = (0..rungs.len()).map(|_| Repeated::default()).collect();
    let mut device: Vec<Repeated> = (0..rungs.len()).map(|_| Repeated::default()).collect();
    for index in 0..ROUNDS {
        eprintln!("[{M}, {N}]: timing round {} of {ROUNDS}", index + 1);
        for (rung, timings) in rungs.iter().zip(warp.iter_mut()) {
            let mut once = || launch(&rung.function, stream, 1, &scores, STEPS, &mut one);
            timings.0.push(round(stream, &mut once)?);
        }
        for (rung, timings) in rungs.iter().zip(device.iter_mut()) {
            let mut once = || launch(&rung.function, stream, blocks, &scores, STEPS, &mut many);
            timings.0.push(round(stream, &mut once)?);
        }
    }

    // The control that says the loop is still there: the same rung at twice
    // the steps, one warp, where there are no wave effects to muddy a ratio.
    eprintln!("[{M}, {N}]: doubling the step count");
    let mut scalings = Vec::with_capacity(rungs.len());
    for (rung, timings) in rungs.iter().zip(warp.iter()) {
        let mut once = || launch(&rung.function, stream, 1, &scores, 2 * STEPS, &mut one);
        scalings.push(round(stream, &mut once)? / timings.best());
    }

    let mut measured = Vec::with_capacity(rungs.len());
    for (index, rung) in rungs.iter().enumerate() {
        measured.push(Measured {
            spelling: rung.spelling,
            registers: function_attribute(
                &rung.function,
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
                "NUM_REGS",
            )?,
            local_bytes: function_attribute(
                &rung.function,
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "LOCAL_SIZE_BYTES",
            )?,
            warps_per_sm: rung.function.max_active_blocks_per_multiprocessor(32, 0)?,
            warp: Repeated(std::mem::take(&mut warp[index].0)),
            device: Repeated(std::mem::take(&mut device[index].0)),
            scaling: scalings[index],
            deviation: deviations[index],
        });
    }
    Ok(measured)
}

/// Nanoseconds of kernel time per band element per step: `milliseconds * 1e6`
/// over `blocks * STEPS * M * N`, the element-steps the launch actually did.
/// Normalizing by the grid is what makes the one-warp and full-device columns
/// each comparable to their own `fused` row.
fn per_element(milliseconds: f64, blocks: u32, elements: usize) -> f64 {
    milliseconds * 1e6 / (blocks as f64 * STEPS as f64 * elements as f64)
}

fn report(shape: (usize, usize), measured: &[Measured], blocks: u32, max_warps_per_sm: u32) {
    let (rows, columns) = shape;
    let elements = rows * columns;
    println!(
        "\n[{rows}, {columns}] — {} fp32 a thread, {elements} elements a step",
        elements / 32
    );
    println!(
        "{:<14}{:>6}{:>8}{:>10}{:>7}{:>12}{:>9}{:>9}{:>12}{:>9}{:>9}",
        "spelling",
        "regs",
        "frame",
        "warps/SM",
        "occ",
        "1 warp ns",
        "spread",
        "x fused",
        "device ps",
        "spread",
        "x fused"
    );
    let Some(fused) = measured.iter().find(|row| row.spelling == "fused") else {
        return;
    };
    let baseline_warp = per_element(fused.warp.best(), 1, elements);
    let baseline_device = per_element(fused.device.best(), blocks, elements);
    for row in measured {
        let warp = per_element(row.warp.best(), 1, elements);
        let device = per_element(row.device.best(), blocks, elements);
        println!(
            "{:<14}{:>6}{:>8}{:>10}{:>6.0}%{:>12.3}{:>8.1}%{:>9.2}{:>12.1}{:>8.1}%{:>9.2}",
            row.spelling,
            row.registers,
            row.local_bytes,
            row.warps_per_sm,
            100.0 * row.warps_per_sm as f64 / max_warps_per_sm as f64,
            warp,
            row.warp.spread(),
            warp / baseline_warp,
            device * 1e3,
            row.device.spread(),
            device / baseline_device,
        );
    }
}

/// The tables that are not the headline: the raw millisecond figures, the
/// linearity control, and how close each check came.
fn appendix(shape: (usize, usize), measured: &[Measured]) {
    let (rows, columns) = shape;
    println!("\n[{rows}, {columns}]");
    println!(
        "{:<14}{:>12}{:>12}{:>13}{:>13}{:>13}{:>16}",
        "spelling", "1 warp ms", "worst ms", "device ms", "worst ms", "t(2S)/t(S)", "max deviation"
    );
    for row in measured {
        println!(
            "{:<14}{:>12.4}{:>12.4}{:>13.4}{:>13.4}{:>13.3}{:>16.2e}",
            row.spelling,
            row.warp.best(),
            row.warp.worst(),
            row.device.best(),
            row.device.worst(),
            row.scaling,
            row.deviation,
        );
    }
}

pub fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ladder bench failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let context: Arc<CudaContext> = CudaContext::new(0)?;
    let owned_stream = context.default_stream();
    let stream: &CudaStream = &owned_stream;
    let module = kernels::load(&context)?;

    let sms = context.multiprocessor_count()?;
    let max_warps_per_sm = device_attribute(
        &context,
        cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        "MAX_THREADS_PER_MULTIPROCESSOR",
    )? / 32;
    let blocks = sms * BLOCKS_PER_SM;

    match (context.device_name(), context.compute_capability()) {
        (Ok(name), Ok((major, minor))) => {
            println!("{name}, sm_{major}{minor}, {sms} SMs, {max_warps_per_sm} warps/SM maximum")
        }
        _ => println!("device 0 ({sms} SMs, attributes unavailable)"),
    }
    println!(
        "{STEPS} steps, k = {K}, {ITERATIONS} timed launches x {ROUNDS} rounds after {WARMUP} \
         warm-up.\n\
         \"1 warp\" is one 32-thread block on one SM: per-thread cost, no occupancy term.\n\
         \"device\" is {blocks} such blocks, {BLOCKS_PER_SM} per SM — twice what any \
         spelling holds\n\
         resident, so a form that fits fewer warps pays for it in waves. Both are time per\n\
         band element per step (ns at one warp, ps across the device), so each column is\n\
         comparable within a shape. regs and frame are the driver's own for the loaded\n\
         function, not quoted from the ladder."
    );

    // A shape that fails is reported and the sweep carries on. One deliberate
    // run of this costs a B200 for the length of it, and losing three good
    // shapes to the fourth's failure would mean paying for the device twice to
    // learn one thing.
    let loaded = module.as_cuda_module();
    let mut failures = 0usize;
    macro_rules! shape {
        ($m:literal, $n:literal) => {{
            let attempt = || -> Result<Vec<Measured>, Box<dyn Error>> {
                let rungs: Vec<Rung> = rungs!($m, $n)
                    .into_iter()
                    .map(|(spelling, ptx_name)| {
                        loaded
                            .load_function(ptx_name)
                            .map(|function| Rung { spelling, function })
                    })
                    .collect::<Result<_, _>>()?;
                measure::<$m, $n>(stream, &rungs, blocks)
            };
            match attempt() {
                Ok(measured) => {
                    report(($m, $n), &measured, blocks, max_warps_per_sm);
                    measured
                }
                Err(error) => {
                    println!("\n[{}, {}] FAILED, no timings reported: {error}", $m, $n);
                    failures += 1;
                    Vec::new()
                }
            }
        }};
    }

    let measured = [
        shape!(32, 96),
        shape!(48, 64),
        shape!(64, 64),
        shape!(32, 128),
    ];

    println!(
        "\nraw kernel times, the linearity control, and how close the checks came.\n\
         t(2S)/t(S) is the same rung timed at {} steps over {STEPS}: a loop the compiler\n\
         hoisted or eliminated does not double. max deviation is the largest gap between\n\
         any element of any block and the f64 reference, against a {TOLERANCE} tolerance.",
        2 * STEPS
    );
    for (shape, rows) in SHAPES.iter().zip(measured.iter()) {
        if !rows.is_empty() {
            appendix(*shape, rows);
        }
    }

    let rungs = measured.iter().flatten().count();
    let worst_spread = measured
        .iter()
        .flatten()
        .map(|row| row.warp.spread().max(row.device.spread()))
        .fold(0.0f64, f64::max);
    let worst_scaling = measured
        .iter()
        .flatten()
        .map(|row| (row.scaling - 2.0).abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nnoise floor: the widest round-to-round spread over all {rungs} rungs and both grids\n\
         is {worst_spread:.1}%. A ratio nearer 1 than that is not a difference this run can\n\
         see. linearity: the furthest any rung's t(2S)/t(S) sits from 2.00 is {worst_scaling:.3}."
    );
    if failures > 0 {
        return Err(format!("{failures} of {} shapes failed their check", SHAPES.len()).into());
    }
    Ok(())
}
