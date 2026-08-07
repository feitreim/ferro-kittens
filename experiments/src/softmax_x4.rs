//! `softmax` with its shared→register loads at `ldmatrix.m8n8.x4` — #131's
//! arm, and the clock it was answered on.
//!
//! The store side has had a `.x4` form since #116 and the load side has not,
//! and the asymmetry was nobody's decision. It cannot be settled by argument:
//! `.x4` on the store side and `.x8` on the TMEM drain both paid by removing
//! *waits*, an `ldmatrix` has no wait to remove, so what is left is issue slots
//! against liveness — four matrices arriving at once where two let the compiler
//! fuse a slot through to its consumer. Nor by a register count: #47, #63, #67,
//! #76 and #81 are five occasions in this repo where that column ordered time
//! backwards.
//!
//! So it is timed, on a kernel that loads. `softmax`'s three passes are all
//! `load_tile` and nothing else in it moves, which makes it the largest share
//! of a launch this question can be asked at.
//!
//! **The kernel is `examples/src/softmax.rs`'s, not a copy.** That file's
//! device body takes the load width as a `const` parameter and the teaching
//! crate emits only `false`; this module is the `true` entry, in this crate's
//! bundle, checked by the same CPU reference and timed by the same clock. Both
//! entries are in one binary, so a pair of arms is adjacent in time rather than
//! two containers apart.
//!
//!     scripts/modal-run bench --case ldmatrix
//!
//! **The answer was a wash, very slightly the narrow load's way**: unordered at
//! 64 and 512 blocks, and 1.1% slower at 8192 with all four pairs agreeing —
//! 4235.6 GB/s against 4281.0, on 40 registers a thread against 32 and the same
//! 6 CTAs an SM. So `load_tile` still issues `.x2`, and this module is what has
//! to be re-run before anyone changes that. The table and the argument are in
//! `docs/library/ldst.md`.

use std::error::Error;
use std::sync::Arc;

use cuda_core::CudaContext;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel};

use crate::bench::{ITERATIONS, Shape, Timings, WARMUP, announce, extremes, middle, time};
use crate::softmax::{self, COLUMNS, run};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `softmax_rows` reading each `[16, 16]` block with one
    /// `ldmatrix.m8n8.x4` instead of two `.x2`s a slot apart. Same body, same
    /// shared plan, same launch envelope — the load width is the whole
    /// difference, which is what makes the pair a measurement.
    #[kernel]
    pub unsafe fn softmax_rows_x4(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        plane: i32,
    ) {
        unsafe { softmax::rows::<true>(source, destination, plane) }
    }
}

/// [`run`] with the `.x4` entry supplied as its launch — `softmax::shipped`'s
/// twin, and the only host code this arm has that the shipped one does not.
fn wide<T>(
    context: &Arc<CudaContext>,
    rows: usize,
    planes: usize,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    let module = kernels::load(context)?;
    let launch = |stream: &cuda_core::CudaStream,
                  config: cuda_core::LaunchConfig,
                  source: *const TmaDescriptor,
                  destination: *const TmaDescriptor|
     -> Result<(), Box<dyn Error>> {
        unsafe { module.softmax_rows_x4(stream, config, source, destination, 0) }?;
        Ok(())
    };
    run(context, rows, planes, &launch, then)
}

/// The `.x4` arm's verify-then-time entry point, with `softmax::bench`'s
/// signature so the two are interchangeable wherever a launch is timed.
pub fn bench(context: &Arc<CudaContext>, shape: Shape) -> Result<Timings, Box<dyn Error>> {
    wide(context, shape.m, shape.k, time).map(|(_, timings)| timings)
}

/// The same entry at the check size, against the same CPU reference the shipped
/// one answers to — `load_tile_x4`'s correctness gate, and the reason
/// `modal_app.py::examples` has a row for it.
pub fn check(context: &Arc<CudaContext>) -> Result<String, Box<dyn Error>> {
    wide(
        context,
        softmax::CHECK_ROWS,
        softmax::CHECK_PLANES,
        |_, _| Ok(()),
    )
    .map(|(note, ())| note)
}

/// Three operating points out of the softmax sweep, not one.
///
/// 64 blocks is under a wave of a 148-SM device and measures the dependent
/// chain at the head of the launch; 512 is past it; 8192 is where the kernel is
/// bandwidth-bound and the number `docs/kernels/softmax.md`'s tables are quoted
/// at. An instruction-count change has no reason to act the same in all three,
/// and a single row could not have told anyone that it did.
const SIZES: &[Shape] = &[
    Shape {
        m: 4096,
        n: COLUMNS,
        k: 2,
    },
    Shape {
        m: 32768,
        n: COLUMNS,
        k: 2,
    },
    Shape {
        m: 524288,
        n: COLUMNS,
        k: 2,
    },
];

/// Whole measurements each arm gets, taken round-robin over the two.
///
/// Round-robin because a device whose clocks step down over a sweep slows
/// whichever arm ran last, and four rather than two because the deliverable is
/// a spread — two points give a range with no interior. Both are `bench
/// repro`'s arguments (#122), applied where the difference under test is
/// expected to be small enough that they decide the answer.
const REPEATS: usize = 4;

/// Bytes a shape moves: every element read once and written once, both bf16 —
/// the `softmax` case's own denominator, so a GB/s here and a GB/s there are
/// the same quantity.
fn bytes(shape: Shape) -> f64 {
    2.0 * 2.0 * (shape.m * shape.n * shape.k) as f64
}

/// `bench ldmatrix` — the two load widths on one clock (#131).
pub fn compare(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "softmax: the shared→register load at `ldmatrix.x2` against `.x4` (#131). Both entries\n\
         are in this crate's bundle off one device body, differing in the load width and in\n\
         nothing else. {REPEATS} whole measurements of each arm, taken round-robin so every\n\
         pair is adjacent in time, each one min ms over {ITERATIONS} timed launches after\n\
         {WARMUP} warm-up, and every one of them checked against the CPU reference first."
    );
    println!(
        "\n{:<20}{:>8}{:>10}{:>10}{:>11}{:>11}{:>10}{:>10}{:>10}",
        "shape", "blocks", "x2 ms", "x4 ms", "x2 GB/s", "x4 GB/s", "x4/x2", "lowest", "highest"
    );

    for &shape in SIZES {
        let (mut narrow, mut wide) = (Vec::new(), Vec::new());
        for pass in 1..=REPEATS {
            announce(format!("{shape} x2 pass {pass}"));
            narrow.push(softmax::bench(context, shape)?.min());
            announce(format!("{shape} x4 pass {pass}"));
            wide.push(bench(context, shape)?.min());
        }
        let paired: Vec<f64> = narrow
            .iter()
            .zip(&wide)
            .map(|(narrow, wide)| wide / narrow)
            .collect();
        let rate = |milliseconds: f64| bytes(shape) / (milliseconds / 1e3) / 1e9;
        let (narrow, wide) = (middle(&narrow), middle(&wide));
        let (lowest, highest) = extremes(&paired);
        println!(
            "{:<20}{:>8}{:>10.4}{:>10.4}{:>11.1}{:>11.1}{:>10.4}{:>10.4}{:>10.4}",
            shape,
            softmax::grid(shape.m, shape.k),
            narrow,
            wide,
            rate(narrow),
            rate(wide),
            middle(&paired),
            lowest,
            highest,
        );
    }

    println!(
        "\nthe last three columns are `.x4`'s milliseconds over `.x2`'s, one paired ratio per\n\
         pass — so below 1.000 is the wide load ahead. A row whose lowest and highest straddle\n\
         1.000 has not ordered the two spellings at this sample count, whichever side its\n\
         median landed on."
    );
    Ok(())
}
