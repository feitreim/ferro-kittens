//! The lab notebook: every rung, every probe, and the clock they were read on.
//!
//! `examples/` shows a reader how to write a kernel against this library, and
//! ships one GEMM. This crate is what that one kernel was chosen *by*. It holds
//! the tile and depth sweep, the traversal sweep, the ablation cube, the four
//! epilogue families and their doubling ladders, the warp-specialized variant,
//! the cuBLASLt denominator, the benchmark harness, and `README.md` — which is
//! the measurement record and is longer than all of the code in `examples/`.
//!
//! **About a third of the entry points here compute a deliberately wrong `C`.**
//! That is not a defect: an epilogue cannot be priced by deleting it, because a
//! dead-code pass then deletes what fed it, so the probes double a pass and
//! aim the extra bytes somewhere they stay in L2. Every one of them says so on
//! its own doc, and [`gemm::Epilogue::exact`] is what keeps them off the
//! correctness gate.
//!
//! ```text
//! scripts/modal-run examples                   # the correctness gate, both crates
//! scripts/modal-run bench --case gemm-depth    # or tile, staged, widths,
//! scripts/modal-run clc_bench                  #    residual, swizzle, repro,
//! scripts/modal-run ws_bench                   #    ablation, epilogue, sol,
//!                                              #    ldmatrix
//! ```
//!
//! **Every rung here is on a correctness gate**, which is [`gemm::check`] and
//! [`gemm_ws::check`] under the `check` argument. Splitting the crates did not
//! split that: `modal_app.py::examples` runs both binaries, so a rung that
//! computes a wrong `C` fails the same job it failed before.
//!
//! # Why this is a fourth crate and not a feature of the third
//!
//! `examples/` and this one want different amounts of polish, different
//! lifetimes and different CI cost. A sweep that answered its question in #98
//! is worth keeping and is not worth reading; a teaching example is the other
//! way round. Two crates is what lets the second one be short.
//!
//! `softmax` and `layernorm` are the seam. Their kernels are the teaching
//! crate's and are compiled from its files — `#[path]` below, not a copy — and
//! their verify-then-time entry points are called from [`bench`], which is
//! this crate's.
//!
//! **The two GEMMs are the other seam, and they are copies on purpose.**
//! [`gemm`] and [`gemm_sol`] each hold the dialed body every arm instantiates;
//! `examples/` ships the same entry points with the dials baked to one
//! configuration. A `#[path]` cannot serve both — the teaching copy is supposed
//! to have no dials in it — so the two crates emit the same kernel names from
//! two files, and `regcount`'s opcode census compares them row for row. That
//! comparison is the check that a shipped kernel and the arms it is measured
//! against are still the same instructions.

use std::process::ExitCode;

pub mod bench;
/// The GEMM's denominator, behind the off-by-default `cublas` feature (#92).
/// Absent from a default build, which is what keeps a crate that anyone can
/// compile from needing a CUDA toolkit to link against.
#[cfg(feature = "cublas")]
pub mod cublaslt;
pub mod gemm;
/// The dialed `gemm_sol`: the shipped body with `ABLATE`, `DRAIN`, `WATCH`,
/// `FOLD` and the warpgroup count still parameters, which is what
/// [`sol_ablate`] and [`sol_watch`] take it apart through. `examples/` ships
/// the same three entries with all five baked in.
pub mod gemm_sol;
/// The kernel [`gemm_sol`] is a port *of*, unported — upstream's device code
/// byte for byte, on this crate's clock, so the port's distance from cuBLASLt
/// can be split into what the port costs and what upstream itself costs.
///
/// Behind an off-by-default feature, and not for tidiness: with it on, the
/// crate's embedded PTX bundle does not load at all unless `opt` is given
/// `-switch-to-lookup=false`. See the feature's comment in `Cargo.toml` and
/// `modal_app.py::upstream_bench`.
#[cfg(feature = "gemm-sol-upstream")]
pub mod gemm_sol_upstream;
pub mod gemm_ws;
/// `softmax` with its loads at `ldmatrix.m8n8.x4` — #131's arm, off the
/// teaching crate's own device body with the load width as its only parameter.
pub mod softmax_x4;
pub mod sol;
pub mod sol_ablate;
pub mod sol_watch;

/// The teaching crate's kernels, compiled here so [`bench`] can time them
/// against their own CPU references.
///
/// `#[path]` and not a second copy: `softmax::bench` and `layernorm::bench` are
/// written against `examples/src/bench.rs`'s clock, which this crate's [`bench`]
/// re-exports, so both crates compile one set of source files and the rows in
/// the sweep are the rows the example ships.
#[path = "../../examples/src/flash_forward.rs"]
pub mod flash_forward;
#[path = "../../examples/src/layernorm.rs"]
pub mod layernorm;
#[path = "../../examples/src/softmax.rs"]
pub mod softmax;

/// What each argument runs, printed when none is given.
const COMMANDS: [(&str, &str); 4] = [
    (
        "check",
        "every rung against a CPU reference — `modal_app.py::examples`",
    ),
    (
        "bench [<case> [<m> <n> <k>]]",
        "the size sweeps and every named sweep — `modal_app.py::bench`",
    ),
    (
        "clc",
        "the GEMM's three item sources on one clock (#88) — `clc_bench`",
    ),
    (
        "ws [shallow]",
        "the warp-specialized GEMM against the one it is a variant of — `ws_bench`; `shallow` is #188's K = 3072 arm",
    ),
];

/// Every rung and probe that computes a GEMM, launched and compared against a
/// CPU reference. `Err` is a wrong number and not a missing GPU — the caller
/// decides what a missing GPU means.
///
/// This is the whole of what `modal_app.py::examples` gained when the crates
/// split. `gemm::check` walks two check sizes, three traversal widths, both
/// schedulers, seven tile rungs and every exact epilogue; `gemm_ws::check` does
/// the same for the warp-specialized variant. The probes that compute a wrong
/// `C` on purpose are excluded by `Epilogue::exact` and `Ablation::exact`,
/// which is stated on each of them rather than assumed here.
fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> ExitCode {
    #[allow(unused_mut)]
    let mut checks = vec![
        (
            "gemm",
            gemm::check(context).map_err(|error| error.to_string()),
        ),
        (
            "gemm_ws",
            gemm_ws::check(context).map_err(|error| error.to_string()),
        ),
        // The rebuilt `gemm_sol` drains. They are the only arms in
        // `sol_ablate` that compute a right `C`, and a batched LDTM that
        // returns its registers in another order is exactly the failure this
        // catches.
        (
            "gemm_sol_drains",
            sol_ablate::check(context).map_err(|error| error.to_string()),
        ),
        // `load_tile_x4`'s gate (#131). The teaching crate checks the shipped
        // load width and this is the other one, on the same reference — an arm
        // whose numbers are quoted has to be an arm that computes a softmax.
        (
            "softmax_x4",
            softmax_x4::check(context).map_err(|error| error.to_string()),
        ),
    ];
    // Only when the vendored upstream copy is compiled in, which is only under
    // `upstream_bench`'s `opt` wrapper — see that entry point.
    #[cfg(feature = "gemm-sol-upstream")]
    checks.push((
        "gemm_sol_upstream",
        gemm_sol_upstream::check(context).map_err(|error| error.to_string()),
    ));
    let mut failures = 0usize;
    for (name, result) in checks {
        match result {
            Ok(note) => println!("pass  {name:<16}  {note}"),
            Err(error) => {
                println!("FAIL  {name:<16}  {error}");
                failures += 1;
            }
        }
    }
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    // The correctness gate every rung in this crate is on.
    if std::env::args().nth(1).as_deref() == Some("check") {
        let Ok(context) = cuda_core::CudaContext::new(0) else {
            println!("check needs a CUDA device");
            return ExitCode::FAILURE;
        };
        return check(&context);
    }

    // The size sweeps and, through their own arguments, every named sweep in
    // `gemm.rs`. This is the argument `modal_app.py::bench` passes through.
    if std::env::args().nth(1).as_deref() == Some("bench") {
        return bench::main();
    }

    // The GEMM's three item sources on one clock, with the ragged-wave
    // prediction beside them (#88). Its own argument rather than a row of
    // `bench`, because it times one kernel three ways and its sizes are chosen
    // by what the prediction does across them rather than by what crosses a
    // regime.
    if std::env::args().nth(1).as_deref() == Some("clc") {
        let Ok(context) = cuda_core::CudaContext::new(0) else {
            println!("clc needs a CUDA device");
            return ExitCode::FAILURE;
        };
        return match gemm::compare(&context) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // The warp-specialized GEMM and the one it is a variant of, on one clock,
    // in one container, with cuBLASLt beside them. Its own argument for the
    // reason `clc` has one — it times two *kernels* against each other rather
    // than sweeping one kernel's sizes, and `bench`'s `Case` is the wrong shape
    // for that.
    if std::env::args().nth(1).as_deref() == Some("ws") {
        let Ok(context) = cuda_core::CudaContext::new(0) else {
            println!("ws needs a CUDA device");
            return ExitCode::FAILURE;
        };
        // `ws shallow` is #188: the same A/B at the K = 3072 geometries
        // oxide-train#80 counts in, where the overlap the design buys is worth
        // the most — every table `compare` prints is at K = M = N, which is
        // where it is worth the least.
        let run = if std::env::args().nth(2).as_deref() == Some("shallow") {
            gemm_ws::shallow(&context, bench::CUBLASLT)
        } else {
            gemm_ws::compare(&context, bench::CUBLASLT)
        };
        return match run {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                println!("FAIL  {error}");
                ExitCode::FAILURE
            }
        };
    }

    // No default run. Every entry point here books a B200 for minutes, and a
    // binary that picks one for you is a binary that spends money on the wrong
    // sweep. `examples/` is the one that runs everything it has when asked for
    // nothing.
    println!("kittens-experiments takes one of:");
    for (command, about) in COMMANDS {
        println!("  {command:<30}{about}");
    }
    ExitCode::FAILURE
}
