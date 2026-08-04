//! Kernels written against the API we want.
//!
//! | Kernel | Status |
//! | --- | --- |
//! | [`gemm`] | runs — checked against a CPU reference by [`gemm::check`] |
//! | [`gemm_sol`] | runs — all three size-specialized entries checked exactly by [`gemm_sol::check`] |
//! | [`softmax`] | runs — checked against a CPU reference by [`softmax::check`] |
//! | [`layernorm`] | runs — both kernels checked against CPU references by [`layernorm::check`] and [`layernorm::check_group`] |
//! | [`flash_forward`] | runs — checked against a CPU reference by [`flash_forward::check`] |
//!
//! **Runs** means the kernel has a launcher and a CPU reference and exits
//! non-zero on a wrong number; **compiles** means it does not. Every kernel is
//! in the default build and there are no cargo features left, which is what
//! makes `scripts/modal-run build` — a real `sm_100a` codegen of this crate — a
//! regression gate on all five.
//!
//! `main` prints the table, then the blocks-per-SM the driver says each
//! kernel's own launch envelope admits, then runs every kernel that has a
//! launcher. On a B200 it reports numbers and exits non-zero when they are
//! wrong; off a GPU it degrades to the table.
//!
//! [`bench`] is the clock and nothing else, because [`softmax::bench`] and
//! [`layernorm::bench`] are written against it and `experiments/src/bench.rs`
//! includes this file's `bench.rs` rather than carrying a second copy. The
//! sweeps, the ablation cube and the cuBLASLt denominator live in
//! `experiments/`, and the design notes behind this crate are in
//! `docs/kernels/`.

use std::process::ExitCode;

pub mod bench;
pub mod flash_forward;
pub mod gemm;
pub mod gemm_sol;
pub mod layernorm;
pub mod softmax;

/// A kernel's launch envelope, as the example derives it from the library's
/// own shape constants.
struct Example {
    name: &'static str,
    status: &'static str,
    threads: u32,
    shared_bytes: usize,
    /// The PTX entry name. `None` where the occupancy query is the wrong
    /// question — it takes a block shape and no cluster, so a
    /// `#[cluster_launch]` kernel would be answered about a launch it never
    /// performs.
    entry: Option<&'static str>,
}

fn examples() -> Vec<Example> {
    vec![
        Example {
            name: "gemm",
            status: "runs",
            threads: gemm::THREADS,
            shared_bytes: gemm::STAGED_SHARED_BYTES,
            entry: None,
        },
        Example {
            name: "gemm_sol",
            status: "runs (three cluster-tile entries)",
            threads: gemm_sol::THREADS,
            shared_bytes: gemm_sol::LARGE_SHARED_BYTES,
            entry: None,
        },
        Example {
            name: "softmax",
            status: "runs",
            threads: softmax::THREADS,
            shared_bytes: softmax::SHARED_BYTES,
            entry: Some("softmax_rows"),
        },
        Example {
            name: "layernorm",
            // A status longer than the table's 38-column field silently pushes
            // the shared-memory column out of the table.
            status: "runs",
            threads: layernorm::THREADS,
            shared_bytes: layernorm::SHARED_BYTES,
            entry: Some("layernorm_rows"),
        },
        Example {
            name: "groupnorm",
            status: "runs",
            threads: layernorm::THREADS,
            shared_bytes: layernorm::SHARED_BYTES,
            entry: Some("groupnorm_tile"),
        },
        Example {
            name: "flash_forward",
            status: "runs",
            threads: flash_forward::THREADS,
            shared_bytes: flash_forward::SHARED_BYTES,
            entry: Some("flash_forward"),
        },
    ]
}

/// Blocks per SM the driver says each kernel's own launch envelope admits, and
/// the warps that comes to.
///
/// A register count off `ptxas` is half an occupancy argument; this is the half
/// that depends on the launch rather than on the code. The driver is asked
/// rather than the arithmetic reproduced here, because the shared-memory
/// carveout it picks is its own business.
fn occupancy(context: &std::sync::Arc<cuda_core::CudaContext>) {
    let Ok(module) = cuda_host::load_embedded_module(context, env!("CARGO_PKG_NAME")) else {
        return;
    };
    println!("\noccupancy at each kernel's own launch envelope, per SM:");
    println!(
        "{:<16}{:>8}{:>14}{:>12}{:>11}",
        "kernel", "threads", "shared", "blocks/SM", "warps/SM"
    );
    let mut notes = Vec::new();
    for example in examples() {
        let measured = measure(&module, &example);
        let (blocks, warps) = match &measured {
            Occupancy::Blocks(blocks) => (
                blocks.to_string(),
                (blocks * example.threads / 32).to_string(),
            ),
            Occupancy::Cluster => ("cluster".to_string(), "—".to_string()),
            Occupancy::TooLarge(_) => ("too large".to_string(), "—".to_string()),
            Occupancy::Failed(_) => ("no answer".to_string(), "—".to_string()),
        };
        match measured {
            Occupancy::TooLarge(error) => notes.push(format!("{}: {error}", example.name)),
            Occupancy::Failed(error) => notes.push(format!("{}: {error}", example.name)),
            Occupancy::Blocks(_) | Occupancy::Cluster => {}
        }
        println!(
            "{:<16}{:>8}{:>12} B{:>12}{:>11}",
            example.name, example.threads, example.shared_bytes, blocks, warps
        );
    }
    for note in notes {
        println!("  {note}");
    }
}

/// What the occupancy query has to say about one kernel — four outcomes and
/// not an `Option`, because "nobody opted this function in", "these tiles do
/// not fit" and "the driver refused something on the way" all used to print as
/// the same zero.
enum Occupancy {
    Blocks(u32),
    Cluster,
    TooLarge(kittens::launch::SharedPlanTooLarge),
    Failed(String),
}

fn measure(module: &std::sync::Arc<cuda_core::CudaModule>, example: &Example) -> Occupancy {
    use kittens::launch::AdmitError;

    let Some(entry) = example.entry else {
        return Occupancy::Cluster;
    };
    let bytes = example.shared_bytes as u32;
    let function = match module.load_function(entry) {
        Ok(function) => function,
        Err(error) => return Occupancy::Failed(format!("no {entry} in the module: {error}")),
    };
    // The opt-in is part of the envelope: without it the driver answers 0 for a
    // plan over 48 KiB exactly as it does for one too big to fit.
    match kittens::launch::admit_shared_plan(&function, bytes) {
        Ok(()) => {}
        Err(AdmitError::TooLarge(error)) => return Occupancy::TooLarge(error),
        Err(AdmitError::Driver(error)) => {
            return Occupancy::Failed(format!("admitting {bytes} B: {error}"));
        }
    }
    match function.max_active_blocks_per_multiprocessor(example.threads, bytes) {
        Ok(blocks) => Occupancy::Blocks(blocks),
        Err(error) => Occupancy::Failed(format!("occupancy query: {error}")),
    }
}

/// Every kernel with a launcher, run and checked. `Err` is a wrong number and
/// not a missing GPU — the caller decides what a missing GPU means.
fn checks(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Vec<(&'static str, Result<String, String>)> {
    vec![
        (
            "gemm",
            gemm::check(context).map_err(|error| error.to_string()),
        ),
        (
            "gemm_sol",
            gemm_sol::check(context).map_err(|error| error.to_string()),
        ),
        (
            "softmax",
            softmax::check(context).map_err(|error| error.to_string()),
        ),
        (
            "layernorm",
            layernorm::check(context).map_err(|error| error.to_string()),
        ),
        (
            "groupnorm",
            layernorm::check_group(context).map_err(|error| error.to_string()),
        ),
        (
            "flash_forward",
            flash_forward::check(context).map_err(|error| error.to_string()),
        ),
    ]
}

fn main() -> ExitCode {
    println!(
        "{:<16}{:<38}{:>8}{:>14}",
        "kernel", "status", "threads", "shared"
    );
    for example in examples() {
        println!(
            "{:<16}{:<38}{:>8}{:>13} B",
            example.name, example.status, example.threads, example.shared_bytes
        );
    }

    // A build box has no driver, and only a device that exists can fail a
    // check.
    let Ok(context) = cuda_core::CudaContext::new(0) else {
        println!("\nno CUDA device: kernels built, not run");
        return ExitCode::SUCCESS;
    };

    occupancy(&context);

    println!();
    let mut failures = 0usize;
    for (name, result) in checks(&context) {
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
