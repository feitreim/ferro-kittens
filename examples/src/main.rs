//! Kernels written against the API we want.
//!
//! Four kernels, each with a header stating whether it **runs** or only
//! **compiles**. There used to be a third status — *aspirational*, naming API
//! that did not exist yet — and an example in it was not a placeholder but a
//! precise statement of a missing surface, in the only terms that matter, which
//! is what a kernel author has to type. The diff between these files and the
//! library *was* the backlog; `examples/README.md` collects what it produced.
//!
//! | Kernel | Status |
//! | --- | --- |
//! | [`gemm`] | runs — checked against a CPU reference by [`gemm::check`] |
//! | [`softmax`] | runs — checked against a CPU reference by [`softmax::check`] |
//! | [`flash_forward`] | compiles — no launcher yet |
//! | [`layernorm`] | compiles — both kernels, no launcher yet |
//!
//! **Every kernel here is in the default build**, and there are no cargo
//! features left. That is what makes `modal run modal_app.py::build` — a real
//! `sm_100a` codegen of this crate — a regression gate on all four: a gated
//! kernel is not in the default feature set, so it was never compiled by the
//! thing that compiles this crate, and the example exercising the most of the
//! library at once was the one nothing checked.
//!
//! **Compiles is not runs**, and the distinction is the point of the column.
//! Two of these have a launcher and a CPU reference and exit non-zero on a
//! wrong number; two have neither. Giving the other two a launcher is real work
//! with a real failure mode — see the seed argument in [`softmax::permutation`]
//! — and not a status to be assigned.
//!
//! Until #8 there was no launcher here at all, because `global.rs` built one
//! shape of tensor map (3-D bf16 panels) and the GEMM's operands are 2-D.
//! `main` now prints the table and then runs every kernel that has one, so on
//! a B200 this binary reports numbers and exits non-zero when they are wrong;
//! off a GPU it degrades to the table it always printed.
//!
//! Pass `bench` and it says how *fast* the ones that run are instead — see
//! [`bench`], which checks each size before it times it.

use std::process::ExitCode;

pub mod bench;
pub mod flash_forward;
pub mod gemm;
pub mod layernorm;
pub mod softmax;

/// A kernel's launch envelope, as the example derives it from the library's
/// own shape constants — the numbers a host launcher would need.
struct Example {
    name: &'static str,
    status: &'static str,
    threads: u32,
    shared_bytes: usize,
    /// The PTX entry name, for the occupancy query — `None` for a kernel the
    /// query cannot describe. `cuOccupancyMaxActiveBlocksPerMultiprocessor`
    /// takes a block shape and no cluster, so a `#[cluster_launch]` kernel
    /// would get an answer about a launch it never performs.
    entry: Option<&'static str>,
}

fn examples() -> Vec<Example> {
    vec![
        Example {
            name: "gemm",
            status: "runs",
            threads: gemm::THREADS,
            shared_bytes: gemm::SHARED_BYTES,
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
            // Kept inside the table's 38-column status field, which a longer
            // string silently pushes the shared-memory column out of.
            status: "compiles (both kernels)",
            threads: layernorm::THREADS,
            shared_bytes: layernorm::SHARED_BYTES,
            entry: Some("layernorm_rows"),
        },
        Example {
            name: "flash_forward",
            status: "compiles (no launcher yet)",
            threads: flash_forward::THREADS,
            shared_bytes: flash_forward::SHARED_BYTES,
            // No launcher, but a plain `#[kernel]` rather than a
            // `#[cluster_launch]`, so the occupancy query describes it
            // faithfully — and this is the kernel carrying the most library
            // surface, which makes its envelope the one worth watching.
            entry: Some("flash_forward"),
        },
    ]
}

/// Blocks per SM the driver says each kernel's own launch envelope admits, and
/// the warps that comes to.
///
/// A register count off `ptxas` is half an occupancy argument; this is the
/// other half, and it is the half that depends on the launch rather than on the
/// code. #47 is the reason it is printed: a kernel can be slow because it is
/// waiting, or because too few of it fit on an SM to have anything to wait
/// *with*, and those want opposite fixes. The driver is asked rather than the
/// arithmetic reproduced here, because the shared-memory carveout it picks is
/// its own business and not a number this file can derive.
///
/// The envelope includes the *opt-in* (#70), which is why
/// [`kittens::launch::admit_shared_plan`] is called before the query and not
/// only before a launch. A plan over 48 KiB is inadmissible on a function
/// nobody opted in for, and the driver answers 0 for that exactly as it
/// answers 0 for a plan too big to fit — so without this line the column
/// reports the omission and reads like the tiles.
fn occupancy(context: &std::sync::Arc<cuda_core::CudaContext>) {
    let Ok(module) = cuda_host::load_embedded_module(context, env!("CARGO_PKG_NAME")) else {
        return;
    };
    println!("\noccupancy at each kernel's own launch envelope, per SM:");
    println!(
        "{:<16}{:>8}{:>14}{:>12}{:>11}",
        "kernel", "threads", "shared", "blocks/SM", "warps/SM"
    );
    for example in examples() {
        let blocks = example.entry.and_then(|entry| {
            let function = module.load_function(entry).ok()?;
            kittens::launch::admit_shared_plan(&function, example.shared_bytes as u32).ok()?;
            function
                .max_active_blocks_per_multiprocessor(example.threads, example.shared_bytes as u32)
                .ok()
        });
        let (blocks, warps) = match blocks {
            Some(blocks) => (
                blocks.to_string(),
                (blocks * example.threads / 32).to_string(),
            ),
            None => ("cluster".to_string(), "—".to_string()),
        };
        println!(
            "{:<16}{:>8}{:>12} B{:>12}{:>11}",
            example.name, example.threads, example.shared_bytes, blocks, warps
        );
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
            "softmax",
            softmax::check(context).map_err(|error| error.to_string()),
        ),
    ]
}

fn main() -> ExitCode {
    // `cargo oxide run kittens-examples -- bench` (#40): the same kernels at
    // several sizes, checked and then timed. It lives behind an argument rather
    // than in the default path so the correctness run stays a few seconds.
    if std::env::args().nth(1).as_deref() == Some("bench") {
        return bench::main();
    }

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

    // A build box has no driver, and the table above is the whole point of the
    // binary there. Only a device that exists can fail a check.
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
