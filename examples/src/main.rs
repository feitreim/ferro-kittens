//! Kernels written against the API we want.
//!
//! Four kernels, each with a header stating whether it **runs**, whether it
//! only **compiles**, or whether it is **aspirational** — naming API that does
//! not exist yet, with the issue numbers it is blocked on. An aspirational
//! example is not a placeholder: it is a precise statement of a missing
//! surface, in the only terms that matter, which is what a kernel author has to
//! type. The diff between these files and the library *is* the backlog.
//! `examples/README.md` collects it.
//!
//! | Kernel | Status |
//! | --- | --- |
//! | [`gemm`] | runs — checked against a CPU reference by [`gemm::check`] |
//! | [`softmax`] | runs — checked against a CPU reference by [`softmax::check`] |
//! | `flash_forward` | aspirational — `--features flash` |
//! | [`layernorm`] | `layernorm_rows` compiles; `groupnorm_tile` aspirational — `--features layernorm` |
//!
//! Each aspirational kernel sits behind its own feature and is off by default,
//! so anything in the default build genuinely compiles and a reader can tell
//! the two apart without running anything. Turning a feature on is how you
//! read that kernel's gap list: the compiler errors *are* the missing API,
//! reported at the call sites that want it.
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
pub mod gemm;
// One of its two kernels compiles in the default build; the `layernorm`
// feature adds the other, which is still aspirational (#3). The gate is on the
// kernel rather than the module, so the file that documents the gap is the
// file that builds.
pub mod layernorm;
pub mod softmax;

#[cfg(feature = "flash")]
pub mod flash_forward;

/// A kernel's launch envelope, as the example derives it from the library's
/// own shape constants — the numbers a host launcher would need.
struct Example {
    name: &'static str,
    status: &'static str,
    threads: u32,
    shared_bytes: usize,
}

/// `mut` is used only when an aspirational feature is on, which is the default
/// build's whole point.
#[allow(unused_mut)]
fn examples() -> Vec<Example> {
    let mut examples = vec![
        Example {
            name: "gemm",
            status: "runs",
            threads: gemm::THREADS,
            shared_bytes: gemm::SHARED_BYTES,
        },
        Example {
            name: "softmax",
            status: "runs",
            threads: softmax::THREADS,
            shared_bytes: softmax::SHARED_BYTES,
        },
        Example {
            name: "layernorm",
            status: if cfg!(feature = "layernorm") {
                "compiles; groupnorm_tile aspirational (#3)"
            } else {
                "compiles (layernorm_rows)"
            },
            threads: layernorm::THREADS,
            shared_bytes: layernorm::SHARED_BYTES,
        },
    ];
    #[cfg(feature = "flash")]
    examples.push(Example {
        name: "flash_forward",
        status: "aspirational (#7, #11, #22, #23, #31)",
        threads: flash_forward::THREADS,
        shared_bytes: flash_forward::SHARED_BYTES,
    });
    examples
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
