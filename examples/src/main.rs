//! Kernels written against the API we want.
//!
//! Four kernels, each with a header stating whether it **compiles** today or
//! is **aspirational** — naming API that does not exist yet, with the issue
//! numbers it is blocked on. An aspirational example is not a placeholder: it
//! is a precise statement of a missing surface, in the only terms that matter,
//! which is what a kernel author has to type. The diff between these files and
//! the library *is* the backlog. `examples/README.md` collects it.
//!
//! | Kernel | Status |
//! | --- | --- |
//! | [`gemm`] | compiles — `cargo oxide build kittens-examples --arch sm_100a` |
//! | `flash_forward` | aspirational — `--features flash` |
//! | `softmax` | aspirational — `--features softmax` |
//! | `layernorm` | aspirational — `--features layernorm` |
//!
//! Each aspirational kernel sits behind its own feature and is off by default,
//! so anything in the default build genuinely compiles and a reader can tell
//! the two apart without running anything. Turning a feature on is how you
//! read that kernel's gap list: the compiler errors *are* the missing API,
//! reported at the call sites that want it.
//!
//! There is no host launcher here. These are statements about kernel code, and
//! the operands two of them need cannot be described host-side anyway:
//! `global.rs` builds one shape of tensor map (3-D bf16 panels), and both the
//! GEMM's 2-D operands and layernorm's parameter vectors want another (#8).

pub mod gemm;

#[cfg(feature = "flash")]
pub mod flash_forward;
#[cfg(feature = "layernorm")]
pub mod layernorm;
#[cfg(feature = "softmax")]
pub mod softmax;

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
    let mut examples = vec![Example {
        name: "gemm",
        status: "compiles",
        threads: gemm::THREADS,
        shared_bytes: gemm::SHARED_BYTES,
    }];
    #[cfg(feature = "flash")]
    examples.push(Example {
        name: "flash_forward",
        status: "aspirational (#5, #6, #7, #11)",
        threads: flash_forward::THREADS,
        shared_bytes: flash_forward::SHARED_BYTES,
    });
    #[cfg(feature = "softmax")]
    examples.push(Example {
        name: "softmax",
        status: "aspirational (#5, #6, #9, #22, #25)",
        threads: softmax::THREADS,
        shared_bytes: softmax::SHARED_BYTES,
    });
    #[cfg(feature = "layernorm")]
    examples.push(Example {
        name: "layernorm",
        status: "aspirational (#3, #5, #6, #9, #13)",
        threads: layernorm::THREADS,
        shared_bytes: layernorm::SHARED_BYTES,
    });
    examples
}

fn main() {
    // Anchors the compiled artifact: the generated loader is what carries the
    // kernels, and nothing else in this binary refers to them.
    let _ = gemm::kernels::load;

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
}
