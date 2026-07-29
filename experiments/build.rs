//! Link `libcublasLt` when the `cublas` feature asks for it, and do nothing at
//! all when it does not.
//!
//! This file is the entire answer to "can the baseline be linked in-process?".
//! `cargo oxide build` and `cargo oxide run` shell out to plain `cargo` — they
//! only inject a codegen backend through `RUSTFLAGS` and never touch the link
//! line — so an ordinary build script's `cargo:rustc-link-*` directives are
//! honoured exactly as they would be under bare `cargo`. That is not an
//! assumption: `cuda-bindings/build.rs` upstream links the CUDA *driver* the
//! same way, and every `cargo oxide` build in this repo already goes through
//! it. cuBLASLt is one more `dylib=` on the same search paths.
//!
//! **Nothing here runs unless the feature is on.** Tier 1 CI has no CUDA
//! toolkit, and a build script that probed for one unconditionally would fail
//! the build it was supposed to leave alone.
//!
//! The rpath is deliberate. The link-time `-L` is enough to *resolve* the
//! symbol, but the produced binary still has to find `libcublasLt.so` at run
//! time, and `cargo oxide run` sets no rpath of its own. Baking the toolkit's
//! own library directory into the binary means the benchmark does not depend
//! on whoever ran it having the right `LD_LIBRARY_PATH`.

use std::path::PathBuf;

/// Where the CUDA toolkit is, in the order upstream's `cuda-bindings/build.rs`
/// consults — same variables, so a box that can build this repo at all can
/// build this feature.
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// Toolkit root when none of [`TOOLKIT_ENV_VARS`] is set.
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";

/// The shared object the `cublas` feature needs, as the loader names it.
const LIBRARY_FILE: &str = "libcublasLt.so";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for variable in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    // Cargo sets this for every enabled feature. Off by default, so the
    // toolkit-less build below never happens on a box that has no toolkit.
    if std::env::var_os("CARGO_FEATURE_CUBLAS").is_none() {
        return;
    }

    let toolkit = TOOLKIT_ENV_VARS
        .iter()
        .find_map(|variable| std::env::var(variable).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string());

    let candidates = library_directories(&toolkit);
    if !candidates
        .iter()
        .any(|dir| dir.join(LIBRARY_FILE).is_file())
    {
        // A linker error naming an unresolved `cublasLtCreate` sends the next
        // person looking at the Rust; the problem is always the toolkit.
        panic!(
            "the `cublas` feature needs {LIBRARY_FILE}, and no candidate directory under the \
             CUDA toolkit at `{toolkit}` has one.\nProbed:\n{}\nA *devel* CUDA image has it; a \
             runtime image does not. Set CUDA_TOOLKIT_PATH or CUDA_HOME, or build without \
             `--features cublas`.",
            candidates
                .iter()
                .map(|dir| format!("  {}", dir.join(LIBRARY_FILE).display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    for directory in &candidates {
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
    }
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}

/// Toolkit directories that may hold [`LIBRARY_FILE`], in search order.
///
/// `lib64` is the classic install; `targets/<triple>/lib` is the layout the
/// redistributable packages use, and CUDA 13's own `lib64` is a symlink into
/// it. Both are offered because which one is real varies by image, and a
/// search path that does not exist costs the linker nothing.
fn library_directories(toolkit: &str) -> Vec<PathBuf> {
    let base = PathBuf::from(toolkit);
    let mut directories = vec![base.join("lib64")];
    if let Ok(target) = std::env::var("TARGET") {
        // `x86_64-unknown-linux-gnu` is `x86_64-linux` to the CUDA packagers.
        let (architecture, rest) = target.split_once('-').unwrap_or((target.as_str(), ""));
        if rest.contains("linux") {
            directories.push(
                base.join("targets")
                    .join(format!("{architecture}-linux"))
                    .join("lib"),
            );
        }
    }
    directories.into_iter().filter(|dir| dir.is_dir()).collect()
}
