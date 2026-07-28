"""Build and run ferro-kittens' device test harness on a Modal B200.

cuda-oxide is a rustc codegen backend (Rust -> PTX). The only place the full
toolchain can live is a Linux box with an NVIDIA GPU + CUDA 13 + LLVM 21, so we
bake all of that into a Modal image once and reuse it. The image recipe below is
a verbatim copy of rust-trainer's; Modal images are content-addressed, so an
identical recipe reuses that project's cached layers instead of rebuilding the
codegen backend from scratch.

Local usage:
    modal run modal_app.py::build     # host tests + a CPU-only device build
    modal run modal_app.py::regcount  # ptxas -v register/spill table, no GPU
    modal run modal_app.py            # the device tests, on a B200
    modal run modal_app.py::examples  # the examples crate's kernels, on a B200
    modal run modal_app.py::bench     # those kernels timed at several sizes
    modal run modal_app.py::doctor    # env / GPU sanity check
"""

import re
import subprocess
from pathlib import Path

import modal

# Keep this revision in sync with the git deps in Cargo.toml: the codegen
# backend and the device/host/core crates must come from the same revision.
CUDA_OXIDE_REF = "b099f64c1a32869b74be99f4f88242fb68655b51"
RUST_TOOLCHAIN = "nightly-2026-04-03"
GIT_REPO = "https://github.com/NVlabs/cuda-oxide.git"

DEFAULT_GPU = "B200"  # the library targets sm_100a exclusively (tcgen05).
PROJECT_DIR = "/root/project"  # the repo, mounted live at run time

# Mirror of the dependency block in device-tests/Cargo.toml. Used only to warm
# the backend + git-dep caches into an image layer so per-run builds are fast.
WARMUP_CARGO_TOML = f"""
[package]
name = "warmup"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
cuda-device = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
cuda-host = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
cuda-core = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
"""

WARMUP_MAIN_RS = """
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = c.get_mut(idx) { *e = a[i] + b[i]; }
    }
}
fn main() { let _ = (CudaContext::new(0), LaunchConfig::for_num_elems(1)); }
"""

image = (
    # CUDA 13 devel base -- same as cuda-oxide's own .devcontainer/Dockerfile.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": "/usr/bin/llc-21",
            "LIBCLANG_PATH": "/usr/lib/llvm-21/lib",
            "LLVM_CONFIG_PATH": "/usr/bin/llvm-config-21",
            "PATH": (
                "/root/.cargo/bin:/usr/lib/llvm-21/bin:"
                "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
        }
    )
    .apt_install(
        "ca-certificates", "curl", "g++", "gcc", "git", "gnupg",
        "libc6-dev", "libssl-dev", "make", "pkg-config", "xz-utils",
    )
    # LLVM 21 toolchain (NVPTX target + clang headers for bindgen).
    .run_commands(
        "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key "
        "| gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg",
        'echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] '
        'https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" '
        "> /etc/apt/sources.list.d/llvm-toolchain-noble-21.list",
        "apt-get update && apt-get install -y --no-install-recommends "
        "clang-21 libclang-common-21-dev lld-21 llvm-21 llvm-21-dev "
        "&& rm -rf /var/lib/apt/lists/*",
    )
    # Pinned nightly Rust with the components the codegen backend needs.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain none --profile minimal",
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal "
        "-c rust-src -c rustc-dev -c llvm-tools",
        f"cargo +{RUST_TOOLCHAIN} install --git {GIT_REPO} --rev {CUDA_OXIDE_REF} cargo-oxide",
    )
    # Build the codegen backend (slow, one time; baked into this image layer)
    # and compile a trivial kernel end-to-end to prove the toolchain works.
    #
    # cargo-oxide links libcuda (the *driver*), which isn't present at build
    # time (no GPU). The toolkit ships a driver *stub* that satisfies the
    # linker; expose it ONLY here via an inline LD_LIBRARY_PATH so it never
    # shadows the real driver injected at run time.
    .run_commands(
        "mkdir -p /opt/warmup/src",
        f"cat > /opt/warmup/Cargo.toml <<'EOF'\n{WARMUP_CARGO_TOML}\nEOF",
        f"cat > /opt/warmup/src/main.rs <<'EOF'\n{WARMUP_MAIN_RS}\nEOF",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        # `cargo oxide build` bootstraps and caches the backend on first use.
        # Do not call `setup` from a standalone project: at this revision that
        # command tries to rebuild the project itself as a backend library.
        "cd /opt/warmup && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cargo oxide build warmup",
    )
    # Lints. A layer of its own, after the expensive ones, so adding it leaves
    # the shared backend-build layers cached. `RUSTUP_TOOLCHAIN` shadows
    # rust-toolchain.toml, so the component list there does not reach here.
    .run_commands(f"rustup component add clippy --toolchain {RUST_TOOLCHAIN}")
    # Live mounts (re-read each run; edits need no image rebuild). The harness
    # path-depends on the library, so both trees come along.
    .add_local_dir(str(Path(__file__).parent / "src"), f"{PROJECT_DIR}/src")
    .add_local_dir(
        str(Path(__file__).parent / "device-tests/src"),
        f"{PROJECT_DIR}/device-tests/src",
    )
    .add_local_file(str(Path(__file__).parent / "Cargo.toml"), f"{PROJECT_DIR}/Cargo.toml")
    .add_local_file(str(Path(__file__).parent / "Cargo.lock"), f"{PROJECT_DIR}/Cargo.lock")
    .add_local_file(
        str(Path(__file__).parent / "device-tests/Cargo.toml"),
        f"{PROJECT_DIR}/device-tests/Cargo.toml",
    )
    # Its lockfile too. Without it cargo re-resolves against crates.io on every
    # run, so this build tests a dependency set nothing in the repo pins and an
    # upstream release can break it with no change of ours.
    .add_local_file(
        str(Path(__file__).parent / "device-tests/Cargo.lock"),
        f"{PROJECT_DIR}/device-tests/Cargo.lock",
    )
    # The examples crate: also standalone, also path-depends on the library.
    .add_local_dir(
        str(Path(__file__).parent / "examples/src"),
        f"{PROJECT_DIR}/examples/src",
    )
    .add_local_file(
        str(Path(__file__).parent / "examples/Cargo.toml"),
        f"{PROJECT_DIR}/examples/Cargo.toml",
    )
    .add_local_file(
        str(Path(__file__).parent / "examples/Cargo.lock"),
        f"{PROJECT_DIR}/examples/Cargo.lock",
    )
    .add_local_file(
        str(Path(__file__).parent / "rust-toolchain.toml"),
        f"{PROJECT_DIR}/rust-toolchain.toml",
    )
)

app = modal.App("ferro-kittens", image=image)

HARNESS_DIR = f"{PROJECT_DIR}/device-tests"
EXAMPLES_DIR = f"{PROJECT_DIR}/examples"
# The driver stub satisfies cargo-oxide's link step where there is no GPU.
STUB_ENV = ["env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"]


def _run(cmd: list[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(cpu=8, timeout=1800)
def build() -> None:
    """Everything that does not need a GPU: the library's host surface (which
    cannot be checked off a CUDA box, `global.rs` needs cuda.h) and a full
    device build of the harness against the driver stub. Iterate compile errors
    here; the B200 function is for running, not for finding typos."""
    _run(["cargo", "clippy", "--features", "host", "--all-targets"], cwd=PROJECT_DIR)
    _run(["cargo", "test", "--features", "host"], cwd=PROJECT_DIR)
    # Docs link to `global`'s types from all over the crate, and those types
    # exist only under `host`. CI's device-only `cargo doc` allows the dangling
    # links (see lib.rs); this is the build where they must actually resolve.
    _run(
        ["env", "RUSTDOCFLAGS=-D warnings",
         "cargo", "doc", "--no-deps", "--features", "host"],
        cwd=PROJECT_DIR,
    )
    # The harness only typechecks where cuda.h is, so its lints live here too.
    _run(["cargo", "clippy", "--all-targets"], cwd=HARNESS_DIR)
    # `build` (unlike `run`) does not auto-detect the GPU arch, and tcgen05
    # exists only on Blackwell -- pin it or the artifact fails to compile.
    _run([*STUB_ENV, "cargo", "oxide", "build", "device-tests", "--arch", "sm_100a"],
         cwd=HARNESS_DIR)
    # The examples crate, same treatment. Only the kernels marked *runs* or
    # *compiles* are in the default feature set, so this is what keeps that
    # claim honest: a post-monomorphization `const { assert!(..) }` in a tile
    # shape is invisible to `cargo check` and shows up only in a real device
    # build. Its host launchers are ordinary Rust and get ordinary lints.
    _run(["cargo", "clippy", "--all-targets"], cwd=EXAMPLES_DIR)
    _run([*STUB_ENV, "cargo", "oxide", "build", "kittens-examples", "--arch", "sm_100a"],
         cwd=EXAMPLES_DIR)


@app.function(cpu=8, timeout=1800)
def gaps() -> None:
    """The aspirational examples' gap lists, as compiler errors.

    `examples/README.md` claims each aspirational kernel's remaining errors by
    name, and that claim is only worth anything if it is read off a compiler
    rather than off the last person's memory. Turning a feature on makes the
    missing API surface *be* the error list, at the call sites that want it.
    Each feature is checked on its own so an error belongs to a known kernel,
    and a non-zero exit is the expected outcome -- reported rather than raised.
    An empty list is the interesting case: the kernel is ready to leave its
    gate, which is a finding and not an error."""
    for feature in ("flash", "layernorm"):
        print(f"\n=== cargo check --features {feature} ===", flush=True)
        checked = subprocess.run(
            ["cargo", "check", "--features", feature, "--message-format", "short"],
            cwd=EXAMPLES_DIR,
        )
        print(f"=== exit {checked.returncode} ===", flush=True)


@app.function(gpu=DEFAULT_GPU, timeout=1800)
def device_tests() -> None:
    """The harness itself. One binary, every case, non-zero exit on failure."""
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _run(["cargo", "oxide", "run", "device-tests"], cwd=HARNESS_DIR)


@app.function(gpu=DEFAULT_GPU, timeout=1800)
def examples() -> None:
    """The examples crate's own launchers. Prints the status table, then runs
    every kernel that has one against its CPU reference; non-zero on a wrong
    number. This is the claim `device-tests` cannot make: not that a primitive
    behaves, but that a whole kernel written against the library computes."""
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _run(["cargo", "oxide", "run", "kittens-examples"], cwd=EXAMPLES_DIR)


@app.function(gpu=DEFAULT_GPU, timeout=1800)
def bench() -> None:
    """The same kernels, timed at several sizes, reporting achieved throughput.

    Separate from `examples` so the correctness path stays a few seconds: this
    one stages problems up to 8192^3 and launches each of them dozens of times.
    Every size is checked against the same CPU reference `examples` uses before
    it is timed at all, so a throughput figure here always belongs to a run that
    computed the right answer -- the harness has no path that prints one for a
    launch it did not verify.

    The clock is CUDA events either side of the launch, not wall clock around
    the driver call, and the headline per size is the minimum over the timed
    launches, with the median and maximum printed beside it. Each example states
    whether it is compute- or memory-bound and the table prints the matching
    metric: a FLOP/s figure for a normalization kernel would be misleading, so
    the harness carries the distinction rather than assuming one.
    """
    _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
            "--format=csv",
        ],
        cwd="/",
    )
    _run(["cargo", "oxide", "run", "kittens-examples", "--", "bench"], cwd=EXAMPLES_DIR)


@app.function(gpu=DEFAULT_GPU, timeout=600)
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


# `ptxas -v` writes one block per entry function on stderr:
#
#     ptxas info    : Compiling entry function 'foo' for 'sm_100a'
#     ptxas info    : Function properties for foo
#         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#     ptxas info    : Used 40 registers, 8192 bytes smem, 356 bytes cmem[0]
#
# The fields drift between CUDA releases (`used N barriers` appeared in 12.x),
# so each is matched independently rather than by one line-shaped regex.
_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PROPERTIES = re.compile(r"Function properties for (\S+)")
_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_REGISTERS = re.compile(r"Used (\d+) registers")
_SMEM = re.compile(r"(\d+) bytes smem")


def _parse_ptxas(log: str) -> dict[str, dict[str, int]]:
    """Per-entry-function counters, keyed by mangled kernel name."""
    kernels: dict[str, dict[str, int]] = {}
    current = None
    for line in log.splitlines():
        named = _ENTRY.search(line) or _PROPERTIES.search(line)
        if named:
            current = kernels.setdefault(
                named.group(1),
                {"registers": 0, "spill_stores": 0, "spill_loads": 0, "stack": 0, "smem": 0},
            )
            continue
        if current is None:
            continue
        if frame := _FRAME.search(line):
            current["stack"] = int(frame.group(1))
            current["spill_stores"] = int(frame.group(2))
            current["spill_loads"] = int(frame.group(3))
        if registers := _REGISTERS.search(line):
            current["registers"] = int(registers.group(1))
            current["smem"] = int(_SMEM.search(line).group(1)) if _SMEM.search(line) else 0
    return kernels


@app.function(cpu=8, timeout=1800)
def regcount(arch: str = "sm_100a", label: str = "") -> None:
    """Register pressure of every kernel the harness emits, from `ptxas -v`.

    Register count is *the* performance number for this library — a tile
    abstraction that costs registers has failed at the thing it exists for, and
    "ptxas says otherwise" is the escape hatch the README promises. `ptxas` is a
    host compiler, so this needs no GPU: build the harness, feed the emitted PTX
    back through `ptxas -v`, and print a sorted table. Run it before and after a
    change and diff the two.

    Only kernels that are actually monomorphized appear — an op no kernel calls
    emits no PTX and measures nothing, so a codegen probe in `device-tests` is
    how a bare library function gets onto this table at all.
    """
    _run([*STUB_ENV, "cargo", "oxide", "build", "device-tests", "--arch", arch], cwd=HARNESS_DIR)

    ptx_files = sorted(Path(HARNESS_DIR).rglob("*.ptx"))
    if not ptx_files:
        listing = sorted(p for p in Path(HARNESS_DIR, "target").rglob("*") if p.is_file())
        raise RuntimeError(
            "no PTX under the harness target dir; cargo-oxide's artifact layout "
            f"must have moved. Files found:\n" + "\n".join(map(str, listing[:200]))
        )

    subprocess.run(["ptxas", "--version"], check=True)
    print(f"\nregisters per thread, {arch}" + (f" — {label}" if label else ""))
    for ptx in ptx_files:
        compiled = subprocess.run(
            ["ptxas", "-v", f"-arch={arch}", "-o", "/dev/null", str(ptx)],
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            raise RuntimeError(f"ptxas failed on {ptx}:\n{compiled.stderr}")
        kernels = _parse_ptxas(compiled.stderr)
        print(f"\n{ptx.relative_to(HARNESS_DIR)}  ({len(kernels)} kernels)")
        print(f"  {'kernel':<44}{'regs':>6}{'spill st':>10}{'spill ld':>10}{'stack':>8}{'smem':>8}")
        for name, counts in sorted(kernels.items()):
            print(
                f"  {name:<44}{counts['registers']:>6}{counts['spill_stores']:>10}"
                f"{counts['spill_loads']:>10}{counts['stack']:>8}{counts['smem']:>8}"
            )


@app.local_entrypoint()
def main() -> None:
    device_tests.remote()
