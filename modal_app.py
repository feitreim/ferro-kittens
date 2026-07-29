"""Build and run ferro-kittens' device test harness on a Modal B200.

cuda-oxide is a rustc codegen backend (Rust -> PTX). The only place the full
toolchain can live is a Linux box with an NVIDIA GPU + CUDA 13 + LLVM 21, so we
bake all of that into a Modal image once and reuse it. The image recipe below is
a verbatim copy of rust-trainer's; Modal images are content-addressed, so an
identical recipe reuses that project's cached layers instead of rebuilding the
codegen backend from scratch.

Local usage:
    modal run modal_app.py::build     # host tests + a CPU-only device build
    modal run modal_app.py::regcount  # ptxas -v register/spill table, the
                                      # shape ladder and its cliff, and the
                                      # occupancy-step gate (#95). No GPU, and
                                      # the only thing here that fails a run on
                                      # a register count.
    modal run modal_app.py            # the device tests, on a B200
    modal run modal_app.py::examples  # both kernel crates' checks, on a B200
    modal run modal_app.py::bench     # those kernels timed at several sizes
    modal run modal_app.py::clc_bench # gemm's three item sources on one clock,
                                      # against the ragged-wave prediction
    modal run modal_app.py::ladder_bench
                                      # four rungs of that ladder on a clock:
                                      # is a streamed band actually slower?
    modal run modal_app.py::bench --case gemm-depth
                                      # one table of that sweep; --m/--n/--k
                                      # narrows it further to a single row
    modal run modal_app.py::bench --case sol
                                      # gemm_sol's cluster tile and N band
                                      # against the wave arithmetic (#138)
    modal run modal_app.py::bench --case sol-small
                                      # the same two rungs below 4096^3, taken
                                      # twice, where the clock is the limit
    modal run modal_app.py::bench --case swizzle
                                      # gemm's item traversal, tile held fixed
    modal run modal_app.py::bench --case tile
                                      # gemm's pair tile and pipeline depth,
                                      # traversal held fixed (#87)
    modal run modal_app.py::bench --case staged
                                      # gemm's epilogue SHAPE: a register drain
                                      # against one staged through shared
                                      # memory by stmatrix (#15)
    modal run modal_app.py::bench --case widths
                                      # gemm's epilogue INSTRUCTION WIDTHS:
                                      # .x8 LDTM and stmatrix .x4 on the staged
                                      # epilogue, ablated and composed (#117)
    modal run modal_app.py::bench --case residual
                                      # where the gap to cuBLASLt lives, with
                                      # every control re-run in one container
                                      # and the residual epilogue decomposed
    modal run modal_app.py::profile   # one launch under Nsight Compute (see
                                      # the note there: no counters on Modal)
    modal run modal_app.py::doctor    # env / GPU sanity check
    modal run modal_app.py::stall     # does nothing, out loud -- the control
                                      # for scripts/modal-run (#103)
"""

import functools
import re
import subprocess
import time
from collections.abc import Callable
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
    # The experiments crate: the lab notebook -- every rung, every ablation, the
    # benchmark sweeps and the cuBLASLt denominator. Standalone like the two
    # above, and it reaches into `examples/src` for two kernels and the clock
    # (`#[path]`), so both directory mounts have to be present for either crate
    # to build.
    .add_local_dir(
        str(Path(__file__).parent / "experiments/src"),
        f"{PROJECT_DIR}/experiments/src",
    )
    .add_local_file(
        str(Path(__file__).parent / "experiments/Cargo.toml"),
        f"{PROJECT_DIR}/experiments/Cargo.toml",
    )
    .add_local_file(
        str(Path(__file__).parent / "experiments/Cargo.lock"),
        f"{PROJECT_DIR}/experiments/Cargo.lock",
    )
    # The `cublas` feature's link step, and the C translation unit that asserts
    # the hand-written cuBLASLt ABI against the real headers. Both sit outside
    # `experiments/src`, so neither arrives with the directory mount above and a
    # missing one fails as a confusing build error rather than as an absence.
    .add_local_file(
        str(Path(__file__).parent / "experiments/build.rs"),
        f"{PROJECT_DIR}/experiments/build.rs",
    )
    .add_local_file(
        str(Path(__file__).parent / "experiments/cublaslt_abi.c"),
        f"{PROJECT_DIR}/experiments/cublaslt_abi.c",
    )
    .add_local_file(
        str(Path(__file__).parent / "rust-toolchain.toml"),
        f"{PROJECT_DIR}/rust-toolchain.toml",
    )
)

app = modal.App("ferro-kittens", image=image)

HARNESS_DIR = f"{PROJECT_DIR}/device-tests"
EXAMPLES_DIR = f"{PROJECT_DIR}/examples"
EXPERIMENTS_DIR = f"{PROJECT_DIR}/experiments"
# The driver stub satisfies cargo-oxide's link step where there is no GPU.
STUB_ENV = ["env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"]


# Every GPU entry point carried `timeout=5400` -- ninety minutes, uniform
# across six functions doing very different amounts of work, and derived from
# nothing. That ceiling is what a wedged container rides to the end, and a
# wedged container is indistinguishable from a slow one from the outside: one
# died after twenty-six minutes having printed fifteen lines of NVIDIA banner
# and never reached Python, billed at B200 rates the whole way.
#
# These are sized from what the runs take, and the elapsed line `_run` prints
# is the evidence for tightening them further. `build` is 155-221 s over ten
# CI runs warm, ~7.5 min cold since #93 gave it a whole CPU.
CHECKING = 900  # compile and lint, no launches: `build`, `regcount`
RUNNING = 1200  # compile and a short harness: `device_tests`, `examples`
# 3600 rather than 2700 since #122 added a seventh table to `bench --case
# residual`: three more rungs at each of 4096^3, 8192^3 and 16384^3, and the
# host cost of that sweep is re-staging and re-checking the operands rather
# than the launches. A ceiling is not a bill -- the container is billed for
# what it runs -- so the only thing raising it costs is a wedge riding longer,
# which is what `scripts/modal-run`'s silence budget is for.
SWEEPING = 3600  # 16384^3 launched dozens of times: `bench` and the profiles
ASKING = 300  # one driver query, nothing built: `doctor`


# Positive evidence that an entry point reached its last line. `scripts/modal-run`
# imports this name and refuses to return 0 without having seen it.
#
# Why not just trust the client's exit code: `modal run` exits **0** when its app
# is stopped out from under it (#103, measured -- the PR quotes the transcript).
# So "the client came back without an error" cannot distinguish a gate that
# passed from a gate that a concurrent agent, a Modal-side eviction, a spend
# limit or the dashboard's stop button killed halfway. Absence of failure is not
# evidence of completion. This line is, and it costs one print.
COMPLETED = "== modal_app: entry point completed =="


def completes(fn: Callable) -> Callable:
    """Print `COMPLETED` once `fn` has returned normally.

    Applied *under* `@app.function`, so it runs in the container and the
    sentinel arrives on the same stream as everything else the run said -- the
    stream the client is already reading. A raised exception skips it, which is
    the point: only the last line of a finished body prints this."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        fn(*args, **kwargs)
        print(COMPLETED, flush=True)

    return wrapped


def _run(cmd: list[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    start = time.monotonic()
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    finally:
        # Printed on the failure path too -- a step that ran for nineteen of a
        # twenty-minute budget and a step that died on contact are different
        # diagnoses, and the exception alone does not distinguish them.
        print(f"  ({time.monotonic() - start:.1f}s)", flush=True)


@app.function(cpu=8, timeout=CHECKING)
@completes
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
    # The examples crate, same treatment, and since #3 this is all four of its
    # kernels rather than the subset that was not behind a feature. That is the
    # point of them having no features left: a post-monomorphization
    # `const { assert!(..) }` in a tile shape is invisible to `cargo check` and
    # shows up only in a real device build, and a gated kernel never reached
    # this line at all. Its host launchers are ordinary Rust and get ordinary
    # lints.
    _run(["cargo", "clippy", "--all-targets"], cwd=EXAMPLES_DIR)
    _run([*STUB_ENV, "cargo", "oxide", "build", "kittens-examples", "--arch", "sm_100a"],
         cwd=EXAMPLES_DIR)

    # The experiments crate, which is where the other twenty-nine GEMM entry
    # points went. It is the expensive half of this function and the half most
    # worth having: a probe that is not on any correctness gate is a probe whose
    # only gate is this codegen.
    _run(["cargo", "clippy", "--all-targets"], cwd=EXPERIMENTS_DIR)
    _run([*STUB_ENV, "cargo", "oxide", "build", "kittens-experiments", "--arch", "sm_100a"],
         cwd=EXPERIMENTS_DIR)

    # The `cublas` feature (#92), which the line above deliberately did NOT
    # have on: the default build has to keep working for anyone without a
    # CUDA toolkit, so "it still builds with the feature off" is the claim
    # that line makes and this one must not weaken.
    #
    # Everything static about the baseline is checkable here, with no GPU. The
    # C file asserts the enum values and struct offsets `cublaslt.rs`
    # transcribes by hand against the real headers -- `-fsyntax-only`, so no
    # linker and no binary -- and the device build then links the real
    # libcublasLt, which is the step that proves FFI through cargo-oxide works
    # at all. Finding that out here costs a CPU container; finding it out in
    # `bench` costs a B200 one.
    _run(["gcc", "-fsyntax-only", "-I/usr/local/cuda/include", "cublaslt_abi.c"],
         cwd=EXPERIMENTS_DIR)
    _run(["cargo", "clippy", "--all-targets", "--features", "cublas"], cwd=EXPERIMENTS_DIR)
    _run([*STUB_ENV, "cargo", "oxide", "build", "kittens-experiments", "--arch", "sm_100a",
          "--features", "cublas"],
         cwd=EXPERIMENTS_DIR)


# `gaps` lived here until #3, and printed each aspirational kernel's remaining
# errors by turning its cargo feature on: the missing API surface *was* the
# error list, at the call sites that wanted it, read off a compiler rather than
# off the last person's memory. It is retired because every list reached empty
# and the features are gone -- `build` above now codegens all four kernels for
# real `sm_100a`, which is the stronger claim the gate was standing in for.
# `experiments/README.md` keeps what the lists said and how to read one, for
# whenever the next aspirational kernel is written.


# `cpu=8` for the same reason `build` has it, and it costs more to omit here
# than there. `cargo oxide run` compiles the crate before it launches anything,
# which `build`'s own comment prices at ten-odd minutes; on Modal's fractional
# default it is closer to forty. A GPU function is billed for the whole
# container, so those minutes are billed at **B200 rates to run a Rust
# compiler** -- the worst ratio in this file, and paid on the two gates every
# change runs. `bench`, `ladder_bench` and `profile` already say it; these two
# were simply missed.
@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=RUNNING)
@completes
def device_tests() -> None:
    """The harness itself. One binary, every case, non-zero exit on failure."""
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _run(["cargo", "oxide", "run", "device-tests"], cwd=HARNESS_DIR)


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=RUNNING)
@completes
def examples() -> None:
    """Both kernel crates' correctness gates. Prints the examples' status table
    and runs every kernel that has a launcher against its CPU reference, then
    runs every rung and probe in `experiments/` against the same one; non-zero
    on a wrong number. This is the claim `device-tests` cannot make: not that a
    primitive behaves, but that a whole kernel written against the library
    computes.

    Two binaries since the crates split, and both are named here rather than in
    one of them, because the gate is what it always was: every entry point that
    computes a GEMM, at both check sizes and all three traversal widths."""
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _run(["cargo", "oxide", "run", "kittens-examples"], cwd=EXAMPLES_DIR)
    _run(["cargo", "oxide", "run", "kittens-experiments", "--", "check"],
         cwd=EXPERIMENTS_DIR)


# 5400 rather than 1800 since #86 put 16384^3 in the sweep: the *device* work is
# eight milliseconds a launch, but staging it is 268 million host-side operand
# values per matrix and the exact check compares 268 million elements of `C`.
@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def bench(case: str = "", m: int = 0, n: int = 0, k: int = 0) -> None:
    """The same kernels, timed at several sizes, reporting achieved throughput.

    Separate from `examples` so the correctness path stays a few seconds: this
    one stages problems up to 8192^3 and launches each of them dozens of times.
    Every size is checked against the same CPU reference `examples` uses before
    it is timed at all, so a throughput figure here always belongs to a run that
    computed the right answer -- the harness has no path that prints one for a
    launch it did not verify.

    **The `gemm*` rows moved in #119.** They launch `gemm::SHIPPED_EPILOGUE`,
    which is `staged84` and not the register drain, so a row here is not
    comparable to the same row taken before that change. The `staged` and
    `widths` cases are where both epilogues appear side by side.

    **`--case repro` is the one that measures this harness rather than a
    kernel** (#122). It is the case to run before quoting a difference between
    two launches: it takes four whole measurements of each arm round-robin and
    prints, beside each difference, the amplification that difference applies to
    its arms' own repeatability. #121 found `whole - no drain` reproducing to
    39-46% at 16384^3, and that is where the number comes from.

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
    # `--case` narrows the sweep to one table, and `--m/--n/--k` to one row of
    # it. Re-staging 16384^3 to re-read a diagnostic row costs more host time
    # than the row does device time.
    narrowed = [case] if case else []
    if case and (m or n or k):
        narrowed += [str(m), str(n), str(k)]
    # `--features cublas` (#92) puts a cuBLASLt column and a ratio beside the
    # `gemm` table. It is off in the crate's default feature set so that tier 1
    # CI and anyone without a devel CUDA toolkit are unaffected, and on here
    # because this image always has one: a GEMM number with no denominator is
    # the thing the feature exists to stop shipping.
    _run(
        ["cargo", "oxide", "run", "kittens-experiments", "--features", "cublas",
         "--", "bench", *narrowed],
        cwd=EXPERIMENTS_DIR,
    )


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def clc_bench() -> None:
    """The GEMM's three item sources on one clock -- issue #88.

    Cluster Launch Control is Blackwell's hardware work-stealing: launch one
    cluster per output tile, and clusters that finish cancel ones the scheduler
    has not launched yet. The static stride it replaces loses only the ragged
    last wave, which is 23% at 4096^3 and 0.3% at 16384^3 -- so the table prints
    that prediction beside each measurement, and a zero at the largest size is
    the predicted result rather than a failure.

    Same 5400 s as `bench` and for the same reason: 16384^3 stages a gigabyte of
    operands and checks every one of 268 million output elements against the CPU
    reference, and this entry point does it once per scheduler.
    """
    _run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
          "--format=csv"], cwd="/")
    _run(["cargo", "oxide", "run", "kittens-experiments", "--", "clc"], cwd=EXPERIMENTS_DIR)


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def ws_bench() -> None:
    """The warp-specialized GEMM against the one it is a variant of.

    `experiments/src/gemm.rs` gets its overlap across CTAs -- two per SM, so one
    CTA's epilogue runs against another's MMA. `experiments/src/gemm_ws.rs` gets it
    inside one CTA, from six warps and a double-buffered TMEM accumulator, which
    is 512 accumulator columns and therefore one CTA per SM. Table 1 holds the
    epilogue identical in both -- the *register* drain on both sides, named
    rather than defaulted since #119 moved each kernel's default onto a staged
    rung -- so that delta is the occupancy/specialization structure and not a
    store path.

    Since #118 it also runs the epilogue ladder on the warp-specialized kernel
    -- #116's staged shape and #117's `.x8` LDTM and `stmatrix.x4` -- with the
    two epilogue-free controls those are subtracted from, and puts each design
    point's *best* rung against the other's. Twelve timed rungs a size, at three
    sizes, every one of them checked element-by-element first except the two
    that write no `C` on purpose and say so in their own label.

    Both kernels and cuBLASLt are measured in the *same container*, because a
    control quoted from another one is not a control: #98 found 2.9% of drift
    between runs of the same tree, and #118's own two sessions moved a 16384^3
    row by 3%.

    `--features cublas` for the same reason `bench` has it -- a GEMM number with
    no denominator is the thing that feature exists to stop shipping. The
    `SWEEPING` timeout covers a cold build plus the sweep; the sweep itself is
    about 90 s of B200, because 16384^3 spends far more host time staging a
    gigabyte of operands and checking 268 million output elements than it spends
    device time launching.
    """
    _run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
          "--format=csv"], cwd="/")
    _run(
        ["cargo", "oxide", "run", "kittens-experiments", "--features", "cublas",
         "--", "ws"],
        cwd=EXPERIMENTS_DIR,
    )


# `cpu=8` because on a cold container the *build* is a long pole in its own
# right — the harness' dependency tree is ten-odd minutes of compilation and
# the B200 is billed through all of it. The timeout covers that plus the sweep:
# twenty rungs, each verified at three configurations and then timed in five
# rounds at two grids, is a further half hour. 1800 s does not fit it, and a
# run that dies between the third shape and the fourth has spent the GPU and
# answered three quarters of the question.
@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def ladder_bench() -> None:
    """Four rungs of #60's register ladder, with a clock on them — issue #63.

    The ladder found the in-place spellings 81 to 130 registers *under* the
    fused one at three shapes, on a stack frame at least as large as the form
    they beat: `ptxas` had decided the band did not fit and left it in local
    memory, streaming it. `ptxas -v` reports what was allocated and never what
    it cost, so this is the first thing in the repo to time a register claim
    rather than count it.

    `[32, 96]`, `[48, 64]` and `[64, 64]` are where the effect is largest;
    `[32, 128]` is the control, where `fused` wins on both static counters and
    ought therefore to win on the clock. Each of the five spellings is checked
    against a CPU reference at every grid and step count it is timed at, timed
    in repeated rounds so the table can state its own noise floor, and reported
    beside the driver's register count, local frame and occupancy.

    This one genuinely needs the GPU. Everything static about it — that the
    timed rungs compile, and that they price identically to the ladder rungs
    they are twins of — is in `::regcount`, which is CPU-only; run that first.
    """
    _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
            "--format=csv",
        ],
        cwd="/",
    )
    _run(["cargo", "oxide", "run", "device-tests", "--", "bench-ladder"], cwd=HARNESS_DIR)


# Nsight Compute ships inside the CUDA 13 devel image already — the binary, the
# chip support and the permissions are all here, and it still cannot read a
# counter on this container. See `profile` below for which file is missing and
# what #86 had to do instead.
NCU = "/usr/local/cuda/bin/ncu"

# Sections rather than a metric list, because a metric name is chip-specific and
# a wrong one fails the whole run: a section asks the profiler for the metrics
# *this* chip has under a heading. SpeedOfLight is the one that answers the
# issue — compute and memory throughput each as a percentage of what the part
# can sustain — and MemoryWorkloadAnalysis is the L1/L2/DRAM traffic table with
# the hit rates in it.
NCU_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "ComputeWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
)

# One `bench` size stages the problem, checks it, warms up five launches and
# times thirty. So the seventh launch is the first timed one, and profiling it
# alone leaves the other thirty-five running at full speed.
NCU_SKIP, NCU_COUNT = "6", "1"


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def profile(kernel: str = "gemm", m: int = 8192, n: int = 8192, k: int = 8192) -> None:
    """One launch of one kernel at one size, under Nsight Compute — issue #86.

    **This does not currently work on Modal, and it is checked in anyway
    because the reason is one missing file and worth writing down.** `ncu`
    2025.3.0 is already in the CUDA 13 image at `/usr/local/cuda/bin/ncu`, it
    lists `gb100` among the chips it knows, `libnvperf_target.so` is in the
    install, `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 0`, and
    the profiler still says:

        Failed to initialize the profiler: LibraryNotLoaded.
        Check that a compatible driver library is loaded.

    The injected driver set under `/usr/lib/x86_64-linux-gnu` carries
    `libnvidia-ml`, `-nvvm`, `-ptxjitcompiler`, `-allocator`, `-gpucomp` and the
    graphics libraries, and **no `libnvidia-pcc.so`** — the performance-counter
    library, which the container runtime does not mount for the compute
    capability set. It is not in the 580.95.05 `.run` package either, so it
    cannot simply be fetched. So there is no counter on this harness, which is
    why #86's L2 hit rate and DRAM byte count are answered by *interventions* —
    `GEMM_FOOTPRINT_SIZES` and `GEMM_DEPTH_SIZES` in `experiments/src/bench.rs`,
    each holding everything constant but one thing — rather than by counters.
    That is a weaker instrument for attribution and a stronger one for cause.

    Whenever the library does appear, this is one command. `--clock-control
    none` is deliberate: the default locks the clocks to base so that two runs
    compare, which also means the duration it reports is not the duration
    `bench` reports, and half the point is to divide a measured byte count by a
    duration that belongs to the same table.
    """
    _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
         "--format=csv"],
        cwd="/",
    )
    subprocess.run(
        ["bash", "-lc", "ls /usr/lib/x86_64-linux-gnu/libnvidia-pcc* 2>/dev/null "
         "|| echo 'libnvidia-pcc.so absent: the profiler will report LibraryNotLoaded'"],
        check=False,
    )
    # Build first: `--target-processes all` would otherwise follow the compiler
    # around for ten minutes looking for a context it never creates.
    _run(["cargo", "oxide", "build", "kittens-experiments", "--arch", "sm_100a"], cwd=EXPERIMENTS_DIR)
    _run(
        [
            NCU, "--target-processes", "all", "--clock-control", "none",
            "--kernel-name", f"regex:{kernel}", "--launch-skip", NCU_SKIP,
            "--launch-count", NCU_COUNT, "--print-details", "all",
            *[argument for section in NCU_SECTIONS for argument in ("--section", section)],
            "cargo", "oxide", "run", "kittens-experiments", "--",
            "bench", kernel, str(m), str(n), str(k),
        ],
        cwd=EXPERIMENTS_DIR,
    )


@app.function(gpu=DEFAULT_GPU, timeout=ASKING)
@completes
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


@app.function(timeout=ASKING)
@completes
def stall(seconds: int = 120) -> None:
    """Do nothing, out loud, for `seconds`. This is `scripts/modal-run`'s
    control, and it is here rather than in a scratch file because a check nobody
    has watched fail is not a check (#67, #76, #95, #99).

    From the wrapper's side it is indistinguishable from a real run: one `$ `
    line proving Python is alive, then a long quiet stretch of work. So

        scripts/modal-run stall &        # in one shell
        modal app stop -y <app id>       # in another, from `modal app list`

    reproduces #103 exactly -- a run killed out from under the client -- for a
    fractional CPU and a minute. No GPU, no build, and nothing else in the repo
    refers to it."""
    _run(["sleep", str(seconds)], cwd="/")


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


# Crates that emit PTX, as `(package, directory)`. The harness carries the
# probes and the ladder; the two kernel crates carry the kernels a *launch*
# configures, and their register and shared-memory counts are what an occupancy
# argument about a real kernel has to be made of (#47).
#
# `examples` and `experiments` both emit `gemm_cg2_staged_x8x4`, because it is
# the same kernel: the teaching crate ships it and the notebook keeps it as the
# arm every A/B is measured against. `_measure` keys on `(ptx file, kernel)` so
# both rows are printed, and the pair is a free check that the extraction moved
# no instruction -- two identical rows are the claim, and a difference between
# them is the finding.
PTX_CRATES = (
    ("device-tests", HARNESS_DIR),
    ("kittens-examples", EXAMPLES_DIR),
    ("kittens-experiments", EXPERIMENTS_DIR),
)


def _measure(arch: str) -> dict[tuple[str, str], dict[str, int]]:
    """Build every PTX-emitting crate and price every emitted entry function,
    keyed by `(ptx file, kernel)`. Building is idempotent, so a caller that
    wants a *fresh* measurement has to invalidate the artifacts first."""
    measured: dict[tuple[str, str], dict[str, int]] = {}
    for package, directory in PTX_CRATES:
        _run([*STUB_ENV, "cargo", "oxide", "build", package, "--arch", arch], cwd=directory)

        ptx_files = sorted(Path(directory).rglob("*.ptx"))
        if not ptx_files:
            listing = sorted(p for p in Path(directory, "target").rglob("*") if p.is_file())
            raise RuntimeError(
                f"no PTX under {package}'s target dir; cargo-oxide's artifact layout "
                f"must have moved. Files found:\n" + "\n".join(map(str, listing[:200]))
            )

        for ptx in ptx_files:
            compiled = subprocess.run(
                ["ptxas", "-v", f"-arch={arch}", "-o", "/dev/null", str(ptx)],
                capture_output=True,
                text=True,
            )
            if compiled.returncode != 0:
                raise RuntimeError(f"ptxas failed on {ptx}:\n{compiled.stderr}")
            for name, counts in _parse_ptxas(compiled.stderr).items():
                measured[(str(ptx.relative_to(PROJECT_DIR)), name)] = counts
    return measured


# The opcode census (#114, #15). `ptxas -v` prices a kernel and never says what
# is in it, and every ablation and epilogue rung in `experiments/src/gemm.rs` is a
# claim about exactly that: that a rung removes what it names, and that a
# re-shaped epilogue issues the instructions its own doc counts. #114 ran this
# by hand and reported it in prose; it is cheap, it is CPU-only, and a claim
# nobody re-derives is a claim that goes stale, so it lives here now.
#
# Substrings of the PTX mnemonic, and deliberately disjoint — `st.global.b32`
# is not a substring of `st.global.v4.b32`, so a plain count per pattern is a
# partition and not a double count.
CENSUS_OPCODES = (
    ("mma", "tcgen05.mma"),
    ("ldtm", "tcgen05.ld"),
    ("tma", "cp.async.bulk.tensor"),
    ("stmatrix", "stmatrix"),
    ("ld.sh.v4", "ld.shared.v4"),
    ("st.g.v4", "st.global.v4"),
    ("st.g.b32", "st.global.b32"),
    ("cvt.bf16x2", "cvt.rn.bf16x2"),
    ("bar.sync", "bar.sync"),
    ("bar.warp", "bar.warp.sync"),
    ("mbar.arrive", "mbarrier.arrive"),
    # Predication a kernel pays for, and the column #125 added it for.
    #
    # `kittens::plan::SharedPlan::reserve` branches on whether the cursor has
    # memory under it, because the device and const eval want different pointer
    # arithmetic and no operation serves both (see its doc). The flag is a
    # literal in both constructors and the claim is that it folds. Two
    # side-effect-free pointer values under an `if` is the shape LLVM
    # if-converts to a `select`, so a surviving branch is a `selp` and the
    # count is the check: `gemm_cg2_idle` calls the whole eight-reservation
    # walk and then does nothing, so if the flag did not fold it would carry
    # eight of them. It carries **zero**, which is what settled #125's last
    # open question. Keep this column: the argument is cheap to re-derive and
    # expensive to re-litigate.
    ("selp", "selp"),
)

# Which entry functions the census prints. The whole table would be forty-odd
# rows of probes with nothing in them; these are the kernels whose instruction
# mix is an argument somebody made.
CENSUS_PREFIX = "gemm_"


def _census(directory: str) -> dict[str, dict[str, int]]:
    """Opcodes per entry function, over the PTX a crate emitted.

    Split on `.visible .entry` rather than parsed: the bodies contain inline
    PTX from `ptx_asm!` verbatim, which is precisely the text worth counting
    and precisely what a structural parser would have to be taught about."""
    counted: dict[str, dict[str, int]] = {}
    for ptx in sorted(Path(directory).rglob("*.ptx")):
        chunks = ptx.read_text().split(".visible .entry ")
        for chunk in chunks[1:]:
            name = chunk.split("(", 1)[0].strip()
            if not name.startswith(CENSUS_PREFIX):
                continue
            counted[name] = {
                column: chunk.count(pattern) for column, pattern in CENSUS_OPCODES
            }
    return counted


def _print_census() -> None:
    # Per crate, and both of them. The rungs a census is *for* are in
    # `experiments/`; the kernel they are read against is the one `examples/`
    # ships, and the two crates emit it under one name because it is one
    # kernel. Two sections rather than one merged table is what lets that be
    # checked instead of assumed -- the two rows must agree opcode for opcode,
    # and a difference between them is the extraction having moved something.
    sections = [
        ("kittens-examples", _census(EXAMPLES_DIR)),
        ("kittens-experiments", _census(EXPERIMENTS_DIR)),
    ]
    if not any(counted for _, counted in sections):
        print(f"\nno `{CENSUS_PREFIX}*` entry functions in the crates' PTX to census.")
        return
    print(
        f"\nopcode census, per entry function — every `{CENSUS_PREFIX}*` kernel the two\n"
        "kernel crates emit, counted in the PTX. A rung that removes a phase must show\n"
        "zero in that phase's column, and an epilogue that re-shapes the store must show\n"
        "the counts its own doc predicts. Counts are static instructions inside one entry\n"
        "function, so a loop the compiler did not unroll shows as one."
    )
    columns = [column for column, _ in CENSUS_OPCODES]
    for package, counted in sections:
        if not counted:
            continue
        print(f"\n  {package}")
        print("  " + f"{'kernel':<30}" + "".join(f"{column:>12}" for column in columns))
        for name in sorted(counted):
            row = "".join(f"{counted[name][column]:>12}" for column in columns)
            print(f"  {name:<30}{row}")
    shared = set(sections[0][1]) & set(sections[1][1])
    disagreed = [name for name in sorted(shared) if sections[0][1][name] != sections[1][1][name]]
    if shared:
        print(
            f"\n  {len(shared)} kernel(s) in both crates: "
            + (
                "identical opcode counts."
                if not disagreed
                else "DIFFER — " + ", ".join(disagreed)
            )
        )


def _print_kernels(measured: dict[tuple[str, str], dict[str, int]]) -> None:
    for ptx in sorted({source for source, _ in measured}):
        kernels = {name: counts for (source, name), counts in measured.items() if source == ptx}
        print(f"\n{ptx}  ({len(kernels)} kernels)")
        print(f"  {'kernel':<44}{'regs':>6}{'spill st':>10}{'spill ld':>10}{'stack':>8}{'smem':>8}")
        for name, counts in sorted(kernels.items()):
            print(
                f"  {name:<44}{counts['registers']:>6}{counts['spill_stores']:>10}"
                f"{counts['spill_loads']:>10}{counts['stack']:>8}{counts['smem']:>8}"
            )


# The register ladder `device-tests` compiles, written out a second time here.
# It has to be: `ptxas` prices the PTX that exists, so a rung that was dropped
# from `ladder!(..)` — or removed because it would not build — is invisible to
# a table derived from the PTX alone, and a sweep that quietly omits what it
# could not compile is worse than no sweep. These two lists are the intent; a
# rung in them with no PTX prints as `not built`.
#
# Mirrors `ladder!(..)` and the `LADDER_*` constants in device-tests/src/main.rs.
LADDER_SHAPES = (
    (32, 16), (32, 32), (32, 48), (32, 64), (32, 96), (32, 128), (32, 192), (32, 256),
    (16, 64), (48, 64), (64, 64), (16, 128), (48, 128), (64, 128),
)  # fmt: skip
LADDER_SPELLINGS = ("fused", "assign", "open_coded", "rebound", "all_in_place")

# The four rungs `::ladder_bench` puts a clock on (#63) — three where in-place
# appears to win by 81-130 registers on a frame no smaller, and `[32, 128]` as
# the control where `fused` wins on both counters. Mirrors `timed_ladder!(..)`.
BENCH_SHAPES = ((32, 96), (48, 64), (64, 64), (32, 128))


def _ladder_pressure(shape: tuple[int, int]) -> int:
    """fp32 values one thread of an `[M, N]` warp tile holds — `SLOTS * VALUES`.

    The ladder's first-order variable, and deliberately not its only one: two
    shapes with the same value here can compile very differently, which is what
    the row sweep is in the ladder to show."""
    rows, columns = shape
    return rows * columns // 32


def _print_ladder(measured: dict[tuple[str, str], dict[str, int]]) -> None:
    """The sweep as one table, then where the cliff is.

    "At which width does spill first go non-zero, per spelling" is the number
    every register issue actually wants, and reading it off fifty-odd rows by
    eye is how it stays unanswered."""
    by_kernel = {name: counts for (_, name), counts in measured.items()}
    rungs = {
        (shape, spelling): by_kernel.get(f"ladder_probe_{shape[0]}x{shape[1]}_{spelling}")
        for shape in LADDER_SHAPES
        for spelling in LADDER_SPELLINGS
    }
    if all(counts is None for counts in rungs.values()):
        return

    def sweep(heading: str, cell: Callable[[dict[str, int]], str], width: int) -> None:
        print(f"\n{heading}")
        print(f"  {'shape':<12}{'per thread':>11}" + "".join(f"{s:>{width}}" for s in LADDER_SPELLINGS))
        for shape in LADDER_SHAPES:
            counts = (rungs[(shape, spelling)] for spelling in LADDER_SPELLINGS)
            cells = ("not built" if count is None else cell(count) for count in counts)
            shape_text = f"[{shape[0]:>2}, {shape[1]:>3}]"
            row = "".join(f"{text:>{width}}" for text in cells)
            print(f"  {shape_text:<12}{_ladder_pressure(shape):>11}{row}")

    sweep(
        "register ladder — regs / spill store bytes, per thread of an [M, N] warp tile",
        lambda counts: f"{counts['registers']}/{counts['spill_stores']}",
        14,
    )
    # A rung can be cheap in registers because the band fits, or because ptxas
    # never promoted it and is addressing it in local memory instead. Those are
    # opposite outcomes and the register column cannot tell them apart, so the
    # frame goes beside it rather than in a footnote.
    sweep("  the same rungs, stack frame bytes", lambda counts: str(counts["stack"]), 14)

    missing = [(shape, spelling) for (shape, spelling), counts in rungs.items() if counts is None]
    if missing:
        print(f"\n  not built — {len(missing)} of {len(rungs)} rungs emitted no PTX, so nothing")
        print("  above measures them. Dropped from `ladder!(..)`, or the shape does not compile:")
        for shape, spelling in missing:
            print(f"    ladder_probe_{shape[0]}x{shape[1]}_{spelling}")

    # Ordered by the pressure a rung actually puts on the file, so "first" means
    # the cheapest shape that hits it rather than the earliest line above.
    ordered = sorted(LADDER_SHAPES, key=lambda shape: (_ladder_pressure(shape), shape))
    print("\n  the cliff, per spelling — the cheapest rung that spills, and that")
    print("  reaches the 255-register ceiling (ladder ordered by values/thread):")
    print(f"    {'spelling':<14}{'first spill':<28}first 255 regs")
    for spelling in LADDER_SPELLINGS:
        edges = []
        for hit in (
            lambda counts: counts["spill_stores"] > 0,
            lambda counts: counts["registers"] >= 255,
        ):
            found = next(
                (
                    shape
                    for shape in ordered
                    if (counts := rungs[(shape, spelling)]) is not None and hit(counts)
                ),
                None,
            )
            edges.append(
                "never on this ladder"
                if found is None
                else f"[{found[0]}, {found[1]}] at {_ladder_pressure(found)}/thread"
            )
        print(f"    {spelling:<14}{edges[0]:<28}{edges[1]}")

    # The column sweep on its own: one row extent, so "where does it break" has
    # a width for an answer, which is the form every earlier issue asked in.
    widths = [shape for shape in LADDER_SHAPES if shape[0] == 32]
    if widths:
        print("\n  the same, as a width — the 32-row column sweep alone:")
        for spelling in LADDER_SPELLINGS:
            spilled = next(
                (
                    shape[1]
                    for shape in sorted(widths, key=lambda shape: shape[1])
                    if (counts := rungs[(shape, spelling)]) is not None
                    and counts["spill_stores"] > 0
                ),
                None,
            )
            answer = "no width on this ladder" if spilled is None else f"{spilled} columns"
            print(f"    {spelling:<14}first spills at {answer}")


def _print_timed_twins(measured: dict[tuple[str, str], dict[str, int]]) -> None:
    """`ladder_probe_*` against `ladder_timed_*`, rung by rung — issue #63.

    `::ladder_bench` times the `ladder_timed_*` kernels and reports what they
    cost; the ladder table above prices the `ladder_probe_*` ones. Those are
    only the same claim if the two compile the same, and they are the same
    source: one `#[inline(always)]` body at the same `(M, N, FORM)`, differing
    in a `const STRIDED: bool` that moves the *final* dump to a per-block
    band so a grid may run it at all. That should cost nothing in the loop
    under test, which is a prediction and therefore something to check rather
    than assert — and to raise on, like the determinism control, because a
    timing printed beside a register count that belongs to a different compiled
    kernel is worse than no timing."""
    by_kernel = {name: counts for (_, name), counts in measured.items()}
    counted = ("registers", "spill_stores", "spill_loads", "stack")

    def cell(counts: dict[str, int] | None) -> str:
        if counts is None:
            return "not built"
        return f"{counts['registers']}/{counts['spill_stores']}/{counts['stack']}"

    rows, disagreements = [], 0
    for shape in BENCH_SHAPES:
        for spelling in LADDER_SPELLINGS:
            suffix = f"{shape[0]}x{shape[1]}_{spelling}"
            priced = by_kernel.get(f"ladder_probe_{suffix}")
            timed = by_kernel.get(f"ladder_timed_{suffix}")
            same = (
                priced is not None
                and timed is not None
                and all(priced[field] == timed[field] for field in counted)
            )
            disagreements += not same
            rows.append((suffix, cell(priced), cell(timed), "same" if same else "DIFFERS"))

    print("\n  the timed twins (#63) — regs/spill/frame of the rung `regcount`")
    print("  prices, beside the rung `::ladder_bench` runs:")
    print(f"    {'rung':<26}{'ladder_probe':>18}{'ladder_timed':>18}{'':>10}")
    for name, priced, timed, verdict in rows:
        print(f"    {name:<26}{priced:>18}{timed:>18}{verdict:>10}")
    if disagreements:
        raise RuntimeError(
            f"{disagreements} of {len(rows)} timed rungs do not price like the ladder rung "
            "they stand in for, so `::ladder_bench` would be timing a kernel the ladder "
            "table above does not describe. Fix the divergence, or say in the bench which "
            "kernel its numbers belong to."
        )
    print(f"\n  identical at all {len(rows)} rungs, every counter.")


# --- the occupancy step (#95) ------------------------------------------------
#
# #94 took `gemm_cg2` from 40 registers to 167 and won 13.7% at 8192^3, and 167
# is one register under the count at which that kernel would hold **two** CTAs
# an SM instead of three. Nothing in the repo was watching, for a good reason:
# until #94 registers were nowhere near binding, and #84's residency census
# found the 73792 B shared plan to be the only cap. They are the joint cap now,
# and #15, #87 and #88 all add live state to that same kernel.
#
# **Read what this gate claims narrowly.** It does not say fewer registers are
# faster. #47, #63, #67, #76 and #94 are five occasions in this repo where the
# register column ordered time backwards, and #94 is one where four times the
# registers bought 13.7% — `ladder_bench` found the 32-register spelling that
# streams its whole band through local memory beating the 252-register one at
# every shape it timed. Crossing the step is a different kind of event: it does
# not make a thread slower, it takes a third of the CTAs off the SM. And for
# `gemm_cg2` that is not even a throughput argument in the abstract — its grid
# is literally `SMS * CTAS_PER_SM / RANKS` clusters, so a residency the kernel
# no longer has is a persistent grid that oversubscribes.
#
# So: gate on the step, and on nothing else. A wobble of a couple of registers
# inside a step is codegen noise and passes.
#
# **And know what it is likely to catch.** When `gemm_cg2` wanted three CTAs an
# SM its ceiling worked out at 168 and `ptxas` put it at 167, which was not a
# coincidence to bank on: several kernels in this tree land at 167-168
# (`flash_forward`, `gemm_256x128_*`, both `global_copy_probe_128_*`), because
# 168 is exactly the largest count that keeps twelve warps on an SM and that is
# `ptxas`' own default target.
#
# #87 then took the third CTA: `[256, 256]` is 256 accumulator columns, so
# `512 / 256` fixes `CTAS_PER_SM` at 2 before shared memory is consulted, and
# two CTAs at 128 threads admits the whole 255-register file. The gated kernels
# read 166, 42 and 80 against a ceiling of 255 today, so nothing here is near
# its step — which is the point of printing `headroom` rather than a verdict.
# The way this gate is most likely to fire is not a kernel drifting up into it:
# it is `THREADS` or `CTAS_PER_SM` moving, a `#[launch_bounds]` or
# `-maxrregcount` being introduced, or a toolkit upgrade changing that default.
# Those are all real and all silent today.

# sm_100a's register file, and the only hardware figure below that is not
# arithmetic: 64 K 32-bit registers per SM, from the CUDA C Programming Guide's
# compute-capability table for 10.0. `ptxas` is the only NVIDIA tool in this
# container that could have been asked instead, and it does not know.
REGISTERS_PER_SM = 65536
WARP_SIZE = 32
# `ptxas` will not allocate past this, so a ceiling that lands on 255 means
# "registers can never cost this kernel that CTA" rather than a real edge.
# `cuda_occupancy.h` admits 256 for compute major >= 7; ptxas' limit binds
# first, and using it keeps the search over counts that can actually be emitted.
MAX_REGISTERS_PER_THREAD = 255
# The two numbers that make the step coarser than `file / threads / CTAs`, both
# read off the toolkit's `cuda_occupancy.h` for compute major 10 and re-checked
# against it on every run by `_check_occupancy_model`: registers are handed to a
# **warp** in units of 256, and the file is split four ways across
# sub-partitions that a warp is assigned to whole.
REGISTER_ALLOCATION_UNIT = 256
SUB_PARTITIONS_PER_SM = 4


def _ctas_by_registers(registers: int, threads: int) -> int:
    """CTAs of `threads` threads an SM holds when a thread wants `registers` of
    them — counting the register term and nothing else.

    Transcribed from `cudaOccMaxBlocksPerSMRegsLimit` in `cuda_occupancy.h`.
    The rounding *is* the content, so it is written the way NVIDIA writes it
    rather than as `REGISTERS_PER_SM // registers // threads`: at 128 threads
    that shortcut says 170 registers still fit three CTAs and the hardware model
    says 168, because 169 registers a thread is 5408 a warp, which rounds to
    5632, of which a 16384-register sub-partition holds two warps and not three.

    The warp and CTA-count ceilings an SM also has are separate terms and are
    deliberately not folded in: this answers "what do the registers allow",
    which is the whole of what a register gate is entitled to say."""
    warps_per_cta = -(-threads // WARP_SIZE)
    per_warp = -(-registers * WARP_SIZE // REGISTER_ALLOCATION_UNIT) * REGISTER_ALLOCATION_UNIT
    warps_per_sub_partition = REGISTERS_PER_SM // SUB_PARTITIONS_PER_SM // per_warp
    return warps_per_sub_partition * SUB_PARTITIONS_PER_SM // warps_per_cta


def _register_ceiling(ctas: int, threads: int) -> int:
    """The most registers a thread may use with `ctas` CTAs of `threads` still
    resident — the step, as a number.

    Searched rather than solved. A closed form is where the granularity gets
    dropped on the floor, and there are 255 candidates."""
    fits = [
        registers
        for registers in range(1, MAX_REGISTERS_PER_THREAD + 1)
        if _ctas_by_registers(registers, threads) >= ctas
    ]
    return max(fits, default=0)


# Kernels this gate watches, and the file that states their launch. Neither
# number it needs is written down here, because both are already written down
# there and a copy would be free to drift: `#[launch_contract(block = ...)]` is
# the exact block shape the host launch is validated against, and `CTAS_PER_SM`
# is the residency the kernel's own grid is sized from — for `gemm_cg2`
# measured twice, by #83 on a clock and by #84 by counting `%smid`, with the
# argument in its doc comment. A kernel joins this table by declaring both.
GATED_KERNELS = (
    ("gemm_cg2", "experiments", "experiments/src/gemm.rs"),
    # #15's staged epilogue, on the same launch geometry and the same grid
    # arithmetic. It reshapes the drained band from `[32, 128]` to `[32, 64]`,
    # which moves peak liveness, so it is exactly the kind of change this gate
    # exists for.
    ("gemm_cg2_staged", "experiments", "experiments/src/gemm.rs"),
    # The kernel `examples/` ships, and the reason it is here: the gate is a
    # gate on the kernel a launch gets by default. `.x8` returns 32 f32 in one
    # instruction and costs +52 registers over `gemm_cg2_staged`'s 42 — the
    # largest single liveness step any epilogue rung in this tree has taken —
    # so this is the row that would go red first.
    #
    # Both crates emit it, so both are gated: the entries carry the crate as
    # well as the source, and `_check_occupancy_step` looks the counts up by
    # `(crate, kernel)`. Keying on the name alone would hand this gate whichever
    # crate `PTX_CRATES` happened to visit last.
    ("gemm_cg2_staged_x8x4", "examples", "examples/src/gemm.rs"),
    # And `experiments/`' copy of it, which is the same source-level kernel in a
    # crate with forty-one others. It is here because the two do *not* have to
    # come out the same and did not: the extraction reads 80 registers against
    # 96. Crate composition alone moves a register count in this tree, which is
    # the whole reason every copy of a gated kernel is a row rather than one of
    # them standing in for the rest.
    ("gemm_cg2_staged_x8x4", "experiments", "experiments/src/gemm.rs"),
    # `groupnorm_tile`, both copies, and it is here because splitting the crates
    # took the `examples/` one 168 -> 236 registers and 3 CTAs an SM -> 2 with
    # nothing watching. One source file, two crates: `experiments/` compiles it
    # through `#[path]`, which is why the crate and the source are two columns.
    # 168 is *exactly* `_register_ceiling(3, 128)`, so it had been sitting on its
    # step without declaring one; `#[launch_bounds(128, 3)]` on the kernel is
    # what declares it now, and this row is what checks it. Shared memory admits
    # six CTAs at its 33 344 B plan, so registers are the binding term and no
    # other resource was going to catch this.
    ("groupnorm_tile", "examples", "examples/src/layernorm.rs"),
    ("groupnorm_tile", "experiments", "examples/src/layernorm.rs"),
)

_CONTRACT_BLOCK = re.compile(r"block\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_CTAS_PER_SM = re.compile(r"\bconst\s+CTAS_PER_SM\s*:\s*u32\s*=\s*(\d+)\s*;")
_LAUNCH_BOUNDS = re.compile(r"#\[launch_bounds\(\s*(\d+)\s*,\s*(\d+)\s*\)\]")


def _launch_geometry(source: str, kernel: str) -> tuple[int, int]:
    """`(threads, CTAs per SM)` for `kernel`, read out of its own source.

    Two spellings, and neither is a number written down here.

    **`#[launch_bounds(threads, ctas)]`**, if the kernel carries one, is read
    first and read alone. It is `__launch_bounds__`: `ptxas` gets the same pair
    the gate does, as `.maxntid` and `.minnctapersm`, and caps registers to
    reach it. So a kernel that declares this way cannot be told one residency by
    the compiler and checked against another, which is the whole failure a
    second constant would reintroduce.

    Otherwise: an exact `#[launch_contract(block = ...)]` above the kernel and
    the file's single `const CTAS_PER_SM`. That is the GEMM's shape, and it is
    the right one there — `CTAS_PER_SM` is what its *grid* is sized from, a host
    fact `launch_bounds` has no way to state, and the two GEMM entries would be
    over-constrained by a `.minnctapersm` they measured their way to rather than
    asked for."""
    text = Path(PROJECT_DIR, source).read_text()
    declared = text.find(f"fn {kernel}(")
    if declared >= 0:
        bounded = _LAUNCH_BOUNDS.findall(text[:declared])
        if bounded:
            threads, ctas = bounded[-1]
            return int(threads), int(ctas)
    blocks = _CONTRACT_BLOCK.findall(text[:declared]) if declared >= 0 else []
    residencies = _CTAS_PER_SM.findall(text)
    if not blocks or len(residencies) != 1:
        raise RuntimeError(
            f"{source} no longer states {kernel}'s launch in either form this gate "
            "reads: a `#[launch_bounds(threads, ctas)]` on the kernel, or an exact "
            "`#[launch_contract(block = (x, y, z))]` above it "
            f"(found {len(blocks)}) plus one `const CTAS_PER_SM: u32 = N;` in the file "
            f"(found {len(residencies)}). Restore one, or drop the kernel from "
            "GATED_KERNELS — an occupancy gate that cannot read the launch is not one."
        )
    x, y, z = (int(extent) for extent in blocks[-1])
    return x * y * z, int(residencies[0])


# `cuda_occupancy.h` is header-only, needs no driver and no GPU, and is the
# authority `_ctas_by_registers` is transcribed from. So compile it and ask it,
# rather than trusting the transcription. Same move as `--determinism`: a
# derived step is only better than a magic number if the derivation is checked,
# and #95's own table gives 170 where this gives 168. One of the two had to be
# wrong, and a hand-copied rounding rule is exactly the kind of thing that is.
OCCUPANCY_MODEL_CPP = r"""
#include <cstdio>
#include <cstdlib>
#include <cuda_occupancy.h>

// argv: <compute major> <compute minor> <registers per SM> <warp size> <threads>
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    cudaOccDeviceProp device;
    device.computeMajor          = atoi(argv[1]);
    device.computeMinor          = atoi(argv[2]);
    device.regsPerBlock          = atoi(argv[3]);
    device.regsPerMultiprocessor = atoi(argv[3]);
    device.warpSize              = atoi(argv[4]);
    int threads = atoi(argv[5]);
    for (int registers = 1; registers <= 255; ++registers) {
        cudaOccFuncAttributes function;
        function.numRegs = registers;
        cudaOccResult result;
        cudaOccPartitionedGCConfig caching = PARTITIONED_GC_OFF;
        int limit = -1;
        if (cudaOccMaxBlocksPerSMRegsLimit(
                &limit, &caching, &result, &device, &function, threads)
            != CUDA_OCC_SUCCESS) {
            return 1;
        }
        printf("%d %d\n", registers, limit);
    }
    return 0;
}
"""


def _check_occupancy_model(threads: int) -> int:
    """`_ctas_by_registers` against NVIDIA's own occupancy model, at every
    register count `ptxas` can emit. Returns how many agreed; raises if any
    did not."""
    source = Path("/tmp/occupancy_model.cpp")
    source.write_text(OCCUPANCY_MODEL_CPP)
    binary = source.with_suffix("")
    subprocess.run(
        ["g++", "-I/usr/local/cuda/include", "-o", str(binary), str(source)], check=True
    )
    reference = subprocess.run(
        [str(binary), "10", "0", str(REGISTERS_PER_SM), str(WARP_SIZE), str(threads)],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split() for line in reference.stdout.splitlines() if line]
    differences = [
        f"  {registers} registers: cuda_occupancy.h says {ctas}, "
        f"_ctas_by_registers says {_ctas_by_registers(int(registers), threads)}"
        for registers, ctas in rows
        if int(ctas) != _ctas_by_registers(int(registers), threads)
    ]
    if differences:
        raise RuntimeError(
            f"the occupancy model in this file disagrees with the toolkit's own at "
            f"{len(differences)} of {len(rows)} register counts, at {threads} threads. "
            "Every step below is derived from it, so none of them mean anything until "
            "this agrees.\n" + "\n".join(differences)
        )
    return len(rows)


def _check_occupancy_step(measured: dict[tuple[str, str], dict[str, int]]) -> None:
    """Registers against the occupancy step, per gated kernel — issue #95."""
    # Keyed by (crate, kernel) and not by kernel alone. Two kernels are emitted
    # by both crates, and a flat map would hand this gate whichever one
    # `PTX_CRATES` happened to visit last -- a gate reading a kernel other than
    # the one it names is exactly the silent failure #95 exists to prevent. The
    # crate is the first path component of a `_measure` key and a column of
    # `GATED_KERNELS`, which is what joins them.
    by_kernel = {
        (ptx.split("/", 1)[0], name): counts for (ptx, name), counts in measured.items()
    }
    print("\n  the occupancy model, against the toolkit's own (`cuda_occupancy.h`,")
    print("  compiled here — it needs no driver, and neither does this whole run):")
    geometries = {kernel: _launch_geometry(source, kernel) for kernel, _, source in GATED_KERNELS}
    for threads in sorted({threads for threads, _ in geometries.values()}):
        agreed = _check_occupancy_model(threads)
        print(f"    {threads:>4} threads: identical at all {agreed} register counts ptxas can emit")

    rows, crossed = [], []
    for kernel, package, source in GATED_KERNELS:
        counts = by_kernel.get((package, kernel))
        if counts is None:
            raise RuntimeError(
                f"{kernel} is gated on its occupancy step but {package} emitted no PTX "
                "for it, so this run measured nothing about it. Either it stopped being "
                "monomorphized, or it was renamed, or it moved crate; a gate that silently "
                "watches an absent kernel is worse than none."
            )
        threads, ctas = _launch_geometry(source, kernel)
        ceiling = _register_ceiling(ctas, threads)
        registers = counts["registers"]
        allowed = _ctas_by_registers(registers, threads)
        rows.append(
            (
                f"{package}/{kernel}",
                threads,
                ctas,
                registers,
                ceiling,
                ceiling - registers,
                allowed,
            )
        )
        if allowed < ctas:
            crossed.append(
                f"  {package}/{kernel}: {registers} registers at {threads} threads "
                f"leaves {allowed} CTAs/SM where the kernel is built for {ctas}. "
                f"{ceiling} is the most a thread may use and keep {ctas} — it is "
                f"{registers - ceiling} over."
            )

    print("\n  the occupancy step (#95) — registers against the residency each kernel's")
    print("  own grid is sized for. `ceiling` is derived, not written down anywhere:")
    print(f"    {'kernel':<36}{'threads':>8}{'wants':>7}{'regs':>6}{'ceiling':>9}{'headroom':>10}{'allows':>8}")
    for kernel, threads, ctas, registers, ceiling, headroom, allowed in rows:
        print(
            f"    {kernel:<36}{threads:>8}{ctas:>7}{registers:>6}"
            f"{ceiling:>9}{headroom:>10}{allowed:>8}"
        )
    if crossed:
        raise RuntimeError(
            "a kernel crossed its occupancy step: it now costs a CTA per SM.\n"
            + "\n".join(crossed)
            + "\n\nThis is not a claim that the kernel got slower per thread — the register "
            "column has ordered time backwards five times here (#47, #63, #67, #76, #94), "
            "and #94 spent 40 -> 167 registers on `gemm_cg2` for +13.7%. It is the one "
            "consequence of a register count that does not depend on any theory of what "
            "the registers are for: CTAs come off the SM. A kernel whose grid is sized "
            "from CTAS_PER_SM (`gemm_cg2`'s is `SMS * CTAS_PER_SM / RANKS` clusters) is "
            "then sized for a residency it does not have. If the trade is worth it, say "
            "so and move CTAS_PER_SM — but move it having measured, not having seen this "
            "go red."
        )
    print("\n  every gated kernel is inside its step.")


@app.function(cpu=8, timeout=CHECKING)
@completes
def regcount(arch: str = "sm_100a", label: str = "", determinism: bool = False) -> None:
    """Register pressure of every kernel the harness and the two kernel crates emit,
    from `ptxas -v`.

    Register count is *the* performance number for this library — a tile
    abstraction that costs registers has failed at the thing it exists for, and
    "ptxas says otherwise" is the escape hatch the README promises. `ptxas` is a
    host compiler, so this needs no GPU: build both PTX-emitting crates, feed
    the emitted PTX back through `ptxas -v`, and print a sorted table. Run it
    before and after a change and diff the two.

    The kernel crates are here because a register count is only half an occupancy
    argument and the other half is a launch: `THREADS` and registers per thread
    together say how many CTAs of a *real* kernel fit on an SM, which is what
    #47 turned out to be about. Probes have no launch and cannot say it.

    Only kernels that are actually monomorphized appear — an op no kernel calls
    emits no PTX and measures nothing, so a codegen probe in `device-tests` is
    how a bare library function gets onto this table at all.

    `device-tests`' `ladder!(..)` generates a probe per (shape, spelling), and
    the second table below is that sweep with the cliff located — the widths
    and *row extents* at which each spelling of the same step starts to spill.

    The third table is #63's: the four rungs `::ladder_bench` puts a clock on,
    each beside the ladder rung it stands in for. A timing only belongs next to
    a register count if the two describe the same compiled kernel, and that is
    a thing to check here, on CPU, before spending a B200 on it.

    The fourth is #95's, and it is a **gate**: for every kernel in
    `GATED_KERNELS`, the registers `ptxas` allocated against the most it may use
    and still keep the CTAs per SM its own grid is sized for. That ceiling is
    derived from the launch and the register file, checked against the toolkit's
    `cuda_occupancy.h` on every run, and it fails the run when it is crossed.
    Nothing else here fails on a register count, and nothing else should: this
    one is about *occupancy*, which is the one consequence of the count that
    does not depend on a theory of what the registers are for.

    `--determinism` measures the same tree twice, with both crates' artifacts
    thrown away in between, and asserts the two tables are identical. It is not
    ceremony: a diff of this table is only evidence if the table is a function
    of the tree, and #31 attributed a surprise 71 -> 32 register swing to its own
    refactor rather than to noise on exactly this control. It costs a second
    build of those two crates (their dependencies stay compiled).
    """
    subprocess.run(["ptxas", "--version"], check=True)
    measured = _measure(arch)
    print(f"\nregisters per thread, {arch}" + (f" — {label}" if label else ""))
    _print_kernels(measured)
    _print_census()
    _print_ladder(measured)
    _print_timed_twins(measured)
    _check_occupancy_step(measured)

    if not determinism:
        return

    print("\n=== determinism control: the same tree, measured again ===", flush=True)
    # Throwing the PTX away is what makes the second pass a measurement rather
    # than a re-read: a build that skipped codegen would hand back the same
    # file and pass this control without having tested anything. `_measure`
    # raises if the artifacts do not come back.
    for _, directory in PTX_CRATES:
        for ptx in Path(directory).rglob("*.ptx"):
            ptx.unlink()
    for package, directory in PTX_CRATES:
        _run(["cargo", "clean", "-p", package], cwd=directory)
    again = _measure(arch)

    differences = [
        f"  {name} ({source}): {measured.get((source, name))} -> {again.get((source, name))}"
        for source, name in sorted(measured.keys() | again.keys())
        if measured.get((source, name)) != again.get((source, name))
    ]
    if differences:
        raise RuntimeError(
            f"regcount is not deterministic on this tree: {len(differences)} of "
            f"{len(measured.keys() | again.keys())} kernels changed across two "
            "builds of the same source. No diff against another tree means "
            "anything until this passes.\n" + "\n".join(differences)
        )
    print(f"identical: {len(measured)} kernels, every counter, across two builds.")


@app.local_entrypoint()
def main() -> None:
    device_tests.remote()
