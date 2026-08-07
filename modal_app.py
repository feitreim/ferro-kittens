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
    modal run modal_app.py::sol_ablate
                                      # gemm_sol's phases priced one at a time,
                                      # with upstream's own K loop as the
                                      # reference the port is short of
    modal run modal_app.py::upstream_bench
                                      # the port, the kernel it is a port of
                                      # unported, and cuBLASLt beside both --
                                      # one container, so one device and one
                                      # day. `--case` takes a comma-separated
                                      # list for exactly this reason
    modal run modal_app.py::upstream_ptx
                                      # why that entry point needs an `opt`
                                      # wrapper: upstream's own crate does not
                                      # assemble in this image. No GPU
    modal run modal_app.py::bench --case swizzle
                                      # gemm's item traversal, tile held fixed
    modal run modal_app.py::bench --case tile
                                      # gemm's pair tile and pipeline depth,
                                      # traversal held fixed (#87)
    modal run modal_app.py::bench --case crossover
                                      # the same two tiles across the sizes
                                      # below 2048^3 that nothing had timed,
                                      # which is the rule gemm::plan_for picks
                                      # by (#105). Minutes, not the hour
                                      # --case tile costs
    modal run modal_app.py::bench --case staged
                                      # gemm's epilogue SHAPE: a register drain
                                      # against one staged through shared
                                      # memory by stmatrix (#15)
    modal run modal_app.py::bench --case widths
                                      # gemm's epilogue INSTRUCTION WIDTHS:
                                      # .x8 LDTM and stmatrix .x4 on the staged
                                      # epilogue, ablated and composed (#117)
    modal run modal_app.py::bench --case ldmatrix
                                      # the OTHER direction's width: softmax's
                                      # loads at ldmatrix .x2 against .x4,
                                      # paired and interleaved (#131)
    modal run modal_app.py::bench --case norm-occupancy
                                      # when a tile walk loses (#222): a
                                      # block-per-row rms norm against the walk
                                      # at both of the walk's levers, with the
                                      # driver's registers and blocks/SM beside
                                      # the clock
    modal run modal_app.py::bench --case residual
                                      # where the gap to cuBLASLt lives, with
                                      # every control re-run in one container
                                      # and the residual epilogue decomposed
    modal run modal_app.py::session --arms clc,ws,bench:gemm-sol
                                      # several of the sweeps above in ONE
                                      # container: one image pull, one build,
                                      # one device, one clock. The commands are
                                      # the entry points' own, off one table
    modal run modal_app.py::profile   # one launch under Nsight Compute (see
                                      # the note there: no counters on Modal)
    modal run modal_app.py::doctor    # env / GPU sanity check
    modal run modal_app.py::stall     # does nothing, out loud -- the control
                                      # for scripts/modal-run (#103)
    modal run modal_app.py::wedge_demo
                                      # launches a kernel that does not return,
                                      # and shows a session surviving it -- the
                                      # control for kittens::watchdog, the way
                                      # stall is the control for modal-run
"""

import functools
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import modal

# Keep this revision in sync with the git deps in Cargo.toml: the codegen
# backend and the device/host/core crates must come from the same revision.
CUDA_OXIDE_REF = "20a56163f258e09f2c51e4c27ae4e4ff17582443"
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

# --- the cargo cache, and the measurement that says where it belongs ---------
#
# Every entry point here used to build into a `target/` that died with its
# container. The image's warm caches were populated for `/opt/warmup`, so the
# project got no benefit from them and each run recompiled the pinned dependency
# tree from nothing. CI.md called that out and left it; this volume is the fix.
# `target/` lives on it now, one directory per crate, so the layout is the
# layout the four crates already had and cargo sees exactly what it saw before
# -- only somewhere that outlives the container.
#
# **It is mounted by the CPU entry points and deliberately not by the GPU ones,
# and that is a measurement rather than a preference.** The obvious plan was the
# other one: fill the cache on `build`'s eight CPUs, then let a B200 skip the
# compile it was being billed for. Run at 41f9e57 on `device_tests`, warm:
#
#     dependency crates compiled     58  ->  1     (the cache works)
#     `cargo` inside the container    53.3 s -> 66.0 s, 60.0 s over two runs
#
# The cache does exactly what it claims and the container still gets slower,
# because the 58 dependencies were never the cost. They compile in parallel on
# `cpu=8` and hide behind the long pole, which is the crate's own device codegen
# -- and cargo-oxide redoes that on **every** invocation, cache or no cache: two
# consecutive `run`s of an unchanged tree both recompiled `device-tests`. So on
# a GPU function the cache trades work that was already free for artifact I/O
# over a network filesystem, and loses. (`regcount --determinism`'s doc has been
# saying the same thing from the other side for a long time: a second pass
# "costs a second build of those two crates -- their dependencies stay
# compiled".)
#
# What that leaves is worth writing down, because it is the whole GPU spending
# story here: **a GPU container's build cannot be cached away, so the only lever
# on it is fewer containers.** That lever is `session` below.
#
# The *registry* is deliberately not on here either, for the reason the GPU
# functions are not: every dependency is already baked into `/root/.cargo` by
# the warmup layer above, which is a content-addressed image layer and therefore
# free, and a second copy on a network volume would be slower than the one in
# the image.
#
# Two runs of two different trees share this, and Modal commits a volume per
# file when the container exits, so the loser of a race sees the other tree's
# artifacts. That is the case cargo is built for and not a hazard: a unit's
# output is filed under a hash of its inputs, so a foreign artifact either has a
# name this build never asks for or is byte-identical to the one it wanted. The
# worst outcome is a rebuild, which is what every run did before this existed.
#
#     modal volume ls ferro-kittens-cargo              # what is on it
#     modal volume rm -r ferro-kittens-cargo /target   # start over
#
# Deleting it is always safe and costs exactly one cold build.
CARGO_CACHE = modal.Volume.from_name("ferro-kittens-cargo", create_if_missing=True)
CACHE_DIR = "/cache"
CACHE = {CACHE_DIR: CARGO_CACHE}

HARNESS_DIR = f"{PROJECT_DIR}/device-tests"
EXAMPLES_DIR = f"{PROJECT_DIR}/examples"
EXPERIMENTS_DIR = f"{PROJECT_DIR}/experiments"
# The driver stub satisfies cargo-oxide's link step where there is no GPU.
STUB_ENV = ["env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"]


# --- the timeouts, and the measurement each one is three times ----------------
#
# Every GPU entry point carried `timeout=5400` -- ninety minutes, uniform across
# six functions doing very different amounts of work, and derived from nothing.
# That ceiling is what a wedged container rides to the end, and a wedged
# container is indistinguishable from a slow one from the outside: one died after
# twenty-six minutes having printed fifteen lines of NVIDIA banner and never
# reached Python, billed at B200 rates the whole way.
#
# The rule since is the one oxide-train#131 wrote down for the twin repo: **a
# timeout is three times a measured baseline, and the baseline is in the comment
# beside it.** An entry point whose work is chosen by its arguments has no
# baseline to state and stays open-ended -- and says so, rather than merely
# being large. Sized:
#
#   entry point       measured                                    timeout
#   build             291.8 s warm, 470.8 s worst (#226)           900   3.1x warm
#   regcount          183-184 s (CI.md's per-line timestamps)      900   4.9x
#   upstream_ptx      a clean clone plus ptxas; unmeasured         900   see below
#   device_tests      83.5 s (#226 corrected tier 3's cost line)   300   3.6x
#   examples          two binaries, both correctness gates;        1200  open-ended
#                     no published measurement
#   the sweeps        `--case`/`--arms` chooses the work           3600  open-ended
#   wedge_demo        three arms, one of them killed at 5 s        1200  see below
#   doctor            one driver query, nothing built              300   open-ended
#   stall             a nap of a length its caller picks           300   open-ended
#
# A ceiling is not a bill: the container is billed for what it runs, so the only
# thing a loose ceiling costs is a wedge riding longer. That is why the numbers
# above are worth tightening at all, and it is also why tightening them is not
# the *first* line of defence any more -- a launch that does not return is now
# `kittens::watchdog`'s 30 s, one process, one row (see `ARMS`).
CHECKING = 900  # compile and lint, no launches: `build`, `regcount`, `upstream_ptx`
# 83.5 s measured on tier 3 at 41f9e57: 57 cases in ~20 s of B200 behind a
# compile costing two and a half times as much. It was on `RUNNING` (1200) with
# `examples`, which is 14x its own measurement -- the two do different amounts
# of work and now carry different ceilings.
HARNESS = 300  # the device-test binary, every case: `device_tests`
# Open-ended, and the reason is that nobody has measured it: `examples` runs
# *two* binaries -- `kittens-examples`' four teaching kernels and
# `kittens-experiments -- check`'s every rung and probe that computes a GEMM, at
# both check sizes and all three traversal widths. That is strictly more work
# than `device_tests` and no run of it has been timed on the record. Put it in a
# session beside a measured arm and this becomes a number.
RUNNING = 1200  # compile and a correctness gate: `examples`
# Open-ended for a reason that will not go away: what a sweep does is chosen by
# its argument. `bench` with no `--case` runs every table including 16384^3;
# `bench --case softmax` is 79.7 s (#226). `session --arms` is however many arms
# were asked for. `profile` replays launches under a profiler. There is no
# single measurement for any of them, so there is no 3x to take -- 3600 rather
# than 2700 since #122 added a seventh table to `bench --case residual`.
SWEEPING = 3600  # work chosen by argument: `bench`, `session`, the named sweeps
# `doctor` is one driver query and a `cargo oxide doctor` against the warmup
# crate. `stall` sleeps for as long as its caller asks, so this is a bound on its
# argument rather than a baseline: `stall --seconds 400` dies at the ceiling,
# which is correct -- the control for a wedge should not be able to become one.
ASKING = 300  # a driver query, or a nap of a stated length: `doctor`, `stall`


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


def _cached_target(cwd: str) -> dict[str, str]:
    """Where the crate rooted at `cwd` compiles to: the cache volume, under the
    crate's own name -- if this container has the cache volume at all.

    Mounting `CACHE` is the whole switch, and it is deliberately the only one.
    An entry point that mounts it builds onto it; one that does not builds where
    it always did, into a `target/` beside the crate. There is no second flag to
    forget to flip, and the mount is visible on the `@app.function` line.

    A directory per crate rather than one shared `CARGO_TARGET_DIR` for all
    four, because that is what the tree already had -- four separate workspaces,
    four `target/`s -- and the point of this change is placement, not a new
    build graph. `_measure` reads the emitted PTX out of the crate directory
    (cargo-oxide writes it to the working directory, not under `target/`), so
    moving `target/` out from under the crates does not move any artifact this
    file reads.

    Anything outside the project -- `upstream_ptx`'s throwaway clone, the warmup
    crate `doctor` asks -- is left alone: a cache is only worth having where the
    same tree comes back."""
    if not cwd.startswith(PROJECT_DIR) or not Path(CACHE_DIR).is_dir():
        return {}
    return {"CARGO_TARGET_DIR": f"{CACHE_DIR}/target/{Path(cwd).name}"}


# Whether this container has already been made safe for the cache volume. Once
# per container, not once per command: `_freshen` forces a rebuild of the
# project's own crates, and doing it between `clippy` and `test` would make the
# second step redo what the first had just done.
_FRESHENED = False


def _freshen() -> None:
    """Give the mounted sources a **new mtime**, so cargo can see that they
    changed.

    Without this the cache volume serves a stale build, silently, and the gate
    that is supposed to reject a bad diff passes a tree it never compiled. It was
    caught on the PR that added this: `cargo clippy --features host
    --all-targets` finished in **1.05 s having compiled nothing**, and the run's
    `cargo test` then reported **127** unit tests -- from a `libkittens.rlib`
    built before the two new ones existed. The tell was `rustdoc`, which is the
    one step that reads the sources unconditionally: it found the new module's
    doctest and failed it against the stale rlib. With the volume emptied, the
    same tree built and ran **129**.

    The mechanism is cargo's freshness check, which for a path crate is *mtime
    against the fingerprint*. Modal's mounts do not hand the container mtimes
    newer than artifacts already on the volume, so an edited file can look older
    than the object built from its predecessor, and cargo is right to believe
    what it was told. Nothing here is cargo's fault or the volume's; it is the
    pairing.

    So the fix is to make the tree honest rather than to distrust cargo: touch
    every file it fingerprints, once, before the first command runs. The
    project's crates are then rebuilt on every invocation -- which is the
    behaviour anyone reading a gate assumes -- and the dependency tree, which is
    what the volume is actually worth (#226: the four `clippy` steps went 67.6 s
    -> 17.3 s), is untouched because none of it lives under `PROJECT_DIR`.

    A no-op without the volume: an entry point that does not mount it builds into
    a `target/` that died with its container, so there is nothing stale to be."""
    global _FRESHENED
    if _FRESHENED or not Path(CACHE_DIR).is_dir():
        return
    _FRESHENED = True
    print(f"$ touch every source under {PROJECT_DIR}  (the cache volume is mounted)", flush=True)
    subprocess.run(
        ["find", PROJECT_DIR, "(", "-name", "*.rs", "-o", "-name", "*.toml",
         "-o", "-name", "*.lock", ")", "-exec", "touch", "{}", "+"],
        check=True,
    )


def _run(cmd: list[str], cwd: str, env: dict[str, str] | None = None) -> None:
    _freshen()
    if env:
        print(f"$ {' '.join(f'{name}={value}' for name, value in env.items())}", flush=True)
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    start = time.monotonic()
    try:
        subprocess.run(
            cmd, cwd=cwd, check=True, env={**os.environ, **_cached_target(cwd), **(env or {})}
        )
    finally:
        # Printed on the failure path too -- a step that ran for nineteen of a
        # twenty-minute budget and a step that died on contact are different
        # diagnoses, and the exception alone does not distinguish them.
        print(f"  ({time.monotonic() - start:.1f}s)", flush=True)


@app.function(cpu=8, timeout=CHECKING, volumes=CACHE)
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

    # The vendored upstream `gemm_sol_final`, which is 1300 lines of device code
    # nothing else in this repository compiles. It is the only kernel here whose
    # gate has to include `ptxas`: `cargo oxide build` emits its PTX happily and
    # `opt -O2` has put sixteen illegal lookup tables in it, so a build alone
    # would pass on a bundle that cannot load. So this step asserts both halves
    # of the workaround at once -- that upstream's code still compiles, and that
    # `-switch-to-lookup=false` still makes what it compiles to assemble.
    # `upstream_ptx` is the diagnosis if this goes red; `OPT_NO_LOOKUP_TABLE` is
    # where the reasoning lives.
    _run(["sh", "-c", WRITE_OPT_WRAPPER], cwd="/")
    _run([*STUB_ENV, "env", f"CUDA_OXIDE_OPT={OPT_NO_LOOKUP_TABLE}",
          "cargo", "oxide", "build", "kittens-experiments", "--arch", "sm_100a",
          "--features", "cublas,gemm-sol-upstream"],
         cwd=EXPERIMENTS_DIR)
    # A real cubin rather than `/dev/null`, because the next two steps read it.
    # `ptxas` succeeding is still the gate; the disassembly is a report.
    _run(["ptxas", "-arch=sm_100a", "-o", "kittens_experiments.cubin",
          "kittens_experiments.ptx"],
         cwd=EXPERIMENTS_DIR)
    # The MMA warp's K-loop issue stream, ours and upstream's, off the one bundle
    # that carries both. It lives here rather than in `regcount` because
    # `regcount` builds without features and so never emits upstream's kernels,
    # and because this is the only step that already pays for that codegen.
    _print_mma_stream()
    # And the same kernels' loops as the machine runs them, which is the one place
    # this repo's PTX counts are checked against SASS at all -- see #150 and #151.
    _print_sass_loops(Path(EXPERIMENTS_DIR, "kittens_experiments.cubin"),
                      MMA_STREAM_KERNELS)


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
@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=HARNESS)
@completes
def device_tests() -> None:
    """The harness itself. One binary, every case, non-zero exit on failure."""
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _arm("device-tests")


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
    _arm("examples")


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
    _run(SMI, cwd="/")
    # `--case` narrows the sweep to one table, and `--m/--n/--k` to one row of
    # it. Re-staging 16384^3 to re-read a diagnostic row costs more host time
    # than the row does device time. `--features cublas` (#92) puts a cuBLASLt
    # column and a ratio beside the `gemm` table: it is off in the crate's
    # default feature set so that tier 1 CI and anyone without a devel CUDA
    # toolkit are unaffected, and on here because this image always has one.
    # Both live in `_bench_arm`, which `session` runs as `bench:<case>`.
    for cmd, cwd in _bench_arm(case, m, n, k):
        _run(cmd, cwd)


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
    _run(SMI, cwd="/")
    _arm("clc")


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def ws_bench(case: str = "") -> None:
    """The warp-specialized GEMM against the one it is a variant of.

    `--case shallow` runs #188's arm instead: the same A/B at the K = 3072
    geometries oxide-train#80 counts in (gate_up fwd, qkv fwd, o_proj fwd, and
    8192^3 as the deep-K control), five arms in one container -- `gemm
    staged84`, `ws staged8`, `ws s6`, the `ws no drain` floor, and cuBLASLt.
    Every prior gemm_ws table is at K = M = N, where the overlap the design
    buys is worth the least; the floor is the kill criterion, since a floor
    already losing to `gemm staged84` closes the design point cheaply.

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
    _run(SMI, cwd="/")
    # An unrecognised `--case` used to be passed through and silently ignored by
    # `main.rs`, which runs `compare` for anything that is not `shallow`. Here it
    # is a missing key and so a failed run, which is the right answer for an
    # argument that asked for a table nobody has.
    _arm(f"ws-{case}" if case else "ws")


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
    _run(SMI, cwd="/")
    _arm("ladder")


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

# Whether the performance-counter library is here at all, printed before the
# profiler is asked for a counter. It is `|| echo` rather than a failure because
# its absence is the *expected* state on this image and the message is the whole
# diagnosis -- see `profile` below.
PCC_PRESENT = (
    "ls /usr/lib/x86_64-linux-gnu/libnvidia-pcc* 2>/dev/null"
    " || echo 'libnvidia-pcc.so absent: the profiler will report LibraryNotLoaded'"
)


def _profile_arm(
    kernel: str = "gemm", m: int = 8192, n: int = 8192, k: int = 8192
) -> list[tuple[list[str], str]]:
    """`profile`'s own commands, so a session and the entry point issue the same
    ones -- `ARMS`' rule, and this arm is new to that table since the launch
    watchdog made a serializing profiler safe to batch."""
    return [
        (["sh", "-c", PCC_PRESENT], "/"),
        # Build first: `--target-processes all` would otherwise follow the
        # compiler around for ten minutes looking for a context it never creates.
        (["cargo", "oxide", "build", "kittens-experiments", "--arch", "sm_100a"],
         EXPERIMENTS_DIR),
        (
            [
                NCU, "--target-processes", "all", "--clock-control", "none",
                "--kernel-name", f"regex:{kernel}", "--launch-skip", NCU_SKIP,
                "--launch-count", NCU_COUNT, "--print-details", "all",
                *[argument for section in NCU_SECTIONS for argument in ("--section", section)],
                "cargo", "oxide", "run", "kittens-experiments", "--",
                "bench", kernel, str(m), str(n), str(k),
            ],
            EXPERIMENTS_DIR,
        ),
    ]


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

    **#138 re-derived this and closed the two obvious escapes.** On a fresh
    B200 container at driver 580.95.05, `ncu` 2025.3.0 fails identically on a
    four-line `nvcc` kernel that has nothing to do with this repo, so it is not
    the codegen. `NVIDIA_DRIVER_CAPABILITIES=all` is *already in the container's
    environment* and the injected set already includes the graphics libraries,
    so the capability set is not the lever it looks like. And NVIDIA's own
    component list for the 580 branch names exactly one `pcc` file — `nvidia-pcc`,
    the VulkanSC pipeline cache compiler — and no performance-counter library at
    all, so this is not something the runtime withheld from a package that has
    it. `libnvperf_host.so` and `libnvperf_target.so` are both present and both
    useless without it; `nsys` is not in the image, and it would trace rather
    than count. The conclusion is the same and now has a floor under it: there
    is no counter here, and an ablation is the instrument.

    Whenever the library does appear, this is one command. `--clock-control
    none` is deliberate: the default locks the clocks to base so that two runs
    compare, which also means the duration it reports is not the duration
    `bench` reports, and half the point is to divide a measured byte count by a
    duration that belongs to the same table.
    """
    _run(SMI, cwd="/")
    # In `_profile_arm`, which `session` runs as the `profile` arm.
    for cmd, cwd in _profile_arm(kernel, m, n, k):
        _run(cmd, cwd)


@app.function(gpu=DEFAULT_GPU, timeout=ASKING)
@completes
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


# The `opt` the vendored upstream kernels need, and why they need one.
#
# `opt -O2` turns each of upstream's four-way stage selects -- `SMEM_A0..3`,
# `TMA_BAR0..3`, and six more -- into a lookup table, and emits that table as a
# `.global` array initialized with the addresses of `.shared` variables. PTX
# forbids exactly that: *"Variable used as initial value not in .global or
# .const state space"*, `ptxas` fatal, no code. Sixteen tables in our build of
# it and sixteen in **upstream's own**, which `upstream_ptx` demonstrates from a
# clean clone -- so this is a defect of the toolchain and not of the vendoring.
#
# `-switch-to-lookup=false` is LLVM's own switch for the transform that makes
# them, and `CUDA_OXIDE_OPT` is cuda-oxide's own hook for supplying the `opt`
# binary. Composing the two is a build-environment change and touches no source:
# with the wrapper, all sixteen tables are gone and every one of the 46 entries
# assembles for `sm_100a`.
#
# It is not free of consequence and is not assumed to be: the flag is global to
# the compilation, so `upstream_bench` measures the port with it and without it
# in one container and prints both. A difference there is the flag's, not the
# kernel's, and belongs in the report.
OPT_NO_LOOKUP_TABLE = "/tmp/opt-no-lookup-table"
WRITE_OPT_WRAPPER = (
    f"printf '#!/bin/sh\\nexec %s -switch-to-lookup=false \"$@\"\\n'"
    f" \"$(which opt)\" > {OPT_NO_LOOKUP_TABLE}; chmod +x {OPT_NO_LOOKUP_TABLE};"
    f" {OPT_NO_LOOKUP_TABLE} --version | head -2"
)


# --- the sweeps as arms, so one container can run several ---------------------
#
# Every GPU entry point in this file is its own Modal function, so asking for
# three tables is three containers: three image pulls, three cargo builds and
# three B200 allocations for a couple of minutes of device time each. `bench.rs`
# already makes this argument one level down -- its `--case` takes a
# comma-separated list precisely so that a kernel, the kernel it is a port of
# and their shared baseline are read off one device on one day -- but the sweeps
# that are not `Case`s cannot ride that list: `clc`, `ws`, `sol-ablate` and
# `upstream` each dispatch on `argv[1]` in `main.rs` and return.
#
# So they are named here instead, and `session` runs any comma-separated set of
# them in one container. The entry points below are thin wrappers over the same
# table, so a session and N separate `modal run`s issue **byte-identical**
# commands and differ only in how many containers pay for them. That is the
# property that makes batching safe to reach for: it cannot change a number.
#
# **Every sweep has an arm now, including the two that used to be barred.**
# `bench --case sol-k` and `profile` were kept out of this table because a launch
# that hangs takes every row after it -- which is how `sol-k` was found (#146:
# `4096x4096x1024` printing nothing for 1200 s until `scripts/modal-run` stopped
# the container). Two things had to be true before they could come in, and both
# now are:
#
#   1. **A launch that does not return costs one row, not a container.** Every
#      host wait in the three kernel crates goes through `kittens::watchdog`,
#      which polls the event behind the work instead of blocking on it and ends
#      the process past 30 s naming the row the sweep last announced. A wedged
#      arm exits -6 in half a minute.
#   2. **An arm's failure is contained by a process boundary.** It always was --
#      `_run` is a `subprocess` -- and `_session` below no longer throws the rest
#      of the run away when one of them fails. The arms after a failure run, and
#      the summary says which failed.
#
# `modal_app.py::wedge_demo` is what says this is true rather than reasoned:
# it launches a kernel that spins for ten minutes and takes a three-arm session
# through it. The rule that replaces the old one is narrower and worth keeping:
# **an arm may fail, and may not take the container with it.**
RUN = ["cargo", "oxide", "run"]
CUBLAS = ["--features", "cublas"]
UPSTREAM = ["--features", "cublas,gemm-sol-upstream"]
WITH_OPT = ["env", f"CUDA_OXIDE_OPT={OPT_NO_LOOKUP_TABLE}"]
WRITE_WRAPPER = (["sh", "-c", WRITE_OPT_WRAPPER], "/")

ARMS: dict[str, list[tuple[list[str], str]]] = {
    "device-tests": [([*RUN, "device-tests"], HARNESS_DIR)],
    "ladder": [([*RUN, "device-tests", "--", "bench-ladder"], HARNESS_DIR)],
    "examples": [
        ([*RUN, "kittens-examples"], EXAMPLES_DIR),
        ([*RUN, "kittens-experiments", "--", "check"], EXPERIMENTS_DIR),
    ],
    "clc": [([*RUN, "kittens-experiments", "--", "clc"], EXPERIMENTS_DIR)],
    "ws": [([*RUN, "kittens-experiments", *CUBLAS, "--", "ws"], EXPERIMENTS_DIR)],
    "ws-shallow": [
        ([*RUN, "kittens-experiments", *CUBLAS, "--", "ws", "shallow"], EXPERIMENTS_DIR)
    ],
    "sol-ablate": [
        ([*RUN, "kittens-experiments", *CUBLAS, "--", "bench", "sol-ablate"], EXPERIMENTS_DIR),
        WRITE_WRAPPER,
        ([*WITH_OPT, *RUN, "kittens-experiments", *UPSTREAM, "--", "bench", "sol-ablate"],
         EXPERIMENTS_DIR),
    ],
    "upstream": [
        ([*RUN, "kittens-experiments", *CUBLAS, "--", "bench", "gemm-sol"], EXPERIMENTS_DIR),
        WRITE_WRAPPER,
        ([*WITH_OPT, *RUN, "kittens-experiments", *UPSTREAM, "--", "bench",
          "gemm-sol,gemm-sol-upstream,gemm-sol-upstream-m512"], EXPERIMENTS_DIR),
    ],
    # The default `gemm` at 8192^3, which is what `profile` with no arguments
    # asks for. A session cannot pass the other three, and that is the right
    # trade: a profile of another kernel or another size is a one-argument
    # `modal run` and does not want company.
    "profile": _profile_arm(),
}


def _arm(name: str) -> None:
    """Run one named sweep in whichever container is already running."""
    for cmd, cwd in ARMS[name]:
        _run(cmd, cwd)


def _bench_arm(case: str, m: int = 0, n: int = 0, k: int = 0) -> list[tuple[list[str], str]]:
    """`bench`'s own commands. Its `--case` is already a comma-separated list, so
    this is one arm however many tables it prints; `--m/--n/--k` narrows it to a
    single row, which is only meaningful with a case named."""
    narrowed = [case] if case else []
    if case and (m or n or k):
        narrowed += [str(m), str(n), str(k)]
    return [([*RUN, "kittens-experiments", *CUBLAS, "--", "bench", *narrowed], EXPERIMENTS_DIR)]


# The four-column form. `device_tests` and `examples` ask for two of them, since
# neither reports a rate and neither reads a clock.
SMI = ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
       "--format=csv"]

OK = "ok"
# What a process ended by `kittens::watchdog` exits with: the module calls
# `abort()`, so the shell sees SIGABRT and `subprocess` reports -6. Named
# because the summary line is where somebody first meets it, and "-6" on its own
# reads like a mystery rather than like the deadline doing its job.
WEDGED = -6


def _steps(name: str) -> list[tuple[list[str], str]]:
    """The commands one arm name runs -- `ARMS`, plus `bench:<case>`."""
    if name.startswith("bench:"):
        return _bench_arm(name.removeprefix("bench:"))
    return ARMS[name]


def _session(arms) -> list[tuple[str, str, float]]:
    """Run named arms in this container, one process each, and say how each went.

    **A failing arm does not end the session.** It used to, and the argument was
    that the arms after it would be reported against a container that had already
    failed. What made that argument bite was the case it could not survive -- a
    launch that hangs, holding the container until Modal's `timeout=` -- and that
    case is now bounded by `kittens::watchdog` at 30 s inside the arm's own
    process. What is left is an ordinary non-zero exit, which is contained by the
    process boundary `_run` has always had. So the session runs on and prints a
    summary; the caller decides what a failed arm means, which is how
    `wedge_demo` can require exactly one.

    Yields nothing and returns everything: a session is minutes long and the
    summary is the thing worth having in one place at the end of it, after the
    thousands of lines the arms themselves printed."""
    outcomes: list[tuple[str, str, float]] = []
    for name, steps, env in arms:
        print(f"\n=== session arm: {name} ===", flush=True)
        start = time.monotonic()
        try:
            for cmd, cwd in steps:
                _run(cmd, cwd, env)
            outcome = OK
        except subprocess.CalledProcessError as failure:
            outcome = f"FAILED (exit {failure.returncode})"
            if failure.returncode == WEDGED:
                outcome += " -- the launch watchdog"
            print(f"=== session arm: {name}: {outcome} ===", flush=True)
        outcomes.append((name, outcome, time.monotonic() - start))
    print("\n=== session summary ===", flush=True)
    for name, outcome, elapsed in outcomes:
        print(f"  {name:<28}{outcome:<40}{elapsed:>8.1f}s", flush=True)
    return outcomes


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def session(arms: str = "") -> None:
    """Several sweeps in one container -- one image pull, one build, one device.

    `--arms` is a comma-separated list of the names in `ARMS` above, plus
    `bench:<case>` for anything `bench` can narrow to. Order is the order given.

        scripts/modal-run session --arms clc,ws,bench:gemm-sol
        scripts/modal-run session --arms sol-ablate,upstream

    The comma is the arm separator, so a `bench:` arm names exactly one case --
    `bench:softmax,bench:layernorm` is two arms and two processes in this one
    container, and `bench --case softmax,layernorm` is the single process that
    prints both tables. Prefer the second when the tables are read against each
    other; the difference is a process start, not a device.

    Each arm runs the commands its own entry point runs, off the one table, so
    this is a placement change and not a measurement change.

    What it buys, measured at 41f9e57 on `bench:softmax,bench:layernorm,
    device-tests` -- three arms that took 79.7 s, 97.5 s and 81.0 s as three
    separate runs and 258.2 s together, against **161.2 s as one session**: 97 s
    of B200, 38% of the bill, for the same three tables. Two things are saved
    and one is not:

      * the image pull and container start, once instead of three times;
      * the *dependency* compile for an arm that follows one in the same crate
        -- `bench:layernorm` built in 28.6 s behind `bench:softmax`'s 40.6 s,
        against 60 s from cold;
      * **not** the crate's own device codegen, which cargo-oxide redoes for
        every invocation even inside one container. That is the same finding
        the cache volume ran into (see `CARGO_CACHE`), and it is why the answer
        to a GPU bill here is fewer containers rather than warmer ones.

    It also buys the thing #98 and #118 keep paying for by hand -- 2.9% of
    drift between two runs of the same tree, and a 16384^3 row that moved 3%
    between two sessions. Arms taken in one container share a device, a driver,
    a clock and a cuBLASLt, so a difference between two of them is theirs.

    **An arm may fail without taking the rest of the session with it**, which is
    the change that let `bench:sol-k` and `profile` into the table at all. Each
    arm is its own process -- `_run` is a `subprocess`, and always was -- and a
    launch that does not return is `kittens::watchdog`'s 30 s rather than the
    container's `timeout=`. So a failed arm is one row of the summary below, the
    arms after it run, and the session as a whole is red. What that does *not*
    license is reading a table from an arm that ran after a failure without
    asking what failed: a build error fails every arm identically, and a device
    that has gone unhealthy fails them for a reason none of them will print.
    The summary is there to be read."""
    _run(SMI, cwd="/")
    requested = [name for name in arms.split(",") if name]
    if not requested:
        raise RuntimeError(
            "session needs --arms: a comma-separated list of "
            f"{', '.join(sorted(ARMS))}, or bench:<case>"
        )
    unknown = [
        name for name in requested
        if name not in ARMS and not name.startswith("bench:")
    ]
    if unknown:
        # Loud, and before the first arm runs. A typo that silently ran two of
        # three sweeps would hand back a plausible-looking session missing the
        # arm somebody asked for -- the same failure `bench.rs` refuses for its
        # own case list, for the same reason.
        raise RuntimeError(
            f"no arm named {', '.join(unknown)}; the arms are "
            f"{', '.join(sorted(ARMS))}, plus bench:<case>"
        )
    failed = [
        name
        for name, outcome, _ in _session(
            (name, _steps(name), {}) for name in requested
        )
        if outcome != OK
    ]
    if failed:
        raise RuntimeError(f"{len(failed)} arm(s) failed: {', '.join(failed)}")


# `--features cublas,wedge`: the crate's usual feature for a session plus the
# spin kernel. It is a separate build of `experiments/` and therefore a separate
# device codegen, which is why this is its own entry point and not a flag on
# `session` -- a session's commands are supposed to be byte-identical to the ones
# its arms' entry points issue, and these deliberately are not.
WEDGE_FEATURES = ["--features", "cublas,wedge"]


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=RUNNING)
@completes
def wedge_demo(seconds: int = 120, budget_ms: int = 5000) -> None:
    """Watch the launch watchdog fire, and watch the session survive it.

    This is to `kittens::watchdog` what `stall` is to `scripts/modal-run`: the
    control that makes the guard something seen rather than something read. A
    deadline nobody has watched fire is not a deadline, and this one is the whole
    reason `bench:sol-k` and `profile` are allowed in `ARMS`.

    Three arms in one container, in this order:

      1. `bench:softmax` -- a green arm before the failure.
      2. `bench:sol-k` with `KITTENS_WEDGE_SECONDS` set, so `crate::wedge`
         queues a `seconds`-long spin *immediately in front of the first row's
         own launch*, on that row's own stream -- everything loaded, everything
         staged, and then a launch that will not finish. That is #146's failure
         exactly. It must fail, with `SIGABRT` from the watchdog, in about
         `budget_ms`.
      3. `device-tests` -- a green arm *after* the failure, which is the claim.
         57 cases, on the device the wedged arm was just using.

    The pass condition is the shape and not merely the exit code: arm 2 failed,
    arms 1 and 3 did not. A run where the wedged arm *passed* is a watchdog that
    did nothing; one where a green arm failed is a wedge that took a neighbour,
    which is the thing this whole change claims cannot happen. Both raise here.

    `budget_ms` is `KITTENS_LAUNCH_BUDGET_MS`, five seconds rather than the
    30 s default so the demonstration costs seconds of B200 rather than a minute
    -- and so the override itself is exercised. `seconds` is twenty-four times
    that; the spin is bounded rather than infinite so that a demonstration that
    goes wrong still gives the device back, which has already been worth having
    three times."""
    _run(SMI, cwd="/")
    green = "bench:softmax"
    wedged = "bench:sol-k"
    after = "device-tests"
    bench = [*RUN, "kittens-experiments", *WEDGE_FEATURES, "--", "bench"]
    outcomes = _session(
        [
            (green, [([*bench, "softmax"], EXPERIMENTS_DIR)], {}),
            (
                wedged,
                [([*bench, "sol-k"], EXPERIMENTS_DIR)],
                {
                    "KITTENS_WEDGE_SECONDS": str(seconds),
                    "KITTENS_LAUNCH_BUDGET_MS": str(budget_ms),
                },
            ),
            (after, ARMS[after], {}),
        ]
    )
    by_name = {name: outcome for name, outcome, _ in outcomes}
    wrong = [
        f"{name} was {by_name[name]}, wanted {want}"
        for name, want in ((green, OK), (wedged, "a failure"), (after, OK))
        if (by_name[name] == OK) != (want == OK)
    ]
    if wrong:
        raise RuntimeError("; ".join(wrong))
    print(
        f"\nthe wedged arm failed in {[e for n, _, e in outcomes if n == wedged][0]:.1f} s and "
        f"`{after}` ran after it. that is the whole claim.",
        flush=True,
    )


# No cache mount: this one builds a throwaway clone under `/tmp`, which is a
# different tree every run and the only thing here a cache could not help.
@app.function(cpu=8, timeout=CHECKING)
@completes
def upstream_ptx() -> None:
    """Build NVLabs' own `gemm_sol_final`, from a clean clone of the pinned
    cuda-oxide, and assemble its PTX. No GPU: `ptxas` is the whole question.

    This exists because the first attempt to time upstream's kernel against ours
    found it does not run, and the obvious suspicion -- that vendoring it into
    `experiments/` broke it -- had to be ruled out before the finding could be
    published. It is ruled out here, from upstream's crate, upstream's
    workspace, upstream's profile, in this image:

        .extern .func llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3
        ptxas gemm_sol_final.ptx, line 10; fatal : Parsing error near '.nvvm'

    and behind that one, sixteen `switch_$_table` arrays of `.shared` addresses.
    Two independent lowering defects, both fatal, both upstream's own. Ours are
    the same two, and `kittens::ldst` had already written the workaround for the
    first one at `b099f64` -- the revision it was observed at, which is no longer
    the pin.

    Run it after a pin bump: if this comes back clean, the two workarounds
    `gemm_sol_upstream.rs` and `OPT_NO_LOOKUP_TABLE` carry can both be dropped.
    It did not come back clean at `20a5616` (#145): same extern, same line. So
    the bump to `20a5616` drops neither workaround, and both stay.
    """
    _run(["git", "clone", "--filter=blob:none", GIT_REPO, "/tmp/oxide"], cwd="/")
    _run(["git", "checkout", CUDA_OXIDE_REF], cwd="/tmp/oxide")
    # The clone's own `.cargo/config.toml` aliases `oxide` to a workspace
    # member, which shadows the installed subcommand.
    _run(["rm", "-f", "/tmp/oxide/.cargo/config.toml"], cwd="/")
    upstream = "/tmp/oxide/crates/rustc-codegen-cuda/examples/gemm_sol_final"
    _run([*STUB_ENV, "cargo", "oxide", "build", "gemm_sol_final", "--arch", "sm_100a"],
         cwd=upstream)
    _run(["sh", "-c",
          "echo '--- unresolved externs ---'; grep -n '^\\.extern \\.func' gemm_sol_final.ptx;"
          " echo '--- lookup tables of shared addresses ---';"
          " grep -c 'switch_\\$_table' gemm_sol_final.ptx || true;"
          " echo '--- ptxas ---';"
          " ptxas -arch=sm_100a -o /dev/null gemm_sol_final.ptx 2>&1 | tail -10"],
         cwd=upstream)


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def upstream_bench() -> None:
    """The port, the kernel it is a port of, and cuBLASLt -- one container, one
    device, one clock.

    `bench --case gemm-sol` published 0.795 / 0.873 / 0.946 of cuBLASLt at
    4096³ / 8192³ / 16384³ with nothing to say whether that gap is the port's or
    the design's. This runs the design too, from upstream's device code
    unmodified (`experiments/src/gemm_sol_upstream.rs`), staged from the same
    generators, checked by the same exact-BF16 reference and timed by the same
    five-warm-up minimum-of-thirty clock, so the only thing left different is
    the code between the events.

    Two invocations, and the first one is the control. `OPT_NO_LOOKUP_TABLE` is
    what makes upstream's kernels assemble at all, and it is global to the
    compilation -- so the port is measured *without* it first, which is exactly
    what `bench --case gemm-sol` measures on any other day, and then again
    *with* it beside upstream. If those two port rows disagree, the flag moved
    the port and every ratio in the second table has to be read against the
    second port row rather than the published one.
    """
    _run(SMI, cwd="/")
    _arm("upstream")


@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=SWEEPING)
@completes
def sol_ablate() -> None:
    """`gemm_sol`'s decomposition with upstream's own kernel as the reference --
    one container, so one device, one cuBLASLt, one clock.

    `bench sol-ablate` prices each phase of the item, then asks the one question
    the phases cannot answer on their own: the `[256, 256]` entry's K loop runs
    at 75.8% of tensor-core peak where `[512, 256]`'s runs at 99.4%, and the two
    run the same K-loop code. Its last table is the same K-depth ladder on
    upstream's `gemm_sol_final` at the same tile, which is what splits that
    deficit into the algorithm's and the port's.

    That table needs `gemm-sol-upstream`, which needs `OPT_NO_LOOKUP_TABLE` to
    assemble at all, and the flag is global to the compilation -- so this runs
    the whole sweep *twice*, once without upstream compiled in and once with. The
    first run is the control: if the port's rows disagree between them, the flag
    moved the port and the upstream comparison has to be read against the second
    run's port rows rather than the first's. #145 measured that difference at
    0.006-0.15%.
    """
    _run(SMI, cwd="/")
    _arm("sol-ablate")


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
# `examples` and `experiments` both emit `gemm_cg2_staged_x8x4` and the three
# `gemm_sol` entries, because they are the same kernels: the teaching crate
# ships them with every const baked to what it ships at, and the notebook keeps
# the dialed body every arm is an instantiation of. `_measure` keys on
# `(ptx file, kernel)` so both rows are printed, and the pair is a free check
# that baking the dials in moved no instruction -- two identical rows are the
# claim, and a difference between them is the finding.
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
            roots = [Path(directory), *map(Path, _cached_target(directory).values())]
            listing = sorted(
                p for root in roots for p in root.rglob("*") if p.is_file()
            )
            raise RuntimeError(
                f"no PTX under {package}'s crate or target dir; cargo-oxide's artifact "
                f"layout must have moved. Files found:\n" + "\n".join(map(str, listing[:200]))
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
    # The *waits*, which are a separate claim from the issues and the only
    # column that can see one.
    #
    # #117 found that the LDTM half of a staged epilogue is its wait and not its
    # issue, and `TmemTile::tile_x8_batched` acts on that by putting a band's
    # issues in flight before a single `tcgen05.wait::ld`. That changes no other
    # count in this table -- same `tcgen05.ld`, same `stmatrix`, same stores --
    # so without this column a batched drain and an unbatched one census
    # identically and the arm cannot be gated at all.
    ("ldtm.wait", "tcgen05.wait::ld"),
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


# The kernels whose MMA warp's issue stream is itself an argument, and the
# opcode that brackets it.
#
# `bench sol-ablate`'s `mma only` arm runs the MMA warp with no barrier wait, no
# TMA and no drain, and the `[256, 256]` entry still reaches only 88.7% of
# tensor-core peak where `[512, 256]` reaches 97.6%. Nothing memory-shaped is
# left in that arm, so what is left is the scalar work the warp does *between*
# `tcgen05.mma` issues -- ring index, byte multiply, two operand descriptors, an
# accumulate predicate -- which is the same per K block at both entries while the
# work per K block is 4 MMA against 8.
#
# That is a claim about instructions, and the PTX is where it is either true or
# not. `mma` counts alone cannot see it: 16 and 32 is exactly what the two
# entries should carry. What matters is how many non-`mma` instructions sit
# between them, so this prints the span rather than a count of it.
MMA_STREAM_KERNELS = (
    "gemm_sol_m256",
    "gemm_sol_m512",
    # The same two with the ring index computed from `global_k` instead of
    # handed in as a const, which is what the port carried before the two were
    # measured against each other. Printing both is what makes the fold
    # visible rather than asserted: the folded stream carries literal barrier
    # offsets and a constant accumulate predicate where this one carries
    # arithmetic.
    "gemm_sol_m256_runtime",
    "gemm_sol_m512_runtime",
    # Upstream's own two entries, when the vendored copy is compiled in. #148
    # closed the port's K-loop deficit against upstream from 6.0 points of peak
    # to 3.5 and could not explain the rest; a static diff of the two issue
    # streams costs no GPU and is the first place to look for it.
    "gemm_sol_clc_multicast_4_stage_pipeline",
    "gemm_sol_clc_multicast_4_stage_pipeline_large",
)


# Every conclusion in this repo's GEMM work so far is a **PTX** count: `regcount`
# parses `ptxas -v`, the opcode censuses that validate every ablation arm are PTX,
# and #148's between-MMA table and its `#[unroll]` fold were both diagnosed and
# verified in PTX. `ptxas` is a real optimizer sitting between all of that and the
# machine, and until this nothing here had ever checked one against the other.
#
# The immediate question is #150 and #151: two spin-loop costs counted in PTX, both
# of which `ptxas` has every reason to fold. But the instrument is general on
# purpose -- point it at a kernel and it prices every tight loop that kernel
# actually executes -- because "is our PTX count what the machine runs" is a
# question about the instrument set and not about those two issues.
#
# Loops are found by **backward branch** rather than by mnemonic: a branch whose
# target address is at or below its own closes a loop, and the instructions from
# that target through the branch are one iteration. That needs no knowledge of how
# an mbarrier wait is spelled in SASS, which is the part nobody here should be
# guessing at.
SASS_LOOP_CEILING = 32
"""Longest loop body printed. A spin loop is a handful of instructions; a K loop is
hundreds, and printing it would bury the thing being looked for."""

# Mnemonic fragments that mark a loop as a barrier spin, used only to *label* rows.
# The loop-finding above does not depend on them, so a Blackwell spelling nobody
# here predicted still gets found and counted -- it just prints without the label.
SASS_BARRIER_HINTS = ("BAR", "ARRIVE", "TRYWAIT", "MBAR")


def _sass(cubin: Path) -> str | None:
    """Disassemble `cubin`, or `None` with a reason printed.

    Tried in the order that gives the most readable output. Neither tool is used
    anywhere else in this file, so a missing one is a fact worth printing rather
    than an exception worth raising -- this runs inside `build`, which gates
    everything, and a disassembler that is absent must not take that gate red.
    """
    for tool, argv in (
        ("nvdisasm", ["nvdisasm", "-c", str(cubin)]),
        ("cuobjdump", ["cuobjdump", "-sass", str(cubin)]),
    ):
        found = subprocess.run(["which", tool], capture_output=True, text=True)
        if found.returncode != 0:
            print(f"  {tool}: not in the image")
            continue
        run = subprocess.run(argv, capture_output=True, text=True)
        if run.returncode != 0:
            print(f"  {tool} failed: {run.stderr.strip()[:300]}")
            continue
        print(f"  disassembled by {tool}")
        return run.stdout
    return None


def _sass_functions(text: str) -> dict[str, tuple[list[tuple[int, str, str]], dict[str, int]]]:
    """Per entry function, its instructions and its label definitions.

    Instructions are `(address, mnemonic, whole line)`; labels map a name to the
    index of the instruction it precedes. Both are needed because `nvdisasm -c`
    names branch targets as labels (`` `(.L_x_5) ``) while `cuobjdump -sass` names
    them as absolute addresses, and a loop finder that understands only one of the
    two silently reports no loops -- which is exactly how this was first written.
    """
    functions: dict[str, tuple[list[tuple[int, str, str]], dict[str, int]]] = {}
    current: tuple[list[tuple[int, str, str]], dict[str, int]] | None = None
    pending: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        named = re.search(r"(?:Function : |\.text\.)([A-Za-z_$][\w$]*)", stripped)
        if named:
            current = functions.setdefault(named.group(1), ([], {}))
            pending = []
            continue
        if current is None:
            continue
        label = re.match(r"^(\.L[\w$.]+):$", stripped)
        if label:
            pending.append(label.group(1))
            continue
        address = re.match(r"^/\*([0-9a-fA-F]+)\*/\s*(.*)$", stripped)
        if not address:
            continue
        body = address.group(2).strip()
        if not body:
            continue
        mnemonic = re.sub(r"^@!?\S+\s+", "", body).split()
        if not mnemonic:
            continue
        instructions, labels = current
        for name in pending:
            labels[name] = len(instructions)
        pending = []
        instructions.append((int(address.group(1), 16), mnemonic[0].rstrip(";"), body))
    return functions


def _sass_loops(
    instructions: list[tuple[int, str, str]], labels: dict[str, int]
) -> list[tuple[int, int, list[str]]]:
    """Every backward branch and the iteration it closes, shortest first.

    A branch resolving to an instruction at or before itself closes a loop, and the
    instructions from there through the branch are one iteration -- the quantity a
    PTX loop-body count is comparable to. Targets are resolved through both
    spellings a disassembler uses, label and absolute address.
    """
    by_address = {address: index for index, (address, _, _) in enumerate(instructions)}
    loops = []
    for index, (address, mnemonic, body) in enumerate(instructions):
        if not mnemonic.startswith(("BRA", "BRX", "JMP", "CBRA")):
            continue
        label = re.search(r"`\((\.L[\w$.]+)\)", body)
        literal = re.search(r"0x([0-9a-fA-F]+)", body)
        if label and label.group(1) in labels:
            start = labels[label.group(1)]
        elif literal and int(literal.group(1), 16) in by_address:
            start = by_address[int(literal.group(1), 16)]
        else:
            continue
        if start > index:
            continue
        loops.append((instructions[start][0], address, [line for _, _, line in instructions[start : index + 1]]))
    loops.sort(key=lambda loop: len(loop[2]))
    return loops


def _print_sass_loops(cubin: Path, kernels: tuple[str, ...]) -> None:
    """Tight loops per kernel, in SASS, with one iteration's instruction count.

    The count is the number to compare against a PTX poll-body count; the body is
    printed so a reader can see *why* it is that number rather than trusting it.
    """
    print(
        "\nSASS loops -- one iteration per row, found by backward branch and not by\n"
        "mnemonic. `insns` is instructions executed per iteration, which is the figure a\n"
        "PTX loop-body count is comparable to; `ptxas` sits between the two and this is\n"
        "the only place in this repo that checks one against the other. Bodies longer\n"
        f"than {SASS_LOOP_CEILING} instructions are counted and not printed."
    )
    text = _sass(cubin)
    if text is None:
        print("  no disassembler in the image; #150 and #151 stay PTX-only claims.")
        return
    functions = _sass_functions(text)
    if not functions:
        print("  the disassembler's output did not parse into entry functions.")
        return
    for name in kernels:
        if name not in functions:
            continue
        instructions, labels = functions[name]
        loops = _sass_loops(instructions, labels)
        tight = [loop for loop in loops if len(loop[2]) <= SASS_LOOP_CEILING]
        print(f"\n  {name}: {len(instructions)} instructions, {len(loops)} loops, "
              f"{len(tight)} of them tight")
        print(f"    {'loop at':>10}{'insns':>7}  kind")
        seen: set[int] = set()
        for target, branch, body in tight:
            if target in seen:
                continue
            seen.add(target)
            barrier = any(hint in line.upper() for line in body for hint in SASS_BARRIER_HINTS)
            print(f"    {hex(target):>10}{len(body):>7}  {'barrier spin' if barrier else 'loop'}"
                  f"  (closed at {hex(branch)})")
        for target, _, body in tight:
            if not any(hint in line.upper() for line in body for hint in SASS_BARRIER_HINTS):
                continue
            print(f"\n    {name} spin at {hex(target)} -- {len(body)} instructions an iteration")
            for line in body:
                print(f"      {line}")
            break


def _print_mma_stream() -> None:
    """The instruction span between the first and last `tcgen05.mma` of a kernel.

    Split on `.visible .entry` exactly as `_census` does, for the same reason: the
    bodies carry `ptx_asm!` verbatim and that text is the thing worth reading.
    `mma` belongs to one warp in a warp-specialized kernel and each warp's path is
    a separate branch, so the span between the first and last of them is that
    warp's K-loop body and nothing else.
    """
    seen: set[str] = set()
    for directory in (EXAMPLES_DIR, EXPERIMENTS_DIR):
        for ptx in sorted(Path(directory).rglob("*.ptx")):
            for chunk in ptx.read_text().split(".visible .entry ")[1:]:
                name = chunk.split("(", 1)[0].strip()
                if name not in MMA_STREAM_KERNELS or name in seen:
                    continue
                lines = [line.strip() for line in chunk.splitlines()]
                # Substring rather than prefix: a `tcgen05.mma` arrives inside a
                # braced `ptx_asm!` body with its predicate in front of it.
                # `_census` counts the same substring, so the two agree by
                # construction.
                issues = [i for i, line in enumerate(lines) if "tcgen05.mma" in line]
                if not issues:
                    continue
                seen.add(name)
                span = lines[issues[0] : issues[-1] + 1]
                code = [line for line in span if line and not line.startswith("//")]
                mma = [line for line in code if "tcgen05.mma" in line]
                between = len(code) - len(mma)
                print(
                    f"\n  {name}: {len(mma)} tcgen05.mma over {len(code)} instructions "
                    f"({between} between them, {between / len(mma):.1f} per issue)"
                )
                for line in span:
                    print(f"    {line}")


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
    # The three `gemm_sol` entries, here since #197 took the two 256-wide ones
    # from six warps to ten. Their residency is fixed by the shared plan and no
    # register count can raise it, so what this gate watches is the *other*
    # direction: `_ctas_by_registers` returning zero. At 320 threads that step
    # is 168 registers a thread, against the 82/83 they read today — and the
    # failure it would catch is a launch the driver refuses, not a slow one.
    # The narrow entry is 192 threads and its own row for the same reason every
    # copy of a gated kernel is a row: the geometries differ.
    ("gemm_sol_m256", "examples", "examples/src/gemm_sol.rs"),
    ("gemm_sol_m256_n128", "examples", "examples/src/gemm_sol.rs"),
    ("gemm_sol_m512", "examples", "examples/src/gemm_sol.rs"),
    # And `experiments/`' copies, which stopped being the same *file* when the
    # teaching copy lost its dials: `examples/src/gemm_sol.rs` is the shipped
    # configuration written out, `experiments/src/gemm_sol.rs` is the body every
    # ablation arm instantiates, and both emit these three entries. Two files
    # can drift where one `#[path]` could not, and `gemm_cg2_staged_x8x4`'s pair
    # is the precedent for what that costs: 80 registers against 96 on the same
    # source, from crate composition alone.
    ("gemm_sol_m256", "experiments", "experiments/src/gemm_sol.rs"),
    ("gemm_sol_m256_n128", "experiments", "experiments/src/gemm_sol.rs"),
    ("gemm_sol_m512", "experiments", "experiments/src/gemm_sol.rs"),
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


# --- the local-memory depot --------------------------------------------------
#
# A stage selection that makes LLVM *choose a value* among distinct shared
# symbols — a `match` yielding pointers or barrier references, one static per
# stage — lowers through a local-memory depot: the candidates are stored to
# `.local` and the winner reloaded, in the hot loop, every iteration. Measured
# on the real work-stealing gemm at 8192³ in cuda-learning's `barrier_bench`,
# three otherwise-identical kernels: the depot spelling reads 710 TFLOP/s
# against 857 for the same pipeline with stage selection as `base.add(i)`.
#
# This library's ring types (`SemaphoreRing`, `SharedTileRing`, `StoreRing`)
# and the single-symbol `SharedPlan` exist so that selection *is* `base.add(i)`
# and that class cannot arise. But `.local` has more than one way in, and the
# depot is LLVM's, so it is in the PTX **text** — a `.local .align` frame
# declaration and the `st.local`/`ld.local` that traffic it. That makes this
# census exactly disjoint from the spill columns above: a ptxas spill happens
# *after* PTX and never appears in it. Nothing in the tree emits `.local` by
# hand — no `ptx_asm!` block contains it — so any occurrence is the compiler's.
#
# **This began life as a gate and its first run is why it is a report.** On the
# tree at #165 every kernel either crate ships carries a frame: 256 B with
# 64 st.local / 16 ld.local for the whole staged-x8x4 family — the examples
# copy, the experiments copy, and every `gemm_sol_*` / `gemm_ws_*` rung beside
# them — 528 B on `gemm_cg2`, 1824 B on `flash_forward`, ~1540 B on
# `groupnorm_tile`. All of it was in ptxas' stack column all along, watched by
# nobody, because every audit (#152 most recently) read the *spill* columns,
# which are ptxas' own decisions and stayed zero. The 64-store/16-load shape —
# scalar stores, a quarter as many loads, so plausibly `.v4` reads of an array
# built element-wise — points at a dynamically-indexed fragment array rather
# than at stage selection, and it recurs across kernels that share `reg.rs`
# types. That guess attributed cleanly (#166): the frames were the register
# tiles themselves, memory-homed because a storage-walking loop past ~32 fp32
# columns stops unrolling, its indices stay runtime values, and SROA cannot
# split an aggregate behind a dynamic GEP. One rolled *reader* was enough to
# home a tile whose builder was fully unrolled — the staged family's 256 B
# frame was built scalar by the drain and re-read by the rolled `stmatrix`
# pack loop, inside the per-tile epilogue.
#
# The *mover* walks (`ldst`, `tmem`, `global`) now carry
# `__unroll_config::<0>()` and an inline-`const` bound — the backend's
# trip-count analysis sees `M / 16` as a runtime division otherwise, warns,
# and leaves the loop rolled — and every GEMM in both crates reads zero here.
# `RegTile::mask` carries it too (#184): a masking write under a
# data-dependent condition keeps the slot array addressable, so that walk
# homed a whole band per *outer* iteration rather than per masked one, and
# it cost oxide-train's tile classifier a 400 B frame off one `right_fill`.
# The `reg.rs` map/reduce walks deliberately do **not**: unrolling them
# blanket was measured (2026-08-03) at 255 registers and 208/208 ptxas
# spill st/ld on `flash_forward` against the 1058-store depot it replaced,
# which is #94's lesson again — a kernel can pay a frame and win. A map
# walk's frame is a tile that does not *fit*, and the fix there is not
# unrolling but streaming: `groupnorm_tile` held its whole [32, 128] band
# and its frame left when the band did (168 regs / 1536 B / 594 GB/s to
# 48 / 0 / 5996 — `docs/kernels/layernorm.md`). What remains on this table
# is `flash_forward`, whose accumulator band is the algorithm's own live
# state, and the arming condition is unchanged: flip the report to a raise
# when the shipped set reads zero, which now waits on that one kernel.
#
# #184 is also where this table's two substrates part company, and a reader
# diffing it should know which one they are reading. The `.local decl` and
# stack columns are *bytes a kernel homes*; `st.local`/`ld.local` are
# **static instruction counts**, and unrolling trades one rolled store for N
# spelled-out ones at constant offsets without changing what runs. So a row
# whose frame shrinks while its `st.local` grows has not got worse:
# `flash_forward` went 1824 B/774 st to 1568/690 (the score band left the
# frame), and `mask_probe_128_causal` — whose load walk is open-coded on
# purpose, so its band stays in the frame the mask now round-trips through —
# held 1536 B while its `st.local` went 556 to 664. Read the frame first.
#
# Substrings of the PTX text, counted per entry function the way the census
# counts opcodes. The frame declaration is one column and its traffic is two
# more, so a row says not just that a frame exists but whether anything still
# reads it.
DEPOT_PATTERNS = (
    (".local decl", ".local .align"),
    ("st.local", "st.local"),
    ("ld.local", "ld.local"),
)


def _local_census() -> dict[tuple[str, str], dict[str, int]]:
    """`.local` declarations and traffic per entry function, over every
    crate's PTX, keyed by `(crate, kernel)` like the occupancy gate."""
    counted: dict[tuple[str, str], dict[str, int]] = {}
    for _, directory in PTX_CRATES:
        for ptx in sorted(Path(directory).rglob("*.ptx")):
            crate = str(ptx.relative_to(PROJECT_DIR)).split("/", 1)[0]
            for chunk in ptx.read_text().split(".visible .entry ")[1:]:
                name = chunk.split("(", 1)[0].strip()
                counted[(crate, name)] = {
                    column: chunk.count(pattern) for column, pattern in DEPOT_PATTERNS
                }
    return counted


def _print_local_depot() -> None:
    """`.local` in the PTX text, per entry function — attributed (#166, the
    section comment) and still a report: the mover walks are fixed, the
    masking walk with them (#184), and the remainder is the map walks, whose
    blanket fix measured worse than the depot on `flash_forward`.

    The `shipped` column marks every kernel `examples/` emits plus the
    `GATED_KERNELS` rows: the kernels a launch gets by default, and the ones
    whose residency is already an argument. Those are the rows the armed gate
    will fail on. Probes will stay report-only either way — a probe exists to
    measure a spelling, and #94's history says a kernel can pay a frame and
    win."""
    counted = _local_census()
    trafficked = {key: counts for key, counts in counted.items() if any(counts.values())}
    shipped = {key for key in counted if key[0] == "examples"}
    shipped |= {(package, kernel) for kernel, package, _ in GATED_KERNELS}

    print("\n  the local-memory depot — `.local` in the PTX text, per entry function.")
    print("  A ptxas spill never appears in PTX, so this is LLVM's local memory and")
    print("  nothing else; the stack column above is where it has been hiding. Diff")
    print("  this table across a change the way the register table is diffed:")
    if not trafficked:
        print(f"    zero everywhere: none of the {len(counted)} entry functions declare")
        print("    or touch `.local`.")
        return

    columns = [column for column, _ in DEPOT_PATTERNS]
    print(f"    {'kernel':<46}" + "".join(f"{column:>13}" for column in columns) + f"{'':>9}")
    for (crate, name), counts in sorted(trafficked.items()):
        row = "".join(f"{counts[column]:>13}" for column in columns)
        print(f"    {crate + '/' + name:<46}{row}{'shipped' if (crate, name) in shipped else '':>9}")
    carrying = sum(1 for key in trafficked if key in shipped)
    print(f"    ({len(counted) - len(trafficked)} further entry functions carry none.)")
    print(f"\n  {carrying} of {len(shipped)} shipped kernels carry local memory.")


# --- the jump-table census ------------------------------------------------
#
# `brx.idx` is a computed branch through a `.branchtargets` label list — LLVM's
# jump table, and the construct a downstream consumer's module died on: ferro
# #219 put four of them into oxide-train's `gemm_tcgen05_bf16_optimized` and the
# driver's PTX JIT refused the module at `cuModuleLoadData` with
# `DriverError(218)`, before any kernel was looked up, while offline `ptxas`
# assembled the same text and reported unchanged register counts (#225,
# oxide-train #127). Every check on this page is `ptxas`, so every check on this
# page was green.
#
# **This is a report and not a gate, and the day-one census is why.** Measured
# at bda3329: 55 of 318 entry functions carry one, and at 3ae07a8 — the commit
# before #219 — 51 of 309 carry one, *the same set*, plus the four kernels that
# did not exist yet. Every count is identical. So the tables are not #219's in
# this tree; they are what the four-rung access ladder has always lowered to
# here, and they load: `device_tests` (57/57), `examples`, `kittens-experiments
# -- check` and `bench --case gemm-sol` all run on a B200 on driver 580.95.05,
# which is the driver in the report. `device-tests`' `shared_drain_quad` was
# built to put the consumer's exact shape — four tables in one entry — in a
# module the harness loads, and that loads too. So a `brx` is not sufficient to
# fail a JIT load, and a gate here would fail every run of a tree that works.
#
# What the census is for is the thing nobody had: a **number to diff**. The
# regression downstream was one instruction class appearing in one module, and
# it took a consumer's bisect to find because ferro measured registers, opcodes
# and `.local` and never this. The column that moves is the finding.
#
# Where it comes from, so a row that moves can be read: LLVM turns a switch into
# a jump table at four cases and into branches at three
# (`MinimumJumpTableEntries`, checked against `llc -mtriple=nvptx64` at
# sm_100a). The bf16 drains have four live rungs — 16, 8, 4 and 2 bytes — and
# table; the fp32 drains have three, because `access_width` cannot return 2 for
# a 4-byte element, and they never appear here. That pair is the in-tree
# control. An exhaustive `match` over `AccessWidth` also *names* every rung, so
# it re-materialises one a caller had constant-folded away — which is how a
# consumer whose rungs folded to three acquired four (#225, `global`).
#
# It arms into a gate when the shipped set reads zero.
JUMP_TABLE_PATTERN = "brx"


def _jump_table_census() -> dict[tuple[str, str], int]:
    """`brx` per entry function, over every crate's PTX, keyed like the depot
    census. Counted in the PTX *text*: nothing in the tree writes `brx` by hand,
    so an occurrence is the compiler's jump table and nothing else. Three per
    table — the list's label, the `brx.idx`, and its reference to the label."""
    counted: dict[tuple[str, str], int] = {}
    for _, directory in PTX_CRATES:
        for ptx in sorted(Path(directory).rglob("*.ptx")):
            crate = str(ptx.relative_to(PROJECT_DIR)).split("/", 1)[0]
            text = ptx.read_text()
            for chunk in text.split(".visible .entry ")[1:]:
                name = chunk.split("(", 1)[0].strip()
                counted[(crate, name)] = chunk.count(JUMP_TABLE_PATTERN)
            # Whatever sits outside an entry function — a `.func` body an entry
            # calls — is nobody's row above and would hide there.
            outside = text.split(".visible .entry ")[0].count(JUMP_TABLE_PATTERN)
            if outside:
                counted[(crate, f"{ptx.name} (outside any entry)")] = outside
    return counted


def _jump_table_excerpt(lines_before: int = 22) -> str:
    """The first table in the tree, with the code that reaches it.

    A count says a module carries one; the excerpt says which dispatch built it.
    The `.branchtargets` list is the arm count, and the `selp` chain above it is
    the ladder LLVM speculated into a value so it could index on it."""
    for _, directory in PTX_CRATES:
        for ptx in sorted(Path(directory).rglob("*.ptx")):
            lines = ptx.read_text().splitlines()
            for index, line in enumerate(lines):
                if "brx.idx" not in line:
                    continue
                start = max(0, index - lines_before)
                body = "\n".join(f"      {text}" for text in lines[start : index + 1])
                return f"\n    the first of them, in {ptx.name}:\n{body}\n"
    return ""


def _print_jump_tables() -> None:
    """`brx` in the PTX text, per entry function — the substrate `ptxas` cannot
    see and the one a runtime compiler can refuse.

    A report, for the reason and with the numbers in the section comment above.
    Diff it across a change the way the register table is diffed: a kernel that
    gains a table has had a dispatch materialised, and a consumer that JIT-loads
    it is the one who finds out."""
    counted = _jump_table_census()
    tabled = {key: count for key, count in counted.items() if count}
    print(
        f"\n  jump tables — `{JUMP_TABLE_PATTERN}` in the PTX text, per entry function.\n"
        "  LLVM tables a switch at four cases and branches at three, so this column is\n"
        "  the drain dispatch's arm count showing through. `ptxas` accepts a table and a\n"
        "  driver's JIT may not (#225): diff it, do not read it."
    )
    if not tabled:
        print(f"    zero everywhere: none of the {len(counted)} entry functions carry one.")
        print("    That is the census this report arms into a gate on — see modal_app.py.")
        return

    shipped = {key for key in counted if key[0] == "examples"}
    shipped |= {(package, kernel) for kernel, package, _ in GATED_KERNELS}
    for (crate, name), count in sorted(tabled.items()):
        mark = "shipped" if (crate, name) in shipped else ""
        print(f"    {crate + '/' + name:<52}{count:>6}{mark:>9}")
    carrying = sum(1 for key in tabled if key in shipped)
    print(f"\n  {len(tabled)} of {len(counted)} entry functions carry one, {carrying} of them shipped.")
    print(_jump_table_excerpt(), end="")


@app.function(cpu=8, timeout=CHECKING, volumes=CACHE)
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

    The fifth reads a different substrate: the PTX *text*, where LLVM's
    local-memory depot lives and a ptxas spill does not. It is a report rather
    than a gate, because its first run found every shipped kernel carrying one
    — see `_print_local_depot`'s section comment for the day-one census, the
    idiom the check was built against, and the condition under which it arms.

    The sixth reads that same substrate for the other thing a runtime compiler
    can refuse and `ptxas` cannot see: `brx` jump tables, which is what a
    four-case dispatch lowers to and what cost a consumer its module at
    `cuModuleLoadData` (#225). Also a report, and its section comment carries
    the day-one census, the driver it was measured on and the arming
    condition.

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
    _print_mma_stream()
    _print_ladder(measured)
    _print_timed_twins(measured)
    _check_occupancy_step(measured)
    _print_local_depot()
    _print_jump_tables()

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
