"""Build and run ferro-kittens' device test harness on a Modal B200.

cuda-oxide is a rustc codegen backend (Rust -> PTX). The only place the full
toolchain can live is a Linux box with an NVIDIA GPU + CUDA 13 + LLVM 21, so we
bake all of that into a Modal image once and reuse it. The image recipe below is
a verbatim copy of rust-trainer's; Modal images are content-addressed, so an
identical recipe reuses that project's cached layers instead of rebuilding the
codegen backend from scratch.

Local usage:
    modal run modal_app.py::build     # host tests + a CPU-only device build
    modal run modal_app.py::regcount  # ptxas -v register/spill table + the
                                      # shape ladder and its cliff, no GPU
    modal run modal_app.py            # the device tests, on a B200
    modal run modal_app.py::examples  # the examples crate's kernels, on a B200
    modal run modal_app.py::bench     # those kernels timed at several sizes
    modal run modal_app.py::ladder_bench
                                      # four rungs of that ladder on a clock:
                                      # is a streamed band actually slower?
    modal run modal_app.py::doctor    # env / GPU sanity check
"""

import re
import subprocess
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
    gate, which is a finding and not an error.

    `layernorm` was the second feature here until #3. Both of that file's
    kernels are in the default build now, so the feature is gone and `build`
    is what holds them -- an empty gap list stops being a finding once there is
    no gate left for it to be about."""
    for feature in ("flash",):
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


# `cpu=8` because on a cold container the *build* is a long pole in its own
# right — the harness' dependency tree is ten-odd minutes of compilation and
# the B200 is billed through all of it. The timeout covers that plus the sweep:
# twenty rungs, each verified at three configurations and then timed in five
# rounds at two grids, is a further half hour. 1800 s does not fit it, and a
# run that dies between the third shape and the fourth has spent the GPU and
# answered three quarters of the question.
@app.function(gpu=DEFAULT_GPU, cpu=8, timeout=5400)
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


# Crates that emit PTX, as `(package, directory)`. The harness carries the
# probes and the ladder; the examples carry the kernels a *launch* configures,
# and their register and shared-memory counts are what an occupancy argument
# about a real kernel has to be made of (#47).
PTX_CRATES = (("device-tests", HARNESS_DIR), ("kittens-examples", EXAMPLES_DIR))


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


@app.function(cpu=8, timeout=1800)
def regcount(arch: str = "sm_100a", label: str = "", determinism: bool = False) -> None:
    """Register pressure of every kernel the harness and the examples emit,
    from `ptxas -v`.

    Register count is *the* performance number for this library — a tile
    abstraction that costs registers has failed at the thing it exists for, and
    "ptxas says otherwise" is the escape hatch the README promises. `ptxas` is a
    host compiler, so this needs no GPU: build both PTX-emitting crates, feed
    the emitted PTX back through `ptxas -v`, and print a sorted table. Run it
    before and after a change and diff the two.

    The examples are here because a register count is only half an occupancy
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
    _print_ladder(measured)
    _print_timed_twins(measured)

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
