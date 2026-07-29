# CI

Three tiers, ordered by what they cost to run. The rule is that the cheapest
tier gates everything and never depends on a credential, and each tier above it
buys something the one below genuinely cannot see.

| | Tier 1 `ci.yml` | Tier 2 `cuda.yml` | Tier 3 `gpu.yml` |
| --- | --- | --- | --- |
| Needs | nothing | CUDA toolkit | B200 |
| Runs on | GitHub runner | Modal, CPU | Modal, B200 |
| Secrets | none | Modal token | Modal token |
| Cost | free | Modal CPU-minutes | ~10 GPU-minutes a run |
| Trigger | every push to main, every PR | push to main, same-repo PRs, manual | labelled PR, `src/`+`device-tests/` on main, manual |

## Tier 1 — the device surface

`fmt`, lockfile freshness, `clippy --all-targets -- -D warnings`, `test`, and
`doc --no-deps` with `RUSTDOCFLAGS=-D warnings`.

**Format with `scripts/fmt`, not `cargo fmt`.** This tier checks three
manifests; a root `cargo fmt` reaches only the library, because `device-tests/`
and `examples/` are separate workspaces. That gap has turned this gate red three
times (#97, #104, #109) on changes that were otherwise correct — the code was
fine and a signature wanted to be on one line. `scripts/fmt --check` is what CI
runs; `scripts/fmt` fixes it.

The library's default feature set is device-only and `cuda-device` is ordinary
Rust, so all of that runs on a stock runner with no toolkit. `device-tests/` and
`examples/` cannot be compiled here — both pull `cuda-core` -> `cuda-bindings`,
whose build script wants `cuda.h` — but formatting and dependency resolution
need neither, so both crates get those two checks for free.

The `host`-gated links in `global`'s docs are allowed to dangle here (see the
`cfg_attr` at the top of `src/lib.rs`); tier 2 is where they must resolve.

## Tier 2 — the host feature, device codegen without a device, and the register gate

`modal run modal_app.py::build`: clippy across all three crates with
`--features host`, the host tests, `cargo doc --features host`, and a real
`cargo oxide build --arch sm_100a` of both kernel crates against the CUDA driver
stub.

That last part is why this tier is worth its money. A post-monomorphization
`const { assert!(..) }` in a tile shape fires at *codegen*. `cargo check` cannot
see it; only an actual device build can. This is the highest value-per-minute
job here, and it needs no GPU.

A second job runs `modal run modal_app.py::regcount`: the `ptxas -v` table, the
shape ladder, #63's timed twins, and #95's **occupancy-step gate**, which fails
when a named kernel's register count would cost it a CTA per SM. That entry
point predates this job by a long way and until #95 ran in no workflow at all,
so a kernel could cross the step on any PR and nothing would say so. It is a
separate job because it goes red for a different reason than `build` does — a
register count rather than a compile — and that is worth seeing on the check
list rather than in a log. It builds both kernel crates in its own container,
so tier 2 now costs roughly twice what it did; still CPU-minutes.

### Why Modal rather than a CUDA container on a GitHub runner

A `nvidia/cuda:13.0.0-devel-ubuntu24.04` job container would supply `cuda.h`
cheaply — but the toolkit is the small half of what this needs. The rest is LLVM
21 from apt.llvm.org, the pinned nightly with `rust-src`/`rustc-dev`/
`llvm-tools`, `cargo-oxide` installed from git, and a bootstrap build of the
cuda-oxide rustc codegen backend. On Modal that is a content-addressed image
layer that already exists. On a fresh GitHub runner it is a cold build every
run, unless we publish our own image to GHCR — which is a second definition of
the toolchain, maintained beside `modal_app.py` and free to drift from it.

So: one definition, in `modal_app.py`, invoked identically from a laptop and
from CI. The price is a credential, and therefore no coverage on fork PRs.

Known inefficiency: each run compiles the pinned dependencies from scratch,
because the image's warm cargo caches were populated for `/opt/warmup` and the
project builds into a fresh `target/`. A Modal Volume mounted at the project's
target directory would cut most of that. Not done here.

## Tier 3 — the harness on a B200

`modal run modal_app.py::device_tests`. The ~20 cases take about 1.5 minutes of
GPU time, but the billed window is the whole container — image pull, harness
build, run — so budget on the order of ten GPU-minutes per invocation.

It runs on a PR only when the PR carries the **`gpu-ci`** label (and on every
later push while the label is on; remove the label to stop the spend), and
automatically on pushes to main that touch `src/**` or `device-tests/**`.

That path filter is deliberately the whole of `src/`, not the four primitive
files. A hand-maintained list rots — `mma.rs` and `sync.rs` are device
primitives too — and it rots in the direction of *not* running a correctness
gate, which is the wrong way for it to fail. Restricting the trigger to
push-to-main bounds the cost far more effectively than narrowing the paths does.

### Running it by hand

```
gh workflow run gpu.yml                          # current default branch
gh workflow run gpu.yml -f ref=refs/pull/17/merge # a specific PR
```

`cuda.yml` takes the same `ref` input.

### Running Modal by hand, and the two ways it costs you money

Use the wrapper rather than `modal run` directly:

```
scripts/modal-run device_tests
scripts/modal-run bench --case gemm-depth
```

A GPU function bills the whole container, including the minutes it spends
compiling before a kernel launches. #93 fixed the worst of that by giving the
GPU entry points a whole CPU — the build went from ~44 minutes to ~7.5 — and
`timeout=` now caps each entry point at what its work actually takes rather than
the flat ninety minutes every one of them used to carry.

That leaves one hole, and the wrapper is what plugs it. **A container can wedge
before it reaches Python**, in which case `timeout=` is the only thing that ever
stops it and nothing inside `modal_app.py` runs to notice: one died after
twenty-six minutes having printed fifteen lines of NVIDIA banner and no
`$ <command>` line at all. From the outside that is indistinguishable from a
cold build. The tell is that a healthy run emits `Compiling …` within a couple
of minutes of the banner, and `scripts/modal-run` watches for exactly that,
stops the app when it does not arrive, and exits 3. It exits 2 on the workspace
spend limit, which fails every entry point at once and reads exactly like a
broken diff.

If you do run `modal run` by hand, check `modal app list` afterwards and confirm
your app is `stopped`. A live app you have stopped paying attention to is still
a live app.

## Pull requests from forks

Fork PRs get no secrets, so tiers 2 and 3 are skipped rather than failed —
their jobs test `head.repo.full_name == github.repository` up front.

There is no automatic way around that, and `pull_request_target` is not used
anywhere here: it would run a fork's code with our credential in the
environment on the strength of a PR being opened. The route is manual. A
maintainer reads the diff, then dispatches against the merge ref:

```
gh workflow run cuda.yml -f ref=refs/pull/<N>/merge
```

The workflow *definition* still comes from the default branch; only the tree
comes from the fork. The maintainer's reading of the diff is the entire
security boundary, which is the point of making it an explicit act.

## Caching

Tier 1 uses `Swatinem/rust-cache`. The cuda-oxide git dependency is pinned to
one revision and is most of the build; keyed on the lockfiles it is fetched and
compiled once and restored until the pin moves.

Tiers 2 and 3 cache nothing on the GitHub side — there is nothing there worth
caching. The expensive artifacts are Modal image layers, content-addressed and
rebuilt only when `modal_app.py`'s recipe changes.

## Secrets

`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, from `modal token new`. Until they are
set on the repo, tiers 2 and 3 fail on their first step with an error naming
them — the `secrets` context is not readable from a job-level `if`, so an
unconfigured repo cannot be turned into a skip, and a check that quietly passes
when it did not run is worse than a red one. Tier 1 is unaffected and gates the
repo from the moment this merges.
