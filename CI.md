# CI

Three tiers, ordered by what they cost to run. The rule is that the cheapest
tier gates everything and never depends on a credential, and each tier above it
buys something the one below genuinely cannot see.

| | Tier 1 `ci.yml` | Tier 2 `cuda.yml` | Tier 3 `gpu.yml` |
| --- | --- | --- | --- |
| Needs | nothing | CUDA toolkit | B200 |
| Runs on | GitHub runner | Modal, CPU | Modal, B200 |
| Secrets | none | Modal token | Modal token |
| Cost | free | Modal CPU-minutes | ~1.5 GPU-minutes a run |
| Trigger | every push to main, every PR | push to main, same-repo PRs, manual | labelled PR, `src/`+`device-tests/` on main, manual |

## Tier 1 — the device surface

`fmt`, lockfile freshness, `clippy --all-targets -- -D warnings`, `test`, and
`doc --no-deps` with `RUSTDOCFLAGS=-D warnings`.

**Format with `scripts/fmt`, not `cargo fmt`.** This tier checks four
manifests; a root `cargo fmt` reaches only the library, because `device-tests/`,
`examples/` and `experiments/` are separate workspaces. That gap has turned this
gate red three times (#97, #104, #109) on changes that were otherwise correct —
the code was fine and a signature wanted to be on one line.
`scripts/fmt --check` is what CI runs; `scripts/fmt` fixes it.

The library's default feature set is device-only and `cuda-device` is ordinary
Rust, so all of that runs on a stock runner with no toolkit. The three kernel
crates cannot be compiled here — all pull `cuda-core` -> `cuda-bindings`, whose
build script wants `cuda.h` — but formatting and dependency resolution need
neither, so each gets those two checks for free.

`experiments/` compiles two of `examples/`' kernel files and its clock through
`#[path]`, so rustfmt visits those files twice. That is idempotent, and it is
why `scripts/fmt` is a loop over manifests rather than anything cleverer.

**This tier does not check intra-doc links at all** — not `global`'s, not any.
The `cfg_attr` at the top of `src/lib.rs` was added so the `host`-gated links
into `global` could dangle without the feature, but `allow` at crate root
applies to the whole crate: off `--features host`, *every* broken link in
`src/` is suppressed. So a green `doc --no-deps` here says nothing about links,
and tier 2's `cargo doc --features host` is the only gate on them. A bad
`[crate::sync::depth_needed]` (the item is `sync::handoff::depth_needed`)
passed a local `RUSTDOCFLAGS=-D warnings` doc run that way, and tier 2 on #164
is what caught it.

To check links before pushing, comment that `cfg_attr` line out and run
`cargo doc --no-deps`. Every error naming `GlobalLayout` is a legitimate
host-gated dangle; anything else is real. Restore the line before committing.

## Tier 2 — the host feature, device codegen without a device, and the register gate

`scripts/modal-run build`: clippy across all four crates, the library's host
tests, `cargo doc --features host`, and a real `cargo oxide build --arch
sm_100a` of all three kernel crates against the CUDA driver stub — plus a second
build of `experiments/` with `--features cublas`, which is the step that proves
FFI through cargo-oxide links at all.

`experiments/` is the expensive half of that and the half most worth having:
about a third of its GEMM entry points compute a deliberately wrong `C` to
isolate one term, so they are on no correctness gate and this codegen is the
only gate they have.

That last part is why this tier is worth its money. A post-monomorphization
`const { assert!(..) }` in a tile shape fires at *codegen*. `cargo check` cannot
see it; only an actual device build can. This is the highest value-per-minute
job here, and it needs no GPU.

A second job runs `scripts/modal-run regcount`: the `ptxas -v` table, the
shape ladder, #63's timed twins, #95's **occupancy-step gate**, which fails
when a named kernel's register count would cost it a CTA per SM, and the
**local-memory depot census** — `.local` in the PTX text, which is upstream of
ptxas and disjoint from the spill columns (a ptxas spill happens after PTX and
never appears in it). The census was built as a gate against the depot a
value-select among per-stage shared symbols lowers to (710 vs 857 TFLOP/s on
otherwise-identical kernels, cuda-learning `kernels/barrier_bench`), and its
first run is why it reports instead: every shipped kernel already carries a
frame nobody had attributed, sitting in the stack column no audit was reading.
It arms into a gate when the shipped set reads zero — the section comment in
`modal_app.py` carries the day-one census and the argument. That entry
point predates this job by a long way and until #95 ran in no workflow at all,
so a kernel could cross the step on any PR and nothing would say so. It is a
separate job because it goes red for a different reason than `build` does — a
register count rather than a compile — and that is worth seeing on the check
list rather than in a log. It builds all three kernel crates in its own
container, so tier 2 now costs roughly twice what it did; still CPU-minutes.

`examples/` and `experiments/` both emit `gemm_cg2_staged_x8x4` and the three
`gemm_sol` entries — the same kernels, shipped by one and kept as every A/B's
control arm by the other — and `_measure` keys on `(ptx file, kernel)`, so the
table carries both rows. Two identical rows is the claim that the teaching copy
moved no instruction; a difference between them is a finding.

For `gemm_sol` that pair is now the *only* thing holding the two copies
together. The teaching copy ships one configuration with the consts baked in and
the arms deleted; the notebook copy keeps `ABLATE`, `DRAIN`, `WATCH`, `FOLD` and
the warpgroup count as parameters, because an arm nobody can build is a verdict
nobody can re-run. They were one file behind a `#[path]` until then, and a
`#[path]` cannot serve a teaching copy that is supposed to have no dials in it.

### What tier 2 could not see, and the census that was added for it

`ptxas` is a host assembler and the driver's PTX JIT is a different compiler.
They disagree, and this tier is entirely the first one: every register count,
every spill column and the occupancy gate come out of `ptxas -v`. A module it
assembles cleanly can still be **refused at `cuModuleLoadData` with
`DriverError(218)`**, and when that happened it happened downstream —
oxide-train's pin bump (#225, oxide-train #127) died at the first module load
with every ferro check green, and the opcode diff that found it was run in the
consumer's repo because ferro had no number that moved.

`regcount` counts `brx` per entry function now, beside the local-memory depot
and for the same reason: a substrate `ptxas` never reports. It is a **report**,
and its section comment in `modal_app.py` carries the census that made it one —
the tables were already there at #218, the same set, and they load. What was
missing was never a gate. It was a column to diff.

The device tier's half of this is `device-tests`' `shared_drain_quad`, which
puts the consumer's shape — four of the ladder's dispatches in one entry — in
a module the harness JIT-loads on every run, at no extra container and no extra
device time. Module load is whole-module, so tier 3 exercises it by starting.

Both jobs go through **`scripts/modal-run`**, so the wedge watchdog and the
completion sentinel cover this tier — see the wrapper's own section below.

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

Each run used to compile the pinned dependencies from scratch, because the
image's warm cargo caches were populated for `/opt/warmup` and the project built
into a fresh `target/`. That is now the `ferro-kittens-cargo` volume — see
[Spending policy](#spending-policy) below.

## Tier 3 — the harness on a B200

`scripts/modal-run device_tests`. The billed window is the whole container —
image pull, harness build, run — and at 41f9e57 that measured **83.5 s wall, of
which 53.3 s was `cargo`**: 57 cases in about 20 s of B200, behind a compile
that costs two and a half times as much. This page used to budget ten
GPU-minutes for it; the number is an eighth of that, and the shape of it is the
argument for rules 1 and 2 of the [spending policy](#spending-policy) — the
harness is cheap and the container is not.

`scripts/modal-run examples` is the other correctness gate here, and it runs
**two** binaries since the crates split: `kittens-examples` for the four
teaching kernels and `kittens-experiments -- check` for every rung and probe
that computes a GEMM. The gate is what it always was — both check sizes, all
three traversal widths, both schedulers where a rung has both — and naming both
binaries in `modal_app.py` rather than in one of them is what keeps it that
way.

Through the wrapper for the reason the wrapper exists: the container that
wedged for twenty-six minutes having printed nothing but a banner was a B200
running this entry point. This is the tier where the watchdog is worth the most
per firing, and until it was wired here it covered only runs started by hand.

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

## Spending policy

Three rules, in the order they save the most. All three come out of one
observation: a Modal function is billed for its whole **container**, and on a
GPU entry point most of that container is usually a Rust compiler.

### 1. Build on CPU, run on GPU

**No compilation belongs on a B200 that can happen anywhere else.** The kernels
never needed a device to build in the first place: `cargo oxide build` emits
**PTX**, the driver's JIT turns it into SASS at load, and the link step is
satisfied by the CUDA toolkit's driver *stub* (`LD_LIBRARY_PATH=…/lib64/stubs`,
which `modal_app.py` calls `STUB_ENV`). A device is needed to *run* a kernel and
never to produce one. That is why tier 2 does a full `cargo oxide build --arch
sm_100a` of all three kernel crates on `cpu=8` — and it always has, so there was
no GPU-held build tier here to reclaim.

What was left was the dependency tree, recompiled from scratch by every run
because the image's warm caches were populated for `/opt/warmup` and the project
built into a fresh `target/`. That is now the **`ferro-kittens-cargo` volume**,
mounted at `/cache`, one directory per crate under `/cache/target/`;
`_cached_target` picks the right one from the working directory `_run` was
given, and mounting the volume is the only switch — an entry point that mounts
it builds onto it, one that does not builds where it always did.

Nothing else moves. cargo-oxide writes the emitted PTX to the *working*
directory rather than under `target/`, so `regcount`'s `_measure`, the opcode
census and the local-memory census all read the same files from the same places.
`regcount` is the check on that, and it is the reason to believe this is a
placement change: run against 41f9e57 and against this tree, its output is
**identical over all 1650 lines** — every register count, the shape ladder, the
timed twins, the occupancy-step gate, the opcode census and the local-memory
depot census. A cache that moved a number would have moved one of those.

The registry is deliberately **not** on the volume: every dependency is already
baked into `/root/.cargo` by the image's warmup layer, which is
content-addressed and therefore free, and a second copy on a network filesystem
would be slower than the one in the image.

#### The volume served a stale build until `_freshen`, and rustdoc is what said so

**A cache volume plus a Modal mount can make a gate pass a tree it never
compiled.** Found on the launch-watchdog PR, on the first `build` after adding a
module to `src/`:

| | `clippy --features host` | unit tests it then ran |
| --- | ---: | ---: |
| volume warm, no `_freshen` | 1.05 s, **compiled nothing** | 127 — the count *before* the new module |
| volume emptied | 13.5 s | 129 |
| volume warm, `_freshen` | 4.4 s | 129 |

The mechanism is cargo's freshness check, which for a path crate is *mtime
against the fingerprint*. Modal's mounts do not hand the container mtimes newer
than artifacts already sitting on the volume, so an edited file can look older
than the object built from its predecessor — and cargo is right to believe what
it was told. Neither cargo nor the volume is at fault; the pairing is.

The tell was `rustdoc`, the one step that reads the sources unconditionally: it
found the new module's doctest and failed it against a `libkittens.rlib` that
had no such module. Without a doctest in the new module, that run would have
been **green on a tree it had not built**.

`_freshen` touches every `.rs`, `.toml` and `.lock` under the project once per
container, before the first command. The project's own crates are then rebuilt
every invocation — which is what anyone reading a gate assumes — and the
dependency tree, which is what the volume is actually worth, is untouched
because none of it lives under `PROJECT_DIR`. Measured on this tree, step time:

| | steps |
| --- | ---: |
| volume empty (cold) | 384.3 s |
| volume warm, `_freshen` | **295.2 s** |

So the volume is still worth 89 s a run, 23%, and #226's warm figure of 291.8 s
survives contact with a build that is actually happening — its saving was in the
dependencies all along, exactly as it argued. What did not survive is the
*guarantee*: any "identical output" comparison taken across a warm volume before
this fix was comparing two runs of possibly the same artifacts.

#### The volume is mounted by the CPU entry points and not the GPU ones

That is a measurement, not a preference, and it is worth keeping written down so
the next person does not redo it. The plan was the obvious one: fill the cache
on `build`'s eight CPUs, then let tier 3 skip the compile it was being billed
for. Measured on `device_tests` at 41f9e57, cache warm from a preceding `build`:

| | dependency crates compiled | `cargo` in the container |
| --- | ---: | ---: |
| no volume | 58 | 53.3 s |
| volume, warm | **1** | 66.0 s |
| volume, warm, run again | **1** | 60.0 s |

The cache does exactly what it claims and the container still gets *slower*,
because the 58 dependencies were never the cost. They compile in parallel on
`cpu=8` and hide behind the long pole, which is the crate's own device codegen —
and **cargo-oxide redoes that on every invocation**, cache or no cache: two
consecutive `run`s of an unchanged tree both recompiled `device-tests` in full.
So on a GPU function the volume trades work that was already free for artifact
I/O over a network filesystem, and loses. (`regcount --determinism` has been
saying the same thing from the other side for a long time: a second pass "costs
a second build of those two crates — their dependencies stay compiled".)

The consequence is the whole of the GPU spending story here: **a GPU
container's build cannot be cached away, so the only lever on it is fewer
containers.** That is rule 2.

#### What it bought

`build`, at 41f9e57, wall clock from `scripts/modal-run` and step time from
`_run`'s own elapsed lines:

| tier 2 `build` (CPU, `cpu=8`) | wall | steps | of which `clippy`×4 |
| --- | ---: | ---: | ---: |
| before, no volume | 470.8 s | 457.1 s | 67.6 s |
| after, volume cold (first fill) | 364.8 s | 352.8 s | 53.8 s |
| after, volume warm | **291.8 s** | **281.1 s** | **17.3 s** |

**Warm saves 176 s of step time per run, 38%**, and cold costs nothing over the
baseline (the two are inside the Modal CPU pool's own run-to-run spread). The
saving is concentrated exactly where the theory says: the four `clippy` steps
collapse from 67.6 s to 17.3 s, while the five `cargo oxide build` steps barely
move — they are codegen, and codegen is redone every time.

Tier 2 runs two of these jobs (`build` and `regcount`) on every push to main and
every same-repo PR, so the volume is worth roughly **350 CPU-seconds per push**,
and **zero GPU-seconds**, which is the honest headline.

The rule that falls out: **if a step can go red without a GPU, it must run
without one.** Compile errors, lints, `ptxas` register counts, the occupancy
gate, the cuBLASLt ABI assertion, upstream's PTX — all of that is tier 2 and
none of it may migrate onto tier 3 for convenience.

### 2. Batch arms into one container

An arm's *device* time is usually a small part of what it costs. `ws_bench`'s
own docstring prices its sweep at about 90 s of B200 inside a container that
also pulls an image and builds a crate; `device_tests` runs its 57 cases in
about 20 s of the 83.5 s above. So the question to ask of two tables is not
"what do they cost" but "how many containers do they need".

Measured at 41f9e57, three arms — `bench:softmax`, `bench:layernorm`,
`device-tests` — as three runs and then as one `session`:

| | wall | note |
| --- | ---: | --- |
| `bench --case softmax` | 79.7 s | 60 s of it cargo |
| `bench --case layernorm` | 97.5 s | 72 s of it cargo |
| `device_tests` | 81.0 s | 52 s of it cargo |
| three separate runs | **258.2 s** | three image pulls, three builds |
| `session --arms …` (same three) | **161.2 s** | one image pull, one dependency compile |

**97 GPU-seconds saved on three arms, 38%.** What a session saves is the image
pull and container start, once instead of three times, plus the *dependency*
compile for any arm that follows one in the same crate — `bench:layernorm` built
in 28.6 s behind `bench:softmax`'s 40.6 s against 72 s from cold. What it does
**not** save is the crate's own device codegen, which cargo-oxide redoes for
every invocation even inside one container. Same finding as rule 1's, from the
other direction.

Two mechanisms, one per level:

* **Inside a sweep**, `bench --case` already takes a comma-separated list, and
  `bench.rs`'s `selection` doc says why: a kernel, the kernel it is a port of
  and their shared baseline are three tables that must be read against each
  other, so they should be taken on one device on one day. Use it — `--case
  gemm,gemm-depth,softmax` is one container.
* **Across sweeps**, `session --arms` runs any comma-separated set of the named
  arms (`clc`, `ws`, `ws-shallow`, `sol-ablate`, `upstream`, `examples`,
  `device-tests`, `ladder`, and `bench:<case>` for anything `bench` narrows to):

  ```
  scripts/modal-run session --arms clc,ws,bench:gemm-sol
  ```

  The arms are one table in `modal_app.py`, and the individual entry points are
  thin wrappers over the same table, so a session and N separate `modal run`s
  issue **byte-identical** commands. Batching cannot change a number — it can
  only change how many containers paid for it.

  It also buys the control that #98 and #118 keep paying for by hand: 2.9% of
  drift between two runs of the same tree, and a 16384³ row that moved 3%
  between two sessions. Arms taken in one container share a device, a driver, a
  clock and a cuBLASLt.

**Every sweep has an arm now, including the two that used to be barred.**
`bench --case sol-k` and `profile` were kept out of the table because a launch
that hangs takes every row after it — which is how `sol-k` was found (#146:
`4096x4096x1024` printing nothing for 1200 s until the wrapper stopped the
container). Two things had to become true first, and both now are:

1. **A launch that does not return costs one row, not a container.** Every host
   wait in the three kernel crates goes through `kittens::watchdog`, which polls
   the event behind the work rather than blocking on it and ends the process
   past 30 s naming the row the sweep last announced. See
   `docs/library/watchdog.md`.
2. **A failed arm no longer throws the rest of the session away.** Each arm was
   always its own process — `_run` is a `subprocess` — and `session` now runs
   the arms after a failure and prints a summary saying which of them failed.
   The session is still red; the rows that ran are still there.

The rule that replaces the old one is narrower and worth keeping: **an arm may
fail, and may not take the container with it.** What it does not license is
reading a table from an arm that ran *after* a failure without asking what
failed — a build error fails every arm identically, and an unhealthy device
fails them for a reason none of them prints. The summary is there to be read.

`modal run modal_app.py::wedge_demo` is what makes that a demonstration rather
than an argument, the way `stall` does for the wrapper below: it launches a
kernel that spins for ten minutes and takes a three-arm session through it,
requiring exactly the middle arm to fail.

### 3. Measure at the cheapest tier that can see the thing

In order. Do not skip a rung because the one above is more convincing; a rung
that can see your change is a rung that can *reject* it before you pay for the
next one.

1. **Locally, no Modal.** The library's default feature set is device-only
   ordinary Rust, so `cargo clippy --all-targets` and `cargo test` need no
   toolkit and no GPU — that is tier 1, and it runs on a laptop. Run
   `scripts/fmt` before you push: tier 1 checks four manifests and goes red in
   twenty seconds.

   The three kernel crates need one more thing and only one: `cuda.h`, for
   `cuda-core` → `cuda-bindings`' build script. That script reads
   `CUDA_TOOLKIT_PATH`, which is why the image sets it, so on a Linux box the
   `cuda_cudart` redistributable unpacked anywhere plus `CUDA_TOOLKIT_PATH`
   pointed at it is enough to `cargo check` and `clippy` all four crates —
   including `--features host`, which nothing else local can reach.

   That gets you the *typecheck* and not the codegen: a device build also wants
   LLVM 21 and the cuda-oxide backend, which is the half the image exists for.
   So a post-monomorphization `const { assert!(..) }` still belongs to tier 2.
   Every ordinary compile error, and every lint, is catchable here for free.
2. **Tier 2, on CPU.** Anything that is a compile, a count or a static gate.
   `scripts/modal-run build` for codegen and lints, `scripts/modal-run
   regcount` for the register table and the occupancy step. Cents, and the
   highest value per minute here.
3. **One container, one arm, for an A/B.** Two spellings of the same kernel
   compared across two runs is not a comparison — #98 measured 2.9% of drift
   between runs of the same tree. Put both arms in one sweep.
4. **A batched session, for a claim that spans sweeps.** `session --arms`, per
   rule 2 above.

## `scripts/modal-run`, and the three ways a Modal run costs you money

Every Modal invocation in this repo goes through the wrapper rather than calling
`modal run` — tiers 2 and 3 above, and you:

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

The third way is the cheapest to pay and the most expensive to believe: **a run
that was killed and read as a passing one**. A concurrent agent stopping an app,
a Modal-side eviction, a spend limit tripping mid-run, the dashboard's stop
button — none of those are your entry point failing, and what the client reports
for them is not one thing. Measured on client 1.5.1 (#103): stopping an app
while the function is running raises `RemoteError` and exits **1**, stopping one
between app creation and the function call makes the client **hang on
`AppHeartbeat` indefinitely**, and the field report that opened #103 saw **0**.

So `scripts/modal-run` does not decide from the exit code. `modal_app.py` prints
a completion sentinel after each entry point's last line — `completes`, one
`print` — and the wrapper returns 0 only having seen it, and **4** otherwise.
Absence of failure is not evidence of completion; the sentinel is. The exit
codes in full:

| code | meaning |
| ---: | --- |
| 0 | the entry point ran to its last line — the sentinel was seen |
| the run's own | the entry point itself failed |
| 2 | the workspace spend limit |
| 3 | a wedge this wrapper detected and stopped |
| 4 | the run ended, without failing and without finishing |

`modal run modal_app.py::stall` is the control: it prints one `$ ` line and then
sleeps, so `modal app stop`ping it from a second shell reproduces exactly this
for a fractional CPU and a minute. Both graces are env-overridable
(`MODAL_RUN_STARTUP_GRACE`, `MODAL_RUN_SILENCE_GRACE`) for the same reason — a
check nobody has watched fail is not a check.

If you do run `modal run` by hand, check `modal app list` afterwards and confirm
your app is `stopped`. A live app you have stopped paying attention to is still
a live app.

### In CI

`cuda.yml`'s two jobs and `gpu.yml`'s call `scripts/modal-run`, and it is the
only command in its step, so the wrapper's code is the step's code and the job
is red on every one of 2, 3 and 4. Nothing catches, remaps, or `|| true`s them.

CI is also where the sentinel earns the most. `pip install modal` is unpinned,
so the runner takes whatever version shipped that morning — 1.5.3 on the first
run of this, against the 1.5.1 the three exit-code behaviours in #103 were
measured on. A gate built on the client's exit code would be a gate that can
change under you between two pushes; a line the container prints cannot.

2, 3 and 4 also arrive as **`::error::` annotations**, so the diagnosis is on
the check rather than three hundred lines into a log. That is worth most for 2:
the spend limit fails every entry point at once, in seconds, and a `build` check
that goes red in fifteen seconds reads exactly like a broken diff — it has cost
diagnosis time before. There is deliberately no retry on any of them; 2 in
particular would spend the rest of the workspace's day retrying nothing.

**The graces keep their defaults in CI (300 / 1200), and the runner's own
timestamps say why.** They are budgets for *silence*, not deadlines on the run:
every line the client prints — `Created objects`, the mount tree, the NVIDIA
banner — resets the clock, so a cold start costs wall time without consuming a
grace. That is what lets one pair of values work on a laptop and on a runner.

Measured on the runs that wired this up, from the per-line timestamps in the
GitHub log — one warm run of each of the three entry points CI invokes:

| | `build` | `regcount` | `device_tests` | budget |
| --- | ---: | ---: | ---: | ---: |
| first line the watchdog reads as alive | 6.5 s | 7.3 s | 9.2 s | — |
| longest silence before it | 3.1 s | 2.9 s | 4.4 s | **300 s** |
| longest silence anywhere in the run | 58.9 s | 55.7 s | 35.8 s | **1200 s** |
| sentinel, and the run's length | 219 / 221 s | 183 / 184 s | 95 / 97 s | its `timeout=` |

Two orders of magnitude of headroom on the startup budget, and the longest quiet
stretch anywhere — the `sm_100a` codegen, which prints nothing while it works —
is a twentieth of the silence budget.

The startup budget is the one doing the work in CI, and that is by construction
rather than by luck: `SILENCE_GRACE` is 1200 s while these three entry points
carry `timeout=` of 900, 900 and **300** (the third was 1200 until the launch
watchdog PR right-sized it against `device_tests`' measured 83.5 s), so a run
that starts and *then* hangs dies on Modal's clock at the same moment or sooner.
What `timeout=` cannot see is a container that never reaches Python — that is
the whole reason the wrapper exists — and 300 s of startup silence against a
worst observed 4.4 s is what covers it.

A *lower* silence budget would still be the change these numbers support: it
would stop a post-start hang at a fraction of the function timeout instead of
dead-heating with it. Not done here, on one warm sample per entry point — and
less urgent than it was, because the hang it was aimed at is now caught two
levels in. A launch that stops making progress is `kittens::watchdog`'s 30 s,
and the silence budget only ever sees the ones with no launch in them.

## Deadlines: three of them, one per level

Nothing here is allowed to run forever, and the three bounds are nested so that
each one covers what the level below cannot see. They are listed inside-out.

| bound | who owns it | budget | what it catches |
| --- | --- | ---: | --- |
| one launch | `kittens::watchdog`, in the process | 30 s | a kernel that stops making progress |
| a container's silence | `scripts/modal-run`, on your laptop or the runner | 300 s to first output, 1200 s between lines | a container that never reaches Python, and a run stopped out from under the client |
| the function | Modal's `timeout=` | per entry point, below | everything else, including a wedge in `cargo` |

**The launch deadline is new and is the one that changed what can be batched.**
Every host wait in `src/`, `examples/`, `experiments/` and `device-tests/` goes
through `kittens::watchdog::wait` or `ReadBack::read_back`, which poll the event
behind the work instead of blocking on it. Past 30 s the process prints the row
the sweep last announced and calls `abort()`, so a wedged arm exits `-6` in half
a minute where it used to hold a B200 until `timeout=`.
`modal_app.py::wedge_demo` is the control that shows it firing, and
`docs/library/watchdog.md` has the design.

### `timeout=`, and the measurement each one is three times

The rule, borrowed from the twin repo (oxide-train#131): **a timeout is about
three times a measured baseline, with the baseline in a comment beside it.** An
entry point whose work is chosen by its arguments has no baseline to state, and
stays open-ended — and *says so*, rather than merely being large. What the file
carries:

| entry point | measured | `timeout=` | |
| --- | ---: | ---: | --- |
| `build` | 291.8 s warm, 470.8 s worst (#226) | 900 | 3.1× warm |
| `regcount` | 183–184 s (the per-line timestamps above) | 900 | 4.9× |
| `upstream_ptx` | a clean clone plus `ptxas`; unmeasured | 900 | rides `CHECKING` |
| `device_tests` | 83.5 s (#226) | **300** | 3.6×, down from 1200 |
| `examples` | two binaries, both correctness gates; unmeasured | 1200 | open-ended |
| `bench`, `session`, the named sweeps | the argument chooses the work | 3600 | open-ended |
| `wedge_demo` | three arms, one killed at 5 s | 1200 | open-ended |
| `doctor` | one driver query | 300 | open-ended |
| `stall` | a nap of a length its caller picks | 300 | a bound on its argument |

`device_tests` is the one that moved: it shared `RUNNING` (1200 s) with
`examples` and is 83.5 s, which made its ceiling fourteen times its own
measurement. The two do different amounts of work and now carry different
constants.

A ceiling is not a bill — the container is billed for what it runs — so the only
thing a loose one costs is a wedge riding longer. That is why these are worth
tightening at all, and it is also why they are no longer the first line of
defence: the 30 s row above is.

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
