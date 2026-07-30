# The examples harness — `main.rs` and `bench.rs`

Source: `examples/src/main.rs` (the binary, the status table and the occupancy
census) and `examples/src/bench.rs` (the clock, and the shape a timed run is
taken at).

## What the crate is for

`examples/` is kernels written against the API we want. Each file carries a
header stating whether it **runs** — it has a launcher and a CPU reference and
exits non-zero on a wrong number — or only **compiles**. There used to be a
third status, *aspirational*, naming API that did not exist yet; an example in
that state was not a placeholder but a precise statement of a missing surface,
in the only terms that matter, which is what a kernel author has to type. The
diff between these files and the library *was* the backlog.

**Every kernel here is in the default build**, and there are no cargo features
left. That is what makes `scripts/modal-run build` — a real `sm_100a` codegen
of this crate — a regression gate on all of them: a gated kernel is not in the
default feature set, so it was never compiled by the thing that compiles this
crate, and the example exercising the most library surface at once was the one
nothing checked.

**Compiles is not runs**, and the distinction is the point of the status
column. Giving a kernel a launcher is real work with a real failure mode — see
the two seed arguments in `docs/kernels/softmax.md` and
`docs/kernels/layernorm.md`, which had to answer different questions — and not
a status to be assigned.

`main` prints the table and then runs every kernel that has a launcher, so on a
B200 the binary reports numbers and exits non-zero when they are wrong; off a
GPU it degrades to the table. A build box has no driver, and the table is the
whole point of the binary there — only a device that exists can fail a check.

### What is *not* here, and where it went

One GEMM: the kernel the library ships, at the epilogue and the two instruction
widths it ships with. The tile rungs it was chosen over, the two schedulers, the
ablation cube, the epilogue families, the doubling probes that compute a
deliberately wrong `C` on purpose to price one term, the warp-specialized
variant, the benchmark sweeps and the cuBLASLt denominator are all in
`experiments/`, which is where the numbers quoted in these docs were measured.
Nothing was deleted; this crate is the third of it that was ever about teaching.

`bench.rs` here is the clock and nothing else, because `softmax::bench` and
`layernorm::bench` are written against it and `experiments/src/bench.rs`
includes this file through `#[path]` rather than carrying a second copy of it.

## The clock — `bench.rs`

Three rules the file exists to enforce, in the order they matter.

**1. A number only ever comes out of a checked run.** `time` takes the launch
as a closure, and every kernel hands it one only from inside its own
verify-then-time entry point — so there is no path that reports throughput for a
kernel that computed the wrong answer. A harness that would happily print
TFLOP/s for garbage is a trap you fall into once, and the number outlives the
run that produced it.

**2. The clock is the device's.** CUDA events either side of the launch, not
wall clock around the driver call. The events measure the kernel's own span on
the device, which is the thing under test; wall clock would measure the driver's
launch path and the host's scheduling as well, and at the small end of a sweep
those are the same order as the kernel.

**3. A single number is not a measurement.** `Timings` keeps the launches both
sorted and in the order they were issued.

### The sample

`WARMUP = 5` launches are discarded before timing begins: the first pays module
load, and the first launch of a given shape pays the driver's own setup for it,
and neither is representative of the next thousand. `ITERATIONS = 30` launches
are then timed. Both are `pub`, because a table that quotes a minimum has to say
how many launches it is the minimum *of*, and `experiments/`' sweeps print both
figures in their headers rather than writing them down a second time.

The headline is the **minimum**: it is the least noise-contaminated estimate of
what the kernel can do, since every source of error on a quiet device adds time
and none subtracts it. The median and maximum are printed beside it because a
gap between them is a finding — a device sharing work, a clock dropping, a
first-touch cost that never amortizes — and the table's job is to surface that,
not to hide it behind one number.

### Why the launch order is kept

Sorting answers only half the question. A row whose minimum will not repeat has
two possible causes with opposite fixes:

- the distribution is wide and `min` of 30 is sampling its left tail, which more
  samples fix;
- the device is slowing down inside the call, which no number of samples fixes.

`Timings::spread` is `max/min − 1` and sees the first. It bounds what `min` of
`ITERATIONS` can be asked to do: a call whose launches all land within 1% of
each other has a minimum that is the floor, while a call spread over 15% has one
that is wherever the luckiest of thirty draws fell, and two such calls will not
agree.

`Timings::drift` is the second half against the first **in launch order**, and
sees the second. Noise that is stationary leaves it at zero however wide the
spread is. A clock stepping down under sustained load does not: it puts the fast
launches at the start and the slow ones at the end, so the sign and the size of
`drift` are the thermal question asked directly rather than inferred from a
spread that cannot tell the two apart.

Neither is visible once the samples are sorted, which is why both orderings are
kept.

### Two crates, one clock

`bench.rs` has two sets of callers and every item in it has one, but not all of
them are in this crate. `time` is reached from `softmax::bench` and
`layernorm::bench` either way; `Timings::median`, `Timings::spread` and
`Timings::drift` are columns only a sweep prints. Nothing in `examples/` calls
those three, and removing them would take the two diagnoses above out of the
harness the sweep shares with the example.

`Shape` prints as `m x n`, or `m x n x k` where a kernel has a reduction depth.
It formats through `Formatter::pad` rather than `write!` so that the table's
column width reaches it.

Progress from `time` goes to stderr and not into the table: a sweep is minutes
long, and a reader watching it should be able to tell a slow size from a stuck
one. Reaching that line at all is the check having passed.

## The occupancy census — `main.rs`

A register count off `ptxas` is half an occupancy argument; the blocks-per-SM
column is the other half, and it is the half that depends on the launch rather
than on the code. It is printed because a kernel can be slow because it is
waiting, or because too few of it fit on an SM to have anything to wait *with*,
and those want opposite fixes. The driver is asked rather than the arithmetic
reproduced, because the shared-memory carveout it picks is its own business and
not a number the file can derive.

The envelope includes the **opt-in**, which is why
`kittens::launch::admit_shared_plan` is called before the query and not only
before a launch. A plan over 48 KiB is inadmissible on a function nobody opted
in for, and the driver answers 0 for that exactly as it answers 0 for a plan too
big to fit — so without that call the column reports the omission and reads like
the tiles.

### Four outcomes, not one

That ambiguity is the reason `Occupancy` has four variants rather than an
`Option`:

| variant | meaning |
| --- | --- |
| `Blocks(n)` | blocks per SM at the kernel's own envelope, opt-in included |
| `Cluster` | no entry point to ask about — an absence on purpose |
| `TooLarge` | the plan is past what this device will admit even with the opt-in; the only outcome that means the tiles are the problem |
| `Failed` | the driver refused something on the way — not an answer about the kernel, and not to be read as one |

The zero that started this meant "nobody opted this function in" and read as
"these tiles do not fit". Collapsing the failures back into one absence would
rebuild the same ambiguity one layer up: a plan the device genuinely will not
admit would print `cluster`, which is what a cluster-launched GEMM prints for a
legitimate reason, and the next reader would spend the same hour.

`Cluster` marks a kernel where `cuOccupancyMaxActiveBlocksPerMultiprocessor` is
the wrong question. It takes a block shape and no cluster, so a
`#[cluster_launch]` kernel would get an answer about a launch it never performs.
That is a limit of *that function*, not of the question:
`cuOccupancyMaxActiveClusters` does take a cluster shape and does describe such
a kernel — `device-tests`' `tmem residency census` calls it, and on a cluster
launch with no allocator in it the answer matches a counted census exactly. So
`Cluster` marks a query nobody has wired into this table, not a residency
nothing can reach.

### Table formatting details that bite

- The status field is 38 columns wide, and a longer status string silently
  pushes the shared-memory column out of the table.
- The blocks/SM cells are four characters wide, and two of the four outcomes
  have something to say that does not fit in one. Saying it in notes under the
  table beats widening the column for the case that should never fire.

## Related

- `docs/kernels/softmax.md`, `docs/kernels/layernorm.md` — the two kernels whose
  `bench` entry points are written against this clock.
- `experiments/README.md` — the sweeps that include this file through `#[path]`.
