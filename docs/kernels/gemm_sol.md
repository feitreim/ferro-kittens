# `gemm_sol`

The design record for `examples/src/gemm_sol.rs`: why the kernel has the shape
it has, every measurement behind a decision, and the alternatives that were
tried and lost.

The kernel is a kittens port of cuda-oxide's canonical Blackwell
`gemm_sol_final`, expressed through this library's typed tiles and pipeline
primitives. **Matching the reference is not parity with cuBLASLt.** The
reference itself only reaches 0.877–0.966 of cuBLASLt across the shapes it is
measured at, so "as fast as `gemm_sol_final`" and "as fast as cuBLASLt" are two
different targets, and only the first of them is what this port is held to. The
ratios the port itself reaches, and the sweeps they come out of, are in
`experiments/` (`bench sol`, `bench sol-small`, `bench sol-ablate`).

## The three entries

All three are the same body at a different cluster tile, and they differ in
nothing else — which is what makes the pair of 256-row entries a controlled
comparison of tile quantization against operand traffic.

| Entry | Cluster tile | Shared bytes | `n` contract |
| --- | --- | --- | --- |
| `gemm_sol_m256_n128` | `[256, 128]` | 131 328 | multiple of 128 |
| `gemm_sol_m256` | `[256, 256]` | 196 864 | multiple of 256 |
| `gemm_sol_m512` | `[512, 256]` | 229 632 | multiple of 256, even M256 tile count |

An SM divides 233 472 B, so every one of these plans declares more than half of
it: residency is **one CTA per SM**, and a `cta_group::2` MMA needs its pair
co-resident, so the device holds `SMs / 2 = 74` clusters at once on 148 SMs.
Both of those are device properties rather than tuning knobs, which is why
`RESIDENT_CLUSTERS` is written down instead of queried — `select_variant` is a
`const fn` that sees a shape and nothing else. `bench sol`'s table 0 prints the
same number off `cuDeviceGetAttribute` beside every row it divides, which is
where a device that disagrees would show up.

`[256, 128]` is the one place the shape contract widens: it is the only entry
whose contract admits an `n` that is not a multiple of 256. It quadruples the
tile count of a square problem against `[256, 256]`'s, which is what a shape too
small to fill a wave of the wider one is for.

## Picking an entry: wave arithmetic

A launch is one cluster per output tile and takes `ceil(tiles / 74)` waves of
them, so `tiles / (waves · 74)` is the fraction of the launch that is not
idling. Halving the tile's `N` doubles the tile count, and that raises the
fraction **only** while both counts fit the same number of waves — at or below
half a wave of wide tiles, i.e. 37 of them. Above that the two are equal on
waves and the narrow tile is left paying its costs for nothing.

`bench sol-small` measures exactly that boundary, both directions reproduced
twice:

| Wide output tiles | narrow / wide |
| --- | --- |
| 16 | 1.27× |
| 36 | 1.27× |
| 64 | 0.93× |
| 144 | 0.81× |

The `[512, 256]` crossover is set above 8192, where upstream takes it at 16384.
`bench sol` has it at 1.12× at 8192³ against a wave efficiency both entries
share, so what it wins there is **operand traffic per flop and not tiles**.

## What the K loop is bound by

`bench sol-ablate` ladders every phase in `K`, and the answer is not the same
thing at the two 256-row entries:

- `issue only` reaches **100.0% of peak a K block** — the tensor core can be
  issued at peak from one warp.
- The barrier round trip is **zero**. (`mma only` is `issue only` with the MMA
  warp's `load.wait` removed as well, so `issue only − mma only` is the round
  trip alone, unfused from issue rate.)
- The whole of `[256, 256]`'s **21% deficit is its feed's duty cycle**: the
  operand pipeline alone needs **95.5%** of the time the tensor core is busy,
  where `[512, 256]`'s needs **55.7%**, because a `[256, 256]` cluster tile
  moves **1.33×** the operand bytes per flop.

That is a property of the tile, so it belongs to the variant choice and not to
the K loop.

The arms that produce those numbers: `feed only` is the TMA and every barrier
with the MMA and the drain removed — what the memory pipeline alone can sustain.
`issue only` is the MMA and every barrier with the loads removed, the producer
arriving on `load` instead of filling it, so the ring recycles on tensor-core
issue and no global traffic occurs at all. `no drain` is the whole kernel except
the drain — the accumulator fills and is released without being read — so
`whole − no drain` is the epilogue's cost *including* whatever of it fails to
overlap. `mma only` is `issue only` with the MMA warp's `load.wait` dropped too:
`issue only` keeps the whole `load`/`free` handshake, so what it measures is
barrier round trip *fused with* issue rate, and this arm separates them. The
producer still keeps `free.wait_recycled` and the `ready` publication still
bounds the MMA warp one output tile at a time, so what runs unthrottled is
exactly the K loop, which is what is being priced.

### The compile-time ring stage

Every entry's shape contract is `k % (STAGES · BLOCK_K) == 0`, so
`global_k % STAGES == k % STAGES` and the four positions of a four-way unroll
are always stages 0, 1, 2, 3 in that order. Handing each position its stage as a
const parameter turns two operand descriptors and two barrier addresses from
runtime arithmetic into folded offsets; the phase parity is the one thing that
genuinely moves with `global_k`, and it moves once a turn instead of four times.

Upstream states the same fact — *"`k_iters % 4 == 0`, so the producer's global
stage and this local stage agree at every tile boundary. Keeping this expression
loop-local lets the unroll pass fold each stage match."* — and spells it as
`#[unroll(4)]` over a loop-local `k_idx & 3` with a `match` on it. That attribute
is only rewritten inside a `#[kernel]` or `#[device_function]` body, so a const
parameter is the spelling reachable from a plain `impl` method, and it does not
depend on an unroll pass firing. The `FOLD` dial keeps both spellings: they
compute the same `C` by construction (the const is `global_k % STAGES` and
nothing else), so the difference between them is entirely how much of the MMA
warp's issue stream is scalar arithmetic.

## The item handoff, and why its depth is four

A cluster's producer publishes a work item's coordinates through a mailbox
behind a phase-parity wait. Such a mailbox is sound only while the producer
leads its slowest consumer by **less than** the ring's depth. The port carried
**one** barrier and **one** cell — upstream spells it that way too — while the
chain of back-pressure between the producer, the MMA warp and the epilogue lets
**four** publications be outstanding.

The derivation, at the shallowest `k` the contract admits:

1. The producer publishes index `p` only after item `p-1`'s loads have all been
   issued, and the last of those passed `free.wait_recycled(p · k_blocks − 1)`,
   which is the MMA warp's `free` commit for K block
   `p · k_blocks − 1 − STAGES`.
2. At `k_blocks == STAGES` (`k = 256`) that block belongs to item `p − 2`. So
   all the producer's own throttle says is that the MMA warp has *finished item
   `p − 2`*.
3. The one-accumulator entry's accumulator ring is two deep, so the MMA warp
   finishing item `p − 2` means the epilogue released `empty(p − 4)` — it has
   read index `p − 4` and is about to wait on `p − 3`.
4. Indices `0..=p` are published against a consumer waiting on `p − 3`: **four
   outstanding**.

(Steps 3 and 4 are what the `ITEMS >= ACCUMULATORS + 2` assert in the source
names.)

A handoff shallower than four therefore either overwrites an item's coordinates
before the epilogue reads them, or lands its wait on a parity the producer has
already passed — and once the producer has published `has_work = false` and
left, on one that will never flip again. The second is the defect that shipped:
`4096x4096x1024` announced and never returned, inside its own shape contract,
while every `k` at and above 1280 won the same race on timing alone.

The `[512, 256]` entry's chain is one step tighter (a one-deep accumulator
interlock per half, so step 3 gives `p − 3`), so four covers it with a margin of
one. Both entries keep the same depth because one shared plan is worth more than
the barrier the large one could save — and **the depth costs no shared memory at
all**: four barriers and four `TileInfo` cells fit inside the padding the
128-byte-aligned staging buffer already left.

The four steps above are an argument, so they are also a test.
`kittens::sync::handoff::depth_needed` enumerates every interleaving the same
three throttles admit and answers **4** at `k_blocks == STAGES`, **3** above it,
and — the part the argument missed — **2 at one item per cluster**. So the
one-deep handoff was never sound at any legal shape; it was winning a race by a
margin that `k` sets.

### Why no gate caught it

`1024x1024x512` alone is why. It is 16, 8 and 4 clusters at the three entries
against 74 resident, so **every cluster gets exactly one item**: its producer
publishes once and then the sentinel, and the handoff is never asked to hold
two. Every timed row in every sweep was at `k >= 2048`. The whole of the regime
the depth is about — more clusters than the device holds, at a `k` short enough
that a tile's K loop is comparable to its drain — was untested at all three
entries.

`SHALLOW_K_GATE` is that regime: one shape per entry (`2048x2048`, `2560x2048`,
`5120x2048`, all at `k = 256`), each above 74 clusters and at
`k_blocks == STAGES`, which is the shape the derivation is tight at. They cost
about 20 million output comparisons between them, a fifth of what one `bench`
row already checks.

## The epilogue

### Lane ownership fixes the row axis

`tcgen05.ld` reaches the 32 tensor-memory lanes of the issuing warp's own
sub-partition, and which sub-partition that is comes from the warp's index
*within its warpgroup*. A `[128, N]` accumulator is 128 lanes, so four warps
tile its rows exactly and a fifth warp of the same warpgroup has no rows left to
own. That is hardware arithmetic, and it is why the epilogue cannot be widened
along the row axis at all.

### The warpgroup split, which lost and now wins

The axis that *is* open is columns: warp 4 is warpgroup 1's warp 0 and owns the
same rows 0–31 warp 0 owns, so a second warpgroup can only split the
accumulator's columns. Warp `w` drains rows `32 · (w % EPILOGUE_ROWS)` of column
half `w / EPILOGUE_ROWS`, and every warp still reads only the lanes its
sub-partition index entitles it to — lane ownership is satisfied twice over
rather than relaxed.

It is free in the resource that binds: residency is one CTA/SM on shared memory,
and doubling the warps halves each one's staging width, so the shared plan is
**byte-identical** at every entry (asserted in the source at all three widths).
That is also what lets one `no drain` control serve both spellings.

**It did not pay, and the reason was a `.local` depot rather than the
hardware.** The original table, two round-robin passes each at 320 threads
against 192:

| | 4096³ | 8192³ |
| --- | --- | --- |
| launch | −1.4%, −1.8% | +0.3%, +0.4% |
| drain alone | −9% | +3% |

That was read as the drain being slower once split, and the mechanism offered
was the resource that fixes the row axis: warps 0 and 4 own the *same* 32
tensor-memory lanes, so a column split does not spread the LDTM traffic over
more sub-partitions, it puts **two requesters on each of the same four**.

The lane argument is still true. It was not what the table was measuring.
Every one of those rows was taken through a `.local` depot that no longer
exists: `store_tile_x4` homed each drained band to local memory until the
rolled walks were marked (#166, landed in #180/#181/#184), and a band *is* the
frame — a `RegTile<32, 64>` is 64 f32 is 256 B, which is the staged family's
frame exactly. The split pays that frame **twice per accumulator** where one
warpgroup pays it once, so the arm that did more of the thing that was broken
looked like the arm that was worse.

Re-run at `f64a27b`, same table, same two round-robin passes, `regcount`
reporting **zero stack and zero spill on every `gemm_sol` rung**:

| | 4096³ `[256, 256]` | 8192³ `[512, 256]` |
| --- | --- | --- |
| launch | **+0.8%, +0.9%** | **+2.1%, +2.1%** |
| drain alone | 3.35 → 3.10 µs, 3.28 → 3.17 µs | 4.72 → 3.05 µs, 4.70 → 3.04 µs |

The sign is the other way at both entries and both passes, and the term that
was blamed is now the term that moves: the drain gets **35% cheaper** at
`[512, 256]`, not slower. The `no drain` control at its own 320 threads is
still identical to the 192-thread one (0.5157 against 0.5155 ms), so the two
extra warps remain free when they do nothing and the whole difference is the
drain.

Both 256-wide entries ship the split as of #196. `[256, 128]` does not, and
that is a gap in the measurement rather than a result: table 6 has arms at
`[256, 256]` and `[512, 256]` only, so the narrow entry has no A/B of its own
and keeps one warpgroup until it does.

The lane ceiling is presumably still there — it is just above this, not below
it. What the split buys is not more sub-partitions but more warps issuing
against them, and that was worth nothing while each warp was also driving a
band through local memory.

### The drain rungs

The shipped drain is a band of 64 columns out of TMEM, `stmatrix` into the
warp's staging tile, the whole tile out to `C`, and the write-after-read the
next band owes itself.

- **`per issue`** — 64-column bands with a wait per LDTM issue. The drain that
  shipped first, and the one the doubling ladder decomposes.
- **`paired`** (shipped) — a 64-column band with both of its LDTM issues in
  flight before one wait, through `TmemTile::tile_x8_batched`: half the waits
  per byte of `C`, the same instruction in every other column of the census, the
  same shared plan. Worth **+0.9% of the launch at 8192³ and nothing at 4096³**,
  in two round-robin passes each, and it ships on that.

  The batched drain takes `BAND` columns and `ISSUES = BAND / 32` `.x8` loads —
  two per 64 columns, one per 16 rows of the warp's 32 — and nothing else moves:
  same `stmatrix`, same staging tile, same `store_shared_rows`, same two
  `warp::sync_mask` a staged pass, same shared plan. The difference against the
  per-issue drain is the wait structure and the band's register liveness and
  nothing besides. `SHIPPED_DRAIN` names the winner in one place, so a rung that
  wins its A/B ships by moving that line — which is what `paired` did.
- **`wide`** — a 128-column band, all four issues in flight before one wait: a
  quarter of `per issue`'s waits per byte of `C`, at 128 f32 a thread live. It
  first measured −1.1 to −1.3% at 4096³ and +0.2 to +0.3% at 8192³, and the note
  on that reading was that the rung is the only one not free: the wider band
  took registers to 176/168 and the stack frame from 256 B to 512, **and the
  loss was not separated from that frame**.

  It is separated now, because the frame is gone — #166's walk marking landed in
  #180/#181/#184 and `regcount` reads **zero stack and zero spill** on every
  `gemm_sol` rung, `wide` included (130 registers at `[256, 256]`, 182 at
  `[512, 256]`, against 91 and 80 for `paired`). Re-run, two round-robin passes:
  **1.001 / 0.998 of `paired` at 4096³ `[256, 256]`, and 1.004 / 1.007 at 8192³
  `[512, 256]`**, where the epilogue alone goes 4.56/4.80 µs to 4.21/4.25. So
  about a point and a quarter of the original loss was the frame, and what is
  left is neutral at the wide-staging entry and a genuine +0.4 to +0.7% at the
  narrow-staging one.

  That split is the mechanism, and it is why this rung is *not* what shipped.
  `wide` widens the register band only — `drain_dial` hands it the entry's own
  `STAGE_N` — so at `[256, 256]`, whose staging tile is already 256 columns, it
  changes the wait count and nothing else and measures as nothing. At
  `[512, 256]`, whose staging tile is 128, a 128-column band makes the drain one
  lift and one store per pass, and that is worth the half point. The warpgroup
  split above is worth four times as much at the same entry and the two do not
  compose: at two warpgroups `[512, 256]`'s per-warp tile is 64 columns, which a
  128-column band does not divide. The bigger win took the slot.
- **`nocvt`** — the shipped drain with the `cvt` pass removed and nothing else:
  same LDTM count, same wait count, same `stmatrix` count, same `ld.shared`,
  same `st.global`, same shared plan, `cvt` zero. Half the band's words go
  unused, which is what keeps the `stmatrix` count equal (a `[32, 64]` band is
  64 f32 a lane and 32 packed words, so a drain that skips the pack writes the
  first half of twice the words it needs). **The convert is free**: this rung is
  0.0 to 1.4% *slower* than `paired`, in two round-robin passes at both shapes.
  Removing 32 `cvt.rn.bf16x2` a band buys nothing.
- **`pack16`** — `tcgen05.ld.16x256b.x8.pack::16b` in place of the `.x8` load
  and no `cvt` at all. A `[32, 64]` band is 2048 elements, which is 64 b16 a
  lane, which is **one** `.x8.pack::16b` arrival of 32 already-packed words
  against the two `.x8` arrivals and 32 `cvt` the same band costs today; the
  words go to the band's eight `[16, 16]` blocks four at a time. On paper
  `whole − pack16` is the `cvt` pass's cost with none of the write-after-write
  the `twice shared` rung pays on its own staging tile.

  **It faults on the device and nothing launches it**: `bench sol-ablate` took
  it once and every SM raised `Xid 13, Out Of Range Address` — 148 SMs of warp
  exceptions and `DriverError(700)`. `.pack::16b` packs the 16-bit elements of
  two consecutive TMEM columns into one register, so against an fp32
  accumulator what it packs is the mantissa halves of two floats — and that is
  not merely the wrong values: the qualifier reads the segment as 16-bit-typed
  and the addressing that follows does not land inside a `[128, N]` fp32
  allocation. The kernel stays because its **census** is the finding —
  `cvt.bf16x2` goes from 8 to 0 while `stmatrix` and every store column hold,
  which is the question a register count could not answer — and `nocvt` is the
  oracle that can be timed.

Two conclusions come out of the pair: the convert is worth nothing, and the 69%
of the drain that the `twice shared` rung prices is the `stmatrix` pass and the
write-after-write a doubled one owes its own staging tile — not the `cvt` beside
it. That also closes `pack16` for good, since it folds the convert into the load
and folding the convert away buys nothing.

### The doubling ladder

`twice global`, `twice shared` and `twice all` are one ladder, and each rung's
difference from the one below prices one third of the
`tcgen05.ld` → `stmatrix` → `st.global` chain. `twice all` against `whole` is a
whole extra drain, and an extra pass has nothing left to hide behind, so it
prices the drain **serially** — which is the term `whole − no drain` cannot
separate from the drain's failure to overlap.

The extra global stores are aimed at the cluster's own first output tile so the
extra bytes stay in L2, and the probe prices instructions rather than a doubled
HBM stream. In the shared half the band is still live, so rung 2 doubles the
`cvt` + `stmatrix` pass alone and rung 3 doubles the LDTM in front of it; the
extra write lands on words nothing has read yet, so no rung owes an extra
`sync_mask` and the ladder holds the syncs fixed.

All three compute a wrong `C` and are on no correctness gate, as every doubling
probe in this repo is.

## The operand feed

`B_BOX` is 64 because `B`'s tensor map is built for a 64-row box and every entry
shares one map: the narrow entry's half-panel *is* 64 rows, so 64 is the only
height all three can name. Nothing about the wide entries asks for it — their
half-panel is `128 x 64`, the same shape as `ATile`, which already arrives in a
single TMA — so they pay two instructions where one would do.

**And it costs nothing.** `bench sol-ablate`'s wide-`B` arms are that map built
per entry instead of once, at byte-for-byte identical traffic. They move the
launch by 0.998 and 1.014 across two passes, the K-block rate from 0.3503 to
0.3484 µs, and `feed only` — where the feed is alone and has nothing to hide
behind — by 1.008 and 0.981. **The feed's ceiling is bytes, not instructions.**

## The traversal band

`group` is the width of the band of `N` that `pipeline::grouped` walks before it
steps in `M`. It was a rule inside the kernel and is a launch parameter now; the
rule is unchanged and the default is what it computed, because the sweep that
could have changed it found nothing to change it to. `bench sol`'s table 3 takes
`G` over `{1, 2, 4, 8, 16}` and gets **1880.3 to 1876.2 TFLOP/s at 8192³** —
flat to 0.2% — and a **1311 to 1367** range at 4096³ against rows whose own
launches spread 1.3 to 3.6%. Tuning on the second of those would be tuning on
noise.

## The watchdog

`WATCH` is the third dial and the only one that is not about what the kernel
does. `ABLATE` says which phase runs and `DRAIN` says how the epilogue issues;
this says whether the launch is allowed to hang. At `WATCH_OFF` the kernel is
byte for byte the shipped one. Above it every spin becomes
`Semaphore::wait_before` and every warp writes a four-word mark past the end of
`C` at each loop head and on each expiry, so a launch that stops making progress
*terminates*, carrying where all six of its warps were.

It exists because a launch that does not return costs a container and says
nothing: the watchdog stops it and every warp's position is lost, which is how
one shape survived three PRs with four candidate mechanisms and no way to tell
them apart. It is its own axis rather than an `ABLATE` arm because it removes no
phase and computes the same `C` — the two are orthogonal, and a reader should be
able to watch any arm.

- The deadline is `1 << 30` SM ticks, about 0.6 s at 1.8 GHz — four orders of
  magnitude past the ~26 µs an output tile takes at 4096³, so no arm can time
  out on being slow.
- Marks are indexed by `blockIdx.x` rather than by cluster, so the two ranks of
  a pair are told apart: a handoff defect is a per-CTA fact and the ranks
  publish separately. Eight slots a CTA covers its six warps.
- A stall also sets a one-word `stop` flag that every role polls at its loop
  head, so a warp that gave up takes the other five out with it and the launch
  reaches `cluster_sync` instead of hanging on it. The flag is reserved in every
  arm, including the shipped ones, because the alternative is two shared plans
  for one kernel — and everything from the barriers to the staging buffer sits
  inside padding that was already there.
- A stalled `wait_load` reports rather than returning: the K loop's shape is
  somebody else's argument and the instrument must not change it, so the loop
  runs its remaining turns against operands nobody promised and the item loop
  notices at its head. Past that point the arm computes a deliberately wrong `C`
  and is on no correctness gate.
- `WATCH_ONE_DEEP` is `WATCH_DEEP` on a one-deep item handoff — a single `ready`
  barrier and a single `info` cell, which is what the port carried before the
  fix. It reproduces the defect under the instrument instead of arguing about
  it, and differs from `WATCH_DEEP` in the handoff's depth and in nothing else:
  same body, same barriers, same shared plan, same `C` to compare against.

## The two rules the dials are held to

Each dial is a bare `u8` because it is a const generic parameter and const
generics take no enums. That is exactly the shape two experiment sets can extend
independently and collide in, and a collision does not fail to build: it
silently remaps one arm onto another, and every number downstream is wrong while
looking fine. It nearly happened — `MMA_ONLY` and `TWICE_GLOBAL` were both `4`,
on two branches, and the merge is the only place it would have been caught.

1. **Values are a permutation of `0..len`**, checked at compile time by
   `dial_is_a_permutation` over `ABLATE_ARMS`, `DRAIN_RUNGS` and `WATCH_RUNGS`.
   Reusing a value fails the assert; adding an arm without widening the array
   fails it too. The lists are also what `arm_name` reads, so they are
   load-bearing rather than decorative — an arm that is not listed has no name
   to print, and prints `UNLISTED DIAL VALUE` in a table instead of a silently
   mislabelled measurement.
2. **Every *combination* of dials must be legal, not just the reachable ones.**
   A dial value that is legal on its own and illegal in combination fails at
   monomorphization, *including in match arms that cannot be reached*: Rust
   monomorphizes both sides of a `match` on a const before anything folds it
   away. `DRAIN_WIDE` drains a 128-column band, a band has to divide the
   per-warp staging tile, and at two warpgroups that tile is 64 wide — so a
   literal `128` inside `drain_dial`'s `DRAIN_WIDE` arm breaks the
   two-warpgroup build even though two warpgroups never take that arm. A shape a
   dial implies has to come from whoever knows all the dials — the entry —
   rather than being written as a literal where one dial can see it and the
   others cannot. That is why the band and its issue count are the `WIDE_BAND` /
   `WIDE_ISSUES` parameters and not constants.

## Correctness

`a_value` and `b_value` generate small integers, so every product and every
partial sum is exact in fp32 and `check_output` compares with `==` against a
reference rounded the same way, rather than against a tolerance. The worst
relative error it returns is bf16's own.

The four staging and checking functions are `pub` because `experiments/`' copy
of the *unported* upstream kernel is staged and checked by them rather than by
re-derivations of them: two kernels compared on one clock have to read
byte-identical operands, and the way to guarantee that is to call the same code.

`check` runs all three entries at `1024x1024x512` and then the three
`SHALLOW_K_GATE` shapes, each exact over every output. `bench_plan` checks at
the gate size before it times anything, which is the order every number quoted
here comes out of.
