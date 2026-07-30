# `gemm` — `C = A·Bᵀ` on the `cta_group::2` cluster path

`examples/src/gemm.rs`, entry point `gemm_cg2_staged_x8x4`. bf16 in, fp32
accumulate, bf16 out — the signature a training GEMM has.

The name is the kernel: `cg2` is the `cta_group::2` cluster path, `staged` is
the shared-memory epilogue, and `x8x4` are its two instruction widths,
`tcgen05.ld.16x256b.x8` and `stmatrix.m8n8.x4`.

This file carries why the kernel is shaped the way it is. The ablation ladder
each number was read off — the probes that compute a deliberately wrong `C` to
isolate a term, the timing harness, the containers a result reproduced in — is
`experiments/README.md` §7.

## The cluster is the unit, not the CTA

A pair of CTAs forms a cluster and shares one `M256_N256` UMMA. Both operands
are split across the pair: each CTA stages its own 128 rows of `A` and its own
128 columns of `B` at the *same* shared offsets, and the instruction reads both
CTAs' shared memory over the cluster interconnect. The accumulator splits the
same way along M, so each CTA drains its own `[128, 256]` band of `C`.

That is why `RANKS` is a constant of the `Tile` job and not a launch parameter:
the item boundary has to be `barrier.cluster` rather than `bar.sync`, so no rank
re-arms a barrier its peer is still filling and no rank fills one its peer has
not re-armed.

The pair shares one barrier *set*, not one barrier each. The peer aims its TMA
at the leader's copy of the stage barrier; the leader's MMA arrives in the
peer's `free` and `done` semaphores.

## The tile shape, and the occupancy step under it

| | value | why |
| --- | ---: | --- |
| `BLOCK_M` | 128 | The pair covers 256, which is the `M` the instruction descriptor names and the widest `M` tcgen05 has. A tile sweep can only move the columns. |
| `BLOCK_N` | 256 | Measured — below. |
| `BLOCK_K` | 64 | One swizzle atom; see `ATOM_K`. |
| `ATOM_K` | 64 | One 128-byte swizzle atom of bf16, and the only width `SharedTile::k_walk` accepts: it carries `const { assert!(C * E::BYTES == S::ATOM_BYTES) }`. A stage wanting more K holds several atoms rather than widening the walk. |
| `STAGES` | 3 | |
| `STAGE_N` | 64 | Measured against the shared budget — below. |
| threads | 128 | One warp per 32 accumulator rows, which is what a `[32, N]` drain wants. |

### `BLOCK_N` was 128, and widening it is worth 11.6–21.6%

`[256, 256]` measured **+11.6% at 8192³ and +21.6% at 16384³** against
`[256, 128]`, on arithmetic intensity and nothing else — `M·N/(M+N)` flops per
operand byte, which doubling `N` buys 1.5× of.

What it costs is generality, and that is not free: a launch must have
`n % 256 == 0` where 128 used to do.

### `BLOCK_K · STAGES` is one number, not two

The shared budget caps `BLOCK_K · STAGES` at 228 for this pair tile, so the two
are a *factorization* of a fixed number of bytes in flight rather than
independent axes. `BLOCK_K` does not move arithmetic intensity at all — a tile
reads `(M + N) · K` bytes for `2 · M · N · K` flops however K is blocked — so
what depth trades against is stage barriers and `expect_tx` charges on one side
and how coarsely the ring recycles on the other.

Dropping to `STAGES` 2 to buy back shared memory was measured and loses:
**−11.8% / −7.3% at unchanged residency**.

### CTAs per SM is 2, and it was measured before it was counted

`cuOccupancyMaxActiveBlocksPerMultiprocessor` takes a block shape and no
cluster, so it cannot answer about a `#[cluster_launch]` kernel — which is why
`main.rs` prints `cluster` in this kernel's occupancy row instead of a number.

Two independent methods, sharing nothing:

- **Bisecting the persistent grid's cap on a clock.** At the earlier
  128-column envelope, 8192³ took **2.1036 ms** capped at one CTA per SM,
  **1.2130** at two and **1.0217** at three — 2.07× between the ends. A timing
  sweep over a persistent grid's cap *is* a residency probe for a cluster
  kernel.
- **`device-tests`' `tmem residency census`**, which timestamps both ends of
  every CTA's tensor-memory allocation off `%globaltimer` and sweeps the
  intervals for the most ever open at once on one SM. It agreed with the clock
  exactly at that envelope.

At `BLOCK_N = 256` the binding term of `min(512 / columns, shared per SM / plan)`
is the **tensor memory** one: `512 / 256` is 2 before shared memory is consulted
at all. That is what makes the staging run's extra 16 KiB free — the census
counts the same 2 at the envelope with it and the envelope without.

`CTAS_PER_SM` is what `MAX_CLUSTERS` is derived from, and `MAX_CLUSTERS` is a
tuning constant rather than a correctness one: `pipeline::run` walks every item
whatever the grid is, so a device with a different SM count computes the same
GEMM off a wave that is not quite a wave.

`shared_per_sm` is *queried* rather than written down. It is 233 472 B on a
B200, and a residency is a floor division by it, so a figure that is only nearly
right moves a kernel across an occupancy step and changes the answer rather than
the third digit. `ctas_per_sm` says the same thing as arithmetic, and `check`
prints it, so the measurement and the arithmetic can disagree out loud.

### The shared plan, and why the staging ring is one buffer deep

Everything before the staging run — two operand rings, two stage barrier rings,
the MMA-complete semaphore and the word `alloc_cluster` stages its result
through — is 98 364 B. The staging run is 128-byte aligned on the end of it, so
the declared total is **114 816 B**.

At two CTAs an SM a CTA gets `233 472 / 2 = 116 736` B. That leaves **1920 B of
headroom**, and a second set of `[32, 64]` staging buffers is 16 384 — so the
ring being one buffer deep is arithmetic, not preference, and the file asserts
it.

The contract is not decoration: 112 KiB is far past the 48 KiB a block gets by
default, and the opt-in (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`) is
issued by the prepared-launch path. `cluster_launch` has nothing to do with it;
`kittens::launch::admit_shared_plan` is the same opt-in for a kernel whose
output partition no contract describes.

The plan is one walk over `SharedPlan`, run twice: against `SharedPlan::sizing()`
it is the byte count, against `SharedPlan::attach()` it is the handles. Before
that it was two programs — an arithmetic byte total and a pointer walk — joined
by a hand-written `const { assert!(..) }` that could only ever check the total,
never an offset. The total moved 98 368 → 98 364 when the `tcgen05.alloc`
staging word became the four-byte `u32` it is, which changes nothing downstream:
the staging run is 128-byte aligned, so both numbers round to the same 98 432
and the declared 114 816 is unmoved.

`SharedPlan::tile_ring` aligning the run to 128 bytes is also not cosmetic. The
run would otherwise start at the end of a four-byte staging word, which is not a
swizzle atom's alignment, and would put the phase `SwizzledChunks` derives from
the base somewhere the `stmatrix` and the read-back would still agree on but no
reader could check.

## The item map is worth 23%

The grid is persistent. One output tile is one work item, `pipeline::run` hands
the job an item index and nothing else, and `pipeline::grouped` turns that into
a tile in blocks of `GROUP = 8` tile-rows rather than row-major.

Five shapes identical in tiles, waves, grid and bytes but differing in `M : N`
moved **1123.7 → 915.7 TFLOP/s** — 23% at a fixed flop count, best and worst
exact transposes of each other. Nothing but the order the output is walked in
can do that.

What the grouped map changes is the *working set of a wave*. 148 clusters walked
row-major sit on `ceil(148 / tiles_n)` rows of tiles and span as much of `N` as
the shape allows; walked in groups they sit on a block whose shape the aspect
ratio no longer controls. `GROUP` is a measurement at this tile shape and not a
preference — a tile change is a reason to re-run the sweep.

A grouped map has to know how many tile-*rows* there are, so it can short the
last group when the width does not divide them. That is why `tiles_m` is a
kernel parameter and not just a factor of the item count: a map that aliases the
tail is a wrong `C` rather than a slow one.

## The K pipeline

`load` is filled by the TMA and drained by the MMA; `free` is released by the
MMA's own commit. **The accumulator instruction, not a thread, is what proves
the operand has been read** — a thread arriving there would only prove the
instruction was issued.

The pair's four TMA loads have to complete on *one* barrier for the leader to
know the whole stage is present, and a plain `cp.async.bulk.tensor` completes on
a barrier in the *issuing* CTA's own shared memory. Both halves are named rather
than open-coded: `Semaphore::at_rank` is the leader's copy of the stage barrier,
and `SharedTile::tma_load_2d_arriving_at` is a load that completes there.

The transaction charge is the leader's alone and is its own charge `RANKS` times
over. Every rank derives the same half-stage charge from the loads it just
issued, because a cluster stage is symmetric; the leader scales its own by
`RANKS` to cover the peer's, and the peer drops it. Nothing orders the charge
against the loads and nothing has to — the transaction count is a signed
accumulator and only the totals must agree, which is what lets the charge follow
the calls it is derived from.

One `mma_walk_cg2` per swizzle atom of the stage: `k_walk` describes exactly
one, so a `BLOCK_K` wider than `ATOM_K` is that many walks over the stacked
subtiles the tile is already stored as — the same bytes, the same chunk count,
one barrier instead of several. At `BLOCK_K = ATOM_K` that loop is one iteration
and folds away.

`pair_shape` is evaluated in a `const` block so a tile whose columns name no
shape is a codegen error rather than a `panic!` lowered into device code. The
failure it prevents does not fault: the wrong descriptor into the right
accumulator computes wrong numbers.

## The epilogue is most of the speed

`Tile::drain` moves this warp's band TMEM → registers → `stmatrix` into a
per-warp `[32, 64]` shared tile → 16-byte stores to `C`, four passes of
`STAGE_N` columns per `[32, 256]` band. The obvious epilogue stores 4 bytes a
thread straight out of registers.

**Total memory issue does not fall — it is 128 instructions a thread either
way.**

| | instructions a thread | what one touches |
| --- | --- | --- |
| straight out of registers | **128** `st.global.b32` | 4 B, on **8 discontiguous 16 B runs** — `BaseLdtm` spreads a warp over 8 rows × 4 column-quads |
| `stmatrix` | 16 `stmatrix.m8n8.x4` | 512 B a warp |
| `ld.shared` | 32 `ld.shared.v4.b32` | 16 B |
| `st.global` | **32** `st.global.v4.b32` | 16 B, on **4 contiguous 128 B runs** |

The whole gain is that the global stores are 4× fewer and land on full 128-byte
lines instead of on eight half-filled 32-byte sectors: the band's writes go from
1024 half-full transactions to 512 full ones.

Two instruction widths on top of that, and both are about the *wait*:

- **`tcgen05.ld.16x256b.x8`** (`TmemTile::tile_x8`). The `.x1` load waits after
  each issue, because the registers it waits on *are* the load's return value —
  a `[32, 64]` band costs 16 loads and 16 fully exposed tensor-memory latencies.
  `.x8` is 2 and 2.
- **`stmatrix.m8n8.x4`** (`kittens::ldst::store_tile_x4`). A `reg::Fragment` is
  four `8x8` b16 matrices and `.x2` names two, so `.x4` halves an instruction
  count at identical addresses.

Together those are **+23.1% / +8.8% / +5.1% at 4096³ / 8192³ / 16384³** over the
staged drain without them, which is **1645.0 and 1754.6 TFLOP/s** and **0.939
and 0.958 of cuBLASLt** at 8192³ and 16384³. `ptxas` reads **80 registers and no
spill**.

### The staging tile is per warp, and there is no proxy fence

`StageTile` is one warp's own 4096 B and nobody else's, which is what keeps the
barrier count at zero. `stmatrix` is `.sync.aligned` and so is a convergence
point for the warp that issues it, and `kittens::global::store_shared_rows` is
cooperative rather than collective — so a warp that writes and then reads back
its own 4096 B needs no `bar.sync` at all, only the `bar.warp.sync` that
separates one pass's read from the next pass's write. A CTA-wide `[128, 64]`
tile is the same 16 384 B and would want two block barriers per pass instead.

Four of these fit in one 16 384 B run because a row-subrange of a swizzled tile
*is* a swizzled tile here: `SWIZZLE_128B`'s period is 8 rows and the XOR is over
the row index, so a tile starting at a multiple of 8 rows reproduces the layout
it would have had on its own.

There is **no proxy fence** in the drain and its absence is not an oversight.
`kittens::shared::publish_to_async_proxy` orders a generic-proxy write against
an *async*-proxy read, and both ends here are generic — `stmatrix` writes and
`ld.shared` reads. `stmatrix.sync.aligned` is the convergence. The one hazard
left is the next pass overwriting a tile this pass is still reading, which is
what the `bar.warp.sync` is for.

`GlobalRows` carries `ldc`, so each band lands at its own `(row, column)` origin
in a `C` wider than the tile.

### Rejected: hand the last hop to the TMA engine

Replacing 32 `ld.shared.v4` + 32 `st.global.v4` a thread a band with **one
`cp.async.bulk.tensor`** was built two ways, both on the correctness gate, both
at the same 114 816 B and the same two CTAs an SM, both keeping the `.x8` LDTM
and the `.x4` `stmatrix`. **Neither wins.**

Measured as a paired ratio at a shortened `K`, where the epilogue is a tenth of
the launch rather than a few tenths of a percent of it — above 1.000 is the TMA
arm ahead, median of four paired ratios, two sessions:

| shape | `1/share` | shipped / warp-scope TMA | range | shipped / CTA-scope TMA | range |
| --- | ---: | ---: | --- | ---: | --- |
| 8192² × k1024 | 61–191 | 0.9885 / 0.9835 | 0.9743 – 0.9952 | 0.9948 / 0.9980 | 0.9924 – 0.9987 |
| 16384² × k1024 | 85–427 | 0.9882 / 0.9899 | 0.9853 – 0.9913 | 0.9977 / 1.0061 | 0.9971 – 1.0069 |
| 8192³ | 294–709 | 0.9978 / 1.0018 | 0.9969 – 1.0023 | 1.0014 / 1.0034 | 0.9984 – 1.0062 |

The **warp-scope** arm — the one that keeps this kernel's per-warp staging tile,
pays no block barrier, and is the layout the shipped epilogue uses — loses by
**1.0–1.7%**, in four independent cells across two geometries and two
containers, none of whose ranges reaches 1.000. The **CTA-scope** arm, one
`[128, 64]` box a band through `kittens::epilogue::StoreRing` and eight
`bar.sync` an item, is a tie to a small loss: 0.2–0.5% behind at 8192², with the
two containers disagreeing in sign at 16384². At 8192³ the ratio against the
library does not move at all — 0.943–0.949 shipped, 0.941–0.948 warp-scope TMA,
0.946 CTA-scope TMA.

**The block barrier was the obvious suspect and it is the wrong way round.** The
opcode census reads `bar.sync` 2 / 5 and `bar.warp.sync` 3 / 0 for the two TMA
arms against the shipped 2 / 1, and *the arm that pays the barriers wins*, in
all four cells, **by 0.6–1.6%**. One `[128, 64]` box a band beats four
`[32, 64]` boxes by more than eight `bar.sync` an item cost: the engine's
per-instruction overhead is the term that matters and the barrier is not.

Where the predicted 2.7% went: a doubling probe (a second TMA store per band,
aimed at the cluster's own tile so its bytes stay in L2) prices the hop at
**1.57 / 1.90 / 1.91 / 1.99 µs a tile** against `ld.shared` + `st.global`'s
**2.33 / 2.41 / 2.56 / 2.58** — 0.42–0.75 µs, 6–11% of a ~7 µs epilogue, under
1% of the launch at its most generous. 2.7% was the value of the term going to
*zero*, and the engine takes it to about three quarters of itself.

And the realised value is below zero, which the doubling probe is structurally
unable to see. A *doubled* store is issued and left in flight beside work that
continues; a *replacing* store is on the critical path of the next band. Depth 1
is forced by the 1920 B of headroom, so `acquire` is
`cp.async.bulk.wait_group.read` at zero groups: band `k + 1`'s `stmatrix` cannot
start until the engine has finished *reading* band `k`'s buffer, four times an
item, where `store_shared_rows` retires into the memory pipeline and blocks on
nothing. The kernel would trade 64 pipelined instructions a thread for one
instruction and one exposed engine latency, and the exposed latency is worth
more.

On this evidence a kernel reaching for `StoreRing` should reach for its `Cta`
scope.

## Numerics, and an exact check

`C` is bf16 and the accumulator is not. `Accumulator` is `BLOCK_N` fp32 columns
of tensor memory, and the single `cvt.rn.bf16x2.f32` is inside
`kittens::ldst::store_tile_x4`, on the way into shared. So there is exactly one
rounding in the kernel, and `check_c` puts the same one in the reference — which
is what lets the comparison stay `==` on 16-bit words with no tolerance at all.

`to_bf16` is round-to-nearest-even, ties to even included, exactly what the
`cvt` does. It is exact for every value the operand generators produce, since
their low 16 mantissa bits are already zero; on the *output* it is not exact,
and that is the whole reason it has to be in the reference.

The alternative — widen the observed bf16 and compare against the fp32
reference within a tolerance — is strictly weaker, because a tolerance wide
enough to admit correct rounding also admits everything smaller than it, and a
wrong tile that happened to land close would pass.

What the `==` *is* blind to is resolution: two fp32 accumulators differing by
less than half an ulp of bf16 round to the same word, so an error under about
0.2% of a value's magnitude is invisible. Every failure mode this gate exists
for — a wrong coordinate, a wrong stride, a dropped or doubled tile, a wrong
operand half, a mis-walked K — moves an element by far more than that or leaves
it at zero.

`check_c` returns the worst relative error against the exact fp32 reference,
which is the representation error of a bf16 `C`. It is reported and not
asserted: it is a property of the output format and of the magnitudes this
reference produces, not of the kernel.

### The operand generators, and why the moduli are 7 and 21

    a_value(row, depth)    = ((row * 5 + depth * 3) % 7)  - 3     ∈ [-3, 3]
    b_value(column, depth) = ((column * 4 + depth * 5) % 21) - 10 ∈ [-10, 10]

Every operand is exact in bf16 (which holds every integer to 256) and every
partial sum stays well under fp32's exact integer range of 2²⁴, so the whole
GEMM is exact and the host compares with `==`. A mismatch is a wrong coordinate,
a wrong stride or a wrong operand half, and never a rounding artifact that has
to be argued about.

**Both generators must depend on `depth`, and that is a bug this reference
already had.** `b_value` once read `(column * 3 + depth * 5) % 5`, and
`depth * 5 % 5` is identically zero — `B` was constant along `K`, so the exact
check was blind to precisely the axis `mma_walk_cg2`'s chunk arithmetic
computes. A kernel reading one plane of `B` every step, walking K backwards, or
aliasing the ring slot for the K index passed anyway.

**The moduli share the factor 7 on purpose.** `A`'s values over one period of
`depth` are `0..7` shifted to sum to zero, so if `B`'s `depth` period were
*coprime* to 7 the sum over one combined period would factor as
`(Σ A)(Σ B) = 0` — the dot product would then be a function of `K mod 7·P`
alone, bounded independently of `K`, and two different K walks would collide by
accident. Sharing the factor defeats that: the partial sums grow with `K`, and a
swept check of every legal `K` up to 8192 finds no wrong K walk this reference
cannot see.

The periods are also what makes the reference cheap. `a_value` repeats every 7
rows and `b_value` every 21 columns, so there are 147 distinct dot products at
any size; every element of `C` is still compared against its own expected value,
in the same summation order.

## The correctness run

Two sizes, four traversals, checked, nothing timed.

- **512×256×256.** Two clusters' worth of `M`, so a cluster's rank arithmetic
  and the `item → tile_m` half of the map both run against a second tile-row.
  `K` is four `BLOCK_K` stages against a three-deep pipeline, so the ring wraps
  and `wait_recycled` is on trial rather than skipped.
- **4096×4096×256**, whose only job is to give every cluster **more than one
  work item**. The first size is two tiles against a 148-cluster grid, so it
  never enters `pipeline::run`'s loop a second time — it would pass identically
  against a launch-per-tile kernel, which is exactly what makes it not a test of
  this one. Every failure mode the persistent scaffold introduces lives at the
  *item boundary*: a barrier re-armed while a peer is still filling it, an
  accumulator not started fresh for the next tile, an epilogue racing the next
  item's first loads. Each is a deadlock or a wrong `C`, and each needs a second
  item to happen at all. 256 tiles over 148 clusters is two items for most and
  one for the rest, so the ragged tail is under test too.

Both sizes report the items their clusters walked, so a size that quietly
stopped exercising the loop — because `MAX_CLUSTERS` moved, or because the
tiling did — says so in the pass line instead of in nobody's memory.

`CHECK_GROUPS` is `[GROUP, 1, 3, 6]`. `GROUP` is first because it is the one
that ships, and a gate that never launches the shipped configuration is not a
gate on it; it is also the least searching of the four. `1` is row-major, so a
regression in the map still fails against the traversal this kernel had before
it was grouped. `3` and `6` are **not powers of two, which is the point**: every
`M` this project runs is, so `tiles_m` is too, so a width of 8 always divides it
and the short last group — the one branch in the map, and the one that turns a
wrong width into a tile computed twice — would never execute. At 4096's 16
tile-rows, `3` leaves a final group of one row and `6` leaves one of four; at
512's two tile-rows both are taller than the grid and take the saturating path.

The traversal is under the same gate for the same reason as the arithmetic: a
wrong item map is a tile computed twice and one computed by nobody, both silent
on the device, and the element-by-element `==` is the only thing that sees
either. The epilogue is under it for a sharper reason still — `stmatrix` and the
read-back address the same swizzled tile through two different derivations, the
four staging tiles are row-subranges of one 16 384 B run, and the loop reuses
each tile four times with only a `bar.warp.sync` between the read of one pass
and the write of the next. Every one of those is a wrong `C` rather than a fault
if it is wrong.

Sizes are rejected rather than launched when they do not divide the tiling: a
cluster owns a whole `2·BLOCK_M` by `BLOCK_N` tile and a stage is a whole
`BLOCK_K`, and the kernel bounds-checks none of it. A traversal width of zero is
rejected for the same reason — `grouped` divides by `group * columns` and the
device checks nothing.

## Launch plumbing

Neither tensor map states a box. `GlobalLayout::tensor_map` takes the tile the
kernel loads into and reads the box, the swizzle and the data type off it, so
`ATile`'s and `BTile`'s own constants are what reach the descriptor and not
numbers the launcher wrote down. `A` is `[k, m]` and `B` is `[k, n]` in the
driver's fastest-first dimension order, which is the same order the kernel gives
its `tma_load_2d` coordinates in — both operands K-major, so the MMA carries no
transpose bits and computes `A·Bᵀ`.

A tensor map's box is `[R, SUBTILE_COLS]`, one swizzle atom wide whatever the
tile's own columns, so **`BLOCK_K` does not reach the descriptor at all**: a
two-atom stage is two boxes through the same map, which is what `tma_load_2d`
already issues per stacked subtile.

Everything outside the item loop is `attach` and `release`. `alloc_cluster` is a
whole-cluster collective with a `cluster_sync` in it and must not be inside
anybody's loop; `release`'s `cluster_sync` is for the cluster that got **no
items at all** — a capped grid can leave a pair having allocated, never looped,
and still owing a deallocation in step with its peer.

## What this kernel had to reach past the library for

Nothing. There is no open-coded index arithmetic and no gap list. The epilogue
was twelve lines of hand-written addressing against `RegTile::coordinate`; the
cluster-scope tensor-memory allocation beside it is `kittens::tmem::alloc_cluster`
/ `dealloc_cluster`, and this file is where that allocator's participation rules
were worked out against silicon.
