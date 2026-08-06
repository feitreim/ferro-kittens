# `ldst.rs` — warp-scope register↔shared movers

## One address derivation, two directions

`ldmatrix` and `stmatrix` share an address convention: the 16 addresses come
from lanes 0..15 while the data is spread over all 32. So the two directions are
one derivation — `fragment_address` — with the data flowing opposite ways, and
they cannot drift apart. A second derivation for the store side is exactly the
thing that could disagree with the load side, and there is not one.

That includes both wide forms. `stmatrix.m8n8.x4` takes 32 addresses from all
lanes, eight per matrix, its four matrices being the two slots' two halves in
order; `.x2` takes 16 from lanes 0..15 and is issued twice, once per slot. Lane
`l` of the `.x4` therefore supplies what lane `l % 16` supplied for slot
`l / 16`, which is what the test
`x4_addresses_are_the_x2_addresses_restacked` pins. `ldmatrix.m8n8.x4` takes
its addresses the same way, so that one test is the derivation for
`store_fragment_x4` and `load_fragment_x4` alike, and the wide pair can no more
drift apart than the narrow pair can.

Because a symmetric derivation feeds both, an accumulating MMA reads a stored
operand exactly like a TMA-loaded tile, and a load sees a TMA-loaded tile
exactly like a drained accumulator.

## Lanes 16..31 form addresses that are ignored

For the `.x2` form, lanes 16..31 land back on the first matrix's addresses. The
instruction ignores them on `sm_100a`, but they are still formed, and they stay
inside the `[row + 16, column + 16]` block the caller promised (the test
`no_lane_addresses_outside_the_block`). That is what lets the safety contract be
a statement about the *tile* rather than about which lanes the hardware happens
to read.

## Bands mean the same thing in TMEM and in shared memory

Each matrix instruction moves one `[16, 16]` block, and `load_tile` /
`store_tile` compose them into an `[M, N]` band out of the same helpers
(`place_block`, `take_block`) that `TmemTile::tile` uses over the drain. A band
therefore cannot mean one thing in TMEM and another in shared memory.

## Why the vector pair is not a matrix instruction

`load_vec` / `store_vec` are plain scalar accesses. A `ColVec`'s values are one
element each, at columns no `ldmatrix` shape describes, and under `BaseLdtm` a
column depends only on `lane % 4` — so the 8 lanes of a column group all want
the same address, which shared memory broadcasts from one bank read. There is no
shuffle involved and nothing for a swizzle to spread.

On the store side the same replication makes every column written by the four
lanes of one quad, all with the same value. The redundancy is the layout's, not
the loop's, and it is what keeps the write a plain store rather than a
lane-masked one.

## Why the row pair is not that pair transposed

`load_row_vec` / `store_row_vec` share the vector pair's shape and none of its
addressing. A `ColVec`'s column depends only on `lane % 4`, so its store is
eight lanes issuing one address; a `RegVec`'s row depends on `lane / 4`, so its
store is `M / 8` elements per lane, 8 rows apart — a scatter, with no run to
widen and nothing for shared memory to broadcast.

The replication is on the other side too, and it costs a lane mask the column
direction does not need: the four lanes of a quad hold the *same* row, so only
the one `RowLayout::owns_row` names writes. That predicate is on the trait for
exactly this, and an impl returning `true` everywhere would turn one store into
`M`-many.

The load direction is symmetric after all. Every lane reads, replicas included,
because one copy per lane holding the row is what makes a `RegVec` well-formed,
and a quad's four identical addresses come out of one bank read.

## The `Element<Unpacked = [f32; 2]>` bound

`stmatrix`/`ldmatrix` `.m8n8.x2` move two *b16* matrices, so the fragment path
holds only for elements packing two fp32 per word. That is expressed as a trait
bound rather than an assertion: a 4-per-word element does not typecheck against
these functions, and so gets the instruction shape it actually needs instead of
quietly moving half the bytes.

## `scatter_tile` is the route for the elements that bound excludes

The bound above is a fact about `stmatrix`, but for a long time it read as a
fact about the *library*: an fp32 band had no way into a shared tile at all, so
an fp32 epilogue stayed on `store_rows`' per-value **global** stores — eight
discontiguous 8-byte runs a warp, where the staged route writes four contiguous
128-byte ones. #116 measured that difference at 20.43 µs/tile against 6.68 for
bf16, and the whole of what was missing was a filling instruction, not a design.

`scatter_tile` is that instruction, and it is deliberately not a new one: a
store into shared memory at any element is an ordinary `st.shared`, so the
function is `global::store_rows`' loop with `SwizzledChunks::element` where the
`GlobalRows::at` was. The two claims it rests on are addressing claims, and both
are host tests — `element` splits a column into the chunk holding it and the
offset inside it (`every_column_owns_its_own_bytes_of_the_chunk_holding_it`),
and a warp's scatter covers its band exactly once
(`a_warps_scatter_covers_the_band_exactly_once`).

It is generic in the layout as well as the element, where `store_tile` is
`BaseLdtm`-only. That asymmetry is the point rather than an oversight:
`store_tile`'s addressing *is* an `ldmatrix` shape and cannot be anything else,
while a scatter only ever asks the layout which `(row, column)` a value is.

**What it costs, per band, against the `stmatrix` route it parallels.** A
`[16, 64]` fp32 band is 64 values a thread; at one store each that is 64
`st.shared.b32`, where the same band at bf16 is 8 `stmatrix.m8n8.x4`. The gap is
the whole risk in this route, and the pair rung halves it: `BaseLdtm`'s
`CONTIGUOUS_VALUES` is 2, so two values are one `st.shared.v2.f32` and the band
costs 32. At bf16 the same rung is not a vector instruction at all — two
adjacent bf16 *are* one 32-bit word — so it is a plain `st.shared.b32` of one
`pack`, which halves the `cvt` count too.

Where `global::store_rows` has to test its cursor at run time to earn the same
pairing, this knows the answer statically: a chunk is 16-byte aligned, a pair
starts at a multiple of `CONTIGUOUS_VALUES` by that constant's own contract, and
`2 * E::BYTES` divides 16 at both element widths. So the decision is a `const`
and neither path carries a branch.

Under `BaseLdtm` a lane's pairs are scattered across the row, so the 32 lanes of
a warp land on half the banks twice over — a 2-way conflict, not a broadcast and
not a clean sweep. The trade is that against, and the global half it buys is
`store_shared_rows`' unchanged 16-byte contiguous stores.

**What it does not do is reduce the band.** A scatter holds exactly what
`store_rows` held; the register pressure an fp32 drain has comes from the band's
width, and it falls when the *epilogue* narrows its pass to the staging tile's
width — which is what having a staging tile makes possible, and not what this
function does by itself. `device-tests`' `scatter drain` / `register drain` pair
is the same rectangle by the two routes at 32 and at 128 columns, so the
`regcount` table reads the difference directly: **122 registers against 56** at
128 fp32 columns, and 44 against 40 at 32. The gap widens with the band because
the band is what the register route has to hold, and that is the same mechanism
that takes the downstream kernel's 512 B frame to zero.

Read those rows only with the probes' identity fill unrolled (#166). Before
`fa70e35` every one of them — including the bf16 `shared_drain_wide`, which has
nothing to do with this — carried the same 272 B frame, because a rolled walk
over a `RegTile` homes the band to a depot and the *fill's* depot swamped every
drain's cost. A probe that means to price one thing has to unroll everything
else, and this pair is the case that made that concrete.

## The measured downstream answer, which is not the one the gap predicted

`oxide-train`'s `gemm_tcgen05_f32_optimized` was converted to this route and
measured on a B200. The register story landed: **120 registers and a 512 B
`.local` frame → 96 and none**, the frame being a band that did not fit, which
staging removes twice over — a pass narrows from `[32, 64]` to `[16, 64]`, and
the accumulating mode stops holding `C`'s band because `accumulate_shared_rows`
does the fold in memory. The milliseconds did not: against three baseline runs
the store arm is ~5% faster at 4096³ and a wash above it, and the accumulate arm
is 15% slower at 4096³ and 8192³.

The cause is the tile's *height*, and it is worth stating because it bounds when
this route is the right one. Four bytes an element buys half the rows in the
same shared bytes, so an fp32 band takes eight passes where the bf16 band takes
four, and a pass is a serial
scatter → `ld.shared` → global → `bar.warp.sync` chain. Twice the passes is
twice that exposed latency, and the accumulating variant puts an `ld.global` on
it as well. #116's −19% for bf16 was measured with a tile wide enough to
amortize the pass; that kernel's operand rings leave 18 344 B for four warps,
which at fp32 is not. **The staged drain is a trade against pass count, not
against element width** — an epilogue that can spend `[32, 64]` fp32 of shared
memory is the one to re-measure this on.

## The load side's `.x4` is a wash, and the asymmetry is now chosen

`load_fragment_x4` / `load_tile_x4` are the load direction at the store side's
width: one `ldmatrix.m8n8.x4` per `[16, 16]` block instead of two `.x2`s a slot
apart. They needed no addressing of their own, per the first section of this
file — one restacking serves both wide forms — and the four destination
registers are the four matrices in the order `store_fragment_x4` supplies them.
The `ldmatrix x4 map` and `ldmatrix x4 map wide` device cases pin that order
against the same host expectation the narrow pair answers to, so a wide load
returning its registers rotated is a named tile position arriving in the wrong
one rather than a tolerance somewhere downstream.

**They are not what ships, and after #131 that is a decision rather than an
omission.** #116's `.x4` and #117's `.x8` both paid by removing *waits*. An
`ldmatrix` has none to remove — it is a plain shared read — so halving the
instruction count can only buy issue slots, and what it costs is liveness of
the shape the drain's `.x8` carries: four matrices arrive at once where two let
the compiler fuse a slot through to its consumer.

The instrument is `softmax`, whose three passes are all `load_tile` and whose
only other work is two `ex2.approx` an element.
`experiments/src/softmax_x4.rs` emits the teaching crate's own device body at
the other load width as a second entry in the same bundle, so the two arms are
one binary and every pair of measurements is adjacent in time rather than two
containers apart — `scripts/modal-run bench --case ldmatrix`, four whole
measurements of each arm round-robin, each the minimum of 30 timed launches
after 5 warm-up, every one of them checked against the CPU reference first.

| rows × 128 × 2 planes | blocks | `.x2` ms | `.x4` ms | `.x2` GB/s | `.x4` GB/s | `.x4`/`.x2` | lowest | highest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 64 | 0.0106 | 0.0105 | 397.2 | 398.4 | 1.0124 | 0.9758 | 1.0249 |
| 32768 | 512 | 0.0166 | 0.0168 | 2020.4 | 1993.5 | 1.0176 | 0.9904 | 1.0212 |
| 524288 | 8192 | 0.1254 | 0.1268 | 4281.0 | 4235.6 | **1.0110** | 1.0097 | 1.0148 |

The last three columns are the four *paired* ratios of `.x4`'s milliseconds to
`.x2`'s, so above 1.000 is the narrow load ahead; the millisecond and GB/s
columns are each arm's own median and can disagree with the pairing, which is
what the first row does. Read that row and the second as unordered: a ratio
below 1.000 is among the four in both, which is the only honest reading of a
difference this size at this sample count. The third row is ordered — all four
pairs land in 1.0097–1.0148 — and it says the wide load is **1.1% slower**
where the kernel is bandwidth-bound, at 4236 GB/s against 4281.

**The liveness cost is the one that showed up**, off
`scripts/modal-run regcount`:

| kernel | regs | spill | frame |
| --- | ---: | ---: | ---: |
| `softmax_rows` (`.x2`, both crates) | 32 | 0 | 0 |
| `softmax_rows_x4` | 40 | 0 | 0 |
| `ldmatrix_map`, `ldmatrix_x4_map`, and both `_wide` | 32 | 0 | 0 |

Eight registers a thread, no spill and no frame either way — and *not* an
occupancy step: the kernel's 32,776 B shared plan fixes it at 6 CTAs an SM, and
40 registers over 128 threads is nowhere near what would move that. The four
`device-tests` probes read identical because a probe that dumps each block as it
arrives holds nothing; the cost is only visible where a real kernel keeps
statistics live across the walk, which is the same thing `docs/kernels/softmax.md`'s
rung ladder says about `CHUNK`.

So the prediction landed on the wrong side of zero rather than on it, and 1.1%
is small enough that what this rules out is worth more than what it says: there
is no issue-slot win here to go looking for on a kernel of this shape. `.x2`
ships, `.x4` is in the library as the measured alternative, and the two
directions differ on purpose — the store side's `.x4` removed a wait, and this
one had none to remove.

## What orders an `stmatrix` before the `ldmatrix` that reads it

`load_fragment`'s safety contract asked the caller for
`fence.proxy.async.shared::cta` before an `ldmatrix` read a tile `stmatrix` had
written. That fence orders the *generic* proxy's writes so the **async** proxy
can see them — the TMA engine, a tcgen05 MMA — and it is what `store_fragment`
and `tma_store` correctly owe. `stmatrix` and `ldmatrix` are both generic-proxy
accesses, warp-level shared-memory stores and loads executed by threads, so
between them that fence orders nothing at all. `publish_to_async_proxy`'s doc
states the rule and names a generic-write/generic-read pair as one of the two
cases where no fence is owed; this pair is that case.

What such a pair owes across threads is a **barrier**, which the contract did
not say — so the sentence was either a redundant demand or an understated
obligation, and those are very different mistakes. Nothing in tree could tell
them apart: every load in the repo — `load_fragment`, `load_tile` and the `.x4`
forms of both — reads a TMA-staged tile, and every `stmatrix` write is read back
by plain `ld.shared`, by the TMA store engine, or by an MMA. The two directions
had never met.

The question is the same one at both widths, and so is the answer. `.x2` and
`.x4` differ in how many addresses the warp supplies and in nothing that touches
a proxy or a barrier, so the obligation below is `load_fragment`'s and
`load_fragment_x4` inherits it verbatim. The case runs `.x2`, which is what
ships.

### Measured

`device-tests`' `stmatrix into ldmatrix`. Warp 0 fills a `[64, 64]` tile twice —
from a *stale* generation of position identities, published with a fence and a
barrier so every warp is known to hold it, and then from the *fresh* one — and a
reader warp reads all sixteen blocks back. The two generations are the same
addresses carrying identities **37 columns apart**: odd, so the rotation is a
bijection on the tile's columns, and not a multiple of 16, so no misaddressed
`ldmatrix` block can produce that displacement. A value therefore names the
generation that reached it rather than merely failing to match, and an
addressing failure is a third answer rather than a second one. B200, `sm_100a`;
4096 values a row.

| what varies | row | result |
|---|---|---|
| control | the read runs before the store, barrier-ordered | **every value stale** |
| **same warp** | nothing between them | every value fresh |
| | the proxy fence only | every value fresh |
| | `bar.sync` only | every value fresh |
| | both — what the contract demanded | every value fresh |
| **cross warp** | `bar.sync` only | **every value fresh** |
| | both — what the contract demanded | every value fresh |
| | nothing between them | **256 of 4096 stale** |
| | the proxy fence only | **128 of 4096 stale** |

So: the fence is redundant and the barrier is the whole of it. Within one warp
the two instructions are that warp's own program order and nothing is owed
between them; across warps `bar.sync` alone is enough, and it is the only thing
that was ever load-bearing. `load_fragment`'s contract now names the barrier,
scopes it to a writer that is not the reading warp, and cites the rule.

Three things worth reading off the table rather than out of the verdict:

- **The two ungated rows fired.** They are races and gate nothing, but they came
  back with real stale values rather than the polite nothing a negative control
  usually returns — so the hazard is not hypothetical and this case observes it
  outside the arrangement built to force it.
- **`the proxy fence only` reading stale is the direct half of the answer.** The
  fence the contract demanded was in the kernel, in the position the contract
  named, and the read still saw the previous generation. That is not an
  inference from which proxy the instructions use; it is the fence failing to
  order the pair, measured. The two rows above it — `bar.sync` alone, clean over
  4096 values — are the other half.
- **The control cannot decline to fire, and that is the point of it.** It reads
  before the store with a barrier making that the order that happens, so it is
  an ordering rather than a race. A dropped-wait style control that merely
  *hopes* to observe a hazard can come back clean and prove nothing — which is
  exactly what happened to `sttm into mma`'s, and to `tmem across warpgroups`'
  before it. The two race rows here happened to fire; the control was never
  going to have to.

The race rows' counts are the only numbers in the table that are not
reproducible. An earlier run of the same nine rows had `nothing between them` at
128 stale and `the proxy fence only` clean — the same verdict, since a race that
resolves in order is not an ordering either way, and since nothing gated rests
on either. What does not move between runs is the seven gated rows.

Not established: the same question with the reading warp in another CTA of a
cluster, across a warpgroup boundary at 256 threads, and any of it on silicon
that is not a B200.

## `store_packed_x4` computes nothing correct in tree

Nothing in tree computes a right answer with `store_packed_x4`, and that is not
a defect. The only producer of pre-packed b16 out of an fp32 accumulator would
be a convert, and if a convert ran then `store_fragment_x4` is the function that
wanted it. It exists so `experiments/`' `pack16` rung can hold every other
instruction in a drain fixed and take the `cvt` column to zero — see
`TmemTile::fragments_pack16_x8`. Because it is `store_fragment_x4` with the four
`E::pack` calls removed and nothing else changed, the pair is an ablation of the
convert rather than a second store path.

## The `stmatrix` inline-PTX workaround

`stmatrix_m8n8_x2` and `stmatrix_m8n8_x4` are hand-written `ptx_asm!` rather
than calls into cuda-oxide, because the LLVM stmatrix declaration cuda-oxide
emits does not resolve for `sm_100a`. Observed at cuda-oxide `b099f64` and
**still present at the pinned `20a5616`** — same `.extern .func`, same `ptxas`
line — so this is not a workaround waiting to be dropped at the next bump.
cuda-oxide `20a5616` does ship `stmatrix_m8n8_x4` in `generated/stmatrix.rs`;
its declaration fails the same way the `.x2` one does.

The load direction needs nothing of the kind: `cuda_device::wmma::ldmatrix_x2`
lowers cleanly for `sm_100a`. That it lives in a `wmma` module is a filing
accident — `ldmatrix` is a plain shared-memory read and has nothing to do with
the wmma MMA path this crate does not use.
