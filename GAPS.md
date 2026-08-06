# Gaps vs ThunderKittens

Measured against HazyResearch/ThunderKittens `main` (~21.3k lines of headers under
`include/`, plus `prototype/`). `wc -l src/*.rs` answers for this side, so that
number is not maintained here: it was 1851 when this file was written and 10 156
on the date below.

The size difference overstates the functional gap and understates the structural
one. TK's surface is a **cross product** — every op is templated over element
type × layout × scope (thread/warp/warpgroup/`group<N>`) × tile-or-vector — so
one conceptual op like `row_sum` expands into dozens of instantiations. This port
is a **vertical slice**: one element type (bf16 operands, fp32 accumulate), one
swizzle (128B), one fragment layout, one MMA family (tcgen05), the ops the
working kernels actually needed. Most of what follows is widening that slice,
not adding new concepts — which is also why quadrupling the line count has
closed rather less of the cross product than the ratio suggests.

Gaps are marked **[NON-GOAL]** where the omission is a deliberate consequence of
targeting Blackwell/tcgen05 only, **[SCOPE]** where it belongs to a different
layer in a Rust world, and left unmarked where it is a genuine hole.

## How to read this file, and when it was last true

**Audited against the source on 2026-07-29.** That pass read all of `src/`
whole for the first time since the library grew, and it widened what this file
is for: sections 1–6 are still the ThunderKittens diff, and **§7 is new** — the
coherence of the surface itself, which no comparison against another library
would have found. Two of the three worst finds were the failure mode below
rather than a missing feature: §2.2 still said `GlobalRows` was fp32-only after
#108 had given it an element, and §4 still said the pipeline scaffold had no
caller after `gemm` had been running on it since #83. Neither is a slow rot —
both landed on 2026-07-28, the same day #65 re-read the entries they falsified.
The lesson is the one #65 drew and not a new one: the entries went stale inside
a day, and no commit message contradicted either.

**Previously audited 2026-07-28** (#65). Every entry was checked by
reading the module it names, not the log — the failure this file keeps having is
an entry that still says *missing* for something that shipped, and no commit
message contradicts one of those. §3.2 read "Missing: … generic `row_reduce`"
for the whole of #6's life; §1.4 read "Absent entirely" three PRs after the type
landed; §1.5 described a builder that had become a type. Each was found
incidentally by someone working nearby.

Two habits follow from that, and they are the only maintenance this file asks
for. **Counts are not restated here** where the tooling already prints them:
`cargo test` reports its own, `device-tests` prints `N of M cases passed`, and
`gh issue list` is the live version of the work order below. **Claims that
something landed are named in `tests/gaps.rs`**, a test whose whole body is a
generic function that is never instantiated — so renaming an exported symbol
breaks the build beside the section asserting it. Absence cannot be checked that
way (there is no way to write "no second `FragmentLayout` exists"), so the
*missing* half stays prose, and stays dated.

---

## 1. Structural gaps

These change type signatures across the crate, so they are cheapest to do early.

### 1.1 No group/scope parameterization — retired as filed (#3)

TK's `group<N>` templates every memory, register, shared, and MMA op over warp
count, with `warp = group<1>` and `warpgroup = group<4>` as aliases. One
implementation serves every collective width.

We hardcode scope per function: `ldst` is warp-scope, `mma` is warpgroup-scope,
`pipeline` is block-scope, and the choice is baked into each function's index
math rather than named in its type.

Rust has no partial specialization, so the equivalent is a `trait Scope { const
WARPS: usize; }` with the lane/warp arithmetic behind associated consts. Worth
doing before the op surface grows, because retrofitting it means touching every
signature. **~150 lines + a mechanical sweep.**

**#3 did not do this**, and closed on the argument against it. The window the
"before the op surface grows" clause names has passed — #5, #6, #7, #31 and #38
roughly tripled the surface — and, more to the point, a `Scope` of
`{ WARPS, THREADS, rank(), sync() }` cannot express the one op that actually
wanted block scope. Warps cannot shuffle to each other, so a fold across them
needs *storage* between two barriers, which no such trait has. What shipped is
`sync::block_reduce::<Op, WARPS>` over a `SharedVec<F32, WARPS>` — the specific
collective, not the parameterization.

**#123 then did it once, for one type, and the argument it settled is not the
one #3 lost.** `epilogue::StoreRing` takes a `Scope` — `Cta` or `Warp`, a trait
of `{ issuing(), converge() }` — because a staging tile need not be CTA-wide and
a ring hard-wired to `bar.sync` could not express `gemm`'s per-warp tile at all.
That is exactly the trait #3 argued against, and it works here for the reason
#3's version did not work for a block reduction: a store ring has *its own*
storage already, so the scope only has to name a barrier and an issuing thread,
which is all `{ WARPS, rank(), sync() }` was ever able to say.

So the crate now spells scope **four ways**, and they do not compose:

| where | spelling | rungs it covers |
| --- | --- | --- |
| `epilogue::StoreRing` | `SC: Scope` | warp, CTA |
| `pipeline::Job` | `const RANKS: u32`, branched in `boundary` | CTA, cluster |
| `sync::block_reduce` | `const WARPS: usize` | CTA, of a named width |
| `global::store_shared_rows` | `const THREADS: u32` + a `thread` argument | any cooperative set |

`Scope` and `RANKS` are the same decision at adjacent rungs of one ladder —
warp < CTA < cluster — implemented twice with no rung in common, and neither can
express the other's. Everything else is still baked into its index math:
`tmem::alloc_block` *is* `warp_id() == 0` plus `sync_threads`, `alloc_cluster`
*is* that plus `cluster_sync`, and neither says so in its type. Filed as
**#124**.

### 1.2 Single element type

| | TK | ferro-kittens |
| --- | --- | --- |
| Operands | bf16, half, fp8 (e4m3/e5m2/e8m0), fp4 (e2m1), int8, tf32, fp32 | bf16, fp16 |
| Accumulate | fp32, fp16, int32 | fp32 only |
| Complex | `crt`/`crv`/`cst`/`csv` throughout | — **[SCOPE]** |

`Element` now carries the byte width, the fp32 → element pack, and (via
`MmaElement`) the tcgen05 operand kind, so shape math, `ldst`'s pack, and
`mma`'s bounds are all generic (issue #2 did the trait work; #16 adds fp8).

**This entry read "`Bf16` is still the only *operand* impl" until #169, and it
had not been true since `F16` landed** — `impl MmaElement for F16` sits eight
lines above `impl MmaElement for Bf16` in the same file, `MMA_KIND = 0` with
`Tcgen05ElementType::F16` telling the two apart, and `experiments`' `gemm_sol`
rung is benchmarked against a cuBLASLt FP16 baseline *because* the port's
operands are fp16. The failure is this file's usual one and not a slow rot: the
line was written when it was true and nothing re-read it. Two operand impls are
also what makes the sentence worth checking — one impl is a sentence about a
type, two is a claim about a trait. tf32 (`KIND = 1`) and fp8 are the genuinely
absent ones, and tf32 is the one downstream keeps asking for: `oxide-train` #67
was filed believing its fp32 GEMM needed it, when that kernel's operands are
bf16 and only its `C` is fp32 (see 2.6). `F32` is the third `Element`
since #3, and deliberately not an `MmaElement`: it exists so a statistic — or,
since #174, a whole epilogue band — can be staged in shared memory without
rounding, which is a storage question and not an operand one. One bf16 fact remains hardcoded behind an assert rather than
guessed at: `mma`'s K=16 chunk geometry assumes 32 bytes per chunk. The
*instruction* left that list with #12 — the walks issue through
`MmaElement::mma`, which routes tcgen05's `KIND` off the element. FP8 is the
consequential one: it
changes the swizzle atom's element count, needs a store path whose `stmatrix`
form matches 4-per-word packing, and adds block-scale operands (see 3.3).

### 1.3 Single swizzle mode, single register layout

TK's `st` takes `swizzle_bytes ∈ {32, 64, 128}` plus an unswizzled mode; `rt`
carries a `row`/`col` layout parameter with `swap_layout`/`transpose` to move
between them; `rv` has three layouts (`align`/`ortho`/`naive`) matching how a
vector pairs with a tile.

We have `Swizzle128B` as the sole `Swizzle` impl (the trait is there) and
`BaseLdtm` as the sole `FragmentLayout`/`RowLayout`/`ColLayout` impl — #1 made
the layout a parameter and #23 opened its shape set to every multiple of 16 up
to 512, but a second impl is still unwritten. The 128B-only restriction
is documented as honest rather than
accidental — the subtile scheme depends on the 128-byte atom — but 64B tiles are
what you want for narrow operands, and no-swizzle for staging tiles that never
feed an MMA.

The swizzle half is **#14**, and it is the larger of the two jobs: `ATOM_BYTES`
is both the swizzle period and a TMA box's width, and an unswizzled mode needs
those separated first — which is why `SharedVec` routes around `Swizzle` rather
than borrowing it (1.4). A second `FragmentLayout` is **filed nowhere**, and
3.4's `transpose`/`swap_layout` wait on it.

`ATOM_BYTES` turns out to be doing a *third* job, and it is the one with a cost
already on the clock: it is also the **walk step**. `SharedTile::k_walk` asserts
`C * E::BYTES == S::ATOM_BYTES`, so at bf16 a K-major walk is 64 wide and
nothing else, and `gemm` reaches a `BLOCK_K = 128` stage by re-wrapping each
subtile as a narrower tile type to get a walk out of it
(`experiments/src/gemm.rs:1199-1209`, and `experiments/src/gemm.rs:5846-5852`
says why). The
arithmetic to cross a stacked subtile already exists — `mma::k_major_offset`
does it for the typed walks — so a multi-atom `OperandWalk` is separable from
the swizzle work. Both halves are recorded on #14.

### 1.4 Shared vectors — the type landed (#13), the ops did not

TK has `sv` as a first-class type with its own maps, reductions, conversions and
TMA paths — it's how row statistics, biases and norm accumulators get staged.
This entry read "Absent entirely" until #65; the type shipped with #53.

`SharedVec<E, N>` is a base pointer and a compile-time length: `at`/`get`/`set`
for scalar access, and four TMA paths (`tma_load`, `tma_load_2d`, `tma_store`,
`tma_store_2d`) that are **one instruction each**, because an unswizzled box has
no atom to be cut into the way `SharedTile::tma_load`'s stacked subtiles are. It
carries a `TileBox` impl, so `GlobalLayout::tensor_map` derives its descriptor
from the destination type exactly as a tile's is derived (1.5), and
`ldst::load_vec`/`store_vec` bridge it to the `ColVec` a per-column op consumes
— which is how `layernorm`'s `gamma`/`beta` reach the registers that multiply by
them. Device cases `shared vector round trip` and `shared vector row` cover
rank 1 and rank 2.

It deliberately does **not** go through `Swizzle`, and that is the decision
worth carrying forward rather than an omission: `ATOM_BYTES` is both the swizzle
period and the width of a TMA box, and a one-row run of elements wants neither —
its box is `N` wide, a number no mode marker can hold. What `ATOM_BYTES` should
mean in the absence of a swizzle is a real question about *tiles*, and it is #14's.

The box rules (`N * E::BYTES` a multiple of 16, `N <= 256`) sit on the four
transfers rather than on `from_raw`, because the one use that never meets a
descriptor is `sync::block_reduce`'s scratch — a two-warp block's partials are
8 bytes and could not otherwise be constructed.

What #13 asked for and did not get is the other half: the map and reduction
instantiations, on shared vectors **and** shared tiles. The issue closed on the
type. **#130** now carries the vector half — with two smaller holes beside it
that the audit found in the same corner: a `SharedVec` cannot be sliced, so
`layernorm` reaches a chunk of its parameter vector by laundering `at()` through
`from_raw`, and `RegVec` had no bridge to memory at all. The **global** half of
that last one landed (2.2, `global::load_row_vec`/`store_row_vec`), and the
`ColVec` side of the same corner closed its global half with #172's
`global::load_cols` (2.2); the shared halves and the slice have not. The
shared-*tile* half is still tracked nowhere (3.1).

### 1.5 Global layout — a type since #8, and still bf16-only

TK's `gl` is a template over element type and up to 4 dims, with each extent
independently compile-time or runtime, and it generates the matching TMA
descriptor as a member. `cgl` (complex) and `pgl` (multi-GPU) wrap it — both
stay **[SCOPE]** (section 5).

`GlobalLayout<E, RANK>` is that type for rank 1 through 5, built `packed` (each
dimension the product of the extents below it) or `strided` (a leading dimension
wider than the columns, or a slice of a larger tensor), with dimension 0
contiguous because the engine reads it that way. Its `tensor_map::<T: TileBox>`
reads the data type, box shape and swizzle mode off the **destination** type, so
the only way to build a descriptor disagreeing with the tile it feeds is to pair
the layout with a different tile than the kernel loads. `check_driver_requirements`
rejects what `cuTensorMapEncodeTiled` would refuse — base alignment, stride
alignment, a box wider than 256 or narrower than a 16-byte line, an extent
smaller than its own box — naming the field and the byte count rather than
returning a bare `CUDA_ERROR_INVALID_VALUE`. `PanelMap` is now a type alias for
`TensorMap`, and `encode_bf16_panels` a five-line wrapper over
`GlobalLayout::<Bf16, 3>::packed`, kept because `softmax` calls it.

What is left is the element, and it is a one-impl hole rather than a design one:
`TensorMapElement` has exactly one impl, `Bf16`. An fp32 buffer cannot be
described even though `F32` has been an `Element` since #3, because nothing in
tree TMAs one. The prerequisite this entry used to name for section 2 is
discharged.

---

## 2. Missing data movement

### 2.1 Register ↔ TMEM — done (#10)

Both directions of TK's `tensor_to_register.cuh`: TMEM → register
(`TmemTile::fragment`, `fragment_tile`, LDTM) and register → TMEM
(`TmemTile::store_fragment`, `store_fragment_tile`, `tmem::store_wait`, STTM).
Both sides are `16x256b` at `[16, 16]` granularity, with `TmemTile::tile` and
`TmemTile::store_tile` composing a whole `[M, N]` band out of them (#22). The
store leaves its wait to the caller, since a store's registers are consumed at
issue.

**Both directions are a warp's, and which lanes are that warp's is measured**
(#193, for oxide-train#94). The safety clauses said "the warp *owning* TMEM rows
`row..row + M`" and never said which rows a warp owns; every caller spelled it
`32 * warp_id()` and every launch had one warpgroup, so the two readings could
not be told apart. They can now: the lane map is `warp_id() % 4`, exported as
`tmem::warp_lanes()`, and a **second warpgroup addresses the same lanes as its
opposite numbers** — reading alone, reading concurrently with the first, over an
MMA-written accumulator as well as a stored one, and storing back. Neither
`tcgen05.fence::before_thread_sync` nor `::after_thread_sync` is required around
that hand-off, which closes the question `store_fragment` carried as open and
closes it against the hardest case rather than the easiest. The bound is real in
the other direction too: a warp cannot reach a quadrant that is not its own, so
`M > 32` was never sound and the clauses now say 32. `device-tests`' `tmem across
warpgroups` is the standing gate; `docs/library/tmem.md` has the table and what
is still unestablished (`cta_group::2`, and silicon that is not a B200).

### 2.1a Shared ↔ register — done (#21)

TK's `shared_to_register.cuh`. `ldst::store_fragment` (`stmatrix`) and
`ldst::load_fragment` (`ldmatrix`) over one shared `SwizzledChunks` derivation,
`ldst::fragment_address`, that both directions call — so the address map is
host-testable and the two cannot drift. Missed by this inventory until the
examples crate (#18) hit it: a kernel whose input is not an MMA operand had no
way to reach registers at all.

Same `[16, 16]` granularity as 2.1, composed into a band by `ldst::load_tile`
and `ldst::store_tile` (#22) out of the same helpers the TMEM side uses, at any
width the cursor can describe (#25).

The two directions are no longer the same width. #116 gave the store side a
`.x4` form — `store_fragment_x4`/`store_tile_x4`, one `stmatrix` per `[16, 16]`
block instead of two — and it is what `gemm`'s shipped epilogue uses. The load
side is still two `ldmatrix_x2` a block, though `ldmatrix_x4` is right there at
the pin. Filed as **#131**, which asks for the measurement as much as the code:
`.x8` on the TMEM side (#117) bought its 23.6% by removing *waits*, and a
shared-memory load has none to remove.

### 2.2 Global ↔ register — done (#11)

TK's `global_to_register.cuh` (`load`/`store`): `global::load_rows` and
`global::store_rows` over a `GlobalRows` cursor — a base address and a leading
dimension, which is all a thread needs to compute its own addresses when there
is no engine to describe the buffer to. Each thread stores the values its
fragment layout gives it, at `L::row_of`/`L::col_of`, one `st.global.f32`
apiece; the row address is formed once per slot, so a band costs one multiply
by the leading dimension per owned row rather than per value. Measured against
the loop it replaces (`regcount`'s `global_copy_probe_*`) it is *cheaper* at
both probe widths — 48 against 56 registers at 32 columns, 44 against 168 at
128 — which is #22's result again.

The two are also the only movers here generic over `FragmentLayout` rather
than pinned to `BaseLdtm`: `ldmatrix`, `stmatrix` and LDTM each fix a lane map
in hardware, and a plain global store fixes nothing.

**This entry said "It is fp32 only" until 2026-07-29, and that is the worst
thing this file did in the audit's window.** `GlobalRows<E: Element>` has
carried an element since #108, which is what makes a bf16 `C` round once in the
store instruction instead of in a round trip through a shared tile — and the
paragraph asserting otherwise was written as the *reason* a bf16 `C` was out of
reach, so it read as an argument rather than as a fact and nothing prompted a
re-read. `ColLayout::CONTIGUOUS_VALUES` (#91) rides with it: a layout that hands
a thread adjacent columns gets `Element::write_pair`, one access for two values,
chosen once per call by `GlobalRows::runs_aligned` rather than promised in a
contract an odd leading dimension would break. Named in `tests/gaps.rs` as
`global_movers_carry_an_element`, which is what stops it going back to fp32 in
prose.

`load_cols` (#172) is the vector shape of the same walk on this cursor: `N`
consecutive elements of **one** row, read into the `ColVec` a `col_map` takes,
at `L::VALUES` registers where broadcasting the same vector through a
stride-zero `load_rows` costs `L::SLOTS * L::VALUES` and holds every value once
per row the thread owns. Its consumer is oxide-train's RMSNorm port, whose
`[dim]` weight is a runtime length and so cannot be a `SharedVec<E, N>` for
`ldst::load_vec` to read; the redundant-tile spelling it replaces put that
kernel on a spill cliff past a 32-column chunk. The pairing test is shared with
the tile movers rather than restated, which is why `pairs_are_one_access` is
bounded on `ColLayout` and not `FragmentLayout`. Device case
`global column map`.

`RegVec` reaches global memory too, since #130's first half:
`global::load_row_vec`/`store_row_vec` write one element per row of a band into
a single column of a row-major buffer — attention's log-sum-exp in the head's
own column of `[rows, heads]`, and the saved statistic a backward pass reads
back. It is a scatter and not a run (consecutive slots are 8 rows apart), so
there is no `CONTIGUOUS_VALUES` analogue to widen it; what the row axis needs
instead is **`RowLayout::owns_row`**, since `row_of` is not injective across a
warp — a row's `f32` lives in every lane holding any of that row's columns, and
a store without an owner would be four threads writing the same value to the
same address. The load has no such rule and every replica issues, which is what
makes the vector well-formed on the way in. Named in `tests/gaps.rs` as
`a_row_statistic_reaches_global_memory`.

**The same statistic reaches the column axis too**, since the flash backward
became the first kernel to want it: `global::load_col_vec` walks `load_row_vec`'s
addresses — one element per band position, down a single global column — and
delivers a `ColVec` instead of a `RegVec`. Which axis a statistic broadcasts
along is a property of the *band*, not of the statistic: a query-parallel
backward scores `Q·Kᵀ` and reads the saved log-sum-exp per row, while the
key-parallel one scores `K·Qᵀ`, whose rows are keys and whose columns are those
same queries. **It is not `load_cols`, and the two returning one type is the
hazard here**: `load_cols` reads a run *along* one row and pairs consecutive
values into one access; this walks rows a stride apart and has no run to widen.
Device case `global column statistic`, kept separate from `global column map`
for exactly that reason — the two share an argument list, and a mover that took
the wrong walk passes the other case. Named in `tests/gaps.rs` beside
`load_row_vec`, grouped by the memory the two share rather than by the type
`load_cols` shares with it.

What is genuinely left. It **bounds-checks nothing**: the extents a TMA
descriptor carries are absent rather than forgotten, since predicating every
value would be paid by the epilogues that do divide. TK's ragged-tail loads want
them back. Then `RegVec`'s *shared*-memory bridge, which is still absent where
`ColVec` now has both of its (`ldst::load_vec`/`store_vec` to shared,
`load_cols` to global) — the remainder of **#130**, and the reason the two
vectors need different code: a `ColVec`'s entries depend only on `lane % 4` and
one contiguous run of memory holds them, where a `RegVec`'s slots are spread
across `lane / 4` and moving one is a scatter. There is no `store_cols`,
deliberately: a kernel that *produces* a per-column result produces it from
`col_reduce` across a grid and wants an atomic, not a plain store. Group scope
is 1.1.

### 2.3 TMA store side — plain stores landed (#9), reductions absent

| TK | ferro-kittens |
| --- | --- |
| `load_async` | `tma_load`, `tma_load_at`, `tma_load_2d`, `tma_load_2d_multicast_cg2`, `tma_load_2d_arriving_at` ✅ |
| `store_async` | `tma_store`, `tma_store_2d` ✅ |
| `store_async_wait`, `store_commit_group`, `store_async_read_wait` | `tma_store_wait::<N>`, `tma_store_commit`, `tma_store_wait_read::<N>` ✅ |
| `store_add_async`, `store_min_async`, `store_max_async` | `tma_store_add_2d` (`add`, rank 2 only — the scoped #42 fallback) |
| `prefetch` | — |
| im2col descriptors | — **[SCOPE]** (conv-only) |

`SharedVec` carries its own `tma_load`/`tma_load_2d`/`tma_store`/`tma_store_2d`
beside these (1.4), at rank 1 and 2 and one instruction each.

What is left of this entry is the *reduction* stores, and this file had them
wrong. It said `cp.reduce.async.bulk.tensor` was absent from cuda-oxide "in
every form — not in the generated crate and not in `intrinsics/imported.json`
either", and argued from that that they were a lowering to contribute rather
than a transcription. **The records are there, and were there at `b099f64`
too**: `imported.json` carries 64 of them at both revisions, spelled
`int_nvvm_cp_async_bulk_tensor_reduce_{add,and,dec,inc,max,min,or,xor}_{tile,im2col}_{1..5}d`
— which is why a grep for `cp_reduce_*` finds nothing and is how the claim
survived as long as it did. What they are not is admitted to the generated
`cuda-device` crate, where the count is zero. So, like prefetch below, this is a
generation-list change upstream or a local `ptx_asm!` intrinsic (`ldst.rs` sets
the precedent). `store_add_async` is what makes split-K and multi-CTA reduction
epilogues cheap; the upstream route is filed as **#42**, and it is a smaller ask
than this file claimed.

The moment a kernel asked for it (the tcgen05 GEMM's accumulate epilogue,
oxide-train#80), #42's sanctioned fallback landed as
`shared::cp_reduce_async_bulk_tensor_2d_s2g_add` behind
`SharedTile::tma_store_add_2d` and `StoreRing::commit_add_2d` — `add`, rank 2,
tile mode, nothing else, gated by `device-tests`' `tma reduce-add store`. The
other seven ops, the other four ranks and im2col stay upstream work; when the
generation list admits the family, the `ptx_asm!` body shrinks to a call.

**Prefetch is not absent for the same reason, and this file said it was.**
`int_nvvm_cp_async_bulk_prefetch_L2` and the tensor prefetch forms *are* in
`intrinsics/imported.json` at the pin, with their PTX spelled out; what is
missing is only the generated Rust wrapper. So it is a generation-list change
upstream, or a `ptx_asm!` locally with none of #42's argument against one.
Recorded on #42 rather than filed separately, since the two share a section and
nothing has asked for prefetch.

### 2.4 Cluster ops stop at cg2

`commit_multicast_cg2` and `mma_walk_cg2` hardcode the 2-CTA pair. TK's
`tma_cluster.cuh` takes a general cluster mask. Larger clusters mostly matter
for the GEMM shapes; the 2-CTA case is the one Blackwell flash kernels use.
Filed as **#49**, which narrows the remaining ask to multicast as *replication*
— the only thing a cluster larger than a pair buys, and the thing the `_cg2`
suffix rules out by name.

**`tma_load_2d_multicast_cg2` is no longer part of the ask.** Its mask was
always the caller's; what was missing was the semantics, and oxide-train#80
measured them (`docs/library/shared.md`, "Where a replicating multicast's bytes
are counted"). A replicating load completes at the given offset in each
destination's own pair, selected by the address's rank parity, so an even-rank
address charges every destination's *pair leader* — one instruction feeding and
accounting for a cluster of several pairs. oxide-train's GEMM runs a 4-CTA
cluster on that and needs nothing new here. The `_cg2` in the name describes the
`cta_group` qualifier the instruction carries, not a limit on the mask.

The barrier-addressing half is no longer part of it. **#50** replaced
`multicast_alias` — which reached the peer's barrier by masking the rank bit
out of a local address, and was therefore correct only for rank 0 of a pair —
with `Semaphore::at_rank`, a `mapa` that takes any rank.

### 2.5 No distributed shared memory (DSMEM)

TK has cluster-scope shared→shared copies. Still absent: #50 brought in the
*addressing* half, since `Semaphore::at_rank` names a peer's shared offset, but
nothing here moves data between two CTAs' shared memory.

### 2.6 Shared → global without an engine — done (#113), folding too (#169), fp32 in (#174), the load inverse absent

Not a TK entry, which is why it had none here: TK's shared→global path is the
TMA and nothing else. `global::store_shared_rows` is the other one — a whole
`[R, C]` swizzled tile copied out to a row-major rectangle by ordinary
`ld.shared.v4` / `st.global.v4` pairs, `THREADS` threads splitting the tile's
16-byte chunks between them, no descriptor and no fence. It exists because a
fragment layout is a bad shape to store *from*: under `BaseLdtm` the widest
thing `store_rows` can issue is a pair and a warp's addresses are scattered
across the row, where a hop through shared memory widens the access to 16 bytes
and makes a warp's addresses one contiguous run. `access_width` walks 16/8/4
down to the element and takes the first the cursor admits, so an odd `ldc` gets
narrower stores rather than a fault.

It is the shipped GEMM epilogue's second half, and #123 measured it **beating**
the TMA route on that kernel.

`global::accumulate_shared_rows` (#169) is the same function folding instead of
overwriting: same signature, same chunk split, same ladder, with the copy
replaced by a load of both sides, an `Element::add_packed` and a store. It
exists because `C += A·Bᵀ` — every backward pass — had no library route at all,
and a hand-rolled read-modify-write drops off the ladder to 4-byte accesses:
`oxide-train`'s tcgen05 GEMM measured **1113.8 TFLOP/s storing and 536.2
accumulating** at 4096³ on one compute pipeline, so the fold cost more than the
whole rest of the kernel. Not bandwidth — re-reading a 4096² bf16 `C` is 33.5 MB
against a 133 µs difference — the *shape*, which is #116's finding again one
level up. The rounding is the element's and not the caller's:
`Element::add_packed` widens both sides, adds in fp32 and rounds the sum **once**,
which is why it is a trait method rather than a snippet at each epilogue. Named
in `tests/gaps.rs` as `a_staged_tile_can_be_folded_into_memory`; held on a B200
by `device-tests`' `shared accumulate` and `shared accumulate wide`.

`ldst::scatter_tile` (#174) is what fills a tile of an element `stmatrix`
cannot move. `store_shared_rows` and `accumulate_shared_rows` were already
generic over `E` and would take a `SharedTile<F32, …>`; nothing could put
anything *in* one, because `ldst::store_tile`/`store_tile_x4` bound
`E: Element<Unpacked = [f32; 2]>` — they are `stmatrix`, which moves b16
matrices and nothing else. So an fp32 accumulator band reached memory only
through `store_rows`' scattered per-value stores, the shape #116 measured at
**20.43 µs/tile** against the staged route's 6.68, and every kernel with an fp32
`C` paid it.

`scatter_tile` is `store_rows`' own loop with a `SwizzledChunks::element` where
the `GlobalRows::at` was: one shared access per owned *pair*
(`Element::write_pair_shared`, `st.shared.v2.f32` at fp32 and a packed
`st.shared.b32` at bf16), generic in the element *and* in the layout, since
nothing about it is an `ldmatrix` shape. An fp32 epilogue is now the two calls a
bf16 one is. Named in `tests/gaps.rs` as `an_fp32_band_reaches_a_staging_tile`,
whose unbounded `E` and free `L` are the claim; held on a B200 by
`device-tests`' `scatter drain` and `scatter drain wide`, against `register
drain`/`register drain wide` — the same rectangle written the old way, which is
the control for the register counts as well as for the bytes.

**What it bought downstream, and what it did not.** `oxide-train`'s
`gemm_tcgen05_f32_optimized` converted to the staged drain goes from **120
registers and a 512 B `.local` frame to 96 and none**: the frame was a band that
did not fit — `store_rows` held a whole `[32, 64]` fp32 band, and the
accumulating mode held `C`'s band beside it — and staging removes both, since
the fold becomes `accumulate_shared_rows`', which holds no band at all. The
milliseconds did **not** follow. Against three baseline runs on a B200, the
store arm is ~5% faster at 4096³ and a wash at 8192³ and 16384³, and the
accumulate arm is **15% slower** at 4096³ and 8192³ — reproducibly, at the shape
where the harness' variance is smallest.

The reason is the tile, not the route. At four bytes an element the same 4096 B
a warp is `[16, 64]` where bf16 gets `[32, 64]`, so an fp32 band is eight passes
where the bf16 one is four, and each pass is a serial
scatter → `ld.shared` → global → `bar.warp.sync` chain. Twice the passes is
twice that exposed latency, and at the accumulating variant the `ld.global` sits
on it too. So the staged drain wins when the staging tile is wide enough to
amortize a pass, and that kernel's shared budget — 98 392 B of operand rings
against a 116 736 B ceiling — does not leave enough for one. The conversion was
therefore **not shipped**; the route is here for the epilogue that can afford
the tile.

Absent: the load direction. Nothing reads a global rectangle into a shared tile
without a descriptor, which is what an irregular operand not worth a host-built
map would want — the mirror of the argument 2.2 makes for `load_rows`.

---

## 3. Missing compute

### 3.1 Elementwise / map ops

TK factors these through op structs (`base_ops.cuh`: `exp exp2 log log2 abs relu
neg sqrt rsqrt copy sum sub mul div min max fma_AxBtC fma_AxCtB`) applied by
generic `unary_map`/`bin_map`/`row_map`/`col_map`, then instantiates the whole
family for **register tiles, register vectors, shared tiles, and shared vectors**.

Done for the register families (#5): `UnaryOp`/`BinaryOp`/`TernaryOp`, the
`unary_map`/`bin_map`/`ternary_map`/`row_map`/`col_map` entry points on
`RegTile` and the two vector types, and the op set — `Exp2Approx` `Exp2Hw`
`Exp` `Log` `Log2` `Abs` `Neg` `Relu` `Sqrt` `Rsqrt` `Recip` `Add` `Sub` `Mul`
`Div` `Max` `Min` `Fma` — with named wrappers (`sqrt`, `mul_row`, `div_col`,
`broadcast_row`, …). TK's nullary `zero`/`one`/`pos_infty`/`neg_infty` are
`splat` with the constant written out; TK's `copy`/`copy2` have no by-value
equivalent to be.

Missing: the shared-tile and shared-vector instantiations. Shared memory is
still a movement target only — nothing in `shared.rs` maps or folds in place, so
data comes back to registers before anything is done to it. #13 asked for these
beside the `SharedVec` type and closed on the type alone (1.4). **#130 carries
the vector half; the tile half is still filed nowhere.** The per-column operand
of `col_map` is no longer the hole
this entry described: `col_reduce` produces one (3.2), `ldst::load_vec` reads
one out of shared memory, and `global::load_cols` reads one out of global memory
without staging it first (#172, 2.2).

### 3.2 Reductions — the register side is done (#6), the shared side is not

TK: `row_max row_min row_sum row_prod row_reduce`, `col_*` mirrors, plus whole-
tile `max min sum prod` — on register **and shared** tiles.

Done for the register tiles (#6). `RegTile::row_reduce`, `col_reduce` and
`tile_reduce` are each generic over a `ReduceOp` — a `BinaryOp` narrowed to the
associative and commutative ones, since the fragment map hands a fold its
operands in the layout's order rather than the tile's, and carrying an
`IDENTITY` so every fold seeds the same way instead of special-casing element
zero. `Add`, `Mul`, `Max` and `Min` implement it; `Sub` and `Div` deliberately
do not, because `row_reduce::<Sub>` has no meaning worth a spelling. The named
wrappers are the whole of TK's list — `row_max`/`row_min`/`row_sum`/`row_prod`,
the four `col_*`, and `tile_max`/`tile_min`/`tile_sum`/`tile_prod`, the last
group prefixed because `max` is already the elementwise binary op and Rust has
no overloading to tell the two apart. `Mul` *is* the product op, so there is no
separate `Prod`. One level down, `RegVec::reduce`, `ColVec::reduce` and the free
`quad_reduce`/`column_group_reduce`/`warp_reduce` are the same folds over a
single value; `quad_max`/`quad_sum` survive as the two named quad shuffles, and
`online_rescale`/`scale_rows` still carry the flash softmax pattern.

The column reduce this entry called the awkward one is written. `col_reduce`
folds a lane's own rows into a `ColVec`, then `column_group_reduce` runs a
three-shuffle butterfly over masks 4, 8 and 16 — the 8 lanes `BaseLdtm::col_of`
spreads a column across — against the two-shuffle quad a row reduce uses, and
all five masks for a whole-tile fold. Which masks those are is the entire
correctness question, so `reduction_masks_are_the_ownership_maps_lane_groups`
derives the lane groups from the maps rather than restating the constants, and
`reductions_fold_exactly_their_logical_axis` folds each axis against the
*logical* tile. On silicon they are the `reduction shuffles` device case.

Still missing: **every shared-tile and shared-vector reduction**, which is 3.1's
other half — the vector half is #130's third item, the tile half is filed
nowhere. Note also that everything above is warp
scope. The fold four warps need is `sync::block_reduce` (1.1) — storage between
two barriers, not a wider `Op`, because warps cannot shuffle to each other.

### 3.3 MMA shape and dtype coverage — shapes done (#12), dtypes not

TK's tcgen05 layer: `{mm, mma} × {AB, ABt, AtB, AtBt} × {1-CTA, 2-CTA}` = 16
entry points, generic over element type, plus `load_mxnv_scale_async` for
block-scaled MXFP.

The operand-order square is closed: `mma_abt`, `mma_ab`, `mma_atb`, `mma_atbt`
(single-CTA) and `mma_walk_cg2` (2-CTA, runtime K-major/MN-major via
`OperandWalk`), each with an `mm_*` twin that starts the accumulator fresh
rather than taking a `bool` — so a call site whose choice is static states it
in the entry point and only gemm's genuinely runtime `k > 0` still threads an
argument. The instruction is `KIND`-generic now too: the walks issue through
`MmaElement::mma`/`mma_cg2`, which each impl writes as
`tcgen05_mma_shared::<{ Self::MMA_KIND }, ..>`, so fp8 is an `Element` impl
and not a second MMA layer. It has to be a *method*: an associated const of a
type parameter is not a legal const-generic argument without
`generic_const_exprs`.

**The estimate was low.** The two walks are 70 lines with their docs, near
enough the ~80 above — but the walks were never the work. Nothing calls them,
and an unused walk that is wrong is worse than an absent one, so the case that
had to be built first was the one that could tell them apart:
`device-tests`' five `mma A·Bᵀ` … `mma transpose control` cases, ~420 lines,
which run all four orders over one pair of logical matrices staged four ways
and require the same product from each. `walk blind spots` is the sixth and
runs on the host: it removes each of the 64 K planes in turn, permutes the K
chunks four ways, and confuses the two stacked MN subtiles five ways, and
fails if the reference cannot see any of it — the standing answer to #48, whose
`depth * 5 % 5` was identically zero. `mma transpose control` is #55's shape:
the same operands under `mma_abt`, required to *disagree*, so the transposed
cases prove their transpose bits and not merely that some walk multiplies.
Total: **70 lines of walks, ~420 of harness.**

Two things that fell out of building it. The MN-major operand has *two*
descriptor spellings in this crate and both are right: banding 64 wide with a
16-byte leading offset, and covering a 128-wide MN in one instruction with the
leading offset set to `SUBTILE_BYTES` — the second subtile reached through the
descriptor, never by a step along the row. `mma_ab` was the last caller of the
first spelling and moved to the second (#175 closed, oxide-train#94), which is
both one shape rule fewer and 1.50x the tensor-core rate on that product; the
banding spelling now survives only in the K-major `k_major_offset` walk. And
`tcgen05_mma_f16` and `tcgen05_mma_shared::<0, 1, 0>` do **not** lower to the
same PTX at the pinned revision, though they are semantically the same
instruction: the f16 wrapper emits the `.disable_output_lane` form with an
all-zero mask (four `mov.u32` of zero), the generic one emits
`.collector::a::discard` and no mask.

Still missing: FP8/FP4 operands (#16) and MXFP scale loading — both dtype
work, not shape work. `tcgen05_mma_sp_*` (sparse), `tcgen05_mma_ws_*`
(weight-stationary) and `tcgen05_shift_down` remain available upstream and
unused.

Two shape observations from the 2026-07-29 audit, neither filed, both weak on
evidence and recorded so the next reader does not re-derive them. The
square is closed at **cta_group::1 for the typed walks and cta_group::2 for the
value walk**, and the other two quadrants do not exist: there is no `mma_abt_cg2`
and no single-CTA `mma_walk`. No kernel has wanted either — `flash_forward`
takes `mm_abt`/`mm_ab` and both GEMMs take `mma_walk_cg2` — so the hole is
symmetrical on paper and unmotivated in practice. The other observation — that
the accumulator was a bare `u32` in all nine entry points where `TmemTile<R, C>`
existed and knew the shape being passed separately — was filed as **#128** and
has landed; every walk takes the tile and derives its own shape.

### 3.4 Tile-shape utilities

Landed with #7: `tril`/`triu`, `make_causal`/`make_causal_t`,
`left_fill`/`right_fill`/`upper_fill`/`lower_fill`, over a `RegTile::mask` that
selects at the layout's own `(row, column)`. Every one takes a **coordinate
origin** TK's signatures do not have — a `diagonal`, or a signed fill index —
because the tile is a sub-block of a larger matrix whose diagonal sits at
`query_base - key_base`; TK's origin-free form is right only for the diagonal
block. `broadcast_row`/`broadcast_col` landed with 3.1.

Both diagonals now carry the *block-origin* form as well, not just the signed
`diagonal`: `make_causal_at` since #7 and `make_causal_t_at` since the flash
backward asked for it. The transposed band is where the argument for the `_at`
form is strongest, and it was the half that did not have one — a CTA owning a
block of keys streams the queries at and after them, so `key_base -
query_base` is negative on every visit but the first, and the `u32` subtraction
a call site would otherwise write wraps and masks nothing. Named in
`tests/gaps.rs` as `every_diagonal_mask_has_a_coordinate_origin`.

Absent: `transpose`, `swap_layout`, `copy` between layouts — all three need a
second `FragmentLayout`, which the crate does not have and no issue asks for
(1.3). **~60 lines**, once one exists. `#7` named them in its title and closed
without them, as its own sequencing asked.

### 3.5 Warp-level MMA fallbacks **[NON-GOAL]**

TK keeps `hmma16816`/`imma16832` (warp `mma.sync`) and the full wgmma warpgroup
path for Hopper and earlier. Deliberately absent — this library is `sm_100a`
only, no arch dispatch.

---

## 4. Missing kernel scaffolding

`pipeline::run` + `trait Job` is TK's `prototype::lcf` (load-compute-finish),
and it's a faithful port of that shape. **This entry read "still uncalled by
anything in `examples/`" until 2026-07-29**, on the argument that `run` strides
items by `blockIdx.x` and a cluster launch needs the whole cluster on one item.
#51 fixed that — the map is `%clusterid` stepping by `%nclusterid`, which at a
one-CTA cluster *is* the old loop — and `gemm` has run on the scaffold since
#83, exact on a B200 and a dead heat with the launch-per-tile grid it replaced.

The module has grown two things since, both of which this entry predates:

- **`run_stealing`** (#88/#97) — the same loop over the hardware's own work
  queue, `clusterlaunchcontrol.try_cancel` multicast to the cluster and
  harvested off an mbarrier. It takes no item count, because the grid *is* the
  item count. What it buys is not speed — the ragged last wave is 1.1–2.4%,
  against a wave model that predicted up to 23% — but the deletion of `SMS` and
  `CTAS_PER_SM`, the two constants that make the scaffold B200-shaped.
- **`grouped`** (#89/#102) — item → `(row, column)` in blocks of `group`
  tile-rows, a bijection for every width including one that does not divide the
  grid. It composes with either item source rather than replacing one, which is
  what lets the swizzle and the scheduler be swept independently.

**#15 closed without `lcsf`**, and the finding is worth keeping: an undrained
accumulator survives the item boundary in tensor memory, so deferring the
epilogue by one item is a reordering *inside* `Job::work` and the scaffold needs
no store stage at all. Measured over two sessions it is −5.4% to +1.2% on the
GEMM and never reliably positive.

What the trait still cannot do, both named by the kernels rather than by this
file: a **warp-asynchronous item stream** — every thread enters `work` with one
`item`, so warps cannot be on different items, which is the shape a reference
warp-specialized GEMM uses — and a **teardown hook**, so the drain a deferred
job owes after `run` returns is remembered by hand at nine entry points in
`gemm_ws.rs` alone. Filed as **#132**.

Still missing from `prototype/`:

- **`lcsf`** — load-compute-**store**-finish, as a *stage the scaffold
  sequences*. That is what remains unbuilt and it is deliberate: the paragraph
  above is why the shape needed no scaffold at all here. A probe holding the
  epilogue's instructions fixed while removing its HBM traffic puts the
  write-bound part of the GEMM's epilogue at **0–1.2%**, so the term `lcsf`
  exists to overlap is not costing this kernel anything to begin with.
  `experiments/README.md` §7 has both tables. A kernel whose epilogue is genuinely
  latency-exposed may still want it; a `Job` whose accumulator is
  single-buffered in TMEM cannot overlap more than one pipeline fill's worth of
  it, which is the constraint that decided this one.
- **`lcsc`** — the store-compute variant.
- **`interpreter`** — TK's instruction-driven "VM" kernel, where a persistent
  grid dispatches over an opcode stream. This is the newest and most speculative
  part of TK; low priority.
- **`common/templates.cuh`** — the shared config/state plumbing the three
  prototypes sit on.

---

## 5. Out of scope for a Rust port

- **`pyutils/`** — `torchutils`, `profiler`, `broker`, `club`, `parallel_tensor`.
  The Rust equivalent is whatever the calling crate uses for host orchestration,
  not this library. **[SCOPE]**
- **`types/system/`** — `pgl`, `ipc`, `vmm`, `multimem` (multi-GPU parallel
  global layouts, IPC handles, virtual memory management). Real functionality,
  but it is host-side allocation plumbing that belongs beside `cuda-core`, not in
  a device tile library. **[SCOPE]** — revisit if multi-GPU lands.
- **`kernels/`** and **`demos/`** — TK ships reference attention, GEMM, mamba2,
  based, hedgehog, fftconv, layernorm, rotary, flux kernels. The kernels this was
  extracted from live in `rust-trainer`. Worth pulling one in as an integration
  test (see below), not worth porting the set.

---

## 6. Missing infrastructure (not a TK gap, a repo gap)

- **Host tests are no longer address arithmetic only** — the claim they were was
  itself stale, along with the count that used to sit here. TK has `tests/` with
  a generated harness over the type/layout cross product; ours are hand-written
  `#[cfg(test)]` modules and `cargo test` reports how many, which is the only
  place that number is worth keeping. Most are still coordinate and pointer
  math, but a real minority are not: `exp2_polynomial_stays_inside_its_error_bound`
  and `log2_series_stays_inside_its_error_bound` bound the two software
  approximations, `unary_ops_are_their_scalar_definitions` and its binary twin
  hold the op set to its scalar meanings, `named_wrappers_resolve_to_their_ops`
  pins each wrapper to the op it claims, and the reduction tests simulate the
  shuffle by folding over the lane groups the *map* names. What none of them do
  is run on silicon: they prove the crate is self-consistent, not that it agrees
  with hardware. `cargo test` needs no toolkit; `--features host` adds `global`'s
  descriptor-shape tests and pulls `cuda-core` → `cuda-bindings`, which wants
  `cuda.h` — so it is a CI tier (below) rather than a laptop command.
- **Device tests reach a good deal more than the six paths this entry used to
  list.** `device-tests/` (a separate kernel crate, run on a B200 through
  `modal_app.py::device_tests`) registers its cases as a `Vec<(&str, closure)>`
  in `run` and prints `N of M cases passed`, so the count is the harness's to
  report and is not restated here. What they reach: the SWIZZLE_128B round trip
  against the TMA engine, narrow and wide and at a short tile whose second
  subtile starts mid-period; `BaseLdtm`'s ownership map against a
  position-encoding MMA at all three layout shapes; `stmatrix` and `ldmatrix`
  in both directions, narrow and wide; STTM both as a register round trip
  through TMEM and by restaging a probed accumulator into a second column band;
  composed `[M, N]` bands (`load_tile` into `store_tile`); all four MMA operand
  orders against one product, with a host-side blind-spot sweep and an
  untransposed control (#12); `load_rows`, the one case whose source reaches
  registers with no descriptor between them; both `SharedVec` ranks through the
  TMA, `load_vec` and `set`; `block_reduce_sum` and `block_reduce::<Max, 4>`;
  the row, column and whole-tile reduction shuffles; the TMA store paths
  including `tma_store_wait_read`'s early recycle; and a repeated-launch probe
  under a watchdog that would catch a TMEM leak across waves.

  Four families have landed since that list was written and are worth naming
  because each covers a mechanism nothing else does. `ldtm x8 map` drains one
  accumulator through both `.x1` and `.x8` and requires equality, which is what
  holds #117's claim about the wide load's *register order* — an ISA-text
  inference otherwise. The `store ring` cases cover depths 1, 2 and 4, the wide
  tile, and both scopes (#123), with `store ring warp phased` the store side of
  `swizzle roundtrip short`: a staging tile that begins mid-swizzle-period, read
  correctly by the engine off its absolute address. `shared drain` covers
  §2.6's mover. And `tmem occupancy ladder` / `tmem residency census` count CTAs
  an SM by `%smid` and `%globaltimer` rather than asking the occupancy query,
  which is what refuted #70's and #74's readings (see `src/tmem.rs`).

  The matching global *store* is checked by the GEMMs instead: the
  register-drain rung (`experiments/src/gemm.rs`) is one `store_rows` compared
  elementwise against an exact CPU reference, and the shipped rung since #119
  (`examples/src/gemm.rs`) is `store_tile_x4` into a per-warp shared tile and
  `store_shared_rows` out of it, on the same gate and the same `==`. Still verified only by downstream kernels being numerically
  correct: the phase-parity rules, the pipeline scaffold, the masking ops
  (`mask_probe` is a codegen probe and says so in its own doc comment), and
  every cluster/multicast path — those last live in `examples/src/gemm.rs` and
  `experiments/src/gemm.rs` and are covered by `modal_app.py::examples` rather
  than by a case here.
- **CI, in three tiers** (#17). `ci.yml` runs `fmt`, lockfile freshness,
  `clippy --all-targets -- -D warnings`, `test` and `cargo doc` under
  `RUSTDOCFLAGS=-D warnings` on every push and pull request, with no toolkit and
  no credential — the library's default features are device-only and
  `cuda-device` is ordinary Rust. `cuda.yml` is `modal_app.py::build`: the
  `host` feature, its tests and its docs, plus a real
  `cargo oxide build --arch sm_100a` of all three kernel crates against the driver
  stub. That last is the tier that earns its money, because a
  post-monomorphization `const { assert!(..) }` in a tile shape fires at
  *codegen* and `cargo check` cannot see it. `gpu.yml` runs the B200 harness
  behind a label. `CI.md` carries the policy and the costs. What is still
  uncovered: fork pull requests get tier 1 only, since the two tiers above it
  need a Modal token.
- **Register cost is swept now, not sampled** (#60). Every claim in this file
  and in `experiments/README.md` used to be measured at 32 and 128 columns, and
  the reason was structural rather than budgetary: `ptxas` is a host compiler
  and a width costs milliseconds, but each width was a *hand-written* probe.
  `device-tests`' `ladder!` generates one per (shape, spelling), so `modal run
  modal_app.py::regcount` now prints a 14-shape × 5-spelling ladder — 16 to 256
  fp32 a thread, deliberately onto the 255-register ceiling and over it — with
  the cliff
  located per spelling instead of left to be eyeballed, and `--determinism`
  measures the same tree twice and asserts an identical table before any diff
  is worth reading. What is still sampled: the ladder varies `M` at two widths
  only, one op (a scale-and-accumulate step), and no shared-memory or MMA path
  at all. And what it found is that the shape response is not smooth — the
  in-place spellings drop 200 registers at some shapes by leaving the band in
  local memory, so a register count on its own no longer settles a comparison.
- **And a register count does not settle it on the clock either** (#63). `modal
  run modal_app.py::ladder_bench` is the first thing in this repo to *time* a
  register claim rather than count it: four of the ladder's shapes on a B200, all
  five spellings each, every one verified against a CPU reference at every grid
  and step count it is timed at, timed in repeated rounds so the table states its
  own noise floor (0.8% worst case), with a `t(2S)/t(S)` control that says the
  loop under test was not hoisted. On that probe the streamed band #60 flagged is **not**
  slower: the fully in-place spelling — 32 registers, band in local memory — is
  the fastest form at all four shapes, 6–10% per warp and 25–42% across a full
  device, and the control shape `[32, 128]`, where `fused` wins on both static
  counters, is one where `fused` *loses* on time. Registers, stack frame and
  occupancy each order part of that table and none of them orders all of it, so
  `ptxas -v` is a record of what was allocated and not a ranking.
- **Put beside #47, that becomes a rule rather than a curiosity.** #47 timed the
  same phenomenon on `softmax` — a `[32, 128]` band at 39 registers on a 1024 B
  frame against a `[32, 32]` band at 66 registers on 256 B — and the streamed
  one cost **2.6×**. Opposite sign, same mechanism, and the difference is what
  the freed registers were able to buy: `softmax` is shared-memory capped at 6
  blocks an SM either way, so streaming bought it nothing and the local traffic
  showed undiluted, where the ladder probe uses no shared memory at all and
  streaming bought it 2–4× the resident warps. So: **a streamed band costs real
  time, and whether it is worth paying is a question about which resource is
  binding in that kernel**, which is why `regcount` now prices the examples
  crate and `kittens-examples` prints occupancy per kernel. Two caveats stay
  attached: the ladder probe is one load-heavy op at one arithmetic intensity,
  and local-memory *traffic* was measured on neither side — no profiler was run,
  so "the band is streamed" is still read off the frame and the register count.

---

## 7. Coherence of the surface itself (new on 2026-07-29)

Sections 1–6 ask what ThunderKittens has that this does not. This one asks what
the crate says about *itself*, which no comparison against another library
finds, and it is where the 2026-07-29 audit spent most of its time. Nothing here
is a bug. All of it is what a new user meets first.

### 7.1 Scope is in the type once, and spelled three other ways

`epilogue::StoreRing` takes a `Scope` — `Cta` or `Warp`, a trait of
`{ issuing(), converge() }` — because #123 needed a staging tile that was not
CTA-wide and a ring hard-wired to `bar.sync` could not express one. That is the
right shape, and it is the only place in the crate a collective's scope is
named. Beside it: `pipeline::Job::RANKS` is a `const u32` branched into
`cluster_sync` or `sync_threads`, `sync::block_reduce` takes a `const WARPS`,
and `global::store_shared_rows` takes a `const THREADS` plus a thread index.
`Scope` and `RANKS` are adjacent rungs of one ladder — warp < CTA < cluster —
built twice with no rung in common. Everything else keeps its scope in its index
math: `tmem::alloc_block` *is* `warp_id() == 0` plus `sync_threads`, and does
not say so. **#124**, and 1.1 for why #3's argument does not reach it.

### 7.2 A kernel cannot call one register-side function without `cuda_device` — closed (#127)

As filed: every warp-scope entry point took `lane: u32` and the crate exposed no
way to get one, so all five kernels imported `cuda_device::warp` in their first
three lines; and `ldst::store_fragment`'s contract said the caller *owed a
`fence.proxy.async.shared::cta`* while the only thing that issued that fence was
the private `StoreRing::publish` — an obligation created by a `kittens` function
and discharged by a `cuda_device` one, at seven sites in the three small kernels.

`kittens::lane()`, `kittens::warp_id()` and `shared::publish_to_async_proxy()`
are those three, wrappers rather than re-exports because the wrapper is where
the doc can say that `lane` is the argument the whole register surface takes,
that `32 * warp_id()` is a `BaseLdtm` band origin, and that a barrier after the
fence is what carries one thread's ordering to the thread that issues the store.
The eight safety contracts that state the obligation now name the function that
discharges it, and `block_reduce`'s — which states that it is *not* owed — names
it too. Named in `tests/gaps.rs`.

`warp::lane_id()` and `warp::warp_id()` are now called in exactly one place in
the repo outside `device-tests`, which is the body of the two wrappers.
`softmax`, `layernorm` and `flash_forward` dropped the `cuda_device::warp` and
`cuda_device::barrier::fence_proxy_async_shared_cta` imports outright; the three
GEMMs keep `cuda_device::warp` for `warp::sync_mask` and nothing else, which is
the epilogue's inter-band convergence and **#126's**.

What is *not* closed and was judged rather than deferred: the
`tcgen05_fence_before_thread_sync` + `cluster_sync` pair three kernels write out
before `dealloc_cluster` stays two lines, documented on `tmem::dealloc_cluster`.
Welding a scope into it would add a fifth spelling of scope beside the four §7.1
counts, and it would close no leak — a kernel calling `cluster_sync` there
already imports `cuda_device::cluster` for `block_rank`. What remains in the
small kernels' `use cuda_device` lines is §7.4's shared plan (#125) and `thread`
(#124), not the register surface.

### 7.3 The handles pick two conventions each

`from_raw` for `SharedTile`/`SharedVec`/`GlobalRows`/`TmemTile`, `attach` for
`SharedTileRing`/`StoreRing`/`Semaphore`/`PhasedSemaphore`/`SemaphoreRing`/`ClcQueue`
— and "rings use `attach`" is the obvious rule and is false. `SharedTile::from_raw`
is `unsafe` and `TmemTile::from_raw` is safe, though neither dereferences
anything, so the crate has no stated rule for where the `unsafe` boundary is.
The TMEM read direction has no verb (`TmemTile::tile` against
`ldst::load_tile`) while its write direction shares one. `lib.rs` re-exports
`ReduceOp` but none of the four types that implement it, and no free function at
all, so `use kittens::*` yields types with no verbs. **#129.**

### 7.4 What a kernel still open-codes, which is the coverage question restated

Three things every kernel wrote for itself when this section was written,
ordered by size. The first of them has since closed.

- **The shared plan — landed (#125), with one half of it left standing.**
  `plan::SharedPlan` is a `Copy` cursor: it starts at the launch's base, hands
  out one typed handle per reservation, aligns each to what that handle's type
  requires, and its `bytes()` is the total the launch declares. All five kernels
  are on it, and the four intrinsic leaks this entry named are gone —
  `DynamicSharedArray::get_raw` is `SharedPlan::attach`, the `*mut Barrier` cast
  is `semaphore`/`semaphores`, `alloc_cluster`'s staging word is `tmem_slot`,
  and `ClcQueue::ALIGNMENT`'s proof obligation is `clc_queue`. `layernorm`'s
  ordering argument is discharged by the type rather than by a comment.
  **What did not close**: a `const fn` over *values* cannot instantiate a
  const-generic tile type, so `experiments`' rung table — which answers for
  rungs no kernel implements — still spells its reservations in bytes through
  `SharedPlan::reserve`, and one `const { assert!(..) }` per GEMM joins the two.
  The three kernels whose plan parameters are module constants have neither.
  That is a language limit (`generic_const_exprs`), and it is the only place in
  the repo a plan is still written twice.
- **The epilogue.** `drain_staged`'s *loop* is the same program in
  `experiments/src/gemm.rs:1371-1403` and `experiments/src/gemm_ws.rs:780-813`
  — and, with the two const parameters resolved, in
  `examples/src/gemm.rs:496-520` — same band selection on
  `WIDE`/`X4`, same `store_tile{,_x4}` into the same `store_shared_rows`, same
  warp-scope write-after-read, same `STAGE_N` stride — while the preambles
  differ, because `gemm_ws` has two accumulator stages to select between and
  resolves the tile origin inline where `gemm` has an `origin()`. The loop is
  the part that would move. It is what `SHIPPED_EPILOGUE` selects, and every
  instruction in it is a library call — what is missing is the loop and the
  inter-band convergence. `epilogue::StoreRing` is the library's only epilogue
  type and it covers the TMA route, which #123 measured *losing* to this one.
  `softmax` and `layernorm` then open-code a third shape, the single-shot
  `tma_store` + commit + `wait::<0>`. **#126.**
- **The band origin.** `32 * warp_id` appears in every kernel in the repo, five
  times inline in `flash_forward.rs`. The `32` is `BaseLdtm`'s rows per warp,
  which the library never names, and `DRAIN_N = 128` — the widest band a thread
  can drain before 256 fp32 crosses the 255-register ceiling — is derived
  independently in both GEMMs' docs (`experiments/src/gemm.rs:412`,
  `experiments/src/gemm_ws.rs:447`). Both are library facts living in
  `examples/` and `experiments/`. Named in #126.

### 7.5 Types that exist and are not used by the operation that needs them

**Landed with #128.** All nine MMA entry points took a bare `u32` accumulator
and a separately-passed `MmaShape` where `TmemTile<R, C>` already knew the
shape, and `tmem::alloc_block` took a runtime `columns: u32` and checked
nothing — which is why `flash_forward` asked for an illegal 192 columns from the
day it was written with nothing in the type system in a position to notice. The
allocators take `const COLUMNS: usize` and assert `tcgen05.alloc`'s
power-of-two-in-`[32, 512]` rule at codegen; the walks take the tile and derive
the shape through `mma::shape` / `mma::pair_shape`, which reject an `[M, N]` the
ISA has no instruction for rather than rounding it. The hand-written
`pair_shape(block_n) -> MmaShape` lookups in `examples/src/gemm.rs`,
`examples/src/gemm_sol.rs` and `experiments/src/gemm.rs` are gone with it.

Still open there: the accumulator has no element type, so fp32 accumulate stays
hardcoded in `mma::descriptor`. #128 left that out deliberately — fp16
accumulate is a `.kind::f16`-only mode with no caller — and the point of the
change is that the type is now in a position to carry one.

---

## Work order

The original order was issues #1–#16, structural items first, since they change
signatures across the crate and get expensive to retrofit. **Thirteen of the
sixteen have closed** — #1 through #13 — and the sections above name the work
beside each. Listing them again here only creates a second place for the status
to go stale, which is how this file earned #65.

What was open on 2026-07-29, and where it lives. `gh issue list` is the live
version; this is a snapshot with a date on it. The 2026-07-28 snapshot listed
#15, #50 and #51 as open and all three had closed by the next day, which is the
argument for keeping this table dated and short rather than complete — and #125
came off this table the same day it went on, which is the same argument again.

| # | Issue | Section |
| --- | --- | --- |
| 14 | Swizzle modes: 32B, 64B, unswizzled — and the walk step | 1.3 |
| 16 | FP8 element support | 1.2 |
| 42 | Reduction TMA stores — upstream rather than a local `ptx_asm!` | 2.3 |
| 49 | Cluster geometry beyond the 2-CTA pair | 2.4 |
| 124 | Scope is in the type once and spelled three other ways | 1.1, 7.1 |
| 126 | The shipped epilogue is open-coded in both GEMMs | 7.4 |
| 128 | TMEM columns unchecked; the MMA takes an address, not a tile | 3.3, 7.5 |
| 129 | Two conventions each for constructors, `unsafe`, verbs, re-exports | 7.3 |
| 130 | `RegVec` cannot reach memory; `SharedVec` cannot be sliced | 1.4, 2.2 |
| 131 | The load side is stuck at `ldmatrix.x2` | 2.1a |
| 132 | `Job` has no teardown hook | 4 |

Kernel-side issues not in this table because they are not library gaps: #81
(the `exp2` default — this PR corrects the doc it names and does not move the
default), #85 (`flash_forward`'s shared plan), #105 (tile selection).

**Wanted and filed nowhere**, each named in its section above: a second
`FragmentLayout` (1.3, and 3.4 waits on it), the map and reduction
instantiations on shared **tiles** (3.1, 3.2 — the vector half is #130), an fp32
`TensorMapElement` (1.5), DSMEM (2.5), and the global→shared mover with no
descriptor (2.6). TMA prefetch is no longer on this list in the form it was:
see 2.3 and the note on #42.

## cuda-oxide support at the pinned rev (`20a5616`)

Checked against the source, since several gaps turn on it. **Re-checked on
2026-07-29 against a checkout whose `HEAD` was confirmed to be
`20a56163f258e09f2c51e4c27ae4e4ff17582443` before anything was read from it** —
which is #106's rule, written after a `~/.cargo` checkout at `4514af2` was cited
for a `clc` doc that our pin does not have. There are now **seven** cuda-oxide
checkouts in that cargo directory — the bump added one — and six of them are not
our pin, `/Users/finn/projects/cuda-oxide` (at `29396b7`) among the ones that
are not.

"Absent" below means absent from the generated `cuda-device` crate. That is the
line that matters for a caller and it is *not* the same as absent upstream — see
the prefetch row.

| Need | Status |
| --- | --- |
| Generic MMA over element kind | ✅ `tcgen05_mma_shared`/`_tensor`, `KIND` = 0 f16, 1 tf32, 2 f8f6f4, 3 i8 |
| Weight-stationary MMA per dtype | ✅ `tcgen05_mma_ws_{bf16,f16,tf32,e4m3,e5m2,e2m1,e2m3,e3m2}` |
| Sparse MMA | ✅ `tcgen05_mma_sp_*` (unused) |
| Register → TMEM (STTM) | ✅ full `tcgen05_st_*` family + `tcgen05_store_wait` |
| TMA store | ✅ `cp_async_bulk_tensor_{1..5}d_s2g` + commit/wait groups |
| TMA reduction store (add/min/max) | ❌ absent from the crate, ✅ **present in `intrinsics/imported.json`** (64 `int_nvvm_cp_async_bulk_tensor_reduce_*` records, 8 ops × `tile`/`im2col` × rank) — a generation-list change or `ptx_asm!` (#42) |
| TMA prefetch | ❌ absent from the crate, ✅ **present in `intrinsics/imported.json`** (`int_nvvm_cp_async_bulk_prefetch_L2`, plain and `.L2::cache_hint`, plus the tensor forms) — a generation-list change, not a lowering (#42's thread) |
| f32 → fp8 | ✅ `cvt_rn_satfinite_{e4m3,e5m2}x2_f32` (+ `relu`) |
| f32 → fp4/fp6/e8m0 | ❌ absent — hand bit-packing |
| MXFP block-scale smem→tmem | ✅ `tcgen05_cp_*_b4x16_p64`, `_b6x16_p32` |
| `stmatrix` | ✅ `x2`/`x4`, plain and `trans` — we use **both** widths since #116, through local `ptx_asm!` either way (the generated declarations do not resolve for `sm_100a`) |
| `ldmatrix` | ✅ `cuda_device::wmma::ldmatrix_{x1,x2,x4}` (+ `trans`); filed under `wmma`, but lowers for `sm_100a` — we use `x2` only, which is #131 |

**FP8 is not blocked upstream.** FP4/FP6 and MXFP block scaling are.
