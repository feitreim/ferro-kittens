# Gaps vs ThunderKittens

Measured against HazyResearch/ThunderKittens `main` (~21.3k lines of headers under
`include/`, plus `prototype/`). `wc -l src/*.rs` answers for this side, so that
number is not maintained here: it was 1851 when this file was written and 7353
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

**Audited against the source on 2026-07-28** (#65). Every entry was checked by
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
collective, not the parameterization. The scope of every other op is still
baked into its index math, and naming it in the types is unfiled.

### 1.2 Single element type

| | TK | ferro-kittens |
| --- | --- | --- |
| Operands | bf16, half, fp8 (e4m3/e5m2/e8m0), fp4 (e2m1), int8, tf32, fp32 | bf16 only |
| Accumulate | fp32, fp16, int32 | fp32 only |
| Complex | `crt`/`crv`/`cst`/`csv` throughout | — **[SCOPE]** |

`Element` now carries the byte width, the fp32 → element pack, and (via
`MmaElement`) the tcgen05 operand kind, so shape math, `ldst`'s pack, and
`mma`'s bounds are all generic — but `Bf16` is still the only *operand* impl
(issue #2 did the trait work; #16 adds fp8). `F32` is the second `Element`
since #3, and deliberately not an `MmaElement`: it exists so a statistic can be
staged in shared memory without rounding, which is a storage question and not
an operand one. One bf16 fact remains hardcoded behind an assert rather than
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
type. See 3.1 — nothing tracks the ops.

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

Two things this deliberately is not. It is **fp32 only**: `GlobalRows` names a
`*mut f32` outright rather than carrying an `Element`, and fp32 out of registers
without a rounding step is the whole point of the entry. The reason first given
here — that `Element` was bf16-only — expired with #2 and #3, which made it a
real trait with `F32` under it; the parameter is available now and nothing has
asked for it. And it **bounds-checks nothing**: the extents a TMA descriptor
carries are absent rather than forgotten, since predicating every value would be
paid by the epilogues that do divide. TK's ragged-tail loads want them back, and
that is what is left of this entry, along with the vector shapes
(`RegVec`/`ColVec` straight out of global memory — the shared-memory bridge
exists, `ldst::load_vec`, but not the global one) and group scope (1.1).

### 2.3 TMA store side — plain stores landed (#9), reductions absent

| TK | ferro-kittens |
| --- | --- |
| `load_async` | `tma_load`, `tma_load_at`, `tma_load_2d`, `tma_load_2d_multicast_cg2` ✅ |
| `store_async` | `tma_store`, `tma_store_2d` ✅ |
| `store_async_wait`, `store_commit_group`, `store_async_read_wait` | `tma_store_wait::<N>`, `tma_store_commit`, `tma_store_wait_read::<N>` ✅ |
| `store_add_async`, `store_min_async`, `store_max_async` | — |
| `prefetch` | — |
| im2col descriptors | — **[SCOPE]** (conv-only) |

`SharedVec` carries its own `tma_load`/`tma_load_2d`/`tma_store`/`tma_store_2d`
beside these (1.4), at rank 1 and 2 and one instruction each.

What is left of this entry is the *reduction* stores, and they are a different
kind of missing: `cp.reduce.async.bulk.tensor` is absent from cuda-oxide at the
pinned revision in every form, so unlike the plain stores they are not a
transcription. `store_add_async` is what makes split-K and multi-CTA reduction
epilogues cheap, and getting it means a local `ptx_asm!` intrinsic (`ldst.rs`
sets the precedent) or an upstream contribution — filed as **#42**, which argues
for the upstream half. Prefetch is absent upstream for the same reason and is
filed nowhere.

### 2.4 Cluster ops stop at cg2

`commit_multicast_cg2`, `tma_load_2d_multicast_cg2` and `mma_walk_cg2` all
hardcode the 2-CTA pair. TK's `tma_cluster.cuh` takes a general cluster mask.
Larger clusters mostly matter for the GEMM shapes; the 2-CTA case is the one
Blackwell flash kernels use. Filed as **#49**, which narrows the remaining ask
to multicast as *replication* — the only thing a cluster larger than a pair
buys, and the thing the `_cg2` suffix rules out by name.

The barrier-addressing half is no longer part of it. **#50** replaced
`multicast_alias` — which reached the peer's barrier by masking the rank bit
out of a local address, and was therefore correct only for rank 0 of a pair —
with `Semaphore::at_rank`, a `mapa` that takes any rank.

### 2.5 No distributed shared memory (DSMEM)

TK has cluster-scope shared→shared copies. Still absent: #50 brought in the
*addressing* half, since `Semaphore::at_rank` names a peer's shared offset, but
nothing here moves data between two CTAs' shared memory.

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
beside the `SharedVec` type and closed on the type alone (1.4); **no open issue
carries the rest**. The per-column operand of `col_map` is no longer the hole
this entry described: `col_reduce` produces one (3.2), and `ldst::load_vec`
reads one out of shared memory.

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
other half and is tracked nowhere. Note also that everything above is warp
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
descriptor spellings in this crate and both are right: `mma_ab` bands 64 wide
with a 16-byte leading offset, and the new walks cover a 128-wide MN in one
instruction with the leading offset set to `SUBTILE_BYTES` — the second
subtile is reached through the descriptor, never by a step along the row. And
`tcgen05_mma_f16` and `tcgen05_mma_shared::<0, 1, 0>` do **not** lower to the
same PTX at the pinned revision, though they are semantically the same
instruction: the f16 wrapper emits the `.disable_output_lane` form with an
all-zero mask (four `mov.u32` of zero), the generic one emits
`.collector::a::discard` and no mask.

Still missing: FP8/FP4 operands (#16) and MXFP scale loading — both dtype
work, not shape work. `tcgen05_mma_sp_*` (sparse), `tcgen05_mma_ws_*`
(weight-stationary) and `tcgen05_shift_down` remain available upstream and
unused.

### 3.4 Tile-shape utilities

Landed with #7: `tril`/`triu`, `make_causal`/`make_causal_t`,
`left_fill`/`right_fill`/`upper_fill`/`lower_fill`, over a `RegTile::mask` that
selects at the layout's own `(row, column)`. Every one takes a **coordinate
origin** TK's signatures do not have — a `diagonal`, or a signed fill index —
because the tile is a sub-block of a larger matrix whose diagonal sits at
`query_base - key_base`; TK's origin-free form is right only for the diagonal
block. `broadcast_row`/`broadcast_col` landed with 3.1.

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

`pipeline::run` + `trait Job` is TK's `prototype::lcf` (load-compute-finish), and
it's a faithful port of that shape — still the only one, and still uncalled by
anything in `examples/`, which is worth knowing before porting a second: the
GEMM declines it deliberately, because `run` strides work items by `blockIdx.x`
and a cluster launch needs the whole cluster on the same item (**#51**).

Missing from `prototype/` (**#15**):

- **`lcsf`** — load-compute-**store**-finish. The store stage is what a training
  kernel with a producer-consumer epilogue needs; without it the store is folded
  into `finish` and can't overlap the next work item's load.
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

  The matching global *store* is checked by `examples/gemm.rs` instead, whose
  whole epilogue is one `store_rows` compared elementwise against an exact CPU
  reference. Still verified only by downstream kernels being numerically
  correct: the phase-parity rules, the pipeline scaffold, the masking ops
  (`mask_probe` is a codegen probe and says so in its own doc comment), and
  every cluster/multicast path — those last live in `examples/src/gemm.rs` and
  are covered by `modal_app.py::examples` rather than by a case here.
- **CI, in three tiers** (#17). `ci.yml` runs `fmt`, lockfile freshness,
  `clippy --all-targets -- -D warnings`, `test` and `cargo doc` under
  `RUSTDOCFLAGS=-D warnings` on every push and pull request, with no toolkit and
  no credential — the library's default features are device-only and
  `cuda-device` is ordinary Rust. `cuda.yml` is `modal_app.py::build`: the
  `host` feature, its tests and its docs, plus a real
  `cargo oxide build --arch sm_100a` of both kernel crates against the driver
  stub. That last is the tier that earns its money, because a
  post-monomorphization `const { assert!(..) }` in a tile shape fires at
  *codegen* and `cargo check` cannot see it. `gpu.yml` runs the B200 harness
  behind a label. `CI.md` carries the policy and the costs. What is still
  uncovered: fork pull requests get tier 1 only, since the two tiers above it
  need a Modal token.
- **Register cost is swept now, not sampled** (#60). Every claim in this file
  and in `examples/README.md` used to be measured at 32 and 128 columns, and
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

## Work order

The original order was issues #1–#16, structural items first, since they change
signatures across the crate and get expensive to retrofit. **Thirteen of the
sixteen have closed** — #1 through #13 — and the sections above name the work
beside each. Listing them again here only creates a second place for the status
to go stale, which is how this file earned #65.

What was open on 2026-07-28, and where it lives. `gh issue list` is the live
version; this is a snapshot with a date on it.

| # | Issue | Section |
| --- | --- | --- |
| 14 | Swizzle modes: 32B, 64B, unswizzled | 1.3 |
| 15 | `lcsf` and the rest of the prototype layer | 4 |
| 16 | FP8 element support | 1.2 |
| 42 | Reduction TMA stores — upstream rather than a local `ptx_asm!` | 2.3 |
| 49 | Cluster geometry beyond the 2-CTA pair | 2.4 |
| 50 | Cluster-scope semaphore arrival | 2.4 |
| 51 | `pipeline::run` cannot schedule a cluster | 4 |
| 29 | `expect_tx` byte accounting is hand-summed | — |

**Wanted and filed nowhere**, each named in its section above: a second
`FragmentLayout` (1.3, and 3.4 waits on it), the map and reduction
instantiations on shared tiles and shared vectors (3.1, 3.2 — the half of #13
that closed without them), an fp32 `TensorMapElement` (1.5), TMA prefetch (2.3),
DSMEM (2.5), and naming each op's scope in its type rather than in its index
math (1.1).

## cuda-oxide support at the pinned rev (`b099f64`)

Checked against the vendored source, since several gaps turn on it:

| Need | Status |
| --- | --- |
| Generic MMA over element kind | ✅ `tcgen05_mma_shared`/`_tensor`, `KIND` = 0 f16, 1 tf32, 2 f8f6f4, 3 i8 |
| Weight-stationary MMA per dtype | ✅ `tcgen05_mma_ws_{bf16,f16,tf32,e4m3,e5m2,e2m1,e2m3,e3m2}` |
| Sparse MMA | ✅ `tcgen05_mma_sp_*` (unused) |
| Register → TMEM (STTM) | ✅ full `tcgen05_st_*` family + `tcgen05_store_wait` |
| TMA store | ✅ `cp_async_bulk_tensor_{1..5}d_s2g` + commit/wait groups |
| TMA reduction store (add/min/max) | ❌ absent — needs `ptx_asm!` or an upstream PR (#42) |
| TMA prefetch | ❌ absent |
| f32 → fp8 | ✅ `cvt_rn_satfinite_{e4m3,e5m2}x2_f32` (+ `relu`) |
| f32 → fp4/fp6/e8m0 | ❌ absent — hand bit-packing |
| MXFP block-scale smem→tmem | ✅ `tcgen05_cp_*_b4x16_p64`, `_b6x16_p32` |
| `stmatrix` | ✅ `x2`/`x4`, plain and `trans` (we use `x2` only) |
| `ldmatrix` | ✅ `cuda_device::wmma::ldmatrix_{x1,x2,x4}` (+ `trans`); filed under `wmma`, but lowers for `sm_100a` — we use `x2` |

**FP8 is not blocked upstream.** FP4/FP6 and MXFP block scaling are.
