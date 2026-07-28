# Gaps vs ThunderKittens

Measured against HazyResearch/ThunderKittens `main` (~21.3k lines of headers under
`include/`, plus `prototype/`). ferro-kittens is 1851 lines.

The size difference overstates the functional gap and understates the structural
one. TK's surface is a **cross product** — every op is templated over element
type × layout × scope (thread/warp/warpgroup/`group<N>`) × tile-or-vector — so
one conceptual op like `row_sum` expands into dozens of instantiations. This port
is a **vertical slice**: one element type (bf16 operands, fp32 accumulate), one
swizzle (128B), one fragment layout, one MMA family (tcgen05), the ops two
working kernels actually needed. Most of what follows is widening that slice,
not adding new concepts.

Gaps are marked **[NON-GOAL]** where the omission is a deliberate consequence of
targeting Blackwell/tcgen05 only, **[SCOPE]** where it belongs to a different
layer in a Rust world, and left unmarked where it is a genuine hole.

---

## 1. Structural gaps

These change type signatures across the crate, so they are cheapest to do early.

### 1.1 No group/scope parameterization

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

### 1.2 Single element type

| | TK | ferro-kittens |
| --- | --- | --- |
| Operands | bf16, half, fp8 (e4m3/e5m2/e8m0), fp4 (e2m1), int8, tf32, fp32 | bf16 only |
| Accumulate | fp32, fp16, int32 | fp32 only |
| Complex | `crt`/`crv`/`cst`/`csv` throughout | — **[SCOPE]** |

`Element` now carries the byte width, the fp32 → element pack, and (via
`MmaElement`) the tcgen05 operand kind, so shape math, `ldst`'s pack, and
`mma`'s bounds are all generic — but `Bf16` is still the only impl (issue #2
did the trait work; #16 adds fp8). One bf16 fact remains hardcoded behind an
assert rather than guessed at: `mma`'s K=16 chunk geometry assumes 32 bytes
per chunk. The *instruction* left that list with #12 — the walks issue through
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
`BaseLdtm` as the sole `FragmentLayout`/`RowLayout` impl (issue #1 made the
layout a parameter; a second one is still unwritten). The 128B-only restriction
is documented as honest rather than
accidental — the subtile scheme depends on the 128-byte atom — but 64B tiles are
what you want for narrow operands, and no-swizzle for staging tiles that never
feed an MMA.

### 1.4 No shared vectors (`sv`)

Absent entirely. TK has `sv` as a first-class type with its own maps,
reductions, conversions, and TMA paths — it's how row statistics, biases, and
norm accumulators get staged. We keep row stats only in registers (`RegVec`),
which works for a fused flash kernel but blocks anything that needs to hand a
vector between warps through shared memory. **~200 lines.**

### 1.5 Global layout is one concrete builder, not a type

TK's `gl` is a template over element type and up to 4 dims, with each extent
independently compile-time or runtime, and it generates the matching TMA
descriptor as a member. `cgl` (complex) and `pgl` (multi-GPU) wrap it.

We have `PanelMap` — bf16, 3-D, tiled, fixed box shape — built by
`encode_bf16_panels`. It's correct and it's the only shape the kernels use, but
the agreement between map and `SharedTile` is enforced by that one builder being
generic, not by a general layout type. Widening this is prerequisite to most of
section 2.

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

Two things this deliberately is not. It is **fp32 only**, because the element
parameter belongs to `Element` and that is bf16-only until #2 — and fp32 out
of registers without a rounding step is the whole point of the entry. And it
**bounds-checks nothing**: the extents a TMA descriptor carries are absent
rather than forgotten, since predicating every value would be paid by the
epilogues that do divide. TK's ragged-tail loads want them back, and that is
what is left of this entry, along with the vector shapes (`RegVec`/`ColVec`
straight out of global memory) and group scope (1.1).

### 2.3 TMA store side — plain stores landed (#9), reductions absent

| TK | ferro-kittens |
| --- | --- |
| `load_async` | `tma_load`, `tma_load_at`, `tma_load_2d`, `tma_load_2d_multicast_cg2` ✅ |
| `store_async` | `tma_store`, `tma_store_2d` ✅ |
| `store_async_wait`, `store_commit_group`, `store_async_read_wait` | `tma_store_wait::<N>`, `tma_store_commit`, `tma_store_wait_read::<N>` ✅ |
| `store_add_async`, `store_min_async`, `store_max_async` | — |
| `prefetch` | — |
| im2col descriptors | — **[SCOPE]** (conv-only) |

What is left of this entry is the *reduction* stores, and they are a different
kind of missing: `cp.reduce.async.bulk.tensor` is absent from cuda-oxide at the
pinned revision in every form, so unlike the plain stores they are not a
transcription. `store_add_async` is what makes split-K and multi-CTA reduction
epilogues cheap, and getting it means a local `ptx_asm!` intrinsic (`ldst.rs`
sets the precedent) or an upstream contribution. Prefetch is absent upstream
for the same reason.

### 2.4 Cluster ops stop at cg2

`multicast_alias`, `commit_multicast_cg2`, `tma_load_2d_multicast_cg2` and
`mma_walk_cg2` all hardcode the 2-CTA pair. TK's `tma_cluster.cuh` takes a
general cluster mask. Larger clusters mostly matter for the GEMM shapes; the
2-CTA case is the one Blackwell flash kernels use.

### 2.5 No distributed shared memory (DSMEM)

TK has cluster-scope shared→shared. Absent here.

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

Missing: the shared-tile and shared-vector instantiations (#13). The per-column
operand of `col_map` can be carried and broadcast today but not yet *produced* —
that reduction is #6.

### 3.2 Reductions

TK: `row_max row_min row_sum row_prod row_reduce`, `col_*` mirrors, plus whole-
tile `max min sum prod` — on register **and shared** tiles.

We have quad-shuffle `quad_max`/`quad_sum` on `RegVec`/`RegTile` (i.e. row-wise
max and sum only), and `online_rescale`/`scale_rows` for the flash softmax
pattern. Missing: everything column-wise, `prod`, `min`, generic `row_reduce`,
and all shared-tile reductions. Column reductions are the awkward ones — the
fragment map replicates rows across a quad, so a column reduce needs a different
shuffle pattern than the one `quad_*` uses. #5 added the destination type
(`ColVec`) and the layout half (`ColLayout`) a column reduce would fill; what is
left is the strided shuffle across the 8 lanes of a column group.

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
second `FragmentLayout`, which the crate does not have. **~60 lines**, once one
exists.

### 3.5 Warp-level MMA fallbacks **[NON-GOAL]**

TK keeps `hmma16816`/`imma16832` (warp `mma.sync`) and the full wgmma warpgroup
path for Hopper and earlier. Deliberately absent — this library is `sm_100a`
only, no arch dispatch.

---

## 4. Missing kernel scaffolding

`pipeline::run` + `trait Job` is TK's `prototype::lcf` (load-compute-finish), and
it's a faithful port of that shape.

Missing from `prototype/`:

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

- **Host tests are address arithmetic only.** TK has `tests/` with a generated
  harness over the type/layout cross product. We have 73 `#[cfg(test)]` unit
  tests (`cargo test --features host`), all of them pure coordinate and pointer
  math — they prove the crate is self-consistent, not that it agrees with
  silicon.
- **Device tests cover six paths.** `device-tests/` (a separate kernel crate,
  run on a B200 through `modal_app.py`) checks the SWIZZLE_128B round trip
  against the TMA engine, `BaseLdtm`'s fragment ownership map against a
  position-encoding MMA at all three layout shapes, the `stmatrix` store path,
  STTM — as a register round trip through TMEM, and by restaging a probed
  accumulator into a second column band and re-draining it — `load_rows`,
  the one case whose source reaches registers with no descriptor between them,
  and (#12) all four MMA operand orders against one product, with a host-side
  blind-spot sweep and an untransposed control. The matching global *store* is
  checked by `examples/gemm.rs` instead, whose whole epilogue is one
  `store_rows` compared elementwise against an exact CPU reference. Everything
  else — the phase-parity rules, the pipeline scaffold, the cluster/multicast
  paths — is still verified only by downstream kernels being numerically
  correct.
- **No CI.** Nothing runs `cargo check` on the device surface automatically, let
  alone `--features host` on a CUDA box or the device tests on a GPU.
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
  Whether that trade is faster is not a `ptxas` question; nothing here times it.

---

## Work order

Tracked as issues #1–#16, filed in priority order. Structural items first, since
they change signatures across the crate and get expensive to retrofit.

| # | Issue | Section |
| --- | --- | --- |
| 1 | Generalize `RegTile` to a logical shape with a layout parameter | 1.3 |
| 2 | `Element` carries MMA kind and conversion, ready for sub-bf16 | 1.2 |
| 3 | Scope parameterization | 1.1 |
| 4 | Device test harness | 6 |
| 5 | Generic map mechanism and the standard elementwise op set | 3.1 |
| 6 | Reductions: column-wise, whole-tile, generic reduce | 3.2 |
| 7 | Masking and shape utilities | 3.4 |
| 8 | Generalize the global layout beyond `PanelMap` | 1.5 |
| 9 | TMA store path | 2.3 |
| 10 | Register to TMEM (STTM) | 2.1 |
| 11 | Direct global ↔ register load/store | 2.2 |
| 12 | MMA: `AtB`/`AtBt` walks, `mm` entry points, generic `KIND` routing — done | 3.3 |
| 13 | Shared vectors, and ops on shared tiles | 1.4 |
| 14 | Swizzle modes: 32B, 64B, unswizzled | 1.3 |
| 15 | `lcsf` and the rest of the prototype layer | 4 |
| 16 | FP8 element support | 1.2 |

## cuda-oxide support at the pinned rev (`b099f64`)

Checked against the vendored source, since several gaps turn on it:

| Need | Status |
| --- | --- |
| Generic MMA over element kind | ✅ `tcgen05_mma_shared`/`_tensor`, `KIND` = 0 f16, 1 tf32, 2 f8f6f4, 3 i8 |
| Weight-stationary MMA per dtype | ✅ `tcgen05_mma_ws_{bf16,f16,tf32,e4m3,e5m2,e2m1,e2m3,e3m2}` |
| Sparse MMA | ✅ `tcgen05_mma_sp_*` (unused) |
| Register → TMEM (STTM) | ✅ full `tcgen05_st_*` family + `tcgen05_store_wait` |
| TMA store | ✅ `cp_async_bulk_tensor_{1..5}d_s2g` + commit/wait groups |
| TMA reduction store (add/min/max) | ❌ absent — needs `ptx_asm!` or an upstream PR |
| TMA prefetch | ❌ absent |
| f32 → fp8 | ✅ `cvt_rn_satfinite_{e4m3,e5m2}x2_f32` (+ `relu`) |
| f32 → fp4/fp6/e8m0 | ❌ absent — hand bit-packing |
| MXFP block-scale smem→tmem | ✅ `tcgen05_cp_*_b4x16_p64`, `_b6x16_p32` |
| `stmatrix` | ✅ `x2`/`x4`, plain and `trans` (we use `x2` only) |
| `ldmatrix` | ✅ `cuda_device::wmma::ldmatrix_{x1,x2,x4}` (+ `trans`); filed under `wmma`, but lowers for `sm_100a` — we use `x2` |

**FP8 is not blocked upstream.** FP4/FP6 and MXFP block scaling are.
