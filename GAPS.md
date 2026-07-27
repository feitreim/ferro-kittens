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
did the trait work; #16 adds fp8). Two bf16 facts remain hardcoded behind
asserts rather than guessed at: `mma` still calls the kind-specialized
`tcgen05_mma_f16` instead of the `KIND`-generic intrinsic (#12), and its K=16
chunk geometry assumes 32 bytes per chunk. FP8 is the consequential one: it
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
Both sides are `16x256b` at `[16, 16]` granularity, so composing a larger
shape stays the caller's on either side. The store leaves its wait to the
caller, since a store's registers are consumed at issue.

### 2.1a Shared ↔ register — done (#21)

TK's `shared_to_register.cuh`. `ldst::store_fragment` (`stmatrix`) and
`ldst::load_fragment` (`ldmatrix`) over one shared `SwizzledChunks` derivation,
`ldst::fragment_address`, that both directions call — so the address map is
host-testable and the two cannot drift. Missed by this inventory until the
examples crate (#18) hit it: a kernel whose input is not an MMA operand had no
way to reach registers at all.

Same `[16, 16]` granularity as 2.1, and the same restriction as every other
`SwizzledChunks` user — one swizzle subtile of width (#25). Composing larger
shapes on either side is #22.

### 2.2 No global ↔ register path

TK has `global_to_register.cuh` (`load`/`store`) at both warp and group scope —
the non-TMA path for small, irregular, or unaligned accesses. We only reach
global memory through TMA into shared. Any epilogue that wants to write fp32
straight out of registers currently has to round-trip through a shared tile.

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

### 3.3 MMA shape and dtype coverage

TK's tcgen05 layer: `{mm, mma} × {AB, ABt, AtB, AtBt} × {1-CTA, 2-CTA}` = 16
entry points, generic over element type, plus `load_mxnv_scale_async` for
block-scaled MXFP.

We have `mma_abt`, `mma_ab` (single-CTA) and `mma_walk_cg2` (2-CTA, runtime
K-major/MN-major via `OperandWalk`).

Missing: `AtB`/`AtBt` walks; the `mm` (zero-initializing) variants as distinct
entry points — we thread `enable_d` by hand, which is the same instruction but
puts the accumulator-init invariant on the caller; FP8/FP4 operands; MXFP scale
loading. Adding `AtB`/`AtBt` is mostly descriptor transpose-bit work in the same
walk skeleton. **~80 lines for the missing walks.**

### 3.4 Tile-shape utilities

Absent: `transpose`, `swap_layout`, `tril`/`triu`, `make_causal`/`make_causal_t`,
`left_fill`/`right_fill`/`upper_fill`/`lower_fill`, `copy` between layouts.
`broadcast_row`/`broadcast_col` landed with 3.1.

`make_causal` is the notable one — every causal attention kernel needs it, and
without it the masking is open-coded in the kernel against raw fragment indices,
which is exactly the index math this library exists to remove. **~100 lines.**

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
  harness over the type/layout cross product. We have 19 `#[cfg(test)]` unit
  tests, all of them pure coordinate and pointer math — they prove the crate is
  self-consistent, not that it agrees with silicon.
- **Device tests cover four paths.** `device-tests/` (a separate kernel crate,
  run on a B200 through `modal_app.py`) checks the SWIZZLE_128B round trip
  against the TMA engine, `BaseLdtm`'s fragment ownership map against a
  position-encoding MMA at all three layout shapes, the `stmatrix` store path,
  and STTM — as a register round trip through TMEM, and by restaging a probed
  accumulator into a second column band and re-draining it. Everything else — the phase-parity rules, the pipeline scaffold, the
  cluster/multicast paths, the MMA walks beyond the single `M128_N64` the probe
  issues — is still verified only by downstream kernels being numerically
  correct.
- **No CI.** Nothing runs `cargo check` on the device surface automatically, let
  alone `--features host` on a CUDA box or the device tests on a GPU.

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
| 12 | MMA: `AtB`/`AtBt` walks, `mm` entry points, generic `KIND` routing | 3.3 |
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
