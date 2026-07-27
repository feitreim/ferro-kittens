# examples

Four kernels written the way we want them to read, and a header on each saying
whether it compiles.

The value of this crate is the **diff** between what the kernels want to say and
what `kittens` can express. An aspirational example is not a placeholder — it is
a statement of a missing API in the only terms that matter, which is what a
kernel author has to type. This file collects that diff across all four.

Its own workspace, like `device-tests`, so `cargo` at the repo root never sees
it. (The root `Cargo.toml` also needs `autoexamples = false`: `examples/` is one
of cargo's own target directory names, and `exclude` does not stop target
auto-discovery, so without it `examples/src/main.rs` gets compiled as an example
target *of the library*.)

## Status

| Kernel | Status | Blocked on |
| --- | --- | --- |
| [`gemm`](src/gemm.rs) | **compiles** | — (three gaps worked around in-file, all marked) |
| [`flash_forward`](src/flash_forward.rs) | aspirational | #7, #11, #22, #23, #31 + scalar broadcasts |
| [`softmax`](src/softmax.rs) | aspirational | **#9, and nothing else** |
| [`layernorm`](src/layernorm.rs) | aspirational | #3, #9, #13, #22 + scalar broadcasts |

Nothing on those lists is arithmetic any more. #5 and #6 between them closed
every elementwise op and every reduction the four kernels asked for, and #21 and
#25 closed both halves of the shared ↔ register path. `softmax` is down to one
item: **#9**, the TMA store, so the result can leave. What remains across the
four is movers (#22, composition only), the store side (#9), masking (#7), and
one structural gap (#3 + #13, layernorm's block-scope statistic).

The one arithmetic leftover is the **scalar broadcast** forms —
`RegTile::scale`/`shift`, `RegVec::scale`/`shift`, a free `rsqrt(f32)`, and
`RegTile::add_assign` — which #5's list named and #5's implementation did not
ship. A tile or vector combined with a bare `f32` still has to splat it into a
whole operand first. They have no issue of their own; #31 covers the in-place
half of the argument.

```sh
cargo oxide build kittens-examples --arch sm_100a   # the default set: gemm
cargo check --features flash                        # read flash's gap list
cargo check --features aspirational                 # all of them
```

Each aspirational kernel is behind its own feature and off by default, so
everything in the default build genuinely compiles and the two kinds are
distinguishable without running anything. Turning a feature on prints the
missing API as compiler errors, at the call sites that want it. Verified: every
error is `unresolved import`, `no method named`, or an unsatisfied
`FragmentLayout` bound — there is nothing in these files that fails for any
reason other than the API not existing.

`gemm` is confirmed to build end-to-end through `cargo oxide build --arch
sm_100a` (`modal run modal_app.py::build`). It has **not been run**: issue #18
needs no GPU. Treat it as a statement about expressiveness, not a numerical
result — in particular the 2-CTA operand split is the first thing to check on
hardware.

---

## What the four kernels collectively demand

### Already filed

**#5 — generic map and the standard elementwise set. Landed.** The map
mechanism and every vector/vector and broadcast form the kernels asked for:
`exp2`, elementwise `mul`, `sub_row`, `mul_row`, `div_row`, `mul_col`,
`add_col`, `RegVec::rsqrt`. **Not** landed, and still named at call sites in
`flash_forward` and `layernorm`: the *scalar* broadcasts `scale`/`shift` on
both `RegTile` and `RegVec`, a free `rsqrt(f32)`, and `RegTile::add_assign`.
A tile or vector combined with a bare `f32` has to splat it into a whole
operand first.

> Note the asymmetry the examples exposed: `RegVec` already had `exp2`, `max`,
> `sub`, `add_assign`, `mul_assign`. `RegTile` had none of them. Every op the
> vector has, the tile wants — which is exactly the argument for the mechanism
> rather than another round of hand-written methods. It still holds for the
> in-place forms: the vector has them and the tile does not (#31).

**#6 — reductions. Landed.** `row_reduce`/`col_reduce`/`tile_reduce` over a
`ReduceOp` (a `BinaryOp` with an identity), with `row_max`/`row_min`/`row_sum`/`row_prod`, the `col_*` mirrors,
and `tile_max`/`tile_min`/`tile_sum`/`tile_prod` derived from them. Both halves
are in the library now: a thread's own registers, then the `shuffle_xor`
butterfly over the lanes the ownership map spreads the folded axis across —
masks 1,2 for a row (what `quad_max`/`quad_sum` already were), 4,8,16 for a
column, all five for a tile.

All of it is **warp scope**. `tile_sum` folds one warp's band; layernorm's
group-norm statistic spans four warps' bands and still needs them to agree,
which is the #3 + #13 entry below.

**#7 — masking.** `make_causal`, for flash. One correction to the issue as
filed: it describes ThunderKittens' signature, which takes no coordinate origin.
A flash kernel masks a `[queries, keys]` band whose diagonal sits at
`query_base - key_base`, so the op it needs is
`make_causal_at(lane, query_base, key_base, fill)`. Without the origin the op
is unusable for anything but a single-block attention.

**#9 — TMA store.** Both normalization kernels end with one. `tma_store`,
`tma_store_commit`, `tma_store_wait::<N>()`.

**#11 — global ↔ register.** Flash's epilogue and the GEMM's. The GEMM's is the
one worth reading: it is fifteen lines of open-coded `RegTile::coordinate`
arithmetic inside a `GAP` fence, and it is exactly the index math this library
exists to delete.

**#13 — shared vectors.** `gamma`/`beta` in layernorm: parameters shared by
every row, wanted in shared memory with their own TMA path (a `[1, N]` box is
not a tile, and making it one wastes a swizzle atom of padding per row).

**#3 — scope parameterization.** Layernorm's whole-tile statistic, and the one
demand #6 deliberately did not meet. A tile owned by four warps needs the warps
to agree.

> A finding from writing #6 against this, worth having before #3 is picked up:
> **`Scope` as filed cannot express the block reduction.** The proposed trait is
> `WARPS` / `THREADS` / `rank()` / `sync()`, and a block-scope fold needs none
> of those so much as it needs *storage* — somewhere for each warp's partial to
> live between the two barriers. `groupnorm_tile` guesses at that today with a
> bare `*mut f32` carved out of its own shared plan, and the library cannot
> allocate one on its behalf: the shared plan belongs to the kernel. So the
> block reduction's signature is either "take a scratch pointer" (not scope
> parameterization at all) or "take #13's `SharedVec`". #3 and #13 have to be
> sequenced together, and #6 does not force either.

**#8 — global layout.** Not reachable from a kernel at all, which is why it does
not appear in the code: `encode_bf16_panels` builds one shape of tensor map (3-D
bf16 panels), and the GEMM's 2-D operands and layernorm's parameter vectors both
want another. This is why the crate ships no host launcher.

**#12 — MMA coverage.** Not demanded. `mma_abt`, `mma_ab` and `mma_walk_cg2`
covered every multiply in all four kernels. Worth recording as a negative
result: the MMA layer is the part of this library that is finished.

---

### Came from writing kernels, not from ThunderKittens' feature list

The most valuable output here. Ordered by how badly it hurts. Items 1–6 were
filed as **#21**, **#22**, **#25**, **#23** and **#24** (which covers both
cluster-scope entries); **1 and 3 have since shipped**. The numbers are noted
inline and the prose is kept as written, because it is the argument rather than
the ticket.

#### 1. ~~There is no shared → register path at all~~ — **closed by #21**

This was the item this section was written for: `ldst` was store-only, so any
kernel whose input is not an MMA operand could not start, and softmax was
blocked before its first line.

`ldst::load_fragment` is now the `ldmatrix` half, sharing the store side's
address derivation (`ldst::fragment_address`, host-tested) and validated
against silicon by the `ldmatrix map` device case. `cuda_device::wmma::
ldmatrix_x2` lowers fine for `sm_100a` — the note that `ldmatrix` "lives only in
the Hopper `wmma` path" was about where the function is filed, not about what it
compiles to.

What remained of this item was shape, not direction — and item 3 below closed
the other half of that: `chunk_writer` spans stacked subtiles, so the path
exists at every width the library can describe. Only item 2 is left, and it is
a cost rather than a hole: `load_fragment`/`store_fragment` move one
`[16, 16]` block, so a band is a loop the kernel writes.

#### 2. The movers are per-`[16, 16]`-block, so every kernel writes the same loop (#22)

`TmemTile::fragment_tile` returns a `[16, 16]` `Fragment`, and
`ldst::store_fragment` takes one. A kernel that wants a `[32, 128]` band writes
a four-deep block-composition loop to assemble it — `gemm.rs` does, the flash
kernel would, and `fragment_probe` in `device-tests` already does. The
composition is a property of the layout; `reg.rs`'s own test
(`fragment_blocks_tile_the_bigger_shapes`) states it as an invariant, and then
no function implements it.

Wanted (filed as #22): `TmemTile::drain::<M, N, L>(row, column) -> RegTile<M, N,
L>`, `ldst::load_tile(tile, row, column, lane) -> RegTile<M, N, L>` and
`ldst::store_tile(tile, row, column, lane, values)`, with the block loop inside.
This is now the *whole* remaining mover gap in both directions: #21 gave the
load, #25 gave both of them the tile's full width, and `softmax.rs` writes the
loop twice — once each way — for want of this one function.

#### 3. ~~`SwizzledChunks` cannot span stacked subtiles~~ — **closed by #25**

`SharedTile::chunk_writer` const-asserted a one-subtile tile, so only tiles 64
bf16 columns wide had a register ↔ shared path at all — both directions, since
`load_fragment` addresses through the same cursor. Softmax's `[128, 128]` tile
could be neither read nor written back.

The cursor now walks subtiles the way `tma_load` does, and for the same reason:
a stacked subtile is `SUBTILE_BYTES = rows * 128` further along, so subtile `i`
row `r` is the tile's 128-byte row `i * rows + r` and the swizzle — which keys
off *physical* address bits `[9:7]` — takes that row's phase. Chunk indices
count across the whole logical row, so `fragment_address` needed no change at
all. Checked against silicon at width by the `swizzle/stmatrix/ldmatrix … wide`
device cases, and by a 4-row tile whose second subtile starts mid-period —
which is the only shape where an absolute phase and a per-subtile one differ.

#### 4. The `RegTile` shape set is closed by the library (#23)

`BaseLdtm` implements `FragmentLayout` for `(16,16)`, `(32,32)` and `(32,128)` —
because each shape is a line of `base_ldtm_shapes!` *inside* `src/reg.rs`. Flash
wants a `[32, 64]` score band, and an out-of-tree kernel cannot add it: the
macro is not exported and the trait cannot be implemented for a foreign type.
Dodging `generic_const_exprs` with an associated type is the right call; making
the set of expressible tiles a closed library decision is a side effect nobody
chose.

Wanted: export the macro, or a blanket impl over the shapes the layout actually
supports.

#### 5. No cluster-scope TMEM allocation (#24)

`tmem::alloc_block` is `tcgen05.alloc.cta_group::1`. A `cg2` accumulator is one
allocation spanning the CTA pair, so exactly one warp in the leader may issue
it — and the peer, which drains its own 128 rows of that same allocation, can
learn the address only by reading the leader's staging word over DSMEM. Both
halves are hardware facts about a cluster accumulator; both are open-coded in
`gemm.rs`'s first `GAP` block. `tcgen05_alloc_cg2` / `tcgen05_dealloc_cg2` /
`tcgen05_relinquish_alloc_permit_cg2` are all present upstream and unused.

Wanted: `tmem::alloc_cluster(slot, columns) -> u32`.

#### 6. No cluster-scope semaphore arrival (#24)

A 2-CTA MMA consumes four tiles staged by two CTAs, so the issuer needs one
barrier that says *the whole stage is present* — the peer's TMA has to complete
on the leader's copy. mbarrier addresses are cluster-mappable and this is the
standard pair-wide producer handoff, but `Semaphore` is CTA-scoped by
construction and says nothing about it. `Semaphore::multicast_alias` is the one
cluster-aware thing in `sync.rs`, and it solves the opposite problem (one
barrier per CTA, one transfer).

Note also that `Semaphore::expect_tx` lowers to `mbarrier.arrive.expect_tx
… .shared::cta`, so it cannot charge a remote barrier even if handed one;
upstream's `mbarrier_arrive_cluster` takes a remote address but carries no
transaction bytes. The shape of the fix is a design question, not a
transcription.

Wanted: `Semaphore::at_rank(rank)`, and a decision about the byte accounting.

#### 7. `pipeline::run` cannot schedule a cluster

The work-item map is `blockIdx.x` strided by `gridDim.x`, so the two CTAs of a
cluster get *different* items. The GEMM therefore does not use the persistent
scaffold at all — it is one cluster per output tile with the K loop as its
pipeline. `prototype::lcf` predates clusters in ThunderKittens too, so this is
not a porting oversight; it is the scaffold needing #3's `Scope` before it can
describe who a work item belongs to.

Related, and cheaper: #15's `lcsf` is filed as depending on #9, which is right,
but the GEMM shows a second reason to want it — with the store folded into
`finish`, a persistent GEMM cannot overlap its epilogue with the next tile's
first K loads.

#### 8. Multicast has no geometry to live in

`tma_load_2d_multicast_cg2`, `commit_multicast_cg2` and `mma_walk_cg2` all
hardcode the 2-CTA pair (GAPS §2.4). But under a 2-CTA UMMA *both* operands are
already split across the pair, so nothing is replicated and there is nothing for
the multicast load to do — the GEMM uses the commit and the walk and cannot use
the load. Multicast starts paying at cluster ≥ 4 (2×2: `A` broadcast along the
N axis, `B` along the M axis), which the `_cg2` suffix rules out. Generalizing
the cluster mask is filed nowhere; §2.4 calls it a nice-to-have on the strength
of flash using pairs, and the GEMM is the counterexample.

#### 9. Smaller things, each one line of API

- **`expect_tx` byte accounting is hand-summed.** Every producer writes
  `(ATile::BYTES + BTile::BYTES) as u32` and has to keep it in step with the
  loads it issued. A tile knows its own size; `Semaphore::expect_tiles` or a
  charge returned by `tma_load` would make the two impossible to disagree.
- **Coordinate-dependent ops need `lane` passed in.** `store_fragment` and
  `RegTile::coordinate` take it explicitly, so the invented `make_causal_at`,
  `load_tile` and `store_tile` do too. Consistent, but every call site writes
  `warp::lane_id()` into a variable that the op could have read itself. Worth
  settling deliberately when #5 lands, since it fixes the convention for
  thirty functions at once.

---

## What the examples confirm already works

Worth recording, because the promotion of an example from aspirational to
compiling is the thing that proves an issue is finished, and half of this
library is already at that bar:

- **The MMA layer.** `mma_abt`, `mma_ab` and `mma_walk_cg2` covered every
  multiply in four kernels, in the layouts they wanted, with no gaps. Each
  builds its own instruction descriptor from the walk it issues and the
  operand element, so a kernel names only its accumulator band's `MmaShape`
  and cannot pair a walk with transpose bits that disagree (#30). The one
  field still stated by a caller is `mma_walk_cg2`'s element
  (`mma_walk_cg2::<Bf16, CHUNKS>`) — an `OperandWalk` has erased it, and #12
  owns whether the walk carries it back.
- **TMEM segment carving.** Flash splits one allocation into its `[128, 64]`
  scores and `[128, 128]` output with `TmemTile::split_columns`, so no kernel
  here adds a column offset to a bare TMEM address (#28).
- **The pipeline primitives.** `SharedTileRing` + `SemaphoreRing` express the
  GEMM's K pipeline exactly, including the subtle part: `free` is released by
  the MMA's own commit, so the accumulator instruction rather than a thread is
  what proves the operand has been read. The `index → (stage, parity)`
  arithmetic never appears in the kernel.
- **`online_rescale`.** The one genuinely subtle piece of flash attention is a
  single call.
- **The swizzle and fragment layers.** No kernel here spells a swizzle phase, a
  chunk index, a subtile stride or a descriptor field — at any tile width,
  since #25. That is the library working.
