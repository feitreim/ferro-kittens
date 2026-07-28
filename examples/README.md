# examples

Four kernels written the way we want them to read, and a header on each saying
whether it runs, whether it only compiles, or whether it is aspirational.

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
| [`gemm`](src/gemm.rs) | **runs** — exact against a CPU reference | — (two gaps worked around in-file, both marked) |
| [`softmax`](src/softmax.rs) | **runs** — within 2⁻⁸ of a CPU reference | — |
| [`flash_forward`](src/flash_forward.rs) | aspirational | #11, #31 |
| [`layernorm`](src/layernorm.rs) | `layernorm_rows` **compiles**; `groupnorm_tile` aspirational | #3 (and #2 under it), for `groupnorm_tile` only |

Two of the four **run** rather than merely compile, which is a strictly stronger
claim and the one worth holding the rest to: a launcher, a CPU reference, and an
exit code.

`softmax`'s blocked-on column is empty, and it is the first example to have
gone the whole way — five gaps, then one, then none. **#9** was the last of
them, and closing it took nothing but the store: `SharedTile::tma_store` walks
the same boxes `tma_load` does, and `tma_store_commit` / `tma_store_wait::<0>()`
are the completion side.

`modal run modal_app.py::bench` is that claim with a clock on it (#40): both
kernels that run, at several sizes each, every one checked against the same CPU
reference *before* it is timed — there is no path through `bench.rs` that prints
a throughput figure for a launch it did not verify. The metric follows what the
kernel is bound by, TFLOP/s for the GEMM and GB/s for the memory-bound softmax,
and the sizes are picked to cross a regime rather than to be large, so what the
table shows is the shape of a curve and not one headline number.

Nothing on the remaining lists is arithmetic, and nothing on them is a mover.
#5 and #6 between them closed every elementwise op and every reduction the four
kernels asked for and **#38** closed the scalar-operand forms they left behind;
#21 and #25 closed both halves of the shared ↔ register path, #22 closed the
composition on top of them, in both directions and against both memories, and
**#13** closed the vector shape of shared memory. **#23 and #7 closed together**
— flash could not name its `[32, 64]` band, and the unsatisfied bound on that
band was what kept `make_causal` off the error list, so shipping either alone
would have left a confusing state. What remains across the four is the global ↔
register path (#11), one in-place form (#31), and one structural gap (#3,
layernorm's block-scope statistic).

`layernorm` is the first example to be **split** by a landed issue rather than
promoted whole, and the split is the honest reading of what #13 bought.
`layernorm_rows` is in the default build and gets a real `sm_100a` compile;
`groupnorm_tile` beside it keeps the `layernorm` feature and its `WANT` block,
because its statistic spans four warps. The `#[cfg]` sits on the *kernel* — the
`#[cuda_module]` macro honours it — so the file that documents the gap is the
file that builds, and one aspirational kernel does not gate a finished one.

The one arithmetic entry still open is `RegTile::add_assign`, and it is **#31**
rather than a leftover: it is an *in-place* form, and #31 exists to land every
in-place map variant at once. #38 deliberately did not add a one-off. See §
"in-place versus by-value" below for what #38 measured about it.

```sh
cargo oxide build kittens-examples --arch sm_100a   # the default set: gemm, softmax, layernorm_rows
cargo oxide run kittens-examples                    # and run the ones with launchers, on a B200
cargo check --features flash                        # read flash's gap list
cargo check --features aspirational                 # every gated kernel's
```

From the repo root: `modal run modal_app.py::build` for the first,
`modal run modal_app.py::examples` for the second, and
`modal run modal_app.py::gaps` for the last two — it prints both gap lists and
never fails, since an empty one is a finding rather than an error.

Each aspirational kernel is behind its own feature and off by default, so
everything in the default build genuinely compiles and the two kinds are
distinguishable without running anything. Turning a feature on prints the
missing API as compiler errors, at the call sites that want it. Verified: every
error is `unresolved import`, `no method named`, or an unsatisfied
`FragmentLayout` bound — there is nothing in these files that fails for any
reason other than the API not existing.

Two kernels are still behind a feature — `flash_forward` and `groupnorm_tile`,
the second of which shares a file with a kernel that is not — and their error
lists are now short enough to write out. Both are read off
`modal run modal_app.py::gaps`, which checks each feature on its own so an
error belongs to a known kernel:

- **`flash_forward`** — `global::store_rows` (#11), `RegTile::add_assign`
  (#31). Both are at the ends of the kernel; everything between the score band
  arriving in registers and the probabilities going back to shared is now
  library API. The list was three errors until #23 landed, and how the third
  went is worth keeping: `make_causal_at` (#7) was wanted all along and never
  appeared as its own error, because it is called on the `[32, 64]` band whose
  `FragmentLayout` bound already failed, and an unsatisfied bound on the
  receiver suppresses method resolution on it. Counting errors would have said
  #7 had landed.
- **`layernorm`** — `sync::block_reduce_sum` (#3), and nothing else. One error,
  in `groupnorm_tile` only; `layernorm_rows` left the feature gate with #13.
  Verified by running the check, not by reading the file.

Nothing named `scale`, `shift` or `rsqrt` is in either list any more (#38), and
nothing named `FragmentLayout` or `make_causal` (#23 + #7). That last pair is
worth keeping as a note on how to read this section: `make_causal_at` never
appeared as its own error, because it was called on the `[32, 64]` band whose
layout bound failed first and an unsatisfied bound on the receiver suppresses
method resolution on it. Counting errors would have said #7 was already
closed. Closing #23 is what made #7 surface.

`softmax` **runs** too, and what its check can claim is different from the
GEMM's in a way worth writing down. A softmax has an `exp2` and a divide in it,
so `==` is not available; instead each row's 128 inputs are a permutation of
the multiples of 1/8 below 16, which makes each row's 128 outputs distinct with
the closest pair 9% apart, against a 2⁻⁸ tolerance. Exactness is not the only
way to make a failure unambiguous — separating the right answer from every
wrong one by forty times the noise floor also works.

`gemm` **runs**, and is the first numerical result this library has produced.
`gemm::check` launches it over `[512, 256] x [256, 256]ᵀ` on a B200 and compares
every element against a CPU reference. The operands are integers in `[-3, 3]`
and `[-2, 2]`, so every product and every partial sum is exact in bf16 and fp32
alike and the comparison is `==` — a mismatch is a wrong coordinate, a wrong
stride or a wrong operand half, and never a rounding artifact to argue about.
`main` runs every kernel that has a launcher and exits non-zero on a wrong
number; off a GPU it prints the status table and stops.

Getting there cost one real bug, recorded here because the aspirational-vs-
compiling distinction is exactly what missed it. The kernel compiled, and hung.
Both CTAs' TMA loads have to complete on the *leader's* stage barrier, and the
kernel mapped that barrier by hand — but a plain `cp.async.bulk.tensor`
completes on a barrier in the **issuing** CTA's own shared memory, so the
peer's 24 KiB never reached the count the leader had charged for 48 KiB. §8
below used to say this kernel "cannot use" `tma_load_2d_multicast_cg2`, on the
grounds that a 2-CTA UMMA replicates nothing. That is true and beside the
point: the multicast form's barrier operand is `.shared::cluster`, which is the
only way one CTA may name another's barrier. With the CTA's own bit as the
whole mask it delivers to exactly one CTA and completes on the leader — no
replication, right address space. `Semaphore::multicast_alias` was already in
the library, filed under the opposite problem.

---

## What the four kernels collectively demand

### Already filed

**#5 — generic map and the standard elementwise set. Landed.** The map
mechanism and every vector/vector and broadcast form the kernels asked for:
`exp2`, elementwise `mul`, `sub_row`, `mul_row`, `div_row`, `mul_col`,
`add_col`, `RegVec::rsqrt`.

> Note the asymmetry the examples exposed: `RegVec` already had `exp2`, `max`,
> `sub`, `add_assign`, `mul_assign`. `RegTile` had none of them. Every op the
> vector has, the tile wants — which is exactly the argument for the mechanism
> rather than another round of hand-written methods. It still holds for the
> in-place forms: the vector has them and the tile does not (#31).

**#38 — scalar operands. Landed.** What #5 listed and did not ship, and the
last arithmetic on any of these lists: `scale`/`shift`/`clamp_min`/`clamp_max`
on `RegTile`, `RegVec` and `ColVec`, plus a free `rsqrt(f32)` for a statistic
that never becomes a vector (`groupnorm_tile`'s variance).

The reason it was missed is worth keeping, because it decided the fix.
`UnaryOp::apply` is an associated function on a unit struct, so `scale(k)` has
**nowhere to put `k`** — the mechanism could not express it, which is a hole in
the trait shape rather than a missed line item. Treating the scalar as the
*second operand* of the existing `BinaryOp` set closes it:
`scalar_map::<Op>(k)` is one method per family, `scale` is `Mul` and `shift` is
`Add`, and `Div`/`Sub`/`Max`/`Min` against a constant are reachable with no new
op at all.

#### in-place versus by-value

#38's probe (`scalar_map_probe_*` in `device-tests`) prices the same tile-loop
three ways — the pre-#38 `bin_map::<Mul>(splat(k))`, the new `scale(k)`, and an
in-place multiply written through `set` — at both probe widths:

| | 32 columns | 128 columns |
| --- | --- | --- |
| `bin_map::<Mul>(splat(k))` | 48 regs, no spill | 255 regs, 124 B spill |
| `scale(k)` | 48 regs, no spill | 255 regs, 60 B spill |
| in place, through `set` | 48 regs, no spill | 252 regs, no spill |

Two things follow. `scale` is strictly better than what a kernel had to write
before it, at every width — half the spill at 128 and free at 32. And the
by-value/in-place gap #31 was filed about **reproduces for a scalar operand**:
at 128 columns both by-value forms spill where the in-place one does not. That
is the argument for #31 covering `scalar_map` alongside `row_map` and
`add_assign`, rather than the three being separate decisions.

But "by-value costs registers" is probably the wrong lesson, and an
intermediate form of the same probe says why. Written as
`out_acc.add(block.scale(k))` — the scaled tile chained inside another map, so
`block` and its scaled copy are both live at the peak — it took 84 B of spill,
where the rebinding `block = block.scale(k)` took none. Same op, same width,
same by-value map; only the peak liveness differs. Read that way, three results
that look unrelated line up: #5's `row_map` cliff (a second tile live beside
the accumulator it replaces), the 60 B above (same), and #22's composed movers
coming out *cheaper* than the loops they replaced (36 against 52 — the
composed form has one band live where the loop kept a band and a fragment). The
variable is how many tiles are live at once, not whether a method takes `self`
by value. If that holds, #31's job is narrower than it looks: in-place forms
pay only where the input is live across the op, and a by-value map whose input
dies at the call already costs nothing.

**#6 — reductions. Landed.** `row_reduce`/`col_reduce`/`tile_reduce` over a
`ReduceOp` (a `BinaryOp` with an identity), with `row_max`/`row_min`/`row_sum`/`row_prod`, the `col_*` mirrors,
and `tile_max`/`tile_min`/`tile_sum`/`tile_prod` derived from them. Both halves
are in the library now: a thread's own registers, then the `shuffle_xor`
butterfly over the lanes the ownership map spreads the folded axis across —
masks 1,2 for a row (what `quad_max`/`quad_sum` already were), 4,8,16 for a
column, all five for a tile.

All of it is **warp scope**. `tile_sum` folds one warp's band; layernorm's
group-norm statistic spans four warps' bands and still needs them to agree,
which is the #3 entry below — #13 has since supplied the storage half of it.

**#7 — masking. Landed**, with the correction the examples surfaced. The issue
as filed describes ThunderKittens' signature, which takes no coordinate origin;
a flash kernel masks a `[queries, keys]` band whose diagonal sits at
`query_base - key_base`, and without the origin the op is unusable for anything
but a single-block attention. What shipped is `RegTile::make_causal(lane,
diagonal, fill)` with `tril`/`triu` on the same parameter, the four fills
(`left`/`right`/`upper`/`lower`) on a signed index for the same reason, and
`mask(lane, fill, keep)` underneath them all — a select at the layout's own
`(row, column)`, in place.

`make_causal_at(lane, query_base, key_base, fill)` is the spelling flash uses,
and taking the two origins rather than their difference is not sugar: the
difference is *negative* for every band above the diagonal, both bases are
`u32` at the call site, and `query_base - key_base` written there wraps to
+4 billion and masks nothing. The host tests carry that case, along with bands
wholly above and wholly below the diagonal — the two the origin-free form gets
silently wrong.

Not landed, as the issue's own sequencing asked: `transpose`, `swap_layout`,
`copy` between layouts. All three need a second `FragmentLayout` to mean
anything, and `BaseLdtm` is still the only one in tree.

**#9 — TMA store. Landed**, and it is what promoted `softmax` to **runs**.
`SharedTile::tma_store` / `tma_store_2d` mirror the load paths box for box;
`tma_store_commit` and `tma_store_wait::<N>()` are the completion side, and
they are *not* a `Semaphore` — a load's destination is shared memory where an
mbarrier can count arriving bytes, a store's is global memory where none can,
so a bulk store completes by counting per-thread groups with no barrier and no
byte accounting. `tma_store_wait_read::<N>()` is the weaker of the two waits:
it says the engine has finished *reading* the shared tile, which is what a
pipelined epilogue needs to recycle its buffer (#15's `lcsf`) and is never
enough as a kernel's last wait.

Not landed, and tracked separately as the issue asked: the **reduction** stores
(`store_add_async` and friends). `cp.reduce.async.bulk.tensor` is absent from
cuda-oxide at the pinned revision in every form, so that one is a `ptx_asm!`
intrinsic here or an upstream contribution — and nothing in the plain store
path prejudges which.

**#11 — global ↔ register.** Flash's epilogue and the GEMM's. The GEMM's is the
one worth reading: it is fifteen lines of open-coded `RegTile::coordinate`
arithmetic inside a `GAP` fence, and it is exactly the index math this library
exists to delete.

**#13 — shared vectors. Landed**, and it is what promoted `layernorm_rows`.
`kittens::shared::SharedVec<E, N>` is one flat run of elements with its own
single-box TMA path — rank 1 or a row of a rank-2 batch, both directions — and
`ldst::load_vec`/`store_vec` are the `ColVec` broadcast on top of it.

The decision worth recording is that it is **unswizzled and does not go through
`Swizzle` at all**, rather than riding a new `SwizzleNone` mode. `ATOM_BYTES`
does two jobs in `shared.rs` — swizzle period *and* TMA box width — and a
vector wants neither: its box is `N` wide, which no mode marker can carry, and
any atom small enough to divide a short vector would split it into stacked
subtiles the engine fetches one instruction each. A 128-byte atom is the
padding the issue was filed about. Making `SwizzleNone` fit would have meant
deciding what `ATOM_BYTES` means for an unswizzled *tile*, which is #14's
question; the vector needed none of it. `SharedVec<Bf16, 32>` is 64 bytes,
where the narrowest tile that even typechecks spends 128 on its single row.

`TileBox` now has its second impl, which is what #8 and #2 were both waiting
on: `BOX_COLS = N`, `BOX_ROWS = 1`, `SWIZZLE = NONE`, so a rank-1
`GlobalLayout` finally has something to hand a box to. Rank 1 is no longer
"expressible but untried" — the `shared vector round trip` device case runs one.

**#3 — scope parameterization.** Layernorm's whole-tile statistic, and the one
demand #6 deliberately did not meet. A tile owned by four warps needs the warps
to agree.

> The storage half of the note below is now answered. `SharedVec` is a type a
> block-reduction signature can name, and its scalar access is deliberately per
> *element* rather than per packed word — warp `w` writing index `w` must not
> read-modify-write a neighbour's value, which is exactly the case a word-wide
> `Element::pack` could not serve. What #3 still owes is the fold itself, and
> what it owes *under* #3 is **#2**: `Element` is implemented only for `Bf16`,
> so `SharedVec<F32, 4>` — the shape a partial actually has — does not exist
> yet. The type is generic over `E: Element` and needs no change when it does.

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
>
> **Settled by #13**: the second one. A `SharedVec` is a type the signature can
> name, and it is still the *kernel's* allocation — `from_raw` takes a base
> pointer like every shared type here — so nothing about the shared plan moved
> into the library.

**#8 — global layout.** ~~Not reachable from a kernel at all~~ — **closed**, and
it is why this crate now has a launcher at all. `kittens::global::GlobalLayout<E,
RANK>` describes a buffer by extents and byte strides at any rank, packed or with
a leading dimension, and `tensor_map::<Tile>()` reads the box, the swizzle and
the data type off the `SharedTile` the map is paired with — the agreement
`encode_bf16_panels` had by being one hardcoded builder, now stated as a type.
`gemm::check` builds both operands' maps from it and differs only in extents.

What #8 does *not* close: `GlobalLayout` is bf16-only, because `Element` is
implemented only for `Bf16` (#2 owns that), so an fp32 buffer still cannot be
described. Two of the three things that entry used to say have since gone:
`TileBox` has a second impl (#13), so a layout has something to hand a
non-tile box to, and rank 1 is no longer untried — the `shared vector round
trip` device case builds one and moves a vector through it. What remains is
only the element type, which is #2 and is ~15 lines.

**#12 — MMA coverage.** Not demanded. `mma_abt`, `mma_ab` and `mma_walk_cg2`
covered every multiply in all four kernels. Worth recording as a negative
result: the MMA layer is the part of this library that is finished.

---

### Came from writing kernels, not from ThunderKittens' feature list

The most valuable output here. Ordered by how badly it hurts. Items 1–6 were
filed as **#21**, **#22**, **#25**, **#23** and **#24** (which covers both
cluster-scope entries); **1, 2, 3, 4 and 5 have since shipped**. The numbers are noted
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
exists at every width the library can describe. Item 2 closed the last of it:
`load_tile`/`store_tile` move a whole band, so nothing about the shared ↔
register path is a loop the kernel writes any more.

#### 2. ~~The movers are per-`[16, 16]`-block, so every kernel writes the same loop~~ — **closed by #22**

`TmemTile::fragment_tile` returned a `[16, 16]` `Fragment` and
`ldst::store_fragment` took one, so a kernel that wanted a `[32, 128]` band
wrote a four-deep block-composition loop to assemble it — `gemm.rs` did, flash
would have, `softmax.rs` wrote it *twice*, once per direction, and the device
harness kept its own copy in the test crate. The composition is a property of
the layout, and `reg.rs`'s own test
(`fragment_blocks_tile_the_bigger_shapes`) stated it as an invariant that no
function implemented.

All four directions landed together: `TmemTile::tile(row, column)` and
`TmemTile::store_tile(row, column, tile)` over TMEM, `ldst::load_tile(chunks,
row, column, lane)` and `ldst::store_tile(chunks, row, column, lane, tile)`
over a swizzled shared tile, each composing the single-block mover it is built
on out of the same two placement helpers — so a band cannot mean one thing in
TMEM and another in shared memory. The harness' `drain_band`/`stage_band` are
gone, replaced by the library calls, which is where the TMEM side's device
coverage comes from; the shared side got a `band round trip` case of its own,
and `softmax` runs on both directions of it.

The composition is free where it was measured to be dangerous (#5's `row_map`
cliff at 128 columns): every TMEM drain kernel in `device-tests` reports the
same register count as its hand-written loop, and the shared-side band is
*cheaper* composed — 36 registers against 52 for the loop `softmax.rs` used to
write (`modal run modal_app.py::regcount`).

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

#### 4. ~~The `RegTile` shape set is closed by the library~~ — **closed by #23**

`BaseLdtm` implemented `FragmentLayout` for `(16,16)`, `(32,32)` and `(32,128)`
and nothing else, because each shape was a line of `base_ldtm_shapes!` *inside*
`src/reg.rs`. Flash wants a `[32, 64]` score band and could not add one.

The issue's cheap option — export the macro — is **not available**, and the
orphan rules are what decide it: `impl FragmentLayout<32, 64> for BaseLdtm`
from another crate is E0117 whatever macro writes it, since `FragmentLayout`
takes only *const* parameters and so no local type can ever appear in the impl
header. Confirmed by compiling it, not by reading the reference.

What landed is the issue's third option, in its trait-projection form.
`RowLayout::Slots<T>` is generic in the element, so a tile's storage is a row
array *of* a column array and `FragmentLayout` is a **blanket impl** over
`RowLayout<M> + ColLayout<N>` — no `(M, N)` needs an impl, no array length is
computed from `M` and `N`, and `generic_const_exprs` stays out of it. The shape
set is the product of the two extent lists, which now run to every multiple of
16 up to 512: 1024 shapes out of 64 impls, and every shape that fits in a
thread's registers at all, since `M * N / 32` is already 256 at the corner.

The blanket impl is also what makes the trait genuinely open: a *layout*
defined out of tree gets `FragmentLayout` for free the moment it has both
halves, which is the one thing orphan rules do allow.

#### 5. ~~No cluster-scope TMEM allocation~~ (#24) — **closed by #46**

`tmem::alloc_block` is `tcgen05.alloc.cta_group::1`. A `cg2` accumulator is one
allocation spanning the CTA pair, and this section used to say that only the
leader may issue it, with the peer reading the leader's staging word over
DSMEM. That was the bug, not the fact: all three `cta_group::2` allocator
instructions want *one full warp in each peer CTA*, the collective writes an
address into each CTA's own staging word, and issuing them from rank 0 alone
hung the kernel's second launch (#40).

Shipped as `tmem::alloc_cluster` / `tmem::dealloc_cluster` — `gemm.rs`'s own
fixed body, moved rather than rewritten, since it is the version that ran on
silicon.

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

Related, and cheaper: #15's `lcsf` was filed as depending on #9, which has now
landed with the wait it needs — `tma_store_wait_read` releases the shared buffer
as soon as the engine has read it, without blocking on global visibility, which
is exactly what lets an epilogue overlap the next tile's first K loads. The
GEMM shows the second reason to want it: with the store folded into `finish`, a
persistent GEMM cannot overlap those today.

#### 8. Multicast has no geometry to live in

`tma_load_2d_multicast_cg2`, `commit_multicast_cg2` and `mma_walk_cg2` all
hardcode the 2-CTA pair (GAPS §2.4). Multicast starts paying, as *replication*,
at cluster ≥ 4 (2×2: `A` broadcast along the N axis, `B` along the M axis),
which the `_cg2` suffix rules out. Generalizing the cluster mask is filed
nowhere.

This item said the GEMM "cannot use the load", since under a 2-CTA UMMA both
operands are already split and nothing is replicated. Running the kernel
corrected that, and the correction is the more useful fact: the GEMM uses
`tma_load_2d_multicast_cg2` with a **one-CTA mask**, replicating nothing, purely
because the multicast form's barrier operand lives in `.shared::cluster` and the
plain form's does not. A pair-wide producer handoff is impossible without it.
The `multicast` in the name describes the delivery; what a cluster kernel
actually needs from it is the address space. Filed as a correction on #24.

#### 9. Smaller things, each one line of API

- **`expect_tx` byte accounting is hand-summed.** Every producer writes
  `(ATile::BYTES + BTile::BYTES) as u32` and has to keep it in step with the
  loads it issued. A tile knows its own size; `Semaphore::expect_tiles` or a
  charge returned by `tma_load` would make the two impossible to disagree.
- **Coordinate-dependent ops need `lane` passed in.** Settled by #27:
  implicit for ops that execute, explicit for pure coordinate queries. The
  masks (#7) take it explicitly, on that rule — `device-tests` and `reg.rs`
  build their expectations by calling the map across all 32 lanes on the
  **host**, where `warp::lane_id()` does not exist, and that is what makes the
  mask's own test non-vacuous rather than a restatement of the kernel.

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
