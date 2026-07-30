# `shared.rs` — design notes and measurements

`shared.rs` holds the shared-memory shapes: `SharedTile` and its ring,
`SharedVec`, `SharedCell` and its ring, the `Element`/`MmaElement`/`Swizzle`
markers, and the swizzled cursor `SwizzledChunks`. The source states the layout
and the contracts; this file states why the layout is that one, which
alternatives were rejected, and where the properties are pinned.

## The subtile layout

The scheme is the one the flash-attention and GEMM kernels validated on B200. A
SWIZZLE_128B tile is stored as stacked 128-byte-row *subtiles* — 64 bf16 columns
each — so that the swizzle phase inside each subtile equals the row index. That
coincidence is what a 64-wide panel gives for free; storing wider tiles as
stacked subtiles is how it is kept by construction at every width. A `[R, 128]`
bf16 operand is two stacked `[R, 64]` subtiles a subtile-stride apart; `[R, 64]`
operands (P/dS) are a single subtile.

Widths that are not a whole number of subtiles are a **compile error**
(`WIDTH_OK`), not a differently-swizzled layout. Restricting honestly beats
pretending generality: there is no correct thing to do with a 96-column bf16
tile under a 128-byte atom, and admitting one would mean a second layout rule
nobody validated.

### The phase is absolute, not per-subtile

SWIZZLE_128B XORs *physical* address bits `[9:7]` into the 16-byte-chunk index.
Two consequences the types keep straight:

- A tile whose base is not 1024-byte aligned starts mid-period, so every manual
  swizzled store must fold in the tile base's own 128-byte row phase
  (`SharedTile::swizzle_phase`).
- Stacked subtiles need no term of their own, because they *are* further rows of
  one flat stack: subtile `i` starts `rows * 128` bytes along, so subtile `i`'s
  row `r` is the tile's 128-byte row `i * rows + r` and takes that row's phase.
  A logical chunk index splits into `(subtile, chunk)` and the two row terms
  simply add. Nothing anywhere assumes a relation between `rows` and the 8-row
  swizzle period.

The second point only *shows up* when the subtile height is not a whole number
of swizzle periods. At every shape the crate ships (R = 64, 128) the "absolute
row" and "own row" readings coincide, and a wrong one would never be noticed —
which is why `a_subtiles_phase_is_its_absolute_row_not_its_own_row` uses a
4-row tile: subtile 1 begins four 128-byte rows down and starts at phase 4, not
at phase 0.

### What the host tests can and cannot see

The chunk-map tests sweep every base phase and check bijectivity onto the tile's
16-byte slots. Note the limit: *any* phase is a permutation of a row's eight
chunks, so a wrong phase stays a bijection. Injectivity catches a wrong subtile
stride; it cannot catch a wrong phase. The phase is pinned by the two explicit
formula tests and, against the TMA engine itself, by the device harness.

## `Element`: why the packing is an associated type

`Element::Unpacked` is an associated type rather than `[f32; PER_WORD]` because
an array length must be a const expression of the parameters, which would need
`generic_const_exprs` — the same dodge `reg::FragmentLayout` takes for its
storage. It also gives store paths a way to state their arity: the `stmatrix`
path moves b16 matrices, so it bounds `E: Element<Unpacked = [f32; 2]>` and a
4-per-word element fails to typecheck there instead of packing silently wrong.
`PER_WORD` is derived from `Unpacked` so the two cannot disagree.

Device code never handles a value of the element type — a tile is bytes in
shared memory and packed words in registers — so the trait describes the packing
rather than a scalar.

### Why there is a scalar half beside the word half

`read`/`write` exist for `SharedVec`, whose consumers index *elements* (one
column's parameter, one warp's partial) where a tile's consumers move whole
32-bit words through `ldmatrix`. `Unpacked` is an opaque `Copy` type with no way
to pick a lane out of it, so the word form cannot serve an element index however
the caller bounds it. `write` takes a byte pointer rather than a word so a
narrow element's neighbours are untouched: two lanes writing adjacent elements
of a vector must not read-modify-write one word and lose each other's value.

### `write_pair`/`read_pair` are global-memory instructions

These pair values *one lane owns*, which is what `reg::ColLayout::CONTIGUOUS_VALUES`
is spent on, and it is the only thing about a fragment mover that depends on the
element. There is deliberately no shared-memory caller: a `SharedVec`'s
neighbouring elements belong to different lanes.

The default is the two scalar writes it replaces, so an element with no wider
spelling is correct without stating one. Both shipped elements do have one, and
they differ in kind:

- Two adjacent bf16 **are** one packed word, so the pair is a plain 4-byte store
  of `pack`: one `cvt.rn.bf16x2.f32` and one `st.global.u32`.
- fp32 needs a vector instruction: `st.global.v2.f32`, moving the pair in eight
  bytes. The instruction count is the same as bf16's and that is the whole of
  what a narrower `C` changes here.

The fp32 pair is written as inline PTX for the reason `ldst::stmatrix_m8n8_x2`
is: the instruction has to be *asked for*. Widening a pair of adjacent stores is
a transformation ptxas may only make when the address is provably aligned, and
an address built from a runtime leading dimension never is — so the caller that
has actually checked (`global::GlobalRows::runs_aligned`) is the only one in a
position to spell it.

### bf16's asymmetry, and why F32's round trip is host-testable

`Bf16::pack` is `cvt_f32x2_bf16x2`, a device intrinsic whose host body is
`unreachable!()`, so its bit layout is only checkable on a GPU. `Bf16::unpack`
is not: bf16 is fp32's leading 16 bits, so widening is a shift and no
instruction — `cuda-device` exposes conversions in the narrowing direction only
and there would be nothing for one to do. That is why the unpack side has an
exhaustive host test and the pack side does not.

`F32`'s two halves are both ordinary bit math, so its round trip is checkable on
the host end to end. It is the identity on the bits, which is exactly the
property a partial staged through shared memory needs.

### Why `F32` is deliberately not an `MmaElement`

tcgen05 reads fp32 operands as tf32, which is a different element with a
different mantissa. Giving `F32` an `MMA_KIND` would let a `[R, C]` fp32 tile be
staged as an operand it is not the bits of.

`F32` is instead the element a *statistic* is held at. It carries no rounding,
which is why `SharedVec<F32, N>` is what `sync::block_reduce` stages its
per-warp partials in: a partial that went through shared memory as bf16 would
come back with eight bits of the sum gone, and the second of two chained
statistics would inherit the error of the first.

## `MmaElement`: why the routing is a method

The element is where the *instruction* is selected, not just the operand format:
`MmaElement::mma` and `mma_cg2` are the whole of `mma`'s routing to silicon.
That is deliberate — it is what makes a new operand type a new impl of this
trait rather than a new MMA layer.

It is a *method* and not `tcgen05_mma_shared::<{ Self::MMA_KIND }, ..>` at the
call site because Rust cannot pass an associated const of a type *parameter* as
a const-generic argument without `generic_const_exprs`. An impl writing
`Self::MMA_KIND` for itself is legal, and is what keeps the const and the
instruction from drifting apart.

The trait is split from `Element` because operand kind is an MMA property and a
tile that only moves bytes never reaches an MMA — the same split, for the same
reason, as `reg::RowLayout` out of `reg::FragmentLayout`.

`MMA_KIND` selects 0 = f16, 1 = tf32, 2 = f8f6f4, 3 = i8; it is a `u32` to match
the const-generic parameter of `tcgen05_mma_shared`/`tcgen05_mma_tensor`
exactly, so an impl's `mma` is a substitution and not a cast. f16 and bf16 share
kind 0 and are told apart only by `ELEMENT_TYPE`, the instruction descriptor's
`atype`/`btype` field. Reading the same bits under the wrong one produces a full
accumulator of wrong numbers, so it belongs to the element rather than to
whoever assembles the descriptor.

`COLLECTOR_A::discard` is the selector every walk here issues under, and is
tcgen05's own default: nothing in this crate reuses an `A` operand across
instructions, so no chain has a collector buffer to keep alive.

## The two TMA completion mechanisms

A load's destination is shared memory, where an mbarrier can live and count the
arriving bytes. A store's destination is global memory, where none can. So the
two directions complete through wholly different machinery, and it is not
cosmetic:

- **Loads** land on a `Semaphore` and hand back a `TransactionBytes` charge to
  be summed into that barrier's `expect_tx`.
- **Stores** complete through `tma_store_commit` plus one of the two waits — the
  issuing thread's outstanding stores committed as one group, waited on by
  counting groups, with no barrier to arrive on and no byte accounting anywhere.

Groups are per *thread* and age in issue order: everything issued since the last
commit becomes the youngest group, and every earlier group's index goes up by
one. A tile's store is one instruction per stacked subtile, so committing per
tile is what makes "wait for that tile" expressible at all — the waits count
groups and cannot name an instruction. The group count is a const parameter
because the instruction's operand is an immediate; a runtime depth is not a
thing the hardware can be asked for.

`tma_store_wait_read` is the cheaper wait and the one a pipelined epilogue is
written around: store stage `i`, recycle its buffer for stage `i + 1`, and let
the global writes retire behind the next tile's work. Blocking on global
visibility to recycle a buffer serializes the overlap the pipeline exists for.
It is never the last wait in a kernel: a result that has only been read out of
shared memory has not arrived anywhere.

### The own-bit multicast, and the deadlock it avoids

`tma_load_2d_arriving_at` issues a cta_group::2 multicast whose mask is the
calling CTA's own bit and nothing else. That looks like a contradiction and is
not — nothing is replicated: one bit, one destination.

What the multicast form supplies that the plain one cannot is the
`.shared::cluster` barrier operand. A plain `cp.async.bulk.tensor` completes on
a barrier in the *issuing* CTA's own shared memory, so a peer staging its half
of a cluster operand has no way to say "my bytes, the leader's barrier" — and
the leader waits forever for a count it charged and nobody paid. That deadlock
is the whole reason the method exists.

`tma_load_2d_multicast_cg2` is the genuine replication form, which is why its
mask is the caller's. The charge it hands back is one tile — what a single
destination receives, which is the whole of it for the own-bit mask and the only
mask any kernel here uses. A caller replicating into several CTAs owns the
question of whether the one barrier it named sees that once or once per
destination; that has not been answered against hardware.

### Stacking boxes taller than the map's

`tma_load_at::<BOX_ROWS>` lands a box at subtile row `dst_row` instead of the
top — how a tile taller than the map's box gets built out of several global row
ranges. The backward kernels stack two adjacent 64-row tiles into one 128-row
operand at `dst_row = 0` then `dst_row = 64`.

The stacking is layout-free precisely because a box height is a whole number of
swizzle periods (8 rows of 128 bytes): landing a box at row 64 of a subtile
reproduces exactly the swizzle rows 64.. of one tall tile would have had.

`BOX_ROWS` is the *map's* box height and so cannot be read off the tile — a
128-row operand built this way is paired with a 64-row map — which is why it is
a parameter rather than `R`. It is a *const* parameter because the charge handed
back is derived from it: the call brings in `SUBTILES` boxes of `BOX_ROWS` rows,
not a whole `BYTES`. That used to be a sentence asking the caller to do the
multiplication.

### The charge is derived, not written down

`SharedTile::CHARGE` counts what the load loop actually issues — one box per
subtile, `R` rows of one swizzle atom each — and `BYTES` counts the tile. They
are the same number for every legal shape, and that identity is the whole reason
a derived charge can replace a hand-written one. A width that stopped being a
whole number of subtiles would break it, and `WIDTH_OK` rejects those first.

Both constants are crate-private, because a kernel able to name a charge without
issuing the load is back to writing the number down.
`every_producers_derived_charge_is_the_sum_it_used_to_write_down` pins one case
per `expect_tx` in `examples/`, with both sides spelled out — a derivation that
agreed with a hand-sum only because both were computed from the same expression
would be testing nothing:

| producer | derived charge | bytes |
| --- | --- | --- |
| gemm, one stage | `GemmA::CHARGE + GemmB::CHARGE` | 24,576 |
| gemm, across 2 ranks | `.across_ranks(2)` | 49,152 |
| flash_forward, Q | `Q::CHARGE` | 32,768 |
| flash_forward, per key block | `KV::CHARGE + KV::CHARGE` | 32,768 |
| softmax / groupnorm's second kernel | `Rows::CHARGE` | 32,768 |
| layernorm | `Rows::CHARGE + 2 * Gamma::CHARGE` | 33,280 |

layernorm's is the sum that mixes a tile with two vectors and so was the easiest
to get wrong: a missed `2 *` under-charges by 256 bytes and the tile is read
while beta is still in flight.

## `publish_to_async_proxy`: the fence half of every fence contract

Shared memory is written two ways and read two ways. `stmatrix`, a plain store,
`ldst::store_tile`, `ldst::store_fragment` and an mbarrier's `init` all go
through the *generic* proxy; the TMA engine and a tcgen05 MMA read through the
*async* one. The two are not coherent on their own and no barrier makes them so:
`sync_threads` orders threads, not proxies, so a `bar.sync` between the write and
the engine's read is a fence that was never issued.

The fence orders the writes of **the thread that executes it**, which is why it
is not a collective and why every thread that wrote a row has to call it. A
barrier afterwards is what carries that ordering to whichever thread issues the
store or the MMA — the pairing `epilogue::StoreRing` performs internally and
three kernels write out as `publish_to_async_proxy(); sync_threads();`.

Both directions of "which proxy" have to be true for the fence to be owed. The
two cases where it is not are worth stating because they look alike:

- A tile that arrived by `tma_load` and leaves by `tma_store` is the async proxy
  on both sides, ordered by the load's own barrier.
- A staged value written and read by ordinary threads — `sync::block_reduce`'s
  scratch — is the generic proxy on both sides, ordered by its barriers.

The function was private until the crate noticed it stated the obligation in six
safety contracts and exported nothing that discharged it, so three kernels
imported `cuda_device::barrier` for the one instruction.

## `OperandWalk`: why the layout is erased to values

An `OperandWalk` carries the chunk step and descriptor leading offset that
distinguish a K-major from an MN-major operand as runtime data instead of a
type. This exists for kernels that select the layout at runtime — gemm's
`transposed` launch parameter.

A value-level walk keeps the issue loop *single*: one select feeding one chain,
matching the hand-written kernel's schedule. A typed two-arm branch would
duplicate the MMA chain per layout and hand ptxas a different instruction stream
to allocate against.

The transpose bit rides on the walk rather than beside it, because "the MMA must
read this operand transposed" is the same fact as its chunk step and leading
offset — an MN-major walk supplies K along rows. Carrying it here means the
instruction descriptor is built from the walk it is issued with.

`k_walk` is restricted to tiles whose K spans exactly one swizzle atom per row
(gemm's `[128, 64]` stage), because a linear step cannot cross stacked subtiles.
`mn_walk`'s leading offset is `SUBTILE_BYTES` — a jump to the stacked subtile
holding MN columns 64..128, not a step along the row.

Both reproduce gemm's hand-written `consume_stage` descriptors exactly:
`build_smem_descriptor(smem + offset, leading, 1024, 2)` with
`(offset, leading) = (chunk * 32, 16)` for K-major and
`(chunk * 16 * 128, SUBTILE_BYTES = 8192)` for MN-major.

## `SharedVec`: why it does not go through `Swizzle`

Swizzling exists to keep a *2-D* tile's 16-byte chunks off the same banks when
successive rows are read together. The XOR is over the row index, and a vector
has one row for the phase to come from. So the layout question is not "which
mode" but "an atom is the wrong unit here at all", and the two jobs
`Swizzle::ATOM_BYTES` does make that concrete. It is the swizzle period **and**
the width of a TMA box, and an unswizzled vector wants neither:

- Its box is `N` wide, a number no mode marker can carry.
- Any atom small enough to divide a short vector splits it into stacked subtiles
  the engine would then fetch one instruction each.
- A 128-byte atom is worse still: the narrowest bf16 tile it admits is 64
  columns, which is the padding per row this type exists to avoid.

`a_vector_costs_no_padding_where_a_tile_would` is that last point measured: a
32-column bf16 tile is not even expressible (`WIDTH_OK` wants a whole 64-column
subtile), and `SharedTile<Bf16, 1, 64, Swizzle128B>` spends 128 bytes on its
single row where `SharedVec<Bf16, 32>` spends 64.

Adding a `SwizzleNone` mode is a *different* job, left open. It owns **tiles**
under narrower and absent swizzles: an unswizzled `[R, C]` staging tile does
need a `Swizzle` impl, and it needs `ATOM_BYTES` to stop meaning "box width"
first. A vector needs none of that, and borrowing the mode marker to get one
would have decided that question by accident.

The TMA still delivers a vector: the box is `[N]` (or `[N, 1]` at higher rank)
with `CU_TENSOR_MAP_SWIZZLE_NONE`, which the engine writes contiguously — the
same bytes `SharedVec::at` addresses. That agreement is what the
`global::TileBox` impl states.

### Where the box rules live, and why they moved off `from_raw`

Two rules bind an unswizzled box:

- `BOX_OK`: its innermost dimension is measured in bytes and must be a whole
  number of 16-byte lines. There is no swizzle atom to round it up to, so the
  length itself has to be legal.
- `LENGTH_OK`: a box dimension is a `u32` field capped at 256 by the descriptor
  format, and a vector is a single box by construction.

Both used to be forced by `from_raw`, which read as "a `SharedVec` is a thing
the TMA can deliver" — a stronger claim than either assert makes. Both are
statements about a *descriptor's box*, and `sync::block_reduce`'s scratch is the
one use of this type that never meets a descriptor. A four-warp block's partials
are 16 bytes, a two-warp block's are 8, and only the first could be constructed
under the old placement — an arbitrary restriction on how many warps a block
reduction supports.

So the rules moved to the four transfer calls they are about, where they still
fire at codegen for every caller they genuinely bind. A vector only ever written
by `set` and read by `get` pays nothing for a box it does not have.

The host side is the other half of this and is unchanged: building a tensor map
runs `GlobalLayout::check_driver_requirements`, which rejects the same shapes at
map-construction time with a message naming the field and the byte count.

`SharedVec<F32, 4>` — a block reduction's four per-warp partials — is 16 bytes
*exactly*, which is the smallest box `BOX_OK` admits. It passes by zero margin,
which is worth an assertion rather than luck.

One direction is not expressible as a host test: `tma_store` on an illegal shape
is a *compile* error, not a failing assertion, so no test can call it. Checked by
hand against `SharedVec<Bf16, 4>` (8 bytes — legal as a handle, illegal as a
box), and the error names the instantiation:

    evaluation of `SharedVec::<Bf16, 4>::BOX_OK` failed
    ... while instantiating `SharedVec::<Bf16, 4>::tma_store`

Making that a permanent case needs `trybuild`, which the crate does not depend
on.

## `SharedCellRing`: the depth is the whole reason the type exists

A mailbox behind a parity wait is only sound while the producer leads the
consumer by less than the ring's depth. A shallower ring either overwrites a
message before it is read, or — since parity has period two — lands the
consumer's wait on a phase the producer has already passed and will never flip
again.

One cell and one barrier is that ring at `N = 1`: it tolerates a lead of exactly
one and reports nothing when the lead is two. So a kernel handing items across
warps has to *derive* the depth from what its own back-pressure guarantees
rather than assume one, and the payload ring and the semaphore ring have to be
built at the same `N` for the parity to name the cell.
`sync::depth_needed` is the derivation for a persistent GEMM's chain.
