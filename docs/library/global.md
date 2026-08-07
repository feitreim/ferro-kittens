# `global` — design notes and measurements

Two ways out of a kernel and into a buffer: a TMA descriptor built on the host,
and ordinary loads and stores issued by the threads themselves. This file is why
each exists, why each is shaped the way it is, and what was measured.

## The descriptor path

### Only the rank is a type parameter

`GlobalLayout` carries extents and byte strides as runtime values and takes only
`RANK` as a const generic. Every buffer a kernel here maps has a runtime shape —
a GEMM's `K`, a batch count — while the rank decides the *arity* of the arrays
the driver call takes, which is exactly what a const generic is for.

The rejected alternative is ThunderKittens' `gl`: a per-dimension
compile-time/runtime marker. It would buy constant folding in address arithmetic
that happens entirely inside the TMA engine, and it would need array lengths
computed from the markers — `generic_const_exprs`, which this crate avoids. The
same dodge is why `GlobalLayout` stores a full `[u64; RANK]` stride array and
hands the driver `strides[1..]`, rather than a `[u64; RANK - 1]`.

### The box comes from the tile, not from the call site

Data type, box shape and swizzle mode all come from the `TileBox` impl of the
shared-memory destination, so a descriptor cannot disagree with the tile it
feeds unless the layout is paired with a different tile type than the kernel
loads. A layout paired with the wrong element does not typecheck at all.

The two `TileBox` impls answer the same question differently, which is the point
of the trait rather than a wart:

- `SharedTile` — the box is one *subtile*, `SUBTILE_COLS` wide, and swizzled,
  because an MMA operand must be. `SharedTile::tma_load` issues one
  `cp.async.bulk.tensor` per stacked subtile, lifting the leading coordinate by
  `SUBTILE_COLS` each time. A wider tile therefore does not widen the box.
- `SharedVec` — the box is the whole vector, one row tall and unswizzled,
  delivered by a single instruction. A swizzle atom is the wrong unit for a
  one-row run of elements: 128 bf16 wants to be one box, not two boxes of 64.
  That is a layout decision, not a missing `Swizzle` mode.

A rank-1 layout has no dimension for a tile's rows, so it admits a `TileBox`
only when `BOX_ROWS == 1` — which is exactly the vector case.

### Why `descriptor_shape` is split out

`CUtensorMap` is 128 opaque bytes and nothing can be read back off it. The
`globalDim`/`boxDim` computation is the whole of the shape agreement and the
only part of a descriptor a host test can inspect, so it lives in a pure
function of its own.

### Driver requirements, checked before the call

`cuTensorMapEncodeTiled` answers a bare `CUDA_ERROR_INVALID_VALUE` for every
violation, so the rules are checked here and the error names the field. The
16-byte innermost-width rule is the interesting one: a swizzled box satisfies it
for free, because its width *is* an atom, and an unswizzled vector box is the
one shape that can violate it — 8 bf16 is 16 bytes and legal, 4 bf16 is 8 bytes
and is not.

### `PanelMap`

An alias for `TensorMap` under the name it had before layouts were a type.
`encode_bf16_panels` is its constructor, and its `rows % R == 0` requirement is
that staging layout's own convention rather than a rule of `GlobalLayout`: the
kernel steps its row coordinate by `R`, so a ragged tail would arrive silently
zero-filled instead of as a short tile. `GlobalLayout` itself leaves edge boxes
to the TMA engine's out-of-bounds fill.

## The direct path

### Why it exists at all

An epilogue that must reach global memory **as fp32** has no descriptor route.
`TensorMapElement` is implemented for the 16-bit elements only and `stmatrix` is
b16, so the staging half of the round trip cannot be spelled: the choice through
shared memory is to round or to widen. And a small irregular operand — a bias
row, a lookup table, a ragged tail — is not worth a descriptor built on the host
in the first place.

An epilogue whose output *is* bf16 has both routes open, and the direct one is
still shorter: the cursor's element is where the rounding happens, and nothing
is staged.

There is no engine here and nothing asynchronous. A thread computes the address
of each value it owns from the fragment layout's own
`(lane, slot, value) -> (row, column)` map and stores it — which is why the
movers are the only things in the crate generic over `FragmentLayout` rather
than pinned to `BaseLdtm`. `ldmatrix`, `stmatrix` and LDTM each fix a lane map
in hardware; a plain `st.global` fixes nothing.

### The cursor's element used to be fp32 and nothing else

The argument was that a register tile *is* fp32 and this path exists to move one
without rounding. A training GEMM's `C` is bf16, which asks the other direction:
there the rounding is the point rather than the loss, and it belongs in the store
instruction rather than in a round trip through a shared tile. The parameter
costs exactly what the doc that predicted it said — a bound, and `E::read` /
`E::write` where a raw `f32` used to be.

### Two values at a time

A layout that gives a thread *adjacent* columns has told the mover something:
those two accesses are one. `ColLayout::CONTIGUOUS_VALUES` is that claim, and
the movers spend it through `Element::write_pair`.

Measured shape of the win, on a `[128, 128]` accumulator band under `BaseLdtm`:

| | memory instructions | 32-byte sectors touched |
|---|---|---|
| scalar | 512 per lane | 8 per warp |
| paired | 256 per lane | 8 per warp |

Half the instructions, and — because the two words of a pair sat in the same
32-byte sector either way — the same transactions carrying twice the useful
bytes.

It is stated on the layout rather than read off `BaseLdtm`'s arithmetic because
the shape set is open: a layout written later gets the default `1` and scalar
accesses, which is wrong about nothing, until it claims better.

What the mover adds is the half a layout cannot know. A *vector* access must be
aligned to its own width, and a cursor's base, stride and column origin are
runtime numbers. So the pairing is tested once per call
(`GlobalRows::runs_aligned`) rather than promised in a safety contract that an
odd leading dimension would quietly break. Under `BaseLdtm` the four values of a
16-column block sit at column offsets `{0, 1, 8, 9}` — two adjacent pairs, hence
`CONTIGUOUS_VALUES = 2`; `2` divides the claimed run rather than equalling it, so
a layout claiming four contains two aligned pairs and one claiming three
contains none.

The choice is made once per call, not once per value: the two spellings are
separate unrolled loops and the alignment test picks between them, so neither
path carries a branch.

The *element* decides only the instruction — fp32 needs `st.global.v2.f32` to
carry two values, bf16 needs a plain 4-byte word because two bf16 already are
one. Same instruction count, half the bytes.

### The alignment test does not move with the element

Not the obvious result. A narrower element halves the byte offset a column costs
*and* halves the width a pair must be aligned to, so both sides scale together
and what survives is the same question at either element: an even stride and an
even column origin, off a base aligned for the element it is. A bf16 `C` cannot
buy the pairing anywhere an fp32 `C` was denied it, and cannot lose it anywhere
an fp32 `C` had it. `the_pairing_test_is_the_same_at_either_element` pins this.

### `store_rows` is not warp-collective

The one way it differs from every other mover in the crate. `stmatrix` is one
instruction fed by 16 lanes' addresses; this is 32 threads each storing their own
values. A lane that does not call it leaves its own values unwritten rather than
making an instruction ill-formed.

### When a row-wise walk over this path loses, and what does not fix it

The idiom this path invites is a **tile walk**: give a warp `load_rows` at
`[16, CHUNK]`, carry a `RegVec` statistic across chunks, `store_rows` the
result. It is short, it needs no shared memory and no barrier, and it is the
right answer often enough that it is worth writing down where it is not.

**Both poles are measured, and they are not the same kernel.**

- **Frame- or depot-bound → the walk wins, by a lot.** `groupnorm_tile` held its
  whole `[32, 128]` band as a value and went **594 → 5996 GB/s**, a factor of
  10.1, when the band was streamed a `CHUNK` at a time — 168 registers and a
  1536-byte frame down to 48 and none. `docs/kernels/layernorm.md` carries that
  table. Note where that walk reads from: a **TMA-staged shared tile**, through
  `ldst::load_tile`, not this path.
- **Row-wise work already coalesced over global → the walk loses, and the loss
  is the fragment map.** oxide-train PR #124 rewrote a block-per-row bf16 RMS
  norm onto this path and measured **+7–14% worse across six runs, one sign**;
  the kernel was reverted. `experiments/src/norm_occupancy.rs` is that
  comparison rebuilt here, five arms on one clock, every arm reading the row
  twice and writing it once so that all five issue identical bytes:

| 24576 × 3072 | threads | live f32 | blocks | waves | regs | blocks/SM | threads/SM | GB/s | vs row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block-per-row | 256 | 2 | 24576 | 166.1 | 23 | 8 | **2048** | 6325 | 1.000 |
| walk `CHUNK` 64, 4 warps | 128 | 32 | 384 | 2.59 | 48 | 10 | 1280 | 3031 | 2.087 |
| walk `CHUNK` 64, 8 warps | 256 | 32 | 192 | 1.30 | 48 | 5 | 1280 | 2504 | 2.526 |
| walk `CHUNK` 16, 4 warps | 128 | 8 | 384 | 2.59 | 32 | 16 | **2048** | 2326 | 2.719 |
| walk `CHUNK` 16, 8 warps | 256 | 8 | 192 | 1.30 | 32 | 8 | **2048** | 2025 | 3.123 |

**The occupancy lever was pulled and it made things worse.** A narrower chunk is
fewer fp32 live in a lane, which is fewer registers, which is more resident
threads: the `CHUNK` 16 arms reach 2048 threads an SM — the block-per-row arm's
own number, at 32 registers against 23 — and they are the *slowest* two rows of
the table. Every shape orders the four walk arms identically (`c64 w4` best,
`c16 w8` worst) at 1024, 3072 and 8192 columns, so what orders this table is
**chunk width**, and occupancy orders it backwards.

That is the same claim the next section makes about storing, now with a clock on
the load side: under `BaseLdtm` a warp's paired access names **8 rows × 16 bytes**
— eight 32-byte sectors, half used — where the block-per-row arm's 32 lanes name
one contiguous 128-byte line. Identical bytes, not identical transactions. A
wider `CHUNK` buys some of that back (at 64 columns a chunk's row is a full
128-byte line, walked by four lanes) which is exactly the direction the table
runs in, and a narrower one spends it for registers nothing was short of.

The 16-row floor costs a second thing, on the launch rather than in the lane: a
CTA owns `16 · WARPS` rows however few rows the problem has. At 6144 rows the
8-warp arms launch **48 CTAs on a 148-SM device** — 0.32 of a wave — and measure
5.7× and 11.0× the reference. That row of the sweep is measuring the launch
shape, and it is a property of the idiom rather than of the harness.

So the decision rule, before rewriting a row-wise kernel onto this path:

1. **Compute the achieved GB/s of what you have.** A kernel near HBM is not
   waiting on registers and a mover rewrite cannot move traffic.
2. **Count what a warp's fragment map names per access**, against what the
   kernel it replaces names. Two adjacent bf16 out of a 16-row block is a
   32-byte sector; a block-per-row loop is a 128-byte line. That ratio is the
   thing the rewrite is spending.
3. **Occupancy is not the answer to (2), and it is available**, so this is a
   measurement rather than a wish: `CHUNK` and the warp count are both const
   parameters of the walk, the narrow-per-lane high-occupancy form needs no API
   this crate does not have, and it is the arm that lost worst.

What is *not* expressible, for the reader who wants the remaining direction: a
band split across warps by **column** rather than by row, so several warps share
one 16-row band at a narrower per-lane width. Its statistic is a `RegVec<16>`
spanning warps, and `sync::block_reduce` folds one **scalar** per warp — there is
no vector form. That is the one structural coupling in the way, and the table
above is the reason nobody should build it before measuring (2) first.

## Out of a staged tile instead

`store_shared_rows` writes the same kind of destination `store_rows` does, out of
a `SharedTile` rather than out of registers.

### Why route a band through shared memory at all

A fragment layout is a bad shape to store *from*. Under `BaseLdtm` a thread owns
columns `{0, 1, 8, 9}` of each 16-column block, so the widest thing `store_rows`
can issue is a pair and the addresses a warp presents are scattered across the
row. Going through shared memory first — `stmatrix` into a swizzled tile, then
this — spends one extra pass over the data to buy back both halves of that: the
accesses widen to 16 bytes, and the addresses of a warp become one contiguous
run. A warp of 32 lanes issuing `st.global.v4.b32` on consecutive chunks writes
512 contiguous bytes, where the same band stored straight out of a fragment
layout is 32 scattered pairs.

The section above is what that costs when nobody pays it: the same scatter on
the *load* side, timed, ordering a five-arm table by how many bytes a lane names
per access.

NVIDIA's own reference kernel takes exactly this route (`gemm_sol_final` in
cuda-oxide at the pinned revision `20a5616`).

### Against the TMA engine

`StoreRing` does the shared→global half of an epilogue with the engine instead.
Neither is strictly better. The engine costs a host-built descriptor and the
fence and group-wait discipline that comes with it, and it lands a whole box;
this costs the CTA's own issue slots, and lands whatever rectangle the
arithmetic names.

### 16 bytes is the chunk, and the swizzle is free

SWIZZLE_128B permutes 16-byte chunks and never the bytes inside one, and 16 bytes
is also the widest access PTX has (`st.global.v4.b32`). So a chunk is
simultaneously the largest run contiguous in shared *and* in global memory, and
the largest one an instruction could have carried anyway.

The tile is swizzled and the destination is not, so one of the two sides has to
be walked out of order. The mover walks the *global* side contiguously and asks
the tile's own cursor where each chunk went. The rejected alternative was laying
the band out linearly in an unswizzled staging tile so the read-back is trivial.

Two arguments against it, both about addresses, **neither of them measured**:

1. The swizzle costs the read-back nothing. SWIZZLE_128B XORs the chunk index
   with the row's position in the swizzle period, so it is a permutation of the
   eight chunks within one 128-byte row and never moves a chunk to another row.
   A 128-byte row is exactly the 32 banks of shared memory once over; eight lanes
   reading the eight chunks of one row therefore touch every bank exactly once,
   permuted or not.
2. An unswizzled staging tile moves the disorder onto the `stmatrix` that fills
   it, where a power-of-two row pitch puts a fragment's eight rows on the same
   banks — the conflict the swizzle exists to prevent. It would also need a
   `Swizzle` impl that does not exist; see `SharedVec`'s docs file for its
   precondition (`ATOM_BYTES` has to stop meaning "TMA box width" first).

### The access ladder

The same question `store_rows` answers at two widths, asked at four: 16, 8, 4,
then the element itself. Each rung is `runs_aligned` at the run of elements that
width is worth, so the three terms it weighs are the three `store_rows` weighs
and the two movers cannot drift apart.

The price is four instantiations of the loop body where `store_rows` has two. The
2-byte rung is guarded on `E::BYTES == 2` and is not emitted at fp32 at all: a
cursor is aligned for its own element by `GlobalRows::from_raw`'s contract, so a
4-byte access is legal at every column a 4-byte element names and the bottom rung
is unreachable there. A narrower access never crosses a chunk, so the shared side
needs no second decision — 16 is `CHUNK_BYTES`, and 8, 4 and 2 divide it.

The two vector widths are inline PTX for the reason `F32::write_pair` is:
widening adjacent accesses into one is a transformation the compiler may only
make when the address is provably aligned, and an address built from a runtime
leading dimension never is. The caller that has actually checked is the only one
in a position to spell it. The two narrow widths need no asking — a 4- or 2-byte
move is one instruction however it is written. The shared-side load is written as
a load into registers rather than fused with the store, so the two are separate
instructions the scheduler can put other work between; a single opaque snippet
spanning both would forbid that.

### The work split is by chunk, not by row

A row split needs `R` to divide by the warps *and* the warp's rows to divide by
32. A chunk split needs only that `THREADS` divide the tile's chunks, which a
const assert checks, and it puts consecutive threads on consecutive chunks of one
destination row — which is the entire point of the round trip.

The arrangement the reference kernel uses is four epilogue warps of a CTA passing
`(threadIdx.x, THREADS = 128)`; a warp-scope caller passes `(lane, 32)`.

### There is no proxy fence, and its absence is not a bug

`StoreRing` needs `fence.proxy.async.shared::cta` because the TMA engine reads
shared memory through the *async* proxy while `stmatrix` wrote it through the
generic one, and nothing but the fence orders two proxies. Both ends of
`store_shared_rows` are generic-proxy — `ld.shared` and `st.global` are ordinary
instructions — so the only thing the writes and the reads need between them is a
`bar.sync`, and a warp reading back its own `stmatrix` (which is `.sync.aligned`)
needs not even that. The barrier is the caller's because only the caller knows
which of the two it is in.

The exit side is shorter than the engine's too: a plain `st.global` is visible to
the host and to the next kernel when the kernel ends, with no counterpart to
`StoreRing::drain` owed to anybody.

Nothing in `store_shared_rows` rounds. `store_rows` narrows fp32 registers in
`Element::write`, and that is where a bf16 `C` is made; a tile is already `E`, so
this moves its bytes and the element only sets how many of them a column is
worth.
