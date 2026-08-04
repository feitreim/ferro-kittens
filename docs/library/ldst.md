# `ldst.rs` — warp-scope register↔shared movers

## One address derivation, two directions

`ldmatrix` and `stmatrix` share an address convention: the 16 addresses come
from lanes 0..15 while the data is spread over all 32. So the two directions are
one derivation — `fragment_address` — with the data flowing opposite ways, and
they cannot drift apart. A second derivation for the store side is exactly the
thing that could disagree with the load side, and there is not one.

That includes the wide store. `stmatrix.m8n8.x4` takes 32 addresses from all
lanes, eight per matrix, its four matrices being the two slots' two halves in
order; `.x2` takes 16 from lanes 0..15 and is issued twice, once per slot. Lane
`l` of the `.x4` therefore supplies what lane `l % 16` supplied for slot
`l / 16`, which is what the test
`stmatrix_x4_addresses_are_the_x2_addresses_restacked` pins.

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
`regcount` table reads the difference directly.

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
