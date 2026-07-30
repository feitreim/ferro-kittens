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
