# `mma` — chained tcgen05 MMA walks

Design notes behind `src/mma.rs`. The source carries the operand shapes and the
safety contract; this file carries the geometry the walks are derived from and
the alternatives that lost.

## Chunk geometry

tcgen05's native step is `D (+)= A·Bᵀ` over a K=16 bf16 chunk, and a full tile
multiply is a chain of those steps with `enable_d` linking every step after the
first into the accumulator.

The geometry is fixed by a 2-byte element and the 128-byte swizzle atom. One
K=16 chunk is 32 bytes along a row, so a subtile row holds four chunks:

- A **K-major** operand (`[MN, K]`, K contiguous) walks
  `(k / 4) * SUBTILE_BYTES + (k % 4) * 32` across its stacked subtiles.
- An **MN-major** operand (`[K, MN]`, MN contiguous — the transpose-bit forms)
  supplies K along rows instead: 16 rows, `16 * ATOM_BYTES` bytes, per chunk. It
  reaches MN past its first 64 columns through the descriptor's *leading* offset
  rather than a step along the row. That is `SharedTile::mn_walk`, and every
  MN-major operand in the module is one.

The leading-offset detail is what lets `mma_atb` cover a 128-wide MN in one
instruction where `mma_ab` issues one per 64-wide band. A test pins it: a leading
offset of 16 would read a `[64, 128]` operand as if MN stopped at 64.

## Walk and descriptor cannot disagree

Which transpose bits the instruction descriptor carries *is* the choice of walk,
so every walk builds its own descriptor from `MmaShape` plus
`MmaElement::ELEMENT_TYPE`, and the transposed walks read their bits back off
`OperandWalk::transposed` rather than restating them.

That pairing used to be prose at the call sites, and getting it wrong does not
fault: the MMA reads the operands under the wrong interpretation and fills the
accumulator with wrong numbers. `shape` is the one descriptor field no walk can
know, so it stays the caller's argument.

The accumulator type is pinned to fp32 because a walk names TMEM by a bare `u32`
and has no accumulator type to read one off. fp16 accumulation is a
`.kind::f16`-only mode no kernel here uses, and threading a typed accumulator
through belongs with whatever gives the walks a `tmem::TmemTile` instead of an
address.

## `mm_*` versus `mma_*(.., false)`

The instruction is the same one either way; what changes is who owes the
invariant.

`mma_abt(.., false)` reads as an argument, so a call site that means "this is the
first MMA into a fresh band" and a call site that means "and I have checked
nothing else wrote there" spell themselves identically — and a `false` that
should have been `true` silently discards an accumulator. Naming the two cases
puts that in the type, and the `mm_*` entry points take no `accumulate` because
there is nothing left to decide.

The `accumulate` parameter stays on the `mma_*` walks for the callers whose
choice is *runtime*: `gemm`'s `k > 0` is one value threaded through one chain,
and splitting it into two typed arms would duplicate the chain for no gain.

## Why `mma_walk_cg2` names its element

The element is the one field the cluster walk cannot take from its operands: an
`OperandWalk` has already erased it, so `E` is named by the caller
(`mma_walk_cg2::<Bf16, CHUNKS>`) rather than derived as in `mma_abt`.

It stays that way now that `E` routes the *instruction* as well as the operand
format — `MmaElement::mma` selects tcgen05's `KIND` from the element. A walk that
carried its element back would have to carry a `KIND` too, which is to say it
would stop being layout-only. Keeping the layout in values is what lets a kernel
selecting K-major vs MN-major at runtime (`gemm`'s `transposed`) issue one loop
either way, with the descriptor's transpose bits moving with the walk.

## What is still bf16-shaped

The *instruction* is no longer on this list: `MmaElement::mma` routes `KIND` from
the element, so the walks are generic over it. What remains is the chunk geometry
above, which assumes 32 bytes per K=16 chunk. The walks assert a 2-byte element
at each site that depends on it rather than generalizing on a guess; widening it
waits on a second element to check a wider form against.
