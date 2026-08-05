# `reg.rs` — register tiles, vectors and the scalar op set

Design notes and measurements behind `src/reg.rs`. The source carries the
contract; this file carries the numbers it was decided on and the alternatives
that lost.

## The layout is a trait, and the shape set is a product

A `RegTile<M, N, L>` stores `L::Storage`, an associated type, rather than
`[[f32; N/4]; M/8]`. An array length has to be a const expression of the
generic parameters, which needs `generic_const_exprs`. Splitting the map into a
`RowLayout<M>` half and a `ColLayout<N>` half makes the storage a *projection*
instead: `RowLayout::Slots<T>` is generic in its element, so a tile's storage is
`Slots<Values>` and no arithmetic on `M` and `N` is performed anywhere.

`FragmentLayout` is then a blanket impl over every `RowLayout × ColLayout` pair.
The consequences:

- A shape costs no line of its own. 32 row extents and 32 column extents give
  1024 tile shapes out of 64 impls.
- A layout defined *outside* this crate is a tile layout as soon as it has both
  halves — which orphan rules put out of reach for a per-shape impl list.
- `FragmentLayout` is never implemented directly, here or downstream.
- The storage the blanket impl projects is byte-for-byte the
  `[[f32; N/4]; M/8]` the per-shape impls named, which is what says the change
  was a spelling and not a representation
  (`the_shape_set_is_the_product_of_the_extents`).

The extent list stops at 512 because of the register file, not because of a
guess at what kernels want: a thread holds `M * N / 32` fp32 values of an
`[M, N]` warp tile, so with the other extent at its 16 minimum, 512 is already
256 registers a thread — one past what the hardware has. No shape outside the
grid fits in registers at all. Unused extents cost nothing; a trait impl no tile
names emits no code.

### `CONTIGUOUS_VALUES` defaults to 1

The default is the answer that is true of *every* map — no two values adjacent —
so a layout written later gets scalar accesses until it claims otherwise, rather
than silently inheriting `BaseLdtm`'s arithmetic. Raising it is a claim about
the map that `global::store_rows` and `global::load_rows` act on, and it is
checked as stated (`base_ldtm_pairs_every_other_value`) rather than by
re-deriving `{0, 1, 8, 9}`.

`BaseLdtm` claims 2 and not 4: `{0, 1, 8, 9}` puts values 1 and 2 seven columns
apart, so the run breaks at the `1 -> 8` step.

## `exp2`: the polynomial against the SFU

Two implementations exist. `exp2_approx` is the FMA polynomial the validated
kernels shipped with; `exp2_hw` is one `ex2.approx.f32` SFU instruction
(`cuda_device::float::ex2_approx_f32`, a generated intrinsic at the pinned
revision, so no libdevice call).

**On a clock the SFU wins by 2.7×.** Timed on `softmax` at `CHUNK = 16` over
8192 blocks:

| | registers | stack frame | throughput |
| --- | --- | --- | --- |
| `exp2_approx` (polynomial) | 50 | 128 B | 1178 GB/s |
| `exp2_hw` (SFU) | 48 | 0 B | **3153 GB/s** |

Accuracy does not separate them: `softmax`'s exactness check measures a worst
relative error of 1.97e-3 either way, because the bf16 round trip dominates and
not the transcendental.

An earlier claim that the measurement did not favour the SFU was taken from a
*register count* on a probe shape no kernel has; the throughput numbers above
are the correction.

**The default has not moved anyway.** `exp2` — the method name, on every
register family — still resolves to `Exp2Approx`, and
`exp2_stays_the_polynomial_everywhere` asserts it. Which implementation a *name*
means is a numerics change, and `flash_forward`, the other caller (twice per
element in its inner loop), has no CPU reference to check one against. `softmax`
calls `exp2_hw` explicitly; ports that must hold "same SASS" keep the
polynomial.

## By-value against in-place maps

Every map comes in two spellings: by value, and through `&mut self` with an
`_assign` suffix. They compute the same thing —
`in_place_tile_maps_are_their_by_value_twins` checks every generated name, not a
sample — and differ only in what the register allocator has to hold.

The surprise is that the calling convention is not what orders them. Measured on
`scalar_map_probe_128` and `softmax_probe_128` (`modal_app.py::regcount`,
sm_100a), what orders the spellings of one `[32, 128]` step is how many whole
bands have to be *materialized between statements*:

| spelling of one step | registers | spill |
| --- | --- | --- |
| `out_acc = out_acc.scale(k).add(block.scale(k))` | 168 | none |
| `out_acc.scale_assign(k); out_acc.add_assign(b)` | 252 | none |
| `out_acc = out_acc.scale(k); .. = ..add(block)` | 255 | 60 B |
| `out_acc = out_acc.add(block.scale(k))` | 255 | 108 B |

The rule that falls out: **say the whole step in one expression if you can;
write it in place if you cannot.** An in-place form is worth reaching for where
the input is the output — an accumulator, or a rescale of one — which is
where a single expression cannot be written and where a by-value map costs a
whole band. Where the by-value spelling already rebinds a dead input it costs
nothing and needs no conversion.

Supporting figures:

- `row_map::<Mul>` against `mul_row_assign` at the flash accumulator's width:
  255 against 168 registers.
- `scale_rows` was hand-written for exactly this. It is `mul_row` by definition
  (`scale_rows_is_the_multiply_row_map`), but swapping in the by-value
  `mul_row` takes `softmax_probe_128` from 168 to 255 registers/thread — the
  allocator does not prove the input band dead. The *in-place* map measures
  identically to the hand-written loop (64 registers at 32 columns, 168 at 128,
  same spills, same stack frame), which is the bar that let the hand-written
  body be deleted.
- A hand-written in-place scalar loop measured 252 registers and no spill where
  the by-value spelling spilled 60 bytes.

Unmeasured: no probe monomorphizes a ternary map at 128 columns, so what
`ternary_map_assign` saves over `ternary_map` there is a guess.

## Why the by-value maps do not delegate to their in-place twins

`fn map(self) -> Self { let mut out = self; out.map_assign(); out }` is the
obvious factoring, and it is not free. Routing the by-value maps through their
in-place twins moved *every* probe in `regcount` that uses a map, in both
directions:

| probe | before | after |
| --- | --- | --- |
| `mask_probe_128_causal` | 71 regs | 32 regs |
| `softmax_probe_32_hand_written` | 64 regs | 80 regs |
| `lane_probe_128_hoisted` | 168 regs | 128 regs, stack frame +512 B |

So the duplication stays, held honest by
`in_place_tile_maps_are_their_by_value_twins` rather than by sharing a body.

## Why the reductions spell their folds out

The three reductions on `RegTile` each write their lane-local fold inline rather
than sharing one `fold(self, slot)` helper. The helper costs a whole tile: even
`#[inline(always)]`, taking `self` by value materializes a second copy of the
storage. `softmax_probe_32` measures 94 registers/thread against 64 for the
written-out loop, and `softmax_probe_128_row_map` picks up 456 bytes of spill
stores on the same change.

Same mechanism as `scale_rows` being `row_map_assign::<Mul>`: what costs is the
copy the shared body is reached through, not the sharing.

The `ReduceOp::IDENTITY` seed costs one extra `apply` per fold. The FMA-seeded
forms fold it away (`Max::apply(-inf, x)` is `x` by construction) and the others
do not — measured at no register cost either way, once the fold is inline.

## `online_rescale`'s fusion

One correction step of the online softmax, in the exact per-slot order of the
hand-written kernels. Fused on purpose: one scalar `next`/`factor` live at a
time, each row's values rescaled before the next row's factor is formed.

The unfused form (`max` / `sub` / `exp2` / `scale_rows`) keeps two full vectors
live across the accumulator scaling and costs registers in register-tight
kernels: **206 → 212 regs/thread on B200** in the persistent forward kernel.

The probes bound the fusion's worth but do not license undoing it —
`softmax_probe_32` reproduces the direction at 56 against 64 regs/thread and
`softmax_probe_128` does not reproduce it at all (168 either way), because the
probe holds its accumulator in registers where the real kernel holds it in TMEM
and drains. The 206 → 212 number stands until a kernel of that shape says
otherwise.

## Hand-written loops that became generic maps

`RegVec`'s `max` / `sub` / `exp2` / `mul_assign` / `add_assign` were hand-written
copies of the compile-time-length slot loop. `modal_app.py::regcount` showed the
generic maps assemble to the same registers and spills at both probe shapes, so
they are the maps now.

## Masks take a coordinate origin

A masked tile is normally a sub-block of a much larger score matrix, so its
diagonal sits at `query_base - key_base`, not at `row == column`. Masking a tile
against its own `row == column` compiles, runs, and is wrong everywhere off the
diagonal block. ThunderKittens' `tril` has no such parameter; that is right for
exactly one block of a tiled kernel and silently wrong for the rest.

Two follow-on decisions:

- **Coordinates are `i32`.** A tiled kernel's bounds are differences of block
  origins and leave `0..M × 0..N` in both directions. That is not an edge case:
  it is how a band wholly above the diagonal takes `keep` false everywhere and
  one wholly below takes it true everywhere, which is most of the bands in a
  causal kernel (`a_band_off_the_diagonal_is_all_or_nothing`).
- **`make_causal_at` takes the two origins, not their difference.** Both are
  `u32` at the call site, so `query_base - key_base` written there wraps —
  `0u32.wrapping_sub(1024)` is +4294966272 — and masks nothing. The helper
  subtracts in `i32`.

Masking is in place because a mask is the innermost thing in an attention loop
and its input is dead the instant it returns; the by-value spelling would put a
second band beside the score band at exactly the width the table above prices.

Lane-local throughout: the map hands a thread whole `(row, column)` pairs, so a
mask is a select on registers it already holds and no lane learns anything from
another.

### `mask` is the one walk in this file that unrolls

Both of its loops carry `__unroll_config::<0>()` and an inline-`const` bound —
#166's pattern, applied here by #184. It is the only walk in `reg.rs` that
does, and the asymmetry is the point.

A conditional write is worse than a map, not the same. `self.set(slot, value,
fill)` under a data-dependent `keep` keeps the slot array *addressable* across
the branch, so the rolled form homes the whole tile to a `.local` frame — and
it homes it on every iteration of whatever loop the mask sits in, not only on
the iterations that mask anything. oxide-train's tile classifier paid a 400 B
frame, 150 `st.local` and 104 `ld.local` for one `right_fill` on the
vocabulary's ragged last chunk, one chunk in 786, with the stores inside the
vocabulary loop. Nothing else here has that shape: a map writes every value
unconditionally, and a reduction writes none.

Two things were measured and rejected on the way in. Writing the mask as an
unconditional select — `self.set(slot, value, if keep {old} else {fill})`, so
there is no branch to keep the array addressable — does *not* fix it on its
own: the walk stays rolled and the frame stays (`flash_forward` 165 → 167
registers, 1824 B either way). With the unroll markers it compiles to exactly
the byte-for-byte same PTX as the branch form, so the extra `get` buys nothing
and the branch stays. Separately, replacing `BaseLdtm::column`'s `[0, 1, 8,
9][value % 4]` — itself a local array in a rolled walk — with the equivalent
`8 * (value / 2) + value % 2` moves register counts across half the probe
table in both directions and takes four of `regcount`'s twenty timed rungs off
their ladder twins. It is a real finding and it is not this change.

The cost is that a tile the mask leaves in registers has to *be* in registers.
`flash_forward`'s score band comes back from a 256 B frame and the kernel goes
165 → 255 registers with no ptxas spill; it can afford that because its own
147 532 B shared plan already holds it to one CTA per SM (the driver says 1
before and after), so the register step from three CTAs to two is not the
binding term. A kernel with room to spare in shared memory would read this
differently, which is what `docs/kernels/flash_forward.md` records.

## A scalar operand is a `BinaryOp`

`scalar_map::<Mul>(k)` is `scale`, `scalar_map::<Add>(k)` is `shift`. The
rejected alternative was a stateful op trait carrying the constant.
`UnaryOp::apply` is an associated function on a unit struct, so a scaling factor
has nowhere to live, and giving it somewhere would put a value in a type
parameter's shadow where the inliner has to rediscover it. Treating the scalar
as the second operand means the whole existing op set — `Div` for a divide by a
constant, `Max`/`Min` for a bound, `Sub` — is reachable the day it is written,
and the named table (`scale`, `shift`, `clamp_min`, `clamp_max`) holds only the
spellings a kernel would otherwise invent a worse name for.

`scalar_map_is_bin_map_against_the_splatted_scalar` pins the claim on all three
register families: it is the two-operand form with the operand splatted, so the
only thing it saves is the splat, and it saves it identically everywhere.

Ops are unit structs rather than `fn` pointers or closures for the same reason:
a type parameter can carry no state to spill and nothing for the inliner to see
through.

## `ReduceOp` is narrower than `BinaryOp` on purpose

A foldable op must be associative and commutative — the fragment map hands a
fold its operands in the layout's order, not the tile's — and must carry an
identity. `Sub` and `Div` are deliberately not members: `row_reduce::<Sub>` has
no meaning worth giving a spelling. The identity lets every fold start the same
way instead of special-casing element zero, and a transposed line in the
identity table is a wrong reduction rather than a slow one (`Mul` seeded from
zero is zero).

## The reduction masks come from the ownership map

`BaseLdtm`'s two halves decide which lanes a fold must reach: `row_of` ignores
`lane % 4`, `col_of` ignores `lane / 4`. So a row reduction shuffles masks 1 and
2, a column reduction 4, 8 and 16, and a whole-tile reduction all five. A wrong
mask yields a plausible wrong number rather than a crash, which is why
`reduction_masks_are_the_ownership_maps_lane_groups` derives the groups from the
maps — a mask set is right exactly when its xor-closure is the set of lanes the
map says share the axis being folded away — instead of restating the constants.

`tile_reduce` gets the whole-tile answer in five shuffles against the
`2 * SLOTS + 3` of routing through `row_reduce` and `RegVec::reduce`; the latter
exists for the case where the row vector is wanted anyway.

All of it is warp scope. A tile several warps own — layernorm's group-norm
statistic over four warps' bands — needs those warps to agree, which is a
shared-memory staging step nothing here helps with.

## Math lowerings, and what is deliberately not libdevice

Nothing in this module makes a libdevice call, and several spellings exist to
keep it that way:

- `fmax` / `fmin` are comparison-selects, retained for their established SASS.
  Libdevice-backed `f32::max` is artifact-safe now; the lowering is kept for the
  SASS, not out of necessity.
- `Abs` clears the sign bit — one `abs.f32` — rather than calling `fabsf`.
- `Sqrt` is `llvm.sqrt.f32`, which NVPTX lowers to native `sqrt.rn.f32`.
- `rsqrt` is a correctly-rounded `sqrt.rn.f32` and a divide, not the SFU
  `rsqrt.approx.f32`: two instructions, but the same number on host and device,
  which is what lets a normalization kernel's host reference compare with `==`
  (`free_rsqrt_is_the_op_struct`). Adopting the SFU form would be a measured
  swap like `exp2_hw`'s.
- The free `rsqrt` function exists beside `exp2_approx` because layernorm's
  scalar variance — already one `f32` — has no vector to reach the `Rsqrt` op
  struct through.
- `exp2_approx`: max relative error 7.5e-5 on the reduced range, checked over
  `-125..125` (`exp2_polynomial_stays_inside_its_error_bound`). The clamp keeps
  the exponent field normal and flushes masked-sentinel inputs to ~2^-125, which
  is what makes a fully-masked row sum to zero rather than to `NaN`.
- `log2_approx`: |error| < 5e-8 on the reduced range; the coefficient literals
  are bit-exact copies of the validated kernel's.
- `Exp` inherits `exp2_approx`'s error bound and its ±125 exponent clamp, so it
  saturates at `x = ±86.6`, just inside where fp32 overflows anyway.
- `Fma` exists because Rust emits no fast-math flags: a separate multiply and
  add stay separate, so the fusion has to be asked for.

## ThunderKittens correspondences

TK names a register vector for the axis it *spans*: its `col_vec` has one entry
per row, its `row_vec` one per column. We name for the axis that *indexes* it,
so TK's `col_vec` is our `RegVec` and TK's `row_vec` is our `ColVec` —
inverted. The op names do agree: TK's `row_map` / `mul_row` and ours both
broadcast a per-row scalar along each row.

Smaller ones:

- TK's nullary `one` / `pos_infty` / `neg_infty` are `splat` with the constant
  written out.
- TK's `fma_AxCtB` is `Fma` with the last two operands swapped at the call site.
  Our maps fix no operand to a broadcast vector, so the second form buys
  nothing.
- `tile_max` is prefixed because `max` is already the elementwise binary op. TK
  distinguishes the two by C++ overloading, which Rust has no equivalent of.
- `row_prod` / `col_prod` / `tile_prod` fold with `Mul`. A separate `Prod` op
  would be the same `a * b` under a second name.
