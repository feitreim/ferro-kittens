# `layernorm` — layernorm over rows, and the whole-tile reduction next to it

Source: `examples/src/layernorm.rs`. Two kernels that differ in exactly one
thing — the axis their statistic is taken over.

| kernel | status |
| --- | --- |
| `layernorm_rows` | runs — `check` verifies it against a CPU reference on a B200, `bench` times it at six sizes |
| `groupnorm_tile` | compiles — no launcher and no reference |

Both are in the default build, so `scripts/modal-run build` is a real `sm_100a`
codegen gate on both.

## What they compute, and how they are laid out

One CTA of four warps owns a `[128, 128]` bf16 tile, in by TMA and out by TMA
from the same shared tile. A warp owns 32 rows; `blockIdx.x` picks the row
band.

`layernorm_rows` computes `y = gamma ⊙ (x − mean(x)) / sqrt(var(x) + eps) +
beta` per row, in three passes over the shared tile, each holding one
`CHUNK = 16` band at a time:

1. the row sum, giving the mean;
2. the centred second moment, giving `1/√(var + ε)`;
3. the output, with `gamma` and `beta` read a chunk at a time.

`row_sum` folds a thread's own values and then the quad holding the rest of the
chunk, and a chunk's quad is the same four lanes whatever the chunk, so the
per-chunk results combine slotwise with no shuffle of their own. `scale`/`shift`
keep `1/COLUMNS` and `epsilon` in a register; before those existed a constant
had to be splatted into a whole `RegVec` to be combined with one.

`groupnorm_tile` takes the same statistic over the *whole* tile, which needs
the four warps to agree. `RegTile::tile_sum` is warp scope: it folds a warp's
own band across all 32 lanes and stops, and warps cannot shuffle to each other,
so what closes the gap is *storage* plus a block barrier rather than another
butterfly. `kittens::sync::block_reduce_sum` folds one warp-uniform value per
warp into one block-uniform value through a `SharedVec<F32, 4>`. The two calls
sit back to back on the same scratch with no barrier between them, which is a
property of the collective and not an accident: it syncs on both sides, so the
variance pass cannot overtake the mean pass's readers.

### Shared memory

A launch of either kernel declares the larger of the two plans, which is
`layernorm_rows`' by two whole parameter vectors: 33,288 bytes. It was 33,344
while the plan was written as `Tile::BYTES + 2 · Parameters::BYTES + 64`; the
64 was slack nobody had a reason for, and a cursor that aligns each handle to
its own type's rule cannot reproduce a round number. This kernel's binding term
is registers in any case.

`gamma` and `beta` are parameters, not activations: loaded once, shared by
every row, and so they belong in shared memory — without a shared vector type
every thread would have re-read them from global.
`kittens::shared::SharedVec` is one flat run of elements with its own
single-box TMA path, unswizzled because a swizzle atom is a statement about a
tile's *rows* and a vector has one. 128 bf16 is 256 bytes: the vector's own
length, where a one-row `SharedTile` would have spent a whole 128-byte swizzle
atom to hold 64 of them. A vector being flat is also why the chunked walk needs
no addressing the library did not already have — a slice of one is a base
address and nothing else.

The four warps' partials are fp32 and not bf16: a partial rounded on its way
through shared memory would lose eight bits of the sum, and the variance pass
would inherit the error of the mean pass. 16 bytes, and never a TMA box — the
whole life of that vector is a `set` then a `get`, one barrier apart.

`groupnorm_tile`'s plan puts the partials before the barrier. That used to be a
correctness argument, when a hand-walked plan put each handle wherever the
arithmetic landed it; `SharedPlan::vec` aligns to 128 whatever precedes it, so
the other order is legal and merely costs 120 bytes of padding. This one is the
cheaper, and `kittens`' own plan tests assert both spellings.

## Why the row is walked in chunks

`layernorm_rows` originally held a whole `[32, 128]` band — 128 fp32 a thread —
and said the algorithm in four lines over it. `regcount` priced that at **255
registers, a 1552-byte stack frame and 16 bytes of spill**, and the occupancy
query read **2 blocks and 8 warps an SM** against `softmax_rows`' 6 and 24 at
33,344 against 32,800 bytes of shared memory.

It is the mirror of `softmax`'s case, and the pair is what says what the rule
actually is. `softmax_rows` was **39** registers on a 1024-byte frame: too few,
the band never promoted, and the cost landed on memory traffic — 2.6×. This
kernel was 255 on a 1552-byte frame: the band *still* did not fit, and on top
of that the registers it did hold were enough to cap the SM at two CTAs.
Neither number is a target. **The band living in registers with the frame at
zero is the target**, and reading either count as a speed is what hid the
problem for as long as it hid.

The fix is one number. The launch geometry, the barrier protocol, the TMA plan
and both waits are untouched, so what changed between the measurements is only
where the band lives.

### The number is 16, and `softmax`'s 32 would have been wrong

`M·N` predicts nothing — `[64, 64]` and `[32, 128]` are both 128 fp32 a thread
and 220 registers apart. Here the *row* extent is not a free parameter at all:
a warp owns 32 rows of a 128-row tile, and changing that is a launch-geometry
change. So the only ladder that applies is the `[32, N]` column sweep — and
this kernel was built at every rung of it rather than handed `softmax`'s
answer, which is the part that mattered:

| `CHUNK` | regs | spill | frame | blocks/SM | GB/s at 8192 blocks |
| --- | --- | --- | --- | --- | --- |
| 128 — the whole band, as filed | 255 | 16 B | 1552 | 2 | 317 |
| 64 | 108 | 0 | 272 | 6 | — |
| 32 — `softmax`'s answer | **40** | 0 | 128 | 6 | 3776 |
| **16 — shipped** | 47 | 0 | **0** | 6 | **5539** |

317 → 5539 GB/s, a factor of **17.5**, and the flat floor at the small end goes
32.4 µs → 8.4 µs with it. That floor is worth a line of its own: `softmax`'s
23 µs at two blocks had been read as a fixed cost every launch on this harness
pays, and this kernel reaches 8.4 µs on the same harness. Whatever that 23 µs
was, it was not the launch — see `docs/kernels/softmax.md`, which later took it
apart microsecond by microsecond.

**`CHUNK = 32` is the cheaper kernel by register count and the slower one by
1.46×.** It is seven registers under the shipped form and carries a 128-byte
frame the shipped form does not, and those seven registers are the band not
fitting. Reading the register column as a speed picks the wrong rung here, and
that is the third time in this repository it would have.

The mechanism was checked from the other side too, on the kernel this change
did not touch: `softmax_rows` at `CHUNK = 16` prices at 50 registers on a
**128**-byte frame against its then-shipped 66 on **256**, and runs 1185
against 950 GB/s. Half the frame, a quarter more throughput, on a kernel whose
arithmetic did not change — so "frame down, time down" is not a property of
this kernel's rewrite alone.

The register-count probe in `experiments/README.md` reports a **zero** frame
for `[32, 32]` in all five of its spellings. This kernel gets 128 bytes at that
shape. The probe is not the kernel: it carries no `ColVec` parameters and no
statistics across chunks, and this kernel's extra live state moves its cliff one
rung down. **The ladder narrows the search to a handful of rungs; it does not
answer for a kernel that has to be built.**

Occupancy is the same 6 blocks an SM at both 16 and 32 — shared memory binds
there, as it does for `softmax` — so the 1.46× is not occupancy. It is the
local-memory traffic itself: a streamed band is cheap when the registers it
frees buy resident warps, and this kernel is past the point where they buy
anything.

`groupnorm_tile` was deliberately left holding the whole band while it had no
launcher and no reference — a rewrite nothing can check is not an improvement —
and its counters *not* moving while `layernorm_rows`' did was the control that
said the change was the band walk and not something the module did. Once it
got its own check (below), the same streaming was applied and the same thing
happened, one column over:

| `groupnorm_tile` | regs | frame | blocks/SM | GB/s at 8192 blocks |
| --- | --- | --- | --- | --- |
| whole band, as filed | 168 | 1536 B | 3 | 594 |
| streamed at [`CHUNK`] | **48** | **0** | 6 | **5996** |

594 → 5996 GB/s, a factor of **10.1**, at a bit-identical worst error
(7.76e-3 both ways — the arithmetic is the same centred form, only where the
band lives changed). The 168 was `#[launch_bounds(128, 3)]` forcing `ptxas`
under a cap; the whole band never fit, and the difference between this
kernel's 594 and `layernorm_rows`' pre-fix 317 is that the cap bought a third
resident CTA to hide some of the local-memory latency behind. The streamed
form runs 8% *above* `layernorm_rows`' 5539 — the same traffic minus two
parameter vectors. The carry across chunks here is one scalar per statistic
rather than a `Rows`: a chunk's `tile_sum` is already a number the warp agrees
on. The rung is inherited from the column sweep above rather than re-swept —
the two kernels differ only in the statistic's axis — and the shipped rung is
measured.

### The other pole of the same idiom

594 → 5996 GB/s is the strongest result in this tree for chunked streaming, and
it is worth stating what it is a result *about*, because the same rewrite has a
measured negative. Both kernels here walk a **TMA-staged shared tile** through
`ldst::load_tile`, and what the streaming fixed was a spilled band: 168
registers and a 1536-byte frame down to 48 and none.

The same walk written against **global memory** — `global::load_rows`, a warp's
16-row band, no staging tile — measures 2.1× a block-per-row RMS norm at
`dim = 3072`, and the narrow-chunk arm that buys back the reference's full 2048
threads an SM measures 2.7× (#222, `experiments/src/norm_occupancy.rs`). Nothing
about that kernel was frame-bound; what it paid was a warp's access width, and
raising occupancy made it worse.

So the two poles are not "walk versus no walk". They are: **a band that does not
fit in registers wants streaming, and a row that is already coalesced in global
memory does not want a fragment map.** `docs/library/global.md` carries the
five-arm table and the decision rule.

### Why the variance pass centres first

The shipped form centres and then squares. The one-pass alternative,
`E[x²] − E[x]²`, would let the mean be computed and the variance accumulated
together — but this seed's rows sit up to 27 away from zero, so it cancels most
of a large number, and the accuracy is spent for a pass that costs
shared-memory reads the TMA has already paid for. This is the one comment left
in the body of that loop.

### Why the parameters are read inside the third pass

Every lane reads the columns its fragment owns, and the 8 lanes of a column
group ask for the same address and shared memory answers them together — which
is the access pattern a swizzle would have been spreading for nothing. Reading
them there rather than before the reductions is what keeps two `ColVec`s off
the live set across two full passes.

### `groupnorm_tile` declares its residency

`#[launch_bounds(128, 6)]` is `__launch_bounds__`: it puts `.maxntid 128, 1, 1`
and `.minnctapersm 6` on the entry, so `ptxas` is told the residency the kernel
is written for. Six is what shared memory admits at the declared plan, which is
the same binding term `layernorm_rows` has — the kernel streams its band now,
and at 48 registers nothing register-side is close to the line.

The attribute said 3 while the kernel held its whole band: three CTAs an SM at
128 threads is 168 registers a thread — `_register_ceiling(3, 128)` in
`modal_app.py` derives the same 168 — and that was a cap `ptxas` had to be
*forced* under, because it had sat on 168 by luck until the file was compiled
into a smaller crate and the same source came out at 236, crossing the step
with nothing in the tree to notice. A count that lands on its ceiling by luck
is not a residency, and the occupancy gate can only watch a kernel that states
one. The attribute takes a literal thread count and the occupancy gate reads it
back out of the source with a digit regex, while `THREADS` derives from `ROWS` —
a tile that changed shape would move one and leave the other, so the file
carries a `const _: () = assert!(THREADS == 128)` to make the two say the same
number out loud.

## What the check has to be, and why it is not `softmax`'s

`softmax`'s seed gives every row the same 128 values in a different order, and
for a softmax that is enough. For a layernorm it is a hole: identical multisets
mean identical row statistics, so the *whole-tile* statistic `groupnorm_tile`
takes is numerically the row statistic, and a kernel that reduced over the
wrong axis would compute the right answer. So would one that never subtracted a
mean, if the mean were zero. Neither is a hypothetical: they are the two
kernels in this file.

The row therefore has to change the *shape* of its distribution and not merely
the order — and it cannot do it by scaling or shifting, because layernorm is
invariant to both. `x → a·x + c` is exactly the transform the kernel divides
out, so a seed of that form makes every row's output bit-identical and the check
blind to every row error there is.

### The seed

    stride      = 11 + 2 · (row / 128 % 63)
    permutation = (37·row + stride·column) % 128

    magnitude = 16 + (p % 32) / 2
    boost     = 1 << (1 + row % 3)                       // 2, 4 or 8
    value(p)  = magnitude · [1, −1/2, boost, −boost/2][p / 32]

The stride construction and its constant 63 are `softmax`'s, for `softmax`'s
reason: a seed of the form `(a·row + b·column + c) % 128` is an affine map of
the column axis, those form a 2-group of exponent 128, and so every row-affine
seed repeats after exactly one 128-row band. An odd cycle is the only thing that
reaches past one, and 63 is the longest available.

What varies on top of it is one scale factor applied to *half* the ladder. `p`
splits into a magnitude and one of four scalings chosen by `p / 32`; the third
and fourth carry a row-dependent boost. That is not an affine image of a fixed
vector, so mean and variance both move with the class — **8.9/28.9, 14.8/53.8,
26.7/105.7** — while the ladder stays 128 *distinct* values at every class,
which is what keeps the permutation's row argument intact on top of it.

Every value is a magnitude of at most six significant bits times a power of
two, so all 384 of them are exact in bf16 and nothing is lost on the way in.
The sign asymmetry (`1, −1/2, boost, −boost/2`) is what makes the mean
non-zero; the ladder's own range ratio of 1.97 being under the smallest boost is
what keeps the four scalings from colliding.

The parameters are `gamma(c) = 1 + c/128` and `beta(c) = 2 + c/64`, injective
over `0..128` and exact in bf16 — injective because a `gamma` or `beta` read
from the wrong column is one of the errors this check exists to see, and a
periodic parameter would hide exactly the misindexing a chunked column walk can
commit.

Because the permutation is a bijection in `column`, a row holds each ladder
entry exactly once and its statistics depend on the row only through its class.
Three class statistics therefore cover every size the harness runs.

### `epsilon`

`1e-5`, stated as a residual rather than sold as coverage: no row this seed
produces has a variance under 835, so `epsilon` is 1.2e-8 of the smallest term
it is added to and a kernel that dropped it entirely would pass. What it guards
is a degenerate row, and a check whose whole job is to police the *arrangement*
of a non-degenerate one cannot also be the test for that.

### The tolerance

2⁻⁷ relative, and the reason it is not `softmax`'s 2⁻⁸ is that half a bf16 ulp
relative is not a constant. A bf16 carries eight significand bits, so an ulp at
`m·2ᵉ` is `2ᵉ⁻⁷` and half an ulp relative is `2⁻⁸/m` — 2⁻⁹ only when `m` is
near 2, and a full 2⁻⁸ when it is near 1. `softmax`'s outputs happen to sit at
the forgiving end; this seed's do not. Its smallest output is 0.552, whose
significand is 1.104, and the correctness run measures a worst relative error of
**3.87e-3 = 0.99 × 2⁻⁸**. That is the rounding floor, not an error, and a
tolerance of 2⁻⁸ would have been sitting on it.

So 2⁻⁷ is one doubling above the floor, which is the same factor of two of
headroom `softmax` has, arrived at honestly rather than inherited. Nothing else
is competing for it: unlike `softmax` there is no `exp2` polynomial, because
`kittens::reg::rsqrt` is a correctly-rounded `sqrt.rn.f32` and a divide rather
than an SFU approximation. What is left over is the order the shuffle butterfly
sums a row in, and the seed keeps every value at least 0.22 away from its row's
mean, so no output is built out of a cancellation the host reference does not
also perform.

### `groupnorm_tile`'s check, and why its error is absolute

The same seed feeds `check_group` — the whole point of its non-identical row
multisets is that the tile statistic and the row statistic differ, so a kernel
that reduced over the wrong axis fails here. The reference needs three tile
statistics the way the row check needs three class statistics: a row's multiset
depends only on its boost class, so a tile's depends only on where its 128 rows
land in the 3-class cycle, and 128 not dividing by 3 is what makes consecutive
tiles differ.

The error is absolute where the row check's is relative, because there is no
`beta`. `groupnorm_tile` has no parameters, its outputs are zero-mean and
unit-variance by construction, and nothing holds one away from zero — a
relative error against a near-zero expected value measures nothing but the
denominator. Against outputs whose scale *is* 1, an absolute error is the
relative error to the tile's scale, said without the division.

The floor is the same rounding argument one octave up: this seed's largest
normalized outputs sit in `[2, 4)`, where a bf16 ulp is 2⁻⁶, and the
correctness run measures a worst absolute error of **7.76e-3 = 0.99 × 2⁻⁷** —
half an ulp at that magnitude, exactly as the row check's 3.87e-3 is half an
ulp at `[1, 2)`. The tolerance is 2⁻⁶, one doubling above.

### The errors it is against

Scored exhaustively over the `256 rows · 128 columns` = 32,768 cells of the
check by the same functions the reference uses. "p10" is the tenth-percentile
relative error among the cells a given error makes wrong, so nine tenths of them
are worse:

| error | cells wrong | p10 | median |
| --- | --- | --- | --- |
| row displaced by 1 | 32,765 | 0.277 | 0.810 |
| row displaced by 32 (a warp band) | 32,738 | 0.256 | 0.776 |
| row displaced by 128 (a CTA band) | 32,620 | 0.118 | 0.501 |
| every CTA reads band 0 | 16,310 | 0.118 | 0.501 |
| third pass pinned to column chunk 0 | 28,672 | 0.155 | 0.560 |
| `gamma`/`beta` pinned to column chunk 0 | 28,672 | 0.108 | 0.321 |
| `gamma` and `beta` swapped | 32,512 | 0.099 | 0.797 |
| statistic over the whole tile | 32,768 | 0.030 | 0.186 |
| mean never subtracted | 32,768 | 0.078 | 0.144 |
| variance as the raw second moment | 23,662 | 0.011 | 0.019 |

The worst row displacement over `1..=129` leaves 31,684 of 32,768 wrong at a p10
of 0.071, which is 9× the tolerance. The last row is the weakest and is named
rather than rounded off: forgetting to centre the *variance* — but not the
output — still fails 72% of the cells, at a p10 of 1.4× the tolerance.

Two rows are worth their own line because they are the two the seed was built
for, and a permuted-multiset seed of `softmax`'s shape scores zero on both:
**statistic over the whole tile** is this file's other kernel, and **mean never
subtracted** is the pass a chunked walk has to carry across chunks. Each fails
every one of the 32,768 cells.

### Two of them were run, not just scored

A table computed by the same functions as the reference is an argument, not a
control — it cannot fail in a way the reference does not also fail. So two rows
were built as kernels and launched:

- **Third pass pinned to column chunk 0**, the error the chunked walk is the
  one thing that can commit. Predicted 28,672; the device failed **28,672**, to
  the cell.
- **Variance as the raw second moment**, chosen because it is the *weakest* row
  in the table — if the check sees this one it sees all of them. Predicted
  23,662; the device failed **24,410**. The 3% is the device rounding through
  fp32 and bf16 where the host model is exact `f64`, which moves cells sitting
  on the tolerance across it.

The residual is `softmax`'s: a constant row displacement is invisible exactly
when it is a multiple of `63 · 128` = 8064 rows, and no power-of-two grid
arithmetic produces one.

## Related

- `docs/kernels/softmax.md` — the same chunked walk, the `exp2` ablation, and
  the seed construction this file's stride is borrowed from.
- `docs/kernels/harness.md` — the clock, and the verify-then-time order.
- `experiments/README.md` — the register-count ladder this kernel's rung sweep
  narrows the search with.
