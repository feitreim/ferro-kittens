# `softmax` — row-wise softmax over a tile

Source: `examples/src/softmax.rs`. Runs on a B200 via
`modal run modal_app.py::examples`; `check` verifies it against a CPU reference
and `bench` times the same launch afterwards.

## What it computes, and how it is laid out

One CTA of four warps owns a `[128, 128]` bf16 tile. The tile arrives by TMA,
is normalized row-wise in registers, and leaves by TMA from the same shared
tile. A warp owns 32 of the 128 rows; `blockIdx.x` picks the row band and
`blockIdx.y` the plane.

Each warp walks its 32 rows a `CHUNK = 16` columns at a time, three passes over
the shared tile:

1. the row peak,
2. the denominator, `Σ exp2(x − peak)`,
3. the output, `exp2(x − peak) · (1/total)`, written back over the input.

Both per-row statistics carry across chunks. `row_max`/`row_sum` fold a
thread's own values and then the quad holding the rest of the chunk, and a
chunk's quad is the same four lanes whatever the chunk, so the per-chunk
results combine slotwise with no shuffle of their own.

Dynamic shared memory is the tile plus one semaphore: 32,776 bytes. It was
32,800 while the plan was written as `Tile::BYTES + 32`; the 32 was slack
nobody had a reason for, and a cursor that aligns each handle to its own type's
rule cannot reproduce a round number. A strictly smaller plan cannot cost
residency, and this kernel is bandwidth-bound on a tile it touches once.

## Why the row is walked in chunks

The body originally held a whole `[32, 128]` band — 128 fp32 a thread — and
said the algorithm in three lines over it. It computed the right answer at
367 GB/s, and the reason was not the memory protocol: `ptxas` never put that
band in registers, and every pass over it was a round trip to the same
hierarchy the TMA was already saturating. Chunking took it to 952 GB/s.

The row extent is not a free parameter — a warp owns 32 rows of a 128-row tile
and changing that is a launch-geometry change — so the `[32, N]` column sweep
is the only ladder that applies. It was measured at every rung with the shipped
arithmetic in place rather than read off a synthetic probe:

| `CHUNK` | regs | spill | frame | blocks/SM | GB/s at 8192 blocks |
| --- | --- | --- | --- | --- | --- |
| 128 | 197 | 0 | 1024 | 6 | 278 |
| 64 | 40 | 0 | 512 | 6 | 554 |
| 32 | 39 | 0 | 256 | 6 | 1004 |
| **16 — shipped** | **32** | 0 | **0** | 6 | **4414** |

The register-count probe in `experiments/README.md` reports a **zero** frame
for `[32, 32]` in all five of its spellings; this kernel carries 256 bytes
there, because the probe carries no statistics across chunks and a real
kernel's extra live state moves its cliff a rung down. **The ladder narrows the
search; it does not answer.**

## The rung was a quarter of it, and `exp2` was the rest

With the rung fixed, `layernorm_rows` ran 5483 GB/s where this kernel ran 950 —
a factor of **5.8** at equal bytes, equal blocks and the same 6 CTAs an SM,
against a kernel doing *more* arithmetic per element. At the then-shipped
`exp2` the rung alone is worth only 950 → 1178, so the rung is not what
explains it.

The rest was found by ablation rather than by argument: the same kernel, the
same geometry, one thing removed at a time, all at `CHUNK = 16` and all at 8192
blocks. The two bottom rows do not compute a softmax and exist only to be
timed.

| at `CHUNK = 16` | GB/s | what it removes |
| --- | --- | --- |
| `exp2` polynomial, `div_row` | 1178 | — (the arithmetic as filed) |
| `exp2_hw`, `div_row` | 3153 | the FMA polynomial |
| **`exp2_hw`, `recip` + `mul_row`** | **4414** | 124 of every 128 divides |
| no transcendental, no divide | 6042 | *both* `exp2` calls |
| one pass, `load_tile` → `store_tile` | 6324 | the two extra passes |

Read downwards, that is the whole 5.8×, and it is three things and not one.

**The FMA `exp2` polynomial is the largest single term, at 2.7×.**
`kittens::reg::exp2_approx` is a clamp, a shift-trick split and a degree-3
minimax polynomial — around a dozen instructions, evaluated twice per element,
128 elements a thread. `exp2_hw` is one `ex2.approx.f32`. This is **not** an
accuracy trade in the direction it looks: the SFU instruction is good to about
2⁻²², the polynomial to 7.5e-5 = 2⁻¹³·⁷, so the swap is *more* accurate and the
check's worst relative error does not move at all (1.97e-3 either way) because
bf16 output rounding dominates both.

**The 128 `div.rn.f32` cost 1.4×.** Folding the row denominator to a single
reciprocal and multiplying leaves four `div.rn.f32` a thread and 128
multiplies, in place of 128 divides.

**The three passes over shared memory cost 4.5%** — 6324 → 6042 GB/s for two
extra full reads of the tile plus every row reduction in the kernel. That is
the one term the memory protocol could have addressed, and the one that was
never worth addressing.

### A control is only valid under the bind it was taken in

The reciprocal swap had been measured before, under the `exp2` polynomial, and
scored **nothing**: 952.5 → 952.1 GB/s. That measurement was not wrong. With
the polynomial in the loop the kernel was so far from memory that removing 124
divides could not show. With the SFU in the loop the identical change is
3153 → 4414 GB/s.

The transferable form: an ablation prices a term *against whatever is currently
binding*. A term that scores zero under one bind has not been shown to be free;
it has been shown to be smaller than the current bottleneck. Every "we measured
that and it did not matter" carries an implicit "…at the time, under that
bind", and the conclusion expires the moment something upstream of it moves.
This is the cleanest instance of it in the repository: the same swap, the same
kernel, the same device, scoring 1.00× and 1.40× depending only on what else
was in the loop.

## What the two changes came to

Every size checked before it was timed, with `layernorm_rows` untouched beside
it at 5483 → 5497 GB/s as the control:

| rows × 128 × 2 planes | blocks | before, GB/s | after, GB/s |
| --- | --- | --- | --- |
| 128 | 2 | 5.2 | 12.0 |
| 512 | 8 | 20.4 | 48.9 |
| 4096 | 64 | 157.9 | 354.2 |
| 32768 | 512 | 632.4 | 2004.9 |
| 262144 | 4096 | 917.5 | 3880.0 |
| 524288 | 8192 | 950.3 | **4389.6** |

**4.6×**, on **66 registers and a 256-byte frame → 32 and zero**. The 5.8× gap
to `layernorm_rows` is now 1.25×, and that residual is not a defect: it is the
two `ex2.approx.f32` this kernel owes its own definition and a layernorm does
not, priced at exactly that by the fourth row of the ablation.

Two things that did not move, checked rather than assumed:

- **Occupancy never moved.** `cuOccupancyMaxActiveBlocksPerMultiprocessor`
  reports 6 blocks and 24 warps an SM at every rung but 128 and for every
  ablation above. Shared memory binds this kernel, and 32 registers a thread is
  nowhere near what would.
- **Nothing was put in flight, and nothing needed to be.** There is still one
  TMA load and one TMA store per CTA and both still block. The kernel runs at
  70% of the rate a pure `load_tile` → `store_tile` over the same tiles
  achieves, so the memory protocol is not what is left.

### The floor was never the launch

The kernel's 23 µs at two blocks was once read as a fixed per-launch cost the
harness pays. It is not. One session, one two-block launch, the arithmetic put
back a piece at a time:

| two-block launch | µs |
| --- | --- |
| copy only | 7.0 |
| both extra passes and every reduction | 8.9 |
| as shipped | 11.6 |
| the old rung and the old `exp2` | 23.7 |

Twelve of those twenty-three microseconds were this kernel's own arithmetic
standing at the head of a dependent chain with two blocks on a 148-SM device,
and none of them were the launch. `layernorm_rows` reaching 8.4 µs was the
first evidence the reading was wrong; the ladder above is the mechanism. The
checked benchmark, in a different session, puts the shipped floor at 10.9 µs
against 25.4 µs before the change.

## Measured and deliberately not shipped

**Folding the normalizer into the exponent.** `exp2(x − peak)/total` is
`exp2(x − peak − log2(total))`, which is four `log2` a thread and no divide at
all. It measured 4390, 4415 and 4462 GB/s across three sweep rounds against the
reciprocal form's 4363 and 4414 — the two are inside each other's run-to-run
spread. A 1% that a rerun erases does not buy putting
`kittens::reg::log2_approx`'s error inside the exponent, where the tolerance
argument would then have to carry it.

**Parking the numerator in the tile.** The third pass recomputes `exp2` rather
than keeping the second pass's result; storing it and only scaling in the third
pass is one `exp2` per element instead of two and measured **4670 GB/s**, the
fastest correct form tried here. The tile is bf16, so the numerator rounds on
the way in and the quotient rounds again: the check's worst relative error goes
**1.97e-3 → 3.44e-3 against a 3.91e-3 tolerance**. 5.8% more throughput for a
check with 12% of its headroom left is the wrong trade, and a tolerance sitting
on its own rounding floor is not a check.

**Reading the band with `ldmatrix.x4`.** Every pass here is a `load_tile`, so
this kernel is the instrument #131 used to price the load side's wide form:
one `ldmatrix.m8n8.x4` per `[16, 16]` block instead of two `.x2`s. The body is
the same body — `examples/src/softmax.rs` takes the load width as a `const`
parameter and `experiments/src/softmax_x4.rs` emits the other value as a second
entry in one bundle — so the arms are paired and adjacent in time
(`scripts/modal-run bench --case ldmatrix`). Four paired measurements at each of
three sizes: unordered at 64 and 512 blocks, and **1.1% slower** at 8192, where
all four pairs agree — 4235.6 GB/s against 4281.0, on 40 registers a thread
against 32 and the same 6 CTAs an SM. Halving the instruction count buys issue
slots and pays for them in liveness, and on this kernel that trade is very
slightly negative. The full table and what it says about the store side's `.x4`
are in `docs/library/ldst.md`.

### A register claim upstream is not a speed claim

`src/reg.rs`'s module header says of the two `exp2`s that "the measurement does
not favour" the SFU, on the evidence that pointing `exp2` at it takes
`softmax_probe_128` from 168 registers to 255 with 112 bytes of spill. That is
true, and it is a statement about *registers*. At this kernel's shape it is 32
registers and a zero frame either way, and the SFU is **2.7× faster**. That is
the fourth time in this repository that the register column has ordered time
backwards, and the note upstream should be read as the register claim it is.

## What the check can prove

A softmax has an `exp2` and a divide in it, so a `==` against a CPU reference is
not available. What `check` does instead is make every *other* kind of failure
loud.

Each row's 128 inputs are a permutation of the multiples of 1/8 below 16, so
each row's 128 outputs are distinct and the closest pair of them is
`2^(1/8) − 1` = 9.05% apart, against a tolerance of 2⁻⁸ = 0.39%. A swapped
column, a swapped row, a wrong plane, or a row normalized against another row's
sum is off by more than an order of magnitude more than rounding can account
for.

Because every row is the same 128 values in a different order, the host
reference is one 128-entry weight table at any size: the peak is always 127/8
and the denominator is always the same sum. Every output is still compared
against its own expected value, which is a permuted lookup into that table.

### The tolerance

2⁻⁸ relative. Half a bf16 ulp *relative* is not a constant: a bf16 carries
eight significand bits, so an ulp at `m·2ᵉ` is `2ᵉ⁻⁷` and half an ulp relative
is `2⁻⁸/m` — 2⁻⁹ only when `m` is near 2, a full 2⁻⁸ when it is near 1. This
seed's outputs sit at the forgiving end of that range, which is why 2⁻⁸ has
room here where `layernorm` needed 2⁻⁷.

Measured rather than argued: the worst relative error over the checked run is
**1.97e-3**, a hair past the 1.95e-3 that rounding to bf16 costs on its own,
and it is unchanged by the `exp2` swap because at 2⁻²² the instruction is
nowhere near what dominates. So the tolerance is 2.0× what a correct kernel
needs, and 23× below the 9.05% gap between two neighbouring outputs of a row.

### The seed, and why the band chooses the stride

The value at `(plane, row, column)` is `permutation(...)/8`, where

    stride      = 11 + 2 · (row / 128 % 63)
    permutation = (53·plane + 37·row + stride·column) % 128

Within a row the exponent walks every multiple of 1/8 below 16 in a stride-`s`
permutation, `s` odd and so a bijection mod 128, which makes all 128 values of
a row distinct and therefore all 128 outputs distinct. `plane` and `row` rotate
that permutation and the band chooses `s`, so a wrong plane, a wrong row or a
wrong band is a wholly different set of numbers rather than a plausible one.

The seed was originally `(53·plane + 37·row + 11·column) % 128`, and the row
axis of it was blind to exactly one thing: a whole CTA band. 37 is odd, so
`37·row` walks all 128 residues — and then repeats, because a step of 128 rows
changes it by `37·128 ≡ 0 (mod 128)`. Any row error that was a multiple of 128
produced bit-identical expected values.

Neither obvious repair works, and the reason is structural rather than
arithmetic. A seed of the form `(a·row + b·column + c) % 128` is an affine map
of the column axis, and those form a group of order 2¹³ under composition — a
2-group, whose exponent is exactly 128 (checked by enumeration). So *every*
row-affine seed has a period in `row` that divides 128, which is one band, and
no choice of multiplier can reach past it:

- `37 → 39`, or any other odd multiplier, already has the maximum period an
  affine row term can have. It changes nothing at all.
- Letting the stride follow the row, `11 + 2·row`, is worse than nothing: the
  stride's period is 64 and the offset's is 128, they turn together, and the
  whole seed still repeats every 128 rows. It also costs the row axis its
  rotation-only errors, dropping a one-row error from *every* cell wrong by
  ≥ 1.0 to every cell wrong by ≥ 0.083.

What the 2-group cannot supply is an **odd** period, so the band supplies one:
`row / 128 % 63` picks the stride, and two rows agree only if they agree both
mod 128 (the offset) and mod 63 bands (the stride). 63 is the largest odd cycle
available — the strides must be distinct mod 128 and there are only 64 odd
residues, so a longer cycle would repeat a stride and hand the blindness
straight back.

### What that buys, in numbers

A kernel whose load row and store row diverge by `δ` rows is caught this
loudly, against a tolerance of 2⁻⁸ = 0.0039. The bounds are exhaustive over
every offset shift and every one of the 63² band pairs, not sampled:

| `δ` | columns of a row wrong | each wrong by at least |
| --- | --- | --- |
| `128 ∤ δ` | ≥ 64 of 128 | 0.083 — 21× the tolerance |
| `δ = 128k`, `63 ∤ k` | ≥ 64 of 128 | 0.159 — 41× the tolerance |
| `δ = 128k`, `63 \| k` | 0 | invisible |

Over the `2 planes · 256 rows · 128 columns` = 65,536 cells of the check, the
four errors worth naming sit far above those floors: a one-row shift makes
every cell wrong, a 32-row one (a warp band) 65,280, a one-band one 64,512, and
every CTA loading band 0 — the positive control — 32,256, which is 126 of the
128 columns of every row that read the wrong band.

The plane axis is untouched and its strength does not depend on any of this: a
plane error shifts the *value* at every cell by `53·Δplane` (mod 128) whatever
the stride is, which is a factor of `2^(53/8) ≈ 99` or `2^(−75/8)` when it
wraps — a relative error of at least 0.9985, 255× the tolerance. All the stride
does is make the column rotation that carries it differ from band to band.

### The residual, stated rather than hoped for

A constant row displacement is invisible exactly when it is a multiple of
`63 · 128` = 8064 rows. That is a rule and not an accident: 63 divides no power
of two, every grid dimension and band offset in the kernel is a power of two,
and every size the project runs is a power of two — so no displacement band
arithmetic can produce is a multiple of 8064.

Two smaller ones, for the same reason. A band displacement of `k ≡ ±32 (mod 63)`
leaves half of each row's columns matching, the most any non-blind `k` leaves;
the other half are wrong by ≥ 0.996. And column 0 of every row is
`53·plane + 37·row` whatever the stride, so that one column on its own cannot
see a band error — the other 127 can.

## The column walk is under that check, and two kernels say so

Halving `CHUNK` doubles the number of chunks a row is walked in, so the errors
the walk can commit are the ones the check has to see. A table computed by the
reference's own functions is an argument and not a control — it cannot fail in
a way the reference does not also fail — so both were **built as kernels and
launched** over the check's 65,536 cells:

| deliberate error | predicted | device |
| --- | --- | --- |
| third pass reads the first chunk for every chunk | 56,960 | **56,960** |
| second pass sums the first chunk twice | 65,536 | **65,536** |

The first is the misindexing a chunked walk is the one thing that can commit,
and predicting it to the cell took two corrections that are the reason it was
worth launching rather than tabulating. The obvious model — "column `c` is
written from column `c % CHUNK`" — predicts 57,344 and is wrong by 384, because
the third pass reads and writes *the same tile*: chunk 0 is overwritten before
chunks 1..7 re-read it, so what they exponentiate is a softmax output and not
an input. Modelling that leaves 64, and the last 64 are cells the device's bf16
intermediate moves across the tolerance where an `f64` host model keeps them
inside it. With both, the model is exact.

The second was chosen as the *weakest* error a chunk walk can commit — one too
many trips round a loop, changing only the denominator — and it turns out not
to be weak: it fails every cell. That is a property of the seed and worth
naming, because it is the row argument paying off somewhere it was not designed
for. The seed spreads a row's chunk across the whole ladder at an odd stride
rather than giving it 16 adjacent values, so the first chunk always carries a
real share of the mass — the smallest shift over all 512 rows of the check is
**4.3%, eleven times the tolerance**.

The check therefore sees a misplaced *column* exactly as loudly as a misplaced
row.

## Related

- `docs/kernels/layernorm.md` — the same chunked walk, its own rung ladder, and
  the seed argument a permuted-multiset input cannot make.
- `docs/kernels/harness.md` — the clock, and the verify-then-time order.
- `experiments/README.md` — the register-count ladder this kernel's rung sweep
  narrows the search with.
