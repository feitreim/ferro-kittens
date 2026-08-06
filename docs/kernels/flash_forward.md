# `flash_forward` — causal flash attention, one query block per CTA

`examples/src/flash_forward.rs`, entry point `flash_forward`.
`O = softmax(scale · Q·Kᵀ + causal mask) · V` for one `[QUERIES, HEAD]` query
block, streamed over `key_blocks` blocks of `KEYS` keys. `blockIdx.x` selects
the query block and `blockIdx.y` the head, which is the plane coordinate of all
three panel maps.

## Status

**Runs**, checked against an f64 causal reference by `flash_forward::check` —
worst relative error **1.66e-3** over two heads and four key blocks — and
**timed**, since #81, by `bench --case flash` in `experiments/`, which checks
every size it times. Earning the first word cost this kernel its second real
bug: the epilogue's output cursor dropped the head term, so every head wrote
query block *x*'s rows of the same panel, and the first checked run failed
50,885 of 65,536 outputs (see below).
The first bug was #175's band overwrite, found by inspection; both lived here
exactly as long as nothing compared `O` to a known-good answer.

It is in the default build deliberately. A kernel behind a cargo feature is
absent from what `modal run modal_app.py::build` compiles, so the one example
that exercises the whole stack at once — both MMA layouts, the swizzled `P`
staging tile, `online_rescale`, masking, and both memory ends — was for a while
the one example CI could not catch a regression in.

Nothing structural is missing. `mm_abt` and `mm_ab` are exactly the two MMAs
this kernel issues, in the layouts and under the accumulator discipline it
issues them: both start a band fresh, which is the entry point's *name* rather
than an argument this file has to get right. The whole register-side body —
drain, mask, scale, row-reduce, correct, accumulate, restage — is library API
line for line, with no index arithmetic left in the file, and so are both memory
ends. What still reaches past the library is `thread`'s `sync_threads` and block
indices, the allocator's column count, and the kernel ABI itself, which no tile
library replaces. None of it is register-side.

## The shared plan

`SHARED_BYTES` is 147 532 — 144 KiB, and 4.5× what `softmax` asks for. By
component:

| component | shape | bytes | share |
| --- | --- | ---: | ---: |
| `Q` tile | `[128, 128]` bf16 | 32768 | 22% |
| `K` ring | 3 × `[64, 128]` bf16 | 49152 | 33% |
| `V` ring | 3 × `[64, 128]` bf16 | 49152 | 33% |
| `P` staging | `[128, 64]` bf16 | 16384 | 11% |
| barriers + TMEM word | 9 × 8 B + 4 B | 76 | — |
| **total** | | **147532** | |

The measurements below were taken at 147 536, before the plan reserved the
`tcgen05.alloc` staging word as the four-byte `u32` it is. Four bytes out of
144 KiB decides nothing here, and the numbers are left as they were taken.

Two thirds of the plan is the pipeline — `STAGES` deep over both `K` and `V` at
16 384 B a stage a side. Dropping to two stages would save 32 768 B, and **that
is the one lever this kernel has**: it is the only change on the table large
enough to put the plan under the step where a second CTA is admitted. It used
to say here that it buys nothing, on #70's reading that shared memory was not
what capped this kernel — see the two sections below, which is where that went
wrong and what replaced it. Nobody has priced the shorter ring yet, and the
harness that would price it (a launcher, a CPU reference, and a row on the
clock) exists only since #182 and this file's `flash` bench case.

## The `0` blocks/SM this used to report was the query, not the plan

The occupancy table printed `0` blocks/SM for this kernel, which reads like a
plan that does not fit and was not one. A block gets 48 KiB of dynamic shared
memory without asking; past that the *function* must be opted in, and
`cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 0 for a function nobody
opted in for in exactly the same way it answers 0 for tiles too big to fit.

On a B200 the same function at the same 147 536 B goes **0 → 1** across a single
`kittens::launch::admit_shared_plan` call, with the device's own ceiling at
232 448 B — **84 912 B of headroom under a plan that was never too large**.

## The `1` is the shared plan, and every query that said otherwise was pinned

Swept on a freshly loaded function per probe, this kernel *answers* **1 block/SM
at 147 536 B, at 73 792, at 32 800, and at 0** — flat, and flat across block
widths of 32, 64, 128 and 256, where `softmax` goes 28/14/7/3 and `layernorm`
8/4/2/1. That was read as a per-CTA resource ignoring both shared memory and
warp count, and the only such resource this kernel holds is tensor memory. A
control agreed: `device-tests`' `tmem occupancy ladder`, a nine-register kernel
whose entire content is `tcgen05.alloc`, answers 1 at every column count and
every block width against 32/32/16/8 for the byte-identical kernel with the
allocator deleted.

**All of that is a fact about `cuOccupancyMaxActiveBlocksPerMultiprocessor`,
and #84 measured the hardware instead.** `device-tests`' `tmem residency census`
counts CTAs an SM directly — every CTA writes down its `%smid` and timestamps
both ends of its allocation off `%globaltimer` — and what it counts is
`min(512 / columns, shared per SM / plan)`, exactly, at every rung. A CTA is
charged the columns it asks for and not the SM's whole tensor memory:

| resource | this kernel's value | CTAs an SM it admits |
| --- | ---: | ---: |
| TMEM columns | 256 | 2 |
| dynamic shared | 147 536 B | **1** |
| both together | | **1** (counted) |

So the 1 CTA/SM is **real**, and shared memory alone causes it — the rung
carrying this plan with *no allocator in the kernel at all* counts 1 too, which
is what isolates it. #70 queried this kernel's occupancy at 147 536, 73 792,
32 800 and 0 bytes, got 1 at all four, and concluded the plan was not the lever;
every one of those four answers was the allocator pinning the query.

**So the lever is the shared plan**, which is the one thing that was ruled out,
and the two remedies that were on the table are not. Shrinking the tensor-memory
allocation buys nothing — 256 columns already admit two CTAs and shared memory
caps the kernel below that. Warps per CTA is not a constant to raise: 4 warps is
`QUERIES / 32`, one warp per 32 accumulator rows, so more warps means a
different decomposition — a producer/consumer split with warps dedicated to the
TMA and the MMA issue — and that is a design to price rather than a change to
make blind.

### Where the step actually is

Two CTAs an SM need a plan under half of the SM's 233 472 B, which is 116 736 —
*arithmetic*, and #85 says so. The census bracketed it between 73 768 B (counts
2) and 147 536 B (counts 1) and did not locate it, and shared-memory allocation
granularity or a per-CTA reservation could move it. Four rungs 1 KiB apart
around the arithmetic value now locate it; `THRESHOLD_PLANS` in
`device-tests/src/tmem_residency.rs` is the ladder and the census prints the
interval it lands in.

On a B200, counted (#81's run of the census):

| declared plan | CTAs an SM |
| ---: | ---: |
| 114 688 B (half − 2 KiB) | 2 |
| **115 712 B (half − 1 KiB)** | **2** |
| 116 736 B (half, #85's arithmetic) | **1** |
| 117 760 B (half + 1 KiB) | 1 |

**The step is in [115 712, 116 736), so #85's arithmetic is a kilobyte too
generous** — at exactly half the SM a second CTA is *not* admitted. The
arithmetic ignores what the driver keeps per CTA, and 2 × (115 712 + 1024) is
233 472 exactly, which is the reading the rungs support and do not prove: a
1 KiB per-CTA reservation, which is what
`CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK` reports on every
architecture that publishes it. Locating it inside that
kilobyte would take another ladder and would not change what this kernel has to
do.

So the ring has to give up **31 824 B**, not the 30 800 B the halving says —
`STAGES` from 3 to 2 is 32 768 B and still clears it, with 944 B to spare.

Nothing here shrinks the ring. That is the follow-up #85 asks for, and what it
was waiting on — a launcher, a CPU reference and a row on the clock — exists
now.

## Tensor memory: 192 is not a legal allocation

The scores `[128, KEYS]` and the output `[128, HEAD]` share one allocation.
`KEYS + HEAD` is 192, and **192 is not a legal `tcgen05.alloc`**: the operand
must be a power of two in `[32, 512]` (cuda-oxide spells the set out as
"32, 64, 128, 256, 512" on `TmemGuard`). This kernel asked for 192 from the day
it was written and nothing caught it, because it has no launcher — the column
count is an argument to an instruction, so no type and no `const` assertion was
ever in a position to see it.

A `const` parameter is. Since #128 `alloc_block::<COLUMNS>` carries the rule
itself, in a `const { assert!(..) }` that fires at codegen, and this file keeps
only the half that is its own: that 256 covers `KEYS + HEAD`. The hand-written
legality assert it grew after the audit is gone, because the entry point it was
standing in front of now refuses the argument.

Rounding up to 256 costs **nothing here**, and the reason this file used to give
was wrong. It is not that the driver charges a CTA the SM's entire tensor memory
the moment it touches the allocator: #84 counts `512 / columns` CTAs holding at
once at every legal count, so these 256 columns admit two. They are free because
the shared plan above admits one before the columns are consulted. Bring the
plan under the step and the two terms meet: 256 columns admit exactly the two
CTAs the smaller plan would, so the allocation is the right size either way and
a *third* CTA would need both to move — and 128 columns do not cover
`KEYS + HEAD`.

The `dealloc_block` at the end is not optional. tcgen05 allocations are not
scoped to the CTA the way shared memory is — a kernel that exits holding them is
a `CUDA_ERROR_TENSOR_MEMORY_LEAK`, and the next CTA scheduled onto the SM is the
one that pays. This kernel had no `dealloc` at all until its allocator was
audited, and nothing could have caught it: `device-tests`' `repeated launch` case
is the standing guard for exactly this class, and a kernel with no launcher never
reaches a guard that works by launching twice.

## Register-side shape

- `ScoreBand` is `[32, KEYS]` — one warp's band of the score tile. Naming it out
  of tree needed `FragmentLayout` to become a blanket impl over the row and
  column extents; the unsatisfied layout bound on that band was also what kept
  `make_causal` off the library's surface.
- `PTile` is exactly one swizzle atom wide, which is what the `stmatrix` store
  path needs.
- `KTile` and `VTile` no longer exist. Their only remaining use was the
  `KTile::BYTES + VTile::BYTES` this kernel charged its stage barrier, and the
  loads hand that back themselves now; a stage's tile is `k_ring.tile(block)`,
  which is where its shape already was.
- `LOG2E` is the library constant rather than a literal. `1.442_695` rounds to
  the same fp32 bit pattern (`0x3fb8aa3b`), so this is not a numerics change,
  but `clippy::approx_constant` is deny-by-default and the literal was a build
  error nobody saw while the file sat behind a cargo feature. It is also what
  `reg::Exp` scales by, so the kernel and the op name one constant.

### `mm_ab`, not `mma`

The running output lives in registers for the whole loop, so tensor memory only
ever holds *this block's* `P·V`. Both MMAs start their accumulator band fresh,
and that is a fact about which entry point is called rather than a `false`
argument a reader has to check.

### In place versus by value

At this kernel's `[32, 128]` accumulator the in-place *accumulate* is **not**
cheaper than the by-value one: `out_acc = out_acc.add(contribution)` costs the
same 168 registers, because that call site rebinds and its input is already
dead. It costs **512 more bytes of stack frame** — 168 regs / 2560 B for
`add_assign` against 168 / 3072 B for the by-value form. The reason to write the
accumulate in place is that the accumulator's input *is* its output.

The place an in-place form is worth registers is the **rescale**: `row_map::<Mul>`
against a hand-written `scale_rows` is **87 registers/thread**, and
`row_map_assign::<Mul>` reaches the hand-written number exactly.

### 255 registers, and why this kernel can spend them (#184)

`make_causal_at` used to be a rolled walk, which homed the `[32, 64]` score
band to a `.local` frame — for the whole loop, not for the diagonal block. It
unrolls now, the band comes back into registers, and this kernel is the one
that pays for it:

| | regs | ptxas spill st/ld | stack | `st.local` | `ld.local` | driver blocks/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 165 | 0 / 0 | 1824 B | 774 | 451 | 1 |
| after | **255** | **0 / 0** | 1568 B | 690 | 383 | **1** |

255 is not a spill and it is not an occupancy step. It is not a spill because
`ptxas` says so in the columns beside it. It is not a step because this
kernel's **own shared plan is what holds it to one CTA per SM**: 147 532 B
against the SM's 227 KiB admits exactly one, and the driver's
`max_active_blocks_per_multiprocessor` answers 1 at both register counts.
Registers would have to reach past two CTAs' worth to matter here and there is
no second CTA to lose. `device-tests`' occupancy ladder makes the same point
from the other side — the 1 CTA/SM is shared memory's, not the tcgen05
allocator's.

What it buys, on a B200, checked before timed, five warm-ups and the minimum
of thirty:

| | before | after | |
| --- | ---: | ---: | ---: |
| 1024 queries × 2 heads | 0.2135 ms | 0.1828 ms | **1.168x** |
| 2048 queries × 4 heads | 0.3966 ms | 0.3347 ms | **1.185x** |

So the register column ordered time backwards again, for the sixth time in
this tree (#47, #63, #67, #76, #94): +90 registers, −15% wall clock, because
the frame it replaced was being written and read in the innermost loop. The
kernel that *would* read this differently is one with room to spare in shared
memory, where the third CTA is real — which is why #184 is recorded here and
not only in `docs/library/reg.md`.

**The 255 is `experiments`' copy now, and `examples`' reads 168** — same source
file, two crates, one compiled through `#[path]` since #81 put this kernel on
the clock, and the frame (1568 B), the `.local` traffic (702 stores, 395 loads)
and the spill columns (zero) are identical between them. Crate composition alone
moving a register count is not new — `groupnorm_tile` went 168 → 236 across the
same split and crossed an occupancy step for it, which is why it is gated
(`modal_app.py`'s `GATED_KERNELS`). It costs nothing here for the reason above —
one CTA an SM either way — and it is recorded because the two rows sit side by
side in `regcount` and an unexplained gap between them is exactly what that
table is for.

### The epilogue

fp32 straight out of registers through `kittens::global::store_rows`, with no
shared tile in the way. The round trip it used to owe — pack to bf16,
`stmatrix`, TMA out — was a precision loss it never asked for, and the
allocation was one this kernel's shared plan could not spare either.

The output is packed `[heads, _, HEAD]` — the panel axis of the three input
maps, spelled on the output's row axis — so the cursor's stride is the band's
own width and its column base is zero: the degenerate case of the stride the
GEMM's epilogue needs, spelled the same way.

The head term of the cursor's row was **missing until the first checked run**.
The TMA maps take the head as a plane coordinate and cannot get it wrong; this
cursor takes the address it is handed, and every head wrote query block *x*'s
rows of panel zero — head 1's panel stayed zeroed and head 0's held whichever
CTA wrote last. The failure was 50,885 of 65,536 outputs at two heads, and it
is why [the check](#the-check) runs two: at one head the collision has nobody
to collide with and the kernel scores a pass it has not earned.

### The exponential is the polynomial, and that is measured (#81)

The inner loop calls `exp2` — `kittens::reg::exp2_approx`, a clamp, a
shift-trick split and a degree-3 minimax — once per element of every key block.
`softmax` calls the SFU `exp2_hw` instead and is **2.7×** faster for it (#76),
which made this kernel the obvious next place to spend the same change: it
exponentiates more elements per byte of traffic than anything else here.

It buys nothing. Timed through `bench --case flash`, each arm checked before it
was timed, and the polynomial run twice so the difference carries its arms' own
repeatability (#122):

| sequence × heads | CTAs | `exp2` | `exp2` again | `exp2_hw` |
| --- | ---: | ---: | ---: | ---: |
| 1024 × 2 | 16 | 0.1789 ms | 0.1741 ms | 0.1820 ms |
| 2048 × 4 | 64 | 0.3336 ms | 0.3254 ms | 0.3343 ms |
| **2048 × 16** | 256 | **0.6513 ms** | **0.6492 ms** | **0.6756 ms** |

The last row is the one with a denominator: 256 CTAs is the first size past a
full wave, it repeats to **0.3%** across two runs of the same arm, and the SFU
is **3.9% slower** there. The other two rows put 16 and 64 CTAs on 148 SMs,
repeat to 2.7%, and say only that nothing large happened either way. The
register table does not distinguish the arms at all — 168 registers in
`examples`, 255 in `experiments`, 1568 B of frame and 702/395 `.local`
stores/loads, identical for both spellings — so a swap made on the register
column would have been made blind in either direction.

So this kernel keeps the polynomial, and the swap is recorded rather than
shipped. The reading is that a change to the exponential is invisible behind the
MMA and TMA pipeline at four warps and one CTA an SM: `softmax` is a bandwidth
loop whose whole arithmetic is the exponential, and this is not. That is a
reading and not a measurement — what would settle it is a warp-count sweep of
one exponentiating loop, which nothing here has, and it is also what the
`ex2.approx` serialization claim would need.

The numerics half is settled: the check measures **1.66e-3** with the polynomial
and **1.73e-3** with the SFU, against a 3.91e-3 tolerance. Neither spelling is
what sets the error, and neither would have been caught by the register column.

### Causal masking

`make_causal_at` takes the query base and the key base separately rather than
their difference. The difference is negative for every band above the diagonal,
which is most of them, and computing `query_base - key_base` in `u32` at that
call site would wrap and mask nothing.

## The check

The reference is the straightforward stable softmax in f64 over bf16-rounded
inputs — scores against keys `0..=row`, subtract the row max, exponentiate,
weight `V`, divide by the sum. The kernel's `exp2`/`LOG2E` folding and its
online rescale are ways of computing exactly this, and the reference deliberately
is neither.

### The seed

Dimension 0 of `Q` and `K` is a carrier pair — `Q` holds `2`, `K` holds
`row / 8` — that puts a **ramp on the key axis**, so later key blocks score
higher and the running max moves at nearly every block. That is the online
rescale being *exercised* rather than merely compiled: against a flat seed, a
kernel that never rescaled its accumulator would be wrong by a factor the
tolerance could miss. The ramp also makes the keys above the diagonal the
best-scored ones a broken mask could admit, so masking failures are large, not
subtle. The other 127 dimensions are zero-centred lattice noise permuted per
`(tensor, head, row)`, which spreads each row's weights over many keys, and
every value is bf16-exact by construction.

`V` sits in `[1, 3)`: the output is a convex combination of its values, so
every output sits in `[1, 3)` too and a relative error is never taken against
a cancellation the reference does not also perform.

The check runs **four key blocks** — one past `STAGES`, so the ring wraps and
`free`'s recycled phase is waited on for real — and **two heads**, which is
what makes a head plane read or written in the wrong place a wrong number
rather than a coincidence. The epilogue bug above is the check-of-the-check:
run at one head, it would have passed.

### The tolerance

2⁻⁸ relative. The floor is `P`'s trip to bf16: the probabilities cross to
shared as the `stmatrix`-staged `A` operand of `O = P·V`, and rounding the
weights of a convex combination moves the result by about a weight ulp —
scale-free in the sequence, since the weights sum to one. The measured worst
over both heads is **1.66e-3 = 0.85 × 2⁻⁹**, under half a bf16 ulp at the
outputs' `[2, 4)` end and the same neighbourhood `softmax`'s floor lives in
(its `ex2` polynomial is in play here too, folded into the same number). 2⁻⁸
is one doubling above the floor, the repository's usual headroom, arrived at
from this kernel's own measurement. The SFU `exp2_hw` moves it to 1.73e-3 and
nothing else (#81), which is the same statement from the other side: the
exponential is not what the tolerance is made of.

## The clock

`experiments/src/bench.rs` carries the `flash` case, at `sequence × HEAD ×
heads`. It was the last entry in that file's `SKIPPED` list — first for having
no reference, then for owing a denominator — and the denominator is the flops
the kernel *issues*: both MMAs run over every key block of every query block, so
one head of `sequence` queries costs `4 · sequence² · HEAD`. The kernel does not
skip a key block it will mask away, so a causal-optimal kernel would do a little
under half of that, and the table quotes what is on the clock rather than what
is useful.

Three sizes, and the ceiling on them is the *host*: the f64 reference every
timed run is checked against is `heads · sequence² · HEAD` multiply-adds, so a
sweep that kept quadrupling would spend its minutes on the CPU. 1024 × 2 and
2048 × 4 are #184's operating points, kept so those numbers and these are the
same rows; 2048 × 16 is 256 CTAs, the first size past a full wave and the only
one of the three whose throughput column means anything.
