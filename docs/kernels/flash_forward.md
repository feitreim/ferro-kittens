# `flash_forward` — causal flash attention, one query block per CTA

`examples/src/flash_forward.rs`, entry point `flash_forward`.
`O = softmax(scale · Q·Kᵀ + causal mask) · V` for one `[QUERIES, HEAD]` query
block, streamed over `key_blocks` blocks of `KEYS` keys. `blockIdx.x` selects
the query block and `blockIdx.y` the head, which is the plane coordinate of all
three panel maps.

## Status

**Compiles**, and is in the default build. It has no launcher and no CPU
reference, which is the only thing between it and *runs* — "runs against an
exact reference" is a claim that has to be earned, and for the two kernels that
make it, earning it was its own piece of work.

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
16 384 B a stage a side. Dropping to two stages would save 32 768 B, **and it
buys nothing.**

## The `0` blocks/SM this used to report was the query, not the plan

The occupancy table printed `0` blocks/SM for this kernel, which reads like a
plan that does not fit and was not one. A block gets 48 KiB of dynamic shared
memory without asking; past that the *function* must be opted in, and
`cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 0 for a function nobody
opted in for in exactly the same way it answers 0 for tiles too big to fit.

On a B200 the same function at the same 147 536 B goes **0 → 1** across a single
`kittens::launch::admit_shared_plan` call, with the device's own ceiling at
232 448 B — **84 912 B of headroom under a plan that was never too large**.

## The `1` is tcgen05, and it is not this kernel's fault

Swept on a freshly loaded function per probe, this kernel answers **1 block/SM
at 147 536 B, at 73 792, at 32 800, and at 0** — flat, and flat across block
widths of 32, 64, 128 and 256, where `softmax` goes 28/14/7/3 and `layernorm`
8/4/2/1. A ceiling that ignores both shared memory and warp count is a *per-CTA*
resource, and the one this kernel holds that neither control does is tensor
memory.

That was a correlation over a sample of one until the control was built:
`device-tests`' `tmem occupancy ladder`, a nine-register kernel whose entire
content is `tcgen05.alloc`, against the byte-identical kernel with the allocator
deleted. On a B200, blocks/SM at zero dynamic shared:

| rung | 32 | 64 | 128 | 256 threads |
| --- | --- | --- | --- | --- |
| no tcgen05 | 32 | 32 | 16 | 8 |
| `alloc` 32 columns | 1 | 1 | 1 | 1 |
| `alloc` 64 / 128 / 192 / 256 / 512 | 1 | 1 | 1 | 1 |

**The guess was right and its obvious remedy is wrong.** A CTA that touches the
allocator is charged the SM's *whole* tensor memory — the smallest allocation
the ISA defines costs exactly what the largest does. The 192 rung takes its
column count as a *kernel argument*, so the driver is not reading a number
`ptxas` recorded; it is pricing the allocator itself. And this kernel confirms it
directly: cut to 32 columns and re-queried, it still answers 1.

So none of the three levers anyone had in hand is one. Shrinking the rings buys
nothing, shrinking the tensor-memory allocation buys nothing, and there is no
cluster shape to blame — `required_cluster_dimensions` is `None` on every rung of
the ladder, and tcgen05 implies no `cta_group::2`.

**What is left is warps per CTA.** At 1 block/SM the CTA's own width is the
entire occupancy of the SM, and this kernel is 128 threads — 4 warps where its
own `max_threads_per_block` is 512 and where the control above holds 32. That is
not a tuning constant to raise: 4 warps is `QUERIES / 32`, one warp per 32
accumulator rows, so more warps means a different decomposition — a
producer/consumer split with warps dedicated to the TMA and the MMA issue, which
is what the shape of a tcgen05 kernel is *for*. With no launcher and no CPU
reference, that is a design to price rather than a change to make blind.

## Tensor memory: 192 is not a legal allocation

The scores `[128, KEYS]` and the output `[128, HEAD]` share one allocation.
`KEYS + HEAD` is 192, and **192 is not a legal `tcgen05.alloc`**: the operand
must be a power of two in `[32, 512]` (cuda-oxide spells the set out as
"32, 64, 128, 256, 512" on `TmemGuard`). This kernel asked for 192 from the day
it was written and nothing caught it, because it has no launcher — the column
count is an argument to an instruction, so no type and no `const` assertion was
ever in a position to see it. The one in the file now is.

Rounding up to 256 costs **nothing**, which is not obvious: the driver charges a
CTA the SM's entire tensor memory the moment it touches the allocator, at 32
columns exactly as at 512, so the 64 columns past `KEYS + HEAD` are free.

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

### The epilogue

fp32 straight out of registers through `kittens::global::store_rows`, with no
shared tile in the way. The round trip it used to owe — pack to bf16,
`stmatrix`, TMA out — was a precision loss it never asked for, and the
allocation was one this kernel's shared plan could not spare either.

The output is packed `[_, HEAD]`, so the cursor's stride is the band's own width
and its column base is zero: the degenerate case of the stride the GEMM's
epilogue needs, spelled the same way.

### Causal masking

`make_causal_at` takes the query base and the key base separately rather than
their difference. The difference is negative for every band above the diagonal,
which is most of them, and computing `query_base - key_base` in `u32` at that
call site would wrap and mask nothing.
