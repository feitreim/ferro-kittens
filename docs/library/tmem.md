# `tmem` — TMEM accumulator views

Design notes and measurements behind `src/tmem.rs`. The source carries the
hardware contract a caller owes; this file carries the numbers those choices
rest on and the alternatives that lost.

## How many columns you ask for is how many CTAs fit on the SM

An SM has **512 columns** of tensor memory and the allocator divides them.
`512 / columns` CTAs hold allocations on one SM at the same time:

| columns | CTAs an SM | who asks for it |
| ---: | ---: | --- |
| 32 | 16 | the allocator's smallest unit |
| 64 | 8 | |
| 128 | 4 | |
| 256 | 2 | `gemm` and `flash_forward` today |
| 512 | 1 | the whole SM |

Tensor memory is one of two per-CTA resources an SM divides, and residency is
**whichever of the two is tighter**:

    min(512 / columns, shared per SM / shared plan)

Read the table as a ceiling another resource may be sitting under.

For `gemm` the table is the term that binds. The pair tile is `[256, 256]`, so
the kernel asks `alloc_cluster` for `BLOCK_N = 256` columns — 2 CTAs — while its
shared plan (114 816 B for the staged epilogue that ships, against the 116 736 B
half of an SM) admits the same 2 with room left over. `examples/src/gemm.rs`'s
`staged_ctas_per_sm` is that `min` written out and says which term is tight.

For `flash_forward` shared memory alone is the cap, at 1 CTA where its 256
columns would allow 2.

### The measurement

`device-tests`' `tmem residency census`, on a B200, counts CTAs rather than
asking about them: every CTA writes down its `%smid` and timestamps both ends of
its allocation off `%globaltimer`, and the host sweeps the intervals for the most
that were ever open at once on one SM. The counted figure is `512 / columns`
exactly, at every legal column count.

### `cta_group::2` is charged the same way

A pair does *not* split the allocation. Each rank is charged its full column
count against its own SM's 512, so `alloc_cluster` at 128 columns leaves room for
four CTAs an SM exactly as `alloc_block` at 128 does. That a cluster allocation
might be a half-share was the leading hypothesis for why `gemm` and
`flash_forward` differ in residency, and it is refuted. The pair's two CTAs are
on two different SMs in any case.

## Do not use the occupancy query to predict this

`cuOccupancyMaxActiveBlocksPerMultiprocessor` returns **1** for any kernel whose
code contains a `tcgen05.alloc` — at every column count, at every block width,
and even for a kernel that allocates and *releases* before doing any work at all.
It is not tracking a resource the CTA holds; it is reacting to the instruction
being present. `cuOccupancyMaxActiveClusters` does the same for a
`#[cluster_launch]` kernel.

Both queries are accurate on kernels with no allocator in them — 32 and 8 CTAs an
SM respectively, matched by the census to the CTA — which is what makes the 1
easy to believe.

Two conclusions were drawn off that 1 and both were wrong:

- That the allocation size could not be a lever and warps per CTA was all that
  was left. A `gemm` was then measured losing **2.07×** to a one-CTA-per-SM grid
  cap, which is impossible if one CTA per SM were the ceiling; that kernel's true
  residency bisected to **3**, a figure the census reproduces exactly at the same
  envelope, from timestamps rather than from a throughput curve.
- That shared memory was not `flash_forward`'s lever. That came from querying
  occupancy at 147 536 B, 73 792, 32 800 and zero and getting 1 block/SM at every
  one of them. Every one of those answers was the allocator pinning the query at
  1. Shared memory is in fact the *only* thing capping that kernel.

So **shrinking an allocation to the next legal power of two doubles the CTAs
tensor memory will hold**. It is a real lever, and it is only worth pulling while
tensor memory is the tighter of the two resources — which for both kernels here
it currently is not. The thing to shrink is the shared plan.

## The extra CTA, and what a blocking allocator looks like

The census counts one *more* CTA resident on the SM than is holding columns — 3
resident against 2 holders at 256 columns, 5 against 4 at 128. That CTA is
admitted and parked inside `tcgen05.alloc`, which blocks until the columns it
asked for come free. It costs a CTA slot and its warps make no progress, so it is
neither absent nor working. Nothing that reads occupancy off a throughput curve
can see the difference, which is why the census timestamps `entered` and
`allocated` separately.

## The allocation permit

`alloc_block` relinquishes the CTA's allocation permit beside the *alloc*, not
beside the dealloc. The permit is the right to call the allocator again, which is
a different thing from the columns; this library allocates once per CTA, and
holding the right to allocate for the rest of a CTA's life is what the
instruction exists to avoid.

PTX requires the permit back before the CTA exits and does not say what happens
otherwise. A deadlock was predicted — the next CTA on that SM stuck in
`tcgen05.alloc` — and on a B200 that is **not** observable: with the relinquish
removed, 4736 CTAs over three launches, each taking all 512 columns, all
completed, and a CTA sitting on an unrelinquished permit for milliseconds did not
delay a co-resident CTA's own allocation by a measurable amount.

So the relinquish is conformance, not a bug fix. The hang that motivated it
belongs to the other half of that defect: a `cta_group::2` dealloc issued by one
CTA of the pair leaks *columns*, and columns do not come back.

## Why both peers drive the cluster allocator

The three `cta_group::2` allocator instructions all say the same thing about who
issues them — *one full warp in each peer CTA* — so both CTAs allocate, both
relinquish, both later deallocate, and each reads the address out of its own
staging word, because the collective writes one into each.

A peer that instead maps the leader's word over distributed shared memory gets
the right address and still leaves the pair's allocator half-driven. That is what
hung a GEMM's second launch. `alloc_cluster`'s body is that kernel's fix lifted
unchanged, because it is the version that ran on silicon.

The `cluster_sync` inside it is what publishes the staging words across the pair;
a caller that stages barriers for its peer before allocating gets that
publication out of the same sync.

## Why the dealloc fence and sync stay two lines at the call site

Retiring a `cta_group::2` accumulator's reads is

```rust,ignore
tcgen05_fence_before_thread_sync();
cluster::cluster_sync();
dealloc_cluster(accumulator.raw(), COLUMNS);
```

and welding the first two into a `kittens` entry point was rejected. The fence is
tcgen05's and would belong there; the *rendezvous* is a scope choice, and folding
`cluster_sync` in would add a fifth spelling of scope beside the four §7.1 of
`GAPS.md` counts — `epilogue::Scope`, `Job::RANKS`, `block_reduce`'s `WARPS`,
`store_shared_rows`' `THREADS` — at the point where unifying them is the open
work. It would also close no leak: a kernel calling `cluster_sync` here already
imports `cuda_device::cluster` for `block_rank` in its `attach`, which all three
call sites in tree do.

`pipeline::run` performs exactly this pair at every item boundary, so a job whose
accumulator dies with the item loop has already had it. Both GEMMs rendezvous
once *outside* `run` — before the dealloc, for the cluster that took no items at
all — and write the two lines out.

## `.x8` and the register list's order

`interleave_x8` asserts that a `tcgen05.ld.16x256b.x8`'s 32 registers are
**repeat-major**: repeat `r` is exactly what a `.x1` at `column + 8r` would have
returned, so `interleave`'s low/high pair for block `j` is repeats `2j` and
`2j + 1`, i.e. register `8j + 4*(value/2) + 2*slot + value%2`.

That ordering is a claim about silicon, not an inference from the ISA text.
`device-tests`' `ldtm x8 map` case is what holds it: it drains one accumulator
both ways and requires the tiles to be equal, so an `.x8` whose registers arrive
in another order fails there rather than in a GEMM's tolerance.

### What `.x8` bought and what it costs

`.x8` returns 32 f32 a thread where `.x1` returns 4 — same bytes, same map, one
eighth of the issues, and one eighth of the *waits*, which is the larger half of
it. `fragment` waits after each of its two loads because the registers it waits
on are its return value, so the `.x1` drain never has more than one load in
flight and pays the full TMEM latency per four values.

What it costs is liveness: 32 f32 arrive at once and all four blocks are live
until the caller consumes them, where the `.x1` path lets the compiler fuse a
single eight-value fragment through to the store.

## Batching the issues behind one wait

`tile_x8_batched` puts every issue of a band in flight before the one wait.
`tcgen05.wait::ld` waits for *prior* loads, plural, so one wait retires the band
however many issues it took.

**Registers.** At two issues it costs nothing: the band holds all `32 * ISSUES`
f32 a thread simultaneously either way, so batching only moves where the wait
sits, and `gemm_sol`'s M256 drain censuses at **96 registers and a 256-byte
frame** both ways. At four issues it does cost: **176 registers and 512 bytes of
frame**. What it needs is register *file*, and at one CTA per SM there is a lot
of it — 192 threads at 255 registers is 48 960 of an SM's 65 536.

**Throughput.** Two issues behind one wait is **+0.9%** of `gemm_sol`'s launch at
8192³ and nothing at 4096³; four issues is a loss. `gemm_sol_ablate`'s doubling
ladder says why: *doubling* the LDTM count and its waits in that drain costs
zero, so the waits `.x8` was worth removing are already covered by the other
three epilogue warps. Batching is the tail of that lever and not a new one.

**`ISSUE_LIMIT` is 4** because of the register file: each `.x8` is 32 f32 a
thread, so four is 128 — the widest band a warp can drain in one pass at all,
which is the same 128-column limit `examples/`' GEMMs derive for themselves from
the 255-register architectural ceiling.

**`ISSUES` is stated rather than derived.** `(M / 16) * (N / 64)` is a generic
const expression and this crate is on stable; the const assert is what keeps a
caller from stating it wrong.

**The issues are spelled out rather than looped over**, and that is not style.
Written as a loop over an `[_; ISSUES]` array, LLVM unrolls it at two issues and
does not at four: the array then needs a runtime index, lands in local memory,
and the arm censuses as **one** `tcgen05.ld` with a **1024-byte stack frame** —
four loads in flight through L1, which is not the thing being measured. Four
literal indices cannot do that.

## `.pack::16b` is not a cheaper epilogue

`.pack::16b` packs the 16-bit elements of two consecutive tensor-memory columns
into one 32-bit register, so it returns the same 32 registers `fragments_x8` does
off **twice** the columns — 64 b16 a thread rather than 32 f32, out of the same
1024 bits of TMEM. That is the whole of what it is: a *reinterpretation*, not a
conversion.

So there is no reading of it that makes it a cheaper epilogue. An fp32
accumulator's columns hold floats; packing two of their low halves yields the
mantissa halves of two numbers, which is not a narrowed value of anything. The
same conclusion was once argued from the register count, which is the weaker
claim — the register count is equal because the *bits* are equal, and equal bits
is exactly why no convert was performed.

What it is good for is **pricing** the convert. Substituted into a drain it holds
the TMEM traffic, the register count, the `stmatrix` count and the stores, and
takes the `cvt` column to zero; the launch difference is then the convert's cost
and nothing else. `experiments/`' `pack16` rung is that substitution, and it is
on no correctness gate because it cannot be.

## A warp's lanes are its quadrant, and two warpgroups share them

Every drain and store here is a warp's own, and until #192 "a warp's own" was
only ever spelled `32 * warp_id()`. That spelling is not the hardware's rule and
it agrees with the hardware's rule for exactly as long as a launch has one
warpgroup. The rule is `warp_id() % 4` — `tmem::warp_lanes()` — and the two
differ the moment a second warpgroup exists.

oxide-train#94 is what made the difference matter. Its flash backward is
pass-bound (register pass 38–43% of a key-tile visit, MMA issue 42–49%, every
wait under 100 ticks out of 4 300), and its remedy is to put a second warpgroup
on the same accumulator's lanes. The library's contract said that was the owning
warp's; the ISA suggested otherwise; a wrong guess is silently wrong gradients.

### Measured

`device-tests`' `tmem across warpgroups` — one accumulator, 256 threads, and a
seed whose value at `(lane, column)` is the integer `lane * 512 + column + 1`, so
a read that lands somewhere else decodes to *where*, and the case reports the
offset rather than the word "wrong". B200, driver 580.95.05, `sm_100a`.

| what varies | row | result |
|---|---|---|
| control | warpgroup 0 reads its own quadrant | every cell |
| **who reads** | warpgroup 1 reads its opposite number's | **every cell** |
| | both warpgroups read the same lanes at once | every cell |
| **fences** | no `fence::after_thread_sync` | every cell |
| | no `fence::before_thread_sync` | every cell |
| | neither fence — `store_wait` and a block barrier | every cell |
| **shape** | #94's split: half the columns each, same lanes | every cell |
| | the backward drain, `tile_x8` at `[32, 128]` | every cell |
| | the forward's `rescale_half`: foreign read *and* store | every cell |
| | an accumulator the **MMA** wrote, not `tcgen05.st` | every cell |
| **the bound** | warpgroup 1 reads a quadrant that is not its own | **every value aliased** |
| | warp 0 reads the block at lane 32 | **every value aliased** |

So: **yes**, and the contract that replaces the ownership language is one
sentence — *a warp addresses the 32 lanes at `warp_lanes()`, which warpgroup it
is in does not matter, and it addresses no others.*

### What the bound does when you cross it

Not a fault, and not garbage. **The lane it reaches is the one at the same offset
inside its own quadrant** — the quadrant bits of the address are dropped and the
low five kept:

```
warp 4 (quadrant 0) asked for lane 32 -> read lane 0
                    asked for lane 40 -> read lane 8
warp 7 (quadrant 3) asked for lane  0 -> read lane 96
warp 0 (quadrant 0) asked for lane 32 -> read lane 0
```

Every one of the 8192 values warpgroup 1 read under a rotated quadrant decoded to
a real cell at that offset, and the last line is inside one warpgroup, so this is
a property of the *warp* and not of the warpgroup boundary.

That is the worst shape a wrong answer could have taken. There is no fault to
catch, no NaN to notice, and the values are plausible accumulator values that
belong to somebody else — a kernel would simply compute the wrong thing. It is
why `TmemTile::tile`, `tile_x8`, `tile_x8_batched` and `store_tile` now carry a
`const { assert!(M <= 32) }`: the composed spelling of exactly this mistake is
`[64, N]`, and it is worth a compile error. No caller in either repository ever
passed more than 32.

Three things that follow, none of them guessable from the first:

- **The fence pair was never needed.** The question `store_fragment` carried as
  **open** is closed in the direction the crate had already bet on, and it is now
  closed against the hardest case rather than the easiest: the consumer is in the
  other warpgroup. `store_wait` plus a block barrier is the whole publication.
- **`store_wait` stays required anyway.** The row that dropped it also read every
  cell — and that row is a race whose two sides happened to arrive in order,
  which is not an ordering. It is reported and gates nothing. This is the one
  place in the table where "it worked" is not evidence.
- **`warp_lanes()` is the spelling the contract is stated in.** `32 * warp_id()`
  is not wrong inside one warpgroup and is not the rule; the two agree there and
  nowhere else, so a kernel that grows a second warpgroup silently stops being
  correct. That is the case for the entry point existing at all — it is four
  characters of arithmetic and the four characters are the whole finding.

Not established: the same question under `cta_group::2`, where one allocation
spans a CTA pair; and any of it on silicon other than a B200.

## Smaller notes

- **Stores go through `_raw` and `to_bits`.** There is no `_pure` store to mirror
  `tcgen05_ld_16x256b_pure`, and the `_unpack16` form splits each register into
  16-bit halves. An accumulator's fp32 has to land bit-exact.
- **`split_columns` takes no offset.** A TMEM segment spans all 128 lanes, so
  segments can only be carved along the column axis and the one that follows a
  `[R, C]` tile begins at `+C`. An explicit offset could name a column inside the
  tile instead, and overlapping segments have no diagnostic: the two MMAs simply
  write each other's accumulator.
- **Block composition is the coordinate map, not a convention.** A thread's two
  slots of a 16-row block are the composed tile's slots `2 * row_block + {0, 1}`,
  and its four values of a 16-column block are its values
  `4 * column_block + {0..4}` — the same map the bigger shape's own
  `RegTile::coordinate` gives, which `reg.rs`'s
  `fragment_blocks_tile_the_bigger_shapes` asserts.
- **`store_wait` is separate from `store_fragment`** because a store's registers
  are consumed at issue: a pass writing many fragments waits once after the last
  one instead of once per fragment.
