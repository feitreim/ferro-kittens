# `sync` — semaphores, transaction bytes, block reductions

cuda-oxide exposes no named barriers, so every cross-warp handoff in a
warp-specialized kernel is an mbarrier phase-parity spin (FA4 does the same).
`sync` makes that idiom first-class: `Semaphore` is a stateless handle over one
mbarrier word, `SemaphoreRing` owns the `index → (stage, parity)` arithmetic that
kernels used to thread by hand, and `TransactionBytes` makes a stage's byte
charge a receipt rather than a claim.

## The soundness rule

Parity arithmetic works because **every barrier's completions lead its waiter by
at most one phase** — each producer's next completion transitively requires the
previous consumer wait. That is a claim about a kernel's protocol, and none of
these types can check it. The `handoff` model at the bottom of this document is
what it takes to establish it for a real kernel.

## Cluster scope, and who charges the bytes

A `Semaphore` is *one CTA's* barrier. A `cta_group::2` MMA consumes an operand
stage of four tiles staged by two CTAs, so its issuer needs one barrier saying
the whole stage is present — which means a CTA has to be able to name its peer's
copy. `Semaphore::at_rank` does that, and it hands back a `ClusterSemaphore`
rather than another `Semaphore`, because a barrier that is not yours is not a
barrier you may do everything to.

That restriction is the ISA's and not ours. `mbarrier.arrive.expect_tx` takes
`.shared::cta` only, and `mbarrier.arrive … .shared::cluster` carries no
transaction count: remote addressing and byte accounting sit in different
instructions and do not compose. So a CTA may charge exactly one barrier — its
own — and may arrive at any rank's. `ClusterSemaphore` offers exactly that pair:
arrive, or hand the address to an engine that will complete transactions on it.
It has no `wait` and no `expect_tx` because there is no instruction for either,
and a type that admitted them would be promising something the hardware does not
do.

What is left to decide is which CTA's barrier a cluster stage completes on, and
there are only two shapes:

1. **The waiter charges the whole stage.** One barrier, in the CTA that waits on
   it. Every rank's TMA completes there — the peers name it through
   `Semaphore::at_rank` — and that CTA charges every byte the stage will bring
   in, including the bytes it does not issue itself.
2. **Every rank charges its own.** Each producer charges the bytes it issues
   against its own barrier, and forwards a `ClusterSemaphore::arrive` to the
   waiter once that barrier flips.

This library takes (1). What (2) has going for it is that the sum stays local — a
CTA charges exactly what it issued — and that is all it has. It costs a barrier
per rank per stage, a thread per rank spinning on that barrier, and a second hop
on the critical path carrying no information: when a producer's own barrier flips
it knows nothing the TMA engine could not have told the waiter directly.

The locality is recoverable under (1) anyway, which is what settles it. A cluster
stage is symmetric by construction — every rank stages the same tile types at the
same shared offsets, which is exactly what lets one MMA descriptor read across
the pair — so the whole-stage charge is the local charge times the number of
ranks staging into it, and the rank count is a launch constant. A charge derived
from the loads a producer issued is still derivable; it is derived once and
multiplied, which is `TransactionBytes::across_ranks`.

One thing (1) looks like it should need and does not: ordering between the charge
and the peers' loads. An mbarrier's transaction count is a signed accumulator, so
completions may land before the `expect_tx` that expects them — only the totals
have to agree. No sync sits between them in the GEMM and none is owed.

## Where the transaction count comes from

`Semaphore::expect_tx` takes a `TransactionBytes` and nothing else, and the only
way to obtain one is to issue the load that will deliver it. A stage's charge is
the sum of the calls that were made.

Every producer used to write that total by hand — `(KTile::BYTES + VTile::BYTES)
as u32` — and keep it in step with the loads above it by reading both.
Under-charging flips the barrier before the last bytes land and the consumer
reads a half-written tile, with no fault and no diagnostic; over-charging hangs
the block. Adding a load and forgetting the sum is one edit, and nothing was
watching for it.

The lighter fix — an `expect_tiles(&[…])` taking the tile types — **was
rejected**, because it is the same fact stated twice with nicer syntax. A list
can omit a tile exactly as a sum can omit a term. What has to agree with the
charge is the *set of loads issued*, so the charge has to come from them and not
from anything that restates them.

Two things fall out, and are the point:

- **Writing a wrong number is not expressible.** `TransactionBytes` has no public
  constructor and is not an integer, so there is nothing to mistype.
- **Issuing a load and dropping its charge is a compile error.** The type is
  `#[must_use]`, so `tile.tma_load(…);` as a statement fails under `-D warnings`,
  and a charge bound to a name that never reaches an `expect_tx` fails as an
  unused binding. It is also not `Copy`, so charging the same bytes to two
  barriers is a use of a moved value.

What the type does not catch, stated plainly: it says how many bytes, never which
barrier. A load aimed at one semaphore whose charge is paid to another still
type-checks — and has to, because that is exactly what a cluster stage does.

## `ClusterSemaphore::arrive` has no caller yet

Nothing in this repo calls it, and the case that looked like it would and did not
is worth recording. A persistent cluster pipeline needs the CTAs of a pair to
agree at each work-item boundary, which is a *rendezvous*: every rank arriving at
every rank. Built out of `ClusterSemaphore::arrive` it is a barrier per rank and
a spin per rank, hand-rolling `barrier.cluster.arrive`/`wait` — which is one
instruction, is what `cuda_device::cluster::cluster_sync` issues, and is what
`pipeline::run` takes. What is left for `arrive` is the signal a cluster barrier
is the wrong shape for: one named rank telling another that a specific thing has
happened, while the rest of the cluster keeps going.

## `block_reduce`, and why it lives here

`block_reduce` is the other thing a kernel needs the block to agree on, and it is
not a barrier idiom but a *collective*: warps cannot shuffle to each other, so a
statistic spanning them is staged through shared memory and folded, with
`sync_threads` on both sides. It lives in `sync` rather than in `reg` because
what it is made of is the barrier and the staging buffer; the register half of
the same fold — `reg::RegTile::tile_reduce` and the `shuffle_xor` butterflies —
is warp scope and stops at 32 lanes.

Warp `w` stages its value at element `w`, the block syncs, and *every* thread
folds all `WARPS` elements in index order from `Op::IDENTITY`. So the result is
bit-for-bit identical in every thread — a scalar operand a kernel may use
immediately, with no broadcast step and no lane that has to be asked. The
trailing barrier is the reusable part: on return no thread is still reading the
scratch, so calling it twice in a row — mean, then variance — needs nothing at
the call site.

`WARPS` is free: any block width from one warp up. It was briefly constrained to
a multiple of four, because `SharedVec` enforced its TMA box rules at
construction and four fp32 is the narrowest legal box — a rule about a descriptor
binding a vector that never becomes one. Those asserts now sit on the four
transfer methods, so a two-warp block's 8-byte scratch is a handle like any
other. A one-warp block is legal too, and degenerate: the fold is over a single
slot and `reg::warp_reduce` is the same answer without the barriers.

The bound is `ReduceOp` rather than `reg::BinaryOp` for the reason `row_reduce`
takes one: the warps' partials arrive in slot order, which is not an order the
caller chose, so a fold over them has to be associative and commutative to mean
anything. `Sub` and `Div` are not members, and the identity is what lets the fold
start the same way whatever `WARPS` is.

## `handoff` — how deep a mailbox ring the one-phase-lead rule needs

The soundness rule at the top of this document is what makes parity arithmetic
work, and it is exactly the claim this crate's types cannot check. What it costs
when the claim is false: `examples/src/gemm_sol.rs` handed work items across
warps through **one** barrier and **one** cell, its producer could lead the
epilogue by four publications, and `4096x4096x1024` stopped returning.

The lead is not a divisibility property and not a `k` threshold — it is what a
chain of independent throttles permits, and the only honest way to get it is to
enumerate the interleavings. `handoff::depth_needed` is that enumeration: an
exhaustive search over the reachable states of a four-stage operand pipeline over
a two-deep accumulator, which is `gemm_sol`'s `[256, N]` entry. It is a model and
it says so; it is `pub` because a kernel's shape contract is where its handoff
depth is decided, and deciding one should be an assertion rather than an
argument.

The roles it models, and nothing else couples them:

- the **producer** publishes item `i`, then issues `k_blocks` loads for it,
  throttled by the MMA warp's release of an operand stage `STAGES` blocks back,
  then queries for the next item and publishes it;
- the **MMA warp** waits for item `i`'s publication and for the epilogue to have
  drained item `i - ACCUMULATORS`, then multiplies `k_blocks` blocks;
- the **epilogue** waits for item `i`'s publication and for the MMA warp to have
  filled it, then drains and releases it.

### The depths it reports

| shape | `depth_needed` |
| --- | --- |
| `k_blocks = 4`, 5 items, 4 stages, 2 accumulators | **4** |
| `k_blocks ∈ {8, 12, 16, 20}`, same otherwise | 3 |
| `k_blocks = 4`, 5 items, 4 stages, 1 accumulator | 3 |
| `k_blocks = 16`, 5 items, 4 stages, 1 accumulator | 2 |
| `k_blocks ∈ {4, 16}`, **1 item**, 4 stages, 2 accumulators | 2 |

**The deepest lead is at the *shallowest* `k`**, which is the opposite of what a
divisibility story predicts and is why every theory of the form "`k_blocks % X ==
0` breaks it" was wrong. At `k_blocks == STAGES` the producer's own throttle only
reaches back into the item before last, so the MMA warp is a whole item further
ahead of the epilogue than it is at any deeper `k`, and one more publication is
outstanding. That is why the shipped depth is four and not two.

A one-deep accumulator makes the chain one step tighter, which is `gemm_sol`'s
`[512, 256]` entry: the same handoff at four cells covers it with a cell to
spare, and that is why one shared plan serves both.

### One item per cluster still needs two

One item per cluster is the shape every correctness gate ran, and it needs a
depth of **two** — so the one-deep handoff was unsound at the gate's own shape as
well. The producer's second publication is the `has_work = false` sentinel and
nothing at all interlocks it against the epilogue's first read, so
`1024x1024x512` was winning a race rather than being correct.

What made it win by a mile is that the epilogue's first poll is a few dozen
cycles after the cluster's launch barrier while the producer's sentinel is a
whole item's TMA behind it. Shortening `k` shortens exactly that margin, which is
the shape of a cliff at one value and not of a degradation.
