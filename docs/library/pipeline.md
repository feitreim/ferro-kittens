# `pipeline` — the persistent-grid harness

`pipeline` is the scaffold of a persistent kernel, extracted: a grid runs a
static strided work-item loop, every mbarrier is re-initialized per item behind
the item boundary, and the kernel supplies a `Job`. ThunderKittens calls this
shape `prototype::lcf`; here the scaffold is `run` and `run_stealing`.

Re-initializing per item is what makes each item's phase arithmetic start from
zero: unbalanced arrivals — a consumer that never arrives for work the item
didn't have — are wiped rather than threaded through parity math.

## A work item belongs to a cluster, not to a CTA

The loop was originally `blockIdx.x` stepping by `gridDim.x`, which hands the two
CTAs of a cluster *different* items — exactly wrong for a cooperative MMA, whose
pair has to be on one output tile. The kernel with the most to gain from a
persistent grid was therefore the one kernel that could not use this scaffold at
all.

The map is now `%clusterid` stepping by `%nclusterid`. That is not a second
schedule beside the old one — it is the same schedule stated at the granularity
the item actually has. A launch that declares no cluster runs clusters of one
CTA, where `%clusterid` *is* `%ctaid` and `%nclusterid` *is* `%nctaid`, so a
block-scope job gets back the exact loop it had. There is one item map, and
`blockIdx.x` was it written one level too low.

### The rank half of the question, and why it needs no `Scope` trait

The other half of "who does this item belong to" is which CTA of the cluster is
asking. That was once filed against a general `Scope` abstraction, and a
block-scope *operation* needs shared storage a `{ WARPS, rank(), sync() }` trait
has no way to supply — so that abstraction could not have done the job anyway.
The two halves separate cleanly, and this one is the easy half:
`cuda_device::cluster::block_rank` reads `%cluster_ctarank` and has no storage,
no collective and no barrier in it. It is already the primitive, the GEMM already
calls it, and wrapping it in a scope trait would add a name without adding a
fact.

## Who the item boundary has to include

Re-initializing barriers per item is what makes each item's parity start at zero,
and it is also the one thing a cluster job cannot do behind a block sync. In a
cluster the CTAs write to *each other's* barriers — the GEMM's peer aims its TMA
at the leader's copy of the stage barrier — so a leader that invalidates on its
own block's schedule can wipe a barrier a peer is still filling, and a peer can
fill a barrier its leader has not yet re-armed. Both are silent: the phase simply
never completes.

`Job::RANKS` is what says which. It costs no extra barriers — the loop has
exactly the two it always had, one publishing the fresh barrier set and one
retiring the finished item — they are just taken at cluster scope, which is a
superset of the block scope they replace. At `RANKS == 1` the branch folds away
at compile time and `sync_threads` is back.

## What using it costs, which is nothing, and was not obvious

`examples/src/gemm.rs` is the caller, and the first this scaffold has ever had.
It runs **exact** on a B200, so the above is verified against silicon and not
only against a compiler, and at 8192³ it is a dead heat with the launch-per-tile
grid it replaces: **1.0217 ms against 1.0204**.

The path there is worth carrying, because it is a trap this module sets for its
next caller. `lcf` at one item per cluster *is* the non-persistent kernel — same
rings, same barriers, drained at the same points — so the only thing `run`
changes is how many clusters exist, and **that number is the whole performance
story**. The GEMM was **2.07× slower at a one-wave grid and 1.19× at a two-wave
one** before a three-wave grid drew level.

Picking that number needs the residency, which
`cuOccupancyMaxActiveBlocksPerMultiprocessor` cannot supply for a
`#[cluster_launch]` kernel — it takes a block shape and no cluster — so the
GEMM's came off a clock by bisection. `experiments/README.md` §7 has the sweep.

## The schedule the hardware picks, and what it is actually worth

`run_stealing` is the same loop over a different item source. Instead of a share
decided before the kernel starts, a cluster runs the item it was launched for and
then *steals* one from the clusters the scheduler has not launched yet —
Blackwell's Cluster Launch Control, which is `clusterlaunchcontrol.try_cancel`
and a 16-byte response delivered on an mbarrier. So the grid is one cluster per
item, `run_stealing` takes no item count at all, and how many clusters are
*resident* is a thing the hardware decides rather than a number this repo writes
down.

### What it is worth, and what it turned out to be worth

The static stride's only loss is the ragged last wave, and at `MAX_CLUSTERS =
222` with the GEMM's `[256, 128]` tiles that is **23% at 4096³, 8% at 8192³ and
0.3% at 16384³** predicted. Measured, it is **1.3%, 1.1% and 2.4%**.

The wave arithmetic is not wrong, and the distinction matters. At 4096³ the
ragged last wave really is 23% of the *grid* idling — 512 tiles over three waves
of 222 leaves 154 clusters with nothing to do. It is not 23% of the *time*,
because the clusters that idle are not the ones on the critical path: the launch
ends when the last busy cluster finishes its tile, and an idle neighbour does not
make that cluster faster. The model is right about a term, and the term is small.

The term that is *not* small was measured separately: the **item boundary**, a
fifth to a third of the launch. A dynamic schedule removes no item boundary at
all; it only moves which cluster pays them. `experiments/README.md` §7 carries the
table.

### The reason to want it anyway is that it deletes the constants

Picking a persistent grid needs `SMS` and `CTAS_PER_SM`, and the second one is a
*measured* property that no device query reports — so it cannot be derived at
compile time and cannot be right on hardware nobody has run. `MAX_CLUSTERS` is
what makes the scaffold B200-shaped. Under `run_stealing` the grid is the
problem's own tile count, the residency is the scheduler's business, and both
constants have nothing left to do.

### Why the steal has to be the cluster's and not the CTA's

The request is `clusterlaunchcontrol.try_cancel…multicast::cluster::all`, and the
multicast is not an optimization. A 2-CTA cluster whose halves each stole would
put the pair on two different output tiles, which is the `blockIdx.x` bug coming
back through a new door — and silently, since both halves would still compute
*something*. The multicast form writes one response into every CTA of the cluster
and completes every CTA's copy of the barrier, so the next item is a fact the
cluster agrees on by construction rather than by a rendezvous someone has to
remember to write.

### The steal could be prefetched, and prefetching it was slower

The response arrives on an mbarrier, so the request need not sit anywhere near
the point the answer is needed. The obvious form issues the *next* item's request
before the current item's `work` and harvests it after, putting a whole tile's K
pipeline between the `try_cancel` and the wait so that no steal is ever on the
critical path. That was expected to be the fast form, and the critical-path form
was expected to be a regression.

**It was the other way round at every size measured, by 8.5% at 16384³ against
numbers that repeat to 0.6%.** Both forms were built and timed;
`experiments/README.md` §7 has the table. The prefetched one is **deleted**, and
`run_stealing` is the form that issues the request after the item it will
replace.

The mechanism is not latency. Prefetching makes every cluster claim its next tile
*before* finishing its current one, so the order tiles are handed out in stops
tracking the order clusters actually become free — which is the ragged tail this
whole mechanism exists to fix, reintroduced by the optimization meant to hide its
latency. That is a hypothesis with one measurement behind it; what would test it
is recording per-cluster item counts and comparing their spread.

### What orders a read of the response against the next request

Every thread reads the response, because every thread needs the item and a shared
read is cheaper than a broadcast. That makes the buffer a thing many threads read
and one thread overwrites, and the harvest is therefore placed *before* the item
boundary that already exists rather than after it: the boundary that retires the
item is also what says no thread is still reading the response the next request
will land on. It costs no extra barrier.

## The store stage `lcf` is supposed to lack, which this already carries

ThunderKittens' next prototype up is `lcsf` — load-compute-**store**-finish — and
it was filed here as a shape this module would have to grow, on the grounds that
with the store folded into the item it cannot overlap the next item's load.
**Nothing here had to change to get that overlap**, and the reason is a property
of what the item boundary touches rather than of the loop above it.

`run` re-initializes mbarriers per item. It does not touch tensor memory, which a
`Job` allocates once *outside* the loop precisely because an allocation spans
items. So a job may finish `work` with its accumulator **undrained**, carry it
across the boundary intact, and drain it at the top of the next item — after that
item's first stages are issued and while they are in flight. The pending item is
job state, `work` is where the phases are ordered, and `lcsf` is therefore a
reordering inside a `Job` rather than a stage this scaffold sequences.
`examples/src/gemm.rs`'s `Lcsf` is one, built against this module unmodified.

Two obligations move to the job with it, and both are silent when missed:

- The drain reads tensor memory the *next* item's MMA will overwrite, so whatever
  rendezvous separates them must cover every CTA that MMA writes — for a
  `cta_group::2` accumulator that is the cluster, not the block, and the item
  boundary's scope has to appear inside `work` and not only around it.
- A job that defers its last item's store owes one drain after `run` returns,
  which is a whole output tile computed by nobody if it is forgotten.

What that buys is a separate question from whether it can be expressed, and for
the GEMM the answer is nothing — `experiments/README.md` §7 has the two sessions
and the probe that says why. The fact worth carrying is that the experiment cost
a reordering, so the next job with a store phase should measure one before anyone
scopes a second scaffold.

## Which item, and which tile: `grouped`

Both loops answer *which item comes next*. Neither says what an item **is**, and
for a job whose items tile a 2-D output that second map is the one the memory
system sees. `grouped` is it: item → `(row, column)`, in blocks of `group`
tile-rows at a time, with `group == 1` the row-major map it generalizes.

It sits in this module rather than in a kernel because it composes with the item
source instead of replacing it. `run` and `run_stealing` each hand a job a
permutation of `0..items`; `grouped` is a bijection of `0..items` onto the tile
grid, and a bijection of a permutation is still every tile exactly once. So a job
that maps its item through it is correct under both loops by construction, and
the swizzle and the scheduler can be swept independently.

**What the width is worth is a property of the shape, not a constant.** The
clusters resident at one time span `ceil(wave / group)` tile-columns and `group`
tile-rows instead of one long row-major run, and the operand bytes they
collectively touch — the working set an L2 has to hold for the reuse to land — is
minimized where those two are balanced against the tile's own aspect.
`experiments/README.md` §7 carries the sweep and the shape it was found at; a
caller should take the width as a parameter and measure, not inherit one.

The map is a bijection for every `group >= 1`, including one that does not divide
`rows`: the last group is short, and its items are laid out over the rows it
actually has rather than over the rows it would have had. Without that the tail
of the grid would alias, which is a wrong `C` and not a slow one.

A width at or past the tile grid's height is the fully column-major walk, whatever
it is nominally set to — which makes an over-large width a saturating parameter
rather than an illegal one, and is why a sweep may carry widths taller than some
of its shapes.
