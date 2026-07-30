# `epilogue` — the shared→global staging ring

`StoreRing` is the shared→global half of an epilogue: `DEPTH` staging tiles that
a drained accumulator is written into and the TMA engine reads out of.

A kernel whose result is computed in registers has no way to hand it to the TMA
engine directly — the engine reads shared memory — so the last move of such a
kernel is `stmatrix` into a swizzled tile (`ldst::store_tile`) and
`cp.async.bulk.tensor` out of it (`SharedTile::tma_store`). `StoreRing` is that
tile, `DEPTH` of them, with the fence and the two waits those two instructions
owe each other stated once instead of at every call site.

## Why the depth is a parameter, and what it buys

At depth 1 there is one buffer: band `k + 1`'s `stmatrix` cannot start until band
`k`'s store has finished *reading* that buffer, so the register→shared move and
the shared→global move strictly alternate. `examples/src/softmax.rs` is that
shape written by hand, and it is the right shape for a kernel that stores once.

At depth 2 the two overlap: band `k + 1` writes the other buffer while the engine
is still draining band `k`'s. What the extra buffer costs is one whole tile of
shared memory, which for a kernel already near its occupancy step is the
expensive kind of byte — a `[128, 64]` bf16 buffer is 16 KiB, so a step of depth
is 16 KiB of the shared plan and nothing else in the type varies with it.

The depth is a parameter so that trade is measured rather than inherited.

## The fence and the two waits

Three mechanisms are involved and none of them subsumes another. The source
carries a compressed statement of all three; this is the long form.

**`publish_to_async_proxy`, before the engine reads a buffer.** `stmatrix` is an
ordinary shared-memory write through the *generic* proxy; the TMA engine reads
through the *async* proxy. Nothing orders the two but this fence, and it orders
the writes of the thread that *executes* it — so every thread that wrote a row
fences, and a barrier after that is what makes the whole scope's writes visible
to the one thread that issues the store. A `bar.sync` alone is not enough: it
orders generic-proxy accesses against each other and says nothing about a proxy
it does not name.

**`cp.async.bulk.wait_group.read`, before a buffer is written again.** A bulk
store completes in two stages: first the engine is done reading shared memory,
later the bytes are visible in global memory. Recycling a buffer needs only the
first, and `StoreRing::acquire` waits for exactly that — blocking on global
visibility instead would serialize the overlap the ring exists for. Groups are
**per thread** and age in issue order, so this wait is taken by the same single
thread that issued every store, and a barrier after it is what releases the other
warps past a wait they did not take.

The group count `acquire` passes is `IN_FLIGHT`: with `DEPTH` groups committed
and ageing in issue order, leaving at most `IN_FLIGHT` with outstanding reads is
exactly the statement that the oldest — this buffer's — is done being read.
Before `DEPTH` bands have been committed the wait is trivially satisfied and
costs an instruction, so the ring needs no fill phase.

**`cp.async.bulk.wait_group`, once, at the end.** Nothing else makes a bulk store
visible to a following kernel or to the host — not the end of the kernel, not a
`bar.sync`, not dropping the handle. `StoreRing::drain` is the only place it
appears, because it is the only obligation the ring cannot discharge
incrementally.

## Why this is not `SharedTileRing`

`SharedTileRing` is addressing and nothing else: `N` tiles and `index % N`, with
the completion mechanism supplied separately by a `SemaphoreRing` the caller
pairs it with. The pairing works for *loads* because a load's destination is
shared memory, where an mbarrier can live and count arriving bytes. A store's
destination is global memory, where none can, so a store ring's completion is a
per-thread group count and there is no semaphore ring to pair with — the
discipline has to live on the ring itself, which is why `StoreRing`'s methods are
collective and `SharedTileRing`'s are pure address arithmetic.

The mechanical half of the same answer: the wait's group count is `DEPTH - 1`,
and `tma_store_wait_read::<{DEPTH - 1}>()` is not something Rust will evaluate in
generic-argument position without `generic_const_exprs` — the same restriction
`shared::Element::Unpacked` and `shared::MmaElement::mma` both dodge the same
way. So the parameter here is `IN_FLIGHT`, the number the *instruction* takes,
and the buffer count is derived from it. A type parameterized that way could not
embed a `SharedTileRing<.., { IN_FLIGHT + 1 }>` either.

## Whose barrier: the `Scope` parameter

Every collective half of the ring is two things: a *convergence*, and the one
thread whose store groups the ring counts. Both are properties of the set of
threads that write a buffer, and the type originally assumed that set was the
whole CTA — `bar.sync` and `threadIdx.x == 0`, written into the methods.

A staging tile need not be CTA-wide. `examples/src/gemm.rs` gives each warp its
own `[32, 64]` tile precisely so that no `bar.sync` is needed: the four warps
share no bytes, `stmatrix.sync.aligned` is already a convergence point for the
warp that issues it, and a block barrier there would synchronize four warps that
have nothing to say to each other. A ring hard-wired to CTA scope cannot express
that layout at all — at that kernel's four bands an item it is **eight `bar.sync`
per output tile against zero**, and the opcode census reads exactly that
difference.

So the scope is a parameter — `Cta` and `Warp`, defaulting to `Cta`. Nothing else
moves: the fence, the two waits and their order are the same instructions in the
same places, because the argument for each of them is about proxies and group
counts and not about how many threads are in the barrier.

Getting the two halves from different scopes is silently wrong rather than a type
error, which is why one trait states them together: a CTA-wide barrier around a
lane-0 issue would work and cost too much, and a warp barrier around a thread-0
issue would let three warps run ahead of a wait they never took. `Warp`'s mask is
the full warp and not `activemask`, because every lane holds part of the buffer
and a ring acquired by a subset of the warp is a buffer nobody finished writing.

### What the warp scope is worth, measured, and it is negative

`gemm` runs both scopes over the same 16 384 B — four warp-scope rings, or one
CTA-scope ring over the same run read as one `[128, 64]` tile — and the
**CTA-scope arm wins, by 0.6–1.6% of the launch in four paired cells across two
containers.** `experiments/README.md` §7 has the rows.

Saving eight `bar.sync` an item does not pay for issuing four `[32, 64]` boxes
where one `[128, 64]` box would do: the engine's per-instruction overhead is the
term that matters and the barrier is not.

So the parameter buys expressiveness and costs speed where it is used, and a
kernel with no other reason should reach for `Cta`. It is kept because a
warp-private staging tile is a real layout — it is what `gemm`'s `stmatrix` path
wants and what `device-tests`' `store ring warp` covers — and because the arm
that lost is what makes the arm that won a measurement rather than an assumption.
