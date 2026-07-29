//! Persistent-grid harness — the scaffold of a persistent flash-attention
//! forward kernel, extracted.
//!
//! A grid runs a static strided work-item loop. Every mbarrier is re-initialized
//! per item behind the item boundary, so each item's phase arithmetic starts
//! from zero and unbalanced arrivals (a consumer that never arrives for work the
//! item didn't have) are wiped, not threaded through parity math.
//! ThunderKittens calls this shape `prototype::lcf`; here the scaffold is [`run`]
//! and the kernel supplies a [`Job`].
//!
//! # A work item belongs to a cluster, not to a CTA
//!
//! The loop used to be `blockIdx.x` stepping by `gridDim.x`, which hands the two
//! CTAs of a cluster *different* items — exactly wrong for a cooperative MMA,
//! whose pair has to be on one output tile. The kernel with the most to gain
//! from a persistent grid was therefore the one kernel that could not use this
//! scaffold at all (#51).
//!
//! The map is now `%clusterid` stepping by `%nclusterid`. That is not a second
//! schedule beside the old one — it is the same schedule stated at the
//! granularity the item actually has. A launch that declares no cluster runs
//! clusters of one CTA, where `%clusterid` *is* `%ctaid` and `%nclusterid` *is*
//! `%nctaid`, so a block-scope job gets back the exact loop it had. There is one
//! item map, and `blockIdx.x` was it written one level too low.
//!
//! ## The rank half of the question, and why it did not need #3
//!
//! The other half of "who does this item belong to" is which CTA of the cluster
//! is asking, and #51 filed that against #3's `Scope` — then noted #6's finding
//! that #3 as written could not do its job, because a block-scope *operation*
//! needs shared storage a `{ WARPS, rank(), sync() }` trait has no way to
//! supply. The two halves separate cleanly, and this one is the easy half:
//! [`cuda_device::cluster::block_rank`] reads `%cluster_ctarank` and has no
//! storage, no collective and no barrier in it. It is already the primitive, the
//! GEMM already calls it, and wrapping it in a scope trait would add a name
//! without adding a fact. #3 shipped a block reduction and no `Scope`, and
//! nothing about that bears on this.
//!
//! # Who the item boundary has to include
//!
//! Re-initializing barriers per item is what makes each item's parity start at
//! zero, and it is also the one thing a cluster job cannot do behind a block
//! sync. In a cluster the CTAs write to *each other's* barriers — the GEMM's
//! peer aims its TMA at the leader's copy of the stage barrier — so a leader
//! that invalidates on its own block's schedule can wipe a barrier a peer is
//! still filling, and a peer can fill a barrier its leader has not yet re-armed.
//! Both are silent: the phase simply never completes.
//!
//! [`Job::RANKS`] is what says which. It costs no extra barriers — the loop has
//! exactly the two it always had, one publishing the fresh barrier set and one
//! retiring the finished item — they are just taken at cluster scope, which is
//! a superset of the block scope they replace. At `RANKS == 1` the branch folds
//! away at compile time and `sync_threads` is back.
//!
//! # What using it costs, which is nothing, and was not obvious
//!
//! `examples/src/gemm.rs` is the caller — the kernel #51 filed this against,
//! and the first this scaffold has ever had. It runs **exact** on a B200, so
//! what is above is verified against silicon and not only against a compiler,
//! and at 8192³ it is a dead heat with the launch-per-tile grid it replaces:
//! 1.0217 ms against 1.0204.
//!
//! The path there is worth carrying, because it is a trap this file sets for
//! its next caller. `lcf` at one item per cluster *is* the non-persistent
//! kernel — same rings, same barriers, drained at the same points — so the only
//! thing [`run`] changes is how many clusters exist, and **that number is the
//! whole performance story**. The GEMM was 2.07× slower at a one-wave grid and
//! 1.19× at a two-wave one before a three-wave grid drew level. Picking it
//! needs the residency, which
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` cannot supply for a
//! `#[cluster_launch]` kernel — it takes a block shape and no cluster — so the
//! GEMM's came off a clock by bisection. `examples/README.md` §7 has the sweep
//! and what it says about #78.
//!
//! # The schedule the hardware picks, and what it is actually worth
//!
//! [`run_stealing`] is the same loop over a different item source. Instead of a
//! share decided before the kernel starts, a cluster runs the item it was
//! launched for and then *steals* one from the clusters the scheduler has not
//! launched yet — Blackwell's Cluster Launch Control, which is
//! `clusterlaunchcontrol.try_cancel` and a 16-byte response delivered on an
//! mbarrier. So the grid is one cluster per item, [`run_stealing`] takes no
//! item count at all, and how many clusters are *resident* is a thing the
//! hardware decides rather than a number this repo writes down.
//!
//! **What it is worth, and what it turned out to be worth.** The static
//! stride's only loss is the ragged last wave, and at `MAX_CLUSTERS = 222` with
//! the GEMM's `[256, 128]` tiles that is 23% at 4096³, 8% at 8192³ and 0.3% at
//! 16384³. Measured, it is **1.3%, 1.1% and 2.4%**.
//!
//! The wave arithmetic is not wrong, and the distinction matters. At 4096³ the
//! ragged last wave really is 23% of the *grid* idling — 512 tiles over three
//! waves of 222 leaves 154 clusters with nothing to do. It is not 23% of the
//! *time*, because the clusters that idle are not the ones on the critical
//! path: the launch ends when the last busy cluster finishes its tile, and an
//! idle neighbour does not make that cluster faster. The model is right about a
//! term, and the term is small. #90 and #94 already measured the one that is
//! not — the *item boundary*, a fifth to a third of the launch — and a dynamic
//! schedule removes no item boundary at all; it only moves which cluster pays
//! them. `examples/README.md` §7 carries the table.
//!
//! **The reason to want it anyway is that it deletes the constants.** Picking a
//! persistent grid needs `SMS` and `CTAS_PER_SM`, and #84 established that the
//! second one is a *measured* property that no query in the table reports — so
//! it cannot be derived at compile time and cannot be right on hardware nobody
//! has run. `MAX_CLUSTERS` is what makes the scaffold B200-shaped. Under
//! [`run_stealing`] the grid is the problem's own tile count, the residency is
//! the scheduler's business, and both constants have nothing left to do.
//!
//! ## Why the steal has to be the cluster's and not the CTA's
//!
//! The request is `clusterlaunchcontrol.try_cancel…multicast::cluster::all`,
//! and the multicast is not an optimization. A 2-CTA cluster whose halves each
//! stole would put the pair on two different output tiles, which is #51's bug
//! coming back through a new door — and silently, since both halves would still
//! compute *something*. The multicast form writes one response into every CTA
//! of the cluster and completes every CTA's copy of the barrier, so the next
//! item is a fact the cluster agrees on by construction rather than by a
//! rendezvous someone has to remember to write.
//!
//! ## The steal could be prefetched, and prefetching it was slower
//!
//! The response arrives on an mbarrier, so the request need not sit anywhere
//! near the point the answer is needed. The obvious form issues the *next*
//! item's request before the current item's `work` and harvests it after,
//! putting a whole tile's K pipeline between the `try_cancel` and the wait so
//! that no steal is ever on the critical path. #88 expected that to be the fast
//! form and expected the critical-path form to be a regression.
//!
//! **It was the other way round at every size measured, by 8.5% at 16384³
//! against numbers that repeat to 0.6%.** Both forms were built and timed;
//! `examples/README.md` §7 has the table. The prefetched one is **deleted**,
//! and [`run_stealing`] is the form that issues the request after the item it
//! will replace.
//!
//! The mechanism is not latency. Prefetching makes every cluster claim its next
//! tile *before* finishing its current one, so the order tiles are handed out
//! in stops tracking the order clusters actually become free — which is the
//! ragged tail this whole mechanism exists to fix, reintroduced by the
//! optimization meant to hide its latency. That is a hypothesis with one
//! measurement behind it; what would test it is recording per-cluster item
//! counts and comparing their spread.
//!
//! ## What orders a read of the response against the next request
//!
//! Every thread reads the response, because every thread needs the item and a
//! shared read is cheaper than a broadcast. That makes the buffer a thing many
//! threads read and one thread overwrites, and the harvest is therefore placed
//! *before* the item boundary that already exists rather than after it: the
//! boundary that retires the item is also what says no thread is still reading
//! the response the next request will land on. It costs no extra barrier.
//!
//! # The store stage `lcf` is supposed to lack, which this already carries
//!
//! ThunderKittens' next prototype up is `lcsf` — load-compute-**store**-finish
//! — and #15 files it as a shape this module would have to grow, on the
//! grounds that with the store folded into the item it cannot overlap the next
//! item's load. **Nothing here had to change to get that overlap**, and the
//! reason is a property of what the item boundary touches rather than of the
//! loop above it.
//!
//! [`run`] re-initializes mbarriers per item. It does not touch tensor memory,
//! which a [`Job`] allocates once *outside* the loop precisely because an
//! allocation spans items. So a job may finish `work` with its accumulator
//! **undrained**, carry it across the boundary intact, and drain it at the top
//! of the next item — after that item's first stages are issued and while they
//! are in flight. The pending item is job state, `work` is where the phases are
//! ordered, and `lcsf` is therefore a reordering inside a [`Job`] rather than a
//! stage this scaffold sequences. `examples/src/gemm.rs`'s `Lcsf` is one, built
//! against this file unmodified.
//!
//! Two obligations move to the job with it, and both are silent when missed.
//! The drain reads tensor memory the *next* item's MMA will overwrite, so
//! whatever rendezvous separates them must cover every CTA that MMA writes —
//! for a `cta_group::2` accumulator that is the cluster, not the block, and
//! the item boundary's scope has to appear inside `work` and not only around it.
//! And a job that defers its last item's store owes one drain after [`run`]
//! returns, which is a whole output tile computed by nobody if it is forgotten.
//!
//! What that buys is a separate question from whether it can be expressed, and
//! for the GEMM the answer is nothing — `examples/README.md` §7 has the two
//! sessions and the probe that says why. The fact worth carrying here is that
//! the experiment cost a reordering, so the next job with a store phase should
//! measure one before anyone scopes a second scaffold.
//!
//! # Which item, and which tile: [`grouped`]
//!
//! Both loops above answer *which item comes next*. Neither says what an item
//! **is**, and for a job whose items tile a 2-D output that second map is the
//! one the memory system sees. [`grouped`] is it: item → `(row, column)`, in
//! blocks of `group` tile-rows at a time, with `group == 1` the row-major map
//! it generalizes.
//!
//! It sits here rather than in a kernel because it composes with the item
//! source instead of replacing it. [`run`] and [`run_stealing`] each hand a job
//! a permutation of `0..items`; [`grouped`] is a bijection of `0..items` onto
//! the tile grid, and a bijection of a permutation is still every tile exactly
//! once. So a job that maps its item through it is correct under both loops by
//! construction, and the swizzle and the scheduler can be swept independently.

use cuda_device::barrier::Barrier;
use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::thread;

use crate::sync::{Semaphore, TransactionBytes};

/// One persistent kernel's work, split at the points the scaffold owns.
/// Implementations are plain structs of tile/semaphore handles built once
/// before [`run`]; every method must be `#[inline(always)]` so the job
/// scalarizes into the kernel like the hand-written loop it replaces.
pub trait Job {
    /// CTAs of the cluster that share this job's barrier set, and so the scope
    /// [`run`]'s item boundary has to be taken at. The default is the block-scope
    /// job: one CTA per item, no peer with an interest in when the barriers are
    /// re-armed.
    ///
    /// This is not the launch's cluster size. A cluster whose CTAs never name
    /// each other's barriers is still `1` here, and pays a `bar.sync` per item
    /// rather than a `barrier.cluster`.
    const RANKS: u32 = 1;

    /// (Re)initialize every mbarrier `item` will use — arrival counts may
    /// depend on the item's shape. Called on thread 0 of each CTA only; [`run`]
    /// fences and syncs before any other thread, in any rank, touches the
    /// barriers.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::sync::Semaphore::init`].
    unsafe fn init(&self, item: u32);

    /// Invalidate the item's barrier set, wiping whatever unbalanced
    /// arrivals the finished item left. Thread 0 of each CTA only.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::sync::Semaphore::inval`].
    unsafe fn inval(&self);

    /// One work item, entered by every thread of every CTA that owns it. Role
    /// dispatch (which warps produce, consume, issue MMAs; which rank leads)
    /// lives here.
    ///
    /// # Safety
    ///
    /// Kernel-specific: whatever the item's barrier protocol requires.
    unsafe fn work(&mut self, item: u32);
}

/// Map a work item to a `(row, column)` of a `rows`x`columns` tile grid,
/// walking `group` tile-rows at a time.
///
/// `group == 1` is the row-major map `(item / columns, item % columns)`, so the
/// control a sweep compares against is a parameter value here and not a second
/// code path. Above 1 the walk fills `group` rows of tiles down the column
/// before moving right, which is what keeps a row-panel of the left operand
/// resident across the columns that re-read it.
///
/// **What that is worth is a property of the shape, not a constant.** The
/// clusters resident at one time span `ceil(wave / group)` tile-columns and
/// `group` tile-rows instead of one long row-major run, and the operand bytes
/// they collectively touch — the working set an L2 has to hold for the reuse to
/// land — is minimized where those two are balanced against the tile's own
/// aspect. `examples/README.md` §7 carries the sweep and the shape it was found
/// at; a caller should take the width as a parameter and measure, not inherit
/// one.
///
/// The map is a bijection for every `group >= 1`, including one that does not
/// divide `rows`: the last group is short, and its items are laid out over the
/// rows it actually has rather than over the rows it would have had. Without
/// that the tail of the grid would alias, which is a wrong `C` and not a slow
/// one.
///
/// **Preconditions, none of them checked**: `group`, `rows` and `columns` at
/// least 1, and `item` less than `rows * columns`. This is ordinary arithmetic
/// and cannot itself be unsound, but a caller that breaks them gets a
/// coordinate off its own tile grid — so on the device the fault lands in
/// whatever `unsafe` addresses `C` with the answer, which is why the launcher
/// rejects a zero width rather than passing it through.
#[inline(always)]
pub fn grouped(item: u32, rows: u32, columns: u32, group: u32) -> (u32, u32) {
    let span = group * columns;
    let first = item / span * group;
    let height = group.min(rows - first);
    let within = item % span;
    (first + within % height, within / height)
}

/// The item boundary, at the scope the job's barriers live at. A cluster
/// barrier subsumes a block one — every thread of every CTA of the cluster
/// arrives — so this is a widening and never a substitution.
#[inline(always)]
fn boundary<J: Job>() {
    if J::RANKS > 1 {
        cluster::cluster_sync();
    } else {
        thread::sync_threads();
    }
}

/// Run `job` over `items` work items on the static strided schedule, one item
/// per cluster. Owns the barrier lifecycle (leader-only inval-then-init,
/// published by a proxy fence and the item boundary) and the per-item tcgen05
/// fence pairing: `work` returns with its MMAs committed, the harness fences
/// them before the boundary. TMEM allocation stays with the caller — it spans
/// items, not one — and a cluster job's allocation is exactly what makes that
/// worth saying, since [`crate::tmem::alloc_cluster`] is a whole-cluster
/// collective that must not be inside the loop.
///
/// # Safety
///
/// Every thread of every CTA of the cluster must call this together, with
/// `job`'s barrier storage unused by anything else for the duration and laid
/// out at the same shared offset in every rank. `items` must be the same in
/// every CTA, and the grid must be a whole number of clusters — which a
/// `#[cluster_launch]` kernel's prepared launch already checks.
#[inline(always)]
pub unsafe fn run<J: Job>(job: &mut J, items: u32) {
    unsafe {
        let leader = thread::threadIdx_x() == 0;
        let mut initialized = false;
        let mut item = cluster::cluster_idx();
        while item < items {
            if leader {
                if initialized {
                    job.inval();
                }
                job.init(item);
                fence_proxy_async_shared_cta();
            }
            initialized = true;
            boundary::<J>();
            job.work(item);
            tcgen05_fence_before_thread_sync();
            boundary::<J>();
            item += cluster::num_clusters();
        }
        if leader && initialized {
            job.inval();
        }
    }
}

/// The hardware's work queue, as the shared memory a cluster asks it through:
/// the 16-byte `try_cancel` response and the mbarrier it is delivered on.
///
/// It is the caller's storage rather than the scaffold's for the same reason a
/// [`Job`]'s barriers are — a kernel owns one shared plan and nothing else may
/// carve out of it — and it must sit at the *same offset in every rank*, which
/// the multicast response requires and a symmetric plan gives for free.
#[derive(Clone, Copy)]
pub struct ClcQueue {
    response: *mut u64,
    sem: Semaphore,
}

impl ClcQueue {
    /// The response's size, which is the ISA's (`.b128`) and is also exactly
    /// the transaction count its mbarrier is charged.
    const RESPONSE_BYTES: usize = 16;
    /// Shared bytes to reserve: the response, then the barrier under it.
    pub const BYTES: usize = Self::RESPONSE_BYTES + 8;
    /// The response is a 128-bit store, so the base is 16-byte aligned.
    pub const ALIGNMENT: usize = 16;

    /// Lay a queue over [`Self::BYTES`] of shared memory.
    ///
    /// # Safety
    ///
    /// `base` must point to [`Self::BYTES`] of shared memory aligned to
    /// [`Self::ALIGNMENT`], used by nothing else for the kernel's duration, and
    /// at the same offset in every CTA of the cluster.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        Self {
            response: base as *mut u64,
            sem: unsafe { Semaphore::attach(base.add(Self::RESPONSE_BYTES) as *mut Barrier) },
        }
    }

    /// Arm this CTA's copy of the response barrier. One thread per CTA, once,
    /// before any rank issues a request.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::init`].
    #[inline(always)]
    unsafe fn arm(self) {
        unsafe { self.sem.init(1) }
    }

    /// Retire it.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::inval`], with no request outstanding.
    #[inline(always)]
    unsafe fn disarm(self) {
        unsafe { self.sem.inval() }
    }

    /// Charge this CTA's barrier for the response it is about to be sent. One
    /// thread per CTA, per request — including the ranks that do not issue,
    /// because a multicast response completes transactions on *every* rank's
    /// copy and `expect_tx` is `.shared::cta`, so no rank can charge another's.
    ///
    /// Nothing orders this against [`Self::issue`] and nothing has to: an
    /// mbarrier's transaction count is a signed accumulator, so a response that
    /// lands before the charge that expects it leaves the same total.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::expect_tx`]; exactly one charge per request, and the
    /// request must actually be issued or this barrier never completes.
    #[inline(always)]
    unsafe fn charge(self) {
        unsafe {
            self.sem
                .expect_tx(TransactionBytes::new(Self::RESPONSE_BYTES))
        }
    }

    /// Ask the hardware to cancel a cluster the scheduler has not launched yet.
    /// **One thread of the whole cluster**, after every rank has charged.
    ///
    /// # Safety
    ///
    /// The kernel must be a cluster launch on sm_100a, every rank must hold
    /// this queue at the same shared offset, and no request may be outstanding.
    #[inline(always)]
    unsafe fn issue(self) {
        unsafe { clc_try_cancel_multicast(self.response as *mut u8, self.sem.raw()) }
    }

    /// Wait out the outstanding request and decode it: the item a cancelled
    /// cluster would have run, or `None` when there was nothing left to cancel
    /// and this cluster is done.
    ///
    /// Every thread of every rank calls it and every one gets the same answer,
    /// which is what the multicast is for. The item is the *cluster* index
    /// behind the CTA coordinate the response names — the same map
    /// [`cluster::cluster_idx`] gives, read back off the grid.
    ///
    /// # Safety
    ///
    /// A request must be outstanding, `parity` must follow the barrier's phase,
    /// and the grid must be one-dimensional — the response is a CTA coordinate
    /// and only a 1-D grid makes `ctaid.x` the whole of it.
    #[inline(always)]
    unsafe fn harvest(self, parity: u32) -> Option<u32> {
        unsafe {
            self.sem.wait(parity);
            // Written by the async proxy and read generically; the barrier
            // phase is what makes it visible, and `read_volatile` is what stops
            // the two halves being hoisted across the request that refills them.
            let (low, high) = (
                self.response.read_volatile(),
                self.response.add(1).read_volatile(),
            );
            if clc_query_is_canceled(low, high) == 0 {
                None
            } else {
                Some(clc_query_get_first_ctaid_x(low, high) / cluster::cluster_nctaidX())
            }
        }
    }
}

/// Run `job` over the grid's own items, taking the first from this cluster's
/// launch position and every one after it from the hardware's pending queue.
///
/// There is no item count: the grid **is** the item count, one cluster per
/// item, and a cluster that gets no steal ran exactly the one it was launched
/// for. That is the whole of how this deletes a tuning constant — nothing here
/// needs to know how many clusters the device holds, because the ones it does
/// not hold are the ones that get stolen.
///
/// The request goes out after the item it will replace, not before it. The
/// module docs carry why, and it is the opposite of what #88 expected.
///
/// # Safety
///
/// Everything [`run`] requires, and four things it does not:
///
/// - **The grid is exactly one cluster per work item**, one-dimensional. A
///   cluster past the last item would run it, and a response naming a CTA in a
///   `y` or `z` the item map cannot see would decode to the wrong item.
/// - **[`Job::RANKS`] is the launch's cluster size.** Unlike [`run`], this is
///   not free: the response is multicast to every CTA of the cluster and read
///   by every thread of it, so the boundary that separates those reads from the
///   next request has to cover the same set. A `RANKS == 1` job under a
///   multi-CTA cluster launch would take a `bar.sync` there and race.
/// - **`queue` is live, exclusive, and at the same offset in every rank**, per
///   [`ClcQueue::attach`].
/// - **The device is sm_100a and the launch is a cluster launch**, which
///   `clusterlaunchcontrol` requires and nothing here can check.
#[inline(always)]
pub unsafe fn run_stealing<J: Job>(job: &mut J, queue: ClcQueue) {
    unsafe {
        let leader = thread::threadIdx_x() == 0;
        let issuer = leader && cluster::block_rank() == 0;
        let mut item = cluster::cluster_idx();
        let mut parity = 0u32;

        if leader {
            queue.arm();
        }
        // Every rank's barrier has to be armed before any rank asks the
        // hardware to complete transactions on it.
        boundary::<J>();

        loop {
            if leader {
                job.init(item);
                fence_proxy_async_shared_cta();
            }
            boundary::<J>();
            job.work(item);
            tcgen05_fence_before_thread_sync();
            // The request goes out here, once the item it will replace is done,
            // rather than ahead of it. See the module docs: issuing it early
            // hides the latency and loses more than the latency is worth.
            if leader {
                queue.charge();
            }
            if issuer {
                queue.issue();
            }
            // Harvested before the boundary, so the boundary that retires the
            // item is also what says every reader is done with the response.
            let stolen = queue.harvest(parity);
            parity ^= 1;
            boundary::<J>();
            if leader {
                job.inval();
            }
            let Some(next) = stolen else { break };
            item = next;
        }

        if leader {
            queue.disarm();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tile once, at every width, including the ones that leave a short
    /// last group.
    ///
    /// This is the whole correctness argument for [`grouped`], and it is worth
    /// saying why it is stated as a bijection rather than as a formula. The
    /// device checks nothing, both item loops hand out a *permutation* of
    /// `0..items`, and the two ways this map can be wrong — a tile computed
    /// twice and a tile computed by nobody — are the same failure seen from
    /// either end. `covered` counting to exactly one everywhere is both claims
    /// at once, and it is what a ragged last group breaks first.
    #[test]
    fn grouped_covers_every_tile_exactly_once() {
        for rows in 1..=17u32 {
            for columns in 1..=17u32 {
                for group in 1..=19u32 {
                    let mut covered = vec![0u32; (rows * columns) as usize];
                    for item in 0..rows * columns {
                        let (row, column) = grouped(item, rows, columns, group);
                        assert!(row < rows && column < columns, "{rows}x{columns} @ {group}");
                        covered[(row * columns + column) as usize] += 1;
                    }
                    assert!(
                        covered.iter().all(|&hits| hits == 1),
                        "{rows}x{columns} @ {group}: {covered:?}"
                    );
                }
            }
        }
    }

    /// Width 1 is the row-major map it generalizes, so a sweep's control row is
    /// the traversal the kernel had before there was a parameter.
    #[test]
    fn width_one_is_row_major() {
        for rows in 1..=9u32 {
            for columns in 1..=9u32 {
                for item in 0..rows * columns {
                    assert_eq!(
                        grouped(item, rows, columns, 1),
                        (item / columns, item % columns)
                    );
                }
            }
        }
    }

    /// A width at or past the tile grid's height is the fully column-major
    /// walk, whatever it is nominally set to — which is what makes an
    /// over-large width a saturating parameter rather than an illegal one, and
    /// is why a sweep may carry widths taller than some of its shapes.
    #[test]
    fn width_past_the_grid_saturates_to_column_major() {
        for rows in 1..=9u32 {
            for columns in 1..=9u32 {
                for group in rows..rows + 5 {
                    for item in 0..rows * columns {
                        assert_eq!(
                            grouped(item, rows, columns, group),
                            (item % rows, item / rows)
                        );
                    }
                }
            }
        }
    }
}
