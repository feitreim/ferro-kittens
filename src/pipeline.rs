//! Persistent-grid harness: a loop over work items that calls back into a
//! [`Job`].
//!
//! Every mbarrier is re-initialized per item behind the item boundary, so each
//! item's phase arithmetic starts from zero and unbalanced arrivals are wiped
//! rather than threaded through parity math. ThunderKittens calls this shape
//! `prototype::lcf`.
//!
//! An item belongs to a *cluster*, not a CTA: [`run`] walks `%clusterid` by
//! `%nclusterid`, which under a launch declaring no cluster is exactly
//! `blockIdx.x` by `gridDim.x`. [`run_stealing`] is the same loop over the
//! hardware's own pending queue. [`grouped`] says what an item *is* for a job
//! tiling a 2-D output, and composes with either loop.
//!
//! How many clusters the grid has is the whole performance story; that, and
//! what the dynamic schedule measured, are in `docs/library/pipeline.md`.
//!
//! ```no_run
//! # use kittens::pipeline::{Job, grouped, run};
//! # use kittens::sync::SemaphoreRing;
//! struct Tile {
//!     stages: SemaphoreRing<3>,
//!     columns: u32,
//! }
//!
//! impl Job for Tile {
//!     #[inline(always)]
//!     unsafe fn init(&self, _item: u32) {
//!         unsafe { self.stages.init_all(1) }
//!     }
//!     #[inline(always)]
//!     unsafe fn inval(&self) {
//!         unsafe { self.stages.inval_all() }
//!     }
//!     #[inline(always)]
//!     unsafe fn work(&mut self, item: u32) {
//!         let (_row, _column) = grouped(item, 64, self.columns, 8);
//!         // ... produce, consume, drain ...
//!     }
//! }
//!
//! # unsafe fn launch(job: &mut Tile, items: u32) {
//! unsafe { run(job, items) }
//! # }
//! ```

use cuda_device::barrier::Barrier;
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::thread;

use crate::shared::publish_to_async_proxy;
use crate::sync::{Semaphore, TransactionBytes};

/// One persistent kernel's work, split at the points the scaffold owns.
///
/// Implementations are plain structs of tile/semaphore handles built once
/// before [`run`]; every method must be `#[inline(always)]` so the job
/// scalarizes into the kernel like the hand-written loop it replaces.
///
/// TMEM allocation stays outside the loop — it spans items — so a job may also
/// end `work` with its accumulator undrained and drain it at the top of the
/// next item, which is a load-compute-**store**-finish pipeline written inside
/// a job rather than a second scaffold. Doing that moves one obligation to the
/// job — the cross-item drain needs a rendezvous covering every CTA the next
/// item's MMA writes — and one to [`Job::finish`], which is where the last
/// item's deferred store goes and which both loops call for the job.
///
/// # What this trait deliberately cannot express
///
/// A **warp-asynchronous item stream**: warp groups of the same cluster sitting
/// at *different* items at once. `work` is entered by every thread with one
/// `item`, so every warp is always on the same item and a job's whole slack is
/// one item — an epilogue can lag its MMA by exactly one, which is what
/// [`Job::finish`] then owes. A kernel that publishes `(row, column)` to its
/// consumers through a shared word and an mbarrier of its own, and runs three
/// warp groups two items apart, is not a `Job` and cannot be made into one by
/// anything in this file.
///
/// That is the price of the guarantee the scaffold exists for, and it is the
/// same fact from both ends: the item boundary is where every mbarrier is
/// re-armed, and it is a rendezvous of the whole cluster, so the producer warps
/// provably cannot run ahead into the next item. Take the boundary away and
/// phase parity has to be threaded from item to item by hand; leave it and the
/// warps are in lockstep. There is no third position, so a scaffold admitting a
/// warp-asynchronous stream is a *second* scaffold beside this one rather than a
/// change to it — different barrier lifecycle, and a job written against either
/// cannot run under the other.
///
/// It is a decision and not an omission because the thing given up is priced:
/// #90 and #94 measured the item boundary at a fifth to a third of the launch,
/// which is what a design with more slack is competing for, and
/// `docs/library/pipeline.md` carries the table. The limit was found by porting
/// a warp-specialized reference kernel onto this trait and recorded here by
/// #132.
pub trait Job {
    /// CTAs of the cluster that share this job's barrier set, and so the scope
    /// [`run`]'s item boundary has to be taken at. The default is the block-scope
    /// job: one CTA per item, no peer with an interest in when the barriers are
    /// re-armed.
    ///
    /// This is not the launch's cluster size. A cluster whose CTAs never name
    /// each other's barriers is still `1` here, and pays a `bar.sync` per item
    /// rather than a `barrier.cluster`.
    ///
    /// It has to be right because a cluster's CTAs write to *each other's*
    /// barriers: a leader invalidating on its own block's schedule can wipe a
    /// barrier a peer is still filling, and a peer can fill a barrier its leader
    /// has not re-armed. Both are silent — the phase simply never completes.
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

    /// Whatever the job's last item still owes, once there is no next item to
    /// overlap it with. Called by [`run`] and [`run_stealing`] after their item
    /// loops, by every thread that called them, exactly once — including in a
    /// cluster the schedule gave no items to, which a job that defers has to
    /// answer for itself and both of this repo's do with a sentinel. The default
    /// is empty, so a job that defers nothing pays nothing for the hook.
    ///
    /// A job that defers its last item's store — see this trait's own doc, and
    /// `Lcsf` in `experiments/` — owes one drain here, and it is a whole output
    /// tile computed by nobody if it is forgotten. That is the reason this hook
    /// is on the trait rather than on the entry point: before #132 one kernel's
    /// nine entry points were nine places to forget it, and forgetting it is a
    /// wrong `C` that no type said anything about.
    ///
    /// It runs *after* the loop's last `inval` — the barrier set the job
    /// re-arms per item is retired by then, and a store that outlives the last
    /// item cannot be one of its readers — and **before** the caller's
    /// `dealloc_*`, since the tensor memory a deferred drain reads is memory the
    /// caller is about to give back.
    ///
    /// # Safety
    ///
    /// Kernel-specific, as [`Self::work`]'s. The default does nothing and has
    /// no contract.
    #[inline(always)]
    unsafe fn finish(&mut self) {}
}

/// Map a work item to a `(row, column)` of a `rows`x`columns` tile grid,
/// walking `group` tile-rows at a time.
///
/// `group == 1` is the row-major map `(item / columns, item % columns)`. Above 1
/// the walk fills `group` rows of tiles down the column before moving right,
/// which keeps a row-panel of the left operand resident across the columns that
/// re-read it. A width at or past `rows` saturates to the fully column-major
/// walk rather than being illegal.
///
/// What a width is worth is a property of the shape and not a constant — take it
/// as a parameter and measure. `docs/library/pipeline.md` has the argument.
///
/// ```
/// # use kittens::pipeline::grouped;
/// // Two tile-rows at a time down a 4x3 grid.
/// let walk: Vec<_> = (0..12).map(|item| grouped(item, 4, 3, 2)).collect();
/// assert_eq!(walk[..4], [(0, 0), (1, 0), (0, 1), (1, 1)]);
/// // Still a bijection when the group does not divide `rows`.
/// let mut tiles: Vec<_> = (0..12).map(|item| grouped(item, 4, 3, 3)).collect();
/// tiles.sort();
/// tiles.dedup();
/// assert_eq!(tiles.len(), 12);
/// ```
///
/// The map is a bijection for every `group >= 1`, including one that does not
/// divide `rows`: the last group is short, and its items are laid out over the
/// rows it actually has rather than over the rows it would have had. Without
/// that the tail of the grid would alias, which is a wrong `C` and not a slow
/// one.
///
/// **Preconditions, none of them checked**: `group`, `rows` and `columns` at
/// least 1, and `item` less than `rows * columns`. This is ordinary arithmetic
/// and cannot itself be unsound, but a caller that breaks them gets a coordinate
/// off its own tile grid — so on the device the fault lands in whatever `unsafe`
/// addresses `C` with the answer, which is why the launcher rejects a zero width
/// rather than passing it through.
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
/// per cluster.
///
/// Owns the barrier lifecycle (leader-only inval-then-init, published by a proxy
/// fence and the item boundary) and the per-item tcgen05 fence pairing: `work`
/// returns with its MMAs committed, the harness fences them before the boundary.
/// TMEM allocation stays with the caller, since it spans items and
/// [`crate::tmem::alloc_cluster`] is a whole-cluster collective that must not be
/// inside the loop — so a job deferring its last item's store owes that drain to
/// [`Job::finish`], which this calls before returning and the caller therefore
/// cannot forget.
///
/// # Safety
///
/// - Every thread of every CTA of the cluster calls this together.
/// - `job`'s barrier storage is unused by anything else for the duration, and
///   laid out at the same shared offset in every rank.
/// - `items` is the same in every CTA.
/// - The grid is a whole number of clusters — which a `#[cluster_launch]`
///   kernel's prepared launch already checks.
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
                publish_to_async_proxy();
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
        job.finish();
    }
}

/// The hardware's work queue, as the shared memory a cluster asks it through:
/// the 16-byte `clusterlaunchcontrol.try_cancel` response and the mbarrier it is
/// delivered on.
///
/// It is the caller's storage rather than the scaffold's for the same reason a
/// [`Job`]'s barriers are — a kernel owns one shared plan and nothing else may
/// carve out of it — and it must sit at the *same offset in every rank*, which
/// the multicast response requires and a symmetric plan gives for free.
///
/// ```no_run
/// # use kittens::pipeline::run_stealing;
/// # use kittens::plan::SharedPlan;
/// # unsafe fn launch<J: kittens::pipeline::Job>(job: &mut J) { unsafe {
/// let (queue, _plan) = SharedPlan::attach().clc_queue();
/// run_stealing(job, queue);
/// # } }
/// ```
#[derive(Clone, Copy)]
pub struct ClcQueue {
    response: *mut u64,
    sem: Semaphore,
}

impl ClcQueue {
    /// The response's size, which is the ISA's (`.b128`) and is also exactly
    /// the transaction count its mbarrier is charged.
    ///
    /// Crate-visible so [`crate::plan::SharedPlan::clc_queue`] can reserve the
    /// response and the barrier as the two objects they are.
    pub(crate) const RESPONSE_BYTES: usize = 16;
    /// Shared bytes to reserve: the response, then the barrier under it.
    pub const BYTES: usize = Self::RESPONSE_BYTES + 8;
    /// The response is a 128-bit store, so the base is 16-byte aligned.
    pub const ALIGNMENT: usize = 16;

    /// Lay a queue over [`Self::BYTES`] of shared memory — the raw form.
    ///
    /// A kernel that has a [`crate::plan::SharedPlan`] should take
    /// [`crate::plan::SharedPlan::clc_queue`] instead, which reserves the two
    /// objects at their own alignments and owes no proof that
    /// [`Self::ALIGNMENT`] was met.
    ///
    /// # Safety
    ///
    /// `base` must point to [`Self::BYTES`] of shared memory aligned to
    /// [`Self::ALIGNMENT`], used by nothing else for the kernel's duration, and
    /// at the same offset in every CTA of the cluster.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        unsafe { Self::from_parts(base, base.add(Self::RESPONSE_BYTES) as *mut Barrier) }
    }

    /// The queue as the two objects it is — the `.b128` response, and the
    /// mbarrier its multicast delivery completes on — placed separately.
    ///
    /// [`crate::plan::SharedPlan::clc_queue`] is the caller: a plan walked for
    /// its *size* has no allocation to offset a pointer inside of, so a
    /// constructor deriving one address from the other cannot be
    /// const-evaluated on the host side.
    ///
    /// # Safety
    ///
    /// As [`Self::attach`], with `sem` [`Self::RESPONSE_BYTES`] past
    /// `response` and both inside the same reservation.
    #[inline(always)]
    pub(crate) const unsafe fn from_parts(response: *mut u8, sem: *mut Barrier) -> Self {
        Self {
            response: response as *mut u64,
            sem: unsafe { Semaphore::attach(sem) },
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
    /// The request is `…multicast::cluster::all`, and the multicast is not an
    /// optimization: a 2-CTA cluster whose halves each stole would put the pair
    /// on two different output tiles, silently. One response into every CTA is
    /// what makes the next item a fact the cluster agrees on by construction.
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
    /// - A request must be outstanding and `parity` must follow the barrier's
    ///   phase.
    /// - The grid must be one-dimensional: the response is a CTA coordinate, and
    ///   only a 1-D grid makes `ctaid.x` the whole of it.
    #[inline(always)]
    unsafe fn harvest(self, parity: u32) -> Option<u32> {
        unsafe {
            self.sem.wait(parity);
            self.harvested()
        }
    }

    /// [`Self::harvest`] with a deadline on the response, per
    /// [`Semaphore::wait_before`].
    ///
    /// # Safety
    ///
    /// As [`Self::harvest`]. A [`Harvest::Stalled`] leaves the request
    /// outstanding, so the only sound thing to do with one is stop querying.
    #[inline(always)]
    unsafe fn harvest_before(self, parity: u32, ticks: u64) -> Harvest {
        unsafe {
            if !self.sem.wait_before(parity, ticks) {
                return Harvest::Stalled;
            }
            match self.harvested() {
                Some(item) => Harvest::Item(item),
                None => Harvest::Done,
            }
        }
    }

    /// Decode a response the barrier has already published.
    ///
    /// # Safety
    ///
    /// The response's phase must have completed.
    #[inline(always)]
    unsafe fn harvested(self) -> Option<u32> {
        unsafe {
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

    /// A stateful producer-side cursor for kernels whose warp roles advance
    /// independently and therefore cannot use [`run_stealing`].
    #[inline(always)]
    pub const fn cursor(self) -> ClcCursor {
        ClcCursor {
            queue: self,
            parity: 0,
        }
    }
}

/// What a bounded CLC query came back with: another item, an empty queue, or a
/// response that never arrived within the deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harvest {
    Item(u32),
    Done,
    Stalled,
}

/// Producer-owned view of a [`ClcQueue`], including its response phase.
///
/// Exactly one designated thread per CTA drives it; both ranks participate in
/// charging and harvesting, while rank zero alone issues the multicast query.
///
/// ```no_run
/// # use kittens::pipeline::ClcQueue;
/// # unsafe fn produce(queue: ClcQueue, first: u32) { unsafe {
/// let mut cursor = queue.cursor();
/// cursor.arm();
/// let mut item = Some(first);
/// while let Some(current) = item {
///     // ... issue this item's loads ...
///     let _ = current;
///     item = cursor.next();
/// }
/// cursor.disarm();
/// # } }
/// ```
pub struct ClcCursor {
    queue: ClcQueue,
    parity: u32,
}

impl ClcCursor {
    /// Initialize this CTA's response barrier once before the role loops begin.
    ///
    /// # Safety
    ///
    /// One designated thread per CTA, before any query and behind a subsequent
    /// cluster-wide publication point.
    #[inline(always)]
    pub unsafe fn arm(&self) {
        unsafe { self.queue.arm() }
    }

    /// Finish the current raw work item and request another from CLC.
    ///
    /// # Safety
    ///
    /// One designated thread per CTA. Both ranks must call once per request in
    /// the same order; no earlier response may still be read.
    #[inline(always)]
    pub unsafe fn next(&mut self) -> Option<u32> {
        unsafe {
            self.queue.charge();
            if cluster::block_rank() == 0 {
                self.queue.issue();
            }
            let next = self.queue.harvest(self.parity);
            self.parity ^= 1;
            next
        }
    }

    /// [`Self::next`] with a deadline on the response.
    ///
    /// [`ClcQueue`] is the one thing in a stealing item loop whose termination
    /// is the hardware's rather than the code's, so it is the one a diagnostic
    /// cannot leave unbounded.
    ///
    /// # Safety
    ///
    /// As [`Self::next`]. A [`Harvest::Stalled`] must end the loop: the parity
    /// is not advanced and the request is still outstanding.
    #[inline(always)]
    pub unsafe fn next_before(&mut self, ticks: u64) -> Harvest {
        unsafe {
            self.queue.charge();
            if cluster::block_rank() == 0 {
                self.queue.issue();
            }
            let next = self.queue.harvest_before(self.parity, ticks);
            if !matches!(next, Harvest::Stalled) {
                self.parity ^= 1;
            }
            next
        }
    }

    /// Invalidate the response barrier after the final query completes.
    ///
    /// # Safety
    ///
    /// One designated thread per CTA, with no request outstanding.
    #[inline(always)]
    pub unsafe fn disarm(&self) {
        unsafe { self.queue.disarm() }
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
/// The request goes out *after* the item it will replace. Prefetching it —
/// issuing the next item's request before the current item's `work` — measured
/// **8.5% slower at 16384³**, and `docs/library/pipeline.md` carries why.
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
                publish_to_async_proxy();
            }
            boundary::<J>();
            job.work(item);
            tcgen05_fence_before_thread_sync();
            // Once the item it will replace is done, rather than ahead of it:
            // issuing early hides the latency and loses more than it is worth.
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
        job.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tile once, at every width, including the ones that leave a short
    /// last group — the whole correctness argument for [`grouped`].
    ///
    /// It is stated as a bijection rather than as a formula because both item
    /// loops hand out a *permutation* of `0..items`, and the two ways this map
    /// can be wrong — a tile computed twice, a tile computed by nobody — are the
    /// same failure seen from either end. `covered` counting to exactly one
    /// everywhere is both claims at once, and it is what a ragged last group
    /// breaks first.
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

    /// A width at or past the tile grid's height is the fully column-major walk,
    /// whatever it is nominally set to — which makes an over-large width a
    /// saturating parameter rather than an illegal one, and is why a sweep may
    /// carry widths taller than some of its shapes.
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
