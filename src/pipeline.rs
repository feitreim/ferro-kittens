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

use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::thread;

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
