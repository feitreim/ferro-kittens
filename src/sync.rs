//! Semaphores over mbarrier intrinsics — the phase-parity idiom, first-class.
//!
//! cuda-oxide exposes no named barriers, so every cross-warp handoff in a
//! warp-specialized kernel is an mbarrier phase-parity spin (FA4 does the
//! same). The soundness rule these types encode: parity arithmetic works
//! because every barrier's completions lead its waiter by at most one phase —
//! each producer's next completion transitively requires the previous
//! consumer wait.
//!
//! A [`Semaphore`] is a stateless handle over one mbarrier word; a
//! [`SemaphoreRing`] owns the `index → (stage, parity)` arithmetic for the
//! `N`-deep producer/consumer rings that today thread parity bits by hand.
//!
//! # Cluster scope, and who charges the bytes
//!
//! A [`Semaphore`] is *one CTA's* barrier. A cta_group::2 MMA consumes an
//! operand stage of four tiles staged by two CTAs, so its issuer needs one
//! barrier saying the whole stage is present — which means a CTA has to be
//! able to name its peer's copy. [`Semaphore::at_rank`] does that, and it
//! hands back a [`ClusterSemaphore`] rather than another [`Semaphore`],
//! because a barrier that is not yours is not a barrier you may do everything
//! to.
//!
//! That restriction is the ISA's and not ours.
//! `mbarrier.arrive.expect_tx` takes `.shared::cta` only, and
//! `mbarrier.arrive … .shared::cluster` carries no transaction count: remote
//! addressing and byte accounting sit in different instructions and do not
//! compose. So a CTA may charge exactly one barrier — its own — and may
//! arrive at any rank's. [`ClusterSemaphore`] offers exactly that pair:
//! arrive, or hand the address to an engine that will complete transactions
//! on it. It has no `wait` and no `expect_tx` because there is no instruction
//! for either, and a type that admitted them would be promising something the
//! hardware does not do.
//!
//! What is left to decide is which CTA's barrier a cluster stage completes on,
//! and there are only two shapes:
//!
//! 1. **The waiter charges the whole stage.** One barrier, in the CTA that
//!    waits on it. Every rank's TMA completes there — the peers name it
//!    through [`Semaphore::at_rank`] — and that CTA charges every byte the
//!    stage will bring in, including the bytes it does not issue itself.
//! 2. **Every rank charges its own.** Each producer charges the bytes it
//!    issues against its own barrier, and forwards a
//!    [`ClusterSemaphore::arrive`] to the waiter once that barrier flips.
//!
//! This library takes (1). What (2) has going for it is that the sum stays
//! local — a CTA charges exactly what it issued, which is the property #29
//! wants — and that is all it has. It costs a barrier per rank per stage, a
//! thread per rank spinning on that barrier, and a second hop on the critical
//! path carrying no information: when a producer's own barrier flips it knows
//! nothing the TMA engine could not have told the waiter directly.
//!
//! The locality is recoverable under (1) anyway, which is what settles it. A
//! cluster stage is symmetric by construction — every rank stages the same
//! tile types at the same shared offsets, which is exactly what lets one MMA
//! descriptor read across the pair — so the whole-stage charge is the local
//! charge times the number of ranks staging into it, and the rank count is a
//! launch constant. A charge derived from the loads a producer issued is
//! still derivable; it is derived once and multiplied.
//!
//! One thing (1) looks like it should need and does not: ordering between the
//! charge and the peers' loads. An mbarrier's transaction count is a signed
//! accumulator, so completions may land before the `expect_tx` that expects
//! them — only the totals have to agree. No sync sits between them in the
//! GEMM and none is owed.
//!
//! # Where the transaction count comes from
//!
//! [`Semaphore::expect_tx`] takes a [`TransactionBytes`] and nothing else, and
//! the only way to obtain one is to issue the load that will deliver it. A
//! stage's charge is the sum of the calls that were made:
//!
//! ```ignore
//! let stage = k_ring.tile(i).tma_load(k_map, base, head, load.sem(i))
//!     + v_ring.tile(i).tma_load(v_map, base, head, load.sem(i));
//! load.sem(i).expect_tx(stage);
//! ```
//!
//! Every producer used to write that total by hand — `(KTile::BYTES +
//! VTile::BYTES) as u32` — and keep it in step with the loads above it by
//! reading both. Under-charging flips the barrier before the last bytes land
//! and the consumer reads a half-written tile, with no fault and no
//! diagnostic; over-charging hangs the block. Adding a load and forgetting the
//! sum is one edit, and nothing was watching for it.
//!
//! The lighter fix — an `expect_tiles(&[…])` taking the tile types — was
//! rejected because it is the same fact stated twice with nicer syntax. A list
//! can omit a tile exactly as a sum can omit a term. What has to agree with
//! the charge is the *set of loads issued*, so the charge has to come from
//! them and not from anything that restates them.
//!
//! Two things that fall out, and are the point:
//!
//! - **Writing a wrong number is not expressible.** [`TransactionBytes`] has
//!   no public constructor and is not an integer, so there is nothing to
//!   mistype.
//! - **Issuing a load and dropping its charge is a compile error.** The type
//!   is `#[must_use]`, so `tile.tma_load(…);` as a statement fails under
//!   `-D warnings`, and a charge bound to a name that never reaches an
//!   `expect_tx` fails as an unused binding. It is also not `Copy`, so
//!   charging the same bytes to two barriers is a use of a moved value.
//!
//! What the type does not catch, stated plainly: it says how many bytes, never
//! which barrier. A load aimed at one semaphore whose charge is paid to
//! another still type-checks — and has to, because that is exactly what a
//! cluster stage does. The peers aim their loads at the leader through
//! [`Semaphore::at_rank`] and the leader charges its own barrier for all of
//! them, which is [`TransactionBytes::across_ranks`]: the local charge, from
//! each of the ranks staging into the barrier. It is the `RANKS *` of the
//! shape-(1) argument above, and it is derived from the loads this CTA issued
//! rather than written down, because a cluster stage is symmetric — that is
//! the same symmetry the MMA descriptor reads across the pair on.
//!
//! [`block_reduce`] is the other thing a kernel needs the block to agree on,
//! and it is not a barrier idiom but a *collective*: warps cannot shuffle to
//! each other, so a statistic spanning them is staged through shared memory
//! and folded, with `sync_threads` on both sides. It lives here rather than in
//! [`crate::reg`] because what it is made of is the barrier and the staging
//! buffer; the register half of the same fold —
//! [`crate::reg::RegTile::tile_reduce`] and the `shuffle_xor` butterflies — is
//! warp scope and stops at 32 lanes.

use cuda_device::barrier::{
    Barrier, mbarrier_arrive, mbarrier_arrive_cluster, mbarrier_arrive_expect_tx, mbarrier_init,
    mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::cluster::map_shared_rank_mut;
use cuda_device::thread;

use crate::reg::{Add, ReduceOp};
use crate::shared::{F32, SharedVec};

/// Transaction bytes owed to an mbarrier, and the only thing
/// [`Semaphore::expect_tx`] will take.
///
/// Handed back by every TMA load in [`crate::shared`] and constructible no
/// other way, so a charge is a receipt for transfers that were issued rather
/// than a claim about them. `+` sums the loads of one stage;
/// [`Self::across_ranks`] scales a symmetric cluster stage. The module docs
/// carry the argument for why this is a type and not a `u32`.
///
/// Deliberately neither `Copy` nor droppable-in-silence: the lints are half of
/// what it is for.
#[must_use = "a load's transaction bytes must reach an expect_tx or its barrier never completes"]
pub struct TransactionBytes(u32);

impl TransactionBytes {
    /// The charge for `bytes` bytes of a transfer being issued now. Private to
    /// the crate: a kernel that could call this could write down any number,
    /// which is the whole of what this type exists to prevent.
    #[inline(always)]
    pub(crate) const fn new(bytes: usize) -> Self {
        Self(bytes as u32)
    }

    /// This charge from each of `ranks` ranks — a cluster stage's whole cost,
    /// as the one CTA that may charge a barrier has to state it.
    ///
    /// Derived and not restated: every rank of a cluster stages the same tile
    /// types at the same shared offsets, so the stage's total is the local
    /// total scaled by the rank count. See the module docs for who charges and
    /// why it cannot be the CTA that issued the bytes.
    #[inline(always)]
    pub const fn across_ranks(self, ranks: u32) -> Self {
        Self(self.0 * ranks)
    }

    /// The count itself, for tests and for intrinsics below this abstraction.
    #[inline(always)]
    pub const fn bytes(&self) -> u32 {
        self.0
    }
}

/// The sum of two stages' loads. Associativity is the transaction counter's,
/// not this type's: the barrier accumulates whatever order the charges and the
/// completions reach it in.
impl core::ops::Add for TransactionBytes {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// One mbarrier, addressed as the 64-bit state word it is. Handles are Copy
/// and carry no phase: parity comes from the caller's tile index (see
/// [`SemaphoreRing`]) so the same storage can back producer and consumer
/// handles at different pipeline positions.
#[derive(Clone, Copy)]
pub struct Semaphore {
    bar: *mut Barrier,
}

impl Semaphore {
    /// Wrap an mbarrier word (typically one slot of a `SharedArray<u64, N, 8>`
    /// static cast to `*mut Barrier`).
    ///
    /// # Safety
    ///
    /// `bar` must point to shared memory that lives as long as every use of
    /// the returned handle.
    #[inline(always)]
    pub const unsafe fn attach(bar: *mut Barrier) -> Self {
        Self { bar }
    }

    /// Initialize with `arriving` expected arrivals per phase. One thread
    /// per barrier, before any use, behind a block sync.
    ///
    /// # Safety
    ///
    /// Must not race any other access to the barrier.
    #[inline(always)]
    pub unsafe fn init(self, arriving: u32) {
        unsafe { mbarrier_init(self.bar, arriving) }
    }

    /// Invalidate before the block exits (or between persistent work items),
    /// wiping whatever unbalanced arrivals the phase left behind.
    ///
    /// # Safety
    ///
    /// No thread may still be arriving at or waiting on the barrier.
    #[inline(always)]
    pub unsafe fn inval(self) {
        unsafe { mbarrier_inval(self.bar) }
    }

    /// Count one arrival.
    ///
    /// # Safety
    ///
    /// The barrier must be initialized and this arrival balanced by the
    /// phase's expected count.
    #[inline(always)]
    pub unsafe fn arrive(self) {
        unsafe {
            mbarrier_arrive(self.bar);
        }
    }

    /// Count one arrival and register the TMA transactions of `bytes` — the
    /// producer side of a `load → wait` handoff, taking the charges the loads
    /// themselves handed back.
    ///
    /// `bytes` is every byte that will complete *against this barrier*, which
    /// under a cluster is not the set the calling CTA issued: a peer's loads
    /// aimed here through [`Semaphore::at_rank`] are counted here, which is
    /// what [`TransactionBytes::across_ranks`] states and the module docs say
    /// why the charge cannot travel with them.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::arrive`]. The charges summed here must be
    /// exactly those of the loads completing on this barrier — the type makes
    /// them derived rather than written down, but it does not know which
    /// barrier a load was aimed at.
    #[inline(always)]
    pub unsafe fn expect_tx(self, bytes: TransactionBytes) {
        unsafe {
            mbarrier_arrive_expect_tx(self.bar, 1, bytes.bytes());
        }
    }

    /// Spin until the barrier completes the phase with the given parity.
    ///
    /// # Safety
    ///
    /// The barrier must be initialized, and `parity` must follow the
    /// one-phase-lead rule from the module docs.
    #[inline(always)]
    pub unsafe fn wait(self, parity: u32) {
        unsafe { while !mbarrier_try_wait_parity(self.bar, parity) {} }
    }

    /// [`Self::wait`] with a deadline: `false` is `ticks` of the SM clock spent
    /// on a phase that did not complete.
    ///
    /// This is the diagnostic for the failure mode [`Self::wait`] cannot report.
    /// A parity spin on a phase whose producer has already left is
    /// indistinguishable, from outside the launch, from a kernel that is merely
    /// slow — the launch never returns and every warp's position is lost. Bound
    /// each spin and the same launch terminates carrying where each warp was.
    ///
    /// `clock64` is per-SM and monotonic within a launch, and the subtraction
    /// wraps, so `ticks` is elapsed and never absolute. It is deliberately not a
    /// wall-clock bound: a caller wanting one should read
    /// [`cuda_device::debug::globaltimer`] itself and say so.
    ///
    /// # Safety
    ///
    /// As [`Self::wait`], with one clause relaxed and one added: the
    /// one-phase-lead rule may be *violated*, since that is what this exists to
    /// catch, and a caller that takes `false` and reads on anyway is reading
    /// memory no barrier has published.
    #[inline(always)]
    pub unsafe fn wait_before(self, parity: u32, ticks: u64) -> bool {
        unsafe {
            let start = cuda_device::debug::clock64();
            while !mbarrier_try_wait_parity(self.bar, parity) {
                if cuda_device::debug::clock64().wrapping_sub(start) > ticks {
                    return false;
                }
            }
            true
        }
    }

    /// The raw barrier word, for intrinsics that consume one directly
    /// (`tcgen05_commit_shared_cluster`, TMA loads).
    #[inline(always)]
    pub const fn raw(self) -> *mut Barrier {
        self.bar
    }

    /// This barrier as the CTA at `rank` holds it: the same shared offset, in
    /// another member of the cluster, addressed through `mapa`. The only thing
    /// one CTA of a cluster may hand another, and see the module docs for why
    /// what comes back can be arrived at but neither waited on nor charged.
    ///
    /// # Safety
    ///
    /// `rank` must name a CTA of the calling block's live cluster, holding an
    /// initialized barrier at this handle's offset. A symmetric shared plan
    /// gives that by construction — every rank lays its barriers out the same
    /// way — which is the same assumption the stage itself rests on.
    #[inline(always)]
    pub unsafe fn at_rank(self, rank: u32) -> ClusterSemaphore {
        ClusterSemaphore {
            bar: unsafe { map_shared_rank_mut(self.bar, rank) },
        }
    }
}

/// One rank's copy of a [`Semaphore`], named from another CTA of the same
/// cluster. It carries everything a CTA may do to a barrier that is not its
/// own and nothing else: arrive at it, or hand its address to an engine —
/// [`crate::shared::SharedTile::tma_load_2d_arriving_at`], an MMA commit —
/// that will complete transactions on it. `wait` and `expect_tx` are missing
/// because the ISA has no remote form of either, not because they were
/// overlooked.
#[derive(Clone, Copy)]
pub struct ClusterSemaphore {
    bar: *mut Barrier,
}

impl ClusterSemaphore {
    /// Count one arrival on the remote barrier — the peer-progress signal
    /// [`Semaphore::arrive`] cannot send, being `.shared::cta`. It carries no
    /// transaction bytes, which is the half of the split the module docs are
    /// about.
    ///
    /// Still nothing in this repo calls it, and #51 is worth recording as the
    /// case that looked like it would and did not. A persistent cluster
    /// pipeline needs the CTAs of a pair to agree at each work-item boundary,
    /// which is a *rendezvous*: every rank arriving at every rank. Built out of
    /// this it is a barrier per rank and a spin per rank, hand-rolling
    /// `barrier.cluster.arrive`/`wait` — which is one instruction, is what
    /// [`cuda_device::cluster::cluster_sync`] issues, and is what
    /// [`crate::pipeline::run`] takes. What is left for this is the signal a
    /// cluster barrier is the wrong shape for: one named rank telling another
    /// that a specific thing has happened, while the rest of the cluster keeps
    /// going.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::arrive`], for the barrier at the far end.
    #[inline(always)]
    pub unsafe fn arrive(self) {
        unsafe { mbarrier_arrive_cluster(self.bar as u64) }
    }

    /// The raw cluster address, for intrinsics that consume a barrier operand
    /// directly.
    #[inline(always)]
    pub const fn raw(self) -> *mut Barrier {
        self.bar
    }
}

/// A semaphore that owns its phase counter: [`Self::wait_next`] spins out the
/// barrier's next completion and advances the parity, so a block-synchronous
/// kernel never spells `phase & 1` — the `tma_phase`/`mma_phase` locals the
/// backward kernels used to thread by hand. Every thread keeps its own copy in
/// registers and they advance in lockstep, which is exactly why those locals
/// worked; a kernel whose warps wait different numbers of times wants
/// [`SemaphoreRing`] (or explicit parities) instead.
#[derive(Clone, Copy)]
pub struct PhasedSemaphore {
    sem: Semaphore,
    phase: u32,
}

impl PhasedSemaphore {
    /// Wrap an mbarrier word at phase zero.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::attach`], and every thread that waits on
    /// this barrier must attach at the same point in the kernel.
    #[inline(always)]
    pub const unsafe fn attach(bar: *mut Barrier) -> Self {
        Self {
            sem: unsafe { Semaphore::attach(bar) },
            phase: 0,
        }
    }

    /// The stateless handle, for the phase-free producer side: `init`,
    /// `expect_tx`, `arrive`, `inval`, and MMA commits.
    #[inline(always)]
    pub const fn sem(self) -> Semaphore {
        self.sem
    }

    /// Spin until the barrier's next completion, then advance the phase.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::wait`].
    #[inline(always)]
    pub unsafe fn wait_next(&mut self) {
        unsafe { self.sem.wait(self.phase & 1) }
        self.phase += 1;
    }
}

/// `N` semaphores backing an `N`-stage pipeline ring, with the
/// `index → (stage, parity)` arithmetic in one place: tile `i` uses stage
/// `i % N`, whose barrier completes once per `N` tiles, so tile `i`'s
/// completion carries parity `(i / N) & 1`.
#[derive(Clone, Copy)]
pub struct SemaphoreRing<const N: usize> {
    base: *mut Barrier,
}

impl<const N: usize> SemaphoreRing<N> {
    /// Wrap `N` consecutive mbarrier words.
    ///
    /// # Safety
    ///
    /// `base` must point to `N` barrier words living as long as every use.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut Barrier) -> Self {
        Self { base }
    }

    /// The semaphore of tile `index`'s stage.
    #[inline(always)]
    pub fn sem(self, index: u32) -> Semaphore {
        unsafe { Semaphore::attach(self.base.add(index as usize % N)) }
    }

    /// Consumer wait: spin until tile `index`'s stage completes its
    /// `(i / N) & 1` phase.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::wait`].
    #[inline(always)]
    pub unsafe fn wait(self, index: u32) {
        unsafe { self.sem(index).wait((index / N as u32) & 1) }
    }

    /// Producer wait for a recycled slot: a producer running the full `N`
    /// stages ahead of its consumer fills tile `index`'s stage only after the
    /// consumer's release from `N` tiles ago — the previous ring cycle, hence
    /// parity `(i / N - 1) & 1`. The first `N` tiles fill virgin slots and
    /// skip the wait.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::wait`]; the consumer must release this
    /// ring exactly once per tile consumed.
    #[inline(always)]
    pub unsafe fn wait_recycled(self, index: u32) {
        unsafe {
            if index as usize >= N {
                self.sem(index).wait((index / N as u32).wrapping_sub(1) & 1);
            }
        }
    }

    /// [`Self::wait`] with a deadline, per [`Semaphore::wait_before`].
    ///
    /// # Safety
    ///
    /// As [`Semaphore::wait_before`].
    #[inline(always)]
    pub unsafe fn wait_before(self, index: u32, ticks: u64) -> bool {
        unsafe { self.sem(index).wait_before((index / N as u32) & 1, ticks) }
    }

    /// [`Self::wait_recycled`] with a deadline, per [`Semaphore::wait_before`].
    /// The first `N` tiles skip the wait and so can never report a stall.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::wait_before`].
    #[inline(always)]
    pub unsafe fn wait_recycled_before(self, index: u32, ticks: u64) -> bool {
        unsafe {
            (index as usize) < N
                || self
                    .sem(index)
                    .wait_before((index / N as u32).wrapping_sub(1) & 1, ticks)
        }
    }

    /// Initialize all `N` barriers with the same expected arrival count.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::init`].
    #[inline(always)]
    pub unsafe fn init_all(self, arriving: u32) {
        unsafe {
            let mut stage = 0u32;
            while (stage as usize) < N {
                self.sem(stage).init(arriving);
                stage += 1;
            }
        }
    }

    /// Invalidate all `N` barriers.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::inval`].
    #[inline(always)]
    pub unsafe fn inval_all(self) {
        unsafe {
            let mut stage = 0u32;
            while (stage as usize) < N {
                self.sem(stage).inval();
                stage += 1;
            }
        }
    }
}

/// Fold one warp-uniform value per warp into one block-uniform value, through
/// `scratch` — the block-scope half of a reduction whose warp-scope half is
/// [`crate::reg::RegTile::tile_reduce`].
///
/// Warp `w` stages its value at element `w`, the block syncs, and *every*
/// thread folds all `WARPS` elements in index order from `Op::IDENTITY`. So the
/// result is bit-for-bit identical in every thread — a scalar operand a kernel
/// may use immediately, with no broadcast step and no lane that has to be
/// asked.
///
/// Two barriers, and the second one is the reusable part: on return no thread
/// is still reading `scratch`, so a second call on the same vector cannot
/// overtake the first's readers. Calling it twice in a row — mean, then
/// variance — needs nothing at the call site.
///
/// `WARPS` is free: any block width from one warp up. It was briefly a
/// multiple of four, because [`SharedVec`] forced its TMA box rules at
/// construction and four fp32 is the narrowest legal box — a rule about a
/// descriptor, binding a vector that never becomes one. Those asserts now sit
/// on the four transfer methods, so a two-warp block's 8-byte scratch is a
/// handle like any other. A one-warp block is legal too, and degenerate: the
/// fold is over a single slot and [`crate::reg::warp_reduce`] is the same
/// answer without the barriers.
///
/// The bound is [`ReduceOp`] rather than [`crate::reg::BinaryOp`] for the
/// reason `row_reduce` takes one: the warps' partials arrive in slot order,
/// which is not an order the caller chose, so a fold over them has to be
/// associative and commutative to mean anything. `Sub` and `Div` are not
/// members, and the identity is what lets the fold start the same way whatever
/// `WARPS` is.
///
/// # Safety
///
/// All five of these are ways to get a plausible wrong number rather than a
/// fault:
///
/// - **Every thread of the block calls it**, with the same `scratch`, the same
///   `WARPS` and the same `Op`. It contains `sync_threads`, so a thread that
///   skips the call hangs the block or lets a partial be read before it is
///   written.
/// - **The block is 1-D and exactly `WARPS * 32` threads.** [`crate::warp_id`]
///   is `threadIdx.x / 32`, so a `WARPS` smaller than the block's warp count
///   writes past the vector, and a larger one folds elements no warp wrote.
/// - **`value` is warp-uniform** — the same in all 32 lanes. This does *not*
///   fold across lanes first: the input is a [`crate::reg::RegTile::tile_sum`]
///   result, which is already the whole warp's, and folding it again would
///   count every lane's copy. A per-lane value must go through
///   [`crate::reg::warp_reduce`] first; passed in raw, only lane 0's is used
///   and the rest are silently dropped.
/// - **`scratch` is `WARPS` live fp32 elements** no other thread and no engine
///   is using across the call. Both the write and the read are ordinary shared
///   accesses through the generic proxy, so the barriers order them and no
///   [`crate::shared::publish_to_async_proxy`] is owed — that fence is for a
///   TMA or an MMA reading the same bytes, which is a use this collective does
///   not have.
/// - **The first call in a kernel owes a barrier only if something else wrote
///   `scratch` first.** The function's own precondition — no thread still
///   reading the vector — is what its trailing barrier establishes, so back to
///   back calls are self-sufficient and a preceding *unrelated* use of the same
///   memory is not.
#[inline(always)]
pub unsafe fn block_reduce<Op: ReduceOp, const WARPS: usize>(
    scratch: SharedVec<F32, WARPS>,
    value: f32,
) -> f32 {
    unsafe {
        if crate::lane() == 0 {
            scratch.set(crate::warp_id() as usize, value);
        }
        thread::sync_threads();
        let folded = fold_partials::<Op, WARPS>(scratch);
        thread::sync_threads();
        folded
    }
}

/// Every element of `scratch`, folded from [`ReduceOp::IDENTITY`] in index
/// order. The half of [`block_reduce`] with no barrier and no thread identity
/// in it, and therefore the half a host test can run: every thread executes
/// exactly this, over exactly these slots, in exactly this order, which is what
/// makes the result block-uniform bit for bit rather than merely equal.
///
/// # Safety
///
/// As [`SharedVec::get`], for every element.
#[inline(always)]
unsafe fn fold_partials<Op: ReduceOp, const WARPS: usize>(scratch: SharedVec<F32, WARPS>) -> f32 {
    unsafe {
        let mut folded = Op::IDENTITY;
        let mut warp = 0usize;
        while warp < WARPS {
            folded = Op::apply(folded, scratch.get(warp));
            warp += 1;
        }
        folded
    }
}

/// [`block_reduce`] over [`Add`] — the whole-tile statistic a group norm takes,
/// once each warp has folded its own band.
///
/// # Safety
///
/// As [`block_reduce`], every clause.
#[inline(always)]
pub unsafe fn block_reduce_sum<const WARPS: usize>(
    scratch: SharedVec<F32, WARPS>,
    value: f32,
) -> f32 {
    unsafe { block_reduce::<Add, WARPS>(scratch, value) }
}

/// How deep a mailbox ring the module's one-phase-lead rule needs, worked out by
/// exhaustive search over a persistent GEMM's own chain of back-pressure.
///
/// The rule at the top of this module — *every barrier's completions lead its
/// waiter by at most one phase* — is what makes parity arithmetic sound, and it is
/// a claim about a kernel's protocol that this crate's types cannot check. #149 is
/// what it costs when the claim is false: `examples/src/gemm_sol.rs` handed work
/// items across warps through **one** barrier and **one** cell, its producer could
/// lead the epilogue by four publications, and `4096x4096x1024` stopped returning.
/// The lead is not a divisibility property and not a `k` threshold — it is what a
/// chain of independent throttles permits, and the only honest way to get it is to
/// enumerate the interleavings.
///
/// So this is a model and it says so. It is `pub` because a kernel's shape
/// contract is where its handoff depth is decided, and deciding one should be an
/// assertion rather than an argument: give it the trip count and it gives back the
/// depth [`SemaphoreRing`] and [`crate::shared::SharedCellRing`] must be built at.
pub mod handoff {
    use std::collections::HashSet;

    /// Warp roles in a four-stage operand pipeline over a two-deep accumulator,
    /// which is `gemm_sol`'s `[256, N]` entry.
    ///
    /// - the **producer** publishes item `i`, then issues `k_blocks` loads for it,
    ///   throttled by the MMA warp's release of an operand stage `STAGES` blocks
    ///   back, then queries for the next item and publishes it;
    /// - the **MMA warp** waits for item `i`'s publication and for the epilogue to
    ///   have drained item `i - ACCUMULATORS`, then multiplies `k_blocks` blocks;
    /// - the **epilogue** waits for item `i`'s publication and for the MMA warp to
    ///   have filled it, then drains and releases it.
    ///
    /// Nothing else couples them, and the depth is entirely a consequence of that.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct State {
        published: u32,
        producer_item: u32,
        producer_block: u32,
        mma_item: u32,
        mma_block: u32,
        mma_seen: u32,
        epilogue_item: u32,
        epilogue_seen: u32,
        released: u32,
    }

    /// The depth a handoff must have for this chain: the largest number of
    /// publications that can be outstanding against an index a consumer has not
    /// read yet.
    ///
    /// A ring of that depth holds every message a consumer might still be owed;
    /// anything shallower loses one, and loses it as an overwritten cell or as a
    /// parity that no longer flips — the second of which is a launch that never
    /// returns.
    pub fn depth_needed(k_blocks: u32, items: u32, stages: u32, accumulators: u32) -> u32 {
        let start = State {
            published: 0,
            producer_item: 0,
            producer_block: 0,
            mma_item: 0,
            mma_block: 0,
            mma_seen: 0,
            epilogue_item: 0,
            epilogue_seen: 0,
            released: 0,
        };
        let mut seen = HashSet::new();
        let mut frontier = vec![start];
        seen.insert(start);
        let mut needed = 1;
        while let Some(state) = frontier.pop() {
            // A consumer about to wait for index `i` has `published - i`
            // publications outstanding against it, and that is the depth.
            for next in [state.mma_seen, state.epilogue_seen] {
                if next == state.mma_item || next == state.epilogue_item {
                    needed = needed.max(state.published.saturating_sub(next));
                }
            }
            for step in steps(state, k_blocks, items, stages, accumulators) {
                if seen.insert(step) {
                    frontier.push(step);
                }
            }
        }
        needed
    }

    /// Every transition enabled in `state` — one per role per barrier it can get
    /// past, with no ordering between the roles, which is what makes this a search
    /// over interleavings rather than a simulation of one.
    fn steps(
        state: State,
        k_blocks: u32,
        items: u32,
        stages: u32,
        accumulators: u32,
    ) -> Vec<State> {
        let mut out = Vec::new();
        let issued = state.producer_item * k_blocks + state.producer_block;
        let committed = state.mma_item * k_blocks + state.mma_block;

        // The producer publishes item `producer_item` before loading it.
        if state.published == state.producer_item && state.producer_item < items {
            out.push(State {
                published: state.published + 1,
                ..state
            });
        }
        // ... then issues its loads, one operand stage behind the MMA warp's
        // release of the stage `stages` blocks back.
        if state.published == state.producer_item + 1
            && (issued < stages || committed + stages > issued)
        {
            let block = state.producer_block + 1;
            out.push(State {
                producer_item: state.producer_item + u32::from(block == k_blocks),
                producer_block: if block == k_blocks { 0 } else { block },
                ..state
            });
        }
        // ... and publishes `has_work = false` when the queue is empty.
        if state.producer_item == items && state.published == items {
            out.push(State {
                published: state.published + 1,
                ..state
            });
        }

        if state.mma_seen == state.mma_item && state.published > state.mma_item {
            out.push(State {
                mma_seen: state.mma_seen + 1,
                ..state
            });
        }
        if state.mma_seen > state.mma_item
            && state.mma_item < items
            && (state.mma_item < accumulators || state.released + accumulators > state.mma_item)
            && issued > committed
        {
            let block = state.mma_block + 1;
            out.push(State {
                mma_item: state.mma_item + u32::from(block == k_blocks),
                mma_block: if block == k_blocks { 0 } else { block },
                ..state
            });
        }

        if state.epilogue_seen == state.epilogue_item && state.published > state.epilogue_item {
            out.push(State {
                epilogue_seen: state.epilogue_seen + 1,
                ..state
            });
        }
        if state.epilogue_seen > state.epilogue_item
            && state.epilogue_item < items
            && state.mma_item > state.epilogue_item
        {
            out.push(State {
                epilogue_item: state.epilogue_item + 1,
                released: state.released + 1,
                ..state
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The depth `gemm_sol`'s item handoff needs, and the shape it is worst at.
    ///
    /// This is the test that would have caught #149 and the reason the fix is
    /// four and not two: **the deepest lead is at the *shallowest* `k`**, which is
    /// the opposite of what a divisibility story predicts and is why every theory
    /// of the form "`k_blocks % X == 0` breaks it" was wrong. At
    /// `k_blocks == STAGES` the producer's own throttle only reaches back into the
    /// item before last, so the MMA warp is a whole item further ahead of the
    /// epilogue than it is at any deeper `k`, and one more publication is
    /// outstanding.
    #[test]
    fn the_item_handoff_needs_four_cells_and_the_shallowest_k_is_why() {
        // `k = 256`, the shallowest the shape contract admits.
        assert_eq!(handoff::depth_needed(4, 5, 4, 2), 4);
        // Every deeper `k` needs three, so a depth-two handoff was never sound
        // either and a depth-one handoff — the spelling #149 is about — was two
        // short of it at every legal shape.
        for k_blocks in [8, 12, 16, 20] {
            assert_eq!(handoff::depth_needed(k_blocks, 5, 4, 2), 3);
        }
    }

    /// A one-deep accumulator makes the chain one step tighter, which is
    /// `gemm_sol`'s `[512, 256]` entry: the same handoff at four cells covers it
    /// with a cell to spare, and that is why one shared plan serves both.
    #[test]
    fn a_shallower_accumulator_needs_a_shallower_handoff() {
        assert_eq!(handoff::depth_needed(4, 5, 4, 1), 3);
        assert_eq!(handoff::depth_needed(16, 5, 4, 1), 2);
    }

    /// One item per cluster is what every correctness gate ran before #149 — and
    /// it needs a depth of **two**, so the one-deep handoff was unsound at the
    /// gate's own shape as well.
    ///
    /// That is the part worth keeping: the producer's second publication is the
    /// `has_work = false` sentinel and nothing at all interlocks it against the
    /// epilogue's first read, so `1024x1024x512` was winning a race rather than
    /// being correct. What made it win by a mile is that the epilogue's first poll
    /// is a few dozen cycles after the cluster's launch barrier while the
    /// producer's sentinel is a whole item's TMA behind it. Shortening `k`
    /// shortens exactly that margin, which is the shape of a cliff at one value
    /// and not of a degradation.
    #[test]
    fn one_item_per_cluster_still_needs_two() {
        assert_eq!(handoff::depth_needed(4, 1, 4, 2), 2);
        assert_eq!(handoff::depth_needed(16, 1, 4, 2), 2);
    }

    #[test]
    fn ring_stage_addressing_wraps() {
        // Pointer math only — no barrier is ever touched.
        let base = 0x100usize as *mut Barrier;
        let ring = unsafe { SemaphoreRing::<3>::attach(base) };
        assert_eq!(ring.sem(0).raw(), base);
        assert_eq!(ring.sem(4).raw(), unsafe { base.add(1) });
        assert_eq!(ring.sem(6).raw(), base);
    }

    use crate::reg::{Max, Mul};

    /// `block_reduce`'s fold over a real four-element buffer. The barriers and
    /// [`crate::warp_id`] are device-only, so this is as far as the host reaches
    /// — but it is the half that decides the *number*, and the device case is
    /// what says the staging under it lands each warp's partial in its own
    /// slot.
    fn fold<Op: ReduceOp, const WARPS: usize>(partials: [f32; WARPS]) -> f32 {
        let scratch = unsafe { SharedVec::<F32, WARPS>::from_raw(partials.as_ptr() as *mut u8) };
        unsafe { fold_partials::<Op, WARPS>(scratch) }
    }

    #[test]
    fn the_fold_reads_every_slot_exactly_once() {
        // Powers of eight, so the total's base-8 digits say how many times each
        // slot was folded: a warp reading its own slot only, a slot read twice,
        // and a slot skipped are three different numbers rather than three ways
        // of being wrong.
        assert_eq!(fold::<Add, 4>([1.0, 8.0, 64.0, 512.0]), 585.0);
        assert_eq!(fold::<Add, 8>([1.0; 8]), 8.0);
        assert_eq!(fold::<Add, 1>([7.0]), 7.0);
        assert_eq!(fold::<Add, 2>([1.0, 8.0]), 9.0);
    }

    #[test]
    fn the_fold_starts_from_the_identity() {
        // What lets one loop serve every width, including the one-warp block
        // where there is nothing to fold against.
        assert_eq!(fold::<Mul, 4>([1.0, 2.0, 4.0, 8.0]), 64.0);
        assert_eq!(fold::<Max, 4>([-3.0, -1.0, -7.0, -2.0]), -1.0);
        assert_eq!(fold::<Max, 8>([f32::NEG_INFINITY; 8]), f32::NEG_INFINITY);
        // A one-warp block folds one slot against the identity — degenerate,
        // and the case that says the loop needs no lower bound.
        assert_eq!(fold::<Mul, 1>([5.0]), 5.0);
        assert_eq!(fold::<Max, 1>([-4.0]), -4.0);
    }

    #[test]
    fn every_warp_count_is_a_scratch_this_can_use() {
        // These four instantiations are the test. Two of them — one warp at 4
        // bytes and two at 8 — were codegen failures until `SharedVec`'s box
        // asserts moved off `from_raw` onto the transfers, and the rule that
        // rejected them is about a TMA descriptor a block reduction's scratch
        // never builds. Nothing about the fold ever cared.
        assert_eq!(SharedVec::<F32, 1>::BYTES, 4);
        assert_eq!(SharedVec::<F32, 2>::BYTES, 8);
        assert_eq!(SharedVec::<F32, 4>::BYTES, 16);
        assert_eq!(SharedVec::<F32, 8>::BYTES, 32);
        assert_eq!(fold::<Add, 1>([1.0]), 1.0);
        assert_eq!(fold::<Add, 2>([1.0, 1.0]), 2.0);
        assert_eq!(fold::<Add, 4>([1.0; 4]), 4.0);
        assert_eq!(fold::<Add, 8>([1.0; 8]), 8.0);
    }

    #[test]
    fn the_fold_is_bit_for_bit_the_same_whoever_runs_it() {
        // Block-uniformity is what `groupnorm_tile` uses the result as a scalar
        // operand on. Every thread folds the same slots in the same order, so
        // the claim is not "equal to within rounding" — these partials are
        // chosen so that a different summation order gives a different f32, and
        // the fold still has to give one answer.
        let partials = [1.0f32, 1.0, 1e8, -1e8];
        assert_eq!(fold::<Add, 4>(partials), 0.0);
        assert_eq!(partials.iter().rev().sum::<f32>(), 2.0);
    }
}
