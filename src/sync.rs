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
//! [`block_reduce`] is the other thing a kernel needs the block to agree on,
//! and it is not a barrier idiom but a *collective*: warps cannot shuffle to
//! each other, so a statistic spanning them is staged through shared memory
//! and folded, with `sync_threads` on both sides. It lives here rather than in
//! [`crate::reg`] because what it is made of is the barrier and the staging
//! buffer; the register half of the same fold —
//! [`crate::reg::RegTile::tile_reduce`] and the `shuffle_xor` butterflies — is
//! warp scope and stops at 32 lanes.

use cuda_device::barrier::{
    Barrier, mbarrier_arrive, mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval,
    mbarrier_try_wait_parity,
};
use cuda_device::{thread, warp};

use crate::reg::{Add, ReduceOp};
use crate::shared::{F32, SharedVec};

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

    /// Count one arrival and register `bytes` of expected TMA transactions —
    /// the producer side of a `load → wait` handoff. The issuing thread
    /// charges every byte its `cp.async.bulk.tensor` calls will complete
    /// against this barrier.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::arrive`]; `bytes` must equal the bytes
    /// actually in flight or the phase never completes.
    #[inline(always)]
    pub unsafe fn expect_tx(self, bytes: u32) {
        unsafe {
            mbarrier_arrive_expect_tx(self.bar, 1, bytes);
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

    /// The raw barrier word, for intrinsics that consume one directly
    /// (`tcgen05_commit_shared_cluster`, TMA loads).
    #[inline(always)]
    pub const fn raw(self) -> *mut Barrier {
        self.bar
    }

    /// The barrier's cluster-multicast alias: a multicast TMA load completes
    /// on *each* receiving CTA's copy of the barrier, addressed by masking
    /// the issuing CTA's rank bit (and sub-word bits) out of the local
    /// address — the validated gemm cta_group::2 idiom. Pass the result to
    /// [`crate::shared::SharedTile::tma_load_2d_multicast_cg2`].
    #[inline(always)]
    pub fn multicast_alias(self) -> Semaphore {
        Semaphore {
            bar: ((self.bar as u32) & 0xFEFF_FFF8) as *mut Barrier,
        }
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
/// `WARPS` has to be a multiple of four, and not because of anything here:
/// [`SharedVec`]'s `BOX_OK` wants a whole number of the TMA's 16-byte lines,
/// which at 4 bytes an element makes 4 warps — 16 bytes exactly — the narrowest
/// scratch that constructs. A 128-thread block is the smallest one that can
/// take this reduction, and a 1- or 2-warp block fails at codegen inside
/// [`SharedVec::from_raw`]. The rule is a statement about a TMA box, and a
/// scratch vector is the one use of the type that never meets the engine.
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
/// - **The block is 1-D and exactly `WARPS * 32` threads.** `warp::warp_id()`
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
///   `fence.proxy.async.shared::cta` is owed — that fence is for a TMA or an
///   MMA reading the same bytes, which is a use this collective does not have.
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
        if warp::lane_id() == 0 {
            scratch.set(warp::warp_id() as usize, value);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    /// `warp::warp_id()` are device-only, so this is as far as the host reaches
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
    }

    #[test]
    fn the_fold_starts_from_the_identity() {
        // What lets one loop serve every width, including the one-warp block
        // where there is nothing to fold against.
        assert_eq!(fold::<Mul, 4>([1.0, 2.0, 4.0, 8.0]), 64.0);
        assert_eq!(fold::<Max, 4>([-3.0, -1.0, -7.0, -2.0]), -1.0);
        assert_eq!(fold::<Max, 8>([f32::NEG_INFINITY; 8]), f32::NEG_INFINITY);
    }

    #[test]
    fn the_widths_a_partials_vector_can_have() {
        // Not every warp count is expressible, and the reason has nothing to do
        // with the fold: `SharedVec::BOX_OK` wants a whole number of the TMA's
        // 16-byte lines, so at 4 bytes an element `WARPS` must be a multiple of
        // four. A 4-warp block — `groupnorm_tile`'s, and every 128-thread
        // kernel's — is the smallest legal one, at 16 bytes exactly; a 1- or
        // 2-warp block cannot construct the scratch at all, and would fail at
        // codegen inside `SharedVec::from_raw` rather than here.
        assert_eq!(SharedVec::<F32, 4>::BYTES, 16);
        assert!(SharedVec::<F32, 4>::BYTES.is_multiple_of(16));
        assert!(!(size_of::<f32>() * 2).is_multiple_of(16));
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
