//! A kernel's shared plan: what it carves out of the launch's dynamic shared
//! memory, in declaration order, as one program.
//!
//! [`SharedPlan`] is a cursor. It starts at the launch's base, hands out one
//! typed handle per reservation, aligns each to what *that handle's type*
//! requires, and accumulates the total as it goes — so the number the launch
//! declares is [`SharedPlan::bytes`] of the same walk that produced the
//! pointers, rather than a second expression asserted equal to it.
//!
//! Every method but [`SharedPlan::attach`] is a `const fn` and
//! [`SharedPlan::sizing`] is a cursor with no memory under it, so the whole
//! walk const-evaluates on the host — which is how a `#[launch_contract]`
//! gets its literal:
//!
//! ```
//! # use kittens::plan::SharedPlan;
//! # use kittens::shared::{Bf16, Swizzle128B};
//! const PLAN: SharedPlan = SharedPlan::sizing()
//!     .tile_ring::<Bf16, 128, 64, Swizzle128B, 3>()
//!     .1
//!     .semaphores::<3>()
//!     .1;
//! const BYTES: usize = PLAN.bytes();
//! assert_eq!(BYTES, 128 * 64 * 2 * 3 + 3 * 8);
//! ```
//!
//! Tiles, rings and vectors land at [`SharedPlan::TILE_ALIGN`] = 128,
//! semaphores at `align_of::<Barrier>()` = 8, [`SharedPlan::tmem_slot`] at 4,
//! and [`SharedPlan::clc_queue`] at [`ClcQueue::ALIGNMENT`] = 16. Each
//! reservation is aligned to its own type's rule, so the order of a walk
//! decides how many bytes the plan spends and never whether a handle is legal.
//!
//! Design notes, the rejected value-parameterized form, and the register-count
//! measurements: `docs/library/plan.md`.

use cuda_device::DynamicSharedArray;
use cuda_device::barrier::Barrier;

use crate::pipeline::ClcQueue;
use crate::shared::{Element, SharedTile, SharedTileRing, SharedVec, Swizzle};
use crate::sync::{Semaphore, SemaphoreRing};

/// A cursor over one launch's dynamic shared memory.
///
/// Built by [`Self::attach`] on the device and [`Self::sizing`] on the host;
/// the two run the same walk and differ only in whether there is memory under
/// it. Each reservation returns the handle **and the advanced cursor**, so a
/// walk reads as a chain of `let (handle, plan) = plan.something();` and the
/// compiler will not let a reservation be skipped or reused.
///
/// ```no_run
/// # use kittens::plan::SharedPlan;
/// # use kittens::shared::{Bf16, Swizzle128B};
/// # unsafe fn demo() {
/// let plan = unsafe { SharedPlan::attach() };
/// let (stages, plan) = plan.tile_ring::<Bf16, 128, 64, Swizzle128B, 3>();
/// let (filled, plan) = plan.semaphores::<3>();
/// let (slot, _) = plan.tmem_slot();
/// # let _ = (stages, filled, slot);
/// # }
/// ```
///
/// # Safety
///
/// - The launch must declare at least [`Self::bytes`] of the finished walk.
///
/// Discharged once, by [`Self::attach`]. Nothing after that is `unsafe`,
/// because a handle is inert: every operation this crate offers on a
/// [`SharedTile`], [`SharedVec`] or [`Semaphore`] is itself `unsafe`, so a
/// handle carved off a [`Self::sizing`] cursor cannot be dereferenced by safe
/// code.
#[derive(Clone, Copy)]
pub struct SharedPlan {
    base: *mut u8,
    offset: usize,
    /// Whether there is memory under [`Self::base`] — `true` from
    /// [`Self::attach`], `false` from [`Self::sizing`]. Read only by
    /// [`Self::reserve`].
    attached: bool,
}

impl SharedPlan {
    /// What a shared **tile** or **vector** base owes: 128 bytes.
    ///
    /// Two rules that happen to be one number, and both have to hold. It is
    /// [`Swizzle::ATOM_BYTES`] for the only swizzle this crate implements, so
    /// a tile starting here reproduces the 128-byte row phase
    /// [`crate::shared::SwizzledChunks`] derives from a base; and it is the
    /// TMA engine's shared-memory destination alignment, which is why a vector
    /// gets it too.
    pub const TILE_ALIGN: usize = 128;
    /// Bytes one mbarrier occupies — the 64-bit state word, and the stride
    /// between the members of a [`SemaphoreRing`].
    pub const BARRIER_BYTES: usize = size_of::<Barrier>();
    /// What an mbarrier's address owes.
    pub const BARRIER_ALIGN: usize = align_of::<Barrier>();
    /// Bytes `tcgen05.alloc` stages its result through, and what it owes: the
    /// one `u32` per CTA the allocator writes and reads back.
    pub const TMEM_SLOT_BYTES: usize = size_of::<u32>();

    /// A cursor over the base of *this* launch's dynamic shared memory.
    ///
    /// # Safety
    ///
    /// - The launch must declare at least the finished walk's [`Self::bytes`];
    ///   every handle this cursor hands out is valid only under that.
    /// - Device code only — `get_raw` is `unreachable!()` on the host.
    #[inline(always)]
    pub unsafe fn attach() -> Self {
        Self {
            base: DynamicSharedArray::<u8, { Self::TILE_ALIGN }>::get_raw(),
            offset: 0,
            attached: true,
        }
    }

    /// The same walk with no memory under it — what a host `const` evaluates
    /// to get the total the launch has to declare.
    ///
    /// The handles it hands out point at a null base and are inert: nothing can
    /// be read or written through one without an `unsafe` block this crate
    /// never writes on the host side.
    ///
    /// ```
    /// # use kittens::plan::SharedPlan;
    /// # use kittens::shared::{Bf16, F32, Swizzle128B};
    /// const PLAN: SharedPlan = SharedPlan::sizing()
    ///     .tile::<Bf16, 128, 128, Swizzle128B>()
    ///     .1
    ///     .vec::<F32, 4>()
    ///     .1
    ///     .semaphore()
    ///     .1;
    /// assert_eq!(PLAN.bytes(), 32_792);
    /// ```
    #[inline(always)]
    pub const fn sizing() -> Self {
        Self {
            base: core::ptr::null_mut(),
            offset: 0,
            attached: false,
        }
    }

    /// Bytes reserved so far — the plan's total once the walk is finished, and
    /// the number the launch declares.
    #[inline(always)]
    pub const fn bytes(self) -> usize {
        self.offset
    }

    /// Reserve `bytes` at `align`, returning the start of the run and the
    /// advanced cursor.
    ///
    /// Every typed reservation below is this call with the type's own two
    /// numbers filled in, and a kernel should prefer one of those. This is the
    /// escape hatch for the one case the type system cannot reach: a `const fn`
    /// over *value* parameters cannot name `SharedTileRing<_, R, C, _, N>`,
    /// because a const parameter is not a function argument, so a host-side
    /// plan answering for shapes no kernel instantiates spells its reservations
    /// in bytes and joins the two forms with an assert.
    ///
    /// `align` must be a power of two. The sizing cursor returns its base
    /// unoffset — a sizing pointer is inert, and its offset lives in
    /// [`Self::bytes`], which is the only thing that path reads. The mode
    /// branch folds at codegen; `docs/library/plan.md` has the census that
    /// shows it.
    ///
    /// ```
    /// # use kittens::plan::SharedPlan;
    /// let plan = SharedPlan::sizing();
    /// let (_, plan) = plan.reserve(3 * 128 * 64 * 2, SharedPlan::TILE_ALIGN);
    /// let (_, plan) = plan.reserve(3 * SharedPlan::BARRIER_BYTES, 8);
    /// assert_eq!(plan.bytes(), 128 * 64 * 2 * 3 + 3 * 8);
    /// ```
    #[inline(always)]
    pub const fn reserve(self, bytes: usize, align: usize) -> (*mut u8, Self) {
        assert!(align.is_power_of_two(), "an alignment is a power of two");
        let offset = self.offset.next_multiple_of(align);
        let at = if self.attached {
            // SAFETY: `attach`'s contract is that the launch declares at least
            // the finished walk's bytes, and `offset + bytes` is inside it.
            unsafe { self.base.add(offset) }
        } else {
            self.base
        };
        (
            at,
            Self {
                base: self.base,
                offset: offset + bytes,
                attached: self.attached,
            },
        )
    }

    /// One [`SharedTile`], 128-byte aligned.
    #[inline(always)]
    pub const fn tile<E: Element, const R: usize, const C: usize, S: Swizzle>(
        self,
    ) -> (SharedTile<E, R, C, S>, Self) {
        let (at, plan) = self.reserve(SharedTile::<E, R, C, S>::BYTES, Self::TILE_ALIGN);
        // SAFETY: `reserve` just handed back a run of exactly `BYTES` at the
        // alignment a tile base owes, inside the plan `attach` promised.
        (unsafe { SharedTile::from_raw(at) }, plan)
    }

    /// `N` consecutive tiles as one [`SharedTileRing`], 128-byte aligned.
    ///
    /// A ring's stride is its tile's `BYTES` and a tile's alignment divides it,
    /// so one alignment at the front covers every buffer in the ring.
    #[inline(always)]
    pub const fn tile_ring<
        E: Element,
        const R: usize,
        const C: usize,
        S: Swizzle,
        const N: usize,
    >(
        self,
    ) -> (SharedTileRing<E, R, C, S, N>, Self) {
        let (at, plan) = self.reserve(SharedTileRing::<E, R, C, S, N>::BYTES, Self::TILE_ALIGN);
        // SAFETY: as `tile`, for `N` of them.
        (unsafe { SharedTileRing::attach(at) }, plan)
    }

    /// One [`SharedVec`], 128-byte aligned — the rule a vector the TMA touches
    /// owes, taken unconditionally so that a plan's correctness never depends
    /// on the order its reservations are written in.
    #[inline(always)]
    pub const fn vec<E: Element, const N: usize>(self) -> (SharedVec<E, N>, Self) {
        let (at, plan) = self.reserve(SharedVec::<E, N>::BYTES, Self::TILE_ALIGN);
        // SAFETY: `reserve` handed back `BYTES` at the engine's destination
        // alignment, which is the strongest a vector can owe.
        (unsafe { SharedVec::from_raw(at) }, plan)
    }

    /// `count` consecutive mbarrier words, 8-byte aligned.
    ///
    /// The value-parameterized form of [`Self::semaphores`], and the only
    /// method here that hands back a `*mut Barrier` — a host plan reserving
    /// `2 * stages + 1` of them has no const parameter to name a ring with.
    /// Device code should take [`Self::semaphore`] or [`Self::semaphores`]:
    /// that is what stops a barrier count being spelled `2 * STAGES * 8` in
    /// bytes at one end of a kernel and `.add(STAGES)` in `Barrier` units at
    /// the other.
    #[inline(always)]
    pub const fn barriers(self, count: usize) -> (*mut Barrier, Self) {
        let (at, plan) = self.reserve(count * Self::BARRIER_BYTES, Self::BARRIER_ALIGN);
        (at as *mut Barrier, plan)
    }

    /// One [`Semaphore`].
    #[inline(always)]
    pub const fn semaphore(self) -> (Semaphore, Self) {
        let (at, plan) = self.barriers(1);
        // SAFETY: one aligned barrier word inside the plan.
        (unsafe { Semaphore::attach(at) }, plan)
    }

    /// An `N`-deep [`SemaphoreRing`].
    #[inline(always)]
    pub const fn semaphores<const N: usize>(self) -> (SemaphoreRing<N>, Self) {
        let (at, plan) = self.barriers(N);
        // SAFETY: `N` aligned barrier words inside the plan.
        (unsafe { SemaphoreRing::attach(at) }, plan)
    }

    /// The staging word [`crate::tmem::alloc_block`] and
    /// [`crate::tmem::alloc_cluster`] write their result into, 4-byte aligned.
    ///
    /// A cluster allocation needs the word at the *same* offset in every rank,
    /// which a symmetric plan gives for free — and is why this is a
    /// reservation rather than a static.
    #[inline(always)]
    pub const fn tmem_slot(self) -> (*mut u32, Self) {
        let (at, plan) = self.reserve(Self::TMEM_SLOT_BYTES, align_of::<u32>());
        (at as *mut u32, plan)
    }

    /// The hardware work queue: its `.b128` response at
    /// [`ClcQueue::ALIGNMENT`], and the mbarrier its multicast delivery
    /// completes on behind that.
    ///
    /// The two objects are reserved separately rather than as one 24-byte run
    /// because that is what lets the whole walk const-evaluate, and it costs
    /// nothing: 16 at 16 then 8 at 8 is [`ClcQueue::BYTES`] on the nose.
    ///
    /// ```
    /// # use kittens::plan::SharedPlan;
    /// # use kittens::pipeline::ClcQueue;
    /// const PLAN: SharedPlan = SharedPlan::sizing().clc_queue().1;
    /// assert_eq!(PLAN.bytes(), ClcQueue::BYTES);
    /// ```
    #[inline(always)]
    pub const fn clc_queue(self) -> (ClcQueue, Self) {
        let (response, plan) = self.reserve(ClcQueue::RESPONSE_BYTES, ClcQueue::ALIGNMENT);
        let (sem, plan) = plan.barriers(1);
        // SAFETY: the barrier is `RESPONSE_BYTES` past the response by the two
        // reservations above, both inside the plan, and both at the same
        // offset in every rank because every rank walks the same plan.
        (unsafe { ClcQueue::from_parts(response, sem) }, plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{Bf16, F32, Swizzle128B};

    /// The GEMM plans the repo's kernels declare, each as its own walk,
    /// against the literal its launch carries: these are the numbers that set
    /// residency, and a change to `reserve`'s alignment rules moves them here
    /// before it moves them on a B200.
    ///
    /// The shapes are copied from the kernels rather than imported — the
    /// library cannot see a kernel crate — so a kernel that changes shape
    /// makes this test stale rather than red.
    #[test]
    fn gemm_pair_tile_plans_are_the_declared_envelopes() {
        // `examples/src/gemm.rs`: BLOCK_M 128, HALF_N 128, BLOCK_K 64,
        // STAGES 3, no work queue.
        const EXAMPLE: SharedPlan = SharedPlan::sizing()
            .tile_ring::<Bf16, 128, 64, Swizzle128B, 3>()
            .1
            .tile_ring::<Bf16, 128, 64, Swizzle128B, 3>()
            .1
            .semaphores::<3>()
            .1
            .semaphores::<3>()
            .1
            .semaphore()
            .1
            .tmem_slot()
            .1;
        assert_eq!(EXAMPLE.bytes(), 98_364);

        // The staged epilogue's run: one `[32, 64]` bf16 tile per warp.
        const STAGED: SharedPlan = EXAMPLE.tile_ring::<Bf16, 32, 64, Swizzle128B, 4>().1;
        assert_eq!(STAGED.bytes(), 114_816);

        // `experiments/src/gemm.rs`: the same walk with the work queue on the
        // end, whose 16-byte alignment absorbs the staging word's four.
        const EXPERIMENT: SharedPlan = EXAMPLE.clc_queue().1;
        assert_eq!(EXPERIMENT.bytes(), 98_392);
        assert_eq!(
            EXPERIMENT
                .tile_ring::<Bf16, 32, 64, Swizzle128B, 4>()
                .1
                .bytes(),
            114_816
        );

        // `experiments/src/gemm_ws.rs`: STAGES 4, and four epilogue warps out
        // of the six its 192 threads make.
        const WS: SharedPlan = SharedPlan::sizing()
            .tile_ring::<Bf16, 128, 64, Swizzle128B, 4>()
            .1
            .tile_ring::<Bf16, 128, 64, Swizzle128B, 4>()
            .1
            .semaphores::<4>()
            .1
            .semaphores::<4>()
            .1
            .semaphore()
            .1
            .tmem_slot()
            .1
            .clc_queue()
            .1;
        assert_eq!(WS.bytes(), 131_176);
        assert_eq!(
            WS.tile_ring::<Bf16, 32, 64, Swizzle128B, 4>().1.bytes(),
            147_584
        );
    }

    #[test]
    fn the_three_small_kernels_plans() {
        // `examples/src/softmax.rs`: one `[128, 128]` bf16 tile and a barrier.
        const SOFTMAX: SharedPlan = SharedPlan::sizing()
            .tile::<Bf16, 128, 128, Swizzle128B>()
            .1
            .semaphore()
            .1;
        assert_eq!(SOFTMAX.bytes(), 32_776);

        // `examples/src/layernorm.rs`, whose constant covers both kernels:
        // `layernorm_rows` takes the tile, `gamma`, `beta` and a barrier.
        const LAYERNORM_ROWS: SharedPlan = SharedPlan::sizing()
            .tile::<Bf16, 128, 128, Swizzle128B>()
            .1
            .vec::<Bf16, 128>()
            .1
            .vec::<Bf16, 128>()
            .1
            .semaphore()
            .1;
        assert_eq!(LAYERNORM_ROWS.bytes(), 33_288);

        // `groupnorm_tile` takes the tile, the four warps' partials and a
        // barrier. Both orders are legal; they differ only in padding.
        const GROUPNORM: SharedPlan = SharedPlan::sizing()
            .tile::<Bf16, 128, 128, Swizzle128B>()
            .1
            .vec::<F32, 4>()
            .1
            .semaphore()
            .1;
        assert_eq!(GROUPNORM.bytes(), 32_792);
        const REVERSED: SharedPlan = SharedPlan::sizing()
            .tile::<Bf16, 128, 128, Swizzle128B>()
            .1
            .semaphore()
            .1
            .vec::<F32, 4>()
            .1;
        assert_eq!(REVERSED.bytes(), 32_912);
        assert!(LAYERNORM_ROWS.bytes() > GROUPNORM.bytes());

        // `examples/src/flash_forward.rs`: Q, the K and V rings, P, five
        // semaphores across a ring pair and three singles, and the slot.
        const FLASH: SharedPlan = SharedPlan::sizing()
            .tile::<Bf16, 128, 128, Swizzle128B>()
            .1
            .tile_ring::<Bf16, 64, 128, Swizzle128B, 3>()
            .1
            .tile_ring::<Bf16, 64, 128, Swizzle128B, 3>()
            .1
            .tile::<Bf16, 128, 64, Swizzle128B>()
            .1
            .semaphores::<3>()
            .1
            .semaphores::<3>()
            .1
            .semaphore()
            .1
            .semaphore()
            .1
            .semaphore()
            .1
            .tmem_slot()
            .1;
        assert_eq!(FLASH.bytes(), 147_532);
    }

    /// A reservation is aligned to its own type's rule and to nothing else, so
    /// the padding is where the alignments say and not where an author
    /// remembered to put it.
    #[test]
    fn each_reservation_aligns_to_its_own_rule() {
        let plan = SharedPlan::sizing();
        // A four-byte slot first, then a tile: 124 bytes of padding.
        let (_, plan) = plan.tmem_slot();
        assert_eq!(plan.bytes(), 4);
        let (_, tiled) = plan.tile::<Bf16, 8, 64, Swizzle128B>();
        assert_eq!(tiled.bytes(), 128 + 8 * 64 * 2);
        // A barrier behind the slot needs only its own eight.
        let (_, barred) = plan.semaphore();
        assert_eq!(barred.bytes(), 8 + 8);
        // And the queue its sixteen. Its two reservations — the `.b128`
        // response at 16, the barrier under it at 8 — come to
        // `ClcQueue::BYTES` between them.
        let (_, queued) = plan.clc_queue();
        assert_eq!(queued.bytes(), 16 + ClcQueue::BYTES);
        assert_eq!(
            SharedPlan::sizing().clc_queue().1.bytes(),
            ClcQueue::BYTES,
            "the response and its barrier are the whole queue, laid out apart"
        );
    }

    /// A sizing cursor hands out the same pointer at every reservation and the
    /// right byte count, which is the whole of what the host path needs — and
    /// is why `reserve` can skip the arithmetic the device does.
    #[test]
    fn a_sizing_cursor_counts_bytes_and_points_nowhere() {
        let plan = SharedPlan::sizing();
        let (first, plan) = plan.reserve(1000, 128);
        let (second, plan) = plan.reserve(24, 16);
        assert!(first.is_null() && second.is_null());
        assert_eq!(plan.bytes(), 1000usize.next_multiple_of(16) + 24);
    }
}
