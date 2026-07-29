//! A kernel's shared plan: what it carves out of the launch's dynamic shared
//! memory, in declaration order, as one program.
//!
//! # Why this exists
//!
//! Every kernel in this repo used to compute its plan **twice** — once as a
//! host-visible `SHARED_BYTES` the launch declares, and once as a pointer walk
//! inside the kernel — with nothing but a hand-written `const { assert!(..) }`
//! relating the two. The walk was raw byte arithmetic over objects whose sizes
//! this crate already knows, and it leaked four things into every kernel that
//! wrote one: `DynamicSharedArray::<u8, 128>::get_raw()`, a
//! `cuda_device::barrier::Barrier` import that existed only as a cast target,
//! the placement and accounting of [`crate::tmem::alloc_cluster`]'s staging
//! word, and an obligation to prove [`ClcQueue::ALIGNMENT`] by hand. #125 is
//! that finding;
//! `GAPS.md` §7.4 ranked it the largest single thing a kernel still open-codes.
//!
//! [`SharedPlan`] is a cursor. It starts at the launch's base, hands out one
//! typed handle per reservation, aligns each to what *that handle's type*
//! requires, and accumulates the total as it goes — so the number the launch
//! declares is [`SharedPlan::bytes`] of the same walk that produced the
//! pointers, rather than a second expression asserted equal to it.
//!
//! # The two constraints it has to survive, and where it only half does
//!
//! **The total must be usable in a `#[launch_contract]` literal.** Every method
//! here but [`SharedPlan::attach`] is a `const fn`, and [`SharedPlan::sizing`]
//! is a cursor with no memory under it, so the whole walk const-evaluates on
//! the host with the handles falling on the floor:
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
//! **A const-generic walk and a value-parameterized total are still two
//! programs.** A `const fn` over *values* — `experiments`' rung table asks for
//! `shared_plan(block_n, block_k, stages)` at runtime, for rungs no kernel
//! implements — cannot instantiate `SharedTileRing<Bf16, R, C, S, N>`, because
//! a const parameter cannot be a function argument. That is a language limit
//! and not a design one, and [`SharedPlan::reserve`] is what it costs: the
//! value form spells the same reservations in bytes and the two are joined by
//! one assert. Kernels whose plan parameters are module constants — every
//! kernel in `examples/` — need no value form and no assert at all, and there
//! the total genuinely *is* the walk.
//!
//! # What each handle is aligned to, and why the caller no longer argues it
//!
//! | reservation | alignment | because |
//! | --- | --- | --- |
//! | [`SharedPlan::tile`], [`SharedPlan::tile_ring`], [`SharedPlan::vec`] | [`SharedPlan::TILE_ALIGN`] = 128 | the TMA's shared destination alignment, and [`crate::shared::Swizzle128B::ATOM_BYTES`] |
//! | [`SharedPlan::semaphore`], [`SharedPlan::semaphores`], [`SharedPlan::barriers`] | 8 | `align_of::<Barrier>()` — an mbarrier is one 64-bit state word |
//! | [`SharedPlan::tmem_slot`] | 4 | `tcgen05.alloc` writes a `u32`, and [`crate::tmem::alloc_block`]'s contract says `SharedArray<u32, 1, 4>` |
//! | [`SharedPlan::clc_queue`] | [`ClcQueue::ALIGNMENT`] = 16 | the response is a `.b128` store |
//!
//! `layernorm.rs` used to carry the general form of this out loud — *"The
//! partials first, because `Tile::BYTES` is the only offset in this plan a
//! vector's 128-byte alignment is promised at; the barrier behind it needs
//! eight"* — which is a correctness argument about **ordering** that the kernel
//! had to make because nothing else would. Under a cursor it is not one: a
//! vector reserved behind a barrier gets padded to 128 rather than landing
//! misaligned, so the order decides how many bytes the plan spends and no
//! longer whether the vector is legal. That is the test this type is meant to
//! pass, and the reason [`SharedPlan::vec`] takes the strong 128-byte rule
//! rather than the `E::BYTES` one a `get`/`set`-only vector could live with:
//! the weaker rule would make a plan's correctness depend on its order again,
//! to save at most 127 bytes.
//!
//! # Cost, as measured
//!
//! `SharedPlan` is a base, an offset and a mode flag, `Copy`, and every method
//! is `#[inline(always)]` arithmetic on a compile-time constant offset.
//! Threading it by value rather than as a `&mut` builder is what is supposed to
//! keep it free, and `scripts/modal-run regcount` diffed row by row against
//! `main` is what was asked. Of 231 register rows and 40 opcode-census rows:
//!
//! - **All 184 `device-tests` kernels are unchanged**, and so is every counter
//!   of the opcode census for every `gemm_*` kernel in both kernel crates — no
//!   instruction moved anywhere.
//! - `gemm_cg2_staged_x8x4`, the kernel `examples/` ships and `experiments/`
//!   keeps as its control arm, is unchanged in both crates (80 and 96), as are
//!   `gemm_cg2` (166), `gemm_cg2_staged` (42), `flash_forward` (168),
//!   `softmax_rows` (32) and `groupnorm_tile` (168).
//! - Six moved, in both directions: `layernorm_rows` 47 → 50 in both crates,
//!   `gemm_cg2_staged_x8x4_2x` 106 → 110, `gemm_cg2_staged_x8x4_hot` 93 → 104,
//!   `gemm_cg2_dry` 20 → 19, `gemm_ws_staged` 44 → 43, `gemm_ws_staged_x8`
//!   94 → 90. No spill and no stack frame moved with any of them.
//!
//! Two of those cross a *register-term* occupancy step by
//! `modal_app.py`'s `_ctas_by_registers` and neither crosses a real one,
//! because another term binds in both: `layernorm_rows` goes 10 → 9 CTAs by
//! registers and is 6 by shared memory either way, and
//! `gemm_cg2_staged_x8x4_hot` goes 5 → 4 and is 2 by tensor memory either way.
//! The PR for #125 has the arithmetic.
//!
//! **[`SharedPlan::reserve`]'s mode branch does fold, and that is measured
//! rather than argued.** A `selp` column in the census is the check — two
//! side-effect-free pointer values under an `if` is what LLVM if-converts to a
//! `select`, so a surviving branch shows up there. `gemm_cg2_idle` runs the
//! whole eight-reservation walk and then does nothing, so an unfolded flag
//! would give it eight; it has **zero**, at an unchanged 17 registers. So the
//! cursor adds no instruction, and the six moved counts are not it.
//!
//! What that still does **not** establish is which of the six they *are*. The
//! movement is bidirectional and the instruction counts are identical, which
//! reads as ptxas allocating rather than as a live value this type added, and
//! the two arms of `attach_staged` that did not move sit beside the two that
//! did. Read it as "no kernel's residency moved, no instruction did, and the
//! branch this type introduces is gone by codegen", which is what was checked.

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
/// # Safety
///
/// The whole plan's obligation is discharged once, by [`Self::attach`]: the
/// launch must declare at least [`Self::bytes`] of the finished walk. Nothing
/// after that is `unsafe`, because a handle is inert — every operation this
/// crate offers on a [`SharedTile`], [`SharedVec`] or [`Semaphore`] is itself
/// `unsafe`, so a handle carved off a [`Self::sizing`] cursor cannot be
/// dereferenced by safe code.
#[derive(Clone, Copy)]
pub struct SharedPlan {
    base: *mut u8,
    offset: usize,
    /// Whether there is memory under [`Self::base`] — `true` from
    /// [`Self::attach`], `false` from [`Self::sizing`].
    ///
    /// It is a flag and not a const parameter because a const parameter would
    /// travel into every kernel's own plan struct, and it is a flag rather
    /// than `base.is_null()` because only one of those folds. See
    /// [`Self::reserve`], which is the only thing that reads it.
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
    /// Bytes `tcgen05.alloc` stages its result through, and what it owes.
    ///
    /// The allocator writes one `u32` per CTA and reads it back; the caller
    /// used to find, align and account for it, which is #125's third leak.
    pub const TMEM_SLOT_BYTES: usize = size_of::<u32>();

    /// A cursor over the base of *this* launch's dynamic shared memory.
    ///
    /// The one place `DynamicSharedArray::<u8, 128>::get_raw()` is spelled for
    /// a kernel that lays out a plan, which is #125's first leak.
    ///
    /// # Safety
    ///
    /// Every handle this cursor goes on to hand out is valid only if the
    /// launch declares at least the finished walk's [`Self::bytes`]. Call it
    /// from device code: `get_raw` is `unreachable!()` on the host.
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
    /// The handles it hands out are offsets from a null base and are inert by
    /// construction: nothing can be read or written through one without an
    /// `unsafe` block this crate never writes on the host side.
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
    /// escape hatch for the case the type system cannot reach: a `const fn`
    /// over *value* parameters cannot name `SharedTileRing<_, R, C, _, N>`,
    /// because a const parameter is not a function argument, so a host-side
    /// plan that has to answer for rungs no kernel instantiates spells its
    /// reservations in bytes. See the module docs.
    ///
    /// `align` must be a power of two.
    ///
    /// # Why the branch is here and what it costs
    ///
    /// The device and the host want *different* pointer arithmetic and there is
    /// no operation that serves both. `wrapping_add` const-evaluates off a
    /// sizing cursor's provenance-free base and **does not compile for the
    /// device** — `rustc_codegen_cuda` rejects `std::intrinsics::arith_offset`
    /// as unsupported, which is the error this method was written twice to fix.
    /// `add` compiles and is what every kernel's hand-written walk used, but
    /// const eval will not let it leave an allocation, and a sizing cursor has
    /// none.
    ///
    /// So the sizing cursor does not offset its base at all: the pointer it
    /// hands back is inert either way, and the offset it is *not* carrying is
    /// in [`Self::bytes`], which is the only thing that path reads. `attached`
    /// is a literal in both constructors and every method here is
    /// `#[inline(always)]`, so this is a branch the optimizer resolves before
    /// it can cost an instruction. It is a flag and not `self.base.is_null()`
    /// for exactly that reason: a literal constant-folds, and a null check
    /// folds only if LLVM can prove `get_raw()` returns a non-null pointer,
    /// which it cannot.
    ///
    /// **That it folds is measured, not assumed.** Two side-effect-free
    /// pointer values under an `if` is the shape LLVM if-converts to a
    /// `select`, so a branch that survived would be a `selp` in the PTX;
    /// `modal_app.py`'s opcode census counts them. `gemm_cg2_idle` runs this
    /// walk's eight reservations and then does nothing, so it would carry
    /// eight — and carries **zero**, at an unchanged 17 registers. See this
    /// module's `Cost, as measured`.
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
    /// owes, taken unconditionally so that a plan's correctness does not
    /// depend on its order. See the module docs.
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
    /// Device code should take [`Self::semaphore`] or [`Self::semaphores`],
    /// which is what stops a barrier count being spelled `2 * STAGES * 8` in
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
    /// The allocator's own resource, which the caller used to place and
    /// account for. A cluster allocation additionally needs it at the *same*
    /// offset in every rank — which a symmetric plan gives for free, and is
    /// why this is a reservation rather than a static.
    #[inline(always)]
    pub const fn tmem_slot(self) -> (*mut u32, Self) {
        let (at, plan) = self.reserve(Self::TMEM_SLOT_BYTES, align_of::<u32>());
        (at as *mut u32, plan)
    }

    /// The hardware work queue: its `.b128` response at
    /// [`ClcQueue::ALIGNMENT`], and the mbarrier its multicast delivery
    /// completes on behind that.
    ///
    /// The alignment used to be the caller's proof obligation, carried as an
    /// identical `const { assert!(offset % ClcQueue::ALIGNMENT == 0) }` block
    /// in both GEMMs (#125's fourth leak). It is this method instead. The two
    /// objects are reserved separately rather than as one 24-byte run because
    /// that is what lets the whole walk const-evaluate — see [`Self::reserve`]
    /// — and it costs nothing: 16 at 16 then 8 at 8 is [`ClcQueue::BYTES`] on
    /// the nose, which the test below asserts.
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

    /// The five plans the repo's kernels declare, each as its own walk, each
    /// against the literal its launch carries. #125's regression suite: these
    /// are the numbers that set residency, and a change to `reserve`'s
    /// alignment rules moves them here before it moves them on a B200.
    ///
    /// The shapes are copied from the kernels rather than imported — the
    /// library cannot see a kernel crate — so a kernel that changes shape
    /// makes this test stale rather than red. What it holds is the *plan
    /// type's* arithmetic, which is the half that lives here.
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

        // The staged epilogue's run: one `[32, 64]` bf16 tile per warp, and
        // the 128-byte alignment that used to be a `next_multiple_of` written
        // beside the constant.
        const STAGED: SharedPlan = EXAMPLE.tile_ring::<Bf16, 32, 64, Swizzle128B, 4>().1;
        assert_eq!(STAGED.bytes(), 114_816);

        // `experiments/src/gemm.rs`: the same walk with the work queue on the
        // end. Its 16-byte alignment absorbs the staging word's four, which is
        // why this total is unchanged from the hand-written one.
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
        // barrier — and, unlike the hand-written walk, is correct in either
        // order. Both spellings are legal; they differ in padding.
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
        // response at 16, the barrier under it at 8 — are `ClcQueue::BYTES`
        // between them, which is the join `clc_queue` relies on.
        let (_, queued) = plan.clc_queue();
        assert_eq!(queued.bytes(), 16 + ClcQueue::BYTES);
        assert_eq!(
            SharedPlan::sizing().clc_queue().1.bytes(),
            ClcQueue::BYTES,
            "the response and its barrier are the whole queue, laid out apart"
        );
    }

    /// A sizing cursor hands out the same pointer at every reservation and
    /// the right byte count, which is the whole of what the host path needs —
    /// and is why [`SharedPlan::reserve`] can skip the arithmetic the device
    /// does. See its docs for why the two cannot share one operation.
    #[test]
    fn a_sizing_cursor_counts_bytes_and_points_nowhere() {
        let plan = SharedPlan::sizing();
        let (first, plan) = plan.reserve(1000, 128);
        let (second, plan) = plan.reserve(24, 16);
        assert!(first.is_null() && second.is_null());
        assert_eq!(plan.bytes(), 1000usize.next_multiple_of(16) + 24);
    }
}
