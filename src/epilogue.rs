//! How a finished accumulator leaves a kernel: a band at a time through a
//! staging tile, in one box through the TMA engine, or through a ring of
//! staging tiles the engine reads out of.
//!
//! A result computed in registers cannot be handed to the engine directly — the
//! engine reads shared memory — so every route here goes through a staging tile
//! that `stmatrix` fills ([`crate::ldst::store_tile`]). What differs is how the
//! tile reaches global memory, and the three ways are three items:
//!
//! - [`Drain::staged`] copies it with ordinary 16-byte stores
//!   ([`crate::global::store_shared_rows`]), walking a warp's whole band of the
//!   accumulator. **The route that ships**: #123 measured the engine against it
//!   on `gemm` and the engine lost.
//! - [`Scope::store_once`] hands it to the engine once, for a kernel that stores a
//!   whole `[R, C]` box per item and has nothing to overlap it with.
//! - [`StoreRing`] is [`StoreRing::DEPTH`] of them cycled, with the fence and
//!   the two waits those instructions owe each other stated once instead of at
//!   every call site.
//!
//! Prefer [`Cta`] where a ring has the choice: the [`Warp`]-scope arm measured
//! 0.6–1.6% slower. That measurement, what a depth buys, and why this is not a
//! [`crate::shared::SharedTileRing`] are in `docs/library/epilogue.md`.
//!
//! ```no_run
//! # use cuda_device::tma::TmaDescriptor;
//! # use kittens::epilogue::StoreRing;
//! # use kittens::ldst::store_tile;
//! # use kittens::shared::{Bf16, Swizzle128B};
//! # use kittens::{BaseLdtm, RegTile, lane, warp_id};
//! # type Ring = StoreRing<Bf16, 128, 64, Swizzle128B, 1>;
//! # unsafe fn epilogue(
//! #     base: *mut u8,
//! #     map: *const TmaDescriptor,
//! #     band: RegTile<32, 64, BaseLdtm>,
//! # ) { unsafe {
//! let mut ring = Ring::attach(base);
//! for row in 0..4 {
//!     let staging = ring.acquire();
//!     store_tile(staging.chunk_writer(), 32 * warp_id(), 0, lane(), band);
//!     ring.commit(map, row, 0);
//! }
//! ring.drain();
//! # } }
//! ```

use core::marker::PhantomData;

use cuda_device::{thread, warp};

use cuda_device::tma::TmaDescriptor;

use crate::global::{GlobalRows, store_shared_rows};
use crate::ldst::{store_tile, store_tile_x4};
use crate::reg::{BaseLdtm, FragmentLayout, RegTile};
use crate::shared::{
    Element, SharedTile, Swizzle, publish_to_async_proxy, tma_store_commit, tma_store_wait,
    tma_store_wait_read,
};
use crate::tmem::TmemTile;

/// The set of threads that fills one staging buffer: how they converge, and
/// which of them owns the buffer's store groups.
///
/// The two are one decision, so the trait states them together — mixing them is
/// silently wrong rather than a type error. A CTA-wide barrier around a lane-0
/// issue works and costs too much; a warp barrier around a thread-0 issue lets
/// three warps run ahead of a wait they never took.
pub trait Scope {
    /// Whether this thread is the one that issues the ring's stores and takes
    /// its group waits. **Exactly one thread of the scope may answer `true`**:
    /// `cp.async.bulk.commit_group` gathers the *calling thread's* outstanding
    /// stores, so the wait that ages a group must be taken by the thread that
    /// committed it.
    fn issuing() -> bool;

    /// Order the scope's memory accesses against each other and hold every
    /// thread until all of them arrive — what carries a writer's proxy fence
    /// to the issuing thread, and what releases the non-issuing threads past a
    /// group wait they did not take.
    fn converge();

    /// One staging tile to global memory through the TMA engine, once — the
    /// fence, the issue, the commit and the visibility wait a kernel that
    /// stores a whole `[R, C]` box per item owes.
    ///
    /// [`StoreRing`] without the ring: an `acquire` with nothing to wait on, a
    /// `commit`, and the [`StoreRing::drain`] that follows it immediately. That
    /// is the right shape when there is no next band to overlap the store with
    /// — `softmax` and `layernorm` each write their tile once and are done —
    /// and the wait is for global *visibility* rather than for the engine's
    /// read, because nothing here recycles the buffer.
    ///
    /// It hangs off the scope rather than taking one because a call has to name
    /// the scope and nothing else: naming any of a generic function's arguments
    /// means naming all of them, and the element, the tile shape and the
    /// swizzle are already in the `tile` argument.
    ///
    /// `row` and `plane` go to [`SharedTile::tma_store`] as it takes them.
    ///
    /// ```no_run
    /// # use cuda_device::tma::TmaDescriptor;
    /// # use kittens::epilogue::{Cta, Scope};
    /// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
    /// # unsafe fn epilogue(
    /// #     tile: SharedTile<Bf16, 128, 128, Swizzle128B>,
    /// #     map: *const TmaDescriptor,
    /// #     row: i32,
    /// # ) { unsafe {
    /// Cta::store_once(tile, map, row, 0);
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// - Every thread of the scope calls this together, with all of its writes
    ///   into `tile` already issued.
    /// - `map` describes a live global buffer whose box shape matches
    ///   `[R, SUBTILE_COLS]`.
    /// - Only the issuing thread is held past the store: there is a convergence
    ///   *before* the engine reads and none after it, exactly as the
    ///   hand-written sequence this replaces had none. A caller that writes
    ///   `tile` again owes itself the barrier that orders that against the
    ///   wait.
    #[inline(always)]
    unsafe fn store_once<E: Element, const R: usize, const C: usize, S: Swizzle>(
        tile: SharedTile<E, R, C, S>,
        map: *const TmaDescriptor,
        row: i32,
        plane: i32,
    ) {
        unsafe {
            // The fence orders the writes of the thread executing it, and the
            // convergence is what carries them to the thread that issues.
            publish_to_async_proxy();
            Self::converge();
            if Self::issuing() {
                tile.tma_store(map, row, plane);
                tma_store_commit();
                tma_store_wait::<0>();
            }
        }
    }
}

/// The whole block writes a buffer: `bar.sync`, and thread 0 issues. The
/// default, what a `[128, N]` staging tile filled by four warps needs, and the
/// scope to reach for absent a measured reason.
pub struct Cta;

impl Scope for Cta {
    #[inline(always)]
    fn issuing() -> bool {
        thread::threadIdx_x() == 0
    }

    #[inline(always)]
    fn converge() {
        thread::sync_threads();
    }
}

/// One warp writes a buffer: `bar.warp.sync`, and lane 0 issues — for a kernel
/// whose warps each own a staging tile and so need no block barrier at all.
/// `bar.warp.sync` orders memory among a warp's lanes exactly as `bar.sync`
/// does among a block's threads, so the proxy fence's reach is the same
/// argument one scope down.
///
/// **Measured 0.6–1.6% slower than [`Cta`]** on the one kernel that has both;
/// prefer [`Cta`] unless the layout is genuinely warp-private.
///
/// The mask is the full warp and not `activemask`: every lane holds part of the
/// buffer, so a ring acquired by a subset of the warp is a buffer nobody
/// finished writing.
///
/// ```no_run
/// # use kittens::epilogue::{StoreRing, Warp};
/// # use kittens::shared::{Bf16, Swizzle128B};
/// # use kittens::warp_id;
/// # unsafe fn demo(base: *mut u8) { unsafe {
/// // A quarter of a `[128, 64]` tile each, and no `bar.sync` between them.
/// type PerWarp = StoreRing<Bf16, 32, 64, Swizzle128B, 0, Warp>;
/// let mut ring = PerWarp::attach(base.add(warp_id() as usize * PerWarp::BYTES));
/// let _staging = ring.acquire();
/// # } }
/// ```
pub struct Warp;

impl Scope for Warp {
    #[inline(always)]
    fn issuing() -> bool {
        crate::lane() == 0
    }

    #[inline(always)]
    fn converge() {
        warp::sync_mask(u32::MAX);
    }
}

/// A drain at one pair of instruction widths — [`Drain::staged`] is the
/// epilogue, and this type is how a call names the widths and nothing else.
///
/// A generic *function* would be the obvious shape and Rust will not have it:
/// naming any of a function's generic arguments means naming all of them, so
/// the widths would drag the element, the two tile shapes and the swizzle to
/// every call site, which are exactly the four the arguments already carry.
/// Naming them on a type leaves the function's own eight to inference.
pub struct Drain<const LDTM_X8: bool, const STMATRIX_X4: bool>;

impl<const LDTM_X8: bool, const STMATRIX_X4: bool> Drain<LDTM_X8, STMATRIX_X4> {
    /// TMEM → registers → `stmatrix` → global, a band at a time: the epilogue
    /// that ships, with the band loop's own convergence inside it.
    ///
    /// One warp drains its own [`BaseLdtm::WARP_ROWS`] rows of `accumulator`, `C`
    /// columns a pass, where `C` is `staging`'s width. A pass reads the band into
    /// registers, writes it into `staging` with `stmatrix`, and copies `staging`
    /// out to `dest` in 16-byte accesses ([`store_shared_rows`]). Every instruction
    /// in that is a call elsewhere in the crate; what lives here is the loop, the
    /// widths, and the one thing in it that is not a call — the [`Warp::converge`]
    /// each pass owes the next.
    ///
    /// # `band`, and why it is not a row
    ///
    /// It names the same thing at both ends: TMEM rows `WARP_ROWS * band` and the
    /// [`WARP_ROWS`](BaseLdtm::WARP_ROWS) after them land in `dest` rows
    /// `row + WARP_ROWS * band` and the [`WARP_ROWS`](BaseLdtm::WARP_ROWS) after
    /// those. So `row` and `column` are the **CTA's** origin in `C`, not the warp's,
    /// and the two cannot drift apart the way `32 * warp_id` written twice can.
    ///
    /// It is a band index rather than a warp index because the two differ across
    /// warpgroups: which TMEM lanes a warp reaches is `warp_id() % LANE_QUADRANTS`
    /// ([`crate::tmem::warp_lanes`], #193), and its rows of `C` are its own. For a
    /// one-warpgroup epilogue — every kernel in tree — both are `warp_id()`.
    ///
    /// # The widths, one per half (#117)
    ///
    /// `LDTM_X8` is the TMEM half: [`TmemTile::tile`] issues
    /// `tcgen05.ld.16x256b.x1` twice per `[16, 16]` block and waits after each,
    /// where [`TmemTile::tile_x8`] takes 64 columns and 32 f32 a thread per issue —
    /// two issues and two waits for a `[32, 64]` band against sixteen of each, and
    /// the waits are the larger half. `STMATRIX_X4` is the `stmatrix` half:
    /// [`store_tile`] issues `stmatrix.m8n8.x2` twice per block and
    /// [`store_tile_x4`] names all four matrices in one, at the same addresses.
    ///
    /// Neither moves a byte — both widths return the same band and write the same
    /// staging tile, which is what puts every combination on the same correctness
    /// gate — and neither touches the global half.
    ///
    /// # There is no proxy fence here and its absence is not a bug
    ///
    /// `fence.proxy.async.shared::cta` orders a generic-proxy write against an
    /// *async*-proxy read, which is what [`StoreRing`] needs. Both ends of this one
    /// are generic — `stmatrix` writes and `ld.shared` reads — and
    /// `stmatrix.sync.aligned` is itself a convergence, so nothing stands between
    /// them. The one hazard left is the *next* pass overwriting a tile this pass is
    /// still reading, which is what [`Warp::converge`] is for.
    ///
    /// # Warp scope, and no other
    ///
    /// `staging` is one warp's tile and nobody else's, so the convergence is
    /// `bar.warp.sync` and the destination rectangle is this warp's rows. The CTA
    /// arrangement — one `[128, C]` tile, `store_shared_rows::<.., 128>`, a
    /// `bar.sync` a pass — is what `crate::global`'s docs describe and nothing in
    /// tree uses; it is a second function when something does.
    ///
    /// ```no_run
    /// # use kittens::epilogue::Drain;
    /// # use kittens::global::GlobalRows;
    /// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
    /// # use kittens::tmem::TmemTile;
    /// # use kittens::{lane, warp_id};
    /// # unsafe fn epilogue(
    /// #     accumulator: TmemTile<128, 256>,
    /// #     staging: SharedTile<Bf16, 32, 64, Swizzle128B>,
    /// #     c: GlobalRows<Bf16>,
    /// #     row: u32,
    /// #     column: u32,
    /// # ) { unsafe {
    /// Drain::<true, true>::staged(accumulator, warp_id(), staging, c, row, column, lane());
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// - Every lane of one warp calls this together, with `accumulator` complete
    ///   and fenced and nothing in flight that will overwrite it.
    /// - `staging` is this warp's alone, and no other warp reads or writes it until
    ///   this returns.
    /// - The rectangle `row..row + R` of `dest`, `column..column + N`, lies inside
    ///   the buffer `dest` names.
    #[inline(always)]
    pub unsafe fn staged<
        E: Element<Unpacked = [f32; 2]>,
        const M: usize,
        const N: usize,
        const R: usize,
        const C: usize,
        S: Swizzle,
    >(
        accumulator: TmemTile<M, N>,
        band: u32,
        staging: SharedTile<E, R, C, S>,
        dest: GlobalRows<E>,
        row: u32,
        column: u32,
        lane: u32,
    ) where
        BaseLdtm: FragmentLayout<R, C>,
    {
        const {
            assert!(
                R == BaseLdtm::WARP_ROWS,
                "a warp's band is BaseLdtm::WARP_ROWS rows, so the staging tile is that tall"
            );
            assert!(
                C <= BaseLdtm::WIDEST_BAND,
                "a pass holds the whole band in registers; BaseLdtm::WIDEST_BAND is what fits"
            );
            assert!(
                N.is_multiple_of(C),
                "the walk is whole staging tiles, so the accumulator's columns are a multiple of them"
            );
        };
        unsafe {
            let chunks = staging.chunk_writer();
            let origin = BaseLdtm::WARP_ROWS as u32 * band;
            let mut at = 0u32;
            while at < N as u32 {
                let pass: RegTile<R, C, BaseLdtm> = if LDTM_X8 {
                    accumulator.tile_x8(origin, at)
                } else {
                    accumulator.tile(origin, at)
                };
                if STMATRIX_X4 {
                    store_tile_x4(chunks, 0, 0, lane, pass);
                } else {
                    store_tile(chunks, 0, 0, lane, pass);
                }
                // The warp's 32 threads carry the whole tile: `staging` is theirs.
                store_shared_rows::<E, R, C, S, 32>(dest, row + origin, column + at, lane, staging);
                Warp::converge();
                at += C as u32;
            }
        }
    }
}

/// `IN_FLIGHT + 1` staging tiles of `[R, C]` `E` under swizzle `S`, cycled by a
/// cursor, with the TMA store discipline below built in.
///
/// Copy and register-resident like every other tile handle here, except that it
/// carries the cursor, so an `acquire`/`commit` pair cannot name different
/// buffers. Every thread of `SC` holds its own copy and they stay in step
/// because they run the same loop, which the collective methods require anyway.
///
/// # The fence and the two waits
///
/// Three obligations, none of which subsumes another. The ring discharges all
/// three; a caller open-coding a staging tile owes all three.
///
/// - **[`publish_to_async_proxy`] on every writing thread, then a convergence,
///   before the engine reads.** `stmatrix` writes through the *generic* proxy
///   and the engine reads through the *async* proxy; nothing but this fence
///   orders the two, and it orders only the writes of the thread executing it.
///   A `bar.sync` alone is **not** sufficient — it orders generic-proxy
///   accesses against each other and says nothing about a proxy it does not
///   name.
/// - **`cp.async.bulk.wait_group.read` before a buffer is rewritten**
///   ([`Self::acquire`]). A bulk store finishes *reading* shared memory long
///   before its bytes reach global memory, and recycling needs only the first
///   of those. Groups are per thread, so the wait falls to the issuing thread
///   and the convergence after it releases everyone else.
/// - **`cp.async.bulk.wait_group` once at the end** ([`Self::drain`]). Nothing
///   else makes a store visible to a following kernel or to the host — not the
///   end of the kernel, not a barrier, not dropping the handle.
///
/// # `IN_FLIGHT`, not `DEPTH`
///
/// `IN_FLIGHT` is the immediate `cp.async.bulk.wait_group.read` takes, and
/// `tma_store_wait_read::<{DEPTH - 1}>()` would need `generic_const_exprs` — so
/// the parameter is the end of the identity the instruction can be handed, and
/// [`Self::DEPTH`] is derived. `IN_FLIGHT = 0` is the single-buffer ring whose
/// `acquire` fully drains the engine's reads; `IN_FLIGHT = 1` overlaps one
/// store with the next band's fill, for one more tile of shared memory.
///
/// `SC` is which threads fill a buffer: [`Cta`] or [`Warp`].
pub struct StoreRing<
    E: Element,
    const R: usize,
    const C: usize,
    S: Swizzle,
    const IN_FLIGHT: u32,
    SC: Scope = Cta,
> {
    base: *mut u8,
    slot: u32,
    _marker: PhantomData<(E, S, SC)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32, SC: Scope> Clone
    for StoreRing<E, R, C, S, IN_FLIGHT, SC>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32, SC: Scope> Copy
    for StoreRing<E, R, C, S, IN_FLIGHT, SC>
{
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32, SC: Scope>
    StoreRing<E, R, C, S, IN_FLIGHT, SC>
{
    /// Staging buffers in the ring, one more than the stores that may be in
    /// flight across a buffer's reuse.
    pub const DEPTH: u32 = IN_FLIGHT + 1;

    /// Bytes of one staging buffer — what a step of depth costs, and the number
    /// a kernel sizing this against its occupancy step is choosing over.
    pub const TILE_BYTES: usize = SharedTile::<E, R, C, S>::BYTES;

    /// Bytes of the whole ring, which is what the kernel's shared plan pays.
    pub const BYTES: usize = Self::DEPTH as usize * Self::TILE_BYTES;

    /// Lay a ring over [`Self::BYTES`] of shared memory.
    ///
    /// ```no_run
    /// # use kittens::epilogue::StoreRing;
    /// # use kittens::plan::SharedPlan;
    /// # use kittens::shared::{Bf16, Swizzle128B};
    /// # type Ring = StoreRing<Bf16, 128, 64, Swizzle128B, 1>;
    /// # unsafe fn demo() { unsafe {
    /// let (at, _plan) = SharedPlan::attach().reserve(Ring::BYTES, SharedPlan::TILE_ALIGN);
    /// let ring = Ring::attach(at);
    /// # let _ = ring;
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// Same contract as [`SharedTile::from_raw`], for [`Self::BYTES`] bytes
    /// used by nothing else while the ring is live.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        Self {
            base,
            slot: 0,
            _marker: PhantomData,
        }
    }

    /// The buffer of stage `index % DEPTH`, for a kernel that wants to name one
    /// without acquiring it.
    ///
    /// It carries none of the ring's discipline: writing through this handle is
    /// writing to whatever the engine may still be reading.
    #[inline(always)]
    pub fn buffer(self, index: u32) -> SharedTile<E, R, C, S> {
        unsafe {
            SharedTile::from_raw(
                self.base
                    .add((index % Self::DEPTH) as usize * Self::TILE_BYTES),
            )
        }
    }

    /// Wait until the current buffer is free of the engine, and hand it back to
    /// be filled.
    ///
    /// The wait is `cp.async.bulk.wait_group.read` at `IN_FLIGHT` groups: with
    /// `DEPTH` groups committed and ageing in issue order, leaving at most
    /// `IN_FLIGHT` with outstanding reads is exactly the statement that the
    /// oldest — this buffer's — is done being read. Before `DEPTH` bands have
    /// been committed the wait is trivially satisfied, so the ring needs no fill
    /// phase.
    ///
    /// It waits for the reads only, not for the bytes to reach global memory —
    /// which is what makes a ring worth having, and why [`Self::drain`] exists.
    ///
    /// # Safety
    ///
    /// - Every thread of `SC` calls this together, and `SC` is the scope whose
    ///   issuing thread issued the ring's stores.
    /// - It contains a [`Scope::converge`], so a thread that skips it either
    ///   hangs the barrier or writes a buffer the engine is still reading — and
    ///   the second is silent.
    #[inline(always)]
    pub unsafe fn acquire(&mut self) -> SharedTile<E, R, C, S> {
        // Groups belong to the thread that committed them, so this wait means
        // something only on the issuing thread; the barrier is what the other
        // threads are released past. (The same reason `tma store early recycle`
        // in `device-tests` syncs between thread 0's wait and everyone's write.)
        if SC::issuing() {
            tma_store_wait_read::<IN_FLIGHT>();
        }
        SC::converge();
        self.buffer(self.slot)
    }

    /// Fence the buffer [`Self::acquire`] handed out and hand it to the engine,
    /// aimed at a [`crate::global`] panel map exactly as
    /// [`SharedTile::tma_store`] aims it, then advance the cursor. The store is
    /// *issued*, not complete.
    ///
    /// Exactly one committed group per call, which is what makes
    /// [`Self::acquire`]'s group count mean "one buffer".
    ///
    /// # Safety
    ///
    /// - Every thread of `SC` calls this together, once per [`Self::acquire`],
    ///   with all of its writes into the buffer already issued.
    /// - `map` describes a live global buffer whose box shape matches
    ///   `[R, SUBTILE_COLS]`.
    /// - The ring's shared memory stays allocated until [`Self::drain`].
    #[inline(always)]
    pub unsafe fn commit(&mut self, map: *const TmaDescriptor, row: i32, plane: i32) {
        unsafe {
            let staging = self.publish();
            if SC::issuing() {
                staging.tma_store(map, row, plane);
                tma_store_commit();
            }
        }
        self.advance();
    }

    /// [`Self::commit`] against a 2-D tensor map, taking its coordinates in the
    /// order [`SharedTile::tma_store_2d`] takes them.
    ///
    /// ```no_run
    /// # use cuda_device::tma::TmaDescriptor;
    /// # use kittens::epilogue::StoreRing;
    /// # use kittens::shared::{Bf16, Swizzle128B};
    /// # type Ring = StoreRing<Bf16, 128, 64, Swizzle128B, 1>;
    /// # unsafe fn demo(base: *mut u8, map: *const TmaDescriptor, tile_row: i32) { unsafe {
    /// let mut ring = Ring::attach(base);
    /// let _staging = ring.acquire();
    /// // ... `store_tile` into `_staging` ...
    /// ring.commit_2d(map, 0, tile_row * 128);
    /// ring.drain();
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// As [`Self::commit`], for a rank-2 map.
    #[inline(always)]
    pub unsafe fn commit_2d(&mut self, map: *const TmaDescriptor, leading: i32, minor: i32) {
        unsafe {
            let staging = self.publish();
            if SC::issuing() {
                staging.tma_store_2d(map, leading, minor);
                tma_store_commit();
            }
        }
        self.advance();
    }

    /// [`Self::commit_2d`] through [`SharedTile::tma_store_add_2d`]: the
    /// buffer's elements are **added into** the destination rather than
    /// overwriting it, which is how an accumulating epilogue emits its tile
    /// without ever reading `C` (#42).
    ///
    /// The ring's discipline is unchanged — the reduction is a bulk store to
    /// the group mechanism, so the same fence, commit and waits govern it.
    ///
    /// # Safety
    ///
    /// - As [`Self::commit_2d`].
    /// - The destination rectangle holds values of the map's data type: a
    ///   reduction reads what a plain store would ignore.
    #[inline(always)]
    pub unsafe fn commit_add_2d(&mut self, map: *const TmaDescriptor, leading: i32, minor: i32) {
        unsafe {
            let staging = self.publish();
            if SC::issuing() {
                staging.tma_store_add_2d(map, leading, minor);
                tma_store_commit();
            }
        }
        self.advance();
    }

    /// Wait until every committed store's bytes are in global memory.
    ///
    /// The last thing a kernel that wrote its result owes, and the one wait
    /// [`Self::acquire`] deliberately does not take. Nothing else discharges
    /// it: a kernel that simply ends has not waited, and the reads being done
    /// says only that shared memory is recyclable.
    ///
    /// Returns with every thread past the wait, so the ring's shared memory is
    /// free for anything the kernel does next.
    ///
    /// # Safety
    ///
    /// Every thread of `SC` calls this together, after the last
    /// [`Self::commit`].
    #[inline(always)]
    pub unsafe fn drain(self) {
        if SC::issuing() {
            tma_store_wait::<0>();
        }
        SC::converge();
    }

    /// Order every thread's `stmatrix` writes ahead of the engine's read of the
    /// current buffer, and hand that buffer to the caller to issue against.
    ///
    /// Both halves are load-bearing and in this order: the fence orders the
    /// writes of the thread executing it, and [`Scope::converge`] is what
    /// carries that ordering to the single thread that issues the store.
    /// Reversing them, or fencing only on the issuing thread, leaves the engine
    /// free to read bytes another thread has not published.
    #[inline(always)]
    unsafe fn publish(self) -> SharedTile<E, R, C, S> {
        unsafe { publish_to_async_proxy() };
        SC::converge();
        self.buffer(self.slot)
    }

    #[inline(always)]
    fn advance(&mut self) {
        self.slot += 1;
        if self.slot == Self::DEPTH {
            self.slot = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{Bf16, Swizzle128B};

    type Single = StoreRing<Bf16, 128, 64, Swizzle128B, 0>;
    type Double = StoreRing<Bf16, 128, 64, Swizzle128B, 1>;
    /// A quarter of the same bytes, acquired and committed by one warp instead
    /// of by the block.
    type PerWarp = StoreRing<Bf16, 32, 64, Swizzle128B, 0, Warp>;

    #[test]
    fn depth_is_one_more_than_the_stores_in_flight() {
        assert_eq!(Single::DEPTH, 1);
        assert_eq!(Double::DEPTH, 2);
        assert_eq!(StoreRing::<Bf16, 128, 64, Swizzle128B, 3>::DEPTH, 4);
    }

    /// A `[128, 64]` bf16 buffer is 16 KiB, so a depth is 16 KiB of the shared
    /// plan and nothing else in the type varies with it.
    #[test]
    fn a_depth_costs_exactly_one_staging_tile() {
        assert_eq!(Single::TILE_BYTES, 128 * 64 * 2);
        assert_eq!(Single::BYTES, 16384);
        assert_eq!(Double::BYTES, 32768);
        assert_eq!(Double::BYTES - Single::BYTES, Single::TILE_BYTES);
        // The shape the width parameter exists for: a full `[128, 128]` band
        // is twice that, per buffer.
        assert_eq!(StoreRing::<Bf16, 128, 128, Swizzle128B, 0>::BYTES, 32768);
    }

    /// The scope is the barrier and the issuing thread and nothing else — in
    /// particular not a byte, which is what lets four warp-scope rings and one
    /// CTA-scope ring occupy the same run and be compared.
    #[test]
    fn the_scope_costs_no_memory() {
        assert_eq!(4 * PerWarp::BYTES, Single::BYTES);
        assert_eq!(PerWarp::DEPTH, 1);
    }

    #[test]
    fn buffers_wrap_at_the_depth() {
        let base = 0x2000usize as *mut u8;
        let ring = unsafe { Double::attach(base) };
        assert_eq!(ring.buffer(0).base(), base);
        assert_eq!(ring.buffer(1).base() as usize, 0x2000 + Double::TILE_BYTES);
        assert_eq!(ring.buffer(2).base(), base);
        assert_eq!(ring.buffer(3).base() as usize, 0x2000 + Double::TILE_BYTES);

        // Depth 1 is one buffer, forever — the degenerate ring, and the one
        // whose `acquire` is a full drain of the engine's reads.
        let single = unsafe { Single::attach(base) };
        for index in 0..5u32 {
            assert_eq!(single.buffer(index).base(), base);
        }
    }

    /// The cursor is what makes an `acquire`/`commit` pair name one buffer, so
    /// its walk is pinned here even though the collective halves need a device.
    #[test]
    fn the_cursor_walks_the_buffers_in_order() {
        let base = 0x4000usize as *mut u8;
        let mut ring = unsafe { StoreRing::<Bf16, 64, 64, Swizzle128B, 2>::attach(base) };
        let tile_bytes = StoreRing::<Bf16, 64, 64, Swizzle128B, 2>::TILE_BYTES;
        for band in 0..7usize {
            assert_eq!(
                ring.buffer(ring.slot).base() as usize,
                0x4000 + (band % 3) * tile_bytes,
                "band {band}"
            );
            ring.advance();
        }
    }
}
