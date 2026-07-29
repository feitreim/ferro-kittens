//! The shared→global half of an epilogue: a staging ring a drained
//! accumulator is written into and the TMA engine reads out of.
//!
//! A kernel whose result is computed in registers has no way to hand it to the
//! TMA engine directly — the engine reads shared memory — so the last move of
//! such a kernel is `stmatrix` into a swizzled tile
//! ([`crate::ldst::store_tile`]) and `cp.async.bulk.tensor` out of it
//! ([`crate::shared::SharedTile::tma_store`]). [`StoreRing`] is that tile,
//! `DEPTH` of them, with the fence and the two waits those two instructions
//! owe each other stated once instead of at every call site.
//!
//! # Why the depth is a parameter, and what it buys
//!
//! At depth 1 there is one buffer: band `k + 1`'s `stmatrix` cannot start until
//! band `k`'s store has finished *reading* that buffer, so the register→shared
//! move and the shared→global move strictly alternate. `examples/src/softmax.rs`
//! is that shape written by hand, and it is the right shape for a kernel that
//! stores once.
//!
//! At depth 2 the two overlap: band `k + 1` writes the other buffer while the
//! engine is still draining band `k`'s. What the extra buffer costs is one
//! whole tile of shared memory, which for a kernel already near its occupancy
//! step is the expensive kind of byte — see [`StoreRing::BYTES`], and the
//! module docs of `examples/src/gemm.rs` for what an occupancy step is worth
//! there. The depth is a parameter so that trade is measured rather than
//! inherited.
//!
//! # The fence and the two waits
//!
//! Three mechanisms are involved and none of them subsumes another.
//!
//! **`fence.proxy.async.shared::cta`, before the engine reads a buffer.**
//! `stmatrix` is an ordinary shared-memory write through the *generic* proxy;
//! the TMA engine reads through the *async* proxy. Nothing orders the two but
//! this fence, and it orders the writes of the thread that *executes* it — so
//! every thread that wrote a row fences, and a `bar.sync` after that is what
//! makes the whole CTA's writes visible to the one thread that issues the
//! store. A `bar.sync` alone is not enough: it orders generic-proxy accesses
//! against each other and says nothing about a proxy it does not name.
//!
//! **`cp.async.bulk.wait_group.read`, before a buffer is written again.** A
//! bulk store completes in two stages: first the engine is done reading shared
//! memory, later the bytes are visible in global memory. Recycling a buffer
//! needs only the first, and [`StoreRing::acquire`] waits for exactly that —
//! blocking on global visibility instead would serialize the overlap the ring
//! exists for. Groups are **per thread** and age in issue order, so this wait
//! is taken by the same single thread that issued every store, and a `bar.sync`
//! after it is what releases the other warps past a wait they did not take.
//!
//! **`cp.async.bulk.wait_group`, once, at the end.** Nothing else makes a bulk
//! store visible to a following kernel or to the host — not the end of the
//! kernel, not a `bar.sync`, not dropping the handle. [`StoreRing::drain`] is
//! the only place it appears, because it is the only obligation the ring cannot
//! discharge incrementally.
//!
//! # Why this is not [`crate::shared::SharedTileRing`]
//!
//! That type is addressing and nothing else: `N` tiles and `index % N`, with
//! the completion mechanism supplied separately by a
//! [`crate::sync::SemaphoreRing`] the caller pairs it with. The pairing works
//! for *loads* because a load's destination is shared memory, where an mbarrier
//! can live and count arriving bytes. A store's destination is global memory,
//! where none can, so a store ring's completion is a per-thread group count and
//! there is no semaphore ring to pair with — the discipline has to live on the
//! ring itself, which is why the methods here are collective and
//! `SharedTileRing`'s are pure address arithmetic.
//!
//! The mechanical half of the same answer: the wait's group count is `DEPTH -
//! 1`, and `tma_store_wait_read::<{DEPTH - 1}>()` is not something Rust will
//! evaluate in generic-argument position without `generic_const_exprs` — the
//! dodge [`crate::shared::Element::Unpacked`] and
//! [`crate::shared::MmaElement::mma`] both take for the same restriction. So
//! the parameter here is `IN_FLIGHT`, the number the *instruction*
//! takes, and the buffer count is derived from it. A type parameterized that way
//! could not embed a `SharedTileRing<.., { IN_FLIGHT + 1 }>` either.

use core::marker::PhantomData;

use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::thread;
use cuda_device::tma::TmaDescriptor;

use crate::shared::{
    Element, SharedTile, Swizzle, tma_store_commit, tma_store_wait, tma_store_wait_read,
};

/// `IN_FLIGHT + 1` staging tiles of `[R, C]` `E` under swizzle `S`, cycled by a
/// cursor, with the fence and wait discipline of the module docs built in.
///
/// The handle is a base pointer, a slot index and compile-time shape — Copy and
/// register-resident like every other tile handle here, except that it carries
/// the cursor, so an `acquire`/`commit` pair cannot name different buffers.
/// Each thread holds its own copy and they stay in step because every thread of
/// the CTA runs the same loop, which the collective methods require anyway.
///
/// # `IN_FLIGHT`, and why it is not the buffer count
///
/// `IN_FLIGHT` is how many committed stores may still have their source reads
/// outstanding while the next band is written — the group count
/// `cp.async.bulk.wait_group.read` takes as an immediate, and therefore the
/// number this type can pass through to it. [`Self::DEPTH`] is `IN_FLIGHT + 1`
/// buffers, and the module docs carry why the parameter is this end of that
/// identity rather than the other.
///
/// So `IN_FLIGHT = 0` is the single-buffer ring — the store is fully drained
/// out of shared memory before the next band's `stmatrix` touches it, which is
/// the shape `examples/src/softmax.rs` writes by hand — and `IN_FLIGHT = 1` is
/// the two-buffer ring that overlaps one store with one drain.
pub struct StoreRing<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32> {
    base: *mut u8,
    slot: u32,
    _marker: PhantomData<(E, S)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32> Clone
    for StoreRing<E, R, C, S, IN_FLIGHT>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32> Copy
    for StoreRing<E, R, C, S, IN_FLIGHT>
{
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const IN_FLIGHT: u32>
    StoreRing<E, R, C, S, IN_FLIGHT>
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

    /// The buffer of stage `index % DEPTH` — the addressing the cursor walks,
    /// exposed because a kernel laying out its shared plan may want to name a
    /// buffer without acquiring it. It carries none of the ring's discipline:
    /// writing through this handle is writing to whatever the engine may still
    /// be reading.
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
    /// The wait is `cp.async.bulk.wait_group.read` at `IN_FLIGHT`
    /// groups: with `DEPTH` groups committed and ageing in issue order, leaving
    /// at most `IN_FLIGHT` with outstanding reads is exactly the statement that
    /// the oldest — this buffer's — is done being read. Before `DEPTH` bands
    /// have been committed the wait is trivially satisfied and costs an
    /// instruction, so the ring needs no fill phase.
    ///
    /// It waits for the reads only, not for the bytes to reach global memory.
    /// That is the whole of what makes a ring worth having, and it is why
    /// [`Self::drain`] still exists.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA must call this together, and the CTA must be the
    /// one whose thread 0 issued the ring's stores. It contains a
    /// `sync_threads`, so a thread that skips it either hangs the block or
    /// writes a buffer the engine is still reading — the second is silent.
    #[inline(always)]
    pub unsafe fn acquire(&mut self) -> SharedTile<E, R, C, S> {
        // Groups belong to the thread that committed them, so this wait means
        // something only on the issuing thread; the barrier is what the other
        // warps are released past. (The same reason `tma store early recycle`
        // in `device-tests` syncs between thread 0's wait and everyone's write.)
        if Self::issuing() {
            tma_store_wait_read::<IN_FLIGHT>();
        }
        thread::sync_threads();
        self.buffer(self.slot)
    }

    /// Publish the buffer [`Self::acquire`] handed out and hand it to the
    /// engine, aimed at a [`crate::global`] panel map exactly as
    /// [`SharedTile::tma_store`] aims it. The cursor moves on to the next
    /// buffer; the store is *issued*, not complete.
    ///
    /// One committed group per call, which is what makes [`Self::acquire`]'s
    /// group count mean "one buffer" — a caller that committed twice per
    /// acquire, or not at all, would be counting something else.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA must call this together, once per
    /// [`Self::acquire`], with every write into the buffer already issued by
    /// the calling thread. `map` must describe a live global buffer whose box
    /// shape matches `[R, SUBTILE_COLS]`, and the ring's memory must stay
    /// allocated until [`Self::drain`].
    #[inline(always)]
    pub unsafe fn commit(&mut self, map: *const TmaDescriptor, row: i32, plane: i32) {
        unsafe {
            let staging = self.publish();
            if Self::issuing() {
                staging.tma_store(map, row, plane);
                tma_store_commit();
            }
        }
        self.advance();
    }

    /// [`Self::commit`] against a 2-D tensor map, taking its coordinates in the
    /// order [`SharedTile::tma_store_2d`] takes them.
    ///
    /// # Safety
    ///
    /// As [`Self::commit`], for a rank-2 map.
    #[inline(always)]
    pub unsafe fn commit_2d(&mut self, map: *const TmaDescriptor, leading: i32, minor: i32) {
        unsafe {
            let staging = self.publish();
            if Self::issuing() {
                staging.tma_store_2d(map, leading, minor);
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
    /// Every thread of the CTA must call this together, after the last
    /// [`Self::commit`].
    #[inline(always)]
    pub unsafe fn drain(self) {
        if Self::issuing() {
            tma_store_wait::<0>();
        }
        thread::sync_threads();
    }

    /// Order every thread's `stmatrix` writes ahead of the engine's read of the
    /// current buffer, and hand that buffer to the caller to issue against.
    ///
    /// Both halves are load-bearing and in this order. The fence orders the
    /// writes of the thread executing it — so every thread that wrote a row has
    /// to execute one — and the barrier is what carries that ordering to the
    /// single thread that issues the store. Reversing them, or fencing only on
    /// the issuing thread, leaves the engine free to read bytes another warp
    /// has not published.
    #[inline(always)]
    unsafe fn publish(self) -> SharedTile<E, R, C, S> {
        unsafe { fence_proxy_async_shared_cta() };
        thread::sync_threads();
        self.buffer(self.slot)
    }

    /// The thread whose store groups the ring counts. One thread and not one
    /// warp: `cp.async.bulk.commit_group` gathers the *calling thread's*
    /// outstanding stores, so a group is a thread's and the wait that ages them
    /// has to be taken by the same one.
    #[inline(always)]
    fn issuing() -> bool {
        thread::threadIdx_x() == 0
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

    #[test]
    fn depth_is_one_more_than_the_stores_in_flight() {
        assert_eq!(Single::DEPTH, 1);
        assert_eq!(Double::DEPTH, 2);
        assert_eq!(StoreRing::<Bf16, 128, 64, Swizzle128B, 3>::DEPTH, 4);
    }

    /// The arithmetic a follow-up wiring this into a kernel with a shared
    /// budget is choosing on: a `[128, 64]` bf16 buffer is 16 KiB, so a depth
    /// is 16 KiB of the plan and nothing else in the type varies with it.
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
    /// its walk is worth pinning even though the collective halves need a
    /// device.
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
