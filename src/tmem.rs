//! TMEM accumulator views.
//!
//! A tcgen05 accumulator lives in tensor memory, addressed as
//! `base + (row << 16) + column`: the high half-word selects one of the
//! 128 TMEM lanes (accumulator rows), the low half-word a 4-byte column.
//! `TmemTile` carries that address plus the logical `[R, C]` fp32 shape,
//! so kernel code names its `S`/`O`/`dP` segments as tiles instead of
//! threading bare `u32` addresses and `(row << 16) + column` arithmetic
//! through every drain loop.
//!
//! MMA shapes with phantom rows (`M128` over 64-row tiles) still
//! type the *drained* shape: `R`/`C` describe what the kernel reads back,
//! not what the instruction touches.

use cuda_device::cusimd::{CuSimd, TmemRegs4};
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_dealloc, tcgen05_ld_16x256b_pure, tcgen05_load_wait,
    tcgen05_st_16x256b_x1_raw, tcgen05_store_wait,
};
use cuda_device::{thread, warp};

use crate::reg::Fragment;

/// CTA-wide TMEM allocation, the lifecycle every tcgen05 kernel opens with:
/// warp 0 allocates `columns` into the shared staging word, a block sync
/// publishes it, and every thread reads back the address.
///
/// # Safety
///
/// Every thread of the block must call this together; `slot` must point to
/// shared memory (a `SharedArray<u32, 1, 4>` static).
#[inline(always)]
pub unsafe fn alloc_block(slot: *mut u32, columns: u32) -> u32 {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_alloc(slot, columns);
        }
        thread::sync_threads();
        *(slot as *const u32)
    }
}

/// Release the CTA's TMEM allocation from warp 0, after the caller's own
/// fence/sync has retired every outstanding read of it.
///
/// # Safety
///
/// No thread may touch the allocation afterwards; `columns` must match the
/// [`alloc_block`] call.
#[inline(always)]
pub unsafe fn dealloc_block(address: u32, columns: u32) {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_dealloc(address, columns);
        }
    }
}

/// Retire the calling warp's outstanding tcgen05 stores.
///
/// Split out of [`TmemTile::store_fragment`] rather than folded into it
/// because a store's registers are consumed at issue: a pass writing many
/// fragments waits once after the last one instead of once per fragment.
///
/// # Safety
///
/// All 32 lanes of the warp must call this together, and it retires only that
/// warp's stores — another warp reading the same TMEM needs its own ordering.
#[inline(always)]
pub unsafe fn store_wait() {
    tcgen05_store_wait()
}

/// Resolve a `16x256b` pair's eight registers into a fragment's
/// `(slot, value)` coordinates.
///
/// A `16x256b` load hands a thread its two rows of the block by two columns,
/// row-major: register `2*slot + pair` is slot `slot`'s column `pair` of that
/// load's 8-column half. The half at `column` is the fragment's values
/// `{0, 1}`, the half at `column + 8` its values `{2, 3}`.
#[inline(always)]
fn interleave(low: TmemRegs4, high: TmemRegs4) -> Fragment {
    let mut tile = Fragment::zero();
    let mut reg = 0;
    while reg < 4 {
        let (slot, pair) = (reg / 2, reg % 2);
        tile.set(slot, pair, low[reg]);
        tile.set(slot, 2 + pair, high[reg]);
        reg += 1;
    }
    tile
}

/// The inverse of [`interleave`] — a fragment back into the two collective
/// accesses' registers, low half first.
#[inline(always)]
fn split(tile: Fragment) -> (TmemRegs4, TmemRegs4) {
    let mut low = [0.0f32; 4];
    let mut high = [0.0f32; 4];
    let mut reg = 0;
    while reg < 4 {
        let (slot, pair) = (reg / 2, reg % 2);
        low[reg] = tile.get(slot, pair);
        high[reg] = tile.get(slot, 2 + pair);
        reg += 1;
    }
    (CuSimd::new(low), CuSimd::new(high))
}

/// fp32 registers as the raw words every `tcgen05_st_*` form takes. There is
/// no `_pure` store to mirror `tcgen05_ld_16x256b_pure`, and the `_unpack16`
/// form splits each register into 16-bit halves — an accumulator's fp32 has to
/// land bit-exact, so it goes through `_raw` and `to_bits`.
#[inline(always)]
fn raw_bits(values: TmemRegs4) -> CuSimd<u32, 4> {
    CuSimd::new([
        values[0].to_bits(),
        values[1].to_bits(),
        values[2].to_bits(),
        values[3].to_bits(),
    ])
}

/// An fp32 accumulator segment in tensor memory.
#[derive(Clone, Copy)]
pub struct TmemTile<const R: usize, const C: usize> {
    address: u32,
}

impl<const R: usize, const C: usize> TmemTile<R, C> {
    /// Wrap a TMEM address (as returned through `tcgen05_alloc`'s shared
    /// staging word, plus any column offset already applied).
    pub const fn from_raw(address: u32) -> Self {
        Self { address }
    }

    /// The raw address, for the MMA issue path.
    pub const fn raw(self) -> u32 {
        self.address
    }

    /// Address of `(row, column)`: rows ride the high half-word.
    pub const fn at(self, row: u32, column: u32) -> u32 {
        self.address + (row << 16) + column
    }

    /// The same-shaped segment `columns` fp32 columns to the right —
    /// accumulator ping-pong stages (gemm's `accum_stage * 256`), or the
    /// N-half a second MMA of a wider allocation writes.
    pub const fn columns_right(self, columns: u32) -> Self {
        Self {
            address: self.address + columns,
        }
    }

    /// The next segment of the same allocation, `C2` columns wide: one
    /// `tcgen05_alloc` of `C + C2` columns carved into a `[R, C]` and a
    /// `[R, C2]` tile, as flash's `S` and `O` share one allocation.
    ///
    /// The offset is `C` rather than a parameter because a TMEM segment
    /// spans all 128 lanes — segments can only be carved along the column
    /// axis, so the one that follows this tile begins where it ends. An
    /// explicit offset could name a column inside this tile instead, and
    /// overlapping segments have no diagnostic: the two MMAs simply write
    /// each other's accumulator.
    pub const fn split_columns<const C2: usize>(self) -> TmemTile<R, C2> {
        TmemTile {
            address: self.address + C as u32,
        }
    }

    /// One thread's eight-value fragment of the 16-row block at `row`:
    /// two `16x256b` collective loads at `column` and `column + 8`, each
    /// drained through `tcgen05_load_wait`. Under the 16x256b map a thread
    /// holds rows `lane/4` and `lane/4 + 8` of the block at column offsets
    /// `2*(lane%4) + {0, 1, 8, 9}` — the low simd carries the `{0, 1}`
    /// offsets of both rows, the high simd the `{8, 9}` offsets (which is
    /// what [`Self::fragment_tile`]'s interleaving spells out).
    ///
    /// # Safety
    ///
    /// All 32 lanes of a warp that owns the TMEM rows `row..row+16` must
    /// call this together, after the MMA writing them has committed.
    #[inline(always)]
    pub unsafe fn fragment(self, row: u32, column: u32) -> (TmemRegs4, TmemRegs4) {
        unsafe {
            let low = tcgen05_ld_16x256b_pure(self.at(row, column));
            tcgen05_load_wait();
            let high = tcgen05_ld_16x256b_pure(self.at(row, column + 8));
            tcgen05_load_wait();
            (low, high)
        }
    }

    /// [`Self::fragment`] as a [`Fragment`] tile — the same eight values with
    /// the simd pair's interleaving resolved into the layout's `(slot, value)`
    /// coordinates, so a register pass indexes by row and column instead of by
    /// which of the two collective loads a value arrived in.
    ///
    /// A `16x256b` load hands a thread its two rows of the block by two
    /// columns, row-major: register `2*slot + pair` is slot `slot`'s column
    /// `pair` of that load's 8-column half. The half at `column` is the
    /// fragment's values `{0, 1}`, the half at `column + 8` its values
    /// `{2, 3}` — the `{0, 1, 8, 9}` offsets of
    /// [`BaseLdtm::column`](crate::reg::BaseLdtm::column).
    ///
    /// # Safety
    ///
    /// As [`Self::fragment`].
    #[inline(always)]
    pub unsafe fn fragment_tile(self, row: u32, column: u32) -> Fragment {
        unsafe {
            let (low, high) = self.fragment(row, column);
            interleave(low, high)
        }
    }

    /// Write one thread's eight-value fragment back into the 16-row block at
    /// `row`: two `16x256b` collective stores at `column` and `column + 8`,
    /// under the same lane → `(row, column)` ownership the load side of
    /// [`Self::fragment`] reads through, so `low` and `high` are exactly what
    /// that call returned.
    ///
    /// Unlike [`Self::fragment`] this does not wait. A load must, because the
    /// registers it waits on *are* its return value; a store's registers are
    /// consumed at issue, so a pass writing many fragments pays one
    /// [`store_wait`] at the end rather than one per fragment.
    ///
    /// # Safety
    ///
    /// All 32 lanes of the warp owning TMEM rows `row..row+16` must call this
    /// together. Ordering the write against whatever reads it — an MMA taking
    /// the segment as accumulator, a later [`Self::fragment`], or
    /// [`dealloc_block`] — is the caller's: [`store_wait`] retires it in the
    /// issuing warp, and a consumer in another warp needs whatever sync it
    /// would need for any warp-private write besides.
    ///
    /// Whether a `tcgen05_fence_before_thread_sync` /
    /// `tcgen05_fence_after_thread_sync` pair is additionally required around
    /// such a hand-off is open: this crate uses neither on any path, and
    /// nothing has been established either way for the store side. Their
    /// absence here is not a guarantee that they are unnecessary.
    #[inline(always)]
    pub unsafe fn store_fragment(self, row: u32, column: u32, low: TmemRegs4, high: TmemRegs4) {
        unsafe {
            tcgen05_st_16x256b_x1_raw(self.at(row, column), raw_bits(low));
            tcgen05_st_16x256b_x1_raw(self.at(row, column + 8), raw_bits(high));
        }
    }

    /// [`Self::store_fragment`] from a [`Fragment`] tile — the store twin of
    /// [`Self::fragment_tile`], undoing the simd pair's interleaving so a
    /// register pass hands back the same `(slot, value)` coordinates it read.
    ///
    /// Only the `[16, 16]` block: composing a taller or wider shape out of
    /// these is the caller's, exactly as it is on the drain side.
    ///
    /// # Safety
    ///
    /// As [`Self::store_fragment`], including the wait it does not do.
    #[inline(always)]
    pub unsafe fn store_fragment_tile(self, row: u32, column: u32, tile: Fragment) {
        unsafe {
            let (low, high) = split(tile);
            self.store_fragment(row, column, low, high);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addressing_rides_the_high_half_word() {
        let tile = TmemTile::<128, 64>::from_raw(0x0001_0000);
        assert_eq!(tile.at(0, 0), 0x0001_0000);
        assert_eq!(tile.at(32, 24), 0x0021_0018);
        assert_eq!(tile.columns_right(64).at(0, 0), 0x0001_0040);
    }

    /// Flash's carve: one `[128, 64 + 128]` allocation as a score segment
    /// and an output segment beside it. The output's first column has to be
    /// the score's last plus one, in both directions — a short offset
    /// overlaps the scores and a long one runs off the allocation, and TMEM
    /// reports neither.
    #[test]
    fn split_columns_carves_an_allocation_without_overlap() {
        const KEYS: usize = 64;
        const HEAD: usize = 128;

        let base = 0x0000_0000;
        let scores = TmemTile::<128, KEYS>::from_raw(base);
        let output: TmemTile<128, HEAD> = scores.split_columns();

        assert_eq!(output.at(0, 0), scores.at(0, KEYS as u32 - 1) + 1);
        assert_eq!(
            output.at(127, HEAD as u32 - 1),
            base + (127 << 16) + (KEYS + HEAD) as u32 - 1
        );

        // A third segment chains off the second, not off the first.
        let spare = output.split_columns::<32>();
        assert_eq!(spare.at(0, 0), base + (KEYS + HEAD) as u32);
    }

    /// The simd-pair interleaving, both directions. `split` is what the store
    /// path hands the hardware and `interleave` what the drain path reads back
    /// from it, so a fragment that survives the pair unchanged is the whole
    /// register-side contract between them — the addressing either shares is
    /// what the device harness is for.
    #[test]
    fn splitting_a_fragment_inverts_the_pair_interleaving() {
        let mut tile = Fragment::zero();
        for slot in 0..Fragment::SLOTS {
            for value in 0..Fragment::VALUES {
                tile.set(slot, value, (Fragment::VALUES * slot + value) as f32);
            }
        }

        let (low, high) = split(tile);
        for reg in 0..4 {
            assert_eq!(low[reg], tile.get(reg / 2, reg % 2));
            assert_eq!(high[reg], tile.get(reg / 2, 2 + reg % 2));
        }

        let restored = interleave(low, high);
        for slot in 0..Fragment::SLOTS {
            for value in 0..Fragment::VALUES {
                assert_eq!(restored.get(slot, value), tile.get(slot, value));
            }
        }
    }

    #[test]
    fn raw_bits_is_the_fp32_pattern_unchanged() {
        let values = TmemRegs4::new([0.0, -0.0, 1.5, f32::from_bits(0x0000_8001)]);
        let bits = raw_bits(values);
        for reg in 0..4 {
            assert_eq!(bits[reg], values[reg].to_bits());
        }
    }
}
