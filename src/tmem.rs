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

use cuda_device::cusimd::TmemRegs4;
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_dealloc, tcgen05_ld_16x256b_pure, tcgen05_load_wait,
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

    /// The segment `columns` fp32 columns to the right — accumulator
    /// ping-pong stages (gemm's `accum_stage * 256`) or a second output
    /// band sharing one allocation.
    pub const fn columns_right(self, columns: u32) -> Self {
        Self {
            address: self.address + columns,
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
}
