//! # Softmax over the rows of a tile
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features softmax`.
//!
//! This is the kernel #5 and #6 were filed for, and it is five lines long once
//! they land. It was also the shortest possible demonstration of the hole
//! underneath both of them — the library could neither get data *out* of a
//! shared tile (#21) nor address one wider than a single swizzle atom (#25), so
//! a kernel whose input is not an MMA operand could not start. Both are closed:
//! [`kittens::ldst::load_fragment`] is the `ldmatrix` half of `ldst`, and
//! [`kittens::shared::SharedTile::chunk_writer`] now walks stacked subtiles, so
//! at `COLUMNS = 128` this tile has a register ↔ shared path in both
//! directions and the two movers below are the real API.
//!
//! Blocked on:
//!
//! - **#6** — `row_max` and `row_sum` on `RegTile`. The whole algorithm is four
//!   lines and two of them do not exist.
//! - **#9** — the TMA store path, so the result can leave.
//!
//! Not blocking, but visible: the movers are per-`[16, 16]`-block, so the band
//! loop is written twice below (#22). It compiles — it is a cost, not a hole.
//!
//! Note what is *not* in that list: nothing about layouts, swizzles,
//! semaphores, or the fragment map. Every gap here is an op or a mover, which
//! is what makes this example worth writing — it says the missing surface is
//! shallow and wide, not deep.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_fragment, store_fragment};
use kittens::reg::{BaseLdtm, Fragment, RegTile};
use kittens::shared::{Bf16, SharedTile, Swizzle128B, tma_store_commit, tma_store_wait};
use kittens::sync::Semaphore;

/// Rows per CTA — four warps of 32.
const ROWS: usize = 128;
/// Columns the softmax runs over. One row of the tile is one distribution.
const COLUMNS: usize = 128;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows of it.
type Band = RegTile<32, COLUMNS, BaseLdtm>;

pub const SHARED_BYTES: usize = Tile::BYTES + 32;
pub const THREADS: u32 = (ROWS / 32) as u32 * 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One warp's band of the tile, read a `[16, 16]` block at a time.
    ///
    /// This is #22's `load_tile(tile, row, 0, lane)` written out by hand: the
    /// mover is real and spans the tile, it just moves one block per call, so
    /// every kernel that wants a band writes this same two-deep loop.
    #[inline(always)]
    unsafe fn load_band(tile: Tile, row: u32, lane: u32) -> Band {
        unsafe {
            let chunks = tile.chunk_writer();
            let mut band = Band::zero();
            let mut row_block = 0usize;
            while row_block < 32 / 16 {
                let mut column_block = 0usize;
                while column_block < COLUMNS / 16 {
                    let fragment = load_fragment::<Bf16>(
                        chunks,
                        row + 16 * row_block as u32,
                        16 * column_block as u32,
                        lane,
                    );
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            band.set(
                                2 * row_block + slot,
                                4 * column_block + value,
                                fragment.get(slot, value),
                            );
                            value += 1;
                        }
                        slot += 1;
                    }
                    column_block += 1;
                }
                row_block += 1;
            }
            band
        }
    }

    /// The mirror of [`load_band`] — #22's `store_tile`, same loop, opposite
    /// direction.
    #[inline(always)]
    unsafe fn store_band(tile: Tile, row: u32, lane: u32, band: Band) {
        unsafe {
            let chunks = tile.chunk_writer();
            let mut row_block = 0usize;
            while row_block < 32 / 16 {
                let mut column_block = 0usize;
                while column_block < COLUMNS / 16 {
                    let mut fragment = Fragment::zero();
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            fragment.set(
                                slot,
                                value,
                                band.get(2 * row_block + slot, 4 * column_block + value),
                            );
                            value += 1;
                        }
                        slot += 1;
                    }
                    store_fragment::<Bf16>(
                        chunks,
                        row + 16 * row_block as u32,
                        16 * column_block as u32,
                        lane,
                        fragment,
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
        }
    }

    /// Row-wise `softmax` of one `[ROWS, COLUMNS]` tile per CTA, in place:
    /// TMA in, normalize in registers, TMA out.
    #[kernel]
    pub unsafe fn softmax_rows(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        mut plane: i32,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::from_raw(smem);
            let loaded = Semaphore::attach(smem.add(Tile::BYTES) as *mut Barrier);

            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let row_base = 32 * warp_id;
            plane += thread::blockIdx_y() as i32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_load(
                    source,
                    (ROWS as u32 * thread::blockIdx_x()) as i32,
                    plane,
                    loaded,
                );
                loaded.expect_tx(Tile::BYTES as u32);
            }
            loaded.wait(0);
            thread::sync_threads();

            let x = load_band(tile, row_base, lane);

            // WANT (#6): the two reductions. Everything else in the algorithm
            // is a `RegTile` op that exists (#5).
            let x = x.sub_row(x.row_max()).exp2();
            let x = x.div_row(x.row_sum());

            store_band(tile, row_base, lane, x);
            // `stmatrix` writes through the generic proxy; the TMA engine reads
            // through the async one.
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            // WANT (#9): the TMA store side. `cp_async_bulk_tensor_*_s2g` and
            // the commit/wait groups are all present upstream at the pinned
            // revision — this is the smallest unblocked item in the backlog.
            if thread::threadIdx_x() == 0 {
                tile.tma_store(
                    destination,
                    (ROWS as u32 * thread::blockIdx_x()) as i32,
                    plane,
                );
                tma_store_commit();
                tma_store_wait::<0>();
                loaded.inval();
            }
        }
    }
}
