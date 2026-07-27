//! # Softmax over the rows of a tile
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features softmax`.
//!
//! This is the kernel #5 and #6 were filed for, and it is five lines long once
//! they land. It was also the shortest possible demonstration of the hole
//! underneath both of them — the library had no way to get data *out* of a
//! shared tile at all, so a kernel whose input is not an MMA operand could not
//! start. #21 closed that: [`kittens::ldst::load_fragment`] is the `ldmatrix`
//! half of `ldst`, and this file now calls it.
//!
//! What is left of that hole is shape, not direction. `load_fragment` moves one
//! `[16, 16]` block, and the cursor it addresses through refuses a tile this
//! wide — so the two `GAP` blocks below are the movers not spanning a tile,
//! rather than the movers not existing.
//!
//! Blocked on:
//!
//! - **#25** — [`kittens::shared::SharedTile::chunk_writer`] refuses any tile
//!   wider than one swizzle atom (64 bf16 columns), so this `[128, 128]` tile
//!   has no register ↔ shared path in *either* direction. The hard blocker:
//!   with the width halved the rest of this kernel would run.
//! - **#22** — the movers are per-`[16, 16]`-block, so filling a `[32, 128]`
//!   band is a composition loop the kernel writes by hand, and there is no
//!   `store_tile` mirror of it at all.
//! - **#5** — `sub_row`, `exp2`, `div_row` on `RegTile`.
//! - **#6** — `row_max`, `row_sum` on `RegTile`.
//! - **#9** — the TMA store path, so the result can leave.
//!
//! Note what is *not* in that list: nothing about layouts, swizzles,
//! semaphores, or the fragment map. Every gap here is an op or a mover, which
//! is what makes this example worth writing — it says the missing surface is
//! shallow and wide, not deep.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_fragment, store_tile};
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

            // ---- GAP (#25, #22: the movers do not span this tile) ----------
            // What the next fifteen lines want to be:
            //
            //     let mut x: Band = load_tile(tile, row_base, 0, lane);
            //
            // `load_fragment` is the real shared → register path (#21) and it
            // is what does the work below — but it moves one [16, 16] block, so
            // the band is a composition loop (#22), and `chunk_writer`
            // const-asserts a one-subtile tile, so at COLUMNS = 128 this line
            // does not compile at all (#25). Halve COLUMNS and it does.
            let chunks = tile.chunk_writer();
            let mut x = Band::zero();
            let mut row_block = 0usize;
            while row_block < 32 / 16 {
                let mut column_block = 0usize;
                while column_block < COLUMNS / 16 {
                    let fragment = load_fragment::<Bf16>(
                        chunks,
                        row_base + 16 * row_block as u32,
                        16 * column_block as u32,
                        lane,
                    );
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            x.set(
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
            // ---- end GAP ---------------------------------------------------

            // WANT (#6, #5): the whole algorithm.
            let m = x.row_max();
            x.sub_row(m);
            x.exp2();
            let total = x.row_sum();
            x.div_row(total);

            // WANT (#22, #25): the store mirror of the loop above. There is no
            // `store_tile`, and `store_fragment` would hit the same
            // one-subtile assertion.
            store_tile(tile, row_base, 0, lane, x);
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
