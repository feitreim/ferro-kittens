//! # Softmax over the rows of a tile
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features softmax`.
//!
//! This is the kernel #5 and #6 were filed for, and it is five lines long once
//! they land. It is also the shortest possible demonstration of the hole
//! underneath both of them: **the library has no way to get data out of a
//! shared tile.** `ldst` is store-only, `SharedTile` hands out raw pointers
//! and a swizzled-chunk cursor, and `TmemTile::fragment_tile` reads TMEM.
//! Nothing reads a shared tile into registers. A kernel whose input is not an
//! MMA operand — every normalization, every activation, every elementwise
//! epilogue — cannot start.
//!
//! Blocked on:
//!
//! - **no open issue** — `ldst::load_tile`, shared → register. GAPS §2.2 asks
//!   for global ↔ register (#11) and §2.1 for register → TMEM (#10); shared →
//!   register is absent from both the gap inventory and the backlog. The
//!   upstream note that `ldmatrix` exists only in cuda-oxide's Hopper `wmma`
//!   path is the reason it was never written, not a reason it is not needed:
//!   the swizzled address math for a plain vectorized load is already in
//!   [`kittens::shared::SwizzledChunks`].
//! - **no open issue** — `ldst::store_tile`, the whole-tile mirror of
//!   [`kittens::ldst::store_fragment`], and one that can span stacked
//!   subtiles. [`kittens::shared::SharedTile::chunk_writer`] refuses any tile
//!   wider than one swizzle atom (64 bf16 columns), so today a `[128, 128]`
//!   tile has no register → shared path at all.
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

use kittens::ldst::{load_tile, store_tile};
use kittens::reg::{BaseLdtm, RegTile};
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

            // WANT (no open issue): shared → register. The inverse of
            // `store_tile`, at the same coordinates, under the same layout.
            //
            //     ldst::load_tile::<E, R, C, S, M, N, L>(tile, row, column, lane)
            //         -> RegTile<M, N, L>
            let mut x: Band = load_tile(tile, row_base as usize, 0, lane);

            // WANT (#6, #5): the whole algorithm.
            let m = x.row_max();
            x.sub_row(m);
            x.exp2();
            let total = x.row_sum();
            x.div_row(total);

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
