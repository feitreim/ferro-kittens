//! # Layernorm over a tile, and the whole-tile reduction next to it
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features layernorm`.
//!
//! Both kernels here end the way softmax does, and that ending is now real:
//! `tma_store` + `tma_store_commit` + `tma_store_wait::<0>()` (#9). What keeps
//! this file aspirational is everything in front of it.
//!
//! Layernorm was softmax's gaps plus two about *vectors* rather than tiles.
//! One of them is closed: [`kittens::reg::ColVec`] exists (#5) and
//! [`kittens::reg::RegTile::col_reduce`] now fills it (#6), folding a column's
//! values across the 8 lanes that share `lane % 4` — the stride-4/8/16
//! butterfly, which is a different shuffle from the quad reduction a row
//! statistic uses, not a reparameterization of it. The other is still open:
//!
//! - **shared vectors (#13).** `gamma`/`beta` are parameters, not activations:
//!   they are loaded once, shared by every row, and belong in shared memory.
//!   With no `SharedVec` they would be re-read from global by every thread.
//!
//! Blocked on: **#13** (`SharedVec` and its TMA path), and — for
//! [`kernels::groupnorm_tile`] only — **#3**, discussed below.
//!
//! Nothing arithmetic is left. The scalar broadcasts this file used to name as
//! a gap — `scale`/`shift` on a tile or a vector, and a free `rsqrt(f32)` for
//! a variance that is already one `f32` — landed with **#38**, so both kernels
//! below spell their normalization in the real API and every remaining `WANT`
//! is a mover or a scope.
//!
//! The second kernel here, [`kernels::groupnorm_tile`], normalizes over the
//! *whole* tile rather than per row, and that is the one place a landed #6
//! does not reach. `RegTile::tile_sum` is warp scope: it folds a warp's own
//! band across all 32 lanes and stops. A `[128, COLUMNS]` tile is four warps'
//! bands, so the warps have to agree, and that needs a shared staging buffer
//! and a block barrier — storage, not a shuffle. **#3**'s `Scope` (`WARPS`,
//! `THREADS`, `rank`, `sync`) supplies neither, which is why #6 stopped at the
//! warp: the block form is **#3** and **#13** meeting, and it is the first
//! place in these four kernels where the missing piece is structural rather
//! than an op.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_tile, store_tile};
use kittens::reg::{BaseLdtm, ColVec, RegTile, rsqrt};
use kittens::shared::{Bf16, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait};
use kittens::sync::{Semaphore, block_reduce_sum};

const ROWS: usize = 128;
const COLUMNS: usize = 128;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows of it.
type Band = RegTile<32, COLUMNS, BaseLdtm>;
/// A per-column statistic or parameter — one value per logical column,
/// replicated across the lanes that hold that column.
type Columns = ColVec<COLUMNS, BaseLdtm>;
/// `gamma` and `beta`, staged once per CTA.
type Parameters = SharedVec<Bf16, COLUMNS>;

pub const SHARED_BYTES: usize = Tile::BYTES + 2 * Parameters::BYTES + 64;
pub const THREADS: u32 = (ROWS / 32) as u32 * 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `y = gamma ⊙ (x - mean(x)) / sqrt(var(x) + eps) + beta`, per row.
    #[kernel]
    pub unsafe fn layernorm_rows(
        source: *const TmaDescriptor,
        gamma_map: *const TmaDescriptor,
        beta_map: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        epsilon: f32,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::from_raw(smem);
            let gamma = Parameters::from_raw(smem.add(Tile::BYTES));
            let beta = Parameters::from_raw(smem.add(Tile::BYTES + Parameters::BYTES));
            let loaded =
                Semaphore::attach(smem.add(Tile::BYTES + 2 * Parameters::BYTES) as *mut Barrier);

            let lane = warp::lane_id();
            let row_base = 32 * warp::warp_id();
            let row = (ROWS as u32 * thread::blockIdx_x()) as i32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_load(source, row, 0, loaded);
                // WANT (#13): a vector's own TMA path. A `[1, COLUMNS]`
                // box is not a tile, and forcing it to be one wastes a whole
                // swizzle atom per row of padding.
                gamma.tma_load(gamma_map, 0, loaded);
                beta.tma_load(beta_map, 0, loaded);
                loaded.expect_tx((Tile::BYTES + 2 * Parameters::BYTES) as u32);
            }
            loaded.wait(0);
            thread::sync_threads();

            // WANT (#13): shared → register for a vector, into the
            // column-indexed register vector `mul_col` consumes.
            let g: Columns = gamma.to_registers(lane);
            let b: Columns = beta.to_registers(lane);

            let x: Band = load_tile(tile.chunk_writer(), row_base, 0, lane);

            // The whole algorithm, in the real API. `scale`/`shift` (#38) keep
            // `1/COLUMNS` and `epsilon` in a register: before them a constant
            // had to be splatted into a whole `RegVec` to be combined with one.
            let mean = x.row_sum().scale(1.0 / COLUMNS as f32);
            let x = x.sub_row(mean);
            let variance = x.mul(x).row_sum().scale(1.0 / COLUMNS as f32);
            let x = x
                .mul_row(variance.shift(epsilon).rsqrt())
                .mul_col(g)
                .add_col(b);

            store_tile(tile.chunk_writer(), row_base, 0, lane, x);
            // `store_tile` writes through the generic proxy and the TMA engine
            // reads through the async one, so the store side owes this fence
            // exactly as an MMA reading the same tile would.
            fence_proxy_async_shared_cta();
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_store(destination, row, 0);
                tma_store_commit();
                tma_store_wait::<0>();
                loaded.inval();
            }
        }
    }

    /// The same normalization over the whole tile instead of per row — group
    /// norm's statistic, and the one that needs the four warps to agree.
    #[kernel]
    pub unsafe fn groupnorm_tile(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        epsilon: f32,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::from_raw(smem);
            let loaded = Semaphore::attach(smem.add(Tile::BYTES) as *mut Barrier);
            let partials = smem.add(Tile::BYTES + 32) as *mut f32;

            let lane = warp::lane_id();
            let row_base = 32 * warp::warp_id();
            let row = (ROWS as u32 * thread::blockIdx_x()) as i32;
            let scale = 1.0 / (ROWS * COLUMNS) as f32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_load(source, row, 0, loaded);
                loaded.expect_tx(Tile::BYTES as u32);
            }
            loaded.wait(0);
            thread::sync_threads();

            let x: Band = load_tile(tile.chunk_writer(), row_base, 0, lane);

            // `tile_sum` (#6) folds this warp's band to one f32, warp-uniform.
            //
            // WANT (#3, #13): and then the four warps have to agree. That step
            // is not a wider shuffle — it is a shared staging buffer plus a
            // block barrier, so #3's `Scope` (`WARPS`, `THREADS`, `rank`,
            // `sync`) does not by itself express it: something has to say
            // where the partials live, and `partials` below is this kernel
            // guessing. Every kernel that needs one writes this by hand today.
            let mean = block_reduce_sum(partials, x.tile_sum()) * scale;
            // `shift`/`scale` against a warp-uniform `f32` (#38), and the free
            // `rsqrt` beside them: this statistic never becomes a vector, so
            // there was nothing for `RegVec::rsqrt` to ride on.
            let x = x.shift(-mean);
            let variance = block_reduce_sum(partials, x.mul(x).tile_sum()) * scale;
            let x = x.scale(rsqrt(variance + epsilon));

            store_tile(tile.chunk_writer(), row_base, 0, lane, x);
            // `store_tile` writes through the generic proxy and the TMA engine
            // reads through the async one, so the store side owes this fence
            // exactly as an MMA reading the same tile would.
            fence_proxy_async_shared_cta();
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_store(destination, row, 0);
                tma_store_commit();
                tma_store_wait::<0>();
                loaded.inval();
            }
        }
    }
}
