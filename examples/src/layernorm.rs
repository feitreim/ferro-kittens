//! # Layernorm over a tile, and the whole-tile reduction next to it
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features layernorm`.
//!
//! Layernorm is softmax's gaps plus two more, and both are about *vectors*
//! rather than tiles:
//!
//! - **a per-column register vector.** [`kittens::reg::RegVec`] is indexed by
//!   *row*: the fragment map replicates a row's statistic across a quad, which
//!   is what makes [`kittens::reg::quad_max`] one shuffle. A per-column
//!   statistic has the opposite access pattern — one column's values are
//!   spread across lanes at stride 4 and across both row blocks a thread owns
//!   — so it needs its own type and its own shuffle, exactly as **#6** says.
//!   #6 names the reduction (`col_sum`); it does not name the type the
//!   reduction returns, and `mul_col`/`add_col` in **#5** need that type to
//!   exist first. `gamma` and `beta` are per-column, so a layernorm cannot be
//!   written without it.
//! - **shared vectors (#13).** `gamma`/`beta` are parameters, not activations:
//!   they are loaded once, shared by every row, and belong in shared memory.
//!   With no `SharedVec` they would be re-read from global by every thread.
//!
//! Blocked on: **#5** (`sub_row`, `mul_row`, `mul_col`, `add_col`, `rsqrt`,
//! scalar `scale`/`shift`, elementwise `mul`), **#6** (`row_sum`, whole-tile
//! `sum`, and the column-vector type above), **#13** (`SharedVec` and its TMA
//! path), **#9** (the store side), and the two movers `softmax.rs` names:
//! `ldst::load_tile` and `ldst::store_tile` — whole-tile forms of the
//! per-`[16, 16]`-block movers, which is **#22**, over a cursor that refuses a
//! 128-wide tile, which is **#25**.
//!
//! The second kernel here, [`kernels::groupnorm_tile`], normalizes over the
//! *whole* tile rather than per row. It costs one more thing than the first:
//! a whole-tile reduction is warp-scope in TK's register-tile form, but a tile
//! spread over four warps needs the warps to agree. That is **#3** (scope
//! parameterization) and **#13** (a shared vector to stage the partials
//! through) meeting — and it is the first place in these four kernels where
//! the missing piece is structural rather than an op.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_tile, store_tile};
use kittens::reg::{BaseLdtm, RegColVec, RegTile, rsqrt};
use kittens::shared::{Bf16, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait};
use kittens::sync::{Semaphore, block_reduce_sum};

const ROWS: usize = 128;
const COLUMNS: usize = 128;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows of it.
type Band = RegTile<32, COLUMNS, BaseLdtm>;
/// A per-column statistic or parameter — one value per logical column,
/// replicated across the lanes that hold that column.
type Columns = RegColVec<COLUMNS, BaseLdtm>;
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
                // WANT (#13, #9): a vector's own TMA path. A `[1, COLUMNS]`
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

            let mut x: Band = load_tile(tile, row_base as usize, 0, lane);

            // WANT (#5, #6). `scale`/`shift` are the scalar forms of #5's
            // `mul`/`sum` ops; `rsqrt` is on its missing list by name.
            let mean = x.row_sum().scale(1.0 / COLUMNS as f32);
            x.sub_row(mean);
            let variance = x.mul(x).row_sum().scale(1.0 / COLUMNS as f32);
            x.mul_row(variance.shift(epsilon).rsqrt());
            x.mul_col(g);
            x.add_col(b);

            store_tile(tile, row_base, 0, lane, x);
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

            let mut x: Band = load_tile(tile, row_base as usize, 0, lane);

            // WANT (#6): whole-tile reductions — `sum` folds a band to one
            // f32 replicated across the warp.
            //
            // WANT (#3, #13): and then the warps have to agree. A whole-tile
            // statistic over a tile that four warps own is a block-scope
            // reduction, which is a `Scope` the library does not have and a
            // shared staging vector it does not have either. Every kernel
            // that needs one writes this by hand today.
            let mean = block_reduce_sum(partials, x.sum()) * scale;
            x.shift(-mean);
            let variance = block_reduce_sum(partials, x.mul(x).sum()) * scale;
            x.scale(rsqrt(variance + epsilon));

            store_tile(tile, row_base, 0, lane, x);
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
