//! # Layernorm over a tile, and the whole-tile reduction next to it
//!
//! **Status: both kernels compile** and both are in the default build. They
//! differ in exactly one thing — the axis their statistic is taken over — and
//! that is what kept them apart for two issues.
//!
//! `layernorm_rows` was blocked on **#13**: `gamma`/`beta` are parameters, not
//! activations. They are loaded once, shared by every row, and belong in shared
//! memory, so with no `SharedVec` every thread would have re-read them from
//! global. [`kittens::shared::SharedVec`] is now that type — one flat run of
//! elements with its own single-box TMA path, unswizzled because a swizzle
//! atom is a statement about a tile's *rows* and a vector has one — and
//! [`kittens::ldst::load_vec`] broadcasts it into the `ColVec` `mul_col`
//! consumes.
//!
//! [`kernels::groupnorm_tile`] normalizes over the *whole* tile rather than per
//! row, and that was the one place a landed #6 did not reach.
//! `RegTile::tile_sum` is warp scope: it folds a warp's own band across all 32
//! lanes and stops. A `[128, COLUMNS]` tile is four warps' bands, so the warps
//! have to agree — and warps cannot shuffle to each other, so what closes the
//! gap is *storage* plus a block barrier rather than another butterfly.
//!
//! **#3** shipped both halves of that: [`kittens::sync::block_reduce_sum`]
//! folds one warp-uniform value per warp into one block-uniform value through
//! a `SharedVec<F32, WARPS>`, and the `impl Element for F32` under it is what
//! lets four fp32 partials be a shared vector at all. The two calls below are
//! back to back on the same scratch with no barrier between them, which is a
//! property of the collective and not an accident: it syncs on both sides, so
//! the variance pass cannot overtake the mean pass's readers.
//!
//! Neither kernel has a launcher or a CPU reference yet, which is the only
//! thing between this file and **runs**.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_tile, load_vec, store_tile};
use kittens::reg::{BaseLdtm, ColVec, RegTile, rsqrt};
use kittens::shared::{
    Bf16, F32, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait,
};
use kittens::sync::{Semaphore, block_reduce_sum};

const ROWS: usize = 128;
const COLUMNS: usize = 128;
/// Warps a CTA runs, and so the number of partials a whole-tile statistic is
/// folded from — one band each.
const WARPS: usize = ROWS / 32;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows of it.
type Band = RegTile<32, COLUMNS, BaseLdtm>;
/// A per-column statistic or parameter — one value per logical column,
/// replicated across the lanes that hold that column.
type Columns = ColVec<COLUMNS, BaseLdtm>;
/// `gamma` and `beta`, staged once per CTA. `COLUMNS` bf16 is 256 bytes: the
/// vector's own length, where a one-row `SharedTile` would have spent a whole
/// 128-byte swizzle atom to hold 64 of them.
type Parameters = SharedVec<Bf16, COLUMNS>;
/// The four warps' partial statistics, between the two barriers a block
/// reduction is made of. fp32 and not bf16: a partial rounded on its way
/// through shared memory would lose eight bits of the sum, and the variance
/// pass would inherit the error of the mean pass. 16 bytes — one TMA line, and
/// the narrowest vector [`SharedVec`] admits.
type Partials = SharedVec<F32, WARPS>;

pub const SHARED_BYTES: usize = Tile::BYTES + 2 * Parameters::BYTES + 64;
pub const THREADS: u32 = (WARPS * 32) as u32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `y = gamma ⊙ (x - mean(x)) / sqrt(var(x) + eps) + beta`, per row.
    ///
    /// # Safety
    ///
    /// Launch with [`THREADS`] threads and [`SHARED_BYTES`] dynamic shared
    /// memory, 128-byte aligned. `source` and `destination` must describe live
    /// `[ROWS * gridDim.x, COLUMNS]` bf16 buffers through a map paired with
    /// [`Tile`]; `gamma_map` and `beta_map` live `[COLUMNS]` buffers through
    /// one paired with [`Parameters`]. A CTA takes its row range from
    /// `blockIdx.x` and never bounds-checks it.
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
                // One instruction each, against a rank-1 map: an unswizzled
                // box has no atom to be cut into, so a vector never walks
                // stacked subtiles the way the tile above does.
                gamma.tma_load(gamma_map, 0, loaded);
                beta.tma_load(beta_map, 0, loaded);
                loaded.expect_tx((Tile::BYTES + 2 * Parameters::BYTES) as u32);
            }
            loaded.wait(0);
            thread::sync_threads();

            // Every lane reads the columns its fragment owns. The 8 lanes of a
            // column group ask for the same address and shared memory answers
            // them together, which is the access pattern a swizzle would have
            // been spreading for nothing.
            let g: Columns = load_vec(gamma, lane);
            let b: Columns = load_vec(beta, lane);

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
    ///
    /// # Safety
    ///
    /// As [`layernorm_rows`], minus the parameter vectors. The block is 1-D and
    /// exactly [`THREADS`] threads, which is what makes each warp's slot in
    /// `partials` its own.
    #[kernel]
    pub unsafe fn groupnorm_tile(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        epsilon: f32,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::from_raw(smem);
            // The partials first, because `Tile::BYTES` is the only offset in
            // this plan a vector's 128-byte alignment is promised at; the
            // barrier behind it needs eight.
            let partials = Partials::from_raw(smem.add(Tile::BYTES));
            let loaded = Semaphore::attach(smem.add(Tile::BYTES + Partials::BYTES) as *mut Barrier);

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

            // `tile_sum` (#6) folds this warp's band to one f32, warp-uniform,
            // and `block_reduce_sum` (#3) folds the four warps' to one that is
            // block-uniform — the shuffle butterfly stops at 32 lanes, so the
            // second step is a staged write and two barriers rather than more
            // shuffles.
            let mean = block_reduce_sum(partials, x.tile_sum()) * scale;
            // `shift`/`scale` against a block-uniform `f32` (#38), and the free
            // `rsqrt` beside them: this statistic never becomes a vector, so
            // there was nothing for `RegVec::rsqrt` to ride on.
            let x = x.shift(-mean);
            // The same scratch again, immediately, with no barrier here: the
            // reduction syncs after its own read, so this warp cannot overwrite
            // a slot another warp has not folded yet.
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
