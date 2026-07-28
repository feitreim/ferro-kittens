//! # Layernorm over a tile, and the whole-tile reduction next to it
//!
//! **Status: [`kernels::layernorm_rows`] compiles** and is in the default
//! build; [`kernels::groupnorm_tile`] beside it is still aspirational and sits
//! behind `--features layernorm`. The two kernels differ in exactly one thing —
//! the axis their statistic is taken over — and that is what separated them.
//!
//! `layernorm_rows` was blocked on **#13**: `gamma`/`beta` are parameters, not
//! activations. They are loaded once, shared by every row, and belong in shared
//! memory, so with no `SharedVec` every thread would have re-read them from
//! global. [`kittens::shared::SharedVec`] is now that type — one flat run of
//! elements with its own single-box TMA path, unswizzled because a swizzle
//! atom is a statement about a tile's *rows* and a vector has one — and
//! [`kittens::ldst::load_vec`] broadcasts it into the `ColVec` `mul_col`
//! consumes. Nothing in this kernel is a workaround any more.
//!
//! [`kernels::groupnorm_tile`] normalizes over the *whole* tile rather than per
//! row, and that is the one place a landed #6 does not reach.
//! `RegTile::tile_sum` is warp scope: it folds a warp's own band across all 32
//! lanes and stops. A `[128, COLUMNS]` tile is four warps' bands, so the warps
//! have to agree, and that needs a shared staging buffer and a block barrier —
//! storage, not a shuffle. **#3**'s `Scope` (`WARPS`, `THREADS`, `rank`,
//! `sync`) supplies none of it.
//!
//! What #13 changed for that kernel is the *shape of the answer*, not its
//! status. The open question on #3 was whether a block reduction takes a
//! scratch pointer or a shared vector; a `SharedVec<F32, WARPS>` is now a type
//! a signature can name, and `set(warp, partial)` / `get(w)` is exactly the
//! access a fold over four partials needs. Two things are still missing and
//! both belong to #3: the reduction itself (`sync::block_reduce_sum`), and an
//! `impl Element for F32` (**#2**) without which the partials cannot be fp32.
//! The `partials` pointer below is this kernel guessing at both.
//!
//! Blocked on: **#3** (and, under it, #2) — for [`kernels::groupnorm_tile`]
//! only.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::ldst::{load_tile, load_vec, store_tile};
#[cfg(feature = "layernorm")]
use kittens::reg::rsqrt;
use kittens::reg::{BaseLdtm, ColVec, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait};
use kittens::sync::Semaphore;
#[cfg(feature = "layernorm")]
use kittens::sync::block_reduce_sum;

const ROWS: usize = 128;
const COLUMNS: usize = 128;

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

pub const SHARED_BYTES: usize = Tile::BYTES + 2 * Parameters::BYTES + 64;
pub const THREADS: u32 = (ROWS / 32) as u32 * 32;

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
    /// As [`layernorm_rows`], minus the parameter vectors.
    #[cfg(feature = "layernorm")]
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
            // WANT (#3): and then the four warps have to agree. #13 supplied
            // the storage half — `SharedVec<F32, 4>` is a type this signature
            // could take, and `set(warp, partial)` writes one element without
            // touching its neighbours, which is why the vector's scalar access
            // is per element and not per packed word. What is still missing is
            // the fold itself, and an `Element` impl for fp32 (#2) to hold it.
            // `partials` below is this kernel guessing at both.
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
