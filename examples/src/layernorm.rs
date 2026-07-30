//! Layernorm over the rows of a tile, and the whole-tile reduction next to it.
//!
//! A CTA of four warps takes a `[128, 128]` bf16 tile in by TMA, normalizes it
//! in registers, and sends the same shared tile back out by TMA. A warp owns
//! 32 of the 128 rows and `blockIdx.x` selects the row band. The two kernels
//! differ in exactly one thing, the axis their statistic is taken over:
//!
//! - [`kernels::layernorm_rows`] runs `y = gamma ⊙ (x - mean) / √(var + ε) +
//!   beta` per row, walking the row a [`CHUNK`]-wide band of columns at a time
//!   in three passes — the mean, the centred variance, then the output — and
//!   carrying both statistics across chunks. `gamma` and `beta` are staged
//!   once per CTA in an unswizzled [`kittens::shared::SharedVec`], since they
//!   are parameters every row shares rather than activations.
//! - [`kernels::groupnorm_tile`] takes the same statistic over the whole tile,
//!   so the four warps have to agree. `tile_sum` stops at a warp, and warps
//!   cannot shuffle to each other, so [`kittens::sync::block_reduce_sum`]
//!   finishes the fold through a shared vector and two barriers. It compiles
//!   and is in the default build; it has no launcher and no reference.
//!
//! [`check`] launches `layernorm_rows` against a CPU reference and [`bench`]
//! times the same launch afterwards, in that order. The seed cannot be a
//! permutation of one multiset the way [`crate::softmax`]'s is: identical row
//! multisets give identical row statistics, and then `groupnorm_tile`'s answer
//! and `layernorm_rows`' coincide.
//!
//!     modal run modal_app.py::examples
//!
//! The rung ladder, the seed and parameter construction, the tolerance
//! argument and the scored error table are in `docs/kernels/layernorm.md`.

use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, launch_bounds, thread};

use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::ldst::{load_tile, load_vec, store_tile};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, ColVec, RegTile, RegVec, rsqrt};
use kittens::shared::{
    Bf16, F32, SharedTile, SharedVec, Swizzle128B, publish_to_async_proxy, tma_store_commit,
    tma_store_wait,
};
use kittens::sync::{Semaphore, block_reduce_sum};
use kittens::{lane, warp_id};

const ROWS: usize = 128;
const COLUMNS: usize = 128;
const WARPS: usize = ROWS / 32;
/// Columns [`kernels::layernorm_rows`] holds in registers at once. Measured,
/// not derived: 32 is the next rung up and 1.46x slower, at seven registers
/// fewer and a 128-byte frame the shipped form does not carry.
const CHUNK: usize = 16;
const CHUNKS: usize = COLUMNS / CHUNK;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
type Band = RegTile<32, COLUMNS, BaseLdtm>;
type Chunk = RegTile<32, CHUNK, BaseLdtm>;
type Rows = RegVec<32, BaseLdtm>;
type Columns = ColVec<CHUNK, BaseLdtm>;
type Parameters = SharedVec<Bf16, COLUMNS>;
type ParameterChunk = SharedVec<Bf16, CHUNK>;
/// fp32 and not bf16: a partial rounded on its way through shared memory would
/// cost the sum eight bits, and the variance pass would inherit the error.
type Partials = SharedVec<F32, WARPS>;

struct RowsShared {
    tile: Tile,
    gamma: Parameters,
    beta: Parameters,
    loaded: Semaphore,
    plan: SharedPlan,
}

struct GroupShared {
    tile: Tile,
    partials: Partials,
    loaded: Semaphore,
    plan: SharedPlan,
}

#[inline(always)]
const fn rows_shared(at: SharedPlan) -> RowsShared {
    let (tile, at) = at.tile::<Bf16, ROWS, COLUMNS, Swizzle128B>();
    let (gamma, at) = at.vec::<Bf16, COLUMNS>();
    let (beta, at) = at.vec::<Bf16, COLUMNS>();
    let (loaded, at) = at.semaphore();
    RowsShared {
        tile,
        gamma,
        beta,
        loaded,
        plan: at,
    }
}

#[inline(always)]
const fn group_shared(at: SharedPlan) -> GroupShared {
    let (tile, at) = at.tile::<Bf16, ROWS, COLUMNS, Swizzle128B>();
    let (partials, at) = at.vec::<F32, WARPS>();
    let (loaded, at) = at.semaphore();
    GroupShared {
        tile,
        partials,
        loaded,
        plan: at,
    }
}

/// Dynamic shared memory a launch of *either* kernel declares — the larger of
/// the two plans.
pub const SHARED_BYTES: usize = {
    let rows = rows_shared(SharedPlan::sizing()).plan.bytes();
    let group = group_shared(SharedPlan::sizing()).plan.bytes();
    if rows > group { rows } else { group }
};
pub const THREADS: u32 = (WARPS * 32) as u32;
/// `#[launch_bounds]` below takes a literal and the occupancy gate reads it
/// back out with a digit regex; this is that literal and the derived form
/// saying the same number out loud.
const _: () = assert!(THREADS == 128);

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
            let RowsShared {
                tile,
                gamma,
                beta,
                loaded,
                ..
            } = rows_shared(SharedPlan::attach());

            let lane = lane();
            let row_base = 32 * warp_id();
            let row = (ROWS as u32 * thread::blockIdx_x()) as i32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                publish_to_async_proxy();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                // Each load hands back its own transaction charge; the barrier
                // is told their sum.
                let staged = tile.tma_load(source, row, 0, loaded)
                    + gamma.tma_load(gamma_map, 0, loaded)
                    + beta.tma_load(beta_map, 0, loaded);
                loaded.expect_tx(staged);
            }
            loaded.wait(0);
            thread::sync_threads();

            let chunks = tile.chunk_writer();

            let mut total = Rows::splat(0.0);
            let mut chunk = 0usize;
            while chunk < CHUNKS {
                let x: Chunk = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                total.add_assign(x.row_sum());
                chunk += 1;
            }
            let mean = total.scale(1.0 / COLUMNS as f32);

            // Centred and then squared rather than `E[x²] - E[x]²`: this seed's
            // rows sit up to 27 away from zero, so the one-pass form cancels
            // most of a large number.
            let mut square = Rows::splat(0.0);
            chunk = 0;
            while chunk < CHUNKS {
                let x: Chunk = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                let x = x.sub_row(mean);
                square.add_assign(x.mul(x).row_sum());
                chunk += 1;
            }
            let deviation = square.scale(1.0 / COLUMNS as f32).shift(epsilon).rsqrt();

            chunk = 0;
            while chunk < CHUNKS {
                let column = (CHUNK * chunk) as u32;
                let x: Chunk = load_tile(chunks, row_base, column, lane);
                let g: Columns =
                    load_vec(ParameterChunk::from_raw(gamma.at(column as usize)), lane);
                let b: Columns = load_vec(ParameterChunk::from_raw(beta.at(column as usize)), lane);
                let x = x.sub_row(mean).mul_row(deviation).mul_col(g).add_col(b);
                store_tile(chunks, row_base, column, lane, x);
                chunk += 1;
            }
            // `store_tile` writes through the generic proxy; the TMA engine
            // reads through the async one.
            publish_to_async_proxy();
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                tile.tma_store(destination, row, 0);
                tma_store_commit();
                tma_store_wait::<0>();
                loaded.inval();
            }
        }
    }

    /// The same normalization over the whole tile instead of per row.
    ///
    /// `#[launch_bounds(128, 3)]` caps `ptxas` at the 168 registers three CTAs
    /// an SM allows. Registers are this kernel's binding term, and it sat on
    /// 168 by luck until a smaller crate compiled the same source at 236.
    ///
    /// # Safety
    ///
    /// As [`layernorm_rows`], minus the parameter vectors. The block is 1-D and
    /// exactly [`THREADS`] threads, which is what makes each warp's slot in
    /// `partials` its own.
    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn groupnorm_tile(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        epsilon: f32,
    ) {
        unsafe {
            let GroupShared {
                tile,
                partials,
                loaded,
                ..
            } = group_shared(SharedPlan::attach());

            let lane = lane();
            let row_base = 32 * warp_id();
            let row = (ROWS as u32 * thread::blockIdx_x()) as i32;
            let scale = 1.0 / (ROWS * COLUMNS) as f32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                publish_to_async_proxy();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                loaded.expect_tx(tile.tma_load(source, row, 0, loaded));
            }
            loaded.wait(0);
            thread::sync_threads();

            let x: Band = load_tile(tile.chunk_writer(), row_base, 0, lane);

            let mean = block_reduce_sum(partials, x.tile_sum()) * scale;
            let x = x.shift(-mean);
            // The same scratch again with no barrier between: the reduction
            // syncs on both sides.
            let variance = block_reduce_sum(partials, x.mul(x).tile_sum()) * scale;
            let x = x.scale(rsqrt(variance + epsilon));

            store_tile(tile.chunk_writer(), row_base, 0, lane, x);
            // `store_tile` writes through the generic proxy; the TMA engine
            // reads through the async one.
            publish_to_async_proxy();
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

pub const CHECK_ROWS: usize = 2 * ROWS;

/// A residual rather than coverage: no row this seed produces has a variance
/// under 835, so a kernel that dropped `epsilon` entirely would still pass.
pub const EPSILON: f32 = 1e-5;

/// Column strides the row axis cycles through, one per CTA band. Odd, and 63
/// because an odd cycle is the only thing that reaches past one band.
const STRIDES: usize = 63;

/// Boost classes a row cycles through. See [`value`].
const BOOSTS: usize = 3;

fn permutation(row: usize, column: usize) -> usize {
    let stride = 11 + 2 * (row / ROWS % STRIDES);
    (37 * row + stride * column) % COLUMNS
}

/// Ladder entry `p`, as a row of class `row % BOOSTS` spells it. Layernorm is
/// invariant to `x -> a·x + c`, so the boost on half the ladder is what makes
/// a row's distribution differ in shape and not merely in order.
fn value(p: usize, row: usize) -> f32 {
    let magnitude = 16.0 + (p % 32) as f32 / 2.0;
    let boost = (1 << (1 + row % BOOSTS)) as f32;
    magnitude * [1.0, -0.5, boost, -boost / 2.0][p / 32]
}

fn input(row: usize, column: usize) -> f32 {
    value(permutation(row, column), row)
}

/// Injective over `0..COLUMNS`, because a parameter read from the wrong column
/// is one of the errors this check exists to see.
fn gamma(column: usize) -> f32 {
    1.0 + column as f32 / 128.0
}

fn beta(column: usize) -> f32 {
    2.0 + column as f32 / 64.0
}

/// The mean and `1/√(variance + ε)` of any row of boost class `class`. Three
/// of these cover every size: [`permutation`] is a bijection in `column`, so a
/// row holds each ladder entry exactly once.
fn statistics(class: usize) -> (f64, f64) {
    let values: Vec<f64> = (0..COLUMNS).map(|p| value(p, class) as f64).collect();
    let mean = values.iter().sum::<f64>() / COLUMNS as f64;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / COLUMNS as f64;
    (mean, 1.0 / (variance + EPSILON as f64).sqrt())
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn packed(count: usize, at: impl Fn(usize) -> f32) -> Vec<u32> {
    (0..count / 2)
        .map(|pair| to_bf16(at(2 * pair)) as u32 | ((to_bf16(at(2 * pair + 1)) as u32) << 16))
        .collect()
}

pub fn grid(rows: usize) -> u32 {
    (rows / ROWS) as u32
}

fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch over `rows`, compare every output against a CPU reference, and only
/// then hand the launch to `then`.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    rows: usize,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig};
    use kittens::global::{GlobalLayout, encode_bf16_panels};
    use kittens::shared::Element;

    /// One doubling above the 3.87e-3 a correct kernel measures, which is
    /// itself 0.99 of the bf16 rounding floor where this seed's outputs sit.
    const TOLERANCE: f32 = 1.0 / 128.0;

    if !rows.is_multiple_of(ROWS) {
        return Err(format!("{rows} rows does not divide the {ROWS} a CTA owns").into());
    }

    let stream = context.default_stream();
    let module = kernels::load(context)?;

    let staged = packed(rows * COLUMNS, |flat| input(flat / COLUMNS, flat % COLUMNS));
    let source = DeviceBuffer::from_host(&stream, &staged)?;
    let destination = DeviceBuffer::<u32>::zeroed(&stream, staged.len())?;
    let gammas = DeviceBuffer::from_host(&stream, &packed(COLUMNS, gamma))?;
    let betas = DeviceBuffer::from_host(&stream, &packed(COLUMNS, beta))?;
    // SAFETY: all four buffers outlive every launch consuming their maps below.
    let (source_map, destination_map, gamma_map, beta_map) = unsafe {
        (
            encode_bf16_panels::<ROWS, COLUMNS>(&stream, source.cu_deviceptr(), rows, 1)?,
            encode_bf16_panels::<ROWS, COLUMNS>(&stream, destination.cu_deviceptr(), rows, 1)?,
            GlobalLayout::<Bf16, 1>::packed(gammas.cu_deviceptr(), [COLUMNS])
                .tensor_map::<Parameters>(&stream)?,
            GlobalLayout::<Bf16, 1>::packed(betas.cu_deviceptr(), [COLUMNS])
                .tensor_map::<Parameters>(&stream)?,
        )
    };

    let config = LaunchConfig {
        grid_dim: (grid(rows), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: SHARED_BYTES as u32,
    };
    let mut launch_once = || -> Result<(), Box<dyn Error>> {
        // SAFETY: the grid covers exactly the rows the tile maps describe, the
        // parameter maps are the rank-1 pair `Parameters` is paired with, and
        // the block and shared plan are the kernel's own.
        unsafe {
            module.layernorm_rows(
                &stream,
                config,
                source_map.as_ptr(),
                gamma_map.as_ptr(),
                beta_map.as_ptr(),
                destination_map.as_ptr(),
                EPSILON,
            )?
        };
        Ok(())
    };
    launch_once()?;

    let classes: Vec<(f64, f64)> = (0..BOOSTS).map(statistics).collect();
    let observed = destination.to_host_vec(&stream)?;
    let (mut wrong, mut sample, mut worst) = (0usize, Vec::new(), 0.0f32);
    for row in 0..rows {
        let (mean, deviation) = classes[row % BOOSTS];
        for column in 0..COLUMNS {
            let index = (row * COLUMNS + column) / 2;
            let word = Bf16::unpack(observed[index]);
            let value = word[column % 2];
            let normalized = (input(row, column) as f64 - mean) * deviation;
            let expected = (gamma(column) as f64 * normalized + beta(column) as f64) as f32;
            let error = (value - expected).abs() / expected.abs();
            worst = worst.max(error);
            // Negated rather than `error > TOLERANCE`: a NaN compares false
            // either way, so only this spelling counts one as wrong.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(error <= TOLERANCE) {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!(
                        "[{row}, {column}] = {value}, want {expected} ({error:.2e})"
                    ));
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!("{wrong} outputs outside 2^-7: {}", sample.join("; ")).into());
    }

    let after = then(&stream, &mut launch_once)?;
    Ok((
        format!("{rows}x{COLUMNS} rows normalized, worst relative error {worst:.2e}"),
        after,
    ))
}

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, CHECK_ROWS, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point. A layernorm [`Shape`] is `rows x COLUMNS`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, time).map(|(_, timings)| timings)
}
