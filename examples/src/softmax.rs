//! Row-wise softmax of one `[128, 128]` bf16 tile per CTA, in place.
//!
//! A CTA of four warps takes the tile in by TMA, normalizes it in registers,
//! and sends the same shared tile back out by TMA. A warp owns 32 of the 128
//! rows; `blockIdx.x` selects the row band and `blockIdx.y` the plane.
//!
//! Each warp walks its rows a [`CHUNK`]-wide band of columns at a time, in
//! three passes over the shared tile — the row peak, then the denominator,
//! then the scaled output written back over the input — carrying both per-row
//! statistics across chunks.
//!
//! [`check`] launches it against a CPU reference and [`bench`] times the same
//! launch afterwards, in that order: a number only ever comes out of a checked
//! run. The seed makes every row a permutation of the multiples of 1/8 below
//! 16, so a row's 128 outputs are distinct and the closest pair of them is
//! 9.05% apart against a 0.39% tolerance — a misplaced column, row, band or
//! plane is orders of magnitude louder than rounding.
//!
//!     modal run modal_app.py::examples
//!
//! The rung ladder, the `exp2` ablation, the tolerance argument and the seed's
//! exhaustive error bounds are in `docs/kernels/softmax.md`.

use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread};

use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::ldst::{load_tile, store_tile};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, RegTile, RegVec};
use kittens::shared::{
    Bf16, SharedTile, Swizzle128B, publish_to_async_proxy, tma_store_commit, tma_store_wait,
};
use kittens::sync::Semaphore;
use kittens::{lane, warp_id};

const ROWS: usize = 128;
const COLUMNS: usize = 128;
/// Columns a warp holds in registers at once. Measured, not derived: 32 is the
/// next rung up and 4.4x slower, at one register more and a 256-byte frame.
const CHUNK: usize = 16;
const CHUNKS: usize = COLUMNS / CHUNK;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
type Band = RegTile<32, CHUNK, BaseLdtm>;
type Rows = RegVec<32, BaseLdtm>;

struct Shared {
    tile: Tile,
    loaded: Semaphore,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (tile, at) = at.tile::<Bf16, ROWS, COLUMNS, Swizzle128B>();
    let (loaded, at) = at.semaphore();
    Shared {
        tile,
        loaded,
        plan: at,
    }
}

pub const SHARED_BYTES: usize = shared(SharedPlan::sizing()).plan.bytes();
pub const THREADS: u32 = (ROWS / 32) as u32 * 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn softmax_rows(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        mut plane: i32,
    ) {
        unsafe {
            let Shared { tile, loaded, .. } = shared(SharedPlan::attach());

            let lane = lane();
            let row_base = 32 * warp_id();
            plane += thread::blockIdx_y() as i32;

            if thread::threadIdx_x() == 0 {
                loaded.init(1);
                publish_to_async_proxy();
            }
            thread::sync_threads();
            if thread::threadIdx_x() == 0 {
                loaded.expect_tx(tile.tma_load(
                    source,
                    (ROWS as u32 * thread::blockIdx_x()) as i32,
                    plane,
                    loaded,
                ));
            }
            loaded.wait(0);
            thread::sync_threads();

            let chunks = tile.chunk_writer();

            let mut peak = Rows::splat(f32::NEG_INFINITY);
            let mut chunk = 0usize;
            while chunk < CHUNKS {
                let x: Band = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                peak.max_assign(x.row_max());
                chunk += 1;
            }

            let mut total = Rows::splat(0.0);
            chunk = 0;
            while chunk < CHUNKS {
                let x: Band = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                total.add_assign(x.sub_row(peak).exp2_hw().row_sum());
                chunk += 1;
            }
            let scale = total.recip();

            chunk = 0;
            while chunk < CHUNKS {
                let column = (CHUNK * chunk) as u32;
                let x: Band = load_tile(chunks, row_base, column, lane);
                let x = x.sub_row(peak).exp2_hw().mul_row(scale);
                store_tile(chunks, row_base, column, lane, x);
                chunk += 1;
            }
            // `stmatrix` writes through the generic proxy; the TMA engine reads
            // through the async one.
            publish_to_async_proxy();
            thread::sync_threads();

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

pub const CHECK_ROWS: usize = 2 * ROWS;
pub const CHECK_PLANES: usize = 2;

/// Column strides the row axis cycles through, one per CTA band. Odd, and 63
/// because an odd cycle is the only thing that reaches past one band.
const STRIDES: usize = 63;

fn permutation(plane: usize, row: usize, column: usize) -> usize {
    let stride = 11 + 2 * (row / ROWS % STRIDES);
    (53 * plane + 37 * row + stride * column) % 128
}

fn input(plane: usize, row: usize, column: usize) -> f32 {
    permutation(plane, row, column) as f32 / 8.0
}

fn staged(rows: usize, planes: usize) -> Vec<u32> {
    let mut staged = Vec::with_capacity(planes * rows * COLUMNS / 2);
    for plane in 0..planes {
        for row in 0..rows {
            for pair in 0..COLUMNS / 2 {
                let (low, high) = (input(plane, row, 2 * pair), input(plane, row, 2 * pair + 1));
                staged.push(to_bf16(low) as u32 | ((to_bf16(high) as u32) << 16));
            }
        }
    }
    staged
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn grid(rows: usize, planes: usize) -> u32 {
    (rows / ROWS * planes) as u32
}

fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch over `planes * rows` rows, compare every output against a CPU
/// reference, and only then hand the launch to `then`.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    rows: usize,
    planes: usize,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig};
    use kittens::global::encode_bf16_panels;
    use kittens::shared::Element;

    /// Twice the 1.97e-3 a correct kernel measures, which is itself the bf16
    /// output rounding floor.
    const TOLERANCE: f32 = 1.0 / 256.0;

    if !rows.is_multiple_of(ROWS) || planes == 0 {
        return Err(
            format!("{rows} rows x {planes} planes does not divide {ROWS} rows a CTA").into(),
        );
    }

    let stream = context.default_stream();
    let module = kernels::load(context)?;

    let staged = staged(rows, planes);
    let source = DeviceBuffer::from_host(&stream, &staged)?;
    let destination = DeviceBuffer::<u32>::zeroed(&stream, staged.len())?;
    // SAFETY: both buffers outlive every launch consuming their maps below.
    let (source_map, destination_map) = unsafe {
        (
            encode_bf16_panels::<ROWS, COLUMNS>(&stream, source.cu_deviceptr(), rows, planes)?,
            encode_bf16_panels::<ROWS, COLUMNS>(&stream, destination.cu_deviceptr(), rows, planes)?,
        )
    };

    let config = LaunchConfig {
        grid_dim: ((rows / ROWS) as u32, planes as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: SHARED_BYTES as u32,
    };
    let mut launch_once = || -> Result<(), Box<dyn Error>> {
        // SAFETY: the grid covers exactly the rows and planes both maps
        // describe, and the block and shared plan are the kernel's own.
        unsafe {
            module.softmax_rows(
                &stream,
                config,
                source_map.as_ptr(),
                destination_map.as_ptr(),
                0,
            )?
        };
        Ok(())
    };
    launch_once()?;

    let weight: Vec<f64> = (0..COLUMNS)
        .map(|value| (value as f64 / 8.0 - 127.0 / 8.0).exp2())
        .collect();
    let total: f64 = weight.iter().sum();
    let observed = destination.to_host_vec(&stream)?;
    let (mut wrong, mut sample, mut worst) = (0usize, Vec::new(), 0.0f32);
    for plane in 0..planes {
        for row in 0..rows {
            for column in 0..COLUMNS {
                let index = ((plane * rows + row) * COLUMNS + column) / 2;
                let word = Bf16::unpack(observed[index]);
                let value = word[column % 2];
                let expected = (weight[permutation(plane, row, column)] / total) as f32;
                let error = (value - expected).abs() / expected;
                worst = worst.max(error);
                // Negated rather than `error > TOLERANCE`: a NaN compares false
                // either way, so only this spelling counts one as wrong.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(error <= TOLERANCE) {
                    wrong += 1;
                    if sample.len() < 8 {
                        sample.push(format!(
                            "[{plane}, {row}, {column}] = {value}, want {expected} ({error:.2e})"
                        ));
                    }
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!("{wrong} outputs outside 2^-8: {}", sample.join("; ")).into());
    }

    let after = then(&stream, &mut launch_once)?;
    Ok((
        format!("{planes}x{rows}x{COLUMNS} rows normalized, worst relative error {worst:.2e}"),
        after,
    ))
}

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, CHECK_ROWS, CHECK_PLANES, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point. A softmax [`Shape`] is `rows x COLUMNS x
/// planes`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, shape.k, time).map(|(_, timings)| timings)
}
