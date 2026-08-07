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
//! The device body takes the width of its shared→register read as a `const`
//! parameter and this crate emits only the shipped value; the other one is
//! `experiments/src/softmax_x4.rs`, which is where #131 timed the two.
//!
//! The rung ladder, the `exp2` ablation, the tolerance argument and the seed's
//! exhaustive error bounds are in `docs/kernels/softmax.md`.

use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread};

use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::epilogue::{Cta, Scope};
use kittens::ldst::{load_tile, load_tile_x4, store_tile};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, RegTile, RegVec};
use kittens::shared::{Bf16, SharedTile, Swizzle128B, SwizzledChunks, publish_to_async_proxy};
use kittens::sync::Semaphore;
use kittens::watchdog::ReadBack;
use kittens::{lane, warp_id};

const ROWS: usize = 128;
pub(crate) const COLUMNS: usize = 128;
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

/// A warp's `[32, CHUNK]` band of the shared tile, by whichever `ldmatrix`
/// width the entry was compiled for.
///
/// The whole of what `X4` moves. Both spellings read the same block through the
/// same [`kittens::ldst::fragment_address`] derivation; `.x4` takes its 32
/// addresses from all lanes and issues one instruction per block where `.x2`
/// takes 16 from the first half and issues two.
#[inline(always)]
unsafe fn band<const X4: bool>(
    chunks: SwizzledChunks<Bf16>,
    row: u32,
    column: u32,
    lane: u32,
) -> Band {
    unsafe {
        if X4 {
            load_tile_x4(chunks, row, column, lane)
        } else {
            load_tile(chunks, row, column, lane)
        }
    }
}

/// One CTA's three passes, with the width of the shared→register read as its
/// only parameter.
///
/// **`X4` is not a dial this kernel ships.** `false` is the entry below and the
/// only one `examples/` emits; `experiments/src/softmax_x4.rs` emits `true` as
/// a second entry so #131's question — whether one `ldmatrix.m8n8.x4` per block
/// beats two `.x2`s on a kernel whose every pass is a `load_tile` — could be
/// asked with both arms in one bundle. The body is shared so that the arm
/// differs from the shipped entry in the load width and in nothing else.
#[inline(always)]
pub(crate) unsafe fn rows<const X4: bool>(
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
            let x = band::<X4>(chunks, row_base, (CHUNK * chunk) as u32, lane);
            peak.max_assign(x.row_max());
            chunk += 1;
        }

        let mut total = Rows::splat(0.0);
        chunk = 0;
        while chunk < CHUNKS {
            let x = band::<X4>(chunks, row_base, (CHUNK * chunk) as u32, lane);
            total.add_assign(x.sub_row(peak).exp2_hw().row_sum());
            chunk += 1;
        }
        let scale = total.recip();

        chunk = 0;
        while chunk < CHUNKS {
            let column = (CHUNK * chunk) as u32;
            let x = band::<X4>(chunks, row_base, column, lane);
            let x = x.sub_row(peak).exp2_hw().mul_row(scale);
            store_tile(chunks, row_base, column, lane, x);
            chunk += 1;
        }
        // The proxy fence, the issue, the commit and the wait for the bytes to
        // be visible: `store_once` is all four, and this kernel stores its tile
        // exactly once.
        Cta::store_once(
            tile,
            destination,
            (ROWS as u32 * thread::blockIdx_x()) as i32,
            plane,
        );
        if thread::threadIdx_x() == 0 {
            loaded.inval();
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn softmax_rows(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        plane: i32,
    ) {
        unsafe { rows::<false>(source, destination, plane) }
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

/// One launch of an entry with this kernel's signature, at the grid and shared
/// plan [`run`] fixes.
///
/// A parameter rather than [`kernels::load`]'s own entry, so that the arm
/// `experiments/` compiles beside the shipped one (#131) is checked by the
/// same CPU reference and timed on the same clock. Nothing else about a run
/// varies with it.
pub(crate) type Launch<'a> = &'a dyn Fn(
    &cuda_core::CudaStream,
    cuda_core::LaunchConfig,
    *const TmaDescriptor,
    *const TmaDescriptor,
) -> Result<(), Box<dyn Error>>;

/// Launch over `planes * rows` rows, compare every output against a CPU
/// reference, and only then hand the launch to `then`.
pub(crate) fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    rows: usize,
    planes: usize,
    launch: Launch,
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
    // The grid covers exactly the rows and planes both maps describe, and the
    // block and shared plan are the kernel's own.
    let mut launch_once = || {
        launch(
            &stream,
            config,
            source_map.as_ptr(),
            destination_map.as_ptr(),
        )
    };
    launch_once()?;

    let weight: Vec<f64> = (0..COLUMNS)
        .map(|value| (value as f64 / 8.0 - 127.0 / 8.0).exp2())
        .collect();
    let total: f64 = weight.iter().sum();
    let observed = destination.read_back(&stream)?;
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

/// [`run`] with the shipped entry supplied as its launch.
fn shipped<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    rows: usize,
    planes: usize,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    let module = kernels::load(context)?;
    let launch = |stream: &cuda_core::CudaStream,
                  config: cuda_core::LaunchConfig,
                  source: *const TmaDescriptor,
                  destination: *const TmaDescriptor|
     -> Result<(), Box<dyn Error>> {
        unsafe { module.softmax_rows(stream, config, source, destination, 0) }?;
        Ok(())
    };
    run(context, rows, planes, &launch, then)
}

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    shipped(context, CHECK_ROWS, CHECK_PLANES, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point. A softmax [`Shape`] is `rows x COLUMNS x
/// planes`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    shipped(context, shape.m, shape.k, time).map(|(_, timings)| timings)
}
