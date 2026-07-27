//! # Softmax over the rows of a tile
//!
//! **Status: runs.** [`check`] launches it against a CPU reference on a B200
//! (`modal run modal_app.py::examples`).
//!
//! This is the kernel #5 and #6 were filed for, and every line of it is now the
//! real API. It was also the shortest possible demonstration of the holes
//! underneath them: the library could neither get data *out* of a shared tile
//! (#21), nor address one wider than a single swizzle atom (#25), nor write a
//! result back to global memory at all (#9) — so a kernel whose input is not an
//! MMA operand could not start, and one whose output is not an MMA accumulator
//! could not finish. All three are closed, and this file is the diff going to
//! zero: five gaps, then one, then none.
//!
//! Not blocking, but visible: the movers are per-`[16, 16]`-block, so the band
//! loop is written twice below (#22). It compiles and it runs — it is a cost,
//! not a hole.
//!
//! Note what is *not* in that list, and never was: nothing about layouts,
//! swizzles, semaphores, the fragment map, or the arithmetic. The missing
//! surface was shallow and wide, not deep, and this is the example that said so
//! first.
//!
//! ## What the numbers can prove
//!
//! A softmax is not the GEMM: it has an `exp2` and a divide in it, so a `==`
//! against a CPU reference is not available and pretending otherwise would be
//! the wrong kind of rigour. What [`check`] does instead is make every *other*
//! kind of failure loud. Each row's 128 inputs are a permutation of the
//! multiples of 1/8 below 16, so each row's 128 outputs are distinct and the
//! closest pair of them is 9% apart, against a tolerance of 0.4%. A swapped
//! column, a swapped row, a wrong plane, a row normalized against another row's
//! sum — all of them are off by more than an order of magnitude more than
//! rounding can account for.

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

            // The whole algorithm, in the real API. `row_max`/`row_sum` (#6)
            // fold a thread's own 32 values and then the quad holding the rest
            // of the row; the broadcasts back down each row are shuffle-free,
            // because the fragment map already gives a thread every one of its
            // rows' values.
            let x = x.sub_row(x.row_max()).exp2();
            let x = x.div_row(x.row_sum());

            store_band(tile, row_base, lane, x);
            // `stmatrix` writes through the generic proxy; the TMA engine reads
            // through the async one.
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            // The store completes on a group, not on a barrier: nothing
            // republishes the tile or the result but this thread's own wait,
            // and the full form rather than `tma_store_wait_read` because what
            // the kernel owes its caller is bytes in global memory.
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

/// Rows the correctness run covers — two CTAs' worth, so `blockIdx.x`'s row
/// stride is on trial and not just assumed.
pub const CHECK_ROWS: usize = 2 * ROWS;
/// Planes it covers: two, so `blockIdx.y` is as well.
pub const CHECK_PLANES: usize = 2;

/// The input at `(plane, row, column)`.
///
/// Within a row the exponent walks every multiple of 1/8 below 16 in a
/// stride-11 permutation — 11 is odd, so it is a bijection mod 128 — which
/// makes all 128 values of a row distinct and therefore all 128 outputs
/// distinct. `plane` and `row` rotate the permutation, so a wrong plane or a
/// wrong row is a wholly different set of numbers rather than a plausible one.
/// Every value is a multiple of 1/8 below 16, so bf16 holds it exactly and
/// nothing is lost on the way in.
fn input(plane: usize, row: usize, column: usize) -> f32 {
    ((53 * plane + 37 * row + 11 * column) % 128) as f32 / 8.0
}

/// The whole input as the packed `u32` words a `[planes, rows, COLUMNS]` bf16
/// staging buffer holds — the layout [`kittens::global::encode_bf16_panels`]
/// describes and the kernel's row/plane coordinates index.
fn staged() -> Vec<u32> {
    let mut staged = Vec::with_capacity(CHECK_PLANES * CHECK_ROWS * COLUMNS / 2);
    for plane in 0..CHECK_PLANES {
        for row in 0..CHECK_ROWS {
            for pair in 0..COLUMNS / 2 {
                let (low, high) = (input(plane, row, 2 * pair), input(plane, row, 2 * pair + 1));
                staged.push(to_bf16(low) as u32 | ((to_bf16(high) as u32) << 16));
            }
        }
    }
    staged
}

/// Round-to-nearest-even fp32 → bf16. Exact for every value [`input`]
/// produces, whose low 16 mantissa bits are already zero.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Launch the kernel over `CHECK_PLANES * CHECK_ROWS` rows and compare every
/// output against a CPU reference.
///
/// The tolerance is relative and it is the honest one: the result is bf16, so
/// half an ulp is already 2⁻⁹, and the kernel's `exp2` is the FMA polynomial
/// [`kittens::reg::exp2_approx`] (7.5e-5 relative) folded by a shuffle
/// butterfly in an order no host loop reproduces. 2⁻⁸ covers all of that with
/// room to spare and is still forty times tighter than the gap between two
/// neighbouring outputs of a row, which is what a misplaced element would have
/// to hide inside.
pub fn check(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<String, Box<dyn std::error::Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig};
    use kittens::global::encode_bf16_panels;
    use kittens::shared::Element;

    /// Half a bf16 ulp is 2⁻⁹; this is the next power of two up.
    const TOLERANCE: f32 = 1.0 / 256.0;

    let stream = context.default_stream();
    let module = kernels::load(context)?;

    let staged = staged();
    let source = DeviceBuffer::from_host(&stream, &staged)?;
    // Zeroed rather than left alone: a store that never lands reads back as
    // bf16 zero, which no softmax output can be.
    let destination = DeviceBuffer::<u32>::zeroed(&stream, staged.len())?;
    // SAFETY: both buffers outlive every launch consuming their maps below.
    let (source_map, destination_map) = unsafe {
        (
            encode_bf16_panels::<ROWS, COLUMNS>(
                &stream,
                source.cu_deviceptr(),
                CHECK_ROWS,
                CHECK_PLANES,
            )?,
            encode_bf16_panels::<ROWS, COLUMNS>(
                &stream,
                destination.cu_deviceptr(),
                CHECK_ROWS,
                CHECK_PLANES,
            )?,
        )
    };

    let config = LaunchConfig {
        grid_dim: ((CHECK_ROWS / ROWS) as u32, CHECK_PLANES as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: SHARED_BYTES as u32,
    };
    // SAFETY: the grid covers exactly the rows and planes both maps describe,
    // and the block and shared plan are the kernel's own constants.
    unsafe {
        module.softmax_rows(
            &stream,
            config,
            source_map.as_ptr(),
            destination_map.as_ptr(),
            0,
        )?
    };

    let observed = destination.to_host_vec(&stream)?;
    let mut wrong = Vec::new();
    let mut worst = 0.0f32;
    for plane in 0..CHECK_PLANES {
        for row in 0..CHECK_ROWS {
            let exponents: Vec<f64> = (0..COLUMNS).map(|c| input(plane, row, c) as f64).collect();
            let peak = exponents.iter().copied().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = exponents.iter().map(|x| (x - peak).exp2()).collect();
            let total: f64 = weights.iter().sum();
            for column in 0..COLUMNS {
                let index = ((plane * CHECK_ROWS + row) * COLUMNS + column) / 2;
                let word = Bf16::unpack(observed[index]);
                let value = word[column % 2];
                let expected = (weights[column] / total) as f32;
                let error = (value - expected).abs() / expected;
                worst = worst.max(error);
                if !(error <= TOLERANCE) && wrong.len() < 8 {
                    wrong.push(format!(
                        "[{plane}, {row}, {column}] = {value}, want {expected} ({error:.2e})"
                    ));
                }
            }
        }
    }
    if !wrong.is_empty() {
        return Err(format!("{} outputs outside 2^-8: {}", wrong.len(), wrong.join("; ")).into());
    }
    Ok(format!(
        "{CHECK_PLANES}x{CHECK_ROWS}x{COLUMNS} rows normalized, worst relative error {worst:.2e}"
    ))
}
