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
//! It used to write the `[16, 16]` block-composition loop twice, once per
//! direction, for want of a band-shaped mover. [`kittens::ldst::load_tile`] and
//! [`kittens::ldst::store_tile`] (#22) are that mover, and the two loops are
//! gone: the body below is four calls and four lines of arithmetic.
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
//! rounding can account for. So is a whole CTA band read from the wrong place,
//! which is the one row error this seed used to reproduce exactly rather than
//! catch (#56); [`permutation`] carries that argument and states what is left
//! over.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

// Host side: the launcher's error type, and the benchmark's size and clock.
use crate::bench::{Shape, Timings, time};
use std::error::Error;

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

            let chunks = tile.chunk_writer();
            let x: Band = load_tile(chunks, row_base, 0, lane);

            // The whole algorithm, in the real API. `row_max`/`row_sum` (#6)
            // fold a thread's own 32 values and then the quad holding the rest
            // of the row; the broadcasts back down each row are shuffle-free,
            // because the fragment map already gives a thread every one of its
            // rows' values.
            let x = x.sub_row(x.row_max()).exp2();
            let x = x.div_row(x.row_sum());

            store_tile(chunks, row_base, 0, lane, x);
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

/// Column strides the row axis cycles through, one per CTA band. Odd, and that
/// is the whole point — see [`permutation`].
const STRIDES: usize = 63;

/// Which multiple of 1/8 sits at `(plane, row, column)`.
///
/// Within a row the exponent walks every multiple of 1/8 below 16 in a
/// stride-`s` permutation, `s` odd and so a bijection mod 128, which makes all
/// 128 values of a row distinct and therefore all 128 outputs distinct.
/// `plane` and `row` rotate that permutation and the band chooses `s`, so a
/// wrong plane, a wrong row or a wrong band is a wholly different set of
/// numbers rather than a plausible one.
///
/// ## Why the band chooses the stride (#56)
///
/// This used to be `(53·plane + 37·row + 11·column) % 128`, and the row axis
/// of it was blind to exactly one thing: a whole CTA band. 37 is odd, so
/// `37·row` walks all 128 residues — and then repeats, because a step of `ROWS`
/// rows changes it by `37·128 ≡ 0 (mod 128)`. Any row error that was a multiple
/// of `ROWS` produced bit-identical expected values.
///
/// Neither obvious repair works, and the reason is structural rather than
/// arithmetic. A seed of the form `(a·row + b·column + c) % 128` is an affine
/// map of the column axis, and those form a group of order 2¹³ under
/// composition — a 2-group, whose exponent is exactly 128 (checked by
/// enumeration). So *every* row-affine seed has a period in `row` that divides
/// 128, which is one band, and no choice of multiplier can reach past it:
///
/// - `37 → 39`, or any other odd multiplier, already has the maximum period an
///   affine row term can have. It changes nothing at all.
/// - Letting the stride follow the row, `(11 + 2·row)`, is worse than nothing:
///   the stride's period is 64 and the offset's is 128, they turn together, and
///   the whole seed still repeats every 128 rows. It also costs the row axis
///   its rotation-only errors, dropping a one-row error from *every* cell wrong
///   by ≥ 1.0 to every cell wrong by ≥ 0.083.
///
/// What the 2-group cannot supply is an **odd** period, so the band supplies
/// one: `row / ROWS % 63` picks the stride, and two rows agree only if they
/// agree both mod 128 (the offset) and mod 63 bands (the stride). 63 is the
/// largest odd cycle available — the strides must be distinct mod 128 and there
/// are only 64 odd residues, so a longer cycle would repeat a stride and hand
/// the blindness straight back.
///
/// ## What that buys, in numbers
///
/// A kernel whose load row and store row diverge by `δ` rows is caught this
/// loudly, against a tolerance of 2⁻⁸ = 0.0039. The bounds are exhaustive over
/// every offset shift and every one of the 63² band pairs, not sampled:
///
/// | `δ` | columns of a row wrong | each wrong by at least |
/// | --- | --- | --- |
/// | `128 ∤ δ` | ≥ 64 of 128 | 0.083 — 21× the tolerance |
/// | `δ = 128k`, `63 ∤ k` | ≥ 64 of 128 | 0.159 — 41× the tolerance |
/// | `δ = 128k`, `63 \| k` | 0 | invisible |
///
/// The four errors worth naming all sit far above those floors, over the
/// `CHECK_PLANES · CHECK_ROWS · COLUMNS` = 65,536 cells of the check: a
/// one-row shift makes every cell wrong, a 32-row one (a warp band) 65,280, a
/// one-band one 64,512, and every CTA loading band 0 — the positive control
/// this issue asked for — 32,256, which is 126 of the 128 columns of every row
/// that read the wrong band.
///
/// The plane axis is untouched and its strength does not depend on any of
/// this: a plane error shifts the *value* at every cell by `53·Δplane`
/// (mod 128) whatever the stride is, which is a factor of `2^(53/8) ≈ 99` or
/// `2^(-75/8)` when it wraps — a relative error of at least 0.9985, 255× the
/// tolerance. All the stride does is make the column rotation that carries it
/// differ from band to band.
///
/// ## The residual, stated rather than hoped for
///
/// A constant row displacement is invisible exactly when it is a multiple of
/// `63 · ROWS` = 8064 rows. That is a rule and not an accident: 63 divides no
/// power of two, every grid dimension and band offset in this file is a power
/// of two, and every size the project runs is a power of two — so no
/// displacement band arithmetic can produce here is a multiple of 8064.
///
/// Two smaller ones, for the same reason. A band displacement of `k ≡ ±32
/// (mod 63)` leaves half of each row's columns matching, the most any non-blind
/// `k` leaves; the other half are wrong by ≥ 0.996. And column 0 of every row
/// is `53·plane + 37·row` whatever the stride, so that one column on its own
/// cannot see a band error — the other 127 can.
fn permutation(plane: usize, row: usize, column: usize) -> usize {
    let stride = 11 + 2 * (row / ROWS % STRIDES);
    (53 * plane + 37 * row + stride * column) % 128
}

/// The input at `(plane, row, column)`. Every value is a multiple of 1/8 below
/// 16, so bf16 holds it exactly and nothing is lost on the way in.
fn input(plane: usize, row: usize, column: usize) -> f32 {
    permutation(plane, row, column) as f32 / 8.0
}

/// The whole input as the packed `u32` words a `[planes, rows, COLUMNS]` bf16
/// staging buffer holds — the layout [`kittens::global::encode_bf16_panels`]
/// describes and the kernel's row/plane coordinates index.
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

/// Round-to-nearest-even fp32 → bf16. Exact for every value [`input`]
/// produces, whose low 16 mantissa bits are already zero.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Blocks a `[planes, rows, COLUMNS]` problem takes: one CTA per `ROWS` of a
/// plane. The benchmark prints it, because at the small end of a size sweep
/// this number and not the bytes is what the run is bound by.
pub fn grid(rows: usize, planes: usize) -> u32 {
    (rows / ROWS * planes) as u32
}

/// The `then` a run that is only being checked passes.
fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch the kernel over `planes * rows` rows, compare every output against a
/// CPU reference, and only then hand the launch to `then`.
///
/// That order is the whole design of [`crate::bench`]: the timing hook cannot
/// be reached from a launch whose output was wrong.
///
/// The tolerance is relative and it is the honest one: the result is bf16, so
/// half an ulp is already 2⁻⁹, and the kernel's `exp2` is the FMA polynomial
/// [`kittens::reg::exp2_approx`] (7.5e-5 relative) folded by a shuffle
/// butterfly in an order no host loop reproduces. 2⁻⁸ covers all of that with
/// room to spare and is still forty times tighter than the gap between two
/// neighbouring outputs of a row, which is what a misplaced element would have
/// to hide inside.
///
/// Measured rather than argued: the worst relative error over the checked run
/// is 1.97e-3, a hair past the 2⁻⁹ = 1.95e-3 that rounding the output to bf16
/// costs on its own. So the tolerance is 2.0× what a correct kernel actually
/// needs, and 46× below the 2^(1/8) − 1 = 9.05% gap between neighbouring
/// outputs — the band the check has to separate a right answer from a wrong
/// one in.
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

    /// Half a bf16 ulp is 2⁻⁹; this is the next power of two up.
    const TOLERANCE: f32 = 1.0 / 256.0;

    // One CTA owns a whole `ROWS` band and the kernel bounds-checks nothing.
    if rows % ROWS != 0 || planes == 0 {
        return Err(
            format!("{rows} rows x {planes} planes does not divide {ROWS} rows a CTA").into(),
        );
    }

    let stream = context.default_stream();
    let module = kernels::load(context)?;

    let staged = staged(rows, planes);
    let source = DeviceBuffer::from_host(&stream, &staged)?;
    // Zeroed rather than left alone: a store that never lands reads back as
    // bf16 zero, which no softmax output can be.
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

    // Every row is the same 128 values in a different order, so the reference
    // is one 128-entry table at any size: the peak is always 127/8 and the
    // denominator is always the same sum. Every output is still compared
    // against its own expected value, which is now a permuted lookup.
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

/// The correctness run: one size, checked, nothing timed.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, CHECK_ROWS, CHECK_PLANES, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point: the same check at `shape`, and then the same
/// launch timed. A softmax [`Shape`] is `rows x COLUMNS x planes`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, shape.k, time).map(|(_, timings)| timings)
}
