//! When a tile walk loses: the occupancy pole of the idiom, on one clock — #222.
//!
//! `scripts/modal-run bench --case norm-occupancy`. **Needs a B200.**
//!
//! This repo's own groupnorm result is one pole of the walk idiom: a kernel
//! holding a whole band went 594 → 5996 GB/s when the band was streamed a
//! `CHUNK` at a time, because it was frame- and depot-bound and the walk was
//! the cure (`docs/kernels/layernorm.md`). oxide-train PR #124 is the other:
//! `rms_norm_forward_tile_bf16`, a faithful tile-walk rewrite of a block-per-row
//! norm, measured **+7–14% worse across six runs, one sign**, and was reverted.
//! Nothing about that kernel was frame-bound; it ran at 1.8 TB/s, 23% of peak,
//! and the rewrite's only effect was to move where the threads were.
//!
//! Neither pole is reproducible from the other's numbers, and the second one
//! was measured in a downstream tree against kernels that are not in this one.
//! So this module builds the losing comparison here, out of this library's own
//! movers, at the shape it lost at — `dim = 3072`, bf16, two passes over the
//! row — and prints the three columns the verdict is made of: registers a
//! thread, threads an SM, and the clock.
//!
//! # The arms
//!
//! Every arm computes `y = x · rsqrt(mean(x²) + ε)` per row, reads the row
//! twice and writes it once, so **all of them issue the same bytes** and a
//! difference between them is never traffic.
//!
//! - [`kernels::rms_norm_per_row`] is the block-per-row reference: one CTA of
//!   [`PER_ROW_THREADS`] threads per row, `dim / 256` elements a lane, a
//!   [`block_reduce_sum`] for the statistic, and no register tile anywhere. It
//!   is what the downstream kernel was before the rewrite.
//! - The four `rms_norm_walk_*` entries are the tile walk, at the two levers
//!   the idiom actually has. A warp owns 16 rows — [`BaseLdtm`]'s floor — and
//!   walks the row a `CHUNK` of columns at a time through [`load_rows`] and
//!   [`store_rows`]; `WARPS` says how many such warps a CTA runs.
//!
//! **Those two levers are the whole of the experiment**, and they are separated
//! on purpose because the issue conflates them. `CHUNK` sets **elements a lane**
//! (`16 · CHUNK / 32`), which is register pressure and therefore occupancy when
//! registers bind. `WARPS` sets **threads a block**, which is occupancy only
//! when something *else* binds — shared memory, or the 32-blocks-an-SM cap.
//! A 2×2 cube says which of the two the downstream rewrite was actually paying:
//!
//! | entry | CHUNK | WARPS | threads/block | fp32 a lane | rows/CTA |
//! | --- | ---: | ---: | ---: | ---: | ---: |
//! | `rms_norm_walk_c64_w4` | 64 | 4 | 128 | 32 | 64 |
//! | `rms_norm_walk_c64_w8` | 64 | 8 | 256 | 32 | 128 |
//! | `rms_norm_walk_c16_w4` | 16 | 4 | 128 | 8 | 64 |
//! | `rms_norm_walk_c16_w8` | 16 | 8 | 256 | 8 | 128 |
//!
//! `c64_w4` is the reverted kernel's shape — 128 threads, 32 values a lane
//! against the reference's twelve. `c16_w8` is the **narrow-per-lane,
//! high-occupancy** form #222 asks whether the walk machinery can express: 256
//! threads a block at a quarter of the per-lane liveness, and the answer to
//! *that* half of the issue is that it needs no new API at all — `CHUNK` and
//! `WARPS` are already what the walk is parameterized on, and the only thing
//! the library fixes is the 16-row floor a warp's band cannot go under.
//!
//! # What the table cannot be read as
//!
//! The second read of a row is L1-resident for the per-row arm (one row, 6 KiB
//! at `dim = 3072`) and is not for any walk arm (a CTA owns 16 rows a warp, so
//! its band is 384 KiB at `c64_w4` and 768 KiB at `w8`). That is not a flaw in
//! the comparison — it is the same 16-row floor showing up in a second place,
//! and it is why `GB/s` here is *issued* bytes and not HBM bytes. A reader who
//! wants those two effects separated wants a profiler, which Modal does not
//! give us; what this file can do is name the confound beside the number.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaFunction, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::bench::{ITERATIONS, Shape, Timings, WARMUP, middle, time};

use kittens::global::{GlobalRows, load_rows, store_rows};
use kittens::plan::SharedPlan;
use kittens::reg::{Add, BaseLdtm, FragmentLayout, RegTile, RegVec, rsqrt, warp_reduce};
use kittens::shared::{Bf16, Element, F32, SharedVec};
use kittens::sync::block_reduce_sum;
use kittens::{lane, warp_id};

/// Rows one warp's band covers, and the thing the walk cannot go under:
/// [`BaseLdtm`] is a 16-row ownership map, so a warp that wants fewer rows has
/// no layout to ask for. Everything the idiom can trade is on the other axis.
const ROWS_PER_WARP: usize = 16;

/// Warps in the block-per-row reference — 256 threads, which is what the
/// downstream kernel this arm stands in for launches, and what puts twelve
/// elements in a lane at `dim = 3072`.
const PER_ROW_WARPS: usize = 8;
pub const PER_ROW_THREADS: u32 = 32 * PER_ROW_WARPS as u32;

/// The per-warp partials the reference's block reduction folds through.
struct PerRowShared {
    partials: SharedVec<F32, PER_ROW_WARPS>,
    plan: SharedPlan,
}

#[inline(always)]
const fn per_row_shared(at: SharedPlan) -> PerRowShared {
    let (partials, at) = at.vec::<F32, PER_ROW_WARPS>();
    PerRowShared { partials, plan: at }
}

/// Dynamic shared memory the reference declares: one fp32 a warp, and the only
/// shared memory in this file. **The walk arms declare none** — their statistic
/// never leaves the warp that owns the rows, which is the code-quality half of
/// the idiom and is not what this measurement is about.
pub const PER_ROW_SHARED_BYTES: usize = per_row_shared(SharedPlan::sizing()).plan.bytes();

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The block-per-row reference: one CTA a row, [`PER_ROW_THREADS`] threads
    /// striding the row in pairs, a shared-memory fold for the statistic.
    ///
    /// The pair stride is what makes a lane's two elements one access and the
    /// warp's sixteen lanes one 128-byte transaction — the coalescing the tile
    /// arms give up, and the reason this arm is the reference rather than a
    /// strawman.
    ///
    /// # Safety
    ///
    /// Launch with [`PER_ROW_THREADS`] threads, [`PER_ROW_SHARED_BYTES`] of
    /// dynamic shared memory and one block per row. `source` and `destination`
    /// are live `[gridDim.x, columns]` bf16 buffers; `columns` is even.
    #[kernel]
    pub unsafe fn rms_norm_per_row(
        source: &[u16],
        columns: u32,
        epsilon: f32,
        mut destination: DisjointSlice<u16>,
    ) {
        unsafe { per_row(source, columns, epsilon, &mut destination) }
    }

    /// The walk at the shape oxide-train reverted: 64 columns a chunk, four
    /// warps, 32 fp32 a lane.
    ///
    /// # Safety
    ///
    /// As [`walk`]: `32 * WARPS` threads, `gridDim.x * 16 * WARPS` rows of live
    /// bf16 at `columns` elements a row, and `columns` a multiple of the chunk.
    #[kernel]
    pub unsafe fn rms_norm_walk_c64_w4(
        source: &[u16],
        columns: u32,
        epsilon: f32,
        mut destination: DisjointSlice<u16>,
    ) {
        unsafe { walk::<64, 4>(source, columns, epsilon, &mut destination) }
    }

    /// The same chunk in a block twice as wide: 256 threads, and the same 32
    /// fp32 a lane. The arm that says whether *block width alone* buys
    /// anything.
    ///
    /// # Safety
    ///
    /// As [`rms_norm_walk_c64_w4`].
    #[kernel]
    pub unsafe fn rms_norm_walk_c64_w8(
        source: &[u16],
        columns: u32,
        epsilon: f32,
        mut destination: DisjointSlice<u16>,
    ) {
        unsafe { walk::<64, 8>(source, columns, epsilon, &mut destination) }
    }

    /// The narrow chunk at the reverted kernel's block width: 8 fp32 a lane,
    /// 128 threads. Register pressure moved without the block moving.
    ///
    /// # Safety
    ///
    /// As [`rms_norm_walk_c64_w4`].
    #[kernel]
    pub unsafe fn rms_norm_walk_c16_w4(
        source: &[u16],
        columns: u32,
        epsilon: f32,
        mut destination: DisjointSlice<u16>,
    ) {
        unsafe { walk::<16, 4>(source, columns, epsilon, &mut destination) }
    }

    /// **The narrow-per-lane, high-occupancy walk** #222 asks about: 8 fp32 a
    /// lane and 256 threads a block, out of the same two const parameters every
    /// other entry here takes.
    ///
    /// # Safety
    ///
    /// As [`rms_norm_walk_c64_w4`].
    #[kernel]
    pub unsafe fn rms_norm_walk_c16_w8(
        source: &[u16],
        columns: u32,
        epsilon: f32,
        mut destination: DisjointSlice<u16>,
    ) {
        unsafe { walk::<16, 8>(source, columns, epsilon, &mut destination) }
    }
}

/// `y = x · rsqrt(mean(x²) + ε)` over one row a block, the way a kernel with no
/// tile types writes it.
///
/// Two passes over the row and a `block_reduce_sum` between them. The row is
/// re-read rather than kept in registers so that this arm issues exactly the
/// bytes the walk arms do — the comparison this file exists for is only a
/// comparison while that holds.
///
/// # Safety
///
/// Every thread of a [`PER_ROW_THREADS`]-thread 1-D block calls it with the
/// same arguments, `source` and `destination` hold `blockIdx.x + 1` rows of
/// `columns` bf16, and `columns` is even.
#[inline(always)]
unsafe fn per_row(
    source: &[u16],
    columns: u32,
    epsilon: f32,
    destination: &mut DisjointSlice<u16>,
) {
    unsafe {
        let PerRowShared { partials, .. } = per_row_shared(SharedPlan::attach());
        let base = thread::blockIdx_x() as usize * columns as usize;
        let read = source.as_ptr().add(base) as *const u8;
        let write = destination.as_mut_ptr().add(base) as *mut u8;

        let stride = 2 * PER_ROW_THREADS;
        let start = 2 * thread::threadIdx_x();

        let mut total = 0.0f32;
        let mut column = start;
        while column < columns {
            let (low, high) = Bf16::read_pair(read.add(2 * column as usize));
            total += low * low + high * high;
            column += stride;
        }
        // `block_reduce_sum` takes a warp-uniform value and folds one per warp,
        // so the lane-local half of the fold is owed here.
        let mean =
            block_reduce_sum::<PER_ROW_WARPS>(partials, warp_reduce::<Add>(total)) / columns as f32;
        let scale = rsqrt(mean + epsilon);

        let mut column = start;
        while column < columns {
            let at = 2 * column as usize;
            let (low, high) = Bf16::read_pair(read.add(at));
            Bf16::write_pair(write.add(at), low * scale, high * scale);
            column += stride;
        }
    }
}

/// The same normalization as a tile walk: a warp owns [`ROWS_PER_WARP`] rows and
/// walks them `CHUNK` columns at a time, twice.
///
/// The statistic is a [`RegVec`] carried across chunks and never leaves the
/// warp — no shared memory, no barrier, and no block-wide anything, which is
/// the idiom's whole case. What it costs is on the launch: `16 · CHUNK / 32`
/// fp32 live in every lane, and a block that is `32 · WARPS` threads wide
/// however few rows the problem has.
///
/// # Safety
///
/// - All 32 lanes of every warp of a `32 * WARPS`-thread 1-D block call it with
///   the same arguments.
/// - `source` and `destination` hold `gridDim.x * ROWS_PER_WARP * WARPS` rows of
///   `columns` bf16 each, and `columns` is a multiple of `CHUNK`.
#[inline(always)]
unsafe fn walk<const CHUNK: usize, const WARPS: usize>(
    source: &[u16],
    columns: u32,
    epsilon: f32,
    destination: &mut DisjointSlice<u16>,
) where
    BaseLdtm: FragmentLayout<ROWS_PER_WARP, CHUNK>,
{
    unsafe {
        let lane = lane();
        let row = ROWS_PER_WARP as u32 * (WARPS as u32 * thread::blockIdx_x() + warp_id());
        let stride = columns as usize;
        let read = GlobalRows::<Bf16>::from_raw(source.as_ptr() as *mut u8, stride);
        let write = GlobalRows::<Bf16>::from_slice(destination, stride);

        let mut squares = RegVec::<ROWS_PER_WARP, BaseLdtm>::splat(0.0);
        let mut column = 0u32;
        while column < columns {
            let band: RegTile<ROWS_PER_WARP, CHUNK, BaseLdtm> = load_rows(read, row, column, lane);
            squares.add_assign(band.mul(band).row_sum());
            column += CHUNK as u32;
        }
        let scale = squares.scale(1.0 / columns as f32).shift(epsilon).rsqrt();

        let mut column = 0u32;
        while column < columns {
            let band: RegTile<ROWS_PER_WARP, CHUNK, BaseLdtm> = load_rows(read, row, column, lane);
            store_rows(write, row, column, lane, band.mul_row(scale));
            column += CHUNK as u32;
        }
    }
}

/// One arm as the host has to know it: what to call it, how wide its block is,
/// what shared memory it declares, how many rows a CTA covers, and the fp32 a
/// lane holds — the last two being the two levers the table is about.
struct Arm {
    name: &'static str,
    entry: &'static str,
    threads: u32,
    shared_bytes: u32,
    rows_per_block: usize,
    values_per_lane: usize,
}

const ARMS: [Arm; 5] = [
    Arm {
        name: "per-row",
        entry: "rms_norm_per_row",
        threads: PER_ROW_THREADS,
        shared_bytes: PER_ROW_SHARED_BYTES as u32,
        rows_per_block: 1,
        // `dim / PER_ROW_THREADS` and therefore shape-dependent; printed from
        // the shape rather than from here, which is why this is zero.
        values_per_lane: 0,
    },
    Arm {
        name: "walk c64 w4",
        entry: "rms_norm_walk_c64_w4",
        threads: 128,
        shared_bytes: 0,
        rows_per_block: 64,
        values_per_lane: 32,
    },
    Arm {
        name: "walk c64 w8",
        entry: "rms_norm_walk_c64_w8",
        threads: 256,
        shared_bytes: 0,
        rows_per_block: 128,
        values_per_lane: 32,
    },
    Arm {
        name: "walk c16 w4",
        entry: "rms_norm_walk_c16_w4",
        threads: 128,
        shared_bytes: 0,
        rows_per_block: 64,
        values_per_lane: 8,
    },
    Arm {
        name: "walk c16 w8",
        entry: "rms_norm_walk_c16_w8",
        threads: 256,
        shared_bytes: 0,
        rows_per_block: 128,
        values_per_lane: 8,
    },
];

/// The residual a correct kernel never reaches: bf16 outputs of order 1 round
/// to 2⁻⁹, and this is two doublings above that.
const EPSILON: f32 = 1e-5;
const TOLERANCE: f32 = 1.0 / 128.0;

/// Distinct residues a row's values cycle through — coprime with the strides
/// below, so a row holds every one of them and two rows a class apart hold the
/// same multiset in a different order.
const CLASSES: usize = 13;

/// The staged value at `(row, column)`: a half-integer in `[-7, 7]`, never
/// zero, and exact in bf16 so the reference's rounding is the kernel's.
fn value(row: usize, column: usize) -> f32 {
    let residue = (7 * column + 3 * row) % CLASSES;
    let magnitude = 0.5 * (residue + 1) as f32;
    if residue.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

/// `rsqrt(mean(x²) + ε)` for any row of class `class`, in f64. Thirteen of
/// these cover every row of every size, since a row's multiset depends only on
/// `row % CLASSES`.
fn deviation(class: usize, columns: usize) -> f64 {
    let squares: f64 = (0..columns)
        .map(|column| {
            let x = value(class, column) as f64;
            x * x
        })
        .sum();
    1.0 / (squares / columns as f64 + EPSILON as f64).sqrt()
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The staged input, row-major bf16.
fn staged(rows: usize, columns: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|flat| to_bf16(value(flat / columns, flat % columns)))
        .collect()
}

/// Every output against the f64 reference, over every element — not a sample.
/// The way a norm goes wrong is one warp's rows, and a sample is exactly what
/// misses that.
fn verify(observed: &[u16], rows: usize, columns: usize) -> Result<f64, Box<dyn Error>> {
    let deviations: Vec<f64> = (0..CLASSES)
        .map(|class| deviation(class, columns))
        .collect();
    let (mut wrong, mut sample, mut worst) = (0usize, Vec::new(), 0.0f64);
    for row in 0..rows {
        let deviation = deviations[row % CLASSES];
        for column in 0..columns {
            let expected = (value(row, column) as f64 * deviation) as f32;
            let seen = from_bf16(observed[row * columns + column]);
            let error = ((seen - expected) / expected).abs() as f64;
            worst = worst.max(error);
            // Negated rather than `>`: a NaN compares false either way, and
            // only this spelling counts one as wrong.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(error <= TOLERANCE as f64) {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!("[{row}, {column}] = {seen}, want {expected}"));
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!(
            "{wrong} of {} outputs outside 2^-7: {}",
            rows * columns,
            sample.join("; ")
        )
        .into());
    }
    Ok(worst)
}

/// One launch of `arm` through the generated launcher for its entry.
///
/// A `match` on the entry name rather than a function pointer: the five
/// launchers are five distinct methods with one signature, and this is the only
/// place that has to know which name is which.
///
/// # Safety
///
/// The grid covers exactly the rows of `source` and `destination` at `columns`
/// elements a row, the block is the arm's own, and the reference arm's shared
/// plan is declared in `config`.
unsafe fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    config: LaunchConfig,
    arm: &Arm,
    source: &DeviceBuffer<u16>,
    columns: u32,
    destination: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        match arm.entry {
            "rms_norm_per_row" => {
                module.rms_norm_per_row(stream, config, source, columns, EPSILON, destination)
            }
            "rms_norm_walk_c64_w4" => {
                module.rms_norm_walk_c64_w4(stream, config, source, columns, EPSILON, destination)
            }
            "rms_norm_walk_c64_w8" => {
                module.rms_norm_walk_c64_w8(stream, config, source, columns, EPSILON, destination)
            }
            "rms_norm_walk_c16_w4" => {
                module.rms_norm_walk_c16_w4(stream, config, source, columns, EPSILON, destination)
            }
            _ => module.rms_norm_walk_c16_w8(stream, config, source, columns, EPSILON, destination),
        }?
    };
    Ok(())
}

/// The five arms staged once at one shape, so a pass over them is five launches
/// against one pair of buffers rather than five stagings.
///
/// Every arm is checked against the CPU reference before anything is timed —
/// `bench.rs`' rule 1, which a file whose whole output is a ratio has the most
/// reason to want to slip.
struct Staged {
    stream: Arc<cuda_core::CudaStream>,
    module: kernels::LoadedModule,
    source: DeviceBuffer<u16>,
    destination: DeviceBuffer<u16>,
    rows: usize,
    columns: usize,
}

impl Staged {
    fn new(
        context: &Arc<CudaContext>,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let widest = ROWS_PER_WARP * 8;
        if !rows.is_multiple_of(widest) {
            return Err(format!("{rows} rows does not divide the {widest} a wide CTA owns").into());
        }
        if !columns.is_multiple_of(64) {
            return Err(format!("{columns} columns does not divide the widest chunk, 64").into());
        }
        let stream = context.default_stream();
        let module = kernels::load(context)?;
        let source = DeviceBuffer::from_host(&stream, &staged(rows, columns))?;
        let destination = DeviceBuffer::<u16>::zeroed(&stream, rows * columns)?;
        Ok(Staged {
            stream,
            module,
            source,
            destination,
            rows,
            columns,
        })
    }

    fn grid(&self, arm: &Arm) -> u32 {
        (self.rows / arm.rows_per_block) as u32
    }

    fn config(&self, arm: &Arm) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.grid(arm), 1, 1),
            block_dim: (arm.threads, 1, 1),
            shared_mem_bytes: arm.shared_bytes,
        }
    }

    /// Launch once and compare every element against the f64 reference.
    ///
    /// The destination is zeroed first, and that is not hygiene: five arms
    /// share one buffer, so an arm that wrote only some of its rows would
    /// otherwise be checked against the previous arm's output and pass.
    fn check(&mut self, arm: &Arm) -> Result<f64, Box<dyn Error>> {
        let config = self.config(arm);
        self.destination.zero_async(&self.stream)?;
        let Staged {
            stream,
            module,
            source,
            destination,
            columns,
            ..
        } = self;
        // SAFETY: as `launch`.
        unsafe {
            launch(
                module,
                stream,
                config,
                arm,
                source,
                *columns as u32,
                destination,
            )?
        };
        let observed = self.destination.to_host_vec(&self.stream)?;
        verify(&observed, self.rows, self.columns)
    }

    /// The clock, after the check — never instead of it.
    fn timed(&mut self, arm: &Arm) -> Result<Timings, Box<dyn Error>> {
        let config = self.config(arm);
        let Staged {
            stream,
            module,
            source,
            destination,
            columns,
            ..
        } = self;
        let columns = *columns as u32;
        let mut once = || -> Result<(), Box<dyn Error>> {
            // SAFETY: as `launch`.
            unsafe { launch(module, stream, config, arm, source, columns, destination) }
        };
        time(stream, &mut once)
    }

    /// Registers a thread, local frame, and the blocks an SM the driver admits
    /// at this arm's own launch envelope — read off the loaded function rather
    /// than derived, the way `device-tests`' ladder reads them.
    fn envelope(&self, arm: &Arm) -> Result<(u32, u32, u32), Box<dyn Error>> {
        let function = self.module.as_cuda_module().load_function(arm.entry)?;
        let blocks =
            function.max_active_blocks_per_multiprocessor(arm.threads, arm.shared_bytes)?;
        Ok((
            attribute(
                &function,
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
                "NUM_REGS",
            )?,
            attribute(
                &function,
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "LOCAL_SIZE_BYTES",
            )?,
            blocks,
        ))
    }
}

fn attribute(
    function: &CudaFunction,
    attribute: cuda_core::sys::CUfunction_attribute,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    let mut value: i32 = 0;
    // SAFETY: `function` is loaded on a live context and the attribute is one
    // of the driver's own enumerators.
    let status = unsafe {
        cuda_core::sys::cuFuncGetAttribute(&mut value, attribute, function.cu_function())
    };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuFuncGetAttribute({name}) failed with driver status {status}").into(),
        );
    }
    Ok(value as u32)
}

/// Bytes a launch issues: every element read twice and written once, all bf16.
/// The same denominator for every arm, because every arm makes the same three
/// accesses — that is what the two-pass structure is holding fixed.
fn bytes(rows: usize, columns: usize) -> f64 {
    3.0 * 2.0 * (rows * columns) as f64
}

/// Whole measurements each arm gets, taken round-robin over the five so that a
/// device stepping its clocks down over the sweep does not order the table.
const REPEATS: usize = 3;

/// The shapes, and why they are these.
///
/// The first is oxide-train's own: `B·T = 24576` rows of `dim = 3072`, which is
/// where `rms_norm_forward_tile_bf16` measured +7–14% and is the row this file
/// exists to reproduce. The second halves the row length with the row count
/// held, so the walk's chunk count halves and its per-lane liveness does not —
/// the arm that says whether the verdict is about `dim`. The third is a long
/// row at a quarter of the rows: more columns a warp walks, fewer CTAs to hide
/// anything behind, which is where a wide chunk should look best if it ever
/// does.
const SIZES: [Shape; 3] = [
    Shape {
        m: 24576,
        n: 3072,
        k: 1,
    },
    Shape {
        m: 24576,
        n: 1024,
        k: 1,
    },
    Shape {
        m: 6144,
        n: 8192,
        k: 1,
    },
];

/// The check size: small enough to stage and verify in a second, large enough
/// that every arm's grid is more than one block and every class of row appears.
const CHECK_ROWS: usize = 512;
const CHECK_COLUMNS: usize = 1024;

/// All five arms against the CPU reference — `modal_app.py::examples`' row for
/// this file, and the gate the timed numbers below are only meaningful behind.
pub fn check(context: &Arc<CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut staged = Staged::new(context, CHECK_ROWS, CHECK_COLUMNS)?;
    let mut worst = 0.0f64;
    for arm in &ARMS {
        worst = worst.max(staged.check(arm)?);
    }
    Ok(format!(
        "{CHECK_ROWS}x{CHECK_COLUMNS} normalized by all {} arms, worst relative error {worst:.2e}",
        ARMS.len()
    ))
}

/// `bench --case norm-occupancy` — the block-per-row reference, the walk that
/// lost, and the walk cube around it, on one clock in one container.
pub fn compare(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "rms norm: a block-per-row kernel against the tile walk, at the two levers the walk\n\
         has — `CHUNK` (fp32 a lane) and `WARPS` (threads a block). Every arm reads the row\n\
         twice and writes it once, so all five issue the same bytes and a difference between\n\
         them is never traffic. {REPEATS} whole measurements of each arm, taken round-robin so\n\
         every pair is adjacent in time, each one min ms over {ITERATIONS} timed launches after\n\
         {WARMUP} warm-up, and every arm checked against the CPU reference before any of it.\n\
         `regs`, `frame` and `blocks/SM` are the driver's, for the loaded function at that\n\
         arm's own launch envelope; `threads/SM` is `blocks/SM · threads`, and it is the column\n\
         #222 is about."
    );

    for shape in SIZES {
        let (rows, columns) = (shape.m, shape.n);
        let mut staged = Staged::new(context, rows, columns)?;
        println!(
            "\n{rows} rows of {columns}, {:.1} MB moved a launch",
            bytes(rows, columns) / 1e6
        );
        println!(
            "{:<14}{:>8}{:>8}{:>9}{:>7}{:>7}{:>11}{:>12}{:>10}{:>10}{:>8}",
            "arm",
            "threads",
            "f32/lane",
            "blocks",
            "regs",
            "frame",
            "blocks/SM",
            "threads/SM",
            "min ms",
            "GB/s",
            "vs row",
        );

        let mut worst = 0.0f64;
        for arm in &ARMS {
            eprintln!("{rows}x{columns} {}: checking", arm.name);
            worst = worst.max(staged.check(arm)?);
        }

        let mut samples: Vec<Vec<f64>> = ARMS.iter().map(|_| Vec::new()).collect();
        for pass in 1..=REPEATS {
            for (index, arm) in ARMS.iter().enumerate() {
                eprintln!("{rows}x{columns} {} pass {pass}", arm.name);
                samples[index].push(staged.timed(arm)?.min());
            }
        }

        let reference = middle(&samples[0]);
        for (index, arm) in ARMS.iter().enumerate() {
            let (registers, frame, blocks_per_sm) = staged.envelope(arm)?;
            let milliseconds = middle(&samples[index]);
            let per_lane = match arm.values_per_lane {
                0 => columns / arm.threads as usize,
                values => values,
            };
            println!(
                "{:<14}{:>8}{:>8}{:>9}{:>7}{:>7}{:>11}{:>12}{:>10.4}{:>10.1}{:>8.3}",
                arm.name,
                arm.threads,
                per_lane,
                staged.grid(arm),
                registers,
                frame,
                blocks_per_sm,
                blocks_per_sm * arm.threads,
                milliseconds,
                bytes(rows, columns) / (milliseconds / 1e3) / 1e9,
                milliseconds / reference,
            );
        }
        println!(
            "worst relative error over all five arms at this shape: {worst:.2e} (tolerance 2^-7)"
        );
    }

    println!(
        "\n`vs row` is each arm's milliseconds over the block-per-row arm's, so above 1.000 is\n\
         the walk losing. GB/s counts *issued* bytes: the per-row arm's second read of a row is\n\
         L1-resident and a walk arm's is not, since a warp owns 16 rows whatever the kernel\n\
         wants — the same floor that sets `f32/lane` at a given chunk."
    );
    Ok(())
}
