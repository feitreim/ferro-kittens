//! Device tests for ferro-kittens: what the host unit tests cannot prove.
//!
//! The library's `#[cfg(test)]` suites establish that its address and
//! coordinate arithmetic is *self-consistent*. Nothing there can show that
//! [`BaseLdtm`]'s ownership map, the SWIZZLE_128B phase math, or the
//! `stmatrix` store path are the ones the silicon actually uses — those were
//! only ever established indirectly, by downstream kernels producing correct
//! numbers. This binary closes that gap by running each of them on a B200 and
//! checking the result against the library's own maps.
//!
//! Every case follows the same discipline: seed a pattern whose value at
//! `(row, column)` identifies that position exactly (no rounding anywhere in
//! the chain), move it through one library path, and report the values by
//! **thread coordinate** — `(warp, lane, slot, value)` — so the host, not the
//! kernel, is what applies the ownership map. A failure is then a wrong
//! coordinate and never a numerical artifact, and the observed map can be read
//! straight back out of the dump.
//!
//! Run it with `modal run modal_app.py` (see `modal_app.py` at the repo root);
//! it exits non-zero if any case fails.

use std::error::Error;
use std::fmt::Write as _;
use std::process::ExitCode;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::DisjointSlice;
use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::global::encode_bf16_panels;
use kittens::ldst::store_fragment;
use kittens::mma::{self, mma_abt};
use kittens::reg::{
    BaseLdtm, Fragment, FragmentLayout, Mul, RegTile, RegVec, fmax, online_rescale,
};
use kittens::shared::{Bf16, SharedTile, Swizzle128B};
use kittens::sync::Semaphore;
use kittens::tmem::{TmemTile, alloc_block, dealloc_block};

/// Edge of the square single-subtile tiles the swizzle and `stmatrix` cases
/// use: 64 bf16 columns is exactly one 128-byte swizzle atom per row, the only
/// width [`SharedTile::chunk_writer`] accepts.
const TILE: usize = 64;
/// 16-byte chunks in one 128-byte swizzled row.
const CHUNKS: usize = 8;
/// 32-bit words in one row of a `[TILE, TILE]` bf16 tile.
const ROW_WORDS: usize = TILE / 2;

/// Rows of the probe accumulator — one full `M128` MMA shape, and the four
/// warps' 32 TMEM rows each.
const ROWS: usize = 128;
/// Columns of the probe accumulator: two `M128_N64` bands side by side, so a
/// warp's drained band is wide enough for the `(32, 128)` layout shape.
const COLUMNS: usize = 128;
/// K of the probe MMA: one swizzle atom of bf16, four chained K=16 chunks.
const DEPTH: usize = 64;

/// The MMA operands and the square tiles, as the library types.
type Square = SharedTile<Bf16, TILE, TILE, Swizzle128B>;
type AOperand = SharedTile<Bf16, ROWS, DEPTH, Swizzle128B>;
type BOperand = SharedTile<Bf16, TILE, DEPTH, Swizzle128B>;
type Accumulator = TmemTile<ROWS, COLUMNS>;

/// Dynamic shared plan of the fragment probe: the A operand, the two B
/// operands, then a 32-byte scratch tail holding the two mbarriers and the
/// TMEM staging word.
const PROBE_SHARED: usize = AOperand::BYTES + 2 * BOperand::BYTES + 32;
/// Dynamic shared plan of the square-tile cases: the tile plus a scratch tail
/// for the swizzle case's TMA barrier.
const SQUARE_SHARED: usize = Square::BYTES + 32;

/// A bf16 bit pattern naming a position in a `[TILE, TILE]` tile.
///
/// Bit 14 is set so the pattern is a *normal* bf16 (exponent field 0x80..0xa0)
/// — a position encoding down in the subnormal range would be at the mercy of
/// flush-to-zero somewhere in the conversion. Nothing here is ever read as a
/// number: the tests compare bit patterns, so the round trip is exact by
/// construction and a mismatch is always a misplaced element.
const fn cell_bits(row: usize, column: usize) -> u16 {
    0x4000 | ((row as u16) << 6) | column as u16
}

/// [`cell_bits`] as the fp32 a register-side path carries. The low 16 bits are
/// zero, so packing it back to bf16 is exact whatever rounding mode the
/// conversion uses.
const fn cell(row: usize, column: usize) -> f32 {
    f32::from_bits((cell_bits(row, column) as u32) << 16)
}

/// The probe accumulator's value at `(row, column)` — `COLUMNS * row + column`,
/// unique over the whole `[ROWS, COLUMNS]` tile and an exact fp32 integer, so a
/// drained value decodes straight back to the coordinate the hardware gave it.
fn accumulator_value(row: usize, column: usize) -> f32 {
    (COLUMNS * row + column) as f32
}

/// Which spelling of the online-softmax correction [`kernels::softmax_probe`]
/// compiles. Only the codegen probe reads these; see its doc comment.
const HAND_WRITTEN: u32 = 0;
/// [`online_rescale`], the deliberately fused form.
const FUSED: u32 = 1;
/// [`HAND_WRITTEN`] with the generic `row_map::<Mul>` in place of `scale_rows`.
const ROW_MAP: u32 = 2;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// TMA one `[TILE, TILE]` bf16 tile into shared memory, then read it back
    /// through [`SharedTile::chunk_writer`] and write it out in *logical*
    /// order — so the host's expectation is simply the buffer it staged.
    ///
    /// This is the swizzle path with nothing else in it: it checks
    /// `swizzle_phase`, the chunk XOR, and the tensor map's box geometry
    /// against the TMA engine, and needs no MMA. Launch with `TILE` threads,
    /// one row each.
    #[kernel]
    pub unsafe fn swizzle_roundtrip(source: *const TmaDescriptor, mut out: DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Square::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Square::BYTES) as *mut Barrier);
            let tid = thread::threadIdx_x();

            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tile.tma_load(source, 0, 0, tma);
                tma.expect_tx(Square::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            dump_rows(tile, tid as usize, TILE, &mut out);

            thread::sync_threads();
            if tid == 0 {
                tma.inval();
            }
        }
    }

    /// Fill a `[TILE, TILE]` shared tile from registers through
    /// [`store_fragment`], then read it back the same way
    /// [`swizzle_roundtrip`] does.
    ///
    /// The fragment of each 16x16 block is built *from the ownership map* —
    /// value `(slot, value)` gets `cell(BaseLdtm::row(..), BaseLdtm::column(..))`
    /// — so a correct round trip means the `stmatrix` addressing and the map
    /// agree, and the tile comes out in plain logical order. Launch with one
    /// warp: `store_fragment` is warp-scope and takes its addresses from lanes
    /// 0..16.
    #[kernel]
    pub unsafe fn stmatrix_roundtrip(mut out: DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Square::from_raw(smem);
            let chunks = tile.chunk_writer();
            let lane = warp::lane_id();

            let mut row_block = 0usize;
            while row_block < TILE / 16 {
                let mut column_block = 0usize;
                while column_block < TILE / 16 {
                    let mut fragment = Fragment::zero();
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let row = 16 * row_block + BaseLdtm::row(lane, slot) as usize;
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            let column = 16 * column_block + BaseLdtm::column(lane, value) as usize;
                            fragment.set(slot, value, cell(row, column));
                            value += 1;
                        }
                        slot += 1;
                    }
                    store_fragment::<Bf16>(
                        chunks,
                        (16 * row_block) as u32,
                        (16 * column_block) as u32,
                        lane,
                        fragment,
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            thread::sync_threads();

            dump_rows(tile, lane as usize, 32, &mut out);
        }
    }

    /// Read `tile`'s rows `first`, `first + stride`, … back through the
    /// swizzle and write them to `out` in logical order.
    #[inline(always)]
    unsafe fn dump_rows(tile: Square, first: usize, stride: usize, out: &mut DisjointSlice<u32>) {
        unsafe {
            let chunks = tile.chunk_writer();
            let mut row = first;
            while row < TILE {
                let mut chunk = 0usize;
                while chunk < CHUNKS {
                    let words = chunks.at(row, chunk) as *const u32;
                    let mut word = 0usize;
                    while word < 4 {
                        *out.get_unchecked_mut(row * ROW_WORDS + chunk * 4 + word) =
                            *words.add(word);
                        word += 1;
                    }
                    chunk += 1;
                }
                row += stride;
            }
        }
    }

    /// The fragment ownership probe, shared by every layout shape.
    ///
    /// `D = A·Bᵀ` over two `M128_N64` bands, seeded (by the host) so that
    /// `D[row, column] == COLUMNS * row + column` exactly. Each of the four
    /// warps drains its own 32 TMEM rows into a `RegTile<M, N, BaseLdtm>`,
    /// composing the `[16, 16]` fragments `TmemTile::fragment_tile` returns by
    /// `(2 * row_block + slot, 4 * column_block + value)` — the composition
    /// every hand-written drain loop spells out.
    ///
    /// The dump is indexed by `(warp, lane, slot, value)` alone. The host
    /// applies `RegTile::coordinate` to it, so this kernel never encodes what
    /// the answer should be.
    #[inline(always)]
    unsafe fn fragment_probe<const M: usize, const N: usize>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<M, N>,
    {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let a = AOperand::from_raw(smem);
            let b_low = BOperand::from_raw(smem.add(AOperand::BYTES));
            let b_high = BOperand::from_raw(smem.add(AOperand::BYTES + BOperand::BYTES));
            // Barriers and the TMEM staging word live in the plan's tail
            // rather than in `static mut` shared arrays, so one generic body
            // can serve every shape's kernel.
            let scratch = smem.add(AOperand::BYTES + 2 * BOperand::BYTES);
            let tma = Semaphore::attach(scratch as *mut Barrier);
            let mma_done = Semaphore::attach(scratch.add(8) as *mut Barrier);
            let tmem_slot = scratch.add(16) as *mut u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let leader = tid == 0;

            if leader {
                tma.init(1);
                mma_done.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(tmem_slot, COLUMNS as u32);
            let accumulator = Accumulator::from_raw(tmem);

            if leader {
                a.tma_load(a_map, 0, 0, tma);
                b_low.tma_load(b_map, 0, 0, tma);
                b_high.tma_load(b_map, 0, 1, tma);
                tma.expect_tx((AOperand::BYTES + 2 * BOperand::BYTES) as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            if leader {
                mma_abt(accumulator.raw(), a, b_low, instruction, false);
                mma_abt(
                    accumulator.columns_right(TILE as u32).raw(),
                    a,
                    b_high,
                    instruction,
                    false,
                );
                mma::commit(mma_done);
            }
            mma_done.wait(0);
            thread::sync_threads();

            let mut tile = RegTile::<M, N, BaseLdtm>::zero();
            let mut row_block = 0usize;
            while row_block < M / 16 {
                let mut column_block = 0usize;
                while column_block < N / 16 {
                    let fragment = accumulator.fragment_tile(
                        32 * warp_id + 16 * row_block as u32,
                        16 * column_block as u32,
                    );
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            tile.set(
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

            let values = RegTile::<M, N, BaseLdtm>::VALUES;
            let base = (32 * warp_id as usize + lane as usize) * M * N / 32;
            let mut slot = 0usize;
            while slot < RegTile::<M, N, BaseLdtm>::SLOTS {
                let mut value = 0usize;
                while value < values {
                    *out.get_unchecked_mut(base + slot * values + value) = tile.get(slot, value);
                    value += 1;
                }
                slot += 1;
            }

            thread::sync_threads();
            dealloc_block(tmem, COLUMNS as u32);
            if leader {
                tma.inval();
                mma_done.inval();
            }
        }
    }

    /// [`fragment_probe`] at the bare `Fragment` shape — one `16x256b` drain,
    /// nothing composed.
    #[kernel]
    pub unsafe fn fragment_map_16x16(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<16, 16>(a_map, b_map, &mut out) }
    }

    /// [`fragment_probe`] over a warp's full 32 rows by two column blocks.
    #[kernel]
    pub unsafe fn fragment_map_32x32(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<32, 32>(a_map, b_map, &mut out) }
    }

    /// [`fragment_probe`] at the flash output accumulator's shape — a warp's
    /// 32 TMEM rows by 128 columns, four slots of 32 values per thread.
    #[kernel]
    pub unsafe fn fragment_map_32x128(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<32, 128>(a_map, b_map, &mut out) }
    }

    /// Each row's max over the values one thread owns, quad-reduced into the
    /// whole row's max. The row reduction the library does not have yet (#6),
    /// written out here because the probe below needs one today.
    #[inline(always)]
    fn quad_row_max<const N: usize>(tile: RegTile<32, N, BaseLdtm>) -> RegVec<32, BaseLdtm>
    where
        BaseLdtm: FragmentLayout<32, N>,
    {
        let mut rows = RegVec::<32, BaseLdtm>::splat(f32::NEG_INFINITY);
        let mut slot = 0usize;
        while slot < RegTile::<32, N, BaseLdtm>::SLOTS {
            let mut lane_max = f32::NEG_INFINITY;
            let mut value = 0usize;
            while value < RegTile::<32, N, BaseLdtm>::VALUES {
                lane_max = fmax(lane_max, tile.get(slot, value));
                value += 1;
            }
            rows.set(slot, lane_max);
            slot += 1;
        }
        rows.quad_max()
    }

    /// Each row's sum, quad-reduced; see [`quad_row_max`].
    #[inline(always)]
    fn quad_row_sum<const N: usize>(tile: RegTile<32, N, BaseLdtm>) -> RegVec<32, BaseLdtm>
    where
        BaseLdtm: FragmentLayout<32, N>,
    {
        let mut rows = RegVec::<32, BaseLdtm>::splat(0.0);
        let mut slot = 0usize;
        while slot < RegTile::<32, N, BaseLdtm>::SLOTS {
            let mut lane_sum = 0.0f32;
            let mut value = 0usize;
            while value < RegTile::<32, N, BaseLdtm>::VALUES {
                lane_sum += tile.get(slot, value);
                value += 1;
            }
            rows.set(slot, lane_sum);
            slot += 1;
        }
        rows.quad_sum()
    }

    /// **A codegen probe, not a test.** No host case launches it and it checks
    /// nothing; it exists so `modal run modal_app.py::regcount` has something
    /// that *monomorphizes* the register-side softmax ops. A library function
    /// no kernel calls emits no PTX, and so measures nothing at all.
    ///
    /// The shape is flash attention's inner loop as the register file sees it:
    /// one `[32, N]` score block per step, a `RegVec` running maximum and sum,
    /// and a `RegTile<32, N, BaseLdtm>` accumulator rescaled into each new
    /// reference — the pattern `online_rescale` and the hand-written `RegVec`
    /// ops were extracted from. `RESCALE` picks which spelling of the
    /// correction is compiled and nothing else changes, so the three forms are
    /// comparable line for line:
    ///
    /// - [`HAND_WRITTEN`]: `max`/`sub`/`exp2`/`mul_assign` and `scale_rows`
    /// - [`FUSED`]: [`online_rescale`], one scalar factor live at a time
    /// - [`ROW_MAP`]: [`HAND_WRITTEN`] with `row_map::<Mul>` for `scale_rows`
    ///
    /// `steps` is a runtime bound so the accumulators stay live across
    /// iterations rather than unrolling into one straight-line block.
    #[inline(always)]
    unsafe fn softmax_probe<const N: usize, const RESCALE: u32>(
        scores: &[f32],
        steps: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<32, N>,
    {
        unsafe {
            let slots = RegTile::<32, N, BaseLdtm>::SLOTS;
            let values = RegTile::<32, N, BaseLdtm>::VALUES;
            let lane = warp::lane_id();

            let mut m_ref = RegVec::<32, BaseLdtm>::splat(f32::NEG_INFINITY);
            let mut running_sum = RegVec::<32, BaseLdtm>::splat(0.0);
            let mut out_acc = RegTile::<32, N, BaseLdtm>::zero();

            let mut step = 0u32;
            while step < steps {
                let mut block = RegTile::<32, N, BaseLdtm>::zero();
                let mut slot = 0usize;
                while slot < slots {
                    let mut value = 0usize;
                    while value < values {
                        let (row, column) =
                            RegTile::<32, N, BaseLdtm>::coordinate(lane, slot, value);
                        let index = step as usize * 32 * N + row as usize * N + column as usize;
                        block.set(slot, value, *scores.get_unchecked(index));
                        value += 1;
                    }
                    slot += 1;
                }

                let row_max = quad_row_max::<N>(block);
                if RESCALE == FUSED {
                    online_rescale(&mut m_ref, row_max, &mut running_sum, &mut out_acc);
                } else {
                    let next = m_ref.max(row_max);
                    let factor = m_ref.sub(next).exp2();
                    m_ref = next;
                    running_sum.mul_assign(factor);
                    if RESCALE == ROW_MAP {
                        out_acc = out_acc.row_map::<Mul>(factor);
                    } else {
                        out_acc.scale_rows(factor);
                    }
                }

                let probabilities = block.sub_row(m_ref).exp2();
                running_sum.add_assign(quad_row_sum::<N>(probabilities));
                out_acc = out_acc.add(probabilities);
                step += 1;
            }

            // Every accumulator has to reach memory or the loop above is dead
            // and the register counts describe nothing.
            let normalized = out_acc.div_row(running_sum);
            let base = lane as usize * slots * values;
            let mut slot = 0usize;
            while slot < slots {
                let reference = m_ref.get(slot);
                let mut value = 0usize;
                while value < values {
                    *out.get_unchecked_mut(base + slot * values + value) =
                        normalized.get(slot, value) + reference;
                    value += 1;
                }
                slot += 1;
            }
        }
    }

    /// [`softmax_probe`] at a 32-wide score block. One line per instantiation:
    /// a generic body compiles nothing on its own.
    #[kernel]
    pub unsafe fn softmax_probe_32_hand_written(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<32, HAND_WRITTEN>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 32 wide, [`online_rescale`].
    #[kernel]
    pub unsafe fn softmax_probe_32_fused(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { softmax_probe::<32, FUSED>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 32 wide, `row_map::<Mul>` for `scale_rows`.
    #[kernel]
    pub unsafe fn softmax_probe_32_row_map(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<32, ROW_MAP>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at the flash accumulator's 128 columns — 32 values a
    /// thread on top of the score block, which is where register pressure
    /// stops being theoretical.
    #[kernel]
    pub unsafe fn softmax_probe_128_hand_written(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<128, HAND_WRITTEN>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 128 wide, [`online_rescale`].
    #[kernel]
    pub unsafe fn softmax_probe_128_fused(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { softmax_probe::<128, FUSED>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 128 wide, `row_map::<Mul>` for `scale_rows`.
    #[kernel]
    pub unsafe fn softmax_probe_128_row_map(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<128, ROW_MAP>(scores, steps, &mut out) }
    }
}

/// The layout shapes with a [`FragmentLayout`] impl, one line each — adding a
/// shape here plus a one-line kernel above is the whole cost of covering it.
#[derive(Clone, Copy)]
enum Shape {
    F16x16,
    F32x32,
    F32x128,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::F16x16, Shape::F32x32, Shape::F32x128];

    /// Logical rows and columns of one warp's drained band.
    fn dimensions(self) -> (usize, usize) {
        match self {
            Shape::F16x16 => (16, 16),
            Shape::F32x32 => (32, 32),
            Shape::F32x128 => (32, 128),
        }
    }

    /// The map under test, as the library computes it: which logical
    /// `(row, column)` of the band `lane`'s `(slot, value)` holds.
    fn coordinate(self, lane: u32, slot: usize, value: usize) -> (usize, usize) {
        let (row, column) = match self {
            Shape::F16x16 => RegTile::<16, 16, BaseLdtm>::coordinate(lane, slot, value),
            Shape::F32x32 => RegTile::<32, 32, BaseLdtm>::coordinate(lane, slot, value),
            Shape::F32x128 => RegTile::<32, 128, BaseLdtm>::coordinate(lane, slot, value),
        };
        (row as usize, column as usize)
    }

    fn name(self) -> &'static str {
        match self {
            Shape::F16x16 => "fragment map (16, 16)",
            Shape::F32x32 => "fragment map (32, 32)",
            Shape::F32x128 => "fragment map (32, 128)",
        }
    }

    unsafe fn launch(
        self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), cuda_core::DriverError> {
        let config = launch_config(ROWS as u32, PROBE_SHARED as u32);
        unsafe {
            match self {
                Shape::F16x16 => module.fragment_map_16x16(stream, config, a_map, b_map, out),
                Shape::F32x32 => module.fragment_map_32x32(stream, config, a_map, b_map, out),
                Shape::F32x128 => module.fragment_map_32x128(stream, config, a_map, b_map, out),
            }
        }
    }
}

fn launch_config(threads: u32, shared_mem_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    }
}

/// Round-to-nearest-even fp32 → bf16, for staging the MMA operands.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn pack(low: u16, high: u16) -> u32 {
    low as u32 | ((high as u32) << 16)
}

/// A `[TILE, TILE]` bf16 tile of position identities, packed as the staging
/// buffer a [`kittens::global::PanelMap`] describes.
fn identity_tile() -> Vec<u32> {
    let mut staged = Vec::with_capacity(TILE * ROW_WORDS);
    for row in 0..TILE {
        for pair in 0..ROW_WORDS {
            staged.push(pack(cell_bits(row, 2 * pair), cell_bits(row, 2 * pair + 1)));
        }
    }
    staged
}

/// The probe's `A` operand, `[ROWS, DEPTH]`: column 0 all ones, column 1 the
/// row index, everything else zero. Against [`probe_b`] this makes
/// `D[row, column] = COLUMNS * row + column`. Every factor is an integer
/// below 256 (exact in bf16) and every product below 2^24 (exact in fp32), so
/// a mismatch is a wrong coordinate and never a rounding artifact.
fn probe_a() -> Vec<u32> {
    let mut staged = vec![0u32; ROWS * DEPTH / 2];
    for row in 0..ROWS {
        staged[row * DEPTH / 2] = pack(to_bf16(1.0), to_bf16(row as f32));
    }
    staged
}

/// The probe's `B` operand as two `[TILE, DEPTH]` planes covering accumulator
/// columns `0..TILE` and `TILE..COLUMNS`: column 0 the accumulator column
/// index, column 1 the row stride `COLUMNS`.
fn probe_b() -> Vec<u32> {
    let mut staged = vec![0u32; 2 * TILE * DEPTH / 2];
    for column in 0..COLUMNS {
        staged[column * DEPTH / 2] = pack(to_bf16(column as f32), to_bf16(COLUMNS as f32));
    }
    staged
}

/// Decode a drained accumulator value back to the `(row, column)` the hardware
/// delivered — the inverse of [`accumulator_value`], and how a failing case
/// reports the map it actually observed.
fn decode(value: f32) -> Option<(usize, usize)> {
    let integer = value as i64;
    if integer as f32 != value || !(0..(ROWS * COLUMNS) as i64).contains(&integer) {
        return None;
    }
    Some((integer as usize / COLUMNS, integer as usize % COLUMNS))
}

fn check_swizzle(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile();
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let map = unsafe { encode_bf16_panels::<TILE, TILE>(stream, source.cu_deviceptr(), TILE, 1)? };
    let mut out = DeviceBuffer::<u32>::zeroed(stream, staged.len())?;
    unsafe {
        module.swizzle_roundtrip(
            stream,
            launch_config(TILE as u32, SQUARE_SHARED as u32),
            map.as_ptr(),
            &mut out,
        )?
    };
    compare_tile(&out.to_host_vec(stream)?, &staged)
}

fn check_stmatrix(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let expected = identity_tile();
    let mut out = DeviceBuffer::<u32>::zeroed(stream, expected.len())?;
    unsafe {
        module.stmatrix_roundtrip(stream, launch_config(32, SQUARE_SHARED as u32), &mut out)?
    };
    compare_tile(&out.to_host_vec(stream)?, &expected)
}

/// Compare a logical-order tile dump against the identities that were staged
/// into it, reporting the first few misplaced elements by coordinate.
fn compare_tile(observed: &[u32], expected: &[u32]) -> Result<String, Box<dyn Error>> {
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, (&got, &want)) in observed.iter().zip(expected).enumerate() {
        if got == want {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let (row, pair) = (index / ROW_WORDS, index % ROW_WORDS);
            let _ = write!(
                report,
                "\n    (row {row}, columns {}..{}): staged {want:#010x}, read back {got:#010x}",
                2 * pair,
                2 * pair + 2
            );
        }
    }
    if mismatches == 0 {
        return Ok(format!("{} elements placed exactly", 2 * expected.len()));
    }
    Err(format!("{mismatches} of {} words misplaced{report}", expected.len()).into())
}

/// The heart of the harness: does [`BaseLdtm`] describe the map the hardware
/// actually uses?
///
/// Every drained value decodes to the accumulator coordinate it came from, so
/// the assertion is `decode(observed) == coordinate(lane, slot, value)` with
/// the warp's row offset folded in. On failure the report is the map the
/// hardware delivered, never a relaxed expectation.
fn check_fragment_map(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    shape: Shape,
) -> Result<String, Box<dyn Error>> {
    let a = DeviceBuffer::from_host(stream, &probe_a())?;
    let b = DeviceBuffer::from_host(stream, &probe_b())?;
    let a_map = unsafe { encode_bf16_panels::<ROWS, DEPTH>(stream, a.cu_deviceptr(), ROWS, 1)? };
    let b_map = unsafe { encode_bf16_panels::<TILE, DEPTH>(stream, b.cu_deviceptr(), TILE, 2)? };

    let (m, n) = shape.dimensions();
    let warps = ROWS / 32;
    let per_lane = m * n / 32;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, warps * 32 * per_lane)?;
    unsafe { shape.launch(module, stream, a_map.as_ptr(), b_map.as_ptr(), &mut out)? };
    let observed = out.to_host_vec(stream)?;

    let slots = per_lane / (n / 4);
    let values = n / 4;
    let mut report = String::new();
    let mut mismatches = 0usize;
    for warp in 0..warps {
        for lane in 0..32u32 {
            for slot in 0..slots {
                for value in 0..values {
                    let index = ((warp * 32 + lane as usize) * slots + slot) * values + value;
                    let (row, column) = shape.coordinate(lane, slot, value);
                    let expected = accumulator_value(32 * warp + row, column);
                    if observed[index] == expected {
                        continue;
                    }
                    mismatches += 1;
                    if mismatches <= 8 {
                        let _ = match decode(observed[index]) {
                            Some((got_row, got_column)) => write!(
                                report,
                                "\n    warp {warp} lane {lane} slot {slot} value {value}: \
                                 map says ({}, {column}), hardware delivered ({got_row}, {got_column})",
                                32 * warp + row
                            ),
                            None => write!(
                                report,
                                "\n    warp {warp} lane {lane} slot {slot} value {value}: \
                                 map says ({}, {column}), hardware delivered {} — not a \
                                 position identity, so the MMA itself is suspect",
                                32 * warp + row,
                                observed[index]
                            ),
                        };
                    }
                }
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "{} values, all at BaseLdtm's coordinates",
            observed.len()
        ));
    }
    report.push_str(&observed_map(&observed, slots, values));
    Err(format!(
        "{mismatches} of {} values misplaced{report}",
        observed.len()
    )
    .into())
}

/// The ownership map the hardware actually delivered, read off warp 0's dump:
/// the row each `(lane, slot)` received and the column each `(lane, value)`
/// received. This is what to compare against `BaseLdtm::row`/`column` when the
/// assertion above fails — the expectation is never the thing to adjust.
fn observed_map(observed: &[f32], slots: usize, values: usize) -> String {
    let mut report =
        String::from("\n    observed map, warp 0 (rows per slot | columns per value):");
    for lane in 0..8u32 {
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        for slot in 0..slots {
            let index = (lane as usize * slots + slot) * values;
            rows.push(decode(observed[index]).map(|(row, _)| row));
        }
        for value in 0..values.min(8) {
            let index = lane as usize * slots * values + value;
            columns.push(decode(observed[index]).map(|(_, column)| column));
        }
        let _ = write!(report, "\n      lane {lane:>2}: {rows:?} | {columns:?}");
    }
    report
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(failures) => {
            eprintln!("{failures} device test case(s) failed");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("harness error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize, Box<dyn Error>> {
    let context = cuda_core::CudaContext::new(0)?;
    let owned_stream = context.default_stream();
    let owned_module = kernels::load(&context)?;
    let stream: &CudaStream = &owned_stream;
    let module: &kernels::LoadedModule = &owned_module;

    type Case<'a> = (
        &'static str,
        Box<dyn Fn() -> Result<String, Box<dyn Error>> + 'a>,
    );
    let mut cases: Vec<Case<'_>> = vec![(
        "swizzle round trip",
        Box::new(|| check_swizzle(stream, module)),
    )];
    for shape in Shape::ALL {
        cases.push((
            shape.name(),
            Box::new(move || check_fragment_map(stream, module, shape)),
        ));
    }
    cases.push((
        "stmatrix round trip",
        Box::new(|| check_stmatrix(stream, module)),
    ));

    let mut failures = 0usize;
    for (name, case) in &cases {
        match case() {
            Ok(note) => println!("pass  {name:<24}  {note}"),
            Err(error) => {
                println!("FAIL  {name:<24}  {error}");
                failures += 1;
            }
        }
    }
    println!("{} of {} cases passed", cases.len() - failures, cases.len());
    Ok(failures)
}
