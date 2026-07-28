//! Device tests for ferro-kittens: what the host unit tests cannot prove.
//!
//! The library's `#[cfg(test)]` suites establish that its address and
//! coordinate arithmetic is *self-consistent*. Nothing there can show that
//! [`BaseLdtm`]'s ownership map, the SWIZZLE_128B phase math, or the
//! `stmatrix`/`ldmatrix` movers are the ones the silicon actually uses — those were
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

// Every `#[kernel]` is an `unsafe fn` whose contract is its launch config, and
// each states that in its own doc; a `# Safety` section per entry point would
// be five lines of ceremony saying the same thing.
#![allow(clippy::missing_safety_doc)]

use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::DisjointSlice;
use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

use kittens::global::{GlobalLayout, encode_bf16_panels};
use kittens::ldst::{load_fragment, load_tile, store_fragment, store_tile};
use kittens::mma::{self, MmaShape, mma_abt};
use kittens::reg::{
    BaseLdtm, ColVec, Fragment, FragmentLayout, Mul, RegTile, RegVec, online_rescale,
};
use kittens::shared::{
    Bf16, SharedTile, Swizzle128B, tma_store_commit, tma_store_wait, tma_store_wait_read,
};
use kittens::sync::Semaphore;
use kittens::tmem::{TmemTile, alloc_block, dealloc_block, store_wait};

/// Edge of the square tiles the swizzle and `stmatrix` cases use: 64 bf16
/// columns is exactly one 128-byte swizzle atom per row, so these tiles are a
/// single subtile and their cursor never leaves it. Every case below runs at
/// this width *and* at [`WIDE`], because the narrow path is the one the
/// hardware has always agreed with and the subtile term must leave it alone.
const TILE: usize = 64;
/// Columns of the wide tiles: two stacked 64-column subtiles, so the cursor
/// has to cross a [`SharedTile::SUBTILE_BYTES`] stride mid-row (#25).
const WIDE: usize = 128;
/// Rows of the short wide tile. Deliberately *not* a multiple of the 8-row
/// swizzle period: its second subtile therefore begins mid-period, and the
/// tile comes back in order only if a subtile's swizzle phase follows its
/// absolute 128-byte row in shared memory rather than restarting per subtile.
/// At every shape the library ships (R = 64, 128) the two readings coincide,
/// so this is the only case that can tell them apart.
const SHORT: usize = 4;

/// Rows of the probe accumulator — one full `M128` MMA shape, and the four
/// warps' 32 TMEM rows each.
const ROWS: usize = 128;
/// Columns of the probe accumulator: two `M128_N64` bands side by side, so a
/// warp's drained band is wide enough for the `(32, 128)` layout shape.
const COLUMNS: usize = 128;
/// K of the probe MMA: one swizzle atom of bf16, four chained K=16 chunks.
const DEPTH: usize = 64;

/// The MMA operands and the tiles the swizzle cases move, as the library
/// types. The shape-generic alias is what lets one probe body serve the narrow
/// and wide cases, so a case cannot drift between widths.
type Tile<const R: usize, const C: usize> = SharedTile<Bf16, R, C, Swizzle128B>;
type AOperand = Tile<ROWS, DEPTH>;
type BOperand = Tile<TILE, DEPTH>;
type Accumulator = TmemTile<ROWS, COLUMNS>;
/// The lane probe's staging tile: a warp's 32 rows by the one swizzle atom
/// [`SharedTile::chunk_writer`] accepts, which is what `store_fragment`
/// addresses into.
type LaneStage = SharedTile<Bf16, 32, TILE, Swizzle128B>;

/// Bytes a swizzle case's plan needs: its tile plus a scratch tail for the TMA
/// barrier.
const fn tile_shared<const R: usize, const C: usize>() -> u32 {
    (Tile::<R, C>::BYTES + 32) as u32
}

/// Dynamic shared plan of the fragment probe: the A operand, the two B
/// operands, then a 32-byte scratch tail holding the two mbarriers and the
/// TMEM staging word.
const PROBE_SHARED: usize = AOperand::BYTES + 2 * BOperand::BYTES + 32;
/// The STTM round trip touches no shared tile at all — its whole plan is the
/// TMEM staging word.
const STTM_SHARED: usize = 32;

/// Columns [`kernels::relaunch_probe`] allocates: the whole SM's tensor
/// memory. No two CTAs can hold an allocation on the same SM at once, so
/// every CTA after the first is served out of columns the one before it gave
/// back — a probe taking a quarter of TMEM would be handed a free quarter
/// without the allocator ever having to hand anything over.
const RELAUNCH_COLUMNS: usize = 512;
/// Blocks in that probe's grid: twice the B200's 148 SMs, so the launch
/// cannot fit in one wave at any occupancy.
const RELAUNCH_BLOCKS: u32 = 296;
/// Launches of it in one process. The class of failure this case exists for
/// is invisible to the first launch by construction, so one launch is not a
/// test; the third is free and covers a resource that takes more than one CTA
/// to exhaust.
const RELAUNCHES: usize = 3;
/// How long a launch may run before the harness calls it deadlocked rather
/// than slow. The probe is microseconds of work per CTA.
const HANG_TIMEOUT: Duration = Duration::from_secs(30);
/// How often the watchdog asks. Long enough to cost nothing, short enough
/// that a passing run is not measurably slower for being watched.
const HANG_POLL: Duration = Duration::from_millis(20);

/// A bf16 bit pattern naming a position in a tile of at most 64 rows by 128
/// columns.
///
/// Bit 14 is set so the pattern is a *normal* bf16 — a position encoding down
/// in the subnormal range would be at the mercy of flush-to-zero somewhere in
/// the conversion. Columns take the low seven bits and rows the next six, so
/// the exponent field lands in 0x80..0xbf: never subnormal, never Inf or NaN.
/// Nothing here is ever read as a number: the tests compare bit patterns, so
/// the round trip is exact by construction and a mismatch is always a
/// misplaced element.
const fn cell_bits(row: usize, column: usize) -> u16 {
    0x4000 | ((row as u16) << 7) | column as u16
}

/// [`cell_bits`] as the fp32 a register-side path carries. The low 16 bits are
/// zero, so packing it back to bf16 is exact whatever rounding mode the
/// conversion uses.
const fn cell(row: usize, column: usize) -> f32 {
    f32::from_bits((cell_bits(row, column) as u32) << 16)
}

/// The word a TMA store's destination is seeded with, and what
/// [`kernels::poison_tile`] fills a recycled tile with. [`cell_bits`] always
/// sets bit 14 and never bits 8..14 or 15, so no identity can be either half of
/// this word — an unwritten destination, a row landed at the wrong stride, and
/// a tile read after it was recycled are all distinguishable from a merely
/// misplaced element.
const POISON: u32 = 0xffff_ffff;

/// Decode a [`cell`] back to the position it names, or `None` if the value is
/// not one — how a failing register-side case reports the coordinate the
/// hardware actually delivered.
fn decode_cell(value: f32) -> Option<(usize, usize)> {
    let bits = (value.to_bits() >> 16) as usize;
    let (row, column) = ((bits >> 7) & 0x3f, bits & 0x7f);
    (bits == 0x4000 | (row << 7) | column).then_some((row, column))
}

/// The probe accumulator's value at `(row, column)` — `COLUMNS * row + column`,
/// unique over the whole `[ROWS, COLUMNS]` tile and an exact fp32 integer, so a
/// drained value decodes straight back to the coordinate the hardware gave it.
fn accumulator_value(row: usize, column: usize) -> f32 {
    (COLUMNS * row + column) as f32
}

/// Row statistics per lane in [`kernels::reduction_probe`]'s dump — the slots
/// of a `RegTile<32, COLUMNS, BaseLdtm>`.
const REDUCTION_ROWS: usize = 32 / 8;
/// Column statistics per lane — that tile's values.
const REDUCTION_COLUMNS: usize = COLUMNS / 4;
/// Per-lane stride of that dump: the row sums, the column sums, the wide
/// band's max and the narrow band's sum.
const REDUCTION_STRIDE: usize = REDUCTION_ROWS + REDUCTION_COLUMNS + 2;

/// Which spelling of the online-softmax correction [`kernels::softmax_probe`]
/// compiles. Only the codegen probe reads these; see its doc comment.
const HAND_WRITTEN: u32 = 0;
/// [`online_rescale`], the deliberately fused form.
const FUSED: u32 = 1;
/// [`HAND_WRITTEN`] with the generic `row_map::<Mul>` in place of `scale_rows`.
const ROW_MAP: u32 = 2;

/// How [`kernels::scalar_map_probe`] spells its two scalar multiplies. The
/// pre-#38 spelling: `bin_map::<Mul>(splat(k))`, the scalar widened into a
/// whole second tile because there was no other way to reach a `BinaryOp` with
/// one. This is the number `scale` has to beat.
const SPLATTED: u32 = 0;
/// `RegTile::scale` — the same `Mul` with the operand left in a register.
const SCALED: u32 = 1;
/// Neither: both multiplies written through `set`, mutating in place. #31's
/// shape, and nothing in `kittens` spells it; it is here as the floor.
const IN_PLACE: u32 = 2;

/// How [`kernels::mask_probe`] masks its score band — issue #7. The three
/// compute the same tile, so a difference in the `regcount` table is the
/// spelling and nothing else.
///
/// No mask at all: what the loop costs before masking enters it.
const UNMASKED: u32 = 0;
/// `RegTile::make_causal_at`, the API.
const CAUSAL: u32 = 1;
/// The same mask open-coded against `RegTile::coordinate`, which is what a
/// kernel writes without #7 — and what `flash_forward`'s gap note describes.
const BY_HAND: u32 = 2;

/// Which lane-passing convention [`kernels::lane_probe`] compiles — issue #27.
/// Today's convention: `warp::lane_id()` read once by the caller and threaded
/// into every coordinate-dependent op.
const HOISTED: u32 = 0;
/// One `warp::lane_id()` per *op*, the op reading the register for itself.
const PER_OP: u32 = 1;
/// One `warp::lane_id()` per coordinate *query* — the pessimistic implicit
/// form, where `coordinate` reads `%laneid` once for the row and again for the
/// column instead of once for the pair.
const PER_QUERY: u32 = 2;

/// Where `(warp, lane, slot, value)` lands in a dump of `slots` × `values` per
/// thread. The STTM round trip seeds each register with its own index here, so
/// this doubles as that case's identity: unique across the launch, and an exact
/// fp32 integer well under 2^24.
const fn dump_index(
    warp: usize,
    lane: u32,
    slot: usize,
    value: usize,
    slots: usize,
    values: usize,
) -> usize {
    ((warp * 32 + lane as usize) * slots + slot) * values + value
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// TMA one `[R, C]` bf16 tile into shared memory, then read it back
    /// through [`SharedTile::chunk_writer`] and write it out in *logical*
    /// order — so the host's expectation is simply the buffer it staged.
    ///
    /// This is the swizzle path with nothing else in it: it checks
    /// `swizzle_phase`, the chunk XOR, the stacked-subtile stride and the
    /// tensor map's box geometry against the TMA engine, and needs no MMA.
    /// Launch with `R` threads, one row each.
    #[inline(always)]
    unsafe fn swizzle_probe<const R: usize, const C: usize>(
        source: *const TmaDescriptor,
        out: &mut DisjointSlice<u32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<R, C>::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Tile::<R, C>::BYTES) as *mut Barrier);
            let tid = thread::threadIdx_x();

            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tile.tma_load(source, 0, 0, tma);
                tma.expect_tx(Tile::<R, C>::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            dump_rows(tile, tid as usize, R, out);

            thread::sync_threads();
            if tid == 0 {
                tma.inval();
            }
        }
    }

    /// [`swizzle_probe`] over one subtile — the width the cursor has always
    /// handled.
    #[kernel]
    pub unsafe fn swizzle_roundtrip(source: *const TmaDescriptor, mut out: DisjointSlice<u32>) {
        unsafe { swizzle_probe::<TILE, TILE>(source, &mut out) }
    }

    /// [`swizzle_probe`] over two stacked subtiles.
    #[kernel]
    pub unsafe fn swizzle_roundtrip_wide(
        source: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
    ) {
        unsafe { swizzle_probe::<TILE, WIDE>(source, &mut out) }
    }

    /// [`swizzle_probe`] over two stacked subtiles only [`SHORT`] rows tall, so
    /// the second one starts mid-swizzle-period. Launch with `SHORT` threads.
    #[kernel]
    pub unsafe fn swizzle_roundtrip_short(
        source: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
    ) {
        unsafe { swizzle_probe::<SHORT, WIDE>(source, &mut out) }
    }

    /// [`swizzle_probe`] against a *rank-2* tensor map.
    ///
    /// The 3-D panel map is the only descriptor shape the library could build
    /// before [`GlobalLayout`], and it is also the only one silicon has ever
    /// checked. Same round trip, same expectation, through
    /// [`SharedTile::tma_load_2d`] — so the box geometry and the swizzle are
    /// held to the standard the panel map is held to, at the rank and stride a
    /// GEMM operand actually has. Launch with `R` threads, one row each.
    #[inline(always)]
    unsafe fn tma_2d_probe<const R: usize, const C: usize>(
        source: *const TmaDescriptor,
        out: &mut DisjointSlice<u32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<R, C>::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Tile::<R, C>::BYTES) as *mut Barrier);
            let tid = thread::threadIdx_x();

            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tile.tma_load_2d(source, 0, 0, tma);
                tma.expect_tx(Tile::<R, C>::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            dump_rows(tile, tid as usize, R, out);

            thread::sync_threads();
            if tid == 0 {
                tma.inval();
            }
        }
    }

    /// [`tma_2d_probe`] over one subtile.
    #[kernel]
    pub unsafe fn tma_2d_roundtrip(source: *const TmaDescriptor, mut out: DisjointSlice<u32>) {
        unsafe { tma_2d_probe::<TILE, TILE>(source, &mut out) }
    }

    /// TMA a tile in and then straight back out to *two* different global
    /// buffers — one packed 3-D panel map through [`SharedTile::tma_store`],
    /// one pitched rank-2 map through [`SharedTile::tma_store_2d`] — under a
    /// single committed group.
    ///
    /// The kernel computes nothing: the host stages position identities and
    /// expects them back, so the whole assertion is that the store side puts
    /// each box where its map says. The pitched destination is the sharp half —
    /// its rows are further apart than they are wide and the gaps hold a bit
    /// pattern no identity can be, so a wrong row stride reports *padding* at a
    /// position rather than a plausible neighbour.
    ///
    /// With `RECYCLE`, the tile is overwritten between
    /// [`tma_store_wait_read`] and [`tma_store_wait`]. That is not a variation
    /// of the same claim but a different one: the `read` wait says the engine
    /// has finished *reading* shared memory, so the poison must not reach
    /// either destination — and if that wait did not mean what the API says it
    /// means, both buffers come back full of it. Launch with `R` threads.
    #[inline(always)]
    unsafe fn tma_store_probe<const R: usize, const C: usize, const RECYCLE: bool>(
        source: *const TmaDescriptor,
        packed: *const TmaDescriptor,
        pitched: *const TmaDescriptor,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<R, C>::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Tile::<R, C>::BYTES) as *mut Barrier);
            let tid = thread::threadIdx_x();

            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tile.tma_load(source, 0, 0, tma);
                tma.expect_tx(Tile::<R, C>::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            // Both proxies are the async one — TMA in, TMA out — so the load's
            // barrier is the whole ordering and no fence belongs here.
            if tid == 0 {
                tile.tma_store(packed, 0, 0);
                tile.tma_store_2d(pitched, 0, 0);
                tma_store_commit();
                if RECYCLE {
                    tma_store_wait_read::<0>();
                }
            }
            if RECYCLE {
                // Groups are per thread, so thread 0's wait above is what the
                // other threads are being released past.
                thread::sync_threads();
                poison_tile(tile, tid as usize, R);
                thread::sync_threads();
            }
            if tid == 0 {
                tma_store_wait::<0>();
                tma.inval();
            }
        }
    }

    /// Overwrite a tile with a pattern the identity encoding cannot produce.
    /// Flat over the tile's bytes rather than through the cursor: what the
    /// recycle case needs is that *nothing* of the old tile survives, which is
    /// a statement about every byte and not about the swizzle.
    #[inline(always)]
    unsafe fn poison_tile<const R: usize, const C: usize>(
        tile: Tile<R, C>,
        first: usize,
        stride: usize,
    ) {
        unsafe {
            let words = tile.base() as *mut u32;
            let mut word = first;
            while word < Tile::<R, C>::BYTES / 4 {
                *words.add(word) = POISON;
                word += stride;
            }
        }
    }

    /// [`tma_store_probe`] over one subtile.
    #[kernel]
    pub unsafe fn tma_store_roundtrip(
        source: *const TmaDescriptor,
        packed: *const TmaDescriptor,
        pitched: *const TmaDescriptor,
    ) {
        unsafe { tma_store_probe::<TILE, TILE, false>(source, packed, pitched) }
    }

    /// [`tma_store_probe`] over two stacked subtiles: the second box has to
    /// leave with its leading coordinate lifted by `SUBTILE_COLS`, which is the
    /// store side's whole share of the subtile walk.
    #[kernel]
    pub unsafe fn tma_store_roundtrip_wide(
        source: *const TmaDescriptor,
        packed: *const TmaDescriptor,
        pitched: *const TmaDescriptor,
    ) {
        unsafe { tma_store_probe::<TILE, WIDE, false>(source, packed, pitched) }
    }

    /// [`tma_store_probe`] recycling its tile as soon as the reads are done.
    #[kernel]
    pub unsafe fn tma_store_recycle(
        source: *const TmaDescriptor,
        packed: *const TmaDescriptor,
        pitched: *const TmaDescriptor,
    ) {
        unsafe { tma_store_probe::<TILE, TILE, true>(source, packed, pitched) }
    }

    /// Fill an `[R, C]` shared tile from registers through [`store_fragment`],
    /// then read it back the same way [`swizzle_probe`] does.
    ///
    /// The fragment of each 16x16 block is built *from the ownership map* —
    /// value `(slot, value)` gets `cell(BaseLdtm::row(..), BaseLdtm::column(..))`
    /// — so a correct round trip means the `stmatrix` addressing and the map
    /// agree, and the tile comes out in plain logical order. Launch with one
    /// warp: `store_fragment` is warp-scope and takes its addresses from lanes
    /// 0..16.
    #[inline(always)]
    unsafe fn stmatrix_probe<const R: usize, const C: usize>(out: &mut DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<R, C>::from_raw(smem);
            let chunks = tile.chunk_writer();
            let lane = warp::lane_id();

            let mut row_block = 0usize;
            while row_block < R / 16 {
                let mut column_block = 0usize;
                while column_block < C / 16 {
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

            dump_rows(tile, lane as usize, 32, out);
        }
    }

    /// [`stmatrix_probe`] over one subtile.
    #[kernel]
    pub unsafe fn stmatrix_roundtrip(mut out: DisjointSlice<u32>) {
        unsafe { stmatrix_probe::<TILE, TILE>(&mut out) }
    }

    /// [`stmatrix_probe`] over two stacked subtiles: the blocks at columns
    /// 64.. are the ones whose addresses cross the subtile stride.
    #[kernel]
    pub unsafe fn stmatrix_roundtrip_wide(mut out: DisjointSlice<u32>) {
        unsafe { stmatrix_probe::<TILE, WIDE>(&mut out) }
    }

    /// TMA the same tile [`swizzle_probe`] stages, then read every `[16, 16]`
    /// block of it into registers through [`load_fragment`] and dump by thread
    /// coordinate.
    ///
    /// **What this proves.** The tile's contents are placed by the TMA engine,
    /// which the `swizzle round trip` case checks independently, and the values
    /// are position identities the register path never computes. So a value
    /// arriving at the register the host expects means `ldmatrix`'s ownership
    /// map *is* [`BaseLdtm`] and `load_fragment`'s addressing is the one the
    /// hardware uses — not merely that the load inverts the store. Nothing here
    /// touches `stmatrix`; the two directions are checked against silicon
    /// separately, and only the shared [`kittens::ldst::fragment_address`]
    /// derivation is common to both.
    ///
    /// Launch with one warp: `load_fragment` is warp-scope and takes its
    /// addresses from lanes 0..16.
    #[inline(always)]
    unsafe fn ldmatrix_probe<const R: usize, const C: usize>(
        source: *const TmaDescriptor,
        out: &mut DisjointSlice<f32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<R, C>::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Tile::<R, C>::BYTES) as *mut Barrier);
            let lane = warp::lane_id();

            if lane == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if lane == 0 {
                tile.tma_load(source, 0, 0, tma);
                tma.expect_tx(Tile::<R, C>::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let chunks = tile.chunk_writer();
            let mut row_block = 0usize;
            while row_block < R / 16 {
                let mut column_block = 0usize;
                while column_block < C / 16 {
                    let fragment = load_fragment::<Bf16>(
                        chunks,
                        (16 * row_block) as u32,
                        (16 * column_block) as u32,
                        lane,
                    );
                    // The dump is one band per block rather than per warp, so
                    // the block index takes the warp's place in `dump_index`.
                    dump_band(
                        fragment,
                        (row_block * (C / 16) + column_block) as u32,
                        lane,
                        out,
                    );
                    column_block += 1;
                }
                row_block += 1;
            }

            thread::sync_threads();
            if lane == 0 {
                tma.inval();
            }
        }
    }

    /// [`ldmatrix_probe`] over one subtile.
    #[kernel]
    pub unsafe fn ldmatrix_map(source: *const TmaDescriptor, mut out: DisjointSlice<f32>) {
        unsafe { ldmatrix_probe::<TILE, TILE>(source, &mut out) }
    }

    /// [`ldmatrix_probe`] over two stacked subtiles.
    #[kernel]
    pub unsafe fn ldmatrix_map_wide(source: *const TmaDescriptor, mut out: DisjointSlice<f32>) {
        unsafe { ldmatrix_probe::<TILE, WIDE>(source, &mut out) }
    }

    /// The composed shared movers, both directions in one trip: `load_tile`
    /// reads a `[32, WIDE]` band out of the TMA-staged tile's rows `0..32`,
    /// and `store_tile` writes that same band into its rows `32..64`.
    ///
    /// The band is 128 columns, so both directions cross the stacked-subtile
    /// stride, and the store's `row` base is non-zero — the two places a
    /// composition loop can be wrong that a single `[16, 16]` block cannot
    /// show. Rows `32..64` must come back as rows `0..32`; the staged values
    /// are position identities, so a block landing in the wrong place names
    /// the row and column it came from.
    ///
    /// The values round-trip through bf16 exactly: [`cell`] has sixteen zero
    /// mantissa bits, so packing it back is the identity whatever the
    /// conversion rounds to.
    ///
    /// Launch with one warp: both movers are warp-scope.
    #[kernel]
    pub unsafe fn band_roundtrip(source: *const TmaDescriptor, mut out: DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<TILE, WIDE>::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Tile::<TILE, WIDE>::BYTES) as *mut Barrier);
            let lane = warp::lane_id();

            if lane == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if lane == 0 {
                tile.tma_load(source, 0, 0, tma);
                tma.expect_tx(Tile::<TILE, WIDE>::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let chunks = tile.chunk_writer();
            let band: RegTile<32, WIDE, BaseLdtm> = load_tile(chunks, 0, 0, lane);
            store_tile(chunks, 32, 0, lane, band);
            thread::sync_threads();

            dump_rows(tile, lane as usize, 32, &mut out);

            thread::sync_threads();
            if lane == 0 {
                tma.inval();
            }
        }
    }

    /// Read `tile`'s rows `first`, `first + stride`, … back through the
    /// swizzle and write them to `out` in logical order. The chunk count is
    /// the cursor's own, so the loop covers every stacked subtile without
    /// restating the tile's shape.
    #[inline(always)]
    unsafe fn dump_rows<const R: usize, const C: usize>(
        tile: Tile<R, C>,
        first: usize,
        stride: usize,
        out: &mut DisjointSlice<u32>,
    ) {
        unsafe {
            let chunks = tile.chunk_writer();
            let mut row = first;
            while row < R {
                let mut chunk = 0usize;
                while chunk < chunks.chunks() {
                    let words = chunks.at(row, chunk) as *const u32;
                    let mut word = 0usize;
                    while word < 4 {
                        *out.get_unchecked_mut(row * (C / 2) + chunk * 4 + word) = *words.add(word);
                        word += 1;
                    }
                    chunk += 1;
                }
                row += stride;
            }
        }
    }

    /// Write a warp's band to `out` indexed by `(warp, lane, slot, value)`
    /// alone, so the host is what applies the ownership map.
    #[inline(always)]
    unsafe fn dump_band<const M: usize, const N: usize>(
        tile: RegTile<M, N, BaseLdtm>,
        warp_id: u32,
        lane: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<M, N>,
    {
        unsafe {
            let slots = RegTile::<M, N, BaseLdtm>::SLOTS;
            let values = RegTile::<M, N, BaseLdtm>::VALUES;
            let mut slot = 0usize;
            while slot < slots {
                let mut value = 0usize;
                while value < values {
                    let index = dump_index(warp_id as usize, lane, slot, value, slots, values);
                    *out.get_unchecked_mut(index) = tile.get(slot, value);
                    value += 1;
                }
                slot += 1;
            }
        }
    }

    /// Round-trip a `[32, COLUMNS]` band of registers through TMEM: seed,
    /// `TmemTile::store_tile`, `store_wait`, `TmemTile::tile`, dump.
    ///
    /// Each thread's value at `(slot, value)` is `dump_index` of its own
    /// `(warp, lane, slot, value)`, so every register in the launch carries a
    /// distinct exact integer naming the thread coordinate that wrote it, and
    /// the host's expectation is just "index `i` came back as `i`".
    ///
    /// **What this proves and what it does not.** On its own a round trip only
    /// shows STTM is the exact inverse of LDTM — a store and a load sharing the
    /// same *wrong* map would pass it just as well. It pins STTM to `BaseLdtm`
    /// only in composition with the `fragment map` cases, which established
    /// LDTM's map against silicon; the `sttm restage` case is what tests the
    /// store's addressing against an independently-known column.
    ///
    /// Launch with `ROWS` threads: four warps, 32 TMEM rows each.
    #[kernel]
    pub unsafe fn sttm_roundtrip(mut out: DisjointSlice<f32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tmem = alloc_block(smem as *mut u32, COLUMNS as u32);
            let band = Accumulator::from_raw(tmem);
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();

            let mut tile = RegTile::<32, COLUMNS, BaseLdtm>::zero();
            let slots = RegTile::<32, COLUMNS, BaseLdtm>::SLOTS;
            let values = RegTile::<32, COLUMNS, BaseLdtm>::VALUES;
            let mut slot = 0usize;
            while slot < slots {
                let mut value = 0usize;
                while value < values {
                    let index = dump_index(warp_id as usize, lane, slot, value, slots, values);
                    tile.set(slot, value, index as f32);
                    value += 1;
                }
                slot += 1;
            }

            band.store_tile(32 * warp_id, 0, tile);
            store_wait();
            dump_band(
                band.tile::<32, COLUMNS>(32 * warp_id, 0),
                warp_id,
                lane,
                &mut out,
            );

            thread::sync_threads();
            dealloc_block(tmem, COLUMNS as u32);
        }
    }

    /// One CTA's whole TMEM lifecycle, over a grid that does not fit in one
    /// wave and a process that launches it more than once.
    ///
    /// Every other case here launches once over one wave, which cannot see a
    /// resource acquired at CTA entry and released at exit: nothing is
    /// scheduled behind the CTA that leaked it. This one is the standing
    /// guard for that whole class — "fine once, broken twice" — and TMEM is
    /// merely the first resource with the shape. Each CTA takes the SM's
    /// entire tensor memory, so every CTA but the first on an SM is served
    /// out of columns its predecessor gave back, in one launch and across
    /// launches.
    ///
    /// It is a guard and not a reproduction: #46's leaked allocation permit
    /// is *not* what it catches, because on a B200 that leak has no
    /// observable effect (see `tmem::alloc_block`). What it does catch is any
    /// leak of the columns themselves, an unbalanced allocator collective,
    /// and whatever the next entry-acquired resource turns out to be.
    ///
    /// The round trip through TMEM is [`sttm_roundtrip`]'s and proves nothing
    /// new about the store map; it is here so a CTA that finished is
    /// distinguishable from one that was never scheduled.
    ///
    /// Launch with `ROWS` threads over [`RELAUNCH_BLOCKS`] blocks.
    #[kernel]
    pub unsafe fn relaunch_probe(mut out: DisjointSlice<f32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tmem = alloc_block(smem as *mut u32, RELAUNCH_COLUMNS as u32);
            let allocation = TmemTile::<ROWS, RELAUNCH_COLUMNS>::from_raw(tmem);
            let warp_id = warp::warp_id();
            // Each thread's registers carry their own index in the dump, so
            // the host's expectation is "index `i` came back as `i`" and a
            // CTA's identity is in the values it writes.
            let base = (thread::blockIdx_x() as usize * ROWS + thread::threadIdx_x() as usize)
                * Fragment::SLOTS
                * Fragment::VALUES;

            let mut fragment = Fragment::zero();
            let mut slot = 0usize;
            while slot < Fragment::SLOTS {
                let mut value = 0usize;
                while value < Fragment::VALUES {
                    fragment.set(slot, value, (base + slot * Fragment::VALUES + value) as f32);
                    value += 1;
                }
                slot += 1;
            }

            // The four warps take a 16-row block each, so the whole CTA is in
            // the allocation it opened.
            allocation.store_fragment_tile(16 * warp_id, 0, fragment);
            store_wait();
            let read = allocation.fragment_tile(16 * warp_id, 0);

            let mut slot = 0usize;
            while slot < Fragment::SLOTS {
                let mut value = 0usize;
                while value < Fragment::VALUES {
                    *out.get_unchecked_mut(base + slot * Fragment::VALUES + value) =
                        read.get(slot, value);
                    value += 1;
                }
                slot += 1;
            }

            thread::sync_threads();
            dealloc_block(tmem, RELAUNCH_COLUMNS as u32);
        }
    }

    /// The fragment ownership probe, shared by every layout shape.
    ///
    /// `D = A·Bᵀ` over two `M128_N64` bands, seeded (by the host) so that
    /// `D[row, column] == COLUMNS * row + column` exactly. Each of the four
    /// warps drains its own 32 TMEM rows into a `RegTile<M, N, BaseLdtm>`.
    ///
    /// With `RESTAGE` the drained band takes a detour: it is written back
    /// through `TmemTile::store_tile` to the columns `COLUMNS..2*COLUMNS` of a
    /// double-width allocation and drained again from there. The values are
    /// absolute position identities the store path never computes, and the
    /// re-drain reads at a column the `fragment map` cases already validated,
    /// so a column-offset bug in STTM shows up as a value from the wrong
    /// column — which the plain round trip, holding `column` fixed, cannot see.
    ///
    /// The dump is indexed by `(warp, lane, slot, value)` alone. The host
    /// applies `RegTile::coordinate` to it, so this kernel never encodes what
    /// the answer should be.
    #[inline(always)]
    unsafe fn fragment_probe<const M: usize, const N: usize, const RESTAGE: bool>(
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
            let columns = (if RESTAGE { 2 * COLUMNS } else { COLUMNS }) as u32;
            let tmem = alloc_block(tmem_slot, columns);
            let accumulator = Accumulator::from_raw(tmem);

            if leader {
                a.tma_load(a_map, 0, 0, tma);
                b_low.tma_load(b_map, 0, 0, tma);
                b_high.tma_load(b_map, 0, 1, tma);
                tma.expect_tx((AOperand::BYTES + 2 * BOperand::BYTES) as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let shape = MmaShape::M128_N64;
            if leader {
                mma_abt(accumulator.raw(), a, b_low, shape, false);
                mma_abt(
                    accumulator.columns_right(TILE as u32).raw(),
                    a,
                    b_high,
                    shape,
                    false,
                );
                mma::commit(mma_done);
            }
            mma_done.wait(0);
            thread::sync_threads();

            let mut tile = accumulator.tile::<M, N>(32 * warp_id, 0);
            if RESTAGE {
                let staged = accumulator.columns_right(COLUMNS as u32);
                staged.store_tile(32 * warp_id, 0, tile);
                store_wait();
                tile = staged.tile::<M, N>(32 * warp_id, 0);
            }
            dump_band(tile, warp_id, lane, out);

            thread::sync_threads();
            dealloc_block(tmem, columns);
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
        unsafe { fragment_probe::<16, 16, false>(a_map, b_map, &mut out) }
    }

    /// [`fragment_probe`] over a warp's full 32 rows by two column blocks.
    #[kernel]
    pub unsafe fn fragment_map_32x32(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<32, 32, false>(a_map, b_map, &mut out) }
    }

    /// [`fragment_probe`] at the flash output accumulator's shape — a warp's
    /// 32 TMEM rows by 128 columns, four slots of 32 values per thread.
    #[kernel]
    pub unsafe fn fragment_map_32x128(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<32, 128, false>(a_map, b_map, &mut out) }
    }

    /// [`fragment_probe`] at the same shape, with the drained band restaged
    /// through STTM into a second column band before it is read back.
    #[kernel]
    pub unsafe fn fragment_restage_32x128(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { fragment_probe::<32, 128, true>(a_map, b_map, &mut out) }
    }

    /// Reductions against silicon: the `shuffle_xor` butterflies, which are
    /// the half of `row_reduce`/`col_reduce`/`tile_reduce` no host test can
    /// reach. A wrong mask there returns a plausible number rather than
    /// failing, so this is the case that has to run.
    ///
    /// The seeding is [`fragment_probe`]'s, unchanged: `D = A·Bᵀ` over two
    /// `M128_N64` bands leaves `D[row, column] == COLUMNS * row + column`
    /// exactly, hardware-delivered. Each warp then reduces its own 32 TMEM
    /// rows and dumps the statistics by `(warp, lane)`; the host applies
    /// [`BaseLdtm`]'s map to say what each of them must be, so nothing here
    /// encodes an expected answer.
    ///
    /// The column sums are the case with no precedent in this crate: they
    /// fold across the 8 lanes sharing `lane % 4` — `shuffle_xor` masks 4, 8
    /// and 16 — a butterfly no kernel here has ever run. `tile_sum` is taken
    /// over a `[32, 32]` band rather than the wide one only so every partial
    /// sum stays an exact fp32 integer: the wide band's total is 58.7M, past
    /// 2^24, where an equality assertion would be measuring fp32 rounding
    /// instead of the reduction.
    ///
    /// Launch with `ROWS` threads, as [`fragment_probe`] is.
    #[kernel]
    pub unsafe fn reduction_probe(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let a = AOperand::from_raw(smem);
            let b_low = BOperand::from_raw(smem.add(AOperand::BYTES));
            let b_high = BOperand::from_raw(smem.add(AOperand::BYTES + BOperand::BYTES));
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

            let shape = MmaShape::M128_N64;
            if leader {
                mma_abt(accumulator.raw(), a, b_low, shape, false);
                mma_abt(
                    accumulator.columns_right(TILE as u32).raw(),
                    a,
                    b_high,
                    shape,
                    false,
                );
                mma::commit(mma_done);
            }
            mma_done.wait(0);
            thread::sync_threads();

            let wide = accumulator.tile::<32, COLUMNS>(32 * warp_id, 0);
            let narrow = accumulator.tile::<32, 32>(32 * warp_id, 0);
            let rows = wide.row_sum();
            let columns = wide.col_sum();

            let base = (warp_id as usize * 32 + lane as usize) * REDUCTION_STRIDE;
            let mut slot = 0usize;
            while slot < REDUCTION_ROWS {
                *out.get_unchecked_mut(base + slot) = rows.get(slot);
                slot += 1;
            }
            let mut value = 0usize;
            while value < REDUCTION_COLUMNS {
                *out.get_unchecked_mut(base + REDUCTION_ROWS + value) = columns.get(value);
                value += 1;
            }
            *out.get_unchecked_mut(base + REDUCTION_ROWS + REDUCTION_COLUMNS) = wide.tile_max();
            *out.get_unchecked_mut(base + REDUCTION_ROWS + REDUCTION_COLUMNS + 1) =
                narrow.tile_sum();

            thread::sync_threads();
            dealloc_block(tmem, COLUMNS as u32);
            if leader {
                tma.inval();
                mma_done.inval();
            }
        }
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

                let row_max = block.row_max();
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
                running_sum.add_assign(probabilities.row_sum());
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

    /// **A codegen probe, not a test.** The scalar-operand half of
    /// [`softmax_probe`]'s question (#38), in the flash inner loop's shape: a
    /// `[32, N]` block per step, scaled and folded into an accumulator that is
    /// itself rescaled and stays live across the runtime loop.
    ///
    /// Both scalar multiplies are spelled the same way in a given
    /// instantiation, and all three instantiations compute the same tile, so a
    /// difference in the table is the spelling and nothing else:
    ///
    /// - [`SPLATTED`]: `bin_map::<Mul>(splat(k))`, the pre-#38 spelling — the
    ///   scalar widened into a whole second tile
    /// - [`SCALED`]: `scale(k)`, the same `Mul` with the operand in a register
    /// - [`IN_PLACE`]: neither, both multiplies written through `set`
    ///
    /// The first pair is #38's own question. The third is #31's, and is here
    /// because a by-value map at this width is exactly what #31 was filed
    /// about.
    #[inline(always)]
    unsafe fn scalar_map_probe<const N: usize, const FORM: u32>(
        scores: &[f32],
        steps: u32,
        k: f32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<32, N>,
    {
        unsafe {
            let slots = RegTile::<32, N, BaseLdtm>::SLOTS;
            let values = RegTile::<32, N, BaseLdtm>::VALUES;
            let lane = warp::lane_id();

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

                if FORM == SCALED {
                    block = block.scale(k);
                    out_acc = out_acc.scale(k);
                } else if FORM == SPLATTED {
                    let widened = RegTile::<32, N, BaseLdtm>::splat(k);
                    block = block.bin_map::<Mul>(widened);
                    out_acc = out_acc.bin_map::<Mul>(widened);
                } else {
                    // Two passes, not one fused pass: the by-value forms above
                    // are two maps, and a floor that fuses them would be
                    // measuring the fusion instead of the operand.
                    let mut slot = 0usize;
                    while slot < slots {
                        let mut value = 0usize;
                        while value < values {
                            block.set(slot, value, block.get(slot, value) * k);
                            value += 1;
                        }
                        slot += 1;
                    }
                    let mut slot = 0usize;
                    while slot < slots {
                        let mut value = 0usize;
                        while value < values {
                            out_acc.set(slot, value, out_acc.get(slot, value) * k);
                            value += 1;
                        }
                        slot += 1;
                    }
                }

                out_acc = out_acc.add(block);
                step += 1;
            }

            let base = lane as usize * slots * values;
            let mut slot = 0usize;
            while slot < slots {
                let mut value = 0usize;
                while value < values {
                    *out.get_unchecked_mut(base + slot * values + value) = out_acc.get(slot, value);
                    value += 1;
                }
                slot += 1;
            }
        }
    }

    /// [`scalar_map_probe`] at a 32-wide block, the pre-#38 splatted operand.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_splatted(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, SPLATTED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at a 32-wide block, `RegTile::scale`.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_scaled(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, SCALED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at a 32-wide block, the in-place floor.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_in_place(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, IN_PLACE>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] with the pre-#38 splatted operand at the flash
    /// accumulator's 128 columns — 32 values a thread, the width
    /// `row_map::<Mul>` failed at.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_splatted(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, SPLATTED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at 128 wide, `RegTile::scale`.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_scaled(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, SCALED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at 128 wide, the in-place floor.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_in_place(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, IN_PLACE>(scores, steps, k, &mut out) }
    }

    /// **A codegen probe, not a test.** Masking is pure lane-local coordinate
    /// arithmetic and is checked exhaustively on the host, across all 32
    /// lanes, in `reg.rs`; what only silicon's compiler can answer is what it
    /// costs, and #7's op sits in flash's innermost loop.
    ///
    /// The shape is that loop: a `[32, N]` score band per step, masked, then
    /// exponentiated and folded into an accumulator that stays live across a
    /// runtime-bounded loop. `FORM` picks [`UNMASKED`], [`CAUSAL`] or
    /// [`BY_HAND`] and nothing else changes. The diagonal moves with the step
    /// so it cannot be folded into the coordinate map.
    ///
    /// **What it measured** (`regcount`, sm_100a, no spills anywhere):
    ///
    /// ```text
    ///                        regs   stack
    ///  32 unmasked             31     256
    ///  32 causal               32     256
    ///  32 by hand              32     256
    /// 128 unmasked            157    1536
    /// 128 causal               71    1536
    /// 128 by hand              71    1536
    /// ```
    ///
    /// The claim this probe exists for is the middle pair against the last:
    /// `make_causal_at` and the loop it replaces compile to the same register
    /// count at both widths, so the op costs nothing over the index math it
    /// deletes. Against no mask at all it is one register at 32 — consistent
    /// with #38's peak-liveness reading, since `mask` takes `&mut self` and
    /// adds one `i32` per slot rather than a second band.
    ///
    /// [`UNMASKED`] is *not* a lower bound at 128, and the 86-register gap is
    /// the probe rather than the op: with no mask pass between them the load
    /// loop and the exponential fuse into one wide batch and ptxas holds more
    /// of the band live at once. Same reading as #27's — where the values are
    /// defined, not how many there are.
    #[inline(always)]
    unsafe fn mask_probe<const N: usize, const FORM: u32>(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<32, N>,
    {
        unsafe {
            let slots = RegTile::<32, N, BaseLdtm>::SLOTS;
            let values = RegTile::<32, N, BaseLdtm>::VALUES;
            let lane = warp::lane_id();

            // A row vector and not a second tile, for `lane_probe`'s reason: a
            // `[32, 128]` accumulator beside the score band puts the probe on
            // the 255-register ceiling, where every form reads alike by force
            // and a null result would mean nothing.
            let mut acc = RegVec::<32, BaseLdtm>::splat(0.0);
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

                let key_base = N as u32 * step;
                if FORM == CAUSAL {
                    block.make_causal_at(lane, query_base, key_base, f32::NEG_INFINITY);
                } else if FORM == BY_HAND {
                    let diagonal = query_base as i32 - key_base as i32;
                    let mut slot = 0usize;
                    while slot < slots {
                        let mut value = 0usize;
                        while value < values {
                            let (row, column) =
                                RegTile::<32, N, BaseLdtm>::coordinate(lane, slot, value);
                            if column as i32 - row as i32 > diagonal {
                                block.set(slot, value, f32::NEG_INFINITY);
                            }
                            value += 1;
                        }
                        slot += 1;
                    }
                }

                acc.add_assign(block.exp2().row_sum());
                step += 1;
            }

            let base = lane as usize * slots;
            let mut slot = 0usize;
            while slot < slots {
                *out.get_unchecked_mut(base + slot) = acc.get(slot);
                slot += 1;
            }
        }
    }

    /// [`mask_probe`] at a 32-wide band, no mask — the floor.
    #[kernel]
    pub unsafe fn mask_probe_32_unmasked(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<32, UNMASKED>(scores, steps, query_base, &mut out) }
    }

    /// [`mask_probe`] at 32 wide, `make_causal_at`.
    #[kernel]
    pub unsafe fn mask_probe_32_causal(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<32, CAUSAL>(scores, steps, query_base, &mut out) }
    }

    /// [`mask_probe`] at 32 wide, the open-coded mask.
    #[kernel]
    pub unsafe fn mask_probe_32_by_hand(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<32, BY_HAND>(scores, steps, query_base, &mut out) }
    }

    /// [`mask_probe`] at the flash accumulator's 128 columns, no mask.
    #[kernel]
    pub unsafe fn mask_probe_128_unmasked(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<128, UNMASKED>(scores, steps, query_base, &mut out) }
    }

    /// [`mask_probe`] at 128 wide, `make_causal_at`.
    #[kernel]
    pub unsafe fn mask_probe_128_causal(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<128, CAUSAL>(scores, steps, query_base, &mut out) }
    }

    /// [`mask_probe`] at 128 wide, the open-coded mask.
    #[kernel]
    pub unsafe fn mask_probe_128_by_hand(
        scores: &[f32],
        steps: u32,
        query_base: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { mask_probe::<128, BY_HAND>(scores, steps, query_base, &mut out) }
    }

    /// The lane an op sees under `CONVENTION`: the caller's already-`hoisted`
    /// value under [`HOISTED`], and otherwise a fresh `%laneid` read — which is
    /// the whole of the question in issue #27. `hoisted` is dead in the
    /// implicit forms, so they carry no lane the ops did not ask for.
    #[inline(always)]
    fn op_lane<const CONVENTION: u32>(hoisted: u32) -> u32 {
        if CONVENTION == HOISTED {
            hoisted
        } else {
            warp::lane_id()
        }
    }

    /// **A codegen probe, not a test.** Like [`softmax_probe`] it is launched
    /// by nothing and checks nothing; it exists so `regcount` can price the
    /// lane-passing convention, and it stays as the regression guard on that
    /// price.
    ///
    /// The shape is a kernel that leans on *every* lane-taking op in one basic
    /// block, which is where hoisting would pay off if it pays off anywhere:
    /// per step, a `coordinate`-addressed load of a `[32, N]` block, a
    /// `RegVec::row` vector into `sub_row`, a `ColVec::column` vector into
    /// `add_col`, a second `coordinate` pass for a causal mask, and a
    /// `store_fragment` of every `[16, 16]` block that fits the staging tile.
    /// `CONVENTION` picks how the lane reaches those ops and nothing else
    /// changes, so the forms are comparable line for line:
    ///
    /// - [`HOISTED`]: one `warp::lane_id()`, threaded through — today's rule
    /// - [`PER_OP`]: each op reads `%laneid` itself
    /// - [`PER_QUERY`]: each row/column *query* reads it, the pessimistic bound
    ///
    /// `steps` is a runtime bound so the block stays live across iterations
    /// rather than unrolling into one straight-line region.
    ///
    /// **What it measured** (`regcount`, sm_100a, no spills anywhere):
    ///
    /// ```text
    ///                    regs   stack
    /// 128 hoisted         198    2816
    /// 128 per_op          168    2816
    /// 128 per_query       168    2816
    ///  32 hoisted          72     256
    ///  32 per_op           40     512
    ///  32 per_query        40     512
    /// ```
    ///
    /// [`PER_OP`] and [`PER_QUERY`] agree on every column at both widths, and
    /// the emitted PTX holds exactly *one* `%laneid` move in all six kernels —
    /// so the count of `warp::lane_id()` calls is free, and reading the
    /// register inside an op costs nothing over receiving it. What is left is
    /// where the value is *defined*: hoisting it into the entry block makes the
    /// whole coordinate map loop-invariant, which ptxas pays for in live
    /// registers (+30 at 128, on a PTX body 5 instructions from identical) and
    /// is repaid in instructions at 32 (517 against 643) and half the local
    /// traffic. That is an allocator trade, not a property of the convention;
    /// this probe exists so it stays visible if the backend's `lane_id`
    /// lowering ever stops being a CSE-able pure read.
    #[inline(always)]
    unsafe fn lane_probe<const N: usize, const CONVENTION: u32>(
        scores: &[f32],
        steps: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<32, N>,
    {
        unsafe {
            let slots = RegTile::<32, N, BaseLdtm>::SLOTS;
            let values = RegTile::<32, N, BaseLdtm>::VALUES;
            // Column blocks of the tile that fit the one-subtile staging tile.
            let column_blocks = if N < TILE { N / 16 } else { TILE / 16 };
            let stage = LaneStage::from_raw(DynamicSharedArray::<u8, 128>::get_raw());
            let chunks = stage.chunk_writer();
            let hoisted = if CONVENTION == HOISTED {
                warp::lane_id()
            } else {
                0
            };

            // A row vector, not a second tile: the accumulator is only here to
            // keep the loop live, and a `[32, 128]` one puts the probe on the
            // 255-register ceiling where every variant looks alike by force.
            let mut acc = RegVec::<32, BaseLdtm>::splat(0.0);
            let mut step = 0u32;
            while step < steps {
                let mut block = RegTile::<32, N, BaseLdtm>::zero();
                let mut slot = 0usize;
                while slot < slots {
                    let mut value = 0usize;
                    while value < values {
                        let (row, column) = if CONVENTION == PER_QUERY {
                            (
                                RegVec::<32, BaseLdtm>::row(op_lane::<CONVENTION>(hoisted), slot),
                                ColVec::<N, BaseLdtm>::column(
                                    op_lane::<CONVENTION>(hoisted),
                                    value,
                                ),
                            )
                        } else {
                            RegTile::<32, N, BaseLdtm>::coordinate(
                                op_lane::<CONVENTION>(hoisted),
                                slot,
                                value,
                            )
                        };
                        let index = step as usize * 32 * N + row as usize * N + column as usize;
                        block.set(slot, value, *scores.get_unchecked(index));
                        value += 1;
                    }
                    slot += 1;
                }

                let mut rows = RegVec::<32, BaseLdtm>::splat(0.0);
                let mut slot = 0usize;
                while slot < slots {
                    let row = RegVec::<32, BaseLdtm>::row(op_lane::<CONVENTION>(hoisted), slot);
                    rows.set(slot, row as f32);
                    slot += 1;
                }
                let mut cols = ColVec::<N, BaseLdtm>::splat(0.0);
                let mut value = 0usize;
                while value < values {
                    let column =
                        ColVec::<N, BaseLdtm>::column(op_lane::<CONVENTION>(hoisted), value);
                    cols.set(value, column as f32);
                    value += 1;
                }
                let mut block = block.sub_row(rows).add_col(cols);

                let mut slot = 0usize;
                while slot < slots {
                    let mut value = 0usize;
                    while value < values {
                        let (row, column) = if CONVENTION == PER_QUERY {
                            (
                                RegVec::<32, BaseLdtm>::row(op_lane::<CONVENTION>(hoisted), slot),
                                ColVec::<N, BaseLdtm>::column(
                                    op_lane::<CONVENTION>(hoisted),
                                    value,
                                ),
                            )
                        } else {
                            RegTile::<32, N, BaseLdtm>::coordinate(
                                op_lane::<CONVENTION>(hoisted),
                                slot,
                                value,
                            )
                        };
                        if column > row + 16 * step {
                            block.set(slot, value, -1.0e30);
                        }
                        value += 1;
                    }
                    slot += 1;
                }

                let mut row_block = 0usize;
                while row_block < 2 {
                    let mut column_block = 0usize;
                    while column_block < column_blocks {
                        let mut fragment = Fragment::zero();
                        let mut slot = 0usize;
                        while slot < Fragment::SLOTS {
                            let mut value = 0usize;
                            while value < Fragment::VALUES {
                                let x = block.get(2 * row_block + slot, 4 * column_block + value);
                                fragment.set(slot, value, x);
                                value += 1;
                            }
                            slot += 1;
                        }
                        store_fragment::<Bf16>(
                            chunks,
                            (16 * row_block) as u32,
                            (16 * column_block) as u32,
                            op_lane::<CONVENTION>(hoisted),
                            fragment,
                        );
                        column_block += 1;
                    }
                    row_block += 1;
                }

                let probabilities = block.exp2();
                let mut slot = 0usize;
                while slot < slots {
                    let mut lane_sum = acc.get(slot);
                    let mut value = 0usize;
                    while value < values {
                        lane_sum += probabilities.get(slot, value);
                        value += 1;
                    }
                    acc.set(slot, lane_sum);
                    slot += 1;
                }
                step += 1;
            }

            // The accumulator has to reach memory or the loop above is dead and
            // the register counts describe nothing.
            let base = op_lane::<CONVENTION>(hoisted) as usize * slots;
            let mut slot = 0usize;
            while slot < slots {
                *out.get_unchecked_mut(base + slot) = acc.get(slot);
                slot += 1;
            }
        }
    }

    /// [`lane_probe`] at 32 columns, today's hoisted lane. One line per
    /// instantiation: a generic body compiles nothing on its own.
    #[kernel]
    pub unsafe fn lane_probe_32_hoisted(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { lane_probe::<32, HOISTED>(scores, steps, &mut out) }
    }

    /// [`lane_probe`] at 32 columns, one `%laneid` read per op.
    #[kernel]
    pub unsafe fn lane_probe_32_per_op(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { lane_probe::<32, PER_OP>(scores, steps, &mut out) }
    }

    /// [`lane_probe`] at 32 columns, one `%laneid` read per coordinate query.
    #[kernel]
    pub unsafe fn lane_probe_32_per_query(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { lane_probe::<32, PER_QUERY>(scores, steps, &mut out) }
    }

    /// [`lane_probe`] at the flash accumulator's 128 columns, today's hoisted
    /// lane. The 32-wide probes did not move under a change worth 87
    /// registers, so this width is the one that decides anything.
    #[kernel]
    pub unsafe fn lane_probe_128_hoisted(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { lane_probe::<128, HOISTED>(scores, steps, &mut out) }
    }

    /// [`lane_probe`] at 128 columns, one `%laneid` read per op.
    #[kernel]
    pub unsafe fn lane_probe_128_per_op(scores: &[f32], steps: u32, mut out: DisjointSlice<f32>) {
        unsafe { lane_probe::<128, PER_OP>(scores, steps, &mut out) }
    }

    /// [`lane_probe`] at 128 columns, one `%laneid` read per coordinate query.
    #[kernel]
    pub unsafe fn lane_probe_128_per_query(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { lane_probe::<128, PER_QUERY>(scores, steps, &mut out) }
    }
}

/// The layout shapes with a [`FragmentLayout`] impl, one line each — adding a
/// shape here plus a one-line kernel above is the whole cost of covering it.
#[derive(Clone, Copy)]
enum Shape {
    F16x16,
    F32x32,
    F32x128,
    /// `F32x128` with the band restaged through STTM into a second column
    /// band — the same expectation, so the same checker.
    Restaged32x128,
}

impl Shape {
    const ALL: [Shape; 4] = [
        Shape::F16x16,
        Shape::F32x32,
        Shape::F32x128,
        Shape::Restaged32x128,
    ];

    /// Logical rows and columns of one warp's drained band.
    fn dimensions(self) -> (usize, usize) {
        match self {
            Shape::F16x16 => (16, 16),
            Shape::F32x32 => (32, 32),
            Shape::F32x128 | Shape::Restaged32x128 => (32, 128),
        }
    }

    /// The map under test, as the library computes it: which logical
    /// `(row, column)` of the band `lane`'s `(slot, value)` holds.
    fn coordinate(self, lane: u32, slot: usize, value: usize) -> (usize, usize) {
        let (row, column) = match self {
            Shape::F16x16 => RegTile::<16, 16, BaseLdtm>::coordinate(lane, slot, value),
            Shape::F32x32 => RegTile::<32, 32, BaseLdtm>::coordinate(lane, slot, value),
            Shape::F32x128 | Shape::Restaged32x128 => {
                RegTile::<32, 128, BaseLdtm>::coordinate(lane, slot, value)
            }
        };
        (row as usize, column as usize)
    }

    fn name(self) -> &'static str {
        match self {
            Shape::F16x16 => "fragment map (16, 16)",
            Shape::F32x32 => "fragment map (32, 32)",
            Shape::F32x128 => "fragment map (32, 128)",
            Shape::Restaged32x128 => "sttm restage (32, 128)",
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
                Shape::Restaged32x128 => {
                    module.fragment_restage_32x128(stream, config, a_map, b_map, out)
                }
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

/// An `[rows, columns]` bf16 tile of position identities, packed as the
/// staging buffer a [`kittens::global::PanelMap`] describes.
fn identity_tile(rows: usize, columns: usize) -> Vec<u32> {
    let mut staged = Vec::with_capacity(rows * columns / 2);
    for row in 0..rows {
        for pair in 0..columns / 2 {
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

/// Does a rank-2 [`GlobalLayout`] over a *pitched* buffer deliver the tile the
/// packed 3-D panel map does?
///
/// Two things the panel map cannot state are on trial: a rank the descriptor
/// builder used to hardcode, and a row stride that is not the row's own extent.
/// The staging buffer is `R` rows of `PITCH` bf16 columns with only the first
/// `C` of each row seeded, and the padding is a bit pattern [`decode_cell`]
/// rejects — so a descriptor that walks rows by the wrong stride reports
/// padding at a position rather than a plausible neighbour.
fn check_tma_2d<const R: usize, const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(
        LaunchConfig,
        *const TmaDescriptor,
        &mut DeviceBuffer<u32>,
    ) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    const fn pitch(columns: usize) -> usize {
        3 * columns / 2
    }
    let staged = identity_tile(R, C);
    let mut pitched = vec![u32::MAX; R * pitch(C) / 2];
    for row in 0..R {
        let (source, destination) = (row * C / 2, row * pitch(C) / 2);
        pitched[destination..destination + C / 2].copy_from_slice(&staged[source..source + C / 2]);
    }
    let source = DeviceBuffer::from_host(stream, &pitched)?;
    let layout =
        unsafe { GlobalLayout::<Bf16, 2>::strided(source.cu_deviceptr(), [C, R], [1, pitch(C)]) };
    let map = layout.tensor_map::<Tile<R, C>>(stream)?;
    let mut out = DeviceBuffer::<u32>::zeroed(stream, staged.len())?;
    launch(
        launch_config(R as u32, tile_shared::<R, C>()),
        map.as_ptr(),
        &mut out,
    )?;
    compare_tile(&out.to_host_vec(stream)?, &staged, C / 2)
}

/// Does the TMA store put each box where its map says?
///
/// A round trip against the load path this harness has already pinned to
/// silicon: stage position identities, TMA them in, TMA them straight back out
/// to two fresh buffers, and expect the staging buffer at both. The kernel
/// applies no map of its own, so a failure is a wrong coordinate, a wrong
/// subtile stride or a wrong destination stride, and nothing else.
///
/// The two destinations are the two forms a kernel writes through and differ in
/// more than rank. The packed one is a 3-D panel map — what `softmax` and
/// `layernorm` end with. The pitched one spaces its rows twice as far apart as
/// they are wide, so a store that inherited the *source's* stride, or the row
/// extent's, lands in the padding: every gap holds [`POISON`], which
/// [`cell_bits`] cannot produce, so the report names padding at a position
/// instead of a plausible neighbouring element.
fn check_tma_store<const R: usize, const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(
        LaunchConfig,
        *const TmaDescriptor,
        *const TmaDescriptor,
        *const TmaDescriptor,
    ) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    const fn pitch(columns: usize) -> usize {
        2 * columns
    }
    let staged = identity_tile(R, C);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let source_map = unsafe { encode_bf16_panels::<R, C>(stream, source.cu_deviceptr(), R, 1)? };

    let packed = DeviceBuffer::from_host(stream, &vec![POISON; staged.len()])?;
    let packed_map = unsafe { encode_bf16_panels::<R, C>(stream, packed.cu_deviceptr(), R, 1)? };

    let mut expected = vec![POISON; R * pitch(C) / 2];
    let pitched = DeviceBuffer::from_host(stream, &expected)?;
    let layout =
        unsafe { GlobalLayout::<Bf16, 2>::strided(pitched.cu_deviceptr(), [C, R], [1, pitch(C)]) };
    let pitched_map = layout.tensor_map::<Tile<R, C>>(stream)?;

    launch(
        launch_config(R as u32, tile_shared::<R, C>()),
        source_map.as_ptr(),
        packed_map.as_ptr(),
        pitched_map.as_ptr(),
    )?;

    for row in 0..R {
        let (source, destination) = (row * C / 2, row * pitch(C) / 2);
        expected[destination..destination + C / 2].copy_from_slice(&staged[source..source + C / 2]);
    }
    compare_tile(&packed.to_host_vec(stream)?, &staged, C / 2)
        .map_err(|error| format!("packed destination: {error}"))?;
    compare_tile(&pitched.to_host_vec(stream)?, &expected, pitch(C) / 2)
        .map_err(|error| format!("pitched destination: {error}").into())
}

/// Does the cursor read back what the TMA engine wrote?
///
/// The staged tile is position identities in logical order and the kernel
/// dumps it in logical order, so the expectation is the staging buffer itself.
/// The engine placed the bytes and `SwizzledChunks` found them again: at
/// `C > 64` that is the statement that the subtile stride and the swizzle
/// phase are the hardware's, which nothing on the host can establish.
fn check_swizzle<const R: usize, const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(
        LaunchConfig,
        *const TmaDescriptor,
        &mut DeviceBuffer<u32>,
    ) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile(R, C);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let map = unsafe { encode_bf16_panels::<R, C>(stream, source.cu_deviceptr(), R, 1)? };
    let mut out = DeviceBuffer::<u32>::zeroed(stream, staged.len())?;
    launch(
        launch_config(R as u32, tile_shared::<R, C>()),
        map.as_ptr(),
        &mut out,
    )?;
    compare_tile(&out.to_host_vec(stream)?, &staged, C / 2)
}

/// Does `stmatrix` put a fragment where the cursor says, at every block of the
/// tile? The read-back path is the one `swizzle round trip` pinned to the TMA
/// engine, so a block landing in the wrong subtile shows up as a whole `[16,
/// 16]` square of misplaced words.
fn check_stmatrix<const R: usize, const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(LaunchConfig, &mut DeviceBuffer<u32>) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let expected = identity_tile(R, C);
    let mut out = DeviceBuffer::<u32>::zeroed(stream, expected.len())?;
    launch(launch_config(32, tile_shared::<R, C>()), &mut out)?;
    compare_tile(&out.to_host_vec(stream)?, &expected, C / 2)
}

/// Do the composed shared movers place a whole `[32, WIDE]` band where the
/// per-block ones would?
///
/// The kernel copies rows `0..32` of a staged tile into its rows `32..64`
/// through `load_tile` and `store_tile`, so the expectation is the staged tile
/// with its second half overwritten by its first — and a composition that
/// transposes a block, drops a row offset or loses the subtile stride shows up
/// as a `[16, 16]` square of identities naming the wrong position.
///
/// The `stmatrix`/`ldmatrix` cases fix each block's own addressing against
/// silicon; this is the case for the loop around them.
fn check_band_roundtrip(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile(TILE, WIDE);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let map = unsafe { encode_bf16_panels::<TILE, WIDE>(stream, source.cu_deviceptr(), TILE, 1)? };
    let (band_rows, words) = (TILE / 2, WIDE / 2);
    let mut expected = staged.clone();
    expected.copy_within(0..band_rows * words, band_rows * words);

    let mut out = DeviceBuffer::<u32>::zeroed(stream, staged.len())?;
    unsafe {
        module.band_roundtrip(
            stream,
            launch_config(32, tile_shared::<TILE, WIDE>()),
            map.as_ptr(),
            &mut out,
        )?
    };
    compare_tile(&out.to_host_vec(stream)?, &expected, words)
}

/// Does `ldmatrix` hand each register the element [`BaseLdtm`] says it owns?
///
/// The tile is TMA-staged position identities, so every drained value names the
/// `(row, column)` it came from. The assertion is that the value at
/// `(block, lane, slot, value)` is the identity of that block's
/// `Fragment::coordinate` — a statement about the hardware's ownership map, not
/// about the load and store being mutually inverse. `stmatrix round trip` is
/// the other direction's case and shares nothing with this one but the
/// `fragment_address` derivation.
fn check_ldmatrix<const R: usize, const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(
        LaunchConfig,
        *const TmaDescriptor,
        &mut DeviceBuffer<f32>,
    ) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile(R, C);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let map = unsafe { encode_bf16_panels::<R, C>(stream, source.cu_deviceptr(), R, 1)? };
    let (row_blocks, column_blocks) = (R / 16, C / 16);
    let (slots, values) = (Fragment::SLOTS, Fragment::VALUES);
    let mut out =
        DeviceBuffer::<f32>::zeroed(stream, row_blocks * column_blocks * 32 * slots * values)?;
    launch(
        launch_config(32, tile_shared::<R, C>()),
        map.as_ptr(),
        &mut out,
    )?;
    let observed = out.to_host_vec(stream)?;

    let mut report = String::new();
    let mut mismatches = 0usize;
    for row_block in 0..row_blocks {
        for column_block in 0..column_blocks {
            let block = row_block * column_blocks + column_block;
            for lane in 0..32u32 {
                for slot in 0..slots {
                    for value in 0..values {
                        let (row, column) = Fragment::coordinate(lane, slot, value);
                        let (row, column) = (
                            16 * row_block + row as usize,
                            16 * column_block + column as usize,
                        );
                        let got = observed[dump_index(block, lane, slot, value, slots, values)];
                        if got == cell(row, column) {
                            continue;
                        }
                        mismatches += 1;
                        if mismatches <= 8 {
                            let _ = match decode_cell(got) {
                                Some((got_row, got_column)) => write!(
                                    report,
                                    "\n    block ({row_block}, {column_block}) lane {lane} \
                                     slot {slot} value {value}: map says ({row}, {column}), \
                                     hardware delivered ({got_row}, {got_column})"
                                ),
                                None => write!(
                                    report,
                                    "\n    block ({row_block}, {column_block}) lane {lane} \
                                     slot {slot} value {value}: map says ({row}, {column}), \
                                     hardware delivered {got}, which names no position"
                                ),
                            };
                        }
                    }
                }
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "{} elements, all at BaseLdtm's coordinates",
            observed.len()
        ));
    }
    Err(format!(
        "{mismatches} of {} values misplaced{report}",
        observed.len()
    )
    .into())
}

/// Does a register fragment survive a trip out to TMEM and back?
///
/// The kernel seeds every register with its own dump index, so the expectation
/// is `observed[i] == i` and a mismatch names both the thread coordinate that
/// should own the value and the one that actually wrote it.
///
/// This establishes that `TmemTile::store_tile` is the exact inverse of
/// `TmemTile::tile` — no more. A store and a load agreeing on the *wrong* lane
/// map would pass identically; it is the `fragment map` cases that fix LDTM's
/// map against silicon, and `sttm restage` that checks the store's column
/// arithmetic against a value it did not compute.
fn check_sttm_roundtrip(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let (slots, values) = (RegTile::<32, COLUMNS, BaseLdtm>::SLOTS, COLUMNS / 4);
    let mut out = DeviceBuffer::<f32>::zeroed(stream, ROWS * slots * values)?;
    unsafe {
        module.sttm_roundtrip(
            stream,
            launch_config(ROWS as u32, STTM_SHARED as u32),
            &mut out,
        )?
    };
    let observed = out.to_host_vec(stream)?;

    let coordinate = |index: usize| {
        let (thread, register) = (index / (slots * values), index % (slots * values));
        (
            thread / 32,
            thread % 32,
            register / values,
            register % values,
        )
    };
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, &got) in observed.iter().enumerate() {
        if got == index as f32 {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let (warp, lane, slot, value) = coordinate(index);
            let _ = match (got as usize as f32 == got && (got as usize) < observed.len())
                .then(|| coordinate(got as usize))
            {
                Some((w, l, s, v)) => write!(
                    report,
                    "\n    warp {warp} lane {lane} slot {slot} value {value}: \
                     read back the register of warp {w} lane {l} slot {s} value {v}"
                ),
                None => write!(
                    report,
                    "\n    warp {warp} lane {lane} slot {slot} value {value}: \
                     read back {got}, which no thread wrote"
                ),
            };
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "{} registers survived TMEM unchanged",
            observed.len()
        ));
    }
    Err(format!("{mismatches} of {} registers wrong{report}", observed.len()).into())
}

/// Wait for everything queued on `stream`, and take the process down if it
/// does not arrive within [`HANG_TIMEOUT`].
///
/// `cuStreamSynchronize` behind a deadlocked launch never returns, and neither
/// would any later call in this harness — a case that can hang is a case that
/// reports nothing and burns the container's full timeout, which is strictly
/// worse than no case at all. So the wait is a polled event, and a hang is
/// aborted where it is found: a context with a stuck kernel in it cannot be
/// torn down cleanly, and `abort` is what gets the diagnosis printed and a
/// non-zero exit out of the run.
fn finish_or_abort(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    what: &str,
) -> Result<(), Box<dyn Error>> {
    let event = context.new_event(None)?;
    event.record(stream)?;
    let deadline = Instant::now() + HANG_TIMEOUT;
    while !event.query()? {
        if Instant::now() >= deadline {
            println!(
                "FAIL  {what}: still running after {HANG_TIMEOUT:?}. \
                 A launch of bounded work that does not finish is blocked on a \
                 resource an earlier CTA did not give back — the SM's TMEM \
                 allocator is the one candidate this harness has. Nothing on \
                 this context can make progress, so the run aborts here."
            );
            let _ = std::io::stdout().flush();
            std::process::abort();
        }
        std::thread::sleep(HANG_POLL);
    }
    Ok(())
}

/// Does a kernel that allocates TMEM still run when it is not the first thing
/// to have run on the SM?
///
/// [`RELAUNCHES`] launches over [`RELAUNCH_BLOCKS`] blocks, the whole dump
/// checked every time, each launch watched by [`finish_or_abort`] so that a
/// launch which never returns is a reported failure instead of a container
/// that sits until its timeout. See [`kernels::relaunch_probe`] for what this
/// does and does not claim.
fn check_relaunch(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let registers = Fragment::SLOTS * Fragment::VALUES;
    let width = ROWS * registers;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, RELAUNCH_BLOCKS as usize * width)?;
    let config = LaunchConfig {
        grid_dim: (RELAUNCH_BLOCKS, 1, 1),
        block_dim: (ROWS as u32, 1, 1),
        shared_mem_bytes: STTM_SHARED as u32,
    };

    for launch in 1..=RELAUNCHES {
        // Wiped between launches: a launch whose CTAs never ran would
        // otherwise pass on the previous one's dump.
        out.zero_async(stream)?;
        unsafe { module.relaunch_probe(stream, config, &mut out)? };
        finish_or_abort(
            context,
            stream,
            &format!("repeated launch {launch}/{RELAUNCHES}"),
        )?;

        let observed = out.to_host_vec(stream)?;
        let mut report = String::new();
        let mut mismatches = 0usize;
        for (index, &got) in observed.iter().enumerate() {
            if got == index as f32 {
                continue;
            }
            mismatches += 1;
            if mismatches <= 8 {
                let thread = index / registers;
                let _ = write!(
                    report,
                    "\n    launch {launch}, block {} thread {} register {}: read back {got}",
                    thread / ROWS,
                    thread % ROWS,
                    index % registers
                );
            }
        }
        if mismatches > 0 {
            return Err(format!(
                "{mismatches} of {} registers wrong on launch {launch}{report}",
                observed.len()
            )
            .into());
        }
    }
    Ok(format!(
        "{RELAUNCH_BLOCKS} blocks × {RELAUNCHES} launches, \
         {RELAUNCH_COLUMNS} TMEM columns per CTA"
    ))
}

/// Compare a logical-order tile dump against the identities that were staged
/// into it, reporting the first few misplaced elements by coordinate.
fn compare_tile(
    observed: &[u32],
    expected: &[u32],
    row_words: usize,
) -> Result<String, Box<dyn Error>> {
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, (&got, &want)) in observed.iter().zip(expected).enumerate() {
        if got == want {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let (row, pair) = (index / row_words, index % row_words);
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
                    let index = dump_index(warp, lane, slot, value, slots, values);
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

/// Every statistic [`kernels::reduction_probe`] dumps, as the host derives it:
/// where it lands in the dump, what it must be, and what to call it in a
/// failure. All of it comes from [`BaseLdtm`]'s map and [`accumulator_value`],
/// so the kernel's shuffles are checked against the library's own claim about
/// which lanes hold which coordinate — never against a relaxed expectation.
fn reduction_expectations() -> Vec<(usize, f64, String)> {
    let mut wanted = Vec::new();
    for warp in 0..ROWS / 32 {
        let band = 32 * warp;
        for lane in 0..32u32 {
            let base = (warp * 32 + lane as usize) * REDUCTION_STRIDE;
            for slot in 0..REDUCTION_ROWS {
                let row = band + BaseLdtm::row(lane, slot) as usize;
                let sum = (0..COLUMNS).map(|c| accumulator_value(row, c) as f64).sum();
                wanted.push((
                    base + slot,
                    sum,
                    format!("row {row} sum over {COLUMNS} columns"),
                ));
            }
            for value in 0..REDUCTION_COLUMNS {
                let column = BaseLdtm::column(lane, value) as usize;
                let sum = (0..32)
                    .map(|r| accumulator_value(band + r, column) as f64)
                    .sum();
                wanted.push((
                    base + REDUCTION_ROWS + value,
                    sum,
                    format!("warp {warp} column {column} sum over 32 rows"),
                ));
            }
            wanted.push((
                base + REDUCTION_ROWS + REDUCTION_COLUMNS,
                accumulator_value(band + 31, COLUMNS - 1) as f64,
                format!("warp {warp} band max"),
            ));
            wanted.push((
                base + REDUCTION_ROWS + REDUCTION_COLUMNS + 1,
                (0..32)
                    .flat_map(|r| (0..32).map(move |c| accumulator_value(band + r, c) as f64))
                    .sum(),
                format!("warp {warp} [32, 32] band sum"),
            ));
        }
    }
    wanted
}

/// Do the shuffle butterflies fold exactly the lanes [`BaseLdtm`] says share a
/// row, a column, or the band?
///
/// The host unit tests establish that the *mask sets* are the map's lane
/// groups and that the register-local halves fold the right registers. What
/// they cannot touch is `shuffle_xor` itself, and a mask that reaches the
/// wrong lanes produces a number rather than a fault — so the column sums
/// here, over the stride-4/8/16 butterfly, are the first evidence that path
/// works at all.
fn check_reductions(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let a = DeviceBuffer::from_host(stream, &probe_a())?;
    let b = DeviceBuffer::from_host(stream, &probe_b())?;
    let a_map = unsafe { encode_bf16_panels::<ROWS, DEPTH>(stream, a.cu_deviceptr(), ROWS, 1)? };
    let b_map = unsafe { encode_bf16_panels::<TILE, DEPTH>(stream, b.cu_deviceptr(), TILE, 2)? };

    let mut out = DeviceBuffer::<f32>::zeroed(stream, ROWS * REDUCTION_STRIDE)?;
    unsafe {
        module.reduction_probe(
            stream,
            launch_config(ROWS as u32, PROBE_SHARED as u32),
            a_map.as_ptr(),
            b_map.as_ptr(),
            &mut out,
        )?
    };
    let observed = out.to_host_vec(stream)?;

    let wanted = reduction_expectations();
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, expected, name) in &wanted {
        // An expectation fp32 cannot hold exactly would make this a rounding
        // measurement rather than a reduction test; the shapes are chosen so
        // that never happens, and this is what says so out loud.
        if *expected as f32 as f64 != *expected {
            return Err(format!("{name} is {expected}, which fp32 cannot hold exactly").into());
        }
        if observed[*index] as f64 == *expected {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let _ = write!(
                report,
                "\n    {name}: expected {expected}, reduced to {}",
                observed[*index]
            );
        }
    }
    if mismatches == 0 {
        return Ok(format!("{} statistics exact", wanted.len()));
    }
    Err(format!("{mismatches} of {} statistics wrong{report}", wanted.len()).into())
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
        Box::new(|| {
            check_swizzle::<TILE, TILE>(stream, |config, map, out| unsafe {
                module.swizzle_roundtrip(stream, config, map, out)
            })
        }),
    )];
    for shape in Shape::ALL {
        cases.push((
            shape.name(),
            Box::new(move || check_fragment_map(stream, module, shape)),
        ));
    }
    cases.push((
        "stmatrix round trip",
        Box::new(|| {
            check_stmatrix::<TILE, TILE>(stream, |config, out| unsafe {
                module.stmatrix_roundtrip(stream, config, out)
            })
        }),
    ));
    cases.push((
        "ldmatrix map",
        Box::new(|| {
            check_ldmatrix::<TILE, TILE>(stream, |config, map, out| unsafe {
                module.ldmatrix_map(stream, config, map, out)
            })
        }),
    ));
    // The same three cases over a tile of two stacked subtiles (#25), plus the
    // short tile that separates a per-subtile swizzle phase from an absolute
    // one. Everything above runs on the narrow path these were derived from,
    // so a regression there is distinguishable from a wrong subtile term.
    cases.push((
        "swizzle round trip wide",
        Box::new(|| {
            check_swizzle::<TILE, WIDE>(stream, |config, map, out| unsafe {
                module.swizzle_roundtrip_wide(stream, config, map, out)
            })
        }),
    ));
    cases.push((
        "swizzle phase carry",
        Box::new(|| {
            check_swizzle::<SHORT, WIDE>(stream, |config, map, out| unsafe {
                module.swizzle_roundtrip_short(stream, config, map, out)
            })
        }),
    ));
    cases.push((
        "stmatrix round trip wide",
        Box::new(|| {
            check_stmatrix::<TILE, WIDE>(stream, |config, out| unsafe {
                module.stmatrix_roundtrip_wide(stream, config, out)
            })
        }),
    ));
    cases.push((
        "ldmatrix map wide",
        Box::new(|| {
            check_ldmatrix::<TILE, WIDE>(stream, |config, map, out| unsafe {
                module.ldmatrix_map_wide(stream, config, map, out)
            })
        }),
    ));
    cases.push((
        "band round trip",
        Box::new(|| check_band_roundtrip(stream, module)),
    ));
    cases.push((
        "sttm round trip",
        Box::new(|| check_sttm_roundtrip(stream, module)),
    ));
    cases.push((
        "reduction shuffles",
        Box::new(|| check_reductions(stream, module)),
    ));
    cases.push((
        "2d tensor map",
        Box::new(|| {
            check_tma_2d::<TILE, TILE>(stream, |config, map, out| unsafe {
                module.tma_2d_roundtrip(stream, config, map, out)
            })
        }),
    ));
    // The store side (#9), against the load side these cases have just pinned.
    cases.push((
        "tma store round trip",
        Box::new(|| {
            check_tma_store::<TILE, TILE>(stream, |config, source, packed, pitched| unsafe {
                module.tma_store_roundtrip(stream, config, source, packed, pitched)
            })
        }),
    ));
    cases.push((
        "tma store round trip wide",
        Box::new(|| {
            check_tma_store::<TILE, WIDE>(stream, |config, source, packed, pitched| unsafe {
                module.tma_store_roundtrip_wide(stream, config, source, packed, pitched)
            })
        }),
    ));
    cases.push((
        "tma store early recycle",
        Box::new(|| {
            check_tma_store::<TILE, TILE>(stream, |config, source, packed, pitched| unsafe {
                module.tma_store_recycle(stream, config, source, packed, pitched)
            })
        }),
    ));
    // Last, and not because it is the least interesting: it is the only case
    // that can take the process down (see `finish_or_abort`), so everything
    // else has reported by the time it runs.
    cases.push((
        "repeated launch",
        Box::new(|| check_relaunch(&context, stream, module)),
    ));

    let mut failures = 0usize;
    for (name, case) in &cases {
        match case() {
            Ok(note) => println!("pass  {name:<26}  {note}"),
            Err(error) => {
                println!("FAIL  {name:<26}  {error}");
                failures += 1;
            }
        }
    }
    println!("{} of {} cases passed", cases.len() - failures, cases.len());
    Ok(failures)
}
