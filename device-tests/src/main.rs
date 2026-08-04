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
//! The [`SharedVec`] cases are the one family whose subject is the *absence*
//! of a layout: an unswizzled box is the shape nothing else here builds, so
//! they are what says the engine writes a vector contiguously rather than
//! saying a phase was computed right. `global rows map` and the `shared drain`
//! pair are the cases whose subject is the absence of an *engine*: a plain
//! pitched matrix on the global side, addressed by the calling threads, with no
//! descriptor between it and either the registers or a staged tile. The drain
//! cases are also the only ones that check what a path was required *not* to
//! write — the poison margins around the rectangle they were asked for.
//!
//! The `mma A·Bᵀ` … `mma transpose control` family is the one whose subject is
//! *arithmetic* rather than addressing, and the one written against a surface
//! no kernel in this repo calls. All four operand orders (#12) multiply the
//! same two logical matrices, staged four ways because no TMA box transposes,
//! and every one of them is required to produce the same product. Two things
//! carry that: `walk blind spots`, which runs on the host and fails if
//! [`walk_reference`] cannot see a dropped K plane, a permuted K chunk or a
//! confused MN subtile — the standing answer to a reference that certifies
//! whatever it is handed — and `mma transpose control`, which is the same
//! operands under the *untransposed* walk and is required to disagree.
//!
//! `block reduction` is the one case whose subject is more than one warp.
//! Everything else here launches 32 or 128 threads and checks a claim about
//! *one* warp's registers; a block reduction is the only thing in the library
//! that no warp can compute alone, so it is the only case where the answer
//! depends on four warps having met at a barrier in the right order.
//!
//! Run it with `modal run modal_app.py` (see `modal_app.py` at the repo root);
//! it exits non-zero if any case fails.

// Every `#[kernel]` is an `unsafe fn` whose contract is its launch config, and
// each states that in its own doc; a `# Safety` section per entry point would
// be five lines of ceremony saying the same thing.
#![allow(clippy::missing_safety_doc)]
// What lets `ladder!` name a kernel after the shape it compiles. A PTX entry
// symbol is its bare function name, so every rung of the sweep needs a
// distinct identifier, and without identifier concatenation each one has to be
// written out by hand — which is exactly why the register table had two widths
// (#60). This crate is nightly-only and pinned to the toolchain the codegen
// backend was built against, so the feature is no new constraint on anything.
#![feature(macro_metavar_expr_concat)]

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
use cuda_device::thread::__unroll_config;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cluster, cluster_launch, cuda_module, debug, kernel, thread, warp};

use kittens::epilogue::{StoreRing, Warp};
use kittens::global::{
    GlobalLayout, GlobalRows, accumulate_shared_rows, encode_bf16_panels, load_col_vec, load_cols,
    load_rows, store_rows, store_shared_rows,
};
use kittens::ldst::{load_fragment, load_tile, load_vec, scatter_tile, store_fragment, store_tile};
use kittens::mma::{self, MmaShape, mm_ab, mm_abt, mm_atb, mm_atbt, mma_abt};
use kittens::reg::{
    BaseLdtm, ColLayout, ColVec, Fragment, FragmentLayout, Max, Mul, RegTile, RegVec,
    online_rescale,
};
use kittens::shared::{
    Bf16, F32, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait,
    tma_store_wait_read,
};
use kittens::sync::{Semaphore, block_reduce, block_reduce_sum};
use kittens::tmem::{
    TmemTile, alloc_block, alloc_cluster, dealloc_block, dealloc_cluster, store_wait,
};

mod ladder_bench;
mod tmem_occupancy;
mod tmem_residency;

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

/// Rows of the [`StoreRing`] probe's staging buffer. Capped at 64 because the
/// band identities are [`cell`]s and a `cell`'s row field is six bits — the
/// case distinguishes bands by rotating the *column* identity ([`ring_column`])
/// rather than by widening the row, so nothing here needs a second encoding.
const RING_ROWS: usize = 64;
/// Rows one warp of that probe owns. Four warps write one staging buffer while
/// a single thread issues its store, which is the arrangement the collective
/// halves of [`StoreRing`] exist for — a one-warp probe would exercise neither
/// the proxy fence's "every writer fences" clause nor the barrier that releases
/// the other warps past thread 0's group wait.
const RING_WARP_ROWS: usize = 16;
/// Threads that probe launches with.
const RING_THREADS: u32 = (RING_ROWS / RING_WARP_ROWS) as u32 * 32;
/// Bands it pushes through the ring. Every buffer of the deepest ring under
/// test is reused at least once at this count, which is the whole point: depth
/// 1 cannot exercise the reuse hazard, and a ring that never wraps is a ring
/// whose `acquire` has nothing to wait for.
const RING_BANDS: usize = 8;
/// How far band `b` rotates its column identities. Odd, so the rotation is a
/// bijection on 64 and on 128 columns alike, and the eight bands' rotations are
/// distinct at both widths — which is what makes a buffer that was recycled too
/// early, or a band stored at another band's rows, decode to a *wrong column*
/// naming the band it came from instead of a plausible right value.
const RING_STRIDE: usize = 37;

/// The column whose identity band `band` parks at column `column` of its
/// staging buffer. The device fills registers with it and the host builds the
/// expected destination from it, so the two cannot disagree about the pattern.
const fn ring_column(column: usize, band: usize, columns: usize) -> usize {
    (column + RING_STRIDE * band) % columns
}

/// Rows of the tile the shared→global drain case stages. Capped at 64 for
/// [`cell`]'s six-bit row field, as [`RING_ROWS`] is.
const DRAIN_ROWS: usize = 64;
/// Rows one warp of that case fills — the same sixteen the store ring's warps
/// take, and for the same reason: four warps writing one staging tile is the
/// arrangement where the barrier before the drain is load-bearing, and a
/// one-warp probe would pass with no barrier at all.
const DRAIN_WARP_ROWS: usize = 16;
/// Threads the drain probe launches with, and the `THREADS` it drains under.
const DRAIN_THREADS: u32 = (DRAIN_ROWS / DRAIN_WARP_ROWS) as u32 * 32;

/// Rows of the pitched bf16 matrix the drain writes into. Taller than
/// [`DRAIN_ROWS`] by more than [`DRAIN_ROW`], so the rectangle has margin above
/// *and* below it that has to come back untouched.
const DRAIN_MATRIX_ROWS: usize = 80;
/// Its leading dimension, in bf16 elements. Wider than either tile width, not a
/// multiple of one, and a multiple of eight elements so that the stride never
/// limits the access ladder — every rung below is the column origin's doing.
const DRAIN_PITCH: usize = 208;
/// The row the rectangle starts at. Non-zero, so a drain that ignored the row
/// origin lands on the poison above it.
const DRAIN_ROW: u32 = 8;

/// The four column origins the case drains at, one per rung of
/// [`kittens::global::store_shared_rows`]' access ladder.
///
/// Against a 256-byte-aligned allocation and [`DRAIN_PITCH`], the widths they
/// select are 16, 8, 4 and 2 bytes — `access_width`'s host tests pin that
/// arithmetic, and these are the same numbers reached through the driver's own
/// pointer. Each is at least 64, so there is a left margin, and each leaves the
/// widest tile inside the pitch, so there is a right one.
const DRAIN_COLUMNS: [u32; 4] = [64, 68, 66, 65];

/// The half-word a drain's destination is seeded with: [`POISON`]'s own halves,
/// and a value [`cell_bits`] can never take.
const POISON_HALF: u16 = 0xffff;

/// Columns of the narrow fp32 staging tile: 32 fp32 is one 128-byte swizzle
/// atom, so it is [`TILE`]'s single-subtile role at four bytes an element.
const F32_TILE: usize = 32;
/// Columns of the wide one — four stacked subtiles rather than [`WIDE`]'s two,
/// which is the same 128 logical columns costing three subtile crossings a row
/// instead of one.
const F32_WIDE: usize = 128;

/// One warp's band of the drain probe, at the staging tile's full width.
type DrainBand<const C: usize> = RegTile<DRAIN_WARP_ROWS, C, BaseLdtm>;

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

/// The reduction-store case's tile (#42): fp32, two stacked 32-column
/// subtiles, at the `[16, 64]` shape whose 4096 bytes are exactly one bf16
/// `[32, 64]` staging buffer — the reinterpretation an accumulating fp32
/// epilogue leans on.
type AddTile = SharedTile<F32, ADD_ROWS, ADD_COLS, Swizzle128B>;
const ADD_ROWS: usize = 16;
const ADD_COLS: usize = 64;
/// The staging tile at fp32 — the one `stmatrix` cannot fill, and the whole
/// subject of the scatter cases.
type F32Tile<const R: usize, const C: usize> = SharedTile<F32, R, C, Swizzle128B>;
type AOperand = Tile<ROWS, DEPTH>;
type BOperand = Tile<TILE, DEPTH>;
type Accumulator = TmemTile<ROWS, COLUMNS>;
/// One warp's band of the [`StoreRing`] probe: the [`RING_WARP_ROWS`] rows it
/// owns of the staging buffer, at the buffer's full width.
type RingBand<const C: usize> = RegTile<RING_WARP_ROWS, C, BaseLdtm>;

/// Shared bytes a [`StoreRing`] probe launches with — the ring and nothing
/// else. A store ring needs no mbarrier: a bulk store completes on a group
/// count and there is no barrier anywhere in this case's plan.
const fn ring_shared<const C: usize, const IN_FLIGHT: u32>() -> u32 {
    StoreRing::<Bf16, RING_ROWS, C, Swizzle128B, IN_FLIGHT>::BYTES as u32
}

/// One warp's own store ring: [`RING_WARP_ROWS`] rows, one buffer, and
/// `bar.warp.sync` where the block-scope ring has `bar.sync`.
///
/// The shape `examples/src/gemm.rs`'s TMA epilogue stages at (#122). Four of
/// these side by side is the same memory as one [`StoreRing`] over
/// [`RING_ROWS`], and the case below is that arrangement against the one the
/// four cases above run.
type WarpRing<const C: usize> = StoreRing<Bf16, RING_WARP_ROWS, C, Swizzle128B, 0, Warp>;

/// Shared bytes a [`WarpRing`] probe launches with: one ring per warp, plus the
/// 128-byte units `PHASE` pushes them all along by.
const fn warp_ring_shared<const C: usize, const PHASE: usize>() -> u32 {
    (PHASE * 128 + (RING_THREADS as usize / 32) * WarpRing::<C>::BYTES) as u32
}

/// The lane probe's staging tile: a warp's 32 rows by the one swizzle atom
/// [`SharedTile::chunk_writer`] accepts, which is what `store_fragment`
/// addresses into.
type LaneStage = SharedTile<Bf16, 32, TILE, Swizzle128B>;

/// Elements of the vector cases' parameter vector. [`WIDE`] bf16 is 256 bytes
/// — one unswizzled TMA box, and twice the 128-byte swizzle atom a tile's box
/// is capped at, so a vector that was quietly cut into atoms comes back with
/// only its first half in place.
const VECTOR: usize = WIDE;
/// Rows of the 2-D parameter buffer the rank-2 vector case slices, and the row
/// it asks for. The row is neither the first nor the last, so a descriptor
/// that ignores the coordinate and one that runs off the end are different
/// failures.
const VECTOR_ROWS: usize = 4;
const VECTOR_ROW: usize = 2;

type Params = SharedVec<Bf16, VECTOR>;
/// The block reduction's staging buffer: one fp32 per warp, 16 bytes at four
/// warps and never a TMA box — written by `set`, read by `get`, one barrier
/// apart. fp32 and not bf16 because a partial rounded on its way through
/// shared memory is a wrong sum, not a wrong layout.
type Partials = SharedVec<F32, BLOCK_WARPS>;
/// One warp's band in that case. `[32, 64]` is 2048 values, the divisor
/// [`block_partial`]'s seeds are scaled by.
type BlockBand = RegTile<32, BLOCK_BAND_COLUMNS, BaseLdtm>;
/// The vector's plan: itself plus the same scratch tail the tiles use.
/// `Params::BYTES` is 128-byte aligned, so the barrier lands where a barrier
/// may.
const VECTOR_SHARED: u32 = (Params::BYTES + 32) as u32;
/// Values one lane holds of a `ColVec<VECTOR, BaseLdtm>`.
const VECTOR_VALUES: usize = <BaseLdtm as ColLayout<VECTOR>>::VALUES;

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

/// The four operand orders' `A` staged K-major, `[M, K]`.
type AKMajor = Tile<ROWS, DEPTH>;
/// The same `A` staged MN-major, `[K, M]` — the transposed walks' operand,
/// and the transpose of [`AKMajor`] in *global* memory rather than in the
/// walk, because no TMA box transposes.
type AMnMajor = Tile<DEPTH, ROWS>;
/// `B` staged K-major, `[N, K]`.
type BKMajor = Tile<COLUMNS, DEPTH>;
/// `B` staged MN-major, `[K, N]`.
type BMnMajor = Tile<DEPTH, COLUMNS>;

/// Dynamic shared plan of an operand-order case: two operand tiles and the
/// same 32-byte tail the fragment probe uses. Every one of the four staged
/// tiles is `[128, 64]` or `[64, 128]` bf16, so one number serves all of
/// them — asserted rather than assumed, since the launch config is written
/// from it.
const WALK_SHARED: usize = 2 * AKMajor::BYTES + 32;
const _: () = assert!(
    AKMajor::BYTES == AMnMajor::BYTES
        && AKMajor::BYTES == BKMajor::BYTES
        && AKMajor::BYTES == BMnMajor::BYTES
);
/// The accumulator band the operand-order cases drain, one per warp.
const WALK_SLOTS: usize = RegTile::<32, COLUMNS, BaseLdtm>::SLOTS;
const WALK_VALUES: usize = RegTile::<32, COLUMNS, BaseLdtm>::VALUES;
/// K=16 chunks one operand-order MMA chains.
const WALK_CHUNKS: usize = DEPTH / 16;
/// The square the untransposed control computes and the host compares over —
/// one swizzle subtile of the accumulator. See
/// [`kernels::walk_untransposed_control`].
const CONTROL_EDGE: usize = 64;

/// Columns [`kernels::relaunch_probe`] allocates: the whole SM's tensor
/// memory. No two CTAs can hold an allocation on the same SM at once, so
/// every CTA after the first is served out of columns the one before it gave
/// back — a probe taking a quarter of TMEM would be handed a free quarter
/// without the allocator ever having to hand anything over.
const RELAUNCH_COLUMNS: usize = 512;
/// Blocks in that probe's grid: twice the B200's 148 SMs, so the launch
/// cannot fit in one wave at any occupancy.
const RELAUNCH_BLOCKS: u32 = 296;
/// What [`kernels::tmem_ladder_none`] leaves in its staging word where the
/// allocator would have left an address. Any value distinguishable from a TMEM
/// address will do; this one is a recognisable non-address.
const TMEM_LADDER_SENTINEL: u32 = 0x7be5_0000;

/// Words each residency census CTA writes: `[smid, entered, allocated, left]`.
/// See [`tmem_residency`], which is the whole argument for why those four.
const CENSUS_FIELDS: usize = 4;

/// Words each CLC probe CTA writes:
/// `[completed, is_canceled, first_ctaid_x, ctaid_x]`.
const CLC_FIELDS: usize = 4;
/// The `try_cancel` response, whose size is the ISA's `.b128` and is also the
/// transaction count its mbarrier is charged.
const CLC_RESPONSE_BYTES: usize = 16;
/// Shared bytes the probe launches with: the response, then its barrier.
const CLC_SHARED_BYTES: u32 = CLC_RESPONSE_BYTES as u32 + 8;
/// Clusters the probe launches. Its shared plan is 24 bytes and it holds no
/// tensor memory, so an SM admits many of it — the grid has to be far larger
/// than the device can hold at once or there is nothing pending to cancel and
/// every row comes back `is_canceled = 0`, which would prove only that the
/// instruction returns.
const CLC_CLUSTERS: u32 = 4096;
/// Nanoseconds a rank waits for its copy of the response before writing down
/// that it never arrived. Generous by three orders of magnitude against a
/// steal, because the only thing a short deadline could produce is a false
/// report of the exact failure this case exists to detect.
const CLC_DEADLINE_NS: u64 = 50_000_000;

/// Iterations [`kernels::census_spin`] runs before giving up on the clock.
///
/// The spin's real exit is `%globaltimer` passing a deadline, and a loop whose
/// only exit is an intrinsic the compiler might hoist would hang a device that
/// costs money. This bounds it at roughly a millisecond of empty iterations on
/// any plausible clock, against the ~10⁴ iterations a working spin needs — and
/// the host checks the *achieved* hold against the requested one, so a spin
/// that ended on the guard is reported rather than quietly believed.
const CENSUS_SPIN_GUARD: u64 = 2_000_000;

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

/// Warps the block-reduction case runs. Four because that is `groupnorm_tile`'s
/// width and the one this library's kernels all launch at, not because the
/// collective needs it — `WARPS` is free.
const BLOCK_WARPS: usize = 4;
/// One warp's band in that case, and the source of its partial: 2048 values,
/// so a seed of `8^w / 2048` folds to exactly `8^w` with every partial sum
/// along the way an exact fp32.
const BLOCK_BAND_COLUMNS: usize = 64;
const BLOCK_BAND_VALUES: f32 = (32 * BLOCK_BAND_COLUMNS) as f32;
/// The three statistics [`kernels::block_reduce_probe`] dumps per thread: the
/// sum, the sum of the doubled partials, and the max.
const BLOCK_REDUCE_STRIDE: usize = 3;

/// Warp `w`'s partial in that case — a distinct power of eight, so the base-8
/// digits of any fold over them say exactly which slots were read and how many
/// times. A warp reading its own slot, a slot read twice and a slot skipped are
/// three different numbers rather than three ways of being wrong.
fn block_partial(warp: usize) -> f32 {
    8f32.powi(warp as i32)
}

/// Which spelling of the online-softmax step [`kernels::softmax_probe`]
/// compiles. Only the codegen probe reads these; see its doc comment.
const HAND_WRITTEN: u32 = 0;
/// [`online_rescale`], the deliberately fused form.
const FUSED: u32 = 1;
/// [`HAND_WRITTEN`] with the generic `row_map::<Mul>` in place of `scale_rows`.
const ROW_MAP: u32 = 2;
/// [`HAND_WRITTEN`] with `mul_row_assign` — #31's in-place row map — in place
/// of `scale_rows`. The pair `ROW_MAP` / this one is the whole of #31's
/// headline claim: same op, same operand, by value against in place.
const ROW_MAP_ASSIGN: u32 = 3;
/// [`HAND_WRITTEN`] with `out_acc.add_assign(p)` for `out_acc = out_acc.add(p)`
/// — the accumulate `flash_forward` was blocked on. Its by-value twin rebinds
/// its input, so under #38's liveness reading this pair should *not* differ,
/// which is the prediction it is here to test.
const ADD_ASSIGN: u32 = 4;

/// How [`kernels::scalar_map_probe`] spells its two scalar multiplies. The
/// pre-#38 spelling: `bin_map::<Mul>(splat(k))`, the scalar widened into a
/// whole second tile because there was no other way to reach a `BinaryOp` with
/// one. This is the number `scale` has to beat.
const SPLATTED: u32 = 0;
/// `RegTile::scale` — the same `Mul` with the operand left in a register.
const SCALED: u32 = 1;
/// Neither: both multiplies written through `set`, mutating in place. #31's
/// shape, hand-written; the floor the library form has to reach.
const IN_PLACE: u32 = 2;
/// `RegTile::scale_assign` — #31's in-place scalar map, which is [`IN_PLACE`]
/// spelled as library API. The pair `IN_PLACE` / this one is what says the API
/// costs nothing over the loop it replaces.
const SCALE_ASSIGN: u32 = 3;
/// [`SCALED`] with the scaled block chained into the accumulate rather than
/// rebound — `out_acc.add(block.scale(k))`, so `block` and its scaled copy are
/// both live where the rebinding form has only one. #38 measured this at 84
/// bytes of spill from a probe edit that was never committed, and it is the
/// evidence the whole peak-liveness reading of #31 rests on; it is a standing
/// form now.
const CHAINED: u32 = 4;
/// The same step as one expression — `out_acc.scale(k).add(block.scale(k))`,
/// nothing rebound at all, which under the liveness reading should be the
/// most expensive form here.
const ONE_EXPRESSION: u32 = 5;

/// How [`kernels::ladder_probe`] spells the step it sweeps — the spellings
/// [`scalar_map_probe`](kernels::scalar_map_probe) prices at two widths, kept
/// as the sweep's only non-shape axis. All five compute the same tile, so a
/// difference across a row of the ladder table is the spelling and nothing
/// else, and the first four differ only in how the two *scales* are written.
///
/// The whole step as one expression, `out_acc.scale(k).add(block.scale(k))`:
/// nothing rebound, no whole band materialized between statements. This is
/// [`ONE_EXPRESSION`].
const LADDER_FUSED: u32 = 0;
/// Both scales through `scale_assign`, then the by-value accumulate the flash
/// call site writes — [`SCALE_ASSIGN`], #31's form.
const LADDER_ASSIGN: u32 = 1;
/// Both scales by value with each result rebound to its input — one whole band
/// materialized per statement, [`SCALED`] and #31's expensive form.
const LADDER_REBOUND: u32 = 2;
/// Both scales as `get`/`set` passes, no library op in them at all —
/// [`IN_PLACE`], statement for statement what [`LADDER_ASSIGN`] compiles, and
/// the floor that form has to reach.
const LADDER_OPEN_CODED: u32 = 3;
/// [`LADDER_ASSIGN`] with the accumulate in place too, so *no* statement in
/// the step materializes a band. The one spelling on this ladder with no twin
/// in `scalar_map_probe`, which shares the by-value `add` across all six of
/// its forms — and the one that shows what the by-value `add` was holding in
/// registers, since without it ptxas is free to leave the band in local memory
/// and stream it.
const LADDER_ALL_IN_PLACE: u32 = 4;

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

/// How [`kernels::global_copy_probe`] moves a band between global memory and
/// registers — issue #11. Both arms read and write the same elements, so a
/// difference in the `regcount` table is the spelling and nothing else.
///
/// The address math a kernel writes without the movers: `RegTile::coordinate`
/// per value, the leading dimension multiplied out at the call site. This is
/// `gemm.rs`'s deleted epilogue, and the load half it never had.
const OPEN_CODED: u32 = 0;
/// [`load_rows`] and [`store_rows`] over a [`GlobalRows`] cursor — the API.
const MOVERS: u32 = 1;

/// Rows of the plain fp32 matrix the global ↔ register case addresses, and its
/// leading dimension. The pitch is wider than the band read out of it and is
/// not a multiple of that band's width, so a mover stepping rows by the band's
/// own width would land on a different element in every row but the first.
const GLOBAL_ROWS: usize = 64;
const GLOBAL_PITCH: usize = 192;
/// The origin of the band [`kernels::global_rows_map`] reads. Neither
/// coordinate is zero and the column is not a multiple of 16: a rectangle of
/// global memory is addressed elementwise, so nothing about the fragment's
/// 16-column blocks should make the origin prefer a block boundary.
const GLOBAL_ROW: u32 = 16;
const GLOBAL_COLUMN: u32 = 40;
/// Columns of the band [`kernels::global_col_vec_map`]'s statistic covers.
/// Thirty-two and not `WIDE`: the band runs down the matrix's rows from
/// `GLOBAL_ROW`, and there are only `GLOBAL_ROWS - GLOBAL_ROW` of them left.
const COLUMN_BAND: usize = 32;

/// The matrix's value at `(row, column)` — its own flat index, unique over the
/// buffer and an exact fp32 integer well under 2^24, so a value that reaches
/// the wrong register names the element it was actually read from.
fn global_cell(row: usize, column: usize) -> f32 {
    (row * GLOBAL_PITCH + column) as f32
}

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
                tma.expect_tx(tile.tma_load(source, 0, 0, tma));
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
                tma.expect_tx(tile.tma_load_2d(source, 0, 0, tma));
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
                tma.expect_tx(tile.tma_load(source, 0, 0, tma));
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

    /// TMA an fp32 tile in and *reduce-store* it out **twice**
    /// ([`SharedTile::tma_store_add_2d`], #42), onto a destination the host
    /// seeded nonzero.
    ///
    /// The double issue is the whole assertion: the expected answer is
    /// `seed + 2 · tile`, which a plain store (`tile`), a single add
    /// (`seed + tile`) and a store that lost the seed (`2 · tile`) all miss —
    /// so the op is pinned as an *add at the destination* rather than any
    /// overwrite that happens to land in the right place. Both reductions ride
    /// one committed group; element-wise fp32 addition is commutative here
    /// because every operand is an exact small integer.
    #[kernel]
    pub unsafe fn tma_store_add_twice(source: *const TmaDescriptor, dest: *const TmaDescriptor) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = AddTile::from_raw(smem);
            let tma = Semaphore::attach(smem.add(AddTile::BYTES) as *mut Barrier);
            let tid = thread::threadIdx_x();

            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tma.expect_tx(tile.tma_load_2d(source, 0, 0, tma));
            }
            tma.wait(0);
            thread::sync_threads();

            // Async proxy on both sides, so the load's barrier is the whole
            // ordering — `tma_store_probe`'s argument, unchanged by the add.
            if tid == 0 {
                tile.tma_store_add_2d(dest, 0, 0);
                tile.tma_store_add_2d(dest, 0, 0);
                tma_store_commit();
                tma_store_wait::<0>();
                tma.inval();
            }
        }
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

    /// Push [`RING_BANDS`] register bands out through a [`StoreRing`], one
    /// acquire/commit pair each, landing band `b` at rows `RING_ROWS * b` of
    /// the destination.
    ///
    /// The whole chain the library owns is in here and nothing else is: the
    /// registers are filled from [`RingBand::coordinate`] and never loaded, so
    /// no TMA load, no mbarrier and no MMA can be the cause of a mismatch.
    /// What remains is `store_tile` into the acquired buffer, the proxy fence
    /// and barrier `commit` takes before handing it to the engine, the group
    /// wait `acquire` takes before the buffer comes round again, and the final
    /// [`StoreRing::drain`].
    ///
    /// Four warps write each buffer and one thread issues each store, which is
    /// the arrangement whose ordering is the point: a fence taken only by the
    /// issuing thread, or a group wait not separated from the other warps'
    /// writes by a barrier, both leave a band partly written and partly stale.
    /// Launch with [`RING_THREADS`].
    #[inline(always)]
    unsafe fn store_ring_probe<const C: usize, const IN_FLIGHT: u32>(
        destination: *const TmaDescriptor,
    ) where
        BaseLdtm: FragmentLayout<RING_WARP_ROWS, C>,
    {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let mut ring = StoreRing::<Bf16, RING_ROWS, C, Swizzle128B, IN_FLIGHT>::attach(smem);
            let lane = warp::lane_id();
            let row_base = RING_WARP_ROWS as u32 * warp::warp_id();

            let mut band = 0usize;
            while band < RING_BANDS {
                let staging = ring.acquire();

                let mut values = RingBand::<C>::zero();
                let mut slot = 0usize;
                while slot < RingBand::<C>::SLOTS {
                    let mut value = 0usize;
                    while value < RingBand::<C>::VALUES {
                        let (row, column) = RingBand::<C>::coordinate(lane, slot, value);
                        values.set(
                            slot,
                            value,
                            cell(
                                (row_base + row) as usize,
                                ring_column(column as usize, band, C),
                            ),
                        );
                        value += 1;
                    }
                    slot += 1;
                }
                store_tile(staging.chunk_writer(), row_base, 0, lane, values);

                ring.commit(destination, (RING_ROWS * band) as i32, 0);
                band += 1;
            }
            // Nothing else makes the bytes visible to the host — not the
            // barriers above and not the kernel ending.
            ring.drain();
        }
    }

    /// [`store_ring_probe`] at depth 1: one buffer, so every band's `stmatrix`
    /// waits out the previous band's store reads. The degenerate ring, and the
    /// shape `softmax` writes by hand.
    #[kernel]
    pub unsafe fn store_ring_depth_1(destination: *const TmaDescriptor) {
        unsafe { store_ring_probe::<TILE, 0>(destination) }
    }

    /// [`store_ring_probe`] at depth 2 — the shallowest ring that can get the
    /// reuse hazard wrong, since band `k + 2` writes the buffer band `k`'s
    /// store is reading.
    #[kernel]
    pub unsafe fn store_ring_depth_2(destination: *const TmaDescriptor) {
        unsafe { store_ring_probe::<TILE, 1>(destination) }
    }

    /// [`store_ring_probe`] at depth 4, where three stores are in flight across
    /// a reuse and the wait is counting groups rather than draining them.
    #[kernel]
    pub unsafe fn store_ring_depth_4(destination: *const TmaDescriptor) {
        unsafe { store_ring_probe::<TILE, 3>(destination) }
    }

    /// [`store_ring_probe`] at depth 2 over buffers of two stacked subtiles, so
    /// each commit issues two boxes into one group — the case that says a
    /// group is a *tile* and not an instruction.
    #[kernel]
    pub unsafe fn store_ring_depth_2_wide(destination: *const TmaDescriptor) {
        unsafe { store_ring_probe::<WIDE, 1>(destination) }
    }

    /// The same bands through **four warp-scope rings** instead of one
    /// block-scope one: each warp owns a [`WarpRing`], fills it alone, and
    /// commits its own [`RING_WARP_ROWS`] rows of every band.
    ///
    /// Two claims live here that [`store_ring_probe`] cannot make.
    ///
    /// **`bar.warp.sync` carries a proxy fence.** The TMA engine reads through
    /// the async proxy and `stmatrix` writes through the generic one, so the
    /// fence every lane takes has to reach lane 0 before it issues. At block
    /// scope a `bar.sync` does that and nobody doubts it; at warp scope the
    /// claim is that `bar.warp.sync` orders memory among a warp's lanes the same
    /// way. If it does not, lane 0 hands the engine a buffer some lane has not
    /// published and the case reports a stale or half-written band.
    ///
    /// **The swizzle phase is the buffer's absolute one.** `PHASE` pushes the
    /// whole run along by that many 128-byte rows, so at `PHASE = 1` every
    /// buffer starts mid-swizzle-period — which is where `gemm`'s staging run
    /// sits, since its offset is a shared plan rounded to 128 and not to 1024.
    /// [`SharedTile::chunk_writer`] folds that phase in; whether the *engine*
    /// derives the same one from the same address is a hardware fact this repo
    /// has only ever checked on the load side (`swizzle roundtrip short`). A
    /// disagreement comes back as chunks permuted within their rows, which
    /// decodes to wrong columns rather than to poison.
    ///
    /// Launch with [`RING_THREADS`].
    #[inline(always)]
    unsafe fn store_ring_warp_probe<const C: usize, const PHASE: usize>(
        destination: *const TmaDescriptor,
    ) where
        BaseLdtm: FragmentLayout<RING_WARP_ROWS, C>,
    {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let warp_id = warp::warp_id();
            let mut ring = WarpRing::<C>::attach(
                smem.add(PHASE * 128 + warp_id as usize * WarpRing::<C>::BYTES),
            );
            let lane = warp::lane_id();
            let row_base = RING_WARP_ROWS as u32 * warp_id;

            let mut band = 0usize;
            while band < RING_BANDS {
                let staging = ring.acquire();

                let mut values = RingBand::<C>::zero();
                let mut slot = 0usize;
                while slot < RingBand::<C>::SLOTS {
                    let mut value = 0usize;
                    while value < RingBand::<C>::VALUES {
                        let (row, column) = RingBand::<C>::coordinate(lane, slot, value);
                        values.set(
                            slot,
                            value,
                            cell(
                                (row_base + row) as usize,
                                ring_column(column as usize, band, C),
                            ),
                        );
                        value += 1;
                    }
                    slot += 1;
                }
                // Row 0 of the buffer, not `row_base`: a warp-scope ring's
                // buffer is only this warp's rows, where the block-scope one
                // holds all four warps' and each warp writes its own slice.
                store_tile(staging.chunk_writer(), 0, 0, lane, values);

                ring.commit(
                    destination,
                    (RING_ROWS * band + RING_WARP_ROWS * warp_id as usize) as i32,
                    0,
                );
                band += 1;
            }
            ring.drain();
        }
    }

    /// [`store_ring_warp_probe`] with the rings 1024-byte aligned — the warp
    /// scope on its own.
    #[kernel]
    pub unsafe fn store_ring_warp(destination: *const TmaDescriptor) {
        unsafe { store_ring_warp_probe::<TILE, 0>(destination) }
    }

    /// [`store_ring_warp_probe`] with the whole run pushed one 128-byte row
    /// along, so every buffer starts mid-swizzle-period — `gemm`'s own offset,
    /// and the case that says the engine reads the phase off the address.
    #[kernel]
    pub unsafe fn store_ring_warp_phased(destination: *const TmaDescriptor) {
        unsafe { store_ring_warp_probe::<TILE, 1>(destination) }
    }

    /// Fill a `[DRAIN_ROWS, C]` staging tile with position identities through
    /// `store_tile`, then push it out to the `(DRAIN_ROW, column)` rectangle of
    /// an ordinary pitched bf16 matrix with
    /// [`kittens::global::store_shared_rows`].
    ///
    /// **What this proves.** There is no descriptor and no engine anywhere on
    /// the way out: the destination is a plain row-major buffer the host seeded
    /// with poison, addressed by the calling threads. The rectangle is narrower
    /// than the matrix's leading dimension and starts at neither a zero row nor
    /// a zero column, so a drain that stepped rows by the tile's own width, or
    /// dropped an origin, lands on poison the host can name. And `column`
    /// arrives as a kernel argument rather than a constant, which is what lets
    /// one launch per rung of the access ladder run the same code.
    ///
    /// The tile is filled by four warps and read by all of them, so the one
    /// obligation the drain does not carry — every warp's `stmatrix` visible to
    /// every warp's reads — is the `sync_threads` below and nothing else. There
    /// is no `fence.proxy.async.shared::cta` in this case at all: both ends are
    /// generic-proxy, which is exactly what distinguishes this path from the
    /// store ring's.
    ///
    /// What it does not establish: that the *width* the ladder chose was the
    /// widest legal one. A narrower access at a wide-enough column would pass
    /// this case unchanged; only `access_width`'s host tests say which rung a
    /// cursor gets. Launch with [`DRAIN_THREADS`].
    #[inline(always)]
    ///
    /// `ACCUMULATE` swaps the drain for
    /// [`kittens::global::accumulate_shared_rows`], which is the same rectangle
    /// by the same addresses at the same widths, folded in rather than
    /// overwritten. Everything the store case proves about the geometry the
    /// accumulate case proves again, and it additionally reads `C`: a fold that
    /// dropped the read leaves the destination's seed nowhere in the answer.
    /// One warp's band of the drain probes, every register holding the
    /// identity of the tile position it owns.
    ///
    /// Shared by the bf16 and fp32 probes so the two fill a band from the same
    /// [`RegTile::coordinate`] and a disagreement between them is the drain's.
    ///
    /// Both walks are fully unrolled for the reason every mover in the library
    /// is (ferro #166): a rolled walk over a `RegTile` keeps its indices
    /// dynamic, SROA cannot split the aggregate, and the whole band is homed to
    /// a `.local` depot. That depot is the *fill's* and would sit in every one
    /// of these probes' `regcount` rows, which is exactly what stops the drains
    /// below being comparable to each other.
    #[inline(always)]
    fn drain_identities<const C: usize>(row_base: u32, lane: u32) -> DrainBand<C>
    where
        BaseLdtm: FragmentLayout<DRAIN_WARP_ROWS, C>,
    {
        let mut values = DrainBand::<C>::zero();
        let mut slot = 0usize;
        while slot < const { DrainBand::<C>::SLOTS } {
            __unroll_config::<0>();
            let mut value = 0usize;
            while value < const { DrainBand::<C>::VALUES } {
                __unroll_config::<0>();
                let (row, tile_column) = DrainBand::<C>::coordinate(lane, slot, value);
                values.set(
                    slot,
                    value,
                    cell((row_base + row) as usize, tile_column as usize),
                );
                value += 1;
            }
            slot += 1;
        }
        values
    }

    #[inline(always)]
    unsafe fn shared_drain_probe<const C: usize, const ACCUMULATE: bool>(
        column: u32,
        out: &mut DisjointSlice<u16>,
    ) where
        BaseLdtm: FragmentLayout<DRAIN_WARP_ROWS, C>,
    {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::<DRAIN_ROWS, C>::from_raw(smem);
            let lane = warp::lane_id();
            let row_base = DRAIN_WARP_ROWS as u32 * warp::warp_id();

            let values = drain_identities::<C>(row_base, lane);
            store_tile(tile.chunk_writer(), row_base, 0, lane, values);
            thread::sync_threads();

            let destination = GlobalRows::<Bf16>::from_slice(out, DRAIN_PITCH);
            let thread = thread::threadIdx_x();
            if ACCUMULATE {
                accumulate_shared_rows::<Bf16, DRAIN_ROWS, C, Swizzle128B, DRAIN_THREADS>(
                    destination,
                    DRAIN_ROW,
                    column,
                    thread,
                    tile,
                );
            } else {
                store_shared_rows::<Bf16, DRAIN_ROWS, C, Swizzle128B, DRAIN_THREADS>(
                    destination,
                    DRAIN_ROW,
                    column,
                    thread,
                    tile,
                );
            }
        }
    }

    /// [`shared_drain_probe`] over one subtile.
    #[kernel]
    pub unsafe fn shared_drain(column: u32, mut out: DisjointSlice<u16>) {
        unsafe { shared_drain_probe::<TILE, false>(column, &mut out) }
    }

    /// [`shared_drain_probe`] over two stacked subtiles: chunks 8.. of every
    /// logical row live a `SUBTILE_BYTES` stride away in shared memory and
    /// eight elements along in global memory, which is the one place the two
    /// sides' notions of "next" disagree.
    #[kernel]
    pub unsafe fn shared_drain_wide(column: u32, mut out: DisjointSlice<u16>) {
        unsafe { shared_drain_probe::<WIDE, false>(column, &mut out) }
    }

    /// [`shared_drain`]'s rectangle folded into its destination instead of
    /// overwriting it.
    #[kernel]
    pub unsafe fn shared_accumulate(column: u32, mut out: DisjointSlice<u16>) {
        unsafe { shared_drain_probe::<TILE, true>(column, &mut out) }
    }

    /// [`shared_drain_wide`]'s rectangle folded in — the stacked-subtile stride
    /// on the accumulating path.
    #[kernel]
    pub unsafe fn shared_accumulate_wide(column: u32, mut out: DisjointSlice<u16>) {
        unsafe { shared_drain_probe::<WIDE, true>(column, &mut out) }
    }

    /// [`shared_drain_probe`] at **fp32**, where `stmatrix` cannot go: the band
    /// reaches the staging tile through
    /// [`kittens::ldst::scatter_tile`] and leaves it through the same
    /// [`kittens::global::store_shared_rows`] (#174).
    ///
    /// `STAGED` is the whole comparison. False is the route an fp32 epilogue
    /// had before this — [`kittens::global::store_rows`] straight out of the
    /// registers, no shared hop and no barrier — and it writes the same
    /// rectangle by the same arithmetic, so one host check holds both and a
    /// difference between them is the staging tile's alone. The two are also
    /// the register-pressure A/B the `regcount` table reads: same band, same
    /// destination, one pass through shared memory.
    ///
    /// The identities are [`cell`]s, whose low 16 bits are zero, so they are
    /// exact at fp32 as they are at bf16 and the comparison is still on bit
    /// patterns with no tolerance in it.
    ///
    /// Nothing here needs a proxy fence, for [`shared_drain_probe`]'s reason
    /// twice over: `scatter_tile` is an ordinary `st.shared` and
    /// `store_shared_rows` an ordinary `ld.shared`, both generic-proxy, so the
    /// `sync_threads` between them is the whole of what the four warps owe each
    /// other. Launch with [`DRAIN_THREADS`].
    #[inline(always)]
    unsafe fn scatter_drain_probe<const C: usize, const STAGED: bool>(
        column: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<DRAIN_WARP_ROWS, C>,
    {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = F32Tile::<DRAIN_ROWS, C>::from_raw(smem);
            let lane = warp::lane_id();
            let row_base = DRAIN_WARP_ROWS as u32 * warp::warp_id();

            let values = drain_identities::<C>(row_base, lane);
            let destination = GlobalRows::<F32>::from_slice(out, DRAIN_PITCH);
            if STAGED {
                scatter_tile(tile.chunk_writer(), row_base, 0, lane, values);
                thread::sync_threads();
                store_shared_rows::<F32, DRAIN_ROWS, C, Swizzle128B, DRAIN_THREADS>(
                    destination,
                    DRAIN_ROW,
                    column,
                    thread::threadIdx_x(),
                    tile,
                );
            } else {
                store_rows(destination, DRAIN_ROW + row_base, column, lane, values);
            }
        }
    }

    /// [`scatter_drain_probe`] over one subtile — 32 fp32 columns is exactly
    /// one 128-byte swizzle atom, so the cursor never leaves it.
    #[kernel]
    pub unsafe fn scatter_drain(column: u32, mut out: DisjointSlice<f32>) {
        unsafe { scatter_drain_probe::<F32_TILE, true>(column, &mut out) }
    }

    /// [`scatter_drain_probe`] over four stacked subtiles: at four bytes an
    /// element a 128-column row crosses a [`SharedTile::SUBTILE_BYTES`] stride
    /// three times, where the bf16 tile of the same width crosses it once.
    #[kernel]
    pub unsafe fn scatter_drain_wide(column: u32, mut out: DisjointSlice<f32>) {
        unsafe { scatter_drain_probe::<F32_WIDE, true>(column, &mut out) }
    }

    /// [`scatter_drain`]'s rectangle by the register route — the control, and
    /// what an fp32 epilogue had to use before #174.
    #[kernel]
    pub unsafe fn register_drain(column: u32, mut out: DisjointSlice<f32>) {
        unsafe { scatter_drain_probe::<F32_TILE, false>(column, &mut out) }
    }

    /// [`scatter_drain_wide`]'s rectangle by the register route.
    #[kernel]
    pub unsafe fn register_drain_wide(column: u32, mut out: DisjointSlice<f32>) {
        unsafe { scatter_drain_probe::<F32_WIDE, false>(column, &mut out) }
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
                tma.expect_tx(tile.tma_load(source, 0, 0, tma));
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
                tma.expect_tx(tile.tma_load(source, 0, 0, tma));
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

    /// Read a `[32, WIDE]` band out of an ordinary pitched fp32 matrix with
    /// [`load_rows`] and dump it by thread coordinate.
    ///
    /// **What this proves.** There is no descriptor, no swizzle and no packing
    /// anywhere in this chain — the source is a plain row-major buffer the host
    /// staged, and every element of it carries its own flat index. The band's
    /// origin is neither the buffer's corner nor a 16-column block boundary,
    /// and the matrix's leading dimension is not the band's width, so a value
    /// arriving in the register the host expects means `load_rows` walked all
    /// three of those independently. Nothing else in the harness reaches global
    /// memory without the TMA engine.
    ///
    /// The *store* direction is covered by `gemm` in the examples crate rather
    /// than by a case here: its epilogue is `store_rows` at a runtime `ldc`
    /// wider than the band, at a non-zero column origin, across four warps, and
    /// every one of its `512 * 256` fp32 outputs is compared for equality
    /// against a CPU reference. A case here could only restate that with fewer
    /// warps.
    ///
    /// Launch with one warp: a band belongs to one, and its 32 lanes between
    /// them own every element of it. Not because the mover is collective —
    /// alone among the movers here it issues no warp-wide instruction, so a
    /// missing lane costs its own values and nothing else.
    #[kernel]
    pub unsafe fn global_rows_map(source: &[f32], mut out: DisjointSlice<f32>) {
        unsafe {
            let lane = warp::lane_id();
            // SAFETY: read-only — `load_rows` is the only thing issued on this
            // cursor, which is what `GlobalRows::from_raw` asks of a caller
            // that reached it by casting away a shared reference.
            let rows = GlobalRows::<F32>::from_raw(source.as_ptr().cast_mut().cast(), GLOBAL_PITCH);
            let band: RegTile<32, WIDE, BaseLdtm> =
                load_rows(rows, GLOBAL_ROW, GLOBAL_COLUMN, lane);
            dump_band(band, 0, lane, &mut out);
        }
    }

    /// Read one row of the same matrix as a [`ColVec`] with [`load_cols`] and
    /// dump it by lane and value.
    ///
    /// **What this proves.** A per-column operand reaches registers from global
    /// memory at exactly the columns `BaseLdtm::column` names — off a row that
    /// is not the buffer's first and a column origin that is neither a
    /// 16-column block boundary nor the row's start, so a value arriving where
    /// the host expects means `load_cols` walked the row origin and the column
    /// map independently. `GLOBAL_COLUMN` is even and the pitch is too, so this
    /// takes the paired `ld.global.v2` arm; the scalar arm is
    /// [`load_rows`]' own and covered by `global rows map`.
    ///
    /// Launch with one warp, for [`global_rows_map`]'s reason.
    #[kernel]
    pub unsafe fn global_cols_map(source: &[f32], mut out: DisjointSlice<f32>) {
        unsafe {
            let lane = warp::lane_id();
            // SAFETY: read-only, as `global_rows_map`.
            let rows = GlobalRows::<F32>::from_raw(source.as_ptr().cast_mut().cast(), GLOBAL_PITCH);
            let cols: ColVec<WIDE, BaseLdtm> = load_cols(rows, GLOBAL_ROW, GLOBAL_COLUMN, lane);
            let values = ColVec::<WIDE, BaseLdtm>::VALUES;
            let mut value = 0usize;
            while value < values {
                *out.get_unchecked_mut(lane as usize * values + value) = cols.get(value);
                value += 1;
            }
        }
    }

    /// Read a `[COLUMN_BAND]` statistic *down* one column of the same pitched
    /// matrix with [`load_col_vec`] and dump it by thread coordinate.
    ///
    /// **What this proves, that the case above cannot.** `load_rows` walks a
    /// rectangle: consecutive values of a thread are consecutive elements, and
    /// a mover that confused the two axes would still deliver something. This
    /// walks one column, so consecutive values are a *pitch* apart and land on
    /// the tile's column axis — the shape an attention backward pass needs
    /// when its score band is `K·Qᵀ` and the saved per-query statistic is
    /// therefore per-column. A mover that read along the row instead would
    /// deliver `GLOBAL_COLUMN + value`'s elements and every one of them names
    /// itself, so the failure reports which walk actually ran.
    ///
    /// One warp, for [`kernels::global_rows_map`]'s reason.
    #[kernel]
    pub unsafe fn global_col_vec_map(source: &[f32], mut out: DisjointSlice<f32>) {
        unsafe {
            let lane = warp::lane_id();
            // SAFETY: read-only, as in `global_rows_map`.
            let rows = GlobalRows::<F32>::from_raw(source.as_ptr().cast_mut().cast(), GLOBAL_PITCH);
            let stat: ColVec<COLUMN_BAND, BaseLdtm> =
                load_col_vec(rows, GLOBAL_ROW, GLOBAL_COLUMN, lane);
            let values = <BaseLdtm as ColLayout<COLUMN_BAND>>::VALUES;
            let mut value = 0usize;
            while value < values {
                *out.get_unchecked_mut(lane as usize * values + value) = stat.get(value);
                value += 1;
            }
        }
    }

    /// What the global ↔ register movers cost against the index math they
    /// delete (#11), at both probe widths.
    ///
    /// Each step copies one `[32, N]` rectangle of a pitched fp32 matrix into
    /// the same rectangle of another — a GEMM epilogue with the MMA taken out,
    /// and the liveness that decides register pressure (#38) is the one a real
    /// epilogue has: a single band, live from where it is defined to where it
    /// is stored.
    ///
    /// `pitch` is a runtime parameter for the reason `gemm.rs` has `ldc` as
    /// one: a destination's leading dimension is not known when the kernel is
    /// compiled, and a probe that folded it would be pricing a store no
    /// epilogue can issue.
    #[inline(always)]
    unsafe fn global_copy_probe<const N: usize, const FORM: u32>(
        source: &[f32],
        steps: u32,
        pitch: u32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<32, N>,
    {
        unsafe {
            let slots = RegTile::<32, N, BaseLdtm>::SLOTS;
            let values = RegTile::<32, N, BaseLdtm>::VALUES;
            let lane = warp::lane_id();

            let mut step = 0u32;
            while step < steps {
                let column = N as u32 * step;
                let band = if FORM == MOVERS {
                    load_rows(
                        GlobalRows::<F32>::from_raw(
                            source.as_ptr().cast_mut().cast(),
                            pitch as usize,
                        ),
                        0,
                        column,
                        lane,
                    )
                } else {
                    let mut band = RegTile::<32, N, BaseLdtm>::zero();
                    let mut slot = 0usize;
                    while slot < slots {
                        let mut value = 0usize;
                        while value < values {
                            let (row, own) =
                                RegTile::<32, N, BaseLdtm>::coordinate(lane, slot, value);
                            let index = row as usize * pitch as usize + (column + own) as usize;
                            band.set(slot, value, *source.get_unchecked(index));
                            value += 1;
                        }
                        slot += 1;
                    }
                    band
                };

                if FORM == MOVERS {
                    store_rows(
                        GlobalRows::<F32>::from_slice(out, pitch as usize),
                        0,
                        column,
                        lane,
                        band,
                    );
                } else {
                    let mut slot = 0usize;
                    while slot < slots {
                        let mut value = 0usize;
                        while value < values {
                            let (row, own) =
                                RegTile::<32, N, BaseLdtm>::coordinate(lane, slot, value);
                            let index = row as usize * pitch as usize + (column + own) as usize;
                            *out.get_unchecked_mut(index) = band.get(slot, value);
                            value += 1;
                        }
                        slot += 1;
                    }
                }
                step += 1;
            }
        }
    }

    /// [`global_copy_probe`] at a 32-wide band, open-coded.
    #[kernel]
    pub unsafe fn global_copy_probe_32_open_coded(
        source: &[f32],
        steps: u32,
        pitch: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { global_copy_probe::<32, OPEN_CODED>(source, steps, pitch, &mut out) }
    }

    /// [`global_copy_probe`] at 32 wide, through the movers.
    #[kernel]
    pub unsafe fn global_copy_probe_32_movers(
        source: &[f32],
        steps: u32,
        pitch: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { global_copy_probe::<32, MOVERS>(source, steps, pitch, &mut out) }
    }

    /// [`global_copy_probe`] at the epilogue width both examples use,
    /// open-coded.
    #[kernel]
    pub unsafe fn global_copy_probe_128_open_coded(
        source: &[f32],
        steps: u32,
        pitch: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { global_copy_probe::<128, OPEN_CODED>(source, steps, pitch, &mut out) }
    }

    /// [`global_copy_probe`] at 128 wide, through the movers.
    #[kernel]
    pub unsafe fn global_copy_probe_128_movers(
        source: &[f32],
        steps: u32,
        pitch: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { global_copy_probe::<128, MOVERS>(source, steps, pitch, &mut out) }
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

    /// [`sttm_roundtrip`] draining through `tcgen05.ld.16x256b.x8` instead of
    /// `.x1` — the case that pins **the order the wide load's 32 registers
    /// arrive in** (#117).
    ///
    /// `TmemTile::tile_x8` gets four `[16, 16]` blocks out of one instruction
    /// by asserting the register list is *repeat-major*: repeat `r` of the
    /// `.x8` is what a `.x1` at `column + 8r` would have returned, so block `j`
    /// is repeats `2j` and `2j + 1`. **No document at this pin says that.**
    /// `intrinsics/generated-reference.md` validates the encoding and the
    /// register *count*; the mapping from those 32 registers to `(row, column)`
    /// is silicon's, and upstream's own evidence table marks the intrinsic
    /// `runtime: unexecuted`.
    ///
    /// So this is the assertion. The seed and the dump are byte-for-byte
    /// [`sttm_roundtrip`]'s — the same `dump_index` in, the same
    /// `observed[i] == i` out — and the *only* thing that differs is which
    /// drain reads it back. A wide load whose registers arrive in any other
    /// order returns a permutation, and the host names both the coordinate that
    /// should own each value and the one that actually wrote it.
    ///
    /// It is a composition, like the case above: `sttm round trip` pins `.x1`
    /// against the store, and this pins `.x8` against the same store, so the
    /// two together say the wide and narrow drains agree. Neither faults if it
    /// is wrong — a `gemm` would simply compute a wrong `C` — which is why the
    /// claim is worth a case rather than a comment.
    ///
    /// Launch with `ROWS` threads, as [`sttm_roundtrip`].
    #[kernel]
    pub unsafe fn ldtm_x8_map(mut out: DisjointSlice<f32>) {
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
                band.tile_x8::<32, COLUMNS>(32 * warp_id, 0),
                warp_id,
                lane,
                &mut out,
            );

            thread::sync_threads();
            dealloc_block(tmem, COLUMNS as u32);
        }
    }

    /// [`ldtm_x8_map`] with all four of the band's `.x8` issues in flight
    /// before the one wait — `TmemTile::tile_x8_batched`.
    ///
    /// Batching moves the wait and nothing else, so the tile it returns must be
    /// the tile [`ldtm_x8_map`] returns, which must be the tile
    /// [`sttm_roundtrip`] returns. What it *could* get wrong is the arrival
    /// order: four loads outstanding at once, resolved after a single
    /// `tcgen05.wait::ld`, is a shape no document at this pin describes, and a
    /// wait that retired only the last issue would return three stale groups.
    /// The seed and the dump are byte-for-byte the two cases above's, so
    /// `observed[i] == i` says both that the wait covers every issue and that
    /// the register order survived the batch.
    ///
    /// Launch with `ROWS` threads, as [`sttm_roundtrip`].
    #[kernel]
    pub unsafe fn ldtm_x8_batched_map(mut out: DisjointSlice<f32>) {
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
                band.tile_x8_batched::<32, COLUMNS, 4>(32 * warp_id, 0),
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

    /// One rung of the TMEM occupancy ladder (#74): allocate `columns` of
    /// tensor memory, publish the address so nothing can fold the allocation
    /// away, give the columns back.
    ///
    /// Deliberately the *smallest* kernel that holds a tcgen05 allocation.
    /// Everything an occupancy query could otherwise blame is held fixed
    /// across the ladder and matched by [`tmem_ladder_none`]: the same 32-byte
    /// shared plan, the same block sync, the same single store. The one thing
    /// that varies from rung to rung is the immediate operand of
    /// `tcgen05.alloc`.
    #[inline(always)]
    unsafe fn tmem_ladder_probe(columns: u32, out: &mut DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tmem = alloc_block(smem as *mut u32, columns);
            if thread::threadIdx_x() == 0 {
                *out.get_unchecked_mut(0) = tmem;
            }
            thread::sync_threads();
            dealloc_block(tmem, columns);
        }
    }

    /// The ladder's control: [`tmem_ladder_probe`]'s shape with the allocator
    /// removed, and nothing else changed.
    ///
    /// Warp 0 writes the staging word, a block sync publishes it, every thread
    /// reads it back — which is `alloc_block`'s body with `tcgen05.alloc` and
    /// the permit relinquish taken out. Whatever this rung answers is what a
    /// kernel of this size and shape is worth on an SM *without* tcgen05, so
    /// the gap between it and the rungs above is attributable to tcgen05 alone.
    #[kernel]
    pub unsafe fn tmem_ladder_none(mut out: DisjointSlice<u32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            if warp::warp_id() == 0 {
                *(smem as *mut u32) = TMEM_LADDER_SENTINEL;
            }
            thread::sync_threads();
            let staged = *(smem as *const u32);
            if thread::threadIdx_x() == 0 {
                *out.get_unchecked_mut(0) = staged;
            }
            thread::sync_threads();
        }
    }

    /// [`tmem_ladder_probe`] at the allocator's smallest unit.
    #[kernel]
    pub unsafe fn tmem_ladder_32(mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(32, &mut out) }
    }

    /// [`tmem_ladder_probe`] at 64 columns.
    #[kernel]
    pub unsafe fn tmem_ladder_64(mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(64, &mut out) }
    }

    /// [`tmem_ladder_probe`] at 128 columns — a quarter of the SM's tensor
    /// memory, and `sttm_roundtrip`'s allocation.
    #[kernel]
    pub unsafe fn tmem_ladder_128(mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(128, &mut out) }
    }

    /// [`tmem_ladder_probe`] at 256 columns.
    #[kernel]
    pub unsafe fn tmem_ladder_256(mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(256, &mut out) }
    }

    /// [`tmem_ladder_probe`] at the whole SM's tensor memory.
    #[kernel]
    pub unsafe fn tmem_ladder_512(mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(512, &mut out) }
    }

    /// [`tmem_ladder_probe`] with the column count arriving as a kernel
    /// argument rather than as an immediate.
    ///
    /// This separates two mechanisms a flat ladder cannot: a driver reading a
    /// column count `ptxas` recorded, and a driver that reserves tensor memory
    /// for any kernel that touches the allocator at all. Only the first can
    /// see a constant, and this rung has none to see.
    #[kernel]
    pub unsafe fn tmem_ladder_runtime(columns: u32, mut out: DisjointSlice<u32>) {
        unsafe { tmem_ladder_probe(columns, &mut out) }
    }

    /// Occupy this CTA's SM until the device's global nanosecond timer passes
    /// `deadline`, and return the time it actually stopped at.
    ///
    /// A *wall-clock* hold rather than a fixed amount of work, which is what
    /// makes the census readable: every CTA occupies its SM for the same
    /// interval no matter how many co-residents are competing with it, so
    /// overlapping intervals count residency and not contention.
    ///
    /// [`CENSUS_SPIN_GUARD`] bounds the loop. `%globaltimer` is an intrinsic
    /// read and a spin whose only exit is a value the compiler might hoist
    /// would hang a device that costs money; the guard makes it terminate on
    /// any hardware, and a CTA that ended on it reports a short interval the
    /// host is required to notice.
    #[inline(always)]
    fn census_spin(deadline: u64) -> u64 {
        let mut guard = 0u64;
        let mut now = debug::globaltimer();
        while now < deadline && guard < CENSUS_SPIN_GUARD {
            guard += 1;
            now = debug::globaltimer();
        }
        now
    }

    /// Publish one CTA's census row: which SM it ran on, and the three times
    /// that bound its residency and its hold on tensor memory.
    #[inline(always)]
    unsafe fn census_record(
        sm: u32,
        entered: u64,
        allocated: u64,
        left: u64,
        out: &mut DisjointSlice<u64>,
    ) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                let base = CENSUS_FIELDS * thread::blockIdx_x() as usize;
                *out.get_unchecked_mut(base) = sm as u64;
                *out.get_unchecked_mut(base + 1) = entered;
                *out.get_unchecked_mut(base + 2) = allocated;
                *out.get_unchecked_mut(base + 3) = left;
            }
        }
    }

    /// Mark this CTA as having run at all, before it waits on anything.
    ///
    /// Without this the probe cannot state its own result. A cancelled cluster
    /// is **never launched**, so its rows keep the zeros the buffer was
    /// allocated with — which is byte-for-byte what a launched rank that never
    /// saw its response would write. The first read of this case conflated the
    /// two and reported 3758 of 8192 CTAs as stalled when most of them were
    /// simply stolen, which is the outcome it was built to observe.
    #[inline(always)]
    unsafe fn clc_entered(out: &mut DisjointSlice<u64>) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                *out.get_unchecked_mut(CLC_FIELDS * thread::blockIdx_x() as usize) = 1;
            }
        }
    }

    /// One row of the CLC probe (#88), per CTA:
    /// `[launched, completed, is_canceled, first_ctaid_x]`.
    ///
    /// `completed` is the whole reason this case exists. A steal's response
    /// arrives on an mbarrier, and `pipeline::run_stealing` has *every* rank
    /// wait on its own copy — which is only sound if
    /// `.multicast::cluster::all` really does complete the transaction in every
    /// CTA of the requesting cluster and not just in the one that issued. If it
    /// does not, the peer waits forever, and a persistent GEMM built on it
    /// hangs with no diagnostic at all. So this probe waits on a **deadline**
    /// rather than on the barrier: a rank whose barrier never flips writes
    /// `completed = 0` and lives to report it, which turns a deadlock into a
    /// row of a table.
    ///
    /// The other three answer the questions a hang would otherwise hide behind:
    /// whether the cancelled unit is a cluster or a CTA (`first_ctaid_x` even,
    /// under a 2-CTA cluster), and whether both ranks of a cluster are told the
    /// *same* thing — which is what stops #51's split pair coming back.
    #[inline(always)]
    unsafe fn clc_record(completed: u64, canceled: u64, first: u64, out: &mut DisjointSlice<u64>) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                let base = CLC_FIELDS * thread::blockIdx_x() as usize;
                *out.get_unchecked_mut(base + 1) = completed;
                *out.get_unchecked_mut(base + 2) = canceled;
                *out.get_unchecked_mut(base + 3) = first;
            }
        }
    }

    /// Ask the hardware for one stolen cluster and write down what came back,
    /// against a deadline — the instrument behind #88's scheduler.
    ///
    /// Deliberately built out of the **raw** `cuda_device::clc` and mbarrier
    /// intrinsics rather than `kittens::pipeline::ClcQueue`. What is under test
    /// here is the instruction and its lowering, so putting the library's
    /// wrapper in the path would make a failure ambiguous between the two.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn clc_probe(deadline_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe {
            use cuda_device::barrier::{
                mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait_parity,
            };
            use cuda_device::clc::{
                clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
            };

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let response = smem as *mut u64;
            let bar = smem.add(CLC_RESPONSE_BYTES) as *mut Barrier;
            let leader = thread::threadIdx_x() == 0;

            clc_entered(&mut out);
            if leader {
                mbarrier_init(bar, 1);
            }
            // Every rank's barrier has to exist before any rank asks the
            // hardware to complete transactions on it.
            thread::sync_threads();
            cluster::cluster_sync();

            // Each rank charges its own barrier; one thread of the cluster
            // issues. Nothing orders the two, and nothing has to — the
            // transaction count is a signed accumulator.
            if leader {
                mbarrier_arrive_expect_tx(bar, 1, CLC_RESPONSE_BYTES as u32);
            }
            if leader && cluster::block_rank() == 0 {
                clc_try_cancel_multicast(response as *mut u8, bar);
            }

            let deadline = debug::globaltimer() + deadline_ns;
            let mut completed = 0u64;
            while debug::globaltimer() < deadline {
                if mbarrier_try_wait_parity(bar, 0) {
                    completed = 1;
                    break;
                }
            }

            let (mut canceled, mut first) = (u64::MAX, u64::MAX);
            if completed == 1 {
                let (low, high) = (response.read_volatile(), response.add(1).read_volatile());
                canceled = clc_query_is_canceled(low, high) as u64;
                if canceled != 0 {
                    first = clc_query_get_first_ctaid_x(low, high) as u64;
                }
            }
            clc_record(completed, canceled, first, &mut out);
        }
    }

    /// One rung of the residency census (#78): hold `columns` of tensor memory
    /// for `hold_ns` nanoseconds and timestamp both ends of it, along with the
    /// SM the CTA landed on.
    ///
    /// `entered` is read *before* the allocator and `allocated` after it, so
    /// the two intervals a row carries answer two different questions.
    /// `[entered, left]` is how long the CTA held a slot on that SM, whatever
    /// it was doing with it; `[allocated, left]` is how long it held tensor
    /// memory. A CTA that is resident but parked inside a blocking
    /// `tcgen05.alloc` waiting for a peer to give columns back has a long
    /// first interval and a short second one, which is the one reading a
    /// throughput curve alone cannot distinguish from not being resident at
    /// all.
    #[inline(always)]
    unsafe fn census_probe(columns: u32, hold_ns: u64, out: &mut DisjointSlice<u64>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let sm = thread::smid();
            let entered = debug::globaltimer();
            let tmem = alloc_block(smem as *mut u32, columns);
            let allocated = debug::globaltimer();
            let left = census_spin(allocated + hold_ns);
            thread::sync_threads();
            dealloc_block(tmem, columns);
            census_record(sm, entered, allocated, left, out);
        }
    }

    /// The census control: [`census_probe`]'s shape with the allocator removed,
    /// as [`tmem_ladder_none`] is the occupancy ladder's.
    ///
    /// Whatever this answers is what an SM will admit of a kernel this size
    /// *without* tcgen05, so the gap between it and the rungs above is
    /// attributable to the allocator alone.
    #[kernel]
    pub unsafe fn residency_census_none(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let sm = thread::smid();
            let entered = debug::globaltimer();
            if warp::warp_id() == 0 {
                *(smem as *mut u32) = TMEM_LADDER_SENTINEL;
            }
            thread::sync_threads();
            let allocated = debug::globaltimer();
            let left = census_spin(allocated + hold_ns);
            thread::sync_threads();
            census_record(sm, entered, allocated, left, &mut out);
        }
    }

    /// [`census_probe`] at the allocator's smallest unit — 16 of these fit the
    /// SM's 512 columns.
    #[kernel]
    pub unsafe fn residency_census_32(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_probe(32, hold_ns, &mut out) }
    }

    /// [`census_probe`] at 64 columns; eight fit.
    #[kernel]
    pub unsafe fn residency_census_64(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_probe(64, hold_ns, &mut out) }
    }

    /// [`census_probe`] at **`gemm`'s** column count. Four fit, and #51 timed
    /// that kernel above two clusters an SM.
    #[kernel]
    pub unsafe fn residency_census_128(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_probe(128, hold_ns, &mut out) }
    }

    /// [`census_probe`] at **`flash_forward`'s** column count. Two fit, which
    /// is the number #78 exists to confirm or refute.
    #[kernel]
    pub unsafe fn residency_census_256(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_probe(256, hold_ns, &mut out) }
    }

    /// [`census_probe`] at the whole SM's tensor memory — the rung where one
    /// CTA an SM is *arithmetically* forced, and therefore the census's own
    /// positive control.
    #[kernel]
    pub unsafe fn residency_census_512(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_probe(512, hold_ns, &mut out) }
    }

    /// The rung that separates a driver's admission decision from the
    /// hardware's allocator: take all 512 columns, give them straight back,
    /// and only then hold the SM.
    ///
    /// The kernel still *contains* a `tcgen05.alloc`, so
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 1 for it exactly
    /// as it does for every other rung (#77). Nothing is held during the
    /// interval that gets counted. If the 1 is a static reservation made when
    /// the CTA is admitted, this rung is pinned at 1 like [`residency_census_512`];
    /// if it is the dynamic cost of holding columns, this rung is free like the
    /// control. Those are different mechanisms and no column sweep can tell
    /// them apart.
    #[kernel]
    pub unsafe fn residency_census_free(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let sm = thread::smid();
            let entered = debug::globaltimer();
            let tmem = alloc_block(smem as *mut u32, 512);
            thread::sync_threads();
            dealloc_block(tmem, 512);
            thread::sync_threads();
            let allocated = debug::globaltimer();
            let left = census_spin(allocated + hold_ns);
            census_record(sm, entered, allocated, left, &mut out);
        }
    }

    /// [`census_probe`]'s `cta_group::2` twin: the same census under
    /// [`alloc_cluster`], which is the allocation `gemm` actually issues and
    /// the one #77's control never covered.
    ///
    /// The pair's two CTAs land on two different SMs, so counting overlap per
    /// `%smid` still counts CTAs an SM — the census needs no special case for
    /// a cluster, only a grid that is a multiple of the cluster size.
    #[inline(always)]
    unsafe fn census_cluster_probe(columns: u32, hold_ns: u64, out: &mut DisjointSlice<u64>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let sm = thread::smid();
            let entered = debug::globaltimer();
            let tmem = alloc_cluster(smem as *mut u32, columns);
            let allocated = debug::globaltimer();
            let left = census_spin(allocated + hold_ns);
            thread::sync_threads();
            cluster::cluster_sync();
            dealloc_cluster(tmem, columns);
            census_record(sm, entered, allocated, left, out);
        }
    }

    /// [`census_cluster_probe`] at `gemm`'s own 128 columns.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn residency_census_cluster_128(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_cluster_probe(128, hold_ns, &mut out) }
    }

    /// [`census_cluster_probe`] at the 256 columns a `[256, 256]` pair tile
    /// allocates — #87's rung, and the first envelope in this repo where
    /// **tensor memory is the tighter of the two per-CTA resources**.
    ///
    /// Every `cta_group::2` rung so far has been capped by shared memory, so
    /// `512 / columns` has never actually bound anything and its half of the
    /// residency formula has never been tested where it decides. At 256
    /// columns it admits two, and a `[256, 256]` kernel two stages deep
    /// declares 65 608 B, which admits three. The prediction this makes is
    /// sharper than the count: the third CTA should be *admitted* and park
    /// inside `tcgen05.alloc`, so `resident` reads three where `holding` reads
    /// two — the first rung here where the census's two columns differ.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn residency_census_cluster_256(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_cluster_probe(256, hold_ns, &mut out) }
    }

    /// [`census_cluster_probe`] at the whole SM's tensor memory — the cluster
    /// side's positive control, and the rung that says whether a `cta_group::2`
    /// allocation is charged to each rank or split across the pair.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn residency_census_cluster_512(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe { census_cluster_probe(512, hold_ns, &mut out) }
    }

    /// A cluster launch with no allocator in it at all — the control for the
    /// two rungs above, so that whatever they cost is attributable to
    /// `tcgen05.alloc.cg2` rather than to being a cluster.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn residency_census_cluster_none(hold_ns: u64, mut out: DisjointSlice<u64>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let sm = thread::smid();
            let entered = debug::globaltimer();
            if warp::warp_id() == 0 {
                *(smem as *mut u32) = TMEM_LADDER_SENTINEL;
            }
            thread::sync_threads();
            cluster::cluster_sync();
            let allocated = debug::globaltimer();
            let left = census_spin(allocated + hold_ns);
            thread::sync_threads();
            cluster::cluster_sync();
            census_record(sm, entered, allocated, left, &mut out);
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
                let staged = a.tma_load(a_map, 0, 0, tma)
                    + b_low.tma_load(b_map, 0, 0, tma)
                    + b_high.tma_load(b_map, 0, 1, tma);
                tma.expect_tx(staged);
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
                let staged = a.tma_load(a_map, 0, 0, tma)
                    + b_low.tma_load(b_map, 0, 0, tma)
                    + b_high.tma_load(b_map, 0, 1, tma);
                tma.expect_tx(staged);
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

    /// The staging every operand-order case shares: the two operand tiles at
    /// the head of the plan, the two barriers and the TMEM staging word in
    /// its 32-byte tail, one TMA per tile charged once, and the CTA
    /// synchronized behind the arrival.
    ///
    /// Returns both tiles, the accumulator's TMEM base, and both semaphores —
    /// the second for the issuing thread to commit its chain to, the first
    /// only so [`walk_drain`] can invalidate it.
    #[inline(always)]
    unsafe fn walk_stage<const AR: usize, const AC: usize, const BR: usize, const BC: usize>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
    ) -> (Tile<AR, AC>, Tile<BR, BC>, u32, Semaphore, Semaphore) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let a = Tile::<AR, AC>::from_raw(smem);
            let b = Tile::<BR, BC>::from_raw(smem.add(Tile::<AR, AC>::BYTES));
            let scratch = smem.add(Tile::<AR, AC>::BYTES + Tile::<BR, BC>::BYTES);
            let tma = Semaphore::attach(scratch as *mut Barrier);
            let mma_done = Semaphore::attach(scratch.add(8) as *mut Barrier);
            let tmem_slot = scratch.add(16) as *mut u32;

            let leader = thread::threadIdx_x() == 0;
            if leader {
                tma.init(1);
                mma_done.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(tmem_slot, COLUMNS as u32);

            if leader {
                tma.expect_tx(a.tma_load(a_map, 0, 0, tma) + b.tma_load(b_map, 0, 0, tma));
            }
            tma.wait(0);
            thread::sync_threads();
            (a, b, tmem, tma, mma_done)
        }
    }

    /// The other half of [`walk_stage`]: wait out the committed MMA chain,
    /// drain a warp's 32 accumulator rows, dump them by `(warp, lane, slot,
    /// value)`, and give the allocation back.
    #[inline(always)]
    unsafe fn walk_drain(
        tmem: u32,
        tma: Semaphore,
        mma_done: Semaphore,
        out: &mut DisjointSlice<f32>,
    ) {
        unsafe {
            mma_done.wait(0);
            thread::sync_threads();

            let warp_id = warp::warp_id();
            let accumulator = Accumulator::from_raw(tmem);
            let band = accumulator.tile::<32, COLUMNS>(32 * warp_id, 0);
            dump_band(band, warp_id, warp::lane_id(), out);

            thread::sync_threads();
            dealloc_block(tmem, COLUMNS as u32);
            if thread::threadIdx_x() == 0 {
                tma.inval();
                mma_done.inval();
            }
        }
    }

    /// `D = A·Bᵀ` with both operands K-major — the walk the fragment probe
    /// already issues, here against a reference that varies along *K* rather
    /// than one that collapses to a single chunk. Launch with `ROWS` threads
    /// and [`WALK_SHARED`] bytes, as all five operand-order cases are.
    #[kernel]
    pub unsafe fn walk_abt(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let (a, b, tmem, tma, done) = walk_stage::<ROWS, DEPTH, COLUMNS, DEPTH>(a_map, b_map);
            if thread::threadIdx_x() == 0 {
                mm_abt(tmem, a, b, MmaShape::M128_N128);
                mma::commit(done);
            }
            walk_drain(tmem, tma, done, &mut out);
        }
    }

    /// `D = A·B` — `A` K-major, `B` MN-major. The banding walk: one `N = 64`
    /// instruction per stacked `B` subtile, into `tmem + 64 * subtile`.
    #[kernel]
    pub unsafe fn walk_ab(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let (a, b, tmem, tma, done) = walk_stage::<ROWS, DEPTH, DEPTH, COLUMNS>(a_map, b_map);
            if thread::threadIdx_x() == 0 {
                mm_ab(tmem, a, b, MmaShape::M128_N64);
                mma::commit(done);
            }
            walk_drain(tmem, tma, done, &mut out);
        }
    }

    /// `D = Aᵀ·B` — both operands MN-major, both reaching their second
    /// stacked subtile through the descriptor's leading offset, in one
    /// `M128_N128` instruction per K chunk.
    #[kernel]
    pub unsafe fn walk_atb(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let (a, b, tmem, tma, done) = walk_stage::<DEPTH, ROWS, DEPTH, COLUMNS>(a_map, b_map);
            if thread::threadIdx_x() == 0 {
                mm_atb(tmem, a, b, MmaShape::M128_N128);
                mma::commit(done);
            }
            walk_drain(tmem, tma, done, &mut out);
        }
    }

    /// `D = Aᵀ·Bᵀ` — `A` MN-major, `B` K-major.
    #[kernel]
    pub unsafe fn walk_atbt(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let (a, b, tmem, tma, done) = walk_stage::<DEPTH, ROWS, COLUMNS, DEPTH>(a_map, b_map);
            if thread::threadIdx_x() == 0 {
                mm_atbt(tmem, a, b, MmaShape::M128_N128);
                mma::commit(done);
            }
            walk_drain(tmem, tma, done, &mut out);
        }
    }

    /// **The positive control.** [`walk_atb`]'s operands, its accumulator
    /// shape, its dump — and the *untransposed* walk. `mma_abt` reads the two
    /// `[K, M]` / `[K, N]` tiles as though K ran along their rows, which is
    /// the failure a transposed walk that quietly dropped its transpose bits
    /// would have; the host requires this dump to disagree with the reference
    /// [`walk_atb`] matches. Without it the transposed cases prove only that
    /// *some* walk computes `Aᵀ·B`, not that this one had to be transposed.
    ///
    /// The shape is [`CONTROL_EDGE`]-square and not the band's, because the
    /// untransposed reading of a `[64, 128]` tile is a `[64, 128]` K-major
    /// matrix: `M64_N64` over `K = 128` touches each tile's 16 KiB exactly
    /// once and nothing past it. The host compares only that quadrant; the
    /// rest of the accumulator is never written and is not read into any
    /// claim.
    #[kernel]
    pub unsafe fn walk_untransposed_control(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let (a, b, tmem, tma, done) = walk_stage::<DEPTH, ROWS, DEPTH, COLUMNS>(a_map, b_map);
            if thread::threadIdx_x() == 0 {
                mm_abt(tmem, a, b, MmaShape::M64_N64);
                mma::commit(done);
            }
            walk_drain(tmem, tma, done, &mut out);
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
    /// ops were extracted from. `FORM` picks which spelling of the correction
    /// and the accumulate is compiled and nothing else changes, so the five
    /// forms are comparable line for line:
    ///
    /// - [`HAND_WRITTEN`]: `max`/`sub`/`exp2`/`mul_assign` and `scale_rows`
    /// - [`FUSED`]: [`online_rescale`], one scalar factor live at a time
    /// - [`ROW_MAP`]: [`HAND_WRITTEN`] with `row_map::<Mul>` for `scale_rows`
    /// - [`ROW_MAP_ASSIGN`]: with `mul_row_assign`, #31's in-place form
    /// - [`ADD_ASSIGN`]: [`HAND_WRITTEN`] with the accumulate in place too
    ///
    /// **What it measured** (`regcount`, sm_100a, no spills in any form):
    ///
    /// ```text
    ///                    32 columns          128 columns
    ///                    regs   stack        regs   stack
    ///  hand_written        64     256         168    3072
    ///  fused               56     416         168    3104
    ///  row_map             64     256         255    2560
    ///  row_map_assign      64     256         168    3072
    ///  add_assign          64     256         168    2560
    /// ```
    ///
    /// #31's claim, and its correction. `row_map` costs 87 registers/thread at
    /// 128 columns and `row_map_assign` costs none — the in-place form reaches
    /// the hand-written loop exactly, at both widths, which is what let
    /// `scale_rows` become a wrapper over it. `add_assign` does *not* buy
    /// registers over the by-value `add` it replaces, because that call site
    /// rebinds and its input is already dead; what it buys there is 512 bytes
    /// of stack frame and a line that says what it means.
    ///
    /// `steps` is a runtime bound so the accumulators stay live across
    /// iterations rather than unrolling into one straight-line block.
    #[inline(always)]
    unsafe fn softmax_probe<const N: usize, const FORM: u32>(
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
                if FORM == FUSED {
                    online_rescale(&mut m_ref, row_max, &mut running_sum, &mut out_acc);
                } else {
                    let next = m_ref.max(row_max);
                    let factor = m_ref.sub(next).exp2();
                    m_ref = next;
                    running_sum.mul_assign(factor);
                    if FORM == ROW_MAP {
                        out_acc = out_acc.row_map::<Mul>(factor);
                    } else if FORM == ROW_MAP_ASSIGN {
                        out_acc.mul_row_assign(factor);
                    } else {
                        out_acc.scale_rows(factor);
                    }
                }

                let probabilities = block.sub_row(m_ref).exp2();
                running_sum.add_assign(probabilities.row_sum());
                if FORM == ADD_ASSIGN {
                    out_acc.add_assign(probabilities);
                } else {
                    out_acc = out_acc.add(probabilities);
                }
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

    /// [`softmax_probe`] at 32 wide, `mul_row_assign` for `scale_rows`.
    #[kernel]
    pub unsafe fn softmax_probe_32_row_map_assign(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<32, ROW_MAP_ASSIGN>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 32 wide, `add_assign` for the accumulate.
    #[kernel]
    pub unsafe fn softmax_probe_32_add_assign(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<32, ADD_ASSIGN>(scores, steps, &mut out) }
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

    /// [`softmax_probe`] at 128 wide, `mul_row_assign` for `scale_rows` — the
    /// pair against `softmax_probe_128_row_map` that #31 is scored on.
    #[kernel]
    pub unsafe fn softmax_probe_128_row_map_assign(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<128, ROW_MAP_ASSIGN>(scores, steps, &mut out) }
    }

    /// [`softmax_probe`] at 128 wide, `add_assign` for the accumulate — the
    /// call site `flash_forward` wanted.
    #[kernel]
    pub unsafe fn softmax_probe_128_add_assign(
        scores: &[f32],
        steps: u32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { softmax_probe::<128, ADD_ASSIGN>(scores, steps, &mut out) }
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
    /// - [`SCALE_ASSIGN`]: [`IN_PLACE`] as library API (#31)
    /// - [`CHAINED`]: [`SCALED`], but the block's scaled copy chained into the
    ///   accumulate instead of rebound, so both bands are live at the peak
    /// - [`ONE_EXPRESSION`]: neither multiply rebound, the whole step chained
    ///
    /// The first pair is #38's own question. [`IN_PLACE`] and
    /// [`SCALE_ASSIGN`] are #31's. The last two are the control that separates
    /// the two readings of all of it: [`CHAINED`] differs from [`SCALED`] in
    /// liveness alone, and [`ONE_EXPRESSION`] pushes that difference as far as
    /// the probe can.
    ///
    /// **What it measured** (`regcount`, sm_100a; every 32-column form is 48
    /// registers with no spill and no stack, so only 128 is tabulated):
    ///
    /// ```text
    ///                  regs   spill   stack
    ///  splatted         255     124    2648
    ///  scaled           255      60    2624
    ///  chained          255     108    2672
    ///  in_place         252       0    2560
    ///  scale_assign     252       0    2560
    ///  one_expression   168       0    2560
    /// ```
    ///
    /// Three things, and the third is not what #31 or #38 predicted.
    /// `scale_assign` reaches the hand-written in-place floor exactly, which
    /// is #31 delivering what it was filed for. [`CHAINED`] is worse than the
    /// [`SCALED`] it differs from by one rebinding, which is #38's 84-byte
    /// result reproduced (108 bytes here, on a probe that scales twice). And
    /// [`ONE_EXPRESSION`] — *strictly more* chained than [`CHAINED`], and so
    /// the most expensive form under the liveness reading — is the cheapest
    /// form in the table by 84 registers, and the only one under the cliff at
    /// all. Peak liveness does not order these. What does, across all six, is
    /// how many whole-tile temporaries have to be *materialized between
    /// statements*: one expression fuses into a single pass over the band and
    /// materializes none, the in-place forms materialize none but make two
    /// passes, and every rebinding form materializes one per statement.
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

                if FORM == ONE_EXPRESSION {
                    // Neither multiply rebinds its input: both scaled copies
                    // stay live beside the bands they came from until the add
                    // consumes them.
                    out_acc = out_acc.scale(k).add(block.scale(k));
                } else if FORM == CHAINED {
                    // [`SCALED`] with one line changed — the block's scale
                    // chained into the add instead of rebound, which is #38's
                    // own comparison and the only difference between the two
                    // forms.
                    out_acc = out_acc.scale(k);
                    out_acc = out_acc.add(block.scale(k));
                } else {
                    if FORM == SCALED {
                        block = block.scale(k);
                        out_acc = out_acc.scale(k);
                    } else if FORM == SPLATTED {
                        let widened = RegTile::<32, N, BaseLdtm>::splat(k);
                        block = block.bin_map::<Mul>(widened);
                        out_acc = out_acc.bin_map::<Mul>(widened);
                    } else if FORM == SCALE_ASSIGN {
                        block.scale_assign(k);
                        out_acc.scale_assign(k);
                    } else {
                        // Two passes, not one fused pass: the by-value forms
                        // above are two maps, and a floor that fuses them
                        // would be measuring the fusion instead of the
                        // operand.
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
                }
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

    /// [`scalar_map_probe`] at a 32-wide block, `RegTile::scale_assign`.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_scale_assign(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, SCALE_ASSIGN>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at a 32-wide block, the chained spelling.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_chained(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, CHAINED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at a 32-wide block, the whole step as one
    /// expression.
    #[kernel]
    pub unsafe fn scalar_map_probe_32_one_expression(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<32, ONE_EXPRESSION>(scores, steps, k, &mut out) }
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

    /// [`scalar_map_probe`] at 128 wide, `RegTile::scale_assign` — the library
    /// form of the line above it, and the pair #31 is scored on.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_scale_assign(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, SCALE_ASSIGN>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at 128 wide, chained — #38's 84-byte spill, as a
    /// form that lives in the tree.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_chained(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, CHAINED>(scores, steps, k, &mut out) }
    }

    /// [`scalar_map_probe`] at 128 wide, the whole step as one expression.
    #[kernel]
    pub unsafe fn scalar_map_probe_128_one_expression(
        scores: &[f32],
        steps: u32,
        k: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe { scalar_map_probe::<128, ONE_EXPRESSION>(scores, steps, k, &mut out) }
    }

    /// **A codegen probe, not a test.** [`scalar_map_probe`]'s step, swept
    /// over tile *shape* instead of measured at two widths — issue #60.
    ///
    /// Every register claim in this repo was measured at 32 and 128 columns,
    /// not because a width is expensive to measure (`ptxas` is a host compiler
    /// and the whole sweep is seconds of it) but because each width was a
    /// hand-written probe. `ladder!` generates them, so a rung costs one
    /// line, and what that buys is the shape of the curve rather than two
    /// points on it.
    ///
    /// The step is the flash inner loop's accumulate, in five spellings that
    /// compute the same tile — [`LADDER_FUSED`], [`LADDER_ASSIGN`],
    /// [`LADDER_REBOUND`], [`LADDER_OPEN_CODED`], [`LADDER_ALL_IN_PLACE`] — so
    /// a row of the table varies only in how many whole bands get materialized
    /// between statements, which is the axis #31 found orders register cost.
    /// The first four are [`scalar_map_probe`]'s own forms, line for line, and
    /// reproduce its two-width table where they meet it. `steps` is a runtime
    /// bound so the accumulator stays live across iterations.
    ///
    /// A thread of an `[M, N]` warp tile holds `M * N / 32` fp32 values —
    /// `SLOTS * VALUES` — and the row extent enters that count exactly as the
    /// column extent does. Whether it enters the *cost* the same way is what
    /// the row half of this ladder is here to ask, and it does not: see below.
    /// The ladder is a column sweep at the warp's 32 rows, plus a
    /// row sweep at two widths, chosen so that four rungs pair off at equal
    /// per-thread volume and different aspect ratio: `[16, 128]` against
    /// `[32, 64]`, `[64, 64]` against `[32, 128]`, `[48, 128]` against
    /// `[32, 192]`, `[64, 128]` against `[32, 256]`. `[32, 256]` and
    /// `[64, 128]` are 256 values a thread, one past the register file, and
    /// are in the ladder because a shape that spills is the informative one.
    ///
    /// `modal run modal_app.py::regcount` prints the sweep as one table and
    /// the per-spelling cliff — the first rung whose spill goes non-zero —
    /// under it; the ladder is written out once more there so a rung that
    /// stops building is reported as missing rather than quietly dropped.
    ///
    /// **What it measured** (`regcount`, sm_100a; registers/thread, spill
    /// store bytes and stack frame bytes; `per thread` is `M * N / 32`):
    ///
    /// ```text
    ///                      fused        assign    open_coded       rebound  all_in_place
    ///  [32,  16]   16       32/0          32/0          32/0          32/0          32/0
    ///  [32,  32]   32       48/0          48/0          48/0          48/0          48/0
    ///  [32,  48]   48       72/0          72/0          72/0          72/0          72/0
    ///  [32,  64]   64       94/0          96/0          96/0          96/0          39/0
    ///  [32,  96]   96      128/0          47/0          47/0         251/0          32/0
    ///  [32, 128]  128      168/0         252/0         252/0        255/60          32/0
    ///  [32, 192]  192    255/900      255/1012      255/1012      255/1096         251/0
    ///  [32, 256]  256   255/1704       255/976       255/976      255/2672         255/0
    ///  [16,  64]   32       56/0          56/0          56/0          55/0          56/0
    ///  [48,  64]   96      128/0          40/0          40/0         202/0          32/0
    ///  [64,  64]  128      162/0          32/0          32/0         168/0          32/0
    ///  [16, 128]   64       96/0         127/0         127/0         105/0          32/0
    ///  [48, 128]  192    255/836       255/996       255/996      255/1868         168/0
    ///  [64, 128]  256   255/1352      255/1584      255/1584      255/2904          96/0
    ///
    ///  stack frame bytes
    ///  [32,  16]   16          0             0             0             0             0
    ///  [32,  32]   32          0             0             0             0             0
    ///  [32,  48]   48        208           208           208           208           208
    ///  [32,  64]   64        272           272           272           272           768
    ///  [32,  96]   96       1536          1920          1920          1536          1152
    ///  [32, 128]  128       2560          2560          2560          2624          1536
    ///  [32, 192]  192       4656          4632          4632          4936          2304
    ///  [32, 256]  256       6760          6056          6056          7704          3072
    ///  [16,  64]   32        144           144           144           144           144
    ///  [48,  64]   96       1536          1920          1920          1536          1152
    ///  [64,  64]  128       2560          2560          2560          2560          1536
    ///  [16, 128]   64       1280          1280          1280          1280           768
    ///  [48, 128]  192       4656          4648          4648          5152          2304
    ///  [64, 128]  256       6384          6600          6600          7784          3072
    /// ```
    ///
    /// Four results, and only the first is the one #60 was filed expecting.
    ///
    /// **`assign` and `open_coded` are identical at all fourteen rungs**, every
    /// counter, including the three where the pair sits 80 to 200 registers
    /// below every other spelling. #31 claimed the `_assign` API compiles to
    /// the loop it replaces on two measurements; this is the same claim on
    /// fourteen, over shapes that spill, and it is the cleanest column here.
    ///
    /// **The row extent is not the column extent.** `[64, 64]` and `[32, 128]`
    /// are both 128 fp32 a thread and they are not the same shape to compile:
    /// `assign` is 32 registers at one and 252 at the other, on the same 2560
    /// bytes of frame. `[16, 128]` against `[32, 64]` and `[48, 64]` against
    /// `[32, 96]` split the same way, smaller. `M * N` is the first-order
    /// variable and the aspect ratio is a strong second one, so a cost read off
    /// one extent does not transfer — which nothing before this ladder could
    /// have said, because nothing before it varied `M`.
    ///
    /// **#31's ordering holds where #31 measured and not everywhere.**
    /// `rebound` is the dearest or joint-dearest spelling at every rung where
    /// the spellings differ at all except the two 16-row ones, where it is the
    /// cheapest of the by-value forms — and at `[32, 128]` the ladder
    /// reproduces [`scalar_map_probe`]'s 168/252/252/255+60 exactly. But
    /// `fused` is *not* the floor: at `[32, 96]`, `[48, 64]` and `[64, 64]` the
    /// in-place forms are 81 to 130 registers under it. Materialization between
    /// statements orders the by-value spellings; it does not order the table.
    ///
    /// **What orders the table is whether the band is in registers at all**,
    /// and that switches on shape. Read the frames beside the registers: at
    /// each of those three rungs the cheap pair keeps a frame at least as
    /// large as the form it beat — `[32, 96]` is 47 registers on 1920 bytes
    /// against `fused`'s 128 on 1536. They did not fit the band; ptxas left it
    /// addressable in local memory and streamed it, and a streamed band is
    /// cheap in the counter this probe reports and not free in fact.
    /// [`LADDER_ALL_IN_PLACE`] is that outcome on purpose — nothing in its
    /// step materializes a band, the accumulate included — and it is 32
    /// registers at six of fourteen rungs on half to three-quarters the frame
    /// of the others from 96 values a thread up. Lower on both counters is
    /// still not a proof of faster: a frame is a *size*, and what a streamed
    /// band costs is local-memory *traffic*, which `ptxas -v` does not report
    /// and only a timed kernel can price.
    ///
    /// **One has now priced it, and the answer is that this table does not
    /// order time** — see [`crate::ladder_bench`], which times four of these
    /// rungs on a B200. On that probe a streamed band is not slower at all:
    /// `all_in_place` is the fastest spelling at all four shapes on both a
    /// single warp and a full device, and at `[32, 128]` — the rung where
    /// `fused` wins on both static counters — `fused` is the *slower* of the
    /// two. On `softmax` the same phenomenon cost 2.6x (#47), because there
    /// shared memory caps occupancy either way and the freed registers buy
    /// nothing. Registers, frame and occupancy each order part of the timing
    /// and none of them orders it all. Read the rows below as what they are, a
    /// record of what `ptxas` allocated, and not as a ranking.
    ///
    /// The cliff, for the record: `rebound` first spills at 128 values a
    /// thread (128 columns at 32 rows), the other three at
    /// 192, and `all_in_place` at no rung on this ladder. Nothing at all is
    /// visible below 64 values a thread — every spelling is identical at
    /// `[32, 16]`, `[32, 32]` and `[32, 48]`, which is why a 32-wide probe
    /// could sit through a change worth 87 registers at 128 (#5) and report
    /// nothing.
    ///
    /// `STRIDED` is the one thing here that is not about registers, and it
    /// exists so that #63 can put a clock on the rows above. The step and the
    /// two bands are identical either way; all it changes is where the *final*
    /// dump lands — at `false` every block writes the same `M * N` elements,
    /// which is fine for one block and an aliasing violation for a grid, and
    /// at `true` each block gets its own band. A timed rung has to be able to
    /// fill the device, so it needs the second; the register ladder is priced
    /// on the first and must keep exactly the numbers recorded above, which
    /// `regcount`'s twin table checks rung by rung.
    #[inline(always)]
    unsafe fn ladder_probe<const M: usize, const N: usize, const FORM: u32, const STRIDED: bool>(
        scores: &[f32],
        steps: u32,
        k: f32,
        out: &mut DisjointSlice<f32>,
    ) where
        BaseLdtm: FragmentLayout<M, N>,
    {
        unsafe {
            let slots = RegTile::<M, N, BaseLdtm>::SLOTS;
            let values = RegTile::<M, N, BaseLdtm>::VALUES;
            let lane = warp::lane_id();

            let mut out_acc = RegTile::<M, N, BaseLdtm>::zero();

            let mut step = 0u32;
            while step < steps {
                let mut block = RegTile::<M, N, BaseLdtm>::zero();
                let mut slot = 0usize;
                while slot < slots {
                    let mut value = 0usize;
                    while value < values {
                        let (row, column) =
                            RegTile::<M, N, BaseLdtm>::coordinate(lane, slot, value);
                        let index = step as usize * M * N + row as usize * N + column as usize;
                        block.set(slot, value, *scores.get_unchecked(index));
                        value += 1;
                    }
                    slot += 1;
                }

                if FORM == LADDER_FUSED {
                    out_acc = out_acc.scale(k).add(block.scale(k));
                } else {
                    if FORM == LADDER_REBOUND {
                        block = block.scale(k);
                        out_acc = out_acc.scale(k);
                    } else if FORM == LADDER_OPEN_CODED {
                        // Two passes, not one fused pass, for
                        // `scalar_map_probe`'s reason: a floor that fused them
                        // would be measuring the fusion and not the spelling.
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
                    } else {
                        block.scale_assign(k);
                        out_acc.scale_assign(k);
                    }
                    // The accumulate every form but [`LADDER_ALL_IN_PLACE`]
                    // shares, so that four of the five columns differ in the
                    // *scales* alone and the fifth differs in this line alone.
                    if FORM == LADDER_ALL_IN_PLACE {
                        out_acc.add_assign(block);
                    } else {
                        out_acc = out_acc.add(block);
                    }
                }
                step += 1;
            }

            // The accumulator has to reach memory or the loop above is dead
            // and the register counts describe nothing.
            let block_base = if STRIDED {
                thread::blockIdx_x() as usize * M * N
            } else {
                0
            };
            let base = block_base + lane as usize * slots * values;
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

    /// One rung: one spelling of [`ladder_probe`] at one `[M, N]`, as a
    /// `#[kernel]` named after the shape it compiles.
    ///
    /// A PTX entry symbol is the bare function name, so every rung needs a
    /// distinct identifier and there is no generic way around writing one —
    /// `${concat}` is what turns that into a shape rather than a signature.
    macro_rules! ladder_rung {
        ($m:literal, $n:literal, $spelling:ident, $form:ident) => {
            #[kernel]
            pub unsafe fn ${concat(ladder_probe_, $m, x, $n, _, $spelling)}(
                scores: &[f32],
                steps: u32,
                k: f32,
                mut out: DisjointSlice<f32>,
            ) {
                unsafe { ladder_probe::<$m, $n, $form, false>(scores, steps, k, &mut out) }
            }
        };
    }

    /// The ladder: one line per shape, the five spellings generated. Both
    /// extents must be multiples of 16 (`BaseLdtm` has an impl per extent);
    /// beyond that a rung costs a line here and nothing else, which is the
    /// whole of #60.
    macro_rules! ladder {
        ($(($m:literal, $n:literal)),* $(,)?) => {$(
            ladder_rung!($m, $n, fused, LADDER_FUSED);
            ladder_rung!($m, $n, assign, LADDER_ASSIGN);
            ladder_rung!($m, $n, rebound, LADDER_REBOUND);
            ladder_rung!($m, $n, open_coded, LADDER_OPEN_CODED);
            ladder_rung!($m, $n, all_in_place, LADDER_ALL_IN_PLACE);
        )*};
    }

    ladder!(
        // The column sweep, at the warp's 32 rows: 16 values a thread up to
        // 256, which is one past the register file.
        (32, 16),
        (32, 32),
        (32, 48),
        (32, 64),
        (32, 96),
        (32, 128),
        (32, 192),
        (32, 256),
        // The row sweep, at two widths, so four of these pair off with a
        // column rung at equal per-thread volume and a different shape.
        (16, 64),
        (48, 64),
        (64, 64),
        (16, 128),
        (48, 128),
        (64, 128),
    );

    /// One rung of the *timed* ladder: [`ladder_probe`] again, at `STRIDED =
    /// true` so a grid of blocks may run it — issue #63.
    ///
    /// The name is the only thing that separates these from the rungs above.
    /// They call the same `#[inline(always)]` body at the same `(M, N, FORM)`,
    /// so `regcount`'s twin table is entitled to expect the same registers,
    /// the same spill and the same frame, and prints both side by side rather
    /// than asking anyone to take it on trust.
    macro_rules! timed_rung {
        ($m:literal, $n:literal, $spelling:ident, $form:ident) => {
            #[kernel]
            pub unsafe fn ${concat(ladder_timed_, $m, x, $n, _, $spelling)}(
                scores: &[f32],
                steps: u32,
                k: f32,
                mut out: DisjointSlice<f32>,
            ) {
                unsafe { ladder_probe::<$m, $n, $form, true>(scores, steps, k, &mut out) }
            }
        };
    }

    /// The four shapes #63 asks for, and only four: three where the in-place
    /// spellings come in 81 to 130 registers under `fused` on a frame no
    /// smaller — the streamed rungs — and `[32, 128]` as the control, where
    /// `fused` wins on both counters (168/0 against 252/0) and a timing that
    /// did *not* favour it would mean the harness, not the shape.
    macro_rules! timed_ladder {
        ($(($m:literal, $n:literal)),* $(,)?) => {$(
            timed_rung!($m, $n, fused, LADDER_FUSED);
            timed_rung!($m, $n, assign, LADDER_ASSIGN);
            timed_rung!($m, $n, rebound, LADDER_REBOUND);
            timed_rung!($m, $n, open_coded, LADDER_OPEN_CODED);
            timed_rung!($m, $n, all_in_place, LADDER_ALL_IN_PLACE);
        )*};
    }

    timed_ladder!((32, 96), (48, 64), (64, 64), (32, 128));

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
    /// 128 hoisted         168    2816
    /// 128 per_op          168    2816
    /// 128 per_query       168    2816
    ///  32 hoisted          56     384
    ///  32 per_op           40     512
    ///  32 per_query        40     512
    /// ```
    ///
    /// [`PER_OP`] and [`PER_QUERY`] agree on every column at both widths, and
    /// the emitted PTX holds exactly *one* `%laneid` move in all six kernels —
    /// so the count of `warp::lane_id()` calls is free, and reading the
    /// register inside an op costs nothing over receiving it. What is left is
    /// where the value is *defined*: hoisting it into the entry block makes
    /// the whole coordinate map loop-invariant, which ptxas pays for in live
    /// registers at 32 (+16) and is repaid in half the local traffic. That is
    /// an allocator trade, not a property of the convention; this probe exists
    /// so it stays visible if the backend's `lane_id` lowering ever stops
    /// being a CSE-able pure read.
    ///
    /// Three cells of this table drifted between #27 and #60 — it recorded
    /// 198 registers for `128 hoisted` and 72/256 for `32 hoisted`, and the
    /// tree measures 168 and 56/384. The re-measurement is #60's, and *only*
    /// the `regcount` columns are re-measured: the instruction counts #27 read
    /// off the emitted PTX are left as its own. What the drift costs is the
    /// 128-column half of the finding — hoisting is free at that width now,
    /// where #27 measured it at 30 registers, so the trade is visible only at
    /// 32 and the direction of the whole result is the same as #60's ladder:
    /// the wide probe sits high enough that everything reads alike.
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

    /// Every path a [`SharedVec`] has, in the order a kernel uses them: TMA a
    /// rank-1 box in, broadcast it to registers, write it back element by
    /// element, TMA it out again.
    ///
    /// The dump is indexed by `(lane, value)` alone — the host applies
    /// [`BaseLdtm::column`], so a wrong broadcast reports the column the
    /// hardware handed that register rather than a bare mismatch. The write
    /// half reverses the vector, so a wrong element stride lands identities at
    /// positions that name other positions, and a store that inherited a
    /// tile's atom leaves the second half of the destination [`POISON`].
    ///
    /// Launch with one warp: [`load_vec`] is warp scope, and 32 lanes writing
    /// four elements each is exactly the neighbour-clobbering case a per-word
    /// scalar write would have failed.
    #[kernel]
    pub unsafe fn shared_vec_roundtrip(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let vec = Params::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Params::BYTES) as *mut Barrier);
            let lane = warp::lane_id();

            if lane == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if lane == 0 {
                tma.expect_tx(vec.tma_load(source, 0, tma));
            }
            tma.wait(0);
            thread::sync_threads();

            let cols: ColVec<VECTOR, BaseLdtm> = load_vec(vec, lane);
            let mut value = 0usize;
            while value < VECTOR_VALUES {
                *out.get_unchecked_mut(lane as usize * VECTOR_VALUES + value) = cols.get(value);
                value += 1;
            }
            thread::sync_threads();

            let mut index = lane as usize;
            while index < VECTOR {
                vec.set(index, cell(0, VECTOR - 1 - index));
                index += 32;
            }
            // `set` writes through the generic proxy and the TMA engine reads
            // through the async one, exactly as `store_tile` does.
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if lane == 0 {
                vec.tma_store(destination, 0);
                tma_store_commit();
                tma_store_wait::<0>();
                tma.inval();
            }
        }
    }

    /// One *row* of a `[VECTOR, VECTOR_ROWS]` buffer through
    /// [`SharedVec::tma_load_2d`], dumped in logical order.
    ///
    /// The rank-2 box is `[VECTOR, 1]`, so this is the case that says a
    /// vector's second box dimension really is one row and that the minor
    /// coordinate selects it: every element carries its source row, so a
    /// descriptor that delivered the wrong row reports *that* row rather than
    /// a misplaced element. Launch with one warp.
    #[kernel]
    pub unsafe fn shared_vec_row(source: *const TmaDescriptor, mut out: DisjointSlice<f32>) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let vec = Params::from_raw(smem);
            let tma = Semaphore::attach(smem.add(Params::BYTES) as *mut Barrier);
            let lane = warp::lane_id();

            if lane == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if lane == 0 {
                tma.expect_tx(vec.tma_load_2d(source, 0, VECTOR_ROW as i32, tma));
            }
            tma.wait(0);
            thread::sync_threads();

            let mut index = lane as usize;
            while index < VECTOR {
                *out.get_unchecked_mut(index) = vec.get(index);
                index += 32;
            }

            thread::sync_threads();
            if lane == 0 {
                tma.inval();
            }
        }
    }

    /// The block reduction against silicon: the one collective in this library
    /// that spans warps, and the only one no shuffle can implement.
    ///
    /// Each warp's band is a splat of the seed the host wrote for *that warp*,
    /// so `tile_sum` hands `block_reduce_sum` a warp-uniform partial the same
    /// way `groupnorm_tile` does — the composition under test is the whole
    /// two-step fold, not the block half alone. The seeds are distinct powers
    /// of eight ([`block_partial`]), so the host reads the returned number as
    /// base-8 digits and says which slot was read how many times.
    ///
    /// Three calls, back to back on the same scratch with no barrier between
    /// them, which is the reuse rule the collective claims. The second folds
    /// the *doubled* partials: if it read the first call's staging instead of
    /// its own, every digit comes back a 1 where a 2 was wanted, per slot. The
    /// third is a `Max` rather than a sum, so the `Op` parameter and the
    /// identity a fold starts from are on the same silicon as the rest.
    ///
    /// Every thread dumps all three by its own `(warp, lane)`, so
    /// block-uniformity is a claim about 128 threads and not about one.
    ///
    /// Launch with `BLOCK_WARPS * 32` threads and [`Partials::BYTES`] of shared
    /// memory.
    #[kernel]
    pub unsafe fn block_reduce_probe(seeds: &[f32], mut out: DisjointSlice<f32>) {
        unsafe {
            let partials = Partials::from_raw(DynamicSharedArray::<u8, 128>::get_raw());
            let warp = warp::warp_id() as usize;
            let thread = thread::threadIdx_x() as usize;

            let band = BlockBand::splat(*seeds.get_unchecked(warp));
            let mine = band.tile_sum();

            let sum = block_reduce_sum(partials, mine);
            let doubled = block_reduce_sum(partials, 2.0 * mine);
            let largest = block_reduce::<Max, BLOCK_WARPS>(partials, mine);

            let base = thread * BLOCK_REDUCE_STRIDE;
            *out.get_unchecked_mut(base) = sum;
            *out.get_unchecked_mut(base + 1) = doubled;
            *out.get_unchecked_mut(base + 2) = largest;
        }
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

/// Does the reduction store **add**, in fp32, exactly where its map says?
///
/// The destination is pitched with poison margins like [`check_tma_store`]'s,
/// but seeded with a large constant inside the rectangle — the two-pass answer
/// `seed + 2 · value` is then computed on the host and expected bit-exactly,
/// every term an integer far below 2^24. A plain store, a single add, an add
/// that dropped the seed, or an add at a neighbouring coordinate each produce
/// a distinct wrong value, and a reduction that strayed into the margins
/// touches poison no seeded element holds.
fn check_tma_store_add(
    stream: &CudaStream,
    launch: impl Fn(
        LaunchConfig,
        *const TmaDescriptor,
        *const TmaDescriptor,
    ) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    const PITCH: usize = 2 * ADD_COLS;
    const SEED: f32 = 1_048_576.0;
    const POISON_F32: f32 = -7.0;

    let tile: Vec<f32> = (0..ADD_ROWS * ADD_COLS).map(|i| (i + 1) as f32).collect();
    let source = DeviceBuffer::from_host(stream, &tile)?;
    let source_map = unsafe {
        GlobalLayout::<F32, 2>::packed(source.cu_deviceptr(), [ADD_COLS, ADD_ROWS])
            .tensor_map::<AddTile>(stream)?
    };

    let mut seeded = vec![POISON_F32; ADD_ROWS * PITCH];
    for row in 0..ADD_ROWS {
        seeded[row * PITCH..row * PITCH + ADD_COLS].fill(SEED);
    }
    let dest = DeviceBuffer::from_host(stream, &seeded)?;
    let dest_map = unsafe {
        GlobalLayout::<F32, 2>::strided(dest.cu_deviceptr(), [ADD_COLS, ADD_ROWS], [1, PITCH])
            .tensor_map::<AddTile>(stream)?
    };

    launch(
        launch_config(32, (AddTile::BYTES + 32) as u32),
        source_map.as_ptr(),
        dest_map.as_ptr(),
    )?;

    let mut expected = seeded;
    for row in 0..ADD_ROWS {
        for column in 0..ADD_COLS {
            expected[row * PITCH + column] = SEED + 2.0 * tile[row * ADD_COLS + column];
        }
    }
    let observed = dest.to_host_vec(stream)?;
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, (&got, &want)) in observed.iter().zip(&expected).enumerate() {
        if got.to_bits() == want.to_bits() {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let _ = write!(
                report,
                "\n    (row {}, column {}): expected {want}, read back {got}",
                index / PITCH,
                index % PITCH
            );
        }
    }
    if mismatches != 0 {
        return Err(format!("{mismatches} of {} elements wrong{report}", expected.len()).into());
    }
    Ok(format!(
        "{} elements at seed + 2·value exactly, margins intact",
        ADD_ROWS * ADD_COLS
    ))
}

/// Does a band written into a [`StoreRing`] reach the global rows it was
/// committed for, at every depth — including after the ring has wrapped?
///
/// The destination is [`RING_BANDS`] tall in staging buffers and seeded with
/// [`POISON`], and band `b` is expected at rows `RING_ROWS * b` holding
/// [`cell`]s whose columns are rotated by [`ring_column`]. Every way this can
/// fail names itself:
///
/// - **A wrong row coordinate** puts a band at another band's rows, where its
///   rotation is the wrong one for every element of the tile.
/// - **A missed store** leaves `POISON`, which no identity can be.
/// - **The reuse hazard** — a buffer written before the engine finished
///   reading it — puts band `k + DEPTH`'s rotation into band `k`'s rows, and
///   the difference of the two decoded columns is `RING_STRIDE * DEPTH`. That
///   is the failure depth 1 cannot produce and depth ≥ 2 can, which is why the
///   deeper cases exist at all.
/// - **A missed final wait** leaves the tail of the destination unwritten when
///   the host reads it back, again as `POISON`.
///
/// What it cannot claim: that a ring built *without* the waits would fail here.
/// A dropped `wait_read` is a race, and a race that does not happen leaves no
/// trace. The case is built so that a violation is a wrong value rather than a
/// plausible one; it is not built to force the violation.
/// `BOX_ROWS` is the descriptor's row count and therefore the *scope* of a commit:
/// [`RING_ROWS`] when the block fills one buffer, [`RING_WARP_ROWS`] when each
/// warp fills its own. The expectation does not move with it — a band is the
/// same [`RING_ROWS`] rows of the same identities either way, assembled by four
/// commits instead of one — which is the whole reason the two arrangements can
/// share this function.
fn check_store_ring<const BOX_ROWS: usize, const C: usize>(
    stream: &CudaStream,
    shared: u32,
    launch: impl Fn(LaunchConfig, *const TmaDescriptor) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let rows = RING_BANDS * RING_ROWS;
    let destination = DeviceBuffer::from_host(stream, &vec![POISON; rows * C / 2])?;
    let map =
        unsafe { encode_bf16_panels::<BOX_ROWS, C>(stream, destination.cu_deviceptr(), rows, 1)? };

    launch(launch_config(RING_THREADS, shared), map.as_ptr())?;

    let mut expected = Vec::with_capacity(rows * C / 2);
    for band in 0..RING_BANDS {
        for row in 0..RING_ROWS {
            for pair in 0..C / 2 {
                expected.push(pack(
                    cell_bits(row, ring_column(2 * pair, band, C)),
                    cell_bits(row, ring_column(2 * pair + 1, band, C)),
                ));
            }
        }
    }
    let note = compare_tile(&destination.to_host_vec(stream)?, &expected, C / 2)?;
    let (box_rows, columns) = (BOX_ROWS, C);
    Ok(format!(
        "{RING_BANDS} bands of [{RING_ROWS}, {columns}] in [{box_rows}, {columns}] boxes \
         through {shared} B of ring, {note}"
    ))
}

/// Does a staged tile reach the rectangle the arithmetic names, by plain
/// stores and at every rung of the access ladder?
///
/// The shared→global half of an epilogue with no engine in it (#15). The
/// destination is a `DRAIN_MATRIX_ROWS × DRAIN_PITCH` bf16 matrix seeded end to
/// end with [`POISON_HALF`], and the drain is asked for a `[DRAIN_ROWS, C]`
/// rectangle at `(DRAIN_ROW, column)` — narrower than the pitch and inset from
/// both ends of it, so **every** failure below names itself:
///
/// - **A wrong row stride** walks rows by the tile's width instead of the
///   matrix's, which past row 0 lands on elements no identity of that row can
///   be, and leaves the rows it skipped as poison.
/// - **A dropped origin** puts the whole rectangle at `(0, 0)`, where the
///   margins the case checks are supposed to still be poison.
/// - **A confused swizzle** — the chunk XOR, the base's phase, the stacked
///   subtile stride at `C = WIDE` — permutes elements *within* a 128-byte row,
///   so it shows up as identities of the right row at the wrong columns rather
///   than as anything numeric.
/// - **A short drain** leaves poison inside the rectangle, which no identity
///   can be.
///
/// The four [`DRAIN_COLUMNS`] run the same kernel at each rung of the width
/// ladder, so a 16-byte access and a 2-byte one are the same case with one
/// argument changed. What that cannot claim is that the rung *chosen* was the
/// widest legal one: a drain that always issued 2-byte stores passes all four.
/// `access_width`'s host tests are what say which rung a cursor gets.
fn check_shared_drain<const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(LaunchConfig, u32, &mut DeviceBuffer<u16>) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let seed = vec![POISON_HALF; DRAIN_MATRIX_ROWS * DRAIN_PITCH];
    for column in DRAIN_COLUMNS {
        let mut destination = DeviceBuffer::from_host(stream, &seed)?;
        launch(
            launch_config(DRAIN_THREADS, Tile::<DRAIN_ROWS, C>::BYTES as u32),
            column,
            &mut destination,
        )?;

        let mut expected = seed.clone();
        for row in 0..DRAIN_ROWS {
            for tile_column in 0..C {
                let at = (DRAIN_ROW as usize + row) * DRAIN_PITCH + column as usize + tile_column;
                expected[at] = cell_bits(row, tile_column);
            }
        }
        compare_matrix(&destination.to_host_vec(stream)?, &expected, column)?;
    }
    let columns = C;
    Ok(format!(
        "[{DRAIN_ROWS}, {columns}] at row {DRAIN_ROW} of a \
         {DRAIN_MATRIX_ROWS}x{DRAIN_PITCH} matrix, at columns {DRAIN_COLUMNS:?}"
    ))
}

/// Does the same rectangle *fold into* its destination, reading `C` and
/// rounding the sum once?
///
/// [`check_shared_drain`] with the seed inside the rectangle changed from
/// poison to the identities themselves, so the answer is every identity
/// doubled. Doubling a bf16 moves the exponent and leaves the mantissa alone,
/// which is why the comparison can stay on bit patterns with no tolerance
/// anywhere: a fold that rounded twice, accumulated at bf16, or added the wrong
/// neighbour lands on a different pattern, not a nearby one.
///
/// Every geometry failure [`check_shared_drain`] names is named here too — the
/// margins are still poison and still checked. The one failure only this case
/// can see is a fold that **dropped the read**: that writes the identities
/// undoubled, which is exactly what the store case expects and this one does
/// not.
fn check_shared_accumulate<const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(LaunchConfig, u32, &mut DeviceBuffer<u16>) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let poison = vec![POISON_HALF; DRAIN_MATRIX_ROWS * DRAIN_PITCH];
    for column in DRAIN_COLUMNS {
        let (mut seed, mut expected) = (poison.clone(), poison.clone());
        for row in 0..DRAIN_ROWS {
            for tile_column in 0..C {
                let at = (DRAIN_ROW as usize + row) * DRAIN_PITCH + column as usize + tile_column;
                seed[at] = cell_bits(row, tile_column);
                expected[at] = to_bf16(2.0 * cell(row, tile_column));
            }
        }
        let mut destination = DeviceBuffer::from_host(stream, &seed)?;
        launch(
            launch_config(DRAIN_THREADS, Tile::<DRAIN_ROWS, C>::BYTES as u32),
            column,
            &mut destination,
        )?;
        compare_matrix(&destination.to_host_vec(stream)?, &expected, column)?;
    }
    let columns = C;
    Ok(format!(
        "[{DRAIN_ROWS}, {columns}] folded into its own identities at row \
         {DRAIN_ROW} of a {DRAIN_MATRIX_ROWS}x{DRAIN_PITCH} matrix, at columns \
         {DRAIN_COLUMNS:?}"
    ))
}

/// Does an **fp32** band reach the same rectangle, through a staging tile
/// `stmatrix` cannot fill?
///
/// [`check_shared_drain`] at four bytes an element (#174), and it names every
/// failure that one does — the rectangle is the same one, inset from a pitch it
/// does not divide, so a wrong row stride, a dropped origin, a confused swizzle
/// and a short drain each land somewhere the identities cannot.
///
/// What is new here is the *element*: at fp32 a 128-column row is four stacked
/// subtiles rather than two, and a column's four bytes sit inside a 16-byte
/// chunk rather than filling a quarter of one — so the byte offset
/// `SwizzledChunks::element` adds is exercised at both widths and both ends of
/// a chunk.
///
/// The same launcher runs the register route (`register_drain`), which writes
/// this rectangle with no shared tile in the way. Passing it says the two
/// routes agree element for element, which is what makes the register counts
/// the `regcount` table reports for them a comparison of one thing.
fn check_scatter_drain<const C: usize>(
    stream: &CudaStream,
    launch: impl Fn(LaunchConfig, u32, &mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let seed = vec![f32::from_bits(POISON); DRAIN_MATRIX_ROWS * DRAIN_PITCH];
    for column in DRAIN_COLUMNS {
        let mut destination = DeviceBuffer::from_host(stream, &seed)?;
        launch(
            launch_config(DRAIN_THREADS, F32Tile::<DRAIN_ROWS, C>::BYTES as u32),
            column,
            &mut destination,
        )?;

        let mut expected = seed.clone();
        for row in 0..DRAIN_ROWS {
            for tile_column in 0..C {
                let at = (DRAIN_ROW as usize + row) * DRAIN_PITCH + column as usize + tile_column;
                expected[at] = cell(row, tile_column);
            }
        }
        let observed = destination.to_host_vec(stream)?;
        compare_f32_matrix(&observed, &expected, column)?;
    }
    let columns = C;
    Ok(format!(
        "[{DRAIN_ROWS}, {columns}] fp32 at row {DRAIN_ROW} of a \
         {DRAIN_MATRIX_ROWS}x{DRAIN_PITCH} matrix, at columns {DRAIN_COLUMNS:?}"
    ))
}

/// [`compare_matrix`] at fp32 — bit patterns, since the identities are exact
/// at this element and poison is a pattern no identity can take.
fn compare_f32_matrix(
    observed: &[f32],
    expected: &[f32],
    column: u32,
) -> Result<(), Box<dyn Error>> {
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, (&got, &want)) in observed.iter().zip(expected).enumerate() {
        if got.to_bits() == want.to_bits() {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let (row, at) = (index / DRAIN_PITCH, index % DRAIN_PITCH);
            let name = |value: f32| match decode_cell(value) {
                Some((r, c)) => format!("the identity of tile ({r}, {c})"),
                None if value.to_bits() == POISON => "poison".to_string(),
                None => format!("{:#010x}, which names no element", value.to_bits()),
            };
            let _ = write!(
                report,
                "\n    ({row}, {at}): wanted {}, found {}",
                name(want),
                name(got)
            );
        }
    }
    if mismatches == 0 {
        return Ok(());
    }
    Err(format!(
        "at column {column}: {mismatches} of {} elements wrong{report}",
        expected.len()
    )
    .into())
}

/// Compare a whole pitched bf16 matrix against what the drain owed it —
/// identities inside the rectangle, poison everywhere else — reporting the
/// first few wrong elements by position, and each one as the position its value
/// actually names.
fn compare_matrix(observed: &[u16], expected: &[u16], column: u32) -> Result<(), Box<dyn Error>> {
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (index, (&got, &want)) in observed.iter().zip(expected).enumerate() {
        if got == want {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let (row, at) = (index / DRAIN_PITCH, index % DRAIN_PITCH);
            let named = match decode_cell(f32::from_bits((got as u32) << 16)) {
                Some((r, c)) => format!("the identity of tile ({r}, {c})"),
                None if got == POISON_HALF => "poison".to_string(),
                None => format!("{got:#06x}, which names no element"),
            };
            let wanted = match decode_cell(f32::from_bits((want as u32) << 16)) {
                Some((r, c)) => format!("tile ({r}, {c})"),
                None if want == POISON_HALF => "poison".to_string(),
                // An accumulating drain's expectation is a doubled identity,
                // which names no position — so say the bits rather than call
                // every one of them poison.
                None => format!("{want:#06x}"),
            };
            let _ = write!(
                report,
                "\n    ({row}, {at}): wanted {wanted}, found {named}"
            );
        }
    }
    if mismatches == 0 {
        return Ok(());
    }
    Err(format!(
        "at column {column}: {mismatches} of {} elements wrong{report}",
        expected.len()
    )
    .into())
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

/// Does [`load_rows`] read the elements the fragment layout says it does?
///
/// The one case in this harness whose source never touches the TMA engine: a
/// pitched fp32 matrix staged by the host, seeded so every element's value is
/// its own flat index, read straight into registers at an origin inside it.
/// The kernel dumps by `(lane, slot, value)` and the host applies
/// [`BaseLdtm`]'s map, so a misplaced value reports the element it actually
/// came from — and because the identity *is* the flat index, a wrong stride, a
/// dropped row origin and a dropped column origin each name a different one.
///
/// See [`kernels::global_rows_map`] for why the store direction is left to
/// `gemm`.
fn check_global_rows(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged: Vec<f32> = (0..GLOBAL_ROWS)
        .flat_map(|row| (0..GLOBAL_PITCH).map(move |column| global_cell(row, column)))
        .collect();
    let source = DeviceBuffer::from_host(stream, &staged)?;

    type Band = RegTile<32, WIDE, BaseLdtm>;
    let (slots, values) = (Band::SLOTS, Band::VALUES);
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 32 * slots * values)?;
    unsafe { module.global_rows_map(stream, launch_config(32, 0), &source, &mut out)? };
    let observed = out.to_host_vec(stream)?;

    let mut report = String::new();
    let mut mismatches = 0usize;
    for lane in 0..32u32 {
        for slot in 0..slots {
            for value in 0..values {
                let (row, column) = Band::coordinate(lane, slot, value);
                let (row, column) = (
                    GLOBAL_ROW as usize + row as usize,
                    GLOBAL_COLUMN as usize + column as usize,
                );
                let got = observed[dump_index(0, lane, slot, value, slots, values)];
                if got == global_cell(row, column) {
                    continue;
                }
                mismatches += 1;
                if mismatches <= 8 {
                    let named = if got >= 0.0 && got.fract() == 0.0 && got < staged.len() as f32 {
                        let index = got as usize;
                        format!("({}, {})", index / GLOBAL_PITCH, index % GLOBAL_PITCH)
                    } else {
                        format!("{got}, which names no element")
                    };
                    let _ = write!(
                        report,
                        "\n    lane {lane} slot {slot} value {value}: map says \
                         ({row}, {column}), memory delivered {named}"
                    );
                }
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "[32, {WIDE}] at ({GLOBAL_ROW}, {GLOBAL_COLUMN}) of a {GLOBAL_ROWS}x{GLOBAL_PITCH} \
             matrix, all at BaseLdtm's coordinates"
        ));
    }
    Err(format!(
        "{mismatches} of {} values misplaced{report}",
        observed.len()
    )
    .into())
}

/// Does [`load_cols`] deliver the columns `BaseLdtm::column` names, off one
/// row of the same pitched matrix `check_global_rows` reads a band out of?
///
/// The vector twin of that case, and the reason it is a separate one: a
/// `ColVec` is `VALUES` registers and not `SLOTS * VALUES`, so a mover that
/// silently read the tile's rows would still pass there and could not pass
/// here.
fn check_global_cols(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged: Vec<f32> = (0..GLOBAL_ROWS)
        .flat_map(|row| (0..GLOBAL_PITCH).map(move |column| global_cell(row, column)))
        .collect();
    let source = DeviceBuffer::from_host(stream, &staged)?;

    type Columns = ColVec<WIDE, BaseLdtm>;
    let values = Columns::VALUES;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 32 * values)?;
    unsafe { module.global_cols_map(stream, launch_config(32, 0), &source, &mut out)? };
    let observed = out.to_host_vec(stream)?;

    let mut report = String::new();
    let mut mismatches = 0usize;
    for lane in 0..32u32 {
        for value in 0..values {
            let column = GLOBAL_COLUMN as usize + Columns::column(lane, value) as usize;
            let got = observed[lane as usize * values + value];
            if got == global_cell(GLOBAL_ROW as usize, column) {
                continue;
            }
            mismatches += 1;
            if mismatches <= 8 {
                let _ = write!(
                    report,
                    "\n    lane {lane} value {value}: map says ({GLOBAL_ROW}, {column}), \
                     memory delivered {got}"
                );
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "[{WIDE}] columns at ({GLOBAL_ROW}, {GLOBAL_COLUMN}) of a \
             {GLOBAL_ROWS}x{GLOBAL_PITCH} matrix, all at BaseLdtm's columns"
        ));
    }
    Err(format!(
        "{mismatches} of {} values misplaced{report}",
        observed.len()
    )
    .into())
}

/// Does [`load_col_vec`] read *down* the column the caller named, onto the
/// axis a `col_map` broadcasts along?
///
/// The same matrix and origin as the two cases above, because that is what
/// makes all three separable: every element carries its own flat index, so a
/// mover that walked the row — which is exactly what [`load_cols`] beside it
/// does, over the same argument list — delivers
/// `GLOBAL_ROW * GLOBAL_PITCH + GLOBAL_COLUMN + value` where this expects
/// `(GLOBAL_ROW + value) * GLOBAL_PITCH + GLOBAL_COLUMN`, and the failure
/// names the element it actually read. Two movers returning one type is the
/// hazard this case exists to pin.
fn check_global_col_vec(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged: Vec<f32> = (0..GLOBAL_ROWS)
        .flat_map(|row| (0..GLOBAL_PITCH).map(move |column| global_cell(row, column)))
        .collect();
    let source = DeviceBuffer::from_host(stream, &staged)?;

    let values = <BaseLdtm as ColLayout<COLUMN_BAND>>::VALUES;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 32 * values)?;
    unsafe { module.global_col_vec_map(stream, launch_config(32, 0), &source, &mut out)? };
    let observed = out.to_host_vec(stream)?;

    let mut report = String::new();
    let mut mismatches = 0usize;
    for lane in 0..32u32 {
        for value in 0..values {
            let column = <BaseLdtm as ColLayout<COLUMN_BAND>>::col_of(lane, value) as usize;
            let row = GLOBAL_ROW as usize + column;
            let got = observed[lane as usize * values + value];
            if got == global_cell(row, GLOBAL_COLUMN as usize) {
                continue;
            }
            mismatches += 1;
            if mismatches <= 8 {
                let named = if got >= 0.0 && got.fract() == 0.0 && got < staged.len() as f32 {
                    let index = got as usize;
                    format!("({}, {})", index / GLOBAL_PITCH, index % GLOBAL_PITCH)
                } else {
                    format!("{got}, which names no element")
                };
                let _ = write!(
                    report,
                    "\n    lane {lane} value {value}: band column {column} is \
                     ({row}, {GLOBAL_COLUMN}), memory delivered {named}"
                );
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "[{COLUMN_BAND}] columns down ({GLOBAL_ROW}.., {GLOBAL_COLUMN}) of a \
             {GLOBAL_ROWS}x{GLOBAL_PITCH} matrix, all at BaseLdtm's columns"
        ));
    }
    Err(format!(
        "{mismatches} of {} values misplaced{report}",
        observed.len()
    )
    .into())
}

/// Does a [`SharedVec`] survive the whole loop — global, shared, registers,
/// shared, global — with every element where the library says it is?
///
/// Four claims in one launch, and each fails distinguishably:
///
/// - **The box.** A rank-1 map with `CU_TENSOR_MAP_SWIZZLE_NONE` and a
///   `[VECTOR]` box is a shape nothing in this harness had built. `VECTOR` is
///   twice a swizzle atom, so a box quietly capped at one atom delivers half
///   the vector and leaves the rest zero.
/// - **The broadcast.** The register dump is indexed by `(lane, value)` and
///   the host applies [`BaseLdtm::column`], so what a failure prints is the
///   column the hardware actually handed that register.
/// - **The scalar write.** 32 lanes write four elements each into a 2-byte
///   element type; a write that went through a 32-bit word would lose one of
///   every adjacent pair.
/// - **The store.** The destination is seeded with [`POISON`], which no
///   identity can be, so an unwritten or short store is not a plausible
///   neighbour.
fn check_shared_vec(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile(1, VECTOR);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let source_map = unsafe { GlobalLayout::<Bf16, 1>::packed(source.cu_deviceptr(), [VECTOR]) }
        .tensor_map::<Params>(stream)?;

    let destination = DeviceBuffer::from_host(stream, &vec![POISON; staged.len()])?;
    let destination_map =
        unsafe { GlobalLayout::<Bf16, 1>::packed(destination.cu_deviceptr(), [VECTOR]) }
            .tensor_map::<Params>(stream)?;

    let mut out = DeviceBuffer::<f32>::zeroed(stream, 32 * VECTOR_VALUES)?;
    unsafe {
        module.shared_vec_roundtrip(
            stream,
            launch_config(32, VECTOR_SHARED),
            source_map.as_ptr(),
            destination_map.as_ptr(),
            &mut out,
        )?
    };

    let observed = out.to_host_vec(stream)?;
    let mut report = String::new();
    let mut mismatches = 0usize;
    for lane in 0..32u32 {
        for value in 0..VECTOR_VALUES {
            let column = BaseLdtm::column(lane, value) as usize;
            let got = observed[lane as usize * VECTOR_VALUES + value];
            if got == cell(0, column) {
                continue;
            }
            mismatches += 1;
            if mismatches <= 8 {
                let _ = match decode_cell(got) {
                    Some((_, delivered)) => write!(
                        report,
                        "\n    lane {lane} value {value}: owns column {column}, was handed {delivered}"
                    ),
                    None => write!(
                        report,
                        "\n    lane {lane} value {value}: owns column {column}, was handed {got}, \
                         which names no position"
                    ),
                };
            }
        }
    }
    if mismatches > 0 {
        return Err(format!(
            "{mismatches} of {} broadcast values wrong{report}",
            32 * VECTOR_VALUES
        )
        .into());
    }

    // The write half: element `i` was overwritten with the identity of
    // `VECTOR - 1 - i`, so the destination is the staged vector reversed.
    let mut reversed = Vec::with_capacity(staged.len());
    for pair in 0..VECTOR / 2 {
        reversed.push(pack(
            cell_bits(0, VECTOR - 1 - 2 * pair),
            cell_bits(0, VECTOR - 2 - 2 * pair),
        ));
    }
    compare_tile(&destination.to_host_vec(stream)?, &reversed, VECTOR / 2)
}

/// Does a vector's rank-2 box really select one row?
///
/// The buffer is `VECTOR_ROWS` rows of position identities carrying their own
/// row index, and the kernel asks for [`VECTOR_ROW`]. A descriptor whose box
/// spans more than one row, or that ignores the minor coordinate, reports the
/// row it delivered instead of a mismatch.
fn check_shared_vec_row(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let staged = identity_tile(VECTOR_ROWS, VECTOR);
    let source = DeviceBuffer::from_host(stream, &staged)?;
    let map =
        unsafe { GlobalLayout::<Bf16, 2>::packed(source.cu_deviceptr(), [VECTOR, VECTOR_ROWS]) }
            .tensor_map::<Params>(stream)?;

    let mut out = DeviceBuffer::<f32>::zeroed(stream, VECTOR)?;
    unsafe {
        module.shared_vec_row(
            stream,
            launch_config(32, VECTOR_SHARED),
            map.as_ptr(),
            &mut out,
        )?
    };

    let observed = out.to_host_vec(stream)?;
    let mut report = String::new();
    let mut mismatches = 0usize;
    for (column, &got) in observed.iter().enumerate() {
        if got == cell(VECTOR_ROW, column) {
            continue;
        }
        mismatches += 1;
        if mismatches <= 8 {
            let _ = match decode_cell(got) {
                Some((row, delivered)) => write!(
                    report,
                    "\n    element {column}: wanted row {VECTOR_ROW}, got row {row} column {delivered}"
                ),
                None => write!(report, "\n    element {column}: {got} names no position"),
            };
        }
    }
    if mismatches == 0 {
        Ok(format!(
            "{VECTOR} elements of row {VECTOR_ROW} placed exactly"
        ))
    } else {
        Err(format!("{mismatches} of {VECTOR} elements wrong{report}").into())
    }
}

/// A fold over the per-warp partials, read back as which slots it touched.
///
/// The partials are distinct powers of eight, so an observed sum's base-8
/// digits are the multiplicity of each slot: `[1, 1, 1, 1]` is the fold that
/// was asked for, `[0, 0, 1, 0]` is a warp that read only its own slot, and a
/// 2 is a slot folded twice. `None` if the value is not a whole number in
/// range, which is what a partial that never reached shared memory looks like.
fn block_digits(value: f32) -> Option<[u32; BLOCK_WARPS]> {
    if value < 0.0 || value.fract() != 0.0 || value >= 8f32.powi(BLOCK_WARPS as i32) {
        return None;
    }
    let mut left = value as u32;
    let mut digits = [0u32; BLOCK_WARPS];
    for digit in digits.iter_mut() {
        *digit = left % 8;
        left /= 8;
    }
    Some(digits)
}

/// Do four warps agree, and does the same scratch survive being folded through
/// three times?
///
/// The host writes one seed per warp and applies the fold itself; the kernel
/// carries no expected total, and every one of its 128 threads reports its own
/// answer, so "block-uniform" is checked rather than sampled. A failure names
/// the slots the fold actually read — see [`block_digits`] — which separates
/// the three ways this can be wrong that all return a plausible number: a warp
/// reading its own slot, a slot read twice, and the second call reading the
/// first call's staging.
fn check_block_reduce(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    // Seeded so that `tile_sum` over a splat band is exactly `block_partial`:
    // every value is a power of two times a power of eight, and every partial
    // sum on the way is an exact fp32 integer multiple of the seed.
    let seeds: Vec<f32> = (0..BLOCK_WARPS)
        .map(|warp| block_partial(warp) / BLOCK_BAND_VALUES)
        .collect();
    let threads = BLOCK_WARPS * 32;
    let device_seeds = DeviceBuffer::from_host(stream, &seeds)?;

    let mut out = DeviceBuffer::<f32>::zeroed(stream, threads * BLOCK_REDUCE_STRIDE)?;
    unsafe {
        module.block_reduce_probe(
            stream,
            launch_config(threads as u32, Partials::BYTES as u32),
            &device_seeds,
            &mut out,
        )?
    };
    let observed = out.to_host_vec(stream)?;

    let sum: f32 = (0..BLOCK_WARPS).map(block_partial).sum();
    let largest = block_partial(BLOCK_WARPS - 1);
    let wanted = [
        ("sum", sum, [1u32; BLOCK_WARPS]),
        ("doubled sum", 2.0 * sum, [2u32; BLOCK_WARPS]),
        ("max", largest, [0, 0, 0, 1]),
    ];

    let mut report = String::new();
    let mut mismatches = 0usize;
    for thread in 0..threads {
        for (offset, (name, expected, digits)) in wanted.iter().enumerate() {
            let got = observed[thread * BLOCK_REDUCE_STRIDE + offset];
            if got == *expected {
                continue;
            }
            mismatches += 1;
            if mismatches <= 8 {
                let _ = match block_digits(got) {
                    Some(read) => write!(
                        report,
                        "\n    warp {} lane {} {name}: wanted {expected} (slots {digits:?}), \
                         got {got} (slots {read:?})",
                        thread / 32,
                        thread % 32
                    ),
                    None => write!(
                        report,
                        "\n    warp {} lane {} {name}: wanted {expected}, got {got}, \
                         which is no fold of the four partials",
                        thread / 32,
                        thread % 32
                    ),
                };
            }
        }
    }
    if mismatches == 0 {
        return Ok(format!(
            "{BLOCK_WARPS} partials folded three ways, identical in {threads} threads"
        ));
    }
    Err(format!(
        "{mismatches} of {} statistics wrong{report}",
        threads * BLOCK_REDUCE_STRIDE
    )
    .into())
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
/// This establishes that `TmemTile::store_tile` is the exact inverse of the
/// drain it is taken through — no more. A store and a load agreeing on the
/// *wrong* lane map would pass identically; it is the `fragment map` cases
/// that fix LDTM's map against silicon, and `sttm restage` that checks the
/// store's column arithmetic against a value it did not compute.
///
/// Two cases ride it. `sttm round trip` reads back through `TmemTile::tile`
/// (`.x1`) and `ldtm x8 map` through `TmemTile::tile_x8` (`.x8`), so the pair
/// says the wide drain returns what the narrow one does — which is the whole
/// of #117's register-order claim, and a claim no document at the pinned
/// cuda-oxide revision makes.
/// `drain` is which read-back the round trip is being taken through, so the
/// `.x1` and `.x8` cases are one derivation with the kernel swapped and cannot
/// drift into checking two different things.
fn check_sttm_roundtrip(
    stream: &CudaStream,
    drain: impl Fn(LaunchConfig, &mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>,
) -> Result<String, Box<dyn Error>> {
    let (slots, values) = (RegTile::<32, COLUMNS, BaseLdtm>::SLOTS, COLUMNS / 4);
    let mut out = DeviceBuffer::<f32>::zeroed(stream, ROWS * slots * values)?;
    drain(launch_config(ROWS as u32, STTM_SHARED as u32), &mut out)?;
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
/// Does Cluster Launch Control do the three things `pipeline::run_stealing`
/// assumes of it? (#88)
///
/// The scheduler rests on three claims about one instruction, and a persistent
/// GEMM built on a wrong one either hangs or drops an output tile — neither of
/// which points at the instruction. So they are asked directly, of a kernel
/// small enough that its only content is the question:
///
/// 1. **`.multicast::cluster::all` completes the transaction in every CTA of
///    the requesting cluster.** Every rank waits on its *own* copy of the
///    barrier while only rank 0 issues, so if the multicast reached only the
///    issuer, the peer would wait forever. Asked as `completed`, against a
///    deadline, so the failing case reports rather than hangs.
/// 2. **Both ranks are told the same thing.** A pair that disagreed would put
///    the two halves of one cooperative MMA on different output tiles, which is
///    #51's bug arriving through a new door and silently.
/// 3. **The cancelled unit is a cluster, not a CTA.** Under a 2-CTA cluster a
///    cluster-granular response has an even `first_ctaid_x`; an odd one would
///    mean the item map has to be something other than `ctaid_x / cluster_size`.
///
/// It also settles the polarity `cuda_device::clc`'s module doc and its
/// function doc disagree about, in the only way that is not an argument: a
/// grid this much larger than the device holds *must* produce cancellations, so
/// if `is_canceled` were 1-means-nothing-left, no row would ever carry a
/// coordinate.
fn check_clc(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let blocks = CLC_CLUSTERS * 2;
    let mut out = DeviceBuffer::<u64>::zeroed(stream, blocks as usize * CLC_FIELDS)?;
    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: CLC_SHARED_BYTES,
    };
    // Safety: one warp, as the kernel's doc names; each CTA writes only the
    // `CLC_FIELDS` words at its own `blockIdx.x`; the grid is a multiple of two,
    // which the `#[cluster_launch(2, 1, 1)]` launcher requires.
    unsafe { module.clc_probe(stream, config, CLC_DEADLINE_NS, &mut out)? };
    finish_or_abort(context, stream, "clc probe")?;
    let rows = out.to_host_vec(stream)?;
    let row = |cta: usize| {
        let base = cta * CLC_FIELDS;
        (rows[base], rows[base + 1], rows[base + 2], rows[base + 3])
    };

    // A cluster that was itself cancelled never ran, so its rows are still the
    // zeros the buffer was allocated with. That is the mechanism under test
    // working, not a failure, and `launched` is what tells the two apart.
    let launched: Vec<usize> = (0..blocks as usize)
        .filter(|&cta| row(cta).0 == 1)
        .collect();
    let stalled = launched.iter().filter(|&&cta| row(cta).1 == 0).count();
    if stalled > 0 {
        return Err(format!(
            "{stalled} of {} launched CTAs never saw their copy of the response within \
             {CLC_DEADLINE_NS} ns. multicast::cluster::all does not complete the \
             transaction in every rank at this rev, so a scheduler that has each rank \
             wait on its own barrier deadlocks",
            launched.len()
        )
        .into());
    }

    let (mut split, mut odd, mut cancelled, mut ran) = (0usize, 0usize, 0usize, 0usize);
    for cluster in 0..CLC_CLUSTERS as usize {
        let (leader_launched, _, leader_canceled, leader_first) = row(2 * cluster);
        let (peer_launched, _, peer_canceled, peer_first) = row(2 * cluster + 1);
        if leader_launched != peer_launched {
            return Err(format!(
                "cluster {cluster} had one rank launched and the other not, which no \
                 cancellation of a whole cluster can produce"
            )
            .into());
        }
        if leader_launched == 0 {
            continue;
        }
        ran += 1;
        if leader_canceled != peer_canceled || leader_first != peer_first {
            split += 1;
        }
        if leader_canceled == 1 {
            cancelled += 1;
            if leader_first % 2 != 0 || leader_first >= blocks as u64 {
                odd += 1;
            }
        }
    }
    if split > 0 {
        return Err(format!(
            "{split} of {ran} launched clusters had their two ranks told different things — \
             a pair built on this would split one output tile across two items (#51)"
        )
        .into());
    }
    if odd > 0 {
        return Err(format!(
            "{odd} of {cancelled} cancellations named a CTA that is not the first of a \
             cluster, so the cancelled unit is not a cluster and item = ctaid_x / \
             cluster_size is the wrong map"
        )
        .into());
    }
    if cancelled == 0 {
        return Err(format!(
            "no cluster of {ran} that ran stole anything, on a grid far larger than the \
             device holds. Either nothing was ever pending, or is_canceled reads 1 for \
             'nothing left' — which is the polarity cuda_device::clc's module example uses"
        )
        .into());
    }
    // The two must account for the whole grid exactly once: every cluster
    // either ran or was cancelled by one that did. That is the property a
    // scheduler needs and the one a dropped tile would break.
    let missing = CLC_CLUSTERS as usize - ran;
    Ok(format!(
        "{ran} clusters ran, {cancelled} of them stole a cluster-aligned item, \
         {missing} were cancelled and never launched, both ranks agreed everywhere, \
         no launched rank stalled"
    ))
}

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

/// `A[m, k]` of the operand-order cases — integers in `[-15, 15]`.
///
/// The pair with [`walk_b`] is chosen against what the reference has to be
/// able to *see*, which is the lesson of #48: `b_value`'s `depth * 5 % 5` was
/// identically zero there, so an exact GEMM check certified a kernel that
/// read one plane of `B` forever. Both generators here vary along every one
/// of their arguments, and [`walk_blind_spots`] is the standing proof of it —
/// it removes each K plane in turn, permutes the K chunks, and swaps the
/// stacked MN subtiles, and requires the reference to notice every time.
///
/// Neither generator is symmetric in its two arguments, and the staged `A` is
/// `[64, 128]` against `[128, 64]`, so `Aᵀ·B` and `A·B` are not the same
/// numbers and cannot be confused by an operand that happens to be its own
/// transpose. Every value is an integer under 32 (exact in bf16, which holds
/// every integer to 256) and every dot product is under `15 · 30 · 64 =
/// 28,800` (exact in fp32, whose integers run to 2²⁴), so the comparison is
/// `==` and a mismatch is an address and never a rounding artifact.
fn walk_a(m: usize, k: usize) -> f32 {
    ((m * 13 + k * 7) % 31) as f32 - 15.0
}

/// `B[k, n]` of the operand-order cases — integers in `[-30, 30]`. See
/// [`walk_a`].
fn walk_b(k: usize, n: usize) -> f32 {
    ((n * 11 + k * 5) % 61) as f32 - 30.0
}

/// A `[rows, columns]` bf16 operand as the packed `u32` words a device buffer
/// holds, row-major — the staging a [`kittens::global::PanelMap`] describes.
/// The four staged buffers differ only in which of `(m, k)` / `(k, n)` this
/// walks fastest, since no TMA box transposes on the way in.
fn stage_walk_operand(
    rows: usize,
    columns: usize,
    value: impl Fn(usize, usize) -> f32,
) -> Vec<u32> {
    let mut staged = Vec::with_capacity(rows * columns / 2);
    for row in 0..rows {
        for pair in 0..columns / 2 {
            staged.push(pack(
                to_bf16(value(row, 2 * pair)),
                to_bf16(value(row, 2 * pair + 1)),
            ));
        }
    }
    staged
}

/// `D[m, n] = Σₖ A[m, remap(k)] · B[k, n]` — the reference at `remap =
/// identity`, and a named wrong K walk otherwise.
fn walk_product(remap: impl Fn(usize) -> usize) -> Vec<f32> {
    let mut product = vec![0.0f32; ROWS * COLUMNS];
    for m in 0..ROWS {
        for n in 0..COLUMNS {
            let mut sum = 0.0f32;
            for k in 0..DEPTH {
                sum += walk_a(m, remap(k)) * walk_b(k, n);
            }
            product[m * COLUMNS + n] = sum;
        }
    }
    product
}

/// The reference every operand-order case is compared against. All four walks
/// stage the *same* two logical matrices — only the majorness of the staging
/// differs — so one reference serves the whole square, and a walk that read
/// its operand under the other order is wrong against the same numbers the
/// other three are right against.
fn walk_reference() -> Vec<f32> {
    walk_product(|k| k)
}

/// A named wrong K walk, as the remap of `A`'s K index that produces it.
type ChunkHypothesis = (&'static str, fn(usize) -> usize);
/// A named confusion of output coordinates, as the permutation of `(m, n)`
/// that produces it.
type CoordinateHypothesis = (&'static str, fn(usize, usize) -> (usize, usize));

/// Does [`walk_reference`] actually depend on everything an operand-order
/// walk computes?
///
/// A reference that does not vary along a coordinate cannot catch an error in
/// that coordinate, and a check built on one certifies whatever it is handed.
/// This is the host case that says it does not: every K plane, every K chunk
/// permutation a chained walk could get wrong, and every confusion between
/// the two stacked MN subtiles the transposed walks reach by leading offset.
/// It runs on the host and needs no GPU.
fn walk_blind_spots() -> Result<String, Box<dyn Error>> {
    let reference = walk_reference();
    let mut visible = 0usize;

    // Every K plane. Dropping plane `k` subtracts `A[m, k] · B[k, n]` from
    // every element exactly (all partial sums are exact integers), so the
    // plane is visible exactly when that product is nonzero somewhere.
    for k in 0..DEPTH {
        let carries_a = (0..ROWS).any(|m| walk_a(m, k) != 0.0);
        let carries_b = (0..COLUMNS).any(|n| walk_b(k, n) != 0.0);
        if !(carries_a && carries_b) {
            return Err(format!(
                "K plane {k} contributes nothing: a walk that skipped it would pass"
            )
            .into());
        }
        visible += 1;
    }

    // K chunk order. A chained walk issues `DEPTH / 16` instructions and
    // pairs `A`'s chunk with `B`'s; these are the ways that pairing goes
    // wrong without going out of bounds.
    let chunk_hypotheses: [ChunkHypothesis; 4] = [
        ("A's K chunks reversed", |k| {
            (WALK_CHUNKS - 1 - k / 16) * 16 + k % 16
        }),
        ("A's K chunks rotated by one", |k| {
            ((k / 16 + 1) % WALK_CHUNKS) * 16 + k % 16
        }),
        ("A's chunk 1 re-reading chunk 0", |k| {
            if k / 16 == 1 { k - 16 } else { k }
        }),
        ("K reversed within each chunk", |k| {
            (k / 16) * 16 + 15 - k % 16
        }),
    ];
    for (name, remap) in chunk_hypotheses {
        if walk_product(remap) == reference {
            return Err(format!("{name} gives the same reference — it is a blind spot").into());
        }
        visible += 1;
    }

    // Coordinate confusions. `M` and `N` are 128 here and a swizzle subtile
    // is 64 wide, so both are two stacked subtiles an MN-major walk reaches
    // only through the descriptor's leading offset. A leading offset of zero
    // reads the first subtile twice; a wrong one swaps them.
    let coordinate_hypotheses: [CoordinateHypothesis; 5] = [
        ("A's MN subtiles swapped", |m, n| (m ^ 64, n)),
        ("B's MN subtiles swapped", |m, n| (m, n ^ 64)),
        ("A's second subtile reading its first", |m, n| (m % 64, n)),
        ("B's second subtile reading its first", |m, n| (m, n % 64)),
        ("the product transposed", |m, n| (n, m)),
    ];
    for (name, permute) in coordinate_hypotheses {
        let permuted: Vec<f32> = (0..ROWS * COLUMNS)
            .map(|index| {
                let (m, n) = permute(index / COLUMNS, index % COLUMNS);
                reference[m * COLUMNS + n]
            })
            .collect();
        if permuted == reference {
            return Err(format!("{name} gives the same reference — it is a blind spot").into());
        }
        visible += 1;
    }

    Ok(format!("{visible} wrong walks, every one of them visible"))
}

/// Which of the four operand orders a case issues — and the control, which is
/// none of them.
#[derive(Clone, Copy)]
enum Order {
    Abt,
    Ab,
    AtB,
    AtBt,
    /// [`Self::AtB`]'s operands under the untransposed walk. Required to
    /// *disagree* with the reference; see
    /// [`kernels::walk_untransposed_control`].
    UntransposedControl,
}

impl Order {
    const ALL: [Order; 5] = [
        Order::Abt,
        Order::Ab,
        Order::AtB,
        Order::AtBt,
        Order::UntransposedControl,
    ];

    fn name(self) -> &'static str {
        match self {
            Order::Abt => "mma AB\u{1d40}",
            Order::Ab => "mma AB",
            Order::AtB => "mma A\u{1d40}B",
            Order::AtBt => "mma A\u{1d40}B\u{1d40}",
            Order::UntransposedControl => "mma transpose control",
        }
    }

    /// The `[rows, columns]` of the accumulator this order actually writes,
    /// and so the only part of the dump that carries a claim. Every real walk
    /// covers the whole band; the control covers one subtile square (see
    /// [`kernels::walk_untransposed_control`]).
    fn region(self) -> (usize, usize) {
        match self {
            Order::UntransposedControl => (CONTROL_EDGE, CONTROL_EDGE),
            _ => (ROWS, COLUMNS),
        }
    }

    /// Which staged majorness each operand wants: `(A transposed, B
    /// transposed)`, where "transposed" is the MN-major staging.
    fn staging(self) -> (bool, bool) {
        match self {
            Order::Abt => (false, false),
            Order::Ab => (false, true),
            Order::AtB | Order::UntransposedControl => (true, true),
            Order::AtBt => (true, false),
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
        let config = launch_config(ROWS as u32, WALK_SHARED as u32);
        unsafe {
            match self {
                Order::Abt => module.walk_abt(stream, config, a_map, b_map, out),
                Order::Ab => module.walk_ab(stream, config, a_map, b_map, out),
                Order::AtB => module.walk_atb(stream, config, a_map, b_map, out),
                Order::AtBt => module.walk_atbt(stream, config, a_map, b_map, out),
                Order::UntransposedControl => {
                    module.walk_untransposed_control(stream, config, a_map, b_map, out)
                }
            }
        }
    }
}

/// Does the walk `order` names compute `A·B` at the coordinates
/// [`BaseLdtm`] claims?
///
/// The kernel dumps by `(warp, lane, slot, value)` and nothing else, so the
/// map is applied here and never there — the same discipline the fragment
/// cases follow, over a product the kernel has no way to encode.
fn check_walk(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    order: Order,
) -> Result<String, Box<dyn Error>> {
    let a_k = DeviceBuffer::from_host(stream, &stage_walk_operand(ROWS, DEPTH, walk_a))?;
    let a_mn = DeviceBuffer::from_host(
        stream,
        &stage_walk_operand(DEPTH, ROWS, |k, m| walk_a(m, k)),
    )?;
    let b_k = DeviceBuffer::from_host(
        stream,
        &stage_walk_operand(COLUMNS, DEPTH, |n, k| walk_b(k, n)),
    )?;
    let b_mn = DeviceBuffer::from_host(stream, &stage_walk_operand(DEPTH, COLUMNS, walk_b))?;

    let a_k_map =
        unsafe { encode_bf16_panels::<ROWS, DEPTH>(stream, a_k.cu_deviceptr(), ROWS, 1)? };
    let a_mn_map =
        unsafe { encode_bf16_panels::<DEPTH, ROWS>(stream, a_mn.cu_deviceptr(), DEPTH, 1)? };
    let b_k_map =
        unsafe { encode_bf16_panels::<COLUMNS, DEPTH>(stream, b_k.cu_deviceptr(), COLUMNS, 1)? };
    let b_mn_map =
        unsafe { encode_bf16_panels::<DEPTH, COLUMNS>(stream, b_mn.cu_deviceptr(), DEPTH, 1)? };

    let (a_transposed, b_transposed) = order.staging();
    let a_map = if a_transposed { &a_mn_map } else { &a_k_map };
    let b_map = if b_transposed { &b_mn_map } else { &b_k_map };

    let mut out = DeviceBuffer::<f32>::zeroed(stream, ROWS * COLUMNS)?;
    unsafe { order.launch(module, stream, a_map.as_ptr(), b_map.as_ptr(), &mut out)? };
    let observed = out.to_host_vec(stream)?;

    let reference = walk_reference();
    let (region_rows, region_columns) = order.region();
    let mut report = String::new();
    let (mut compared, mut mismatches) = (0usize, 0usize);
    for warp in 0..ROWS / 32 {
        for lane in 0..32u32 {
            for slot in 0..WALK_SLOTS {
                for value in 0..WALK_VALUES {
                    let index = dump_index(warp, lane, slot, value, WALK_SLOTS, WALK_VALUES);
                    let (row, column) =
                        RegTile::<32, COLUMNS, BaseLdtm>::coordinate(lane, slot, value);
                    let (m, n) = (32 * warp + row as usize, column as usize);
                    if m >= region_rows || n >= region_columns {
                        continue;
                    }
                    compared += 1;
                    let expected = reference[m * COLUMNS + n];
                    if observed[index] == expected {
                        continue;
                    }
                    mismatches += 1;
                    if mismatches <= 8 && !matches!(order, Order::UntransposedControl) {
                        let _ = write!(
                            report,
                            "\n    D[{m}, {n}] (warp {warp} lane {lane} slot {slot} \
                             value {value}): got {}, want {expected}",
                            observed[index]
                        );
                    }
                }
            }
        }
    }

    if matches!(order, Order::UntransposedControl) {
        return if mismatches == 0 {
            Err(
                "the untransposed walk reproduced Aᵀ·B exactly, so the transposed \
                 cases prove nothing about their transpose bits"
                    .into(),
            )
        } else {
            Ok(format!(
                "{mismatches} of {compared} values differ from Aᵀ·B, as they must"
            ))
        };
    }
    if mismatches == 0 {
        return Ok(format!(
            "{compared} values exact, {WALK_CHUNKS} chained K chunks"
        ));
    }
    Err(format!("{mismatches} of {compared} values wrong{report}").into())
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
    // `cargo oxide run device-tests -- bench-ladder` (#63): the ladder's own
    // rungs, checked and then *timed*, so a register claim can be compared
    // against a clock. Behind an argument because it needs minutes of a B200
    // where the correctness run needs seconds.
    if std::env::args().nth(1).as_deref() == Some("bench-ladder") {
        return ladder_bench::main();
    }
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
    // The four operand orders (#12), against a reference that varies along
    // every coordinate they compute — and the control that says the
    // transposed ones had to be transposed. The blind-spot sweep goes first
    // because everything after it is only as good as it is.
    cases.push(("walk blind spots", Box::new(walk_blind_spots)));
    for order in Order::ALL {
        cases.push((
            order.name(),
            Box::new(move || check_walk(stream, module, order)),
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
        Box::new(|| {
            check_sttm_roundtrip(stream, |config, out| unsafe {
                module.sttm_roundtrip(stream, config, out)
            })
        }),
    ));
    // The same round trip read back through `tcgen05.ld.16x256b.x8` (#117).
    // It goes here rather than beside the `fragment map` cases because it is
    // the case above with one line changed, and the pair is what says the wide
    // drain and the narrow one agree.
    cases.push((
        "ldtm x8 map",
        Box::new(|| {
            check_sttm_roundtrip(stream, |config, out| unsafe {
                module.ldtm_x8_map(stream, config, out)
            })
        }),
    ));
    // And the same tile again with every issue of the band outstanding at once,
    // which is a claim about what one `tcgen05.wait::ld` retires.
    cases.push((
        "ldtm x8 batched map",
        Box::new(|| {
            check_sttm_roundtrip(stream, |config, out| unsafe {
                module.ldtm_x8_batched_map(stream, config, out)
            })
        }),
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
    // The path with no descriptor in it (#11). Only the load direction: the
    // store's is `gemm`'s epilogue, checked exactly.
    cases.push((
        "global rows map",
        Box::new(|| check_global_rows(stream, module)),
    ));
    // Its vector twin (#172): the per-column operand, off the same matrix.
    cases.push((
        "global column map",
        Box::new(|| check_global_cols(stream, module)),
    ));
    // The other vector twin: the same `ColVec`, read *down* a column instead
    // of along a row. Separate from the case above because the two share a
    // type and an argument list and differ only in the walk.
    cases.push((
        "global column statistic",
        Box::new(|| check_global_col_vec(stream, module)),
    ));
    // The vector shape (#13): an unswizzled box, at rank 1 and rank 2.
    cases.push((
        "shared vector round trip",
        Box::new(|| check_shared_vec(stream, module)),
    ));
    cases.push((
        "shared vector row",
        Box::new(|| check_shared_vec_row(stream, module)),
    ));
    // The fold no shuffle can do (#3): four warps, one shared vector, three
    // reductions in a row on it.
    cases.push((
        "block reduction",
        Box::new(|| check_block_reduce(stream, module)),
    ));
    cases.push((
        "tma store early recycle",
        Box::new(|| {
            check_tma_store::<TILE, TILE>(stream, |config, source, packed, pitched| unsafe {
                module.tma_store_recycle(stream, config, source, packed, pitched)
            })
        }),
    ));
    // The reduction store (#42): the same engine path with an add at the
    // destination, seeded nonzero so nothing that overwrites can pass.
    cases.push((
        "tma reduce-add store",
        Box::new(|| {
            check_tma_store_add(stream, |config, source, dest| unsafe {
                module.tma_store_add_twice(stream, config, source, dest)
            })
        }),
    ));
    // The epilogue's shared→global half as one primitive (#15): `stmatrix`
    // into a staging ring and a bulk store out of it, with the proxy fence and
    // the two group waits inside. Depth 1 is what `softmax` does by hand;
    // everything above it is the reuse hazard, which depth 1 cannot exercise.
    cases.push((
        "store ring depth 1",
        Box::new(|| {
            check_store_ring::<RING_ROWS, TILE>(
                stream,
                ring_shared::<TILE, 0>(),
                |config, map| unsafe { module.store_ring_depth_1(stream, config, map) },
            )
        }),
    ));
    cases.push((
        "store ring depth 2",
        Box::new(|| {
            check_store_ring::<RING_ROWS, TILE>(
                stream,
                ring_shared::<TILE, 1>(),
                |config, map| unsafe { module.store_ring_depth_2(stream, config, map) },
            )
        }),
    ));
    cases.push((
        "store ring depth 4",
        Box::new(|| {
            check_store_ring::<RING_ROWS, TILE>(
                stream,
                ring_shared::<TILE, 3>(),
                |config, map| unsafe { module.store_ring_depth_4(stream, config, map) },
            )
        }),
    ));
    cases.push((
        "store ring depth 2 wide",
        Box::new(|| {
            check_store_ring::<RING_ROWS, WIDE>(
                stream,
                ring_shared::<WIDE, 1>(),
                |config, map| unsafe { module.store_ring_depth_2_wide(stream, config, map) },
            )
        }),
    ));
    // The same ring at **warp scope** (#122): four warps, four buffers, four
    // commits a band, and `bar.warp.sync` carrying the proxy fence to the lane
    // that issues. `gemm`'s TMA epilogue stages this way because its staging
    // tiles are per warp, and the phased rung is that kernel's own offset — a
    // shared plan rounded to 128 and not to 1024, so every buffer starts
    // mid-swizzle-period and the engine has to read the phase off the address.
    cases.push((
        "store ring warp",
        Box::new(|| {
            check_store_ring::<RING_WARP_ROWS, TILE>(
                stream,
                warp_ring_shared::<TILE, 0>(),
                |config, map| unsafe { module.store_ring_warp(stream, config, map) },
            )
        }),
    ));
    cases.push((
        "store ring warp phased",
        Box::new(|| {
            check_store_ring::<RING_WARP_ROWS, TILE>(
                stream,
                warp_ring_shared::<TILE, 1>(),
                |config, map| unsafe { module.store_ring_warp_phased(stream, config, map) },
            )
        }),
    ));
    // The same half of an epilogue without the engine: `stmatrix` into a
    // staging tile and plain vectorized stores out of it, at every rung of the
    // access ladder. No descriptor, no proxy fence, and no wait owed at the end
    // — which is most of what distinguishes it from the four cases above.
    cases.push((
        "shared drain",
        Box::new(|| {
            check_shared_drain::<TILE>(stream, |config, column, out| unsafe {
                module.shared_drain(stream, config, column, out)
            })
        }),
    ));
    cases.push((
        "shared drain wide",
        Box::new(|| {
            check_shared_drain::<WIDE>(stream, |config, column, out| unsafe {
                module.shared_drain_wide(stream, config, column, out)
            })
        }),
    ));
    // The same two rectangles folded in rather than overwritten (#169). Same
    // addresses, same widths, one extra load — so what these two add over the
    // pair above is the read and the single rounding, and nothing about the
    // geometry.
    cases.push((
        "shared accumulate",
        Box::new(|| {
            check_shared_accumulate::<TILE>(stream, |config, column, out| unsafe {
                module.shared_accumulate(stream, config, column, out)
            })
        }),
    ));
    cases.push((
        "shared accumulate wide",
        Box::new(|| {
            check_shared_accumulate::<WIDE>(stream, |config, column, out| unsafe {
                module.shared_accumulate_wide(stream, config, column, out)
            })
        }),
    ));
    // The same rectangle at fp32, which had no staged route at all until #174:
    // `stmatrix` moves b16 matrices, so an fp32 band reaches its staging tile
    // one value at a time through `scatter_tile`. The two `register drain`
    // cases are the route it replaces, writing the same rectangle from the same
    // band with no shared hop — so they are the control for correctness here
    // and for register pressure in the `regcount` table.
    cases.push((
        "scatter drain",
        Box::new(|| {
            check_scatter_drain::<F32_TILE>(stream, |config, column, out| unsafe {
                module.scatter_drain(stream, config, column, out)
            })
        }),
    ));
    cases.push((
        "scatter drain wide",
        Box::new(|| {
            check_scatter_drain::<F32_WIDE>(stream, |config, column, out| unsafe {
                module.scatter_drain_wide(stream, config, column, out)
            })
        }),
    ));
    cases.push((
        "register drain",
        Box::new(|| {
            check_scatter_drain::<F32_TILE>(stream, |config, column, out| unsafe {
                module.register_drain(stream, config, column, out)
            })
        }),
    ));
    cases.push((
        "register drain wide",
        Box::new(|| {
            check_scatter_drain::<F32_WIDE>(stream, |config, column, out| unsafe {
                module.register_drain_wide(stream, config, column, out)
            })
        }),
    ));
    // What a tcgen05 allocation costs an SM (#74), against a control that is
    // the same kernel with the allocator removed. Not a round trip through a
    // library path like everything above it — the claim is about the *driver's*
    // occupancy answer, which is why the probe is small enough that nothing in
    // it can be the cause but the one line that varies.
    cases.push((
        "tmem occupancy ladder",
        Box::new(|| tmem_occupancy::check(stream, module)),
    ));
    // And what the SM *actually* holds (#78), which the case above cannot say:
    // it reports the driver's answer, and #51 showed that answer failing to
    // predict `gemm`'s residency by a factor of two. This one counts CTAs by
    // `%smid` and `%globaltimer` instead of asking.
    cases.push((
        "tmem residency census",
        Box::new(|| tmem_residency::check(stream, module)),
    ));
    // #88's three assumptions about `clusterlaunchcontrol`, asked of the
    // instruction rather than of a GEMM built on it. It goes here, next to the
    // census, because both are probes of hardware behaviour that no query
    // reports — and before `repeated launch`, which can take the process down.
    cases.push((
        "clc work stealing",
        Box::new(|| check_clc(&context, stream, module)),
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
