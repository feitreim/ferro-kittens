//! # Layernorm over a tile, and the whole-tile reduction next to it
//!
//! **Status: [`kernels::layernorm_rows`] runs** — [`check`] launches it against
//! a CPU reference on a B200 (`modal run modal_app.py::examples`) and
//! [`bench`] times it at six sizes. [`kernels::groupnorm_tile`] still only
//! compiles. Both are in the default build. They differ in exactly one thing —
//! the axis their statistic is taken over — and that is what kept them apart
//! for two issues.
//!
//! ## Why it walks the row in chunks (#67)
//!
//! `layernorm_rows` held a whole `[32, COLUMNS]` band — 128 fp32 a thread —
//! and said the algorithm in four lines over it. `regcount` priced that at
//! **255 registers, a 1552-byte stack frame and 16 bytes of spill**, and the
//! occupancy query in `main` read **2 blocks and 8 warps an SM** against
//! `softmax_rows`' 6 and 24 at 33,344 against 32,800 bytes of shared memory.
//!
//! It is the mirror of #47 and it is worth stating as a pair, because between
//! them they say what the rule actually is. `softmax_rows` was **39** registers
//! on a 1024-byte frame: too few, the band never promoted, and the cost landed
//! on memory traffic — 2.6×. This kernel was 255 on a 1552-byte frame: the band
//! *still* did not fit, and on top of that the registers it did hold were
//! enough to cap the SM at two CTAs. Neither number is a target. **The band
//! living in registers with the frame at zero is the target**, and reading
//! either count as a speed is what hid #47 for as long as it hid.
//!
//! The fix is [`CHUNK`] and it is one number, exactly as #47's was: the row is
//! walked a chunk at a time and `total`/`square` carry across the chunks. The
//! launch geometry, the barrier protocol, the TMA plan and both waits are
//! untouched, so what changed between the measurements is only where the band
//! lives.
//!
//! ## The number is 16, and #47's 32 would have been wrong
//!
//! #60's warning is that `M·N` predicts nothing — `[64, 64]` and `[32, 128]`
//! are both 128 fp32 a thread and 220 registers apart. Here the *row* extent is
//! not a free parameter at all: a warp owns 32 rows of a 128-row tile, and
//! changing that is a launch-geometry change and a different issue. So the only
//! ladder that applies is the `[32, N]` column sweep — and this kernel was
//! built at every rung of it rather than handed softmax's answer, which is the
//! part that mattered:
//!
//! | `CHUNK` | regs | spill | frame | blocks/SM | GB/s at 8192 blocks |
//! | --- | --- | --- | --- | --- | --- |
//! | 128 — the whole band, as filed | 255 | 16 B | 1552 | 2 | 317 |
//! | 64 | 108 | 0 | 272 | 6 | — |
//! | 32 — softmax's answer | **40** | 0 | 128 | 6 | 3776 |
//! | **16** | 47 | 0 | **0** | 6 | **5539** |
//!
//! 317 → 5539 GB/s, a factor of **17.5**, and the flat floor at the small end
//! goes 32.4 µs → 8.4 µs with it. That floor is worth a line of its own: #47
//! read `softmax`'s 23 µs at two blocks as "a fixed cost every launch on this
//! harness pays", and this kernel reaches 8.4 µs on the same harness. Whatever
//! that 23 µs is, it is not the launch.
//!
//! **`CHUNK = 32` is the cheaper kernel by register count and the slower one by
//! 1.46×.** It is seven registers under the shipped form and carries a 128-byte
//! frame the shipped form does not, which is #47's defect in miniature: those
//! seven registers are the band not fitting. Reading the register column as a
//! speed picks the wrong rung here, and this is the third time in this
//! repository that it would have (#47, #63, and now this).
//!
//! The mechanism was checked from the other side too, on the kernel this file
//! does not touch: `softmax_rows` at `CHUNK = 16` prices at 50 registers on a
//! **128**-byte frame against its shipped 66 on **256**, and runs 1185 against
//! 950 GB/s. Half the frame, a quarter more throughput, on a kernel whose
//! arithmetic did not change — so "frame down, time down" is not a property of
//! this kernel's rewrite. That probe is deliberately **not** in this pull
//! request: `softmax`'s own header argues for 32 at length and rewriting it
//! belongs to its own issue, with its own controls.
//!
//! Note also what the ladder in `experiments/README.md` says about `[32, 32]` — a
//! **zero** frame, in all five spellings. This kernel gets 128 bytes at that
//! shape. The probe is not the kernel: it carries no `ColVec` parameters and no
//! statistics across chunks, and this kernel's extra live state moves its cliff
//! one rung down. The ladder narrows the search to a handful of rungs; it does
//! not answer for a kernel that has to be built.
//!
//! Occupancy is the same 6 blocks an SM at both 16 and 32 — shared memory binds
//! there, as it does for `softmax` — so the 1.46× is not occupancy. It is the
//! local-memory traffic itself, which is #63's answer read the other way: a
//! streamed band is cheap when the registers it frees buy resident warps, and
//! this kernel is past the point where they buy anything.
//!
//! [`kernels::groupnorm_tile`] is deliberately left holding the whole band, at
//! 236 registers on a 1536-byte frame. It has no launcher and no reference, and
//! a rewrite nothing can check is not an improvement — but it shares this
//! file, the `SharedVec` staging and the `[32, COLUMNS]` band, so its counters
//! *not* moving while `layernorm_rows`' do is the control that says the change
//! is the band walk and not something the module did.
//!
//! `layernorm_rows` was blocked on **#13**: `gamma`/`beta` are parameters, not
//! activations. They are loaded once, shared by every row, and belong in shared
//! memory, so with no `SharedVec` every thread would have re-read them from
//! global. [`kittens::shared::SharedVec`] is now that type — one flat run of
//! elements with its own single-box TMA path, unswizzled because a swizzle
//! atom is a statement about a tile's *rows* and a vector has one — and
//! [`kittens::ldst::load_vec`] broadcasts it into the `ColVec` `mul_col`
//! consumes.
//!
//! [`kernels::groupnorm_tile`] normalizes over the *whole* tile rather than per
//! row, and that was the one place a landed #6 did not reach.
//! `RegTile::tile_sum` is warp scope: it folds a warp's own band across all 32
//! lanes and stops. A `[128, COLUMNS]` tile is four warps' bands, so the warps
//! have to agree — and warps cannot shuffle to each other, so what closes the
//! gap is *storage* plus a block barrier rather than another butterfly.
//!
//! **#3** shipped both halves of that: [`kittens::sync::block_reduce_sum`]
//! folds one warp-uniform value per warp into one block-uniform value through
//! a `SharedVec<F32, WARPS>`, and the `impl Element for F32` under it is what
//! lets four fp32 partials be a shared vector at all. The two calls below are
//! back to back on the same scratch with no barrier between them, which is a
//! property of the collective and not an accident: it syncs on both sides, so
//! the variance pass cannot overtake the mean pass's readers.
//!
//! ## What the check has to be, and why it is not softmax's
//!
//! `layernorm_rows` had no launcher, no CPU reference and no row in
//! `bench.rs`, so "faster" could not be told from "wrong faster" — and that is
//! why it is fixed first here rather than last. The seed is not a copy of
//! [`crate::softmax`]'s, and the reason is on [`value`]: a seed that gives
//! every row the same values in a different order leaves the row statistics
//! identical, and against *that* input the whole-tile kernel below computes
//! layernorm's answer exactly. A softmax seed cannot see the difference
//! between the two kernels in this file.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, launch_bounds, thread, warp};

// Host side: the launcher's error type, and the benchmark's size and clock.
use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::ldst::{load_tile, load_vec, store_tile};
use kittens::reg::{BaseLdtm, ColVec, RegTile, RegVec, rsqrt};
use kittens::shared::{
    Bf16, F32, SharedTile, SharedVec, Swizzle128B, tma_store_commit, tma_store_wait,
};
use kittens::sync::{Semaphore, block_reduce_sum};

const ROWS: usize = 128;
const COLUMNS: usize = 128;
/// Warps a CTA runs, and so the number of partials a whole-tile statistic is
/// folded from — one band each.
const WARPS: usize = ROWS / 32;
/// Columns [`kernels::layernorm_rows`] holds in registers at once, and the
/// whole of #67.
///
/// The widest `[32, N]` band this kernel gets a **zero** stack frame at —
/// measured at 16, 32, 64 and 128 rather than inherited from
/// [`crate::softmax`], whose answer is 32 and is 1.46× slower here. The module
/// header has the table and the reason.
const CHUNK: usize = 16;
/// Chunks a row is walked in.
const CHUNKS: usize = COLUMNS / CHUNK;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows by the whole row — what [`kernels::groupnorm_tile`]
/// still holds, and what `layernorm_rows` used to.
type Band = RegTile<32, COLUMNS, BaseLdtm>;
/// One warp's 32 rows by [`CHUNK`] columns.
type Chunk = RegTile<32, CHUNK, BaseLdtm>;
/// A per-row scalar of those 32 rows — the running sum, then the deviation.
type Rows = RegVec<32, BaseLdtm>;
/// A per-column parameter over one chunk — one value per logical column,
/// replicated across the lanes that hold that column.
type Columns = ColVec<CHUNK, BaseLdtm>;
/// `gamma` and `beta`, staged once per CTA. `COLUMNS` bf16 is 256 bytes: the
/// vector's own length, where a one-row `SharedTile` would have spent a whole
/// 128-byte swizzle atom to hold 64 of them.
type Parameters = SharedVec<Bf16, COLUMNS>;
/// One chunk's slice of one of them. A vector is flat and unswizzled, so a
/// slice of it is a base address and nothing else — which is why the chunked
/// walk below needs no addressing the library did not already have.
type ParameterChunk = SharedVec<Bf16, CHUNK>;
/// The four warps' partial statistics, between the two barriers a block
/// reduction is made of. fp32 and not bf16: a partial rounded on its way
/// through shared memory would lose eight bits of the sum, and the variance
/// pass would inherit the error of the mean pass. 16 bytes, and never a TMA
/// box — the whole life of this vector is [`SharedVec::set`] then
/// [`SharedVec::get`], one barrier apart.
type Partials = SharedVec<F32, WARPS>;

pub const SHARED_BYTES: usize = Tile::BYTES + 2 * Parameters::BYTES + 64;
pub const THREADS: u32 = (WARPS * 32) as u32;

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
                // One instruction each for the vectors, against a rank-1 map:
                // an unswizzled box has no atom to be cut into, so a vector
                // never walks stacked subtiles the way the tile does. Each
                // load hands back its own charge and the barrier is told their
                // sum, so the three-term total cannot drift from the three
                // calls above it (#29).
                let staged = tile.tma_load(source, row, 0, loaded)
                    + gamma.tma_load(gamma_map, 0, loaded)
                    + beta.tma_load(beta_map, 0, loaded);
                loaded.expect_tx(staged);
            }
            loaded.wait(0);
            thread::sync_threads();

            let chunks = tile.chunk_writer();

            // Three passes over the shared tile, each holding one [`CHUNK`]-wide
            // band at a time. `row_sum` (#6) folds a thread's own values and
            // then the quad holding the rest of the chunk, and a chunk's quad
            // is the same four lanes whatever the chunk, so the per-chunk
            // results combine slotwise with no shuffle of their own.
            //
            // `scale`/`shift` (#38) keep `1/COLUMNS` and `epsilon` in a
            // register: before them a constant had to be splatted into a whole
            // `RegVec` to be combined with one.
            let mut total = Rows::splat(0.0);
            let mut chunk = 0usize;
            while chunk < CHUNKS {
                let x: Chunk = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                total.add_assign(x.row_sum());
                chunk += 1;
            }
            let mean = total.scale(1.0 / COLUMNS as f32);

            // Centred and then squared, rather than the raw second moment minus
            // the square of the mean. The one-pass form would let the mean be
            // computed and the variance accumulated together — but this seed's
            // rows sit up to 27 away from zero, so `E[x²] - E[x]²` cancels most
            // of a large number and the accuracy is spent for a pass that costs
            // shared-memory reads the TMA has already paid for.
            let mut square = Rows::splat(0.0);
            chunk = 0;
            while chunk < CHUNKS {
                let x: Chunk = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                let x = x.sub_row(mean);
                square.add_assign(x.mul(x).row_sum());
                chunk += 1;
            }
            let deviation = square.scale(1.0 / COLUMNS as f32).shift(epsilon).rsqrt();

            // The parameters are read a chunk at a time and out of the same
            // shared vectors the TMA filled — every lane reads the columns its
            // fragment owns, and the 8 lanes of a column group ask for the same
            // address and shared memory answers them together, which is the
            // access pattern a swizzle would have been spreading for nothing.
            // Reading them here rather than before the reductions is what keeps
            // two `ColVec`s off the live set across two full passes.
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
    /// # The residency is declared, because it used to be luck
    ///
    /// `#[launch_bounds(THREADS, 3)]` is `__launch_bounds__`: it puts
    /// `.maxntid 128, 1, 1` and `.minnctapersm 3` on the entry, so **`ptxas` is
    /// told the residency this kernel is written for** and caps registers to
    /// reach it. Three CTAs an SM at 128 threads is 168 registers a thread,
    /// which is exactly the step — `_register_ceiling(3, 128)` in
    /// `modal_app.py` derives the same 168 — and shared memory permits six at
    /// [`SHARED_BYTES`], so **registers are this kernel's binding term** and
    /// nothing else was going to hold the line.
    ///
    /// It sat on 168 without saying so until this file was compiled into a
    /// smaller crate, at which point the same source came out at 236 and the
    /// step was crossed with nothing in the tree to notice. A count that lands
    /// on its ceiling by luck is not a residency, and #95's gate can only watch
    /// a kernel that states one — this is that statement, in the one place the
    /// compiler reads it too, so the gate and `ptxas` cannot be told different
    /// numbers.
    ///
    /// # Safety
    ///
    /// As [`layernorm_rows`], minus the parameter vectors. The block is 1-D and
    /// exactly [`THREADS`] threads, which is what makes each warp's slot in
    /// `partials` its own — and what `#[launch_bounds]` above now also declares.
    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn groupnorm_tile(
        source: *const TmaDescriptor,
        destination: *const TmaDescriptor,
        epsilon: f32,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = Tile::from_raw(smem);
            // The partials first, because `Tile::BYTES` is the only offset in
            // this plan a vector's 128-byte alignment is promised at; the
            // barrier behind it needs eight.
            let partials = Partials::from_raw(smem.add(Tile::BYTES));
            let loaded = Semaphore::attach(smem.add(Tile::BYTES + Partials::BYTES) as *mut Barrier);

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
                loaded.expect_tx(tile.tma_load(source, row, 0, loaded));
            }
            loaded.wait(0);
            thread::sync_threads();

            let x: Band = load_tile(tile.chunk_writer(), row_base, 0, lane);

            // `tile_sum` (#6) folds this warp's band to one f32, warp-uniform,
            // and `block_reduce_sum` (#3) folds the four warps' to one that is
            // block-uniform — the shuffle butterfly stops at 32 lanes, so the
            // second step is a staged write and two barriers rather than more
            // shuffles.
            let mean = block_reduce_sum(partials, x.tile_sum()) * scale;
            // `shift`/`scale` against a block-uniform `f32` (#38), and the free
            // `rsqrt` beside them: this statistic never becomes a vector, so
            // there was nothing for `RegVec::rsqrt` to ride on.
            let x = x.shift(-mean);
            // The same scratch again, immediately, with no barrier here: the
            // reduction syncs after its own read, so this warp cannot overwrite
            // a slot another warp has not folded yet.
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

/// Rows the correctness run covers — two CTAs' worth, so `blockIdx.x`'s row
/// stride is on trial and not just assumed.
pub const CHECK_ROWS: usize = 2 * ROWS;

/// The `epsilon` every launcher passes.
///
/// Stated as a residual rather than sold as coverage: no row this seed
/// produces has a variance under 835, so `epsilon` is 1.2e-8 of the smallest
/// term it is added to and a kernel that dropped it entirely would pass. What
/// it guards is a degenerate row, and a check whose whole job is to police the
/// *arrangement* of a non-degenerate one cannot also be the test for that.
pub const EPSILON: f32 = 1e-5;

/// Column strides the row axis cycles through, one per CTA band, and the
/// reason 63 rather than any other number is [`crate::softmax::permutation`]'s
/// — a seed of the form `(a·row + b·column + c) % 128` is an affine map of the
/// column axis, those form a 2-group of exponent 128, and so every row-affine
/// seed repeats after exactly one `ROWS` band. An odd cycle is the only thing
/// that reaches past one, and 63 is the longest available.
const STRIDES: usize = 63;

/// Boost classes a row cycles through — the one thing a softmax seed did not
/// have to supply and a layernorm seed does. See [`value`].
const BOOSTS: usize = 3;

/// Which of the 128 ladder entries sits at `(row, column)`.
fn permutation(row: usize, column: usize) -> usize {
    let stride = 11 + 2 * (row / ROWS % STRIDES);
    (37 * row + stride * column) % COLUMNS
}

/// Ladder entry `p`, as a row of class `row % BOOSTS` spells it.
///
/// ## Why a permutation alone is not a layernorm seed
///
/// [`crate::softmax`] gives every row the same 128 values in a different
/// order, and for a softmax that is enough. For a layernorm it is a hole:
/// identical multisets mean identical row statistics, so the *whole-tile*
/// statistic [`kernels::groupnorm_tile`] takes is numerically the row
/// statistic, and a kernel that reduced over the wrong axis would compute the
/// right answer. So would one that never subtracted a mean, if the mean were
/// zero. Neither is a hypothetical: they are the two kernels in this file.
///
/// The row therefore has to change the *shape* of its distribution and not
/// merely the order — and it cannot do it by scaling or shifting, because
/// layernorm is invariant to both. `x → a·x + c` is exactly the transform the
/// kernel divides out, so a seed of that form makes every row's output
/// bit-identical and the check blind to every row error there is.
///
/// What varies here is one scale factor applied to *half* the ladder. `p`
/// splits into a magnitude `16 + (p % 32)/2` and one of four scalings chosen
/// by `p / 32`; the third and fourth carry a row-dependent `boost` of 2, 4 or
/// 8. That is not an affine image of a fixed vector, so mean and variance both
/// move with the class — 8.9/28.9, 14.8/53.8, 26.7/105.7 — while the ladder
/// stays 128 *distinct* values at every class, which is what keeps the
/// permutation's row argument intact on top of it.
///
/// Every value is a magnitude of at most six significant bits times a power of
/// two, so all 384 of them are exact in bf16 and nothing is lost on the way
/// in. The sign asymmetry (`1, -1/2, boost, -boost/2`) is what makes the mean
/// non-zero; the ladder's own range ratio of 1.97 being under the smallest
/// boost is what keeps the four scalings from colliding.
fn value(p: usize, row: usize) -> f32 {
    let magnitude = 16.0 + (p % 32) as f32 / 2.0;
    let boost = (1 << (1 + row % BOOSTS)) as f32;
    magnitude * [1.0, -0.5, boost, -boost / 2.0][p / 32]
}

/// The input at `(row, column)`.
fn input(row: usize, column: usize) -> f32 {
    value(permutation(row, column), row)
}

/// The two parameter vectors, injective over `0..COLUMNS` and exact in bf16 —
/// injective because a `gamma` or `beta` read from the wrong column is one of
/// the errors this check exists to see, and a periodic parameter would hide
/// exactly the misindexing a chunked column walk can commit.
fn gamma(column: usize) -> f32 {
    1.0 + column as f32 / 128.0
}

fn beta(column: usize) -> f32 {
    2.0 + column as f32 / 64.0
}

/// The mean and `1/√(variance + ε)` of any row of boost class `class`.
///
/// Three of them cover every size the harness runs: [`permutation`] is a
/// bijection in `column`, so a row holds each ladder entry exactly once and
/// its statistics depend on the row only through its class.
fn statistics(class: usize) -> (f64, f64) {
    let values: Vec<f64> = (0..COLUMNS).map(|p| value(p, class) as f64).collect();
    let mean = values.iter().sum::<f64>() / COLUMNS as f64;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / COLUMNS as f64;
    (mean, 1.0 / (variance + EPSILON as f64).sqrt())
}

/// Round-to-nearest-even fp32 → bf16. Exact for every value this file stages,
/// whose low 16 mantissa bits are already zero — each example carries its own
/// copy rather than reaching into another's, which is what keeps a file here
/// readable end to end.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// A run of `count` elements as the packed `u32` words a bf16 buffer holds.
fn packed(count: usize, at: impl Fn(usize) -> f32) -> Vec<u32> {
    (0..count / 2)
        .map(|pair| to_bf16(at(2 * pair)) as u32 | ((to_bf16(at(2 * pair + 1)) as u32) << 16))
        .collect()
}

/// Blocks a `[rows, COLUMNS]` problem takes: one CTA per `ROWS`.
pub fn grid(rows: usize) -> u32 {
    (rows / ROWS) as u32
}

/// The `then` a run that is only being checked passes.
fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch the kernel over `rows`, compare every output against a CPU
/// reference, and only then hand the launch to `then`.
///
/// That order is the whole design of [`crate::bench`]: the timing hook cannot
/// be reached from a launch whose output was wrong.
///
/// ## The tolerance, and what it has to separate
///
/// 2⁻⁷ relative, and the reason it is not [`crate::softmax`]'s 2⁻⁸ is worth
/// stating because the number 2⁻⁹ that file quotes is **not a constant**. A
/// bf16 carries eight significand bits, so an ulp at `m·2ᵉ` is `2ᵉ⁻⁷` and half
/// an ulp *relative* is `2⁻⁸/m` — 2⁻⁹ only when `m` is near 2, and a full 2⁻⁸
/// when it is near 1. Softmax's outputs happen to sit at the forgiving end;
/// this seed's do not. Its smallest output is 0.552, whose significand is
/// 1.104, and the correctness run measures a worst relative error of
/// **3.87e-3 = 0.99 × 2⁻⁸**. That is the rounding floor, not an error, and a
/// tolerance of 2⁻⁸ would have been sitting on it.
///
/// So 2⁻⁷ is one doubling above the floor, which is the same factor of two of
/// headroom softmax has, arrived at honestly rather than inherited. Nothing
/// else is competing for it: unlike softmax there is no `exp2` polynomial,
/// because [`kittens::reg::rsqrt`] is a correctly-rounded `sqrt.rn.f32` and a
/// divide rather than the SFU approximation. What is left over is the order
/// the shuffle butterfly sums a row in, and the seed keeps every value at
/// least 0.22 away from its row's mean, so no output is built out of a
/// cancellation the host reference does not also perform.
///
/// The errors it is against, scored exhaustively over the
/// `CHECK_ROWS · COLUMNS` = 32,768 cells of the check by the same functions
/// the reference uses. "p10" is the tenth-percentile relative error among the
/// cells a given error makes wrong, so nine tenths of them are worse:
///
/// | error | cells wrong | p10 | median |
/// | --- | --- | --- | --- |
/// | row displaced by 1 | 32,765 | 0.277 | 0.810 |
/// | row displaced by 32 (a warp band) | 32,738 | 0.256 | 0.776 |
/// | row displaced by 128 (a CTA band) | 32,620 | 0.118 | 0.501 |
/// | every CTA reads band 0 | 16,310 | 0.118 | 0.501 |
/// | third pass pinned to column chunk 0 | 28,672 | 0.155 | 0.560 |
/// | `gamma`/`beta` pinned to column chunk 0 | 28,672 | 0.108 | 0.321 |
/// | `gamma` and `beta` swapped | 32,512 | 0.099 | 0.797 |
/// | statistic over the whole tile | 32,768 | 0.030 | 0.186 |
/// | mean never subtracted | 32,768 | 0.078 | 0.144 |
/// | variance as the raw second moment | 23,662 | 0.011 | 0.019 |
///
/// The worst row displacement over `1..=129` leaves 31,684 of 32,768 wrong at
/// a p10 of 0.071, which is 9× the tolerance. The last row is the weakest and
/// is named rather than rounded off: forgetting to centre the *variance* — but
/// not the output — still fails 72% of the cells, at a p10 of 1.4× the
/// tolerance.
///
/// Two are worth their own line because they are the two the seed was built
/// for, and a permuted-multiset seed of [`crate::softmax`]'s shape scores zero
/// on both: **statistic over the whole tile** is this file's other kernel, and
/// **mean never subtracted** is the pass that a chunked walk has to carry
/// across chunks. Each fails every one of the 32,768 cells.
///
/// ## Two of them were run, not just scored
///
/// A table computed by the same functions as the reference is an argument, not
/// a control — it cannot fail in a way the reference does not also fail. So
/// two rows were built as kernels and launched:
///
/// - **Third pass pinned to column chunk 0**, the error this pull request's own
///   change is the one that can commit. Predicted 28,672; the device failed
///   **28,672**, to the cell.
/// - **Variance as the raw second moment**, chosen because it is the *weakest*
///   row in the table — if the check sees this one it sees all of them.
///   Predicted 23,662; the device failed **24,410**. The 3% is the device
///   rounding through fp32 and bf16 where the host model is exact `f64`, which
///   moves cells sitting on the tolerance across it.
///
/// The residual is the one [`crate::softmax::permutation`] states: a constant
/// row displacement is invisible exactly when it is a multiple of
/// `STRIDES · ROWS` = 8064 rows, and no power-of-two grid arithmetic produces
/// one.
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

    /// One bf16 ulp. Half an ulp is 2⁻⁸ where this seed's outputs sit, and the
    /// run measures 0.99 of that — see the header above.
    const TOLERANCE: f32 = 1.0 / 128.0;

    // One CTA owns a whole `ROWS` band and the kernel bounds-checks nothing.
    if !rows.is_multiple_of(ROWS) {
        return Err(format!("{rows} rows does not divide the {ROWS} a CTA owns").into());
    }

    let stream = context.default_stream();
    let module = kernels::load(context)?;

    let staged = packed(rows * COLUMNS, |flat| input(flat / COLUMNS, flat % COLUMNS));
    let source = DeviceBuffer::from_host(&stream, &staged)?;
    // Zeroed rather than left alone: a store that never lands reads back as
    // bf16 zero, and no output of this seed is within a tolerance of zero —
    // the smallest is 0.55.
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
            // Negated rather than `error > TOLERANCE`, and that is the point: a
            // NaN compares false either way, so only this spelling counts one
            // as wrong.
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

/// The correctness run: one size, checked, nothing timed.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, CHECK_ROWS, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point: the same check at `shape`, and then the same
/// launch timed. A layernorm [`Shape`] is `rows x COLUMNS`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, time).map(|(_, timings)| timings)
}
