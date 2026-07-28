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
//! gone: the body below is calls and arithmetic.
//!
//! Note what is *not* in that list, and never was: nothing about layouts,
//! swizzles, semaphores, the fragment map, or the arithmetic. The missing
//! surface was shallow and wide, not deep, and this is the example that said so
//! first.
//!
//! ## Why it walks the row in chunks (#47, #76)
//!
//! The body used to hold a whole `[32, 128]` band — 128 fp32 a thread — and
//! say the algorithm in three lines over it. It computed the right answer at
//! 367 GB/s, and the reason was not the memory protocol #47 blamed: `ptxas`
//! never put that band in registers, and every pass over it was a round trip
//! to the same hierarchy the TMA was already saturating. [`CHUNK`] is the fix
//! and it is one number — the row is walked a chunk at a time and the
//! reductions carry across chunks — which took the kernel to 952 GB/s.
//!
//! That is where #47 stopped, and it stopped one rung short and for the wrong
//! reason. #76 built this kernel at **every** `[32, N]` rung rather than
//! taking the ladder's answer, exactly as #67 had to for `layernorm_rows`.
//! The row extent is not free — a warp owns 32 rows of a 128-row tile and
//! changing that is a launch-geometry change — so the column sweep is the only
//! ladder that applies, and it is the shipped arithmetic at each rung:
//!
//! | `CHUNK` | regs | spill | frame | blocks/SM | GB/s at 8192 blocks |
//! | --- | --- | --- | --- | --- | --- |
//! | 128 | 197 | 0 | 1024 | 6 | 278 |
//! | 64 | 40 | 0 | 512 | 6 | 554 |
//! | 32 — #47's answer | 39 | 0 | 256 | 6 | 1004 |
//! | **16 — shipped** | **32** | 0 | **0** | 6 | **4414** |
//!
//! `[32, 32]` has a **zero** frame in all five spellings of #60's probe and
//! this kernel gets 256 bytes there, for #67's reason: the probe carries no
//! statistics across chunks and a real kernel's extra live state moves its
//! cliff a rung down. **The ladder narrows the search; it does not answer.**
//!
//! ## The rung was a quarter of it, and `exp2` was the rest (#76)
//!
//! After #67, `layernorm_rows` ran 5483 GB/s where this kernel ran 950 — a
//! factor of **5.8** at equal bytes, equal blocks and the same 6 CTAs an SM,
//! against a kernel doing *more* arithmetic per element. So the ordering was
//! backwards, and the rung is not what explains it: at the shipped `exp2` the
//! rung is worth only 950 → 1178.
//!
//! The rest was found by ablation rather than by argument — the same kernel,
//! the same geometry, one thing removed at a time, all at `CHUNK = 16` and all
//! at 8192 blocks. The two bottom rows do not compute a softmax and exist only
//! to be timed:
//!
//! | at `CHUNK = 16` | GB/s | what it removes |
//! | --- | --- | --- |
//! | `exp2` polynomial, `div_row` | 1178 | — (the shipped arithmetic) |
//! | `exp2_hw`, `div_row` | 3153 | the FMA polynomial |
//! | **`exp2_hw`, `recip` + `mul_row`** | **4414** | 124 of every 128 divides |
//! | no transcendental, no divide | 6042 | *both* `exp2` calls |
//! | one pass, `load_tile` → `store_tile` | 6324 | the two extra passes |
//!
//! Read downwards that is the whole 5.8×, and it is three things and not one:
//!
//! - **The FMA `exp2` polynomial is the largest single term, at 2.7×.**
//!   [`kittens::reg::exp2_approx`] is a clamp, a shift-trick split and a
//!   degree-3 minimax polynomial — around a dozen instructions, evaluated
//!   twice per element, 128 elements a thread. `exp2_hw` is one
//!   `ex2.approx.f32`. This is **not** an accuracy trade in the direction it
//!   looks: the SFU instruction is good to about 2⁻²², the polynomial to
//!   7.5e-5 = 2⁻¹³·⁷, so the swap is *more* accurate and the check's worst
//!   relative error does not move at all (1.97e-3 either way) because bf16
//!   output rounding dominates both.
//! - **The 128 `div.rn.f32` cost 1.4×, and #47 measured them at nothing.**
//!   That measurement was not wrong; it was taken under a different binding
//!   constraint. With the polynomial in the loop the kernel was so far from
//!   memory that removing 124 divides moved it 952.5 → 952.1 GB/s. With the
//!   SFU in the loop the same change is 3153 → 4414. **A control is only valid
//!   under the bind it was taken in**, and this is the cleanest instance of
//!   that in the repository.
//! - **The three passes over shared memory cost 4.5%**, which is the one thing
//!   #47 proposed to fix and the one thing that was never worth fixing:
//!   6324 → 6042 GB/s for two extra full reads of the tile plus every row
//!   reduction in the kernel.
//!
//! What the two changes come to, at all six benchmark sizes, every one checked
//! before it was timed and `layernorm_rows` untouched beside it at 5483 → 5497
//! GB/s as the control:
//!
//! | rows × 128 × 2 planes | blocks | before, GB/s | after, GB/s |
//! | --- | --- | --- | --- |
//! | 128 | 2 | 5.2 | 12.0 |
//! | 512 | 8 | 20.4 | 48.9 |
//! | 4096 | 64 | 157.9 | 354.2 |
//! | 32768 | 512 | 632.4 | 2004.9 |
//! | 262144 | 4096 | 917.5 | 3880.0 |
//! | 524288 | 8192 | 950.3 | **4389.6** |
//!
//! **4.6×**, on **66 registers and a 256-byte frame → 32 and zero**. The 5.8×
//! is now 1.25×, and that residual is not a defect: it is the two
//! `ex2.approx.f32` this kernel owes its own definition and a layernorm does
//! not, priced at exactly that by the fourth row of the ablation.
//!
//! Two things that are *not* in that list, and were checked rather than
//! assumed:
//!
//! - **Occupancy never moved.** `cuOccupancyMaxActiveBlocksPerMultiprocessor`
//!   says 6 blocks and 24 warps an SM at every rung but 128 and for every
//!   ablation above — shared memory binds this kernel and 32 registers a
//!   thread is nowhere near what would.
//! - **Nothing was put in flight, and nothing needed to be.** There is still
//!   one TMA load and one TMA store per CTA and both still block. The kernel
//!   now runs at 70% of the rate a pure `load_tile` → `store_tile` over the
//!   same tiles achieves, so the memory protocol is not what is left.
//!
//! ### The floor was never the launch
//!
//! #47 read this kernel's 23 µs at two blocks as "a fixed cost every launch on
//! this harness pays". It is not, and the ablation says so directly — one
//! session, one two-block launch, the arithmetic put back a piece at a time:
//! **7.0 µs** for the copy, 8.9 µs with both extra passes and every reduction,
//! **11.6 µs** as shipped, and 23.7 µs at the old rung and the old `exp2`. So
//! twelve of those twenty-three microseconds were this kernel's own arithmetic
//! standing at the head of a dependent chain with two blocks on a 148-SM
//! device, and none of them were the launch. `layernorm_rows` reaching 8.4 µs
//! (#67) was the first evidence the reading was wrong; this is the mechanism.
//! The checked benchmark, a different session, puts the shipped floor at
//! 10.9 µs against the 25.4 µs it measured before this change.
//!
//! ### What was measured and deliberately not shipped
//!
//! - **Folding the normalizer into the exponent.** `exp2(x - peak)/total` is
//!   `exp2(x - peak - log2(total))`, which is four `log2` a thread and no
//!   divide at all. It measured 4390, 4415 and 4462 GB/s across three sweep
//!   rounds against `recip`'s 4363 and 4414 — the two forms are inside each
//!   other's run-to-run spread. It is not here because a 1% that a rerun
//!   erases does not buy putting [`kittens::reg::log2_approx`]'s error inside
//!   the exponent, where the tolerance argument would then have to carry it.
//! - **Parking the numerator in the tile.** The third pass recomputes `exp2`
//!   rather than keeping the second pass's result; storing it and only scaling
//!   in the third pass is one `exp2` per element instead of two and measured
//!   **4670 GB/s**, the fastest correct form here. It is not shipped, and the
//!   reason is now a number rather than a worry: the tile is bf16, so the
//!   numerator rounds on the way in and the quotient rounds again, and the
//!   check's worst relative error goes **1.97e-3 → 3.44e-3 against a 3.91e-3
//!   tolerance**. 5.8% more throughput for a check with 12% of its headroom
//!   left is the wrong trade, and a tolerance sitting on its own rounding
//!   floor is not a check (#75).
//!
//! ### One claim upstream that this contradicts
//!
//! `reg.rs`'s module header says of the two `exp2`s that "the measurement does
//! not favour" the SFU, on the evidence that pointing `exp2` at it takes
//! `softmax_probe_128` from 168 registers to 255 with 112 bytes of spill. That
//! is true, and it is a statement about *registers*. At this kernel's shape it
//! is 32 registers and a zero frame either way, and the SFU is **2.7× faster**.
//! That is the fourth time in this repository the register column has ordered
//! time backwards (#47, #63, #67, now this), and the note upstream should be
//! read as the register claim it is rather than as advice about speed.
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
//!
//! ## The column walk is under that check, and two kernels say so
//!
//! Halving [`CHUNK`] doubles the number of chunks a row is walked in, so the
//! errors the walk can commit are the ones this check has to see. A table
//! computed by the reference's own functions is an argument and not a control
//! — it cannot fail in a way the reference does not also fail — so both were
//! **built as kernels and launched** over the check's 65,536 cells:
//!
//! | deliberate error | predicted | device |
//! | --- | --- | --- |
//! | third pass reads the first chunk for every chunk | 56,960 | **56,960** |
//! | second pass sums the first chunk twice | 65,536 | **65,536** |
//!
//! The first is the misindexing this change is the one that can commit, and
//! predicting it to the cell took two corrections that are the reason it was
//! worth launching rather than tabulating. The obvious model — "column `c` is
//! written from column `c % CHUNK`" — predicts 57,344 and is wrong by 384,
//! because the third pass reads and writes *the same tile*: chunk 0 is
//! overwritten before chunks 1..7 re-read it, so what they exponentiate is a
//! softmax output and not an input. Modelling that leaves 64, and the last 64
//! are cells the device's bf16 intermediate moves across the tolerance where
//! an `f64` host model keeps them inside it. With both, the model is exact.
//!
//! The second was chosen as the *weakest* error a chunk walk can commit — one
//! too many trips round a loop, changing only the denominator — and it turns
//! out not to be weak: it fails every cell. That is a property of the seed and
//! worth naming, because it is the row argument paying off somewhere it was
//! not designed for. [`permutation`] spreads a row's chunk across the whole
//! ladder at an odd stride rather than giving it 16 adjacent values, so the
//! first chunk always carries a real share of the mass — the smallest shift
//! over all 512 rows of the check is **4.3%, eleven times the tolerance**.
//!
//! The check the seed was strengthened for (#56, #61) therefore sees a
//! misplaced *column* exactly as loudly as a misplaced row.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, thread, warp};

// Host side: the launcher's error type, and the benchmark's size and clock.
use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::ldst::{load_tile, store_tile};
use kittens::reg::{BaseLdtm, RegTile, RegVec};
use kittens::shared::{Bf16, SharedTile, Swizzle128B, tma_store_commit, tma_store_wait};
use kittens::sync::Semaphore;

/// Rows per CTA — four warps of 32.
const ROWS: usize = 128;
/// Columns the softmax runs over. One row of the tile is one distribution.
const COLUMNS: usize = 128;
/// Columns a warp holds in registers at once — #47's fix, at #76's rung.
///
/// The widest `[32, N]` band **this kernel** gets a zero stack frame at,
/// measured at 16, 32, 64 and 128 rather than read off #60's ladder: the
/// ladder says `[32, 32]` is frame-free in all five spellings and this kernel
/// carries 256 bytes there, because the probe holds no statistics across
/// chunks. 32 is one rung too wide and **4.4× slower**; the module header has
/// the table.
const CHUNK: usize = 16;
/// Chunks a row is walked in.
const CHUNKS: usize = COLUMNS / CHUNK;

type Tile = SharedTile<Bf16, ROWS, COLUMNS, Swizzle128B>;
/// One warp's 32 rows by [`CHUNK`] columns of it.
type Band = RegTile<32, CHUNK, BaseLdtm>;
/// A per-row scalar of those 32 rows — the maximum, then the denominator.
type Rows = RegVec<32, BaseLdtm>;

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

            // Three passes over the shared tile, each holding one [`CHUNK`]-wide
            // band at a time. `row_max`/`row_sum` (#6) fold a thread's own
            // values and then the quad holding the rest of the chunk, and a
            // chunk's quad is the same four lanes whatever the chunk, so the
            // per-chunk results combine slotwise with no shuffle of their own.
            let mut peak = Rows::splat(f32::NEG_INFINITY);
            let mut chunk = 0usize;
            while chunk < CHUNKS {
                let x: Band = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                peak.max_assign(x.row_max());
                chunk += 1;
            }

            // `exp2_hw` and not `exp2`: one `ex2.approx.f32` rather than the
            // dozen-instruction FMA polynomial, twice per element over 128
            // elements a thread, and it is 2.7× (#76). It is also the more
            // accurate of the two — the SFU is good to ~2⁻²² and the
            // polynomial to 7.5e-5 — so the check's worst error does not move.
            let mut total = Rows::splat(0.0);
            chunk = 0;
            while chunk < CHUNKS {
                let x: Band = load_tile(chunks, row_base, (CHUNK * chunk) as u32, lane);
                total.add_assign(x.sub_row(peak).exp2_hw().row_sum());
                chunk += 1;
            }
            // Four `div.rn.f32` a thread and 128 multiplies, rather than
            // `div_row`'s 128 divides. #47 measured that swap at nothing —
            // 952.5 against 952.1 GB/s — and it was measuring a kernel the
            // `exp2` polynomial had already made compute-bound. With the SFU
            // in the loop the same swap is 3153 → 4414 GB/s.
            let scale = total.recip();

            // The third pass re-reads the input and recomputes `exp2` rather
            // than keeping the second pass's result. Parking the numerator in
            // the tile instead is one `exp2` per element and 4670 GB/s, and it
            // is not here because the tile is bf16: rounding the numerator and
            // then the quotient takes the check's worst relative error from
            // 1.97e-3 to 3.44e-3 against a 3.91e-3 tolerance. The exponential
            // is arithmetic; the precision is not recoverable.
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
/// The tolerance is relative and it is the honest one. The result is bf16, and
/// half an ulp *relative* is `2⁻⁸/m` for a significand `m` — 2⁻⁹ only when `m`
/// is near 2 and a full 2⁻⁸ when it is near 1 (#75 established that, correcting
/// an earlier reading of this file's own comment as a constant). This seed's
/// outputs sit at the forgiving end of that range, which is why 2⁻⁸ has room
/// here where `layernorm` needed 2⁻⁷.
///
/// Measured rather than argued: the worst relative error over the checked run
/// is **1.97e-3**, a hair past the 1.95e-3 that rounding to bf16 costs on its
/// own, and it is unchanged by #76 swapping the `exp2` polynomial for the SFU
/// — because at 2⁻²² the instruction is nowhere near what dominates. So the
/// tolerance is 2.0× what a correct kernel needs, and 23× below the
/// 2^(1/8) − 1 = 9.05% gap between two neighbouring outputs of a row, which is
/// the band the check has to separate a right answer from a wrong one in.
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
    if !rows.is_multiple_of(ROWS) || planes == 0 {
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
                // Negated rather than `error > TOLERANCE`, and that is the
                // point: a NaN compares false either way, so only this spelling
                // counts one as wrong. A kernel that wrote NaN passing its own
                // check is the failure mode worth spending a lint allow on.
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
