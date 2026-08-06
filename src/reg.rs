//! Register tiles and vectors over a fragment ownership map, plus the scalar
//! ops they compose with.
//!
//! A [`RegTile<M, N, L>`](RegTile) is a *logical* `[M, N]` fp32 tile spread
//! across the 32 lanes of one warp; the layout `L` owns the
//! `(lane, slot, value) -> (row, column)` map, so ops are written against the
//! logical shape. [`BaseLdtm`] is the crate's only layout and its doc carries
//! the map every op here assumes. Elementwise work goes through [`UnaryOp`] /
//! [`BinaryOp`] / [`TernaryOp`] and the `*_map` methods, each of which has an
//! `_assign` twin that rewrites the receiver instead of returning a tile.
//!
//! Everything here is **warp scope** — nothing makes several warps agree — and
//! every mask takes a **coordinate origin**, because a tile is normally a
//! sub-block of a much larger matrix ([`RegTile::make_causal_at`]).
//!
//! `docs/library/reg.md` carries the measurements: by-value against in-place,
//! the two `exp2`s, and the ThunderKittens naming inversion.
//!
//! **Which `exp2` a kernel should call is a measurement that kernel has to
//! take** (#81). The SFU spelling [`exp2_hw`] is one `ex2.approx.f32` against
//! [`exp2_approx`]'s clamp, shift-trick split and degree-3 minimax, and it is
//! worth **2.7× on `softmax`** (#76) and **nothing on `flash_forward`** — 3.9%
//! slower there, at the shape that fills the device, against 0.3% run-to-run.
//! Neither is the general answer: `softmax` is a bandwidth loop whose arithmetic
//! is the exponential, and flash spends its time in the MMA and TMA pipeline at
//! four warps a CTA. Accuracy does not decide it either — both kernels' checks
//! move by less than a bf16 ulp between the two spellings, because what
//! dominates is the round trip and not the transcendental.
//!
//! So the *name* `exp2` is the polynomial, on every register family, and the
//! reason is structural rather than a verdict: `ex2.approx.f32` exists only
//! under device compilation and panics anywhere else, so the polynomial is the
//! only one of the two a host test, a doctest or a CPU reference can evaluate,
//! and [`Exp`] and [`online_rescale`] are on it for that reason too. An earlier
//! version of this header said the measurement did not favour the SFU on the
//! evidence of a *register count* on a probe shape no kernel has; the register
//! column does not separate the two spellings in either shipped kernel, and
//! #76's and #81's clocks are what the sentence above is made of.
//!
//! ```no_run
//! # use kittens::lane;
//! # use kittens::reg::{BaseLdtm, RegTile};
//! # unsafe fn softmax(mut scores: RegTile<32, 64, BaseLdtm>) -> RegTile<32, 64, BaseLdtm> {
//! scores.make_causal_at(lane(), 0, 0, -1.0e30);
//! let probs = scores.sub_row(scores.row_max()).exp2();
//! probs.div_row(probs.row_sum())
//! # }
//! ```

use cuda_device::thread::__unroll_config;
use cuda_device::warp;

/// NaN-free float max, as a comparison-select rather than a libdevice call.
#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// NaN-free float min; see [`fmax`].
#[inline(always)]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `2^x` on FMA units — max relative error 7.5e-5, and what every `exp2` in
/// the crate resolves to, because it is the one of the two that has a value on
/// the host. [`exp2_hw`] is the SFU alternative: 2.7× on `softmax` (#76), 3.9%
/// slower on `flash_forward` (#81), a different rounding, and a panic if a host
/// build evaluates it.
///
/// The input is clamped to ±125, so a masked-score sentinel flushes to a
/// harmless ~2^-125 instead of overflowing: a fully masked row sums to zero
/// rather than to `NaN`.
///
/// ```
/// # use kittens::reg::exp2_approx;
/// assert!((exp2_approx(3.5) / 3.5f32.exp2() - 1.0).abs() < 1.0e-4);
/// assert!(exp2_approx(-1.0e30) <= 2.0f32.powi(-124));
/// ```
#[inline(always)]
pub fn exp2_approx(x: f32) -> f32 {
    const SHIFT: f32 = 12582912.0; // 1.5 * 2^23
    const C0: f32 = 0.999_928_07;
    const C1: f32 = 0.693_260_99;
    const C2: f32 = 0.242_611_12;
    const C3: f32 = 0.055_171_67;
    let x = fmin(fmax(x, -125.0), 125.0);
    let shifted = x + SHIFT;
    let integer = (shifted.to_bits() as i32).wrapping_sub(0x4b40_0000);
    let fraction = x - (shifted - SHIFT);
    let poly = C0 + fraction * (C1 + fraction * (C2 + fraction * C3));
    f32::from_bits((poly.to_bits() as i32).wrapping_add(integer << 23) as u32)
}

/// `2^x` as one `ex2.approx.f32` SFU instruction, and a different rounding —
/// 2.7× the throughput of [`exp2_approx`] on `softmax` (#76) and 3.9% *slower*
/// on `flash_forward` (#81). Worth trying in a kernel whose arithmetic is the
/// exponential; not a swap to make on a ladder someone else measured.
///
/// A spelling a kernel asks for rather than the default, because the intrinsic
/// exists only under device compilation and panics anywhere else. Adopting it
/// is a numerics change and not a refactor, so it wants a check in front of it:
/// both kernels' worst relative errors move by less than a bf16 ulp, since what
/// dominates them is the round trip and not the transcendental.
#[inline(always)]
pub fn exp2_hw(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x)
}

/// `log2(x)` for positive normal `x`: exponent extraction, mantissa
/// renormalized to `[√½, √2]`, then a four-term atanh series in
/// `t = (m-1)/(m+1)`. |error| < 5e-8 on the reduced range. Undefined on zero,
/// negatives and subnormals.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn log2_approx(x: f32) -> f32 {
    const C0: f32 = 2.885_390_1;
    const C1: f32 = 0.961_796_7;
    const C2: f32 = 0.577_078_02;
    const C3: f32 = 0.412_198_58;
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) as i32) - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > core::f32::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let t2 = t * t;
    exponent as f32 + t * (C0 + t2 * (C1 + t2 * (C2 + t2 * C3)))
}

/// `1/√x` as a correctly-rounded `sqrt.rn.f32` and a divide rather than the SFU
/// `rsqrt.approx.f32`: two instructions, but the same number on host and
/// device, which is what lets a normalization kernel's host reference compare
/// with `==`. Free-standing for the scalar case — layernorm's variance is
/// already one `f32` and has no vector to reach [`Rsqrt`] through.
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Fold across the 4 lanes of a quad (`shuffle_xor` masks 1 and 2) — under
/// [`BaseLdtm`] the lanes holding the rest of one row's columns, so this is the
/// second half of a row reduction; the first is folding a thread's own `VALUES`
/// registers.
///
/// All four lanes of the quad must call it, and the result lands in all four:
/// that is what makes a [`RegVec`] a whole-row statistic rather than a partial.
#[inline(always)]
pub fn quad_reduce<Op: ReduceOp>(value: f32) -> f32 {
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 1));
    Op::apply(value, warp::shuffle_xor_f32(value, 2))
}

/// Fold across the 8 lanes sharing a `lane % 4` (masks 4, 8 and 16) — under
/// [`BaseLdtm`] the lanes holding the rest of one column's rows, so this is the
/// second half of a column reduction. All 8 must call it; three shuffles rather
/// than [`quad_reduce`]'s two.
#[inline(always)]
pub fn column_group_reduce<Op: ReduceOp>(value: f32) -> f32 {
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 4));
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 8));
    Op::apply(value, warp::shuffle_xor_f32(value, 16))
}

/// Fold across all 32 lanes — the full butterfly, leaving the result
/// warp-uniform. Every lane of the warp must call it.
#[inline(always)]
pub fn warp_reduce<Op: ReduceOp>(value: f32) -> f32 {
    column_group_reduce::<Op>(quad_reduce::<Op>(value))
}

/// Max across the 4 lanes of a quad; see [`quad_reduce`].
#[inline(always)]
pub fn quad_max(value: f32) -> f32 {
    quad_reduce::<Max>(value)
}

/// Sum across the 4 lanes of a quad; see [`quad_reduce`].
#[inline(always)]
pub fn quad_sum(value: f32) -> f32 {
    quad_reduce::<Add>(value)
}

/// A scalar function named as a *type*, so one definition instantiates for
/// every register family through the `unary_map` methods.
///
/// Implementors are unit structs — a type parameter can carry no state to
/// spill and nothing for the inliner to see through.
///
/// ```
/// # use kittens::reg::{RegVec, BaseLdtm, UnaryOp};
/// pub struct Square;
/// impl UnaryOp for Square {
///     fn apply(x: f32) -> f32 { x * x }
/// }
/// let squared = RegVec::<32, BaseLdtm>::splat(3.0).unary_map::<Square>();
/// assert_eq!(squared.get(0), 9.0);
/// ```
pub trait UnaryOp {
    /// The scalar function.
    fn apply(x: f32) -> f32;
}

/// Two-operand [`UnaryOp`]. Also what the scalar and row/column broadcast maps
/// take, with `b` the constant or the per-row/per-column scalar.
pub trait BinaryOp {
    /// The scalar function.
    fn apply(a: f32, b: f32) -> f32;
}

/// Three-operand [`UnaryOp`] — the fused multiply-add family. Rust emits no
/// fast-math flags, so a separate multiply and add never contract on their own.
pub trait TernaryOp {
    /// The scalar function.
    fn apply(a: f32, b: f32, c: f32) -> f32;
}

/// A [`BinaryOp`] a reduction may fold with.
///
/// An implementor owes associativity and commutativity — the fragment map hands
/// a fold its operands in the layout's order, not the tile's — and an identity
/// every fold can seed from. `Sub` and `Div` are deliberately not members.
pub trait ReduceOp: BinaryOp {
    /// The value with `apply(IDENTITY, x) == x` for every `x` in the fold.
    const IDENTITY: f32;
}

macro_rules! scalar_ops {
    ($trait:ident: $($(#[$meta:meta])* $name:ident($($arg:ident),+) = $body:expr;)*) => {$(
        $(#[$meta])*
        pub struct $name;

        impl $trait for $name {
            #[inline(always)]
            fn apply($($arg: f32),+) -> f32 {
                $body
            }
        }
    )*};
}

scalar_ops! { UnaryOp:
    /// The FMA polynomial ([`exp2_approx`]) — what `exp2` means everywhere,
    /// and the only spelling a host build can evaluate.
    Exp2Approx(x) = exp2_approx(x);
    /// The SFU instruction ([`exp2_hw`]) — a different rounding, and a
    /// per-kernel measurement rather than a faster twin (#81).
    Exp2Hw(x) = exp2_hw(x);
    /// `e^x` on [`Exp2Approx`], inheriting its error bound and its ±125
    /// exponent clamp — so this saturates at `x = ±86.6`, just inside where
    /// fp32 overflows anyway.
    Exp(x) = exp2_approx(x * core::f32::consts::LOG2_E);
    Log2(x) = log2_approx(x);
    /// `ln(x)` on [`log2_approx`]; see [`Exp`].
    Log(x) = log2_approx(x) * core::f32::consts::LN_2;
    /// Sign-bit clear — one `abs.f32`, no libdevice `fabsf`.
    Abs(x) = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    Neg(x) = -x;
    Relu(x) = fmax(x, 0.0);
    /// `llvm.sqrt.f32`, which NVPTX lowers to the native `sqrt.rn.f32`.
    Sqrt(x) = x.sqrt();
    /// [`rsqrt`] — correctly rounded, not the SFU `rsqrt.approx.f32`.
    Rsqrt(x) = rsqrt(x);
    Recip(x) = 1.0 / x;
}

scalar_ops! { BinaryOp:
    Add(a, b) = a + b;
    Sub(a, b) = a - b;
    Mul(a, b) = a * b;
    Div(a, b) = a / b;
    Max(a, b) = fmax(a, b);
    Min(a, b) = fmin(a, b);
}

/// The identity of each foldable op; see [`ReduceOp`].
macro_rules! reduce_ops {
    ($($op:ty = $identity:expr;)*) => {$(
        impl ReduceOp for $op {
            const IDENTITY: f32 = $identity;
        }
    )*};
}

reduce_ops! {
    Add = 0.0;
    Mul = 1.0;
    Max = f32::NEG_INFINITY;
    Min = f32::INFINITY;
}

scalar_ops! { TernaryOp:
    /// `a*b + c` in one `fma.rn.f32`; the fusion has to be asked for, since
    /// Rust emits no fast-math flags.
    Fma(a, b, c) = a.mul_add(b, c);
}

/// The row half of a fragment ownership map: how a warp's 32 lanes divide the
/// `M` logical rows of a tile into per-thread *slots*, and where the one
/// `f32` per owned row lives. Separate from the column half so a [`RegVec`] can
/// name a row count without inventing a column count.
pub trait RowLayout<const M: usize> {
    /// Per-thread storage, one `T` per owned row (`[T; SLOTS]`). Generic in the
    /// element because a tile's storage is this array of a
    /// [`ColLayout::Values`].
    type Slots<T: Copy>: Copy;

    /// Rows of the `[M, _]` tile one thread owns.
    const SLOTS: usize;

    /// The logical row in `0..M` that `lane` holds in `slot`.
    fn row_of(lane: u32, slot: usize) -> u32;

    /// Whether `lane` is the single writer of the rows it holds.
    ///
    /// A row statistic is *replicated*: [`Self::row_of`] is not injective across
    /// the warp, because a row's `f32` lives in every lane that holds any of
    /// that row's columns. Reading one back is a broadcast and needs no owner,
    /// but writing one out does — every replica storing would be the same value
    /// to the same address from several threads, which is idempotent and still
    /// not a single writer.
    ///
    /// This is the row axis' answer to what
    /// [`ColLayout::CONTIGUOUS_VALUES`] is on the column axis: a claim about the
    /// map that only the map can make. [`crate::global::store_row_vec`] acts on
    /// it, and an impl that returned `true` everywhere would turn one store into
    /// `M`-many.
    fn owns_row(lane: u32) -> bool;

    /// Every slot set to `value`.
    fn splat_slots<T: Copy>(value: T) -> Self::Slots<T>;

    /// The entry in `slot`. By reference so a tile's row — a whole
    /// [`ColLayout::Values`] — is reached without copying it.
    fn get_slot<T: Copy>(slots: &Self::Slots<T>, slot: usize) -> &T;

    /// The entry in `slot`, to write through; see [`Self::get_slot`].
    fn get_slot_mut<T: Copy>(slots: &mut Self::Slots<T>, slot: usize) -> &mut T;
}

/// The column half of a fragment ownership map, mirroring [`RowLayout`]: the
/// `N` logical columns a thread holds per row, and where one `f32` per owned
/// column lives.
///
/// Under [`BaseLdtm`] a column depends only on `lane % 4`, so the 8 lanes of a
/// column group each hold their own copy of the same `N/4` columns. A
/// [`ColVec`] is that per-lane copy, and the copies agree only once something
/// has folded across the group.
pub trait ColLayout<const N: usize> {
    /// Per-thread storage, one `f32` per owned column (`[f32; VALUES]`).
    type Values: Copy;

    /// Values one thread owns per slot — the columns of `0..N` it holds.
    const VALUES: usize;

    /// How many consecutive values land on consecutive columns, so that a run
    /// of them is one vector memory access rather than that many scalar ones.
    ///
    /// **An impl raising this owes both halves of what a vector access needs:**
    /// that the addresses are adjacent, and that the first is aligned for the
    /// width. A run starts at every multiple of the constant.
    ///
    /// ```text
    /// run % CONTIGUOUS_VALUES == 0  ⟹  col_of(lane, run + i) == col_of(lane, run) + i
    ///                                   for every i < CONTIGUOUS_VALUES,
    ///                                   and CONTIGUOUS_VALUES divides both
    ///                                   col_of(lane, run) and VALUES.
    /// ```
    ///
    /// [`crate::global::store_rows`] and [`crate::global::load_rows`] act on
    /// the claim. The default is `1` — the answer true of every map — so a new
    /// layout gets scalar accesses until it says otherwise.
    const CONTIGUOUS_VALUES: usize = 1;

    /// The logical column in `0..N` that `lane` holds in `value`.
    fn col_of(lane: u32, value: usize) -> u32;

    /// Every value set to `value`.
    fn splat_values(value: f32) -> Self::Values;

    /// The value in `value`.
    fn get_value(values: &Self::Values, value: usize) -> f32;

    /// Write `x` into `value`.
    fn set_value(values: &mut Self::Values, value: usize, x: f32);
}

/// A fragment ownership map for a logical `[M, N]` fp32 tile: which `(row,
/// column)` each of a lane's `SLOTS * VALUES` registers holds, plus the
/// storage they live in. The two coordinate halves come from [`RowLayout`] and
/// [`ColLayout`]; what a tile adds is the joint storage.
///
/// **Never implemented directly.** The blanket impl below covers every
/// [`RowLayout`] × [`ColLayout`] pair, in this crate and downstream.
pub trait FragmentLayout<const M: usize, const N: usize>: RowLayout<M> + ColLayout<N> {
    /// Per-thread storage, `VALUES` values for each of `SLOTS` rows.
    type Storage: Copy;

    /// Every value of every slot set to `value`.
    fn splat(value: f32) -> Self::Storage;

    /// The value at `(slot, value)`.
    fn get(values: &Self::Storage, slot: usize, value: usize) -> f32;

    /// Write `x` at `(slot, value)`.
    fn set(values: &mut Self::Storage, slot: usize, value: usize, x: f32);
}

/// Every [`RowLayout`] × [`ColLayout`] pair *is* a tile layout: a thread's rows
/// of values are its row array of its column array, which needs no arithmetic
/// on `M` and `N` to name. So a shape costs no line anywhere — adding a row
/// extent and a column extent adds every tile between them.
impl<const M: usize, const N: usize, L: RowLayout<M> + ColLayout<N>> FragmentLayout<M, N> for L {
    type Storage = L::Slots<L::Values>;

    #[inline(always)]
    fn splat(value: f32) -> Self::Storage {
        L::splat_slots(L::splat_values(value))
    }

    #[inline(always)]
    fn get(values: &Self::Storage, slot: usize, value: usize) -> f32 {
        L::get_value(L::get_slot(values, slot), value)
    }

    #[inline(always)]
    fn set(values: &mut Self::Storage, slot: usize, value: usize, x: f32) {
        L::set_value(L::get_slot_mut(values, slot), value, x);
    }
}

/// The base-LDTM `16x256b` ownership map — the tcgen05 drain shape the
/// validated kernels use, and the crate's only layout.
///
/// Within each 16-row block of its warp's 32 TMEM rows a thread owns rows
/// `lane/4` and `lane/4 + 8`, and columns `2*(lane%4)` and `+1` of each
/// 8-column half. So per 16 logical rows a thread holds two slots —
/// `2*block + {0, 1}`, the second being the `+8` row — and per 16 logical
/// columns four values, at offsets `{0, 1, 8, 9}` from its column pair base:
///
/// ```text
/// SLOTS  = M / 8    row_of(lane, slot)  = 16*(slot/2) + 8*(slot%2) + lane/4
/// VALUES = N / 4    col_of(lane, value) = 2*(lane%4) + 16*(value/4) + [0,1,8,9][value%4]
/// ```
///
/// **`M` and `N` are per warp**, both multiples of 16 and at most 512, and the
/// warp's 32 lanes cover each `(row, column)` exactly once. Slots count the
/// rows a *thread* owns, not a warpgroup's rows: the flash output accumulator
/// is `RegTile<32, 128, BaseLdtm>` — one warp's 32 TMEM rows by 128 columns,
/// four slots of 32 values in each thread.
///
/// ```
/// # use kittens::reg::{BaseLdtm, RegTile};
/// let (rows, values) = (
///     RegTile::<32, 128, BaseLdtm>::SLOTS,
///     RegTile::<32, 128, BaseLdtm>::VALUES,
/// );
/// assert_eq!((rows, values), (4, 32));
/// assert_eq!(RegTile::<32, 128, BaseLdtm>::coordinate(5, 1, 2), (9, 10));
/// ```
pub struct BaseLdtm;

impl BaseLdtm {
    /// Rows of a tile one warp holds — the band origin every drain in the crate
    /// is a multiple of.
    ///
    /// The layout gives a warp 32 consecutive rows, so `WARP_ROWS * warp_id()`
    /// is where its band starts in a shared tile, in a TMEM accumulator
    /// ([`crate::tmem::TmemTile::tile`]) and in the global rows a drain writes.
    /// Every kernel in the repo spells that product out; this is the `32` in it.
    pub const WARP_ROWS: usize = 32;

    /// The widest band of [`Self::WARP_ROWS`] rows a warp can hold in registers
    /// at once — why a 256-column accumulator drains in two passes and a
    /// 128-column one in the single pass it always did.
    ///
    /// A `RegTile<WARP_ROWS, N, BaseLdtm>` is `WARP_ROWS * N / 32` = `N` fp32
    /// values a thread, so 256 columns at once would want 256 registers before
    /// any of the kernel's own live state — past the 255 the architecture has,
    /// **at any occupancy**. It is an architectural bound and not a budget: no
    /// launch bound, residency or register headroom moves it.
    pub const WIDEST_BAND: usize = 128;

    /// The logical row `lane` holds in `slot` — the `lane/4` row of the
    /// slot's 16-row block, or its `+8` twin for odd slots.
    #[inline(always)]
    pub const fn row(lane: u32, slot: usize) -> u32 {
        16 * (slot as u32 / 2) + 8 * (slot as u32 % 2) + lane / 4
    }

    /// The logical column `lane` holds in `value`: fours at offsets
    /// `{0, 1, 8, 9}` of successive 16-column blocks, from the lane's own
    /// column pair — the inverse of the packing
    /// [`crate::ldst::store_fragment`] undoes.
    #[inline(always)]
    pub const fn column(lane: u32, value: usize) -> u32 {
        2 * (lane % 4) + 16 * (value as u32 / 4) + [0, 1, 8, 9][value % 4]
    }
}

/// One [`RowLayout`] impl per logical row count; `SLOTS` is `M / 8` because
/// each 16-row block gives a thread two rows.
macro_rules! base_ldtm_rows {
    ($($m:literal),* $(,)?) => {$(
        const _: () = assert!($m % 16 == 0, "BaseLdtm rows come in 16-row blocks");

        impl RowLayout<$m> for BaseLdtm {
            type Slots<T: Copy> = [T; $m / 8];
            const SLOTS: usize = $m / 8;

            #[inline(always)]
            fn row_of(lane: u32, slot: usize) -> u32 {
                Self::row(lane, slot)
            }

            // A row depends on `lane / 4`, so the four lanes of a quad share
            // it and the first of them writes for all four.
            #[inline(always)]
            fn owns_row(lane: u32) -> bool {
                lane % 4 == 0
            }

            #[inline(always)]
            fn splat_slots<T: Copy>(value: T) -> Self::Slots<T> {
                [value; $m / 8]
            }

            #[inline(always)]
            fn get_slot<T: Copy>(slots: &Self::Slots<T>, slot: usize) -> &T {
                &slots[slot]
            }

            #[inline(always)]
            fn get_slot_mut<T: Copy>(slots: &mut Self::Slots<T>, slot: usize) -> &mut T {
                &mut slots[slot]
            }
        }
    )*};
}

/// One [`ColLayout`] impl per logical column count; `VALUES` is `N / 4`
/// because each 16-column block gives a thread four values.
macro_rules! base_ldtm_cols {
    ($($n:literal),* $(,)?) => {$(
        const _: () = assert!($n % 16 == 0, "BaseLdtm columns come in 16-column blocks");

        impl ColLayout<$n> for BaseLdtm {
            type Values = [f32; $n / 4];
            const VALUES: usize = $n / 4;
            // `{0, 1, 8, 9}` is two adjacent pairs from an even base. Two and
            // not four: the run breaks at the `1 -> 8` step.
            const CONTIGUOUS_VALUES: usize = 2;

            #[inline(always)]
            fn col_of(lane: u32, value: usize) -> u32 {
                Self::column(lane, value)
            }

            #[inline(always)]
            fn splat_values(value: f32) -> Self::Values {
                [value; $n / 4]
            }

            #[inline(always)]
            fn get_value(values: &Self::Values, value: usize) -> f32 {
                values[value]
            }

            #[inline(always)]
            fn set_value(values: &mut Self::Values, value: usize, x: f32) {
                values[value] = x;
            }
        }
    )*};
}

// Every multiple of 16 up to 512, in both extents — 1024 tile shapes out of 64
// impls, since `FragmentLayout` is the product of the two. The bound is the
// register file: at the other extent's 16 minimum, 512 is already 256
// registers a thread, one past what the hardware has.
base_ldtm_rows!(
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
);
base_ldtm_cols!(
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
);

/// [`BaseLdtm::WIDEST_BAND`]'s derivation, checked rather than asserted in
/// prose: a thread holds one fp32 per column of its band, so the widest band is
/// the one that fits the registers the architecture has and the next one up is
/// past them before the kernel's own live state.
const _: () = {
    /// Registers a thread has, architecturally.
    const REGISTERS: usize = 255;
    type Widest = RegTile<{ BaseLdtm::WARP_ROWS }, { BaseLdtm::WIDEST_BAND }, BaseLdtm>;
    type Twice = RegTile<{ BaseLdtm::WARP_ROWS }, { 2 * BaseLdtm::WIDEST_BAND }, BaseLdtm>;
    assert!(Widest::SLOTS * Widest::VALUES == BaseLdtm::WIDEST_BAND);
    assert!(Widest::SLOTS * Widest::VALUES <= REGISTERS);
    assert!(Twice::SLOTS * Twice::VALUES > REGISTERS);
};

/// The named half of the op set: one line per exposed name, so a new op costs
/// a [`scalar_ops`] line and one of these. `should_implement_trait` is allowed
/// wholesale because every op device code takes must stay a direct
/// `#[inline(always)]` call, not an operator impl.
macro_rules! op_methods {
    (unary $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[allow(clippy::should_implement_trait)]
        #[inline(always)]
        pub fn $name(self) -> Self {
            self.unary_map::<$op>()
        }
    )*};
    (binary $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[allow(clippy::should_implement_trait)]
        #[inline(always)]
        pub fn $name(self, other: Self) -> Self {
            self.bin_map::<$op>(other)
        }
    )*};
    (scalar $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self, k: f32) -> Self {
            self.scalar_map::<$op>(k)
        }
    )*};
    (row $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self, rows: RegVec<M, L>) -> Self {
            self.row_map::<$op>(rows)
        }
    )*};
    (col $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self, cols: ColVec<N, L>) -> Self {
            self.col_map::<$op>(cols)
        }
    )*};
    (assign $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(&mut self, other: Self) {
            self.bin_map_assign::<$op>(other);
        }
    )*};
    (scalar_assign $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(&mut self, k: f32) {
            self.scalar_map_assign::<$op>(k);
        }
    )*};
    (row_assign $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(&mut self, rows: RegVec<M, L>) {
            self.row_map_assign::<$op>(rows);
        }
    )*};
    (col_assign $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(&mut self, cols: ColVec<N, L>) {
            self.col_map_assign::<$op>(cols);
        }
    )*};
    (row_reduce $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self) -> RegVec<M, L> {
            self.row_reduce::<$op>()
        }
    )*};
    (col_reduce $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self) -> ColVec<N, L> {
            self.col_reduce::<$op>()
        }
    )*};
    (tile_reduce $($(#[$meta:meta])* $name:ident = $op:ty;)*) => {$(
        $(#[$meta])*
        #[inline(always)]
        pub fn $name(self) -> f32 {
            self.tile_reduce::<$op>()
        }
    )*};
}

/// The unary names every register family carries.
macro_rules! unary_op_methods {
    () => {
        op_methods! { unary
            /// Software `2^x` ([`exp2_approx`]) — what every `exp2` in the
            /// crate resolves to, and the spelling that also has a value off a
            /// device.
            exp2 = Exp2Approx;
            /// SFU `2^x` ([`exp2_hw`]) — a different result from `exp2`, and
            /// faster in a loop whose arithmetic is the exponential (#81).
            exp2_hw = Exp2Hw;
            exp = Exp;
            log = Log;
            log2 = Log2;
            abs = Abs;
            neg = Neg;
            relu = Relu;
            sqrt = Sqrt;
            rsqrt = Rsqrt;
            recip = Recip;
        }
    };
}

/// The scalar-operand names every register family carries. [`RegTile::scalar_map`]
/// already reaches every [`BinaryOp`], so this table holds only the spellings a
/// kernel would otherwise invent a worse name for.
macro_rules! scalar_op_methods {
    () => {
        op_methods! { scalar
            /// Multiply by a warp-uniform constant — attention's `1/√d`, a
            /// normalization's `1/N`.
            scale = Mul;
            /// Add a warp-uniform constant; `shift(-mean)` is the centering
            /// step, and the reason there is no separate `Sub` name.
            shift = Add;
            /// Lower-bound every value (`Max` against `k`) — a leaky floor, or
            /// a variance kept off zero.
            clamp_min = Max;
            /// Upper-bound every value; the mirror of [`Self::clamp_min`].
            clamp_max = Min;
        }
    };
}

/// The in-place twin of [`scalar_op_methods`], name for name.
macro_rules! scalar_assign_op_methods {
    () => {
        op_methods! { scalar_assign
            /// [`Self::scale`] rewriting the receiver.
            scale_assign = Mul;
            /// [`Self::shift`] rewriting the receiver.
            shift_assign = Add;
            /// [`Self::clamp_min`] rewriting the receiver.
            clamp_min_assign = Max;
            /// [`Self::clamp_max`] rewriting the receiver.
            clamp_max_assign = Min;
        }
    };
}

/// Per-thread row statistics of an `[M, _]` fragment-mapped tile: one `f32`
/// per owned row (slot), replicated across each quad. A 32-row warp tile is 4
/// slots (2 per 16-row block × 2 blocks per warp).
///
/// What [`RegTile::row_reduce`] returns and what [`RegTile::row_map`] takes.
/// Every op is a compile-time-length loop over the slot array — straight-line
/// code after inlining.
///
/// ```no_run
/// # use kittens::lane;
/// # use kittens::reg::{BaseLdtm, RegTile, RegVec};
/// # unsafe fn demo(scores: RegTile<32, 64, BaseLdtm>) {
/// let row_max: RegVec<32, BaseLdtm> = scores.row_max();
/// let probs = scores.sub_row(row_max).exp2();
/// let mut running_sum = probs.row_sum();
/// running_sum.mul_assign(RegVec::splat(0.5));
/// # }
/// ```
pub struct RegVec<const M: usize, L: RowLayout<M>>(pub L::Slots<f32>);

impl<const M: usize, L: RowLayout<M>> Clone for RegVec<M, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const M: usize, L: RowLayout<M>> Copy for RegVec<M, L> {}

impl<const M: usize, L: RowLayout<M>> RegVec<M, L> {
    /// Rows of the tile this thread owns.
    pub const SLOTS: usize = L::SLOTS;

    /// Wrap this thread's slots.
    #[inline(always)]
    pub fn from_slots(slots: L::Slots<f32>) -> Self {
        Self(slots)
    }

    /// All slots set to `value` (e.g. a masked-score sentinel, or zero).
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self(L::splat_slots(value))
    }

    /// The statistic of row-slot `slot`.
    #[inline(always)]
    pub fn get(&self, slot: usize) -> f32 {
        *L::get_slot(&self.0, slot)
    }

    /// Write the statistic of row-slot `slot`.
    #[inline(always)]
    pub fn set(&mut self, slot: usize, value: f32) {
        *L::get_slot_mut(&mut self.0, slot) = value;
    }

    /// The logical row in `0..M` that `lane`'s `slot` holds.
    #[inline(always)]
    pub fn row(lane: u32, slot: usize) -> u32 {
        L::row_of(lane, slot)
    }

    /// True if any slot exceeds `reference + slack` — the correction-vote
    /// predicate.
    ///
    /// **This lane's vote only.** Combining the votes across the warp or
    /// warpgroup is the caller's collective step.
    #[inline(always)]
    pub fn any_exceeds(self, reference: Self, slack: f32) -> bool {
        let mut exceed = false;
        let mut slot = 0;
        while slot < L::SLOTS {
            exceed = exceed || self.get(slot) > reference.get(slot) + slack;
            slot += 1;
        }
        exceed
    }

    /// Complete each slot's lane-local partial into a whole-row statistic by
    /// folding across the quad — the second half of [`RegTile::row_reduce`],
    /// exposed for a caller that formed its own partials (a running softmax
    /// sum, say). All four lanes of each quad must call it.
    #[inline(always)]
    pub fn quad_reduce<Op: ReduceOp>(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, quad_reduce::<Op>(self.get(slot)));
            slot += 1;
        }
        out
    }

    /// Quad-reduce each slot's lane-local max into a whole-row max.
    #[inline(always)]
    pub fn quad_max(self) -> Self {
        self.quad_reduce::<Max>()
    }

    /// Quad-reduce each slot's lane-local partial sum into a whole-row sum.
    #[inline(always)]
    pub fn quad_sum(self) -> Self {
        self.quad_reduce::<Add>()
    }

    /// Fold every row's statistic into one warp-uniform scalar: this thread's
    /// slots, then [`column_group_reduce`] across the 8 lanes holding the
    /// tile's other rows. Warp-collective.
    ///
    /// **Only meaningful on a whole-row statistic** — one replicated across
    /// each quad, which is what [`RegTile::row_reduce`] returns and what a
    /// lane-local partial is not. Starting from a tile,
    /// [`RegTile::tile_reduce`] gets the same answer in five shuffles instead
    /// of `2 * SLOTS + 3`.
    #[inline(always)]
    pub fn reduce<Op: ReduceOp>(self) -> f32 {
        let mut folded = Op::IDENTITY;
        let mut slot = 0;
        while slot < L::SLOTS {
            folded = Op::apply(folded, self.get(slot));
            slot += 1;
        }
        column_group_reduce::<Op>(folded)
    }

    /// `Op` on every slot, rewriting this vector.
    #[inline(always)]
    pub fn unary_map_assign<Op: UnaryOp>(&mut self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, Op::apply(self.get(slot)));
            slot += 1;
        }
    }

    /// `Op` on every slot.
    #[inline(always)]
    pub fn unary_map<Op: UnaryOp>(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, Op::apply(self.get(slot)));
            slot += 1;
        }
        out
    }

    /// `Op` slotwise against `other`, rewriting this vector.
    #[inline(always)]
    pub fn bin_map_assign<Op: BinaryOp>(&mut self, other: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, Op::apply(self.get(slot), other.get(slot)));
            slot += 1;
        }
    }

    /// `Op` slotwise against `other`.
    #[inline(always)]
    pub fn bin_map<Op: BinaryOp>(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, Op::apply(self.get(slot), other.get(slot)));
            slot += 1;
        }
        out
    }

    /// `Op` slotwise against one scalar, rewriting this vector.
    #[inline(always)]
    pub fn scalar_map_assign<Op: BinaryOp>(&mut self, k: f32) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, Op::apply(self.get(slot), k));
            slot += 1;
        }
    }

    /// `Op` slotwise against one scalar — `scalar_map::<Div>(k)` for a divide,
    /// `scalar_map::<Max>(k)` for a floor; see [`RegTile::scalar_map`].
    #[inline(always)]
    pub fn scalar_map<Op: BinaryOp>(self, k: f32) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, Op::apply(self.get(slot), k));
            slot += 1;
        }
        out
    }

    /// `Op` slotwise across `self`, `b` and `c`, rewriting this vector.
    #[inline(always)]
    pub fn ternary_map_assign<Op: TernaryOp>(&mut self, b: Self, c: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, Op::apply(self.get(slot), b.get(slot), c.get(slot)));
            slot += 1;
        }
    }

    /// `Op` slotwise across three vectors; [`Fma`] is the only such op today.
    #[inline(always)]
    pub fn ternary_map<Op: TernaryOp>(self, b: Self, c: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, Op::apply(self.get(slot), b.get(slot), c.get(slot)));
            slot += 1;
        }
        out
    }

    /// Slotwise `self * b + c`, fused.
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        self.ternary_map::<Fma>(b, c)
    }

    /// Slotwise `self = self * b + c`, fused.
    #[inline(always)]
    pub fn fma_assign(&mut self, b: Self, c: Self) {
        self.ternary_map_assign::<Fma>(b, c);
    }

    unary_op_methods!();
    scalar_op_methods!();
    scalar_assign_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
    }

    op_methods! { assign
        /// Slotwise `self += other`.
        add_assign = Add;
        sub_assign = Sub;
        /// Slotwise `self *= other`.
        mul_assign = Mul;
        div_assign = Div;
        max_assign = Max;
        min_assign = Min;
    }
}

/// Per-thread column statistics of a `[_, N]` fragment-mapped tile: one `f32`
/// per owned column (value). The mirror of [`RegVec`] across the transpose.
///
/// Under [`BaseLdtm`] a lane's columns depend only on `lane % 4`, so the 8
/// lanes of a column group hold 8 copies of the same `N/4` entries. Those
/// copies agree only after a fold across the group, which is what
/// [`RegTile::col_reduce`] does. A vector built any other way (splatted, or
/// from [`Self::column`]) is a legitimate `col_map` operand but carries no such
/// guarantee, and [`Self::reduce`] is only meaningful on one that does.
///
/// ```no_run
/// # use kittens::reg::{BaseLdtm, ColVec, RegTile};
/// # unsafe fn demo(tile: RegTile<32, 64, BaseLdtm>) {
/// let means: ColVec<64, BaseLdtm> = tile.col_sum().scale(1.0 / 32.0);
/// let centered = tile.sub_col(means);
/// # }
/// ```
pub struct ColVec<const N: usize, L: ColLayout<N>>(pub L::Values);

impl<const N: usize, L: ColLayout<N>> Clone for ColVec<N, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, L: ColLayout<N>> Copy for ColVec<N, L> {}

impl<const N: usize, L: ColLayout<N>> ColVec<N, L> {
    /// Columns of the tile this thread owns.
    pub const VALUES: usize = L::VALUES;

    /// Wrap this thread's values.
    #[inline(always)]
    pub fn from_values(values: L::Values) -> Self {
        Self(values)
    }

    /// All values set to `value`.
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self(L::splat_values(value))
    }

    /// The statistic of column-value `value`.
    #[inline(always)]
    pub fn get(&self, value: usize) -> f32 {
        L::get_value(&self.0, value)
    }

    /// Write the statistic of column-value `value`.
    #[inline(always)]
    pub fn set(&mut self, value: usize, x: f32) {
        L::set_value(&mut self.0, value, x);
    }

    /// The logical column in `0..N` that `lane`'s `value` holds.
    #[inline(always)]
    pub fn column(lane: u32, value: usize) -> u32 {
        L::col_of(lane, value)
    }

    /// Fold every column's statistic into one warp-uniform scalar: this
    /// thread's values, then [`quad_reduce`] across the 4 lanes holding the
    /// tile's other columns. Warp-collective.
    ///
    /// **Only meaningful once the 8 copies agree** — i.e. on a vector
    /// [`RegTile::col_reduce`] produced. The mirror of [`RegVec::reduce`].
    #[inline(always)]
    pub fn reduce<Op: ReduceOp>(self) -> f32 {
        let mut folded = Op::IDENTITY;
        let mut value = 0;
        while value < L::VALUES {
            folded = Op::apply(folded, self.get(value));
            value += 1;
        }
        quad_reduce::<Op>(folded)
    }

    /// Complete each value's lane-local partial into a whole-column statistic
    /// by folding across the column group — the step that makes the 8 copies of
    /// a [`ColVec`] agree. All 8 lanes of the group must call it.
    #[inline(always)]
    pub fn column_group_reduce<Op: ReduceOp>(self) -> Self {
        let mut out = self;
        let mut value = 0;
        while value < L::VALUES {
            out.set(value, column_group_reduce::<Op>(self.get(value)));
            value += 1;
        }
        out
    }

    /// `Op` on every value, rewriting this vector.
    #[inline(always)]
    pub fn unary_map_assign<Op: UnaryOp>(&mut self) {
        let mut value = 0;
        while value < L::VALUES {
            self.set(value, Op::apply(self.get(value)));
            value += 1;
        }
    }

    /// `Op` on every value.
    #[inline(always)]
    pub fn unary_map<Op: UnaryOp>(self) -> Self {
        let mut out = self;
        let mut value = 0;
        while value < L::VALUES {
            out.set(value, Op::apply(self.get(value)));
            value += 1;
        }
        out
    }

    /// `Op` valuewise against `other`, rewriting this vector.
    #[inline(always)]
    pub fn bin_map_assign<Op: BinaryOp>(&mut self, other: Self) {
        let mut value = 0;
        while value < L::VALUES {
            self.set(value, Op::apply(self.get(value), other.get(value)));
            value += 1;
        }
    }

    /// `Op` valuewise against `other`.
    #[inline(always)]
    pub fn bin_map<Op: BinaryOp>(self, other: Self) -> Self {
        let mut out = self;
        let mut value = 0;
        while value < L::VALUES {
            out.set(value, Op::apply(self.get(value), other.get(value)));
            value += 1;
        }
        out
    }

    /// `Op` valuewise against one scalar, rewriting this vector.
    #[inline(always)]
    pub fn scalar_map_assign<Op: BinaryOp>(&mut self, k: f32) {
        let mut value = 0;
        while value < L::VALUES {
            self.set(value, Op::apply(self.get(value), k));
            value += 1;
        }
    }

    /// `Op` valuewise against one scalar; see [`RegTile::scalar_map`].
    #[inline(always)]
    pub fn scalar_map<Op: BinaryOp>(self, k: f32) -> Self {
        let mut out = self;
        let mut value = 0;
        while value < L::VALUES {
            out.set(value, Op::apply(self.get(value), k));
            value += 1;
        }
        out
    }

    unary_op_methods!();
    scalar_op_methods!();
    scalar_assign_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
    }

    op_methods! { assign
        add_assign = Add;
        sub_assign = Sub;
        mul_assign = Mul;
        div_assign = Div;
        max_assign = Max;
        min_assign = Min;
    }
}

/// One thread's slice of a logical `[M, N]` fp32 tile held across a warp under
/// the ownership map `L`: `L::SLOTS` owned rows × `L::VALUES` owned values per
/// row. `L` maps `(lane, slot, value)` back to the logical `(row, column)`, so
/// nothing here has to know how the fragment is scattered.
///
/// `M` and `N` are one warp's extents. Elementwise ops are lane-local;
/// reductions are warp-collective and every lane must reach them.
///
/// ```no_run
/// # use kittens::lane;
/// # use kittens::reg::{BaseLdtm, RegTile};
/// # unsafe fn demo(scores: RegTile<32, 64, BaseLdtm>) -> RegTile<32, 64, BaseLdtm> {
/// let mut acc = RegTile::<32, 64, BaseLdtm>::zero();
/// let mut band = scores.scale(0.125);
/// band.right_fill(lane(), 48, f32::NEG_INFINITY);
/// acc.add_assign(band.exp2());
/// acc
/// # }
/// ```
pub struct RegTile<const M: usize, const N: usize, L: FragmentLayout<M, N>>(pub L::Storage);

impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> Clone for RegTile<M, N, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> Copy for RegTile<M, N, L> {}

/// The `[16, 16]` tile one [`crate::tmem::TmemTile`] drain returns: 2 slots ×
/// 4 values per thread. Block `(row_block, column_block)`'s `(slot, value)`
/// sits at `(2*row_block + slot, 4*column_block + value)` of a bigger tile.
pub type Fragment = RegTile<16, 16, BaseLdtm>;

impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> RegTile<M, N, L> {
    /// Rows of the tile this thread owns.
    pub const SLOTS: usize = L::SLOTS;
    /// Values this thread owns per row.
    pub const VALUES: usize = L::VALUES;

    /// Wrap this thread's values, slot-major.
    #[inline(always)]
    pub fn from_values(values: L::Storage) -> Self {
        Self(values)
    }

    /// The additive identity — a fresh accumulator.
    #[inline(always)]
    pub fn zero() -> Self {
        Self(L::splat(0.0))
    }

    /// Every value set to `value`.
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self(L::splat(value))
    }

    /// Every column of row-slot `s` set to `rows` slot `s`.
    #[inline(always)]
    pub fn broadcast_row(rows: RegVec<M, L>) -> Self {
        let mut out = Self::zero();
        let mut slot = 0;
        while slot < L::SLOTS {
            let row = rows.get(slot);
            let mut value = 0;
            while value < L::VALUES {
                out.set(slot, value, row);
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// Every row's `value` set to `cols` value `value`.
    #[inline(always)]
    pub fn broadcast_col(cols: ColVec<N, L>) -> Self {
        let mut out = Self::zero();
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(slot, value, cols.get(value));
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// The value at `(slot, value)`.
    #[inline(always)]
    pub fn get(&self, slot: usize, value: usize) -> f32 {
        L::get(&self.0, slot, value)
    }

    /// Write `x` at `(slot, value)`.
    #[inline(always)]
    pub fn set(&mut self, slot: usize, value: usize, x: f32) {
        L::set(&mut self.0, slot, value, x);
    }

    /// The logical `(row, column)` in `0..M × 0..N` that `lane`'s
    /// `(slot, value)` holds — what a masking or per-column pass indexes by.
    #[inline(always)]
    pub fn coordinate(lane: u32, slot: usize, value: usize) -> (u32, u32) {
        (L::row_of(lane, slot), L::col_of(lane, value))
    }

    /// Replace every value whose logical coordinate `keep` rejects with
    /// `fill`, in place — the mechanism under the masks and fills below.
    ///
    /// `keep` sees the **logical** `(row, column)` in `0..M × 0..N`, not the
    /// storage position, and must be pure: it runs once per owned value.
    /// Lane-local — no lane learns anything from another — so `lane` must be
    /// the calling thread's own.
    ///
    /// Coordinates are `i32`, not `u32`, because a tiled kernel's bounds are
    /// differences of block origins and land outside `0..M × 0..N` in both
    /// directions. That is the common case, not the edge one: a band wholly
    /// above the diagonal takes `keep` false everywhere.
    ///
    /// **Both walks unroll**, which is not decoration: a masking write under a
    /// data-dependent condition keeps the slot array addressable, so a rolled
    /// walk here homes the whole tile to a `.local` frame — and it homes it on
    /// every iteration of whatever loop the mask sits in, not only the masked
    /// one. `#166`'s pattern, the marker plus an inline-`const` bound, is what
    /// makes every `set` a constant slot and leaves the tile in registers.
    /// This is the only walk in this file that carries it; the maps and
    /// reductions are the same shape and are deliberately left rolled, because
    /// a map walk's frame is a band that does not *fit* and unrolling it buys
    /// a spill (`#166`'s measurement, `modal_app.py`'s census comment).
    ///
    /// ```no_run
    /// # use kittens::lane;
    /// # use kittens::reg::{BaseLdtm, RegTile};
    /// # unsafe fn demo(scores: &mut RegTile<32, 64, BaseLdtm>) {
    /// // A sliding window of 16 keys, in this band's own coordinates.
    /// scores.mask(lane(), -1.0e30, |row, column| {
    ///     column <= row && column > row - 16
    /// });
    /// # }
    /// ```
    #[inline(always)]
    pub fn mask(&mut self, lane: u32, fill: f32, keep: impl Fn(i32, i32) -> bool) {
        let mut slot = 0;
        while slot < const { L::SLOTS } {
            __unroll_config::<0>();
            let row = L::row_of(lane, slot) as i32;
            let mut value = 0;
            while value < const { L::VALUES } {
                __unroll_config::<0>();
                let column = L::col_of(lane, value) as i32;
                if !keep(row, column) {
                    self.set(slot, value, fill);
                }
                value += 1;
            }
            slot += 1;
        }
    }

    /// Keep the lower triangle — `column - row <= diagonal` — and fill the
    /// rest.
    ///
    /// `diagonal` is where the boundary crosses this tile's own `(0, 0)`, so a
    /// tile that is a sub-block of a larger matrix passes the difference of its
    /// origins and gets the larger matrix's diagonal. Passing `0` masks the
    /// tile against itself, which is right for exactly one block of a tiled
    /// kernel; see [`Self::make_causal_at`].
    #[inline(always)]
    pub fn tril(&mut self, lane: u32, diagonal: i32, fill: f32) {
        self.mask(lane, fill, |row, column| column - row <= diagonal);
    }

    /// Keep the upper triangle, `column - row >= diagonal`; the mirror of
    /// [`Self::tril`] and its exact complement at `diagonal + 1`.
    #[inline(always)]
    pub fn triu(&mut self, lane: u32, diagonal: i32, fill: f32) {
        self.mask(lane, fill, |row, column| column - row >= diagonal);
    }

    /// Causal attention's mask: a query attends to no key after it, so this is
    /// [`Self::tril`] under the name every attention kernel knows it by, with
    /// `fill` the pre-softmax sentinel (`f32::NEG_INFINITY`, or a finite
    /// `-1e30` if the row could be entirely masked).
    #[inline(always)]
    pub fn make_causal(&mut self, lane: u32, diagonal: i32, fill: f32) {
        self.tril(lane, diagonal, fill);
    }

    /// [`Self::make_causal`] for a transposed score band — `Kᵀ·Q` rather than
    /// `Q·Kᵀ` — which is [`Self::triu`] about the same diagonal.
    #[inline(always)]
    pub fn make_causal_t(&mut self, lane: u32, diagonal: i32, fill: f32) {
        self.triu(lane, diagonal, fill);
    }

    /// [`Self::make_causal`] taking the two block origins a flash kernel
    /// already holds: the band covers queries `query_base..query_base + M`
    /// against keys `key_base..key_base + N`, and its diagonal sits at
    /// `query_base - key_base`.
    ///
    /// **Prefer this to computing the difference at the call site.** Both
    /// origins are `u32` there, and the difference is negative for every band
    /// above the diagonal — the fully-masked ones — so a `u32` subtraction
    /// wraps to a huge positive number and masks nothing. This subtracts in
    /// `i32`.
    ///
    /// ```no_run
    /// # use kittens::lane;
    /// # use kittens::reg::{BaseLdtm, RegTile};
    /// # unsafe fn demo(scores: &mut RegTile<32, 64, BaseLdtm>, key_block: u32) {
    /// scores.make_causal_at(lane(), 32 * kittens::warp_id(), 64 * key_block, -1.0e30);
    /// # }
    /// ```
    #[inline(always)]
    pub fn make_causal_at(&mut self, lane: u32, query_base: u32, key_base: u32, fill: f32) {
        self.make_causal(lane, query_base as i32 - key_base as i32, fill);
    }

    /// [`Self::make_causal_at`] for a transposed score band: the band covers
    /// keys `key_base..key_base + M` against queries `query_base..query_base +
    /// N`, so its rows are the keys and its diagonal sits at
    /// `key_base - query_base`.
    ///
    /// Row origin first, as in [`Self::make_causal_at`] — the two differ in
    /// which axis is which, not in the order they are named. The reason to
    /// prefer it over [`Self::make_causal_t`] is the same and is if anything
    /// stronger here: an attention backward pass that owns a block of *keys*
    /// streams the queries at and after them, so `key_base - query_base` is
    /// negative for every band but the first one it visits, and a `u32`
    /// subtraction there masks nothing.
    ///
    /// ```no_run
    /// # use kittens::lane;
    /// # use kittens::reg::{BaseLdtm, RegTile};
    /// # unsafe fn demo(scores: &mut RegTile<32, 64, BaseLdtm>, query_block: u32) {
    /// scores.make_causal_t_at(lane(), 32 * kittens::warp_id(), 64 * query_block, -1.0e30);
    /// # }
    /// ```
    #[inline(always)]
    pub fn make_causal_t_at(&mut self, lane: u32, key_base: u32, query_base: u32, fill: f32) {
        self.make_causal_t(lane, key_base as i32 - query_base as i32, fill);
    }

    /// Fill the columns at and right of `column`, keeping the rest — the
    /// ragged-tail mask, with `column` the number of real keys left at this
    /// band's origin (`keys - key_base`, which may be negative or past `N`).
    #[inline(always)]
    pub fn right_fill(&mut self, lane: u32, column: i32, fill: f32) {
        self.mask(lane, fill, |_, c| c < column);
    }

    /// Fill the columns left of `column`; the mirror of [`Self::right_fill`],
    /// and a sliding window's other edge.
    #[inline(always)]
    pub fn left_fill(&mut self, lane: u32, column: i32, fill: f32) {
        self.mask(lane, fill, |_, c| c >= column);
    }

    /// Fill the rows above `row` — [`Self::left_fill`] on the query axis.
    #[inline(always)]
    pub fn upper_fill(&mut self, lane: u32, row: i32, fill: f32) {
        self.mask(lane, fill, |r, _| r >= row);
    }

    /// Fill the rows at and below `row`: [`Self::right_fill`] on the query
    /// axis, for a band whose last rows run past the sequence.
    #[inline(always)]
    pub fn lower_fill(&mut self, lane: u32, row: i32, fill: f32) {
        self.mask(lane, fill, |r, _| r < row);
    }

    /// Scale every value in row-slot `s` by `factors` slot `s` — the
    /// running-max rescale of an online-softmax accumulator.
    ///
    /// The same arithmetic as [`Self::mul_row`], but in place: at the flash
    /// accumulator's width the by-value spelling costs 255 registers/thread
    /// against this one's 168.
    #[inline(always)]
    pub fn scale_rows(&mut self, factors: RegVec<M, L>) {
        self.row_map_assign::<Mul>(factors);
    }

    /// `Op` on every owned value, rewriting this tile.
    ///
    /// Each by-value map below computes exactly this into a fresh copy; the
    /// copy is the whole of the difference, and it is the difference that
    /// costs. [`Self::bin_map_assign`] carries the rule for choosing.
    #[inline(always)]
    pub fn unary_map_assign<Op: UnaryOp>(&mut self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                self.set(slot, value, Op::apply(self.get(slot, value)));
                value += 1;
            }
            slot += 1;
        }
    }

    // This loop is written out rather than delegating to `unary_map_assign` on
    // a copy, as is every by-value map below. The delegating form is the
    // obvious factoring and it moves register counts in both directions:
    // 71 -> 32 on one probe, 64 -> 80 on another, 168 -> 128 with a 512-byte
    // stack frame on a third.

    /// `Op` on every owned value.
    ///
    /// ```no_run
    /// # use kittens::reg::{BaseLdtm, Relu, RegTile};
    /// # unsafe fn demo(tile: RegTile<32, 64, BaseLdtm>) -> RegTile<32, 64, BaseLdtm> {
    /// tile.unary_map::<Relu>()
    /// # }
    /// ```
    #[inline(always)]
    pub fn unary_map<Op: UnaryOp>(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(slot, value, Op::apply(self.get(slot, value)));
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// `Op` against `other` at the same logical coordinate, rewriting this
    /// tile.
    ///
    /// **When to write this instead of [`Self::bin_map`].** What costs
    /// registers is a whole band materialized *between statements*, so the rule
    /// at a call site is: say the whole step in one expression if you can,
    /// write it in place if you cannot. That is where the input is the output —
    /// an accumulator, or a rescale of one — and at `[32, 128]` the difference
    /// is 168 registers against 255 plus a spill. Where a by-value spelling
    /// already rebinds a dead input it costs nothing.
    ///
    /// ```no_run
    /// # use kittens::reg::{Add, BaseLdtm, RegTile};
    /// # unsafe fn demo(mut acc: RegTile<32, 128, BaseLdtm>, block: RegTile<32, 128, BaseLdtm>) {
    /// acc.bin_map_assign::<Add>(block);
    /// # }
    /// ```
    ///
    /// `docs/library/reg.md` has the table these come off.
    #[inline(always)]
    pub fn bin_map_assign<Op: BinaryOp>(&mut self, other: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                self.set(
                    slot,
                    value,
                    Op::apply(self.get(slot, value), other.get(slot, value)),
                );
                value += 1;
            }
            slot += 1;
        }
    }

    /// `Op` against `other` at the same logical coordinate — which is the same
    /// `(slot, value)`, since both tiles ride the same map.
    #[inline(always)]
    pub fn bin_map<Op: BinaryOp>(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(
                    slot,
                    value,
                    Op::apply(self.get(slot, value), other.get(slot, value)),
                );
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// `Op` against one warp-uniform scalar broadcast over every owned value —
    /// `k` in a register, not a tile, so the operand costs one register rather
    /// than `SLOTS * VALUES` and no [`Self::splat`] runs.
    ///
    /// The scalar is the op's *second* operand, so every [`BinaryOp`] is
    /// reachable this way — the named wrappers ([`Self::scale`],
    /// [`Self::shift`], the clamps) are only the common spellings.
    ///
    /// ```no_run
    /// # use kittens::reg::{BaseLdtm, Div, RegTile};
    /// # unsafe fn demo(tile: RegTile<32, 64, BaseLdtm>, n: f32) -> RegTile<32, 64, BaseLdtm> {
    /// tile.scalar_map::<Div>(n)
    /// # }
    /// ```
    #[inline(always)]
    pub fn scalar_map<Op: BinaryOp>(self, k: f32) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(slot, value, Op::apply(self.get(slot, value), k));
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// `Op` against one scalar, rewriting this tile — [`Self::scalar_map`]'s
    /// in-place form. At `[32, 128]` it holds 252 registers and no spill where
    /// the by-value spelling spills 60 bytes.
    #[inline(always)]
    pub fn scalar_map_assign<Op: BinaryOp>(&mut self, k: f32) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                self.set(slot, value, Op::apply(self.get(slot, value), k));
                value += 1;
            }
            slot += 1;
        }
    }

    /// `Op` across `self`, `b` and `c`, rewriting this tile. What it saves over
    /// [`Self::ternary_map`] at 128 columns is unmeasured.
    #[inline(always)]
    pub fn ternary_map_assign<Op: TernaryOp>(&mut self, b: Self, c: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                self.set(
                    slot,
                    value,
                    Op::apply(
                        self.get(slot, value),
                        b.get(slot, value),
                        c.get(slot, value),
                    ),
                );
                value += 1;
            }
            slot += 1;
        }
    }

    /// `Op` across three tiles; see [`Fma`].
    #[inline(always)]
    pub fn ternary_map<Op: TernaryOp>(self, b: Self, c: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(
                    slot,
                    value,
                    Op::apply(
                        self.get(slot, value),
                        b.get(slot, value),
                        c.get(slot, value),
                    ),
                );
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// Elementwise `self * b + c`, fused.
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        self.ternary_map::<Fma>(b, c)
    }

    /// Elementwise `self = self * b + c`, fused.
    #[inline(always)]
    pub fn fma_assign(&mut self, b: Self, c: Self) {
        self.ternary_map_assign::<Fma>(b, c);
    }

    /// `Op` against a per-row scalar broadcast along that row, rewriting this
    /// tile — what [`Self::scale_rows`] is `Mul` of.
    ///
    /// One row's factor is read once and spent before the next row's is formed,
    /// so a `RegVec` operand costs one live register rather than a second band.
    #[inline(always)]
    pub fn row_map_assign<Op: BinaryOp>(&mut self, rows: RegVec<M, L>) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let row = rows.get(slot);
            let mut value = 0;
            while value < L::VALUES {
                self.set(slot, value, Op::apply(self.get(slot, value), row));
                value += 1;
            }
            slot += 1;
        }
    }

    /// `Op` against a per-row scalar broadcast along that row: every value of
    /// row-slot `s` sees `rows` slot `s`. No shuffle — the map already gives a
    /// thread every one of its rows' values, so the operand is lane-local.
    ///
    /// ```no_run
    /// # use kittens::reg::{BaseLdtm, RegTile, Sub};
    /// # unsafe fn demo(scores: RegTile<32, 64, BaseLdtm>) -> RegTile<32, 64, BaseLdtm> {
    /// scores.row_map::<Sub>(scores.row_max()).exp2()
    /// # }
    /// ```
    #[inline(always)]
    pub fn row_map<Op: BinaryOp>(self, rows: RegVec<M, L>) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let row = rows.get(slot);
            let mut value = 0;
            while value < L::VALUES {
                out.set(slot, value, Op::apply(self.get(slot, value), row));
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// `Op` against a per-column scalar broadcast down that column: the value
    /// at `(slot, value)` sees `cols` value `value`, i.e. the scalar for
    /// logical column [`ColVec::column`]`(lane, value)`. Also shuffle-free —
    /// but see [`ColVec`] on which operands carry a whole-column meaning.
    ///
    /// The column walk is the outer one, for [`Self::col_map_assign`]'s reason.
    #[inline(always)]
    pub fn col_map<Op: BinaryOp>(self, cols: ColVec<N, L>) -> Self {
        let mut out = self;
        let mut value = 0;
        while value < L::VALUES {
            let col = cols.get(value);
            let mut slot = 0;
            while slot < L::SLOTS {
                out.set(slot, value, Op::apply(self.get(slot, value), col));
                slot += 1;
            }
            value += 1;
        }
        out
    }

    /// [`Self::col_map`] rewriting this tile — the column mirror of
    /// [`Self::row_map_assign`], down to the loop order.
    ///
    /// The column walk is the *outer* one, so one column's scalar is read once
    /// and spent before the next column's is formed — `L::VALUES` reads of the
    /// operand rather than `L::SLOTS × L::VALUES` of them (#196). That is the
    /// same trade [`Self::row_map_assign`] makes, and it matters here for a
    /// reason that one does not have: `ColVec::get` indexes `[f32; N/4]` with
    /// the loop variable, and these maps are the walks #166 measured *worse*
    /// unrolled, so the index stays a runtime one and the array is a candidate
    /// for the local depot rather than a register. Every element sees the same
    /// operand under an elementwise, independent `Op`, so the result is
    /// bit-identical and only the order the storage is walked in changes.
    ///
    /// **What the hoist is worth** (`device-tests`' `axis_map_probe`,
    /// `regcount`, sm_100a): nothing at all on a `[32, 16]` chunk, where four
    /// slots and four values price the same both ways down to the static
    /// `.local` counts; 167 → 39 registers a thread and 600 → 519 `ld.local`
    /// on a `[32, 128]` one, on an unchanged 2304-byte frame.
    #[inline(always)]
    pub fn col_map_assign<Op: BinaryOp>(&mut self, cols: ColVec<N, L>) {
        let mut value = 0;
        while value < L::VALUES {
            let col = cols.get(value);
            let mut slot = 0;
            while slot < L::SLOTS {
                self.set(slot, value, Op::apply(self.get(slot, value), col));
                slot += 1;
            }
            value += 1;
        }
    }

    // The three reductions below each spell their lane-local fold out rather
    // than sharing one `fold(self, slot)` helper. Taking `self` by value
    // materializes a second copy of the storage even at `#[inline(always)]`:
    // 94 registers/thread against 64 for the written-out loop, and 456 bytes
    // of spill stores at 128 columns.

    /// Fold each row across all `N` columns: this thread's columns of the row,
    /// then [`quad_reduce`] over the quad that holds the rest of them. Two
    /// shuffles per owned row, so **every lane of the warp must call it**.
    ///
    /// `Op` is applied in the *layout's* order, not left to right along the
    /// row. The result is a whole-row statistic replicated across each quad,
    /// which is exactly the operand [`Self::row_map`] wants.
    ///
    /// ```no_run
    /// # use kittens::reg::{Add, BaseLdtm, Max, RegTile};
    /// # unsafe fn demo(scores: RegTile<32, 64, BaseLdtm>) {
    /// let row_max = scores.row_reduce::<Max>();
    /// let probs = scores.sub_row(row_max).exp2();
    /// let denominator = probs.row_reduce::<Add>();
    /// # }
    /// ```
    #[inline(always)]
    pub fn row_reduce<Op: ReduceOp>(self) -> RegVec<M, L> {
        let mut partials = RegVec::<M, L>::splat(Op::IDENTITY);
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut folded = Op::IDENTITY;
            let mut value = 0;
            while value < L::VALUES {
                folded = Op::apply(folded, self.get(slot, value));
                value += 1;
            }
            partials.set(slot, folded);
            slot += 1;
        }
        partials.quad_reduce::<Op>()
    }

    /// Fold each column across all `M` rows: this thread's rows of the column,
    /// then [`column_group_reduce`] over the 8 lanes that hold the rest of
    /// them. Every lane of the warp must call it.
    ///
    /// The op that makes a [`ColVec`] whole — before it, the 8 lanes of a
    /// column group hold 8 independent partials of the same column.
    #[inline(always)]
    pub fn col_reduce<Op: ReduceOp>(self) -> ColVec<N, L> {
        let mut partials = ColVec::<N, L>::splat(Op::IDENTITY);
        let mut value = 0;
        while value < L::VALUES {
            let mut folded = Op::IDENTITY;
            let mut slot = 0;
            while slot < L::SLOTS {
                folded = Op::apply(folded, self.get(slot, value));
                slot += 1;
            }
            partials.set(value, folded);
            value += 1;
        }
        partials.column_group_reduce::<Op>()
    }

    /// Fold the whole tile to one warp-uniform scalar: every register this
    /// thread owns, then [`warp_reduce`] over all 32 lanes. Five shuffles
    /// total, against the `2 * SLOTS + 3` of routing through
    /// [`Self::row_reduce`] and [`RegVec::reduce`]. Every lane must call it.
    ///
    /// **Warp scope.** A statistic over a tile several warps own — layernorm's
    /// group norm across four warps' bands — needs a shared-memory staging
    /// step this gives no help with.
    #[inline(always)]
    pub fn tile_reduce<Op: ReduceOp>(self) -> f32 {
        let mut folded = Op::IDENTITY;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                folded = Op::apply(folded, self.get(slot, value));
                value += 1;
            }
            slot += 1;
        }
        warp_reduce::<Op>(folded)
    }

    unary_op_methods!();
    scalar_op_methods!();
    scalar_assign_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
    }

    op_methods! { assign
        /// `self += other` — an output accumulator taking one block's
        /// contribution.
        add_assign = Add;
        sub_assign = Sub;
        mul_assign = Mul;
        div_assign = Div;
        max_assign = Max;
        min_assign = Min;
    }

    op_methods! { row_reduce
        row_max = Max;
        row_min = Min;
        row_sum = Add;
        /// The product of each row.
        row_prod = Mul;
    }

    op_methods! { col_reduce
        col_max = Max;
        col_min = Min;
        col_sum = Add;
        /// See [`Self::row_prod`].
        col_prod = Mul;
    }

    op_methods! { tile_reduce
        /// Prefixed `tile_` because [`Self::max`] is already the elementwise
        /// binary op.
        tile_max = Max;
        tile_min = Min;
        tile_sum = Add;
        tile_prod = Mul;
    }

    op_methods! { row
        add_row = Add;
        sub_row = Sub;
        /// The value form of [`Self::scale_rows`].
        mul_row = Mul;
        div_row = Div;
    }

    op_methods! { row_assign
        add_row_assign = Add;
        sub_row_assign = Sub;
        /// The in-place form of [`Self::mul_row`]; [`Self::scale_rows`] is its
        /// name at an online-softmax accumulator.
        mul_row_assign = Mul;
        div_row_assign = Div;
    }

    op_methods! { col
        add_col = Add;
        sub_col = Sub;
        mul_col = Mul;
        div_col = Div;
    }

    op_methods! { col_assign
        add_col_assign = Add;
        sub_col_assign = Sub;
        mul_col_assign = Mul;
        div_col_assign = Div;
    }
}

/// One correction step of the online softmax: advance `m_ref` to cover
/// `row_max`, then rescale `running_sum` and `out_acc` into the new reference.
///
/// Lane-local — `row_max` must already be a whole-row statistic
/// ([`RegTile::row_reduce`]) — and all three operands are rewritten in place.
///
/// Fused on purpose: one `next`/`factor` scalar live at a time. Writing the
/// four steps separately (`max` / `sub` / `exp2` / `scale_rows`) keeps two full
/// vectors live across the accumulator scaling and costs 206 → 212
/// registers/thread in the persistent forward kernel on B200.
///
/// ```no_run
/// # use kittens::reg::{BaseLdtm, RegTile, RegVec, online_rescale};
/// # unsafe fn demo(
/// #     scores: RegTile<32, 64, BaseLdtm>,
/// #     m_ref: &mut RegVec<32, BaseLdtm>,
/// #     running_sum: &mut RegVec<32, BaseLdtm>,
/// #     out_acc: &mut RegTile<32, 64, BaseLdtm>,
/// # ) {
/// online_rescale(m_ref, scores.row_max(), running_sum, out_acc);
/// # }
/// ```
#[inline(always)]
pub fn online_rescale<const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    m_ref: &mut RegVec<M, L>,
    row_max: RegVec<M, L>,
    running_sum: &mut RegVec<M, L>,
    out_acc: &mut RegTile<M, N, L>,
) {
    let mut slot = 0;
    while slot < L::SLOTS {
        let next = fmax(m_ref.get(slot), row_max.get(slot));
        let factor = exp2_approx(m_ref.get(slot) - next);
        m_ref.set(slot, next);
        running_sum.set(slot, running_sum.get(slot) * factor);
        let mut value = 0;
        while value < L::VALUES {
            out_acc.set(slot, value, out_acc.get(slot, value) * factor);
            value += 1;
        }
        slot += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32-row × 32-column warp tile: 4 slots of 8 values, the shape the
    /// recurrence tests replay.
    type Scores = RegTile<32, 32, BaseLdtm>;
    /// Its row statistics.
    type Rows = RegVec<32, BaseLdtm>;
    /// Its column statistics.
    type Cols = ColVec<32, BaseLdtm>;

    /// The flash score band.
    type Band = RegTile<32, 64, BaseLdtm>;

    /// A tile whose value at `(row, column)` names that coordinate exactly, so
    /// a map that reads the wrong operand shows up as a wrong coordinate.
    fn indexed<const M: usize, const N: usize, L: FragmentLayout<M, N>>(
        lane: u32,
    ) -> RegTile<M, N, L> {
        let mut tile = RegTile::<M, N, L>::zero();
        for slot in 0..L::SLOTS {
            for value in 0..L::VALUES {
                let (row, column) = RegTile::<M, N, L>::coordinate(lane, slot, value);
                tile.set(slot, value, (256 * row + column) as f32);
            }
        }
        tile
    }

    /// [`indexed`] at the shape most of these tests run on.
    fn coordinate_tile(lane: u32) -> Scores {
        indexed(lane)
    }

    /// The row vector holding each row's own index.
    fn row_indices(lane: u32) -> Rows {
        let mut rows = Rows::splat(0.0);
        for slot in 0..Rows::SLOTS {
            rows.set(slot, Rows::row(lane, slot) as f32);
        }
        rows
    }

    /// The column vector holding each column's own index.
    fn column_indices(lane: u32) -> Cols {
        let mut cols = Cols::splat(0.0);
        for value in 0..Cols::VALUES {
            cols.set(value, Cols::column(lane, value) as f32);
        }
        cols
    }

    #[test]
    fn exp2_polynomial_stays_inside_its_error_bound() {
        // The 7.5e-5 relative-error claim, plus the clamp semantics the
        // masked-score sentinel depends on.
        let mut x = -125.0f32;
        while x <= 125.0 {
            let approx = exp2_approx(x) as f64;
            let exact = (x as f64).exp2();
            assert!(
                ((approx - exact) / exact).abs() < 1.0e-4,
                "exp2_approx({x}) = {approx}, expected {exact}"
            );
            x += 0.137;
        }
        assert!(exp2_approx(-1.0e30) <= 2.0f32.powi(-124));
    }

    #[test]
    fn log2_series_stays_inside_its_error_bound() {
        let mut x = 1.0e-3f32;
        while x < 1.0e6 {
            let approx = log2_approx(x) as f64;
            let exact = (x as f64).log2();
            assert!(
                (approx - exact).abs() < 1.0e-6 + 1.0e-7 * exact.abs(),
                "log2_approx({x}) = {approx}, expected {exact}"
            );
            x *= 1.7;
        }
    }

    #[test]
    fn regvec_ops_match_the_hand_written_recurrence() {
        // The correction rescale from softmax_tile, replayed on both forms.
        let m_ref = Rows::from_slots([-3.0f32, 0.5, 2.0, -1.0e30]);
        let row_max = Rows::from_slots([1.0f32, 0.25, 9.0, -2.0]);
        let next = m_ref.max(row_max);
        let factor = m_ref.sub(next).exp2();
        for slot in 0..Rows::SLOTS {
            let expected_next = if m_ref.get(slot) > row_max.get(slot) {
                m_ref.get(slot)
            } else {
                row_max.get(slot)
            };
            assert_eq!(next.get(slot), expected_next);
            assert_eq!(
                factor.get(slot),
                exp2_approx(m_ref.get(slot) - expected_next)
            );
        }
        assert!(row_max.any_exceeds(m_ref, 8.0));
        assert!(!Rows::splat(0.0).any_exceeds(Rows::splat(0.0), 8.0));

        let mut tile = Scores::zero();
        tile.set(2, 5, 4.0);
        tile.scale_rows(factor);
        assert_eq!(tile.get(2, 5), 4.0 * factor.get(2));
    }

    #[test]
    fn value_columns_are_the_fragment_maps_offsets() {
        // The {0,1,8,9}-per-16-block pattern every drain loop spells by hand,
        // read off lane 0 (whose column pair base is 0).
        assert_eq!(
            (0..8).map(|v| BaseLdtm::column(0, v)).collect::<Vec<_>>(),
            [0, 1, 8, 9, 16, 17, 24, 25]
        );
        // A RegTile<32, 128>'s 32 values cover 128 columns exactly once.
        assert_eq!(RegTile::<32, 128, BaseLdtm>::VALUES, 32);
        let mut columns = (0..32).map(|v| BaseLdtm::column(0, v)).collect::<Vec<_>>();
        columns.sort();
        columns.dedup();
        assert_eq!(columns.len(), 32);
        assert_eq!(columns[31], 121);
    }

    /// The property that makes a layout a valid ownership map: the warp's 32
    /// lanes hold every logical coordinate of the tile, and no coordinate
    /// twice.
    fn covers_each_coordinate_once<const M: usize, const N: usize, L: FragmentLayout<M, N>>() {
        let mut owners = vec![0u32; M * N];
        for lane in 0..32 {
            for slot in 0..L::SLOTS {
                for value in 0..L::VALUES {
                    let (row, column) = RegTile::<M, N, L>::coordinate(lane, slot, value);
                    assert!(
                        (row as usize) < M && (column as usize) < N,
                        "lane {lane} slot {slot} value {value} -> ({row}, {column}) outside [{M}, {N}]"
                    );
                    owners[row as usize * N + column as usize] += 1;
                }
            }
        }
        for row in 0..M {
            for column in 0..N {
                assert_eq!(
                    owners[row * N + column],
                    1,
                    "({row}, {column}) in [{M}, {N}]"
                );
            }
        }
        assert_eq!(32 * L::SLOTS * L::VALUES, M * N);
    }

    #[test]
    fn the_shape_set_is_the_product_of_the_extents() {
        // A shape costs no line of its own, and the storage the blanket impl
        // projects is exactly the `[[f32; N/4]; M/8]` a per-shape impl would
        // name — a spelling, not a representation.
        assert_eq!(size_of::<Band>(), 32 * 64 / 32 * 4);
        assert_eq!(size_of::<RegTile<16, 512, BaseLdtm>>(), 16 * 512 / 32 * 4);
        assert_eq!(size_of::<RegTile<512, 16, BaseLdtm>>(), 512 * 16 / 32 * 4);
        assert_eq!(size_of::<RegVec<48, BaseLdtm>>(), 48 / 8 * 4);
        assert_eq!(size_of::<ColVec<80, BaseLdtm>>(), 80 / 4 * 4);
        // Shapes nothing in this repo names, and a map is still a map on them.
        covers_each_coordinate_once::<48, 80, BaseLdtm>();
        covers_each_coordinate_once::<16, 512, BaseLdtm>();
    }

    #[test]
    fn base_ldtm_covers_each_coordinate_once() {
        covers_each_coordinate_once::<16, 16, BaseLdtm>();
        covers_each_coordinate_once::<32, 32, BaseLdtm>();
        covers_each_coordinate_once::<32, 64, BaseLdtm>();
        covers_each_coordinate_once::<32, 128, BaseLdtm>();
        // Slots follow the rows a thread owns: two per 16-row block.
        assert_eq!(Fragment::SLOTS, 2);
        assert_eq!(Fragment::VALUES, 4);
        assert_eq!(RegTile::<32, 128, BaseLdtm>::SLOTS, 4);
    }

    /// The claim [`ColLayout::CONTIGUOUS_VALUES`] makes, checked as stated
    /// rather than by re-deriving `{0, 1, 8, 9}`: every run of that many
    /// values, from a run-aligned start, is a run of consecutive columns from
    /// a run-aligned column. That is what `store_rows` and `load_rows` turn
    /// into a vector access.
    #[test]
    fn base_ldtm_pairs_every_other_value() {
        fn values_run_in_column_order<const N: usize, L: ColLayout<N>>() {
            let run = L::CONTIGUOUS_VALUES;
            assert!(L::VALUES.is_multiple_of(run), "a partial run has no home");
            for lane in 0..32u32 {
                for start in (0..L::VALUES).step_by(run) {
                    let base = L::col_of(lane, start);
                    assert!(
                        (base as usize).is_multiple_of(run),
                        "N = {N}, lane {lane}, value {start}: column {base} is not run-aligned"
                    );
                    for step in 0..run {
                        assert_eq!(
                            L::col_of(lane, start + step),
                            base + step as u32,
                            "N = {N}, lane {lane}, value {}",
                            start + step
                        );
                    }
                }
            }
        }

        assert_eq!(<BaseLdtm as ColLayout<128>>::CONTIGUOUS_VALUES, 2);
        values_run_in_column_order::<16, BaseLdtm>();
        values_run_in_column_order::<64, BaseLdtm>();
        values_run_in_column_order::<128, BaseLdtm>();
        values_run_in_column_order::<512, BaseLdtm>();
        // And the run stops at two: `{0, 1, 8, 9}` puts values 1 and 2 seven
        // columns apart, which is why nothing here widens to four.
        assert_eq!(BaseLdtm::column(0, 2) - BaseLdtm::column(0, 1), 7);
    }

    /// A `[16, 16]` drain block's own coordinates, which every composed tile
    /// offsets by its block position.
    fn fragment_coordinate(lane: u32, slot: usize, value: usize) -> (u32, u32) {
        Fragment::coordinate(lane, slot, value)
    }

    #[test]
    fn fragment_blocks_tile_the_bigger_shapes() {
        // A TMEM drain only ever returns `Fragment`s, so kernel drain loops
        // assemble a bigger tile by placing block (row_block, column_block)'s
        // (slot, value) at (2*row_block + slot, 4*column_block + value). That
        // must be the same map the bigger shape's own `coordinate` gives.
        fn composes<const M: usize, const N: usize, L: FragmentLayout<M, N>>() {
            for lane in 0..32 {
                for row_block in 0..M / 16 {
                    for column_block in 0..N / 16 {
                        for slot in 0..Fragment::SLOTS {
                            for value in 0..Fragment::VALUES {
                                let (row, column) = fragment_coordinate(lane, slot, value);
                                assert_eq!(
                                    RegTile::<M, N, L>::coordinate(
                                        lane,
                                        2 * row_block + slot,
                                        4 * column_block + value
                                    ),
                                    (
                                        16 * row_block as u32 + row,
                                        16 * column_block as u32 + column
                                    )
                                );
                            }
                        }
                    }
                }
            }
        }
        composes::<16, 16, BaseLdtm>();
        composes::<32, 32, BaseLdtm>();
        composes::<32, 64, BaseLdtm>();
        composes::<32, 128, BaseLdtm>();
    }

    #[test]
    fn unary_ops_are_their_scalar_definitions() {
        // Exp2Hw is absent on purpose: `ex2_approx_f32` has no host body, so
        // only a device kernel can exercise it.
        for x in [0.25f32, 1.0, 2.0, 3.75, 40.0] {
            assert_eq!(Sqrt::apply(x), x.sqrt());
            assert_eq!(Rsqrt::apply(x), 1.0 / x.sqrt());
            assert_eq!(Recip::apply(x), 1.0 / x);
            assert_eq!(Log2::apply(x), log2_approx(x));
            assert_eq!(Exp2Approx::apply(x), exp2_approx(x));
            let ln = Log::apply(x) as f64;
            assert!((ln - (x as f64).ln()).abs() < 1.0e-6 * (1.0 + ln.abs()));
            let exp = Exp::apply(x) as f64;
            assert!((exp / (x as f64).exp() - 1.0).abs() < 2.0e-4);
        }
        for x in [-3.0f32, -0.0, 0.0, 2.5, f32::INFINITY] {
            assert_eq!(Abs::apply(x), x.abs());
            assert_eq!(Neg::apply(x), -x);
            assert_eq!(Relu::apply(x), if x > 0.0 { x } else { 0.0 });
        }
    }

    #[test]
    fn binary_ops_are_their_scalar_definitions() {
        // Tautological per op, but it is what pins the generated name table:
        // a transposed line in `scalar_ops!` makes `div` mean `mul`.
        for (a, b) in [(1.0f32, 2.0f32), (-3.5, 0.25), (7.0, -7.0)] {
            assert_eq!(Add::apply(a, b), a + b);
            assert_eq!(Sub::apply(a, b), a - b);
            assert_eq!(Mul::apply(a, b), a * b);
            assert_eq!(Div::apply(a, b), a / b);
            assert_eq!(Max::apply(a, b), a.max(b));
            assert_eq!(Min::apply(a, b), a.min(b));
            assert_eq!(Fma::apply(a, b, 0.5), a * b + 0.5);
        }
    }

    #[test]
    fn maps_reach_every_owned_value() {
        let tile = coordinate_tile(0);
        let negated = tile.neg();
        let doubled = tile.bin_map::<Add>(tile);
        let fused = tile.fma(Scores::splat(3.0), Scores::splat(1.0));
        for slot in 0..Scores::SLOTS {
            for value in 0..Scores::VALUES {
                let x = tile.get(slot, value);
                assert_eq!(negated.get(slot, value), -x);
                assert_eq!(doubled.get(slot, value), x + x);
                assert_eq!(fused.get(slot, value), x * 3.0 + 1.0);
            }
        }
    }

    #[test]
    fn named_wrappers_resolve_to_their_ops() {
        let vec = Rows::from_slots([0.25f32, 2.0, 4.0, 9.0]);
        let other = Rows::from_slots([1.0f32, -2.0, 0.5, 3.0]);
        for slot in 0..Rows::SLOTS {
            assert_eq!(vec.sqrt().get(slot), Sqrt::apply(vec.get(slot)));
            assert_eq!(vec.rsqrt().get(slot), Rsqrt::apply(vec.get(slot)));
            assert_eq!(vec.recip().get(slot), Recip::apply(vec.get(slot)));
            assert_eq!(vec.relu().get(slot), Relu::apply(vec.get(slot)));
            assert_eq!(vec.abs().get(slot), Abs::apply(vec.get(slot)));
            assert_eq!(vec.neg().get(slot), Neg::apply(vec.get(slot)));
            assert_eq!(vec.log2().get(slot), Log2::apply(vec.get(slot)));
            assert_eq!(vec.log().get(slot), Log::apply(vec.get(slot)));
            assert_eq!(vec.exp().get(slot), Exp::apply(vec.get(slot)));
            assert_eq!(vec.add(other).get(slot), vec.get(slot) + other.get(slot));
            assert_eq!(vec.mul(other).get(slot), vec.get(slot) * other.get(slot));
            assert_eq!(vec.div(other).get(slot), vec.get(slot) / other.get(slot));
            assert_eq!(
                vec.min(other).get(slot),
                fmin(vec.get(slot), other.get(slot))
            );
            assert_eq!(
                vec.max(other).get(slot),
                fmax(vec.get(slot), other.get(slot))
            );
            assert_eq!(vec.sub(other).get(slot), vec.get(slot) - other.get(slot));
        }

        let mut scaled = vec;
        scaled.mul_assign(other);
        let mut summed = vec;
        summed.add_assign(other);
        for slot in 0..Rows::SLOTS {
            assert_eq!(scaled.get(slot), vec.get(slot) * other.get(slot));
            assert_eq!(summed.get(slot), vec.get(slot) + other.get(slot));
        }
    }

    #[test]
    fn scalar_map_is_bin_map_against_the_splatted_scalar() {
        // The claim that makes `scalar_map` a `BinaryOp` map and not a second
        // mechanism: it is the vector/vector form with the operand splatted,
        // for every op, on all three families — so the only thing it saves is
        // the splat, and it saves it identically everywhere.
        let k = -0.75f32;
        let tiles_agree = |mapped: Scores, splatted: Scores| {
            for slot in 0..Scores::SLOTS {
                for value in 0..Scores::VALUES {
                    assert_eq!(mapped.get(slot, value), splatted.get(slot, value));
                }
            }
        };
        let rows_agree = |mapped: Rows, splatted: Rows| {
            for slot in 0..Rows::SLOTS {
                assert_eq!(mapped.get(slot), splatted.get(slot));
            }
        };
        let cols_agree = |mapped: Cols, splatted: Cols| {
            for value in 0..Cols::VALUES {
                assert_eq!(mapped.get(value), splatted.get(value));
            }
        };

        for lane in 0..32 {
            let (tile, wide) = (coordinate_tile(lane), Scores::splat(k));
            tiles_agree(tile.scalar_map::<Add>(k), tile.bin_map::<Add>(wide));
            tiles_agree(tile.scalar_map::<Sub>(k), tile.bin_map::<Sub>(wide));
            tiles_agree(tile.scalar_map::<Mul>(k), tile.bin_map::<Mul>(wide));
            tiles_agree(tile.scalar_map::<Div>(k), tile.bin_map::<Div>(wide));
            tiles_agree(tile.scalar_map::<Max>(k), tile.bin_map::<Max>(wide));
            tiles_agree(tile.scalar_map::<Min>(k), tile.bin_map::<Min>(wide));

            let (vec, wide) = (row_indices(lane), Rows::splat(k));
            rows_agree(vec.scalar_map::<Mul>(k), vec.bin_map::<Mul>(wide));
            rows_agree(vec.scalar_map::<Max>(k), vec.bin_map::<Max>(wide));

            let (cols, wide) = (column_indices(lane), Cols::splat(k));
            cols_agree(cols.scalar_map::<Add>(k), cols.bin_map::<Add>(wide));
            cols_agree(cols.scalar_map::<Div>(k), cols.bin_map::<Div>(wide));
        }
    }

    #[test]
    fn scalar_names_resolve_to_their_ops() {
        // The generated name table, per family: a transposed line here makes
        // `shift` mean `scale`, which is a wrong kernel and not a slow one.
        let k = 2.5f32;
        let tile = coordinate_tile(7);
        let vec = Rows::from_slots([-3.0f32, 0.25, 2.0, 9.0]);
        let cols = column_indices(7);
        for slot in 0..Scores::SLOTS {
            for value in 0..Scores::VALUES {
                let x = tile.get(slot, value);
                assert_eq!(tile.scale(k).get(slot, value), x * k);
                assert_eq!(tile.shift(k).get(slot, value), x + k);
                assert_eq!(tile.clamp_min(k).get(slot, value), fmax(x, k));
                assert_eq!(tile.clamp_max(k).get(slot, value), fmin(x, k));
            }
        }
        for slot in 0..Rows::SLOTS {
            let x = vec.get(slot);
            assert_eq!(vec.scale(k).get(slot), x * k);
            assert_eq!(vec.shift(k).get(slot), x + k);
            assert_eq!(vec.clamp_min(k).get(slot), fmax(x, k));
            assert_eq!(vec.clamp_max(k).get(slot), fmin(x, k));
        }
        for value in 0..Cols::VALUES {
            let x = cols.get(value);
            assert_eq!(cols.scale(k).get(value), x * k);
            assert_eq!(cols.shift(k).get(value), x + k);
            assert_eq!(cols.clamp_min(k).get(value), fmax(x, k));
            assert_eq!(cols.clamp_max(k).get(value), fmin(x, k));
        }
        // And that the ops it does *not* name stay reachable, which is the
        // whole argument for `scalar_map` over a wrapper per op.
        assert_eq!(vec.scalar_map::<Div>(k).get(1), vec.get(1) / k);
        assert_eq!(vec.scalar_map::<Sub>(k).get(1), vec.get(1) - k);
    }

    #[test]
    fn free_rsqrt_is_the_op_struct() {
        // Layernorm's variance is one f32 with no vector to reach `Rsqrt`
        // through, so the two spellings have to be the same number — a host
        // reference compares them with `==`.
        for x in [0.25f32, 1.0, 2.0, 3.75, 1.0e-6, 1.0e12] {
            assert_eq!(rsqrt(x), Rsqrt::apply(x));
            assert_eq!(rsqrt(x), 1.0 / x.sqrt());
            assert_eq!(Rows::splat(x).rsqrt().get(0), rsqrt(x));
        }
    }

    #[test]
    fn exp2_stays_the_polynomial_everywhere() {
        // Which of the two `exp2`s a *name* resolves to is a numerics
        // decision, so it gets an assertion rather than a convention.
        let vec = Rows::from_slots([-3.0f32, 0.0, 1.5, 7.25]);
        for slot in 0..Rows::SLOTS {
            assert_eq!(vec.exp2().get(slot), exp2_approx(vec.get(slot)));
        }
        let mut tile = Scores::zero();
        tile.set(1, 2, 3.5);
        assert_eq!(tile.exp2().get(1, 2), exp2_approx(3.5));
        assert_eq!(Cols::splat(3.5).exp2().get(0), exp2_approx(3.5));
    }

    #[test]
    fn row_map_broadcasts_along_the_layouts_rows() {
        for lane in 0..32 {
            let tile = coordinate_tile(lane);
            let sums = tile.add_row(row_indices(lane));
            let products = tile.mul_row(row_indices(lane));
            for slot in 0..Scores::SLOTS {
                for value in 0..Scores::VALUES {
                    let (row, _) = Scores::coordinate(lane, slot, value);
                    let x = tile.get(slot, value);
                    assert_eq!(sums.get(slot, value), x + row as f32);
                    assert_eq!(products.get(slot, value), x * row as f32);
                }
            }
        }
    }

    #[test]
    fn col_map_broadcasts_down_the_layouts_columns() {
        for lane in 0..32 {
            let tile = coordinate_tile(lane);
            let sums = tile.add_col(column_indices(lane));
            let differences = tile.sub_col(column_indices(lane));
            for slot in 0..Scores::SLOTS {
                for value in 0..Scores::VALUES {
                    let (_, column) = Scores::coordinate(lane, slot, value);
                    let x = tile.get(slot, value);
                    assert_eq!(sums.get(slot, value), x + column as f32);
                    assert_eq!(differences.get(slot, value), x - column as f32);
                }
            }
        }
    }

    #[test]
    fn broadcasts_are_constant_along_their_free_axis() {
        for lane in 0..32 {
            let rows = Scores::broadcast_row(row_indices(lane));
            let cols = Scores::broadcast_col(column_indices(lane));
            for slot in 0..Scores::SLOTS {
                for value in 0..Scores::VALUES {
                    let (row, column) = Scores::coordinate(lane, slot, value);
                    assert_eq!(rows.get(slot, value), row as f32);
                    assert_eq!(cols.get(slot, value), column as f32);
                }
            }
        }
    }

    #[test]
    fn column_values_are_a_lane_groups_share_of_the_columns() {
        // A ColVec entry belongs to `col_of(lane, value)`, which depends only
        // on `lane % 4` — so the four groups' entries tile 0..N exactly once
        // and the other 28 lanes hold duplicates of them.
        for lane in 0..32u32 {
            for value in 0..Cols::VALUES {
                assert_eq!(Cols::column(lane, value), Cols::column(lane % 4, value));
            }
        }
        let mut columns: Vec<u32> = (0..4)
            .flat_map(|lane| (0..Cols::VALUES).map(move |value| Cols::column(lane, value)))
            .collect();
        columns.sort();
        assert_eq!(columns, (0..32).collect::<Vec<_>>());
    }

    /// The lanes `shuffle_xor` reaches from `lane` with the given masks: the
    /// xor-closure, which for a set of distinct bits is every subset xor.
    fn xor_group(lane: u32, masks: &[u32]) -> Vec<u32> {
        let mut group: Vec<u32> = (0..1u32 << masks.len())
            .map(|subset| {
                masks.iter().enumerate().fold(lane, |l, (bit, mask)| {
                    if subset >> bit & 1 == 1 { l ^ mask } else { l }
                })
            })
            .collect();
        group.sort();
        group.dedup();
        group
    }

    /// The lanes holding the same rows as `lane` — where a row reduction's
    /// shuffle must reach, read off the ownership map rather than assumed.
    fn lanes_sharing_rows(lane: u32) -> Vec<u32> {
        (0..32)
            .filter(|&other| (0..Rows::SLOTS).all(|s| Rows::row(other, s) == Rows::row(lane, s)))
            .collect()
    }

    /// The lanes holding the same columns as `lane`; the column mirror.
    fn lanes_sharing_columns(lane: u32) -> Vec<u32> {
        (0..32)
            .filter(|&other| {
                (0..Cols::VALUES).all(|v| Cols::column(other, v) == Cols::column(lane, v))
            })
            .collect()
    }

    #[test]
    fn reduction_masks_are_the_ownership_maps_lane_groups() {
        // The whole correctness argument for the reductions, and the half a
        // wrong answer would look plausible under: a shuffle_xor mask set is
        // right exactly when its xor-closure is the set of lanes the map says
        // share the axis being folded away. Masks 1,2 for rows (`row_of`
        // ignores lane % 4), 4,8,16 for columns (`col_of` ignores lane / 4),
        // all five for the whole tile.
        for lane in 0..32u32 {
            assert_eq!(lanes_sharing_rows(lane), xor_group(lane, &[1, 2]));
            assert_eq!(lanes_sharing_columns(lane), xor_group(lane, &[4, 8, 16]));
            assert_eq!(
                xor_group(lane, &[1, 2, 4, 8, 16]),
                (0..32).collect::<Vec<_>>()
            );
        }
        // And that those groups are complete: a quad's values cover every
        // column of a row exactly once, a column group's slots every row.
        let mut columns: Vec<u32> = xor_group(0, &[1, 2])
            .into_iter()
            .flat_map(|lane| (0..Cols::VALUES).map(move |v| Cols::column(lane, v)))
            .collect();
        columns.sort();
        assert_eq!(columns, (0..32).collect::<Vec<_>>());
        let mut rows: Vec<u32> = xor_group(0, &[4, 8, 16])
            .into_iter()
            .flat_map(|lane| (0..Rows::SLOTS).map(move |s| Rows::row(lane, s)))
            .collect();
        rows.sort();
        assert_eq!(rows, (0..32).collect::<Vec<_>>());
    }

    /// The device half of a reduction, simulated: each lane's contribution
    /// combined over the lane group the *map* names, so nothing here depends
    /// on the mask constants the previous test pins.
    fn fold_over<F: Fn(f32, f32) -> f32>(
        group: &[u32],
        contribution: impl Fn(u32) -> f32,
        op: F,
    ) -> f32 {
        group
            .iter()
            .map(|&lane| contribution(lane))
            .reduce(op)
            .unwrap()
    }

    /// One lane's own registers of a row-slot, folded.
    fn lane_row_partial<Op: ReduceOp>(tile: Scores, slot: usize) -> f32 {
        (0..Scores::VALUES).fold(Op::IDENTITY, |folded, value| {
            Op::apply(folded, tile.get(slot, value))
        })
    }

    /// The column mirror of [`lane_row_partial`].
    fn lane_column_partial<Op: ReduceOp>(tile: Scores, value: usize) -> f32 {
        (0..Scores::SLOTS).fold(Op::IDENTITY, |folded, slot| {
            Op::apply(folded, tile.get(slot, value))
        })
    }

    #[test]
    fn reductions_fold_exactly_their_logical_axis() {
        // The lane-local halves are pure arithmetic and run on the host; the
        // shuffle halves are simulated by folding over the lane groups the
        // *map* names, so nothing here depends on the mask constants the
        // previous test pins. The expectation is the reduction of the
        // *logical* tile — 256*row + column over every column of a row, every
        // row of a column, every coordinate of the tile — so a fold reaching
        // the wrong registers or the wrong lanes lands on a different number.
        let value = |row: u32, column: u32| (256 * row + column) as f32;
        for lane in 0..32u32 {
            for slot in 0..Scores::SLOTS {
                let row = Scores::coordinate(lane, slot, 0).0;
                let quad = lanes_sharing_rows(lane);
                let sum = fold_over(
                    &quad,
                    |l| lane_row_partial::<Add>(coordinate_tile(l), slot),
                    |a, b| a + b,
                );
                let max = fold_over(
                    &quad,
                    |l| lane_row_partial::<Max>(coordinate_tile(l), slot),
                    fmax,
                );
                assert_eq!(sum, (0..32).map(|c| value(row, c)).sum::<f32>());
                assert_eq!(max, value(row, 31));
            }

            for v in 0..Scores::VALUES {
                let column = Scores::coordinate(lane, 0, v).1;
                let group = lanes_sharing_columns(lane);
                let sum = fold_over(
                    &group,
                    |l| lane_column_partial::<Add>(coordinate_tile(l), v),
                    |a, b| a + b,
                );
                let min = fold_over(
                    &group,
                    |l| lane_column_partial::<Min>(coordinate_tile(l), v),
                    fmin,
                );
                assert_eq!(sum, (0..32).map(|r| value(r, column)).sum::<f32>());
                assert_eq!(min, value(0, column));
            }

            let whole = fold_over(
                &(0..32).collect::<Vec<_>>(),
                |l| {
                    (0..Scores::SLOTS)
                        .map(|s| lane_row_partial::<Add>(coordinate_tile(l), s))
                        .sum::<f32>()
                },
                |a, b| a + b,
            );
            let total: f32 = (0..32)
                .flat_map(|r| (0..32).map(move |c| value(r, c)))
                .sum();
            assert_eq!(whole, total);
        }
    }

    #[test]
    fn reduce_op_identities_are_neutral() {
        // Every fold in the crate starts from `IDENTITY` and folds all of its
        // elements, so a transposed line in `reduce_ops!` is not a slow
        // reduction, it is a wrong one — `Mul` seeded from zero is zero.
        for x in [-3.5f32, -0.0, 0.0, 1.0, 2.5, 1.0e30] {
            assert_eq!(Add::apply(Add::IDENTITY, x), x);
            assert_eq!(Mul::apply(Mul::IDENTITY, x), x);
            assert_eq!(Max::apply(Max::IDENTITY, x), x);
            assert_eq!(Min::apply(Min::IDENTITY, x), x);
        }
        // And that they are the identities of the right ops, not just neutral
        // for the values above.
        assert_eq!(Add::IDENTITY, 0.0);
        assert_eq!(Mul::IDENTITY, 1.0);
        assert_eq!(Max::IDENTITY, f32::NEG_INFINITY);
        assert_eq!(Min::IDENTITY, f32::INFINITY);
    }

    #[test]
    fn scale_rows_is_the_multiply_row_map() {
        // `scale_rows` is `mul_row_assign`, so that half is trivial. It is
        // kept against the *by-value* `row_map` — the one it is not, and the
        // one that costs 87 registers/thread more at the flash width.
        for lane in 0..32 {
            let tile = coordinate_tile(lane);
            let factors = row_indices(lane);
            let mut scaled = tile;
            scaled.scale_rows(factors);
            same_tile(scaled, tile.row_map::<Mul>(factors), "row_map");
            let mut mapped = tile;
            mapped.mul_row_assign(factors);
            same_tile(scaled, mapped, "mul_row_assign");
        }
    }

    /// Assert two tiles hold the same value at every coordinate this lane owns.
    fn same_tile(left: Scores, right: Scores, what: &str) {
        for slot in 0..Scores::SLOTS {
            for value in 0..Scores::VALUES {
                assert_eq!(
                    left.get(slot, value),
                    right.get(slot, value),
                    "{what} at ({slot}, {value})"
                );
            }
        }
    }

    /// [`same_tile`] for row statistics.
    fn same_rows(left: Rows, right: Rows, what: &str) {
        for slot in 0..Rows::SLOTS {
            assert_eq!(left.get(slot), right.get(slot), "{what} at slot {slot}");
        }
    }

    /// [`same_tile`] for column statistics.
    fn same_cols(left: Cols, right: Cols, what: &str) {
        for value in 0..Cols::VALUES {
            assert_eq!(left.get(value), right.get(value), "{what} at value {value}");
        }
    }

    /// `subject.name(args)` and `subject.name_assign(args)` compute the same
    /// thing — the pairing every line of the in-place tables asserts.
    macro_rules! twins {
        ($subject:expr, $same:ident, $($name:ident / $assign:ident($($arg:expr),*);)*) => {$({
            let mut in_place = $subject;
            in_place.$assign($($arg),*);
            $same($subject.$name($($arg),*), in_place, stringify!($assign));
        })*};
    }

    /// [`twins`] for the generic maps, which name their op rather than
    /// carrying it in the method name.
    macro_rules! generic_twins {
        ($subject:expr, $same:ident, $($map:ident / $assign:ident::<$op:ty>($($arg:expr),*);)*) => {$({
            let mut in_place = $subject;
            in_place.$assign::<$op>($($arg),*);
            $same($subject.$map::<$op>($($arg),*), in_place, stringify!($assign));
        })*};
    }

    #[test]
    fn in_place_tile_maps_are_their_by_value_twins() {
        // A kernel picks between the two spellings on register pressure alone,
        // so the one thing that must never differ is the number. Every
        // generated name is here, not a sample: a transposed line in an
        // `op_methods!` table is a wrong kernel.
        let k = -0.75f32;
        for lane in 0..32 {
            let tile = coordinate_tile(lane);
            let other = coordinate_tile(31 - lane).shift(1.0);
            let rows = row_indices(lane).shift(1.0);
            let cols = column_indices(lane).shift(1.0);

            twins! { tile, same_tile,
                add / add_assign(other);
                sub / sub_assign(other);
                mul / mul_assign(other);
                div / div_assign(other);
                max / max_assign(other);
                min / min_assign(other);
                scale / scale_assign(k);
                shift / shift_assign(k);
                clamp_min / clamp_min_assign(k);
                clamp_max / clamp_max_assign(k);
                add_row / add_row_assign(rows);
                sub_row / sub_row_assign(rows);
                mul_row / mul_row_assign(rows);
                div_row / div_row_assign(rows);
                add_col / add_col_assign(cols);
                sub_col / sub_col_assign(cols);
                mul_col / mul_col_assign(cols);
                div_col / div_col_assign(cols);
                fma / fma_assign(other, tile);
            }

            generic_twins! { tile, same_tile,
                unary_map / unary_map_assign::<Exp2Approx>();
                unary_map / unary_map_assign::<Neg>();
                bin_map / bin_map_assign::<Sub>(other);
                scalar_map / scalar_map_assign::<Div>(k);
                ternary_map / ternary_map_assign::<Fma>(other, tile);
                row_map / row_map_assign::<Mul>(rows);
                col_map / col_map_assign::<Add>(cols);
            }
        }
    }

    #[test]
    fn in_place_vector_maps_are_their_by_value_twins() {
        let k = 2.5f32;
        for lane in 0..32 {
            let rows = row_indices(lane).shift(1.0);
            let other_rows = row_indices(lane).neg().shift(0.5);
            let cols = column_indices(lane).shift(1.0);
            let other_cols = column_indices(lane).neg().shift(0.5);

            twins! { rows, same_rows,
                add / add_assign(other_rows);
                sub / sub_assign(other_rows);
                mul / mul_assign(other_rows);
                div / div_assign(other_rows);
                max / max_assign(other_rows);
                min / min_assign(other_rows);
                scale / scale_assign(k);
                shift / shift_assign(k);
                clamp_min / clamp_min_assign(k);
                clamp_max / clamp_max_assign(k);
                fma / fma_assign(other_rows, rows);
            }
            generic_twins! { rows, same_rows,
                unary_map / unary_map_assign::<Exp2Approx>();
                bin_map / bin_map_assign::<Sub>(other_rows);
                scalar_map / scalar_map_assign::<Div>(k);
                ternary_map / ternary_map_assign::<Fma>(other_rows, rows);
            }

            twins! { cols, same_cols,
                add / add_assign(other_cols);
                sub / sub_assign(other_cols);
                mul / mul_assign(other_cols);
                div / div_assign(other_cols);
                max / max_assign(other_cols);
                min / min_assign(other_cols);
                scale / scale_assign(k);
                shift / shift_assign(k);
                clamp_min / clamp_min_assign(k);
                clamp_max / clamp_max_assign(k);
            }
            generic_twins! { cols, same_cols,
                unary_map / unary_map_assign::<Exp2Approx>();
                bin_map / bin_map_assign::<Sub>(other_cols);
                scalar_map / scalar_map_assign::<Div>(k);
            }
        }
    }

    #[test]
    fn splat_fills_every_owned_value() {
        let tile = Scores::splat(f32::NEG_INFINITY);
        for slot in 0..Scores::SLOTS {
            for value in 0..Scores::VALUES {
                assert_eq!(tile.get(slot, value), f32::NEG_INFINITY);
            }
        }
        for value in 0..Cols::VALUES {
            assert_eq!(Cols::splat(1.0).get(value), 1.0);
        }
    }

    /// The sentinel a masked score carries into `exp2` — finite, so a fully
    /// masked row sums to zero rather than to `NaN`.
    const MASKED: f32 = -1.0e30;

    #[test]
    fn causal_masks_against_the_bands_global_origin() {
        // The band is a [32, 64] sub-block of a much larger score matrix, so
        // what decides an element is whether its *global* key index is at or
        // before its global query index. An origin-free mask agrees with that
        // only when the two bases are equal.
        for (query_base, key_base) in [(0u32, 0u32), (32, 0), (0, 32), (128, 64), (64, 128)] {
            for lane in 0..32 {
                let unmasked: Band = indexed(lane);
                let mut band = unmasked;
                band.make_causal_at(lane, query_base, key_base, MASKED);
                for slot in 0..Band::SLOTS {
                    for value in 0..Band::VALUES {
                        let (row, column) = Band::coordinate(lane, slot, value);
                        let attends = key_base + column <= query_base + row;
                        assert_eq!(
                            band.get(slot, value),
                            if attends {
                                unmasked.get(slot, value)
                            } else {
                                MASKED
                            },
                            "({query_base}, {key_base}) lane {lane} at ({row}, {column})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_transposed_band_masks_about_the_same_diagonal() {
        // The key-parallel half of an attention backward: rows are keys and
        // columns are queries, so the surviving elements are the transpose of
        // the ones `make_causal_at` keeps for the same block pair. A CTA
        // owning keys streams the queries at and after them, so every base
        // pair here but the first has `key_base > query_base` — the sign an
        // origin-free `u32` difference cannot carry.
        for (key_base, query_base) in [(0u32, 0u32), (64, 0), (0, 64), (128, 128), (256, 64)] {
            for lane in 0..32 {
                let unmasked: Band = indexed(lane);
                let mut band = unmasked;
                band.make_causal_t_at(lane, key_base, query_base, MASKED);
                for slot in 0..Band::SLOTS {
                    for value in 0..Band::VALUES {
                        let (row, column) = Band::coordinate(lane, slot, value);
                        let attends = key_base + row <= query_base + column;
                        assert_eq!(
                            band.get(slot, value),
                            if attends {
                                unmasked.get(slot, value)
                            } else {
                                MASKED
                            },
                            "({key_base}, {query_base}) lane {lane} at ({row}, {column})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_band_off_the_diagonal_is_all_or_nothing() {
        // The two common cases in a tiled kernel, and the two the origin-free
        // signature cannot express at all. `key_base > query_base + M` is
        // wholly after every query in the band; `query_base > key_base + N`
        // wholly before every key.
        for lane in 0..32 {
            let unmasked: Band = indexed(lane);

            let mut above = unmasked;
            above.make_causal_at(lane, 0, 1024, MASKED);
            let mut below = unmasked;
            below.make_causal_at(lane, 1024, 0, MASKED);

            for slot in 0..Band::SLOTS {
                for value in 0..Band::VALUES {
                    assert_eq!(above.get(slot, value), MASKED);
                    assert_eq!(below.get(slot, value), unmasked.get(slot, value));
                }
            }
        }
        // And that the origin arrives as a signed difference: the same band
        // through the `u32` subtraction a call site would write instead
        // wraps to +4294966272 and masks nothing.
        let mut wrapped: Band = indexed(0);
        wrapped.make_causal(0, 0u32.wrapping_sub(1024) as i32, MASKED);
        assert_eq!(wrapped.get(0, 0), MASKED);
    }

    #[test]
    fn tril_and_triu_partition_the_tile() {
        // Complements about `diagonal`, at every diagonal that crosses the
        // tile and two that miss it entirely.
        for lane in 0..32 {
            for diagonal in [-64, -1, 0, 1, 17, 31, 64] {
                let full = coordinate_tile(lane);
                let mut lower = full;
                lower.tril(lane, diagonal, 0.0);
                let mut upper = full;
                upper.triu(lane, diagonal + 1, 0.0);
                let mut causal = full;
                causal.make_causal(lane, diagonal, MASKED);
                let mut transposed = full;
                transposed.make_causal_t(lane, diagonal, MASKED);
                let mut mirrored = full;
                mirrored.triu(lane, diagonal, MASKED);

                for slot in 0..Scores::SLOTS {
                    for value in 0..Scores::VALUES {
                        let (row, column) = Scores::coordinate(lane, slot, value);
                        let below = column as i32 - row as i32 <= diagonal;
                        let x = full.get(slot, value);
                        assert_eq!(lower.get(slot, value), if below { x } else { 0.0 });
                        assert_eq!(upper.get(slot, value), if below { 0.0 } else { x });
                        assert_eq!(causal.get(slot, value), if below { x } else { MASKED });
                        assert_eq!(transposed.get(slot, value), mirrored.get(slot, value));
                    }
                }
            }
        }
    }

    #[test]
    fn fills_cut_at_their_index_and_saturate_off_the_tile() {
        // Bounds outside `0..N` are the ragged tail's ordinary case — a band
        // wholly past the sequence end, or wholly inside it — so they are
        // asserted alongside the ones that cut.
        for lane in 0..32 {
            for bound in [-16, 0, 1, 9, 32, 64, 96] {
                let full: Band = indexed(lane);
                let (mut right, mut left, mut upper, mut lower) = (full, full, full, full);
                right.right_fill(lane, bound, MASKED);
                left.left_fill(lane, bound, MASKED);
                upper.upper_fill(lane, bound, MASKED);
                lower.lower_fill(lane, bound, MASKED);

                for slot in 0..Band::SLOTS {
                    for value in 0..Band::VALUES {
                        let (row, column) = Band::coordinate(lane, slot, value);
                        let (row, column) = (row as i32, column as i32);
                        let x = full.get(slot, value);
                        let cut = |keep: bool| if keep { x } else { MASKED };
                        assert_eq!(right.get(slot, value), cut(column < bound));
                        assert_eq!(left.get(slot, value), cut(column >= bound));
                        assert_eq!(upper.get(slot, value), cut(row >= bound));
                        assert_eq!(lower.get(slot, value), cut(row < bound));
                    }
                }
            }
        }
    }

    #[test]
    fn masks_are_the_predicate_over_the_layouts_coordinates() {
        // Every mask above is `mask` with a predicate, and `mask` is a select
        // at the coordinate the *layout* gives — not at `(slot, value)`. A
        // mask that indexed by storage position instead would pass every test
        // that only checks the diagonal block, so this checks the mechanism
        // directly, on a shape whose slots and values are both plural.
        for lane in 0..32 {
            let full: Band = indexed(lane);
            let mut masked = full;
            masked.mask(lane, MASKED, |row, column| (row + column) % 3 == 0);
            for slot in 0..Band::SLOTS {
                for value in 0..Band::VALUES {
                    let (row, column) = Band::coordinate(lane, slot, value);
                    assert_eq!(
                        masked.get(slot, value),
                        if (row + column) % 3 == 0 {
                            full.get(slot, value)
                        } else {
                            MASKED
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn online_rescale_matches_the_unfused_ops() {
        let mut m_ref = Rows::from_slots([-3.0f32, 0.5, 2.0, -1.0e30]);
        let row_max = Rows::from_slots([1.0f32, 0.25, 9.0, -2.0]);
        let mut running_sum = Rows::from_slots([1.0f32, 2.0, 3.0, 0.0]);
        let mut tile = Scores::zero();
        for slot in 0..Scores::SLOTS {
            tile.set(slot, 3, 1.0 + slot as f32);
        }

        let mut m_unfused = m_ref;
        let mut sum_unfused = running_sum;
        let mut tile_unfused = tile;
        let next = m_unfused.max(row_max);
        let factor = m_unfused.sub(next).exp2();
        m_unfused = next;
        sum_unfused.mul_assign(factor);
        tile_unfused.scale_rows(factor);

        online_rescale(&mut m_ref, row_max, &mut running_sum, &mut tile);
        for slot in 0..Scores::SLOTS {
            assert_eq!(m_ref.get(slot), m_unfused.get(slot));
            assert_eq!(running_sum.get(slot), sum_unfused.get(slot));
            assert_eq!(tile.get(slot, 3), tile_unfused.get(slot, 3));
        }
    }
}
