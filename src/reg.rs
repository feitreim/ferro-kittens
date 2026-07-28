//! Register vectors/tiles over a fragment ownership map, plus the scalar maps
//! they compose with.
//!
//! A [`RegTile<M, N, L>`](RegTile) is a *logical* `[M, N]` fp32 tile spread
//! across the 32 lanes of a warp. The layout `L` owns the
//! `(lane, slot, value) -> (row, column)` map and the per-thread storage that
//! map implies, so ops are written against the logical shape and never spell
//! the map by hand. [`BaseLdtm`] — the tcgen05 `16x256b` drain the validated
//! kernels use — is the crate's only layout, and its doc carries the ownership
//! contract every op here assumes.
//!
//! `exp2` exists twice —
//! [`exp2_approx`] (the FMA polynomial, bit-identical to what the flash
//! kernels shipped with) and [`exp2_hw`] (one `ex2.approx` SFU instruction,
//! also pure-PTX-safe post-#56). Ports that must hold "same SASS" keep the
//! polynomial; swapping to the SFU is a separate, measured change — and the
//! measurement does not favour it: pointing `exp2` at the SFU takes
//! `softmax_probe_128` from 168 registers and no spill to 255 and 112 bytes
//! of spill stores (`modal_app.py::regcount`). One instruction, but the FMA
//! chain schedules where `ex2.approx` serializes.
//!
//! Elementwise work goes through [`UnaryOp`] / [`BinaryOp`] / [`TernaryOp`]
//! and the `*_map` methods, so a scalar function is written once and reaches
//! [`RegTile`], [`RegVec`] and [`ColVec`] alike. The named methods (`exp2`,
//! `mul_row`, `scale`, …) are wrappers over those.
//!
//! Every map comes in two spellings: by value, and in place through
//! `&mut self` with an `_assign` suffix (#31). They compute the same thing —
//! `in_place_tile_maps_are_their_by_value_twins` is what says so — and differ
//! only in what the register allocator has to hold. Which to write is a
//! measured question with a surprising answer; [`RegTile::bin_map_assign`]
//! carries the table and the rule that falls out of it.
//!
//! A *scalar* operand is a [`BinaryOp`] too — `scalar_map::<Mul>(k)` is
//! `scale`, `scalar_map::<Add>(k)` is `shift` — rather than a stateful op
//! trait, because a `UnaryOp` is a unit struct with an associated function and
//! has nowhere to keep a `k`. The consequence is that every tile-against-scalar
//! form the op set can express is reachable without a new op.
//!
//! Masking ([`RegTile::make_causal`], `tril`/`triu`, the fills) is the other
//! use of the ownership map: a select at each value's logical `(row, column)`,
//! in place, with no lane learning anything from another. Every one of them
//! takes a **coordinate origin** — a `diagonal`, or a fill index — because the
//! tile being masked is a sub-block of a much larger score matrix and its
//! diagonal sits at `query_base - key_base`. Masking a tile against its own
//! `row == column` instead compiles, runs, and is wrong everywhere off the
//! diagonal block; see [`RegTile::make_causal_at`].
//!
//! Reductions take the same [`BinaryOp`] and come in two halves: a thread's own
//! registers, then a `shuffle_xor` butterfly over the lanes the map spreads the
//! folded axis across. Which lanes those are is the entire correctness
//! question, and it is decided by [`BaseLdtm`]'s two maps — `row_of` ignores
//! `lane % 4`, `col_of` ignores `lane / 4` — so a row reduction shuffles masks
//! 1 and 2, a column reduction 4, 8 and 16, and a whole-tile reduction all
//! five. A wrong mask there yields a plausible wrong number rather than a
//! crash, which is why `reduction_masks_are_the_ownership_maps_lane_groups`
//! derives the groups from the maps instead of restating the constants.
//! All of it is **warp scope**; nothing here makes several warps agree.
//!
//! **ThunderKittens naming.** TK names a register vector for the axis it
//! *spans*: its `col_vec` has one entry per row, its `row_vec` one per column.
//! We name for the axis that *indexes* it, so TK's `col_vec` is our
//! [`RegVec`] and TK's `row_vec` is our [`ColVec`] — inverted. The op names
//! do agree: TK's `row_map`/`mul_row` and ours both broadcast a per-row scalar
//! along each row.

use cuda_device::warp;

/// NaN-free float max. The comparison-select lowering is retained for its
/// established SASS; libdevice-backed `f32::max` is now artifact-safe.
#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// NaN-free float min; see [`fmax`].
#[inline(always)]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `2^x` on FMA units: round-to-nearest split via the 1.5·2²³ shift trick,
/// exponent-bit insertion for the integer part, and a degree-3 minimax
/// polynomial (max relative error 7.5e-5 on the reduced range) for the
/// fraction. The clamp keeps the exponent field in the normal range and
/// flushes masked-sentinel inputs to a harmless ~2^-125.
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

/// `2^x` as one `ex2.approx.f32` SFU instruction — FA4's SFU offload.
/// Different rounding than [`exp2_approx`]; adopting it in a gated kernel is
/// a numerics change, not a refactor.
#[inline(always)]
pub fn exp2_hw(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x)
}

/// `log2(x)` for positive normal `x`: exponent extraction, mantissa
/// renormalized to `[√½, √2]`, then the atanh series in `t = (m-1)/(m+1)`
/// (four terms; |error| < 5e-8 on the reduced range). The coefficient
/// literals are bit-exact copies of the validated kernel's.
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
/// `rsqrt.approx.f32`: two instructions, but the same number on host and device,
/// which is what lets a normalization kernel's host reference be `==`. Free
/// beside [`exp2_approx`] because a scalar variance — layernorm's, over a
/// statistic that is already one `f32` — has no vector to reach [`Rsqrt`] with.
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Fold across the 4 lanes of a quad — the lanes differing only in `lane % 4`,
/// which under [`BaseLdtm`] is the axis one row's columns are spread along
/// (`row_of` ignores `lane % 4` entirely). So this is the second half of a row
/// reduction: the first is folding a thread's own `VALUES` registers.
///
/// The result lands in all four lanes, which is what makes a [`RegVec`] a
/// whole-row statistic rather than a partial one.
/// `reduction_masks_are_the_ownership_maps_lane_groups` is what pins masks
/// 1 and 2 to that claim.
#[inline(always)]
pub fn quad_reduce<Op: ReduceOp>(value: f32) -> f32 {
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 1));
    Op::apply(value, warp::shuffle_xor_f32(value, 2))
}

/// Fold across the 8 lanes sharing a `lane % 4` — the lanes differing only in
/// `lane / 4`, which under [`BaseLdtm`] is the axis one column's rows are
/// spread along (`col_of` ignores `lane / 4`). The second half of a column
/// reduction, and three shuffles rather than [`quad_reduce`]'s two: this is
/// the concrete sense in which a column reduction is a different shuffle
/// rather than a reparameterization of the row one.
#[inline(always)]
pub fn column_group_reduce<Op: ReduceOp>(value: f32) -> f32 {
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 4));
    let value = Op::apply(value, warp::shuffle_xor_f32(value, 8));
    Op::apply(value, warp::shuffle_xor_f32(value, 16))
}

/// Fold across all 32 lanes — both axes, the full butterfly, leaving the
/// result warp-uniform.
#[inline(always)]
pub fn warp_reduce<Op: ReduceOp>(value: f32) -> f32 {
    column_group_reduce::<Op>(quad_reduce::<Op>(value))
}

/// Max across the 4 lanes of a quad — how a fragment row's statistic
/// becomes whole-row (each quad's lanes hold disjoint columns of one row).
#[inline(always)]
pub fn quad_max(value: f32) -> f32 {
    quad_reduce::<Max>(value)
}

/// Sum across the 4 lanes of a quad; see [`quad_max`].
#[inline(always)]
pub fn quad_sum(value: f32) -> f32 {
    quad_reduce::<Add>(value)
}

/// A scalar function named as a *type*, so one definition instantiates for
/// every register family through the `unary_map` methods. Unit structs rather
/// than `fn` pointers or closures: a type parameter can carry no state to
/// spill and nothing for the inliner to see through.
pub trait UnaryOp {
    /// The scalar function.
    fn apply(x: f32) -> f32;
}

/// Two-operand [`UnaryOp`]. Also what the row/column broadcast maps take, with
/// `b` the per-row or per-column scalar.
pub trait BinaryOp {
    /// The scalar function.
    fn apply(a: f32, b: f32) -> f32;
}

/// Three-operand [`UnaryOp`] — the fused multiply-add family, which exists
/// because a separate multiply and add do not contract on their own.
pub trait TernaryOp {
    /// The scalar function.
    fn apply(a: f32, b: f32, c: f32) -> f32;
}

/// A [`BinaryOp`] a reduction may fold with: associative and commutative — the
/// fragment map hands a fold its operands in the layout's order, not the
/// tile's — and carrying an identity to seed from. `Sub` and `Div` are
/// deliberately not members.
///
/// The bound is narrower than [`BinaryOp`] on purpose: `row_reduce::<Sub>` has
/// no meaning worth giving it a spelling, and the identity lets every fold
/// start the same way instead of special-casing element zero. It costs one
/// extra `apply`, which the FMA-seeded forms fold away
/// (`Max::apply(-inf, x)` is `x` by construction) and the others do not —
/// measured at no register cost either way, once the fold is written inline
/// (see the note above the reductions on `RegTile`).
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
    /// The FMA polynomial ([`exp2_approx`]) — what the validated kernels ship.
    Exp2Approx(x) = exp2_approx(x);
    /// The SFU instruction ([`exp2_hw`]). Rounds differently from
    /// [`Exp2Approx`]; the choice between them is a numerics decision.
    Exp2Hw(x) = exp2_hw(x);
    /// `e^x` on [`Exp2Approx`], inheriting that polynomial's error bound and
    /// its ±125 exponent clamp — so this saturates at `x = ±86.6`, just inside
    /// where fp32 overflows anyway.
    Exp(x) = exp2_approx(x * core::f32::consts::LOG2_E);
    Log2(x) = log2_approx(x);
    /// `ln(x)` on [`log2_approx`]; see [`Exp`].
    Log(x) = log2_approx(x) * core::f32::consts::LN_2;
    /// Sign-bit clear rather than a libdevice `fabsf` — one `abs.f32`, and it
    /// keeps the op usable in a pure-PTX artifact regardless of #56's fate.
    Abs(x) = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    Neg(x) = -x;
    Relu(x) = fmax(x, 0.0);
    /// `llvm.sqrt.f32`, which NVPTX lowers to the native `sqrt.rn.f32` with no
    /// libdevice call.
    Sqrt(x) = x.sqrt();
    /// [`rsqrt`] — correctly rounded, not the SFU `rsqrt.approx.f32`, whose
    /// adoption would be a measured swap like [`Exp2Hw`]'s.
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
    // `1.0` and not `0.0`: a product folded from an additive identity is zero.
    Mul = 1.0;
    Max = f32::NEG_INFINITY;
    Min = f32::INFINITY;
}

scalar_ops! { TernaryOp:
    /// `a*b + c` in one `fma.rn.f32`. Rust emits no fast-math flags, so a
    /// separate multiply and add stay separate — the fusion has to be asked
    /// for. TK's `fma_AxCtB` is this op with the last two operands swapped at
    /// the call site; our maps fix no operand to a broadcast vector, so the
    /// second form buys nothing.
    Fma(a, b, c) = a.mul_add(b, c);
}

/// The row half of a fragment ownership map: how a warp's 32 lanes divide the
/// `M` logical rows of a tile into per-thread *slots*, and where the one
/// `f32` per owned row lives.
///
/// Split out of [`FragmentLayout`] so a [`RegVec`] can name a row count
/// without inventing a column count, and so `scale_rows` can check a vector
/// and a tile against the *same* `M`.
pub trait RowLayout<const M: usize> {
    /// Per-thread storage, one `T` per owned row (`[T; SLOTS]`).
    ///
    /// Generic in the element, and that is what opens the shape set: a tile's
    /// storage is this array of a [`ColLayout::Values`], so
    /// [`FragmentLayout::Storage`] is a *projection* out of the two extents
    /// rather than an array whose length is `M / 8` — a length no impl can
    /// compute from a generic `M` without `generic_const_exprs`.
    type Slots<T: Copy>: Copy;

    /// Rows of the `[M, _]` tile one thread owns.
    const SLOTS: usize;

    /// The logical row in `0..M` that `lane` holds in `slot`.
    fn row_of(lane: u32, slot: usize) -> u32;

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
/// Unlike a [`RowLayout`] slot, a value is *not* warp-uniform in the same
/// sense: under [`BaseLdtm`] a column depends only on `lane % 4`, so the 8
/// lanes of a column group each hold their own copy of the same `N/4` columns.
/// A [`ColVec`] is that per-lane copy.
pub trait ColLayout<const N: usize> {
    /// Per-thread storage, one `f32` per owned column (`[f32; VALUES]`).
    type Values: Copy;

    /// Values one thread owns per slot — the columns of `0..N` it holds.
    const VALUES: usize;

    /// How many consecutive values land on consecutive columns, so that a run
    /// of them is one vector memory access rather than that many scalar ones.
    ///
    /// A run starts at every multiple of the constant, and the contract is
    /// both halves of what a vector access needs — that the addresses are
    /// adjacent, and that the first of them is aligned for the width:
    ///
    /// ```text
    /// run % CONTIGUOUS_VALUES == 0  ⟹  col_of(lane, run + i) == col_of(lane, run) + i
    ///                                   for every i < CONTIGUOUS_VALUES,
    ///                                   and CONTIGUOUS_VALUES divides both
    ///                                   col_of(lane, run) and VALUES.
    /// ```
    ///
    /// The default is `1` — no two values adjacent — because that is the
    /// answer that is true of every map, so a layout written later gets scalar
    /// accesses until it claims otherwise rather than silently inheriting
    /// [`BaseLdtm`]'s arithmetic (#23, #91). Raising it is a claim about the
    /// map that [`crate::global::store_rows`] and
    /// [`crate::global::load_rows`] act on; `base_ldtm_pairs_every_other_value`
    /// is what checks it for the one layout that does.
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
/// The storage is an associated type rather than `[[f32; VALUES]; SLOTS]`
/// because an array length must be a const expression of the generic
/// parameters, which would need `generic_const_exprs`. It is nevertheless
/// *derived*: the blanket impl below projects it as `Slots<Values>`, so no
/// `(M, N)` needs an impl of its own and the shape set is the product of the
/// two extent sets rather than a list of pairs (#23).
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
/// of values are its row array of its column array, which is the same
/// `[[f32; VALUES]; SLOTS]` a per-shape impl would have written and needs no
/// arithmetic on `M` and `N` to name.
///
/// Blanket, so a shape costs no line anywhere: adding a row extent and a column
/// extent adds every tile between them, and a layout defined *outside* this
/// crate is a tile layout as soon as it has both halves — which is the part
/// orphan rules put out of reach for [`BaseLdtm`] (#23). The consequence is
/// that `FragmentLayout` is never implemented directly, here or downstream.
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

/// The base-LDTM `16x256b` ownership map — the only drain shape the validated
/// kernels use, and the crate's only layout.
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
/// `M` and `N` are per *warp* and both multiples of 16, and across the warp's
/// 32 lanes the map covers each `(row, column)` of the tile exactly once
/// (`base_ldtm_covers_each_coordinate_once`). Note that slots count the rows a
/// thread owns, not a warpgroup's rows: the flash output accumulator is
/// `RegTile<32, 128, BaseLdtm>` — one warp's 32 TMEM rows by 128 columns, four
/// slots of 32 values in each thread.
///
/// Row statistics live once per owned row, replicated across the 4 lanes of a
/// quad by shuffle reductions ([`quad_max`], [`quad_sum`]).
pub struct BaseLdtm;

impl BaseLdtm {
    /// The logical row `lane` holds in `slot` — the `lane/4` row of the
    /// slot's 16-row block, or its `+8` twin for odd slots.
    #[inline(always)]
    pub const fn row(lane: u32, slot: usize) -> u32 {
        16 * (slot as u32 / 2) + 8 * (slot as u32 % 2) + lane / 4
    }

    /// The logical column `lane` holds in `value`: fours at offsets
    /// `{0, 1, 8, 9}` of successive 16-column blocks, from the lane's own
    /// column pair. The coordinate a masking or per-column-statistic pass
    /// needs, and the inverse of the packing
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
            // `{0, 1, 8, 9}` is two adjacent pairs, from an even base
            // (`2*(lane%4) + 16*(value/4)`). Two and not four: the run breaks
            // at the `1 -> 8` step.
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
// impls, since `FragmentLayout` is the product of the two.
//
// The bound is the register file, not a guess at what kernels want: a thread
// holds `M * N / 32` fp32 values of an `[M, N]` warp tile, so with the other
// extent at its 16 minimum, 512 is already 256 registers a thread — one past
// what the hardware has. No shape outside this grid fits in registers at all,
// which is the sense in which the set is open rather than merely bigger.
// Unused extents cost nothing: a trait impl no tile names emits no code.
base_ldtm_rows!(
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
);
base_ldtm_cols!(
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
);

/// The named half of the op set: one line per exposed name, so a new op costs
/// a [`scalar_ops`] line and one of these. `should_implement_trait` is allowed
/// wholesale because every op the device code takes must stay a direct
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
            /// Software `2^x` ([`exp2_approx`]) — the shipped kernels'
            /// rounding, and what every `exp2` in the crate resolves to.
            exp2 = Exp2Approx;
            /// SFU `2^x` ([`exp2_hw`]); a different result from `exp2`.
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

/// The scalar-operand names every register family carries. The generic
/// [`RegTile::scalar_map`] already reaches every [`BinaryOp`] — `Div`, `Sub`
/// and the rest need no wrapper to be callable — so this table holds only the
/// spellings a kernel would otherwise invent a worse name for.
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

/// The in-place twin of [`scalar_op_methods`], name for name. A separate table
/// rather than a second name generated by the first, so that adding a scalar op
/// is still one line and the two tables can be read against each other.
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
/// Every op is a compile-time-length loop over the slot array — plain
/// straight-line FMA/select code after inlining. `max`/`sub`/`exp2`/
/// `mul_assign`/`add_assign` were hand-written copies of exactly that loop
/// until `modal_app.py::regcount` showed the generic maps assemble to the same
/// registers and spills at both probe shapes; they are the maps now.
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

    /// Wrap this thread's slots. Named rather than the tuple constructor
    /// because a type alias (`Fragment`-style) can't spell one.
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
    /// predicate (this lane's vote only; the warp/warpgroup OR is the
    /// caller's collective step).
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
    /// folding across the quad ([`quad_reduce`]) — the second half of
    /// [`RegTile::row_reduce`], and the half a caller holding its own
    /// partials (a running softmax sum, say) is the one that needs.
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
    /// tile's other rows.
    ///
    /// Only meaningful on a vector that is already a *whole*-row statistic —
    /// one replicated across each quad, which is what
    /// [`RegTile::row_reduce`] returns and what a lane-local partial is not.
    /// [`RegTile::tile_reduce`] gets the same answer in five shuffles instead
    /// of `2 * SLOTS + 3`; this exists for the case where the row vector is
    /// wanted anyway.
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

    /// `Op` on every slot, rewriting this vector; see
    /// [`RegTile::unary_map_assign`].
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

    /// `Op` slotwise against one scalar; see [`RegTile::scalar_map`].
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

    /// `Op` slotwise across three vectors — [`Fma`] and nothing else so far.
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
        /// Slotwise `self *= other` — the running sum's rescale.
        mul_assign = Mul;
        div_assign = Div;
        max_assign = Max;
        min_assign = Min;
    }
}

/// Per-thread column statistics of a `[_, N]` fragment-mapped tile: one `f32`
/// per owned column (value). The mirror of [`RegVec`] across the transpose,
/// and TK's `row_vec` — see the module docs on that inversion.
///
/// Under [`BaseLdtm`] a lane's columns depend only on `lane % 4`, so the 8
/// lanes of a column group hold 8 copies of the same `N/4` entries; a
/// whole-warp column statistic is consistent only once those copies agree.
/// [`RegTile::col_reduce`] is what makes them agree — it folds across those 8
/// lanes ([`column_group_reduce`]) and so returns a vector every lane of the
/// group reads the same way. A vector built any other way (splatted, or from
/// [`Self::column`]) is a legitimate `col_map` operand but carries no such
/// guarantee, and [`Self::reduce`] is only meaningful on one that does.
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

    /// Wrap this thread's values; see [`RegVec::from_slots`].
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
    /// tile's other columns. The mirror of [`RegVec::reduce`], and subject to
    /// the same precondition — the vector must already hold whole-column
    /// statistics, i.e. the 8 copies must agree, which is what
    /// [`RegTile::col_reduce`] establishes.
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
    /// by folding across the column group ([`column_group_reduce`]) — the
    /// mirror of [`RegVec::quad_reduce`], and the step that makes the 8 copies
    /// of a [`ColVec`] agree.
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
pub struct RegTile<const M: usize, const N: usize, L: FragmentLayout<M, N>>(pub L::Storage);

impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> Clone for RegTile<M, N, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> Copy for RegTile<M, N, L> {}

/// The `[16, 16]` tile one [`crate::tmem::TmemTile`] drain returns: the two
/// rows a thread owns in a 16-row block, times the four values it owns in a
/// 16-column block. Every register pass over a TMEM accumulator is a loop over
/// these.
pub type Fragment = RegTile<16, 16, BaseLdtm>;

impl<const M: usize, const N: usize, L: FragmentLayout<M, N>> RegTile<M, N, L> {
    /// Rows of the tile this thread owns.
    pub const SLOTS: usize = L::SLOTS;
    /// Values this thread owns per row.
    pub const VALUES: usize = L::VALUES;

    /// Wrap this thread's values, slot-major. Named rather than the tuple
    /// constructor because a type alias ([`Fragment`]) can't spell one.
    #[inline(always)]
    pub fn from_values(values: L::Storage) -> Self {
        Self(values)
    }

    /// The additive identity — a fresh accumulator.
    #[inline(always)]
    pub fn zero() -> Self {
        Self(L::splat(0.0))
    }

    /// Every value set to `value`. TK's nullary `one`/`pos_infty`/`neg_infty`
    /// ops are this with the constant written out.
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
    /// Lane-local: the map hands a thread whole `(row, column)` pairs, so a
    /// mask is a select on registers it already holds and no lane learns
    /// anything from another. In place because a mask is the innermost thing
    /// in an attention loop and its input is dead the instant it returns; the
    /// by-value spelling would put a second band beside the score band at
    /// exactly the width #5 and #38 measured that to cost.
    ///
    /// Coordinates are `i32`, not `u32`: a tiled kernel's bounds are
    /// differences of block origins and leave `0..M × 0..N` in both
    /// directions. That is not an edge case — it is how a band wholly above
    /// the diagonal takes `keep` false everywhere and one wholly below takes
    /// it true everywhere, which is most of the bands in a causal kernel.
    #[inline(always)]
    pub fn mask(&mut self, lane: u32, fill: f32, keep: impl Fn(i32, i32) -> bool) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let row = L::row_of(lane, slot) as i32;
            let mut value = 0;
            while value < L::VALUES {
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
    /// `diagonal` is where the boundary crosses this tile's own `(0, 0)`, so
    /// a tile that is a sub-block of a larger matrix passes the difference of
    /// its origins and gets the larger matrix's diagonal. TK's `tril` has no
    /// such parameter because it masks a tile against itself, which is right
    /// for exactly one block of a tiled kernel and silently wrong for the
    /// rest; see [`Self::make_causal_at`].
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
    /// already holds instead of their difference: the band covers queries
    /// `query_base..query_base + M` against keys `key_base..key_base + N`, and
    /// its diagonal sits at `query_base - key_base`.
    ///
    /// Taking them separately is not sugar. The difference is negative for
    /// every band above the diagonal — the fully-masked ones — and both
    /// origins are `u32` at the call site, so `query_base - key_base` written
    /// there wraps to a huge positive number and masks nothing. This subtracts
    /// in `i32`.
    #[inline(always)]
    pub fn make_causal_at(&mut self, lane: u32, query_base: u32, key_base: u32, fill: f32) {
        self.make_causal(lane, query_base as i32 - key_base as i32, fill);
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
    /// A wrapper as of #31, and it was hand-written until then for a reason
    /// worth keeping. It is `mul_row` by definition
    /// (`scale_rows_is_the_multiply_row_map`), but the *by-value* `mul_row`
    /// builds a second tile and leaves the allocator to prove the first one
    /// dead, and at the flash accumulator's width that proof does not land:
    /// `softmax_probe_128` goes 168 → 255 registers/thread on that swap
    /// (`modal_app.py::regcount`). What it may safely become is the *in-place*
    /// map, which measures identically to this loop — 64 registers at 32
    /// columns and 168 at 128, same spills, same stack frame — which is the
    /// bar #31 set for deleting the hand-written body.
    #[inline(always)]
    pub fn scale_rows(&mut self, factors: RegVec<M, L>) {
        self.row_map_assign::<Mul>(factors);
    }

    /// `Op` on every owned value, rewriting this tile.
    ///
    /// The in-place half of the map mechanism (#31). Each by-value map below
    /// computes exactly this into a fresh copy — the copy is the whole of the
    /// difference, and it is the difference that costs. See
    /// [`Self::bin_map_assign`] for when a call site should prefer which.
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

    /// `Op` on every owned value.
    ///
    /// The loop is written out rather than delegating to
    /// [`Self::unary_map_assign`] on a copy, and the same goes for every
    /// by-value map below. The two spell the same arithmetic and the
    /// delegating form is the obvious factoring, but it is not free: routing
    /// the by-value maps through their in-place twins moved *every* probe in
    /// `regcount` that uses a map, in both directions — `mask_probe_128_causal`
    /// 71 → 32 registers, `softmax_probe_32_hand_written` 64 → 80, and
    /// `lane_probe_128_hoisted` 168 → 128 registers while its stack frame grew
    /// 512 bytes. Same reason the reductions below spell their folds out; the
    /// duplication is held honest by
    /// `in_place_tile_maps_are_their_by_value_twins`, not by sharing a body.
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
    /// **When this is worth writing instead of [`Self::bin_map`].** Not
    /// always, and the reason is not the one #31 was filed under. Measured on
    /// `scalar_map_probe_128` and `softmax_probe_128` (`regcount`, sm_100a),
    /// what orders the spellings of one `[32, 128]` step is how many whole
    /// bands have to be *materialized between statements* — which the calling
    /// convention only correlates with:
    ///
    /// ```text
    /// out_acc = out_acc.scale(k).add(block.scale(k))   168 regs, no spill
    /// out_acc.scale_assign(k); out_acc.add_assign(b)   252 regs, no spill
    /// out_acc = out_acc.scale(k); .. = ..add(block)    255 regs,  60 B
    /// out_acc = out_acc.add(block.scale(k))            255 regs, 108 B
    /// ```
    ///
    /// So the rule at a call site is: *say the whole step in one expression if
    /// you can; write it in place if you cannot.* An in-place form is worth
    /// reaching for where the input is the output — an accumulator, or a
    /// rescale of one — which is where a single expression cannot be written
    /// and where a by-value map would otherwise cost a whole band
    /// (`row_map::<Mul>` against [`Self::mul_row_assign`]: 255 against 168).
    /// Where the by-value spelling already rebinds a dead input, it costs
    /// nothing and needs no conversion.
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

    /// `Op` against one scalar broadcast over every owned value — `k` in a
    /// register, not a tile, so the operand costs one register rather than
    /// `SLOTS * VALUES` and no [`Self::splat`] runs.
    ///
    /// A [`BinaryOp`] and not a stateful op trait: `UnaryOp::apply` is an
    /// associated function on a unit struct, so a scaling factor has nowhere to
    /// live, and making it live somewhere would put a value in a type
    /// parameter's shadow where the inliner has to rediscover it. Treating the
    /// scalar as the second operand instead means the whole existing op set —
    /// `Div` for a divide by a constant, `Max`/`Min` for a bound, `Sub` — is
    /// reachable the day it is written.
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
    /// in-place form, and the one #38 measured a hand-written loop of at 252
    /// registers and no spill where the by-value spelling spilled 60 bytes.
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

    /// `Op` across `self`, `b` and `c`, rewriting this tile. Unmeasured: no
    /// probe monomorphizes a ternary map at 128 columns, so what it saves over
    /// [`Self::ternary_map`] there is a guess and not a number.
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
    /// tile — the form [`Self::scale_rows`] is `Mul` of, and the one #31 was
    /// filed to get.
    ///
    /// One row's factor is read once and spent before the next row's is
    /// formed, exactly as in the hand-written loop, so a `RegVec` operand
    /// costs one live register rather than a second band.
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
    /// logical column [`ColVec::column`]`(lane, value)`. Also shuffle-free,
    /// for the mirror reason — but see [`ColVec`] on where the operand can
    /// legitimately come from today.
    #[inline(always)]
    pub fn col_map<Op: BinaryOp>(self, cols: ColVec<N, L>) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                out.set(
                    slot,
                    value,
                    Op::apply(self.get(slot, value), cols.get(value)),
                );
                value += 1;
            }
            slot += 1;
        }
        out
    }

    /// [`Self::col_map`] rewriting this tile — the column mirror of
    /// [`Self::row_map_assign`].
    #[inline(always)]
    pub fn col_map_assign<Op: BinaryOp>(&mut self, cols: ColVec<N, L>) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let mut value = 0;
            while value < L::VALUES {
                self.set(
                    slot,
                    value,
                    Op::apply(self.get(slot, value), cols.get(value)),
                );
                value += 1;
            }
            slot += 1;
        }
    }

    // The three reductions below each spell their lane-local fold out rather
    // than sharing one `fold(self, slot)` helper. The helper is the obvious
    // factoring and it costs a whole tile: even `#[inline(always)]`, taking
    // `self` by value materializes a second copy of the storage, and
    // `softmax_probe_32` measures 94 registers/thread against 64 for the
    // written-out loop, with `softmax_probe_128_row_map` picking up 456 bytes
    // of spill stores on the same change (`modal_app.py::regcount`). Same
    // reason `scale_rows` is `row_map_assign::<Mul>` and not `row_map::<Mul>`:
    // what costs is the copy the shared body is reached through, not the
    // sharing.

    /// Fold each row across all `N` columns: this thread's columns of the row,
    /// then [`quad_reduce`] over the quad that holds the rest of them. Two
    /// shuffles per owned row.
    ///
    /// `Op` is applied in the *layout's* order, not left to right along the
    /// row, so a non-associative one gets a well-defined but unhelpful answer.
    /// The result is a whole-row statistic replicated across each quad, which
    /// is exactly the operand [`Self::row_map`] wants.
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
    /// them. Three shuffles per owned column, and the op that makes a
    /// [`ColVec`] whole — before it, the 8 lanes of a column group hold 8
    /// independent partials of the same column.
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
    /// [`Self::row_reduce`] and [`RegVec::reduce`].
    ///
    /// Warp scope. A tile that several warps own — layernorm's group-norm
    /// statistic over four warps' bands — needs those warps to agree, which is
    /// a shared-memory staging step this returns no help with; see the crate's
    /// #3/#13 discussion.
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
        /// `self += other` — flash's output accumulator taking one key
        /// block's contribution, and the name that example was blocked on.
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
        /// The product of each row. [`Mul`] *is* the product op — a separate
        /// `Prod` would be the same `a * b` under a second name.
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
        /// binary op; TK distinguishes the two by C++ overloading, which Rust
        /// has no equivalent of.
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
        /// The in-place form of [`Self::mul_row`], and the op
        /// [`Self::scale_rows`] had to be hand-written to avoid.
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

/// One correction step of the online softmax, in the exact per-slot order of
/// the hand-written kernels: advance `m_ref` to cover `row_max`, and rescale
/// `running_sum` and `out_acc` into the new reference. Fused on purpose —
/// one scalar `next`/`factor` live at a time, each row's values rescaled
/// before the next row's factor is formed. The unfused form (`max`/`sub`/
/// `exp2`/`scale_rows`) keeps two full vectors live across the accumulator
/// scaling and measurably costs registers in register-tight kernels
/// (persistent forward: 206 → 212 regs/thread on B200).
///
/// `softmax_probe_32` reproduces that direction at 56 vs 64 regs/thread and
/// `softmax_probe_128` does not reproduce it at all (168 either way) — the
/// probe holds its accumulator in registers, where the real kernel holds it in
/// TMEM and drains, so the probe bounds the fusion's worth but does not
/// license undoing it. The 206 → 212 number stands until a kernel of that
/// shape says otherwise.
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

    /// The flash score band, and the shape #23 was filed about.
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
        // #23. `FragmentLayout` is a blanket impl over `RowLayout × ColLayout`
        // now, so a shape costs no line of its own — and the storage it
        // projects is still exactly the `[[f32; N/4]; M/8]` the per-shape
        // impls named, which is what says the change is a spelling and not a
        // representation.
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
    /// a run-aligned column. That is exactly what `store_rows` and `load_rows`
    /// turn into a vector access, and the reason the constant is a property of
    /// the map and not of the mover (#91).
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
        // A TMEM drain only ever returns `Fragment`s, so every tile wider or
        // taller than [16, 16] is assembled by placing block (row_block,
        // column_block)'s (slot, value) at (2*row_block + slot, 4*column_block
        // + value). That composition is spelled by hand in kernel drain loops
        // and by the device harness; this is the assertion that it is the same
        // map the bigger shape's own `coordinate` gives.
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

    /// One lane's own registers of a row-slot, folded. The reductions spell
    /// this out inline rather than share it — see the note in `RegTile` on
    /// what the shared helper cost — so the host test carries its own copy.
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
        // `scale_rows` is `mul_row_assign` now and the assertion is trivial on
        // that pair — it is kept against the *by-value* `row_map`, which is
        // the one it is still not, and which costs 87 registers/thread at the
        // flash width for being a different program rather than a different
        // spelling.
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
        // #31. The in-place forms exist for register pressure and a kernel
        // picks between the two spellings on that basis alone, so the one
        // thing that must never differ is the number. Every generated name is
        // here, not a sample: a transposed line in an `op_methods!` table is a
        // wrong kernel, and the by-value tables are only pinned to their ops
        // one test above.
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
        // The whole of #7's correction. The band is a [32, 64] sub-block of a
        // much larger score matrix, so what decides an element is whether its
        // *global* key index is at or before its global query index — and the
        // origin-free mask agrees with that only when the two bases are equal,
        // which is one block of a tiled kernel and no others.
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
