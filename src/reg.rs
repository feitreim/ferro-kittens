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
//! `mul_row`, …) are wrappers over those.
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

/// Max across the 4 lanes of a quad — how a fragment row's statistic
/// becomes whole-row (each quad's lanes hold disjoint columns of one row).
#[inline(always)]
pub fn quad_max(value: f32) -> f32 {
    let value = fmax(value, warp::shuffle_xor_f32(value, 1));
    fmax(value, warp::shuffle_xor_f32(value, 2))
}

/// Sum across the 4 lanes of a quad; see [`quad_max`].
#[inline(always)]
pub fn quad_sum(value: f32) -> f32 {
    let value = value + warp::shuffle_xor_f32(value, 1);
    value + warp::shuffle_xor_f32(value, 2)
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
    /// `1/√x` as a correctly-rounded sqrt and divide, not the SFU
    /// `rsqrt.approx.f32`. Two instructions, but the same numerics on host and
    /// device; the SFU form is a measured swap, like [`Exp2Hw`].
    Rsqrt(x) = 1.0 / x.sqrt();
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
    /// Per-thread storage, one `f32` per owned row (`[f32; SLOTS]`).
    type Slots: Copy;

    /// Rows of the `[M, _]` tile one thread owns.
    const SLOTS: usize;

    /// The logical row in `0..M` that `lane` holds in `slot`.
    fn row_of(lane: u32, slot: usize) -> u32;

    /// Every slot set to `value`.
    fn splat_slots(value: f32) -> Self::Slots;

    /// The value in `slot`.
    fn get_slot(slots: &Self::Slots, slot: usize) -> f32;

    /// Write `value` into `slot`.
    fn set_slot(slots: &mut Self::Slots, slot: usize, value: f32);
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
/// parameters, which would need `generic_const_exprs`. Each implemented
/// `(M, N)` shape names its own storage instead — one macro line per shape.
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
            type Slots = [f32; $m / 8];
            const SLOTS: usize = $m / 8;

            #[inline(always)]
            fn row_of(lane: u32, slot: usize) -> u32 {
                Self::row(lane, slot)
            }

            #[inline(always)]
            fn splat_slots(value: f32) -> Self::Slots {
                [value; $m / 8]
            }

            #[inline(always)]
            fn get_slot(slots: &Self::Slots, slot: usize) -> f32 {
                slots[slot]
            }

            #[inline(always)]
            fn set_slot(slots: &mut Self::Slots, slot: usize, value: f32) {
                slots[slot] = value;
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

/// One [`FragmentLayout`] impl per logical `(M, N)` shape, joining a
/// [`RowLayout`] to a [`ColLayout`]. Adding a shape is one line — do it when a
/// call site needs it.
macro_rules! base_ldtm_shapes {
    ($(($m:literal, $n:literal)),* $(,)?) => {$(
        impl FragmentLayout<$m, $n> for BaseLdtm {
            type Storage = [[f32; $n / 4]; $m / 8];

            #[inline(always)]
            fn splat(value: f32) -> Self::Storage {
                [[value; $n / 4]; $m / 8]
            }

            #[inline(always)]
            fn get(values: &Self::Storage, slot: usize, value: usize) -> f32 {
                values[slot][value]
            }

            #[inline(always)]
            fn set(values: &mut Self::Storage, slot: usize, value: usize, x: f32) {
                values[slot][value] = x;
            }
        }
    )*};
}

base_ldtm_rows!(16, 32);
base_ldtm_cols!(16, 32, 128);
base_ldtm_shapes!((16, 16), (32, 32), (32, 128));

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
            *self = self.bin_map::<$op>(other);
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

/// Per-thread row statistics of an `[M, _]` fragment-mapped tile: one `f32`
/// per owned row (slot), replicated across each quad. A 32-row warp tile is 4
/// slots (2 per 16-row block × 2 blocks per warp).
///
/// Every op is a compile-time-length loop over the slot array — plain
/// straight-line FMA/select code after inlining. `max`/`sub`/`exp2`/
/// `mul_assign`/`add_assign` were hand-written copies of exactly that loop
/// until `modal_app.py::regcount` showed the generic maps assemble to the same
/// registers and spills at both probe shapes; they are the maps now.
pub struct RegVec<const M: usize, L: RowLayout<M>>(pub L::Slots);

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
    pub fn from_slots(slots: L::Slots) -> Self {
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
        L::get_slot(&self.0, slot)
    }

    /// Write the statistic of row-slot `slot`.
    #[inline(always)]
    pub fn set(&mut self, slot: usize, value: f32) {
        L::set_slot(&mut self.0, slot, value);
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

    /// Quad-reduce each slot's lane-local max into a whole-row max.
    #[inline(always)]
    pub fn quad_max(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, quad_max(self.get(slot)));
            slot += 1;
        }
        out
    }

    /// Quad-reduce each slot's lane-local partial sum into a whole-row sum.
    #[inline(always)]
    pub fn quad_sum(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, quad_sum(self.get(slot)));
            slot += 1;
        }
        out
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

    unary_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
    }

    op_methods! { assign
        /// Slotwise `self *= other` — the running sum's rescale.
        mul_assign = Mul;
        /// Slotwise `self += other`.
        add_assign = Add;
    }
}

/// Per-thread column statistics of a `[_, N]` fragment-mapped tile: one `f32`
/// per owned column (value). The mirror of [`RegVec`] across the transpose,
/// and TK's `row_vec` — see the module docs on that inversion.
///
/// Under [`BaseLdtm`] a lane's columns depend only on `lane % 4`, so the 8
/// lanes of a column group hold 8 identical copies of the same `N/4` entries;
/// a whole-warp column statistic is consistent only once those 8 copies agree.
/// Nothing here produces one — the strided shuffle reduction that would is
/// issue #6. What this type does today is *carry* a column vector (splatted,
/// or built from `column`) into [`RegTile::col_map`].
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

    unary_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
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

    /// Scale every value in row-slot `s` by `factors` slot `s` — the
    /// running-max rescale of an online-softmax accumulator.
    ///
    /// Not [`Self::mul_row`], despite meaning exactly that
    /// (`scale_rows_is_the_multiply_row_map`): this form rewrites the
    /// accumulator in place, while a `row_map` builds a second tile and leaves
    /// the allocator to prove the first one dead. At the flash accumulator's
    /// width that proof does not land — `softmax_probe_128` goes 168 → 255
    /// registers/thread on the swap (`modal_app.py::regcount`). The two are
    /// the same at 32 columns; the difference is a whole tile wide.
    #[inline(always)]
    pub fn scale_rows(&mut self, factors: RegVec<M, L>) {
        let mut slot = 0;
        while slot < L::SLOTS {
            let factor = factors.get(slot);
            let mut value = 0;
            while value < L::VALUES {
                self.set(slot, value, self.get(slot, value) * factor);
                value += 1;
            }
            slot += 1;
        }
    }

    /// `Op` on every owned value.
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

    unary_op_methods!();

    op_methods! { binary
        add = Add;
        sub = Sub;
        mul = Mul;
        div = Div;
        max = Max;
        min = Min;
    }

    op_methods! { row
        add_row = Add;
        sub_row = Sub;
        /// The value form of [`Self::scale_rows`].
        mul_row = Mul;
        div_row = Div;
    }

    op_methods! { col
        add_col = Add;
        sub_col = Sub;
        mul_col = Mul;
        div_col = Div;
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

    /// A tile whose value at `(row, column)` names that coordinate exactly, so
    /// a map that reads the wrong operand shows up as a wrong coordinate.
    fn coordinate_tile(lane: u32) -> Scores {
        let mut tile = Scores::zero();
        for slot in 0..Scores::SLOTS {
            for value in 0..Scores::VALUES {
                let (row, column) = Scores::coordinate(lane, slot, value);
                tile.set(slot, value, (256 * row + column) as f32);
            }
        }
        tile
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
    fn base_ldtm_covers_each_coordinate_once() {
        covers_each_coordinate_once::<16, 16, BaseLdtm>();
        covers_each_coordinate_once::<32, 32, BaseLdtm>();
        covers_each_coordinate_once::<32, 128, BaseLdtm>();
        // Slots follow the rows a thread owns: two per 16-row block.
        assert_eq!(Fragment::SLOTS, 2);
        assert_eq!(Fragment::VALUES, 4);
        assert_eq!(RegTile::<32, 128, BaseLdtm>::SLOTS, 4);
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

    #[test]
    fn scale_rows_is_the_multiply_row_map() {
        // `scale_rows` stays in place because the by-value `row_map` costs 87
        // registers/thread at the flash width; this is what says the two
        // nevertheless mean the same thing.
        for lane in 0..32 {
            let tile = coordinate_tile(lane);
            let factors = row_indices(lane);
            let mut scaled = tile;
            scaled.scale_rows(factors);
            let mapped = tile.row_map::<Mul>(factors);
            for slot in 0..Scores::SLOTS {
                for value in 0..Scores::VALUES {
                    assert_eq!(scaled.get(slot, value), mapped.get(slot, value));
                }
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
