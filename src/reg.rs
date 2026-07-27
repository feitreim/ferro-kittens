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
//! polynomial; swapping to the SFU is a separate, measured change.

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
/// Once #6 can produce one, a
/// column vector is that per-lane copy.
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

/// Per-thread row statistics of an `[M, _]` fragment-mapped tile: one `f32`
/// per owned row (slot), replicated across each quad. A 32-row warp tile is 4
/// slots (2 per 16-row block × 2 blocks per warp).
///
/// Every op is a compile-time-length loop over the slot array — plain
/// straight-line FMA/select code after inlining, nothing the register
/// allocator can see through less clearly than the hand-written form.
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

    /// Slotwise max with `other`.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, fmax(self.get(slot), other.get(slot)));
            slot += 1;
        }
        out
    }

    /// Slotwise `self - other`. A plain method rather than `ops::Sub` so
    /// every op the device code takes stays a direct `#[inline(always)]`
    /// call.
    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, self.get(slot) - other.get(slot));
            slot += 1;
        }
        out
    }

    /// Slotwise software `2^x` ([`exp2_approx`]).
    #[inline(always)]
    pub fn exp2(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < L::SLOTS {
            out.set(slot, exp2_approx(self.get(slot)));
            slot += 1;
        }
        out
    }

    /// Slotwise `self *= other`.
    #[inline(always)]
    pub fn mul_assign(&mut self, other: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, self.get(slot) * other.get(slot));
            slot += 1;
        }
    }

    /// Slotwise `self += other`.
    #[inline(always)]
    pub fn add_assign(&mut self, other: Self) {
        let mut slot = 0;
        while slot < L::SLOTS {
            self.set(slot, self.get(slot) + other.get(slot));
            slot += 1;
        }
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
}

/// One correction step of the online softmax, in the exact per-slot order of
/// the hand-written kernels: advance `m_ref` to cover `row_max`, and rescale
/// `running_sum` and `out_acc` into the new reference. Fused on purpose —
/// one scalar `next`/`factor` live at a time, each row's values rescaled
/// before the next row's factor is formed. The unfused form (`max`/`sub`/
/// `exp2`/`scale_rows`) keeps two full vectors live across the accumulator
/// scaling and measurably costs registers in register-tight kernels
/// (persistent forward: 206 → 212 regs/thread on B200).
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
