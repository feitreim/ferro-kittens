//! Chained tcgen05 MMA walks over shared-tile operands.
//!
//! tcgen05's native step is `D (+)= A·Bᵀ` over a K=16 chunk; a tile multiply is
//! a chain of those steps, every one after the first linked into the
//! accumulator. Each walk names its operand layout in its types and builds its
//! own descriptor from that layout, so a caller cannot pair a walk with a
//! descriptor that disagrees — which does not fault, it fills the accumulator
//! with wrong numbers.
//!
//! | Walk | `A` | `B` | transpose `(a, b)` |
//! | --- | --- | --- | --- |
//! | [`mma_abt`] | `[M, K]` | `[N, K]` | `(false, false)` |
//! | [`mma_ab`] | `[M, K]` | `[K, N]` | `(false, true)` |
//! | [`mma_atb`] | `[K, M]` | `[K, N]` | `(true, true)` |
//! | [`mma_atbt`] | `[K, M]` | `[N, K]` | `(true, false)` |
//!
//! Each has an [`mm_abt`]-style twin that *replaces* the accumulator rather than
//! adding to it, and [`mma_walk_cg2`] is the cluster-pair form, taking
//! [`OperandWalk`] values so a kernel choosing K-major vs MN-major at runtime
//! issues one loop either way. Operands are generic over their [`MmaElement`],
//! which also routes tcgen05's `KIND`; the chunk geometry is still 2-byte-only
//! and asserted as such at every walk that depends on it.
//!
//! A K loop, published to its consumers once:
//!
//! ```no_run
//! # use kittens::mma::{MmaShape, commit, mma_abt};
//! # use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
//! # use kittens::Semaphore;
//! # unsafe fn demo(
//! #     accumulator: u32,
//! #     a: SharedTile<Bf16, 128, 64, Swizzle128B>,
//! #     b_ring: SharedTileRing<Bf16, 64, 64, Swizzle128B, 3>,
//! #     done: Semaphore,
//! # ) { unsafe {
//! let mut block = 0u32;
//! while block < 8 {
//!     mma_abt(accumulator, a, b_ring.tile(block % 3), MmaShape::M128_N64, block > 0);
//!     block += 1;
//! }
//! commit(done);
//! # } }
//! ```
//!
//! `docs/library/mma.md` has the chunk geometry the walks are derived from, and
//! why `mm_*` is a separate entry point rather than a `false`.

use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05InstructionDescriptor, tcgen05_commit_multicast_cg2,
    tcgen05_commit_shared_cluster,
};

use crate::shared::{MmaElement, OperandWalk, SharedTile, Swizzle};
use crate::sync::Semaphore;

/// The `M`×`N` an MMA instruction covers — the accumulator band the caller
/// allocated and, under `cta_group::2`, the pair's shared `M256` class. The one
/// descriptor field no walk can derive from its operands.
pub use cuda_device::tcgen05::Tcgen05MmaShape as MmaShape;

/// K elements per chained-MMA chunk (one 16-bit core-matrix step).
const K_CHUNK: usize = 16;
/// Bytes of one K chunk along a row.
const K_CHUNK_BYTES: usize = K_CHUNK * 2;
/// K chunks per swizzle-atom row (four for a 2-byte element under a 128-byte
/// atom).
const CHUNKS_PER_ROW: usize = 128 / K_CHUNK_BYTES;

/// The chunk offsets below assume 32 bytes per K=16 chunk, which is a 2-byte
/// element. Asserted at the walks that depend on it rather than generalized on a
/// guess.
const fn assert_two_byte_element<E: MmaElement>() {
    assert!(
        E::BYTES == 2,
        "chained-MMA chunk geometry is 2-byte-element only"
    );
}

/// The instruction descriptor for a walk that reads its operands with the given
/// transpose configuration: everything the walk and its element already fix,
/// leaving `shape` as the caller's one degree of freedom.
///
/// The accumulator stays fp32 because a walk names TMEM by a bare `u32` and has
/// no accumulator type to read one off.
#[inline(always)]
fn descriptor<E: MmaElement>(shape: MmaShape, transpose_a: bool, transpose_b: bool) -> u32 {
    Tcgen05InstructionDescriptor::builder()
        .shape(shape)
        .element_type(E::ELEMENT_TYPE)
        .accumulator_type(Tcgen05AccumulatorType::F32)
        .transpose_a(transpose_a)
        .transpose_b(transpose_b)
        .build()
        .raw()
}

/// Byte offset of K chunk `k` in a K-major operand whose stacked subtiles
/// are `subtile_bytes` apart.
const fn k_major_offset(k: usize, subtile_bytes: usize) -> usize {
    (k / CHUNKS_PER_ROW) * subtile_bytes + (k % CHUNKS_PER_ROW) * K_CHUNK_BYTES
}

/// Byte offset of K chunk `k` in a `transpose_b` operand (K along rows).
const fn k_rows_offset(k: usize, atom_bytes: usize) -> usize {
    k * K_CHUNK * atom_bytes
}

/// `D (+)= A·Bᵀ` with both operands K-major `[rows, K]` — a flash forward's
/// `S = Q·Kᵀ` walk.
///
/// One chained MMA per K=16 chunk, no transpose bits. `A` and `B` may stack
/// different row counts; each walks its own subtile stride.
///
/// ```no_run
/// # use kittens::mma::{MmaShape, mma_abt};
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # unsafe fn demo(
/// #     tmem: u32,
/// #     q: SharedTile<Bf16, 128, 64, Swizzle128B>,
/// #     k: SharedTile<Bf16, 64, 64, Swizzle128B>,
/// # ) { unsafe {
/// mma_abt(tmem, q, k, MmaShape::M128_N64, false);
/// # } }
/// ```
///
/// # Safety
///
/// - Exactly one thread issues this.
/// - `tmem` names an accumulator `shape` fits.
/// - Both tiles hold committed operand data until the MMA's own commit is
///   observed.
#[inline(always)]
pub unsafe fn mma_abt<
    E: MmaElement,
    const AR: usize,
    const BR: usize,
    const K: usize,
    S: Swizzle,
>(
    tmem: u32,
    a: SharedTile<E, AR, K, S>,
    b: SharedTile<E, BR, K, S>,
    shape: MmaShape,
    accumulate: bool,
) {
    const { assert_two_byte_element::<E>() };
    let instruction = descriptor::<E>(shape, false, false);
    unsafe {
        let mut chunk = 0;
        while chunk < K / K_CHUNK {
            E::mma(
                tmem,
                a.operand_descriptor(k_major_offset(
                    chunk,
                    SharedTile::<E, AR, K, S>::SUBTILE_BYTES,
                )),
                b.operand_descriptor(k_major_offset(
                    chunk,
                    SharedTile::<E, BR, K, S>::SUBTILE_BYTES,
                )),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// `D (+)= A·B` — a flash forward's `O = P·V` walk. `A` is K-major `[rows, K]`;
/// `B` supplies K along its rows, so this sets `transpose_b`.
///
/// One 64-wide output band per `B` subtile, accumulated into
/// `tmem + subtile * 64`; `accumulate` false starts every band fresh.
///
/// **`shape`'s `N` is the band's width and not the tile's** — 64 under a
/// 128-byte swizzle atom and a 2-byte element, for every `B` this walk accepts.
/// The `[M, N]` the operands describe is the *whole* accumulation, which this
/// walk reaches in `B::SUBTILES` instructions rather than one; passing it here
/// makes every band overlap the next by half and the last run `N / 2` columns
/// past the accumulator. That is a wrong `D` and not a fault, since the columns
/// are usually still inside the allocation ([#175]).
///
/// It is the one place in this module where the shape is not the operands'
/// logical product, and the reason it is an argument at all is [#128].
///
/// [#175]: https://github.com/feitreim/ferro-kittens/issues/175
/// [#128]: https://github.com/feitreim/ferro-kittens/issues/128
///
/// ```no_run
/// # use kittens::mma::{MmaShape, mma_ab};
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # unsafe fn demo(
/// #     tmem: u32,
/// #     p: SharedTile<Bf16, 128, 64, Swizzle128B>,
/// #     v: SharedTile<Bf16, 64, 128, Swizzle128B>,
/// # ) { unsafe {
/// // `[128, 128]` of accumulator, in two `M128_N64` bands.
/// mma_ab(tmem, p, v, MmaShape::M128_N64, true);
/// # } }
/// ```
///
/// # Safety
///
/// - As [`mma_abt`].
/// - `shape`'s `N` is `B::SUBTILE_COLS`, per the paragraph above.
/// - `tmem` owns `64 * B::SUBTILES` fp32 columns.
#[inline(always)]
pub unsafe fn mma_ab<E: MmaElement, const AR: usize, const K: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<E, AR, K, S>,
    b: SharedTile<E, K, N, S>,
    shape: MmaShape,
    accumulate: bool,
) {
    const { assert_two_byte_element::<E>() };
    let instruction = descriptor::<E>(shape, false, true);
    unsafe {
        let mut band = 0;
        while band < SharedTile::<E, K, N, S>::SUBTILES {
            let band_base = band * SharedTile::<E, K, N, S>::SUBTILE_BYTES;
            let mut chunk = 0;
            while chunk < K / K_CHUNK {
                E::mma(
                    tmem + (band as u32) * 64,
                    a.operand_descriptor(k_major_offset(
                        chunk,
                        SharedTile::<E, AR, K, S>::SUBTILE_BYTES,
                    )),
                    b.operand_descriptor(band_base + k_rows_offset(chunk, S::ATOM_BYTES)),
                    instruction,
                    accumulate || chunk > 0,
                );
                chunk += 1;
            }
            band += 1;
        }
    }
}

/// `D (+)= Aᵀ·B` — both operands MN-major, `[K, M]` and `[K, N]`, so both
/// transpose bits are set.
///
/// Both go through [`SharedTile::mn_walk`]: a K=16 chunk is 16 rows, and MN past
/// the first 64 columns is reached through the descriptor's leading offset, so
/// **one instruction covers the whole `[M, N]` band** rather than one per
/// stacked subtile as in [`mma_ab`].
///
/// # Safety
///
/// - As [`mma_abt`].
/// - `shape` is the `[M, N]` the two tiles describe.
#[inline(always)]
pub unsafe fn mma_atb<E: MmaElement, const K: usize, const M: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<E, K, M, S>,
    b: SharedTile<E, K, N, S>,
    shape: MmaShape,
    accumulate: bool,
) {
    const { assert_two_byte_element::<E>() };
    let (a, b) = (a.mn_walk(), b.mn_walk());
    let instruction = descriptor::<E>(shape, a.transposed(), b.transposed());
    unsafe {
        let mut chunk = 0;
        while chunk < K / K_CHUNK {
            E::mma(
                tmem,
                a.chunk_descriptor(chunk),
                b.chunk_descriptor(chunk),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// `D (+)= Aᵀ·Bᵀ` — `A` MN-major `[K, M]`, `B` K-major `[N, K]`. The mirror of
/// [`mma_ab`]: each walks one operand the way the other walks its second.
///
/// # Safety
///
/// As [`mma_atb`].
#[inline(always)]
pub unsafe fn mma_atbt<
    E: MmaElement,
    const K: usize,
    const M: usize,
    const N: usize,
    S: Swizzle,
>(
    tmem: u32,
    a: SharedTile<E, K, M, S>,
    b: SharedTile<E, N, K, S>,
    shape: MmaShape,
    accumulate: bool,
) {
    const { assert_two_byte_element::<E>() };
    let a = a.mn_walk();
    let instruction = descriptor::<E>(shape, a.transposed(), false);
    unsafe {
        let mut chunk = 0;
        while chunk < K / K_CHUNK {
            E::mma(
                tmem,
                a.chunk_descriptor(chunk),
                b.operand_descriptor(k_major_offset(
                    chunk,
                    SharedTile::<E, N, K, S>::SUBTILE_BYTES,
                )),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// A `cta_group::2` chained MMA over [`OperandWalk`] operands: one instruction
/// per chunk drives the CTA pair's shared `M256`-class accumulator, each CTA
/// contributing its own operand halves.
///
/// The layout lives in the walk values ([`SharedTile::k_walk`] /
/// [`SharedTile::mn_walk`]) and the descriptor's transpose bits come from
/// [`OperandWalk::transposed`], so a runtime layout choice moves the walk and the
/// instruction together. `E` is named by the caller rather than derived, because
/// an [`OperandWalk`] has already erased it.
///
/// ```no_run
/// # use kittens::mma::{MmaShape, mma_walk_cg2};
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # unsafe fn demo(
/// #     tmem: u32,
/// #     a: SharedTile<Bf16, 128, 64, Swizzle128B>,
/// #     b: SharedTile<Bf16, 64, 128, Swizzle128B>,
/// #     transposed: bool,
/// # ) { unsafe {
/// let a_walk = if transposed { a.mn_walk() } else { a.k_walk() };
/// mma_walk_cg2::<Bf16, 4>(tmem, a_walk, b.mn_walk(), MmaShape::M256_N128, true);
/// # } }
/// ```
///
/// # Safety
///
/// - As [`mma_abt`], from the *leader* CTA's issuing thread only.
/// - The cluster's peer holds its operand halves at the same shared offsets.
/// - Both walks cover `CHUNKS` K=16 chunks of committed data, of `E`.
#[inline(always)]
pub unsafe fn mma_walk_cg2<E: MmaElement, const CHUNKS: usize>(
    tmem: u32,
    a: OperandWalk,
    b: OperandWalk,
    shape: MmaShape,
    accumulate: bool,
) {
    const { assert_two_byte_element::<E>() };
    let instruction = descriptor::<E>(shape, a.transposed(), b.transposed());
    unsafe {
        let mut chunk = 0;
        while chunk < CHUNKS {
            E::mma_cg2(
                tmem,
                a.chunk_descriptor(chunk),
                b.chunk_descriptor(chunk),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// `D = A·Bᵀ` — [`mma_abt`] with the accumulator *replaced* rather than added
/// to, which is tcgen05's `mm` half of `{mm, mma}`.
///
/// Reach for the `mm_*` form whenever the choice is static: a `false` that
/// should have been `true` silently discards an accumulator, and these take no
/// `accumulate` because there is nothing left to decide. The `mma_*` walks keep
/// the parameter for callers whose choice is *runtime*, such as a GEMM's
/// `k > 0`.
///
/// ```no_run
/// # use kittens::mma::{MmaShape, mm_abt};
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # unsafe fn demo(
/// #     tmem: u32,
/// #     q: SharedTile<Bf16, 128, 64, Swizzle128B>,
/// #     k: SharedTile<Bf16, 64, 64, Swizzle128B>,
/// # ) { unsafe {
/// mm_abt(tmem, q, k, MmaShape::M128_N64);
/// # } }
/// ```
///
/// # Safety
///
/// As [`mma_abt`].
#[inline(always)]
pub unsafe fn mm_abt<
    E: MmaElement,
    const AR: usize,
    const BR: usize,
    const K: usize,
    S: Swizzle,
>(
    tmem: u32,
    a: SharedTile<E, AR, K, S>,
    b: SharedTile<E, BR, K, S>,
    shape: MmaShape,
) {
    unsafe { mma_abt(tmem, a, b, shape, false) }
}

/// `D = A·B` — [`mma_ab`] starting every output band fresh. See [`mm_abt`],
/// and [`mma_ab`] for why `shape`'s `N` is the band's width rather than the
/// tile's.
///
/// # Safety
///
/// As [`mma_ab`], including the band-width clause on `shape`.
#[inline(always)]
pub unsafe fn mm_ab<E: MmaElement, const AR: usize, const K: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<E, AR, K, S>,
    b: SharedTile<E, K, N, S>,
    shape: MmaShape,
) {
    unsafe { mma_ab(tmem, a, b, shape, false) }
}

/// `D = Aᵀ·B` — [`mma_atb`] starting the accumulator fresh. See [`mm_abt`].
///
/// # Safety
///
/// As [`mma_atb`].
#[inline(always)]
pub unsafe fn mm_atb<E: MmaElement, const K: usize, const M: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<E, K, M, S>,
    b: SharedTile<E, K, N, S>,
    shape: MmaShape,
) {
    unsafe { mma_atb(tmem, a, b, shape, false) }
}

/// `D = Aᵀ·Bᵀ` — [`mma_atbt`] starting the accumulator fresh. See [`mm_abt`].
///
/// # Safety
///
/// As [`mma_atbt`].
#[inline(always)]
pub unsafe fn mm_atbt<E: MmaElement, const K: usize, const M: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<E, K, M, S>,
    b: SharedTile<E, N, K, S>,
    shape: MmaShape,
) {
    unsafe { mma_atbt(tmem, a, b, shape, false) }
}

/// [`mma_walk_cg2`] starting the pair's accumulator fresh. See [`mm_abt`].
///
/// # Safety
///
/// As [`mma_walk_cg2`].
#[inline(always)]
pub unsafe fn mm_walk_cg2<E: MmaElement, const CHUNKS: usize>(
    tmem: u32,
    a: OperandWalk,
    b: OperandWalk,
    shape: MmaShape,
) {
    unsafe { mma_walk_cg2::<E, CHUNKS>(tmem, a, b, shape, false) }
}

/// Publish the issued MMA chain to `sem`: every consumer that `wait`s the
/// semaphore afterward observes the accumulator complete.
///
/// This is what makes an accumulator safe to drain — no LDTM may read a band
/// before the MMA writing it has been committed and waited on.
///
/// # Safety
///
/// - Same issuing thread as the MMAs it commits.
/// - `sem` is initialized.
#[inline(always)]
pub unsafe fn commit(sem: Semaphore) {
    unsafe { tcgen05_commit_shared_cluster(sem.raw() as *mut u64) }
}

/// Publish a `cta_group::2` MMA chain to every CTA in `cta_mask`'s copy of
/// `sem` — the pair-UMMA commit (`0b11` for both halves of the pair).
///
/// ```no_run
/// # use kittens::mma::commit_multicast_cg2;
/// # use kittens::Semaphore;
/// # unsafe fn demo(done: Semaphore) { unsafe {
/// commit_multicast_cg2(done, 0b11);
/// # } }
/// ```
///
/// # Safety
///
/// - As [`commit`], from the leader CTA's issuing thread.
/// - Each masked CTA holds an initialized barrier at `sem`'s address.
#[inline(always)]
pub unsafe fn commit_multicast_cg2(sem: Semaphore, cta_mask: u16) {
    unsafe { tcgen05_commit_multicast_cg2(sem.raw() as *mut u64, cta_mask) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{Bf16, Swizzle128B};

    /// The PTX ISA descriptor layout spelled out independently of the builder
    /// under test: dtype `[5:4]` = 1 (fp32), atype `[9:7]` and btype `[12:10]`
    /// = 1 (bf16), transpose A/B at bits 15/16, `N >> 3` at `[22:17]`,
    /// `M >> 4` at `[28:24]`.
    fn bf16_f32_descriptor(m: u32, n: u32, transpose_a: bool, transpose_b: bool) -> u32 {
        (1 << 4)
            | (1 << 7)
            | (1 << 10)
            | ((transpose_a as u32) << 15)
            | ((transpose_b as u32) << 16)
            | ((n >> 3) << 17)
            | ((m >> 4) << 24)
    }

    #[test]
    fn each_walks_descriptor_encodes_its_own_transpose_configuration() {
        // Flash's score MMA, its output MMA, and a GEMM's cta_group::2 stage.
        assert_eq!(
            descriptor::<Bf16>(MmaShape::M128_N64, false, false),
            bf16_f32_descriptor(128, 64, false, false)
        );
        assert_eq!(
            descriptor::<Bf16>(MmaShape::M128_N128, false, true),
            bf16_f32_descriptor(128, 128, false, true)
        );
        assert_eq!(
            descriptor::<Bf16>(MmaShape::M256_N128, true, true),
            bf16_f32_descriptor(256, 128, true, true)
        );
    }

    #[test]
    fn walks_carry_the_transpose_bit_their_layout_needs() {
        let base = 0x4000 as *mut u8;
        let k_major = unsafe { SharedTile::<Bf16, 128, 64, Swizzle128B>::from_raw(base) };
        let mn_major = unsafe { SharedTile::<Bf16, 64, 128, Swizzle128B>::from_raw(base) };
        assert!(!k_major.k_walk().transposed());
        assert!(mn_major.mn_walk().transposed());
    }

    #[test]
    fn k_major_walk_matches_flash_score_mma() {
        // flash: (chunk / 4) * SUBTILE_BYTES + (chunk % 4) * 32 over 8 chunks.
        let subtile = 64 * 128;
        for chunk in 0..8 {
            assert_eq!(
                k_major_offset(chunk, subtile),
                (chunk / 4) * subtile + (chunk % 4) * 32
            );
        }
    }

    #[test]
    fn row_walk_matches_flash_grad_mma() {
        // flash: chunk * 2048 (16 rows of 128-byte atoms) over 4 chunks.
        for chunk in 0..4 {
            assert_eq!(k_rows_offset(chunk, 128), chunk * 2048);
        }
    }

    /// The transposed walks read their bits off [`OperandWalk::transposed`],
    /// so what this pins is that the four operand orders really are four
    /// distinct descriptors and not two spelled twice.
    #[test]
    fn the_four_operand_orders_are_four_descriptors() {
        let orders = [
            descriptor::<Bf16>(MmaShape::M128_N128, false, false),
            descriptor::<Bf16>(MmaShape::M128_N128, false, true),
            descriptor::<Bf16>(MmaShape::M128_N128, true, true),
            descriptor::<Bf16>(MmaShape::M128_N128, true, false),
        ];
        for (i, &left) in orders.iter().enumerate() {
            for &right in &orders[i + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    /// An MN-major operand reaches its second stacked subtile through the
    /// descriptor's *leading* offset, never through the chunk step — which is
    /// what lets [`mma_atb`] cover a 128-wide MN in one instruction where
    /// [`mma_ab`] issues one per 64-wide band. A leading offset of 16 here
    /// would read a `[64, 128]` operand as if MN stopped at 64.
    #[test]
    fn an_mn_major_walk_reaches_the_second_subtile_by_leading_offset() {
        type Operand = SharedTile<Bf16, 64, 128, Swizzle128B>;
        let base = 0x4000 as *mut u8;
        let walk = unsafe { Operand::from_raw(base) }.mn_walk();
        let leading = |descriptor: u64| ((descriptor >> 16) & 0x3fff) << 4;
        assert_eq!(
            leading(walk.chunk_descriptor(0)),
            Operand::SUBTILE_BYTES as u64
        );
        // And the chunk step is 16 rows of the atom, the same one `mma_ab`
        // walks its MN-major `B` by.
        let address = |descriptor: u64| (descriptor & 0x3fff) << 4;
        for chunk in 0..4 {
            assert_eq!(
                address(walk.chunk_descriptor(chunk)) - address(walk.chunk_descriptor(0)),
                k_rows_offset(chunk, 128) as u64
            );
        }
    }
}
