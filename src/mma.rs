//! Chained tcgen05 MMA walks over shared-tile operands.
//!
//! tcgen05's native step is `D (+)= A·Bᵀ` over a K=16 bf16 chunk; a full
//! tile multiply is a chain of those steps with `enable_d` linking every
//! step after the first into the accumulator. The single-CTA walks name the
//! layout in their types — [`mma_abt`]/[`mma_ab`] from a flash-attention
//! forward, [`mma_atb`]/[`mma_atbt`] completing that square — and the
//! cluster-pair [`mma_walk_cg2`] of a GEMM names it in [`OperandWalk`]
//! values, because that kernel selects K-major vs MN-major at runtime.
//!
//! Chunk geometry, fixed by a 2-byte element and the 128-byte swizzle atom:
//! one K=16 chunk is 32 bytes along a row, so a subtile row holds four
//! chunks. A **K-major** operand (`[MN, K]`, K contiguous) walks
//! `(k / 4) * SUBTILE_BYTES + (k % 4) * 32` across its stacked subtiles; an
//! **MN-major** operand (`[K, MN]`, MN contiguous — the transpose-bit forms)
//! supplies K along rows instead, 16 rows (`16 * ATOM_BYTES` bytes) per
//! chunk, and reaches MN past its first 64 columns through the descriptor's
//! *leading* offset rather than a step along the row: that is
//! [`SharedTile::mn_walk`], and every MN-major operand here is one.
//!
//! Which transpose bits the instruction descriptor carries is that same
//! choice of walk, so every walk builds its own descriptor ([`MmaShape`] plus
//! [`MmaElement::ELEMENT_TYPE`]) and a caller has no way to pair a walk with
//! a descriptor that disagrees. That pairing used to be prose, and getting it
//! wrong does not fault: the MMA reads the operands under the wrong
//! interpretation and fills the accumulator with wrong numbers.
//!
//! The four operand orders, and what each one's tiles are shaped like:
//!
//! | Walk | `A` | `B` | transpose `(a, b)` |
//! | --- | --- | --- | --- |
//! | [`mma_abt`] | `[M, K]` | `[N, K]` | `(false, false)` |
//! | [`mma_ab`] | `[M, K]` | `[K, N]` | `(false, true)` |
//! | [`mma_atb`] | `[K, M]` | `[K, N]` | `(true, true)` |
//! | [`mma_atbt`] | `[K, M]` | `[N, K]` | `(true, false)` |
//!
//! Every one of them comes in an [`mm_abt`]-style twin that starts the
//! accumulator fresh — see [`mm_abt`] for why that is a separate entry point
//! rather than a `false`.
//!
//! The tile operands are generic over their [`MmaElement`], and so is the
//! *instruction*: the walks issue through [`MmaElement::mma`], which routes
//! tcgen05's `KIND` from the element (#12). The chunk geometry above is what
//! is still bf16-shaped, and the walks assert a 2-byte element until issue
//! #16 has a second one to check a wider form against.

use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05InstructionDescriptor, tcgen05_commit_multicast_cg2,
    tcgen05_commit_shared_cluster,
};

use crate::shared::{MmaElement, OperandWalk, SharedTile, Swizzle};
use crate::sync::Semaphore;

/// The `M`×`N` an MMA instruction covers — the accumulator band the caller
/// allocated and, under `cta_group::2`, the pair's shared `M256` class. The
/// one descriptor field no walk can know.
pub use cuda_device::tcgen05::Tcgen05MmaShape as MmaShape;

/// K elements per chained-MMA chunk (one 16-bit core-matrix step).
const K_CHUNK: usize = 16;
/// Bytes of one K chunk along a row.
const K_CHUNK_BYTES: usize = K_CHUNK * 2;
/// K chunks per swizzle-atom row (four for a 2-byte element under a 128-byte
/// atom).
const CHUNKS_PER_ROW: usize = 128 / K_CHUNK_BYTES;

/// The one thing in this module that is still bf16-shaped, asserted at the
/// walks that depend on it rather than generalized on a guess: the chunk
/// geometry above assumes 32 bytes per K=16 chunk. Issue #16 widens it,
/// against a second element it can be checked with. (The *instruction* is no
/// longer on this list — [`MmaElement::mma`] routes `KIND` from the element.)
const fn assert_two_byte_element<E: MmaElement>() {
    assert!(
        E::BYTES == 2,
        "chained-MMA chunk geometry is 2-byte-element only"
    );
}

/// The instruction descriptor for a walk that reads its operands with the
/// given transpose configuration: everything the walk and its element already
/// fix, leaving `shape` as the caller's one degree of freedom.
///
/// The accumulator stays fp32 because a walk names TMEM by a bare `u32` and
/// has no accumulator type to read one off; fp16 accumulation is a
/// `.kind::f16`-only mode no kernel here uses, and threading a typed
/// accumulator through belongs with whatever gives the walks a [`crate::tmem`]
/// tile instead of an address.
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

/// `D (+)= A·Bᵀ` with both operands K-major `[rows, K]` — flash's
/// `S = Q·Kᵀ` walk. One chained MMA per K=16 chunk from the current leader
/// thread, under a descriptor this builds with no transpose bits; `shape`
/// names the caller's accumulator band. The operands may stack different row
/// counts (7e15's paired-`[128, K]` A against an unpaired `[64, K]` B) —
/// each walks its own subtile stride.
///
/// # Safety
///
/// Exactly one thread issues this; `tmem` names an accumulator `shape` fits;
/// both tiles hold committed operand data until the MMA's own commit is
/// observed.
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

/// `D (+)= A·B` — flash's `O = P·V` / gradient walk. `A` is K-major
/// `[rows, K]`; `B` supplies K along its rows, so the descriptor this builds
/// sets `transpose_b`. One 64-wide output band per `B` subtile, accumulated
/// into `tmem + subtile * 64`. `accumulate` false starts every band's
/// accumulator fresh.
///
/// # Safety
///
/// As [`mma_abt`]; `tmem` must own `64 * B::SUBTILES` fp32 columns.
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
/// transpose bits are set. The walk this issues is the one the *pair* path
/// already had as a value ([`SharedTile::mn_walk`]): a K=16 chunk is 16 rows,
/// and MN past the first 64 columns is reached through the descriptor's
/// leading offset, so one instruction covers the whole `[M, N]` band rather
/// than one per stacked subtile.
///
/// The transpose bits are read back off the walks rather than written down
/// beside them, exactly as [`mma_walk_cg2`] does — the operand order and the
/// descriptor cannot disagree because only one of them is stated.
///
/// # Safety
///
/// As [`mma_abt`]; `shape` must be the `[M, N]` the two tiles describe.
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

/// `D (+)= Aᵀ·Bᵀ` — `A` MN-major `[K, M]`, `B` K-major `[N, K]`. The last of
/// the four operand orders, and the mirror of [`mma_ab`]: each walks one
/// operand the way the other walks its second.
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

/// A cta_group::2 chained MMA over [`OperandWalk`] operands — one
/// instruction per chunk drives the CTA pair's shared `M256`-class
/// accumulator, each CTA contributing its own operand halves. The layout
/// lives in the walk values ([`SharedTile::k_walk`] /
/// [`SharedTile::mn_walk`]), so a kernel selecting K-major vs MN-major at
/// runtime (gemm's `transposed`) issues one loop either way — and the
/// descriptor's transpose bits come from [`OperandWalk::transposed`], so a
/// runtime layout choice moves the walk and the instruction together.
///
/// The element is the one field this cannot take from its operands: an
/// [`OperandWalk`] has already erased it, so `E` is named by the caller
/// (`mma_walk_cg2::<Bf16, CHUNKS>`) rather than derived as in [`mma_abt`].
/// It stays that way now that `E` routes the instruction as well as the
/// operand format (#12): a walk that carried its element back would have to
/// carry a `KIND` too, which is to say it would stop being layout-only.
///
/// # Safety
///
/// As [`mma_abt`], from the *leader* CTA's issuing thread only, with the
/// cluster's peer holding its operand halves at the same shared offsets;
/// both walks must cover `CHUNKS` K=16 chunks of committed data, of `E`.
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
/// The instruction is the same one either way; what changes is who owes the
/// invariant. `mma_abt(.., false)` reads as an argument, so a call site that
/// means "this is the first MMA into a fresh band" and a call site that means
/// "and I have checked nothing else wrote there" spell themselves
/// identically, and a `false` that should have been `true` silently discards
/// an accumulator. Naming the two cases puts that in the type: everything
/// below takes no `accumulate` because there is nothing left to decide.
///
/// The `accumulate` parameter stays on the `mma_*` walks for the callers
/// whose choice is *runtime* — gemm's `k > 0` is one value threaded through
/// one chain, and splitting it into two typed arms would duplicate the chain
/// for no gain (see [`OperandWalk`]).
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

/// `D = A·B` — [`mma_ab`] starting every output band fresh. See [`mm_abt`].
///
/// # Safety
///
/// As [`mma_ab`].
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
/// # Safety
///
/// Same issuing thread as the MMAs it commits; `sem` initialized.
#[inline(always)]
pub unsafe fn commit(sem: Semaphore) {
    unsafe { tcgen05_commit_shared_cluster(sem.raw() as *mut u64) }
}

/// Publish a cta_group::2 MMA chain to every CTA in `cta_mask`'s copy of
/// `sem` — the pair-UMMA commit (`0b11` for both halves of the pair).
///
/// # Safety
///
/// As [`commit`], from the leader CTA's issuing thread, with each masked
/// CTA holding an initialized barrier at `sem`'s address.
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
        // The three descriptors the call sites used to hand-build: flash's
        // score MMA, its output MMA, and gemm's cta_group::2 stage.
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
