//! Chained tcgen05 MMA walks over shared-tile operands.
//!
//! tcgen05's native step is `D (+)= A·Bᵀ` over a K=16 bf16 chunk; a full
//! tile multiply is a chain of those steps with `enable_d` linking every
//! step after the first into the accumulator. Only walks extracted from
//! working kernels live here, never ones designed in the abstract: the
//! single-CTA [`mma_abt`]/[`mma_ab`] of a flash-attention forward, with the
//! layout in the types, and the cluster-pair (`M256_N256` multicast)
//! [`mma_walk_cg2`] of a GEMM, with the layout in [`OperandWalk`] values
//! because that kernel selects K-major vs MN-major at runtime.
//!
//! Chunk geometry, fixed by a 2-byte element and the 128-byte swizzle atom:
//! one K=16 chunk is 32 bytes along a row, so a subtile row holds four
//! chunks. A K-major operand walks `(k / 4) * SUBTILE_BYTES + (k % 4) * 32`
//! across its stacked subtiles; an MN-major operand (transpose-bit forms)
//! supplies K along rows instead, 16 rows (`16 * ATOM_BYTES` bytes) per
//! chunk. Which transpose bits the instruction descriptor carries is that
//! same choice of walk — none for [`mma_abt`], `transpose_b` for
//! [`mma_ab`], both for an MN-major [`mma_walk_cg2`] — so every walk builds
//! its own descriptor ([`MmaShape`] plus [`MmaElement::ELEMENT_TYPE`]) and a
//! caller has no way to pair a walk with a descriptor that disagrees. That
//! pairing used to be prose, and getting it wrong does not fault: the MMA
//! reads the operands under the wrong interpretation and fills the
//! accumulator with wrong numbers.
//!
//! The tile operands are generic over their [`MmaElement`], but the geometry
//! above is not: the walks assert a 2-byte element until issue #16 has a
//! second one to check a wider form against.

use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05InstructionDescriptor, tcgen05_commit_multicast_cg2,
    tcgen05_commit_shared_cluster, tcgen05_mma_f16, tcgen05_mma_f16_cg2,
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

/// The two things in this module that are still bf16-shaped, asserted at the
/// walks that depend on them rather than generalized on a guess: the chunk
/// geometry above assumes 32 bytes per K=16 chunk, and `tcgen05_mma_f16` is
/// the kind-specialized wrapper. Issue #12 swaps that for the `KIND`-generic
/// intrinsic driven by `MmaElement::MMA_KIND`; issue #16 widens the geometry,
/// against a second element it can be checked with.
const fn assert_two_byte_element<E: MmaElement>() {
    assert!(
        E::BYTES == 2,
        "chained-MMA chunk geometry and tcgen05_mma_f16 are 2-byte-element only"
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
            tcgen05_mma_f16(
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
                tcgen05_mma_f16(
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
/// Issue #12 owns that — routing `KIND` needs the element as a type here
/// anyway, and whether it comes back from the walk is its call to make.
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
            tcgen05_mma_f16_cg2(
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
}
