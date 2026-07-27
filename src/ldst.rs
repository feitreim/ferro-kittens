//! Warp-scope register↔shared tile movers.
//!
//! The store side of the swizzled-fragment path: a thread packs fp32
//! fragment values to the tile's element ([`Element::pack`]) and stores them
//! through `stmatrix` at addresses from
//! [`crate::shared::SharedTile::swizzled_chunk`], so an accumulating MMA reads
//! the operand exactly like a TMA-loaded tile.

use cuda_device::ptx_asm;

use crate::reg::Fragment;
use crate::shared::{Element, SwizzledChunks};

/// Pack one thread's [`Fragment`] to `E` and store it into a single-subtile
/// swizzled tile — the store twin of
/// [`crate::tmem::TmemTile::fragment_tile`], addressed by the same
/// `(row, column)` the drain used.
///
/// `stmatrix.m8n8.x2` moves two *b16* matrices, so this path holds only for
/// elements that pack two fp32 per word. That is a bound
/// (`Element<Unpacked = [f32; 2]>`) rather than an assertion: a 4-per-word
/// element does not typecheck here and gets the store its own instruction
/// shape needs, instead of quietly writing half the bytes.
///
/// One `stmatrix.m8n8.x2` write per slot covers the fragment's two owned rows,
/// each write pairing the two values of one 8-column half:
/// [`BaseLdtm::column`](crate::reg::BaseLdtm::column) puts values `{0, 1}` in
/// the half at `column` and `{2, 3}` in the half at `column + 8`, so those are
/// the two matrices of one `.x2`.
///
/// The addresses come from lanes 0..15 only (lanes 0..7 the first matrix,
/// 8..15 the second) — a different lane set than the one holding the data,
/// which is why the 16-byte chunk is the column's chunk index plus one for the
/// upper half-warp, and the row is `lane % 8` into the block: address lane `l`
/// supplies the row that data lanes `4*(l%8)..+4` own in this slot.
///
/// # Safety
///
/// All 32 lanes of the warp owning TMEM rows `row..row+16` must call this
/// together, `chunks` must belong to a tile at least `row + 16` rows tall, and
/// the caller owes a `fence.proxy.async.shared::cta` before any MMA reads the
/// tile.
#[inline(always)]
pub unsafe fn store_fragment<E: Element<Unpacked = [f32; 2]>>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
    fragment: Fragment,
) {
    unsafe {
        let chunk = (column / 8) as usize + usize::from((8..16).contains(&lane));
        let low = row as usize + (lane % 8) as usize;
        let mut slot = 0usize;
        while slot < Fragment::SLOTS {
            stmatrix_m8n8_x2(
                chunks.at(low + 8 * slot, chunk),
                E::pack([fragment.get(slot, 0), fragment.get(slot, 1)]),
                E::pack([fragment.get(slot, 2), fragment.get(slot, 3)]),
            );
            slot += 1;
        }
    }
}

/// Store two packed b16 matrix fragments (`stmatrix.sync.aligned.m8n8.x2`)
/// without routing through the unresolved LLVM stmatrix declaration emitted
/// by cuda-oxide b099f64.
///
/// # Safety
///
/// `smem_ptr` must be a 16-byte-aligned shared-memory address with 32 bytes
/// writable, and all 32 lanes of the warp must call this together.
#[inline(always)]
pub unsafe fn stmatrix_m8n8_x2(smem_ptr: *mut u8, r0: u32, r1: u32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 smem; cvta.to.shared.u64 smem, %0; stmatrix.sync.aligned.m8n8.x2.shared.b16 [smem], {%1, %2}; }",
            in("l") smem_ptr as u64,
            in("r") r0,
            in("r") r1,
            clobber("memory"),
        );
    }
}
