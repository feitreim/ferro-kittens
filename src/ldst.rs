//! Warp-scope register↔shared tile movers.
//!
//! Both directions of the swizzled-fragment path. Storing, a thread packs fp32
//! fragment values to the tile's element ([`Element::pack`]) and writes them
//! through `stmatrix`; loading, `ldmatrix` reads the same words back and
//! [`Element::unpack`] widens them. Addresses come from
//! [`crate::shared::SharedTile::swizzled_chunk`] in both cases, so an
//! accumulating MMA reads a stored operand exactly like a TMA-loaded tile, and
//! a load sees a TMA-loaded tile exactly like a drained accumulator.
//!
//! `ldmatrix` and `stmatrix` share an address convention — the 16 addresses
//! come from lanes 0..15 while the data is spread over all 32 — so the two
//! directions are one derivation ([`fragment_address`]) with the data flowing
//! opposite ways, and cannot drift apart.

use cuda_device::ptx_asm;
use cuda_device::wmma::ldmatrix_x2;

use crate::reg::Fragment;
use crate::shared::{Element, SwizzledChunks};

/// The `(row, chunk)` that `lane` supplies as slot `slot`'s `m8n8.x2` address,
/// for a fragment at `(row, column)` of a single-subtile tile.
///
/// `ldmatrix`/`stmatrix` take their 16 addresses from lanes 0..15 only — lanes
/// 0..7 the first 8x8 matrix, 8..15 the second — a different lane set than the
/// one holding the data. So the 16-byte chunk is the column's own chunk index
/// plus one for the upper half-warp (the second matrix is the 8-column half at
/// `column + 8`, which [`BaseLdtm::column`](crate::reg::BaseLdtm::column) puts
/// values `{2, 3}` in), and the row is `lane % 8` into the block: address lane
/// `l` supplies the row that data lanes `4*(l%8)..+4` own in this slot.
///
/// Lanes 16..31 land back on the first matrix's addresses. The instruction
/// ignores them on sm_100a, but they stay inside the tile, which is what makes
/// the safety contract below a statement about the tile rather than about
/// which lanes the hardware happens to read.
#[inline(always)]
pub const fn fragment_address(row: u32, column: u32, lane: u32, slot: usize) -> (usize, usize) {
    let chunk = (column / 8) as usize + (lane >= 8 && lane < 16) as usize;
    (row as usize + (lane % 8) as usize + 8 * slot, chunk)
}

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
        let mut slot = 0usize;
        while slot < Fragment::SLOTS {
            let (address_row, chunk) = fragment_address(row, column, lane, slot);
            stmatrix_m8n8_x2(
                chunks.at(address_row, chunk),
                E::pack([fragment.get(slot, 0), fragment.get(slot, 1)]),
                E::pack([fragment.get(slot, 2), fragment.get(slot, 3)]),
            );
            slot += 1;
        }
    }
}

/// Read a `[16, 16]` block at `(row, column)` of a single-subtile swizzled tile
/// into one thread's [`Fragment`] — the inverse of [`store_fragment`], and the
/// way a kernel whose input is not an MMA operand gets its data into registers
/// at all.
///
/// `ldmatrix.m8n8.x2` moves two b16 matrices, so this carries the same
/// `Element<Unpacked = [f32; 2]>` bound as the store for the same reason: a
/// 4-per-word element wants its own instruction shape and should fail to
/// typecheck rather than read half the bytes it was asked for.
///
/// The two returned words are the two matrices of the `.x2`, so they land in
/// the fragment as the value pairs `{0, 1}` and `{2, 3}` — exactly the halves
/// [`BaseLdtm::column`](crate::reg::BaseLdtm::column) places at `column` and
/// `column + 8`.
///
/// # Safety
///
/// All 32 lanes of the warp must call this together, `chunks` must belong to a
/// tile at least `row + 16` rows tall whose bytes are already visible to the
/// generic proxy (a TMA load needs its barrier waited on, a `stmatrix` needs
/// `fence.proxy.async.shared::cta`), and `column + 16` must fit the tile.
#[inline(always)]
pub unsafe fn load_fragment<E: Element<Unpacked = [f32; 2]>>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
) -> Fragment {
    unsafe {
        let mut fragment = Fragment::zero();
        let mut slot = 0usize;
        while slot < Fragment::SLOTS {
            let (address_row, chunk) = fragment_address(row, column, lane, slot);
            let [low, high] = ldmatrix_x2(chunks.at(address_row, chunk) as *const u32);
            let low = E::unpack(low);
            let high = E::unpack(high);
            fragment.set(slot, 0, low[0]);
            fragment.set(slot, 1, low[1]);
            fragment.set(slot, 2, high[0]);
            fragment.set(slot, 3, high[1]);
            slot += 1;
        }
        fragment
    }
}

/// Store two packed b16 matrix fragments (`stmatrix.sync.aligned.m8n8.x2`)
/// without routing through the unresolved LLVM stmatrix declaration emitted
/// by cuda-oxide b099f64.
///
/// The load direction needs no such workaround:
/// [`cuda_device::wmma::ldmatrix_x2`] lowers cleanly for `sm_100a`, so
/// [`load_fragment`] calls it directly. That it lives in a `wmma` module is a
/// filing accident — `ldmatrix` is a plain shared-memory read and has nothing
/// to do with the wmma MMA path this crate does not use.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reg::BaseLdtm;

    /// The 16 addressing lanes of one slot name the slot's 8 rows in each of
    /// the two 8-column halves — the `.x2`'s two matrices — and nothing else.
    #[test]
    fn addressing_lanes_cover_both_matrices_of_each_slot() {
        for row in [0u32, 16, 48] {
            for column in [0u32, 8, 32, 48] {
                for slot in 0..Fragment::SLOTS {
                    let mut seen = Vec::new();
                    for lane in 0..16u32 {
                        seen.push(fragment_address(row, column, lane, slot));
                    }
                    let expected: Vec<_> = (0..2)
                        .flat_map(|matrix| {
                            (0..8).map(move |r| {
                                (row as usize + 8 * slot + r, (column / 8) as usize + matrix)
                            })
                        })
                        .collect();
                    assert_eq!(seen, expected, "row {row} column {column} slot {slot}");
                }
            }
        }
    }

    /// The row an address lane supplies is the row the *data* lanes of its
    /// quad own in that slot — the whole reason the two lane sets differ.
    #[test]
    fn address_rows_are_the_rows_the_data_lanes_own() {
        for slot in 0..Fragment::SLOTS {
            for lane in 0..16u32 {
                let (address_row, _) = fragment_address(0, 0, lane, slot);
                for data_lane in 4 * (lane % 8)..4 * (lane % 8) + 4 {
                    assert_eq!(BaseLdtm::row(data_lane, slot) as usize, address_row);
                }
            }
        }
    }

    /// The chunk of the second matrix holds the columns `BaseLdtm` gives
    /// values `{2, 3}`, which is what makes those the fragment's high word.
    #[test]
    fn the_two_chunks_are_the_two_column_halves() {
        for column in [0u32, 16, 32, 48] {
            for lane in 0..8u32 {
                let (_, low) = fragment_address(0, column, lane, 0);
                let (_, high) = fragment_address(0, column, lane + 8, 0);
                assert_eq!(high, low + 1);
                let owned = |value| column + BaseLdtm::column(4 * lane, value);
                assert_eq!(owned(0) as usize / 8, low);
                assert_eq!(owned(1) as usize / 8, low);
                assert_eq!(owned(2) as usize / 8, high);
                assert_eq!(owned(3) as usize / 8, high);
            }
        }
    }

    /// Every lane's address stays inside the `[row + 16, column + 16]` block
    /// the caller promised — including lanes 16..31, whose addresses the
    /// instruction ignores but still forms.
    #[test]
    fn no_lane_addresses_outside_the_block() {
        for row in [0u32, 16, 48] {
            for column in [0u32, 16, 48] {
                for slot in 0..Fragment::SLOTS {
                    for lane in 0..32u32 {
                        let (address_row, chunk) = fragment_address(row, column, lane, slot);
                        assert!((row as usize..row as usize + 16).contains(&address_row));
                        assert!(
                            (column as usize / 8..column as usize / 8 + 2).contains(&chunk),
                            "lane {lane} chunk {chunk}"
                        );
                    }
                }
            }
        }
    }
}
