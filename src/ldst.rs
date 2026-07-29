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
//!
//! Each instruction moves one `[16, 16]` block, so [`load_tile`] and
//! [`store_tile`] compose them into a whole `[M, N]` band — the same
//! composition [`crate::tmem::TmemTile::tile`] does over the drain, and out of
//! the same helpers, so a band cannot mean one thing in TMEM and another in
//! shared memory.
//!
//! [`load_vec`] and [`store_vec`] are the vector pair, and they are plain
//! scalar loads rather than a matrix instruction: a [`ColVec`]'s values are
//! one element each at columns no `ldmatrix` shape describes, and the lanes of
//! a column group all want the same address, which shared memory broadcasts.

use cuda_device::ptx_asm;
use cuda_device::wmma::ldmatrix_x2;

use crate::reg::{BaseLdtm, ColLayout, ColVec, Fragment, FragmentLayout, RegTile};
use crate::shared::{Element, SharedVec, SwizzledChunks};
use crate::tmem::{place_block, take_block};

/// The `(row, chunk)` that `lane` supplies as slot `slot`'s `m8n8.x2` address,
/// for a fragment at `(row, column)` of a swizzled tile.
///
/// The chunk index is the column's, counted across the tile's whole logical
/// row — which is the index [`SwizzledChunks::at`] takes, so a column past the
/// first stacked subtile needs nothing extra here.
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

/// Pack one thread's [`Fragment`] to `E` and store it into a swizzled tile —
/// the store twin of
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
/// All 32 lanes of the warp holding the fragment must call this together —
/// `stmatrix` takes its addresses from lanes 0..15 and its data from all 32, so
/// a lane that skips it makes the instruction ill-formed rather than leaving
/// its own values unwritten. `row` and `column` are the *shared tile's*
/// coordinates and nothing here reads tensor memory: `chunks` must belong to a
/// tile at least `row + 16` rows tall into which `column + 16` fits. The caller
/// owes a `fence.proxy.async.shared::cta` before any MMA or TMA reads the tile.
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

/// [`store_fragment`]'s four matrices in one instruction, rather than two
/// `.x2`s a slot apart.
///
/// A fragment is exactly four `8x8` b16 matrices — two slots by the two
/// 8-column halves — which is the shape `stmatrix.m8n8.x4` takes. The `.x2`
/// form has to be issued per slot because it names two matrices and a
/// fragment's are four; that is the only reason [`store_fragment`] loops at
/// all, so naming all four at once removes the loop rather than unrolling it.
///
/// **The addressing is [`fragment_address`] with one substitution and no new
/// derivation.** `.x2` takes its 16 addresses from lanes 0..15 and is issued
/// twice, once per slot; `.x4` takes 32 from all lanes, eight per matrix, and
/// its four matrices are the two slots' two halves in that order. So lane `l`
/// supplies what lane `l % 16` supplied for slot `l / 16` — the identity
/// `stmatrix_x4_addresses_are_the_x2_addresses_restacked` pins. A second
/// address derivation for the wider instruction is exactly the thing that
/// could drift from the load side, and there isn't one.
///
/// # Safety
///
/// As [`store_fragment`]: all 32 lanes of the warp holding the fragment must
/// call this together — all 32 of them supply addresses here, not just the
/// first 16 — `chunks` must belong to a tile at least `row + 16` rows tall into
/// which `column + 16` fits, and the caller owes a
/// `fence.proxy.async.shared::cta` before any MMA or TMA reads the tile.
#[inline(always)]
pub unsafe fn store_fragment_x4<E: Element<Unpacked = [f32; 2]>>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
    fragment: Fragment,
) {
    unsafe {
        let (address_row, chunk) = fragment_address(row, column, lane % 16, (lane / 16) as usize);
        stmatrix_m8n8_x4(
            chunks.at(address_row, chunk),
            E::pack([fragment.get(0, 0), fragment.get(0, 1)]),
            E::pack([fragment.get(0, 2), fragment.get(0, 3)]),
            E::pack([fragment.get(1, 0), fragment.get(1, 1)]),
            E::pack([fragment.get(1, 2), fragment.get(1, 3)]),
        );
    }
}

/// [`store_tile`] over [`store_fragment_x4`] — the same band, at one
/// `stmatrix` per `[16, 16]` block instead of two.
///
/// # Safety
///
/// As [`store_fragment_x4`], for every block of the band — including the
/// `fence.proxy.async.shared::cta` the caller owes before any MMA or TMA reads
/// the tile.
#[inline(always)]
pub unsafe fn store_tile_x4<E: Element<Unpacked = [f32; 2]>, const M: usize, const N: usize>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
    tile: RegTile<M, N, BaseLdtm>,
) where
    BaseLdtm: FragmentLayout<M, N>,
{
    unsafe {
        let mut row_block = 0usize;
        while row_block < M / 16 {
            let mut column_block = 0usize;
            while column_block < N / 16 {
                store_fragment_x4(
                    chunks,
                    row + 16 * row_block as u32,
                    column + 16 * column_block as u32,
                    lane,
                    take_block(&tile, row_block, column_block),
                );
                column_block += 1;
            }
            row_block += 1;
        }
    }
}

/// Read a `[16, 16]` block at `(row, column)` of a swizzled tile into one
/// thread's [`Fragment`] — the inverse of [`store_fragment`], and the
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

/// A whole `[M, N]` band of a swizzled tile at `(row, column)`, composed out of
/// the `M/16 × N/16` blocks [`load_fragment`] returns — the shared-side twin of
/// [`crate::tmem::TmemTile::tile`], and what a kernel whose input is not an MMA
/// operand reads its band with.
///
/// The blocks compose along both axes because chunk indices count across the
/// tile's whole logical row: a column past the first stacked subtile is the
/// cursor's problem ([`SwizzledChunks::at`]) and not this loop's.
///
/// # Safety
///
/// As [`load_fragment`], for every block of the band: all 32 lanes of the warp
/// must call this together, `chunks` must belong to a tile at least `row + M`
/// rows tall into which `column + N` fits, and its bytes must already be
/// visible to the generic proxy.
#[inline(always)]
pub unsafe fn load_tile<E: Element<Unpacked = [f32; 2]>, const M: usize, const N: usize>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
) -> RegTile<M, N, BaseLdtm>
where
    BaseLdtm: FragmentLayout<M, N>,
{
    unsafe {
        let mut tile = RegTile::<M, N, BaseLdtm>::zero();
        let mut row_block = 0usize;
        while row_block < M / 16 {
            let mut column_block = 0usize;
            while column_block < N / 16 {
                place_block(
                    &mut tile,
                    row_block,
                    column_block,
                    load_fragment(
                        chunks,
                        row + 16 * row_block as u32,
                        column + 16 * column_block as u32,
                        lane,
                    ),
                );
                column_block += 1;
            }
            row_block += 1;
        }
        tile
    }
}

/// The inverse of [`load_tile`]: a whole `[M, N]` band packed to `E` and
/// written back block by block through [`store_fragment`].
///
/// # Safety
///
/// As [`store_fragment`], for every block of the band — including the
/// `fence.proxy.async.shared::cta` the caller owes before any MMA or TMA reads
/// the tile.
#[inline(always)]
pub unsafe fn store_tile<E: Element<Unpacked = [f32; 2]>, const M: usize, const N: usize>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
    tile: RegTile<M, N, BaseLdtm>,
) where
    BaseLdtm: FragmentLayout<M, N>,
{
    unsafe {
        let mut row_block = 0usize;
        while row_block < M / 16 {
            let mut column_block = 0usize;
            while column_block < N / 16 {
                store_fragment(
                    chunks,
                    row + 16 * row_block as u32,
                    column + 16 * column_block as u32,
                    lane,
                    take_block(&tile, row_block, column_block),
                );
                column_block += 1;
            }
            row_block += 1;
        }
    }
}

/// Broadcast a [`SharedVec`] into the [`ColVec`] a per-column op consumes:
/// each lane reads the `L::VALUES` columns [`ColLayout::col_of`] says it holds.
///
/// The shared-memory twin of [`load_tile`] for the vector shape, and the way a
/// parameter loaded once per CTA — layernorm's `gamma`, an attention bias —
/// reaches the registers that multiply by it. Under [`BaseLdtm`] a column
/// depends only on `lane % 4`, so the 8 lanes of a column group issue the same
/// address and shared memory answers all of them from one bank read; there is
/// no shuffle here and nothing for a swizzle to spread.
///
/// # Safety
///
/// The vector's bytes must already be visible to the generic proxy (a TMA load
/// needs its barrier waited on, another warp's [`SharedVec::set`] a barrier),
/// and `L`'s columns must fit `N` — which every [`ColLayout<N>`] impl
/// guarantees.
#[inline(always)]
pub unsafe fn load_vec<E: Element, const N: usize, L: ColLayout<N>>(
    vec: SharedVec<E, N>,
    lane: u32,
) -> ColVec<N, L> {
    unsafe {
        let mut cols = ColVec::<N, L>::splat(0.0);
        let mut value = 0usize;
        while value < L::VALUES {
            cols.set(value, vec.get(L::col_of(lane, value) as usize));
            value += 1;
        }
        cols
    }
}

/// The inverse of [`load_vec`]: each lane writes back the columns it holds.
///
/// Every column of the vector is written by the four lanes of one quad rather
/// than by one lane, and they write the same value — a [`ColVec`] is
/// replicated across the lanes of a column group, so the redundancy is the
/// layout's and not this loop's. That makes the write idempotent, which is
/// what keeps it a plain store instead of a lane-masked one.
///
/// # Safety
///
/// All 32 lanes of the warp must call this together, and the caller owes a
/// `fence.proxy.async.shared::cta` before the TMA engine reads the vector.
#[inline(always)]
pub unsafe fn store_vec<E: Element, const N: usize, L: ColLayout<N>>(
    vec: SharedVec<E, N>,
    lane: u32,
    cols: ColVec<N, L>,
) {
    unsafe {
        let mut value = 0usize;
        while value < L::VALUES {
            vec.set(L::col_of(lane, value) as usize, cols.get(value));
            value += 1;
        }
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

/// [`stmatrix_m8n8_x2`]'s four-matrix form, through the same `ptx_asm!`
/// workaround and for the same reason: cuda-oxide `b099f64` ships
/// `stmatrix_m8n8_x4` in `generated/stmatrix.rs`, and its LLVM declaration
/// does not resolve for `sm_100a` any more than the `.x2` one does.
///
/// # Safety
///
/// `smem_ptr` must be a 16-byte-aligned shared-memory address with 16 bytes
/// writable at each of the 32 lanes' addresses, and all 32 lanes of the warp
/// must call this together.
#[inline(always)]
pub unsafe fn stmatrix_m8n8_x4(smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 smem; cvta.to.shared.u64 smem, %0; stmatrix.sync.aligned.m8n8.x4.shared.b16 [smem], {%1, %2, %3, %4}; }",
            in("l") smem_ptr as u64,
            in("r") r0,
            in("r") r1,
            in("r") r2,
            in("r") r3,
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

    /// The `.x4`'s 32 addresses are the `.x2`'s two sets of 16, stacked in
    /// slot order — which is the whole of why [`store_fragment_x4`] needs no
    /// address derivation of its own, and the thing that would have to break
    /// for the wide store to disagree with the narrow one.
    #[test]
    fn stmatrix_x4_addresses_are_the_x2_addresses_restacked() {
        for row in [0u32, 16, 48] {
            for column in [0u32, 16, 32, 48] {
                for lane in 0..32u32 {
                    // Eight lanes per matrix, four matrices: the lane's group
                    // picks the slot and the half, its position in the group
                    // the row.
                    let (address_row, chunk) =
                        fragment_address(row, column, lane % 16, (lane / 16) as usize);
                    assert_eq!(
                        address_row,
                        row as usize + 8 * (lane / 16) as usize + (lane % 8) as usize,
                        "lane {lane}"
                    );
                    assert_eq!(
                        chunk,
                        (column / 8) as usize + ((lane % 16) / 8) as usize,
                        "lane {lane}"
                    );
                }
            }
        }
    }

    /// The four matrices of one `.x4` are exactly the four the two `.x2`s
    /// wrote, one lane group each: nothing is dropped and nothing is written
    /// twice.
    #[test]
    fn the_x4_covers_both_slots_of_both_halves() {
        for row in [0u32, 32] {
            for column in [0u32, 16] {
                let wide: Vec<_> = (0..32)
                    .map(|lane| fragment_address(row, column, lane % 16, (lane / 16) as usize))
                    .collect();
                let narrow: Vec<_> = (0..Fragment::SLOTS)
                    .flat_map(|slot| {
                        (0..16).map(move |lane| fragment_address(row, column, lane, slot))
                    })
                    .collect();
                assert_eq!(wide, narrow, "row {row} column {column}");
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
