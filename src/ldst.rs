//! Warp-scope register↔shared tile movers.
//!
//! Storing, a thread packs fp32 fragment values to the tile's element
//! ([`Element::pack`]) and writes them through `stmatrix`; loading, `ldmatrix`
//! reads the same words back and [`Element::unpack`] widens them. Both
//! directions address through [`crate::shared::SharedTile::swizzled_chunk`], so
//! a stored operand and a TMA-loaded tile are the same tile to either side.
//!
//! One instruction moves one `[16, 16]` block. [`load_tile`] and [`store_tile`]
//! compose those blocks into an `[M, N]` band — the same composition
//! [`crate::tmem::TmemTile::tile`] does over the drain.
//!
//! [`load_vec`] and [`store_vec`] are the vector pair, and are plain scalar
//! accesses rather than a matrix instruction: a [`ColVec`]'s values are one
//! element each at columns no `ldmatrix` shape describes.
//!
//! Design notes and measurements: `docs/library/ldst.md`.

use cuda_device::ptx_asm;
use cuda_device::thread::__unroll_config;
use cuda_device::wmma::ldmatrix_x2;

use crate::reg::{BaseLdtm, ColLayout, ColVec, Fragment, FragmentLayout, RegTile};
use crate::shared::{Element, SharedVec, SwizzledChunks};
use crate::tmem::{place_block, take_block};

/// The `(row, chunk)` that `lane` supplies as slot `slot`'s `m8n8.x2` address,
/// for a fragment at `(row, column)` of a swizzled tile.
///
/// `ldmatrix`/`stmatrix` take their 16 addresses from lanes 0..15 only — lanes
/// 0..7 the first 8x8 matrix, 8..15 the second at `column + 8` — a different
/// lane set than the one holding the data: address lane `l` supplies the row
/// that data lanes `4*(l%8)..+4` own in this slot. Chunks count across the
/// tile's whole logical row (the index [`SwizzledChunks::at`] takes), so a
/// column past the first stacked subtile needs nothing extra here.
///
/// Lanes 16..31 land back on the first matrix's addresses; the instruction
/// ignores them on sm_100a, and they stay inside the tile either way.
#[inline(always)]
pub const fn fragment_address(row: u32, column: u32, lane: u32, slot: usize) -> (usize, usize) {
    let chunk = (column / 8) as usize + (lane >= 8 && lane < 16) as usize;
    (row as usize + (lane % 8) as usize + 8 * slot, chunk)
}

/// Pack one thread's [`Fragment`] to `E` and store it into a swizzled tile —
/// the store twin of [`crate::tmem::TmemTile::fragment_tile`], addressed by the
/// same `(row, column)` the drain used.
///
/// One `stmatrix.m8n8.x2` per slot covers the fragment's two owned rows, each
/// write pairing the two values of one 8-column half. `.x2` moves two *b16*
/// matrices, hence the `Element<Unpacked = [f32; 2]>` bound: a 4-per-word
/// element fails to typecheck here rather than quietly writing half its bytes.
///
/// # Safety
///
/// - All 32 lanes of the warp holding the fragment must call this together;
///   `stmatrix` takes addresses from lanes 0..15 and data from all 32, so a lane
///   that skips it makes the instruction ill-formed.
/// - `row` and `column` are the *shared tile's* coordinates: `chunks` must
///   belong to a tile at least `row + 16` rows tall into which `column + 16`
///   fits.
/// - The caller owes a [`crate::shared::publish_to_async_proxy`] before any MMA
///   or TMA reads the tile, and a barrier after it to carry this thread's
///   ordering to whichever thread issues the read.
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
/// 8-column halves — which is the shape `stmatrix.m8n8.x4` takes, so naming all
/// four at once removes [`store_fragment`]'s loop rather than unrolling it.
/// The addressing is still [`fragment_address`]: lane `l` supplies here what
/// lane `l % 16` supplied for slot `l / 16`.
///
/// # Safety
///
/// As [`store_fragment`], except that all 32 lanes supply addresses here rather
/// than only the first 16:
///
/// - All 32 lanes of the warp holding the fragment must call this together.
/// - `chunks` must belong to a tile at least `row + 16` rows tall into which
///   `column + 16` fits.
/// - The caller owes a [`crate::shared::publish_to_async_proxy`] before any MMA
///   or TMA reads the tile.
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

/// [`store_fragment_x4`] taking words that are **already packed**, so the
/// `stmatrix` runs with no `cvt` in front of it.
///
/// The addressing is [`store_fragment_x4`]'s, unchanged: this is that function
/// with the four `E::pack` calls removed and the four words passed in.
///
/// **Nothing in tree computes a right answer with this.** It exists so
/// `experiments/`' `pack16` rung can hold every other instruction in a drain
/// fixed and take the `cvt` column to zero — see
/// [`crate::tmem::TmemTile::fragments_pack16_x8`].
///
/// # Safety
///
/// As [`store_fragment_x4`], and the words must be the b16 pairs the destination
/// block expects — a caller that hands over something else writes something else.
#[inline(always)]
pub unsafe fn store_packed_x4<E: Element>(
    chunks: SwizzledChunks<E>,
    row: u32,
    column: u32,
    lane: u32,
    words: [u32; 4],
) {
    unsafe {
        let (address_row, chunk) = fragment_address(row, column, lane % 16, (lane / 16) as usize);
        stmatrix_m8n8_x4(
            chunks.at(address_row, chunk),
            words[0],
            words[1],
            words[2],
            words[3],
        );
    }
}

/// [`store_tile`] over [`store_fragment_x4`] — the same band, at one
/// `stmatrix` per `[16, 16]` block instead of two.
///
/// # Safety
///
/// As [`store_fragment_x4`], for every block of the band — including the
/// [`crate::shared::publish_to_async_proxy`] the caller owes before any MMA
/// or TMA reads the tile.
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
        // Both walks fully unrolled, or the counters stay runtime values, the
        // reads of `tile` stay dynamic GEPs SROA cannot split, and the whole
        // aggregate is homed to a `.local` depot (#166). LLVM stops unrolling
        // on its own near 32 fp32 columns.
        let mut row_block = 0usize;
        while row_block < const { M / 16 } {
            __unroll_config::<0>();
            let mut column_block = 0usize;
            while column_block < const { N / 16 } {
                __unroll_config::<0>();
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
/// thread's [`Fragment`] — the inverse of [`store_fragment`].
///
/// `ldmatrix.m8n8.x2` moves two b16 matrices, hence the same
/// `Element<Unpacked = [f32; 2]>` bound the store carries. The two words are
/// the `.x2`'s two matrices and land in the fragment as the value pairs
/// `{0, 1}` and `{2, 3}` — the halves
/// [`BaseLdtm::column`](crate::reg::BaseLdtm::column) places at `column` and
/// `column + 8`.
///
/// ```no_run
/// # use kittens::ldst::load_fragment;
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # use kittens::{lane, warp_id};
/// # unsafe fn demo(tile: SharedTile<Bf16, 128, 64, Swizzle128B>) {
/// let block = unsafe { load_fragment(tile.chunk_writer(), 32 * warp_id(), 16, lane()) };
/// let sum = block.get(0, 0) + block.get(1, 3);
/// # }
/// ```
///
/// # Safety
///
/// - All 32 lanes of the warp must call this together.
/// - `chunks` must belong to a tile at least `row + 16` rows tall into which
///   `column + 16` fits.
/// - Its bytes must already be visible to the generic proxy: a TMA load needs
///   its barrier waited on, a `stmatrix` needs
///   `fence.proxy.async.shared::cta`.
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
/// [`crate::tmem::TmemTile::tile`].
///
/// The blocks compose along both axes because chunk indices count across the
/// tile's whole logical row: a column past the first stacked subtile is
/// [`SwizzledChunks::at`]'s problem and not this loop's.
///
/// ```no_run
/// # use kittens::ldst::load_tile;
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # use kittens::{BaseLdtm, RegTile, lane, warp_id};
/// # unsafe fn demo(tile: SharedTile<Bf16, 128, 64, Swizzle128B>) {
/// let band: RegTile<32, 64, BaseLdtm> =
///     unsafe { load_tile(tile.chunk_writer(), 32 * warp_id(), 0, lane()) };
/// let row_maxima = band.row_max();
/// # }
/// ```
///
/// # Safety
///
/// As [`load_fragment`], for every block of the band:
///
/// - All 32 lanes of the warp must call this together.
/// - `chunks` must belong to a tile at least `row + M` rows tall into which
///   `column + N` fits.
/// - Its bytes must already be visible to the generic proxy.
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
        // Fully unrolled for `tile`'s sake — see `store_tile_x4` (#166).
        let mut row_block = 0usize;
        while row_block < const { M / 16 } {
            __unroll_config::<0>();
            let mut column_block = 0usize;
            while column_block < const { N / 16 } {
                __unroll_config::<0>();
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
/// ```no_run
/// # use kittens::ldst::store_tile;
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B, publish_to_async_proxy};
/// # use kittens::{BaseLdtm, RegTile, lane, warp_id};
/// # unsafe fn demo(tile: SharedTile<Bf16, 128, 64, Swizzle128B>, band: RegTile<32, 64, BaseLdtm>) {
/// unsafe { store_tile(tile.chunk_writer(), 32 * warp_id(), 0, lane(), band) };
/// unsafe { publish_to_async_proxy() };
/// # }
/// ```
///
/// # Safety
///
/// As [`store_fragment`], for every block of the band — including the
/// [`crate::shared::publish_to_async_proxy`] the caller owes before any MMA
/// or TMA reads the tile.
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
        // Fully unrolled for `tile`'s sake — see `store_tile_x4` (#166).
        let mut row_block = 0usize;
        while row_block < const { M / 16 } {
            __unroll_config::<0>();
            let mut column_block = 0usize;
            while column_block < const { N / 16 } {
                __unroll_config::<0>();
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
/// The way a parameter loaded once per CTA — layernorm's `gamma`, an attention
/// bias — reaches the registers that multiply by it. Under [`BaseLdtm`] a
/// column depends only on `lane % 4`, so the 8 lanes of a column group issue
/// the same address and shared memory answers all of them from one bank read.
///
/// ```no_run
/// # use kittens::ldst::load_vec;
/// # use kittens::shared::{Bf16, SharedVec};
/// # use kittens::{BaseLdtm, ColVec, RegTile, lane};
/// # unsafe fn demo(gamma: SharedVec<Bf16, 64>, band: RegTile<32, 64, BaseLdtm>) {
/// let scale: ColVec<64, BaseLdtm> = unsafe { load_vec(gamma, lane()) };
/// let scaled = band.mul_col(scale);
/// # }
/// ```
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
        while value < const { L::VALUES } {
            __unroll_config::<0>();
            cols.set(value, vec.get(L::col_of(lane, value) as usize));
            value += 1;
        }
        cols
    }
}

/// The inverse of [`load_vec`]: each lane writes back the columns it holds.
///
/// A [`ColVec`] is replicated across the lanes of a column group, so every
/// column is written by four lanes with the same value. The write is idempotent
/// for that reason, and needs no lane mask.
///
/// # Safety
///
/// All 32 lanes of the warp must call this together, and the caller owes a
/// [`crate::shared::publish_to_async_proxy`] before the TMA engine reads the
/// vector.
#[inline(always)]
pub unsafe fn store_vec<E: Element, const N: usize, L: ColLayout<N>>(
    vec: SharedVec<E, N>,
    lane: u32,
    cols: ColVec<N, L>,
) {
    unsafe {
        let mut value = 0usize;
        while value < const { L::VALUES } {
            __unroll_config::<0>();
            vec.set(L::col_of(lane, value) as usize, cols.get(value));
            value += 1;
        }
    }
}

/// Store two packed b16 matrix fragments (`stmatrix.sync.aligned.m8n8.x2`),
/// written as inline PTX because cuda-oxide's stmatrix declaration does not
/// resolve for `sm_100a`.
///
/// The load direction needs no such workaround:
/// [`cuda_device::wmma::ldmatrix_x2`] lowers cleanly, so [`load_fragment`]
/// calls it directly.
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
/// workaround and for the same reason.
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

    /// The `.x4`'s 32 addresses are the `.x2`'s two sets of 16, stacked in slot
    /// order — why [`store_fragment_x4`] needs no derivation of its own.
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
