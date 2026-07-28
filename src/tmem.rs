//! TMEM accumulator views.
//!
//! A tcgen05 accumulator lives in tensor memory, addressed as
//! `base + (row << 16) + column`: the high half-word selects one of the
//! 128 TMEM lanes (accumulator rows), the low half-word a 4-byte column.
//! `TmemTile` carries that address plus the logical `[R, C]` fp32 shape,
//! so kernel code names its `S`/`O`/`dP` segments as tiles instead of
//! threading bare `u32` addresses and `(row << 16) + column` arithmetic
//! through every drain loop.
//!
//! MMA shapes with phantom rows (`M128` over 64-row tiles) still
//! type the *drained* shape: `R`/`C` describe what the kernel reads back,
//! not what the instruction touches.
//!
//! # How many columns you ask for is how many CTAs fit on the SM
//!
//! An SM has **512 columns** of tensor memory and the allocator divides them.
//! `512 / columns` CTAs hold allocations on one SM at the same time:
//!
//! | columns | CTAs an SM | who asks for it |
//! | ---: | ---: | --- |
//! | 32 | 16 | the allocator's smallest unit |
//! | 64 | 8 | |
//! | 128 | 4 | `gemm`, through [`alloc_cluster`] |
//! | 256 | 2 | `flash_forward`, through [`alloc_block`] |
//! | 512 | 1 | the whole SM |
//!
//! Tensor memory is one of two per-CTA resources an SM divides, and residency
//! is **whichever of the two is tighter**: `min(512 / columns, shared per SM /
//! shared plan)`. Both of this repo's tcgen05 kernels are capped by the shared
//! side rather than this one — `gemm` at 3 CTAs where its columns would allow 4,
//! `flash_forward` at 1 where its columns would allow 2 — so read the table
//! above as a ceiling that another resource may be sitting under.
//!
//! Measured on a B200 by `device-tests`' `tmem residency census`, which counts
//! CTAs rather than asking about them: every CTA writes down its `%smid` and
//! timestamps both ends of its allocation off `%globaltimer`, and the host
//! sweeps the intervals for the most that were ever open at once on one SM. The
//! counted figure is `512 / columns` exactly, at every legal column count.
//!
//! **`cta_group::2` is charged the same way.** A pair does *not* split the
//! allocation: each rank is charged its full column count against its own SM's
//! 512, so [`alloc_cluster`] at 128 columns leaves room for four CTAs an SM
//! exactly as [`alloc_block`] at 128 does. That was the leading hypothesis for
//! why `gemm` and `flash_forward` might differ (#78) and it is refuted.
//!
//! ## Do not use the occupancy query to predict this
//!
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` returns **1** for any kernel
//! whose code contains a `tcgen05.alloc` — at every column count, at every
//! block width, and even for a kernel that allocates and *releases* before
//! doing any work at all. It is not tracking a resource the CTA holds; it is
//! reacting to the instruction being present. `cuOccupancyMaxActiveClusters`
//! does the same for a `#[cluster_launch]` kernel. Both queries are accurate on
//! kernels with no allocator in them (32 and 8 CTAs an SM respectively, matched
//! by the census to the CTA), which is what makes the 1 easy to believe.
//!
//! #74 and #77 read that 1 as hardware and concluded that the allocation size
//! could not be a lever and that warps per CTA was all that was left. #51 then
//! measured `gemm` losing 2.07× to a one-CTA-per-SM grid cap, which is
//! impossible if one CTA per SM were the ceiling, and #83 bisected that kernel's
//! true residency to 3 — a figure the census reproduces exactly at the same
//! envelope, from timestamps rather than from a throughput curve.
//!
//! So **shrinking an allocation to the next legal power of two doubles the CTAs
//! tensor memory will hold**, and that is a real lever where #74 and #77 said
//! there was none. It is only worth pulling while tensor memory is the tighter
//! of the two resources, which for both kernels here it currently is not: the
//! thing to shrink is the shared plan.
//!
//! The same trap is worth naming once. #70 tried to price `flash_forward`'s
//! shared plan by querying occupancy at 147536 B, 73792, 32800 and zero, got 1
//! block/SM at every one of them, and concluded shared memory was not the
//! lever. Every one of those answers was the allocator pinning the query at 1.
//! Shared memory is in fact the *only* thing capping that kernel.
//!
//! ## The extra CTA, and what a blocking allocator looks like
//!
//! The census counts one *more* CTA resident on the SM than is holding columns
//! — 3 resident against 2 holders at 256 columns, 5 against 4 at 128. That CTA
//! is admitted and parked inside `tcgen05.alloc`, which blocks until the
//! columns it asked for come free. It costs a CTA slot and its warps make no
//! progress, so it is neither absent nor working. Nothing that reads occupancy
//! off a throughput curve can see the difference, which is why the census
//! timestamps `entered` and `allocated` separately.

use cuda_device::cusimd::{CuSimd, TmemRegs4};
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_alloc_cg2, tcgen05_dealloc, tcgen05_dealloc_cg2,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_relinquish_alloc_permit,
    tcgen05_relinquish_alloc_permit_cg2, tcgen05_st_16x256b_x1_raw, tcgen05_store_wait,
};
use cuda_device::{cluster, thread, warp};

use crate::reg::{BaseLdtm, Fragment, FragmentLayout, RegTile};

/// CTA-wide TMEM allocation, the lifecycle every tcgen05 kernel opens with:
/// warp 0 allocates `columns` into the shared staging word, a block sync
/// publishes it, and every thread reads back the address.
///
/// That warp also gives up the CTA's *allocation permit*, which is a
/// different thing from giving back the columns — those are
/// [`dealloc_block`]'s. The permit is the right to call the allocator again,
/// so it goes back beside the alloc rather than beside the dealloc: this
/// library allocates once per CTA, and holding the right to allocate for the
/// rest of a CTA's life is what the instruction exists to avoid.
///
/// PTX requires the permit back before the CTA exits and does not say what
/// happens otherwise. #46 predicted a deadlock — the next CTA on that SM
/// stuck in `tcgen05.alloc` — and on a B200 that is *not* observable: with
/// this call removed, 4736 CTAs over three launches, each taking all 512
/// columns, all completed, and a CTA sitting on an unrelinquished permit for
/// milliseconds did not delay a co-resident CTA's own allocation by a
/// measurable amount. So this is conformance, not a bug fix, and the hang
/// that motivated the issue (#40) belongs to the other half of that defect:
/// a `cta_group::2` dealloc issued by one CTA of the pair leaks *columns*,
/// and columns do not come back.
///
/// # Safety
///
/// Every thread of the block must call this together, and at most once per
/// CTA: allocating again after the permit is relinquished is illegal.
/// `slot` must point to shared memory (a `SharedArray<u32, 1, 4>` static).
#[inline(always)]
pub unsafe fn alloc_block(slot: *mut u32, columns: u32) -> u32 {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_alloc(slot, columns);
            tcgen05_relinquish_alloc_permit();
        }
        thread::sync_threads();
        *(slot as *const u32)
    }
}

/// Release the CTA's TMEM allocation from warp 0, after the caller's own
/// fence/sync has retired every outstanding read of it.
///
/// # Safety
///
/// No thread may touch the allocation afterwards; `columns` must match the
/// [`alloc_block`] call.
#[inline(always)]
pub unsafe fn dealloc_block(address: u32, columns: u32) {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_dealloc(address, columns);
        }
    }
}

/// Cluster-wide TMEM allocation: [`alloc_block`]'s `cta_group::2` twin, for
/// the single allocation a 2-CTA MMA accumulates into.
///
/// The three `cta_group::2` allocator instructions all say the same thing
/// about who issues them — *one full warp in each peer CTA* — so both CTAs
/// allocate, both relinquish, both later [`dealloc_cluster`], and each reads
/// the address out of its own staging word, because the collective writes one
/// into each. A peer that instead maps the leader's word over distributed
/// shared memory gets the right address and still leaves the pair's allocator
/// half-driven, which is what hung the GEMM's second launch (#40); this body
/// is that kernel's fix, lifted (#24), unchanged because it is the version
/// that ran on silicon.
///
/// The `cluster_sync` is what publishes the staging words across the pair; a
/// caller that stages barriers for its peer before allocating gets that
/// publication out of the same sync.
///
/// Each rank is charged `columns` against **its own SM's 512**, so the pair
/// costs residency exactly as two independent [`alloc_block`] calls would —
/// see the module docs. A `cta_group::2` allocation is not a half-share of one
/// SM's tensor memory, and the pair's two CTAs are on two different SMs in any
/// case.
///
/// # Safety
///
/// Every thread of every CTA in the cluster must call this together and at
/// most once, with `slot` at the same shared offset in both CTAs.
#[inline(always)]
pub unsafe fn alloc_cluster(slot: *mut u32, columns: u32) -> u32 {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_alloc_cg2(slot, columns);
        }
        thread::sync_threads();
        cluster::cluster_sync();
        if warp::warp_id() == 0 {
            tcgen05_relinquish_alloc_permit_cg2();
        }
        *(slot as *const u32)
    }
}

/// Release the cluster's TMEM allocation, warp 0 of each peer CTA taking its
/// half — [`dealloc_block`]'s `cta_group::2` twin, and as with that one the
/// fence and the syncs that retire the pair's reads are the caller's.
///
/// # Safety
///
/// No thread of either CTA may touch the allocation afterwards; `address` and
/// `columns` must be that CTA's own [`alloc_cluster`] result and argument.
#[inline(always)]
pub unsafe fn dealloc_cluster(address: u32, columns: u32) {
    unsafe {
        if warp::warp_id() == 0 {
            tcgen05_dealloc_cg2(address, columns);
        }
    }
}

/// Retire the calling warp's outstanding tcgen05 stores.
///
/// Split out of [`TmemTile::store_fragment`] rather than folded into it
/// because a store's registers are consumed at issue: a pass writing many
/// fragments waits once after the last one instead of once per fragment.
///
/// # Safety
///
/// All 32 lanes of the warp must call this together, and it retires only that
/// warp's stores — another warp reading the same TMEM needs its own ordering.
#[inline(always)]
pub unsafe fn store_wait() {
    tcgen05_store_wait()
}

/// Resolve a `16x256b` pair's eight registers into a fragment's
/// `(slot, value)` coordinates.
///
/// A `16x256b` load hands a thread its two rows of the block by two columns,
/// row-major: register `2*slot + pair` is slot `slot`'s column `pair` of that
/// load's 8-column half. The half at `column` is the fragment's values
/// `{0, 1}`, the half at `column + 8` its values `{2, 3}`.
#[inline(always)]
fn interleave(low: TmemRegs4, high: TmemRegs4) -> Fragment {
    let mut tile = Fragment::zero();
    let mut reg = 0;
    while reg < 4 {
        let (slot, pair) = (reg / 2, reg % 2);
        tile.set(slot, pair, low[reg]);
        tile.set(slot, 2 + pair, high[reg]);
        reg += 1;
    }
    tile
}

/// The inverse of [`interleave`] — a fragment back into the two collective
/// accesses' registers, low half first.
#[inline(always)]
fn split(tile: Fragment) -> (TmemRegs4, TmemRegs4) {
    let mut low = [0.0f32; 4];
    let mut high = [0.0f32; 4];
    let mut reg = 0;
    while reg < 4 {
        let (slot, pair) = (reg / 2, reg % 2);
        low[reg] = tile.get(slot, pair);
        high[reg] = tile.get(slot, 2 + pair);
        reg += 1;
    }
    (CuSimd::new(low), CuSimd::new(high))
}

/// fp32 registers as the raw words every `tcgen05_st_*` form takes. There is
/// no `_pure` store to mirror `tcgen05_ld_16x256b_pure`, and the `_unpack16`
/// form splits each register into 16-bit halves — an accumulator's fp32 has to
/// land bit-exact, so it goes through `_raw` and `to_bits`.
#[inline(always)]
fn raw_bits(values: TmemRegs4) -> CuSimd<u32, 4> {
    CuSimd::new([
        values[0].to_bits(),
        values[1].to_bits(),
        values[2].to_bits(),
        values[3].to_bits(),
    ])
}

/// Write `fragment` into the `[16, 16]` block at `(row_block, column_block)`
/// of a composed `[M, N]` tile.
///
/// A thread's two slots of a 16-row block are the composed tile's slots
/// `2 * row_block + {0, 1}`, and its four values of a 16-column block are its
/// values `4 * column_block + {0..4}`. That is not a convention the movers
/// chose: it is the same map the bigger shape's own [`RegTile::coordinate`]
/// gives, which is what `reg.rs`'s `fragment_blocks_tile_the_bigger_shapes`
/// asserts.
#[inline(always)]
pub(crate) fn place_block<const M: usize, const N: usize>(
    tile: &mut RegTile<M, N, BaseLdtm>,
    row_block: usize,
    column_block: usize,
    fragment: Fragment,
) where
    BaseLdtm: FragmentLayout<M, N>,
{
    let mut slot = 0usize;
    while slot < Fragment::SLOTS {
        let mut value = 0usize;
        while value < Fragment::VALUES {
            tile.set(
                2 * row_block + slot,
                4 * column_block + value,
                fragment.get(slot, value),
            );
            value += 1;
        }
        slot += 1;
    }
}

/// The inverse of [`place_block`]: one `[16, 16]` block of a composed tile, as
/// the [`Fragment`] a single-block mover takes.
#[inline(always)]
pub(crate) fn take_block<const M: usize, const N: usize>(
    tile: &RegTile<M, N, BaseLdtm>,
    row_block: usize,
    column_block: usize,
) -> Fragment
where
    BaseLdtm: FragmentLayout<M, N>,
{
    let mut fragment = Fragment::zero();
    let mut slot = 0usize;
    while slot < Fragment::SLOTS {
        let mut value = 0usize;
        while value < Fragment::VALUES {
            fragment.set(
                slot,
                value,
                tile.get(2 * row_block + slot, 4 * column_block + value),
            );
            value += 1;
        }
        slot += 1;
    }
    fragment
}

/// An fp32 accumulator segment in tensor memory.
#[derive(Clone, Copy)]
pub struct TmemTile<const R: usize, const C: usize> {
    address: u32,
}

impl<const R: usize, const C: usize> TmemTile<R, C> {
    /// Wrap a TMEM address (as returned through `tcgen05_alloc`'s shared
    /// staging word, plus any column offset already applied).
    pub const fn from_raw(address: u32) -> Self {
        Self { address }
    }

    /// The raw address, for the MMA issue path.
    pub const fn raw(self) -> u32 {
        self.address
    }

    /// Address of `(row, column)`: rows ride the high half-word.
    pub const fn at(self, row: u32, column: u32) -> u32 {
        self.address + (row << 16) + column
    }

    /// The same-shaped segment `columns` fp32 columns to the right —
    /// accumulator ping-pong stages (gemm's `accum_stage * 256`), or the
    /// N-half a second MMA of a wider allocation writes.
    pub const fn columns_right(self, columns: u32) -> Self {
        Self {
            address: self.address + columns,
        }
    }

    /// The next segment of the same allocation, `C2` columns wide: one
    /// `tcgen05_alloc` of `C + C2` columns carved into a `[R, C]` and a
    /// `[R, C2]` tile, as flash's `S` and `O` share one allocation.
    ///
    /// The offset is `C` rather than a parameter because a TMEM segment
    /// spans all 128 lanes — segments can only be carved along the column
    /// axis, so the one that follows this tile begins where it ends. An
    /// explicit offset could name a column inside this tile instead, and
    /// overlapping segments have no diagnostic: the two MMAs simply write
    /// each other's accumulator.
    pub const fn split_columns<const C2: usize>(self) -> TmemTile<R, C2> {
        TmemTile {
            address: self.address + C as u32,
        }
    }

    /// One thread's eight-value fragment of the 16-row block at `row`:
    /// two `16x256b` collective loads at `column` and `column + 8`, each
    /// drained through `tcgen05_load_wait`. Under the 16x256b map a thread
    /// holds rows `lane/4` and `lane/4 + 8` of the block at column offsets
    /// `2*(lane%4) + {0, 1, 8, 9}` — the low simd carries the `{0, 1}`
    /// offsets of both rows, the high simd the `{8, 9}` offsets (which is
    /// what [`Self::fragment_tile`]'s interleaving spells out).
    ///
    /// # Safety
    ///
    /// All 32 lanes of a warp that owns the TMEM rows `row..row+16` must
    /// call this together, after the MMA writing them has committed.
    #[inline(always)]
    pub unsafe fn fragment(self, row: u32, column: u32) -> (TmemRegs4, TmemRegs4) {
        unsafe {
            let low = tcgen05_ld_16x256b_pure(self.at(row, column));
            tcgen05_load_wait();
            let high = tcgen05_ld_16x256b_pure(self.at(row, column + 8));
            tcgen05_load_wait();
            (low, high)
        }
    }

    /// [`Self::fragment`] as a [`Fragment`] tile — the same eight values with
    /// the simd pair's interleaving resolved into the layout's `(slot, value)`
    /// coordinates, so a register pass indexes by row and column instead of by
    /// which of the two collective loads a value arrived in.
    ///
    /// A `16x256b` load hands a thread its two rows of the block by two
    /// columns, row-major: register `2*slot + pair` is slot `slot`'s column
    /// `pair` of that load's 8-column half. The half at `column` is the
    /// fragment's values `{0, 1}`, the half at `column + 8` its values
    /// `{2, 3}` — the `{0, 1, 8, 9}` offsets of
    /// [`BaseLdtm::column`](crate::reg::BaseLdtm::column).
    ///
    /// # Safety
    ///
    /// As [`Self::fragment`].
    #[inline(always)]
    pub unsafe fn fragment_tile(self, row: u32, column: u32) -> Fragment {
        unsafe {
            let (low, high) = self.fragment(row, column);
            interleave(low, high)
        }
    }

    /// Write one thread's eight-value fragment back into the 16-row block at
    /// `row`: two `16x256b` collective stores at `column` and `column + 8`,
    /// under the same lane → `(row, column)` ownership the load side of
    /// [`Self::fragment`] reads through, so `low` and `high` are exactly what
    /// that call returned.
    ///
    /// Unlike [`Self::fragment`] this does not wait. A load must, because the
    /// registers it waits on *are* its return value; a store's registers are
    /// consumed at issue, so a pass writing many fragments pays one
    /// [`store_wait`] at the end rather than one per fragment.
    ///
    /// # Safety
    ///
    /// All 32 lanes of the warp owning TMEM rows `row..row+16` must call this
    /// together. Ordering the write against whatever reads it — an MMA taking
    /// the segment as accumulator, a later [`Self::fragment`], or
    /// [`dealloc_block`] — is the caller's: [`store_wait`] retires it in the
    /// issuing warp, and a consumer in another warp needs whatever sync it
    /// would need for any warp-private write besides.
    ///
    /// Whether a `tcgen05_fence_before_thread_sync` /
    /// `tcgen05_fence_after_thread_sync` pair is additionally required around
    /// such a hand-off is open: this crate uses neither on any path, and
    /// nothing has been established either way for the store side. Their
    /// absence here is not a guarantee that they are unnecessary.
    #[inline(always)]
    pub unsafe fn store_fragment(self, row: u32, column: u32, low: TmemRegs4, high: TmemRegs4) {
        unsafe {
            tcgen05_st_16x256b_x1_raw(self.at(row, column), raw_bits(low));
            tcgen05_st_16x256b_x1_raw(self.at(row, column + 8), raw_bits(high));
        }
    }

    /// [`Self::store_fragment`] from a [`Fragment`] tile — the store twin of
    /// [`Self::fragment_tile`], undoing the simd pair's interleaving so a
    /// register pass hands back the same `(slot, value)` coordinates it read.
    ///
    /// Only the `[16, 16]` block; [`Self::store_tile`] is the composed form.
    ///
    /// # Safety
    ///
    /// As [`Self::store_fragment`], including the wait it does not do.
    #[inline(always)]
    pub unsafe fn store_fragment_tile(self, row: u32, column: u32, tile: Fragment) {
        unsafe {
            let (low, high) = split(tile);
            self.store_fragment(row, column, low, high);
        }
    }

    /// A whole `[M, N]` band at `(row, column)`, composed out of the
    /// `M/16 × N/16` blocks [`Self::fragment_tile`] returns — the drain a warp
    /// holding a `RegTile<32, 128, BaseLdtm>` accumulator wants, rather than
    /// the four-deep block loop it used to write for itself.
    ///
    /// `M` and `N` are the *warp's* logical shape and `row` its first TMEM
    /// lane, so a warp of an `M128` accumulator drains `tile::<32, N>(32 *
    /// warp_id, 0)`. The shape set is the one [`BaseLdtm`] implements
    /// [`FragmentLayout`] for.
    ///
    /// # Safety
    ///
    /// As [`Self::fragment`], for every block of the band: all 32 lanes of a
    /// warp owning TMEM rows `row..row + M` must call this together after the
    /// MMA writing them has committed, and `column + N` must fit the
    /// allocation.
    #[inline(always)]
    pub unsafe fn tile<const M: usize, const N: usize>(
        self,
        row: u32,
        column: u32,
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
                        self.fragment_tile(
                            row + 16 * row_block as u32,
                            column + 16 * column_block as u32,
                        ),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            tile
        }
    }

    /// The inverse of [`Self::tile`]: a whole `[M, N]` band written back
    /// block by block through [`Self::store_fragment_tile`].
    ///
    /// The stores are left outstanding, as the single-block form leaves them —
    /// one [`store_wait`] retires the whole band.
    ///
    /// # Safety
    ///
    /// As [`Self::store_fragment`], for every block of the band, including the
    /// wait it does not do.
    #[inline(always)]
    pub unsafe fn store_tile<const M: usize, const N: usize>(
        self,
        row: u32,
        column: u32,
        tile: RegTile<M, N, BaseLdtm>,
    ) where
        BaseLdtm: FragmentLayout<M, N>,
    {
        unsafe {
            let mut row_block = 0usize;
            while row_block < M / 16 {
                let mut column_block = 0usize;
                while column_block < N / 16 {
                    self.store_fragment_tile(
                        row + 16 * row_block as u32,
                        column + 16 * column_block as u32,
                        take_block(&tile, row_block, column_block),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addressing_rides_the_high_half_word() {
        let tile = TmemTile::<128, 64>::from_raw(0x0001_0000);
        assert_eq!(tile.at(0, 0), 0x0001_0000);
        assert_eq!(tile.at(32, 24), 0x0021_0018);
        assert_eq!(tile.columns_right(64).at(0, 0), 0x0001_0040);
    }

    /// Flash's carve: one `[128, 64 + 128]` allocation as a score segment
    /// and an output segment beside it. The output's first column has to be
    /// the score's last plus one, in both directions — a short offset
    /// overlaps the scores and a long one runs off the allocation, and TMEM
    /// reports neither.
    #[test]
    fn split_columns_carves_an_allocation_without_overlap() {
        const KEYS: usize = 64;
        const HEAD: usize = 128;

        let base = 0x0000_0000;
        let scores = TmemTile::<128, KEYS>::from_raw(base);
        let output: TmemTile<128, HEAD> = scores.split_columns();

        assert_eq!(output.at(0, 0), scores.at(0, KEYS as u32 - 1) + 1);
        assert_eq!(
            output.at(127, HEAD as u32 - 1),
            base + (127 << 16) + (KEYS + HEAD) as u32 - 1
        );

        // A third segment chains off the second, not off the first.
        let spare = output.split_columns::<32>();
        assert_eq!(spare.at(0, 0), base + (KEYS + HEAD) as u32);
    }

    /// The simd-pair interleaving, both directions. `split` is what the store
    /// path hands the hardware and `interleave` what the drain path reads back
    /// from it, so a fragment that survives the pair unchanged is the whole
    /// register-side contract between them — the addressing either shares is
    /// what the device harness is for.
    #[test]
    fn splitting_a_fragment_inverts_the_pair_interleaving() {
        let mut tile = Fragment::zero();
        for slot in 0..Fragment::SLOTS {
            for value in 0..Fragment::VALUES {
                tile.set(slot, value, (Fragment::VALUES * slot + value) as f32);
            }
        }

        let (low, high) = split(tile);
        for reg in 0..4 {
            assert_eq!(low[reg], tile.get(reg / 2, reg % 2));
            assert_eq!(high[reg], tile.get(reg / 2, 2 + reg % 2));
        }

        let restored = interleave(low, high);
        for slot in 0..Fragment::SLOTS {
            for value in 0..Fragment::VALUES {
                assert_eq!(restored.get(slot, value), tile.get(slot, value));
            }
        }
    }

    /// The block composition, against the coordinate map rather than against
    /// itself: a fragment placed at `(row_block, column_block)` must land in
    /// the registers whose `RegTile::coordinate` is that block's `(row,
    /// column)` offset by `16 * (row_block, column_block)`, and must come back
    /// out of [`take_block`] unchanged.
    ///
    /// `reg.rs` asserts that map is what the bigger shape's own `coordinate`
    /// gives; this is what ties the movers' storage indexing to it, in both
    /// directions, so a composed band cannot be a transposition of the drain
    /// that filled it.
    #[test]
    fn block_placement_is_the_composed_coordinate() {
        type Band = RegTile<32, 128, BaseLdtm>;

        let mut fragment = Fragment::zero();
        for slot in 0..Fragment::SLOTS {
            for value in 0..Fragment::VALUES {
                fragment.set(slot, value, (Fragment::VALUES * slot + value) as f32 + 1.0);
            }
        }

        for row_block in 0..32 / 16 {
            for column_block in 0..128 / 16 {
                let mut band = Band::zero();
                place_block(&mut band, row_block, column_block, fragment);

                for lane in 0..32u32 {
                    for slot in 0..Fragment::SLOTS {
                        for value in 0..Fragment::VALUES {
                            let (row, column) = Fragment::coordinate(lane, slot, value);
                            let composed = (2 * row_block + slot, 4 * column_block + value);
                            assert_eq!(
                                Band::coordinate(lane, composed.0, composed.1),
                                (
                                    16 * row_block as u32 + row,
                                    16 * column_block as u32 + column
                                )
                            );
                            assert_eq!(band.get(composed.0, composed.1), fragment.get(slot, value));
                        }
                    }
                }

                let taken = take_block(&band, row_block, column_block);
                for slot in 0..Fragment::SLOTS {
                    for value in 0..Fragment::VALUES {
                        assert_eq!(taken.get(slot, value), fragment.get(slot, value));
                    }
                }
            }
        }
    }

    #[test]
    fn raw_bits_is_the_fp32_pattern_unchanged() {
        let values = TmemRegs4::new([0.0, -0.0, 1.5, f32::from_bits(0x0000_8001)]);
        let bits = raw_bits(values);
        for reg in 0..4 {
            assert_eq!(bits[reg], values[reg].to_bits());
        }
    }
}
