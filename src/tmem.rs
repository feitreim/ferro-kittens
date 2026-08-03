//! TMEM accumulator views.
//!
//! A tcgen05 accumulator lives in tensor memory, addressed as
//! `base + (row << 16) + column`: the high half-word selects one of the 128
//! TMEM lanes (accumulator rows), the low half-word a 4-byte column.
//! [`TmemTile`] carries that address plus the logical `[R, C]` fp32 shape, so
//! kernel code names its segments as tiles instead of threading bare `u32`
//! addresses through every drain loop.
//!
//! Two things a caller owes:
//!
//! - **The columns must go back.** A CTA that exits still holding them is a
//!   `CUDA_ERROR_TENSOR_MEMORY_LEAK`, and the next CTA on that SM is the one
//!   that pays.
//! - **How many columns you ask for is how many CTAs fit.** An SM has 512, so
//!   `512 / columns` CTAs hold allocations on it at once — a ceiling the shared
//!   plan may already be sitting under.
//!
//! MMA shapes with phantom rows (`M128` over 64-row tiles) still type the
//! *drained* shape: `R`/`C` describe what the kernel reads back, not what the
//! instruction touches.
//!
//! Allocate, multiply, drain, release:
//!
//! ```no_run
//! # use cuda_device::thread;
//! # use kittens::mma::{MmaShape, commit, mm_abt};
//! # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
//! # use kittens::tmem::{TmemTile, alloc_block, dealloc_block};
//! # use kittens::{BaseLdtm, RegTile, Semaphore, warp_id};
//! # unsafe fn demo(
//! #     slot: *mut u32,
//! #     a: SharedTile<Bf16, 128, 64, Swizzle128B>,
//! #     b: SharedTile<Bf16, 128, 64, Swizzle128B>,
//! #     done: Semaphore,
//! # ) { unsafe {
//! const COLUMNS: u32 = 128;
//! let accumulator = TmemTile::<128, 128>::from_raw(alloc_block(slot, COLUMNS));
//! if thread::threadIdx_x() == 0 {
//!     mm_abt(accumulator.raw(), a, b, MmaShape::M128_N128);
//!     commit(done);
//! }
//! done.wait(0);
//! let band: RegTile<32, 128, BaseLdtm> = accumulator.tile(32 * warp_id(), 0);
//! thread::sync_threads();
//! dealloc_block(accumulator.raw(), COLUMNS);
//! # let _ = band;
//! # } }
//! ```
//!
//! `docs/library/tmem.md` has the residency census, the occupancy-query trap
//! that must not be used to predict it, and what `.x8` batching measures at.

use cuda_device::cusimd::{CuSimd, TmemRegs4};
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_alloc_cg2, tcgen05_dealloc, tcgen05_dealloc_cg2,
    tcgen05_ld_16x256b_pure, tcgen05_ld_16x256b_x8_pack16, tcgen05_ld_16x256b_x8_pure,
    tcgen05_load_wait, tcgen05_relinquish_alloc_permit, tcgen05_relinquish_alloc_permit_cg2,
    tcgen05_st_16x256b_x1_raw, tcgen05_store_wait,
};
use cuda_device::thread::__unroll_config;
use cuda_device::{cluster, thread};

use crate::reg::{BaseLdtm, Fragment, FragmentLayout, RegTile};

/// Take `columns` of this CTA's tensor memory and return their base address.
///
/// Warp 0 allocates into the shared staging word, a block sync publishes it,
/// and every thread reads the address back. That warp also gives up the CTA's
/// *allocation permit* — the right to call the allocator again, which is a
/// different thing from giving back the columns ([`dealloc_block`]'s job).
///
/// ```no_run
/// # use kittens::tmem::{TmemTile, alloc_block, dealloc_block};
/// # unsafe fn demo(slot: *mut u32) { unsafe {
/// const COLUMNS: u32 = 256;
/// let accumulator = TmemTile::<128, 256>::from_raw(alloc_block(slot, COLUMNS));
/// // ... MMA into `accumulator`, drain it ...
/// dealloc_block(accumulator.raw(), COLUMNS);
/// # } }
/// ```
///
/// # Safety
///
/// - Every thread of the block calls this together.
/// - At most once per CTA: allocating again after the permit is relinquished is
///   illegal.
/// - `slot` points to shared memory (a `SharedArray<u32, 1, 4>` static).
#[inline(always)]
pub unsafe fn alloc_block(slot: *mut u32, columns: u32) -> u32 {
    unsafe {
        if crate::warp_id() == 0 {
            tcgen05_alloc(slot, columns);
            tcgen05_relinquish_alloc_permit();
        }
        thread::sync_threads();
        *(slot as *const u32)
    }
}

/// Give the CTA's columns back, from warp 0.
///
/// Retiring the outstanding reads first is the caller's: at block scope that is
/// `tcgen05_fence_before_thread_sync()` immediately in front of a
/// `sync_threads()`, the same pair [`dealloc_cluster`] shows at cluster scope.
///
/// # Safety
///
/// - The caller's fence/sync has retired every outstanding read of the
///   allocation, and no thread touches it afterwards.
/// - `address` and `columns` are the [`alloc_block`] result and argument.
#[inline(always)]
pub unsafe fn dealloc_block(address: u32, columns: u32) {
    unsafe {
        if crate::warp_id() == 0 {
            tcgen05_dealloc(address, columns);
        }
    }
}

/// [`alloc_block`]'s `cta_group::2` twin: the single allocation a 2-CTA MMA
/// accumulates into.
///
/// Both peers drive the allocator — both allocate, both relinquish, both later
/// [`dealloc_cluster`] — and each reads the address out of *its own* staging
/// word, because the collective writes one into each. The `cluster_sync` inside
/// is what publishes those words across the pair, so a caller that stages
/// barriers for its peer before allocating gets that publication for free.
///
/// Each rank is charged `columns` against **its own SM's 512**, so the pair
/// costs residency exactly as two independent [`alloc_block`] calls would; it is
/// not a half-share of one SM's tensor memory.
///
/// ```no_run
/// # use kittens::tmem::{TmemTile, alloc_cluster};
/// # unsafe fn demo(slot: *mut u32) { unsafe {
/// const COLUMNS: u32 = 256;
/// let pair_accumulator = TmemTile::<128, 256>::from_raw(alloc_cluster(slot, COLUMNS));
/// # let _ = pair_accumulator;
/// # } }
/// ```
///
/// # Safety
///
/// - Every thread of every CTA in the cluster calls this together, at most once.
/// - `slot` is at the same shared offset in both CTAs.
#[inline(always)]
pub unsafe fn alloc_cluster(slot: *mut u32, columns: u32) -> u32 {
    unsafe {
        if crate::warp_id() == 0 {
            tcgen05_alloc_cg2(slot, columns);
        }
        thread::sync_threads();
        cluster::cluster_sync();
        if crate::warp_id() == 0 {
            tcgen05_relinquish_alloc_permit_cg2();
        }
        *(slot as *const u32)
    }
}

/// Give the cluster's columns back, warp 0 of each peer CTA taking its half.
///
/// A tcgen05 read — an LDTM drain, an MMA — is retired against a thread
/// rendezvous by `tcgen05_fence_before_thread_sync()` issued *immediately* in
/// front of it. For a `cta_group::2` allocation that rendezvous is
/// `cluster_sync`, because the reads being retired live in two CTAs, and both
/// lines are the caller's:
///
/// ```no_run
/// # use cuda_device::cluster;
/// # use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
/// # use kittens::tmem::{TmemTile, dealloc_cluster};
/// # unsafe fn demo(accumulator: TmemTile<128, 256>) { unsafe {
/// # const COLUMNS: u32 = 256;
/// tcgen05_fence_before_thread_sync();
/// cluster::cluster_sync();
/// dealloc_cluster(accumulator.raw(), COLUMNS);
/// # } }
/// ```
///
/// [`crate::pipeline::run`] performs exactly this pair at every item boundary,
/// so a job whose accumulator dies with the item loop has already had it.
///
/// # Safety
///
/// - The fence/rendezvous pair above has retired the pair's reads, and no
///   thread of either CTA touches the allocation afterwards.
/// - `address` and `columns` are *that* CTA's own [`alloc_cluster`] result and
///   argument.
#[inline(always)]
pub unsafe fn dealloc_cluster(address: u32, columns: u32) {
    unsafe {
        if crate::warp_id() == 0 {
            tcgen05_dealloc_cg2(address, columns);
        }
    }
}

/// Retire the calling warp's outstanding tcgen05 stores.
///
/// Separate from [`TmemTile::store_fragment`] because a store's registers are
/// consumed at issue: a pass writing many fragments waits once at the end
/// instead of once per fragment.
///
/// ```no_run
/// # use kittens::tmem::{TmemTile, store_wait};
/// # use kittens::{BaseLdtm, RegTile, warp_id};
/// # unsafe fn demo(accumulator: TmemTile<128, 128>, band: RegTile<32, 128, BaseLdtm>) { unsafe {
/// accumulator.store_tile(32 * warp_id(), 0, band);
/// store_wait();
/// # } }
/// ```
///
/// # Safety
///
/// - All 32 lanes of the warp call this together.
/// - It retires only *that* warp's stores; another warp reading the same TMEM
///   needs its own ordering.
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

/// Resolve one `16x256b.x8` load's 32 registers into the four `[16, 16]`
/// blocks they cover.
///
/// `.x8` is eight base accesses laid end to end along the *column* axis, and
/// the register list is repeat-major: repeat `r` is what
/// [`tcgen05_ld_16x256b_pure`] would have returned at `column + 8r`, so
/// [`interleave`]'s low/high pair for block `j` is repeats `2j` and `2j + 1` —
/// register `8j + 4*(value/2) + 2*slot + value%2`.
///
/// That ordering is a claim about silicon rather than the ISA text;
/// `device-tests`' `ldtm x8 map` case is what holds it.
#[inline(always)]
fn interleave_x8(regs: CuSimd<f32, 32>) -> [Fragment; 4] {
    let mut blocks = [Fragment::zero(); 4];
    let mut block = 0usize;
    while block < 4 {
        let mut slot = 0usize;
        while slot < Fragment::SLOTS {
            let mut value = 0usize;
            while value < Fragment::VALUES {
                blocks[block].set(
                    slot,
                    value,
                    regs[8 * block + 4 * (value / 2) + 2 * slot + value % 2],
                );
                value += 1;
            }
            slot += 1;
        }
        block += 1;
    }
    blocks
}

/// `.x8` loads [`TmemTile::tile_x8_batched`] will hold in flight at once.
///
/// The bound is the register file: each load is 32 f32 a thread, so four is 128
/// — the widest band a warp can drain in one pass at all.
pub const ISSUE_LIMIT: usize = 4;

/// One `.x8` arrival into its four `[16, 16]` blocks of a band, at the block
/// coordinates issue number `issue` covers.
///
/// Split out of [`TmemTile::tile_x8_batched`] so the batch can name its issues
/// at literal indices — see that method on why a loop is not equivalent.
#[inline(always)]
fn place_x8_group<const M: usize, const N: usize>(
    tile: &mut RegTile<M, N, BaseLdtm>,
    issue: usize,
    arrival: CuSimd<f32, 32>,
) where
    BaseLdtm: FragmentLayout<M, N>,
{
    let groups = N / 64;
    let (row_block, group) = (issue / groups, issue % groups);
    let blocks = interleave_x8(arrival);
    let mut block = 0usize;
    while block < 4 {
        place_block(tile, row_block, 4 * group + block, blocks[block]);
        block += 1;
    }
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

/// fp32 registers as the raw words every `tcgen05_st_*` form takes.
///
/// There is no `_pure` store, and `_unpack16` splits each register into 16-bit
/// halves; an accumulator's fp32 has to land bit-exact, so it goes through
/// `_raw` and `to_bits`.
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
/// values `4 * column_block + {0..4}` — the same map the bigger shape's own
/// [`RegTile::coordinate`] gives.
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
    /// allocation of `C + C2` columns carved into a `[R, C]` and a `[R, C2]`
    /// tile, as flash's scores and output share one.
    ///
    /// There is no offset parameter. A TMEM segment spans all 128 lanes, so
    /// segments can only be carved along the column axis and the next one begins
    /// exactly where this tile ends; an explicit offset could name a column
    /// *inside* this tile, and overlapping segments have no diagnostic — the two
    /// MMAs simply write each other's accumulator.
    ///
    /// ```no_run
    /// # use kittens::tmem::{TmemTile, alloc_block};
    /// # unsafe fn demo(slot: *mut u32) { unsafe {
    /// const KEYS: usize = 64;
    /// const HEAD: usize = 128;
    /// let scores = TmemTile::<128, KEYS>::from_raw(alloc_block(slot, (KEYS + HEAD) as u32));
    /// let output: TmemTile<128, HEAD> = scores.split_columns();
    /// # let _ = output;
    /// # } }
    /// ```
    pub const fn split_columns<const C2: usize>(self) -> TmemTile<R, C2> {
        TmemTile {
            address: self.address + C as u32,
        }
    }

    /// One thread's eight-value fragment of the 16-row block at `row`, as the
    /// two `16x256b` collective loads it arrives in.
    ///
    /// Under the `16x256b` map a thread holds rows `lane/4` and `lane/4 + 8` of
    /// the block at column offsets `2*(lane%4) + {0, 1, 8, 9}`: the low simd
    /// carries the `{0, 1}` offsets of both rows, the high simd the `{8, 9}`.
    /// [`Self::fragment_tile`] resolves that into `(slot, value)` coordinates.
    ///
    /// # Safety
    ///
    /// - All 32 lanes of the warp owning TMEM rows `row..row + 16` call this
    ///   together.
    /// - The MMA writing those rows has committed.
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

    /// [`Self::fragment`] as a [`Fragment`] tile — the same eight values indexed
    /// by `(slot, value)` instead of by which collective load they arrived in.
    ///
    /// The half at `column` is the fragment's values `{0, 1}`, the half at
    /// `column + 8` its values `{2, 3}` — the `{0, 1, 8, 9}` offsets of
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
    /// `row`, under the same lane → `(row, column)` ownership [`Self::fragment`]
    /// reads through: `low` and `high` are exactly what that call returned.
    ///
    /// Does **not** wait. [`store_wait`] retires it, once per pass.
    ///
    /// # Safety
    ///
    /// - All 32 lanes of the warp owning TMEM rows `row..row + 16` call this
    ///   together.
    /// - Ordering the write against whatever reads it — an MMA taking the
    ///   segment as accumulator, a later [`Self::fragment`], or
    ///   [`dealloc_block`] — is the caller's. [`store_wait`] retires it in the
    ///   issuing warp; a consumer in another warp needs whatever sync any
    ///   warp-private write would need besides.
    ///
    /// Whether a `tcgen05_fence_before_thread_sync` /
    /// `tcgen05_fence_after_thread_sync` pair is additionally required around
    /// such a hand-off is **open**. This crate uses neither on any path, and
    /// that is not a guarantee that they are unnecessary.
    #[inline(always)]
    pub unsafe fn store_fragment(self, row: u32, column: u32, low: TmemRegs4, high: TmemRegs4) {
        unsafe {
            tcgen05_st_16x256b_x1_raw(self.at(row, column), raw_bits(low));
            tcgen05_st_16x256b_x1_raw(self.at(row, column + 8), raw_bits(high));
        }
    }

    /// [`Self::store_fragment`] from a [`Fragment`] tile — the store twin of
    /// [`Self::fragment_tile`], taking back the same `(slot, value)`
    /// coordinates it read.
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
    /// `M/16 × N/16` blocks [`Self::fragment_tile`] returns.
    ///
    /// `M` and `N` are the *warp's* logical shape and `row` its first TMEM lane,
    /// so each warp of an `M128` accumulator drains its own quarter. The shape
    /// set is the one [`BaseLdtm`] implements [`FragmentLayout`] for.
    ///
    /// ```no_run
    /// # use kittens::tmem::TmemTile;
    /// # use kittens::{BaseLdtm, RegTile, warp_id};
    /// # unsafe fn demo(accumulator: TmemTile<128, 128>) { unsafe {
    /// let band: RegTile<32, 128, BaseLdtm> = accumulator.tile(32 * warp_id(), 0);
    /// # let _ = band;
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// - All 32 lanes of the warp owning TMEM rows `row..row + M` call this
    ///   together, after the MMA writing them has committed.
    /// - `column + N` fits the allocation.
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
            // Both walks fully unrolled, or `tile`'s indices stay dynamic,
            // SROA cannot split it, and the aggregate is homed to a `.local`
            // depot (#166). LLVM stops unrolling on its own near 32 fp32
            // columns.
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

    /// The four `[16, 16]` blocks at `(row, column..column + 64)` in **one**
    /// LDTM, against the eight [`Self::fragment`] issues and eight waits that
    /// cover the same 64 columns.
    ///
    /// Same bytes, same map. What it costs is liveness: 32 f32 arrive at once
    /// and all four blocks stay live until the caller consumes them, where the
    /// `.x1` path lets the compiler fuse a single eight-value [`Fragment`]
    /// through to the store.
    ///
    /// # Safety
    ///
    /// - All 32 lanes of the warp owning TMEM rows `row..row + 16` call this
    ///   together, after the MMA writing them has committed.
    /// - `column + 64` fits the allocation.
    #[inline(always)]
    pub unsafe fn fragments_x8(self, row: u32, column: u32) -> [Fragment; 4] {
        unsafe {
            let regs = tcgen05_ld_16x256b_x8_pure(self.at(row, column));
            tcgen05_load_wait();
            interleave_x8(regs)
        }
    }

    /// One `.x8.pack::16b` arrival: 32 registers of **already-packed 16-bit
    /// pairs**, off twice the columns [`Self::fragments_x8`] covers.
    ///
    /// `.pack::16b` is a *reinterpretation*, not a conversion — **it is not a
    /// cheaper epilogue.** Over an fp32 accumulator it yields the mantissa
    /// halves of pairs of floats, which is not a narrowed value of anything.
    /// What it is good for is pricing the convert; see
    /// `docs/library/tmem.md`.
    ///
    /// # Safety
    ///
    /// - As [`Self::fragments_x8`].
    /// - The values are meaningful only if the segment holds 16-bit data.
    #[inline(always)]
    pub unsafe fn fragments_pack16_x8(self, row: u32, column: u32) -> CuSimd<u32, 32> {
        unsafe {
            let regs = tcgen05_ld_16x256b_x8_pack16(self.at(row, column));
            tcgen05_load_wait();
            regs
        }
    }

    /// [`Self::tile`] over [`Self::fragments_x8`] — the same `[M, N]` band,
    /// drained 64 columns at a time instead of 16.
    ///
    /// `N` must be a multiple of 64, which is what makes the `.x8` land on whole
    /// blocks; the shape set is otherwise [`Self::tile`]'s. The two return the
    /// same tile for every shape both accept, and `device-tests`' `ldtm x8 map`
    /// is that assertion.
    ///
    /// ```no_run
    /// # use kittens::tmem::TmemTile;
    /// # use kittens::{BaseLdtm, RegTile, warp_id};
    /// # unsafe fn demo(accumulator: TmemTile<128, 128>) { unsafe {
    /// let band: RegTile<32, 128, BaseLdtm> = accumulator.tile_x8(32 * warp_id(), 0);
    /// # let _ = band;
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// As [`Self::fragments_x8`], for every 64-column group of the band.
    #[inline(always)]
    pub unsafe fn tile_x8<const M: usize, const N: usize>(
        self,
        row: u32,
        column: u32,
    ) -> RegTile<M, N, BaseLdtm>
    where
        BaseLdtm: FragmentLayout<M, N>,
    {
        const {
            assert!(
                N.is_multiple_of(64),
                "tcgen05.ld.16x256b.x8 covers 64 columns; a band it drains has to be a multiple of that"
            )
        };
        unsafe {
            let mut tile = RegTile::<M, N, BaseLdtm>::zero();
            // Fully unrolled for `tile` and `blocks`' sake — see `tile` (#166).
            let mut row_block = 0usize;
            while row_block < const { M / 16 } {
                __unroll_config::<0>();
                let mut group = 0usize;
                while group < const { N / 64 } {
                    __unroll_config::<0>();
                    let blocks =
                        self.fragments_x8(row + 16 * row_block as u32, column + 64 * group as u32);
                    let mut block = 0usize;
                    while block < 4 {
                        __unroll_config::<0>();
                        place_block(&mut tile, row_block, 4 * group + block, blocks[block]);
                        block += 1;
                    }
                    group += 1;
                }
                row_block += 1;
            }
            tile
        }
    }

    /// [`Self::tile_x8`] with **every issue of the band in flight before the one
    /// wait**, rather than a wait per issue.
    ///
    /// `ISSUES` is the band's `.x8` count, `(M / 16) * (N / 64)`, stated rather
    /// than derived because a generic const expression is not on stable; the
    /// const asserts keep a caller from stating it wrong. At most
    /// [`ISSUE_LIMIT`], which is the register file and not a convenience.
    ///
    /// Two issues behind one wait cost no registers and are worth **+0.9%** of
    /// `gemm_sol`'s launch at 8192³; four cost 176 registers against 96 and are
    /// a loss. `docs/library/tmem.md` has the ladder, and why the issues are
    /// spelled out rather than looped over.
    ///
    /// ```no_run
    /// # use kittens::tmem::TmemTile;
    /// # use kittens::{BaseLdtm, RegTile, warp_id};
    /// # unsafe fn demo(accumulator: TmemTile<128, 128>) { unsafe {
    /// // [32, 128] is 2 row blocks x 2 column groups = 4 issues.
    /// let band: RegTile<32, 128, BaseLdtm> =
    ///     accumulator.tile_x8_batched::<32, 128, 4>(32 * warp_id(), 0);
    /// # let _ = band;
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// As [`Self::tile_x8`]: all 32 lanes of the warp owning TMEM rows
    /// `row..row + M` call this together after the MMA writing them has
    /// committed, and `column + N` fits the allocation.
    #[inline(always)]
    pub unsafe fn tile_x8_batched<const M: usize, const N: usize, const ISSUES: usize>(
        self,
        row: u32,
        column: u32,
    ) -> RegTile<M, N, BaseLdtm>
    where
        BaseLdtm: FragmentLayout<M, N>,
    {
        const {
            assert!(
                N.is_multiple_of(64),
                "tcgen05.ld.16x256b.x8 covers 64 columns; a band it drains has to be a multiple of that"
            );
            assert!(
                ISSUES == (M / 16) * (N / 64),
                "ISSUES is the band's `.x8` count: one per 16 rows per 64 columns"
            );
            assert!(
                ISSUES <= ISSUE_LIMIT,
                "a batch is at most ISSUE_LIMIT x8 loads: more than that is past the register file"
            );
        };
        unsafe {
            let address = |issue: usize| {
                let groups = N / 64;
                self.at(
                    row + 16 * (issue / groups) as u32,
                    column + 64 * (issue % groups) as u32,
                )
            };
            let mut arrivals = [CuSimd::<f32, 32>::new([0.0; 32]); ISSUE_LIMIT];
            if ISSUES > 0 {
                arrivals[0] = tcgen05_ld_16x256b_x8_pure(address(0));
            }
            if ISSUES > 1 {
                arrivals[1] = tcgen05_ld_16x256b_x8_pure(address(1));
            }
            if ISSUES > 2 {
                arrivals[2] = tcgen05_ld_16x256b_x8_pure(address(2));
            }
            if ISSUES > 3 {
                arrivals[3] = tcgen05_ld_16x256b_x8_pure(address(3));
            }
            tcgen05_load_wait();

            let mut tile = RegTile::<M, N, BaseLdtm>::zero();
            if ISSUES > 0 {
                place_x8_group(&mut tile, 0, arrivals[0]);
            }
            if ISSUES > 1 {
                place_x8_group(&mut tile, 1, arrivals[1]);
            }
            if ISSUES > 2 {
                place_x8_group(&mut tile, 2, arrivals[2]);
            }
            if ISSUES > 3 {
                place_x8_group(&mut tile, 3, arrivals[3]);
            }
            tile
        }
    }

    /// The inverse of [`Self::tile`]: a whole `[M, N]` band written back block
    /// by block through [`Self::store_fragment_tile`].
    ///
    /// The stores are left outstanding — one [`store_wait`] retires the band.
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
            // Fully unrolled for `tile`'s sake — see `Self::tile` (#166).
            let mut row_block = 0usize;
            while row_block < const { M / 16 } {
                __unroll_config::<0>();
                let mut column_block = 0usize;
                while column_block < const { N / 16 } {
                    __unroll_config::<0>();
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
    /// path hands the hardware and `interleave` what the drain path reads back,
    /// so a fragment that survives the pair unchanged is the whole
    /// register-side contract between them; the addressing either shares is
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
    /// out of [`take_block`] unchanged — so a composed band cannot be a
    /// transposition of the drain that filled it.
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
