//! # GEMM — `C = A·Bᵀ` on the cta_group::2 cluster path
//!
//! **Status: compiles.** Builds under `cargo oxide build kittens-examples --arch
//! sm_100a`. It has never been run: issue #18 needs no GPU, so this is a
//! statement about what the library can *express*, not a numerical result.
//!
//! A pair of CTAs forms a cluster and shares one `M256_N128` UMMA. Both
//! operands are split across the pair — each CTA stages its own 128 rows of
//! `A` and its own 64 columns of `B` at the *same* shared offsets, and the
//! instruction reads both CTAs' shared memory over the cluster interconnect.
//! The accumulator splits the same way along M, so each CTA drains its own
//! `[128, 128]` band of `C`.
//!
//! K is software-pipelined `STAGES` deep over a [`SharedTileRing`] pair, with
//! [`SemaphoreRing`] owning the `index → (stage, parity)` arithmetic on both
//! sides: `load` is filled by the TMA and drained by the MMA, `free` is
//! released by the MMA's own commit — the accumulator instruction, not a
//! thread, is what proves the operand has been read.
//!
//! ## What this kernel had to reach past the library for
//!
//! Three `GAP` blocks below, none of which has an open issue behind it:
//!
//! - **cluster-scope TMEM allocation.** [`kittens::tmem::alloc_block`] is
//!   `tcgen05.alloc.cta_group::1`; a 2-CTA accumulator needs the `cg2` form
//!   plus a way for the peer to learn the address.
//! - **cluster-scope semaphore arrival.** The pair's four TMA loads have to
//!   land on *one* barrier for the leader to know the whole stage is present,
//!   and [`Semaphore`] is CTA-scoped by construction.
//! - **global stores from registers.** The epilogue is open-coded index math
//!   against `RegTile::coordinate`, which is the arithmetic this library
//!   exists to delete (#11).
//!
//! And one thing the library has that this kernel *cannot* use:
//! [`SharedTile::tma_load_2d_multicast_cg2`] delivers the same bytes to both
//! CTAs, but under a 2-CTA UMMA both operands are already split, so nothing
//! is replicated. Multicast pays off at cluster > 2 (GAPS §2.4), which the
//! `_cg2` suffix rules out. See `examples/README.md`.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::cluster;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::{
    tcgen05_alloc_cg2, tcgen05_dealloc_cg2, tcgen05_fence_before_thread_sync,
};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, thread, warp};

use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::reg::{BaseLdtm, Fragment, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::TmemTile;

/// Rows of `C` one CTA owns. The pair covers `2 * BLOCK_M`, which is the `M`
/// the instruction descriptor names.
const BLOCK_M: usize = 128;
/// Columns of `C` the *pair* computes, split `BLOCK_N / 2` per CTA along the
/// operand's N axis.
const BLOCK_N: usize = 128;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K per pipeline stage: one 128-byte swizzle atom of bf16, the only width
/// [`SharedTile::k_walk`] accepts, and four chained K=16 MMA chunks.
const BLOCK_K: usize = 64;
/// Chained MMAs per stage.
const CHUNKS: usize = BLOCK_K / 16;
/// Pipeline depth over K.
const STAGES: usize = 3;
/// One warp per 32 accumulator rows, which is what a `[32, N]` drain wants.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32;
/// The CTA mask naming both halves of the pair.
const PAIR: u16 = 0b11;

/// This CTA's `A` rows, K-major.
type ATile = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// This CTA's `B` columns, also K-major — so the MMA carries no transpose
/// bits and computes `A·Bᵀ`.
type BTile = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// This CTA's half of the pair's accumulator: 128 TMEM lanes by `BLOCK_N`
/// fp32 columns.
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;
/// One warp's band of it, drained.
type Band = RegTile<32, BLOCK_N, BaseLdtm>;

/// Barriers and the TMEM staging word, in the tail of the shared plan: the two
/// `STAGES`-deep rings, the MMA-complete semaphore, and one `u32`.
const SCRATCH_BYTES: usize = 2 * STAGES * 8 + 8 + 8;
/// Dynamic shared memory the launch must provide.
pub const SHARED_BYTES: usize = ARing::BYTES + BRing::BYTES + SCRATCH_BYTES;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]` over a `(2·BLOCK_M, BLOCK_N)` output
    /// tile per cluster, `k_blocks` stages of `BLOCK_K` deep.
    ///
    /// The grid is one CTA pair per output tile: `blockIdx.x / 2` selects the
    /// tile, and `cluster::block_rank()` says which half of it this CTA owns.
    /// `a_map` describes `A` as `[rows, K]` bf16, `b_map` describes `B` as
    /// `[columns, K]`; both are boxed `[R, 64]` the way
    /// [`SharedTile::tma_load_2d`] walks them.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_cg2(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let a_ring = ARing::attach(smem);
            let b_ring = BRing::attach(smem.add(ARing::BYTES));
            let scratch = smem.add(ARing::BYTES + BRing::BYTES);
            let load = SemaphoreRing::<STAGES>::attach(scratch as *mut Barrier);
            let free = SemaphoreRing::<STAGES>::attach((scratch as *mut Barrier).add(STAGES));
            let done = Semaphore::attach((scratch as *mut Barrier).add(2 * STAGES));
            let tmem_slot = scratch.add(2 * STAGES * 8 + 8) as *mut u32;

            let rank = cluster::block_rank();
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let tile = thread::blockIdx_x() / 2;
            let (tile_m, tile_n) = (tile / tiles_n, tile % tiles_n);

            if thread::threadIdx_x() == 0 {
                load.init_all(1);
                free.init_all(1);
                done.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            // Also publishes the barriers above to the peer, which writes to
            // them before it writes to anything of its own.
            let accumulator = Accumulator::from_raw(alloc_cluster(tmem_slot, BLOCK_N as u32, rank));

            // `M` is the pair's 256 rows and `N` its 128 columns. The rest of
            // the descriptor is the walk's: both operands are K-major, so the
            // MMA takes no transpose bits, and bf16 comes from the tiles.
            let shape = MmaShape::M256_N128;

            if warp_id == 0 && lane == 0 {
                // Producer. The pair's four tiles all complete on the leader's
                // copy of the stage barrier, which the leader charges in full.
                let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * rank) as i32;
                let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * rank) as i32;
                let mut k = 0u32;
                while k < k_blocks {
                    free.wait_recycled(k);
                    let arrival = at_leader(load.sem(k), rank);
                    let column = (BLOCK_K as u32 * k) as i32;
                    a_ring.tile(k).tma_load_2d(a_map, column, a_row, arrival);
                    b_ring.tile(k).tma_load_2d(b_map, column, b_row, arrival);
                    if rank == 0 {
                        load.sem(k)
                            .expect_tx(2 * (ATile::BYTES + BTile::BYTES) as u32);
                    }
                    k += 1;
                }
            }

            if rank == 0 && warp_id == 1 && lane == 0 {
                // The pair's single MMA issuer. Every chunk of every stage
                // chains into the one accumulator, so only the very first
                // instruction of the very first stage starts it fresh.
                let mut k = 0u32;
                while k < k_blocks {
                    load.wait(k);
                    mma_walk_cg2::<Bf16, CHUNKS>(
                        accumulator.raw(),
                        a_ring.tile(k).k_walk(),
                        b_ring.tile(k).k_walk(),
                        shape,
                        k > 0,
                    );
                    // The MMA releases its own operands, in both CTAs: a
                    // thread arriving here would only prove the instruction
                    // was *issued*.
                    commit_multicast_cg2(free.sem(k), PAIR);
                    k += 1;
                }
                commit_multicast_cg2(done, PAIR);
            }

            done.wait(0);
            thread::sync_threads();

            let mut band = Band::zero();
            let mut row_block = 0usize;
            while row_block < 32 / 16 {
                let mut column_block = 0usize;
                while column_block < BLOCK_N / 16 {
                    let fragment = accumulator.fragment_tile(
                        32 * warp_id + 16 * row_block as u32,
                        16 * column_block as u32,
                    );
                    let mut slot = 0usize;
                    while slot < Fragment::SLOTS {
                        let mut value = 0usize;
                        while value < Fragment::VALUES {
                            band.set(
                                2 * row_block + slot,
                                4 * column_block + value,
                                fragment.get(slot, value),
                            );
                            value += 1;
                        }
                        slot += 1;
                    }
                    column_block += 1;
                }
                row_block += 1;
            }

            // ---- GAP (#11, direct global ↔ register store) -----------------
            // What the next twelve lines want to be:
            //
            //     store_tile(c_layout, band, row_base, column_base, lane);
            //
            // The library reaches global memory only through TMA into shared,
            // so an fp32 epilogue either round-trips through a shared tile
            // (losing precision to bf16 on the way) or open-codes the address
            // arithmetic below. This is exactly the index math the library
            // exists to delete, and `RegTile::coordinate` being public is the
            // library admitting it.
            let row_base = 2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * rank + 32 * warp_id;
            let column_base = BLOCK_N as u32 * tile_n;
            let mut slot = 0usize;
            while slot < Band::SLOTS {
                let mut value = 0usize;
                while value < Band::VALUES {
                    let (row, column) = Band::coordinate(lane, slot, value);
                    let index =
                        (row_base + row) as usize * ldc as usize + (column_base + column) as usize;
                    *c.get_unchecked_mut(index) = band.get(slot, value);
                    value += 1;
                }
                slot += 1;
            }
            // ---- end GAP ---------------------------------------------------

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            cluster::cluster_sync();
            if rank == 0 && warp_id == 0 {
                tcgen05_dealloc_cg2(accumulator.raw(), BLOCK_N as u32);
            }
            if thread::threadIdx_x() == 0 {
                load.inval_all();
                free.inval_all();
                done.inval();
            }
        }
    }

    /// ---- GAP (cluster-scope TMEM allocation; no open issue) --------------
    ///
    /// What this wants to be: `tmem::alloc_cluster(slot, columns)`, the
    /// cta_group::2 twin of [`kittens::tmem::alloc_block`].
    ///
    /// A `cg2` accumulator is one allocation spanning the pair, so exactly one
    /// warp in the leader CTA may issue it — and the peer, which drains its own
    /// 128 rows of the same allocation, has no way to learn the address except
    /// to read the leader's staging word over distributed shared memory. Both
    /// halves of that (the `_cg2` intrinsic and the DSMEM read) are hardware
    /// facts about a cluster accumulator, and both belong in `tmem.rs` rather
    /// than in every kernel that wants one.
    ///
    /// The `cluster_sync` here is load-bearing twice over: it publishes the
    /// staging word, and it is the point after which the peer's TMA may write
    /// to the leader's barriers.
    ///
    /// # Safety
    ///
    /// Every thread of every CTA in the cluster must call this together, with
    /// `slot` pointing at a shared `u32` at the same offset in both.
    #[inline(always)]
    unsafe fn alloc_cluster(slot: *mut u32, columns: u32, rank: u32) -> u32 {
        unsafe {
            if rank == 0 && warp::warp_id() == 0 {
                tcgen05_alloc_cg2(slot, columns);
            }
            thread::sync_threads();
            cluster::cluster_sync();
            if rank == 0 {
                *(slot as *const u32)
            } else {
                cluster::dsmem_read_u32(slot as *const u32, 0)
            }
        }
    }

    /// ---- GAP (cluster-scope semaphore arrival; no open issue) ------------
    ///
    /// What this wants to be: `Semaphore::at_rank(rank)`.
    ///
    /// A 2-CTA MMA consumes four tiles staged by two CTAs, so the issuer needs
    /// one barrier that says *the whole stage is present* — which means the
    /// peer's TMA has to complete on the leader's copy. mbarrier addresses are
    /// cluster-mappable and this is the standard way to build a pair-wide
    /// producer handoff, but nothing in `sync.rs` says so: [`Semaphore`] is
    /// CTA-scoped by construction, and [`Semaphore::multicast_alias`] — the
    /// one cluster-aware thing it has — solves the opposite problem (one
    /// barrier per CTA, one transfer).
    ///
    /// # Safety
    ///
    /// `sem` must be at the same shared offset in every CTA of a live cluster,
    /// and rank 0's copy must be initialized before this CTA arrives on it.
    #[inline(always)]
    unsafe fn at_leader(sem: Semaphore, rank: u32) -> Semaphore {
        unsafe {
            if rank == 0 {
                sem
            } else {
                Semaphore::attach(cluster::map_shared_rank_mut(sem.raw(), 0))
            }
        }
    }
}
