//! # GEMM — `C = A·Bᵀ` on the cta_group::2 cluster path
//!
//! **Status: runs.** [`check`] launches it against a CPU reference on a B200
//! (`modal run modal_app.py::examples`). The operands are small integers, so
//! every product and every partial sum is exact and the comparison is `==`
//! rather than a tolerance — a mismatch is a wrong index, never rounding.
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
//! ## The grid is persistent, and the cluster is the work item
//!
//! One output tile used to be one launch's worth of cluster: the grid was
//! `2 * tiles` and a CTA read its tile out of `blockIdx.x / 2`. It is now
//! [`kittens::pipeline::run`] over a [`Tile`] job, on a grid capped at
//! [`MAX_CLUSTERS`] — a cluster that runs out of items is the whole schedule,
//! and a cluster that has more walks them.
//!
//! That was blocked until #51 for a reason worth keeping: the scaffold's item
//! map was `blockIdx.x` strided by `gridDim.x`, which gives the two halves of
//! a pair *different* output tiles. It is `%clusterid` strided by `%nclusterid`
//! now, which is the same map at the granularity a cooperative MMA's item
//! actually has. What the job supplies beyond `lcf`'s three methods is one
//! constant — `RANKS = 2` — and that constant buys the thing per-item barrier
//! re-initialization needs across a pair: the item boundary is
//! `barrier.cluster` rather than `bar.sync`, so no rank re-arms a barrier its
//! peer is still filling and no rank fills one its peer has not re-armed.
//!
//! The grid's cap is the entire performance story, and it is a *measurement*
//! rather than a derivation — see [`CTAS_PER_SM`]. `lcf` at one item per
//! cluster is the launch-per-tile kernel exactly: same rings, same barriers,
//! drained at the same points. So a persistent grid changes only how many
//! clusters exist, and picking that number wrong costs 2×.
//!
//! What the scaffold does *not* buy is overlap between items. `lcf` folds the
//! epilogue into the item, so this kernel's `store_rows` and the next tile's
//! first K loads are still separated by a boundary that drains the pipeline —
//! #15's `lcsf` is the shape that would let them cross. So the honest claim for
//! this port is a **dead heat**: 1.0217 ms at 8192³ against the launch-per-tile
//! grid's 1.0204, which is 1076 TFLOP/s against 1077. It is not faster. What it
//! buys is that the scaffold has a caller, and therefore a regression test.
//! `examples/README.md` §7 is the sweep that found the cap and what it says
//! about #78.
//!
//! **That boundary has since been priced, and it is the largest single term in
//! this kernel's time** (#86). Holding `M` and `N` at 8192 and sweeping `K`
//! holds the tile count, the wave count, the grid and the `C` traffic fixed
//! while changing only how much arithmetic sits between two boundaries; the
//! result is 152.8 / 508.6 / 1074.0 / **1369.9** TFLOP/s at `K` of 512 / 2048 /
//! 8192 / 32768. Milliseconds against `K` is a line whose intercept is the
//! boundary; over the ten items a cluster walks, and over every reasonable
//! choice of which points to fit, that is **28–37 µs per output tile and 27–36%
//! of the 8192³ launch**, on a steady state of 1480–1550 TFLOP/s. The same
//! ~30 µs is the flat floor at the small end of the benchmark: 256x128x256 is
//! one cluster running one item, and it costs 23.4 µs. See `examples/README.md`
//! §7 for the point-selection table, for why the fit's residuals rule *out* the
//! obvious pipeline-fill explanation, and for the other two sweeps — the
//! aspect-ratio one is worth 23% and belongs to #89.
//!
//! ## What this kernel had to reach past the library for
//!
//! **Nothing.** There is no `GAP` block left in this file, and the last one to
//! go is worth recording because it was the most arithmetic of the three. The
//! epilogue was twelve lines of open-coded index math against
//! `RegTile::coordinate` — the library reached global memory only through TMA
//! into shared, so an fp32 band either round-tripped through bf16 or was
//! addressed by hand. It is now one [`kittens::global::store_rows`] over a
//! [`kittens::global::GlobalRows`] cursor holding `ldc` (#11), and the exact
//! CPU check below is what says the two address the same elements: every one
//! of `M * N` is compared, so a coordinate this kernel no longer computes is
//! still a coordinate the reference would catch.
//!
//! The cluster-scope TMEM allocation that used to sit beside it is now
//! [`kittens::tmem::alloc_cluster`] / [`kittens::tmem::dealloc_cluster`]
//! (#24, #46) — this file was where the `cta_group::2` allocator's
//! participation rules and its relinquish were worked out against silicon,
//! and they are hardware facts about a cluster accumulator rather than
//! anything this kernel chose.
//!
//! The third is gone too, and how it went is worth recording. The pair's four
//! TMA loads have to complete on *one* barrier for the leader to know the
//! whole stage is present, and this file used to map the leader's barrier by
//! hand. That deadlocks: a plain `cp.async.bulk.tensor` completes on a barrier
//! in the *issuing* CTA's own shared memory, so the peer's bytes never reached
//! the leader's count. It then said so through a multicast load with a
//! degenerate mask, which worked and did not read as what it was. Both halves
//! are now named: [`Semaphore::at_rank`] is the leader's copy of the stage
//! barrier, and [`SharedTile::tma_load_2d_arriving_at`] is a load that
//! completes there (#50). The charge stays the leader's, and is its own charge
//! `RANKS` times over — `sync.rs`'s module docs carry the argument for why the
//! byte accounting can only sit at the barrier and not travel with the loads.
//!
//! What the leader's own charge *is* it no longer says: the two loads hand it
//! back, and `across_ranks` is the only arithmetic left (#29). This was the
//! producer with the most to lose from a hand-written total, since a stage
//! here is four tiles issued by two CTAs and the number covering them was
//! written once, in the CTA that issues half of them.

use cuda_device::barrier::Barrier;
use cuda_device::cluster;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{
    DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, thread, warp,
};

// Host side: the launcher's error type, and the benchmark's size and clock.
use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::global::{GlobalRows, store_rows};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcQueue, Job};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};

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
/// CTAs in the cluster. Also the multiplier on a stage's transaction charge:
/// both ranks stage the same two tile types at the same shared offsets, so the
/// whole stage is one rank's charge twice over.
const RANKS: u32 = 2;
/// The CTA mask naming every half of the pair.
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
/// The rank that owns the pair's MMA, its accumulator and its stage barriers.
const LEADER: u32 = 0;

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

/// Barriers, the TMEM staging word and the scheduler's queue, in the tail of
/// the shared plan: the two `STAGES`-deep rings, the MMA-complete semaphore,
/// one `u32`, and [`ClcQueue::BYTES`] for the hardware work queue.
const SCRATCH_BYTES: usize = 2 * STAGES * 8 + 8 + 8 + ClcQueue::BYTES;
/// Where the work queue sits, and the one thing about it a constant has to
/// say: `try_cancel` writes a `.b128`, so the offset has to be aligned for one.
/// This assert fires at codegen, which is the only place the ring byte counts
/// under it are known.
const QUEUE_OFFSET: usize = ARing::BYTES + BRing::BYTES + 2 * STAGES * 8 + 8 + 8;
const _: () = assert!(QUEUE_OFFSET % ClcQueue::ALIGNMENT == 0);
/// Dynamic shared memory the launch must provide.
///
/// Every scheduler below launches with the *same* plan, including the static
/// one that never touches the queue. Twenty-four bytes is not worth a second
/// envelope, and paying them on both sides is what keeps the A/B a comparison
/// of schedules rather than of residencies — 73 816 B still admits the three
/// CTAs per SM that #84 counted at 73 792.
pub const SHARED_BYTES: usize = ARing::BYTES + BRing::BYTES + SCRATCH_BYTES;

/// `#[launch_contract]` takes literals, so the envelope is written twice; this
/// is what keeps the two in step. The contract is not decoration: 72 KiB is
/// past the 48 KiB a block gets by default, and the opt-in
/// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`) is issued by the
/// prepared-launch path. #70 asked whether this kernel was silently relying on
/// something and the answer is no — it relies on this line, which is why it
/// launches at 72 KiB while `flash_forward`, a plain `#[kernel]`, reported
/// zero blocks per SM at 144. `cluster_launch` has nothing to do with it.
/// [`kittens::launch::admit_shared_plan`] is the same opt-in for a kernel
/// whose output partition no contract describes.
const _: () = assert!(THREADS == 128 && SHARED_BYTES == 73_816);

/// SMs on the device this project targets and measures on — a B200, as
/// `modal run modal_app.py::bench` prints in its header.
const SMS: u32 = 148;
/// CTAs of this kernel one SM holds at once — **measured, first on a clock and
/// since confirmed by counting.**
///
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` takes a block shape and no
/// cluster, so it cannot answer about a `#[cluster_launch]` kernel; `main.rs`
/// prints `cluster` in this kernel's row for exactly that reason. The one
/// figure it does give is #77's, and extrapolating that here would say **1**: a
/// CTA that touches `tcgen05.alloc` is charged the SM's whole tensor memory,
/// and this kernel allocates.
///
/// That extrapolation is false, and the cap sweep in `examples/README.md` §7 is
/// what refutes it. Capping the grid at one CTA per SM makes 8192³ take 2.1036
/// ms against the launch-per-tile grid's 1.0204, which cannot happen if the
/// device could only ever hold that many CTAs — the two schedules would then be
/// the same schedule. Two per SM gives 1.2130, and three gives **1.0217**: a
/// dead heat, and where the curve stops. So the residency is three, it was
/// arrived at by bisection rather than by asking, and it is the reason a
/// persistent grid costs nothing here rather than 2×.
///
/// **A second instrument agrees, and it names the resource.** #78's
/// `tmem residency census` counts CTAs per SM outright — every CTA records its
/// `%smid` and timestamps both ends of its allocation off `%globaltimer`, and
/// the host sweeps those intervals for the most ever open at once. At this
/// kernel's own envelope it counts **3**. Two methods sharing nothing, a
/// throughput curve on a real GEMM and timestamps from a nine-register probe,
/// landing on one integer is much stronger evidence than either alone.
///
/// The census also says *which* resource sets it, which the bisection could
/// not: 128 columns of tensor memory would admit 4 CTAs an SM and this
/// kernel's 73792 B shared plan admits 3, so **shared memory caps this kernel
/// and `tcgen05` does not**. It priced `alloc_cluster` against `alloc_block` at
/// equal columns too and found them identical, so none of this is a cluster
/// effect. And a query *can* describe a cluster launch —
/// `cuOccupancyMaxActiveClusters` takes the shape the block query has no
/// argument for. It was never called here; it is not that nothing could answer.
const CTAS_PER_SM: u32 = 3;
/// Clusters the persistent grid launches at most, past which a cluster takes a
/// second work item rather than the scheduler holding a pair back.
///
/// This is a *tuning* constant and not a correctness one: [`pipeline::run`]
/// walks every item whatever the grid is, so a device with a different SM
/// count computes the same GEMM off a wave that is not quite a wave. Which is
/// the only reason a hardware figure may sit in it — and, per
/// [`CTAS_PER_SM`], the reason getting it wrong is a benchmark row and not a
/// wrong `C`.
///
/// **It is also the constant [`Scheduler::Stealing`] exists to delete**, and
/// the case against carrying it is now stronger than "a benchmark row": #84
/// showed [`CTAS_PER_SM`] is measured and underivable, so this line is right
/// for a B200 and cannot be known to be right for anything else. Under CLC the
/// grid is the tile count and the residency is the scheduler's business. This
/// constant survives here only for as long as the static path is the control.
const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / RANKS;
const _: () = assert!(MAX_CLUSTERS == 222);

/// Which item source a launch runs on. The [`Tile`] job is identical under
/// both — that is the point of it being a [`Job`] — and what changes is the
/// grid, and where the next item comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scheduler {
    /// [`pipeline::run`]: a grid capped at [`MAX_CLUSTERS`], each cluster
    /// taking a fixed share decided before the kernel starts. The control.
    Static,
    /// [`pipeline::run_stealing`]: a grid of one cluster per tile, and a
    /// cluster that finishes cancels one the scheduler has not launched yet.
    Stealing,
}

impl Scheduler {
    /// What the benchmark prints, and the only place these names are spelled.
    pub fn name(self) -> &'static str {
        match self {
            Scheduler::Static => "static",
            Scheduler::Stealing => "clc",
        }
    }
}

/// One output tile of `C`, as the persistent grid's work item.
///
/// Every field is what the item needs and does not depend on *which* item it
/// is: the pair's rings and barriers, the two operand maps, this CTA's half of
/// the accumulator, and the thread's own coordinates. The item index is the
/// only thing [`Job::work`] takes, and `(item / tiles_n, item % tiles_n)` is
/// the whole of what it does with it — the same map `blockIdx.x / 2` used to
/// carry, now asked of a number that means one tile per *cluster*.
#[derive(Clone, Copy)]
struct Tile {
    a_ring: ARing,
    b_ring: BRing,
    /// Filled by the TMA, drained by the MMA. In the leader's copy the whole
    /// pair's four tiles complete on one barrier; the peer's own copy is
    /// unused, and initialized anyway because the plan is symmetric.
    load: SemaphoreRing<STAGES>,
    /// Released by the MMA's own commit, in both CTAs.
    free: SemaphoreRing<STAGES>,
    /// The pair's accumulator complete, likewise multicast by the MMA.
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    /// `C` with `ldc` in it — built once, since a persistent CTA writes bands
    /// of the same output through every item it runs.
    c: GlobalRows,
    tiles_n: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl Job for Tile {
    /// The pair shares one barrier set — the peer aims its TMA at the leader's
    /// stage barrier and the leader's MMA arrives in the peer's `free` and
    /// `done` — so the item boundary that re-arms them has to be the cluster's.
    const RANKS: u32 = crate::gemm::RANKS;

    /// Every barrier of the item takes exactly one arrival: the leader's stage
    /// barrier from the TMA transaction count, `free` and `done` from the MMA
    /// commit. Nothing here depends on `item`, since every tile is the same
    /// `k_blocks` deep.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::init`]; [`pipeline::run`] owns the thread and the
    /// ordering.
    #[inline(always)]
    unsafe fn init(&self, _item: u32) {
        unsafe {
            self.load.init_all(1);
            self.free.init_all(1);
            self.done.init(1);
        }
    }

    /// # Safety
    ///
    /// As [`Semaphore::inval`]. The arrivals this wipes are real and not
    /// hypothetical: the last `STAGES` MMA commits release `free` slots no
    /// producer will ever wait on again.
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe {
            self.load.inval_all();
            self.free.inval_all();
            self.done.inval();
        }
    }

    /// # Safety
    ///
    /// Every thread of both CTAs of the cluster must enter with the same
    /// `item`, which is what [`pipeline::run`]'s cluster-strided map gives, and
    /// the maps must cover the tile it names.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let Tile {
                a_ring,
                b_ring,
                load,
                free,
                done,
                a_map,
                b_map,
                accumulator,
                c,
                tiles_n,
                k_blocks,
                rank,
                warp_id,
                lane,
            } = *self;
            let (tile_m, tile_n) = (item / tiles_n, item % tiles_n);

            // `M` is the pair's 256 rows and `N` its 128 columns. The rest of
            // the descriptor is the walk's: both operands are K-major, so the
            // MMA takes no transpose bits, and bf16 comes from the tiles.
            let shape = MmaShape::M256_N128;

            if warp_id == 0 && lane == 0 {
                // Producer. Both CTAs load their own halves, and all four
                // tiles complete on the leader's copy of the stage barrier —
                // one barrier is what the MMA issuer needs to know the whole
                // stage is present. Only the leader charges, and it charges
                // the whole stage: `expect_tx` is `.shared::cta`, so a peer
                // could not charge this barrier even holding its address.
                //
                // Every rank derives the same half-stage charge from the loads
                // it just issued, because a cluster stage is symmetric; the
                // leader scales its own by `RANKS` to cover the peer's, and
                // the peer drops it. Nothing orders the charge and the loads,
                // and nothing has to — the transaction count is a signed
                // accumulator and only the totals must agree, which is what
                // lets the charge follow the calls it is derived from.
                let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * rank) as i32;
                let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * rank) as i32;
                let mut k = 0u32;
                while k < k_blocks {
                    free.wait_recycled(k);
                    let stage = load.sem(k).at_rank(LEADER);
                    let column = (BLOCK_K as u32 * k) as i32;
                    let a_bytes = a_ring
                        .tile(k)
                        .tma_load_2d_arriving_at(a_map, column, a_row, stage);
                    let b_bytes = b_ring
                        .tile(k)
                        .tma_load_2d_arriving_at(b_map, column, b_row, stage);
                    if rank == LEADER {
                        load.sem(k)
                            .expect_tx((a_bytes + b_bytes).across_ranks(RANKS));
                    }
                    k += 1;
                }
            }

            if rank == LEADER && warp_id == 1 && lane == 0 {
                // The pair's single MMA issuer. Every chunk of every stage
                // chains into the one accumulator, so only the very first
                // instruction of the very first stage starts it fresh — and
                // "first" is per *item*, because the epilogue below drains the
                // accumulator before the next item starts filling it.
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

            // The whole band in one call (#22): this warp's 32 TMEM lanes by
            // every column of the accumulator, composed out of the `[16, 16]`
            // blocks LDTM delivers.
            let band: Band = accumulator.tile(32 * warp_id, 0);

            // The fp32 epilogue, straight out of registers (#11). `ldc` is the
            // destination's leading dimension and `C` is wider than this
            // tile's columns, so the cursor carries the stride and the band
            // lands at its own `(row, column)` origin — no shared staging
            // tile, no descriptor, and no rounding to bf16 on the way out.
            let row_base = 2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * rank + 32 * warp_id;
            let column_base = BLOCK_N as u32 * tile_n;
            store_rows(c, row_base, column_base, lane, band);
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The item and the work queue, laid over the one shared plan both entry
    /// points launch with. Everything here spans items rather than
    /// belonging to one, which is why it is hoisted out of every scheduler
    /// alike: the rings, the barriers, the operand maps, and the pair's TMEM
    /// allocation, whose `alloc_cluster` is a whole-cluster collective with a
    /// `cluster_sync` in it and must not be inside anybody's item loop.
    ///
    /// # Safety
    ///
    /// The launch geometry's, and the operands': both maps must describe live
    /// buffers covering `k_blocks * BLOCK_K` along K and the full extent the
    /// item loop walks, and `c` must hold `ldc` columns for every row of it.
    #[inline(always)]
    unsafe fn attach(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<f32>,
    ) -> (Tile, ClcQueue) {
        unsafe {
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let scratch = smem.add(ARing::BYTES + BRing::BYTES);
            let tmem_slot = scratch.add(2 * STAGES * 8 + 8) as *mut u32;

            let tile = Tile {
                a_ring: ARing::attach(smem),
                b_ring: BRing::attach(smem.add(ARing::BYTES)),
                load: SemaphoreRing::<STAGES>::attach(scratch as *mut Barrier),
                free: SemaphoreRing::<STAGES>::attach((scratch as *mut Barrier).add(STAGES)),
                done: Semaphore::attach((scratch as *mut Barrier).add(2 * STAGES)),
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster(tmem_slot, BLOCK_N as u32)),
                c: GlobalRows::from_slice(c, ldc as usize),
                tiles_n,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
            };
            (tile, ClcQueue::attach(smem.add(QUEUE_OFFSET)))
        }
    }

    /// Give the pair's accumulator back.
    ///
    /// The scaffold's last item boundary already retired the pair's reads, and
    /// this `cluster_sync` is for the cluster that got no items at all —
    /// [`Scheduler::Static`] can leave a pair having allocated, never looped,
    /// and still owing a deallocation in step with its peer. Under CLC no
    /// cluster is ever launched without an item, and the sync is kept anyway
    /// because it costs one barrier at the end of a kernel and the alternative
    /// is a scheduler-shaped hole in a resource protocol.
    ///
    /// # Safety
    ///
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired.
    #[inline(always)]
    unsafe fn release(tile: &Tile) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), BLOCK_N as u32);
        }
    }

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `(2·BLOCK_M, BLOCK_N)` output
    /// tile per work item, `k_blocks` stages of `BLOCK_K` deep.
    ///
    /// The grid is persistent and the item map is [`pipeline::run`]'s: a
    /// *cluster* takes item `%clusterid` and steps by `%nclusterid` until the
    /// `tiles` are gone, and `cluster::block_rank()` says which half of the
    /// pair this CTA owns. `a_map` describes `A` as `[rows, K]` bf16, `b_map`
    /// describes `B` as `[columns, K]`. Both come from a rank-2
    /// [`kittens::global::GlobalLayout`] paired with the tile it feeds, so
    /// their `[R, 64]` boxes are `ATile`'s and `BTile`'s own constants and not
    /// numbers [`check`] wrote down.
    ///
    /// Everything outside the item loop is `attach` and `release`; what is
    /// left here is the schedule.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus: the grid must be a whole number of clusters and
    /// `tiles` the item count they are to cover — see [`grid`], which is what
    /// the launcher below sizes both from.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 73_816,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_cg2(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) = attach(a_map, b_map, tiles_n, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles);
            release(&tile);
        }
    }

    /// The same GEMM on the hardware's schedule: a grid of one cluster per
    /// output tile, of which the device launches as many as it holds, and the
    /// rest are cancelled out from under the scheduler by clusters that have
    /// finished — [`pipeline::run_stealing`].
    ///
    /// **There is no `tiles` argument, and that absence is the feature.** The
    /// static entry point above needs the item count because its grid is capped
    /// at a constant that had to be measured on one device (#84). Here the grid
    /// *is* the item count, and neither [`SMS`] nor [`CTAS_PER_SM`] appears
    /// anywhere on the path.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus [`pipeline::run_stealing`]'s: the grid is exactly
    /// `RANKS` × the tile count, one-dimensional, on sm_100a.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 73_816,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_cg2_clc(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, queue) = attach(a_map, b_map, tiles_n, k_blocks, ldc, &mut c);
            pipeline::run_stealing(&mut tile, queue);
            release(&tile);
        }
    }
}

/// Rows of `C` the correctness run computes — two clusters' worth of `M`, so
/// the `item → (tile_m, tile_n)` map is exercised in both axes.
pub const M: usize = 512;
/// Columns of `C`: two `BLOCK_N` tiles.
pub const N: usize = 256;
/// Reduction depth: four `BLOCK_K` stages against a three-deep pipeline, so
/// the ring wraps and `wait_recycled` is on trial rather than skipped.
pub const K: usize = 256;

/// A second correctness size, whose only job is to give every cluster **more
/// than one work item**.
///
/// [`M`]`x`[`N`] is four tiles and the grid holds 222 clusters, so it never
/// enters [`pipeline::run`]'s loop a second time — it would pass identically
/// against the pre-#51 kernel, which is exactly what makes it not a test of
/// this one. The failure modes the persistent scaffold introduces all live at
/// the *item boundary*: a barrier re-armed while a peer is still filling it,
/// an accumulator that is not started fresh for the next tile, an epilogue
/// racing the next item's first loads. Each of those is a deadlock or a wrong
/// `C`, and each needs a second item to happen at all.
///
/// 512 tiles over 222 clusters is three items for most and two for the rest,
/// so the ragged tail — clusters that leave the loop an item early, while
/// their neighbours are still running one — is under test too. `K` stays at
/// the wrapping four stages and the shape stays cheap: the operands are 2 MiB
/// each and the only large buffer is `C`.
const ITEMS_M: usize = 4096;
const ITEMS_N: usize = 4096;
const ITEMS_K: usize = 256;

/// `A[m, k]` and `B[n, k]`: integers in `[-3, 3]` and `[-10, 10]`.
///
/// Every operand is exact in bf16 (which holds every integer to 256) and every
/// partial sum stays under `3 * 10 * K` — 983,040 at the benchmark's largest
/// `K` of 32768, against fp32's exact integer range of 2²⁴ = 16,777,216. So the
/// whole GEMM is exact and the host compares with `==`. That is the point: a
/// mismatch is a wrong coordinate, a wrong stride or a wrong operand half, and
/// never a rounding artifact that has to be argued about.
///
/// Both generators depend on `depth`, and #48 is why that is spelled out
/// rather than assumed. `b_value` used to read `(column * 3 + depth * 5) % 5`,
/// and `depth * 5 % 5` is identically zero — `B` was constant along `K`, so
/// the exact check was blind to precisely the axis `mma_walk_cg2`'s chunk
/// arithmetic computes. A kernel reading one plane of `B` every step, walking
/// K backwards, or aliasing the ring slot for the K index passed anyway.
///
/// The moduli are 7 for `A` and 21 for `B`, and the pair is chosen rather than
/// convenient. `A`'s values over one period of `depth` are `0..7` shifted to
/// sum to zero, so if `B`'s `depth` period were *coprime* to 7 the sum over
/// one combined period would factor as `(Σ A)(Σ B) = 0` — the dot product
/// would then be a function of `K mod 7·P` alone, bounded independently of
/// `K`, and two different K walks would collide by accident. Sharing the
/// factor 7 defeats that: the partial sums grow with `K`, and a swept check of
/// every legal `K` up to 8192 (every multiple of `BLOCK_K`) finds no wrong K
/// walk this reference cannot see — same plane, reversed, off by a plane, off
/// by a chunk, off by a stage, chunk order permuted, stage order reversed, or
/// the three-deep ring slot mistaken for the K index.
///
/// The residual is worth stating because it is bounded and provable rather
/// than hoped for: when `K` is a multiple of 21 the tail vanishes and `B`'s
/// column period collapses from 21 to 7, so a column error that is a multiple
/// of 7 becomes invisible. No `K` this project runs is a multiple of 21 — they
/// are all powers of two — and the K axis stays exact even there.
pub(crate) fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

pub(crate) fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

/// Round-to-nearest-even fp32 → bf16. Exact for every value [`a_value`] and
/// [`b_value`] produce, since their low 16 mantissa bits are already zero.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// A `[rows, k]` bf16 operand as the packed `u32` words a device buffer holds,
/// K contiguous — which is what makes both operands K-major and the MMA
/// transpose-free.
pub(crate) fn stage(rows: usize, k: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u32> {
    let mut staged = Vec::with_capacity(rows * k / 2);
    for row in 0..rows {
        for pair in 0..k / 2 {
            let (low, high) = (value(row, 2 * pair), value(row, 2 * pair + 1));
            staged.push(to_bf16(low) as u32 | ((to_bf16(high) as u32) << 16));
        }
    }
    staged
}

/// Tiles of `C` a `[m, n]` output has, which is the item count the persistent
/// grid walks: one per `2·BLOCK_M` by `BLOCK_N` tile.
fn tiles(m: usize, n: usize) -> u32 {
    (m / (2 * BLOCK_M) * (n / BLOCK_N)) as u32
}

/// Blocks the launch asks for — [`MAX_CLUSTERS`] pairs, or fewer where the
/// problem has fewer tiles than that.
///
/// The benchmark prints it, because at the small end of a size sweep this
/// number and not the arithmetic is what the run is bound by. Since #51 it is
/// also where the sweep stops growing: past 222 clusters the grid is flat and
/// the extra tiles arrive as extra *items*, which is the whole difference
/// between this kernel and the one that launched a pair per tile.
pub fn grid(m: usize, n: usize) -> u32 {
    grid_for(Scheduler::Static, m, n)
}

/// Blocks a `scheduler`'s launch asks for, which is the one host-visible
/// difference between them.
///
/// The static grid is capped and the stealing grid is not: CLC cancels
/// clusters out of the *pending* queue, so a cluster that never launches is
/// exactly how the hardware caps it, and asking for fewer than one cluster per
/// tile would leave items nothing can reach. This is the shape of the
/// deletion — one branch reads a measured constant and the other reads the
/// problem.
pub fn grid_for(scheduler: Scheduler, m: usize, n: usize) -> u32 {
    let clusters = match scheduler {
        Scheduler::Static => tiles(m, n).min(MAX_CLUSTERS),
        Scheduler::Stealing => tiles(m, n),
    };
    RANKS * clusters
}

/// Waves the static grid takes at `m`x`n`, and how full the last one is.
///
/// This is the entire quantity CLC has to win back, and it is worth computing
/// rather than measuring against: a cluster's item takes the same time whoever
/// scheduled it, so the static stride's only loss is the clusters idle through
/// the ragged last wave. At [`MAX_CLUSTERS`] the efficiency is 57.7% at 2048³,
/// 76.9% at 4096³, 92.3% at 8192³ and **99.7% at 16384³** — so the predicted
/// gain runs from 23% down to nothing across the sweep, and the largest size,
/// the one a peak claim would be made on, predicts nothing.
pub fn wave_efficiency(m: usize, n: usize) -> (u32, f64) {
    let tiles = tiles(m, n);
    let waves = tiles.div_ceil(MAX_CLUSTERS);
    (waves, tiles as f64 / (waves * MAX_CLUSTERS) as f64)
}

/// The `then` a run that is only being checked passes: nothing follows the
/// comparison.
fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch `[m, k] · [n, k]ᵀ`, compare every element of `C` against a CPU
/// reference, and only then hand the launch to `then`.
///
/// That order is the whole design: `then` — [`crate::bench::time`] for a timed
/// run — is unreachable from a launch whose output was wrong, so no throughput
/// figure can be printed for a kernel that did not compute. It receives the
/// launch as a closure over buffers that are already staged, so a repeat costs
/// a launch and nothing else.
///
/// The two operands differ only in their extents, and neither states a box:
/// [`kittens::global::GlobalLayout::tensor_map`] takes the tile the kernel
/// loads into and reads the box, the swizzle and the data type off it. `A` is
/// `[k, m]` in the driver's fastest-first dimension order and `B` is `[k, n]`,
/// which is the same order the kernel gives its
/// [`SharedTile::tma_load_2d`] coordinates in.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    scheduler: Scheduler,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    use kittens::global::GlobalLayout;

    // The tiling is the constraint on what sizes exist: a cluster owns a whole
    // `2·BLOCK_M` by `BLOCK_N` tile and a stage is a whole `BLOCK_K`, and the
    // kernel bounds-checks none of it. A size that does not divide is rejected
    // rather than launched into somebody else's memory.
    if m % (2 * BLOCK_M) != 0 || n % BLOCK_N != 0 || k % BLOCK_K != 0 {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{BLOCK_N}x{BLOCK_K} tiling",
            2 * BLOCK_M
        )
        .into());
    }

    let stream = context.default_stream();
    // SAFETY: the artifact is this crate's own, and `gemm_cg2` is the only
    // entry point in it — the ABI the contract declares is the one compiled.
    let module = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::from_host(&stream, &stage(m, k, a_value))?;
    let b = DeviceBuffer::from_host(&stream, &stage(n, k, b_value))?;
    // SAFETY: both buffers outlive every launch consuming their maps below.
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<Bf16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<Bf16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BTile>(&stream)?;

    let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
    // All three are prepared and one is launched. Preparing is a driver call
    // opting the entry point into its 72 KiB, and doing it for the entry points
    // this call will not use costs nothing measurable and keeps the dispatch
    // below a `match` over launches rather than over types the kernel macro
    // named.
    let config =
        |scheduler| LaunchConfig1D::new(grid_for(scheduler, m, n), THREADS, SHARED_BYTES as u32);
    let static_launch = module.prepare_gemm_cg2(config(Scheduler::Static))?;
    let clc_launch = module.prepare_gemm_cg2_clc(config(Scheduler::Stealing))?;
    let tiles_n = (n / BLOCK_N) as u32;
    let k_blocks = (k / BLOCK_K) as u32;
    let launch_once = |c: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn Error>> {
        // SAFETY: both maps describe live buffers covering the walk the grid
        // above takes, and `c` holds `n` columns for every row of it. The
        // stealing entries additionally take a grid of exactly one cluster per
        // tile, which is what `grid_for` gives them.
        unsafe {
            match scheduler {
                Scheduler::Static => module.gemm_cg2(
                    &stream,
                    &static_launch,
                    a_map.as_ptr(),
                    b_map.as_ptr(),
                    tiles(m, n),
                    tiles_n,
                    k_blocks,
                    n as u32,
                    c,
                )?,
                Scheduler::Stealing => module.gemm_cg2_clc(
                    &stream,
                    &clc_launch,
                    a_map.as_ptr(),
                    b_map.as_ptr(),
                    tiles_n,
                    k_blocks,
                    n as u32,
                    c,
                )?,
            }
        };
        Ok(())
    };
    launch_once(&mut c)?;
    check_c(&c.to_host_vec(&stream)?, m, n, k)?;

    let after = then(&stream, &mut || launch_once(&mut c))?;
    Ok((format!("{m}x{n}x{k} exact"), after))
}

/// Compare an observed `[m, n]` row-major fp32 `C` against the CPU reference
/// for `[m, k] · [n, k]ᵀ`, element by element and with `==`.
///
/// `a_value` repeats every 7 rows and `b_value` every 21 columns, so the
/// reference has 147 distinct dot products at any size and the naive
/// `O(m·n·k)` form is pure waste — minutes of host time at the sizes the
/// benchmark reaches, for the same 147 numbers. Every element of `C` is still
/// compared against its own expected value, in the same summation order, so the
/// comparison is the one it always was. The sum stays over the *full* `k`,
/// since both generators vary along it.
///
/// It is a function rather than a block inside [`run`] because the **cuBLASLt
/// baseline** (#92) is checked by it too. A denominator produced by a different
/// GEMM is worth nothing, and the way that happens is a transposed operand or a
/// wrong leading dimension — which computes the wrong matrix at full speed and
/// looks like a plausible number. Sharing this function rather than copying it
/// is what makes "checked against the same CPU reference" a property of the
/// code instead of a claim in a comment.
pub(crate) fn check_c(
    observed: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let reference: Vec<f32> = (0..7 * 21)
        .map(|cell| {
            (0..k)
                .map(|depth| a_value(cell / 21, depth) * b_value(cell % 21, depth))
                .sum()
        })
        .collect();
    let (mut wrong, mut sample) = (0usize, Vec::new());
    for row in 0..m {
        for column in 0..n {
            let expected = reference[(row % 7) * 21 + column % 21];
            let value = observed[row * n + column];
            if value != expected {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!("C[{row}, {column}] = {value}, want {expected}"));
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!("{wrong} of {} elements wrong: {}", m * n, sample.join("; ")).into());
    }
    Ok(())
}

/// A scheduler's own failure modes, named so a passing line says what it
/// proved. All three are silent on the device and all three reach the host as
/// a wrong `C`, which is why the exact comparison above is the only gate that
/// sees them:
///
/// - a **dropped tile** — a steal that succeeded and was read as "no work
///   left", so the cancelled cluster's item is computed by nobody. The
///   `is_canceled` polarity is exactly this bug, and cuda-oxide's own module
///   doc has the sense inverted against its function doc and its lowering.
/// - a **tile computed twice**, which is only visible through the tile it
///   displaces, since the epilogue stores rather than accumulates.
/// - a **split pair** — #51 again, if the two CTAs of a cluster ever steal
///   separately and land on different output tiles.
///
/// A deadlock is the fourth and reports itself.
const SCHEDULERS: [Scheduler; 2] = [Scheduler::Static, Scheduler::Stealing];

/// The correctness run: two sizes, checked, nothing timed.
///
/// The second one is [`ITEMS_M`]`x`[`ITEMS_N`], and it is here because the
/// first cannot fail the way the persistent grid can. Both report the items
/// their clusters walked, so a size that quietly stopped exercising the loop —
/// because [`MAX_CLUSTERS`] moved, or because the tiling did — says so in the
/// pass line instead of in nobody's memory.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        for scheduler in SCHEDULERS {
            let (note, ()) = run(context, m, n, k, scheduler, nothing_after)?;
            let clusters = grid_for(scheduler, m, n) / RANKS;
            notes.push(format!(
                "{note} on {} ({} tiles over {clusters} clusters)",
                scheduler.name(),
                tiles(m, n)
            ));
        }
    }
    Ok(notes.join(", "))
}

/// The benchmark's entry point: the same check at `shape`, and then the same
/// launch timed.
///
/// Still the static schedule, deliberately. This row is what a reader compares
/// against every earlier run of this table, and moving it to a different
/// scheduler in the same change that introduces one would make the comparison
/// unreadable. [`compare`] is where the schedulers are put beside each other.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, shape.n, shape.k, Scheduler::Static, time).map(|(_, timings)| timings)
}

/// Sizes the scheduler comparison runs at, which are the sizes the prediction
/// varies across — 23% at 4096³ down to nothing at 16384³. The largest is here
/// because a result of *zero* at it is the finding, and a sweep that stopped
/// before the point where the static stride is already 99.7% efficient would
/// be a sweep chosen to flatter this work.
const COMPARE_SIZES: &[Shape] = &[
    Shape {
        m: 2048,
        n: 2048,
        k: 2048,
    },
    Shape {
        m: 4096,
        n: 4096,
        k: 4096,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
    Shape {
        m: 16384,
        n: 16384,
        k: 16384,
    },
];

/// Both schedulers on one clock, with the ragged-wave prediction beside them —
/// `cargo oxide run kittens-examples -- clc`.
///
/// The prediction column is computed from [`wave_efficiency`] and not fitted to
/// anything, and it is printed *first* so the table is read as a test of a
/// stated number rather than as a result looking for an explanation. Every size
/// is checked against the CPU reference under every scheduler before it is
/// timed, by the same entry point the rest of the harness uses.
pub fn compare(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm schedulers — min ms over 30 timed launches, both on one shared plan\n\
         `predicted` is the ragged last wave the static grid idles through, which is the\n\
         whole of what a dynamic schedule has to win back: 1 - tiles/(waves x {MAX_CLUSTERS})."
    );
    println!(
        "{:<18}{:>7}{:>7}{:>10}{:>12}{:>11}{:>11}{:>11}",
        "shape", "tiles", "waves", "wave eff", "predicted", "static ms", "clc ms", "measured"
    );
    for &shape in COMPARE_SIZES {
        let Shape { m, n, k } = shape;
        let (waves, efficiency) = wave_efficiency(m, n);
        let mut milliseconds = Vec::new();
        for scheduler in SCHEDULERS {
            eprintln!("{shape} on {}: staging and checking", scheduler.name());
            let (_, timings) = run(context, m, n, k, scheduler, time)?;
            milliseconds.push(timings.min());
        }
        let (static_ms, clc_ms) = (milliseconds[0], milliseconds[1]);
        println!(
            "{:<18}{:>7}{:>7}{:>9.1}%{:>11.1}%{:>11.4}{:>11.4}{:>10.1}%",
            shape,
            tiles(m, n),
            waves,
            100.0 * efficiency,
            100.0 * (1.0 - efficiency),
            static_ms,
            clc_ms,
            100.0 * (1.0 - clc_ms / static_ms)
        );
    }
    Ok(())
}
