//! # GEMM, warp-specialized — the same `C = A·Bᵀ` at one CTA per SM
//!
//! **Status: runs.** [`check`] launches it against the same CPU reference
//! [`crate::gemm::check`] uses, on a B200, with the same exact `==` on bf16
//! words. It is a **variant of [`crate::gemm`] and not a replacement** — that
//! kernel is untouched, and this file exists so the two can be A/B'd.
//!
//! ## What is different, and it is one thing
//!
//! Both kernels compute the same `[256, 256]` pair tile with the same
//! `cta_group::2` UMMA over the same four-chunk `BLOCK_K = 64` stages, and both
//! drain the accumulator straight to global memory through
//! [`kittens::global::store_rows`]. **The epilogue is deliberately identical**,
//! down to the instruction — see "what this PR does not change" below.
//!
//! What moves is *where the overlap comes from*.
//!
//! [`crate::gemm`] gets it **across CTAs**. It is 98392 B of shared memory and
//! 256 accumulator columns, which is two CTAs an SM, so one CTA's epilogue runs
//! against another CTA's MMA. That is what forces its whole budget: 18344 B of
//! shared headroom and 166 registers against an occupancy step at 168.
//!
//! This kernel gets it **inside one CTA**, from warp specialization plus a
//! double-buffered accumulator — which is NVIDIA's `gemm_sol_final`
//! (cuda-oxide `b099f64`,
//! `crates/rustc-codegen-cuda/examples/gemm_sol_final/`), reproduced in this
//! repo's idiom. Six warps per CTA:
//!
//! | warp | role |
//! | ---: | --- |
//! | 0–3 | the epilogue — LDTM the accumulator, store `C` |
//! | 4 | the TMA producer |
//! | 5 | the pair-UMMA issuer (leader rank only) |
//!
//! and **two accumulator stages** in tensor memory ([`ACCUM_STAGES`]), so the
//! MMA warp fills stage `s` while the epilogue warps drain stage `1 - s` from
//! the item before.
//!
//! ## Tensor memory picks the residency, it is not imposed
//!
//! Residency is `min(512 / accumulator_columns, shared per SM / plan)` (#84).
//! Two stages of [`BLOCK_N`] columns is **512 columns — the whole SM's tensor
//! memory** — so `512 / 512 = 1` and one CTA per SM falls out of the
//! double-buffer rather than being chosen. [`kittens::tmem::TmemTile`] already
//! says this: [`kittens::tmem::TmemTile::columns_right`] is the ping-pong.
//!
//! Having given up the second CTA, the kernel then spends the budget the second
//! CTA was holding. The shared plan is [`SHARED_BYTES`] against the 233 472 B
//! an SM divides, so shared memory lands on the same 1 and neither resource is
//! the odd one out. It is worth noting that **`shared_plan(STAGES)` here is
//! byte-for-byte [`crate::gemm`]'s `[256, 256] @ STAGES = 4` rung** — the one
//! its `RUNGS` table computes and refuses to build, because at four warps that
//! plan buys one CTA an SM and nothing to do with it. At six warps and two
//! accumulator stages there is something to do with it, which is the whole
//! argument of this file in one sentence.
//!
//! ## Registers stop binding
//!
//! The register file is 65536 per SM over four sub-partitions — 16384 each,
//! granted per warp in units of 256. [`crate::gemm`] runs 2 CTAs × 4 warps = 8
//! warps an SM, 2 per sub-partition, 8192 registers each: 256 a thread, which
//! is why 168 is a step and why that kernel sits at 166 with two registers of
//! headroom. Here it is 1 CTA × 6 warps = 6 warps an SM, so the *binding*
//! sub-partition still holds 2 and **255 registers a thread is reachable**. The
//! epilogue warps can hold real state; whether they need to is a separate
//! question and `regcount` is what answers it.
//!
//! ## Why [`kittens::pipeline`] did not have to change
//!
//! [`kittens::pipeline::Job::work`]'s doc says role dispatch lives in the job,
//! and it does: [`run`](kittens::pipeline::run),
//! [`run_stealing`](kittens::pipeline::run_stealing),
//! [`ClcQueue`](kittens::pipeline::ClcQueue) and
//! [`grouped`](kittens::pipeline::grouped) are used **unmodified**, and the
//! whole rewrite is inside [`Ws::work`]. Two things are worth recording about
//! how that lands, because neither was obvious.
//!
//! **The item boundary is what lets the pending accumulator be safe.** The
//! epilogue warps drain item `i - 1` while the MMA warp fills item `i`, and
//! nothing inside the item orders those against each other — they touch
//! disjoint TMEM columns, so nothing has to. What *does* need ordering is item
//! `i + 1`'s MMA against item `i - 1`'s drain, since those share a stage, and
//! [`kittens::pipeline::run`]'s per-item `cluster_sync` (with its
//! `tcgen05_fence_before_thread_sync` in front) is exactly that rendezvous, at
//! exactly the right scope. This is [`crate::gemm`]'s `Lcsf` finding again —
//! `lcsf` wants the boundary's *scope* moved inside the item — and here the
//! scaffold's own boundary supplies it for free.
//!
//! **The one thing the trait cannot express is a warp-asynchronous item
//! stream.** The reference has no per-item rendezvous at all: its TMA warp
//! publishes `(tile_m, tile_n)` through a shared `TILE_INFO` word and a
//! `TILE_READY` mbarrier, and its three warp groups sit at *different* items
//! simultaneously, bounded only by the two accumulator stages. A [`Job`] is
//! entered by every thread with one `item`, so the warps here are always on the
//! same item and the epilogue can lag the MMA by exactly one. That is a real
//! difference in the *slack* the two designs have — one item against two — and
//! it is the part of `gemm_sol_final` this port does not reproduce. It is not a
//! bug in the trait; it is the price of the trait's guarantee that barriers can
//! be re-armed per item.
//!
//! Which brings the second consequence, and it is in our favour: because every
//! item re-arms its own barrier set behind the boundary, **every item starts at
//! ring stage 0 and phase 0**. The reference needs `k % 256 == 0` so that
//! `k_iters % 4 == 0` and its producer's global stage index agrees with its
//! consumer's local one across a tile boundary. We need no such thing. This
//! kernel's shape contract is `m % 256 == 0`, `n % 256 == 0`, `k % 64 == 0` —
//! the reference's on `n`, weaker on `k`, and the item boundary is what buys
//! the difference.
//!
//! ## What this file deliberately does not change
//!
//! The reference's epilogue is `stmatrix` into a 64 KiB shared staging buffer
//! and 64-bit vectorized stores out of it. **That is not here.** This kernel
//! keeps [`crate::gemm`]'s register → global epilogue exactly, because
//! otherwise the measurement would move the occupancy/specialization structure
//! and the epilogue in one change and no row of the table would say which one
//! paid. `kittens::epilogue::StoreRing` (#111) exists and the shared→global
//! vectorized store it wants is being built elsewhere; swapping it in is a
//! **follow-up measured on its own**.
//!
//! The epilogue needed no modification to run at six warps, which is worth
//! stating because it was the open question: warps 0–3 cover the CTA's 128
//! accumulator rows at 32 rows each exactly as they did at four warps, and
//! warps 4 and 5 simply do not enter it. [`kittens::global::store_rows`] is not
//! warp-collective — its own docs say so — so a warp that does not call it
//! leaves nothing ill-formed.
//!
//! ## What it measured
//!
//! `cargo oxide run kittens-examples -- ws`, which is
//! `scripts/modal-run ws_bench`, runs this kernel and [`crate::gemm`] and
//! cuBLASLt in one container at 4096³, 8192³ and 16384³ — the three sizes the
//! brief names, and no smaller, because below 4096³ cuBLASLt's own run-to-run
//! spread reaches 77% and no ratio there is quotable. The control is re-run in
//! the same session rather than quoted from `examples/README.md` §7: this
//! change does not touch [`crate::gemm`], but #98 found 2.9% of drift between
//! containers and #109 nearly published a false +3.6% by trusting a baseline
//! from another one.
//!
//! The question this file answers is narrow and worth stating plainly: **does
//! giving up the second CTA for warp specialization win, lose, or wash on a
//! B200 with our epilogue?** The reference's 1524 / 1868 / 2162 TFLOP/s are a
//! B300 against an FP16-in/FP16-out cuBLASLt; ours is a B200 at bf16 where
//! cuBLASLt does 1.77–1.89 PFLOP/s. Their numbers are not a target this has any
//! right to expect, and the honest deliverable is the sign of the delta.

use cuda_device::barrier::Barrier;
use cuda_device::cluster;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, warp};

use crate::bench::{Baseline, Shape, Timings, time};
use crate::gemm::{Scheduler, a_value, b_value, check_c, stage};
use std::error::Error;

use kittens::global::{GlobalRows, store_rows};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcQueue, Job};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};

/// Rows of `C` one CTA owns; the pair covers `2 * BLOCK_M`, which is the `M`
/// the widest `cta_group::2` instruction descriptor names.
const BLOCK_M: usize = 128;
/// Columns of `C` the pair computes, and this CTA's columns per accumulator
/// **stage**. Fixed at 256 rather than swept: [`ACCUM_STAGES`] stages of it
/// have to be the 512 columns an SM has, which is the whole design point.
const BLOCK_N: usize = 256;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K per pipeline stage: one 128-byte swizzle atom of bf16, the only width
/// [`SharedTile::k_walk`] accepts, and four chained K=16 MMA chunks.
const BLOCK_K: usize = 64;
/// Chained MMAs per stage.
const CHUNKS: usize = BLOCK_K / 16;
/// Pipeline depth over K in the shipped entry point — the reference's four.
const STAGES: usize = 4;
/// Accumulator stages in tensor memory: the MMA warp fills one while the
/// epilogue warps drain the other. `ACCUM_STAGES * BLOCK_N` must be the 512
/// columns an SM has, which is asserted below.
const ACCUM_STAGES: u32 = 2;
/// Columns [`alloc_cluster`] is asked for — both stages at once, because a CTA
/// allocates tensor memory exactly once and a stage is a column offset into
/// that allocation rather than an allocation of its own.
const ACCUM_COLUMNS: u32 = ACCUM_STAGES * BLOCK_N as u32;

/// Warps that run the epilogue: 0..[`EPILOGUE_WARPS`], one per 32 accumulator
/// rows, which is the `[32, N]` band [`kittens::tmem::TmemTile::tile`] drains.
const EPILOGUE_WARPS: u32 = (BLOCK_M / 32) as u32;
/// The warp that issues this CTA's TMA loads.
const TMA_WARP: u32 = EPILOGUE_WARPS;
/// The warp that issues the pair's UMMA — in the leader rank only, since a
/// `cta_group::2` MMA is one instruction driving both CTAs' operands.
const MMA_WARP: u32 = EPILOGUE_WARPS + 1;
/// Warps per CTA, and the whole of what "warp-specialized" costs in threads.
const WARPS: u32 = MMA_WARP + 1;
/// Threads the launch declares.
pub const THREADS: u32 = 32 * WARPS;

/// Accumulator columns one warp drains in a single band.
///
/// A `RegTile<32, N, BaseLdtm>` is `32 * N / 32` fp32 values a thread, so 256
/// columns at once would want 256 registers before any of the kernel's own
/// live state — past the 255 the architecture has, at any occupancy. The
/// register headroom this design buys does not move this number; 128 is the
/// widest band that fits and a 256-column stage drains in two of them, exactly
/// as [`crate::gemm`] does.
const DRAIN_N: usize = 128;
/// CTAs in the cluster, and the multiplier on a stage's transaction charge.
const RANKS: u32 = 2;
/// The CTA mask naming every half of the pair.
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
/// The rank that owns the pair's MMA and its stage barriers.
const LEADER: u32 = 0;

type ATile = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
type BTile = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing<const STAGES: usize> = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing<const STAGES: usize> = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// One accumulator stage: this CTA's 128 TMEM lanes by [`BLOCK_N`] fp32
/// columns. Two of these are the SM's whole tensor memory.
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;
/// One warp's band of it, drained [`DRAIN_N`] columns at a time.
type Band = RegTile<32, DRAIN_N, BaseLdtm>;

/// Barriers, the TMEM staging word and the scheduler's queue, in the tail of
/// the shared plan — the same tail [`crate::gemm`] has, because the pipeline
/// this kernel drives is the same pipeline.
const fn scratch_bytes(stages: usize) -> usize {
    2 * stages * 8 + 8 + 8 + ClcQueue::BYTES
}

/// Dynamic shared memory a `stages`-deep launch asks for: the two operand
/// rings and the scratch tail.
///
/// Stated as arithmetic because `#[launch_contract]` takes a literal and the
/// host needs the same number outside any monomorphization. It is not trusted:
/// [`kernels::attach`] carries a codegen-time assert that this agrees with the
/// rings' own `BYTES`, which is the only place the two could drift.
pub const fn shared_plan(stages: usize) -> usize {
    let a = BLOCK_M * BLOCK_K * 2 * stages;
    let b = HALF_N * BLOCK_K * 2 * stages;
    a + b + scratch_bytes(stages)
}

/// Dynamic shared memory the shipped entry point's launch must provide.
pub const SHARED_BYTES: usize = shared_plan(STAGES);

/// `#[launch_contract]` takes literals, so the envelope is written twice and
/// this is what keeps the two in step — the same join [`crate::gemm`]'s
/// `SHARED_BYTES` assert makes, and for the same reason: past 48 KiB the plan
/// needs the opt-in the prepared-launch path issues (#70).
///
/// The tensor-memory line is the one that matters to the design. Two
/// accumulator stages of [`BLOCK_N`] columns is all 512 an SM has; a third
/// stage or a wider `N` would be inexpressible rather than slow.
const _: () = {
    assert!(THREADS == 192);
    assert!(SHARED_BYTES == 131_176);
    assert!(shared_plan(6) == 196_744);
    assert!(ACCUM_COLUMNS == 512);
    assert!(BLOCK_N % DRAIN_N == 0);
};

/// SMs on the device this project targets and measures on — a B200.
const SMS: u32 = 148;
/// CTAs of this kernel one SM holds. **Predicted from
/// `min(512 / columns, shared per SM / plan)` and confirmed by counting**, not
/// queried: `cuOccupancyMaxActiveBlocksPerMultiprocessor` returns 1 for any
/// kernel containing a `tcgen05.alloc` (#77) whatever the true figure is, so it
/// would agree here by accident.
///
/// Both terms give 1. [`ACCUM_COLUMNS`] is 512, so `512 / 512 = 1`; and
/// [`SHARED_BYTES`] is 131 176 against the 233 472 B an SM divides, so
/// `233472 / 131176 = 1` as well. `device-tests`' `tmem residency census`
/// already carries a `cg2 512` rung — a `cta_group::2` launch holding all 512
/// columns across the counted interval — and what it counts there is this
/// kernel's residency, on an instrument that timestamps `%globaltimer` on real
/// CTAs rather than asking a driver.
const CTAS_PER_SM: u32 = 1;
/// Clusters the persistent grid launches at most. A tuning constant and not a
/// correctness one — [`pipeline::run`] walks every item whatever the grid is —
/// and the constant [`Scheduler::Stealing`] exists to delete.
const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / RANKS;
const _: () = assert!(MAX_CLUSTERS == 74);

/// [`pipeline::grouped`]'s width in tile-rows, carried across from
/// [`crate::gemm`]'s measured value.
///
/// **This is inherited and not measured here**, and that is a stated
/// limitation rather than a claim. The width works by shaping a *wave*'s
/// operand footprint, and this kernel's wave is 74 clusters where
/// [`crate::gemm`]'s is 148 — so the value that won there is not the value that
/// should win here, and a sweep is the obvious follow-up. Carrying 8 keeps the
/// A/B a comparison of kernels rather than of traversals, which is the more
/// important property for one table.
const GROUP: u32 = 8;

/// One output tile of `C`, as the persistent grid's work item — the same
/// fields [`crate::gemm`]'s job carries, plus the second accumulator stage.
#[derive(Clone, Copy)]
struct Tile<const STAGES: usize> {
    a_ring: ARing<STAGES>,
    b_ring: BRing<STAGES>,
    /// Filled by the TMA, drained by the MMA. In the leader's copy the whole
    /// pair's four tiles complete on one barrier.
    load: SemaphoreRing<STAGES>,
    /// Released by the MMA's own commit, in both CTAs.
    free: SemaphoreRing<STAGES>,
    /// The item's accumulator complete, multicast by the MMA. Waited by
    /// **every** thread at the end of the item, which is what lets the next
    /// item's epilogue read it with no barrier of its own.
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    /// Stage 0 of the accumulator; stage 1 is [`Self::accumulator`]'s
    /// `columns_right`.
    accumulator: Accumulator,
    c: GlobalRows<Bf16>,
    tiles_m: u32,
    tiles_n: u32,
    group: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl<const STAGES: usize> Tile<STAGES> {
    /// The accumulator stage `index` names — the ping-pong, and the whole of
    /// what [`ACCUM_STAGES`] buys.
    #[inline(always)]
    fn accumulator(&self, index: u32) -> Accumulator {
        if index % ACCUM_STAGES == 0 {
            self.accumulator
        } else {
            self.accumulator.columns_right(BLOCK_N as u32)
        }
    }

    /// Issue this rank's half of every K block, charging the leader's stage
    /// barrier for the whole pair.
    ///
    /// Unlike [`crate::gemm`]'s producer this issues the *entire* walk in one
    /// call and never yields to an epilogue, because it is not the warp that
    /// runs one: that is the point of a dedicated TMA warp, and it is the
    /// difference between this and `Lcsf`, which had to split its walk at
    /// `FILL` so warp 0 could go and store.
    ///
    /// # Safety
    ///
    /// One thread of the TMA warp, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
        unsafe {
            let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = 0u32;
            while k < self.k_blocks {
                self.free.wait_recycled(k);
                let stage = self.load.sem(k).at_rank(LEADER);
                let column = (BLOCK_K as u32 * k) as i32;
                let a_bytes = self
                    .a_ring
                    .tile(k)
                    .tma_load_2d_arriving_at(self.a_map, column, a_row, stage);
                let b_bytes = self
                    .b_ring
                    .tile(k)
                    .tma_load_2d_arriving_at(self.b_map, column, b_row, stage);
                if self.rank == LEADER {
                    self.load
                        .sem(k)
                        .expect_tx((a_bytes + b_bytes).across_ranks(RANKS));
                }
                k += 1;
            }
        }
    }

    /// Chain the whole K walk into `stage` of the accumulator and publish it.
    ///
    /// # Safety
    ///
    /// One thread of the MMA warp of the leader rank, with `stage` holding
    /// nothing any epilogue is still reading — which the item boundary two
    /// items back is what guarantees.
    #[inline(always)]
    unsafe fn multiply(&self, stage: u32) {
        unsafe {
            let accumulator = self.accumulator(stage);
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                mma_walk_cg2::<Bf16, CHUNKS>(
                    accumulator.raw(),
                    self.a_ring.tile(k).k_walk(),
                    self.b_ring.tile(k).k_walk(),
                    MmaShape::M256_N256,
                    k > 0,
                );
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// This warp's band of `stage`, stored to the tile `item` names —
    /// [`crate::gemm::Tile::drain`] unchanged, on an epilogue warp.
    ///
    /// # Safety
    ///
    /// Every lane of a warp below [`EPILOGUE_WARPS`], with `stage` holding
    /// `item`'s completed accumulator and nothing in flight that will
    /// overwrite it.
    #[inline(always)]
    unsafe fn drain(&self, item: u32, stage: u32) {
        unsafe {
            let accumulator = self.accumulator(stage);
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
            let row_base =
                2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * self.warp_id;
            let column_base = BLOCK_N as u32 * tile_n;
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band: Band = accumulator.tile(32 * self.warp_id, column);
                store_rows(self.c, row_base, column_base + column, self.lane, band);
                column += DRAIN_N as u32;
            }
        }
    }
}

/// The warp-specialized job: one output tile per item, with the epilogue one
/// item behind the MMA and on warps of its own.
///
/// The deferral is [`crate::gemm`]'s `Lcsf` — the accumulator lives outside the
/// item loop, so an undrained one survives the item boundary — and the warp
/// split is what makes the deferral pay. Under `Lcsf` the drain sits *between*
/// the producer's fill prefix and the rest of its walk, because warp 0 is both
/// the producer and a quarter of the epilogue; here the producer never stops.
#[derive(Clone, Copy)]
struct Ws<const STAGES: usize> {
    tile: Tile<STAGES>,
    /// The item whose accumulator is still in tensor memory, or [`Self::NONE`].
    pending: u32,
    /// Items this **cluster** has run, which is what selects the accumulator
    /// stage — and it cannot be the item index.
    ///
    /// [`pipeline::run`] strides items by [`MAX_CLUSTERS`], which is 74 and
    /// even, so `item % 2` is *constant* for a given cluster and would put
    /// every item in the same stage; [`pipeline::run_stealing`] hands out
    /// whatever the hardware cancelled, which has no parity at all. The
    /// ping-pong is a property of the sequence a cluster walks, so it counts
    /// the sequence.
    sequence: u32,
}

impl<const STAGES: usize> Ws<STAGES> {
    /// No accumulator owed. `u32::MAX` is not a reachable item: the tile grid
    /// is `tiles_m * tiles_n` and both are `u32`.
    const NONE: u32 = u32::MAX;

    #[inline(always)]
    fn new(tile: Tile<STAGES>) -> Self {
        Self {
            tile,
            pending: Self::NONE,
            sequence: 0,
        }
    }

    /// The stage [`Self::pending`] sits in — the one the current item is *not*
    /// using.
    ///
    /// `(sequence + 1) % 2` reads as the next stage and is also the previous
    /// one, which is what makes the same expression right in [`Self::work`]
    /// (where `sequence` is the current item's) and in [`Self::finish`] (where
    /// it has already been advanced past the last).
    #[inline(always)]
    fn pending_stage(&self) -> u32 {
        (self.sequence + 1) % ACCUM_STAGES
    }

    /// Store the last item's accumulator, which no later item is coming to
    /// overlap. Once, after [`pipeline::run`] returns and before the pair gives
    /// its tensor memory back.
    ///
    /// A cluster that ran no items owes nothing, which is the static-schedule
    /// case where [`MAX_CLUSTERS`] exceeds the tile count.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA, before `release`.
    #[inline(always)]
    unsafe fn finish(&self) {
        unsafe {
            if self.pending != Self::NONE && self.tile.warp_id < EPILOGUE_WARPS {
                self.tile.drain(self.pending, self.pending_stage());
            }
        }
    }
}

impl<const STAGES: usize> Job for Ws<STAGES> {
    /// The pair shares one barrier set — the peer aims its TMA at the leader's
    /// stage barrier and the leader's MMA arrives in the peer's `free` and
    /// `done` — so the item boundary that re-arms them is the cluster's.
    const RANKS: u32 = crate::gemm_ws::RANKS;

    /// Every barrier takes exactly one arrival: the leader's stage barrier from
    /// the TMA transaction count, `free` and `done` from the MMA commit.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::init`]; [`pipeline::run`] owns the thread and the
    /// ordering.
    #[inline(always)]
    unsafe fn init(&self, _item: u32) {
        unsafe {
            self.tile.load.init_all(1);
            self.tile.free.init_all(1);
            self.tile.done.init(1);
        }
    }

    /// # Safety
    ///
    /// As [`Semaphore::inval`].
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe {
            self.tile.load.inval_all();
            self.tile.free.inval_all();
            self.tile.done.inval();
        }
    }

    /// One item, with the three roles side by side.
    ///
    /// The ordering argument, in full, because every way it can be wrong is
    /// silent:
    ///
    /// - The epilogue warps read stage `1 - s` (item `i - 1`); the MMA warp
    ///   writes stage `s` (item `i`). **Disjoint columns**, so nothing inside
    ///   the item orders them and nothing has to — that is what the second
    ///   accumulator stage is for.
    /// - Item `i + 1`'s MMA writes stage `1 - s`, which item `i - 1`'s epilogue
    ///   was reading. Those are separated by [`pipeline::run`]'s item boundary
    ///   — a `cluster_sync`, so it covers the *peer's* epilogue too, which a
    ///   `bar.sync` would not — with the harness'
    ///   `tcgen05_fence_before_thread_sync` in front of it retiring the LDTM.
    /// - The epilogue of item `i` needs item `i`'s MMA complete. It gets that
    ///   from the `done.wait` **every thread** takes at the end of this
    ///   function, one item earlier, plus that same boundary. So the drain at
    ///   the top of the next item takes no barrier of its own, and `done` can
    ///   be one semaphore re-armed per item rather than a ring.
    ///
    /// # Safety
    ///
    /// Every thread of both CTAs must enter with the same `item`, which is what
    /// [`pipeline::run`]'s cluster-strided map gives, and the maps must cover
    /// the tile it names.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            let stage = self.sequence % ACCUM_STAGES;

            if tile.warp_id == TMA_WARP && tile.lane == 0 {
                tile.produce(tile_m, tile_n);
            }
            if tile.rank == LEADER && tile.warp_id == MMA_WARP && tile.lane == 0 {
                tile.multiply(stage);
            }
            if tile.warp_id < EPILOGUE_WARPS && self.pending != Self::NONE {
                tile.drain(self.pending, self.pending_stage());
            }

            // Every thread, including the epilogue warps that have just
            // finished the previous item. This is the only rendezvous inside
            // the item, and it is what makes the *next* item's drain free.
            tile.done.wait(0);
            self.pending = item;
            self.sequence += 1;
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The item and the work queue, laid over the one shared plan every entry
    /// point launches with. Everything here spans items — the rings, the
    /// barriers, the operand maps, and the pair's TMEM allocation, whose
    /// `alloc_cluster` is a whole-cluster collective with a `cluster_sync` in
    /// it and must not sit inside anybody's item loop.
    ///
    /// # Safety
    ///
    /// The launch geometry's, and the operands': both maps must describe live
    /// buffers covering `k_blocks * BLOCK_K` along K and the full extent the
    /// item loop walks, and `c` must hold `ldc` columns for every row of it.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach<const STAGES: usize>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<STAGES>, ClcQueue) {
        // Fired at codegen, which is the only place the ring byte counts are
        // known — and the reason `cargo check` cannot stand in for
        // `scripts/modal-run build`.
        const {
            assert!(
                shared_plan(STAGES)
                    == ARing::<STAGES>::BYTES + BRing::<STAGES>::BYTES + scratch_bytes(STAGES)
            );
            assert!(
                (ARing::<STAGES>::BYTES + BRing::<STAGES>::BYTES + 2 * STAGES * 8 + 8 + 8)
                    % ClcQueue::ALIGNMENT
                    == 0
            );
        };
        unsafe {
            let rings = ARing::<STAGES>::BYTES + BRing::<STAGES>::BYTES;
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let scratch = smem.add(rings);
            let tmem_slot = scratch.add(2 * STAGES * 8 + 8) as *mut u32;

            let tile = Tile {
                a_ring: ARing::<STAGES>::attach(smem),
                b_ring: BRing::<STAGES>::attach(smem.add(ARing::<STAGES>::BYTES)),
                load: SemaphoreRing::<STAGES>::attach(scratch as *mut Barrier),
                free: SemaphoreRing::<STAGES>::attach((scratch as *mut Barrier).add(STAGES)),
                done: Semaphore::attach((scratch as *mut Barrier).add(2 * STAGES)),
                a_map,
                b_map,
                // Both stages in one allocation. The allocator's unit is the
                // CTA, so a second stage is a column offset and never a second
                // `tcgen05.alloc`.
                accumulator: Accumulator::from_raw(alloc_cluster(tmem_slot, ACCUM_COLUMNS)),
                c: GlobalRows::<Bf16>::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                group,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
            };
            (
                tile,
                ClcQueue::attach(smem.add(rings + 2 * STAGES * 8 + 8 + 8)),
            )
        }
    }

    /// Give the pair's accumulator back.
    ///
    /// The `cluster_sync` is not decoration and the reference carries the same
    /// one with a comment on it: without a whole-cluster rendezvous before
    /// exit, a fast CTA can leave while its partner is still driving a
    /// cross-CTA operation, which faults as CUDA_EXCEPTION_17 rather than
    /// producing a wrong number. Here it also covers the cluster that got no
    /// items at all and still owes a deallocation in step with its peer.
    ///
    /// # Safety
    ///
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired — which is `finish` having run.
    #[inline(always)]
    unsafe fn release<const STAGES: usize>(tile: &Tile<STAGES>) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), ACCUM_COLUMNS);
        }
    }

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `[256, 256]` output tile per work
    /// item, six warps to a CTA and one CTA to an SM.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus: the grid must be a whole number of clusters and
    /// `tiles_m * tiles_n` the item count they are to cover.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_176,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            job.finish();
            release(&job.tile);
        }
    }

    /// The same kernel on the hardware's schedule — one cluster per output
    /// tile, and a cluster that finishes cancels one the scheduler has not
    /// launched yet.
    ///
    /// It matters more here than it does for [`crate::gemm`]. A wave is 74
    /// clusters where that kernel's is 148, so the ragged last wave is a
    /// coarser quantization of the same tile grid and the static stride has
    /// more to lose.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus [`pipeline::run_stealing`]'s: the grid is exactly
    /// `RANKS` × the tile count, one-dimensional, on sm_100a.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_176,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_clc(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, queue) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::new(tile);
            pipeline::run_stealing(&mut job, queue);
            job.finish();
            release(&job.tile);
        }
    }

    /// The same kernel **six** pipeline stages deep — 196 744 B.
    ///
    /// It is here because the depth is *free* and that is a consequence of the
    /// design worth measuring rather than asserting. Tensor memory has already
    /// fixed residency at one CTA an SM, so shared memory has 102 296 B doing
    /// nothing at four stages; the reference spends its own surplus on a 64 KiB
    /// `stmatrix` staging buffer, and with the register epilogue kept there is
    /// nothing to spend it on but pipeline. Under [`crate::gemm`]'s two CTAs an
    /// SM the same two stages would be an occupancy step and #98 priced that at
    /// 25–44%; here they cost nothing that any resource can see.
    ///
    /// # Safety
    ///
    /// As [`gemm_ws`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_744,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_s6(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<6>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            job.finish();
            release(&job.tile);
        }
    }
}

/// Which entry point a plan launches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Entry {
    /// `gemm_ws` / `gemm_ws_clc` — four stages, the reference's depth.
    S4,
    /// `gemm_ws_s6` — six, static only.
    S6,
}

impl Entry {
    /// Pipeline depth, which is the only thing an entry varies.
    pub fn stages(self) -> usize {
        match self {
            Entry::S4 => STAGES,
            Entry::S6 => 6,
        }
    }

    /// Dynamic shared memory its launch declares.
    pub fn shared(self) -> usize {
        shared_plan(self.stages())
    }

    /// What the tables call it.
    pub fn name(self) -> String {
        format!("ws s{}", self.stages())
    }
}

/// How a launch takes its work.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub scheduler: Scheduler,
    pub group: u32,
    pub entry: Entry,
}

impl Plan {
    /// The kernel as it ships.
    pub fn new(scheduler: Scheduler) -> Self {
        Plan {
            scheduler,
            group: GROUP,
            entry: Entry::S4,
        }
    }
}

/// Tiles of `C` a `[m, n]` output has along each axis.
fn tile_grid(m: usize, n: usize) -> (u32, u32) {
    ((m / (2 * BLOCK_M)) as u32, (n / BLOCK_N) as u32)
}

/// Tiles of `C` a `[m, n]` output has, which is the item count.
fn tiles(m: usize, n: usize) -> u32 {
    let (rows, columns) = tile_grid(m, n);
    rows * columns
}

/// Blocks a launch asks for. The static grid is capped at [`MAX_CLUSTERS`]; the
/// stealing grid is one cluster per tile, since CLC caps it by leaving clusters
/// unlaunched — one branch reads a measured constant and the other reads the
/// problem.
fn grid_for(scheduler: Scheduler, m: usize, n: usize) -> u32 {
    let clusters = match scheduler {
        Scheduler::Static => tiles(m, n).min(MAX_CLUSTERS),
        Scheduler::Stealing => tiles(m, n),
    };
    RANKS * clusters
}

/// The `then` a run that is only being checked passes.
fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch `[m, k] · [n, k]ᵀ`, compare every element of `C` against the CPU
/// reference, and only then hand the launch to `then`.
///
/// The order is the design and it is [`crate::gemm::run`]'s: `then` — the
/// clock — is unreachable from a launch whose output was wrong, so no
/// throughput figure can be printed for a kernel that did not compute. The
/// reference itself is shared with [`crate::gemm`] rather than copied, through
/// `a_value` / `b_value` / `check_c`, which is what makes "the two kernels are
/// checked against the same thing" a property of the code.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    plan: Plan,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    use kittens::global::GlobalLayout;

    // The shape contract, and the kernel bounds-checks none of it. `k % 64`
    // rather than the reference's `k % 256`: barriers are re-armed per item, so
    // every item starts at ring stage 0 and no tile boundary has to land on a
    // pipeline cycle.
    if m % (2 * BLOCK_M) != 0 || n % BLOCK_N != 0 || k % BLOCK_K != 0 {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{BLOCK_N}x{BLOCK_K} tiling",
            2 * BLOCK_M
        )
        .into());
    }
    if plan.group == 0 {
        return Err("a traversal group width of 0 has no tiles in it".into());
    }

    let stream = context.default_stream();
    // SAFETY: the artifact is this crate's own and the contracts declare the
    // ABI compiled.
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

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let blocks = grid_for(plan.scheduler, m, n);
    let (tiles_m, tiles_n) = tile_grid(m, n);
    let k_blocks = (k / BLOCK_K) as u32;
    let config = LaunchConfig1D::new(blocks, THREADS, plan.entry.shared() as u32);

    let (stream_ref, module_ref) = (&stream, &module);
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    // SAFETY (every arm): both maps describe live buffers covering the walk the
    // grid takes, and `c` holds `n` columns for every row of it. The stealing
    // entry additionally takes a grid of exactly one cluster per tile, which is
    // what `grid_for` gives it.
    macro_rules! launcher {
        ($prepare:ident, $launch:ident) => {{
            let prepared = module_ref.$prepare(config)?;
            let launch = move |c: &mut DeviceBuffer<u16>| -> Result<(), Box<dyn Error>> {
                unsafe {
                    module_ref.$launch(
                        stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, plan.group,
                        k_blocks, n as u32, c,
                    )?
                };
                Ok(())
            };
            Box::new(launch) as Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>>
        }};
    }
    let launch_once = match (plan.entry, plan.scheduler) {
        (Entry::S4, Scheduler::Static) => launcher!(prepare_gemm_ws, gemm_ws),
        (Entry::S4, Scheduler::Stealing) => launcher!(prepare_gemm_ws_clc, gemm_ws_clc),
        (Entry::S6, Scheduler::Static) => launcher!(prepare_gemm_ws_s6, gemm_ws_s6),
        // A depth comparison under a moving scheduler would be two variables.
        (entry, scheduler) => {
            return Err(format!(
                "{entry:?} has no entry point on the {} schedule",
                scheduler.name()
            )
            .into());
        }
    };
    launch_once(&mut c)?;
    let worst = check_c(&c.to_host_vec(&stream)?, m, n, k)?;
    let label = format!("{m}x{n}x{k} exact, worst |rel| {worst:.2e} against the fp32 reference");

    let after = then(&stream, &mut || launch_once(&mut c))?;
    Ok((label, after))
}

/// A small correctness size — four tiles, so both axes of the item map move.
const M: usize = 512;
const N: usize = 512;
const K: usize = 256;

/// The size whose only job is to give every cluster **more than one work
/// item**, which is where this kernel's own failure modes live.
///
/// [`M`]x[`N`] is four tiles over 74 clusters, so it never enters
/// [`pipeline::run`]'s loop twice and would pass against a kernel with no
/// deferred epilogue and no accumulator ping-pong at all. Everything warp
/// specialization adds needs a second item to happen: a stage the MMA
/// overwrites while an epilogue warp is still reading it, a `pending` dropped
/// at the end of the loop, a stage index taken from the item rather than from
/// the sequence. 256 tiles over 74 clusters is three or four items for every
/// cluster with a ragged tail, at a `K` that wraps the ring.
const ITEMS_M: usize = 4096;
const ITEMS_N: usize = 4096;
const ITEMS_K: usize = 256;

/// Traversal widths the correctness run walks, chosen to break
/// [`pipeline::grouped`]'s short last group rather than to be fast — the same
/// three [`crate::gemm::check`] uses and for the same reason.
const CHECK_GROUPS: [u32; 3] = [1, 3, 6];

/// The correctness run: two sizes, three traversals, both schedulers, both
/// depths, checked and nothing timed.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        let mut rounding = None;
        for scheduler in [Scheduler::Static, Scheduler::Stealing] {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler,
                    group,
                    entry: Entry::S4,
                };
                let (label, _) = run(context, m, n, k, plan, nothing_after)?;
                rounding.get_or_insert(label);
            }
            let clusters = grid_for(scheduler, m, n) / RANKS;
            notes.push(format!(
                "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} tiles over {clusters} \
                 clusters, one deferred accumulator per cluster in flight)",
                scheduler.name(),
                tiles(m, n)
            ));
        }
        // The deeper rung under the same gate: a ring depth is a different
        // `wait_recycled` parity and a different producer/consumer distance,
        // and both go wrong as a wrong `C` rather than as a fault.
        for group in CHECK_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                entry: Entry::S6,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} ({} B)",
            Entry::S6.name(),
            Entry::S6.shared()
        ));
        if let Some(label) = rounding {
            notes.push(label);
        }
    }
    Ok(notes.join(", "))
}

/// Sizes the comparison runs at, and no smaller.
///
/// Below 4096³ cuBLASLt's own run-to-run spread reaches 77%, so a ratio there
/// says more about the library's launch path than about either kernel. These
/// are also the three sizes the reference publishes, which is the only thing
/// making its numbers and these numbers even nominally about the same question.
const SIZES: [Shape; 3] = [
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

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// Waves the static grid takes, and how full the last one is.
fn wave_efficiency(m: usize, n: usize) -> (u32, f64) {
    let tiles = tiles(m, n);
    let waves = tiles.div_ceil(MAX_CLUSTERS);
    (waves, tiles as f64 / (waves * MAX_CLUSTERS) as f64)
}

fn timed(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
    plan: Plan,
) -> Result<Timings, Box<dyn Error>> {
    eprintln!(
        "{shape} on {} / {}: staging and checking",
        plan.entry.name(),
        plan.scheduler.name()
    );
    run(context, shape.m, shape.n, shape.k, plan, time).map(|(_, timings)| timings)
}

/// The A/B — `cargo oxide run kittens-examples -- ws`, which is
/// `scripts/modal-run ws_bench`.
///
/// **The control is re-measured in this session, not quoted.** Nothing here
/// changes [`crate::gemm`], so its numbers ought to be the ones
/// `examples/README.md` §7 already carries; #98 found 2.9% of drift between
/// containers and #109 came within one paragraph of publishing a false +3.6%
/// against a baseline that had moved under it. A control that is not moving has
/// to be one measured beside the thing it controls.
pub fn compare(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm warp specialization — min ms over 30 timed launches, every row checked\n\
         element-by-element against the same CPU reference before it was timed.\n\
         `gemm` is examples/src/gemm.rs unchanged: [256,256] at 3 stages, 4 warps, 98392 B,\n\
         2 CTAs an SM, 256 accumulator columns. `ws` is this file: the same pair tile and\n\
         the same register epilogue at 6 warps, {} B, 1 CTA an SM, 512 accumulator\n\
         columns in two ping-ponged stages.\n\
         One variable moves — where the overlap comes from — and the epilogue is\n\
         deliberately identical, so a delta here is the occupancy/specialization\n\
         structure and nothing else.",
        SHARED_BYTES
    );

    println!("\n1. the two designs, at the three sizes the reference publishes");
    println!(
        "{:<18}{:>8}{:>8}{:>10}{:>11}{:>12}{:>11}{:>12}{:>10}",
        "shape",
        "ws tiles",
        "waves",
        "wave eff",
        "gemm ms",
        "gemm TF/s",
        "ws ms",
        "ws TF/s",
        "ws vs gemm"
    );
    let mut measured = Vec::new();
    for shape in SIZES {
        eprintln!("{shape} on gemm (the control): staging and checking");
        let control = crate::gemm::bench(context, shape)?.min();
        let ours = timed(context, shape, Plan::new(Scheduler::Static))?.min();
        let (waves, efficiency) = wave_efficiency(shape.m, shape.n);
        println!(
            "{:<18}{:>8}{:>8}{:>9.1}%{:>11.4}{:>12.1}{:>11.4}{:>12.1}{:>10}",
            shape,
            tiles(shape.m, shape.n),
            waves,
            100.0 * efficiency,
            control,
            tflops(shape, control),
            ours,
            tflops(shape, ours),
            format!("{:+.1}%", 100.0 * (control / ours - 1.0))
        );
        measured.push((shape, control, ours));
    }

    println!(
        "\n2. the two levers this design point unlocks, both free in resources.\n\
         `clc` is the hardware schedule: a wave is 74 clusters here against gemm's 148, so\n\
         the ragged last wave is a coarser quantization of the same tile grid.\n\
         `s6` is six pipeline stages instead of four — 196744 B, which tensor memory has\n\
         already made free, since 512 accumulator columns fixed residency at 1 CTA/SM\n\
         before shared memory was consulted at all."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "shape", "ws ms", "ws clc ms", "vs static", "ws s6 ms", "s6 TF/s", "vs s4"
    );
    for &(shape, _, ours) in &measured {
        let stealing = timed(
            context,
            shape,
            Plan {
                scheduler: Scheduler::Stealing,
                group: GROUP,
                entry: Entry::S4,
            },
        )?
        .min();
        let deep = timed(
            context,
            shape,
            Plan {
                scheduler: Scheduler::Static,
                group: GROUP,
                entry: Entry::S6,
            },
        )?
        .min();
        println!(
            "{:<18}{:>12.4}{:>12.4}{:>12}{:>12.4}{:>12.1}{:>12}",
            shape,
            ours,
            stealing,
            format!("{:+.1}%", 100.0 * (ours / stealing - 1.0)),
            deep,
            tflops(shape, deep),
            format!("{:+.1}%", 100.0 * (ours / deep - 1.0))
        );
    }

    println!(
        "\n3. against cuBLASLt on the same device in the same container — the denominator,\n\
         and the drift control that says how much of the delta above is the session."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>16}{:>16}",
        "shape", "cuBLASLt ms", "theirs TF/s", "gemm/theirs", "ws/theirs"
    );
    for &(shape, control, ours) in &measured {
        let Some(baseline) = baseline else {
            println!(
                "no cuBLASLt column: built without --features cublas. modal_app.py::ws_bench\n\
                 turns it on, and a ratio is the point of this table."
            );
            break;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        println!(
            "{:<18}{:>14.4}{:>14.1}{:>16.3}{:>16.3}",
            shape,
            theirs,
            tflops(shape, theirs),
            theirs / control,
            theirs / ours
        );
    }
    Ok(())
}
