//! # GEMM — `C = A·Bᵀ` on the cta_group::2 cluster path
//!
//! **Status: runs.** [`check`] launches it against a CPU reference on a B200
//! (`modal run modal_app.py::examples`). The operands are small integers, so
//! every product and every partial sum is exact and the comparison is `==`
//! rather than a tolerance — a mismatch is a wrong index, never rounding.
//!
//! A pair of CTAs forms a cluster and shares one `M256_N256` UMMA. Both
//! operands are split across the pair — each CTA stages its own 128 rows of
//! `A` and its own 128 columns of `B` at the *same* shared offsets, and the
//! instruction reads both CTAs' shared memory over the cluster interconnect.
//! The accumulator splits the same way along M, so each CTA drains its own
//! `[128, 256]` band of `C`.
//!
//! **The pair tile and the pipeline depth are const parameters now** (#87).
//! [`Tile`] takes the pair's columns and `STAGES`, [`RUNGS`] is the sweep those
//! two were chosen by, and [`SHIPPED`] names this rung. `[256, 256]` at three
//! stages measured **+11.6% at 8192³ and +21.6% at 16384³** against the
//! `[256, 128]` tile this file carried through #102, which is 1457.1 and
//! **1622.5 TFLOP/s** and 0.826 and **0.886 of cuBLASLt**. The mechanism is
//! arithmetic intensity and nothing else: re-running `gemm-depth` on the new
//! tile moves the fit's *slope* 21–23% and moves its intercept the wrong way,
//! so the "half the tiles, half the item boundaries" argument — which looks
//! like the stronger one — is worth less than nothing here. `examples/README.md`
//! §7 has the four losing rungs, the control that separates the two, and the
//! small sizes this cost.
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
//! **Every number in the paragraph above belongs to a kernel that no longer
//! exists**, and it is kept because two later sections are quoted against it.
//! #91 halved the boundary, #102 moved the mid-`K` rows, and #87 changed the
//! tile the whole sweep was run at. Re-fitted on this kernel the same sweep
//! gives **1826–1862 TFLOP/s of steady state and 21.5–27.2 µs a tile**, and the
//! boundary is *larger* per tile than it was, on a third fewer tiles. §7 has
//! both fits side by side; the one thing that survives unchanged is that the
//! intercept is everything not scaling with `K`, which after #102 is no longer
//! only the item boundary.
//!
//! ## The item map is grouped, not row-major
//!
//! That aspect-ratio sweep is what #89 was started from, and it is worth
//! restating why it is evidence rather than a suggestion. Five shapes, every one
//! of them identical in flops, tiles, waves, grid, `C` bytes, operand bytes and
//! arithmetic intensity, differing only in `M : N` — and throughput moved
//! **1123.7 → 915.7 TFLOP/s**, with the best and worst rows exact transposes of
//! each other. Nothing but the order the output is walked in can explain a 23%
//! swing across rows that request the same bytes from the same memory system.
//!
//! So the item map is [`pipeline::grouped`] at [`GROUP`] tile-rows rather than
//! `(item / tiles_n, item % tiles_n)`. What that changes is the *working set of
//! a wave*: 222 clusters walked row-major sit on `ceil(222 / tiles_n)` rows of
//! tiles and span as much of `N` as the shape allows, where grouped they sit on
//! a block whose shape the aspect ratio no longer controls. [`swizzle`] sweeps
//! the width and keeps the losers, and `examples/README.md` §7 has both.
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
/// Columns of `C` the *pair* computes in the kernel this file ships, split
/// `BLOCK_N / 2` per CTA along the operand's N axis.
///
/// Since #87 this is one value of a swept parameter rather than the only one
/// the file knows: [`Tile`] takes the pair's columns and the pipeline depth as
/// const parameters, [`RUNGS`] is the sweep, and this names the rung the
/// headline entry points are built at.
///
/// **It was 128 through #102 and the sweep moved it.** `[256, 256]` at three
/// stages measured **+11.6% at 8192³ and +21.6% at 16384³** against `[256,
/// 128]` at three, taking the kernel to 1457.1 and **1622.5 TFLOP/s** and from
/// 0.740 to 0.826 and 0.728 to **0.886 of cuBLASLt**. It costs one CTA an SM
/// and no registers, and `examples/README.md` §7 has the four losing rungs and
/// the control that separates the two mechanisms.
///
/// **What it also costs is generality, and that is not free.** A launch must
/// now have `n % 256 == 0` where 128 used to do, so this kernel computes a
/// narrower set of shapes than it did — the direction #92 already named as
/// where a like-for-like rate against a general library flatters us.
const BLOCK_N: usize = 256;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K per pipeline stage: one 128-byte swizzle atom of bf16, the only width
/// [`SharedTile::k_walk`] accepts, and four chained K=16 MMA chunks.
const BLOCK_K: usize = 64;
/// Chained MMAs per stage.
const CHUNKS: usize = BLOCK_K / 16;
/// Pipeline depth over K, in the shipped kernel.
const STAGES: usize = 3;
/// One warp per 32 accumulator rows, which is what a `[32, N]` drain wants.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32;
/// Accumulator columns one warp drains in a single band.
///
/// A `RegTile<32, N, BaseLdtm>` is `32 * N / 32` fp32 values a thread, so a
/// warp draining 256 columns at once would want **256 registers** before any
/// of the kernel's own live state — past the 255 the architecture has, and the
/// whole of what a `BLOCK_N = 256` pair tile costs that #87's table does not
/// mention. 128 is the widest band that fits, so a 256-column tile drains in
/// two of them and a 128-column tile drains in the one band it always did:
/// the loop below is a single iteration at `BLOCK_N = 128` and the shipped
/// kernel's codegen is unchanged, which `regcount` is what confirms.
const DRAIN_N: usize = 128;
/// CTAs in the cluster. Also the multiplier on a stage's transaction charge:
/// both ranks stage the same two tile types at the same shared offsets, so the
/// whole stage is one rank's charge twice over.
const RANKS: u32 = 2;
/// The CTA mask naming every half of the pair.
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
/// The rank that owns the pair's MMA, its accumulator and its stage barriers.
const LEADER: u32 = 0;

/// This CTA's `A` rows, K-major. The one operand tile no rung moves: the
/// pair's `M` is fixed at 256 by the widest `MmaShape` there is.
type ATile = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// This CTA's `B` columns, also K-major — so the MMA carries no transpose
/// bits and computes `A·Bᵀ`.
type BTile<const HALF_N: usize> = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing<const STAGES: usize> = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing<const HALF_N: usize, const STAGES: usize> =
    SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// This CTA's half of the pair's accumulator: 128 TMEM lanes by `BLOCK_N`
/// fp32 columns. The column count is what charges tensor memory, so it is also
/// the `512 / columns` half of the residency this kernel gets.
type Accumulator<const BLOCK_N: usize> = TmemTile<BLOCK_M, BLOCK_N>;
/// One warp's band of it, drained — [`DRAIN_N`] columns at a time.
type Band = RegTile<32, DRAIN_N, BaseLdtm>;

/// Barriers, the TMEM staging word and the scheduler's queue, in the tail of
/// the shared plan: the two `stages`-deep rings' semaphores, the MMA-complete
/// semaphore, one `u32`, and [`ClcQueue::BYTES`] for the hardware work queue.
const fn scratch_bytes(stages: usize) -> usize {
    2 * stages * 8 + 8 + 8 + ClcQueue::BYTES
}

/// Dynamic shared memory a `[2·BLOCK_M, block_n]` pair tile `stages` deep asks
/// for: the two operand rings and the scratch tail.
///
/// Stated as arithmetic because `#[launch_contract]` takes a literal and a
/// host-side rung table needs the same number outside any monomorphization.
/// It is not trusted: [`attach`] carries a codegen-time assert that this agrees
/// with the rings' own `BYTES`, per rung, which is the only place the two
/// could ever drift.
pub const fn shared_plan(block_n: usize, stages: usize) -> usize {
    let a = BLOCK_M * BLOCK_K * 2 * stages;
    let b = (block_n / 2) * BLOCK_K * 2 * stages;
    a + b + scratch_bytes(stages)
}

/// The UMMA shape a pair tile of `block_n` columns issues.
///
/// `M` is 256 in both — the pair's rows, and the widest `M` tcgen05 has, which
/// is why the tile sweep can only move `N`. A rung whose columns name no shape
/// fails at codegen rather than issuing the wrong descriptor into the right
/// accumulator, which does not fault and computes wrong numbers.
const fn pair_shape(block_n: usize) -> MmaShape {
    match block_n {
        128 => MmaShape::M256_N128,
        256 => MmaShape::M256_N256,
        _ => panic!("no cta_group::2 MmaShape covers this pair tile's columns"),
    }
}

/// Dynamic shared memory the shipped kernel's launch must provide.
///
/// Every scheduler below launches with the *same* plan, including the static
/// one that never touches the queue. Twenty-four bytes is not worth a second
/// envelope, and paying them on both sides is what keeps the A/B a comparison
/// of schedules rather than of residencies — 73 816 B still admits the three
/// CTAs per SM that #84 counted at 73 792.
pub const SHARED_BYTES: usize = shared_plan(BLOCK_N, STAGES);

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
const _: () = assert!(THREADS == 128 && SHARED_BYTES == 98_392);

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
/// The census also says *which* resource set it, which the bisection could
/// not: 128 columns of tensor memory would admit 4 CTAs an SM and that
/// kernel's 73792 B shared plan admitted 3, so **shared memory capped it and
/// `tcgen05` did not**. It priced `alloc_cluster` against `alloc_block` at
/// equal columns too and found them identical, so none of this is a cluster
/// effect. And a query *can* describe a cluster launch —
/// `cuOccupancyMaxActiveClusters` takes the shape the block query has no
/// argument for. It was never called here; it is not that nothing could answer.
///
/// **All of the above is about the kernel this file shipped through #102, and
/// it is 2 now, for a different reason.** #87 widened the pair tile to
/// `[256, 256]`, which is **256 accumulator columns**, and `512 / 256` is 2
/// before shared memory is consulted at all. So this is the first kernel in
/// this repo whose residency is set by the *tensor memory* half of
/// `min(512 / columns, shared per SM / plan)` — every one before it was capped
/// by the shared half. The census counts 2 at the 98392 B plan, agreeing.
///
/// The step down from 3 was paid for and the sweep is what says at what price.
/// #98 priced a 3 → 2 step at **13.6–16.1%** on bytes no code touched; here the
/// bytes are a third of a wider pipeline and a wider tile, and the net is
/// **+11.6% at 8192³ and +21.6% at 16384³**. `examples/README.md` §7 separates
/// the two with `[256, 128] @ STAGES = 4` — the same 2 CTAs an SM at the *old*
/// tile — which lands at −7.7% and +2.1%.
const CTAS_PER_SM: u32 = 2;
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
const _: () = assert!(MAX_CLUSTERS == 148);

/// Tile-rows the item map walks before moving right — [`pipeline::grouped`]'s
/// width, and this kernel's answer to #89.
///
/// **A measurement, at one tile shape, and not a preference.** [`swizzle`]
/// sweeps `1, 2, 4, 8, 16, 32` and `examples/README.md` §7 keeps every row
/// including the ones that lost. `1` is the row-major map this kernel had
/// through #97; the value here is what won at the `[256, 128]` pair tile the
/// rest of this file is written around, and #87 moves that tile — so a tile
/// change is a reason to re-run the sweep and not a reason to trust this line.
///
/// What it changes is the *working set of a wave*, which is the quantity #90
/// measured without a counter. 222 clusters walked row-major sit on
/// `ceil(222 / tiles_n)` rows of tiles and span the whole of `N` if `N` is
/// narrow enough; walked in groups they sit on `GROUP` rows and
/// `ceil(222 / GROUP)` columns, which is a shape the aspect ratio no longer
/// controls. [`wave_reuse`] is that arithmetic, printed beside every row of the
/// sweep.
const GROUP: u32 = 8;

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

/// Which entry point a rung is compiled into.
///
/// A rung is a pair tile and a pipeline depth, and both are const parameters
/// of [`Tile`] — but `#[launch_contract]` takes a literal shared plan, so each
/// combination is its own `#[kernel]` and this is the host's name for it. The
/// four sweep entries are static-only; [`Scheduler::Stealing`] exists on the
/// shipped rung alone, because a scheduler comparison at a moving tile would
/// be two variables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Entry {
    /// `gemm_cg2` / `gemm_cg2_clc` — `[256, 256] @ STAGES = 3` since #87, the
    /// kernel this file ships and the only rung with a stealing twin.
    Shipped,
    N128S2,
    /// The kernel this file shipped through #102, kept as the control.
    N128S3,
    N128S4,
    N256S2,
    /// A rung the table computes and no kernel implements — see [`UNBUILT`].
    /// Launching it is an error rather than a missing arm, because the reason
    /// it is not built is a measurement and not an oversight.
    Unbuilt,
}

/// One point of #87's tile and depth sweep.
///
/// The two numbers that move are the pair tile's columns and the pipeline's
/// depth over K. Everything else about a rung — its shared plan, its tensor
/// memory, the residency those two admit, its arithmetic intensity and how
/// many output tiles a problem has — is arithmetic on them, which is the whole
/// reason this is a table and not six kernels written out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rung {
    /// Columns of `C` the *pair* computes, and this CTA's accumulator columns.
    pub block_n: usize,
    /// Pipeline depth over K.
    pub stages: usize,
    pub entry: Entry,
}

impl Rung {
    /// Dynamic shared memory this rung's launch declares.
    pub const fn shared(self) -> usize {
        shared_plan(self.block_n, self.stages)
    }

    /// **Arithmetic intensity**, `M·N/(M+N)` over the pair tile: flops per
    /// operand byte the tile reads, which is what a wider `N` buys.
    ///
    /// Doubling `N` buys 1.5× and not 2×, because `M·N/(M+N)` is not linear in
    /// `N`. `M` is 256 in every rung — the widest `M` tcgen05 has — so `N` is
    /// the only axis this sweep can move at all.
    pub fn intensity(self) -> f64 {
        let (m, n) = ((2 * BLOCK_M) as f64, self.block_n as f64);
        m * n / (m + n)
    }

    /// CTAs of this rung an SM holds: **#84's `min(512 / columns, shared per
    /// SM / plan)`**, predicted from the two per-CTA resources.
    ///
    /// Predicted, and the sweep prints `device-tests`' *counted* figure beside
    /// it rather than instead of it — #84 found the formula exact at fourteen
    /// rungs, and predicted and counted disagreeing would be a finding rather
    /// than a rounding error. `shared_per_sm` is queried from the driver and
    /// not written down: the number matters to the column and a constant that
    /// is only nearly right would move rungs across a step.
    ///
    /// The `512 / columns` term has never bound anything in this repo. At
    /// `block_n = 256` it does.
    pub fn ctas_per_sm(self, shared_per_sm: usize) -> u32 {
        let tmem = 512 / self.block_n;
        let shared = shared_per_sm / self.shared();
        tmem.min(shared) as u32
    }

    /// Clusters this rung's persistent grid launches at most — the same
    /// `SMS * CTAS_PER_SM / RANKS` the shipped kernel uses, at the residency
    /// this rung actually gets.
    ///
    /// Sizing the grid from the rung's own residency is what makes the sweep a
    /// comparison of tiles rather than of grids: a rung held at 222 clusters
    /// while the device admits 296 of it would be measured on a schedule
    /// chosen for a different kernel.
    pub fn max_clusters(self, shared_per_sm: usize) -> u32 {
        SMS * self.ctas_per_sm(shared_per_sm) / RANKS
    }

    /// What the table calls it.
    pub fn name(self) -> String {
        format!("[256,{}] s{}", self.block_n, self.stages)
    }
}

/// The kernel this file ships, as a rung: `[256, 256] @ STAGES = 3` (#87).
pub const SHIPPED: Rung = Rung {
    block_n: BLOCK_N,
    stages: STAGES,
    entry: Entry::Shipped,
};

/// #87's sweep, and the losers stay in it.
///
/// Five of the issue's six rungs are here. The sixth, `[256, 256] @ STAGES =
/// 4`, is [`UNBUILT`]: it is one CTA an SM, and #98 measured that step at a
/// further 25–44% under a step already worth 14–16%, so it is computed and not
/// launched. Booking a B200 on a rung whose answer two prior measurements
/// already give is how a sweep spends money to confirm itself.
pub const RUNGS: [Rung; 5] = [
    Rung {
        block_n: 128,
        stages: 2,
        entry: Entry::N128S2,
    },
    CONTROL,
    Rung {
        block_n: 128,
        stages: 4,
        entry: Entry::N128S4,
    },
    Rung {
        block_n: 256,
        stages: 2,
        entry: Entry::N256S2,
    },
    SHIPPED,
];

/// The kernel this file shipped through #102 — `[256, 128] @ STAGES = 3`, and
/// the control every #87 row is read against.
pub const CONTROL: Rung = Rung {
    block_n: 128,
    stages: 3,
    entry: Entry::N128S3,
};

/// The rung this sweep computes and does not build — see [`RUNGS`].
pub const UNBUILT: Rung = Rung {
    block_n: 256,
    stages: 4,
    entry: Entry::Unbuilt,
};

/// The shared plans the four sweep entry points declare, against the
/// arithmetic every host table reads.
///
/// `#[launch_contract]` takes a literal, so each rung's plan is written twice —
/// once there and once as [`shared_plan`]. `attach` asserts the arithmetic
/// against the *rings'* own byte counts at codegen; this asserts it against the
/// literals, which is the other half of the same join.
const _: () = {
    assert!(shared_plan(128, 2) == 49_224);
    assert!(shared_plan(128, 4) == 98_408);
    assert!(shared_plan(256, 2) == 65_608);
    assert!(shared_plan(256, 3) == 98_392);
    assert!(shared_plan(256, 4) == 131_176);
};

/// One output tile of `C`, as the persistent grid's work item.
///
/// Every field is what the item needs and does not depend on *which* item it
/// is: the pair's rings and barriers, the two operand maps, this CTA's half of
/// the accumulator, the shape of the tile grid, and the thread's own
/// coordinates. The item index is the only thing [`Job::work`] takes, and
/// [`pipeline::grouped`] is the whole of what it does with it — the same map
/// `blockIdx.x / 2` used to carry, now asked of a number that means one tile
/// per *cluster* and answered in groups of [`GROUP`] tile-rows rather than
/// row-major.
#[derive(Clone, Copy)]
struct Tile<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> {
    a_ring: ARing<STAGES>,
    b_ring: BRing<HALF_N, STAGES>,
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
    accumulator: Accumulator<BLOCK_N>,
    /// `C` with `ldc` in it — built once, since a persistent CTA writes bands
    /// of the same output through every item it runs.
    c: GlobalRows,
    /// The tile grid, both axes. `tiles_m` is here only so the item map can
    /// short the last group of tile-rows when [`GROUP`] does not divide it —
    /// row-major never needed it, and a map that aliases the tail is a wrong
    /// `C` rather than a slow one.
    tiles_m: u32,
    tiles_n: u32,
    /// [`pipeline::grouped`]'s width. A launch parameter and not a constant, so
    /// [`swizzle`] can sweep it against one staged pair of operands.
    group: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

/// The three phases of an output tile, named so that the two epilogue shapes
/// below can order them differently without either one owning a second copy of
/// the K walk.
///
/// The split is [`Lcsf`]'s reason for existing and nothing else moves because
/// of it: [`Job::work`] on [`Tile`] calls all three back to back and is the
/// kernel that shipped through #87, instruction for instruction.
impl<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> Tile<BLOCK_N, HALF_N, STAGES> {
    /// K blocks the producer can issue before any of them has to be recycled.
    ///
    /// [`SemaphoreRing::wait_recycled`] is a no-op below `STAGES`, so this
    /// prefix of the walk is the part that can be issued and then *left* — the
    /// producer does not block in it, and the loads it starts are in flight
    /// across whatever runs next. That is the whole of what [`Lcsf`] overlaps
    /// its store with, and it is why the split is at `STAGES` and not
    /// somewhere tunable.
    const FILL: u32 = STAGES as u32;

    /// Issue this rank's half of K blocks `from..to`, charging the leader's
    /// stage barrier for the whole pair.
    ///
    /// # Safety
    ///
    /// One thread of the CTA, with `from..to` inside `0..k_blocks` and every
    /// block issued exactly once across the calls that cover the walk.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32, from: u32, to: u32) {
        unsafe {
            // Both CTAs load their own halves, and all four tiles complete on
            // the leader's copy of the stage barrier — one barrier is what the
            // MMA issuer needs to know the whole stage is present. Only the
            // leader charges, and it charges the whole stage: `expect_tx` is
            // `.shared::cta`, so a peer could not charge this barrier even
            // holding its address.
            //
            // Every rank derives the same half-stage charge from the loads it
            // just issued, because a cluster stage is symmetric; the leader
            // scales its own by `RANKS` to cover the peer's, and the peer drops
            // it. Nothing orders the charge and the loads, and nothing has to —
            // the transaction count is a signed accumulator and only the totals
            // must agree, which is what lets the charge follow the calls it is
            // derived from.
            let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = from;
            while k < to {
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

    /// Chain the whole K walk into the pair's accumulator and publish it.
    ///
    /// # Safety
    ///
    /// One thread of the leader rank, with the accumulator's previous contents
    /// already read — every chunk of every stage chains into the one
    /// accumulator, so only the very first instruction of the very first stage
    /// starts it fresh, and "first" is per *item*.
    #[inline(always)]
    unsafe fn multiply(&self) {
        unsafe {
            // `MmaShape` is a re-export of `Tcgen05MmaShape` and `mma_walk_cg2`
            // takes the shape as a value, so widening the pair tile needs
            // nothing from `src/mma.rs`. In a `const` block so a rung whose
            // columns name no shape is a codegen error rather than a `panic!`
            // lowered into device code.
            let shape = const { pair_shape(BLOCK_N) };
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                mma_walk_cg2::<Bf16, CHUNKS>(
                    self.accumulator.raw(),
                    self.a_ring.tile(k).k_walk(),
                    self.b_ring.tile(k).k_walk(),
                    shape,
                    k > 0,
                );
                // The MMA releases its own operands, in both CTAs: a thread
                // arriving here would only prove the instruction was *issued*.
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// The fp32 epilogue, straight out of registers (#11) — this warp's band of
    /// the accumulator, LDTM'd and stored to the tile `item` names.
    ///
    /// `ldc` is the destination's leading dimension and `C` is wider than this
    /// tile's columns, so the cursor carries the stride and each band lands at
    /// its own `(row, column)` origin — no shared staging tile, no descriptor,
    /// and no rounding to bf16 on the way out.
    ///
    /// A band at a time rather than the whole accumulator at once, and
    /// [`DRAIN_N`] is why: 256 columns in one `RegTile` is 256 fp32 a thread.
    /// At `BLOCK_N = 128` this is the single band (#22) it has always been, and
    /// the loop folds away.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA, with the accumulator complete and fenced, and
    /// nothing that will overwrite it in flight.
    #[inline(always)]
    unsafe fn drain(&self, item: u32) {
        unsafe {
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
            let row_base =
                2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * self.warp_id;
            let column_base = BLOCK_N as u32 * tile_n;
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                // This warp's 32 TMEM lanes by `DRAIN_N` columns of the
                // accumulator, composed out of the `[16, 16]` blocks LDTM
                // delivers.
                let band: Band = self.accumulator.tile(32 * self.warp_id, column);
                store_rows(self.c, row_base, column_base + column, self.lane, band);
                column += DRAIN_N as u32;
            }
        }
    }
}

impl<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> Job
    for Tile<BLOCK_N, HALF_N, STAGES>
{
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
            // The item map, and the only line in this kernel #89 changes. It is
            // a bijection under both schedulers alike — `run` hands out a
            // strided share and `run_stealing` hands out whatever the hardware
            // cancelled, and a bijection of either is still every tile once.
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);

            if self.warp_id == 0 && self.lane == 0 {
                self.produce(tile_m, tile_n, 0, self.k_blocks);
            }
            if self.rank == LEADER && self.warp_id == 1 && self.lane == 0 {
                self.multiply();
            }
            self.done.wait(0);
            thread::sync_threads();
            self.drain(item);
        }
    }
}

/// [`Epilogue::HotStore`]'s job: [`Tile`] with every store aimed at the
/// cluster's own first tile, so the epilogue's bytes stay in L2.
///
/// **It computes a wrong `C` and is never checked.** See [`Epilogue::HotStore`]
/// for what it is for and why an exact version of it does not exist.
#[derive(Clone, Copy)]
struct HotStore<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> {
    tile: Tile<BLOCK_N, HALF_N, STAGES>,
    /// The item this cluster was launched for, and the only tile of `C` it ever
    /// writes. Read once outside the loop: [`pipeline::run`]'s first item *is*
    /// `cluster_idx`, so this is a tile the cluster would have written anyway
    /// and the probe stays inside the buffer without a bounds check.
    home: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> Job
    for HotStore<BLOCK_N, HALF_N, STAGES>
{
    const RANKS: u32 = crate::gemm::RANKS;

    /// # Safety
    ///
    /// As [`Tile`]'s.
    #[inline(always)]
    unsafe fn init(&self, item: u32) {
        unsafe { self.tile.init(item) }
    }

    /// # Safety
    ///
    /// As [`Tile`]'s.
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe { self.tile.inval() }
    }

    /// # Safety
    ///
    /// As [`Tile`]'s. The store is to a tile of `C` this cluster owns, so it is
    /// in bounds; it is the *wrong* tile, which is the point and not a hazard.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply();
            }
            tile.done.wait(0);
            thread::sync_threads();
            // The one line that differs from `Tile::work`, and the whole probe.
            tile.drain(self.home);
        }
    }
}

/// The same output tile with its store phase moved into the *next* item —
/// ThunderKittens' `prototype::lcsf`, and #15.
///
/// # The store stage is a phase of the item, not a stage of the scaffold
///
/// #15 files `lcsf` as a shape [`pipeline::run`] would have to grow, and #94
/// priced it as a TMEM → shared → TMA store epilogue needing an fp32
/// `TensorMapElement`, an fp32 `SharedTile` swizzle and a shared staging buffer
/// nobody had the bytes for. **None of that is what the overlap needs here.**
///
/// The thing `lcf` forbids is the epilogue of item `i` running while item
/// `i + 1`'s first loads are in flight. What actually stands between them is
/// not the shape of the scaffold — it is that [`Tile::drain`] is the last thing
/// `work` does. The pair's accumulator lives in tensor memory allocated once
/// *outside* the item loop, and the item boundary re-arms mbarriers and touches
/// no tensor memory at all, so an undrained accumulator survives the boundary
/// intact. Deferring the drain by one item is therefore a change of phase order
/// inside `work`, and [`pipeline::run`] is unmodified: the pending item is job
/// state, and `lcf`'s scaffold already admits an `lcsf` job.
///
/// So the whole epilogue — the LDTM *and* the scattered fp32 stores — runs
/// after [`Tile::FILL`] stages of the next item's loads have been issued and
/// while they are in flight. It costs **no shared memory, no deferred
/// registers, no fp32 TMA path and no occupancy step**, which is what makes it
/// measurable without first building the 250–400 lines #94 scoped.
///
/// # It does need a synchronisation the item boundary does not supply
///
/// The MMA is `cta_group::2` and is issued by the leader alone, writing *both*
/// ranks' halves of the accumulator. Under the fused epilogue the pair is
/// separated by [`pipeline::run`]'s cluster boundary, so a peer is provably
/// done reading before a leader can issue. Here the peer's drain of item `i`
/// and the leader's MMA for item `i + 1` are inside the same `work`, and a
/// `bar.sync` orders only one CTA — the leader would overwrite tensor memory
/// the peer is still reading, silently and as a wrong `C`. The rendezvous below
/// is therefore `cluster_sync`, which is the finding: `lcsf` does not want an
/// extra barrier at the boundary, it wants the boundary's *scope* moved inside
/// the item.
#[derive(Clone, Copy)]
struct Lcsf<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> {
    tile: Tile<BLOCK_N, HALF_N, STAGES>,
    /// The item whose accumulator is still sitting in tensor memory, or
    /// [`Self::NONE`] before the first one and after the last.
    pending: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> Lcsf<BLOCK_N, HALF_N, STAGES> {
    /// No accumulator owed. `u32::MAX` is not a reachable item: the tile grid
    /// is `tiles_m * tiles_n` and both are `u32`, so a real item is strictly
    /// less than their product and cannot be this.
    const NONE: u32 = u32::MAX;

    #[inline(always)]
    fn new(tile: Tile<BLOCK_N, HALF_N, STAGES>) -> Self {
        Self {
            tile,
            pending: Self::NONE,
        }
    }

    /// Store the last item's accumulator, which no later item is coming to
    /// overlap. Called once, after the loop and before the pair gives its
    /// tensor memory back.
    ///
    /// A cluster that ran no items at all owes nothing, which is the
    /// [`Scheduler::Static`] case where `MAX_CLUSTERS` exceeds the tile count.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA, after [`pipeline::run`] has returned and before
    /// `release`.
    #[inline(always)]
    unsafe fn finish(&self) {
        unsafe {
            if self.pending != Self::NONE {
                self.tile.drain(self.pending);
            }
        }
    }
}

impl<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize> Job
    for Lcsf<BLOCK_N, HALF_N, STAGES>
{
    const RANKS: u32 = crate::gemm::RANKS;

    /// # Safety
    ///
    /// As [`Tile`]'s.
    #[inline(always)]
    unsafe fn init(&self, item: u32) {
        unsafe { self.tile.init(item) }
    }

    /// # Safety
    ///
    /// As [`Tile`]'s.
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe { self.tile.inval() }
    }

    /// # Safety
    ///
    /// As [`Tile`]'s, plus: the accumulator must still hold [`Self::pending`]'s
    /// result, which is [`pipeline::run`] calling this once per item in order.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            let fill = Tile::<BLOCK_N, HALF_N, STAGES>::FILL.min(tile.k_blocks);

            // Fill the pipe first, so the loads are in flight across the store
            // below. The producer does not block in this prefix — a stage
            // barrier below `STAGES` has nothing to recycle — so it reaches the
            // drain rather than sitting in `free.wait_recycled`.
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce(tile_m, tile_n, 0, fill);
            }
            // Reconvergence, and it is load-bearing rather than tidy: `LDTM` is
            // warp-collective and the producer branch above leaves warp 0's
            // lane 0 somewhere its other 31 lanes are not. Under the fused
            // epilogue the `sync_threads` after `done.wait` did this job; here
            // there is no wait between the branch and the drain, so the barrier
            // has to be written down. It is the *same* barrier count as `lcf` —
            // what `lcsf` adds is the cluster-scope one below, not this.
            thread::sync_threads();

            // The previous item's epilogue, overlapped with those loads. This
            // is the whole of `lcsf`.
            if self.pending != Self::NONE {
                tile.drain(self.pending);
            }

            // Cluster-scope, and the type doc says why: the leader's MMA below
            // writes the peer's half of the accumulator the peer was just
            // reading. The fence is the same pairing [`pipeline::run`] takes
            // around its own boundary — tcgen05 work (here the drain's LDTM)
            // retired before a thread sync that publishes it.
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();

            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce(tile_m, tile_n, fill, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply();
            }
            tile.done.wait(0);
            self.pending = item;
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
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<f32>,
    ) -> (Tile<BLOCK_N, HALF_N, STAGES>, ClcQueue) {
        // Everything a rung has to be true about, fired at codegen — which is
        // the only place the ring byte counts are known, and the reason `cargo
        // check` cannot stand in for `modal_app.py::build`.
        //
        // `HALF_N` is a second parameter rather than `BLOCK_N / 2` because a
        // const parameter cannot be arithmetic on another one without
        // `generic_const_exprs`; the assert is what keeps the two in step.
        // `shared_plan` is host-side arithmetic and the rings' `BYTES` are the
        // library's own count of the same bytes, so the third assert is what
        // says a `#[launch_contract]` literal describes the plan `attach` lays
        // out. And the queue's offset must be aligned for the `.b128`
        // `try_cancel` writes.
        const {
            assert!(BLOCK_N == 2 * HALF_N);
            assert!(BLOCK_N % DRAIN_N == 0);
            assert!(
                shared_plan(BLOCK_N, STAGES)
                    == ARing::<STAGES>::BYTES
                        + BRing::<HALF_N, STAGES>::BYTES
                        + scratch_bytes(STAGES)
            );
            assert!(
                (ARing::<STAGES>::BYTES + BRing::<HALF_N, STAGES>::BYTES + 2 * STAGES * 8 + 8 + 8)
                    % ClcQueue::ALIGNMENT
                    == 0
            );
        };
        unsafe {
            let rings = ARing::<STAGES>::BYTES + BRing::<HALF_N, STAGES>::BYTES;
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let scratch = smem.add(rings);
            let tmem_slot = scratch.add(2 * STAGES * 8 + 8) as *mut u32;

            let tile = Tile {
                a_ring: ARing::<STAGES>::attach(smem),
                b_ring: BRing::<HALF_N, STAGES>::attach(smem.add(ARing::<STAGES>::BYTES)),
                load: SemaphoreRing::<STAGES>::attach(scratch as *mut Barrier),
                free: SemaphoreRing::<STAGES>::attach((scratch as *mut Barrier).add(STAGES)),
                done: Semaphore::attach((scratch as *mut Barrier).add(2 * STAGES)),
                a_map,
                b_map,
                accumulator: Accumulator::<BLOCK_N>::from_raw(alloc_cluster(
                    tmem_slot,
                    BLOCK_N as u32,
                )),
                c: GlobalRows::from_slice(c, ldc as usize),
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
    unsafe fn release<const BLOCK_N: usize, const HALF_N: usize, const STAGES: usize>(
        tile: &Tile<BLOCK_N, HALF_N, STAGES>,
    ) {
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
    /// tiles are gone, and `cluster::block_rank()` says which half of the
    /// pair this CTA owns. `a_map` describes `A` as `[rows, K]` bf16, `b_map`
    /// describes `B` as `[columns, K]`. Both come from a rank-2
    /// [`kittens::global::GlobalLayout`] paired with the tile it feeds, so
    /// their `[R, 64]` boxes are `ATile`'s and `BTile`'s own constants and not
    /// numbers [`check`] wrote down.
    ///
    /// **The separate `tiles` argument is gone**, and #89 is why. A grouped
    /// item map has to know how many tile-*rows* there are, to short the last
    /// group when the width does not divide them — so `tiles_m` is now a
    /// parameter, `tiles_m * tiles_n` is the item count, and the launcher no
    /// longer passes a total it had already told the kernel twice over.
    ///
    /// Everything outside the item loop is `attach` and `release`; what is
    /// left here is the schedule.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus: the grid must be a whole number of clusters and
    /// `tiles_m * tiles_n` the item count they are to cover — see [`grid`],
    /// which is what the launcher below sizes both from.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<BLOCK_N, HALF_N, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// The same GEMM with its store phase deferred one item — #15's `lcsf`.
    ///
    /// Identical to [`gemm_cg2`] in grid, shared plan, tensor memory, operand
    /// maps and item map; the only difference is that [`Lcsf`] drains the
    /// previous item's accumulator after the next item's first stages are in
    /// flight rather than before them. The shared plan is *unchanged* — the
    /// overlap needs no staging buffer — so this is a comparison of epilogue
    /// placement at a fixed residency, which is what makes the A/B readable.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s exactly.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_lcsf(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (tile, _) = attach::<BLOCK_N, HALF_N, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Lcsf::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            // The last item has no successor to overlap its store with, so it
            // pays an un-hidden epilogue — one per cluster over the whole
            // launch, against one per *item* under the fused shape.
            job.finish();
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`Epilogue::HotStore`]'s probe, which
    /// splits the epilogue's cost into the part that is issue and the part that
    /// is bandwidth. Never checked, never shipped, and its number is an upper
    /// bound rather than a throughput.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s. It writes fewer tiles of `C` than that one, not more.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_hot(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (tile, _) = attach::<BLOCK_N, HALF_N, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = HotStore {
                tile,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// The same GEMM on the hardware's schedule: a grid of one cluster per
    /// output tile, of which the device launches as many as it holds, and the
    /// rest are cancelled out from under the scheduler by clusters that have
    /// finished — [`pipeline::run_stealing`].
    ///
    /// **Neither [`SMS`] nor [`CTAS_PER_SM`] appears anywhere on this path, and
    /// that absence is the feature.** The static entry point above sizes its
    /// grid from a constant that had to be measured on one device (#84); here
    /// the grid *is* the tile count and the residency is the scheduler's
    /// business.
    ///
    /// Through #97 the visible marker of that was this entry point taking no
    /// `tiles`. It is not any more — #89 made `tiles_m` a parameter of the item
    /// map, so *both* entry points now take the tile grid and derive the count.
    /// The difference was never the argument list; it is [`grid_for`], where
    /// one branch reads a measured constant and the other reads the problem.
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
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_clc(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, queue) = attach::<BLOCK_N, HALF_N, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            pipeline::run_stealing(&mut tile, queue);
            release(&tile);
        }
    }

    // #87's sweep is the four entry points below: the static schedule on the
    // same `Tile`, at one `(BLOCK_N, STAGES)` each. The only reason they are
    // written out rather than swept from a parameter is that
    // `#[launch_contract]`'s `dynamic_shared` takes a literal — so a rung's
    // shared plan is spelled twice and `attach`'s codegen assert is what holds
    // the two spellings together.
    //
    // `[256, 256] @ STAGES = 4` is deliberately **not** here. It is 131 176 B,
    // which is one CTA an SM, and #98 measured that step at a further 25–44%
    // under a step already worth 14–16%. `RUNGS` computes it and no B200 time
    // is booked on it.

    /// `[256, 128]` two stages deep — 49 224 B, and the only rung in the sweep
    /// that steps residency *up*: four CTAs an SM where the shipped kernel
    /// holds three. Tensor memory admits four at 128 columns too, so nothing
    /// is left binding but the pipeline being a stage shallower.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 49_224,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x128_s2(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) =
                attach::<128, 64, 2>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 128]` four stages deep — 98 408 B, two CTAs an SM.
    ///
    /// **The control that makes the rest of the sweep readable.** It is one
    /// occupancy step down at *unchanged* pair tile, unchanged arithmetic
    /// intensity and unchanged tile count, so whatever it costs is the step
    /// and the fourth stage together and nothing else. #98 priced the same step
    /// on bytes no code touched; here the bytes are a live pipeline stage, so
    /// the two together say what a stage is worth against what it costs.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_408,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x128_s4(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) =
                attach::<128, 64, 4>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 256]` two stages deep — 65 608 B, **the rung where tensor memory
    /// binds first**.
    ///
    /// Shared memory admits three CTAs here and 256 accumulator columns admit
    /// two, so this is the first kernel in this repo whose residency is set by
    /// the `512 / columns` half of `min(512 / columns, shared per SM / plan)`.
    /// `src/tmem.rs` makes a sharper prediction than the count alone: the third
    /// CTA is *admitted* and parks inside `tcgen05.alloc`, so the census should
    /// read three resident against two holding — the first rung in this repo
    /// where those two columns differ.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 65_608,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x256_s2(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) =
                attach::<256, 128, 2>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 128]` three stages deep — 73 816 B, three CTAs an SM. **The
    /// kernel this file shipped through #102, kept as the sweep's control.**
    ///
    /// It is no longer [`gemm_cg2`]: #87 moved the pair tile to `[256, 256]`,
    /// and this entry point is what every table in `examples/README.md` before
    /// that was measured on. Keeping it launchable is what makes the tile
    /// comparison a measurement against the previous kernel rather than
    /// against a number remembered from another session — and #98's own method
    /// is the argument, since it found a 2.9% drift between containers large
    /// enough to change a verdict.
    ///
    /// It is also the rung that pairs with [`gemm_256x128_s4`] to price the
    /// occupancy step at unchanged tile, and with [`gemm_cg2`] to price the
    /// tile at unchanged everything else.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 73_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x128_s3(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            let (mut tile, _) =
                attach::<128, 64, 3>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles_m * tiles_n);
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

/// Tiles of `C` a `[m, n]` output has along each axis — the grid
/// [`pipeline::grouped`] walks, and the pair of numbers the kernel needs
/// because a grouped walk has to know where the tile-rows run out.
fn tile_grid(m: usize, n: usize, block_n: usize) -> (u32, u32) {
    ((m / (2 * BLOCK_M)) as u32, (n / block_n) as u32)
}

/// Tiles of `C` a `[m, n]` output has, which is the item count the persistent
/// grid walks: one per `2·BLOCK_M` by `BLOCK_N` tile.
fn tiles(m: usize, n: usize, block_n: usize) -> u32 {
    let (rows, columns) = tile_grid(m, n, block_n);
    rows * columns
}

/// Operand bytes the clusters resident at one time collectively touch at a
/// single K block, and the reuse that implies — **the quantity #89 is about,
/// and the closest this harness can get to an L2 hit rate.**
///
/// It is arithmetic on the item map and not a counter, and that limitation is
/// not a choice: `modal_app.py::profile` is checked in *not working*, because
/// the container's injected driver set carries no `libnvidia-pcc.so` and Nsight
/// Compute cannot start without it. So there is no measured hit rate on this
/// harness, for this change or for #90's, and the substitute is stated rather
/// than quietly dropped.
///
/// What it computes: [`MAX_CLUSTERS`] consecutive items are resident together
/// and march through K in step, so at one K block they read the tile-rows of
/// `A` and the tile-columns of `B` those items span. It walks the map rather
/// than closing a form over it, because a grouped walk straddles group
/// boundaries and the closed form would be four cases and a place to be wrong.
///
/// Returns the tile-rows and tile-columns a wave spans, the distinct bytes that
/// comes to, and the reuse — bytes requested over bytes distinct, so 1.0 would
/// be a wave that shares nothing.
fn wave_reuse(
    m: usize,
    n: usize,
    group: u32,
    block_n: usize,
    clusters: u32,
) -> (usize, usize, f64, f64) {
    let (rows, columns) = tile_grid(m, n, block_n);
    let wave = clusters.min(rows * columns);
    let (mut spans_row, mut spans_column) =
        (vec![false; rows as usize], vec![false; columns as usize]);
    for item in 0..wave {
        let (row, column) = pipeline::grouped(item, rows, columns, group);
        spans_row[row as usize] = true;
        spans_column[column as usize] = true;
    }
    let spanned = |flags: &[bool]| flags.iter().filter(|&&hit| hit).count();
    let (walked_rows, walked_columns) = (spanned(&spans_row), spanned(&spans_column));
    let depth = (BLOCK_K * 2) as f64;
    let distinct = (walked_rows * 2 * BLOCK_M + walked_columns * block_n) as f64 * depth;
    let requested = wave as f64 * (2 * BLOCK_M + block_n) as f64 * depth;
    (walked_rows, walked_columns, distinct, requested / distinct)
}

/// How a launch takes its work: where the next item comes from, and which tile
/// an item names. The two are orthogonal by construction — [`pipeline::grouped`]
/// is a bijection and both item sources hand out permutations — which is what
/// makes sweeping one against the other a two-column table rather than a
/// combinatorial argument.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub scheduler: Scheduler,
    /// [`pipeline::grouped`]'s width in tile-rows. `1` is row-major.
    pub group: u32,
    /// The pair tile and pipeline depth to launch — #87's variable, and the
    /// one #89's `GROUP` is only measured *at*.
    pub rung: Rung,
    /// Where the store phase sits — #15's variable.
    pub epilogue: Epilogue,
}

impl Plan {
    /// The kernel as it ships: whichever schedule `scheduler` names, walked at
    /// the measured [`GROUP`], on the [`SHIPPED`] rung, with the epilogue fused
    /// into the item.
    fn new(scheduler: Scheduler) -> Self {
        Plan {
            scheduler,
            group: GROUP,
            rung: SHIPPED,
            epilogue: Epilogue::Fused,
        }
    }

    /// The same plan with its store phase somewhere else — #15's variable.
    fn with(self, epilogue: Epilogue) -> Self {
        Plan { epilogue, ..self }
    }
}

/// Where an item's store phase runs — #15, and the axis [`Lcsf`] adds.
///
/// It is not a scheduler and not a rung: the grid, the shared plan, the tensor
/// memory, the item map and the tile are identical across the two, and the only
/// thing that moves is whether [`Tile::drain`] runs at the end of its own item
/// or at the start of the next one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    /// `lcf`: the store folds into the item and cannot overlap the next item's
    /// loads. The control, and what this kernel shipped through #87.
    Fused,
    /// `lcsf`: the store is deferred one item and runs while the next item's
    /// first [`Tile::FILL`] stages are in flight.
    Deferred,
    /// **Not a GEMM. This computes a wrong `C` on purpose and is never checked.**
    ///
    /// The fused epilogue with every item's store aimed at the *cluster's own
    /// first tile* instead of the item's. Identical in every instruction — same
    /// LDTM, same count of `st.global.v2.f32`, same addresses within a tile —
    /// and the only thing that moves is that 148 clusters rewrite 38 MB of `C`
    /// over and over instead of streaming 268 MB or 1 GB of it, so the writes
    /// stay in L2 and never press on HBM.
    ///
    /// It exists to split the epilogue's cost in two, which no exact
    /// intervention here can. If this is much faster than [`Epilogue::Fused`]
    /// the epilogue is **bandwidth-bound**, and staging through shared memory
    /// so a TMA engine does the global write — #15's original route — has that
    /// much to win. If it is not, the epilogue is **issue-bound**, the threads
    /// are paying for the LDTM and the stores themselves rather than for where
    /// they land, and no amount of decoupling the write recovers it.
    ///
    /// The number it produces is an upper bound and not a throughput. It is
    /// excluded from [`check`] deliberately: a probe that computed the right
    /// answer would not be measuring what this one is for.
    HotStore,
}

impl Epilogue {
    /// What the benchmark prints, and the only place these names are spelled.
    pub fn name(self) -> &'static str {
        match self {
            Epilogue::Fused => "lcf",
            Epilogue::Deferred => "lcsf",
            Epilogue::HotStore => "hot",
        }
    }

    /// Whether a launch on this epilogue computes the GEMM. Only
    /// [`Epilogue::HotStore`] does not, and it says so everywhere it appears.
    fn exact(self) -> bool {
        self != Epilogue::HotStore
    }
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
    grid_for(Scheduler::Static, m, n, SHIPPED, MAX_CLUSTERS)
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
pub fn grid_for(scheduler: Scheduler, m: usize, n: usize, rung: Rung, cap: u32) -> u32 {
    let clusters = match scheduler {
        Scheduler::Static => tiles(m, n, rung.block_n).min(cap),
        Scheduler::Stealing => tiles(m, n, rung.block_n),
    };
    RANKS * clusters
}

/// Waves the static grid takes at `m`x`n`, and how full the last one is.
///
/// This is the entire quantity CLC has to win back, and it is worth computing
/// rather than measuring against: a cluster's item takes the same time whoever
/// scheduled it, so the static stride's only loss is the clusters idle through
/// the ragged last wave.
///
/// #87 moved every number here, and in the favourable direction: at the
/// `[256, 128]` tile's 222 clusters the efficiency ran 57.7% at 2048³, 76.9% at
/// 4096³, 92.3% at 8192³ and 99.7% at 16384³; at `[256, 256]`'s 148 it is
/// **98.8% at both 8192³ and 16384³**, because halving the tiles and halving
/// the clusters leaves a wave that divides the work much more evenly. That is a
/// confound in #87's own sweep rather than a win to claim — #97 measured the
/// ragged wave as worth ~1% of the *time* where it is 8% of the grid — and §7
/// says so beside the table.
pub fn wave_efficiency(m: usize, n: usize) -> (u32, f64) {
    wave_efficiency_of(m, n, SHIPPED, MAX_CLUSTERS)
}

/// [`wave_efficiency`] at a rung's own tile and grid cap — the two things #87
/// moves, and both of them move this.
fn wave_efficiency_of(m: usize, n: usize, rung: Rung, cap: u32) -> (u32, f64) {
    let tiles = tiles(m, n, rung.block_n);
    let waves = tiles.div_ceil(cap);
    (waves, tiles as f64 / (waves * cap) as f64)
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
    plan: Plan,
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
    let rung = plan.rung;
    if m % (2 * BLOCK_M) != 0 || n % rung.block_n != 0 || k % BLOCK_K != 0 {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{}x{BLOCK_K} tiling",
            2 * BLOCK_M,
            rung.block_n
        )
        .into());
    }
    // `grouped` divides by `group * columns` and the device checks nothing, so
    // a zero here is a launch that faults rather than a launch that is wrong.
    if plan.group == 0 {
        return Err("a traversal group width of 0 has no tiles in it".into());
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
    // The `B` tile is the one operand type a rung moves, and its box comes off
    // the type — so the descriptor a `[256, 256]` rung loads through is the
    // library's arithmetic on `BTile<128>` and not a number written here.
    let b_map = match rung.block_n {
        128 => b_layout.tensor_map::<BTile<64>>(&stream)?,
        256 => b_layout.tensor_map::<BTile<128>>(&stream)?,
        columns => return Err(format!("no rung has {columns} pair columns").into()),
    };

    let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
    let cap = rung.max_clusters(shared_per_sm(context)?);
    let blocks = grid_for(plan.scheduler, m, n, rung, cap);
    let (tiles_m, tiles_n) = tile_grid(m, n, rung.block_n);
    let k_blocks = (k / BLOCK_K) as u32;
    let config = LaunchConfig1D::new(blocks, THREADS, rung.shared() as u32);

    // One prepared launch per call, boxed because the handle's type is the
    // kernel's. Preparing is a driver call opting the entry point into its
    // >48 KiB plan; it happens once, outside the clock, and what the timed
    // closure does is a launch.
    //
    // SAFETY (every arm): both maps describe live buffers covering the walk
    // the grid takes, and `c` holds `n` columns for every row of it. The
    // stealing entry additionally takes a grid of exactly one cluster per
    // tile, which is what `grid_for` gives it.
    //
    // The closure borrows the stream and the module and *owns* the prepared
    // handle, which is why the pointers are taken out of the descriptors here:
    // the descriptors are locals of this function and outlive the closure, and
    // a raw pointer is `Copy` where they are not.
    let (stream_ref, module_ref) = (&stream, &module);
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    macro_rules! launcher {
        ($prepare:ident, $launch:ident) => {{
            let prepared = module_ref.$prepare(config)?;
            let launch = move |c: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn Error>> {
                unsafe {
                    module_ref.$launch(
                        stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, plan.group,
                        k_blocks, n as u32, c,
                    )?
                };
                Ok(())
            };
            Box::new(launch) as Box<dyn Fn(&mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>>>
        }};
    }
    let launch_once = match (rung.entry, plan.scheduler, plan.epilogue) {
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused) => {
            launcher!(prepare_gemm_cg2, gemm_cg2)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Deferred) => {
            launcher!(prepare_gemm_cg2_lcsf, gemm_cg2_lcsf)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::HotStore) => {
            launcher!(prepare_gemm_cg2_hot, gemm_cg2_hot)
        }
        (Entry::Shipped, Scheduler::Stealing, Epilogue::Fused) => {
            launcher!(prepare_gemm_cg2_clc, gemm_cg2_clc)
        }
        (Entry::N128S2, Scheduler::Static, Epilogue::Fused) => {
            launcher!(prepare_gemm_256x128_s2, gemm_256x128_s2)
        }
        (Entry::N128S4, Scheduler::Static, Epilogue::Fused) => {
            launcher!(prepare_gemm_256x128_s4, gemm_256x128_s4)
        }
        (Entry::N128S3, Scheduler::Static, Epilogue::Fused) => {
            launcher!(prepare_gemm_256x128_s3, gemm_256x128_s3)
        }
        (Entry::N256S2, Scheduler::Static, Epilogue::Fused) => {
            launcher!(prepare_gemm_256x256_s2, gemm_256x256_s2)
        }
        (Entry::Unbuilt, _, _) => {
            return Err("[256,256] s4 is one CTA an SM and is computed, not built".into());
        }
        // Only the shipped rung has a stealing twin, and deliberately: a
        // scheduler comparison at a moving tile would be two variables. The
        // deferred epilogue is the same rule for the same reason — it is the
        // one variable #15 moves, so it moves against the shipped kernel on the
        // static schedule and nothing else.
        (entry, scheduler, epilogue) => {
            return Err(format!(
                "{entry:?} has no {} entry point on the {} schedule",
                epilogue.name(),
                scheduler.name()
            )
            .into());
        }
    };
    launch_once(&mut c)?;
    // Rule 1 of `crate::bench`, and the one exception to it is stated in the
    // label rather than hidden: `HotStore` writes the wrong tile of `C` on
    // purpose, so checking it would fail by construction. Every other plan —
    // every schedule, every rung, every traversal width, and #15's deferred
    // epilogue — goes through the element-by-element `==` before a clock can
    // reach it.
    let label = if plan.epilogue.exact() {
        check_c(&c.to_host_vec(&stream)?, m, n, k)?;
        format!("{m}x{n}x{k} exact")
    } else {
        format!(
            "{m}x{n}x{k} UNCHECKED ({} is not a GEMM)",
            plan.epilogue.name()
        )
    };

    let after = then(&stream, &mut || launch_once(&mut c))?;
    Ok((label, after))
}

/// Bytes of shared memory an SM divides between its resident CTAs — the
/// denominator of #84's `shared per SM / plan`, **queried rather than written
/// down**.
///
/// #87 asks for it to be derived, and the reason is not tidiness: it is 233 472
/// on a B200, and a rung's residency is a floor division by it, so a figure
/// that is only nearly right moves a rung across an occupancy step and changes
/// the answer rather than the third digit.
fn shared_per_sm(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<usize, Box<dyn Error>> {
    let mut bytes = 0i32;
    // SAFETY: the attribute is an `int` and `context` names a live device.
    let status = unsafe {
        cuda_core::sys::cuDeviceGetAttribute(
            &mut bytes,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            context.cu_device(),
        )
    };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) = {status}"
        )
        .into());
    }
    Ok(bytes as usize)
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

/// Traversal widths the correctness run walks every size at, chosen to break
/// [`pipeline::grouped`] rather than to be fast.
///
/// `1` is row-major, the map this kernel had through #97, so a regression in
/// the rest of the file still fails against the traversal it always used. The
/// other two are **not powers of two, and that is the point**: every `M` this
/// project runs is, so `tiles_m` is too, so a swept width of 8 or 16 always
/// divides it and the short last group — the one branch in the map, and the one
/// that turns a wrong width into a tile computed twice — would never execute.
/// At [`ITEMS_M`]'s 16 tile-rows, `3` leaves a final group of one row and `6`
/// leaves one of four; at [`M`]'s two tile-rows both are taller than the grid
/// and take the saturating path instead. Three widths against two shapes covers
/// full groups, short groups and over-tall widths, under both schedulers.
const CHECK_GROUPS: [u32; 3] = [1, 3, 6];

/// The correctness run: two sizes, three traversals, two schedulers, checked,
/// nothing timed.
///
/// The second size is [`ITEMS_M`]`x`[`ITEMS_N`], and it is here because the
/// first cannot fail the way the persistent grid can. Both report the items
/// their clusters walked, so a size that quietly stopped exercising the loop —
/// because [`MAX_CLUSTERS`] moved, or because the tiling did — says so in the
/// pass line instead of in nobody's memory.
///
/// The traversal is under the same gate for the same reason the scheduler is:
/// a wrong item map is a tile computed twice and one computed by nobody, both
/// silent on the device, and the element-by-element `==` against the CPU
/// reference is the only thing that sees either.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    let per_sm = shared_per_sm(context)?;
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        let (rows, _) = tile_grid(m, n, BLOCK_N);
        for scheduler in SCHEDULERS {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler,
                    group,
                    rung: SHIPPED,
                    epilogue: Epilogue::Fused,
                };
                run(context, m, n, k, plan, nothing_after)?;
            }
            let clusters = grid_for(scheduler, m, n, SHIPPED, MAX_CLUSTERS) / RANKS;
            notes.push(format!(
                "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} of {rows} tile-rows \
                 ({} tiles over {clusters} clusters)",
                scheduler.name(),
                tiles(m, n, BLOCK_N)
            ));
        }
        // #15's deferred store phase, under the same widths and the same `==`.
        //
        // It is on this gate rather than beside a benchmark row because the two
        // ways it can go wrong are both silent and neither is a fault: a
        // deferred drain that is not ordered against the next item's MMA reads
        // an accumulator the leader has begun overwriting, and a cluster that
        // forgets its last item drops one tile of `C` entirely. The second is
        // exactly what `ITEMS_M x ITEMS_N` is here to catch — a shape with more
        // tiles than clusters, so every cluster carries a pending item across
        // several boundaries and one past the loop.
        for group in CHECK_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                rung: SHIPPED,
                epilogue: Epilogue::Deferred,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} tiles, one deferred \
             accumulator per cluster in flight)",
            Epilogue::Deferred.name(),
            tiles(m, n, BLOCK_N)
        ));
        // #87's rungs, on the static schedule they are swept on. A wider pair
        // tile is a different item map, a different accumulator and a
        // different epilogue loop, and every way those go wrong is a wrong `C`
        // rather than a fault — so no rung reaches a clock without passing the
        // same element-by-element `==` the shipped kernel does. The traversal
        // widths go with them, because `grouped`'s short last group is a
        // function of `tiles_m`, which a rung's `block_n` moves.
        for rung in RUNGS.into_iter().filter(|rung| *rung != SHIPPED) {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler: Scheduler::Static,
                    group,
                    rung,
                    epilogue: Epilogue::Fused,
                };
                run(context, m, n, k, plan, nothing_after)?;
            }
            notes.push(format!(
                "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} \
                 ({} tiles, {} B, {} CTAs/SM)",
                rung.name(),
                tiles(m, n, rung.block_n),
                rung.shared(),
                rung.ctas_per_sm(per_sm)
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
    run(
        context,
        shape.m,
        shape.n,
        shape.k,
        Plan::new(Scheduler::Static),
        time,
    )
    .map(|(_, timings)| timings)
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
///
/// **Both columns now walk at [`GROUP`], where #97's ran row-major**, so the
/// absolute milliseconds here are not comparable to that table and the
/// *difference* between the columns still is — which is what this table is for.
/// [`swizzle`] is where the traversal is the variable.
pub fn compare(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm schedulers — min ms over 30 timed launches, both on one shared plan,\n\
         both walking pipeline::grouped at width {GROUP} (#89) rather than row-major.\n\
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
            let (_, timings) = run(context, m, n, k, Plan::new(scheduler), time)?;
            milliseconds.push(timings.min());
        }
        let (static_ms, clc_ms) = (milliseconds[0], milliseconds[1]);
        println!(
            "{:<18}{:>7}{:>7}{:>9.1}%{:>11.1}%{:>11.4}{:>11.4}{:>10.1}%",
            shape,
            tiles(m, n, BLOCK_N),
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

/// Traversal widths [`swizzle`] sweeps, in tile-rows. `1` is the row-major map
/// this kernel had through #97 and is the control; the rest are kept in the
/// table whatever they measure.
const GROUPS: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// The shape the width is chosen at, and the one [`GROUP`] should be read as
/// belonging to. 8192³ because it is the size every other table in this repo
/// is quoted at, and because its 32 x 64 tile grid is tall enough for the
/// widths above to be distinguishable and short enough that 32 saturates —
/// which puts both ends of the parameter's range inside the sweep.
const SWEEP: Shape = Shape {
    m: 8192,
    n: 8192,
    k: 8192,
};

/// The two sizes #89's acceptance names, measured either side of the change.
const HEADLINE: [Shape; 2] = [
    SWEEP,
    Shape {
        m: 16384,
        n: 16384,
        k: 16384,
    },
];

/// One checked, timed launch. The only way a number reaches any table below,
/// which is what keeps [`crate::bench`]'s rule 1 in force here: `run` compares
/// every element of `C` against the CPU reference before `time` is reachable,
/// so a traversal that dropped or doubled a tile reports a failure rather than
/// a throughput.
fn timed(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
    plan: Plan,
) -> Result<f64, Box<dyn Error>> {
    eprintln!(
        "{shape} on {} at group {}: staging and checking",
        plan.scheduler.name(),
        plan.group
    );
    let (_, timings) = run(context, shape.m, shape.n, shape.k, plan, time)?;
    Ok(timings.min())
}

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

fn same(left: Shape, right: Shape) -> bool {
    (left.m, left.n, left.k) == (right.m, right.n, right.k)
}

/// Traversal widths the headline `[256, 256]` rung is re-swept at.
///
/// [`GROUP`] was measured at `[256, 128]`, and a wider pair tile halves
/// `tiles_n` and changes how many clusters a wave holds — both inputs to
/// [`wave_reuse`] — so carrying 8 across would be assuming the answer this
/// sweep exists to move. Four widths rather than [`GROUPS`]' six, because each
/// one is a staged 8192³ and the two ends of that range are already known to
/// lose.
const TILE_GROUPS: [u32; 4] = [1, 4, 8, 16];

/// #87's tile and depth sweep — `modal run modal_app.py::bench --case tile`.
///
/// Three tables, and the design is what makes them separable. #87 argues for
/// `[256, 256]` on **arithmetic intensity** — 128.0 FLOP/byte against 85.3, a
/// 1.5× cut in operand traffic. There is a second mechanism the issue does not
/// name and which is probably the stronger one: a wider tile **halves the
/// output tile count**, and therefore halves the number of item boundaries,
/// which is the term #90 and #94 priced. The two are confounded in the obvious
/// experiment, because both scale with `N`.
///
/// What separates them is `[256, 128] @ STAGES = 4`. It has the *same* shared
/// plan as `[256, 256] @ STAGES = 3` (98 408 B against 98 392) and therefore
/// the same two CTAs per SM, at unchanged intensity and unchanged tile count.
/// So the step down in occupancy is priced by that rung on its own, and what
/// the `[256, 256]` rung has over it is exactly the two tile mechanisms and
/// nothing else.
///
/// The other prediction stated in advance: `[256, 256] @ STAGES = 2` is the
/// first rung in this repo whose residency is set by **tensor memory**. Shared
/// memory admits three CTAs there and 256 accumulator columns admit two, so
/// `src/tmem.rs` says the census should read *three resident against two
/// holding* — a CTA admitted and parked inside a blocking `tcgen05.alloc`.
/// Every rung in this repo so far has had those two columns equal.
///
/// Every row is checked against the CPU reference before it is timed, by the
/// same [`run`] the rest of the harness uses. The losers stay in the table.
pub fn tile_sweep(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    let per_sm = shared_per_sm(context)?;
    println!(
        "gemm pair tile and pipeline depth (#87) — min ms over 30 timed launches, every row\n\
         checked against the CPU reference first, static schedule throughout.\n\
         `CTA/SM` is min(512/columns, {per_sm}/plan) — #84's formula, with the shared\n\
         memory an SM divides *queried* rather than written down. It is predicted; the\n\
         counted figure is `device-tests`' `tmem residency census`, and predicted and\n\
         counted disagreeing is a finding rather than a rounding error.\n\
         `intensity` is the pair tile's M*N/(M+N), flops per operand byte."
    );

    println!("\n1. the rungs, and what each one costs before anything is launched");
    println!(
        "{:<16}{:>8}{:>10}{:>12}{:>10}{:>10}{:>12}",
        "rung", "stages", "shared B", "TMEM cols", "CTA/SM", "clusters", "intensity"
    );
    for rung in RUNGS.into_iter().chain([UNBUILT]) {
        let built = if rung == UNBUILT { "  (not built)" } else { "" };
        println!(
            "{:<16}{:>8}{:>10}{:>12}{:>10}{:>10}{:>12.1}{built}",
            rung.name(),
            rung.stages,
            rung.shared(),
            rung.block_n,
            rung.ctas_per_sm(per_sm),
            rung.max_clusters(per_sm),
            rung.intensity()
        );
    }

    println!(
        "\n2. the sweep, at the two sizes every other table here is quoted at.\n\
         wave eff moves with the rung and is not a nuisance to divide out: a residency\n\
         step resizes the persistent grid, so the ragged last wave moves with it — here\n\
         in the *favourable* direction for the rungs that lose residency, which means\n\
         this table if anything understates what an occupancy step costs them."
    );
    println!(
        "{:<16}{:<18}{:>8}{:>7}{:>10}{:>9}{:>11}{:>12}{:>9}",
        "rung", "shape", "tiles", "waves", "wave eff", "reuse", "min ms", "TFLOP/s", "vs #102"
    );
    let mut measured: Vec<(Rung, Shape, f64)> = Vec::new();
    for &shape in &HEADLINE {
        // The control first at every size, so the `vs #102` column has its
        // denominator from the row above rather than from a later one. The
        // reference is the kernel that shipped *before* this sweep, which is
        // what makes every delta a comparison against something measured many
        // times rather than against the sweep's own winner.
        let order = [CONTROL]
            .into_iter()
            .chain(RUNGS.into_iter().filter(|rung| *rung != CONTROL));
        for rung in order {
            let cap = rung.max_clusters(per_sm);
            let plan = Plan {
                scheduler: Scheduler::Static,
                group: GROUP,
                rung,
                epilogue: Epilogue::Fused,
            };
            eprintln!("{shape} on {}: staging and checking", rung.name());
            let (_, timings) = run(context, shape.m, shape.n, shape.k, plan, time)?;
            let milliseconds = timings.min();
            measured.push((rung, shape, milliseconds));
            let (waves, efficiency) = wave_efficiency_of(shape.m, shape.n, rung, cap);
            let (_, _, _, reuse) = wave_reuse(shape.m, shape.n, GROUP, rung.block_n, cap);
            let reference = measured
                .iter()
                .find(|row| row.0 == CONTROL && same(row.1, shape))
                .map(|row| row.2);
            println!(
                "{:<16}{:<18}{:>8}{:>7}{:>9.1}%{:>8.1}x{:>11.4}{:>12.1}{:>9}",
                rung.name(),
                shape,
                tiles(shape.m, shape.n, rung.block_n),
                waves,
                100.0 * efficiency,
                reuse,
                milliseconds,
                tflops(shape, milliseconds),
                match reference {
                    Some(before) => format!("{:+.1}%", 100.0 * (before / milliseconds - 1.0)),
                    None => "—".to_string(),
                }
            );
        }
    }

    println!(
        "\n3. the traversal width at the widest rung, at {SWEEP}.\n\
         GROUP = {GROUP} was measured at [256,128] (#89) and a wider pair tile halves\n\
         tiles_n and changes a wave's cluster count, so carrying it across would assume\n\
         the answer. The [256,128] row is the control."
    );
    println!(
        "{:<16}{:>8}{:>11}{:>11}{:>9}{:>12}{:>12}",
        "rung", "group", "wave rows", "wave cols", "reuse", "min ms", "TFLOP/s"
    );
    for rung in [CONTROL, SHIPPED] {
        let cap = rung.max_clusters(per_sm);
        for group in TILE_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                rung,
                epilogue: Epilogue::Fused,
            };
            eprintln!(
                "{SWEEP} on {} at group {group}: staging and checking",
                rung.name()
            );
            let (_, timings) = run(context, SWEEP.m, SWEEP.n, SWEEP.k, plan, time)?;
            let milliseconds = timings.min();
            let (rows, columns, _, reuse) = wave_reuse(SWEEP.m, SWEEP.n, group, rung.block_n, cap);
            println!(
                "{:<16}{:>8}{:>11}{:>11}{:>8.1}x{:>12.4}{:>12.1}",
                rung.name(),
                group,
                rows,
                columns,
                reuse,
                milliseconds,
                tflops(SWEEP, milliseconds)
            );
        }
    }

    println!(
        "\n4. against cuBLASLt on the same device in the same container — the denominator,\n\
         and the drift control (#98) that says how much of a delta above is the session."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>16}{:>16}",
        "shape", "cuBLASLt ms", "theirs TF/s", "#102/theirs", "shipped/theirs"
    );
    for &shape in &HEADLINE {
        let Some(baseline) = baseline else {
            println!(
                "no cuBLASLt column: built without --features cublas. modal_app.py::bench\n\
                 turns it on, and a ratio is the point of this table."
            );
            break;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        let at = |rung: Rung| {
            measured
                .iter()
                .find(|row| row.0 == rung && same(row.1, shape))
                .map(|row| row.2)
        };
        println!(
            "{:<18}{:>14.4}{:>14.1}{:>16}{:>16}",
            shape,
            theirs,
            tflops(shape, theirs),
            match at(CONTROL) {
                Some(ours) => format!("{:.3}", theirs / ours),
                None => "—".to_string(),
            },
            match at(SHIPPED) {
                Some(ours) => format!("{:.3}", theirs / ours),
                None => "—".to_string(),
            }
        );
    }
    Ok(())
}

/// Sizes the epilogue sweep runs, chosen so the answer has to be a *shape* and
/// not a number.
///
/// The item boundary is a fixed cost per output tile, so what [`Epilogue`]
/// moves is amortized over the arithmetic between one boundary and the next.
/// That makes the gain a decreasing function of `K` per item and of tiles per
/// cluster, and these five sizes span both: 1024³ is 16 tiles over 148
/// clusters — one item each, nothing to amortize and **no successor to overlap
/// with at all**, so `lcsf` should do nothing there or lose — while 16384³ is
/// 4096 tiles over 148 clusters at 256 K blocks apiece.
///
/// A gain that is flat across this range is not the item boundary and would
/// refute the mechanism rather than confirm the win.
const EPILOGUE_SIZES: [Shape; 5] = [
    Shape {
        m: 1024,
        n: 1024,
        k: 1024,
    },
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

/// #15's sweep — `cargo oxide run kittens-examples -- bench epilogue`, which is
/// `modal run modal_app.py::bench --case epilogue`.
///
/// One variable moves: whether [`Tile::drain`] runs at the end of its own item
/// or at the start of the next one. The grid, the shared plan, the tensor
/// memory, the pair tile, the traversal width and the schedule are identical in
/// both columns, and the shared plan being *unchanged* is what makes this
/// readable at all — #94 and #101 could not separate a staging buffer's cost
/// from its benefit precisely because a real one moves the epilogue and the
/// residency at once. Here the residency cannot move: [`CTAS_PER_SM`] is 2
/// because 256 accumulator columns are half of tensor memory, and nothing below
/// touches a column or a byte.
///
/// # The prediction, written down before it ran
///
/// #104's re-fit puts the item boundary at **8.6–18.3 µs per tile and 20.0–25.3%
/// of 8192³**, and that fit is a poor instrument — the intercept now carries a
/// K-dependent locality term, so the spread is point selection and not
/// precision. What `lcsf` can hide is the part of that which is the *epilogue*
/// rather than the drain of the last MMAs or the barrier round trip: the LDTM
/// and the scattered fp32 stores, which #91's evidence puts at roughly half the
/// boundary.
///
/// So: **+4% to +11% at 8192³ and 16384³, nothing at 1024³, and monotone
/// between.** The 1024³ row is the falsifiable one — one item per cluster means
/// every epilogue is the *last* epilogue, [`Lcsf::finish`] pays it un-hidden,
/// and the extra `cluster_sync` inside the item is a pure cost. If 1024³ gains,
/// the mechanism is not the one this function claims.
///
/// The second prediction is about registers and it is the one that would stop
/// this: the deferred drain reads tensor memory the previous item wrote, and
/// nothing is held in registers across the boundary, so peak liveness is
/// unchanged and `regcount` should report **168 and no spill**. A band held
/// across the boundary would be 128 fp32 a thread and would show up there
/// instead of here.
pub fn epilogue_sweep(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm epilogue placement (#15) — min ms over 30 timed launches, every row checked\n\
         element-by-element against the CPU reference before it was timed, static schedule\n\
         and GROUP = {GROUP} throughout.\n\
         `lcf` folds the store into the item; `lcsf` defers it one item, so it runs while\n\
         the next item's first {} stages are in flight. Same grid, same {SHARED_BYTES} B\n\
         shared plan, same {BLOCK_N} accumulator columns, same {CTAS_PER_SM} CTAs an SM —\n\
         the store stage needs no staging buffer, so there is no occupancy step in this\n\
         table and nothing has to be argued for out of a budget.",
        STAGES
    );
    println!("\n1. the two epilogues, over sizes chosen so the gain has to fall with amortization");
    println!(
        "{:<18}{:>8}{:>10}{:>12}{:>12}{:>12}{:>12}{:>10}",
        "shape", "tiles", "items/cl", "lcf ms", "lcf TF/s", "lcsf ms", "lcsf TF/s", "vs lcf"
    );
    let mut measured = Vec::new();
    for shape in EPILOGUE_SIZES {
        let plan = Plan::new(Scheduler::Static);
        let fused = timed(context, shape, plan)?;
        let deferred = timed(context, shape, plan.with(Epilogue::Deferred))?;
        let hot = timed(context, shape, plan.with(Epilogue::HotStore))?;
        let items = tiles(shape.m, shape.n, BLOCK_N);
        let clusters = grid_for(Scheduler::Static, shape.m, shape.n, SHIPPED, MAX_CLUSTERS) / RANKS;
        println!(
            "{:<18}{:>8}{:>10.1}{:>12.4}{:>12.1}{:>12.4}{:>12.1}{:>10}",
            shape,
            items,
            items as f64 / clusters as f64,
            fused,
            tflops(shape, fused),
            deferred,
            tflops(shape, deferred),
            format!("{:+.1}%", 100.0 * (fused / deferred - 1.0))
        );
        measured.push((shape, fused, deferred, hot));
    }

    println!(
        "\n2. what the gain implies about the item boundary. #104 fits the boundary at\n\
         8.6-18.3 us per tile over three point selections; `hidden` divides the measured\n\
         saving per item by that range, so it is the fraction of the *fitted* boundary\n\
         this placement removed. Two numbers because the fit's own spread is 2.1x, which\n\
         is why this column is a range and not a figure."
    );
    println!(
        "{:<18}{:>12}{:>14}{:>16}{:>18}",
        "shape", "items/cl", "saved ms", "saved us/item", "hidden of boundary"
    );
    for &(shape, fused, deferred, _) in &measured {
        let items = tiles(shape.m, shape.n, BLOCK_N);
        let clusters = grid_for(Scheduler::Static, shape.m, shape.n, SHIPPED, MAX_CLUSTERS) / RANKS;
        // Items on the critical path, which is what #90 and #104 divide a
        // fitted intercept by — the deepest-loaded cluster, not the average.
        let critical = items.div_ceil(clusters) as f64;
        let saved = fused - deferred;
        let per_item = saved * 1e3 / critical;
        println!(
            "{:<18}{:>12.1}{:>14.4}{:>16.2}{:>18}",
            shape,
            items as f64 / clusters as f64,
            saved,
            per_item,
            format!(
                "{:.0}%-{:.0}%",
                100.0 * per_item / 18.3,
                100.0 * per_item / 8.6
            )
        );
    }

    println!(
        "\n2b. what the epilogue costs, split into issue and bandwidth. `hot` is the fused\n\
         kernel with every store aimed at the cluster's own first tile — same LDTM, same\n\
         store count, same addresses within a tile, 38 MB of C rewritten instead of 268 MB\n\
         or 1 GB streamed, so the writes stay in L2. IT COMPUTES A WRONG C AND IS NOT\n\
         CHECKED; the number is an upper bound on what decoupling the global write could\n\
         ever be worth, and that is exactly what #15's TMEM -> shared -> TMA route buys\n\
         over the deferred drain above.\n\
         A `hot` close to `lcf` means the epilogue is issue-bound and no staging buffer\n\
         recovers it. A `hot` far below means it is bandwidth-bound and the route has\n\
         that much, and no more, to win."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>14}{:>18}",
        "shape", "lcf ms", "hot ms", "hot TF/s", "write-bound part"
    );
    for &(shape, fused, _, hot) in &measured {
        println!(
            "{:<18}{:>12.4}{:>12.4}{:>14.1}{:>18}",
            shape,
            fused,
            hot,
            tflops(shape, hot),
            format!("{:.1}%", 100.0 * (fused - hot) / fused)
        );
    }

    println!(
        "\n3. against cuBLASLt on the same device in the same container — the denominator,\n\
         and the drift control that says how much of the delta above is the session."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>14}{:>16}",
        "shape", "cuBLASLt ms", "theirs TF/s", "lcf/theirs", "lcsf/theirs"
    );
    for &(shape, fused, deferred, _) in &measured {
        if shape.m < 8192 {
            continue;
        }
        let Some(baseline) = baseline else {
            println!(
                "no cuBLASLt column: built without --features cublas. modal_app.py::bench\n\
                 turns it on, and a ratio is the point of this table."
            );
            break;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        println!(
            "{:<18}{:>14.4}{:>14.1}{:>14.3}{:>16.3}",
            shape,
            theirs,
            tflops(shape, theirs),
            theirs / fused,
            theirs / deferred
        );
    }
    Ok(())
}

/// The traversal sweep — `cargo oxide run kittens-examples -- bench swizzle`,
/// which is `modal run modal_app.py::bench --case swizzle`.
///
/// Four tables, and each one exists to test a claim that was written down
/// before it ran. In order:
///
/// 1. **The width sweep**, at [`SWEEP`], under both schedulers. The losers stay
///    in the table. The width the rest of the run uses is picked from the
///    static column here rather than from [`GROUP`], so this function measures
///    the parameter instead of confirming the constant.
/// 2. **The mechanism**, which is the table worth reading first. If the width
///    works by leaving operands in L2, then at a `K` whose operands fit L2
///    *whole* it can do nothing at all — there is no miss left to avoid — and
///    at a `K` far past L2 it must do the most. So the same sweep runs over
///    [`crate::bench::GEMM_DEPTH_SIZES`], where `A` and `B` are 8 MiB each at
///    `K = 512` against 512 MiB each at `K = 32768`. **A width that helps at
///    both ends is not doing what this issue claims**, and that is a result
///    this table can produce.
/// 3. **The aspect ratio**, over #90's five shapes — identical in flops, tiles,
///    waves, grid, `C` bytes and arithmetic intensity, differing only in
///    `M : N`. Row-major throughput across them moved 1123.7 → 915.7 TFLOP/s,
///    monotone in the reuse the map leaves L2. A grouped walk makes the wave's
///    footprint nearly independent of the shape, so the prediction is that the
///    **spread collapses**: the worst rows gain most and the best row, which is
///    already walking a near-square wave, barely moves.
/// 4. **Before and after** at the two sizes #89 asks for, under both
///    schedulers, with cuBLASLt beside them where the feature is on.
///
/// The `reuse` column throughout is [`wave_reuse`] — arithmetic on the item
/// map, not a counter, because this harness has none. See [`wave_reuse`].
pub fn swizzle(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm traversal — min ms over 30 timed launches, every row checked against the CPU\n\
         reference first. `group` is pipeline::grouped's width in tile-rows: 1 is the\n\
         row-major map through #97, and is the control rather than a separate code path.\n\
         `reuse` is operand bytes a wave requests over the distinct bytes it touches at one\n\
         K block — arithmetic on the item map, not a counter. There is no L2 hit rate in\n\
         this table because there is none on this harness: Nsight Compute cannot start in\n\
         the Modal image for want of libnvidia-pcc.so (modal_app.py::profile)."
    );

    println!("\n1. width, at {SWEEP} — {} clusters a wave", MAX_CLUSTERS);
    println!(
        "{:<8}{:>11}{:>11}{:>12}{:>9}{:>12}{:>12}{:>12}{:>12}",
        "group",
        "wave rows",
        "wave cols",
        "distinct MB",
        "reuse",
        "static ms",
        "static TF/s",
        "clc ms",
        "clc TF/s"
    );
    let mut swept = Vec::new();
    for group in GROUPS {
        let static_ms = timed(
            context,
            SWEEP,
            Plan {
                scheduler: Scheduler::Static,
                group,
                rung: SHIPPED,
                epilogue: Epilogue::Fused,
            },
        )?;
        let clc_ms = timed(
            context,
            SWEEP,
            Plan {
                scheduler: Scheduler::Stealing,
                group,
                rung: SHIPPED,
                epilogue: Epilogue::Fused,
            },
        )?;
        let (rows, columns, distinct, reuse) =
            wave_reuse(SWEEP.m, SWEEP.n, group, BLOCK_N, MAX_CLUSTERS);
        println!(
            "{:<8}{:>11}{:>11}{:>12.2}{:>8.1}x{:>12.4}{:>12.1}{:>12.4}{:>12.1}",
            group,
            rows,
            columns,
            distinct / 1e6,
            reuse,
            static_ms,
            tflops(SWEEP, static_ms),
            clc_ms,
            tflops(SWEEP, clc_ms)
        );
        swept.push((group, static_ms, clc_ms));
    }

    let &(best, _, _) = swept
        .iter()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("GROUPS is not empty");
    let &(clc_best, _, _) = swept
        .iter()
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .expect("GROUPS is not empty");
    println!(
        "\nfastest static width at {SWEEP}: group {best}. fastest under clc: group {clc_best}.\n\
         the rest of this run uses {best}, and GROUP in this file is set from it — at this\n\
         tile shape only. #87 moves the tile, which moves this."
    );

    // Every table below asks for 8192³ rows the sweep already paid for, and
    // re-staging a shape to re-time a plan that has already been timed would
    // add half an hour of host time for numbers that would only disagree by
    // the benchmark's own spread. Anything else is measured.
    let measured =
        |shape: Shape, group: u32, scheduler: Scheduler| -> Result<f64, Box<dyn Error>> {
            if same(shape, SWEEP)
                && let Some(&(_, static_ms, clc_ms)) = swept.iter().find(|row| row.0 == group)
            {
                return Ok(match scheduler {
                    Scheduler::Static => static_ms,
                    Scheduler::Stealing => clc_ms,
                });
            }
            timed(
                context,
                shape,
                Plan {
                    scheduler,
                    group,
                    rung: SHIPPED,
                    epilogue: Epilogue::Fused,
                },
            )
        };

    println!(
        "\n2. the mechanism — the same two widths against the operand footprint, static.\n\
         M and N are 8192 throughout, so tiles, waves, grid, C bytes and the wave's own\n\
         working set are identical in every row and only K moves. A and B are 8 MiB each\n\
         at K=512 and 512 MiB each at K=32768, so the first row's operands are L2-resident\n\
         whole and the last row's cannot be. **Prediction: no gain at K=512, the largest\n\
         gain at K=32768.** A width that gains at both ends is not working through L2."
    );
    println!(
        "{:<18}{:>13}{:>12}{:>12}{:>12}{:>12}{:>11}",
        "shape", "operand MiB", "group 1 ms", "group 1 TF/s", "best ms", "best TF/s", "gain"
    );
    for &shape in crate::bench::GEMM_DEPTH_SIZES {
        let row_major = measured(shape, 1, Scheduler::Static)?;
        let grouped = measured(shape, best, Scheduler::Static)?;
        let operand = 2.0 * (shape.m * shape.k) as f64 * 2.0 / (1 << 20) as f64;
        println!(
            "{:<18}{:>13.0}{:>12.4}{:>12.1}{:>12.4}{:>12.1}{:>10.1}%",
            shape,
            operand,
            row_major,
            tflops(shape, row_major),
            grouped,
            tflops(shape, grouped),
            100.0 * (row_major / grouped - 1.0)
        );
    }

    println!(
        "\n3. the aspect ratio — #90's five shapes, identical but for M : N, static.\n\
         row-major moved 1123.7 -> 915.7 TFLOP/s across these, monotone in `reuse`.\n\
         **Prediction: the spread collapses.** a grouped wave's footprint barely depends\n\
         on the shape, so the bottom rows gain most and the top row, already walking a\n\
         near-square wave, barely moves."
    );
    println!(
        "{:<18}{:>9}{:>9}{:>12}{:>12}{:>9}{:>12}{:>12}{:>10}",
        "shape",
        "tiles_n",
        "reuse 1",
        "group 1 ms",
        "group 1 TF/s",
        "reuse",
        "best ms",
        "best TF/s",
        "gain"
    );
    for &shape in crate::bench::GEMM_FOOTPRINT_SIZES {
        let row_major = measured(shape, 1, Scheduler::Static)?;
        let grouped = measured(shape, best, Scheduler::Static)?;
        let (_, columns) = tile_grid(shape.m, shape.n, BLOCK_N);
        let (_, _, _, was) = wave_reuse(shape.m, shape.n, 1, BLOCK_N, MAX_CLUSTERS);
        let (_, _, _, now) = wave_reuse(shape.m, shape.n, best, BLOCK_N, MAX_CLUSTERS);
        println!(
            "{:<18}{:>9}{:>8.1}x{:>12.4}{:>12.1}{:>8.1}x{:>12.4}{:>12.1}{:>9.1}%",
            shape,
            columns,
            was,
            row_major,
            tflops(shape, row_major),
            now,
            grouped,
            tflops(shape, grouped),
            100.0 * (row_major / grouped - 1.0)
        );
    }

    println!(
        "\n4. before and after at the sizes #89 names, under both schedulers.\n\
         the swizzle applies to the item, not to the item source, so it composes with CLC\n\
         rather than replacing it — these two columns are the check on that claim."
    );
    println!(
        "{:<18}{:>7}{:>12}{:>12}{:>12}{:>12}{:>14}{:>14}",
        "shape",
        "group",
        "static ms",
        "static TF/s",
        "clc ms",
        "clc TF/s",
        "cuBLASLt ms",
        "static/theirs"
    );
    for shape in HEADLINE {
        let against = match baseline {
            Some(baseline) => {
                eprintln!("{shape}: staging and checking {}", baseline.name);
                Some((baseline.bench)(context, shape)?.0.min())
            }
            None => None,
        };
        for group in [1, best] {
            let static_ms = measured(shape, group, Scheduler::Static)?;
            let clc_ms = measured(shape, group, Scheduler::Stealing)?;
            println!(
                "{:<18}{:>7}{:>12.4}{:>12.1}{:>12.4}{:>12.1}{:>14}{:>14}",
                shape,
                group,
                static_ms,
                tflops(shape, static_ms),
                clc_ms,
                tflops(shape, clc_ms),
                match against {
                    Some(milliseconds) => format!("{milliseconds:.4}"),
                    None => "—".to_string(),
                },
                match against {
                    Some(milliseconds) => format!("{:.3}", milliseconds / static_ms),
                    None => "—".to_string(),
                }
            );
        }
    }
    if baseline.is_none() {
        println!(
            "\nno cuBLASLt column: built without --features cublas. modal_app.py::bench\n\
             turns it on, and a ratio is the point of the last column."
        );
    }
    Ok(())
}
