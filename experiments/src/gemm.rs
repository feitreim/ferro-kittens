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
//! like the stronger one — is worth less than nothing here. `experiments/README.md`
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
//! What the scaffold does *not* buy is overlap between items. Every epilogue
//! this file ships folds into the item — `staged84` moved the drain's *shape*
//! and not its placement — so this kernel's stores and the next tile's
//! first K loads are still separated by a boundary that drains the pipeline —
//! #15's `lcsf` is the shape that would let them cross. So the honest claim for
//! this port is a **dead heat**: 1.0217 ms at 8192³ against the launch-per-tile
//! grid's 1.0204, which is 1076 TFLOP/s against 1077. It is not faster. What it
//! buys is that the scaffold has a caller, and therefore a regression test.
//! `experiments/README.md` §7 is the sweep that found the cap and what it says
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
//! one cluster running one item, and it costs 23.4 µs. See `experiments/README.md`
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
//! Both fits are the register drain's, and `bench --case gemm-depth` no longer
//! runs it: since #119 that sweep launches [`SHIPPED_EPILOGUE`] like every
//! other `gemm` row, so a re-fit taken today is against `staged84` and belongs
//! beside these two rather than in place of either.
//!
//! ## What the item boundary is, ablated rather than fitted
//!
//! Every figure above for a *part* of this kernel is a share of that intercept,
//! and [`Ablation`] is what measures the parts directly: the same kernel with
//! one phase of the item removed, priced at more than one corner so that
//! **serial** and **exposed** costs are told apart rather than assumed. On the
//! bf16 kernel the fit re-measures at **18.0–25.0 µs a tile**, and the
//! decomposition of it is not the five-member chain this file has described
//! since #86:
//!
//! - **The epilogue is the whole of it.** Delete the LDTM and the stores and
//!   leave every barrier, boundary and refill where it is, and the intercept
//!   goes to zero at every point selection. The barrier re-arm, the two cluster
//!   boundaries and the pipeline refill are all inside the `K`-proportional
//!   term.
//! - **And it is exposed, not hidden.** It costs 20.4 µs a tile with the
//!   multiply beside it and 20.6 without — a ratio of 1.01 — and #108's `2x`
//!   measured the same quantity *serially* at 21.4 by a route sharing none of
//!   this arithmetic. #109's reading, that the epilogue is partly overlapped
//!   and the fit sees its residue, is refuted.
//! - **So the gap to cuBLASLt is the epilogue.** [`Ablation::NoDrain`] measures
//!   **1850 TFLOP/s at 8192³ and 1841 at `K = 32768`** — 82% of dense bf16
//!   peak, flat across a 4× change in depth — against the library's 1808 in the
//!   same container. The epilogue costs 111% of the entire distance between the
//!   two kernels, and every other term this file has ranked lives inside the
//!   part that is already faster than the library.
//!
//! `experiments/README.md` §7 has the tables, the PTX census that says each rung
//! removes what it names, and the three corners that do not run.
//!
//! **Every figure in the three bullets above is the register drain's, and that
//! epilogue is no longer the default.** The two sections below are what was
//! spent against them: the same subtraction reads 20.43 µs a tile for `lcf`,
//! 14.96 for `staged` and 6.68 for the shipped `staged84` at 8192³, so the
//! epilogue is now roughly a third of the term these bullets rank everything
//! by. The ladder itself is unchanged and still runs on `lcf`, which is what
//! keeps the two comparable.
//!
//! ## And the epilogue can be staged, which is worth 2–8%
//!
//! [`Epilogue::Staged`] is what that conclusion was worth spending:
//! [`Tile::drain_staged`] moves the band TMEM → registers → `stmatrix` into a
//! per-warp `[32, 64]` shared tile → 16-byte stores, where [`Tile::drain`]
//! stores 4 bytes a thread straight out of registers. **+8.0% at 4096³, +4.2%
//! at 8192³ and +1.9% at 16384³**, taking the kernel to 1558.0 and 1692.7
//! TFLOP/s and from 0.822 to 0.856 and 0.877 to 0.893 of cuBLASLt.
//!
//! Measured on the *epilogue* rather than on the launch it is amortized into,
//! by #114's own `whole − no drain` subtraction at a fixed envelope, it is
//! **−19.0% / −19.6% / −24.4%**. The `lcf` arm of that subtraction reproduces
//! #114's 20.43 µs a tile at 20.14 in a different container.
//!
//! **Total memory issue does not fall — it is 128 instructions a thread in
//! both**, which corrects the arithmetic #15 was scoped from; the whole of the
//! gain is that global stores fall 4× and land on full 128-byte lines. The
//! `stmatrix` ceiling of ~6.2% at 8192³ was set by a premise this does not
//! meet (that the stores *halve*), and 4.2% of it is what arrived.
//!
//! Two things it does not cost. **Residency**: 114 816 B against 98 392, still
//! two CTAs an SM because 256 accumulator columns bind first, counted by
//! `device-tests`' census and controlled for by
//! [`gemm_cg2_staged_no_drain`](kernels::gemm_cg2_staged_no_drain). And
//! **registers**: `ptxas` reads **42 against `gemm_cg2`'s 166** with no spill,
//! because a block-at-a-time LDTM → `stmatrix` never materializes the band that
//! [`kittens::global::store_rows`]' slot-major walk forces live.
//!
//! `gemm_ws` carries the same epilogue and gains 2.5–4.1% from it — which is
//! the control that says the win is the store's *shape* and not its placement,
//! since that kernel's epilogue was already deferred and already on warps of
//! its own. §7 has both.
//!
//! ## And the widths on top of it, which is what this file now ships
//!
//! [`SHIPPED_EPILOGUE`] is `staged84` — [`Epilogue::StagedWideX4`], the staged
//! drain with the LDTM half at `tcgen05.ld.16x256b.x8` and the `stmatrix` half
//! at `.m8n8.x4` (#117). [`bench`] launches it; [`gemm_cg2`](kernels::gemm_cg2)
//! and its register drain are the control every A/B above is taken against.
//!
//! **The mechanism is the wait and not the issue.** [`TmemTile::fragment`]
//! waits after *each* `.x1` load, because the registers it waits on are the
//! load's return value — so a `[32, 64]` band is 16 loads and 16 fully exposed
//! tensor-memory latencies where `.x8` is 2 and 2. That alone is
//! **+23.6% / +8.6% / +3.6%** over `staged`. `.x4` halves an instruction count
//! and is worth −0.6% / −0.3% / −1.1% *alone*, a clean null with a consistent
//! sign; composed the pair is **+23.1% / +8.8% / +5.1%**, which is 1668 and
//! 1723 TFLOP/s and **0.944 and 0.922 of cuBLASLt** at 8192³ and 16384³.
//! #119 re-measured that in its own container against its own cuBLASLt and got
//! 1645.0 and 1754.6 TFLOP/s, **0.939 and 0.958** — the largest ratio this
//! kernel has reached, and a reminder that the denominator moves too.
//!
//! **The composition gain is liveness, and the register column is what says
//! so.** `ptxas` reads 42 for `staged`, 94 for `staged8` — the 32 f32 that now
//! arrive at once — and 80 for `staged84`, because `.x4` consumes all four of
//! a fragment's matrices in one instruction. Those 14 registers are the whole
//! of why the composed rung beats `.x8` alone at 16384³. Zero spill in every
//! rung, and 80 is far inside the 255 that two CTAs at 128 threads admit —
//! `regcount`'s ceiling for this kernel has been 255 rather than 168 since #87
//! gave up the third CTA. Residency here is tensor-memory-bound at 256
//! accumulator columns and was never the count.
//!
//! **`gemm_ws` ships `staged8` instead, and the same column is why.** There
//! `.x8` costs the same +50 and `.x4` recovers **2** registers (94 → 92), so
//! the two do not compose, `staged8` and `staged84` sit within 1.1% and trade
//! places between sessions, and the rung with less surface wins the tie. The
//! two files differ here on purpose (#118, #119).
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
//! the width and keeps the losers, and `experiments/README.md` §7 has both.
//!
//! ## `C` is bf16, and the accumulator is not
//!
//! `bf16` in, fp32 accumulate, `bf16` out is the signature a training GEMM has.
//! This kernel accumulated in fp32 and *wrote* fp32 through #107, which is the
//! unusual half, and #108 moved the output alone: [`Accumulator`] is still
//! [`BLOCK_N`] fp32 columns of tensor memory, [`Band`] is still 128 fp32 a
//! thread, and the single `cvt.rn.bf16x2.f32` is inside the store
//! ([`kittens::global::store_rows`] over a [`kittens::global::GlobalRows`] of
//! [`Bf16`]). The shipped drain converts in [`kittens::ldst::store_tile_x4`]
//! on the way into shared instead, and the count is the same 128 a thread —
//! see the widths section above.
//!
//! **The cuBLASLt baseline moved in the same change and had to.** At 8192³ an
//! fp32 `C` is 268 MB and a bf16 `C` is 134 MB, so a baseline left at
//! `CUDA_R_32F` would write twice what this kernel writes and the difference
//! would read as ours. Every ratio in `experiments/README.md` §7 published before
//! #108 is against the fp32 pair; the section #108 adds re-measures both
//! columns in one session and says which of the older rows survive.
//!
//! What it changes in the epilogue is **not the instruction count**. Under
//! [`BaseLdtm`] a thread's four values of a 16-column block sit at column
//! offsets `{0, 1, 8, 9}` — two adjacent pairs, `CONTIGUOUS_VALUES = 2` — so a
//! pair store is one instruction at either element and only its width moves,
//! 8 bytes to 4. The `cvt` is new and the stores are not fewer, which is why
//! the prediction was neutral-to-slightly-negative on an epilogue #107 measured
//! as issue-bound rather than write-bound.
//!
//! **Measured, against an fp32 control run back to back: about +1% at 8192³ and
//! nothing resolvable at 16384³**, with the ratio to cuBLASLt flat to two points
//! down because the library gains at least as much from the cheaper output as we
//! do. Throughput was not the reason to do this and it did not have to pay for
//! itself; that it costs nothing is the result.
//!
//! What it *does* change is what the epilogue could be. `stmatrix` is b16 and
//! was unusable here for exactly one reason — `C` was fp32 — and #94 and #107
//! both rejected the TMEM → shared → TMA route partly on that. That constraint
//! is gone: [`kittens::ldst::store_tile`] already writes a
//! `RegTile<M, N, BaseLdtm>` into a swizzled [`Bf16`] tile through `stmatrix`,
//! and [`SharedTile::tma_store_2d`] already takes it out. §7 is where that is
//! priced rather than assumed.
//!
//! Both halves of that route are now built and both are on the correctness
//! gate: [`Tile::drain_staged`] is the `stmatrix` half (#116) and
//! [`Tile::drain_staged_tma`] is the engine (#123). The first was worth
//! +8.0% / +4.2% / +1.9% and ships; the second is **1.0–1.7% slower** than what
//! it replaces, so the route is half taken on purpose, and §7 is where that is
//! a measurement rather than the sentence above.
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
//! still a coordinate the reference would catch. The shipped drain reaches the
//! same cursor a hop later — [`kittens::ldst::store_tile_x4`] into a per-warp
//! shared tile and [`kittens::global::store_shared_rows`] out of it — and no
//! index arithmetic came back with it.
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

use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{
    DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, thread, warp,
};

// Host side: the launcher's error type, and the benchmark's size and clock.
use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::epilogue::{StoreRing, Warp};
use kittens::global::{GlobalRows, store_rows, store_shared_rows};
use kittens::ldst::{store_tile, store_tile_x4};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcQueue, Job};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{lane, warp_id};

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
/// and no registers, and `experiments/README.md` §7 has the four losing rungs and
/// the control that separates the two mechanisms.
///
/// **What it also costs is generality, and that is not free.** A launch must
/// now have `n % 256 == 0` where 128 used to do, so this kernel computes a
/// narrower set of shapes than it did — the direction #92 already named as
/// where a like-for-like rate against a general library flatters us.
const BLOCK_N: usize = 256;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K one linear MMA walk covers: **one 128-byte swizzle atom of bf16, and the
/// only width [`SharedTile::k_walk`] accepts.** Not a parameter, and not
/// because nobody swept it —
///
/// ```text
/// const { assert!(C * E::BYTES == S::ATOM_BYTES,
///     "a linear K-major walk needs K to span exactly one swizzle atom") }
/// ```
///
/// — is in `k_walk` itself, and `Swizzle128B` is the only mode in tree. So 64
/// is what a *walk* is, at bf16, and a stage that wants more K than that gets
/// it by holding several of these atoms rather than by widening the walk. That
/// is [`BLOCK_K`], which is a parameter.
const ATOM_K: usize = 64;
/// Chained K=16 MMA chunks in one atom's walk.
const ATOM_CHUNKS: usize = ATOM_K / 16;
/// K a pipeline stage carries, in the kernel this file ships — a whole number
/// of [`ATOM_K`] atoms, loaded by one TMA per operand and multiplied by one
/// walk per atom.
///
/// **Swept since this issue, and the sweep is more constrained than it looks.**
/// A stage's shared bytes are `512 · BLOCK_K · STAGES` for the pair tile this
/// kernel ships, so at the 116 736 B an SM divides two ways it is the *product*
/// `BLOCK_K · STAGES` that is capped, at 228 — and `BLOCK_K` does not move
/// arithmetic intensity at all, since a tile reads `(M + N) · K` bytes for
/// `2 · M · N · K` flops however K is blocked. What it does move is the number
/// of stage barriers, `expect_tx` charges and loop iterations an item pays, and
/// how coarsely the ring recycles.
///
/// So the sweep is a *factorization* of a fixed budget rather than an
/// independent axis, and [`RUNGS`] carries both of the pairs that hold the
/// bytes fixed and move only the factorization: `k64 s2` against `k128 s1` at
/// 65 KiB, and `k64 s3` against the 131 KiB `k128 s2` that is computed and not
/// built. `experiments/README.md` §7 has what they measured.
const BLOCK_K: usize = 64;
/// Pipeline depth over K, in the shipped kernel.
const STAGES: usize = 3;
/// One warp per 32 accumulator rows, which is what a `[32, N]` drain wants.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32;
/// Accumulator columns one warp drains in a single band of the
/// **register-drain** epilogue — [`Epilogue::Fused`] and [`Epilogue::Deferred`],
/// which since #119 are the control rather than the default. [`STAGE_N`] is
/// the shipped drain's band.
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
/// Accumulator columns one warp drains in a single band of the **staged**
/// epilogue ([`Epilogue::Staged`] and #117's three widths on it, which is
/// [`SHIPPED_EPILOGUE`]) — and not a swept parameter.
///
/// The band goes to shared memory through `stmatrix`, so its width is the
/// staging tile's width, and `SharedTile::WIDTH_OK` wants a whole swizzle
/// subtile: **at bf16 under `Swizzle128B` 64 columns is the narrowest tile that
/// exists**, and the widest one the budget below admits. So this is 64 because
/// both bounds meet there.
///
/// The budget is the whole of the arithmetic. At 2 CTAs an SM a CTA gets
/// `233 472 / 2 = 116 736` B and [`SHARED_BYTES`] spends 98 392 of it, leaving
/// 18 344. Four warps × `[32, 64]` bf16 is 16 384 B and fits; anything wider is
/// 32 768 and does not, and **`STAGES` is not available to buy the difference**
/// — `experiments/README.md` §7 prices a 3 → 2 step at −11.8% / −7.3% at
/// *unchanged* residency, which is pipeline depth and not an occupancy step.
const STAGE_N: usize = 64;
/// One warp's staging tile: its own 32 rows of `C` by [`STAGE_N`] columns,
/// 4096 B, and nobody else's.
///
/// **Per warp and not per CTA**, which is what keeps the barrier count at zero.
/// `stmatrix` is `.sync.aligned` and so is a convergence point for the warp
/// that issues it, and [`kittens::global::store_shared_rows`] is cooperative
/// rather than collective — so a warp that writes and then reads back its own
/// 4096 B needs no `bar.sync` at all, only the `bar.warp.sync` that separates
/// one pass's read from the next pass's write. A CTA-wide `[128, 64]` tile is
/// the same 16 384 B and would want two block barriers per pass instead.
///
/// A row-subrange of a swizzled tile *is* a swizzled tile here, which is why
/// four of these can be carved out of one 16 384 B run: `SWIZZLE_128B`'s period
/// is 8 rows and the XOR is over the row index, so a tile starting at a
/// multiple of 8 rows reproduces the layout it would have had on its own.
type StageTile = SharedTile<Bf16, 32, STAGE_N, Swizzle128B>;
/// [`StageTile`] as a depth-1 **warp-scope** store ring — the same 4096 B, with
/// the proxy fence, the `cp.async.bulk.tensor` issue and the read-wait a TMA
/// store owes stated by [`kittens::epilogue::StoreRing`] instead of here.
///
/// Depth 1 and not 2, and the shared budget is the whole of why: a second
/// buffer of this shape is another 16 384 B across the four warps and
/// [`STAGED_SHARED_BYTES`] has 1920 B left. So there is no overlap to be had
/// here — band `k + 1`'s `stmatrix` waits out band `k`'s store reads, which is
/// the shape `examples/src/softmax.rs` writes by hand.
///
/// **That wait is the leading explanation for why the TMA rungs lose**, and the
/// experiment that would settle it fits in the bytes this one already has: a
/// depth-2 ring of `[16, 64]` buffers is `2 × 2048 = 4096 B` a warp, exactly
/// what this costs, so a warp could drain its 32 rows as two 16-row halves with
/// the second half's `stmatrix` overlapping the first half's store and the
/// envelope would not move. What it needs is a 16-row band out of
/// [`TmemTile::tile_x8`], whose `.x8` shape is 32 — so the obstacle is the LDTM
/// width and not shared memory, and it is a follow-up rather than a variant of
/// this type.
///
/// **Warp scope is the point.** The CTA-scope ring the type shipped with would
/// put a `bar.sync` on both sides of every band, and [`StageTile`] is per warp
/// precisely so that there are none — see [`Tile::drain_staged_tma`], which is
/// the arm that measures whether that matters.
type StageRing = StoreRing<Bf16, 32, STAGE_N, Swizzle128B, 0, Warp>;
/// The four [`StageTile`]s read as **one CTA-wide tile**: the same 16 384 B run
/// at the same offset, addressed as `[BLOCK_M, STAGE_N]` rather than as four
/// row-quarters of it.
///
/// A row-subrange of a `Swizzle128B` tile is a swizzled tile whenever it starts
/// at a multiple of the 8-row period, which is what lets one run be read both
/// ways ([`StageTile`] says the same thing from the other side). So the two TMA
/// rungs differ in the *box* — `[128, 64]` in one instruction against `[32, 64]`
/// in four — and in the barriers that box implies, and in nothing else.
type CtaStage = SharedTile<Bf16, BLOCK_M, STAGE_N, Swizzle128B>;
/// [`CtaStage`] as a depth-1 CTA-scope store ring — [`kittens::epilogue::StoreRing`]
/// exactly as #111 built it, unparameterized.
type CtaRing = StoreRing<Bf16, BLOCK_M, STAGE_N, Swizzle128B, 0>;
/// The band a staged pass drains — [`Band`] at [`STAGE_N`] columns, so **64
/// fp32 a thread where the register epilogue holds 128**.
type StagedBand = RegTile<32, STAGE_N, BaseLdtm>;
/// CTAs in the cluster. Also the multiplier on a stage's transaction charge:
/// both ranks stage the same two tile types at the same shared offsets, so the
/// whole stage is one rank's charge twice over.
const RANKS: u32 = 2;
/// The CTA mask naming every half of the pair.
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
/// The rank that owns the pair's MMA, its accumulator and its stage barriers.
const LEADER: u32 = 0;

/// This CTA's `A` rows, K-major. The pair's `M` is fixed at 256 by the widest
/// `MmaShape` there is, so `BLOCK_K` is the only extent a rung moves here.
type ATile<const BLOCK_K: usize> = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// This CTA's `B` columns, also K-major — so the MMA carries no transpose
/// bits and computes `A·Bᵀ`.
type BTile<const HALF_N: usize, const BLOCK_K: usize> =
    SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing<const BLOCK_K: usize, const STAGES: usize> =
    SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing<const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> =
    SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// One swizzle atom of an operand tile, which is what a [`SharedTile::k_walk`]
/// can describe — a `BLOCK_K` wider than [`ATOM_K`] is this many of these,
/// stacked, and [`SharedTile::subtile`] is where each one starts.
type Atom<const R: usize> = SharedTile<Bf16, R, ATOM_K, Swizzle128B>;
/// This CTA's half of the pair's accumulator: 128 TMEM lanes by `BLOCK_N`
/// fp32 columns. The column count is what charges tensor memory, so it is also
/// the `512 / columns` half of the residency this kernel gets.
type Accumulator<const BLOCK_N: usize> = TmemTile<BLOCK_M, BLOCK_N>;
/// One warp's band of it, drained — [`DRAIN_N`] columns at a time.
type Band = RegTile<32, DRAIN_N, BaseLdtm>;

/// Warps in the block, and so buffers in the staging run.
const WARPS: usize = THREADS as usize / 32;
/// The staging run, as a ring of one [`StageTile`] per warp — the per-warp
/// offset is [`SharedTileRing::tile`]'s arithmetic and not a written-down
/// `warp_id * StageTile::BYTES`.
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, WARPS>;

/// Everything the launch's dynamic shared memory holds before the staging run,
/// in declaration order: the two operand rings, the two `STAGES`-deep barrier
/// rings, the MMA-complete semaphore, the word `alloc_cluster` stages its
/// result through, and the hardware work queue.
///
/// One walk. Before #125 this was two — an arithmetic `shared_plan` and a
/// pointer walk in [`attach`] — joined by a hand-written `const { assert!(..) }`
/// that could check the total and never an offset, with the barrier count
/// spelled `2 * STAGES * 8` in bytes at one end and `.add(STAGES)` in `Barrier`
/// units at the other. The queue's 16-byte alignment was a third `const` block,
/// and [`SharedPlan::clc_queue`] owns it now.
struct Shared<const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> {
    a_ring: ARing<BLOCK_K, STAGES>,
    b_ring: BRing<HALF_N, BLOCK_K, STAGES>,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    tmem_slot: *mut u32,
    queue: ClcQueue,
    /// The cursor past the queue — [`SharedPlan::bytes`] of it is this rung's
    /// [`shared_plan`], and it is where [`staged`] picks up.
    plan: SharedPlan,
}

/// The rung's plan, as one walk.
///
/// Run against [`SharedPlan::attach`] it is the handles; run against
/// [`SharedPlan::sizing`] it is the envelope, which is what `attach`'s one
/// remaining `const` assert compares to [`shared_plan`].
#[inline(always)]
const fn shared<const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize>(
    at: SharedPlan,
) -> Shared<HALF_N, BLOCK_K, STAGES> {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (done, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    let (queue, at) = at.clc_queue();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        done,
        tmem_slot,
        queue,
        plan: at,
    }
}

/// The staging run on the end of it — 128-byte aligned, which is the
/// `next_multiple_of(128)` [`staged_plan`] used to carry beside a
/// `STAGE_OFFSET` constant.
#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, WARPS>()
}

/// Dynamic shared memory a `[2·BLOCK_M, block_n]` pair tile `stages` deep asks
/// for: the two operand rings and the scratch tail.
///
/// **The one place in this file the plan is still written twice, and the reason
/// is a language limit rather than a design one.** [`shared`] is the walk, and
/// it is generic over `HALF_N`, `BLOCK_K` and `STAGES` because that is what a
/// `SharedTileRing` needs to know its own `BYTES`. A const parameter cannot be
/// a function argument, so this — which answers for every rung of the sweep at
/// runtime, including the ones [`UNBUILT`] names and no kernel instantiates —
/// cannot call it. [`SharedPlan::reserve`] is what it spells instead: the same
/// cursor, the same alignment rules, the sizes in bytes. [`attach`] asserts the
/// two agree, per rung, at codegen.
///
/// Returns the cursor rather than the total so [`staged_plan`] can continue it
/// through the one walk both forms *do* share, [`staged`], whose type
/// parameters are all module constants.
const fn shared_cursor(block_n: usize, block_k: usize, stages: usize) -> SharedPlan {
    let at = SharedPlan::sizing();
    let (_, at) = at.reserve(BLOCK_M * block_k * 2 * stages, SharedPlan::TILE_ALIGN);
    let (_, at) = at.reserve((block_n / 2) * block_k * 2 * stages, SharedPlan::TILE_ALIGN);
    let (_, at) = at.barriers(2 * stages + 1);
    let (_, at) = at.tmem_slot();
    let (_, at) = at.clc_queue();
    at
}

/// [`shared_cursor`]'s total — see there for why it is not [`shared`]'s.
pub const fn shared_plan(block_n: usize, block_k: usize, stages: usize) -> usize {
    shared_cursor(block_n, block_k, stages).bytes()
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

/// Dynamic shared memory a **register-drain** launch must provide — `lcf`,
/// `lcsf`, every [`Ablation`] rung and every rung of #87's tile sweep.
///
/// **Not the shipped envelope since #119.** [`SHIPPED_EPILOGUE`] is `staged84`
/// and declares [`STAGED_SHARED_BYTES`]; this is the plan the staged one is
/// laid on top of, byte for byte, which is what keeps the two arms of every
/// epilogue A/B differing in the drain and in 16 424 declared bytes and in
/// nothing else.
///
/// Every scheduler below launches with the *same* plan, including the static
/// one that never touches the queue. Twenty-four bytes is not worth a second
/// envelope, and paying them on both sides is what keeps the A/B a comparison
/// of schedules rather than of residencies — 73 816 B still admits the three
/// CTAs per SM that #84 counted at 73 792.
pub const SHARED_BYTES: usize = shared_plan(BLOCK_N, BLOCK_K, STAGES);

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

/// Dynamic shared memory a **staged** epilogue's launch declares: the shipped
/// plan, plus one [`StageTile`] per warp.
///
/// The 128-byte alignment is [`SharedPlan::tile_ring`]'s and used to be a
/// `next_multiple_of(128)` written here. It is 40 bytes at the shipped rung and
/// is not free-floating: [`SHARED_BYTES`] ends on a 24-byte [`ClcQueue`], so
/// the staging run would otherwise start at 98 392, which is not a swizzle
/// atom's alignment and would put the phase
/// [`kittens::shared::SwizzledChunks`] derives from the base somewhere the
/// `stmatrix` and the read-back would still agree on but no reader could check.
pub const fn staged_plan(block_n: usize, block_k: usize, stages: usize) -> usize {
    staged(shared_cursor(block_n, block_k, stages)).1.bytes()
}

/// CTAs of a staged launch an SM holds — [`Rung::ctas_per_sm`]'s
/// `min(512 / columns, shared per SM / plan)` at [`STAGED_SHARED_BYTES`].
///
/// It is 2, the same integer the register-drain envelope gets, because the
/// tensor-memory term binds at [`BLOCK_N`] columns and the staging tiles come
/// out of slack the shared term still had. `device-tests`' `tmem residency
/// census` counts the same 2 at this envelope, which is what makes the A/B an
/// epilogue comparison rather than a residency one — and, since #119, what
/// says [`CTAS_PER_SM`] still describes the shipped launch after the default
/// moved onto this plan.
fn staged_ctas_per_sm(shared_per_sm: usize) -> u32 {
    (512 / BLOCK_N).min(shared_per_sm / STAGED_SHARED_BYTES) as u32
}

/// The staged epilogue's envelope, the second literal `#[launch_contract]`
/// needs — and since #119 **the envelope the shipped launch declares**, since
/// [`SHIPPED_EPILOGUE`] is one of the four rungs that carry staging tiles.
///
/// All four declare it: #117's two instruction widths change what the epilogue
/// issues and not what it occupies, so `staged`, `staged8`, `staged4` and
/// `staged84` share one number and one `no drain` control.
///
/// **It must stay at or under 116 736 B**, which is the 233 472 an SM has
/// divided by the 2 CTAs [`CTAS_PER_SM`] counts — and the residency itself does
/// not move, because at [`BLOCK_N`] `= 256` the binding term of
/// `min(512 / columns, shared per SM / plan)` is the tensor-memory one and
/// stays 2 until shared memory passes that line. 1920 B of headroom is left.
pub const STAGED_SHARED_BYTES: usize = staged_plan(BLOCK_N, BLOCK_K, STAGES);
const _: () = assert!(STAGED_SHARED_BYTES == 114_816 && STAGED_SHARED_BYTES <= 116_736);

/// The three readings of the staging run agree, which is what makes the TMA
/// rungs an epilogue A/B and not an envelope one: four per-warp [`StageTile`]s,
/// four [`StageRing`] buffers and one [`CtaRing`] buffer are the same
/// 16 384 bytes at the same offset, and every rung that carries any of them
/// declares [`STAGED_SHARED_BYTES`].
///
/// **A second staging buffer does not fit and this is where that is stated.**
/// The envelope leaves 1920 B under the 116 736 a CTA gets at two per SM;
/// another depth is 16 384. So both TMA rungs are depth 1 by arithmetic rather
/// than by preference.
const _: () = {
    let run = WARPS * StageTile::BYTES;
    assert!(run == 16_384);
    assert!(WARPS * StageRing::BYTES == run);
    assert!(CtaRing::BYTES == run);
    assert!(STAGED_SHARED_BYTES + run > 116_736);
};

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
/// That extrapolation is false, and the cap sweep in `experiments/README.md` §7 is
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
/// by the shared half. The census counts 2 at the 98392 B plan, agreeing —
/// **and the same 2 at the 114 816 B plan [`SHIPPED_EPILOGUE`] declares**,
/// which is why moving the default onto the staged epilogue in #119 left this
/// constant, [`MAX_CLUSTERS`] and the grid exactly where they were.
///
/// The step down from 3 was paid for and the sweep is what says at what price.
/// #98 priced a 3 → 2 step at **13.6–16.1%** on bytes no code touched; here the
/// bytes are a third of a wider pipeline and a wider tile, and the net is
/// **+11.6% at 8192³ and +21.6% at 16384³**. `experiments/README.md` §7 separates
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
/// sweeps `1, 2, 4, 8, 16, 32` and `experiments/README.md` §7 keeps every row
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
/// A rung is a pair tile, a stage's K and a pipeline depth, and all three are
/// const parameters of [`Tile`] — but `#[launch_contract]` takes a literal
/// shared plan, so each combination is its own `#[kernel]` and this is the
/// host's name for it. Every sweep entry is static-only;
/// [`Scheduler::Stealing`] exists on the shipped rung alone, because a
/// scheduler comparison at a moving tile would be two variables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Entry {
    /// `gemm_cg2` / `gemm_cg2_clc` — `[256, 256] @ BLOCK_K = 64, STAGES = 3`
    /// since #87, the kernel this file ships and the only rung with a stealing
    /// twin.
    Shipped,
    N128S2,
    /// The kernel this file shipped through #102, kept as the control.
    N128S3,
    N128S4,
    N256S2,
    /// `[256, 256] @ BLOCK_K = 64, STAGES = 1` — the shallow end of the depth
    /// family, and the rung the `BLOCK_K` rung is read against at fixed depth.
    N256K64S1,
    /// `[256, 256] @ BLOCK_K = 128, STAGES = 1` — the same bytes in flight as
    /// `N256S2`, factorized the other way.
    N256K128S1,
    /// A rung the table computes and no kernel implements — see [`UNBUILT`].
    /// Launching it is an error rather than a missing arm, because the reason
    /// it is not built is a measurement and not an oversight.
    Unbuilt,
}

/// One point of the tile, stage-K and depth sweep.
///
/// The three numbers that move are the pair tile's columns, the K a stage
/// carries and the pipeline's depth over K. Everything else about a rung — its
/// shared plan, its tensor memory, the residency those admit, its arithmetic
/// intensity, the K blocks an item walks and how many output tiles a problem
/// has — is arithmetic on them, which is the whole reason this is a table and
/// not eight kernels written out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rung {
    /// Columns of `C` the *pair* computes, and this CTA's accumulator columns.
    pub block_n: usize,
    /// K one pipeline stage carries — a whole number of [`ATOM_K`] atoms.
    pub block_k: usize,
    /// Pipeline depth over K.
    pub stages: usize,
    pub entry: Entry,
}

impl Rung {
    /// Dynamic shared memory this rung's launch declares.
    pub const fn shared(self) -> usize {
        shared_plan(self.block_n, self.block_k, self.stages)
    }

    /// K in flight in one CTA's rings — `block_k * stages`, and the quantity
    /// the shared budget actually caps.
    ///
    /// It is worth a name because it is what makes this sweep a
    /// *factorization* rather than two free axes: two rungs with the same value
    /// here declare the same shared plan to within the scratch tail and get the
    /// same residency, and differ only in how that K is divided into barriers.
    pub fn k_in_flight(self) -> usize {
        self.block_k * self.stages
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
        format!("[256,{}] k{} s{}", self.block_n, self.block_k, self.stages)
    }
}

/// The kernel this file ships, as a rung: `[256, 256] @ BLOCK_K = 64,
/// STAGES = 3` (#87).
pub const SHIPPED: Rung = Rung {
    block_n: BLOCK_N,
    block_k: BLOCK_K,
    stages: STAGES,
    entry: Entry::Shipped,
};

/// The epilogue this file ships — `staged84`, both of #117's instruction
/// widths on #116's staged drain, and what [`bench`] launches.
///
/// **It was [`Epilogue::Fused`] through #119**, on the precedent that a rung
/// beside the shipped kernel keeps every A/B in `experiments/README.md` §7
/// quotable against one launch. The evidence is three sections deep and points
/// one way: staging the drain is +8.0% / +4.2% / +1.9% at 4096³ / 8192³ /
/// 16384³ (#116) and the two widths compose on top of that for
/// **+23.1% / +8.8% / +5.1%** (#117). Measured default against default in one
/// container, with cuBLASLt re-measured in it, that is
/// **+37.2% / +13.5% / +10.0%** and **0.583 → 0.801, 0.827 → 0.939 and
/// 0.871 → 0.958 of cuBLASLt**. Every rung passed the same element-by-element
/// `==` on bf16 words at both check sizes and all three traversal widths.
///
/// It carries [`STAGED_SHARED_BYTES`] rather than [`SHARED_BYTES`] and the
/// residency does not move with it — `min(512 / columns, shared per SM /
/// plan)` binds on the tensor-memory term at [`BLOCK_N`] columns, so
/// [`CTAS_PER_SM`] is 2 at both envelopes.
///
/// **`gemm_ws` ships a different rung on purpose** — `staged8`, without
/// `.x4`. There `.x4` recovers 2 registers where it recovers 14 here, so the
/// composition gain that makes this the winner does not exist on that design
/// point; see [`crate::gemm_ws::SHIPPED_ENTRY`].
pub const SHIPPED_EPILOGUE: Epilogue = Epilogue::StagedWideX4;

/// #87's sweep, and the losers stay in it.
///
/// Five of the issue's six rungs are here. The sixth, `[256, 256] @ STAGES =
/// 4`, is [`UNBUILT`]: it is one CTA an SM, and #98 measured that step at a
/// further 25–44% under a step already worth 14–16%, so it is computed and not
/// launched. Booking a B200 on a rung whose answer two prior measurements
/// already give is how a sweep spends money to confirm itself.
pub const RUNGS: [Rung; 7] = [
    Rung {
        block_n: 128,
        block_k: 64,
        stages: 2,
        entry: Entry::N128S2,
    },
    CONTROL,
    Rung {
        block_n: 128,
        block_k: 64,
        stages: 4,
        entry: Entry::N128S4,
    },
    K64S1,
    K128S1,
    Rung {
        block_n: 256,
        block_k: 64,
        stages: 2,
        entry: Entry::N256S2,
    },
    SHIPPED,
];

/// The kernel this file shipped through #102 — `[256, 128] @ STAGES = 3`, and
/// the control every #87 row is read against.
pub const CONTROL: Rung = Rung {
    block_n: 128,
    block_k: 64,
    stages: 3,
    entry: Entry::N128S3,
};

/// One stage of one swizzle atom — the shallow end of the depth family, and
/// the rung [`K128S1`] is read against at fixed depth.
///
/// At `STAGES = 1` there is no pipeline at all: the producer's `wait_recycled`
/// cannot clear until the MMA reading the one stage has committed, so loads and
/// arithmetic serialize. It is in the sweep because the `BLOCK_K` comparison
/// has to be taken at fixed depth, and one stage is the only depth at which
/// both 64 and 128 fit under two CTAs an SM.
pub const K64S1: Rung = Rung {
    block_n: 256,
    block_k: 64,
    stages: 1,
    entry: Entry::N256K64S1,
};

/// Two atoms in one stage — **the `BLOCK_K` rung**, and it is worth a B200
/// because it pairs two ways.
///
/// Against [`K64S1`] it is the same depth carrying twice the K per stage: half
/// the stage barriers, half the `expect_tx` charges and half the loop
/// iterations, for the same bytes, the same TMA instruction count and the same
/// chunk count. Against `[256, 256] @ STAGES = 2` it is the same **65 KiB of
/// shared memory and the same 128 K in flight**, factorized the other way — one
/// barrier over two atoms against two barriers over one atom each. That second
/// pair is the control that says whether a K block's fixed cost or the
/// pipelining is what the shared budget is better spent on.
pub const K128S1: Rung = Rung {
    block_n: 256,
    block_k: 128,
    stages: 1,
    entry: Entry::N256K128S1,
};

/// The rungs this sweep computes and does not build — see [`RUNGS`].
///
/// Both are one CTA an SM, and #98 measured that step at a further 25–44% under
/// a step already worth 14–16%. `[256, 256] @ k128 s2` is the one a reader asks
/// for first — it is the shipped kernel's 192 K in flight arriving as two
/// barriers instead of three — and it declares 131 144 B against the 116 736 B
/// two CTAs an SM leaves. **That is the whole constraint on `BLOCK_K` in one
/// line:** at this pair tile the shared budget is already spent, so a wider
/// stage is only reachable by giving depth back.
pub const UNBUILT: [Rung; 2] = [
    Rung {
        block_n: 256,
        block_k: 64,
        stages: 4,
        entry: Entry::Unbuilt,
    },
    Rung {
        block_n: 256,
        block_k: 128,
        stages: 2,
        entry: Entry::Unbuilt,
    },
];

/// The shared plans the four sweep entry points declare, against the
/// arithmetic every host table reads.
///
/// `#[launch_contract]` takes a literal, so each rung's plan is written twice —
/// once there and once as [`shared_plan`]. `attach` asserts the arithmetic
/// against the *rings'* own byte counts at codegen; this asserts it against the
/// literals, which is the other half of the same join.
const _: () = {
    assert!(shared_plan(128, 64, 2) == 49_224);
    assert!(shared_plan(128, 64, 4) == 98_408);
    assert!(shared_plan(256, 64, 1) == 32_824);
    assert!(shared_plan(256, 64, 2) == 65_608);
    assert!(shared_plan(256, 64, 3) == 98_392);
    assert!(shared_plan(256, 128, 1) == 65_592);
    // The two the sweep computes and does not build, both one CTA an SM.
    assert!(shared_plan(256, 64, 4) == 131_176);
    assert!(shared_plan(256, 128, 2) == 131_144);
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
struct Tile<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> {
    a_ring: ARing<BLOCK_K, STAGES>,
    b_ring: BRing<HALF_N, BLOCK_K, STAGES>,
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
    /// of the same output through every item it runs. The cursor's element is
    /// the output's, so the fp32 → bf16 rounding is the store instruction and
    /// nothing else (#108).
    c: GlobalRows<Bf16>,
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
impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize>
    Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>
{
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
    /// **`LOADS` is [`Ablation`]'s operand-traffic switch and is `true` in
    /// every kernel this file ships.** At `false` the stage barrier is
    /// completed by one bare arrival instead of by the bytes of four TMA
    /// loads, so the loop, the recycling handshake, the barrier and the
    /// consumer's wait on it are all exactly what they were and the only thing
    /// gone is the traffic. It is a const parameter rather than a second loop
    /// because a second loop is a place for the two to drift, and the shipped
    /// kernel's codegen is what `regcount` checks did not move.
    ///
    /// # Safety
    ///
    /// One thread of the CTA, with `from..to` inside `0..k_blocks` and every
    /// block issued exactly once across the calls that cover the walk.
    #[inline(always)]
    unsafe fn produce<const LOADS: bool>(&self, tile_m: u32, tile_n: u32, from: u32, to: u32) {
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
                if LOADS {
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
                } else if self.rank == LEADER {
                    // The one arrival the barrier was init'd for, with no
                    // transaction bytes attached — so the stage completes on
                    // the same barrier at the same point in the loop, having
                    // moved nothing.
                    self.load.sem(k).arrive();
                }
                k += 1;
            }
        }
    }

    /// Chain the whole K walk into the pair's accumulator and publish it.
    ///
    /// **`MMA` is [`Ablation`]'s arithmetic switch and is `true` in every
    /// kernel this file ships.** At `false` the `tcgen05.mma` chain is gone and
    /// everything around it stays: the same loop, the same wait on each stage,
    /// the same `tcgen05.commit` releasing the operands and the same commit
    /// publishing the accumulator. A `commit` with no outstanding MMA to cover
    /// completes at once, which is what makes the rung run rather than hang —
    /// and what makes it price the multiply and not the protocol.
    ///
    /// # Safety
    ///
    /// One thread of the leader rank, with the accumulator's previous contents
    /// already read — every chunk of every stage chains into the one
    /// accumulator, so only the very first instruction of the very first stage
    /// starts it fresh, and "first" is per *item*.
    #[inline(always)]
    unsafe fn multiply<const MMA: bool>(&self) {
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
                if MMA {
                    // One walk per swizzle atom of the stage: `k_walk` describes
                    // exactly one, so a `BLOCK_K` wider than `ATOM_K` is that
                    // many walks over the stacked subtiles the tile is already
                    // stored as — the same bytes, the same chunk count, one
                    // barrier instead of several. At `BLOCK_K = ATOM_K` this
                    // loop is one iteration and folds away, which is what keeps
                    // the shipped kernel's codegen where `regcount` found it.
                    let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                    let mut atom = 0usize;
                    while atom < ATile::<BLOCK_K>::SUBTILES {
                        mma_walk_cg2::<Bf16, ATOM_CHUNKS>(
                            self.accumulator.raw(),
                            Atom::<BLOCK_M>::from_raw(a.subtile(atom)).k_walk(),
                            Atom::<HALF_N>::from_raw(b.subtile(atom)).k_walk(),
                            shape,
                            k > 0 || atom > 0,
                        );
                        atom += 1;
                    }
                }
                // The MMA releases its own operands, in both CTAs: a thread
                // arriving here would only prove the instruction was *issued*.
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// The epilogue, straight out of registers (#11) — this warp's band of the
    /// accumulator, LDTM'd and stored to the tile `item` names.
    ///
    /// `ldc` is the destination's leading dimension and `C` is wider than this
    /// tile's columns, so the cursor carries the stride and each band lands at
    /// its own `(row, column)` origin — no shared staging tile and no
    /// descriptor.
    ///
    /// **The band is fp32 and `C` is bf16** (#108), so the one rounding in this
    /// kernel is the `cvt.rn.bf16x2.f32` inside [`kittens::global::store_rows`]'
    /// pair store. The accumulator does not move: `bf16` in, fp32 accumulate,
    /// `bf16` out is the signature a training GEMM has, and it is the *output*
    /// that was the unusual choice here rather than the accumulator. What it
    /// costs in instructions is one `cvt` per pair and no fewer stores — a pair
    /// of adjacent columns is 4 bytes where it was 8, and `CONTIGUOUS_VALUES`
    /// is 2 either way, so the store count is unchanged and only the width
    /// halves.
    ///
    /// A band at a time rather than the whole accumulator at once, and
    /// [`DRAIN_N`] is why: 256 columns in one `RegTile` is 256 fp32 a thread.
    /// At `BLOCK_N = 128` this is the single band (#22) it has always been, and
    /// the loop folds away.
    ///
    /// # `WIDE` is #117's LDTM width on *this* path, and it is a control
    ///
    /// [`Self::drain_staged`] changed two things at once against this function
    /// — the store's shape and, later, the load's instruction width — and #116
    /// priced the first of them with the second still at `.x1`. `WIDE` puts
    /// `tcgen05.ld.16x256b.x8` on the register drain, so the shared round trip
    /// can be A/B'd against *no* round trip at the same LDTM width and the two
    /// levers stop being confounded. Every byte written to `C` is identical:
    /// `TmemTile::tile_x8` and `TmemTile::tile` return the same band (that is
    /// `device-tests`' `ldtm x8 map`), so this rung is on the correctness gate
    /// like the register drain it is a width of.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA, with the accumulator complete and fenced, and
    /// nothing that will overwrite it in flight.
    #[inline(always)]
    unsafe fn drain<const WIDE: bool>(&self, item: u32) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                // This warp's 32 TMEM lanes by `DRAIN_N` columns of the
                // accumulator, composed out of the `[16, 16]` blocks LDTM
                // delivers.
                let band: Band = if WIDE {
                    self.accumulator.tile_x8(32 * self.warp_id, column)
                } else {
                    self.accumulator.tile(32 * self.warp_id, column)
                };
                store_rows(self.c, row_base, column_base + column, self.lane, band);
                column += DRAIN_N as u32;
            }
        }
    }

    /// [`Self::drain`] **staged through shared memory**: TMEM → registers →
    /// `stmatrix` into this warp's own [`StageTile`] → plain 16-byte stores out
    /// to `C`. #15's route, minus the engine.
    ///
    /// # What it is trying to buy, counted per warp band of `[32, 256]`
    ///
    /// | | instructions a thread | what one touches |
    /// |---|---|---|
    /// | [`Self::drain`] | **128** `st.global.b32` | 4 B a thread, on **8 discontiguous 16 B runs** — [`BaseLdtm`] spreads a warp over 8 rows × 4 column-quads |
    /// | staged, `stmatrix` | 64 `stmatrix.m8n8.x2` | 256 B a warp |
    /// | staged, `ld.shared` | 32 `ld.shared.v4.b32` | 16 B a thread |
    /// | staged, `st.global` | **32** `st.global.v4.b32` | 16 B a thread, on **4 contiguous 128 B runs** |
    ///
    /// **Total memory issue does not fall — it is 128 either way.** The version
    /// of this table #15 was scoped from counted 96, by leaving out the
    /// `ld.shared` half of [`kittens::global::store_shared_rows`]' chunk copy;
    /// that mover is deliberately a load and a store rather than one opaque
    /// snippet, and both are instructions. So the entire hypothesis is the
    /// **shape** of the global write and not its count: 4× fewer store
    /// instructions, and each one landing on full 128-byte lines instead of on
    /// eight half-filled 32-byte sectors. Counting sectors rather than
    /// instructions, the band's global writes go from 1024 half-full to 512
    /// full.
    ///
    /// It is strictly more total work — a whole extra pass over the data
    /// through shared memory — so it wins only if that recoalescing is worth
    /// more than 64 `stmatrix` plus 32 `ld.shared`. `experiments/README.md` §7 is
    /// where that is measured rather than argued.
    ///
    /// The `cvt.rn.bf16x2.f32` count is unchanged at 128: [`kittens::shared::Element::pack`] is
    /// called twice per `stmatrix` where `Element::write_pair` was called once
    /// per pair store.
    ///
    /// # The LDTM half was untouched, was 54% of the cost, and `WIDE` is it
    ///
    /// #109 split the epilogue into 13.1 µs of stores and 8.3 µs of LDTM at
    /// 8192³. Every column above is the store half. What changed on the load
    /// side was only the *band*: [`StagedBand`] is 64 fp32 a thread where
    /// [`Band`] is 128, because the pass is as wide as the staging tile — the
    /// same TMEM reads in twice as many passes, and `regcount` is what says
    /// where peak liveness went.
    ///
    /// `WIDE` is what removes it (#117), and the measurement closed #109's
    /// split from the other end: at 8192³ this epilogue costs 14.96 µs a tile
    /// and `WIDE` takes it to 6.89, so **8.07 µs of it was LDTM** against
    /// #109's 8.3 — two instruments, two containers, four issues apart. The
    /// mechanism is the *wait* rather than the issue, which is why `X4` halves
    /// an instruction count and is worth −0.6% to −1.1% on its own.
    ///
    /// # There is no proxy fence here and its absence is not a bug
    ///
    /// `fence.proxy.async.shared::cta` orders a generic-proxy write against an
    /// *async*-proxy read, which is what the TMA path in
    /// [`kittens::epilogue::StoreRing`] needs. Both ends of this one are
    /// generic — `stmatrix` writes and `ld.shared` reads — so nothing but
    /// convergence stands between them, and `stmatrix.sync.aligned` is
    /// convergence. The one hazard left is the *next* pass overwriting a tile
    /// this pass is still reading, which is why the `bar.warp.sync` below is
    /// there and why it is a warp barrier and not a block one.
    ///
    /// # `WIDE` and `X4` are the two instruction widths, one per half (#117)
    ///
    /// Both default off, and both leave every byte of this loop where it was —
    /// same bands, same staging tile, same 114 816 B envelope, same order of
    /// writes to `C`. What changes is how many instructions carry them.
    ///
    /// **`WIDE` is the LDTM half.** [`TmemTile::tile`] issues
    /// `tcgen05.ld.16x256b.x1` twice per `[16, 16]` block and waits after each
    /// one, because the registers it waits on *are* the load's return value —
    /// so a `[32, 64]` band costs 16 loads and 16 fully exposed TMEM
    /// latencies. [`TmemTile::tile_x8`] is `.x8`: 64 columns and 32 f32 a
    /// thread per issue, so the same band is **2 loads and 2 waits**. The
    /// waits are the larger half of that and the reason this is worth a rung
    /// at all.
    ///
    /// **`X4` is the `stmatrix` half.** A [`kittens::reg::Fragment`] is four
    /// `8x8` b16 matrices and `stmatrix.m8n8.x2` names two, so
    /// [`kittens::ldst::store_tile`] issues two per block;
    /// [`kittens::ldst::store_tile_x4`] names all four in one. Half the
    /// `stmatrix`, identical addresses.
    ///
    /// Neither touches the global half, which is the half #116 already moved:
    /// `store_shared_rows` issues the same 32 × 16 B stores on the same four
    /// contiguous 128 B runs whatever these two are set to. That is what makes
    /// this an ablation of the *other* half rather than a second pass at the
    /// same one.
    ///
    /// # Safety
    ///
    /// As [`Self::drain`], plus: `stage` must be 4096 B of shared memory no
    /// other warp writes.
    #[inline(always)]
    unsafe fn drain_staged<const WIDE: bool, const X4: bool>(&self, item: u32, stage: StageTile) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let chunks = stage.chunk_writer();
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band: StagedBand = if WIDE {
                    self.accumulator.tile_x8(32 * self.warp_id, column)
                } else {
                    self.accumulator.tile(32 * self.warp_id, column)
                };
                if X4 {
                    store_tile_x4(chunks, 0, 0, self.lane, band);
                } else {
                    store_tile(chunks, 0, 0, self.lane, band);
                }
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    row_base,
                    column_base + column,
                    self.lane,
                    stage,
                );
                // The write-after-read the loop owes itself, at warp scope
                // because the tile is this warp's alone. `bar.warp.sync` orders
                // memory among the lanes it synchronizes, so the next pass's
                // `stmatrix` cannot overtake a lane still reading this one's
                // chunks.
                warp::sync_mask(u32::MAX);
                column += STAGE_N as u32;
            }
        }
    }

    /// [`Self::drain_staged`] with a **second pass** appended to each band, and
    /// `STAGE` naming how much of the pass is repeated — the instrument that
    /// splits the staged epilogue into the three things it is made of.
    ///
    /// # Why doubling rather than deleting
    ///
    /// The staged epilogue is a chain: LDTM into registers, `stmatrix` into the
    /// staging tile, then `ld.shared` + `st.global` out of it. Every ablation
    /// that *removes* a link removes the ones above it too — nothing consumes
    /// the band, so a dead-code pass takes the LDTM with the `stmatrix` — which
    /// is the hazard [`Epilogue::DoubleDrain`]'s own doc states and the reason
    /// #108 built a doubling probe instead of a subtractive one. Doubling
    /// deletes nothing, so what a rung measures is what it names whatever the
    /// compiler believes, and `regcount`'s opcode census is what says the extra
    /// pass survived.
    ///
    /// | `STAGE` | the second pass runs | the difference from the rung below |
    /// | ---: | --- | --- |
    /// | 1 | `store_shared_rows` | `ld.shared` + `st.global.v4` — the global half |
    /// | 2 | `stmatrix` + `store_shared_rows` | the `cvt` + `stmatrix` pass and its `bar.warp.sync` |
    /// | 3 | LDTM + `stmatrix` + `store_shared_rows` | the LDTM half: the issue *and* the wait |
    ///
    /// So `STAGE = 3` minus [`Epilogue::StagedWideX4`] is one whole staged
    /// epilogue measured **serially**, in the way #108's `2x` measured the
    /// register one, and the three differences add up to it by construction.
    ///
    /// # `again` is the cluster's own first tile
    ///
    /// The second pass's global stores go there rather than to the item's, for
    /// [`DoubleDrain::home`]'s reason: the extra bytes have to stay in L2 or
    /// this would price a doubled HBM stream on top of a doubled instruction
    /// count. The `ld.shared` half is unaffected either way — it reads the same
    /// staging tile.
    ///
    /// **It computes a wrong `C` and is never checked**, as every doubling
    /// probe here is.
    ///
    /// # The syncs, which are part of what is being counted
    ///
    /// At `STAGE = 1` the second pass only *reads* the staging tile, so the
    /// pass keeps the one `bar.warp.sync` [`Self::drain_staged`] has: two reads
    /// with no write between them need nothing. At `STAGE >= 2` the second pass
    /// writes the tile again, so it owes the read-before-write barrier as well,
    /// and that second `bar.warp.sync` is charged to the `stmatrix` half — it
    /// is what a second `stmatrix` pass costs, not an artifact beside it.
    ///
    /// # Safety
    ///
    /// As [`Self::drain_staged`], for both `item` and `again`.
    #[inline(always)]
    unsafe fn drain_staged_twice<const WIDE: bool, const X4: bool, const STAGE: u32>(
        &self,
        item: u32,
        again: u32,
        stage: StageTile,
    ) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let (again_row, again_column) = self.origin(again);
            let chunks = stage.chunk_writer();
            let load = |column: u32| -> StagedBand {
                if WIDE {
                    self.accumulator.tile_x8(32 * self.warp_id, column)
                } else {
                    self.accumulator.tile(32 * self.warp_id, column)
                }
            };
            let write = |band: StagedBand| {
                if X4 {
                    store_tile_x4(chunks, 0, 0, self.lane, band);
                } else {
                    store_tile(chunks, 0, 0, self.lane, band);
                }
            };
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band = load(column);
                write(band);
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    row_base,
                    column_base + column,
                    self.lane,
                    stage,
                );
                if STAGE >= 2 {
                    warp::sync_mask(u32::MAX);
                    write(if STAGE >= 3 { load(column) } else { band });
                }
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    again_row,
                    again_column + column,
                    self.lane,
                    stage,
                );
                warp::sync_mask(u32::MAX);
                column += STAGE_N as u32;
            }
        }
    }

    /// [`Self::drain_staged`] with the **shared→global hop handed to the TMA
    /// engine**: the same `.x8` LDTM, the same `.x4` `stmatrix` into the same
    /// per-warp [`StageTile`], and `cp.async.bulk.tensor.2d.global.shared::cta`
    /// where `ld.shared` + `st.global.v4` were. #15's route, finally with the
    /// engine in it.
    ///
    /// # What it replaces, per warp band of `[32, 64]`
    ///
    /// | | instructions a thread |
    /// |---|---|
    /// | `store_shared_rows` | 32 `ld.shared.v4.b32` + 32 `st.global.v4.b32` |
    /// | this | **one** `cp.async.bulk.tensor`, on lane 0 |
    ///
    /// Plus, on every lane, one `fence.proxy.async.shared::cta`; plus, on lane
    /// 0, one `cp.async.bulk.commit_group` and one
    /// `cp.async.bulk.wait_group.read`; plus two `bar.warp.sync` where the plain
    /// path had one. So the trade is 64 memory instructions a thread for a
    /// fence a thread and a barrier, and the whole claim is issue.
    ///
    /// **The bandwidth claim is dead and this rung does not make it.** #121's
    /// `s84 hot` deleted the epilogue's entire HBM traffic and moved throughput
    /// by −2.3% to +1.8%, so there is no memory pressure for an engine to
    /// relieve. What #15 said this would buy is not what it can buy.
    ///
    /// # And the issue claim came out below zero
    ///
    /// **This rung is 1.0–1.7% slower than `staged84`**, in four paired cells
    /// across two containers at the shortened geometries #122 built, none of
    /// whose ranges reaches 1.000. The full-depth 8192³ A/B cannot see it —
    /// `1/share` of 458–709 there, so three containers of it read `+0.2`,
    /// `+0.4` and `−0.2` percent and mean nothing.
    ///
    /// The doubling probe never promised the ~+2.7% either: `ld.shared` +
    /// `st.global` is 2.33–2.58 µs a tile and this hop is 1.57–1.99, so the
    /// most the change was ever worth is 0.42–0.75 µs — 2.7% was that term
    /// going to *zero*, and the engine takes it to about three quarters of
    /// itself.
    ///
    /// **Why the realised value is below even that, and why the probe cannot
    /// see it.** A *doubled* store is an extra one left in flight beside work
    /// that continues; a *replacing* store is on the critical path of the next
    /// band. At [`StageRing`]'s `IN_FLIGHT = 0` the `acquire` below is
    /// `cp.async.bulk.wait_group.read` at zero groups, so band `k + 1`'s
    /// `stmatrix` waits out the engine's read of band `k`'s buffer — four times
    /// an item — where `store_shared_rows` retires into the memory pipeline and
    /// blocks on nothing. The kernel trades 64 pipelined instructions a thread
    /// for one instruction and one exposed engine latency, and the latency is
    /// dearer. That is a mechanism consistent with every number and not a
    /// measured one; [`StageRing`] carries what measuring it would take.
    ///
    /// # The proxy fence, which the plain path correctly does not have
    ///
    /// `stmatrix` writes through the generic proxy and the TMA engine reads
    /// through the async one, so nothing but `fence.proxy.async.shared::cta`
    /// orders them — and it orders the writes of the thread that *executes* it,
    /// which is why every lane fences and a barrier carries that to the lane
    /// that issues. [`Self::drain_staged`]'s doc says there is no fence there
    /// and that its absence is not a bug; both ends are generic there. This is
    /// the other case, and the fence is part of what is being measured.
    ///
    /// # Why the ring is warp-scope
    ///
    /// [`StageTile`] is one warp's 4096 B and nobody else's, so the convergence
    /// the fence needs is `bar.warp.sync` and the thread that owns the store
    /// group is lane 0. [`Self::drain_staged_tma_cta`] is the same epilogue at
    /// CTA scope through the ring exactly as #111 built it, and the difference
    /// between the two is what a block barrier costs here.
    ///
    /// # `TWICE` is the doubling instrument, and it doubles the whole hop
    ///
    /// A second `acquire`-less `commit_2d` per band, aimed at the cluster's own
    /// first tile so the extra bytes stay in L2. It adds a link rather than
    /// deleting one, for [`Tile::drain_staged_twice`]'s reason, and what it
    /// adds is a fence, a barrier, an issue and a commit — the *whole* of the
    /// TMA global half, which is what makes its difference comparable with the
    /// 2.0–2.3 µs a tile `s84 2g` prices `ld.shared` + `st.global` at.
    ///
    /// At `TWICE` it computes a wrong `C` and is never checked.
    ///
    /// # Safety
    ///
    /// As [`Self::drain_staged`], plus [`kittens::epilogue::StoreRing`]'s: every
    /// lane of the warp calls this together, `ring` is this warp's alone, and
    /// the ring is drained before the kernel ends — nothing else makes a bulk
    /// store's bytes visible. `c_map` must describe `C` with a `[32, 64]` box.
    #[inline(always)]
    unsafe fn drain_staged_tma<const TWICE: bool>(
        &self,
        item: u32,
        again: u32,
        ring: &mut StageRing,
        c_map: *const TmaDescriptor,
    ) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let (again_row, again_column) = self.origin(again);
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                // The engine must be done *reading* the buffer before this
                // band's `stmatrix` refills it. Before the first band the wait
                // is trivially satisfied, so there is no fill phase.
                let staging = ring.acquire();
                let band: StagedBand = self.accumulator.tile_x8(32 * self.warp_id, column);
                store_tile_x4(staging.chunk_writer(), 0, 0, self.lane, band);
                ring.commit_2d(c_map, (column_base + column) as i32, row_base as i32);
                if TWICE {
                    ring.commit_2d(c_map, (again_column + column) as i32, again_row as i32);
                }
                column += STAGE_N as u32;
            }
        }
    }

    /// [`Self::drain_staged_tma`] with the four warps' tiles read as **one
    /// [`CtaStage`]** and one `cp.async.bulk.tensor` a band, through
    /// [`kittens::epilogue::StoreRing`] at the CTA scope it was written for.
    ///
    /// Same bytes into `C`, same LDTM, same `stmatrix` addresses — warp `w`
    /// still writes rows `32w..32w + 32`, of a taller tile — and the same
    /// 16 384 B of staging. What moves is the box, `[128, 64]` in one
    /// instruction against `[32, 64]` in four, and what that box costs: the
    /// fence has to reach a thread in another warp, so `acquire` and `commit`
    /// are a `bar.sync` each and the item pays **eight** block barriers where
    /// the shipped epilogue pays none.
    ///
    /// That is the trade this rung exists to price, and it is the one #116
    /// decided the other way without an engine in the picture. It is also the
    /// answer to whether the ring was usable as built: it was, at this scope,
    /// and the epilogue wanted the other one.
    ///
    /// **The eight barriers are cheaper than the three boxes they save, and the
    /// hypothesis was backwards.** This arm *wins* against
    /// [`Self::drain_staged_tma`] by 0.6–1.6% of the launch in four paired
    /// cells across two containers, and it is the one that ties `staged84`
    /// rather than losing by 1.0–1.7%. The opcode census reads 5 `bar.sync` and
    /// 0 `bar.warp.sync` here against 2 and 3 there, so the barriers are real
    /// and visible; what they buy is one `[128, 64]` box a band instead of
    /// four `[32, 64]` ones, and the engine's per-instruction overhead is the
    /// term that decides this pair.
    ///
    /// # Safety
    ///
    /// As [`Self::drain_staged_tma`], at CTA scope: every thread of the block
    /// calls this together, and `c_map` must describe `C` with a `[128, 64]`
    /// box.
    #[inline(always)]
    unsafe fn drain_staged_tma_cta(
        &self,
        item: u32,
        ring: &mut CtaRing,
        c_map: *const TmaDescriptor,
    ) {
        unsafe {
            let (row_base, column_base) = self.cta_origin(item);
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let staging = ring.acquire();
                let band: StagedBand = self.accumulator.tile_x8(32 * self.warp_id, column);
                store_tile_x4(
                    staging.chunk_writer(),
                    32 * self.warp_id,
                    0,
                    self.lane,
                    band,
                );
                ring.commit_2d(c_map, (column_base + column) as i32, row_base as i32);
                column += STAGE_N as u32;
            }
        }
    }

    /// Where in `C` this **CTA's** tile of `item` starts — the item index's
    /// whole meaning to the epilogue, before the warp's own rows are added.
    ///
    /// Split from [`Self::origin`] because a CTA-wide staging tile is aimed by
    /// this and a per-warp one by that, and the four warps' bands are
    /// contiguous: warp `w` owns rows `32w..32w + 32` of the same 128 this
    /// names, which is what lets [`CtaStage`] and [`StageTile`] describe one
    /// run.
    #[inline(always)]
    fn cta_origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank,
            BLOCK_N as u32 * tile_n,
        )
    }

    /// Where in `C` this warp's band of `item` starts — split out so
    /// [`Self::drain`] and [`Self::drain_storing_twice`] cannot disagree about
    /// it.
    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (row, column) = self.cta_origin(item);
        (row + 32 * self.warp_id, column)
    }

    /// [`Self::drain`] with **one LDTM and two sets of stores** — the second
    /// probe of #108's pair, and the one that splits the epilogue.
    ///
    /// [`Epilogue::DoubleDrain`] runs the whole epilogue twice and prices it;
    /// this runs the load once and the stores twice, so the difference between
    /// the two probes is **the LDTM**, and what is left is what the store loop
    /// costs on its own. That is the number `stmatrix` is worth arguing about,
    /// because a `stmatrix` epilogue keeps the LDTM exactly as it is and halves
    /// the stores.
    ///
    /// The band is held across both stores, which is the one thing to watch: it
    /// was live across the first store already, so peak liveness does not move
    /// and `regcount` is what says so rather than this comment.
    ///
    /// # Safety
    ///
    /// As [`Self::drain`], for both `item` and `again`.
    #[inline(always)]
    unsafe fn drain_storing_twice(&self, item: u32, again: u32) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let (again_row, again_column) = self.origin(again);
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band: Band = self.accumulator.tile(32 * self.warp_id, column);
                store_rows(self.c, row_base, column_base + column, self.lane, band);
                store_rows(self.c, again_row, again_column + column, self.lane, band);
                column += DRAIN_N as u32;
            }
        }
    }
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> Job
    for Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>
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
                self.produce::<true>(tile_m, tile_n, 0, self.k_blocks);
            }
            if self.rank == LEADER && self.warp_id == 1 && self.lane == 0 {
                self.multiply::<true>();
            }
            self.done.wait(0);
            thread::sync_threads();
            self.drain::<false>(item);
        }
    }
}

/// [`Epilogue::HotStore`]'s job: [`Tile`] with every store aimed at the
/// cluster's own first tile, so the epilogue's bytes stay in L2.
///
/// **It computes a wrong `C` and is never checked.** See [`Epilogue::HotStore`]
/// for what it is for and why an exact version of it does not exist.
#[derive(Clone, Copy)]
struct HotStore<
    const BLOCK_N: usize,
    const HALF_N: usize,
    const BLOCK_K: usize,
    const STAGES: usize,
> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    /// The item this cluster was launched for, and the only tile of `C` it ever
    /// writes. Read once outside the loop: [`pipeline::run`]'s first item *is*
    /// `cluster_idx`, so this is a tile the cluster would have written anyway
    /// and the probe stays inside the buffer without a bounds check.
    home: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> Job
    for HotStore<BLOCK_N, HALF_N, BLOCK_K, STAGES>
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
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            // The one line that differs from `Tile::work`, and the whole probe.
            tile.drain::<false>(self.home);
        }
    }
}

/// [`Epilogue::DoubleDrain`]'s job: [`Tile`] with the epilogue run a **second
/// time**, aimed at the cluster's own first tile.
///
/// **It computes a wrong `C` and is never checked.** See
/// [`Epilogue::DoubleDrain`] for what the second drain is measuring and why the
/// second one goes to the home tile rather than to the item's.
#[derive(Clone, Copy)]
struct DoubleDrain<
    const BLOCK_N: usize,
    const HALF_N: usize,
    const BLOCK_K: usize,
    const STAGES: usize,
> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    /// As [`HotStore::home`], and here for a second reason: the extra epilogue's
    /// bytes have to stay in L2 or this probe would price a doubled HBM stream
    /// as well as a doubled instruction count.
    home: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> Job
    for DoubleDrain<BLOCK_N, HALF_N, BLOCK_K, STAGES>
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
    /// As [`Tile`]'s, and the extra drain's: both tiles are ones this cluster
    /// owns, so both are in bounds, and the accumulator is read twice rather
    /// than written — nothing between the two drains touches tensor memory.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            self.tile.work(item);
            // The whole probe. `item` and `home` are runtime values a compiler
            // cannot prove equal, so this is a second epilogue and not a dead
            // store the first one subsumes.
            self.tile.drain::<false>(self.home);
        }
    }
}

/// [`Epilogue::DoubleStore`]'s job: [`Tile`] with the epilogue's **stores**
/// doubled and its LDTM not.
///
/// **It computes a wrong `C` and is never checked.** Paired with
/// [`DoubleDrain`], whose extra pass includes the load: the difference between
/// the two is what the LDTM costs, and the remainder is the store loop.
#[derive(Clone, Copy)]
struct DoubleStore<
    const BLOCK_N: usize,
    const HALF_N: usize,
    const BLOCK_K: usize,
    const STAGES: usize,
> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    /// As [`DoubleDrain::home`], and for both of its reasons.
    home: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> Job
    for DoubleStore<BLOCK_N, HALF_N, BLOCK_K, STAGES>
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
    /// As [`Tile`]'s, and [`Tile::drain_storing_twice`]'s: both tiles are ones
    /// this cluster owns.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            tile.drain_storing_twice(item, self.home);
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
/// (Two of those three have since stopped existing as obstacles: #108 made `C`
/// bf16, so the staging tile and the tensor map are both the element the
/// library already has. What that changes is the *other* route's price, not
/// this one's, and it is §7's question rather than this type's.)
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
struct Lcsf<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    /// The item whose accumulator is still sitting in tensor memory, or
    /// [`Self::NONE`] before the first one and after the last.
    pending: u32,
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize>
    Lcsf<BLOCK_N, HALF_N, BLOCK_K, STAGES>
{
    /// No accumulator owed. `u32::MAX` is not a reachable item: the tile grid
    /// is `tiles_m * tiles_n` and both are `u32`, so a real item is strictly
    /// less than their product and cannot be this.
    const NONE: u32 = u32::MAX;

    #[inline(always)]
    fn new(tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>) -> Self {
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
                self.tile.drain::<false>(self.pending);
            }
        }
    }
}

impl<const BLOCK_N: usize, const HALF_N: usize, const BLOCK_K: usize, const STAGES: usize> Job
    for Lcsf<BLOCK_N, HALF_N, BLOCK_K, STAGES>
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
            let fill = Tile::<BLOCK_N, HALF_N, BLOCK_K, STAGES>::FILL.min(tile.k_blocks);

            // Fill the pipe first, so the loads are in flight across the store
            // below. The producer does not block in this prefix — a stage
            // barrier below `STAGES` has nothing to recycle — so it reaches the
            // drain rather than sitting in `free.wait_recycled`.
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, fill);
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
                tile.drain::<false>(self.pending);
            }

            // Cluster-scope, and the type doc says why: the leader's MMA below
            // writes the peer's half of the accumulator the peer was just
            // reading. The fence is the same pairing [`pipeline::run`] takes
            // around its own boundary — tcgen05 work (here the drain's LDTM)
            // retired before a thread sync that publishes it.
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();

            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, fill, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            self.pending = item;
        }
    }
}

/// [`Ablation`]'s job: [`Tile`] with one phase of the item switched off per
/// rung, and everything else — the launch geometry, the shared plan, the
/// tensor memory, the item map, the barrier protocol and the scaffold —
/// identical to the shipped kernel's.
///
/// **Every rung but `<true, true, true>` computes a wrong `C` and is never
/// checked.** They are not GEMMs and are not supposed to be; see [`Ablation`]
/// for what each one is subtracted from what.
///
/// The three switches are const parameters of the *shipped* phases rather than
/// copies of them ([`Tile::produce`], [`Tile::multiply`], [`Tile::drain`]), so
/// a rung cannot drift from the kernel it is meant to be a rung of, and
/// `<true, true, true>` is `Tile::work` instruction for instruction. The tile
/// is [`SHIPPED`]'s and not a parameter: this ladder decomposes one kernel, and
/// re-running it at another shape is a change to [`BLOCK_N`] and [`STAGES`]
/// rather than a second axis to sweep against.
#[derive(Clone, Copy)]
struct Ablated<const LOADS: bool, const MMA: bool, const DRAIN: bool> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
}

impl<const LOADS: bool, const MMA: bool, const DRAIN: bool> Job for Ablated<LOADS, MMA, DRAIN> {
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
    /// As [`Tile`]'s. At `DRAIN = false` nothing reads the accumulator, which
    /// is a wrong `C` and not a hazard; at `MMA = false` the drain reads
    /// whatever the allocation held, which is the same.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<LOADS>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<MMA>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            if DRAIN {
                tile.drain::<false>(item);
            }
        }
    }
}

/// [`Ablation::Idle`]'s job: an item with **no phases at all**.
///
/// It is a separate type rather than a fourth switch on [`Ablated`] because it
/// differs from that family in more than one thing — no producer, no consumer,
/// no wait on `done`, no reconvergence — and a rung that pretends to be one
/// step from its neighbour when it is four would be exactly the ladder defect
/// this file's method section warns about. It prices the floor: the persistent
/// grid's item loop, the per-item barrier `init`/`inval`, and the two
/// cluster-scope boundaries [`pipeline::run`] takes around every item.
///
/// **It computes nothing and is never checked.**
#[derive(Clone, Copy)]
struct Idle {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
}

impl Job for Idle {
    const RANKS: u32 = crate::gemm::RANKS;

    /// # Safety
    ///
    /// As [`Tile`]'s. The barriers are still armed and retired per item —
    /// that is the whole of what this rung measures.
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
    /// Nothing arrives at a barrier and nothing waits on one, so there is no
    /// protocol left to violate.
    #[inline(always)]
    unsafe fn work(&mut self, _item: u32) {}
}

/// [`Epilogue::Staged`]'s job: the shipped item with [`Tile::drain_staged`]
/// where [`Tile::drain`] was, and nothing else moved.
///
/// The three phases are the *shipped* ones ([`Tile::produce`],
/// [`Tile::multiply`]) rather than copies of them, for [`Ablated`]'s reason: a
/// second spelling of the K walk is a place for the A/B's two arms to drift.
/// What differs from `Tile::work` is one call and one extra field.
///
/// `DRAIN` is the ablation switch, and it is here rather than reached through
/// [`Ablated`] because the two rungs must declare the *same* 114 816 B shared
/// plan: `whole − no drain` at the staged envelope is what makes the staged
/// epilogue's exposed cost comparable with #114's 20.43 µs, and a control at a
/// different envelope would be measuring the plan as well as the drain. At
/// `false` it computes a wrong `C` and is never checked.
///
/// `WIDE` and `X4` are [`Tile::drain_staged`]'s two instruction widths (#117).
/// They change no byte of the shared plan, so all four combinations declare
/// the same 114 816 B and share **one** `no drain` control — the ablation is
/// clean in a way #116's was not, since that one had to price a 16 424-byte
/// envelope change before it could price the epilogue.
#[derive(Clone, Copy)]
struct Staged<const DRAIN: bool, const WIDE: bool, const X4: bool> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    /// This warp's 4096 B of the staging run — see [`StageTile`] for why it is
    /// the warp's and not the CTA's.
    stage: StageTile,
}

impl<const DRAIN: bool, const WIDE: bool, const X4: bool> Job for Staged<DRAIN, WIDE, X4> {
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
    /// As [`Tile`]'s, plus [`Tile::drain_staged`]'s: `stage` is this warp's
    /// alone, which [`kernels::attach_staged`] is what establishes.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            if DRAIN {
                tile.drain_staged::<WIDE, X4>(item, self.stage);
            }
        }
    }
}

/// [`Epilogue::FusedWide`]'s job: the shipped fused item with the **register**
/// epilogue at `.x8`.
///
/// It exists to un-confound #116 from #117. #116 measured the shared round trip
/// against the register drain with both at `.x1`, and #117 then took ~8 µs a
/// tile of exposed tensor-memory latency out of the staged arm only. Whether
/// the round trip is still worth its `stmatrix` and its `ld.shared` is a
/// question about *this* pair — register `.x8` against staged `.x8` — and
/// nothing in this file has ever run the left-hand side of it.
///
/// It computes the GEMM: [`TmemTile::tile_x8`] and [`TmemTile::tile`] return
/// the same band, so this is on the correctness gate.
#[derive(Clone, Copy)]
struct Wide {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
}

impl Job for Wide {
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
    /// As [`Tile`]'s.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            // The one line that differs from `Tile::work`.
            tile.drain::<true>(item);
        }
    }
}

/// [`Epilogue::StagedHot`]'s job: [`Staged`] at both of #117's widths with
/// every global store aimed at the cluster's own first tile, so the epilogue's
/// bytes stay in L2.
///
/// [`HotStore`] is this probe on the register drain and #107 ran it there. It
/// is worth running again on the staged path for the reason that section gives
/// for re-running it after #108: the question is whether the global half is
/// **bandwidth-bound or issue-bound**, and #116 changed the write from 1024
/// half-full sector transactions to 512 full ones without changing the bytes.
/// A shape change that helps a bandwidth-bound store and a shape change that
/// helps an issue-bound one are different claims, and this is what separates
/// them: if `staged84` and this are the same time, the stores are not waiting
/// on HBM and a TMA engine has nothing to win.
///
/// **It computes a wrong `C` and is never checked.**
#[derive(Clone, Copy)]
struct StagedHot {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    stage: StageTile,
    /// As [`HotStore::home`].
    home: u32,
}

impl Job for StagedHot {
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
    /// As [`Staged`]'s. The store is to a tile of `C` this cluster owns, so it
    /// is in bounds; it is the *wrong* tile, which is the point.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            tile.drain_staged::<true, true>(self.home, self.stage);
        }
    }
}

/// [`Tile::drain_staged_twice`]'s job — the staged epilogue's own `2x`/`2s`
/// ladder, at both of #117's widths.
///
/// `STAGE` is how much of the second pass runs; see
/// [`Tile::drain_staged_twice`] for the table and for why this is a doubling
/// probe rather than a subtractive one.
///
/// **It computes a wrong `C` and is never checked.**
#[derive(Clone, Copy)]
struct StagedTwice<const STAGE: u32> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    stage: StageTile,
    /// As [`DoubleDrain::home`]: the extra pass's bytes have to stay in L2 or
    /// this would price a doubled HBM stream as well as a doubled issue.
    home: u32,
}

impl<const STAGE: u32> Job for StagedTwice<STAGE> {
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
    /// As [`Staged`]'s, plus [`Tile::drain_staged_twice`]'s: both tiles are
    /// ones this cluster owns, and the accumulator is read rather than written.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            tile.drain_staged_twice::<true, true, STAGE>(item, self.home, self.stage);
        }
    }
}

/// [`Epilogue::StagedTma`]'s job: the shipped item with
/// [`Tile::drain_staged_tma`] where [`Tile::drain_staged`] was, and nothing
/// else moved.
///
/// The ring is a field rather than a local because its cursor and its
/// outstanding store groups span *items*: a persistent cluster's last band of
/// item `k` is still being read out of shared memory when item `k + 1`'s first
/// `stmatrix` wants the buffer, and `acquire` is what stands between them. At
/// depth 1 the cursor never moves, so what the field really carries is the
/// obligation — which is why the entry point drains it after the item loop and
/// not inside one.
///
/// `TWICE` is [`Tile::drain_staged_tma`]'s doubling instrument; at `true` it
/// computes a wrong `C` and is never checked.
#[derive(Clone, Copy)]
struct StagedTma<const TWICE: bool> {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    ring: StageRing,
    /// The descriptor for `C`, with a `[32, 64]` box — the one operand this
    /// epilogue needs that no other one does, and the reason the TMA entry
    /// points take an argument the rest do not.
    c_map: *const TmaDescriptor,
    /// As [`DoubleDrain::home`], and only read at `TWICE`.
    home: u32,
}

impl<const TWICE: bool> Job for StagedTma<TWICE> {
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
    /// As [`Staged`]'s, plus [`Tile::drain_staged_tma`]'s.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            tile.drain_staged_tma::<TWICE>(item, self.home, &mut self.ring, self.c_map);
        }
    }
}

/// [`Epilogue::StagedTmaCta`]'s job: [`StagedTma`] with the four staging tiles
/// read as one and the ring at CTA scope — see [`Tile::drain_staged_tma_cta`].
#[derive(Clone, Copy)]
struct StagedTmaCta {
    tile: Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    ring: CtaRing,
    /// The descriptor for `C`, with a `[128, 64]` box — a different map from
    /// [`StagedTma`]'s, because a box is the tile's own shape.
    c_map: *const TmaDescriptor,
}

impl Job for StagedTmaCta {
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
    /// As [`StagedTma`]'s, at CTA scope.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            if tile.warp_id == 0 && tile.lane == 0 {
                tile.produce::<true>(tile_m, tile_n, 0, tile.k_blocks);
            }
            if tile.rank == LEADER && tile.warp_id == 1 && tile.lane == 0 {
                tile.multiply::<true>();
            }
            tile.done.wait(0);
            thread::sync_threads();
            tile.drain_staged_tma_cta(item, &mut self.ring, self.c_map);
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
    unsafe fn attach<
        const BLOCK_N: usize,
        const HALF_N: usize,
        const BLOCK_K: usize,
        const STAGES: usize,
    >(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>, ClcQueue) {
        // Everything a rung has to be true about, fired at codegen — which is
        // the only place the ring byte counts are known, and the reason `cargo
        // check` cannot stand in for `modal_app.py::build`.
        //
        // `HALF_N` is a second parameter rather than `BLOCK_N / 2` because a
        // const parameter cannot be arithmetic on another one without
        // `generic_const_exprs`; the assert is what keeps the two in step.
        //
        // The last one is the join between the two spellings of this rung's
        // plan: `shared` is the walk `SharedPlan` lays out, `shared_plan` is
        // the value-parameterized arithmetic the host rung table needs, and
        // #125 could not make them one expression — see `shared_plan` for why.
        // Everything else that used to be here is gone with the pointer walk:
        // the queue's `ALIGNMENT` is `SharedPlan::clc_queue`'s and the ring
        // byte counts are the walk's own.
        const {
            assert!(BLOCK_N == 2 * HALF_N);
            assert!(BLOCK_N % DRAIN_N == 0);
            assert!(
                shared::<HALF_N, BLOCK_K, STAGES>(SharedPlan::sizing())
                    .plan
                    .bytes()
                    == shared_plan(BLOCK_N, BLOCK_K, STAGES)
            );
        };
        unsafe {
            let shared = shared::<HALF_N, BLOCK_K, STAGES>(SharedPlan::attach());

            let tile = Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                done: shared.done,
                a_map,
                b_map,
                accumulator: Accumulator::<BLOCK_N>::from_raw(alloc_cluster(
                    shared.tmem_slot,
                    BLOCK_N as u32,
                )),
                c: GlobalRows::<Bf16>::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                group,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id: warp_id(),
                lane: lane(),
            };
            (tile, shared.queue)
        }
    }

    /// The shipped rung's staging run, past the end of [`shared`]'s walk.
    ///
    /// # Safety
    ///
    /// The launch must declare [`STAGED_SHARED_BYTES`], not [`SHARED_BYTES`].
    #[inline(always)]
    unsafe fn stage_run() -> StageRun {
        unsafe { staged(shared::<HALF_N, BLOCK_K, STAGES>(SharedPlan::attach()).plan).0 }
    }

    /// [`attach`] on the shipped rung, plus this warp's [`StageTile`] out of
    /// the run that follows the shipped plan.
    ///
    /// The staging run is at the *end* of the envelope rather than folded into
    /// [`shared_plan`], and that is what keeps the A/B honest: every offset
    /// `attach` lays out is byte for byte the one [`gemm_cg2`] uses, so the two
    /// arms differ in the epilogue and in 16 424 declared bytes and in nothing
    /// else.
    ///
    /// # Safety
    ///
    /// [`attach`]'s, and the launch must declare [`STAGED_SHARED_BYTES`].
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach_staged(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>, StageTile) {
        unsafe {
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, c,
            );
            (tile, stage_run().tile(tile.warp_id))
        }
    }

    /// [`attach_staged`] with the warp's staging tile handed back as a
    /// [`StageRing`] — the same base address, since a ring of one buffer *is*
    /// its buffer and taking it from [`SharedTile::base`] is what stops the
    /// per-warp offset being written down twice.
    ///
    /// # Safety
    ///
    /// [`attach_staged`]'s.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach_staged_ring(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>, StageRing) {
        unsafe {
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, c);
            (tile, StageRing::attach(stage.base()))
        }
    }

    /// [`attach_staged`] with the whole staging run handed back as one
    /// CTA-scope [`CtaRing`] — the same 16 384 B from the same offset, read as
    /// one `[BLOCK_M, STAGE_N]` tile rather than as four.
    ///
    /// # Safety
    ///
    /// [`attach_staged`]'s.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach_staged_cta_ring(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>, CtaRing) {
        unsafe {
            let (tile, _) = attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, c);
            // The run's own base, which is buffer 0's — the whole point of
            // reading the same 16 384 B as one `[BLOCK_M, STAGE_N]` tile.
            (tile, CtaRing::attach(stage_run().tile(0).base()))
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
    unsafe fn release<
        const BLOCK_N: usize,
        const HALF_N: usize,
        const BLOCK_K: usize,
        const STAGES: usize,
    >(
        tile: &Tile<BLOCK_N, HALF_N, BLOCK_K, STAGES>,
    ) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), BLOCK_N as u32);
        }
    }

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `(2·BLOCK_M, BLOCK_N)` output
    /// tile per work item, `k_blocks` stages of `BLOCK_K` deep — **with the
    /// register drain, which since #119 is the control and not the default.**
    /// [`gemm_cg2_staged_x8x4`] is what [`bench`] launches; this is the arm
    /// every epilogue A/B in `experiments/README.md` §7 is measured against, and
    /// the only entry point with a [`Scheduler::Stealing`] twin.
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
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

    /// The same GEMM with its epilogue **staged through shared memory** —
    /// `stmatrix` into a per-warp tile and 16-byte stores out of it, where
    /// [`gemm_cg2`] stores 4 bytes a thread straight out of registers.
    ///
    /// Identical to [`gemm_cg2`] in grid, tensor memory, operand maps, item map
    /// and schedule, and in every byte of the shared plan `attach` lays out.
    /// What differs is [`Tile::drain_staged`] and 16 424 more declared bytes,
    /// which is 114 816 against 98 392 and still **2 CTAs an SM** — the tensor
    /// memory term of `min(512 / columns, shared per SM / plan)` is what binds
    /// at 256 accumulator columns, so shared memory has 1920 B left to give
    /// before anything moves.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s, at [`STAGED_SHARED_BYTES`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Staged::<true, false, false> { tile, stage };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_cg2_staged`] with the LDTM half at `.x8` — #117's first lever,
    /// and the only thing that differs from that kernel.
    ///
    /// `tcgen05.ld.16x256b.x8` returns 32 f32 a thread where the `.x1` this
    /// crate has always issued returns 4, so a `[32, 64]` staged band is 2
    /// loads and 2 waits instead of 16 and 16. Same bytes out of tensor
    /// memory, same `stmatrix`, same stores, same 114 816 B — and
    /// [`gemm_cg2_staged_no_drain`] is its control as much as it is
    /// [`gemm_cg2_staged`]'s, since the envelope does not move.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Staged::<true, true, false> { tile, stage };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_cg2_staged`] with the `stmatrix` half at `.x4` — #117's third
    /// lever, alone.
    ///
    /// A fragment is four `8x8` b16 matrices and `.x2` names two, so the
    /// shipped staged path issues two `stmatrix` per `[16, 16]` block where
    /// this issues one. The addresses are the same 32; only the lane grouping
    /// that supplies them changes.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x4(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Staged::<true, false, true> { tile, stage };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// Both of #117's widths at once — the composition rung, and **the kernel
    /// this file ships since #119** ([`SHIPPED_EPILOGUE`]).
    ///
    /// It is the only rung that can say whether the two widths add, and they
    /// do: `.x8` alone is +23.6% / +8.6% / +3.6% over `staged` and the pair is
    /// +23.1% / +8.8% / +5.1%, the gain at the largest size coming entirely
    /// from `.x4` handing back 14 of the 52 registers `.x8` costs (94 → 80).
    /// That recovery is what makes this the shipped rung here and *not* on
    /// `gemm_ws`, where the same `.x4` recovers 2.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Staged::<true, true, true> { tile, stage };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_cg2_staged_x8x4`] with the **shared→global hop handed to the TMA
    /// engine** — #15's route, and the one lever #121 left with a number on it.
    ///
    /// Identical to that kernel in grid, tensor memory, operand maps, item map,
    /// schedule, shared plan, LDTM width and `stmatrix` width; identical in
    /// every byte written to `C`. What differs is one `cp.async.bulk.tensor` a
    /// band where 32 `ld.shared.v4` and 32 `st.global.v4` a thread were, and
    /// the fence, the barrier and the group wait that instruction owes. See
    /// [`Tile::drain_staged_tma`].
    ///
    /// The extra `c_map` argument is the whole of what the host has to build
    /// for this: a rank-2 descriptor for `C` with a `[32, 64]` box. The plain
    /// staged path writes through a `GlobalRows` cursor carrying `ldc`, which
    /// is still passed because [`attach`] builds one either way.
    ///
    /// **It computes the GEMM and is on the correctness gate.**
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s, plus: `c_map` must describe the same `C` as `c`,
    /// `[n, m]` bf16 with a `[32, 64]` box.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_tma(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        c_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, ring) =
                attach_staged_ring(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedTma::<false> {
                tile,
                ring,
                c_map,
                home: 0,
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            // The one obligation the ring cannot discharge incrementally, and
            // the only thing that makes the last bands' bytes readable at all —
            // not the kernel ending, and not the `cluster_sync` in `release`.
            job.ring.drain();
            release(&job.tile);
        }
    }

    /// [`gemm_cg2_staged_x8x4_tma`] with the four staging tiles read as **one
    /// `[128, 64]` tile** and the ring at CTA scope — the same epilogue through
    /// [`kittens::epilogue::StoreRing`] exactly as #111 built it.
    ///
    /// One `cp.async.bulk.tensor` a band instead of four, at the price of eight
    /// `bar.sync` an item where the shipped epilogue has none. Its difference
    /// from the rung above is that price and nothing else. See
    /// [`Tile::drain_staged_tma_cta`].
    ///
    /// **It computes the GEMM and is on the correctness gate.**
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged_x8x4_tma`]'s, for a `[128, 64]` box.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_tma_cta(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        c_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, ring) = attach_staged_cta_ring(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = StagedTmaCta { tile, ring, c_map };
            pipeline::run(&mut job, tiles_m * tiles_n);
            job.ring.drain();
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_cg2_staged_x8x4_tma`] with a
    /// second TMA store per band: the TMA global half doubled and nothing else,
    /// so this minus that kernel is what the hop costs, measured by addition.
    ///
    /// The twin of [`gemm_cg2_staged_x8x4_2g`], which is the same question
    /// asked of `ld.shared` + `st.global` — and the comparison the two are for
    /// is between those two differences, not between either and a whole
    /// epilogue. See [`Tile::drain_staged_tma`].
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged_x8x4_tma`]'s. Both tiles it writes are tiles this
    /// cluster owns.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_tma_2g(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        c_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, ring) =
                attach_staged_ring(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedTma::<true> {
                tile,
                ring,
                c_map,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            job.ring.drain();
            release(&job.tile);
        }
    }

    /// [`gemm_cg2`] with the **register** epilogue at `.x8` — the left-hand
    /// side of #116's A/B, taken at #117's LDTM width instead of at the `.x1`
    /// one #116 had to take it at.
    ///
    /// Same 98 392 B as [`gemm_cg2`], since nothing here stages anything: the
    /// only difference from that kernel is [`TmemTile::tile_x8`] where
    /// [`TmemTile::tile`] was. It computes the GEMM and is checked.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_lcf8(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Wide { tile };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_cg2_staged_x8x4`] with every
    /// global store aimed at the cluster's own first tile, so the epilogue's
    /// bytes never leave L2.
    ///
    /// [`gemm_cg2_hot`] is this probe on the register drain. Bandwidth against
    /// issue, on the store shape #116 built: see [`StagedHot`].
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_hot(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedHot {
                tile,
                stage,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_cg2_staged_x8x4`] with a second
    /// `store_shared_rows` per band: the staged epilogue's **global half**,
    /// doubled and nothing else. See [`Tile::drain_staged_twice`].
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_2g(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedTwice::<1> {
                tile,
                stage,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_cg2_staged_x8x4_2g`] with the
    /// `stmatrix` pass doubled as well, so the difference between the two is
    /// the `stmatrix` half. See [`Tile::drain_staged_twice`].
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_2m(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedTwice::<2> {
                tile,
                stage,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — the whole staged epilogue run twice, so
    /// this minus [`gemm_cg2_staged_x8x4`] is one of them measured *serially*
    /// and this minus [`gemm_cg2_staged_x8x4_2m`] is the LDTM half. #108's `2x`
    /// on the staged path. See [`Tile::drain_staged_twice`].
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_x8x4_2x(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = StagedTwice::<3> {
                tile,
                stage,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_cg2_staged`] with the epilogue
    /// removed, which is the staged arm's own `no drain` control.
    ///
    /// [`gemm_cg2_no_drain`] would not do. It declares 98 392 B where this one
    /// declares 114 816, and the whole point of the subtraction is that
    /// everything but the drain is held: `staged − staged no drain` is then the
    /// staged epilogue's exposed cost on exactly the instrument that measured
    /// the register epilogue's at 20.43 µs a tile (#114), and the difference
    /// between the two subtractions is the answer this issue is after.
    ///
    /// Running it *against* [`gemm_cg2_no_drain`] is the second thing it buys:
    /// two kernels that differ only in 16 424 bytes of shared memory nothing
    /// touches, which is what says the envelope itself costs nothing at this
    /// residency.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2_staged`]'s, less the epilogue: it writes no `C` at all.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 114_816,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_staged_no_drain(
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
            let (tile, stage) =
                attach_staged(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Staged::<false, false, false> { tile, stage };
            pipeline::run(&mut job, tiles_m * tiles_n);
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
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

    /// **A deliberately wrong GEMM** — [`Epilogue::DoubleDrain`]'s probe, which
    /// prices the epilogue by running it twice. Never checked, never shipped.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s. Both tiles it writes are tiles this cluster owns.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_2x(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = DoubleDrain {
                tile,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`Epilogue::DoubleStore`]'s probe, which
    /// doubles the epilogue's stores and not its LDTM. Never checked, never
    /// shipped.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s. Both tiles it writes are tiles this cluster owns.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_2s(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = DoubleStore {
                tile,
                home: cluster::cluster_idx().min(tiles_m * tiles_n - 1),
            };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    // The ablation ladder is the five entry points below, and they are five
    // rather than one because a `#[kernel]` cannot be generic. Every one of
    // them launches on the shipped rung's geometry — `[256, 256] @ STAGES = 3`,
    // 98 392 B, 2 CTAs an SM, `GROUP = 8`, the static schedule — so the only
    // thing that differs between neighbours is which phase of the item runs.
    // See `Ablation` for the lattice and for what each edge prices.
    //
    // **None of them computes a GEMM** and none is on the correctness gate.

    /// **A deliberately wrong GEMM** — the ladder's `no drain` rung: the loads
    /// and the MMA of the shipped kernel, with the epilogue removed.
    ///
    /// The MMA cannot be deleted along with the drain that consumed it: it is
    /// `tcgen05.mma` behind inline PTX, its operands are released by a
    /// `tcgen05.commit` the producer's `wait_recycled` blocks on, and its final
    /// commit is what `done.wait` below is waiting for. Delete any of that and
    /// this kernel hangs rather than gets faster. The ladder says so too — the
    /// `no mma` rung below is a different number.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s, less the epilogue: it writes no `C` at all.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_no_drain(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Ablated::<true, true, false> { tile };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — the ladder's `no mma` rung: the loads
    /// and the epilogue, with the `tcgen05.mma` chain removed and every
    /// `tcgen05.commit` around it kept.
    ///
    /// The drain reads an accumulator nothing wrote, which is the wrong `C`
    /// this rung is for. It is also what keeps the epilogue in the kernel: the
    /// LDTM and the stores are issued whatever the values are, and no
    /// instruction in either is data-dependent.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s. It writes the tile the item names, with wrong numbers
    /// in it.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_no_mma(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Ablated::<true, false, true> { tile };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — the ladder's `loads` rung: the operand
    /// stream and the barrier protocol, with neither the MMA nor the epilogue.
    ///
    /// It is one step from [`gemm_cg2_no_drain`] (the MMA) and one step from
    /// [`gemm_cg2_no_mma`] (the epilogue), which is what closes the ladder's
    /// 2 x 2: the two paths to it price the same two phases in the presence
    /// and in the absence of each other.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s, less the epilogue: it writes no `C`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_loads(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Ablated::<true, false, false> { tile };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — the ladder's `dry` rung: the whole
    /// barrier protocol with **no operand traffic**.
    ///
    /// One step from [`gemm_cg2_loads`], and the step is the four TMA loads a
    /// stage issues and the transaction count they are charged as. The stage
    /// barrier is still armed, arrived at and waited on once per K block, the
    /// ring still recycles against the consumer's commits, and the item still
    /// walks its `k_blocks`. So `loads - dry` is what delivering the operands
    /// costs *exposed*, and `dry` itself is a K loop with nothing in it.
    ///
    /// # Safety
    ///
    /// [`gemm_cg2`]'s, less the loads and the epilogue: the operand maps are
    /// never read and no `C` is written.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_dry(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Ablated::<false, false, false> { tile };
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **Not a GEMM at all** — the ladder's floor: [`pipeline::run`] over items
    /// whose `work` does nothing.
    ///
    /// The TMEM allocation, the shared plan, the grid, the item count and the
    /// per-item barrier lifecycle are the shipped kernel's; the item's body is
    /// empty. It differs from [`gemm_cg2_dry`] in the whole barrier protocol
    /// rather than in one phase, and is reported as a floor for that reason.
    ///
    /// # Safety
    ///
    /// The launch geometry's. It reads neither operand and writes no `C`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 98_392,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_cg2_idle(
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
            let (tile, _) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            let mut job = Idle { tile };
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, queue) = attach::<BLOCK_N, HALF_N, BLOCK_K, STAGES>(
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<128, 64, 64, 2>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<128, 64, 64, 4>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<256, 128, 64, 2>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 256]` one atom deep, one stage — 32 824 B, and the shallow end of
    /// the depth family.
    ///
    /// Tensor memory still caps it at two CTAs an SM, so the 33 KiB it does not
    /// spend buys nothing: this rung gives up the pipeline for a resource that
    /// was not binding. It exists to be the fixed-depth control for
    /// [`gemm_256x256_k128_s1`], because a `BLOCK_K` comparison taken across
    /// different `STAGES` would be two variables.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 32_824,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x256_k64_s1(
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
            let (mut tile, _) = attach::<256, 128, 64, 1>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 256]` **two atoms in one stage** — 65 592 B, and the only rung in
    /// this file where `BLOCK_K` is not 64.
    ///
    /// A stage is two stacked swizzle subtiles per operand, so one TMA per
    /// operand brings both, one stage barrier covers both, and the MMA walks
    /// each atom in turn — `SharedTile::k_walk` describes exactly one atom, so
    /// the widening lives in the *walk count* and not in the walk. Instruction
    /// for instruction against `gemm_256x256_k64_s1` at the same `K`, this rung
    /// issues the same TMA loads and the same MMA chunks and **half the stage
    /// barriers, `expect_tx` charges and loop iterations**.
    ///
    /// It has the same 65 KiB and the same 128 K in flight as
    /// [`gemm_256x256_s2`], which is what makes that pair the control: same
    /// bytes, same residency, opposite factorization.
    ///
    /// # Safety
    ///
    /// As [`gemm_cg2`]. `k_blocks` counts `BLOCK_K = 128` blocks here, which
    /// the launcher derives from the rung rather than from a constant.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 65_592,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_256x256_k128_s1(
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
            let (mut tile, _) = attach::<256, 128, 128, 1>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }

    /// `[256, 128]` three stages deep — 73 816 B, three CTAs an SM. **The
    /// kernel this file shipped through #102, kept as the sweep's control.**
    ///
    /// It is no longer [`gemm_cg2`]: #87 moved the pair tile to `[256, 256]`,
    /// and this entry point is what every table in `experiments/README.md` before
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
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (mut tile, _) = attach::<128, 64, 64, 3>(
                a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c,
            );
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
///
/// Since #108 it rounds the *output* too, and there it is not exact — see
/// [`check_c`]. It is the same rounding `cvt.rn.bf16x2.f32` does, ties to even
/// included, which is what lets the comparison stay `==`.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// bf16 → fp32, which is a shift and loses nothing: the way an observed `C`
/// re-enters arithmetic the host can print and divide.
fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    // Bytes of one swizzle atom of K, which is the unit this is quoted in for
    // every rung: `reuse` is a ratio and does not care, and `distinct` is
    // comparable across rungs only if the depth it is taken at is the same one.
    let depth = (ATOM_K * 2) as f64;
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
    /// Which phases of the item run at all — the ablation ladder's variable,
    /// and [`Ablation::Whole`] in every plan that computes a GEMM.
    pub ablation: Ablation,
}

impl Plan {
    /// The [`SHIPPED`] rung on whichever schedule `scheduler` names, walked at
    /// the measured [`GROUP`], **with the epilogue fused into the item**.
    ///
    /// That last clause used to say "the kernel as it ships" and no longer
    /// does: [`SHIPPED_EPILOGUE`] is `staged84` since #119, and [`bench`] is
    /// what applies it. What this constructor gives is the register-drain
    /// base every sweep below takes its control arm from — the only epilogue
    /// with a [`Scheduler::Stealing`] entry point, and the only one the
    /// [`Ablation`] cube is built at — so a caller wanting the shipped launch
    /// says `.with(SHIPPED_EPILOGUE)` and a caller wanting the control says
    /// nothing.
    fn new(scheduler: Scheduler) -> Self {
        Plan {
            scheduler,
            group: GROUP,
            rung: SHIPPED,
            epilogue: Epilogue::Fused,
            ablation: Ablation::Whole,
        }
    }

    /// The same plan with its store phase somewhere else — #15's variable.
    fn with(self, epilogue: Epilogue) -> Self {
        Plan { epilogue, ..self }
    }

    /// The same plan with one phase of the item switched off — the ablation
    /// ladder's variable.
    fn ablated(self, ablation: Ablation) -> Self {
        Plan { ablation, ..self }
    }

    /// Dynamic shared memory this plan's launch declares.
    ///
    /// The rung's own plan for every epilogue but the staged family, which
    /// carries a staging tile per warp on the end of it — the one place a
    /// launch's envelope is not a function of the rung alone, and the reason
    /// this is a method rather than `rung.shared()` at the call site.
    ///
    /// All four staged rungs answer the same number. #117's two instruction
    /// widths change what the epilogue issues and not what it occupies, so the
    /// staged A/Bs after #116 are taken at one envelope as well as at one
    /// residency.
    fn shared(self) -> usize {
        match self.epilogue {
            Epilogue::Staged
            | Epilogue::StagedWide
            | Epilogue::StagedX4
            | Epilogue::StagedWideX4
            | Epilogue::StagedHot
            | Epilogue::StagedTwiceGlobal
            | Epilogue::StagedTwiceShared
            | Epilogue::StagedTwiceAll
            | Epilogue::StagedTma
            | Epilogue::StagedTmaCta
            | Epilogue::StagedTmaTwiceGlobal => {
                staged_plan(self.rung.block_n, self.rung.block_k, self.rung.stages)
            }
            _ => self.rung.shared(),
        }
    }
}

/// Which phases of an output tile a launch runs — the ablation ladder.
///
/// Every figure this file has ever published for a *part* of the kernel came
/// from one of two instruments, and they disagree by construction. The
/// `gemm-depth` fit's intercept is everything that does not scale with `K`,
/// which is the part of the item boundary the pipeline is actually **exposed**
/// to; #108's `2x`/`2s` probes add a second epilogue to an item and measure
/// what one costs **serially**, with nothing in flight to hide it. #108
/// measured 21.4 µs of serial epilogue against #104's 8.6–18.3 µs of *whole*
/// fitted boundary, which is only consistent if most of the epilogue is
/// already overlapped with something.
///
/// This axis asks the third question: with the kernel's own launch geometry
/// and its own schedule, **what does the launch stop costing when a phase is
/// taken out of it**. That is the exposed cost, measured by subtraction rather
/// than by a fit, and it is what a change to that phase can hope to recover.
///
/// # It is a cube, and that is the whole design
///
/// An item has three phases — the operand traffic, the multiply, the epilogue
/// — and each is either in the kernel or not, so the rungs are the eight
/// corners of `{loads} × {mma} × {drain}` and every edge of the cube is one
/// phase's cost *in one context*. A phase priced at a single corner has no
/// error bar on the thing that matters: a ladder that peels phases off the end
/// measures the epilogue in a kernel that still has its multiply, and the
/// multiply in a kernel that has already lost its epilogue, and cannot say
/// which of those attributions is the overlap.
///
/// Pricing every phase at all four of its corners is what separates
/// **serial** from **exposed** without assuming either. The epilogue in the
/// empty kernel (`drain - dry`) is as serial as this harness can make it; the
/// epilogue in the whole kernel (`whole - no drain`) is what the shipped
/// launch is exposed to; and if those two agree, the phase is overlapped with
/// nothing, whatever a fit says about it.
///
/// **Five of the eight corners are built, and the missing three are an
/// observation.** Every rung that runs the epilogue with the operand loads
/// switched off fails to return; two were launched and stopped by hand, and
/// [`Ablation::at`] has what separates them from the rungs that run and why the
/// first explanation for it was wrong. The tables print `—` there. The epilogue
/// keeps both of the contexts the question turns on — with the multiply and
/// without — and what is lost is its cost in an *empty* kernel and the operand
/// traffic's exposed cost.
///
/// [`Ablation::Idle`] sits outside the cube — it is a floor rather than a
/// corner, differing from `dry` in the whole barrier protocol rather than in
/// one phase, and is reported as one.
///
/// **Every rung but [`Ablation::Whole`] computes a wrong `C` on purpose and is
/// excluded from the correctness gate**, which is the same exception
/// [`Epilogue::HotStore`] and its two siblings carry, stated in the launch's
/// own label.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ablation {
    /// Every phase — the kernel this file ships, and the cube's `(1, 1, 1)`.
    Whole,
    /// The epilogue removed: no LDTM and no stores.
    NoDrain,
    /// The `tcgen05.mma` chain removed, and every barrier around it kept.
    NoMma,
    /// The operand stream and the barrier protocol, and nothing else.
    LoadsOnly,
    /// No phase at all: the barrier protocol walked once per K block, moving
    /// nothing and computing nothing. The cube's `(0, 0, 0)`.
    Dry,
    /// An item with **no barrier protocol either** — the persistent grid's
    /// loop, its per-item `init`/`inval`, and the two cluster boundaries around
    /// each item. A floor rather than a corner, since it differs from
    /// [`Ablation::Dry`] in the whole protocol rather than in one phase.
    Idle,
}

impl Ablation {
    /// What the benchmark prints, and the only place these names are spelled.
    pub fn name(self) -> &'static str {
        match self {
            Ablation::Whole => "whole",
            Ablation::NoDrain => "no drain",
            Ablation::NoMma => "no mma",
            Ablation::LoadsOnly => "loads",
            Ablation::Dry => "dry",
            Ablation::Idle => "idle",
        }
    }

    /// The corner of the cube this rung sits at, as `(loads, mma, drain)`.
    /// [`Ablation::Idle`] is not one and answers `None`, which is what keeps
    /// the floor out of every difference the sweep takes.
    fn corner(self) -> Option<(bool, bool, bool)> {
        Some(match self {
            Ablation::Whole => (true, true, true),
            Ablation::NoDrain => (true, true, false),
            Ablation::NoMma => (true, false, true),
            Ablation::LoadsOnly => (true, false, false),
            Ablation::Dry => (false, false, false),
            Ablation::Idle => return None,
        })
    }

    /// The rung at a corner, where one is built — the inverse of
    /// [`Ablation::corner`], and how the sweep names the far end of an edge it
    /// wants to difference.
    ///
    /// **Three corners answer `None`, and it is an observation rather than an
    /// omission: every rung that runs the epilogue with the operand loads
    /// switched off fails to return.**
    ///
    /// Two of them were built and launched at `8192x8192x512`, each produced no
    /// result, and each was stopped by hand — `(loads 0, mma 1, drain 1)` and
    /// `(loads 0, mma 0, drain 1)`. Their neighbours run: `(0, 0, 0)` is `dry`
    /// and finishes in 23 µs, `(1, 0, 1)` is `no mma` and finishes at every
    /// depth. So what separates the hanging rungs from the running ones is
    /// **`loads = 0` and `drain = 1`**, in both MMA states, and the third
    /// corner sharing that property is not built on this evidence rather than
    /// on a third B200.
    ///
    /// **What hangs is not established, and the first guess was wrong.** That
    /// guess was `tcgen05.mma` against never-written shared memory, and the
    /// second hang refutes it — that rung issues no MMA at all. What is
    /// recorded is the shape and not a mechanism: an epilogue in a kernel whose
    /// TMA never ran is what does not come back, and neither the LDTM nor the
    /// stores have any argument that depends on a load.
    ///
    /// What it costs the sweep is stated where it costs it. The epilogue keeps
    /// both of the contexts the question turns on — with the multiply beside it
    /// and without — so exposed-against-serial is unaffected. What is lost is
    /// the epilogue in an *empty* kernel, which would have reproduced #108's
    /// `2x` by a second route, and the operand traffic's exposed cost, whose
    /// contexts with the drain present are two of these three.
    fn at(corner: (bool, bool, bool)) -> Option<Ablation> {
        LADDER
            .into_iter()
            .find(|rung| rung.corner() == Some(corner))
    }

    /// What the rung still runs, printed beside it so a row says what it is
    /// rather than what it is called.
    fn phases(self) -> &'static str {
        match self.corner() {
            None => "an empty item",
            Some((true, true, true)) => "loads + mma + drain",
            Some((true, true, false)) => "loads + mma",
            Some((true, false, true)) => "loads + drain",
            Some((true, false, false)) => "loads",
            Some((false, false, false)) => "barriers only",
            Some((false, _, _)) => "not built — see Ablation::at",
        }
    }

    /// Whether a launch on this rung computes the GEMM. Only the whole kernel
    /// does, and every other rung says so everywhere it appears.
    fn exact(self) -> bool {
        self == Ablation::Whole
    }
}

/// One axis of [`Ablation`]'s cube: a phase an item either runs or does not.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Loads,
    Mma,
    Drain,
}

/// The three of them, in the order the report prices them.
const PHASES: [Phase; 3] = [Phase::Loads, Phase::Mma, Phase::Drain];

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Loads => "operand traffic",
            Phase::Mma => "the mma",
            Phase::Drain => "the epilogue",
        }
    }

    /// This phase set to `on`, with the other two axes left at `context` — the
    /// corner whose difference from its twin is this phase's cost right there.
    fn corner(self, context: (bool, bool), on: bool) -> (bool, bool, bool) {
        match self {
            Phase::Loads => (on, context.0, context.1),
            Phase::Mma => (context.0, on, context.1),
            Phase::Drain => (context.0, context.1, on),
        }
    }

    /// What the other two axes are called, for the column that names a context.
    fn context_of(self, context: (bool, bool)) -> String {
        let (first, second) = match self {
            Phase::Loads => ("mma", "drain"),
            Phase::Mma => ("loads", "drain"),
            Phase::Drain => ("loads", "mma"),
        };
        match context {
            (false, false) => "in an empty kernel".to_string(),
            (true, false) => format!("with {first}"),
            (false, true) => format!("with {second}"),
            (true, true) => format!("with {first} + {second}"),
        }
    }
}

/// **What an item's store phase is, and where it runs** — #15, and the axis
/// [`Lcsf`] adds.
///
/// It is not a scheduler and not a rung: the grid, the tensor memory, the item
/// map and the tile are identical across all of it, and between
/// [`Epilogue::Fused`] and [`Epilogue::Deferred`] so is the shared plan — the
/// only thing that moves there is whether [`Tile::drain`] runs at the end of
/// its own item or at the start of the next one.
///
/// [`Epilogue::Staged`] moves a second thing and says so in its own doc: it
/// keeps the placement and changes the *shape* of the store, which costs
/// 16 424 declared bytes it cannot get any other way. Residency does not move
/// with them, which is what keeps this one axis rather than two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    /// `lcf`: the store folds into the item and cannot overlap the next item's
    /// loads. The control, and what this kernel shipped through #119.
    Fused,
    /// `lcsf`: the store is deferred one item and runs while the next item's
    /// first [`Tile::FILL`] stages are in flight.
    Deferred,
    /// `staged`: the store stays where [`Epilogue::Fused`] puts it and changes
    /// **shape** — TMEM → registers → `stmatrix` into a per-warp shared tile →
    /// 16-byte stores to `C` ([`Tile::drain_staged`]).
    ///
    /// This is the one member of this axis that moves the shared plan, and it
    /// has to: the staging tiles are 16 384 B that did not exist. 114 816 B is
    /// still 2 CTAs an SM — the tensor-memory term binds at 256 accumulator
    /// columns — so the A/B is still taken at one residency, but the sentence
    /// above about the plan being identical across this axis is false here and
    /// [`gemm_cg2_staged_no_drain`](kernels::gemm_cg2_staged_no_drain) is the
    /// control that prices the difference.
    ///
    /// **It computes the GEMM and is on the correctness gate**, unlike the
    /// three probes at the end of this enum.
    Staged,
    /// `staged8`: [`Epilogue::Staged`] with the LDTM half at
    /// `tcgen05.ld.16x256b.x8` — 32 f32 a thread an issue instead of 4, so a
    /// `[32, 64]` band is 2 loads and 2 waits rather than 16 and 16 (#117).
    ///
    /// The waits are the point. [`TmemTile::fragment`] waits after *each* of
    /// its two `.x1` loads because the registers it waits on are the load's
    /// return value, so a `.x1` drain never has more than one LDTM in
    /// flight and pays the full tensor-memory latency per four values.
    ///
    /// **The shared plan does not move**, unlike [`Epilogue::Staged`]'s: this
    /// is the same 114 816 B, the same staging tiles and the same two CTAs an
    /// SM, so [`gemm_cg2_staged`](kernels::gemm_cg2_staged) is a clean control
    /// with no envelope term in it, and
    /// [`gemm_cg2_staged_no_drain`](kernels::gemm_cg2_staged_no_drain) serves
    /// this rung's `whole − no drain` subtraction unchanged.
    ///
    /// On the correctness gate.
    StagedWide,
    /// `staged4`: [`Epilogue::Staged`] with the `stmatrix` half at
    /// `.m8n8.x4` — one instruction per `[16, 16]` block instead of two (#117).
    ///
    /// A [`kittens::reg::Fragment`] is four `8x8` b16 matrices, which is
    /// exactly what `.x4` takes; `.x2` names two and therefore has to be
    /// issued once per slot. The 32 addresses are the same 32
    /// ([`kittens::ldst::store_fragment_x4`] reuses `fragment_address`
    /// outright), so this halves an instruction count and moves no byte.
    ///
    /// Same plan and same control as [`Epilogue::StagedWide`]. On the
    /// correctness gate.
    StagedX4,
    /// `staged84`: both of #117's widths — the composition rung, and
    /// [`SHIPPED_EPILOGUE`] since #119.
    ///
    /// They do add, and the register column is why: `.x4` hands back 14 of the
    /// 52 registers `.x8` costs (94 → 80), which is the whole of why this beats
    /// [`Epilogue::StagedWide`] at 16384³. On `gemm_ws` the same `.x4` recovers
    /// 2 and that kernel ships `staged8` instead.
    ///
    /// Same plan and same control as [`Epilogue::StagedWide`]. On the
    /// correctness gate.
    StagedWideX4,
    /// `lcf8`: the **register** epilogue at #117's LDTM width — the arm #116's
    /// A/B never had.
    ///
    /// #116 measured the shared round trip against [`Epilogue::Fused`] with
    /// both arms at `.x1`, and #117 then took the exposed tensor-memory latency
    /// out of the staged arm alone. So the published comparison prices *two*
    /// changes at once, and whether the round trip still earns its `stmatrix`
    /// and its `ld.shared` is a question about `lcf8` against
    /// [`Epilogue::StagedWide`] — a pair nothing has run.
    ///
    /// Same 98 392 B as [`Epilogue::Fused`] and the same bytes into `C`, so it
    /// is on the correctness gate and needs no envelope control.
    FusedWide,
    /// **Not a GEMM. Never checked.** [`Epilogue::StagedWideX4`] with every
    /// global store aimed at the cluster's own first tile.
    ///
    /// [`Epilogue::HotStore`] on the staged path, and it asks that variant's
    /// question about the shape #116 built rather than about the one it
    /// replaced: with the band's global writes now 512 full sector
    /// transactions instead of 1024 half-full ones, is what is left
    /// **bandwidth-bound or issue-bound**? If this and `staged84` are the same
    /// time, nothing is waiting on HBM and handing the write to a TMA engine
    /// (#15's original route, `kittens::epilogue::StoreRing`) has nothing to
    /// win.
    StagedHot,
    /// **Not a GEMM. Never checked.** [`Epilogue::StagedWideX4`] with a second
    /// `store_shared_rows` per band — the global half doubled and nothing else.
    ///
    /// The first rung of [`Tile::drain_staged_twice`]'s ladder: this minus
    /// `staged84` is what `ld.shared` + `st.global.v4` cost, measured by
    /// addition so no dead-code pass gets a vote.
    StagedTwiceGlobal,
    /// **Not a GEMM. Never checked.** [`Epilogue::StagedTwiceGlobal`] with the
    /// `stmatrix` pass doubled too, so the difference is that pass — which is
    /// the `cvt` as well as the `stmatrix`, since `Element::pack` is inside it
    /// and the census reads 8 `cvt.rn.bf16x2` against 16 across this step.
    StagedTwiceShared,
    /// **Not a GEMM. Never checked.** The whole staged epilogue twice, so this
    /// minus `staged84` is one of them **serially** and this minus
    /// [`Epilogue::StagedTwiceShared`] is the LDTM half — the issue and the
    /// wait together, which is the term #117 found and did not exhaust.
    StagedTwiceAll,
    /// `s84 tma`: [`Epilogue::StagedWideX4`] with the shared→global hop handed
    /// to the **TMA engine** — one `cp.async.bulk.tensor` a band where 32
    /// `ld.shared.v4` and 32 `st.global.v4` a thread were (#15, #111).
    ///
    /// The staging tile stays per warp, so the ring is warp-scope and the item
    /// still pays no block barrier; what it adds is a
    /// `fence.proxy.async.shared::cta` per lane, a `bar.warp.sync`, a commit
    /// and a group wait. See [`Tile::drain_staged_tma`].
    ///
    /// **The claim was issue and not bandwidth, and it came out below zero.**
    /// #121's `s84 hot` deleted the epilogue's entire HBM traffic for −2.3% to
    /// +1.8%, which retired the premise this rung was originally scoped from;
    /// what was left is the 2.0–2.3 µs a tile that `ld.shared` + `st.global`
    /// cost at 8192³, worth about +2.7% of the launch *if it went to zero*. It
    /// does not go to zero — the engine's hop is 1.57–1.99 µs a tile against
    /// 2.33–2.58 — and this rung is **1.0–1.7% slower than `staged84`** at the
    /// geometries where a difference this size can be measured at all. #15's
    /// TMA-store idea is retired with a number.
    ///
    /// Same 114 816 B, same two CTAs an SM, same `staged no drain` control. On
    /// the correctness gate.
    StagedTma,
    /// `s84 tmac`: [`Epilogue::StagedTma`] with the four per-warp staging tiles
    /// read as **one `[128, 64]` tile** and the ring at CTA scope — one TMA
    /// instruction a band instead of four, at eight `bar.sync` an item instead
    /// of none.
    ///
    /// This is [`kittens::epilogue::StoreRing`] as #111 built it, unmodified,
    /// and the pair with the rung above is what says which scope the epilogue
    /// wanted. **It wanted this one**: this arm beats the warp-scope one by
    /// 0.6–1.6% in four paired cells across two containers, so the eight block
    /// barriers cost less than the three extra `[32, 64]` boxes they save. It
    /// still does not beat `staged84` — 0.2–0.5% behind at 8192², and the two
    /// containers disagree in sign at 16384². Same plan, same control, on the
    /// correctness gate.
    StagedTmaCta,
    /// **Not a GEMM. Never checked.** [`Epilogue::StagedTma`] with a second TMA
    /// store per band — the TMA global half doubled and nothing else.
    ///
    /// The twin of [`Epilogue::StagedTwiceGlobal`], and the two differences are
    /// what the comparison is: `2g − staged84` prices `ld.shared` +
    /// `st.global`, this minus `s84 tma` prices the engine's hop, and a lever
    /// that is worth anything has the second smaller than the first.
    StagedTmaTwiceGlobal,
    /// **Not a GEMM. This computes a wrong `C` on purpose and is never checked.**
    ///
    /// The fused epilogue with every item's store aimed at the *cluster's own
    /// first tile* instead of the item's. Identical in every instruction — same
    /// LDTM, same count of pair stores, same addresses within a tile — and the
    /// only thing that moves is that 148 clusters rewrite 19 MB of `C` over and
    /// over instead of streaming 134 MB or 512 MB of it, so the writes stay in
    /// L2 and never press on HBM. Every one of those figures halved when #108
    /// made `C` bf16, which is exactly the reason to re-run this probe rather
    /// than carry #107's answer across: the bytes it deletes are half what they
    /// were.
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
    /// **Not a GEMM either. This computes a wrong `C` on purpose and is never
    /// checked.**
    ///
    /// The fused epilogue run **twice** per item — the same LDTM and the same
    /// stores a second time, the second aimed at the cluster's own first tile
    /// so the extra bytes stay in L2 and what the extra pass costs is issue.
    ///
    /// It exists because nothing here has priced *the epilogue*. Every figure
    /// this file quotes for it is a share of the **item boundary**, read off the
    /// `gemm-depth` fit's intercept — and #104 says plainly that the intercept
    /// is now "everything that does not scale with `K`", which since #102
    /// includes a locality term, so the boundary reads 8.6 to 18.3 µs a tile
    /// depending on which points the line is drawn through. A 2.1× spread is
    /// not an instrument you can rank a lever with.
    ///
    /// This is the direct measurement instead: `2x - lcf`, over the items on the
    /// critical path, is **one epilogue**, with the grid, the shared plan, the
    /// tensor memory, the tile, the traversal, the schedule and the arithmetic
    /// all held fixed and no fit involved. It caps the whole family of epilogue
    /// changes at once — an epilogue that cost nothing could not save more than
    /// this — which is what #108 needs to answer whether `stmatrix` is worth
    /// reaching for now that a bf16 `C` has made it spellable.
    ///
    /// **What it does not separate is LDTM from stores.** [`Tile::drain`] is
    /// both, and the obvious probe that keeps one and drops the other is a probe
    /// whose result a dead-code pass decides: with nothing consuming the band,
    /// the LDTM goes too and the measurement is of an empty loop. So this
    /// number bounds the epilogue and does not decompose it, and it is quoted
    /// that way. [`Epilogue::DoubleStore`] is what does.
    DoubleDrain,
    /// **Not a GEMM either, and never checked.**
    ///
    /// The other half of [`Epilogue::DoubleDrain`]'s pair: one LDTM and **two**
    /// sets of stores, where that one is two of each. So `2s - lcf` is the
    /// store loop on its own and `2x - 2s` is the LDTM, and the epilogue is
    /// split into the two things it is made of without either probe having to
    /// delete anything a dead-code pass could then delete more of.
    ///
    /// That split is the whole question `stmatrix` turns on. A `stmatrix`
    /// epilogue keeps the LDTM exactly as it is and halves the stores — 128
    /// pair stores a thread an item become 64 warp-collective `stmatrix.x2`,
    /// with the same `cvt` count and the global write handed to a TMA engine.
    /// Its ceiling is therefore **half of `2s - lcf`**, and no part of
    /// `2x - 2s` is available to it at all.
    DoubleStore,
}

impl Epilogue {
    /// What the benchmark prints, and the only place these names are spelled.
    pub fn name(self) -> &'static str {
        match self {
            Epilogue::Fused => "lcf",
            Epilogue::Deferred => "lcsf",
            Epilogue::Staged => "staged",
            Epilogue::StagedWide => "staged8",
            Epilogue::StagedX4 => "staged4",
            Epilogue::StagedWideX4 => "staged84",
            Epilogue::FusedWide => "lcf8",
            Epilogue::StagedHot => "s84 hot",
            Epilogue::StagedTwiceGlobal => "s84 2g",
            Epilogue::StagedTwiceShared => "s84 2m",
            Epilogue::StagedTwiceAll => "s84 2x",
            Epilogue::StagedTma => "s84 tma",
            Epilogue::StagedTmaCta => "s84 tmac",
            Epilogue::StagedTmaTwiceGlobal => "s84t 2g",
            Epilogue::HotStore => "hot",
            Epilogue::DoubleDrain => "2x",
            Epilogue::DoubleStore => "2s",
        }
    }

    /// Whether a launch on this epilogue computes the GEMM. The probes do not,
    /// and each says so everywhere it appears.
    fn exact(self) -> bool {
        !matches!(
            self,
            Epilogue::HotStore
                | Epilogue::DoubleDrain
                | Epilogue::DoubleStore
                | Epilogue::StagedHot
                | Epilogue::StagedTwiceGlobal
                | Epilogue::StagedTwiceShared
                | Epilogue::StagedTwiceAll
                | Epilogue::StagedTmaTwiceGlobal
        )
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
    use kittens::global::{GlobalLayout, TensorMap};

    // The tiling is the constraint on what sizes exist: a cluster owns a whole
    // `2·BLOCK_M` by `BLOCK_N` tile and a stage is a whole `BLOCK_K`, and the
    // kernel bounds-checks none of it. A size that does not divide is rejected
    // rather than launched into somebody else's memory.
    let rung = plan.rung;
    if m % (2 * BLOCK_M) != 0 || n % rung.block_n != 0 || k % rung.block_k != 0 {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{}x{} tiling",
            2 * BLOCK_M,
            rung.block_n,
            rung.block_k
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
    // A tensor map's box is `[R, SUBTILE_COLS]` — one swizzle atom wide,
    // whatever the tile's own `C` — so **`BLOCK_K` does not reach the
    // descriptor at all**: a two-atom stage is two boxes through the same map,
    // which is what `tma_load_2d` already issues per stacked subtile. The one
    // extent that does reach it is the tile's rows, so the `B` map is the only
    // one a rung moves and it moves with `block_n` and not with `block_k`.
    let a_map = a_layout.tensor_map::<ATile<ATOM_K>>(&stream)?;
    let b_map = match rung.block_n {
        128 => b_layout.tensor_map::<BTile<64, ATOM_K>>(&stream)?,
        256 => b_layout.tensor_map::<BTile<128, ATOM_K>>(&stream)?,
        columns => return Err(format!("no rung has {columns} pair columns").into()),
    };

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    // The TMA epilogues are the only ones that need a descriptor for the
    // *output* — every other rung writes `C` through a `GlobalRows` cursor
    // carrying `ldc` and nothing else — so one is built only where one is
    // launched, and every other row of every sweep costs the two encodes it
    // always did. The box is the staging tile's own shape, which is why the two
    // scopes need two maps: `[32, 64]` for the per-warp ring and `[128, 64]`
    // for the CTA-wide one. `C` is row-major `[m, n]`, so the layout's extents
    // are `[n, m]` in the driver's fastest-first order, exactly as `A`'s are
    // `[k, m]`.
    //
    // SAFETY: `c` outlives every launch consuming the map, and holds `m * n`
    // bf16 elements.
    let c_layout = unsafe { GlobalLayout::<Bf16, 2>::packed(c.cu_deviceptr(), [n, m]) };
    let c_map = match plan.epilogue {
        Epilogue::StagedTma | Epilogue::StagedTmaTwiceGlobal => {
            Some(c_layout.tensor_map::<StageTile>(&stream)?)
        }
        Epilogue::StagedTmaCta => Some(c_layout.tensor_map::<CtaStage>(&stream)?),
        _ => None,
    };
    let c_ptr = c_map.as_ref().map_or(core::ptr::null(), TensorMap::as_ptr);
    let cap = rung.max_clusters(shared_per_sm(context)?);
    let blocks = grid_for(plan.scheduler, m, n, rung, cap);
    let (tiles_m, tiles_n) = tile_grid(m, n, rung.block_n);
    let k_blocks = (k / rung.block_k) as u32;
    let config = LaunchConfig1D::new(blocks, THREADS, plan.shared() as u32);

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
    // The same thing for an entry point that also takes a descriptor for `C`.
    // A second macro rather than an `Option` threaded through the first: the
    // argument list is the kernel's ABI and the arms that use it are exactly
    // the arms that built a map above.
    macro_rules! tma_launcher {
        ($prepare:ident, $launch:ident) => {{
            let prepared = module_ref.$prepare(config)?;
            let launch = move |c: &mut DeviceBuffer<u16>| -> Result<(), Box<dyn Error>> {
                unsafe {
                    module_ref.$launch(
                        stream_ref, &prepared, a_ptr, b_ptr, c_ptr, tiles_m, tiles_n, plan.group,
                        k_blocks, n as u32, c,
                    )?
                };
                Ok(())
            };
            Box::new(launch) as Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>>
        }};
    }
    let launch_once = match (rung.entry, plan.scheduler, plan.epilogue, plan.ablation) {
        // The ablation ladder first, because it is the one axis that is *only*
        // ever taken at the shipped rung, the static schedule and the fused
        // epilogue — a rung with a phase missing is not a thing to combine
        // with a scheduler comparison or a store placement, and the catch-all
        // below is what says so if anyone tries.
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::NoDrain) => {
            launcher!(prepare_gemm_cg2_no_drain, gemm_cg2_no_drain)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::NoMma) => {
            launcher!(prepare_gemm_cg2_no_mma, gemm_cg2_no_mma)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::LoadsOnly) => {
            launcher!(prepare_gemm_cg2_loads, gemm_cg2_loads)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::Dry) => {
            launcher!(prepare_gemm_cg2_dry, gemm_cg2_dry)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::Idle) => {
            launcher!(prepare_gemm_cg2_idle, gemm_cg2_idle)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2, gemm_cg2)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Deferred, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_lcsf, gemm_cg2_lcsf)
        }
        // The staged epilogue and its own `no drain` control. They are the one
        // pair on this axis that declares a different envelope, which
        // `Plan::shared` is what supplies to the launch config above.
        (Entry::Shipped, Scheduler::Static, Epilogue::Staged, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged, gemm_cg2_staged)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::Staged, Ablation::NoDrain) => {
            launcher!(prepare_gemm_cg2_staged_no_drain, gemm_cg2_staged_no_drain)
        }
        // #117's two instruction widths, separately and together. They declare
        // the same 114 816 B as the arm above, so `Ablation::NoDrain` is not
        // repeated for each of them — `staged no drain` is the control for all
        // four, which is the whole reason the widths were built on this axis.
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedWide, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8, gemm_cg2_staged_x8)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedX4, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x4, gemm_cg2_staged_x4)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedWideX4, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8x4, gemm_cg2_staged_x8x4)
        }
        // The register drain at #117's LDTM width — 98 392 B, so
        // `gemm_cg2_no_drain` is its control and not the staged one.
        (Entry::Shipped, Scheduler::Static, Epilogue::FusedWide, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_lcf8, gemm_cg2_lcf8)
        }
        // The staged epilogue's own decomposition: one hot-store probe and
        // three rungs of `Tile::drain_staged_twice`'s doubling ladder. All four
        // declare the same 114 816 B as `staged84`, so the same `staged no
        // drain` control serves every subtraction taken against them.
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedHot, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8x4_hot, gemm_cg2_staged_x8x4_hot)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTwiceGlobal, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8x4_2g, gemm_cg2_staged_x8x4_2g)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTwiceShared, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8x4_2m, gemm_cg2_staged_x8x4_2m)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTwiceAll, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_staged_x8x4_2x, gemm_cg2_staged_x8x4_2x)
        }
        // The TMA store, at both scopes, and its own doubling probe. Same
        // 114 816 B once more, so `staged no drain` is their control too — and
        // the three are the only arms in this match that take a descriptor for
        // `C`.
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTma, Ablation::Whole) => {
            tma_launcher!(prepare_gemm_cg2_staged_x8x4_tma, gemm_cg2_staged_x8x4_tma)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTmaCta, Ablation::Whole) => {
            tma_launcher!(
                prepare_gemm_cg2_staged_x8x4_tma_cta,
                gemm_cg2_staged_x8x4_tma_cta
            )
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::StagedTmaTwiceGlobal, Ablation::Whole) => {
            tma_launcher!(
                prepare_gemm_cg2_staged_x8x4_tma_2g,
                gemm_cg2_staged_x8x4_tma_2g
            )
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::HotStore, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_hot, gemm_cg2_hot)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::DoubleDrain, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_2x, gemm_cg2_2x)
        }
        (Entry::Shipped, Scheduler::Static, Epilogue::DoubleStore, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_2s, gemm_cg2_2s)
        }
        (Entry::Shipped, Scheduler::Stealing, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_cg2_clc, gemm_cg2_clc)
        }
        (Entry::N128S2, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x128_s2, gemm_256x128_s2)
        }
        (Entry::N128S4, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x128_s4, gemm_256x128_s4)
        }
        (Entry::N128S3, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x128_s3, gemm_256x128_s3)
        }
        (Entry::N256S2, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x256_s2, gemm_256x256_s2)
        }
        (Entry::N256K64S1, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x256_k64_s1, gemm_256x256_k64_s1)
        }
        (Entry::N256K128S1, Scheduler::Static, Epilogue::Fused, Ablation::Whole) => {
            launcher!(prepare_gemm_256x256_k128_s1, gemm_256x256_k128_s1)
        }
        (Entry::Unbuilt, _, _, _) => {
            return Err(format!(
                "[256,{}] k{} s{} is one CTA an SM and is computed, not built",
                rung.block_n, rung.block_k, rung.stages
            )
            .into());
        }
        // Only the shipped rung has a stealing twin, and deliberately: a
        // scheduler comparison at a moving tile would be two variables. The
        // deferred epilogue is the same rule for the same reason — it is the
        // one variable #15 moves, so it moves against the shipped kernel on the
        // static schedule and nothing else. The ablation ladder is the same
        // rule once more: a rung with a phase missing exists to be subtracted
        // from the shipped kernel and from nothing else.
        (entry, scheduler, epilogue, ablation) => {
            return Err(format!(
                "{entry:?} has no {} entry point with {} on the {} schedule",
                epilogue.name(),
                ablation.name(),
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
    let label = match (plan.epilogue.exact(), plan.ablation.exact()) {
        (true, true) => {
            // Exact on the 16-bit words, and the number beside it is the
            // rounding those words carry — the representation error of a bf16
            // `C`, which is a property of the output format and not of this
            // launch. See `check_c` for why the comparison is still `==`.
            let worst = check_c(&c.to_host_vec(&stream)?, m, n, k)?;
            format!("{m}x{n}x{k} exact, worst |rel| {worst:.2e} against the fp32 reference")
        }
        (false, _) => format!(
            "{m}x{n}x{k} UNCHECKED ({} is not a GEMM)",
            plan.epilogue.name()
        ),
        (_, false) => format!(
            "{m}x{n}x{k} UNCHECKED ({} is not a GEMM)",
            plan.ablation.name()
        ),
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

/// Compare an observed `[m, n]` row-major **bf16** `C` against the CPU
/// reference for `[m, k] · [n, k]ᵀ`, element by element and with `==`.
///
/// `a_value` repeats every 7 rows and `b_value` every 21 columns, so the
/// reference has 147 distinct dot products at any size and the naive
/// `O(m·n·k)` form is pure waste — minutes of host time at the sizes the
/// benchmark reaches, for the same 147 numbers. Every element of `C` is still
/// compared against its own expected value, in the same summation order, so the
/// comparison is the one it always was. The sum stays over the *full* `k`,
/// since both generators vary along it.
///
/// # Where the rounding is, and what that makes this blind to
///
/// The output is bf16 since #108 and the accumulator is not, so exactly one
/// rounding happens and this function has to agree with it. **It is put in the
/// reference**: the exact fp32 dot product is rounded by [`to_bf16`] — the same
/// round-to-nearest-even `cvt.rn.bf16x2.f32` performs — and the comparison
/// stays `==` on the 16-bit words. **No tolerance was introduced and none was
/// widened.** That is available because the sum itself is still exact: every
/// operand is exact in bf16 and every partial sum stays under 2²⁴, so the fp32
/// value reaching the `cvt` is the same integer whatever order it was summed
/// in, and rounding it once is a deterministic function of that integer.
///
/// The alternative — widen the observed bf16 and compare to the fp32 reference
/// within a tolerance — would have been strictly weaker, because a tolerance
/// wide enough to admit correct rounding also admits everything smaller than
/// it, and a wrong tile that happened to land close would pass.
///
/// What this *is* blind to is resolution, and it is worth being precise: two
/// fp32 accumulators differing by less than half an ulp of bf16 round to the
/// same word, so an error under about 0.2% of a value's magnitude is now
/// invisible where it was not before. Every failure mode this gate exists for —
/// a wrong coordinate, a wrong stride, a dropped or doubled tile, a wrong
/// operand half, a mis-walked K — moves an element by far more than that or
/// leaves it at zero, so none of them has become harder to see. A kernel that
/// accumulated in bf16 instead of fp32 would be the case this cannot promise to
/// catch at every element, and at these `K` it would still fail most of them.
///
/// The return value is the **worst relative error against the exact fp32
/// reference** over the elements compared, which is the representation error
/// and is reported rather than asserted: it is a property of bf16 and of the
/// magnitudes this reference produces, not of the kernel.
///
/// It is a function rather than a block inside [`run`] because the **cuBLASLt
/// baseline** (#92) is checked by it too. A denominator produced by a different
/// GEMM is worth nothing, and the way that happens is a transposed operand or a
/// wrong leading dimension — which computes the wrong matrix at full speed and
/// looks like a plausible number. Sharing this function rather than copying it
/// is what makes "checked against the same CPU reference" a property of the
/// code instead of a claim in a comment. It also holds the library to the same
/// rounding: cuBLASLt is asked for a `CUDA_R_16BF` output off a
/// `CUBLAS_COMPUTE_32F` accumulator, so if it rounded any other way — or
/// rounded a partial sum — its `C` would fail here rather than be timed.
pub(crate) fn check_c(
    observed: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, Box<dyn Error>> {
    let exact: Vec<f32> = (0..7 * 21)
        .map(|cell| {
            (0..k)
                .map(|depth| a_value(cell / 21, depth) * b_value(cell % 21, depth))
                .sum()
        })
        .collect();
    let reference: Vec<u16> = exact.iter().copied().map(to_bf16).collect();
    let (mut wrong, mut sample, mut worst) = (0usize, Vec::new(), 0.0f64);
    for row in 0..m {
        for column in 0..n {
            let cell = (row % 7) * 21 + column % 21;
            let value = observed[row * n + column];
            if value != reference[cell] {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!(
                        "C[{row}, {column}] = {}, want {}",
                        from_bf16(value),
                        from_bf16(reference[cell])
                    ));
                }
            }
            if exact[cell] != 0.0 {
                let error = (from_bf16(value) - exact[cell]) as f64 / exact[cell] as f64;
                worst = worst.max(error.abs());
            }
        }
    }
    if wrong > 0 {
        return Err(format!("{wrong} of {} elements wrong: {}", m * n, sample.join("; ")).into());
    }
    Ok(worst)
}

/// A scheduler's own failure modes, named so a passing line says what it
/// proved. All three are silent on the device and all three reach the host as
/// a wrong `C`, which is why the exact comparison above is the only gate that
/// sees them:
///
/// - a **dropped tile** — a steal that succeeded and was read as "no work
///   left", so the cancelled cluster's item is computed by nobody. Reading the
///   `is_canceled` polarity backwards produces exactly this, which is why the
///   probe in `device-tests` establishes it by counting rather than by trusting
///   a doc. (An earlier version of this comment said cuda-oxide's module doc
///   had the sense inverted. It did, at v0.2.1; upstream fixed it before the
///   revision we pin, and that claim was read off a stale cargo checkout.
///   `experiments/README.md` §7 has the correction.)
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
        // The rounding a bf16 `C` carries, reported once per size from the
        // first checked launch of it (#108). It is measured against the exact
        // fp32 reference rather than asserted, and it is the same number
        // whichever plan produced the output, since every plan that reaches
        // this line was `==` on the words.
        let mut rounding = None;
        for scheduler in SCHEDULERS {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler,
                    group,
                    rung: SHIPPED,
                    epilogue: Epilogue::Fused,
                    ablation: Ablation::Whole,
                };
                let (label, _) = run(context, m, n, k, plan, nothing_after)?;
                rounding.get_or_insert(label);
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
                ablation: Ablation::Whole,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} tiles, one deferred \
             accumulator per cluster in flight)",
            Epilogue::Deferred.name(),
            tiles(m, n, BLOCK_N)
        ));
        // The staged store phase, under the same widths and the same `==`.
        //
        // It is on this gate and not beside a benchmark row for a sharper
        // reason than the deferred one: `stmatrix` and the read-back address
        // the same swizzled tile through two different derivations, the
        // per-warp staging tiles are four row-subranges of one 16 384 B run,
        // and the loop reuses each tile four times with only a `bar.warp.sync`
        // between the read of one pass and the write of the next. Every one of
        // those is a wrong `C` rather than a fault if it is wrong, and this is
        // the only thing that looks.
        for group in CHECK_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                rung: SHIPPED,
                epilogue: Epilogue::Staged,
                ablation: Ablation::Whole,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} tiles, {} B, \
             {} CTAs/SM)",
            Epilogue::Staged.name(),
            tiles(m, n, BLOCK_N),
            STAGED_SHARED_BYTES,
            staged_ctas_per_sm(per_sm)
        ));
        // #117's two instruction widths, on the same gate and for a sharper
        // reason still: both are claims about a *register map* that no
        // compile-time type states.
        //
        // `.x8` returns 32 f32 in one instruction and the order they arrive in
        // is silicon's, not the ISA text's — `kittens::tmem::interleave_x8`
        // asserts repeat-major and `device-tests`' `ldtm x8 map` is what holds
        // it against the `.x1` drain. `.x4` names four `stmatrix` matrices
        // where `.x2` names two, and a wrong lane-to-matrix grouping writes a
        // transposed or overlapping tile.
        //
        // Neither faults. Both compute a wrong `C`, and `check_c` compares
        // bf16 words with `==` and no tolerance at all, so a lever that cost a
        // mantissa bit would fail here rather than pass quietly — which is
        // what makes this gate, and not a benchmark's error column, the thing
        // that decides whether a width is shippable.
        for epilogue in [
            Epilogue::StagedWide,
            Epilogue::StagedX4,
            Epilogue::StagedWideX4,
        ] {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler: Scheduler::Static,
                    group,
                    rung: SHIPPED,
                    epilogue,
                    ablation: Ablation::Whole,
                };
                run(context, m, n, k, plan, nothing_after)?;
            }
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {}, {} and {} at groups {CHECK_GROUPS:?} \
             ({} ships, same {} B)",
            Epilogue::StagedWide.name(),
            Epilogue::StagedX4.name(),
            Epilogue::StagedWideX4.name(),
            SHIPPED_EPILOGUE.name(),
            STAGED_SHARED_BYTES
        ));
        // `.x8` on the *register* drain, which is the same repeat-major claim
        // one band wider: `Band` is 128 columns where `StagedBand` is 64, so
        // this walks two `.x8` groups per band and a wrong interleave that
        // happened to cancel within one group would not survive two.
        for group in CHECK_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                rung: SHIPPED,
                epilogue: Epilogue::FusedWide,
                ablation: Ablation::Whole,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} B, the register drain \
             at .x8)",
            Epilogue::FusedWide.name(),
            SHARED_BYTES
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
                    ablation: Ablation::Whole,
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
        if let Some(label) = rounding {
            notes.push(label);
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
///
/// **The epilogue it launches is [`SHIPPED_EPILOGUE`], which since #119 is
/// `staged84` and not `lcf`.** Every `bench --case gemm*` row published before
/// that is against the register drain and is not comparable to one taken after
/// it; `experiments/README.md` §7 says which tables moved.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    bench_with(context, shape, SHIPPED_EPILOGUE)
}

/// [`bench`] on a named epilogue — the shipped rung and schedule, checked
/// first, with only #15's variable moved.
///
/// It exists for [`crate::gemm_ws::compare`], which has to quote *this*
/// kernel's best rung as a control and cannot quote it from
/// `experiments/README.md`: #98 found 2.9% of drift between containers and #109
/// came within a paragraph of publishing a false +3.6% against a baseline that
/// had moved under it. A control that is not moving is one measured beside the
/// thing it controls, which means the other file needs a way to ask for one.
pub fn bench_with(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
    epilogue: Epilogue,
) -> Result<Timings, Box<dyn Error>> {
    run(
        context,
        shape.m,
        shape.n,
        shape.k,
        Plan::new(Scheduler::Static).with(epilogue),
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
/// `cargo oxide run kittens-experiments -- clc`.
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
/// # And the stage's K, which had never been swept — the second half
///
/// `BLOCK_K` was 64 from this file's first commit and no issue had moved it.
/// Two facts about it turned up before any rung was built, and both are worth
/// more than the sweep:
///
/// - **It cannot be swept as a walk width.** `SharedTile::k_walk` carries
///   `const { assert!(C * E::BYTES == S::ATOM_BYTES) }` — a linear K-major walk
///   spans exactly one swizzle atom — and `Swizzle128B` is the only mode in
///   tree, so 64 is what a walk *is* at bf16. A wider stage is several atoms
///   walked in turn, which is what [`Tile::multiply`] now does, and the
///   descriptor never sees it: a tensor map's box is `[R, SUBTILE_COLS]`
///   whatever the tile's `C`.
/// - **It does not move arithmetic intensity.** A tile reads `(M + N) · K`
///   bytes to do `2 · M · N · K` flops however K is blocked, so the mechanism
///   #87 found — the whole of what the pair tile was worth — is not on this
///   axis at all. What `BLOCK_K` moves is the *count* of stage barriers,
///   `expect_tx` charges and loop iterations an item pays, and how coarsely the
///   ring recycles.
///
/// And the shared budget makes it a factorization rather than an axis: a stage
/// is `512 · BLOCK_K` bytes at this pair tile, so two CTAs an SM cap
/// `BLOCK_K · STAGES` at 228 and every extra atom in a stage is a stage given
/// back. [`K128S1`] and `[256, 256] @ s2` are the pair that holds the bytes
/// fixed and moves only the factorization.
///
/// ## The predictions, written down before the rungs ran
///
/// 1. **`k64 s1` is a catastrophe.** One stage is no pipeline: the producer's
///    `wait_recycled` cannot clear until the MMA reading that stage has
///    committed, so loads and arithmetic serialize outright. Predicted
///    **−35% to −55%** against the shipped kernel at 8192³.
///    **Confirmed: −44.5% and −45.6%.**
/// 2. **`k128 s1` loses to `k64 s2` at equal bytes**, and by most of what the
///    load/MMA overlap is worth. Predicted **−20% to −45%**. **Refuted in
///    magnitude: −14.4% and −16.6%** — right sign, half the size.
/// 3. **`k128 s1` beats `k64 s1`**, and *this is the actual `BLOCK_K`
///    measurement* — same depth, twice the K per barrier. Predicted **+5% to
///    +20%**, from a halved per-K-block fixed cost. **Refuted upward, and it is
///    the informative one: +38.2% and +42.4%.** Far too large to be barrier
///    issue: the ablation's `dry` rung prices the *whole* barrier protocol at
///    22 µs a tile against a 105 µs launch, and halving that cannot buy 38%.
///    What a wider stage halves at one stage is **the number of times the
///    pipeline is exposed to a load latency**, because the producer cannot
///    refill until the MMA releases the one buffer. `BLOCK_K` is a
///    latency-amortization lever and not a fixed-cost one, and it competes for
///    the same bytes as the better lever.
/// 4. **Nothing here beats the shipped kernel.** Confirmed.
/// 5. **The register column says nothing again.** Confirmed: `k128 s1` prices
///    at **246 registers** against `k64 s1`'s 168, zero spill either way, and
///    is 38% faster. Eighth time (#47, #63, #67, #76, #94, #100, #109).
///
/// **What the sweep bought is a design rule, not a win**, and the 65 KiB pair
/// is what states it: `k128 s1` and `k64 s2` are sixteen bytes apart, at
/// identical residency, tiles, waves, reuse and K in flight, differing only in
/// one barrier over two atoms against two barriers over one atom each — and
/// **two shallow stages beat one deep one by 14–17%**. At a fixed shared
/// budget, spend it on stages rather than on stage width. Which is why
/// `BLOCK_K = 64`, the narrowest a walk admits, is right here rather than
/// merely inherited: **it is swept, and the answer is that it was already
/// right**, for a reason this file had not stated.
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
         `intensity` is the pair tile's M*N/(M+N), flops per operand byte — a\n\
         property of the pair tile alone, so BLOCK_K does not move it and the two k128\n\
         rungs are there to move something else. `K in flight` is block_k * stages, which\n\
         is what the shared budget actually caps: two rungs sharing it declare the same\n\
         plan and differ only in how that K is divided into stage barriers."
    );

    println!("\n1. the rungs, and what each one costs before anything is launched");
    println!(
        "{:<18}{:>8}{:>8}{:>12}{:>10}{:>12}{:>10}{:>10}{:>12}",
        "rung",
        "block_k",
        "stages",
        "K in flight",
        "shared B",
        "TMEM cols",
        "CTA/SM",
        "clusters",
        "intensity"
    );
    for rung in RUNGS.into_iter().chain(UNBUILT) {
        let built = if rung.entry == Entry::Unbuilt {
            "  (not built)"
        } else {
            ""
        };
        println!(
            "{:<18}{:>8}{:>8}{:>12}{:>10}{:>12}{:>10}{:>10}{:>12.1}{built}",
            rung.name(),
            rung.block_k,
            rung.stages,
            rung.k_in_flight(),
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
                ablation: Ablation::Whole,
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
                ablation: Ablation::Whole,
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

/// The cube, ordered by how many phases a rung still runs, then by which —
/// which is the order the report prints and the order a reader can subtract in.
/// [`Ablation::Idle`] is last and is the floor rather than a corner.
const LADDER: [Ablation; 6] = [
    Ablation::Whole,
    Ablation::NoDrain,
    Ablation::NoMma,
    Ablation::LoadsOnly,
    Ablation::Dry,
    Ablation::Idle,
];

/// Items a cluster on the critical path walks at `m`x`n` — the divisor every
/// per-tile figure in this file uses, and the deepest-loaded cluster rather
/// than the average, because the launch ends when that one finishes.
fn items_on_critical_path(m: usize, n: usize) -> f64 {
    let items = tiles(m, n, BLOCK_N);
    let clusters = grid_for(Scheduler::Static, m, n, SHIPPED, MAX_CLUSTERS) / RANKS;
    items.div_ceil(clusters) as f64
}

/// Least squares of `y` on `x`, as `(slope, intercept)`.
///
/// The one statistical object in this file, kept to four lines because every
/// fit it is asked for has between two and four points and the interesting
/// question is never the arithmetic — it is which points went in, which is why
/// every caller prints all of its selections rather than a preferred one.
fn least_squares(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (sxx, sxy) = points
        .iter()
        .fold((0.0, 0.0), |(sxx, sxy), (x, y)| (sxx + x * x, sxy + x * y));
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (slope, (sy - slope * sx) / n)
}

/// The ablation ladder — `modal run modal_app.py::bench --case ablation`.
///
/// # What this measures that nothing else here does
///
/// Every figure this file publishes for a *part* of the kernel comes from one
/// of two instruments, and they answer different questions. #104's `gemm-depth`
/// fit puts the item boundary at **8.6–18.3 µs a tile** — the intercept, which
/// is everything that does not scale with `K`, so it sees only what the
/// pipeline is **exposed** to. #108's `2x`/`2s` probes run a second epilogue
/// inside an item and price it at **21.4 µs, of which 13.1 is stores and 8.3
/// is LDTM** — which is more than the whole fitted boundary, and can only be,
/// because the second epilogue has nothing in flight to hide behind and so
/// measures the epilogue's **serial** cost.
///
/// So most of the epilogue is overlapped with something and neither instrument
/// says how much. This one does: it takes a phase *out* of the shipped kernel
/// and measures what the launch stops costing. That is the exposed cost, by
/// subtraction, with the launch geometry, the shared plan, the tensor memory,
/// the residency, the traversal, the schedule and the item count all held
/// fixed — and it is the quantity a change to that phase can hope to recover.
///
/// # The predictions, written down before the first run, and how they went
///
/// The first session ran the four corners with the loads always on. Two of
/// these were refuted and the refutations are the result; they are kept as
/// written, because a pre-registration edited after the fact is worth nothing.
///
/// 1. **The epilogue's exposed cost at 8192³ is well under #108's serial
///    21.4 µs** — the fit sees 8.6–18.3 µs for the *whole* boundary and the
///    epilogue is only part of it. Predicted **4–12 µs a tile**.
///    **Refuted: 20.4 µs**, which is the serial figure to within 5%.
/// 2. **The epilogue costs more with the multiply gone**, because there is then
///    nothing left to cover it. Predicted a ratio of **1.5x to 4x**. **Refuted:
///    1.01x.** The epilogue is overlapped with nothing, so #108's serial number
///    and #104's fitted boundary have no reconciliation of the kind this
///    function was built expecting — and the fit is the instrument that is
///    wrong, not the probe.
/// 3. **The MMA, exposed, is most of the launch.** Predicted 60–85% of 8192³.
///    **Refuted downward: 24.6%**, and the reason is the same one #98 found —
///    what a lone CTA's stall is covered by is the *other CTA*, not by another
///    phase of its own item.
/// 4. **The operand traffic is small.** Predicted under 15% of the launch at
///    `K = 8192`. **Refuted: 33% — but only serially.** The corner that would
///    have priced it with the multiply in place is one of the two that hang,
///    so this number is the traffic in a kernel with no arithmetic to cover it
///    and the prediction is not, strictly, settled. Stated that way rather
///    than quoted as a share.
/// 5. **`idle` is a few µs a tile and `K`-independent.** Confirmed: 1.7 µs a
///    tile, 0.0122 ms a launch at every depth to three digits.
/// 6. **Registers move and nothing follows them.** Confirmed, and it is the
///    eighth time in this repo (#47, #63, #67, #76, #94, #100, #109): a rung
///    with no epilogue is **28 registers on a zero frame** against the shipped
///    kernel's 166 on 528 B, and the epilogue is the whole of both.
///
/// **And the result the whole thing is for.** The epilogue's exposed cost is
/// **20.4 µs a tile at 8192³ against #108's serial 21.4**, and the `no drain`
/// rung's own fit has **no fixed per-tile cost at all** — so the item boundary
/// is the epilogue and nothing else, and it is overlapped with nothing. `no
/// drain` measures **1850 TFLOP/s** where cuBLASLt in the same container
/// measures 1808, and the epilogue is **111% of the whole gap to the library**.
/// `experiments/README.md` §7 has the tables and what they do to `stmatrix`.
///
/// # What a rung cannot prove
///
/// The decomposition is **context-dependent**, and the cube is what bounds
/// that rather than hides it: an edge attributes to its phase everything the
/// launch stops paying when that phase goes, which includes whatever the phase
/// was *making other work wait for*. Two phases that overlap have a cost
/// between them that neither subtraction owns — which is why every phase is
/// priced at all four corners rather than one, and why the four numbers are
/// printed rather than averaged.
///
/// It also cannot see **inside** a phase. `no mma` removes the whole
/// `tcgen05.mma` chain, not its issue rate; `no loads` removes the traffic, not
/// the TMA instruction; and no rung here separates the epilogue's LDTM from its
/// stores, which is what #108's `2s` is for and this sweep deliberately does
/// not duplicate. Every rung but the first computes a wrong `C`, so none is on
/// the correctness gate and none of their numbers is a throughput.
///
/// # The hazard this ladder is most likely to fail on
///
/// A deleted drain can let a compiler delete the MMA that fed it. Four things
/// say it did not, and the argument is the standard #101 set:
///
/// - **Counted, in the PTX.** An opcode census over `kittens_examples.ptx`,
///   per entry function, is the direct check and it is unambiguous:
///   `gemm_cg2_no_drain` carries **4 `tcgen05.mma`** — the same four chained
///   chunks `gemm_cg2` has — with **0 `tcgen05.ld`, 0 `st.global` and 0
///   `cvt.rn.bf16x2`**. The epilogue is gone and the arithmetic is not.
///   `gemm_cg2_no_mma` is the mirror: **0 mma**, with all four LDTM, five
///   stores and five `cvt` still there. `gemm_cg2_dry` is the only rung
///   carrying a bare `mbarrier.arrive` and the only one with zero
///   `cp.async.bulk.tensor`. Each rung removes exactly what it names.
/// - **Structural.** The MMA is inline PTX writing tensor memory; its operands
///   are released by a `tcgen05.commit` the producer's `wait_recycled` blocks
///   on, and its last commit is what `done.wait` is waiting for. A dead-code
///   pass that removed it would produce a kernel that hangs.
/// - **Measured, by the cube itself.** `no drain` and `loads` differ only in
///   the MMA. If the MMA had been deleted from `no drain` the two rungs would
///   be the same kernel and would measure the same time. They differ by 26 µs
///   a tile at 8192³.
/// - **Priced.** `regcount` reports every one of these entry points off
///   `ptxas -v` in the same run: 166 registers on a 528 B frame wherever the
///   epilogue is, 20–28 on a zero frame wherever it is not, and no spill
///   anywhere.
pub fn ablation_ladder(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    let sizes = crate::bench::GEMM_DEPTH_SIZES;
    println!(
        "gemm ablation ladder — min ms over 30 timed launches, static schedule, GROUP = {GROUP},\n\
         the shipped [256,{BLOCK_N}] @ STAGES = {STAGES} rung throughout: same grid, same\n\
         {SHARED_BYTES} B shared plan, same {BLOCK_N} accumulator columns, same {CTAS_PER_SM} CTAs\n\
         an SM, same item count. One phase of the item is removed per rung and NOTHING ELSE\n\
         MOVES.\n\
         ONLY THE `whole` ROW COMPUTES A GEMM. It is checked element-by-element against the\n\
         CPU reference before it is timed; every other row is UNCHECKED and computes a wrong\n\
         `C` on purpose, exactly as `hot`, `2x` and `2s` do.\n\
         M and N are 8192 in every row, so tiles, waves, grid, `C` bytes and the wave's own\n\
         working set are identical everywhere and only the arithmetic between two item\n\
         boundaries moves — which is what lets each rung carry its own fit."
    );

    println!("\n1. the ladder, at four reduction depths");
    println!(
        "{:<12}{:<22}{:>10}{:>10}{:>10}{:>10}",
        "rung", "what it runs", "K=512", "K=2048", "K=8192", "K=32768"
    );
    let mut measured: Vec<(Ablation, Vec<f64>)> = Vec::new();
    for rung in LADDER {
        let mut row = Vec::new();
        for &shape in sizes {
            let plan = Plan::new(Scheduler::Static).ablated(rung);
            eprintln!("{shape} on ablation rung `{}`: staging", rung.name());
            let (_, timings) = run(context, shape.m, shape.n, shape.k, plan, time)?;
            row.push(timings.min());
        }
        println!(
            "{:<12}{:<22}{}",
            rung.name(),
            rung.phases(),
            row.iter()
                .map(|ms| format!("{ms:>10.4}"))
                .collect::<String>()
        );
        measured.push((rung, row));
    }
    let at = |rung: Ablation, column: usize| {
        measured
            .iter()
            .find(|row| row.0 == rung)
            .map(|row| row.1[column])
            .expect("every rung of LADDER was measured above")
    };

    println!(
        "\n2. every phase priced at all four corners of the cube, in microseconds per output\n\
         tile on the critical path. A row is one edge — two rungs differing in exactly that\n\
         phase — so it is what the launch stops paying when the phase goes, IN THAT CONTEXT.\n\
         The bottom row of each block is the phase in an otherwise empty kernel, which is as\n\
         SERIAL as this harness can make it; the top row is the phase in the whole kernel,\n\
         which is what the shipped launch is EXPOSED to. Those two agreeing means the phase\n\
         is overlapped with nothing."
    );
    println!(
        "{:<18}{:<24}{:>10}{:>10}{:>10}{:>10}",
        "phase", "context", "K=512", "K=2048", "K=8192", "K=32768"
    );
    let per_tile = |shape: Shape, milliseconds: f64| {
        milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n)
    };
    const CONTEXTS: [(bool, bool); 4] =
        [(true, true), (true, false), (false, true), (false, false)];
    let edge = |phase: Phase, context: (bool, bool), column: usize| {
        let with = Ablation::at(phase.corner(context, true))?;
        let without = Ablation::at(phase.corner(context, false))?;
        Some(at(with, column) - at(without, column))
    };
    for phase in PHASES {
        for (row, context) in CONTEXTS.into_iter().enumerate() {
            let cells: String = sizes
                .iter()
                .enumerate()
                .map(|(column, &shape)| match edge(phase, context, column) {
                    Some(milliseconds) => format!("{:>10.2}", per_tile(shape, milliseconds)),
                    None => format!("{:>10}", "—"),
                })
                .collect();
            println!(
                "{:<18}{:<24}{cells}",
                if row == 0 { phase.name() } else { "" },
                phase.context_of(context)
            );
        }
    }
    let floor: String = sizes
        .iter()
        .enumerate()
        .map(|(column, &shape)| format!("{:>10.2}", per_tile(shape, at(Ablation::Idle, column))))
        .collect();
    println!("{:<18}{:<24}{floor}", "the floor", "an empty item");
    let protocol: String = sizes
        .iter()
        .enumerate()
        .map(|(column, &shape)| {
            format!(
                "{:>10.2}",
                per_tile(
                    shape,
                    at(Ablation::Dry, column) - at(Ablation::Idle, column)
                )
            )
        })
        .collect();
    println!("{:<18}{:<24}{protocol}", "the protocol", "over the floor");

    println!(
        "\n3. the same thing as the answer it is for: SERIAL against EXPOSED, at each depth,\n\
         with `hidden` the fraction of the phase's serial cost that the rest of the kernel\n\
         was covering. `serial` is the phase in an empty kernel and `exposed` is the phase\n\
         in the whole one, both in microseconds per tile.\n\
         This is the column that reconciles — or does not — #108's 21.4 us of serial\n\
         epilogue with #104's 8.6-18.3 us of whole fitted item boundary."
    );
    println!(
        "{:<18}{:<10}{:>12}{:>12}{:>10}{:>12}",
        "phase", "shape", "serial us", "exposed us", "ratio", "hidden"
    );
    for phase in PHASES {
        for (column, &shape) in sizes.iter().enumerate() {
            let (Some(serial), Some(exposed)) = (
                edge(phase, (false, false), column),
                edge(phase, (true, true), column),
            ) else {
                continue;
            };
            let (serial, exposed) = (per_tile(shape, serial), per_tile(shape, exposed));
            println!(
                "{:<18}{:<10}{:>12.2}{:>12.2}{:>10}{:>12}",
                if column == 0 { phase.name() } else { "" },
                format!("K={}", shape.k),
                serial,
                exposed,
                format!("{:.2}x", exposed / serial),
                format!("{:.0}%", 100.0 * (1.0 - exposed / serial))
            );
        }
    }

    println!(
        "\n3b. the exposed cost as a share of the launch it was taken out of, which is the\n\
         form a lever is ranked in: it is the most a change to that phase could ever be\n\
         worth at that size, and every real change recovers some fraction of it."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}",
        "phase", "K=512", "K=2048", "K=8192", "K=32768"
    );
    for phase in PHASES {
        let cells: String = sizes
            .iter()
            .enumerate()
            .map(|(column, _)| match edge(phase, (true, true), column) {
                Some(exposed) => {
                    format!("{:>11.1}%", 100.0 * exposed / at(Ablation::Whole, column))
                }
                None => format!("{:>12}", "—"),
            })
            .collect();
        println!("{:<18}{cells}", phase.name());
    }

    println!(
        "\n4. a fit per rung, so the FIXED per-tile cost is decomposed and not only the\n\
         launch. ms against K is a line whose intercept is everything that does not scale\n\
         with K; #104 warns that since #102 that bucket also holds a K-dependent locality\n\
         term, which is why all three point selections are printed and the spread between\n\
         them is the honest uncertainty. `per tile` divides the intercept by the {} items a\n\
         cluster on the critical path walks. The `whole` row is `gemm-depth` re-fitted, and\n\
         has to reproduce #104's 21.5-27.2 us on the fp32 kernel for the rest to be read as\n\
         a decomposition of that fit.",
        items_on_critical_path(8192, 8192)
    );
    println!(
        "{:<12}{:>20}{:>14}{:>14}{:>16}",
        "rung", "points", "fixed ms", "per tile us", "steady TFLOP/s"
    );
    let selections: [(&str, &[usize]); 3] = [
        ("8192, 32768", &[2, 3]),
        ("2048, 8192, 32768", &[1, 2, 3]),
        ("all four", &[0, 1, 2, 3]),
    ];
    let mut fits: Vec<(Ablation, Vec<(f64, f64)>)> = Vec::new();
    for rung in LADDER {
        let mut per_selection = Vec::new();
        for (row, (label, points)) in selections.into_iter().enumerate() {
            let taken: Vec<(f64, f64)> = points
                .iter()
                .map(|&column| (sizes[column].k as f64, at(rung, column)))
                .collect();
            let (slope, intercept) = least_squares(&taken);
            per_selection.push((slope, intercept));
            // A rung with no arithmetic in it has a slope near zero, and
            // dividing by one produces a rate with no meaning rather than a
            // large one. It is printed as absent instead.
            let steady = 2.0 * 8192.0 * 8192.0 / (slope / 1e3) / 1e12;
            println!(
                "{:<12}{:>20}{:>14.4}{:>14.1}{:>16}",
                if row == 0 { rung.name() } else { "" },
                label,
                intercept,
                intercept * 1e3 / items_on_critical_path(8192, 8192),
                if slope > 1e-6 {
                    format!("{steady:.0}")
                } else {
                    "—".to_string()
                }
            );
        }
        fits.push((rung, per_selection));
    }

    println!(
        "\n4b. the fits differenced along the cube's edges — how much of each phase's cost\n\
         is FIXED per tile rather than proportional to K, which is the direct answer to what\n\
         the item boundary is MADE OF. A phase whose intercept difference is the whole\n\
         kernel's intercept is the whole boundary; one whose difference is zero is not in\n\
         the boundary at all, whatever it costs per launch."
    );
    println!(
        "{:<18}{:<24}{:>16}{:>16}{:>16}",
        "phase", "context", "fixed us/tile", "3pt", "all four"
    );
    let fit_at = |rung: Ablation, selection: usize| {
        fits.iter()
            .find(|row| row.0 == rung)
            .map(|row| row.1[selection].1)
            .expect("every rung of LADDER was fitted above")
    };
    for phase in PHASES {
        for (row, context) in [(true, true), (false, false)].into_iter().enumerate() {
            let cells: String = (0..3)
                .map(|selection| {
                    let difference = Ablation::at(phase.corner(context, true))
                        .zip(Ablation::at(phase.corner(context, false)))
                        .map(|(with, without)| {
                            fit_at(with, selection) - fit_at(without, selection)
                        });
                    match difference {
                        Some(milliseconds) => format!(
                            "{:>16.2}",
                            milliseconds * 1e3 / items_on_critical_path(8192, 8192)
                        ),
                        None => format!("{:>16}", "—"),
                    }
                })
                .collect();
            println!(
                "{:<18}{:<24}{cells}",
                if row == 0 { phase.name() } else { "" },
                phase.context_of(context)
            );
        }
    }

    let Some(baseline) = baseline else {
        println!(
            "\nno cuBLASLt drift control: built without --features cublas. modal_app.py::bench\n\
             turns it on, and #109 is why this table wants one."
        );
        return Ok(());
    };
    println!(
        "\n5. the drift control. Every rung here is the REGISTER-drain kernel — the ladder\n\
         is built at `lcf` and #119 moved the default to `staged84` without touching it —\n\
         so `whole` at 8192³ has to reproduce the published 0.812-0.825 of cuBLASLt for\n\
         the rest of this run to be readable as a decomposition of THAT kernel, which is\n\
         #109's lesson about a control the change itself moved."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>16}",
        "shape", "cuBLASLt ms", "theirs TF/s", "whole/theirs"
    );
    for (column, &shape) in sizes.iter().enumerate() {
        if shape.k != 8192 {
            continue;
        }
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        println!(
            "{:<18}{:>14.4}{:>14.1}{:>16.3}",
            shape,
            theirs,
            tflops(shape, theirs),
            theirs / at(Ablation::Whole, column)
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

/// #15's sweep — `cargo oxide run kittens-experiments -- bench epilogue`, which is
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
///
/// # And the prediction for `2x`, written down before that ran (#108)
///
/// [`Epilogue::DoubleDrain`] is new and measures a quantity nothing here has:
/// the epilogue itself, rather than its share of a fitted boundary. The
/// pre-registered guess is **5–12 µs per tile at 8192³, so 5–12% of that
/// launch** — the boundary fits at 8.6–18.3 µs and #91's evidence puts the
/// store loop at roughly half of it, with the LDTM most of the rest, so an
/// epilogue somewhat under the whole boundary is what the existing numbers
/// imply. A `2x - lcf` at or above the boundary's own upper fit would mean the
/// boundary is *mostly* epilogue and the fit has been mis-attributing it; well
/// under 5 µs would mean the epilogue is already nearly free and no epilogue
/// change — `stmatrix` included — is worth building.
///
/// It is not free of assumptions and the one it makes is worth naming: running
/// the epilogue twice measures the *marginal* epilogue, and a marginal one can
/// be cheaper than the first if the second finds its addresses in L2 and its
/// issue slots idle. That biases this number **down**, which is the safe
/// direction for a ceiling.
///
/// **It measured 21.4 µs and 20.2% of 8192³ — refuted, by about a factor of
/// two, and the bias argument above is refuted with it.** 21.4 µs is *above* the
/// whole item boundary as #104 fits it, and the epilogue is only part of that
/// boundary; both can be true only if the epilogue is partly overlapped in the
/// real kernel and not at all in the probe, which makes this the epilogue's
/// **serial** cost and the fit's intercept its **exposed** residue. `2s` — one
/// LDTM, two store loops — then splits it: 13.1 µs of stores against 8.3 µs of
/// LDTM at 8192³. `experiments/README.md` §7 has both, and what they say about
/// `stmatrix`.
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
         table and nothing has to be argued for out of a budget.\n\
         `C` is bf16 since #108, so every byte figure below is half what #107 measured\n\
         and table 2c is new: it prices the epilogue directly instead of as a share of a\n\
         fitted item boundary.",
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
        let twice = timed(context, shape, plan.with(Epilogue::DoubleDrain))?;
        let restored = timed(context, shape, plan.with(Epilogue::DoubleStore))?;
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
        measured.push((shape, fused, deferred, hot, twice, restored));
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
    for &(shape, fused, deferred, _, _, _) in &measured {
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
         store count, same addresses within a tile, 19 MB of C rewritten instead of 134 MB\n\
         or 512 MB streamed, so the writes stay in L2. IT COMPUTES A WRONG C AND IS NOT\n\
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
    for &(shape, fused, _, hot, _, _) in &measured {
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
        "\n2c. what the epilogue costs *at all*, measured rather than fitted (#108). `2x` runs\n\
         the fused epilogue TWICE per item — same LDTM, same stores, the second aimed at\n\
         the cluster's home tile so the extra bytes stay in L2 and what the extra pass\n\
         costs is issue. IT COMPUTES A WRONG C AND IS NOT CHECKED.\n\
         `2x - lcf` over the items on the critical path is one epilogue, with the grid,\n\
         the plan, the tile, the traversal, the schedule and the arithmetic all fixed and\n\
         no fit involved — where every figure this file has quoted for the epilogue is a\n\
         share of the item boundary read off an intercept whose own spread is 2.1x.\n\
         It is the CEILING on every epilogue change at once, `stmatrix` included: an\n\
         epilogue that cost nothing could not save more than this.\n\
         `2s` is the other half of the pair — ONE LDTM and TWO sets of stores, also to\n\
         the home tile and also unchecked — so `2s - lcf` is the store loop alone and\n\
         `2x - 2s` is the LDTM alone. Neither probe deletes anything, which is why a\n\
         dead-code pass has no say in either: the version that keeps the load and drops\n\
         the stores would measure whatever the optimizer left of it.\n\
         `stmatrix` keeps the LDTM and halves the stores, so its ceiling is half the\n\
         `stores` column and none of the `LDTM` one."
    );
    println!(
        "{:<18}{:>10}{:>10}{:>10}{:>14}{:>12}{:>12}{:>14}",
        "shape", "lcf ms", "2x ms", "2s ms", "epilogue us", "stores us", "LDTM us", "epilogue %"
    );
    for &(shape, fused, _, _, twice, restored) in &measured {
        let items = tiles(shape.m, shape.n, BLOCK_N);
        let clusters = grid_for(Scheduler::Static, shape.m, shape.n, SHIPPED, MAX_CLUSTERS) / RANKS;
        let critical = items.div_ceil(clusters) as f64;
        let per_item = |milliseconds: f64| milliseconds * 1e3 / critical;
        println!(
            "{:<18}{:>10.4}{:>10.4}{:>10.4}{:>14.2}{:>12.2}{:>12.2}{:>14}",
            shape,
            fused,
            twice,
            restored,
            per_item(twice - fused),
            per_item(restored - fused),
            per_item(twice - restored),
            format!("{:.1}%", 100.0 * (twice - fused) / fused)
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
    for &(shape, fused, deferred, _, _, _) in &measured {
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

/// Sizes the staged epilogue is measured at.
///
/// **Nothing below 4096³**, and that is a property of the denominator rather
/// than a choice about ours: `experiments/README.md` §7 records cuBLASLt's own
/// run-to-run spread reaching 77% at the small end, where a ratio is not a
/// quantity. The three here are the three every recent epilogue claim in this
/// file is quoted at.
const STAGED_SIZES: [Shape; 3] = [
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

/// The staged epilogue's A/B — `cargo oxide run kittens-experiments -- bench
/// staged`, which is `modal run modal_app.py::bench --case staged`.
///
/// # What is being asked, and what the answer is bounded by
///
/// #114 left the epilogue as the *whole* of the gap to cuBLASLt: the
/// epilogue-free kernel measures 1850 TFLOP/s at 8192³ against the library's
/// 1808 in the same container, and `whole − no drain` is 20.43 µs a tile
/// against #108's 21.4 µs measured serially by an unrelated route — a ratio of
/// 1.01, so the epilogue is exposed rather than hidden and cutting it pays at
/// close to full value. #109 then split it 13.1 µs of stores against 8.3 µs of
/// LDTM.
///
/// [`Epilogue::Staged`] attacks the store half only. **Its ceiling at 8192³ is
/// about 6.2%** — the stores' share of the launch — and it cannot reach even
/// that, because it does not delete the stores, it re-shapes them and buys 64
/// `stmatrix` and 32 `ld.shared` a thread to do it. [`Tile::drain_staged`] has
/// the instruction count, and the headline of it is that **total memory issue
/// does not move**: 128 either way. What moves is that global stores fall 4×
/// and land on full 128-byte lines.
///
/// # The three tables
///
/// 1. **The A/B**, staged against fused at the three sizes, with cuBLASLt in
///    the same container as the denominator both are quoted against.
/// 2. **The epilogue's exposed cost, in each arm, by the same subtraction**.
///    `staged − staged no drain` against `lcf − no drain`, in microseconds per
///    output tile on the critical path. This is the measurement the whole issue
///    turns on, and it is the one that can say *where* a null result went: an
///    epilogue that got cheaper while the launch did not is a different finding
///    from one that did not get cheaper.
/// 3. **The envelope control.** `staged no drain` against `no drain` is two
///    kernels differing in 16 424 bytes of shared memory nothing touches. It
///    should be zero — residency is 2 CTAs an SM on both, set by the
///    tensor-memory term — and if it is not, table 2's subtraction is
///    attributing a plan to a drain.
pub fn staged_sweep(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm staged epilogue (#15) — min ms over 30 timed launches, static schedule,\n\
         GROUP = {GROUP}, the shipped [256,{BLOCK_N}] @ k{BLOCK_K} s{STAGES} rung throughout.\n\
         `lcf` drains TMEM -> registers -> {} scattered 4-byte stores a thread per warp band;\n\
         `staged` drains TMEM -> registers -> stmatrix into a per-warp [32,{STAGE_N}] bf16 tile\n\
         -> 32 contiguous 16-byte stores. Same grid, same tensor memory, same item map, same\n\
         {CTAS_PER_SM} CTAs an SM; the shared plan is {SHARED_BYTES} B against {STAGED_SHARED_BYTES} B,\n\
         which is the staging tiles and is the one thing this axis cannot hold fixed.\n\
         TOTAL MEMORY ISSUE IS 128 INSTRUCTIONS A THREAD IN BOTH — see Tile::drain_staged.\n\
         The `lcf` and `staged` rows compute the GEMM and are checked element-by-element\n\
         against the CPU reference before they are timed; both `no drain` rows are UNCHECKED\n\
         and write no `C` at all.",
        BLOCK_N / 2
    );

    println!("\n1. the A/B");
    println!(
        "{:<18}{:>8}{:>12}{:>12}{:>12}{:>14}{:>10}",
        "shape", "tiles", "lcf ms", "lcf TF/s", "staged ms", "staged TF/s", "vs lcf"
    );
    let mut measured = Vec::new();
    for shape in STAGED_SIZES {
        let plan = Plan::new(Scheduler::Static);
        let fused = timed(context, shape, plan)?;
        let staged = timed(context, shape, plan.with(Epilogue::Staged))?;
        let bare = timed(context, shape, plan.ablated(Ablation::NoDrain))?;
        let staged_bare = timed(
            context,
            shape,
            plan.with(Epilogue::Staged).ablated(Ablation::NoDrain),
        )?;
        println!(
            "{:<18}{:>8}{:>12.4}{:>12.1}{:>12.4}{:>14.1}{:>10}",
            shape,
            tiles(shape.m, shape.n, BLOCK_N),
            fused,
            tflops(shape, fused),
            staged,
            tflops(shape, staged),
            format!("{:+.1}%", 100.0 * (fused / staged - 1.0))
        );
        measured.push((shape, fused, staged, bare, staged_bare));
    }

    println!(
        "\n2. what the epilogue costs in each arm, by #114's own subtraction: the launch with\n\
         the drain minus the launch without it, over the items on the critical path. The\n\
         `lcf` column is the 20.43 us/tile #114 measured at 8192^3, re-measured here as its\n\
         own control rather than quoted across sessions."
    );
    println!(
        "{:<18}{:>16}{:>18}{:>14}{:>18}",
        "shape", "lcf us/tile", "staged us/tile", "change", "of the launch"
    );
    for &(shape, fused, staged, bare, staged_bare) in &measured {
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        let (theirs, ours) = (per_tile(fused - bare), per_tile(staged - staged_bare));
        println!(
            "{:<18}{:>16.2}{:>18.2}{:>14}{:>18}",
            shape,
            theirs,
            ours,
            format!("{:+.1}%", 100.0 * (ours / theirs - 1.0)),
            format!(
                "{:.1}% -> {:.1}%",
                100.0 * (fused - bare) / fused,
                100.0 * (staged - staged_bare) / staged
            )
        );
    }

    println!(
        "\n3. the envelope control — `no drain` at {SHARED_BYTES} B against `no drain` at\n\
         {STAGED_SHARED_BYTES} B. The extra bytes are declared and never touched, so this row\n\
         is what says table 2 is comparing two drains and not two shared plans. Both are\n\
         {CTAS_PER_SM} CTAs an SM by `min(512 / columns, shared per SM / plan)`, whose binding\n\
         term at {BLOCK_N} accumulator columns is the tensor-memory one."
    );
    println!(
        "{:<18}{:>16}{:>20}{:>12}",
        "shape", "no drain ms", "staged no drain ms", "delta"
    );
    for &(shape, _, _, bare, staged_bare) in &measured {
        println!(
            "{:<18}{:>16.4}{:>20.4}{:>12}",
            shape,
            bare,
            staged_bare,
            format!("{:+.1}%", 100.0 * (staged_bare / bare - 1.0))
        );
    }

    println!(
        "\n4. against cuBLASLt on the same device in the same container. #114 measured the\n\
         epilogue-free kernel at 1850 TF/s against the library's 1808 at 8192^3, so this\n\
         column is the whole of what the epilogue is being asked to buy."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>14}{:>16}",
        "shape", "cuBLASLt ms", "theirs TF/s", "lcf/theirs", "staged/theirs"
    );
    for &(shape, fused, staged, _, _) in &measured {
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
            theirs / staged
        );
    }
    Ok(())
}

/// The four staged rungs, and the label each is printed under.
///
/// `staged` is first because it is the control: #116 shipped it and every
/// column after it is a delta against that row, not against `lcf`.
const WIDTH_RUNGS: [Epilogue; 4] = [
    Epilogue::Staged,
    Epilogue::StagedWide,
    Epilogue::StagedX4,
    Epilogue::StagedWideX4,
];

/// #117's instruction-width ablation — `cargo oxide run kittens-experiments --
/// bench widths`, which is `modal run modal_app.py::bench --case widths`.
///
/// # What is left, and why this is the half that was not touched
///
/// #116 cut the epilogue's *store* half by ~20% and left the LDTM half exactly
/// as it found it. #114 had already established what that is worth: the item
/// boundary **is** the epilogue and it is **exposed**, at a `whole − no drain`
/// to serial ratio of 1.01, and the epilogue-free kernel beats cuBLASLt
/// outright. So the LDTM half is the largest identified lever left and it pays
/// at close to full value.
///
/// # Three levers were scoped and one of them does not exist
///
/// **`.x8` (`staged8`) is real.** [`TmemTile::tile`] issues
/// `tcgen05.ld.16x256b.x1` twice per `[16, 16]` block and waits after each
/// one, so a `[32, 64]` staged band costs 16 loads and — the part that
/// matters — 16 fully exposed tensor-memory latencies, since a load's
/// registers *are* what the wait waits on and a `.x1` drain therefore
/// never has two in flight. [`TmemTile::tile_x8`] is 2 and 2.
///
/// **`.x4` (`staged4`) is real.** A [`kittens::reg::Fragment`] is four `8x8`
/// b16 matrices; `.x2` names two and `.x4` names four, at the same 32
/// addresses.
///
/// **`.pack::16b` is not a lever and the pin says so before any device time is
/// spent on it.** The scoping read it as folding the fp32→bf16 convert into
/// the load, halving the registers reaching `stmatrix`. At
/// `b099f64c1a32869b74be99f4f88242fb68655b51`,
/// `intrinsics/abi-v1.toml` gives `tcgen05_ld_16x256b_x8_pack16` the result
/// type `[u32; 32]` — **the same 32 registers as `_x8_raw`** — and
/// `intrinsics/generated-reference.md` validates it as
/// `tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 <register-list:32>`.
/// The register count does not fall, so there is no packing of two values into
/// one word for the same span. `.pack::16b` is the load-side twin of
/// `tcgen05.st`'s `.unpack::16b`: it moves **16-bit-typed** tensor memory,
/// pairing adjacent columns' half-words. Against an fp32 accumulator it is not
/// a rounding mode, it is the wrong instruction — and it is `cvt.rn.bf16x2.f32`
/// that rounds to nearest even, which is what [`check_c`] compares against with
/// `==`. No rung is built for it.
///
/// # The `.x8` hazard, and where it is actually resolved
///
/// `crates/cuda-device/src/tcgen05.rs` carries a note that
/// `tcgen05_ld_16x256b_x4/x8/x16/x32` were removed as broken — "stored to SMEM
/// instead of returning registers" — while `generated/tcgen05.rs` exposes
/// `_x4_raw`, `_x8_raw`, `_x16_raw` and `_x32_raw` at every shape. The note is
/// **stale history and not a live warning**: the removed intrinsics took a
/// shared-memory pointer, and every generated variant at this pin is
/// `status = "active"` with an array *result*, `_x8_pure` and `_x8_raw` sharing
/// one LLVM intrinsic (`llvm.nvvm.tcgen05.ld.16x256b.x8`) and one validated
/// encoding. This rung uses `_x8_pure`, which upstream blesses either way.
///
/// What is genuinely unresolved by any document is the **order** the 32
/// registers arrive in. `kittens::tmem::interleave_x8` asserts repeat-major —
/// repeat `r` is what a `.x1` at `column + 8r` would have returned — and that
/// is a claim about silicon. `device-tests`' `ldtm x8 map` holds it by draining
/// one accumulator both ways and requiring equality, and [`check`] holds it
/// again end to end, with `==` on bf16 words and no tolerance.
///
/// # The two tables
///
/// 1. **The A/B**, all four staged rungs at [`STAGED_SIZES`], each quoted
///    against `staged` and against cuBLASLt in the same container. The
///    composition row is what says whether the two widths add.
/// 2. **The epilogue's exposed cost per arm**, by #114's subtraction. There is
///    **one** control for all four: the widths change what the epilogue issues
///    and not what it occupies, so every rung here declares the same
///    [`STAGED_SHARED_BYTES`] and `staged no drain` serves them all. That is
///    the ablation #116 could not have — it had a 16 424-byte envelope change
///    to price first — and it is why table 3 of [`staged_sweep`] has no twin
///    here.
pub fn widths_sweep(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm epilogue instruction widths (#117) — min ms over 30 timed launches, static\n\
         schedule, GROUP = {GROUP}, the shipped [256,{BLOCK_N}] @ k{BLOCK_K} s{STAGES} rung\n\
         throughout. All four rows are the #116 staged epilogue — TMEM -> registers ->\n\
         stmatrix into a per-warp [32,{STAGE_N}] bf16 tile -> 32 contiguous 16-byte stores —\n\
         and differ only in how many instructions carry it:\n\
         `staged`   .x1 LDTM (16 loads, 16 waits a band), stmatrix .x2 (16 a band)\n\
         `staged8`  .x8 LDTM ( 2 loads,  2 waits a band), stmatrix .x2\n\
         `staged4`  .x1 LDTM,                             stmatrix .x4 ( 8 a band)\n\
         `staged84` .x8 LDTM,                             stmatrix .x4\n\
         Same grid, same tensor memory, same item map, same {CTAS_PER_SM} CTAs an SM, and —\n\
         unlike #116's A/B — THE SAME {STAGED_SHARED_BYTES} B SHARED PLAN in every row, so one\n\
         `no drain` control serves all four. The global half is untouched: 32 x 16 B stores\n\
         on four contiguous 128 B runs whatever the widths are.\n\
         Every row computes the GEMM and is checked element-by-element against the CPU\n\
         reference before it is timed; `check_c` compares bf16 words with `==` and no\n\
         tolerance, so a width that cost a mantissa bit would fail rather than place."
    );

    println!("\n1. the A/B, against `staged`");
    print!("{:<18}{:>8}", "shape", "tiles");
    for rung in WIDTH_RUNGS {
        print!("{:>12}", format!("{} ms", rung.name()));
    }
    for rung in WIDTH_RUNGS.into_iter().skip(1) {
        print!("{:>12}", format!("{} vs", rung.name()));
    }
    println!();

    let mut measured = Vec::new();
    for shape in STAGED_SIZES {
        let plan = Plan::new(Scheduler::Static);
        let mut arms = Vec::new();
        for rung in WIDTH_RUNGS {
            arms.push(timed(context, shape, plan.with(rung))?);
        }
        let bare = timed(
            context,
            shape,
            plan.with(Epilogue::Staged).ablated(Ablation::NoDrain),
        )?;
        print!("{:<18}{:>8}", shape, tiles(shape.m, shape.n, BLOCK_N));
        for &arm in &arms {
            print!("{:>12.4}", arm);
        }
        for &arm in arms.iter().skip(1) {
            print!("{:>12}", format!("{:+.1}%", 100.0 * (arms[0] / arm - 1.0)));
        }
        println!();
        measured.push((shape, arms, bare));
    }

    println!("\n   the same rows as throughput");
    print!("{:<18}", "shape");
    for rung in WIDTH_RUNGS {
        print!("{:>14}", format!("{} TF/s", rung.name()));
    }
    println!();
    for (shape, arms, _) in &measured {
        print!("{:<18}", shape);
        for &arm in arms {
            print!("{:>14.1}", tflops(*shape, arm));
        }
        println!();
    }

    println!(
        "\n2. what the epilogue costs in each arm, by #114's subtraction: the launch with the\n\
         drain minus the launch without it, over the items on the critical path. One control\n\
         for all four arms, because all four declare {STAGED_SHARED_BYTES} B. This is the\n\
         measurement the issue turns on — an epilogue that got cheaper while the launch did\n\
         not is a different finding from one that did not get cheaper."
    );
    print!("{:<18}", "shape");
    for rung in WIDTH_RUNGS {
        print!("{:>16}", format!("{} us/tile", rung.name()));
    }
    print!("{:>16}", "best change");
    println!();
    for (shape, arms, bare) in &measured {
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        print!("{:<18}", shape);
        let control = per_tile(arms[0] - bare);
        let mut best = control;
        for &arm in arms {
            let cost = per_tile(arm - bare);
            best = best.min(cost);
            print!("{:>16.2}", cost);
        }
        print!("{:>16}", format!("{:+.1}%", 100.0 * (best / control - 1.0)));
        println!();
    }

    let Some(baseline) = baseline else {
        println!(
            "\nno cuBLASLt column: built without --features cublas. modal_app.py::bench\n\
             turns it on, and a ratio is the point of the third table."
        );
        return Ok(());
    };
    println!(
        "\n3. against cuBLASLt on the same device in the same container. #114 measured the\n\
         epilogue-free kernel at 1850 TF/s against the library's 1808 at 8192^3, so this\n\
         column is the whole of what these widths are being asked to buy."
    );
    print!("{:<18}{:>14}{:>14}", "shape", "cuBLASLt ms", "theirs TF/s");
    for rung in WIDTH_RUNGS {
        print!("{:>16}", format!("{}/theirs", rung.name()));
    }
    println!();
    for (shape, arms, _) in &measured {
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, *shape)?.0.min();
        print!(
            "{:<18}{:>14.4}{:>14.1}",
            shape,
            theirs,
            tflops(*shape, theirs)
        );
        for &arm in arms {
            print!("{:>16.3}", theirs / arm);
        }
        println!();
    }
    Ok(())
}

/// The four rungs of the staged epilogue's own decomposition, in the order the
/// ladder differences them.
const RESIDUAL_RUNGS: [Epilogue; 4] = [
    Epilogue::StagedWideX4,
    Epilogue::StagedTwiceGlobal,
    Epilogue::StagedTwiceShared,
    Epilogue::StagedTwiceAll,
];

/// What each step of that ladder is the price of.
/// The `cvt` travels with the `stmatrix` and the column says so: doubling that
/// pass doubles `cvt.rn.bf16x2` too, because [`kittens::shared::Element::pack`]
/// is inside it. `regcount`'s census reads 8 against 16 across that step.
const RESIDUAL_PARTS: [&str; 3] = [
    "ld.shared + st.global",
    "cvt + stmatrix",
    "LDTM (issue + wait)",
];

/// Where the remaining gap to cuBLASLt lives, re-derived — `cargo oxide run
/// kittens-experiments -- bench residual`, which is `modal run
/// modal_app.py::bench --case residual`.
///
/// # Why this is a re-derivation and not a new lever
///
/// The ranking this file carries was assembled out of three containers. #114
/// measured an epilogue-free `gemm` at 1850 TFLOP/s against cuBLASLt's 1808 —
/// a ratio of **1.02**, and a ceiling, since a kernel with no epilogue cannot
/// be moved by epilogue work. #116 and #117 then cut the epilogue from 20.43 to
/// ~6.9 µs a tile, and `staged84` sits at 0.944 of the library at 8192³. On
/// that arithmetic the residual epilogue is worth about 7.7% and the gap is
/// **still 100% epilogue** — but the 1.02 predates both cuts and every number
/// in the chain was taken somewhere else.
///
/// So table 1 re-measures the ceiling, the shipped rung and the library **in
/// one container**, which is the control #109 would have needed: that section
/// would have reported a false +3.6% had it not re-run its parent after its own
/// commit moved the baseline.
///
/// # And what is inside the residual, which nothing has opened
///
/// #117 named the LDTM half and removed most of it; what remains is three
/// things nobody has priced apart. `Tile::drain_staged_twice` is the
/// instrument, and it adds rather than deletes for the reason
/// [`Epilogue::DoubleDrain`] gives: an epilogue nobody observes is an epilogue
/// the compiler is entitled to delete, so the rung that keeps the LDTM and
/// drops the stores measures an empty loop. Doubling one link of the chain at a
/// time keeps every instruction and prices the added one by subtraction.
///
/// # The five tables
///
/// 1. **The ceiling, the shipped rung and the library**, in one container, at
///    every size from 1024³ up. Both `no drain` controls run: the staged
///    envelope's, which is `staged84`'s own control, and the fused one, which
///    is the kernel #114 clocked at 1850. If `no drain / theirs` is still above
///    1 the framing holds; if it is not, that is the headline and the rest of
///    this file's ranking is wrong.
/// 2. **cuBLASLt's own spread**, because a ratio is only as stable as its
///    denominator. Min, median and max over one call's 30 launches, and a
///    second independent call's min beside them.
/// 3. **The residual epilogue, decomposed**, by the doubling ladder. Every
///    difference is a *serial* cost, and the exposed total from #114's
///    subtraction is printed beside it: the two agreeing is what says the split
///    describes the shipped launch and not just the probe.
/// 4. **Bandwidth or issue**, from `s84 hot`. #116 changed the shape of the
///    global write without changing its bytes, so if the writes were waiting on
///    HBM this rung is much faster than `staged84` and a TMA epilogue has that
///    to win; if it is not, what is left is issue and decoupling the write
///    recovers nothing.
/// 5. **The shared round trip, re-priced at #117's LDTM width.** #116's A/B was
///    taken with both arms at `.x1`, and #117 then took ~8 µs a tile out of the
///    staged arm alone. `lcf8` is the arm that A/B never had, and `lcf8`
///    against `staged8` is whether the round trip still earns its `stmatrix`
///    and its `ld.shared`.
/// 6. **The instrument's own reproducibility**, from the two places tables 1
///    and 3 both take the same subtraction. No extra launch.
/// 7. **The TMA store** (#123), which is the one thing table 3 leaves with a
///    number on it. Two arms — [`Epilogue::StagedTma`] at warp scope and
///    [`Epilogue::StagedTmaCta`] at CTA scope — against `staged84`, and then
///    the hop itself priced by addition on both store shapes so the two
///    doubling differences can be read against each other rather than against
///    a whole epilogue.
/// 8. **The same A/B at a share it can be measured at** — [`tma_ordering`],
///    #122's correction applied to table 7 rather than to the epilogue
///    subtraction, and the table that decides whether table 7's fraction of a
///    percent is a result or a rounding.
pub fn residual_sweep(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<crate::bench::Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm: where the gap to cuBLASLt lives, re-derived in one container — min ms over\n\
         30 timed launches, static schedule, GROUP = {GROUP}, the shipped [256,{BLOCK_N}] @ k{BLOCK_K}\n\
         s{STAGES} rung throughout. Every ratio this file quotes for the epilogue was\n\
         assembled across three containers and the 1.02 ceiling predates both #116 and\n\
         #117, so nothing below is inherited: the controls are re-run beside the arms."
    );

    let Some(baseline) = baseline else {
        println!(
            "\nthis sweep is a denominator and cannot run without one: built with no\n\
             --features cublas. modal_app.py::bench turns it on."
        );
        return Ok(());
    };
    println!("\nthe denominator: {}", (baseline.about)());

    println!(
        "\n1. the ceiling, the shipped rung and the library, all in this container. `no drain`\n\
         is the kernel with its epilogue deleted — the fused one is #114's 1850 TFLOP/s rung\n\
         at {SHARED_BYTES} B, the staged one is `staged84`'s own control at {STAGED_SHARED_BYTES} B,\n\
         and neither computes a GEMM. A `no drain / theirs` above 1 is what makes \"the gap is\n\
         the epilogue\" arithmetic rather than a hope."
    );
    println!(
        "{:<18}{:>7}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>9}{:>9}{:>9}{:>9}",
        "shape",
        "tiles",
        "wave",
        "s84 ms",
        "s8 ms",
        "s84nd ms",
        "lcfnd ms",
        "theirs ms",
        "s84/th",
        "s8/th",
        "s84nd/th",
        "lcfnd/th"
    );
    let plan = Plan::new(Scheduler::Static);
    let mut ceiling = Vec::new();
    for shape in EPILOGUE_SIZES {
        let shipped = timed(context, shape, plan.with(Epilogue::StagedWideX4))?;
        let wide = timed(context, shape, plan.with(Epilogue::StagedWide))?;
        let staged_bare = timed(
            context,
            shape,
            plan.with(Epilogue::Staged).ablated(Ablation::NoDrain),
        )?;
        let fused_bare = timed(context, shape, plan.ablated(Ablation::NoDrain))?;
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0;
        let (waves, efficiency) = wave_efficiency(shape.m, shape.n);
        println!(
            "{:<18}{:>7}{:>8}{:>10.4}{:>10.4}{:>10.4}{:>10.4}{:>10.4}{:>9.3}{:>9.3}{:>9.3}{:>9.3}",
            shape,
            tiles(shape.m, shape.n, BLOCK_N),
            format!("{waves}x{:.0}%", 100.0 * efficiency),
            shipped,
            wide,
            staged_bare,
            fused_bare,
            theirs.min(),
            theirs.min() / shipped,
            theirs.min() / wide,
            theirs.min() / staged_bare,
            theirs.min() / fused_bare
        );
        ceiling.push((shape, shipped, staged_bare, fused_bare, theirs));
    }

    println!("\n   the same rows as throughput");
    println!(
        "{:<18}{:>14}{:>14}{:>14}{:>14}",
        "shape", "s84 TF/s", "s84nd TF/s", "lcfnd TF/s", "theirs TF/s"
    );
    for (shape, shipped, staged_bare, fused_bare, theirs) in &ceiling {
        println!(
            "{:<18}{:>14.1}{:>14.1}{:>14.1}{:>14.1}",
            shape,
            tflops(*shape, *shipped),
            tflops(*shape, *staged_bare),
            tflops(*shape, *fused_bare),
            tflops(*shape, theirs.min())
        );
    }

    println!(
        "\n2. the denominator's own spread. Min, median and max over the 30 launches of one\n\
         call, then a second independent call — new buffers, new handle, new heuristic query\n\
         — so a size where the library is not one number says so here rather than in a ratio."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>10}{:>12}{:>10}",
        "shape", "min ms", "median", "max", "max/min", "again ms", "call/call"
    );
    for (shape, _, _, _, theirs) in &ceiling {
        eprintln!("{shape}: staging and checking {} again", baseline.name);
        let again = (baseline.bench)(context, *shape)?.0.min();
        println!(
            "{:<18}{:>12.4}{:>12.4}{:>12.4}{:>10.2}{:>12.4}{:>10.3}",
            shape,
            theirs.min(),
            theirs.median(),
            theirs.max(),
            theirs.max() / theirs.min(),
            again,
            again / theirs.min()
        );
    }

    println!(
        "\n3. the residual epilogue, decomposed. Each rung doubles one more link of\n\
         `Tile::drain_staged_twice`'s chain, so each difference is what the added link\n\
         costs **serially**, with the second pass's stores aimed at the cluster's own tile so\n\
         its bytes stay in L2. `exposed` is #114's subtraction — `staged84` minus `staged no\n\
         drain` — and the two columns agreeing is what says the split describes the shipped\n\
         launch. Microseconds per output tile on the critical path."
    );
    print!("{:<18}", "shape");
    for part in RESIDUAL_PARTS {
        print!("{:>24}", part);
    }
    println!("{:>14}{:>14}", "serial total", "exposed");
    let mut ladder = Vec::new();
    for shape in STAGED_SIZES {
        let mut arms = Vec::new();
        for rung in RESIDUAL_RUNGS {
            arms.push(timed(context, shape, plan.with(rung))?);
        }
        let bare = timed(
            context,
            shape,
            plan.with(Epilogue::Staged).ablated(Ablation::NoDrain),
        )?;
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        print!("{:<18}", shape);
        for step in 0..3 {
            print!("{:>24.2}", per_tile(arms[step + 1] - arms[step]));
        }
        println!(
            "{:>14.2}{:>14.2}",
            per_tile(arms[3] - arms[0]),
            per_tile(arms[0] - bare)
        );
        ladder.push((shape, arms, bare));
    }

    println!("\n   the rungs themselves, as min ms, so the differences above can be checked");
    print!("{:<18}", "shape");
    for rung in RESIDUAL_RUNGS {
        print!("{:>14}", format!("{} ms", rung.name()));
    }
    println!("{:>14}", "s no drain");
    for (shape, arms, bare) in &ladder {
        print!("{:<18}", shape);
        for arm in arms {
            print!("{:>14.4}", arm);
        }
        println!("{:>14.4}", bare);
    }

    println!(
        "\n4. bandwidth or issue, on the store shape #116 built. `s84 hot` is `staged84` with\n\
         every global store aimed at the cluster's own first tile — identical in every\n\
         instruction, and the only thing that moves is that 148 clusters rewrite one tile\n\
         each instead of streaming the whole of `C`. A large gain means the writes are\n\
         waiting on HBM and a TMA epilogue has that to win; a small one means what is left\n\
         is issue, and decoupling the write recovers none of it."
    );
    println!(
        "{:<18}{:>14}{:>14}{:>12}{:>18}{:>18}",
        "shape", "staged84 ms", "s84 hot ms", "hot vs", "s84 us/tile", "hot us/tile"
    );
    for (shape, arms, bare) in &ladder {
        let hot = timed(context, *shape, plan.with(Epilogue::StagedHot))?;
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        println!(
            "{:<18}{:>14.4}{:>14.4}{:>12}{:>18.2}{:>18.2}",
            shape,
            arms[0],
            hot,
            format!("{:+.1}%", 100.0 * (arms[0] / hot - 1.0)),
            per_tile(arms[0] - bare),
            per_tile(hot - bare)
        );
    }

    println!(
        "\n5. the shared round trip, re-priced at #117's LDTM width. #116's A/B ran both arms\n\
         at `.x1` and won +4.2% at 8192³; #117 then took ~8 us a tile out of the staged arm\n\
         alone. `lcf8` is the register drain at `.x8` — the arm that A/B never had — and the\n\
         question is whether `stmatrix` + `ld.shared` still pay for themselves against a\n\
         drain that no longer waits sixteen times a band. Every row here computes the GEMM."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}{:>14}{:>14}",
        "shape", "lcf ms", "lcf8 ms", "staged ms", "staged8 ms", "x1: st vs lcf", "x8: st vs lcf"
    );
    let mut trip = Vec::new();
    for (shape, arms, _) in &ladder {
        let lcf = timed(context, *shape, plan.with(Epilogue::Fused))?;
        let lcf8 = timed(context, *shape, plan.with(Epilogue::FusedWide))?;
        let staged = timed(context, *shape, plan.with(Epilogue::Staged))?;
        let staged8 = timed(context, *shape, plan.with(Epilogue::StagedWide))?;
        println!(
            "{:<18}{:>12.4}{:>12.4}{:>12.4}{:>12.4}{:>14}{:>14}",
            shape,
            lcf,
            lcf8,
            staged,
            staged8,
            format!("{:+.1}%", 100.0 * (lcf / staged - 1.0)),
            format!("{:+.1}%", 100.0 * (lcf8 / staged8 - 1.0)),
        );
        trip.push((*shape, lcf, lcf8, staged, staged8, arms[0]));
    }

    println!("\n   and every arm of table 5 against the library, from table 1's own denominator");
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "shape", "lcf/th", "lcf8/th", "staged/th", "staged8/th", "s84/th"
    );
    for (shape, lcf, lcf8, staged, staged8, shipped) in &trip {
        let Some((_, _, _, _, theirs)) = ceiling.iter().find(|row| same(row.0, *shape)) else {
            continue;
        };
        println!(
            "{:<18}{:>12.3}{:>12.3}{:>12.3}{:>12.3}{:>12.3}",
            shape,
            theirs.min() / lcf,
            theirs.min() / lcf8,
            theirs.min() / staged,
            theirs.min() / staged8,
            theirs.min() / shipped
        );
    }

    println!(
        "\n6. the instrument's own reproducibility, which nothing in this file has ever\n\
         measured. Tables 1 and 3 both timed `staged84` and both timed `staged no drain`,\n\
         minutes apart in this container, so the same subtraction was taken twice — and an\n\
         epilogue figure is only as good as the spread between them. No extra launch: this\n\
         is the two tables read against each other."
    );
    println!(
        "{:<18}{:>10}{:>10}{:>10}{:>10}{:>14}{:>14}{:>10}",
        "shape", "s84 (1)", "s84 (3)", "nd (1)", "nd (3)", "us/tile (1)", "us/tile (3)", "spread"
    );
    for (shape, arms, bare) in &ladder {
        let Some((_, shipped, staged_bare, _, _)) = ceiling.iter().find(|row| same(row.0, *shape))
        else {
            continue;
        };
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        let (first, second) = (per_tile(shipped - staged_bare), per_tile(arms[0] - bare));
        println!(
            "{:<18}{:>10.4}{:>10.4}{:>10.4}{:>10.4}{:>14.2}{:>14.2}{:>10}",
            shape,
            shipped,
            arms[0],
            staged_bare,
            bare,
            first,
            second,
            format!("{:+.0}%", 100.0 * (second / first - 1.0))
        );
    }

    println!(
        "\n7. the TMA store, which is the one lever table 3 left with a number on it. Both arms\n\
         keep `staged84`'s `.x8` LDTM, its `.x4` `stmatrix` and its staging tiles, and replace\n\
         32 `ld.shared.v4` + 32 `st.global.v4` a thread a band with one `cp.async.bulk.tensor`.\n\
         `s84 tma` keeps the tile **per warp**, so the ring is warp-scope and the item still\n\
         pays no block barrier; `s84 tmac` reads the same 16 384 B as one [128,64] tile through\n\
         `kittens::epilogue::StoreRing` at the CTA scope #111 built it at — one TMA instruction\n\
         a band instead of four, and eight `bar.sync` an item instead of none. Both compute the\n\
         GEMM and both are checked. Same 114 816 B and the same `staged no drain` control, so\n\
         the µs/tile columns are the same subtraction table 3's `exposed` is."
    );
    println!(
        "{:<18}{:>11}{:>11}{:>11}{:>10}{:>10}{:>12}{:>12}{:>12}",
        "shape",
        "s84 ms",
        "tma ms",
        "tmac ms",
        "tma vs",
        "tmac vs",
        "s84 us/t",
        "tma us/t",
        "tmac us/t"
    );
    let mut engine = Vec::new();
    for (shape, arms, bare) in &ladder {
        let tma = timed(context, *shape, plan.with(Epilogue::StagedTma))?;
        let cta = timed(context, *shape, plan.with(Epilogue::StagedTmaCta))?;
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        println!(
            "{:<18}{:>11.4}{:>11.4}{:>11.4}{:>10}{:>10}{:>12.2}{:>12.2}{:>12.2}",
            shape,
            arms[0],
            tma,
            cta,
            format!("{:+.1}%", 100.0 * (arms[0] / tma - 1.0)),
            format!("{:+.1}%", 100.0 * (arms[0] / cta - 1.0)),
            per_tile(arms[0] - bare),
            per_tile(tma - bare),
            per_tile(cta - bare)
        );
        engine.push((*shape, arms[0], arms[1], tma, cta, *bare));
    }

    println!(
        "\n   and the hop itself, priced by addition on both store shapes. `s84 2g` is table 3's\n\
         first rung — a second `store_shared_rows` per band — and `s84t 2g` is the same probe\n\
         with a second TMA store instead, fence and barrier and commit included, both aimed at\n\
         the cluster's own tile so the extra bytes stay in L2. The lever is worth something\n\
         exactly when the second column is smaller than the first, and the difference between\n\
         them is the most this change can ever recover."
    );
    println!(
        "{:<18}{:>18}{:>18}{:>14}{:>16}",
        "shape", "ld+st us/tile", "tma hop us/tile", "the lever", "of the epilogue"
    );
    for (shape, shipped, doubled, tma, _, bare) in &engine {
        let doubled_tma = timed(context, *shape, plan.with(Epilogue::StagedTmaTwiceGlobal))?;
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        let (plain, engine_hop) = (per_tile(doubled - shipped), per_tile(doubled_tma - tma));
        println!(
            "{:<18}{:>18.2}{:>18.2}{:>14.2}{:>16}",
            shape,
            plain,
            engine_hop,
            plain - engine_hop,
            format!(
                "{:.0}%",
                100.0 * (plain - engine_hop) / per_tile(shipped - bare)
            )
        );
    }

    println!("\n   and both arms against the library, from table 1's own denominator");
    println!(
        "{:<18}{:>12}{:>12}{:>12}",
        "shape", "s84/th", "tma/th", "tmac/th"
    );
    for (shape, shipped, _, tma, cta, _) in &engine {
        let Some((_, _, _, _, theirs)) = ceiling.iter().find(|row| same(row.0, *shape)) else {
            continue;
        };
        println!(
            "{:<18}{:>12.3}{:>12.3}{:>12.3}",
            shape,
            theirs.min() / shipped,
            theirs.min() / tma,
            theirs.min() / cta
        );
    }

    tma_ordering(context)
}

/// The two shortened geometries #122 built, and the cube they replace.
///
/// `M` and `N` are what an epilogue's cost is a function of — the tile grid,
/// the waves, the items a cluster walks and the whole of `C` — and `K` is the
/// only axis it does not sit on. So these hold the output geometry of the two
/// sizes every epilogue claim in this file is quoted at and shorten the
/// reduction, which leaves the difference the same absolute size over a launch
/// it is a tenth of instead of a thousandth. `bench repro`'s own reasoning,
/// applied to the comparison this section is about.
const TMA_SIZES: [Shape; 3] = [
    Shape {
        m: 8192,
        n: 8192,
        k: 1024,
    },
    Shape {
        m: 16384,
        n: 16384,
        k: 1024,
    },
    // The anchor: the one full-depth size §7 trusts, so the shortened rows are
    // judged against a number this file already carries.
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
];

/// `staged84` and the two TMA arms, in the order the ratios are taken.
const TMA_ARMS: [Epilogue; 3] = [
    Epilogue::StagedWideX4,
    Epilogue::StagedTma,
    Epilogue::StagedTmaCta,
];

/// Whole measurements each arm gets, round-robin over the arms — `bench
/// repro`'s count and its reason. A device whose clocks step down over a sweep
/// slows whichever arm ran last, and `A,A,A,A,B,B,B,B` cannot tell that from a
/// difference between `A` and `B`; `A,B,C,A,B,C,…` puts each triple adjacent in
/// time, so the four paired ratios carry any drift as a common term.
const TMA_REPEATS: usize = 4;

/// The TMA A/B taken the way a *ratio between two whole launches* has to be
/// taken — table 8 of [`residual_sweep`].
///
/// # Why the tables above are not enough on their own
///
/// #122 established that a difference carries its arms' error divided by its
/// own share of them, and that the correction is to take it where the share is
/// large rather than to sample harder. That argument is usually made about the
/// epilogue *subtraction*, but it binds hardest here: `s84 tma` against
/// `staged84` is a fraction of a percent between two 0.65 ms launches, which is
/// a smaller share than the epilogue is and therefore a *harder* measurement
/// than `whole − no drain`. Table 7's `+0.2% / −0.1%` is one min against
/// another, arm by arm, at the depth where that share is worst.
///
/// So this repeats it at the geometry the question is about — same tiles, same
/// waves, same items, same `C`, same epilogue in absolute terms — with `K`
/// shortened until the difference is visible, four paired ratios a cell, and
/// the arms interleaved. It is the shape of #122's own `staged8`/`staged84`
/// ordering table, asked of a different pair.
///
/// A ratio whose lowest-to-highest range straddles 1.000 orders nothing, and
/// the table says so in its own column rather than leaving it to be read off
/// the digits.
fn tma_ordering(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\n8. the same A/B where the difference is a tenth of the launch instead of a\n\
         thousandth. #122: a difference carries its arms' error over its own share of them,\n\
         and `tma` against `staged84` is a *smaller* share of the launch than the epilogue\n\
         is — so table 7 is the harder measurement, not the easier one. The two `k1024`\n\
         shapes are 8192² and 16384²'s own output geometries with the reduction shortened:\n\
         same tiles, same waves, same items a cluster walks, same `C`, same epilogue in\n\
         absolute terms, over a launch it is a large fraction of. {TMA_REPEATS} whole\n\
         measurements of each arm, taken round-robin so every triple is adjacent in time and\n\
         a drift enters all three sides of it."
    );
    println!(
        "{:<20}{:<12}{:>11}{:>11}{:>11}{:>10}{:>10}",
        "shape", "arm", "best ms", "worst ms", "call/call", "in-call", "drift"
    );
    let mut sweep = Vec::new();
    for shape in TMA_SIZES {
        let mut taken: Vec<Vec<Timings>> = TMA_ARMS.iter().map(|_| Vec::new()).collect();
        for pass in 1..=TMA_REPEATS {
            for (arm, into) in TMA_ARMS.iter().zip(taken.iter_mut()) {
                eprintln!("{shape} {} pass {pass}: staging and checking", arm.name());
                into.push(bench_with(context, shape, *arm)?);
            }
        }
        for (arm, calls) in TMA_ARMS.iter().zip(taken.iter()) {
            let bests: Vec<f64> = calls.iter().map(Timings::min).collect();
            let (best, worst) = span(&bests);
            let widest = calls.iter().map(Timings::spread).fold(0.0, f64::max);
            let drifted = calls
                .iter()
                .map(|call| call.drift())
                .fold(
                    0.0f64,
                    |most, next| {
                        if next.abs() > most.abs() { next } else { most }
                    },
                );
            println!(
                "{:<20}{:<12}{:>11.4}{:>11.4}{:>10.2}%{:>9.2}%{:>9.2}%",
                shape,
                arm.name(),
                best,
                worst,
                100.0 * (worst / best - 1.0),
                100.0 * widest,
                100.0 * drifted,
            );
        }
        sweep.push((shape, taken));
    }

    println!(
        "\n   the paired ratios, `staged84` over each TMA arm — above 1.000 is the TMA arm\n\
         ahead. Four pairs a cell, each one taken adjacent in time, printed as the median\n\
         and the whole range. `1/share` is what the difference multiplies the arms' error\n\
         by, and a range that straddles 1.000 orders nothing whatever its median reads."
    );
    println!(
        "{:<20}{:>10}{:>20}{:>10}{:>10}{:>20}{:>10}",
        "shape", "s84/tma", "lowest–highest", "1/share", "s84/tmac", "lowest–highest", "1/share"
    );
    for (shape, taken) in &sweep {
        print!("{shape:<20}");
        for arm in 1..TMA_ARMS.len() {
            let ratios: Vec<f64> = taken[0]
                .iter()
                .zip(&taken[arm])
                .map(|(base, other)| base.min() / other.min())
                .collect();
            let (low, high) = span(&ratios);
            let centre = middle(&ratios);
            print!(
                "{:>10.4}{:>20}{:>10.0}",
                centre,
                format!("{low:.4} – {high:.4}"),
                1.0 / (centre - 1.0).abs()
            );
        }
        println!();
    }
    Ok(())
}

/// Lowest and highest of a set of repeats.
fn span(over: &[f64]) -> (f64, f64) {
    over.iter()
        .copied()
        .fold((f64::MAX, f64::MIN), |(low, high), next| {
            (low.min(next), high.max(next))
        })
}

/// The middle of a set of repeats, which is what a range is quoted around.
fn middle(of: &[f64]) -> f64 {
    let mut sorted = of.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// The traversal sweep — `cargo oxide run kittens-experiments -- bench swizzle`,
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
                ablation: Ablation::Whole,
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
                ablation: Ablation::Whole,
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
                    ablation: Ablation::Whole,
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
