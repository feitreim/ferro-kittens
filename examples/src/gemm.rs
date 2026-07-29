//! # GEMM — `C = A·Bᵀ` on the cta_group::2 cluster path
//!
//! **Status: runs.** [`check`] launches it against a CPU reference on a B200
//! (`scripts/modal-run examples`). The operands are small integers, so every
//! product and every partial sum is exact and the comparison is `==` rather
//! than a tolerance — a mismatch is a wrong index, never rounding.
//!
//! This is **one kernel**: the one the library ships, written out. Every rung
//! it was chosen over, every ablation that priced a phase of it and every probe
//! that computes a deliberately wrong `C` to isolate a term lives in
//! `experiments/`, and `experiments/README.md` §7 is where each number below
//! comes from. What is here is the shape a reader has to hold in their head to
//! write a kernel like it.
//!
//! ## The cluster is the unit, not the CTA
//!
//! A pair of CTAs forms a cluster and shares one `M256_N256` UMMA. Both
//! operands are split across the pair — each CTA stages its own 128 rows of
//! `A` and its own 128 columns of `B` at the *same* shared offsets, and the
//! instruction reads both CTAs' shared memory over the cluster interconnect.
//! The accumulator splits the same way along M, so each CTA drains its own
//! `[128, 256]` band of `C`.
//!
//! That is why [`RANKS`] is a constant of the [`Job`] and not a launch
//! parameter: the item boundary has to be `barrier.cluster` rather than
//! `bar.sync`, so no rank re-arms a barrier its peer is still filling and no
//! rank fills one its peer has not re-armed.
//!
//! ## The grid is persistent, and the work item is an output tile
//!
//! One output tile used to be one launch's worth of cluster. It is now
//! [`kittens::pipeline::run`] over a [`Tile`] job on a grid capped at
//! [`MAX_CLUSTERS`]: a cluster that runs out of items is the whole schedule,
//! and a cluster that has more walks them. The cap is a *measurement* — see
//! [`CTAS_PER_SM`] — and picking it wrong costs 2×.
//!
//! `pipeline::run` hands the job an item index and nothing else.
//! [`pipeline::grouped`] turns that into a tile, in blocks of [`GROUP`]
//! tile-rows rather than row-major, and that map is worth 23% at a fixed flop
//! count: five shapes identical in tiles, waves, grid and bytes but differing
//! in `M : N` moved 1123.7 → 915.7 TFLOP/s, best and worst exact transposes of
//! each other. Nothing but the order the output is walked in can do that.
//!
//! ## K is software-pipelined, and the MMA is what releases the operand
//!
//! K is [`STAGES`] deep over a [`SharedTileRing`] pair, with [`SemaphoreRing`]
//! owning the `index → (stage, parity)` arithmetic on both sides. `load` is
//! filled by the TMA and drained by the MMA; `free` is released by the MMA's
//! own commit — **the accumulator instruction, not a thread, is what proves the
//! operand has been read**. A thread arriving there would only prove the
//! instruction was issued.
//!
//! The pair's four TMA loads have to complete on *one* barrier for the leader
//! to know the whole stage is present, and a plain `cp.async.bulk.tensor`
//! completes on a barrier in the *issuing* CTA's own shared memory. Both halves
//! are named rather than open-coded: [`Semaphore::at_rank`] is the leader's copy
//! of the stage barrier and [`SharedTile::tma_load_2d_arriving_at`] is a load
//! that completes there. The charge stays the leader's and is its own charge
//! [`RANKS`] times over.
//!
//! ## The epilogue goes through shared memory, and that is most of the speed
//!
//! [`Tile::drain`] moves this warp's band TMEM → registers → `stmatrix` into a
//! per-warp `[32, 64]` shared tile → 16-byte stores to `C`, where the obvious
//! epilogue stores 4 bytes a thread straight out of registers. **Total memory
//! issue does not fall — it is 128 instructions a thread either way.** The whole
//! gain is that the global stores are 4× fewer and land on full 128-byte lines
//! instead of on eight half-filled 32-byte sectors: the band's writes go from
//! 1024 half-full transactions to 512 full ones.
//!
//! Two instruction widths on top of it, and both are about the *wait*:
//!
//! - **`tcgen05.ld.16x256b.x8`**, [`TmemTile::tile_x8`]. The `.x1` load waits
//!   after each issue because the registers it waits on *are* the load's return
//!   value, so a `[32, 64]` band costs 16 loads and 16 fully exposed
//!   tensor-memory latencies. `.x8` is 2 and 2.
//! - **`stmatrix.m8n8.x4`**, [`kittens::ldst::store_tile_x4`]. A
//!   [`kittens::reg::Fragment`] is four `8x8` b16 matrices and `.x2` names two,
//!   so `.x4` halves an instruction count at identical addresses.
//!
//! Together those are +23.1% / +8.8% / +5.1% at 4096³ / 8192³ / 16384³ over the
//! staged drain without them, which is 1645.0 and 1754.6 TFLOP/s and **0.939
//! and 0.958 of cuBLASLt** at 8192³ and 16384³. `ptxas` reads 80 registers and
//! no spill.
//!
//! There is **no proxy fence** in the drain and its absence is not an oversight:
//! `fence.proxy.async.shared::cta` orders a generic-proxy write against an
//! *async*-proxy read, and both ends here are generic — `stmatrix` writes and
//! `ld.shared` reads. `stmatrix.sync.aligned` is the convergence. The one hazard
//! left is the next pass overwriting a tile this pass is still reading, which is
//! what the `bar.warp.sync` is for, and it is a warp barrier because
//! [`StageTile`] is one warp's 4096 B and nobody else's. Handing the same hop to
//! the TMA engine instead was built, measured and lost by 1.0–1.7% (#123).
//!
//! ## `C` is bf16 and the accumulator is not
//!
//! `bf16` in, fp32 accumulate, `bf16` out is the signature a training GEMM has.
//! [`Accumulator`] is [`BLOCK_N`] fp32 columns of tensor memory and the single
//! `cvt.rn.bf16x2.f32` is inside [`kittens::ldst::store_tile_x4`], on the way
//! into shared. So there is exactly one rounding in the kernel, and [`check_c`]
//! puts the same one in the reference — which is what lets the comparison stay
//! `==` on 16-bit words with no tolerance at all.
//!
//! ## What this kernel had to reach past the library for
//!
//! **Nothing.** There is no open-coded index arithmetic and no `GAP` block. The
//! epilogue was twelve lines of hand-written addressing against
//! `RegTile::coordinate` until #11; the cluster-scope tensor-memory allocation
//! beside it is [`kittens::tmem::alloc_cluster`] / [`dealloc_cluster`] since #24
//! and #46, and this file is where that allocator's participation rules were
//! worked out against silicon.

use cuda_device::barrier::Barrier;
use cuda_device::cluster;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{
    DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, thread, warp,
};

use std::error::Error;

use kittens::global::{GlobalRows, store_shared_rows};
use kittens::ldst::store_tile_x4;
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, Job};
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};

/// Rows of `C` one CTA owns. The pair covers `2 * BLOCK_M`, which is the `M`
/// the instruction descriptor names — and the widest `M` tcgen05 has, which is
/// why a tile sweep can only move the columns.
const BLOCK_M: usize = 128;
/// Columns of `C` the *pair* computes, split [`HALF_N`] per CTA along the
/// operand's N axis.
///
/// It was 128 through #102 and a sweep moved it: `[256, 256]` measured +11.6%
/// at 8192³ and +21.6% at 16384³ against `[256, 128]`, on arithmetic intensity
/// and nothing else — `M·N/(M+N)` flops per operand byte, which doubling `N`
/// buys 1.5× of. What it costs is generality, and that is not free: a launch
/// must have `n % 256 == 0` where 128 used to do.
const BLOCK_N: usize = 256;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K one linear MMA walk covers: **one 128-byte swizzle atom of bf16, and the
/// only width [`SharedTile::k_walk`] accepts.**
///
/// `k_walk` carries `const { assert!(C * E::BYTES == S::ATOM_BYTES, ..) }`, so
/// 64 is what a *walk* is at bf16 and a stage wanting more K than that gets it
/// by holding several of these atoms rather than by widening the walk.
const ATOM_K: usize = 64;
/// Chained K=16 MMA chunks in one atom's walk.
const ATOM_CHUNKS: usize = ATOM_K / 16;
/// K a pipeline stage carries — a whole number of [`ATOM_K`] atoms, loaded by
/// one TMA per operand and multiplied by one walk per atom.
const BLOCK_K: usize = 64;
/// Pipeline depth over K.
///
/// `BLOCK_K · STAGES` is what the shared budget actually caps, at 228 for this
/// pair tile, so the two are a *factorization* of a fixed number of bytes in
/// flight rather than independent axes. `BLOCK_K` does not move arithmetic
/// intensity at all — a tile reads `(M + N) · K` bytes for `2 · M · N · K`
/// flops however K is blocked — so what it trades against depth is stage
/// barriers and `expect_tx` charges against how coarsely the ring recycles.
const STAGES: usize = 3;
/// One warp per 32 accumulator rows, which is what a `[32, N]` drain wants.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32;
/// Accumulator columns one warp drains in a single band.
///
/// The band goes to shared memory through `stmatrix`, so its width is the
/// staging tile's width and `SharedTile::WIDTH_OK` wants a whole swizzle
/// subtile: **at bf16 under `Swizzle128B` 64 columns is the narrowest tile that
/// exists**, and the widest one the budget admits. At 2 CTAs an SM a CTA gets
/// `233 472 / 2 = 116 736` B and [`SHARED_BYTES`] spends 98 368 of it; four
/// warps × `[32, 64]` bf16 is 16 384 B and fits, and anything wider is 32 768
/// and does not.
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
/// The band a drain pass holds: [`STAGE_N`] columns, so 64 fp32 a thread.
type StagedBand = RegTile<32, STAGE_N, BaseLdtm>;
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
/// This CTA's `B` columns, also K-major — so the MMA carries no transpose bits
/// and computes `A·Bᵀ`.
type BTile = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// One swizzle atom of an operand tile, which is what a [`SharedTile::k_walk`]
/// can describe — a `BLOCK_K` wider than [`ATOM_K`] is this many of these,
/// stacked, and [`SharedTile::subtile`] is where each one starts.
type Atom<const R: usize> = SharedTile<Bf16, R, ATOM_K, Swizzle128B>;
/// This CTA's half of the pair's accumulator: 128 TMEM lanes by [`BLOCK_N`]
/// fp32 columns. The column count is what charges tensor memory, so it is also
/// the `512 / columns` half of the residency this kernel gets.
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

/// Barriers and the TMEM staging word, in the tail of the shared plan: the two
/// `stages`-deep rings' semaphores, the MMA-complete semaphore, and one `u32`.
const fn scratch_bytes(stages: usize) -> usize {
    2 * stages * 8 + 8 + 8
}

/// Dynamic shared memory the operand rings and the scratch tail ask for.
///
/// Stated as arithmetic because `#[launch_contract]` takes a literal and this
/// is where the two have to agree. It is not trusted either: [`kernels::attach`]
/// carries a codegen-time assert against the rings' own `BYTES`, which is the
/// only place the two could ever drift.
const fn shared_plan(block_n: usize, block_k: usize, stages: usize) -> usize {
    let a = BLOCK_M * block_k * 2 * stages;
    let b = (block_n / 2) * block_k * 2 * stages;
    a + b + scratch_bytes(stages)
}

/// The UMMA shape this pair tile issues.
///
/// A tile whose columns name no shape fails at codegen rather than issuing the
/// wrong descriptor into the right accumulator, which does not fault and
/// computes wrong numbers.
const fn pair_shape(block_n: usize) -> MmaShape {
    match block_n {
        128 => MmaShape::M256_N128,
        256 => MmaShape::M256_N256,
        _ => panic!("no cta_group::2 MmaShape covers this pair tile's columns"),
    }
}

/// The operand half of the envelope: everything but the staging run.
const SHARED_BYTES: usize = shared_plan(BLOCK_N, BLOCK_K, STAGES);
const _: () = assert!(THREADS == 128 && SHARED_BYTES == 98_368);

/// Where the first warp's [`StageTile`] starts, as a byte offset into the
/// launch's dynamic shared memory.
///
/// Rounded up to the 128-byte alignment a [`SharedTile`] base owes. The
/// staging run would otherwise start at the end of an 8-byte semaphore, which
/// is not a swizzle atom's alignment and would put the phase
/// [`kittens::shared::SwizzledChunks`] derives from the base somewhere the
/// `stmatrix` and the read-back would still agree on but no reader could check.
const STAGE_OFFSET: usize = SHARED_BYTES.next_multiple_of(128);

/// Dynamic shared memory the launch declares — the second literal
/// `#[launch_contract]` needs.
///
/// **It must stay at or under 116 736 B**, which is the 233 472 an SM has
/// divided by the 2 CTAs [`CTAS_PER_SM`] counts. 1920 B of headroom is left,
/// which is why the staging ring is one buffer deep: another is 16 384.
pub const STAGED_SHARED_BYTES: usize = STAGE_OFFSET + (THREADS as usize / 32) * (32 * STAGE_N * 2);

/// The envelope, the staging run and the alignment are three spellings of the
/// same bytes, and this is where they have to agree.
///
/// The contract is not decoration: 112 KiB is far past the 48 KiB a block gets
/// by default, and the opt-in (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`)
/// is issued by the prepared-launch path. `cluster_launch` has nothing to do
/// with it; [`kittens::launch::admit_shared_plan`] is the same opt-in for a
/// kernel whose output partition no contract describes.
const _: () = {
    let run = (THREADS as usize / 32) * StageTile::BYTES;
    assert!(run == 16_384);
    assert!(STAGE_OFFSET >= SHARED_BYTES && STAGE_OFFSET.is_multiple_of(128));
    assert!(STAGED_SHARED_BYTES == 114_816 && STAGED_SHARED_BYTES <= 116_736);
    assert!(STAGE_OFFSET + run == STAGED_SHARED_BYTES);
    assert!(STAGED_SHARED_BYTES + run > 116_736);
};

/// SMs on the device this project targets and measures on — a B200.
const SMS: u32 = 148;
/// CTAs of this kernel one SM holds at once — **measured, and then counted.**
///
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` takes a block shape and no
/// cluster, so it cannot answer about a `#[cluster_launch]` kernel; `main.rs`
/// prints `cluster` in this kernel's row for exactly that reason.
///
/// It was found by bisecting the grid cap on a clock — one CTA per SM made
/// 8192³ take 2.1036 ms, two gave 1.2130, three 1.0217 — and confirmed by
/// `device-tests`' `tmem residency census`, which timestamps both ends of every
/// CTA's tensor-memory allocation off `%globaltimer` and sweeps the intervals
/// for the most ever open at once. Two methods sharing nothing, landing on one
/// integer.
///
/// At [`BLOCK_N`] `= 256` the binding term of #84's
/// `min(512 / columns, shared per SM / plan)` is the **tensor memory** one:
/// `512 / 256` is 2 before shared memory is consulted at all. That is what
/// makes [`STAGED_SHARED_BYTES`]' extra 16 KiB free — the census counts the
/// same 2 at both envelopes.
const CTAS_PER_SM: u32 = 2;
/// Clusters the persistent grid launches at most, past which a cluster takes a
/// second work item rather than the scheduler holding a pair back.
///
/// This is a *tuning* constant and not a correctness one: [`pipeline::run`]
/// walks every item whatever the grid is, so a device with a different SM count
/// computes the same GEMM off a wave that is not quite a wave.
const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / RANKS;
const _: () = assert!(MAX_CLUSTERS == 148);

/// Tile-rows the item map walks before moving right — [`pipeline::grouped`]'s
/// width, and this kernel's answer to the 23% aspect-ratio swing.
///
/// What it changes is the *working set of a wave*. 148 clusters walked
/// row-major sit on `ceil(148 / tiles_n)` rows of tiles and span as much of `N`
/// as the shape allows; walked in groups they sit on a block whose shape the
/// aspect ratio no longer controls. It is a measurement at this tile shape and
/// not a preference — a tile change is a reason to re-run the sweep.
const GROUP: u32 = 8;

/// One output tile of `C`, as the persistent grid's work item.
///
/// Every field is what the item needs and does not depend on *which* item it
/// is: the pair's rings and barriers, the two operand maps, this CTA's half of
/// the accumulator, this warp's staging tile, the shape of the tile grid, and
/// the thread's own coordinates. The item index is the only thing [`Job::work`]
/// takes, and [`pipeline::grouped`] is the whole of what it does with it.
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
    /// This warp's 4096 B of the staging run — see [`StageTile`] for why it is
    /// the warp's and not the CTA's.
    stage: StageTile,
    /// `C` with `ldc` in it — built once, since a persistent CTA writes bands
    /// of the same output through every item it runs.
    c: GlobalRows<Bf16>,
    /// The tile grid, both axes. `tiles_m` is here so the item map can short
    /// the last group of tile-rows when [`GROUP`] does not divide it — a map
    /// that aliases the tail is a wrong `C` rather than a slow one.
    tiles_m: u32,
    tiles_n: u32,
    /// [`pipeline::grouped`]'s width, a launch parameter rather than a constant
    /// so the correctness run can walk every size at three of them.
    group: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl Tile {
    /// Issue this rank's half of the item's K blocks, charging the leader's
    /// stage barrier for the whole pair.
    ///
    /// # Safety
    ///
    /// One thread of the CTA, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
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
            // nothing from `src/mma.rs`. In a `const` block so a tile whose
            // columns name no shape is a codegen error rather than a `panic!`
            // lowered into device code.
            let shape = const { pair_shape(BLOCK_N) };
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                // One walk per swizzle atom of the stage: `k_walk` describes
                // exactly one, so a `BLOCK_K` wider than `ATOM_K` is that many
                // walks over the stacked subtiles the tile is already stored
                // as — the same bytes, the same chunk count, one barrier
                // instead of several. At `BLOCK_K = ATOM_K` this loop is one
                // iteration and folds away.
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                let mut atom = 0usize;
                while atom < ATile::SUBTILES {
                    mma_walk_cg2::<Bf16, ATOM_CHUNKS>(
                        self.accumulator.raw(),
                        Atom::<BLOCK_M>::from_raw(a.subtile(atom)).k_walk(),
                        Atom::<HALF_N>::from_raw(b.subtile(atom)).k_walk(),
                        shape,
                        k > 0 || atom > 0,
                    );
                    atom += 1;
                }
                // The MMA releases its own operands, in both CTAs: a thread
                // arriving here would only prove the instruction was *issued*.
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// The epilogue: this warp's band of the accumulator, staged through shared
    /// memory on its way to the tile `item` names.
    ///
    /// TMEM → registers at `tcgen05.ld.16x256b.x8` → `stmatrix.m8n8.x4` into
    /// this warp's own [`StageTile`] → 16-byte stores out to `C`. Four passes
    /// of [`STAGE_N`] columns cover the CTA's `[32, 256]` band, and the tile is
    /// reused by each of them.
    ///
    /// | | instructions a thread | what one touches |
    /// |---|---|---|
    /// | straight out of registers | **128** `st.global.b32` | 4 B, on **8 discontiguous 16 B runs** — [`BaseLdtm`] spreads a warp over 8 rows × 4 column-quads |
    /// | `stmatrix` | 16 `stmatrix.m8n8.x4` | 512 B a warp |
    /// | `ld.shared` | 32 `ld.shared.v4.b32` | 16 B |
    /// | `st.global` | **32** `st.global.v4.b32` | 16 B, on **4 contiguous 128 B runs** |
    ///
    /// `ldc` is the destination's leading dimension and `C` is wider than this
    /// tile's columns, so [`GlobalRows`] carries the stride and each band lands
    /// at its own `(row, column)` origin.
    ///
    /// # Safety
    ///
    /// Every thread of the CTA, with the accumulator complete and fenced, and
    /// nothing that will overwrite it in flight.
    #[inline(always)]
    unsafe fn drain(&self, item: u32) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            let chunks = self.stage.chunk_writer();
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band: StagedBand = self.accumulator.tile_x8(32 * self.warp_id, column);
                store_tile_x4(chunks, 0, 0, self.lane, band);
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c,
                    row_base,
                    column_base + column,
                    self.lane,
                    self.stage,
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

    /// Where in `C` this warp's band of `item` starts — the item index's whole
    /// meaning to the epilogue.
    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * self.warp_id,
            BLOCK_N as u32 * tile_n,
        )
    }
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
            // The item map. It is a bijection, so every tile is computed once
            // whatever `group` is.
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);

            if self.warp_id == 0 && self.lane == 0 {
                self.produce(tile_m, tile_n);
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

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The item, laid over the launch's dynamic shared memory. Everything here
    /// spans items rather than belonging to one, which is why it is hoisted out
    /// of the item loop: the rings, the barriers, the operand maps, this warp's
    /// staging tile, and the pair's TMEM allocation — whose `alloc_cluster` is
    /// a whole-cluster collective with a `cluster_sync` in it and must not be
    /// inside anybody's loop.
    ///
    /// The staging run sits at the *end* of the envelope rather than folded
    /// into [`shared_plan`], so the operand offsets are the same ones a kernel
    /// with no staging tiles would lay out.
    ///
    /// # Safety
    ///
    /// The launch geometry's, and the operands': both maps must describe live
    /// buffers covering `k_blocks * BLOCK_K` along K and the full extent the
    /// item loop walks, `c` must hold `ldc` columns for every row of it, and
    /// the launch must declare [`STAGED_SHARED_BYTES`].
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> Tile {
        // The join `shared_plan` cannot make on its own, fired at codegen —
        // which is the only place the ring byte counts are known, and the
        // reason `cargo check` cannot stand in for a device build.
        const {
            assert!(
                shared_plan(BLOCK_N, BLOCK_K, STAGES)
                    == ARing::BYTES + BRing::BYTES + scratch_bytes(STAGES)
            );
        };
        unsafe {
            let rings = ARing::BYTES + BRing::BYTES;
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let scratch = smem.add(rings);
            let tmem_slot = scratch.add(2 * STAGES * 8 + 8) as *mut u32;
            let warp_id = warp::warp_id();

            Tile {
                a_ring: ARing::attach(smem),
                b_ring: BRing::attach(smem.add(ARing::BYTES)),
                load: SemaphoreRing::<STAGES>::attach(scratch as *mut Barrier),
                free: SemaphoreRing::<STAGES>::attach((scratch as *mut Barrier).add(STAGES)),
                done: Semaphore::attach((scratch as *mut Barrier).add(2 * STAGES)),
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster(tmem_slot, BLOCK_N as u32)),
                stage: StageTile::from_raw(
                    smem.add(STAGE_OFFSET + warp_id as usize * StageTile::BYTES),
                ),
                c: GlobalRows::<Bf16>::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                group,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id,
                lane: warp::lane_id(),
            }
        }
    }

    /// Give the pair's accumulator back.
    ///
    /// The scaffold's last item boundary already retired the pair's reads, and
    /// this `cluster_sync` is for the cluster that got no items at all — a
    /// capped grid can leave a pair having allocated, never looped, and still
    /// owing a deallocation in step with its peer.
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

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `(2·BLOCK_M, BLOCK_N)` output tile
    /// per work item, `k_blocks` stages of `BLOCK_K` deep.
    ///
    /// The name is the kernel: `cg2` is the `cta_group::2` cluster path,
    /// `staged` is the shared-memory epilogue, and `x8x4` are its two
    /// instruction widths — `tcgen05.ld.16x256b.x8` and `stmatrix.m8n8.x4`.
    ///
    /// The grid is persistent and the item map is [`pipeline::run`]'s: a
    /// *cluster* takes item `%clusterid` and steps by `%nclusterid` until the
    /// tiles are gone, and `cluster::block_rank()` says which half of the pair
    /// this CTA owns. `a_map` describes `A` as `[rows, K]` bf16, `b_map`
    /// describes `B` as `[columns, K]`. Both come from a rank-2
    /// [`kittens::global::GlobalLayout`] paired with the tile it feeds, so
    /// their `[R, 64]` boxes are `ATile`'s and `BTile`'s own constants and not
    /// numbers [`check`] wrote down.
    ///
    /// A grouped item map has to know how many tile-*rows* there are, to short
    /// the last group when the width does not divide them — so `tiles_m` is a
    /// parameter and `tiles_m * tiles_n` is the item count.
    ///
    /// Everything outside the item loop is [`attach`] and [`release`]; what is
    /// left here is the schedule.
    ///
    /// # Safety
    ///
    /// [`attach`]'s, plus: the grid must be a whole number of clusters and
    /// `tiles_m * tiles_n` the item count they are to cover — see [`grid`],
    /// which is what the launcher sizes both from.
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
            let mut tile = attach(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            pipeline::run(&mut tile, tiles_m * tiles_n);
            release(&tile);
        }
    }
}

/// Rows of `C` the correctness run computes — two clusters' worth of `M`, so
/// the `item → (tile_m, tile_n)` map is exercised in both axes.
const M: usize = 512;
/// Columns of `C`: two [`BLOCK_N`] tiles.
const N: usize = 256;
/// Reduction depth: four [`BLOCK_K`] stages against a three-deep pipeline, so
/// the ring wraps and `wait_recycled` is on trial rather than skipped.
const K: usize = 256;

/// A second correctness size, whose only job is to give every cluster **more
/// than one work item**.
///
/// [`M`]`x`[`N`] is four tiles and the grid holds 148 clusters, so it never
/// enters [`pipeline::run`]'s loop a second time — it would pass identically
/// against a launch-per-tile kernel, which is exactly what makes it not a test
/// of this one. The failure modes the persistent scaffold introduces all live
/// at the *item boundary*: a barrier re-armed while a peer is still filling it,
/// an accumulator that is not started fresh for the next tile, an epilogue
/// racing the next item's first loads. Each of those is a deadlock or a wrong
/// `C`, and each needs a second item to happen at all.
///
/// 256 tiles over 148 clusters is two items for most and one for the rest, so
/// the ragged tail — clusters that leave the loop an item early, while their
/// neighbours are still running one — is under test too. `K` stays at the
/// wrapping four stages and the shape stays cheap.
const ITEMS_M: usize = 4096;
const ITEMS_N: usize = 4096;
const ITEMS_K: usize = 256;

/// `A[m, k]` and `B[n, k]`: integers in `[-3, 3]` and `[-10, 10]`.
///
/// Every operand is exact in bf16 (which holds every integer to 256) and every
/// partial sum stays well under fp32's exact integer range of 2²⁴. So the whole
/// GEMM is exact and the host compares with `==`. That is the point: a mismatch
/// is a wrong coordinate, a wrong stride or a wrong operand half, and never a
/// rounding artifact that has to be argued about.
///
/// Both generators depend on `depth`, and #48 is why that is spelled out rather
/// than assumed. `b_value` used to read `(column * 3 + depth * 5) % 5`, and
/// `depth * 5 % 5` is identically zero — `B` was constant along `K`, so the
/// exact check was blind to precisely the axis `mma_walk_cg2`'s chunk
/// arithmetic computes. A kernel reading one plane of `B` every step, walking K
/// backwards, or aliasing the ring slot for the K index passed anyway.
///
/// The moduli are 7 for `A` and 21 for `B`, and the pair is chosen rather than
/// convenient. `A`'s values over one period of `depth` are `0..7` shifted to
/// sum to zero, so if `B`'s `depth` period were *coprime* to 7 the sum over one
/// combined period would factor as `(Σ A)(Σ B) = 0` — the dot product would
/// then be a function of `K mod 7·P` alone, bounded independently of `K`, and
/// two different K walks would collide by accident. Sharing the factor 7
/// defeats that: the partial sums grow with `K`, and a swept check of every
/// legal `K` up to 8192 finds no wrong K walk this reference cannot see.
fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

/// Round-to-nearest-even fp32 → bf16 — the same rounding `cvt.rn.bf16x2.f32`
/// does, ties to even included, which is what lets [`check_c`] stay `==`.
///
/// Exact for every value [`a_value`] and [`b_value`] produce, since their low
/// 16 mantissa bits are already zero. On the *output* it is not exact, and that
/// is the whole reason it has to be in the reference.
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
fn stage(rows: usize, k: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u32> {
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
fn tile_grid(m: usize, n: usize) -> (u32, u32) {
    ((m / (2 * BLOCK_M)) as u32, (n / BLOCK_N) as u32)
}

/// Blocks the launch asks for — [`MAX_CLUSTERS`] pairs, or fewer where the
/// problem has fewer tiles than that.
///
/// Past that cap the grid is flat and the extra tiles arrive as extra *items*,
/// which is the whole difference between this kernel and one that launched a
/// pair per tile.
pub fn grid(m: usize, n: usize) -> u32 {
    let (rows, columns) = tile_grid(m, n);
    RANKS * (rows * columns).min(MAX_CLUSTERS)
}

/// Bytes of shared memory an SM divides between its resident CTAs — the
/// denominator of #84's `shared per SM / plan`, **queried rather than written
/// down**.
///
/// It is 233 472 on a B200, and a residency is a floor division by it, so a
/// figure that is only nearly right moves a kernel across an occupancy step and
/// changes the answer rather than the third digit.
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

/// CTAs of this launch an SM holds — #84's `min(512 / columns, shared per SM /
/// plan)`, which is [`CTAS_PER_SM`] said as arithmetic instead of as a
/// measurement, and printed by [`check`] so the two can disagree out loud.
fn ctas_per_sm(shared_per_sm: usize) -> u32 {
    (512 / BLOCK_N).min(shared_per_sm / STAGED_SHARED_BYTES) as u32
}

/// Launch `[m, k] · [n, k]ᵀ` at a traversal width and compare every element of
/// `C` against a CPU reference.
///
/// The two operands differ only in their extents, and neither states a box:
/// [`kittens::global::GlobalLayout::tensor_map`] takes the tile the kernel
/// loads into and reads the box, the swizzle and the data type off it. `A` is
/// `[k, m]` in the driver's fastest-first dimension order and `B` is `[k, n]`,
/// which is the same order the kernel gives its
/// [`SharedTile::tma_load_2d`] coordinates in.
fn run(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    group: u32,
) -> Result<f64, Box<dyn Error>> {
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
    // `grouped` divides by `group * columns` and the device checks nothing, so
    // a zero here is a launch that faults rather than a launch that is wrong.
    if group == 0 {
        return Err("a traversal group width of 0 has no tiles in it".into());
    }

    let stream = context.default_stream();
    // SAFETY: the artifact is this crate's own, and `gemm_cg2_staged_x8x4` is
    // the only entry point in it — the ABI the contract declares is the one
    // compiled.
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
    // descriptor at all**: a two-atom stage would be two boxes through the same
    // map, which is what `tma_load_2d` already issues per stacked subtile.
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BTile>(&stream)?;

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let (tiles_m, tiles_n) = tile_grid(m, n);
    let config = LaunchConfig1D::new(grid(m, n), THREADS, STAGED_SHARED_BYTES as u32);
    // Preparing is a driver call opting the entry point into its >48 KiB plan;
    // it happens once and the launch is what follows.
    let prepared = module.prepare_gemm_cg2_staged_x8x4(config)?;
    // SAFETY: both maps describe live buffers covering the walk the grid takes,
    // `c` holds `n` columns for every row of it, and the grid is a whole number
    // of clusters covering `tiles_m * tiles_n` items.
    unsafe {
        module.gemm_cg2_staged_x8x4(
            &stream,
            &prepared,
            a_map.as_ptr(),
            b_map.as_ptr(),
            tiles_m,
            tiles_n,
            group,
            (k / BLOCK_K) as u32,
            n as u32,
            &mut c,
        )?
    };
    check_c(&c.to_host_vec(&stream)?, m, n, k)
}

/// Compare an observed `[m, n]` row-major **bf16** `C` against the CPU
/// reference for `[m, k] · [n, k]ᵀ`, element by element and with `==`.
///
/// [`a_value`] repeats every 7 rows and [`b_value`] every 21 columns, so the
/// reference has 147 distinct dot products at any size and the naive
/// `O(m·n·k)` form is pure waste. Every element of `C` is still compared
/// against its own expected value, in the same summation order.
///
/// # Where the rounding is, and what that makes this blind to
///
/// The output is bf16 and the accumulator is not, so exactly one rounding
/// happens and this function has to agree with it. **It is put in the
/// reference**: the exact fp32 dot product is rounded by [`to_bf16`] and the
/// comparison stays `==` on the 16-bit words, with no tolerance at all. That is
/// available because the sum itself is still exact — every operand is exact in
/// bf16 and every partial sum stays under 2²⁴, so the fp32 value reaching the
/// `cvt` is the same integer whatever order it was summed in.
///
/// The alternative — widen the observed bf16 and compare to the fp32 reference
/// within a tolerance — would have been strictly weaker, because a tolerance
/// wide enough to admit correct rounding also admits everything smaller than
/// it, and a wrong tile that happened to land close would pass.
///
/// What this *is* blind to is resolution: two fp32 accumulators differing by
/// less than half an ulp of bf16 round to the same word, so an error under
/// about 0.2% of a value's magnitude is invisible. Every failure mode this gate
/// exists for — a wrong coordinate, a wrong stride, a dropped or doubled tile,
/// a wrong operand half, a mis-walked K — moves an element by far more than
/// that or leaves it at zero.
///
/// The return value is the **worst relative error against the exact fp32
/// reference**, which is the representation error of a bf16 `C` and is reported
/// rather than asserted: it is a property of the output format and of the
/// magnitudes this reference produces, not of the kernel.
fn check_c(observed: &[u16], m: usize, n: usize, k: usize) -> Result<f64, Box<dyn Error>> {
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

/// Traversal widths the correctness run walks every size at, chosen to break
/// [`pipeline::grouped`] rather than to be fast.
///
/// `1` is row-major, so a regression in the map still fails against the
/// traversal this kernel had before it was grouped. The other two are **not
/// powers of two, and that is the point**: every `M` this project runs is, so
/// `tiles_m` is too, so a swept width of 8 or 16 always divides it and the
/// short last group — the one branch in the map, and the one that turns a wrong
/// width into a tile computed twice — would never execute. At [`ITEMS_M`]'s 16
/// tile-rows, `3` leaves a final group of one row and `6` leaves one of four;
/// at [`M`]'s two tile-rows both are taller than the grid and take the
/// saturating path instead.
const CHECK_GROUPS: [u32; 3] = [1, 3, 6];

/// The correctness run: two sizes, three traversals, checked, nothing timed.
///
/// The second size is [`ITEMS_M`]`x`[`ITEMS_N`], and it is here because the
/// first cannot fail the way the persistent grid can. Both report the items
/// their clusters walked, so a size that quietly stopped exercising the loop —
/// because [`MAX_CLUSTERS`] moved, or because the tiling did — says so in the
/// pass line instead of in nobody's memory.
///
/// The traversal is under the same gate for the same reason: a wrong item map
/// is a tile computed twice and one computed by nobody, both silent on the
/// device, and the element-by-element `==` is the only thing that sees either.
/// The epilogue is under it for a sharper reason still — `stmatrix` and the
/// read-back address the same swizzled tile through two different derivations,
/// the four staging tiles are row-subranges of one 16 384 B run, and the loop
/// reuses each tile four times with only a `bar.warp.sync` between the read of
/// one pass and the write of the next. Every one of those is a wrong `C` rather
/// than a fault if it is wrong.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let per_sm = shared_per_sm(context)?;
    let mut notes = Vec::new();
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        let (rows, columns) = tile_grid(m, n);
        // The rounding a bf16 `C` carries, reported once per size from the
        // first checked launch of it. It is the same number whichever traversal
        // produced the output, since every one that reaches this line was `==`
        // on the words.
        let mut rounding = None;
        for group in CHECK_GROUPS {
            let worst = run(context, m, n, k, group)?;
            rounding.get_or_insert(worst);
        }
        notes.push(format!(
            "{m}x{n}x{k} exact at groups {CHECK_GROUPS:?} of {rows} tile-rows \
             ({} tiles over {} clusters, {STAGED_SHARED_BYTES} B, {} CTAs/SM, \
             worst |rel| {:.2e} against the fp32 reference)",
            rows * columns,
            grid(m, n) / RANKS,
            ctas_per_sm(per_sm),
            rounding.expect("CHECK_GROUPS is not empty"),
        ));
    }
    Ok(notes.join(", "))
}
