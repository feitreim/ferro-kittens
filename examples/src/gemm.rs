//! # GEMM — `C = A·Bᵀ` on the `cta_group::2` cluster path
//!
//! bf16 in, fp32 accumulate, bf16 out. A cluster of two CTAs shares one
//! `M256_N256` UMMA: each rank stages its own [`BLOCK_M`] rows of `A` and its
//! own [`HALF_N`] columns of `B` at the same shared offsets, the instruction
//! reads both ranks' shared memory, and each rank drains its own
//! `[BLOCK_M, BLOCK_N]` band of `C`. Rank [`LEADER`] owns the MMA, the
//! accumulator and the stage barriers, so the item boundary is
//! `barrier.cluster` rather than `bar.sync`.
//!
//! The grid is persistent — [`MAX_CLUSTERS`] clusters, two CTAs an SM on a
//! B200 — and one work item is one `[2 * BLOCK_M, BLOCK_N]` output tile:
//! [`pipeline::run`] hands out item indices, [`pipeline::grouped`] maps them to
//! tiles in blocks of [`GROUP`] tile-rows, and K is [`STAGES`] deep over a pair
//! of tile rings. The TMA fills `load` and the MMA's own commit releases
//! `free`: the accumulator instruction, not a thread, is what proves an operand
//! has been read.
//!
//! The epilogue goes through shared memory, and that is most of the speed —
//! TMEM → registers (`tcgen05.ld.16x256b.x8`) → `stmatrix.m8n8.x4` into a
//! per-warp `[32, STAGE_N]` shared tile → 16-byte stores to `C`, four passes a
//! band, 128 threads and one warp per 32 accumulator rows.
//!
//! [`check`] runs it against a CPU reference on a B200 (`scripts/modal-run
//! examples`) with small-integer operands, so the comparison is `==` and a
//! mismatch is a wrong index, never rounding. Measurements, the shapes that
//! lost and the occupancy arithmetic: `docs/kernels/gemm.md`.

use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract, thread};

use std::error::Error;

use kittens::epilogue::{Drain, X4, X8};
use kittens::global::GlobalRows;
use kittens::mma::{commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, Job};
use kittens::plan::SharedPlan;
use kittens::reg::BaseLdtm;
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::watchdog::{self, ReadBack};
use kittens::{lane, warp_id};

const BLOCK_M: usize = 128;
const BLOCK_N: usize = 256;
const HALF_N: usize = BLOCK_N / 2;
/// One 128-byte swizzle atom of bf16, and the only width [`SharedTile::k_walk`]
/// accepts.
const ATOM_K: usize = 64;
const ATOM_CHUNKS: usize = ATOM_K / 16;
const BLOCK_K: usize = 64;
const STAGES: usize = 3;
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32;
/// The narrowest tile `Swizzle128B` admits at bf16, and the widest the shared
/// budget leaves room for four of.
const STAGE_N: usize = 64;
/// One warp's staging tile: its own [`BaseLdtm::WARP_ROWS`] rows of `C` by
/// [`STAGE_N`] columns, 4096 B, and nobody else's — which is what keeps the
/// epilogue's barrier count at zero.
type StageTile = SharedTile<Bf16, { BaseLdtm::WARP_ROWS }, STAGE_N, Swizzle128B>;
const RANKS: u32 = 2;
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
const LEADER: u32 = 0;

type ATile = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
type BTile = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type Atom<const R: usize> = SharedTile<Bf16, R, ATOM_K, Swizzle128B>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const WARPS: usize = THREADS as usize / 32;
type StageRun = SharedTileRing<Bf16, { BaseLdtm::WARP_ROWS }, STAGE_N, Swizzle128B, WARPS>;

struct Shared {
    a_ring: ARing,
    b_ring: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (done, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        done,
        tmem_slot,
        plan: at,
    }
}

#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, { BaseLdtm::WARP_ROWS }, STAGE_N, Swizzle128B, WARPS>()
}

const SHARED_BYTES: usize = shared(SharedPlan::sizing()).plan.bytes();
const _: () = assert!(THREADS == 128 && SHARED_BYTES == 98_364);

/// Dynamic shared memory the launch declares. It must stay at or under the
/// 116 736 B an SM's 233 472 leaves a CTA at [`CTAS_PER_SM`].
pub const STAGED_SHARED_BYTES: usize = staged(shared(SharedPlan::sizing()).plan).1.bytes();

const _: () = {
    let run = WARPS * StageTile::BYTES;
    assert!(run == 16_384);
    assert!(STAGED_SHARED_BYTES == 114_816 && STAGED_SHARED_BYTES <= 116_736);
    assert!(STAGED_SHARED_BYTES + run > 116_736);
};

/// SMs on a B200.
const SMS: u32 = 148;
/// Measured on a clock and counted by `device-tests`' `tmem residency census`:
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` takes no cluster, so it cannot
/// answer for this kernel.
const CTAS_PER_SM: u32 = 2;
const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / RANKS;
const _: () = assert!(MAX_CLUSTERS == 148);

/// [`pipeline::grouped`]'s width, swept at this tile shape — a tile change is a
/// reason to re-run the sweep.
const GROUP: u32 = 8;

#[derive(Clone, Copy)]
struct Tile {
    a_ring: ARing,
    b_ring: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    stage: StageTile,
    c: GlobalRows<Bf16>,
    tiles_m: u32,
    tiles_n: u32,
    group: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl Tile {
    /// # Safety
    /// One thread of the CTA, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
        unsafe {
            // All four of the pair's tiles complete on the leader's copy of the
            // stage barrier, and only the leader charges it: `expect_tx` is
            // `.shared::cta`, so a peer could not charge that barrier even
            // holding its address. Both ranks derive the same half-stage charge
            // from the loads they just issued; the leader scales its own by
            // `RANKS` to cover the peer's, and the peer drops it.
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

    /// The instruction is `M256_N256`, and this file no longer says so: the
    /// walk reads it off [`Accumulator`] through [`kittens::mma::pair_shape`],
    /// which is where the `const fn` from a pair tile's columns to a shape
    /// belongs (#128). A kernel-side one had `panic!` for its default arm and
    /// was the only thing checking a `[BLOCK_M, BLOCK_N]` against the ISA.
    ///
    /// # Safety
    /// One thread of the leader rank, with the accumulator's previous contents
    /// already read: only the first chunk of the first stage of an item starts
    /// it fresh.
    #[inline(always)]
    unsafe fn multiply(&self) {
        unsafe {
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                let mut atom = 0usize;
                while atom < ATile::SUBTILES {
                    mma_walk_cg2::<Bf16, ATOM_CHUNKS, _, _>(
                        self.accumulator,
                        Atom::<BLOCK_M>::from_raw(a.subtile(atom)).k_walk(),
                        Atom::<HALF_N>::from_raw(b.subtile(atom)).k_walk(),
                        k > 0 || atom > 0,
                    );
                    atom += 1;
                }
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// [`Drain::staged`] over this CTA's tile: the band walk and the
    /// `bar.warp.sync` between passes are the library's, and what stays here is
    /// where in `C` the tile goes.
    ///
    /// `X8` and `X4` are #117's two instruction widths, one per half of the
    /// drain — `tcgen05.ld.16x256b.x8` reading the band and `stmatrix.m8n8.x4`
    /// writing it. Both at their wide end is what ships; the four combinations
    /// are `experiments/`' rungs.
    ///
    /// # Safety
    /// Every thread of the CTA, with the accumulator complete and fenced, and
    /// nothing that will overwrite it in flight.
    #[inline(always)]
    unsafe fn drain(&self, item: u32) {
        unsafe {
            let (row_base, column_base) = self.origin(item);
            Drain::<X8, X4>::staged(
                self.accumulator,
                self.warp_id,
                self.stage,
                self.c,
                row_base,
                column_base,
                self.lane,
            );
        }
    }

    /// Where in `C` this **CTA's** tile of `item` starts. The warp's own rows
    /// are not added here: the drain adds them to the accumulator's rows and to
    /// these together, which is what keeps one band index from meaning two
    /// different things.
    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank,
            BLOCK_N as u32 * tile_n,
        )
    }
}

impl Job for Tile {
    const RANKS: u32 = crate::gemm::RANKS;

    /// # Safety
    /// As [`Semaphore::init`].
    #[inline(always)]
    unsafe fn init(&self, _item: u32) {
        unsafe {
            self.load.init_all(1);
            self.free.init_all(1);
            self.done.init(1);
        }
    }

    /// # Safety
    /// As [`Semaphore::inval`].
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe {
            self.load.inval_all();
            self.free.inval_all();
            self.done.inval();
        }
    }

    /// # Safety
    /// Every thread of both CTAs of the cluster must enter with the same
    /// `item`, and the maps must cover the tile it names.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
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

    /// # Safety
    /// Both maps must describe live buffers covering the walk the item loop
    /// takes, `c` must hold `ldc` columns for every row of it, and the launch
    /// must declare [`STAGED_SHARED_BYTES`]. `alloc_cluster` is a whole-cluster
    /// collective with a `cluster_sync` in it, so this must not be reached from
    /// inside the item loop.
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
        unsafe {
            let shared = shared(SharedPlan::attach());
            let (run, _) = staged(shared.plan);
            let warp_id = warp_id();

            Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                done: shared.done,
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster::<BLOCK_N>(shared.tmem_slot)),
                stage: run.tile(warp_id),
                c: GlobalRows::<Bf16>::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                group,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id,
                lane: lane(),
            }
        }
    }

    /// # Safety
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired.
    #[inline(always)]
    unsafe fn release(tile: &Tile) {
        unsafe {
            // The item boundary already retired the pair's reads; this
            // `cluster_sync` is for the cluster that got no items at all, which
            // a capped grid can leave having allocated and never looped.
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster::<BLOCK_N>(tile.accumulator.raw());
        }
    }

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `(2·BLOCK_M, BLOCK_N)` output tile
    /// per work item, `k_blocks` stages of `BLOCK_K` deep.
    ///
    /// `a_map` describes `A` as `[rows, K]` bf16 and `b_map` describes `B` as
    /// `[columns, K]`. A grouped item map has to know how many tile-*rows*
    /// there are, so `tiles_m` is a parameter and `tiles_m * tiles_n` is the
    /// item count.
    ///
    /// # Safety
    /// [`attach`]'s, plus: the grid must be a whole number of clusters and
    /// `tiles_m * tiles_n` the item count they are to cover — see [`grid`].
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

const M: usize = 512;
const N: usize = 256;
const K: usize = 256;

/// A second correctness size, whose only job is to give every cluster more than
/// one work item: the persistent grid's failure modes are all at that boundary.
const ITEMS_M: usize = 4096;
const ITEMS_N: usize = 4096;
const ITEMS_K: usize = 256;

/// `A[m, k]` and `B[n, k]`: integers exact in bf16, with a shared factor of 7
/// in the two moduli so that no K walk can collide by accident — see the docs.
fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

/// Round-to-nearest-even fp32 → bf16, ties to even included — the same rounding
/// `cvt.rn.bf16x2.f32` does, which is what lets [`check_c`] stay `==`.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// A `[rows, k]` bf16 operand as the packed `u32` words a device buffer holds,
/// K contiguous — which is what makes the MMA transpose-free.
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

fn tile_grid(m: usize, n: usize) -> (u32, u32) {
    ((m / (2 * BLOCK_M)) as u32, (n / BLOCK_N) as u32)
}

/// Blocks the launch asks for — [`MAX_CLUSTERS`] pairs, or fewer where the
/// problem has fewer tiles. Past the cap the extra tiles arrive as extra items.
pub fn grid(m: usize, n: usize) -> u32 {
    let (rows, columns) = tile_grid(m, n);
    RANKS * (rows * columns).min(MAX_CLUSTERS)
}

fn run(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    group: u32,
) -> Result<f64, Box<dyn Error>> {
    use cuda_core::LaunchConfig1D;
    use kittens::global::GlobalLayout;

    if !m.is_multiple_of(2 * BLOCK_M) || !n.is_multiple_of(BLOCK_N) || !k.is_multiple_of(BLOCK_K) {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{BLOCK_N}x{BLOCK_K} tiling",
            2 * BLOCK_M
        )
        .into());
    }
    if group == 0 {
        return Err("a traversal group width of 0 has no tiles in it".into());
    }

    let stream = context.default_stream();
    // SAFETY: the artifact is this crate's own, and `gemm_cg2_staged_x8x4` is
    // the only entry point in it.
    let module = unsafe { kernels::load(context)? };

    let a = watchdog::stage(&stream, &stage(m, k, a_value))?;
    let b = watchdog::stage(&stream, &stage(n, k, b_value))?;
    // SAFETY: both buffers outlive every launch consuming their maps below.
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<Bf16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<Bf16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BTile>(&stream)?;

    let mut c = watchdog::cleared::<u16>(&stream, m * n)?;
    let (tiles_m, tiles_n) = tile_grid(m, n);
    let config = LaunchConfig1D::new(grid(m, n), THREADS, STAGED_SHARED_BYTES as u32);
    // Preparing is a driver call opting the entry point into its >48 KiB plan.
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
    check_c(&c.read_back(&stream)?, m, n, k)
}

/// Compare an observed bf16 `C` element by element with `==`: the reference
/// carries the same single rounding the kernel does, so no tolerance is
/// involved. The return value is the worst relative error against the exact
/// fp32 reference, reported rather than asserted.
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

/// Traversal widths every size is walked at: the one that ships, then three
/// chosen to break [`pipeline::grouped`] rather than to be fast. `3` and `6`
/// are not powers of two, which is what executes the short last group at all.
const CHECK_GROUPS: [u32; 4] = [GROUP, 1, 3, 6];

/// The correctness run: two sizes, four traversals, checked, nothing timed.
///
/// It reports the launch envelope it ran at and nothing derived from a clock or
/// from the driver. [`CTAS_PER_SM`] against what the device actually admits is
/// `experiments/`' sweep, which prints predicted and counted side by side at
/// every tile rung, and `device-tests`' `tmem residency census`.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        let (rows, columns) = tile_grid(m, n);
        let mut rounding = None;
        for group in CHECK_GROUPS {
            let worst = run(context, m, n, k, group)?;
            rounding.get_or_insert(worst);
        }
        notes.push(format!(
            "{m}x{n}x{k} exact at groups {CHECK_GROUPS:?} of {rows} tile-rows \
             ({} tiles over {} clusters, {STAGED_SHARED_BYTES} B, \
             worst |rel| {:.2e} against the fp32 reference)",
            rows * columns,
            grid(m, n) / RANKS,
            rounding.expect("CHECK_GROUPS is not empty"),
        ));
    }
    Ok(notes.join(", "))
}
