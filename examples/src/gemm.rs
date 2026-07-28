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
//! ## What this kernel had to reach past the library for
//!
//! One `GAP` block below, with no open issue behind it:
//!
//! - **global stores from registers.** The epilogue is open-coded index math
//!   against `RegTile::coordinate`, which is the arithmetic this library
//!   exists to delete (#11).
//!
//! The cluster-scope TMEM allocation that used to sit beside it is now
//! [`kittens::tmem::alloc_cluster`] / [`kittens::tmem::dealloc_cluster`]
//! (#24, #46) — this file was where the `cta_group::2` allocator's
//! participation rules and its relinquish were worked out against silicon,
//! and they are hardware facts about a cluster accumulator rather than
//! anything this kernel chose.
//!
//! A third one is gone, and how it went is worth recording. The pair's four
//! TMA loads have to complete on *one* barrier for the leader to know the
//! whole stage is present, and this file used to map the leader's barrier by
//! hand and call it a missing `Semaphore::at_rank`. That deadlocks: a plain
//! `cp.async.bulk.tensor` completes on a barrier in the *issuing* CTA's own
//! shared memory, so the peer's bytes never reached the leader's count. The
//! fix is [`SharedTile::tma_load_2d_multicast_cg2`] with the CTA's own bit as
//! the whole mask, onto [`Semaphore::multicast_alias`] — the primitive was
//! already here, filed under the opposite problem (`examples/README.md` §8
//! said this kernel "cannot use" the multicast load). What the pair needs
//! from multicast is not replication; it is the right address space for the
//! barrier operand.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
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

use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
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

/// `#[launch_contract]` takes literals, so the envelope is written twice; this
/// is what keeps the two in step. The contract is not decoration: 72 KiB is
/// past the 48 KiB a block gets by default, and the opt-in
/// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`) is issued by the
/// prepared-launch path and by nothing else.
const _: () = assert!(THREADS == 128 && SHARED_BYTES == 73_792);

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]` over a `(2·BLOCK_M, BLOCK_N)` output
    /// tile per cluster, `k_blocks` stages of `BLOCK_K` deep.
    ///
    /// The grid is one CTA pair per output tile: `blockIdx.x / 2` selects the
    /// tile, and `cluster::block_rank()` says which half of it this CTA owns.
    /// `a_map` describes `A` as `[rows, K]` bf16, `b_map` describes `B` as
    /// `[columns, K]`. Both come from a rank-2 [`kittens::global::GlobalLayout`]
    /// paired with the tile it feeds, so their `[R, 64]` boxes are `ATile`'s
    /// and `BTile`'s own constants and not numbers [`check`] wrote down.
    ///
    /// # Safety
    ///
    /// The launch geometry is the contract's, but the operands are not: both
    /// maps must describe live buffers covering `k_blocks * BLOCK_K` along K
    /// and the full `tiles_m`/`tiles_n` extent the grid walks, and `c` must
    /// hold `ldc` columns for every row of that walk. The grid must be
    /// `2 * tiles_m * tiles_n` blocks, since a CTA derives its output tile
    /// from `blockIdx.x` and never bounds-checks it.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 73_792,
        dynamic_shared_alignment = 128
    )]
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
            let accumulator = Accumulator::from_raw(alloc_cluster(tmem_slot, BLOCK_N as u32));

            // `M` is the pair's 256 rows and `N` its 128 columns. The rest of
            // the descriptor is the walk's: both operands are K-major, so the
            // MMA takes no transpose bits, and bf16 comes from the tiles.
            let shape = MmaShape::M256_N128;

            if warp_id == 0 && lane == 0 {
                // Producer. The pair's four tiles all complete on the leader's
                // copy of the stage barrier, which the leader charges in full
                // before either CTA issues, so the count is never short.
                //
                // The loads are the *multicast* form with each CTA's own bit
                // and nothing else, which looks pointless and is not: a plain
                // `cp.async.bulk.tensor` completes on a barrier in the issuing
                // CTA's own shared memory, so rank 1 cannot name the leader's.
                // The multicast form takes a `.shared::cluster` barrier, which
                // is the only way to say "my bytes, the leader's barrier".
                // Nothing is replicated — `1 << rank` delivers to one CTA.
                let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * rank) as i32;
                let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * rank) as i32;
                let mut k = 0u32;
                while k < k_blocks {
                    free.wait_recycled(k);
                    if rank == 0 {
                        load.sem(k)
                            .expect_tx(2 * (ATile::BYTES + BTile::BYTES) as u32);
                    }
                    let arrival = load.sem(k).multicast_alias();
                    let column = (BLOCK_K as u32 * k) as i32;
                    let mine = 1u16 << rank;
                    a_ring
                        .tile(k)
                        .tma_load_2d_multicast_cg2(a_map, column, a_row, arrival, mine);
                    b_ring
                        .tile(k)
                        .tma_load_2d_multicast_cg2(b_map, column, b_row, arrival, mine);
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

            // The whole band in one call (#22): this warp's 32 TMEM lanes by
            // every column of the accumulator, composed out of the `[16, 16]`
            // blocks LDTM delivers.
            let band: Band = accumulator.tile(32 * warp_id, 0);

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
            dealloc_cluster(accumulator.raw(), BLOCK_N as u32);
            if thread::threadIdx_x() == 0 {
                load.inval_all();
                free.inval_all();
                done.inval();
            }
        }
    }
}

/// Rows of `C` the correctness run computes — two clusters' worth of `M`, so
/// the `blockIdx → (tile_m, tile_n)` map is exercised in both axes.
pub const M: usize = 512;
/// Columns of `C`: two `BLOCK_N` tiles.
pub const N: usize = 256;
/// Reduction depth: four `BLOCK_K` stages against a three-deep pipeline, so
/// the ring wraps and `wait_recycled` is on trial rather than skipped.
pub const K: usize = 256;

/// `A[m, k]` and `B[n, k]`: integers in `[-3, 3]` and `[-10, 10]`.
///
/// Every operand is exact in bf16 (which holds every integer to 256) and every
/// partial sum stays under `3 * 10 * K` — 245,760 at the benchmark's largest
/// `K` of 8192, against fp32's exact integer range of 2²⁴ = 16,777,216. So the
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
fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

fn b_value(column: usize, depth: usize) -> f32 {
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

/// Blocks a `[m, n]` output takes: one CTA pair per `2·BLOCK_M` by `BLOCK_N`
/// tile of `C`. The benchmark prints it, because at the small end of a size
/// sweep this number and not the arithmetic is what the run is bound by.
pub fn grid(m: usize, n: usize) -> u32 {
    2 * (m / (2 * BLOCK_M) * (n / BLOCK_N)) as u32
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
    let launch = module.prepare_gemm_cg2(LaunchConfig1D::new(
        grid(m, n),
        THREADS,
        SHARED_BYTES as u32,
    ))?;
    let tiles_n = (n / BLOCK_N) as u32;
    let launch_once = |c: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn Error>> {
        // SAFETY: both maps describe live buffers covering the walk the grid
        // above takes, and `c` holds `n` columns for every row of it.
        unsafe {
            module.gemm_cg2(
                &stream,
                &launch,
                a_map.as_ptr(),
                b_map.as_ptr(),
                tiles_n,
                (k / BLOCK_K) as u32,
                n as u32,
                c,
            )?
        };
        Ok(())
    };
    launch_once(&mut c)?;

    // `a_value` repeats every 7 rows and `b_value` every 21 columns, so the
    // reference has 147 distinct dot products at any size and the naive
    // `O(m·n·k)` form is pure waste — minutes of host time at the sizes the
    // benchmark reaches, for the same 147 numbers. Every element of `C` is
    // still compared against its own expected value, in the same summation
    // order, so the comparison is the one it always was. The sum stays over
    // the *full* `k`, since both generators vary along it.
    let reference: Vec<f32> = (0..7 * 21)
        .map(|cell| {
            (0..k)
                .map(|depth| a_value(cell / 21, depth) * b_value(cell % 21, depth))
                .sum()
        })
        .collect();
    let observed = c.to_host_vec(&stream)?;
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

    let after = then(&stream, &mut || launch_once(&mut c))?;
    Ok((format!("{m}x{n}x{k} exact"), after))
}

/// The correctness run: one size, checked, nothing timed.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, M, N, K, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point: the same check at `shape`, and then the same
/// launch timed.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, shape.n, shape.k, time).map(|(_, timings)| timings)
}
