//! # Flash-attention forward, causal, one query block per CTA
//!
//! `O = softmax(scale · Q·Kᵀ + causal mask) · V` for one `[QUERIES, HEAD]`
//! query block, streamed over `key_blocks` blocks of [`KEYS`] keys.
//! `blockIdx.x` selects the query block and `blockIdx.y` the head, which is the
//! plane coordinate of all three panel maps.
//!
//! One CTA of [`THREADS`] threads, four warps of 32 accumulator rows each. `Q`
//! is loaded once; `K` and `V` are [`STAGES`] deep over their own rings. The
//! leader thread issues every TMA load and both MMAs — [`mm_abt`] for
//! `Q·Kᵀ` into the `[QUERIES, KEYS]` score tile, then [`mm_ab`] for `P·V` into
//! the `[QUERIES, HEAD]` output tile beside it in the same tensor-memory
//! allocation. The running max, the running sum and the output accumulator stay
//! in registers for the whole loop, so tensor memory only ever holds one
//! block's `P·V` and [`online_rescale`] folds the correction in. `P` goes back
//! to shared through `stmatrix` to be the `A` operand of the second MMA, and
//! the epilogue stores fp32 straight out of registers.
//!
//! **Status: compiles**, and is in the default build. There is no launcher and
//! no CPU reference, so nothing here is measured against a known-good `O`.
//!
//! The shared plan by component, the 1 block/SM ceiling and why it is tcgen05
//! rather than this plan, and the register-side measurements:
//! `docs/kernels/flash_forward.md`.

use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use kittens::global::{GlobalRows, store_rows};
use kittens::ldst::store_tile;
use kittens::mma::{MmaShape, commit, mm_ab, mm_abt};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, RegTile, RegVec, online_rescale};
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, Swizzle128B, publish_to_async_proxy};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_block, dealloc_block};
use kittens::{lane, warp_id};

const QUERIES: usize = 128;
const KEYS: usize = 64;
const HEAD: usize = 128;
const STAGES: usize = 3;
/// Folded into the score scale so the exponential is `exp2`.
const LOG2E: f32 = core::f32::consts::LOG2_E;

type QTile = SharedTile<Bf16, QUERIES, HEAD, Swizzle128B>;
/// The probabilities, staged back to shared as the `A` operand of `O = P·V`.
/// Exactly one swizzle atom wide, which is what the `stmatrix` store path
/// needs.
type PTile = SharedTile<Bf16, QUERIES, KEYS, Swizzle128B>;
type KRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;
type VRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;

type Scores = TmemTile<QUERIES, KEYS>;
type Output = TmemTile<QUERIES, HEAD>;

type ScoreBand = RegTile<32, KEYS, BaseLdtm>;
type OutBand = RegTile<32, HEAD, BaseLdtm>;
type Rows = RegVec<32, BaseLdtm>;

/// Columns of tensor memory the CTA allocates for the scores and the output
/// beside them. `KEYS + HEAD` is 192 and `tcgen05.alloc` takes a power of two
/// in `[32, 512]`; rounding up is free, since the driver charges a CTA the SM's
/// whole tensor memory at 32 columns exactly as at 512.
const TMEM_COLUMNS: u32 = 256;
const _: () = assert!(
    TMEM_COLUMNS as usize >= KEYS + HEAD
        && TMEM_COLUMNS.is_power_of_two()
        && TMEM_COLUMNS >= 32
        && TMEM_COLUMNS <= 512,
    "tcgen05.alloc takes a power of two in [32, 512] that covers the scores and the output"
);

struct Shared {
    q: QTile,
    k_ring: KRing,
    v_ring: VRing,
    p: PTile,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    q_loaded: Semaphore,
    scored: Semaphore,
    accumulated: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (q, at) = at.tile::<Bf16, QUERIES, HEAD, Swizzle128B>();
    let (k_ring, at) = at.tile_ring::<Bf16, KEYS, HEAD, Swizzle128B, STAGES>();
    let (v_ring, at) = at.tile_ring::<Bf16, KEYS, HEAD, Swizzle128B, STAGES>();
    let (p, at) = at.tile::<Bf16, QUERIES, KEYS, Swizzle128B>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (q_loaded, at) = at.semaphore();
    let (scored, at) = at.semaphore();
    let (accumulated, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    Shared {
        q,
        k_ring,
        v_ring,
        p,
        load,
        free,
        q_loaded,
        scored,
        accumulated,
        tmem_slot,
        plan: at,
    }
}

/// Dynamic shared memory the launch declares — 144 KiB and change, which is why
/// this kernel needs [`kittens::launch::admit_shared_plan`].
pub const SHARED_BYTES: usize = shared(SharedPlan::sizing()).plan.bytes();
pub const THREADS: u32 = (QUERIES / 32) as u32 * 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `O = softmax(scale · Q·Kᵀ + causal mask) · V` for one `[QUERIES, HEAD]`
    /// query block, streamed over `key_blocks` blocks of `KEYS` keys.
    #[kernel]
    pub unsafe fn flash_forward(
        q_map: *const TmaDescriptor,
        k_map: *const TmaDescriptor,
        v_map: *const TmaDescriptor,
        key_blocks: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        unsafe {
            let Shared {
                q,
                k_ring,
                v_ring,
                p,
                load,
                free,
                q_loaded,
                scored,
                accumulated,
                tmem_slot,
                ..
            } = shared(SharedPlan::attach());

            let warp_id = warp_id();
            let lane = lane();
            let leader = thread::threadIdx_x() == 0;
            let query_base = QUERIES as u32 * thread::blockIdx_x();
            let head = thread::blockIdx_y() as i32;

            if leader {
                load.init_all(1);
                free.init_all(1);
                q_loaded.init(1);
                scored.init(1);
                accumulated.init(1);
                publish_to_async_proxy();
            }
            thread::sync_threads();
            let tmem = alloc_block(tmem_slot, TMEM_COLUMNS);
            let scores = Scores::from_raw(tmem);
            let output: Output = scores.split_columns();

            if leader {
                q_loaded.expect_tx(q.tma_load(q_map, query_base as i32, head, q_loaded));
            }

            let score_shape = MmaShape::M128_N64;
            let output_shape = MmaShape::M128_N128;

            let mut running_max = Rows::splat(f32::NEG_INFINITY);
            let mut running_sum = Rows::splat(0.0);
            let mut out_acc = OutBand::zero();
            q_loaded.wait(0);

            let mut block = 0u32;
            while block < key_blocks {
                let key_base = KEYS as u32 * block;

                if leader {
                    free.wait_recycled(block);
                    let keys =
                        k_ring
                            .tile(block)
                            .tma_load(k_map, key_base as i32, head, load.sem(block));
                    let values =
                        v_ring
                            .tile(block)
                            .tma_load(v_map, key_base as i32, head, load.sem(block));
                    load.sem(block).expect_tx(keys + values);
                }
                load.wait(block);
                thread::sync_threads();

                if leader {
                    mm_abt(scores.raw(), q, k_ring.tile(block), score_shape);
                    commit(scored);
                }
                scored.wait(block & 1);

                let mut s: ScoreBand = scores.tile(32 * warp_id, 0);

                // The two bases go in separately rather than as their
                // difference: that is negative for every band above the
                // diagonal, and computing it in `u32` here would wrap and mask
                // nothing.
                s.make_causal_at(lane, query_base + 32 * warp_id, key_base, f32::NEG_INFINITY);

                let s = s.scale(scale * LOG2E);
                let row_max = s.row_max();
                online_rescale(&mut running_max, row_max, &mut running_sum, &mut out_acc);
                let s = s.sub_row(running_max).exp2();
                running_sum.add_assign(s.row_sum());

                store_tile(p.chunk_writer(), 32 * warp_id, 0, lane, s);
                thread::sync_threads();

                if leader {
                    mm_ab(output.raw(), p, v_ring.tile(block), output_shape);
                    commit(accumulated);
                }
                accumulated.wait(block & 1);

                let contribution: OutBand = output.tile(32 * warp_id, 0);
                out_acc.add_assign(contribution);
                if leader {
                    free.sem(block).arrive();
                }
                block += 1;
            }

            let out_acc = out_acc.div_row(running_sum);

            store_rows(
                GlobalRows::<F32>::from_slice(&mut out, HEAD),
                query_base + 32 * warp_id,
                0,
                lane,
                out_acc,
            );

            thread::sync_threads();
            // tcgen05 allocations are not scoped to the CTA the way shared
            // memory is: a kernel that exits holding them is a
            // `CUDA_ERROR_TENSOR_MEMORY_LEAK`, and the next CTA scheduled onto
            // the SM is the one that pays.
            dealloc_block(tmem, TMEM_COLUMNS);
            if leader {
                load.inval_all();
                free.inval_all();
                q_loaded.inval();
                scored.inval();
                accumulated.inval();
            }
        }
    }
}
