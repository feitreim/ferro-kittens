//! # Flash-attention forward, causal, one query block per CTA
//!
//! **Status: aspirational.** Excluded from the default build; read its gap
//! list with `cargo check --features flash`.
//!
//! Blocked on:
//!
//! - **#7** (masking) — no `make_causal`. And #7 as filed describes TK's
//!   signature, which takes no coordinate origin. A flash kernel masks a
//!   `[queries, keys]` band whose diagonal sits at `query_base - key_base`,
//!   so what it needs is `make_causal_at(lane, query_base, key_base, fill)`.
//! - **#11** (global ↔ register) — the epilogue writes fp32 out of registers.
//!   Today that means packing to bf16, `stmatrix` into a shared tile, and a
//!   TMA store (#9): a precision loss the epilogue never asked for.
//! - **#31** — `RegTile::add_assign`. `RegTile` has the by-value `add` and no
//!   in-place form, and this kernel's `[32, 128]` output accumulator is
//!   exactly the shape the by-value cost was measured on. #38 measured the
//!   same thing for a scalar operand and got the same direction: at 128
//!   columns the by-value spelling spills where the in-place one does not.
//!   The scalar broadcast `scale` that used to share this entry landed with
//!   #38 and is the real API below.
//!
//! Plus one thing named at its call site below:
//!
//! - **#23, the `RegTile` shape set is closed.** `BaseLdtm` implements
//!   `FragmentLayout` for `(16,16)`, `(32,32)` and `(32,128)` only, because
//!   each shape is a line of `base_ldtm_shapes!` *inside* `src/reg.rs`. This
//!   kernel's score band is `[32, 64]`, and an out-of-tree kernel has no way
//!   to add it.
//!
//! ## What already works
//!
//! Everything structural. [`mma_abt`] and [`mma_ab`] are exactly the two MMAs
//! this kernel issues, in the layouts it issues them; the semaphore protocol,
//! the swizzled `P` staging tile, and [`online_rescale`] — the fused
//! running-max correction, the one genuinely subtle piece of flash — are all
//! first-class. The gap is not the hard part of the algorithm. It is that
//! between `S` arriving in registers and `P` going back to shared, the kernel
//! has to spell out every elementwise operation by hand.

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use kittens::global::store_rows;
use kittens::ldst::store_tile;
use kittens::mma::{MmaShape, commit, mma_ab, mma_abt};
use kittens::reg::{BaseLdtm, RegTile, RegVec, online_rescale};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_block};

/// Queries per CTA — one `M128` MMA, four warps of 32 accumulator rows each.
const QUERIES: usize = 128;
/// Keys per pipeline stage.
const KEYS: usize = 64;
/// Head dimension.
const HEAD: usize = 128;
/// Pipeline depth over the key blocks.
const STAGES: usize = 3;
/// `log2(e)`, folded into the score scale so the exponential is `exp2`.
const LOG2E: f32 = 1.442_695;

type QTile = SharedTile<Bf16, QUERIES, HEAD, Swizzle128B>;
type KTile = SharedTile<Bf16, KEYS, HEAD, Swizzle128B>;
type VTile = SharedTile<Bf16, KEYS, HEAD, Swizzle128B>;
/// The probabilities, staged back to shared as the `A` operand of `O = P·V`.
/// Exactly one swizzle atom wide, which is what the `stmatrix` store path
/// needs.
type PTile = SharedTile<Bf16, QUERIES, KEYS, Swizzle128B>;
type KRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;
type VRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;

type Scores = TmemTile<QUERIES, KEYS>;
type Output = TmemTile<QUERIES, HEAD>;

/// One warp's band of the score tile.
///
/// GAP (#23): `BaseLdtm` has no `FragmentLayout<32, 64>` impl. The
/// implemented shapes are a `base_ldtm_shapes!` invocation inside
/// `src/reg.rs`, so the set of tiles a kernel may hold is fixed by the
/// library and this line does not compile out of tree. Dodging
/// `generic_const_exprs` with an associated type is the right call; closing
/// the shape set over it is not.
type ScoreBand = RegTile<32, KEYS, BaseLdtm>;
/// One warp's band of the output accumulator.
type OutBand = RegTile<32, HEAD, BaseLdtm>;
/// Row statistics of either band.
type Rows = RegVec<32, BaseLdtm>;

/// The two rings, `q_loaded`, `scored`, `accumulated`, and the TMEM staging
/// word.
const SCRATCH_BYTES: usize = 2 * STAGES * 8 + 3 * 8 + 8;
pub const SHARED_BYTES: usize =
    QTile::BYTES + KRing::BYTES + VRing::BYTES + PTile::BYTES + SCRATCH_BYTES;
pub const THREADS: u32 = (QUERIES / 32) as u32 * 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `O = softmax(scale · Q·Kᵀ + causal mask) · V` for one `[QUERIES, HEAD]`
    /// query block, streamed over `key_blocks` blocks of `KEYS` keys.
    ///
    /// One CTA per query block per head: `blockIdx.x` selects the block and
    /// `blockIdx.y` the head, which is the plane coordinate of the panel maps.
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
            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q = QTile::from_raw(smem);
            let k_ring = KRing::attach(smem.add(QTile::BYTES));
            let v_ring = VRing::attach(smem.add(QTile::BYTES + KRing::BYTES));
            let p = PTile::from_raw(smem.add(QTile::BYTES + KRing::BYTES + VRing::BYTES));
            let scratch =
                smem.add(QTile::BYTES + KRing::BYTES + VRing::BYTES + PTile::BYTES) as *mut Barrier;
            let load = SemaphoreRing::<STAGES>::attach(scratch);
            let free = SemaphoreRing::<STAGES>::attach(scratch.add(STAGES));
            let q_loaded = Semaphore::attach(scratch.add(2 * STAGES));
            let scored = Semaphore::attach(scratch.add(2 * STAGES + 1));
            let accumulated = Semaphore::attach(scratch.add(2 * STAGES + 2));
            let tmem_slot = scratch.add(2 * STAGES + 3) as *mut u32;

            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let leader = thread::threadIdx_x() == 0;
            let query_base = QUERIES as u32 * thread::blockIdx_x();
            let head = thread::blockIdx_y() as i32;

            if leader {
                load.init_all(1);
                free.init_all(1);
                q_loaded.init(1);
                scored.init(1);
                accumulated.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            // Scores and output share one allocation: `[128, KEYS]` then
            // `[128, HEAD]` beside it.
            let tmem = alloc_block(tmem_slot, (KEYS + HEAD) as u32);
            let scores = Scores::from_raw(tmem);
            let output: Output = scores.split_columns();

            if leader {
                q.tma_load(q_map, query_base as i32, head, q_loaded);
                q_loaded.expect_tx(QTile::BYTES as u32);
            }

            // Only the accumulator bands' shapes: `mma_abt`/`mma_ab` each
            // carry the transpose configuration their own walk reads under,
            // and the element comes from the tiles.
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
                    k_ring
                        .tile(block)
                        .tma_load(k_map, key_base as i32, head, load.sem(block));
                    v_ring
                        .tile(block)
                        .tma_load(v_map, key_base as i32, head, load.sem(block));
                    load.sem(block)
                        .expect_tx((KTile::BYTES + VTile::BYTES) as u32);
                }
                load.wait(block);
                thread::sync_threads();

                if leader {
                    mma_abt(scores.raw(), q, k_ring.tile(block), score_shape, false);
                    commit(scored);
                }
                scored.wait(block & 1);

                // The whole score band in one call (#22), composed out of the
                // `[16, 16]` blocks LDTM delivers.
                let mut s: ScoreBand = scores.tile(32 * warp_id, 0);

                // WANT (#7): causal masking against the band's own origin.
                // Without it this is a loop over `ScoreBand::coordinate`
                // comparing raw indices — the index math this library exists
                // to delete, in the one place every attention kernel needs it.
                s.make_causal_at(lane, query_base + 32 * warp_id, key_base, f32::NEG_INFINITY);

                // The softmax numerator, in the real API (#5, #6, #38).
                // `scale` folds `1/√d` and `log2(e)` into one register-operand
                // `Mul`; `row_max`/`row_sum` are both halves of the reduction,
                // the thread's own 16 values and then the quad.
                let s = s.scale(scale * LOG2E);
                let row_max = s.row_max();
                online_rescale(&mut running_max, row_max, &mut running_sum, &mut out_acc);
                let s = s.sub_row(running_max).exp2();
                running_sum.add_assign(s.row_sum());

                // The mirror of the drain above, on the shared side (#22).
                store_tile(p.chunk_writer(), 32 * warp_id, 0, lane, s);
                thread::sync_threads();

                if leader {
                    // `accumulate` is false: the running output lives in
                    // registers, so TMEM only ever holds this block's `P·V`.
                    mma_ab(output.raw(), p, v_ring.tile(block), output_shape, false);
                    commit(accumulated);
                }
                accumulated.wait(block & 1);

                let contribution: OutBand = output.tile(32 * warp_id, 0);
                // WANT (#31): `RegTile` has the by-value `add` but no
                // in-place form, and this is the accumulator #31 measured —
                // at 128 columns the by-value spelling cost 87 registers.
                // `RegVec` has had `add_assign` since #5; #38 deliberately did
                // not add the tile's, so that #31 lands every in-place form at
                // once rather than unifying a one-off later.
                out_acc.add_assign(contribution);
                if leader {
                    free.sem(block).arrive();
                }
                block += 1;
            }

            // The softmax denominator, broadcast down the rows (#5).
            let out_acc = out_acc.div_row(running_sum);

            // WANT (#11): fp32 straight out of registers, at the coordinates
            // the fragment layout already knows.
            store_rows(&mut out, out_acc, query_base + 32 * warp_id, lane);

            thread::sync_threads();
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
