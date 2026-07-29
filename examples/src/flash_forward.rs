//! # Flash-attention forward, causal, one query block per CTA
//!
//! **Status: compiles**, and is in the default build. It has no launcher and
//! no CPU reference, which is the only thing between it and **runs** — and the
//! reason it says *compiles*: "runs against an exact reference" is a claim that
//! has to be earned, and #48 and #56 are what earning it looked like for the
//! two kernels that make it.
//!
//! It sat behind `--features flash` until its gap list emptied, and then for a
//! while after — which was the bug. A gated kernel is not in what
//! `modal run modal_app.py::build` compiles, so the one example exercising the
//! whole stack at once (both MMA layouts, the swizzled `P` staging tile,
//! [`online_rescale`], masking, and both memory ends) was the one example CI
//! could not catch a regression in.
//!
//! Blocked on:
//!
//! **Nothing.** #11 and #31 were the last two entries and landed together,
//! from opposite ends of the kernel.
//!
//! **#31** shipped `RegTile::add_assign` — the accumulate below — along with
//! an in-place twin for every by-value map, generated from the same
//! `op_methods!` table. What it measured on the way is worth carrying and is
//! not what the issue predicted: at this kernel's `[32, 128]` accumulator the
//! in-place *accumulate* is **not** cheaper than the by-value one, because
//! that call site rebinds and its input is already dead. The place an
//! in-place form is worth 87 registers/thread is the rescale —
//! `row_map::<Mul>` against `scale_rows`, which is now `row_map_assign::<Mul>`
//! and reaches the hand-written number exactly.
//!
//! **#23 and #7 have landed**, together, because they were the same blocker
//! from two sides: this kernel could not name its `[32, 64]` score band, and
//! the unsatisfied layout bound on that band was what kept `make_causal` off
//! the error list. `FragmentLayout` is now a blanket impl over the row and
//! column extents, so the band is a type an out-of-tree crate may write, and
//! the mask takes the coordinate origin a tiled kernel has to give it.
//!
//! **#11 has landed too**, and it is what took this list to one entry. The
//! epilogue writes fp32 out of registers through
//! [`kittens::global::store_rows`], with no shared tile in the way — the
//! round trip it used to owe (pack to bf16, `stmatrix`, TMA out) was a
//! precision loss it never asked for, and the allocation was one this
//! kernel's shared plan could not spare either.
//!
//! ## The shared plan, and the 0 it used to report
//!
//! [`SHARED_BYTES`] is 147532 B — 144 KiB, and 4.5x what `softmax` asks for.
//! It is worth seeing by component rather than as one number:
//!
//! | component | shape | bytes | share |
//! | --- | --- | ---: | ---: |
//! | `Q` tile | `[128, 128]` bf16 | 32768 | 22% |
//! | `K` ring | 3 x `[64, 128]` bf16 | 49152 | 33% |
//! | `V` ring | 3 x `[64, 128]` bf16 | 49152 | 33% |
//! | `P` staging | `[128, 64]` bf16 | 16384 | 11% |
//! | barriers + TMEM word | 9 x 8 B + 4 B | 76 | — |
//! | **total** | | **147532** | |
//!
//! The numbers #70 measured below are quoted at the 147536 this plan was
//! before #125 gave it to [`kittens::plan::SharedPlan`], which reserves the
//! `tcgen05.alloc` staging word as the four-byte `u32` it is. Four bytes out
//! of 144 KiB decides nothing here and the measurements are left as they were
//! taken.
//!
//! Two thirds of it is the pipeline: [`STAGES`] deep over both `K` and `V`,
//! at 16384 B a stage a side. Dropping to two stages would save 32768 B.
//!
//! **It buys nothing, and #70 is what measured that.** The occupancy table
//! used to print `0` blocks/SM for this kernel, which reads like a plan that
//! does not fit and was not one. A block gets 48 KiB of dynamic shared memory
//! without asking; past that the *function* must be opted in, and
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 0 for a function
//! nobody opted in for in exactly the same way it answers 0 for tiles too big
//! to fit. On a B200 the same function at the same 147536 B goes **0 -> 1**
//! across a single [`kittens::launch::admit_shared_plan`] call, with the
//! device's own ceiling at 232448 B — 84912 B of headroom under a plan that
//! was never too large.
//!
//! And 1 is not the shared plan's doing either: swept on a freshly loaded
//! function per probe, this kernel answers 1 block/SM at 147536 B, at 73792,
//! at 32800, and at **0** — flat, and flat across block widths of 32, 64, 128
//! and 256 where `softmax` goes 28/14/7/3 and `layernorm` 8/4/2/1. A ceiling
//! that ignores both shared memory and warp count is a *per-CTA* resource, and
//! the one this kernel holds that neither control does is TMEM: it and the
//! GEMM are the only `tcgen05` entries here.
//!
//! ## The 1 is tcgen05, and it is not this kernel's fault
//!
//! That last paragraph was a guess when #70 wrote it — a correlation over a
//! sample of one, which is the shape #47 exists to warn about. #74 built the
//! control it was missing, and it is `device-tests`' `tmem occupancy ladder`:
//! a nine-register kernel whose entire content is `tcgen05.alloc`, against the
//! byte-identical kernel with the allocator deleted. On a B200, blocks/SM at
//! zero dynamic shared:
//!
//! | rung | 32 | 64 | 128 | 256 threads |
//! | --- | --- | --- | --- | --- |
//! | no tcgen05 | 32 | 32 | 16 | 8 |
//! | `alloc` 32 columns | 1 | 1 | 1 | 1 |
//! | `alloc` 64 / 128 / 192 / 256 / 512 | 1 | 1 | 1 | 1 |
//!
//! **The guess was right and its obvious remedy is wrong.** A CTA that touches
//! the allocator is charged the SM's *whole* tensor memory — the smallest
//! allocation the ISA defines costs exactly what the largest does. The 192 rung
//! takes its column count as a *kernel argument*, so the driver is not reading
//! a number `ptxas` recorded; it is pricing the allocator itself. And this
//! kernel confirms it directly: cut to 32 columns and re-queried, it still
//! answers 1.
//!
//! So none of the three levers anyone had in hand is one. Shrinking the rings
//! buys nothing (#70 measured that), shrinking the TMEM allocation buys nothing
//! (above), and there is no cluster shape to blame — `required_cluster_dimensions`
//! is `None` on every rung of the ladder and tcgen05 implies no `cta_group::2`.
//!
//! **What is left is the only thing left: warps per CTA.** At 1 block/SM the
//! CTA's own width is the entire occupancy of the SM, and this kernel is 128
//! threads — 4 warps where its own `max_threads_per_block` is 512, and where
//! the control above holds 32. That is not a tuning constant to raise: 4 warps
//! is [`QUERIES`] / 32, one warp per 32 accumulator rows, so more warps means a
//! different decomposition — a producer/consumer split with warps dedicated to
//! the TMA and the MMA issue, which is what the shape of a tcgen05 kernel is
//! *for*. This kernel has no launcher and no CPU reference, so that is a design
//! to price rather than a change to make blind, and #74 is where it is written
//! down.
//!
//! ## What already works
//!
//! Everything structural. [`mm_abt`] and [`mm_ab`] are exactly the two MMAs
//! this kernel issues, in the layouts it issues them — and, since #12, in the
//! *accumulator discipline* it issues them under: both start a band fresh,
//! which is now the entry point's name rather than an argument this file has
//! to get right. The semaphore protocol,
//! the swizzled `P` staging tile, and [`online_rescale`] — the fused
//! running-max correction, the one genuinely subtle piece of flash — are all
//! first-class. And the whole register-side body — drain, mask, scale,
//! row-reduce, correct, accumulate, restage — is now the library's own API,
//! line for line, with no index arithmetic left in this file. Both ends are
//! library API too: the drain in, the fp32 store out.
//!
//! This paragraph used to end "nothing in this file reaches past the library
//! any more", and the 2026-07-29 audit caught that being false in the kernel's
//! first three lines: the thread identities and the proxy fence were
//! `cuda_device` calls, because `kittens` exposed neither. They are
//! [`kittens::lane`], [`kittens::warp_id`] and
//! [`kittens::shared::publish_to_async_proxy`] since #127, and the shared plan
//! — `DynamicSharedArray::get_raw`, the `*mut Barrier` cast, the byte offsets
//! — is [`kittens::plan::SharedPlan`] since #125. What still reaches past the
//! library is `thread`'s `sync_threads` and block indices (#124), the
//! allocator's column count (#85), and the kernel ABI itself, which no tile
//! library replaces. None of it is register-side.

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

/// Queries per CTA — one `M128` MMA, four warps of 32 accumulator rows each.
const QUERIES: usize = 128;
/// Keys per pipeline stage.
const KEYS: usize = 64;
/// Head dimension.
const HEAD: usize = 128;
/// Pipeline depth over the key blocks.
const STAGES: usize = 3;
/// `log2(e)`, folded into the score scale so the exponential is `exp2`.
///
/// The library constant rather than a literal: `1.442_695` rounds to the same
/// fp32 bit pattern (`0x3fb8aa3b`), so this is not a numerics change, but
/// `clippy::approx_constant` is deny-by-default and the literal was a build
/// error nobody saw while this file sat behind a cargo feature. It is also what
/// `reg::Exp` scales by, so the kernel and the op now name one constant.
const LOG2E: f32 = core::f32::consts::LOG2_E;

type QTile = SharedTile<Bf16, QUERIES, HEAD, Swizzle128B>;
/// The probabilities, staged back to shared as the `A` operand of `O = P·V`.
/// Exactly one swizzle atom wide, which is what the `stmatrix` store path
/// needs.
type PTile = SharedTile<Bf16, QUERIES, KEYS, Swizzle128B>;
/// `KTile` and `VTile` used to sit beside these, and #29 is what retired them:
/// their only remaining use was the `KTile::BYTES + VTile::BYTES` this kernel
/// charged its stage barrier, and the loads hand that back themselves now. A
/// stage's tile is `k_ring.tile(block)`, which is where its shape already was.
type KRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;
type VRing = SharedTileRing<Bf16, KEYS, HEAD, Swizzle128B, STAGES>;

type Scores = TmemTile<QUERIES, KEYS>;
type Output = TmemTile<QUERIES, HEAD>;

/// One warp's band of the score tile — `[32, 64]`, the shape #23 was filed
/// about and the reason this line is out of tree at all.
type ScoreBand = RegTile<32, KEYS, BaseLdtm>;
/// One warp's band of the output accumulator.
type OutBand = RegTile<32, HEAD, BaseLdtm>;
/// Row statistics of either band.
type Rows = RegVec<32, BaseLdtm>;

/// Columns of tensor memory the CTA allocates: the `[128, KEYS]` scores and
/// the `[128, HEAD]` output beside them, rounded up to the allocator's grain.
///
/// `KEYS + HEAD` is 192 and **192 is not a legal `tcgen05.alloc`** — the
/// operand must be a power of two in `[32, 512]` (cuda-oxide spells the set out
/// as "32, 64, 128, 256, 512" on `TmemGuard`). This kernel asked for 192 from
/// the day it was written and nothing caught it, because it has no launcher:
/// the column count is an argument to an instruction, so no type and no
/// `const` assertion was ever in a position to see it. The one below is.
///
/// Rounding up costs **nothing**, which is not obvious and is #74's finding:
/// the driver charges a CTA the SM's *entire* tensor memory the moment it
/// touches the allocator, at 32 columns exactly as at 512, so the 64 columns
/// past `KEYS + HEAD` are free. Measured — see the module doc.
const TMEM_COLUMNS: u32 = 256;
const _: () = assert!(
    TMEM_COLUMNS as usize >= KEYS + HEAD
        && TMEM_COLUMNS.is_power_of_two()
        && TMEM_COLUMNS >= 32
        && TMEM_COLUMNS <= 512,
    "tcgen05.alloc takes a power of two in [32, 512] that covers the scores and the output"
);

/// Everything the launch's dynamic shared memory holds, in declaration order.
///
/// This file used to state the layout twice in the same 40 lines — once as a
/// `SCRATCH_BYTES` in bytes and once as a pointer walk in `Barrier` units, so
/// the barrier count read `2 * STAGES * 8 + 3 * 8` at one end and
/// `.add(2 * STAGES + 3)` at the other. #125 is that finding; it is one walk
/// now, and the units are the handles'.
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

/// Dynamic shared memory the launch declares — 144 KiB and change, which is
/// 4.5× what `softmax` asks for and why this kernel needs
/// [`kittens::launch::admit_shared_plan`].
///
/// 147 532 rather than the 147 536 this file carried until #125: the staging
/// word `tcgen05.alloc` writes is a `u32` and [`SharedPlan::tmem_slot`]
/// reserves four bytes for it at four-byte alignment, where the hand-written
/// `SCRATCH_BYTES` charged eight. Nothing else in the plan moved, and the
/// residency cannot: this plan is over the 116 736 B half of an SM either way,
/// so it is one CTA per SM at both numbers.
pub const SHARED_BYTES: usize = shared(SharedPlan::sizing()).plan.bytes();
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
            // Scores and output share one allocation: `[128, KEYS]` then
            // `[128, HEAD]` beside it.
            let tmem = alloc_block(tmem_slot, TMEM_COLUMNS);
            let scores = Scores::from_raw(tmem);
            let output: Output = scores.split_columns();

            if leader {
                q_loaded.expect_tx(q.tma_load(q_map, query_base as i32, head, q_loaded));
            }

            // Only the accumulator bands' shapes: `mm_abt`/`mm_ab` each carry
            // the transpose configuration their own walk reads under, and the
            // element comes from the tiles.
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

                // The whole score band in one call (#22), composed out of the
                // `[16, 16]` blocks LDTM delivers.
                let mut s: ScoreBand = scores.tile(32 * warp_id, 0);

                // Causal masking against this band's own origin (#7). The two
                // bases go in separately: their difference is negative for
                // every band above the diagonal, which is most of them, and
                // `query_base - key_base` in `u32` at this call site would
                // wrap and mask nothing.
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
                    // `mm` and not `mma` (#12): the running output lives in
                    // registers, so TMEM only ever holds this block's `P·V`,
                    // and that is now a fact about which entry point this
                    // calls rather than a `false` a reader has to check.
                    mm_ab(output.raw(), p, v_ring.tile(block), output_shape);
                    commit(accumulated);
                }
                accumulated.wait(block & 1);

                let contribution: OutBand = output.tile(32 * warp_id, 0);
                // The accumulate, in place (#31). `out_acc = out_acc.add(c)`
                // costs the same registers at this width — its input is dead
                // at the rebinding, which is the case #38 predicted would be
                // free — and 512 more bytes of stack frame
                // (`softmax_probe_128_add_assign`, 168 regs / 2560 B against
                // 168 / 3072). The reason to write it this way is that the
                // accumulator's input *is* its output.
                out_acc.add_assign(contribution);
                if leader {
                    free.sem(block).arrive();
                }
                block += 1;
            }

            // The softmax denominator, broadcast down the rows (#5).
            let out_acc = out_acc.div_row(running_sum);

            // fp32 straight out of registers, at the coordinates the fragment
            // layout already knows (#11). The output is packed `[_, HEAD]`, so
            // the cursor's stride is the band's own width and its column base
            // is zero — the degenerate case of the stride the GEMM's epilogue
            // needs, spelled the same way.
            store_rows(
                GlobalRows::<F32>::from_slice(&mut out, HEAD),
                query_base + 32 * warp_id,
                0,
                lane,
                out_acc,
            );

            thread::sync_threads();
            // The columns go back. tcgen05 allocations are not scoped to the
            // CTA the way shared memory is — a kernel that exits holding them
            // is a `CUDA_ERROR_TENSOR_MEMORY_LEAK`, and the next CTA scheduled
            // onto the SM is the one that pays. This kernel had no `dealloc`
            // at all until #74 went looking at its allocator, which nothing
            // could have caught: `device-tests`' `repeated launch` case is the
            // standing guard for exactly this class, and a kernel with no
            // launcher never reaches a guard that works by launching twice.
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
