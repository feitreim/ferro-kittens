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
//! [`check`] launches it against a CPU reference and [`bench`] times the same
//! launch afterwards, in that order: a number only ever comes out of a checked
//! run. The seed puts a score ramp on the key axis, so the running max moves
//! at nearly every key block and the keys above the diagonal are the
//! best-scored ones a broken mask could admit; the check runs four key blocks,
//! one past [`STAGES`], so the ring wraps.
//!
//!     modal run modal_app.py::examples
//!
//! The shared plan by component, the 1 block/SM ceiling and why it is tcgen05
//! rather than this plan, the register-side measurements, the seed argument
//! and the tolerance: `docs/kernels/flash_forward.md`.

use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::bench::{Shape, Timings, time};
use std::error::Error;

use kittens::global::{GlobalRows, store_rows};
use kittens::ldst::store_tile;
use kittens::mma::{commit, mm_ab, mm_abt};
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
///
/// The power-of-two half of that used to be asserted here, because the count
/// was an argument to an instruction and no type could see it. `alloc_block`
/// takes it as a `const` parameter now (#128) and carries its own rule; what is
/// left here is the half that is this kernel's, that the two segments fit.
const TMEM_COLUMNS: usize = 256;
const _: () = assert!(
    TMEM_COLUMNS >= KEYS + HEAD,
    "the allocation must cover the scores and the output beside them"
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
            let tmem = alloc_block::<TMEM_COLUMNS>(tmem_slot);
            let scores = Scores::from_raw(tmem);
            let output: Output = scores.split_columns();

            if leader {
                q_loaded.expect_tx(q.tma_load(q_map, query_base as i32, head, q_loaded));
            }

            // Neither MMA names a shape. `mm_abt` derives `M128_N64` from
            // `Scores` and `mm_ab` derives `M128_N128` from `Output` (#128).
            // The output's used to be the *band's* — `M128_N128` had band 1
            // overwrite half of band 0 (kittens #175) — and this kernel had it
            // wrong from the day it was written, which having no reference `O`
            // is why (#85). A shape read off the accumulator cannot be that.
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
                    mm_abt(scores, q, k_ring.tile(block));
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
                    mm_ab(output, p, v_ring.tile(block));
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

            // The output is packed [heads, queries, HEAD] — the panel axis of
            // the three input maps, spelled on the output's row axis. The head
            // term of this sum was missing until the first checked run: the
            // TMA maps take the head as a plane coordinate and could not get
            // it wrong, but this cursor takes the address it is handed, and
            // every head wrote query block x's rows of the same panel.
            let out_base = QUERIES as u32 * thread::gridDim_x() * thread::blockIdx_y() + query_base;
            store_rows(
                GlobalRows::<F32>::from_slice(&mut out, HEAD),
                out_base + 32 * warp_id,
                0,
                lane,
                out_acc,
            );

            thread::sync_threads();
            // tcgen05 allocations are not scoped to the CTA the way shared
            // memory is: a kernel that exits holding them is a
            // `CUDA_ERROR_TENSOR_MEMORY_LEAK`, and the next CTA scheduled onto
            // the SM is the one that pays.
            dealloc_block::<TMEM_COLUMNS>(tmem);
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

/// One key block past [`STAGES`], so the ring wraps and `free`'s recycled
/// phase is waited on for real.
pub const CHECK_SEQUENCE: usize = 4 * KEYS;
/// Two, so a head plane read or written in the wrong place is a wrong number
/// rather than a coincidence.
pub const CHECK_HEADS: usize = 2;
/// A power of two, so folding [`LOG2E`] into it is the only rounding the
/// score scale carries.
pub const SCALE: f32 = 0.125;

/// Column strides the row axis cycles through. Odd, and per row rather than
/// per band: two rows of one warp band must not share a stride, or their
/// scores against a common key correlate.
const STRIDES: usize = 63;

/// The seed's one ladder: 128 lattice points, permuted per (tensor, head,
/// row). `tensor` shifts both the stride cycle and the offset so Q, K and V
/// draw different permutations of the same lattice.
fn permutation(tensor: usize, head: usize, row: usize, column: usize) -> usize {
    let stride = 11 + 2 * ((row + 17 * tensor) % STRIDES);
    (53 * head + 37 * row + stride * column + 71 * tensor) % 128
}

/// Dimension 0 of `Q` and `K` is a carrier pair: `2 · row/8` puts a ramp on
/// the key axis, so later key blocks score higher and the running max moves —
/// which is the online rescale actually being exercised, not merely compiled.
/// The other 127 dimensions are zero-centred lattice noise, which is what
/// spreads each row's weights over many keys. Every value is bf16-exact.
fn q_value(head: usize, row: usize, dim: usize) -> f32 {
    if dim == 0 {
        2.0
    } else {
        (permutation(0, head, row, dim) as f32 - 64.0) / 64.0
    }
}

fn k_value(head: usize, row: usize, dim: usize) -> f32 {
    if dim == 0 {
        row as f32 / 8.0
    } else {
        (permutation(1, head, row, dim) as f32 - 64.0) / 64.0
    }
}

/// `V` in `[1, 3)`: the output is a convex combination of these, so every
/// output sits in `[1, 3)` too and a relative error is never against a
/// cancellation the reference does not also perform.
fn v_value(head: usize, row: usize, dim: usize) -> f32 {
    1.0 + permutation(2, head, row, dim) as f32 / 64.0
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The value the device actually sees: the seed is bf16-exact by
/// construction, but the reference states its inputs as "bf16-rounded" rather
/// than trusting that.
fn rounded(value: f32) -> f64 {
    from_bf16(to_bf16(value)) as f64
}

/// `[heads, sequence, HEAD]` bf16 staging words for one input tensor — the
/// panel layout all three maps describe, head-major.
fn staged(sequence: usize, heads: usize, at: impl Fn(usize, usize, usize) -> f32) -> Vec<u32> {
    let mut staged = Vec::with_capacity(heads * sequence * HEAD / 2);
    for head in 0..heads {
        for row in 0..sequence {
            for pair in 0..HEAD / 2 {
                let (low, high) = (at(head, row, 2 * pair), at(head, row, 2 * pair + 1));
                staged.push(to_bf16(low) as u32 | ((to_bf16(high) as u32) << 16));
            }
        }
    }
    staged
}

/// One head's inputs in f64 over the bf16-rounded seed, flat `[sequence, HEAD]`.
struct Panels {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
}

fn panels(head: usize, sequence: usize) -> Panels {
    let tensor = |at: &dyn Fn(usize, usize, usize) -> f32| -> Vec<f64> {
        (0..sequence * HEAD)
            .map(|flat| rounded(at(head, flat / HEAD, flat % HEAD)))
            .collect()
    };
    Panels {
        q: tensor(&q_value),
        k: tensor(&k_value),
        v: tensor(&v_value),
    }
}

/// The straightforward stable causal softmax times `V` for one query row, in
/// f64: scores against keys `0..=row`, subtract the row max, exponentiate,
/// weight `V`, divide by the sum. The kernel's `exp2`/[`LOG2E`] folding and
/// its online rescale are ways of computing exactly this.
fn reference_row(panels: &Panels, row: usize) -> Vec<f64> {
    let q = &panels.q[row * HEAD..(row + 1) * HEAD];
    let scores: Vec<f64> = (0..=row)
        .map(|key| {
            let k = &panels.k[key * HEAD..(key + 1) * HEAD];
            SCALE as f64 * q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>()
        })
        .collect();
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    (0..HEAD)
        .map(|dim| {
            weights
                .iter()
                .enumerate()
                .map(|(key, w)| w * panels.v[key * HEAD + dim])
                .sum::<f64>()
                / total
        })
        .collect()
}

fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch over `sequence` queries and keys by `heads` heads, compare every
/// output against the CPU reference, and only then hand the launch to `then`.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    sequence: usize,
    heads: usize,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig};
    use kittens::global::encode_bf16_panels;

    /// One doubling above the 1.66e-3 a correct kernel measures, which is
    /// `P`'s trip to bf16 on its way to being the second MMA's `A` operand;
    /// the argument is in `docs/kernels/flash_forward.md`.
    const TOLERANCE: f32 = 1.0 / 256.0;

    if !sequence.is_multiple_of(QUERIES) || heads == 0 {
        return Err(format!(
            "{sequence} queries x {heads} heads does not divide the {QUERIES} queries a CTA owns"
        )
        .into());
    }
    let key_blocks = (sequence / KEYS) as u32;

    let stream = context.default_stream();
    let module = kernels::load(context)?;
    // 144 KiB of dynamic shared memory: past 48 KiB the function must be
    // opted in or the launch is inadmissible, not slow.
    let function = module.as_cuda_module().load_function("flash_forward")?;
    kittens::launch::admit_shared_plan(&function, SHARED_BYTES as u32)?;

    let q = DeviceBuffer::from_host(&stream, &staged(sequence, heads, q_value))?;
    let k = DeviceBuffer::from_host(&stream, &staged(sequence, heads, k_value))?;
    let v = DeviceBuffer::from_host(&stream, &staged(sequence, heads, v_value))?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, heads * sequence * HEAD)?;
    // SAFETY: all three buffers outlive every launch consuming their maps
    // below.
    let (q_map, k_map, v_map) = unsafe {
        (
            encode_bf16_panels::<QUERIES, HEAD>(&stream, q.cu_deviceptr(), sequence, heads)?,
            encode_bf16_panels::<KEYS, HEAD>(&stream, k.cu_deviceptr(), sequence, heads)?,
            encode_bf16_panels::<KEYS, HEAD>(&stream, v.cu_deviceptr(), sequence, heads)?,
        )
    };

    let config = LaunchConfig {
        grid_dim: ((sequence / QUERIES) as u32, heads as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: SHARED_BYTES as u32,
    };
    // Takes the output buffer as an argument rather than capturing it: the
    // check below reads `out` between the verifying launch and the timed
    // ones, so the closure must not hold the mutable borrow across it.
    let launch = |out: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn Error>> {
        // SAFETY: the grid covers exactly the query blocks and heads the
        // three maps describe, `out` holds a `[sequence, HEAD]` fp32 panel
        // per head, and the block and shared plan are the kernel's own.
        unsafe {
            module.flash_forward(
                &stream,
                config,
                q_map.as_ptr(),
                k_map.as_ptr(),
                v_map.as_ptr(),
                key_blocks,
                SCALE,
                out,
            )?
        };
        Ok(())
    };
    launch(&mut out)?;

    let observed = out.to_host_vec(&stream)?;
    let (mut wrong, mut sample, mut worst) = (0usize, Vec::new(), 0.0f32);
    for head in 0..heads {
        let panels = panels(head, sequence);
        for row in 0..sequence {
            let expected = reference_row(&panels, row);
            for (column, expected) in expected.iter().enumerate() {
                let value = observed[(head * sequence + row) * HEAD + column];
                let error = ((value as f64 - expected) / expected).abs() as f32;
                worst = worst.max(error);
                // Negated rather than `error > TOLERANCE`: a NaN compares
                // false either way, so only this spelling counts one as wrong.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(error <= TOLERANCE) {
                    wrong += 1;
                    if sample.len() < 8 {
                        sample.push(format!(
                            "[{head}, {row}, {column}] = {value}, want {expected} ({error:.2e})"
                        ));
                    }
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!("{wrong} outputs outside 2^-8: {}", sample.join("; ")).into());
    }

    let mut launch_once = || launch(&mut out);
    let after = then(&stream, &mut launch_once)?;
    Ok((
        format!("{heads}x{sequence}x{HEAD} rows attended, worst relative error {worst:.2e}"),
        after,
    ))
}

pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    run(context, CHECK_SEQUENCE, CHECK_HEADS, nothing_after).map(|(note, ())| note)
}

/// The benchmark's entry point. A flash-forward [`Shape`] is `sequence x HEAD
/// x heads`.
pub fn bench(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
) -> Result<Timings, Box<dyn Error>> {
    run(context, shape.m, shape.k, time).map(|(_, timings)| timings)
}
