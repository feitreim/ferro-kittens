//! # NVLabs' own `gemm_sol_final`, unported, on this harness' clock
//!
//! [`crate::gemm_sol`] is a *port*: the same design rebuilt through
//! ferro-kittens' typed tiles and pipeline primitives. It measures 0.806 /
//! 0.873 / 0.946 of cuBLASLt at 4096³ / 8192³ / 16384³, and that gap had two
//! candidate explanations nobody could separate — the port lost something the
//! original has, or the original itself does not reach cuBLASLt on a B200.
//!
//! **It is the second, mostly.** Upstream unported measures 0.877 / 0.911 /
//! 0.966 of the same live cuBLASLt at the same variant, so 62–69% of the
//! shortfall is where `gemm_sol_final` itself stands on this device and 31–38%
//! is the port's. And at 8192³ under upstream's *shipped* selector — M256xN256,
//! where #138 crosses over to M512xN256 — upstream measures 0.839 and the port
//! beats it by 4.0%. `experiments/README.md` §7 holds all of it.
//!
//! This module is the second reading the separation needs. The device code
//! below is **byte-identical to upstream's**: `gemm_sol_upstream_kernels.rs` is
//! `cuda-oxide`'s `crates/rustc-codegen-cuda/examples/gemm_sol_final/src/
//! kernels.rs` copied without a character changed, `include!`d here exactly as
//! upstream `include!`s it into its own `main.rs` and for the same reason —
//! `#[cuda_module]` wants an inline module with a brace body. Both entry
//! points, both shape contracts, both epilogues, upstream's `swizzle_g`,
//! upstream's `PEER_BIT_MASK` barrier aliasing. `sha256` of the copy is in the
//! pull request that added it, against the checkout it came from.
//!
//! Upstream's revision is **`b099f64c1a32869b74be99f4f88242fb68655b51`**, which
//! is what the port was built and measured against. `Cargo.toml` now pins
//! `20a56163f258e09f2c51e4c27ae4e4ff17582443`, and `gemm_sol_final/` is
//! *byte-identical* at the two — upstream's `kernels.rs` and the vendored
//! `gemm_sol_upstream_kernels.rs` all hash to
//! `6746e517eca19fdcc01cb0d5003e924bb638f5ee42e8c333d457a0d6f334d6e9` — so this
//! measurement does not turn on which of the two is in the manifest.
//!
//! ## What is upstream's and what is this harness'
//!
//! Everything between the two CUDA events is upstream's. Everything outside
//! them is this crate's, deliberately, because the comparison is worthless
//! otherwise — it is the same rule `cublaslt.rs` states for the library
//! baseline, applied a third time:
//!
//! - **The clock.** [`crate::bench::time`], so five discarded warm-ups and the
//!   minimum of thirty timed launches on the default stream. Upstream's own
//!   `main.rs` takes one event pair around a hundred launches and divides —
//!   a *mean*, which cannot be compared to a minimum, and it is upstream's own
//!   figures that would be the ones understated.
//! - **The operands, and the check.** [`crate::gemm_sol::stage_f16`] and
//!   [`crate::gemm_sol::check_output`] — the same functions the port is staged
//!   and checked by, not copies of them. So the two kernels read
//!   byte-identical `A` and `B` and are held to the same exact-BF16 output
//!   comparison, and rule 1 of `bench.rs` reaches this module too: nothing
//!   here returns a time that did not first pass that check.
//! - **The launch geometry and the descriptors are upstream's**, verbatim:
//!   `2 · (m/256) · (n/128)` CTAs of 192 threads and no dynamic shared memory
//!   for *both* variants, and the two `cuTensorMapEncodeTiled` calls copied out
//!   of upstream's `main.rs`. This kernel does not take a kittens tensor map
//!   and is not given one.
//! - **The variant policy is upstream's.** [`bench`] is upstream's selector —
//!   `preferred_from_tiles_m` is 0 for M256xN256 and 64 for M512xN256 over
//!   `tiles_m = m/256`, so M256 serves 4K and 8K and M512 serves 16K.
//!   [`bench_m512`] forces the large entry at every size, because the port
//!   crosses over at 8K and a row that compares two kernels at two different
//!   tile shapes answers a question nobody asked.
//!
//! ## The one thing that is not upstream's and could have been
//!
//! Upstream loads its PTX from a file it built beside the binary; this loads
//! the embedded bundle `#[cuda_module]` generates, as every other kernel in
//! this crate does. Neither is inside the events.

use std::error::Error;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
    mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
    tcgen05_relinquish_alloc_permit_cg2,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2};
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, thread, warp};
// The one name below that does not resolve to what upstream's `main.rs`
// resolves it to, and the reason this module exists at all rather than a copy
// of upstream's binary.
//
// `cuda_device::tcgen05::stmatrix_m8n8_x2` does not lower for `sm_100a` on
// this toolchain: it reaches the NVPTX back end as
// `llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3`, is selected by nothing, and
// is emitted as a call to an `.extern .func` that does not exist. `ptxas`
// stops on it — *"line 10; fatal: Parsing error near '.nvvm'"* — and since a
// package's kernels share one embedded bundle, one such call takes down the
// load of every kernel in this crate, the port included. It is the same defect
// `kittens::ldst` documents at `b099f64` and `20a5616` alike and works around
// with `ptx_asm!`, and
// `kittens::ldst::stmatrix_m8n8_x2` has upstream's exact signature and emits
// upstream's exact instruction, `stmatrix.sync.aligned.m8n8.x2.shared.b16`.
//
// So this is a one-line change to *which implementation of one instruction*
// the device code links against. `gemm_sol_upstream_kernels.rs` is still not
// modified — it cannot be, and the alternative is no measurement at all.
use kittens::ldst::stmatrix_m8n8_x2;

use crate::bench::{Shape, Timings, time};
use crate::gemm_sol::{check_output, stage_f16};

/// Upstream's shared-memory matrix descriptor encoder, copied out of its
/// `main.rs`. The device code below calls it and nothing else of the host's.
///
/// ```text
///   [0:13]  base_addr >> 4
///   [16:29] LBO >> 4 (leading byte offset — stride to next core matrix RIGHT)
///   [32:45] SBO >> 4 (stride byte offset — stride to next core matrix DOWN)
///   [46]    fixed 0b1
///   [61:63] swizzle mode
/// ```
#[inline(always)]
fn build_smem_descriptor(
    smem_addr: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    let swizzle_bits = (swizzle as u64) << 61;

    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit | swizzle_bits
}

// Upstream's device code, not one character of it this repository's. `include!`
// rather than a module for upstream's own reason: `#[cuda_module]` rejects a
// file module and wants a brace body.
include!("gemm_sol_upstream_kernels.rs");

/// Upstream's `A` map: `[k, m]` f16, a `64 × 128` box, 128-byte swizzle.
/// Copied from its `main.rs`.
fn create_tma_descriptor_f16_swizzled(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn Error>> {
    create_tma_descriptor_f16_swizzled_box(global_address, global_width, global_height, 64, 128)
}

/// Upstream's `B` map, and the general form of the one above: `[k, n]` f16 with
/// the box stated, 128-byte swizzle. Copied from its `main.rs`.
fn create_tma_descriptor_f16_swizzled_box(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [box_k, box_mn];
    let element_strides: [u32; 2] = [1, 1];

    // SAFETY: every pointer is to a live local of the width the driver's
    // documented signature takes, and `global_address` is a device allocation
    // of `global_width * global_height` f16 that outlives every launch made
    // through the resulting map.
    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "cuTensorMapEncodeTiled (SWIZZLE_128B, box {box_k}x{box_mn}) failed: {result:?}"
        )
        .into());
    }

    // SAFETY: the call returned `CUDA_SUCCESS`, which is the driver's statement
    // that it wrote the map.
    Ok(unsafe { tensor_map.assume_init() })
}

/// Upstream's two compiled entry points, which differ in their M macro tile.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    M256xN256,
    M512xN256,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::M256xN256 => "upstream M256xN256",
            Variant::M512xN256 => "upstream M512xN256",
        }
    }

    fn m_tile(self) -> usize {
        match self {
            Variant::M256xN256 => 256,
            Variant::M512xN256 => 512,
        }
    }
}

/// Upstream's own selector, restated: `preferred_from_tiles_m` is 0 for the
/// small entry and 64 for the large one, over `tiles_m = m / 256`. So M256
/// serves 4096 and 8192, and M512 serves 16384.
///
/// The port crosses over one size earlier (#138 measured that crossover worth
/// ~190 TFLOP/s at 8192³ *to the port*), which is exactly why [`bench_m512`]
/// exists beside this.
pub const fn select_variant(m: usize) -> Variant {
    if m / 256 >= 64 {
        Variant::M512xN256
    } else {
        Variant::M256xN256
    }
}

/// Upstream's shape contract, which its `KernelProblem::validate` states as
/// `m % m_tile == 0`, `n % 256 == 0` and `k % 256 == 0`.
fn validate_shape(m: usize, n: usize, k: usize, variant: Variant) -> Result<(), Box<dyn Error>> {
    if m < variant.m_tile()
        || !m.is_multiple_of(variant.m_tile())
        || n < 256
        || !n.is_multiple_of(256)
        || k < 256
        || !k.is_multiple_of(256)
    {
        return Err(format!(
            "{m}x{n}x{k} violates {}'s M{}xN256, K%256 contract",
            variant.name(),
            variant.m_tile(),
        )
        .into());
    }
    Ok(())
}

/// The grid upstream launches, for both variants: work IDs are M256xN128 and
/// each entry drains the ones it does not own through CLC.
pub fn grid(shape: Shape) -> u32 {
    2 * (shape.m / 256 * (shape.n / 128)) as u32
}

/// Stage upstream's operands, launch once, check every output, and only then
/// hand the launch to `then`.
///
/// The same shape as `gemm_sol::run` on purpose: the two differ in the
/// kernel and in the descriptors it is given, and in nothing else a timing
/// could come out of.
fn run<T>(
    context: &Arc<CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    variant: Variant,
    initialize: bool,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    validate_shape(m, n, k, variant)?;
    let stream = context.default_stream();
    // Safe, unlike every other `load` in this crate: upstream's kernels carry
    // no `#[launch_contract]`, so the macro emits the unchecked launch path and
    // a safe loader, and it is each launch below that is `unsafe`.
    let module = kernels::load(context)?;

    let a = if initialize {
        DeviceBuffer::from_host(&stream, &stage_f16(m, k, crate::gemm_sol::a_value))?
    } else {
        DeviceBuffer::<u16>::zeroed(&stream, m * k)?
    };
    let b = if initialize {
        DeviceBuffer::from_host(&stream, &stage_f16(n, k, crate::gemm_sol::b_value))?
    } else {
        DeviceBuffer::<u16>::zeroed(&stream, n * k)?
    };

    let a_map = create_tma_descriptor_f16_swizzled(
        a.cu_deviceptr() as *mut std::ffi::c_void,
        k as u64,
        m as u64,
    )?;
    let b_map = create_tma_descriptor_f16_swizzled_box(
        b.cu_deviceptr() as *mut std::ffi::c_void,
        k as u64,
        n as u64,
        64,
        64,
    )?;
    // Named, not shadowed: the maps live in device memory for as long as any
    // launch can read them, and binding the pointer over the buffer would free
    // what the pointer points at.
    let a_map_buffer = DeviceBuffer::from_host(&stream, &a_map.opaque)?;
    let b_map_buffer = DeviceBuffer::from_host(&stream, &b_map.opaque)?;
    let a_map = a_map_buffer.cu_deviceptr() as *const TmaDescriptor;
    let b_map = b_map_buffer.cu_deviceptr() as *const TmaDescriptor;

    // Two BF16 outputs per word, which is the shape of upstream's epilogue
    // store and therefore of its output parameter.
    let mut c = DeviceBuffer::<u32>::zeroed(&stream, m * n / 2)?;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let config = LaunchConfig {
        grid_dim: (grid(Shape { m, n, k }), 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };
    let (stream_ref, module_ref) = (&stream, &module);
    let (n_arg, k_arg) = (n as i32, k as i32);

    let launch_once: Box<dyn Fn(&mut DeviceBuffer<u32>) -> Result<(), Box<dyn Error>>> =
        match variant {
            Variant::M256xN256 => Box::new(move |out| {
                // SAFETY: the shape passed `validate_shape`, the maps describe
                // the live `A` and `B` above, and `out` holds `m * n / 2` words
                // — which is upstream's own launch, argument for argument.
                unsafe {
                    module_ref.gemm_sol_clc_multicast_4_stage_pipeline(
                        stream_ref, config, a_map, b_map, out, n_arg, k_arg, tiles_m, tiles_n,
                    )?
                };
                Ok(())
            }),
            Variant::M512xN256 => Box::new(move |out| {
                // SAFETY: as above, for the large entry.
                unsafe {
                    module_ref.gemm_sol_clc_multicast_4_stage_pipeline_large(
                        stream_ref, config, a_map, b_map, out, n_arg, k_arg, tiles_m, tiles_n,
                    )?
                };
                Ok(())
            }),
        };

    launch_once(&mut c)?;
    stream.synchronize()?;
    let label = if initialize {
        // Upstream packs column `2i` in the low half of word `i`, so the words
        // read back little-endian *are* `C` row-major in BF16 — the same slice
        // the port's own check takes.
        let words = c.to_host_vec(&stream)?;
        let observed: Vec<u16> = words
            .iter()
            .flat_map(|&word| [word as u16, (word >> 16) as u16])
            .collect();
        let worst = check_output(&observed, m, n, k)?;
        format!(
            "{} {m}x{n}x{k} exact over {} BF16 outputs, worst |rel| {worst:.2e}",
            variant.name(),
            m * n,
        )
    } else {
        format!("{} {m}x{n}x{k}", variant.name())
    };
    let mut launch = || launch_once(&mut c);
    let result = then(&stream, &mut launch)?;
    Ok((label, result))
}

fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Both upstream entry points against the same CPU reference the port is held
/// to, at the size `modal_app.py::examples` checks everything else at.
pub fn check(context: &Arc<CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for variant in [Variant::M256xN256, Variant::M512xN256] {
        notes.push(run(context, 1024, 1024, 512, variant, true, nothing_after)?.0);
    }
    Ok(notes.join("; "))
}

fn timed(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
) -> Result<Timings, Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    run(context, 1024, 1024, 512, variant, true, nothing_after)?;
    Ok(run(context, m, n, k, variant, false, time)?.1)
}

/// Upstream's own variant policy: M256xN256 at 4K and 8K, M512xN256 at 16K.
pub fn bench(context: &Arc<CudaContext>, shape: Shape) -> Result<Timings, Box<dyn Error>> {
    timed(context, shape, select_variant(shape.m))
}

/// The large entry at every size, which is the row that lines up with the
/// port's own crossover at 8192³.
pub fn bench_m512(context: &Arc<CudaContext>, shape: Shape) -> Result<Timings, Box<dyn Error>> {
    timed(context, shape, Variant::M512xN256)
}
