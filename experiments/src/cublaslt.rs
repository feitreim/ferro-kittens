//! # cuBLASLt beside the GEMM — the denominator `bench` did not have
//!
//! `bench` reported the GEMM at 1069–1179 TFLOP/s and had nothing to divide by
//! except a spec sheet. "48% of dense peak" is a real number and the wrong
//! comparison: what a kernel is worth is how far it sits from a tuned library
//! **on the same device, at the same shape, on the same day**. This module is
//! that library, called in-process, so every future GEMM change carries a
//! ratio instead of an absolute somebody has to re-contextualize.
//!
//! ## Why FFI and not a separate binary
//!
//! Upstream cuda-oxide compiles a C program and shells out to it. That measures
//! the right kernel and the wrong thing: a subprocess times on its own stream,
//! in its own process, with its own warm-up count and its own reduction over
//! its own samples, so the difference between the two figures is two
//! *harnesses* as much as two kernels. Here the baseline goes through
//! [`crate::bench::time`] — the same CUDA events, the same stream, the same
//! five discarded warm-ups, the same minimum over thirty timed launches — and
//! the only thing left different is the code that runs between the events.
//!
//! It costs about a dozen `extern "C"` declarations and two structs. See
//! `examples/build.rs` for the link step and `examples/cublaslt_abi.c` for the
//! constants, asserted against the real headers rather than trusted.
//!
//! ## The configuration, and the trap in it
//!
//! [`crate::gemm`] computes `C = A·Bᵀ` with **bf16 operands, an fp32
//! accumulator and a bf16 `C`**, `C` row-major with `ldc = n`. So:
//! `CUDA_R_16BF` in, `CUBLAS_COMPUTE_32F` across, `CUDA_R_16BF` out.
//!
//! **The output was fp32 on both sides through #107, and #108 moved both.**
//! This paragraph used to say that upstream's C file uses a 2-byte output and
//! is therefore not the same measurement — which was true, and had the
//! asymmetry the wrong way round: it is a training GEMM's ordinary signature
//! that upstream was using and our fp32 `C` that was unusual. The two halves
//! have to move together or the comparison is worthless: at 8192³ an fp32 `C`
//! is 268 MB and a bf16 one is 134 MB, so leaving this line at `CUDA_R_32F`
//! would time a baseline writing twice what the kernel it divides writes, and
//! report the difference as ours. Every ratio published before #108 is against
//! the fp32 pair and is not comparable to one measured after it.
//!
//! cuBLASLt is column-major and ours is row-major, and reconciling that is
//! where a baseline goes wrong. Written out, with `Â` meaning "the bytes of
//! buffer `a`, read as column-major":
//!
//! - `a` holds `[m, k]` row-major with `k` contiguous, so `Â` is `k × m`,
//!   `ld = k`, and `Â = Aᵀ`.
//! - `b` holds `[n, k]` row-major, so `B̂` is `k × n`, `ld = k`, and `B̂ = Bᵀ`.
//! - `c` holds `[m, n]` row-major, so `Ĉ` is `n × m`, `ld = n`, and `Ĉ = Cᵀ`.
//!
//! The product cuBLASLt must be asked for is therefore not `C` but its
//! transpose, `Ĉ = Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ = (B̂)ᵀ · Â`. Reading that right to
//! left: **cuBLASLt's first operand is our `b`** under `CUBLAS_OP_T`, its
//! second is our `a` under `CUBLAS_OP_N`, and the output layout is `(n, m, n)`.
//! The operands are *swapped* relative to the obvious spelling.
//!
//! The obvious spelling is not slower and does not fail. It computes a
//! different matrix at full speed, and a baseline built on it would be a
//! number with nothing wrong on its face. Which is why nothing in this module
//! reports a time before [`gemm::check_c`] has compared its output — the same
//! function, not a copy of it, that our own kernel is checked by.
//!
//! ### The check really is exact, and that is not luck
//!
//! [`gemm::a_value`] and [`gemm::b_value`] are small integers, every product is
//! exact in bf16, and every partial sum stays under 983,040 against fp32's
//! exact integer range of 2²⁴. Integer addition in fp32 below that bound is
//! exact **in any order**, so it does not matter which tiling cuBLASLt picks,
//! whether it splits K, or in what order it reduces the pieces: the fp32 value
//! reaching the epilogue is the same integer ours is.
//!
//! **A bf16 output does not cost that**, which is the part worth stating
//! plainly. Rounding one exactly-known integer to bf16 is a deterministic
//! function of it, so the comparison stays `==` on the 16-bit words against a
//! reference [`gemm::check_c`] rounds the same way, and no tolerance was
//! introduced on either side. A tolerance is what would have let a subtly wrong
//! baseline through — and here it would also have hidden a library that rounded
//! a *partial* sum, which is the one new way this baseline could differ from
//! ours and the one this comparison still catches.
//!
//! ## What is fair here, and what is not
//!
//! - **Layout.** Both read *byte-identical* operands: a plain packed bf16
//!   `[m, k]` and `[n, k]`, K contiguous. This is worth stating because the
//!   opposite was expected — our kernel does not use
//!   `kittens::global::encode_bf16_panels` and needs no rearranged input. What
//!   it needs is a TMA tensor map built over that plain buffer, which is a
//!   descriptor and not a copy, and it is built once outside the clock exactly
//!   as cuBLASLt's descriptors, layouts and heuristic are.
//! - **What ours cannot do, which is the real asymmetry.** Our kernel computes
//!   this one form: both operands K-major, `α = 1`, `β = 0`, no epilogue, and
//!   `m % 256 == n % 128 == k % 64 == 0`. cuBLASLt takes any of that. A
//!   like-for-like rate against a library that is also general is a comparison
//!   in our favour, and the ratio below should be read knowing it.
//! - **Workspace.** cuBLASLt gets [`WORKSPACE_BYTES`] (32 MiB, upstream's
//!   figure); ours takes none. It is allocated **outside the timed region**,
//!   before the warm-up, which is the fair choice — a real pipeline keeps a
//!   workspace around rather than allocating one per call — and it is a
//!   genuine allowance ours does not use.
//! - **Algorithm.** Whatever `cublasLtMatmulAlgoGetHeuristic` ranks first, with
//!   no search and no tuning pass. [`describe`] prints its configuration so the
//!   baseline is reproducible rather than merely quoted.

use std::error::Error;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer};

use crate::bench::{Shape, Timings, time};
use crate::gemm;

/// Scratch cuBLASLt may use, and the cap the heuristic is chosen under.
///
/// 32 MiB is upstream's figure, kept so this baseline is comparable with the
/// one in `cuda-oxide`'s `gemm_sol`. It bounds which algorithms the heuristic
/// is allowed to return: a larger workspace admits split-K schemes that a
/// smaller one does not, so this number is part of the configuration and not
/// an implementation detail.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

// The ABI. Every constant here is asserted against the real headers by
// `examples/cublaslt_abi.c`, which `modal_app.py::build` compiles.

type Handle = *mut c_void;
type MatmulDesc = *mut c_void;
type MatrixLayout = *mut c_void;
type Preference = *mut c_void;
type Status = c_int;

const SUCCESS: Status = 0;

const CUDA_R_32F: c_int = 0;
const CUDA_R_16F: c_int = 2;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: c_int = 3;
const DESC_TRANSB: c_int = 4;
const PREF_MAX_WORKSPACE_BYTES: c_int = 1;

/// `cublasLtMatmulAlgo_t` — opaque to callers, and copied by value into
/// `cublasLtMatmul`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Algo {
    /// Read only by cuBLASLt, through the pointer [`describe`] and
    /// `cublasLtMatmul` are handed — never by name from Rust, which is what
    /// `dead_code` is objecting to.
    #[allow(dead_code)]
    data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Heuristic {
    algo: Algo,
    workspace_bytes: usize,
    /// Whether *this entry* is usable. A returned count above zero is not the
    /// same claim, and the header is explicit that the per-result status has
    /// to be read separately.
    state: Status,
    waves: f32,
    /// Padding the header declares. It exists so the struct is the size
    /// `cublaslt_abi.c` asserts; nothing reads it.
    #[allow(dead_code)]
    reserved: [c_int; 4],
}

#[link(name = "cublasLt")]
unsafe extern "C" {
    fn cublasLtCreate(handle: *mut Handle) -> Status;
    fn cublasLtDestroy(handle: Handle) -> Status;
    fn cublasLtGetVersion() -> usize;
    fn cublasLtGetStatusString(status: Status) -> *const c_char;

    fn cublasLtMatmulDescCreate(
        desc: *mut MatmulDesc,
        compute_type: c_int,
        scale_type: c_int,
    ) -> Status;
    fn cublasLtMatmulDescDestroy(desc: MatmulDesc) -> Status;
    fn cublasLtMatmulDescSetAttribute(
        desc: MatmulDesc,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Status;

    fn cublasLtMatrixLayoutCreate(
        layout: *mut MatrixLayout,
        element_type: c_int,
        rows: u64,
        columns: u64,
        leading_dimension: i64,
    ) -> Status;
    fn cublasLtMatrixLayoutDestroy(layout: MatrixLayout) -> Status;

    fn cublasLtMatmulPreferenceCreate(preference: *mut Preference) -> Status;
    fn cublasLtMatmulPreferenceDestroy(preference: Preference) -> Status;
    fn cublasLtMatmulPreferenceSetAttribute(
        preference: Preference,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Status;

    fn cublasLtMatmulAlgoGetHeuristic(
        handle: Handle,
        desc: MatmulDesc,
        a: MatrixLayout,
        b: MatrixLayout,
        c: MatrixLayout,
        d: MatrixLayout,
        preference: Preference,
        requested: c_int,
        results: *mut Heuristic,
        returned: *mut c_int,
    ) -> Status;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const Algo,
        attribute: c_int,
        buffer: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> Status;

    fn cublasLtMatmul(
        handle: Handle,
        desc: MatmulDesc,
        alpha: *const c_void,
        a: *const c_void,
        a_layout: MatrixLayout,
        b: *const c_void,
        b_layout: MatrixLayout,
        beta: *const c_void,
        c: *const c_void,
        c_layout: MatrixLayout,
        d: *mut c_void,
        d_layout: MatrixLayout,
        algo: *const Algo,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> Status;
}

/// Turn a non-zero status into an error naming both the call and what cuBLASLt
/// calls the code, since the numbers alone are not memorable.
fn checked(status: Status, call: &str) -> Result<(), Box<dyn Error>> {
    if status == SUCCESS {
        return Ok(());
    }
    // SAFETY: cuBLASLt returns a static string for every status, including
    // codes it does not recognize.
    let text = unsafe { CStr::from_ptr(cublasLtGetStatusString(status)) };
    Err(format!(
        "{call}: cuBLASLt status {status} ({})",
        text.to_string_lossy()
    )
    .into())
}

/// Every cuBLASLt object one measurement creates, destroyed in one place.
///
/// The fields start null and are filled in order, so a failure part way
/// through setup drops exactly what had been created. The alternative is
/// upstream's straight-line C, which leaks every handle it already made on any
/// early return — survivable in a program that then exits, not in a sweep that
/// does this once per size.
struct Session {
    handle: Handle,
    desc: MatmulDesc,
    a: MatrixLayout,
    b: MatrixLayout,
    d: MatrixLayout,
    preference: Preference,
}

impl Session {
    fn new() -> Self {
        Session {
            handle: std::ptr::null_mut(),
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: each field is either null or an object this session made,
        // and nothing outlives it — the timed closure borrows the session.
        unsafe {
            if !self.preference.is_null() {
                let _ = cublasLtMatmulPreferenceDestroy(self.preference);
            }
            for layout in [self.d, self.b, self.a] {
                if !layout.is_null() {
                    let _ = cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                let _ = cublasLtMatmulDescDestroy(self.desc);
            }
            if !self.handle.is_null() {
                let _ = cublasLtDestroy(self.handle);
            }
        }
    }
}

/// A `CUBLASLT_MATMUL_DESC_*` attribute that is one `int`, which both of the
/// two this benchmark sets are.
fn set_transpose(desc: MatmulDesc, attribute: c_int, value: i32) -> Result<(), Box<dyn Error>> {
    // SAFETY: `desc` is live and the buffer is exactly the width the attribute
    // is documented to take.
    checked(
        unsafe {
            cublasLtMatmulDescSetAttribute(
                desc,
                attribute,
                (&raw const value).cast(),
                size_of::<i32>(),
            )
        },
        "cublasLtMatmulDescSetAttribute",
    )
}

/// One column-major matrix layout: `rows × columns` with leading dimension
/// `ld`, in `element_type`.
fn layout(
    slot: &mut MatrixLayout,
    element_type: c_int,
    rows: usize,
    columns: usize,
    leading_dimension: usize,
) -> Result<(), Box<dyn Error>> {
    // SAFETY: `slot` is a live null pointer this call fills in.
    checked(
        unsafe {
            cublasLtMatrixLayoutCreate(
                slot,
                element_type,
                rows as u64,
                columns as u64,
                leading_dimension as i64,
            )
        },
        "cublasLtMatrixLayoutCreate",
    )
}

/// The `uint32_t` configuration attributes that identify a chosen algorithm.
///
/// Deliberately not the whole set: `INNER_SHAPE_ID` and `CLUSTER_SHAPE_ID` are
/// `uint16_t` in the header, and reading them through the same `u32` buffer
/// would be the exact class of transcription bug `cublaslt_abi.c` exists to
/// prevent. What is here is enough to name the algorithm again.
const CONFIG_ATTRIBUTES: [(&str, c_int); 6] = [
    ("id", 0),
    ("tile", 1),
    ("splitk", 2),
    ("reduction", 3),
    ("swizzle", 4),
    ("stages", 6),
];

/// The chosen algorithm, in one line, so the baseline can be reproduced rather
/// than merely believed. A baseline nobody else can obtain is worth much less
/// than one they can.
fn describe(heuristic: &Heuristic) -> String {
    let fields: Vec<String> = CONFIG_ATTRIBUTES
        .iter()
        .map(|&(name, attribute)| {
            let mut value = 0u32;
            let mut written = 0usize;
            // SAFETY: the buffer is `u32`-wide and the attributes above are the
            // `uint32_t` ones; a width cuBLASLt disagrees with is an error
            // status, not a write past the end.
            let status = unsafe {
                cublasLtMatmulAlgoConfigGetAttribute(
                    &heuristic.algo,
                    attribute,
                    (&raw mut value).cast(),
                    size_of::<u32>(),
                    &mut written,
                )
            };
            match status {
                SUCCESS => format!("{name}={value}"),
                _ => format!("{name}=?"),
            }
        })
        .collect();
    format!(
        "{} waves={:.2} workspace={} B",
        fields.join(" "),
        heuristic.waves,
        heuristic.workspace_bytes
    )
}

/// `cublasLt <version>` — printed once above the comparison table.
pub fn about() -> String {
    // SAFETY: a pure version query, safe to call before any handle exists.
    format!("cuBLASLt version {}", unsafe { cublasLtGetVersion() })
}

/// Stage the same operands, check the same `C` against the same CPU reference,
/// and only then time the same way [`crate::gemm::bench`] does.
///
/// Returns the timings and the chosen algorithm's identity.
///
/// The order is the point and it is the same order [`crate::gemm`] enforces:
/// the launch is verified before [`time`] ever sees it, so there is no path
/// here that prints a throughput for a GEMM that computed something else. For
/// this baseline that rule is doing more work than usual — the plausible bug
/// is a transposed operand, which is fast and wrong rather than slow and
/// obvious.
pub fn bench(
    context: &Arc<CudaContext>,
    shape: Shape,
) -> Result<(Timings, String), Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    let stream = context.default_stream();

    // Byte-identical to what `gemm::run` stages, from the same generators. Not
    // a re-implementation of the operands: the same function.
    let a = DeviceBuffer::from_host(&stream, &gemm::stage(m, k, gemm::a_value))?;
    let b = DeviceBuffer::from_host(&stream, &gemm::stage(n, k, gemm::b_value))?;
    let c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    // Outside the timed region, as every allocation on both sides is.
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, WORKSPACE_BYTES)?;

    let mut session = Session::new();
    // SAFETY (this block): each call fills or consumes a field of `session`,
    // which owns every handle and destroys it exactly once.
    checked(
        unsafe { cublasLtCreate(&mut session.handle) },
        "cublasLtCreate",
    )?;
    checked(
        unsafe { cublasLtMatmulDescCreate(&mut session.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) },
        "cublasLtMatmulDescCreate",
    )?;

    // `Ĉ = (B̂)ᵀ · Â`, per the module header. cuBLASLt's first operand is our
    // `b`, transposed; its second is our `a`, not.
    set_transpose(session.desc, DESC_TRANSA, CUBLAS_OP_T)?;
    set_transpose(session.desc, DESC_TRANSB, CUBLAS_OP_N)?;
    layout(&mut session.a, CUDA_R_16BF, k, n, k)?;
    layout(&mut session.b, CUDA_R_16BF, k, m, k)?;
    // bf16 out since #108, and this line is half of that change: ours writes
    // 134 MB of `C` at 8192³ where it wrote 268, and a baseline left at
    // `CUDA_R_32F` would be writing twice as much as the kernel it is the
    // denominator for. The compute type is untouched — both sides still
    // accumulate in fp32.
    layout(&mut session.d, CUDA_R_16BF, n, m, n)?;

    checked(
        unsafe { cublasLtMatmulPreferenceCreate(&mut session.preference) },
        "cublasLtMatmulPreferenceCreate",
    )?;
    // A local, not the constant: cuBLASLt reads this through a pointer, and
    // the address of a `const` is the address of a temporary.
    let workspace_bytes = WORKSPACE_BYTES;
    checked(
        unsafe {
            cublasLtMatmulPreferenceSetAttribute(
                session.preference,
                PREF_MAX_WORKSPACE_BYTES,
                (&raw const workspace_bytes).cast(),
                size_of::<usize>(),
            )
        },
        "cublasLtMatmulPreferenceSetAttribute",
    )?;

    let mut heuristic = Heuristic {
        algo: Algo { data: [0; 8] },
        workspace_bytes: 0,
        state: SUCCESS,
        waves: 0.0,
        reserved: [0; 4],
    };
    let mut returned: c_int = 0;
    checked(
        unsafe {
            cublasLtMatmulAlgoGetHeuristic(
                session.handle,
                session.desc,
                session.a,
                session.b,
                session.d,
                session.d,
                session.preference,
                1,
                &mut heuristic,
                &mut returned,
            )
        },
        "cublasLtMatmulAlgoGetHeuristic",
    )?;
    if returned == 0 {
        return Err(format!("cuBLASLt has no algorithm for {m}x{n}x{k}").into());
    }
    // A returned count above zero does not by itself mean the entry is usable.
    checked(heuristic.state, "the algorithm the heuristic returned")?;

    // Host-pointer scale mode is the default, so these stay on the stack and
    // must outlive every launch below — which they do, being declared here.
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let mut launch = || -> Result<(), Box<dyn Error>> {
        // SAFETY: every handle is live for the whole closure, the device
        // pointers name buffers sized by the layouts above, and `c` is both
        // `C` and `D` — legal, and unread since `beta` is zero.
        checked(
            unsafe {
                cublasLtMatmul(
                    session.handle,
                    session.desc,
                    (&raw const alpha).cast(),
                    b.cu_deviceptr() as *const c_void,
                    session.a,
                    a.cu_deviceptr() as *const c_void,
                    session.b,
                    (&raw const beta).cast(),
                    c.cu_deviceptr() as *const c_void,
                    session.d,
                    c.cu_deviceptr() as *mut c_void,
                    session.d,
                    &heuristic.algo,
                    workspace.cu_deviceptr() as *mut c_void,
                    WORKSPACE_BYTES,
                    stream.cu_stream().cast(),
                )
            },
            "cublasLtMatmul",
        )
    };

    launch()?;
    stream.synchronize()?;
    // `==` on the bf16 words, against a reference rounded the way
    // `cvt.rn.bf16x2.f32` rounds — so this asserts the library's epilogue
    // rounds once, from an fp32 accumulator, exactly as ours does. The worst
    // relative error it returns is the same property of bf16 both sides carry,
    // and `gemm::check` is where it is reported.
    gemm::check_c(&c.to_host_vec(&stream)?, m, n, k)?;

    Ok((time(&stream, &mut launch)?, describe(&heuristic)))
}

/// Live FP16-input/FP32-accumulate/FP16-output reference for
/// [`crate::gemm_sol`]. The port writes BF16 because that is the upstream
/// kernel's epilogue; cuBLASLt's closest supported path differs only in the
/// final 16-bit conversion and writes the same number of bytes.
pub fn bench_f16(
    context: &Arc<CudaContext>,
    shape: Shape,
) -> Result<(Timings, String), Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    let stream = context.default_stream();
    let a = DeviceBuffer::<u16>::zeroed(&stream, m * k)?;
    let b = DeviceBuffer::<u16>::zeroed(&stream, n * k)?;
    let c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, WORKSPACE_BYTES)?;

    let mut session = Session::new();
    checked(
        unsafe { cublasLtCreate(&mut session.handle) },
        "cublasLtCreate",
    )?;
    checked(
        unsafe { cublasLtMatmulDescCreate(&mut session.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) },
        "cublasLtMatmulDescCreate",
    )?;
    set_transpose(session.desc, DESC_TRANSA, CUBLAS_OP_T)?;
    set_transpose(session.desc, DESC_TRANSB, CUBLAS_OP_N)?;
    layout(&mut session.a, CUDA_R_16F, k, n, k)?;
    layout(&mut session.b, CUDA_R_16F, k, m, k)?;
    layout(&mut session.d, CUDA_R_16F, n, m, n)?;

    checked(
        unsafe { cublasLtMatmulPreferenceCreate(&mut session.preference) },
        "cublasLtMatmulPreferenceCreate",
    )?;
    let workspace_bytes = WORKSPACE_BYTES;
    checked(
        unsafe {
            cublasLtMatmulPreferenceSetAttribute(
                session.preference,
                PREF_MAX_WORKSPACE_BYTES,
                (&raw const workspace_bytes).cast(),
                size_of::<usize>(),
            )
        },
        "cublasLtMatmulPreferenceSetAttribute",
    )?;

    let mut heuristic = Heuristic {
        algo: Algo { data: [0; 8] },
        workspace_bytes: 0,
        state: SUCCESS,
        waves: 0.0,
        reserved: [0; 4],
    };
    let mut returned = 0;
    checked(
        unsafe {
            cublasLtMatmulAlgoGetHeuristic(
                session.handle,
                session.desc,
                session.a,
                session.b,
                session.d,
                session.d,
                session.preference,
                1,
                &mut heuristic,
                &mut returned,
            )
        },
        "cublasLtMatmulAlgoGetHeuristic",
    )?;
    if returned == 0 {
        return Err(format!("cuBLASLt has no FP16 algorithm for {m}x{n}x{k}").into());
    }
    checked(heuristic.state, "the FP16 algorithm the heuristic returned")?;

    let alpha = 1.0f32;
    let beta = 0.0f32;
    let mut launch = || -> Result<(), Box<dyn Error>> {
        checked(
            unsafe {
                cublasLtMatmul(
                    session.handle,
                    session.desc,
                    (&raw const alpha).cast(),
                    b.cu_deviceptr() as *const c_void,
                    session.a,
                    a.cu_deviceptr() as *const c_void,
                    session.b,
                    (&raw const beta).cast(),
                    c.cu_deviceptr() as *const c_void,
                    session.d,
                    c.cu_deviceptr() as *mut c_void,
                    session.d,
                    &heuristic.algo,
                    workspace.cu_deviceptr() as *mut c_void,
                    WORKSPACE_BYTES,
                    stream.cu_stream().cast(),
                )
            },
            "cublasLtMatmul(FP16)",
        )
    };

    Ok((time(&stream, &mut launch)?, describe(&heuristic)))
}
