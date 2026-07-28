/*
 * The cuBLASLt ABI that `examples/src/cublaslt.rs` hand-transcribes, checked
 * against the real headers.
 *
 * That file declares about a dozen `extern "C"` functions and a dozen integer
 * constants by hand rather than generating them, which is the right trade for
 * a surface this small -- but a hand-copied enumerator is exactly the kind of
 * thing that is wrong silently. A wrong `CUBLAS_COMPUTE_*` asks for a
 * different arithmetic; a wrong struct offset reads the heuristic's algorithm
 * out of its padding. Neither is a compile error in Rust, and the second need
 * not even be a wrong *answer* -- it can be a baseline that is quietly slow.
 *
 * So the numbers are asserted here instead, where the headers can see them.
 * `modal_app.py::build` compiles this with `-fsyntax-only`, which costs
 * milliseconds, needs no GPU and no linker, and fails the CPU gate rather than
 * the B200 one.
 *
 *   gcc -fsyntax-only -I$CUDA_HOME/include examples/cublaslt_abi.c
 *
 * The correctness path catches the same class of error a second time -- the
 * baseline's output is compared element by element against the same CPU
 * reference our kernel is -- but it catches it minutes later, on a device, and
 * reports it as a wrong number rather than as a wrong constant.
 */

#include <cublasLt.h>
#include <stddef.h>

/* `rust` is the literal in cublaslt.rs; `c` is what the header calls it. */
#define SAME(rust, c) \
    _Static_assert((rust) == (c), #c " moved; examples/src/cublaslt.rs still has " #rust)

/* Element types: bf16 operands, fp32 scales and fp32 output (library_types.h). */
SAME(0, CUDA_R_32F);
SAME(14, CUDA_R_16BF);

/* fp32 accumulation over bf16 inputs -- not one of the _FAST_ variants, which
 * would quietly narrow the arithmetic and break the exact `==` check. */
SAME(68, CUBLAS_COMPUTE_32F);

SAME(0, CUBLAS_OP_N);
SAME(1, CUBLAS_OP_T);
SAME(0, CUBLAS_STATUS_SUCCESS);

/* The two attributes that decide *which GEMM* gets computed. Getting these
 * wrong is the failure mode #92 called out by name. */
SAME(3, CUBLASLT_MATMUL_DESC_TRANSA);
SAME(4, CUBLASLT_MATMUL_DESC_TRANSB);

SAME(1, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);

/* The algorithm identity the benchmark prints, so the baseline is reproducible. */
SAME(0, CUBLASLT_ALGO_CONFIG_ID);
SAME(1, CUBLASLT_ALGO_CONFIG_TILE_ID);
SAME(2, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
SAME(3, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
SAME(4, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
SAME(6, CUBLASLT_ALGO_CONFIG_STAGES_ID);

/* `struct Algo` and `struct Heuristic` in cublaslt.rs. `offsetof` takes a
 * comma, so these do not go through SAME. */
SAME(64, sizeof(cublasLtMatmulAlgo_t));
_Static_assert(sizeof(cublasLtMatmulHeuristicResult_t) == 96,
               "cublasLtMatmulHeuristicResult_t resized; struct Heuristic must follow");
_Static_assert(offsetof(cublasLtMatmulHeuristicResult_t, workspaceSize) == 64,
               "workspaceSize moved; struct Heuristic must follow");
_Static_assert(offsetof(cublasLtMatmulHeuristicResult_t, state) == 72,
               "state moved; struct Heuristic must follow");
_Static_assert(offsetof(cublasLtMatmulHeuristicResult_t, wavesCount) == 76,
               "wavesCount moved; struct Heuristic must follow");
