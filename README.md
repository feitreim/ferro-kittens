# ferro-kittens

<p align="center">
  <img src="assets/ferris_w_cat_ears.png" alt="Ferris the crab wearing cat ears" width="400">
  <br>
  <em>Ferris trying his best to fit in with the other kittens</em>
</p>

A ThunderKittens-style tile library for [cuda-oxide](https://github.com/NVlabs/cuda-oxide),
targeting Blackwell (`sm_100a`) exclusively. Kernels are written against typed
shared/register/TMEM tiles with warp- and warpgroup-scoped ops instead of raw
intrinsics and hand-threaded index math.

The MMA layer is tcgen05 — no wmma/wgmma backends, no arch dispatch.

Right now GEMM performance is at about ~90% of cublasLT and/or the upstream cuda-oxide example.
There are examples in `/examples/`

## Modules

| Module | What it holds |
| --- | --- |
| `shared` | Shared-memory tiles with the SWIZZLE_128B layout in the type |
| `plan` | One `const` walk over the shared block, handing back the total a launch contract needs |
| `reg` | Register vectors/tiles over a parameterized fragment ownership map |
| `tmem` | TMEM accumulator views (`base + (row << 16) + column`) |
| `mma` | Chained tcgen05 MMA walks over shared-tile operands |
| `ldst` | Warp-scope register↔shared movers (`stmatrix` on swizzled chunks) |
| `sync` | Semaphores over mbarrier intrinsics, and the block-scope fold warps cannot shuffle |
| `pipeline` | Persistent-grid harness (ThunderKittens' `prototype::lcf` shape) |
| `epilogue` | The staging ring a drained accumulator crosses on its way to the TMA engine |
| `global` | Global layouts and their TMA tensor maps (host-only) |
| `launch` | The >48 KiB shared-memory opt-in a large tile plan needs (host-only) |

## Usage

```toml
[dependencies]
kittens = { package = "ferro-kittens", git = "https://github.com/feitreim/ferro-kittens" }
```

The `host` feature enables `global`'s `cuTensorMapEncodeTiled` builders and pulls
in `cuda-core`; it is off by default so the device-only surface can be
`cargo check`ed without a CUDA toolkit installed.

The toolchain is pinned in `rust-toolchain.toml` — cuda-oxide's proc macros need
nightly features and the codegen backend is built against exactly that nightly.

Everything the device-only surface can be held to runs on every pull request
with no toolkit and no credential; the host feature, a real `sm_100a` codegen,
and the B200 harness are progressively more expensive tiers with their own
triggers. `CI.md` has the policy, the costs, and how to ask for a GPU run.

## Examples, and experiments

`examples/` holds real kernels — two GEMMs, flash-attention forward, softmax,
layernorm — written the way we want them to read, each with a header saying
whether it **runs** against a CPU reference or only **compiles**. All five
compile, the crate has no cargo features left, and `cargo oxide build
kittens-examples --arch sm_100a` therefore codegens every one of them.

The two GEMMs are not variants of each other. `gemm.rs` is the library's own, a
two-CTA cluster kernel in about six hundred lines; `gemm_sol.rs` is cuda-oxide's
canonical `gemm_sol_final` ported through this API, which is how the abstraction
gets held to a kernel it did not get to design.

`experiments/` is where `gemm.rs` was chosen. Every tile rung, every scheduler,
the ablation cube, the four epilogue families, the doubling probes that compute
a deliberately wrong `C` to isolate one term, the warp-specialized variant, the
benchmark harness and the cuBLASLt denominator — thirty GEMM entry points, all
still launchable and all still on the same correctness gate.
`experiments/README.md` is the measurement record: what each of them measured,
and the missing library surface the four kernels demanded while they were still
asking for it, mapped to issues, with the part no issue covered called out
separately.

## Design notes

Source carries what a *caller* needs — what a thing does, the contract it owes,
and an example of calling it. The measurement a decision rests on lives in
`docs/`: `docs/library/` one file per module, `docs/kernels/` one per kernel.
A one-line conclusion stays behind wherever the number changes what a caller
would write, so `softmax.rs` says `CHUNK = 16` and that 32 is 4.4× slower,
while `docs/kernels/softmax.md` has the ladder that came off, the `exp2`
ablation beside it, and why a control taken under one bottleneck expires when
the bottleneck moves.

The library's own examples are doctests rather than prose, which is the point:
`cargo test --doc` compiles all 74 of them against the real signatures, so an
example that drifts fails a gate instead of misleading a reader.
