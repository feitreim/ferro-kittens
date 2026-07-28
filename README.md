# ferro-kittens

A ThunderKittens-style tile library for [cuda-oxide](https://github.com/NVlabs/cuda-oxide),
targeting Blackwell (`sm_100a`) exclusively. Kernels are written against typed
shared/register/TMEM tiles with warp- and warpgroup-scoped ops instead of raw
intrinsics and hand-threaded index math.

The MMA layer is tcgen05 — no wmma/wgmma backends, no arch dispatch.

## Zero-cost by construction

Everything here is a plain `#[inline(always)]` function or a `Copy` struct of
pointers and const generics. The crate ships no kernels and no `#[cuda_module]`;
device code monomorphizes into the *calling* crate's artifact the same way
`cuda-device` does, so a kernel crate pays nothing for the abstraction unless
ptxas says otherwise.

Asking ptxas is `modal run modal_app.py::regcount`: it builds the device-test
harness, runs `ptxas -v -arch=sm_100a` over the emitted PTX, and prints a
sorted registers/spills/shared table. `ptxas` is a host compiler, so this needs
no GPU. Run it either side of a change and diff. Only kernels that are actually
monomorphized appear, so a library function with no caller gets a codegen probe
in `device-tests` to put it on the table.

It also prints a **register ladder**: the same step in five spellings across
fourteen tile shapes, from 16 fp32 a thread to 256 — one past the file — with
the spill cliff located per spelling. A rung is one line of `ladder!` in
`device-tests`, which is the point: the number that decides a design here is a
function of tile *shape*, and it is not a smooth one. `--determinism` measures
the same tree twice and asserts an identical table, which is what makes a diff
of two runs evidence.

**And read that table as allocation, not as a ranking.** `modal run
modal_app.py::ladder_bench` puts a clock on four of those rungs — the first
timed register claim in this repo, on a B200 — and none of registers, stack
frame or occupancy orders the resulting times. On that probe the spelling that
leaves the whole band in local memory at 32 registers is the *fastest* at all
four shapes, and the one shape where the static counters favour the fused
expression is a shape where it loses. On `softmax`, #47 timed the same
phenomenon and it cost **2.6×**. Both are right: a streamed band costs real
time, and whether it is worth paying turns on whether the registers it frees are
the resource actually binding that kernel — shared memory caps `softmax` either
way, where the probe has none. So a register count is half an argument and
occupancy is the other half. `examples/README.md` § "in-place versus by-value"
carries both tables and reads them against each other.

## Modules

| Module | What it holds |
| --- | --- |
| `shared` | Shared-memory tiles with the SWIZZLE_128B layout in the type |
| `reg` | Register vectors/tiles over a parameterized fragment ownership map |
| `tmem` | TMEM accumulator views (`base + (row << 16) + column`) |
| `mma` | Chained tcgen05 MMA walks over shared-tile operands |
| `ldst` | Warp-scope register↔shared movers (`stmatrix` on swizzled chunks) |
| `sync` | Semaphores over mbarrier intrinsics, and the block-scope fold warps cannot shuffle |
| `pipeline` | Persistent-grid harness (ThunderKittens' `prototype::lcf` shape) |
| `global` | Global layouts and their TMA tensor maps (host-only) |

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

## Examples

`examples/` holds real kernels — a cluster GEMM, flash-attention forward,
softmax, layernorm — written the way we want them to read, each with a header
saying whether it compiles today or names API that does not exist yet.
`examples/README.md` collects the missing surface all four demand, mapped to
issues, with the part no issue covers called out separately.
