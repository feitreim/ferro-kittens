# `launch` — design notes

One function, and three reasons it is not redundant.

## The 48 KiB cliff

A CUDA block gets 48 KiB of dynamic shared memory without asking. Every plan
past that is *opt-in*: `cuFuncSetAttribute` with
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, per function, before the
launch. A launch that skips it is not slow, it is inadmissible.

This library's whole point is tiles big enough to keep tcgen05 fed, so its
kernels cross the cliff as a matter of course:

| kernel | dynamic shared plan |
|---|---|
| `gemm` | 72 KiB |
| `flash_forward` | 144 KiB |

## Why this exists when cuda-oxide already has one

`PreparedLaunch::__prepare` issues the same opt-in, and `CudaFunction`'s setter
is `pub(crate)` so that it is the only thing that can. That is the right default
— it validates the plan against the device before mutating the function — but it
is reachable only from a `#[launch_contract]` kernel's generated `prepare_*`.

A contract is a claim about the whole launch: its domain, its block shape, and
the index space each output slice is partitioned by. A kernel that partitions its
output some other way — warp bands through a raw `GlobalRows` cursor, which is
what `global::store_rows` is for — cannot state that claim honestly, and should
not have to invent one to get 144 KiB of shared memory.

So `admit_shared_plan` is the same opt-in with the same check in front of it, on
the path that does not go through a contract. Nothing here replaces `prepare_*`:
a kernel that *can* state a contract should, and gets this for free.

## Reading a zero

`cuOccupancyMaxActiveBlocksPerMultiprocessor` answers **0** both for a plan the
device cannot fit and for a plan nobody opted into. The two want opposite fixes —
shrink the tiles, or call `admit_shared_plan` — and an hour was once spent on the
wrong one.

Hence the error split. `admit_shared_plan` returns `SharedPlanTooLarge` with the
device's own `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` ceiling
beside the ask, so a plan that genuinely does not fit says so in bytes rather
than in a zero. `SharedPlanTooLarge` is deliberately not a `DriverError`: the
driver is working fine and the answer is that the tiles are too big.
