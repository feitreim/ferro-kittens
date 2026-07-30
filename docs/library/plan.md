# `plan.rs` — design notes and measurements

`SharedPlan` is a cursor over one launch's dynamic shared memory. It hands out
one typed handle per reservation, aligns each to what that handle's type
requires, and accumulates the total as it goes. The source states the contract;
this file states why the type exists in that shape, what it replaced, and what
it was measured to cost.

## What it replaced

Every kernel in this repo used to compute its plan **twice** — once as a
host-visible `SHARED_BYTES` the launch declares, and once as a pointer walk
inside the kernel — with nothing but a hand-written `const { assert!(..) }`
relating the two. The walk was raw byte arithmetic over objects whose sizes the
crate already knows, and it leaked four things into every kernel that wrote one:

1. `DynamicSharedArray::<u8, 128>::get_raw()`, spelled per kernel.
2. A `cuda_device::barrier::Barrier` import that existed only as a cast target.
3. The placement and accounting of `tmem::alloc_cluster`'s staging word — the
   allocator's own resource, which the caller had to find, align and count.
4. An obligation to prove `ClcQueue::ALIGNMENT` by hand, carried as an identical
   `const { assert!(offset % ClcQueue::ALIGNMENT == 0) }` block in both GEMMs.

`GAPS.md` §7.4 ranked this the largest single thing a kernel still open-coded.
Under the cursor, the number the launch declares is `SharedPlan::bytes` of the
same walk that produced the pointers, rather than a second expression asserted
equal to it.

## Alignment, and why the caller no longer argues it

| reservation | alignment | because |
| --- | --- | --- |
| `tile`, `tile_ring`, `vec` | `TILE_ALIGN` = 128 | the TMA's shared destination alignment, and `Swizzle128B::ATOM_BYTES` |
| `semaphore`, `semaphores`, `barriers` | 8 | `align_of::<Barrier>()` — an mbarrier is one 64-bit state word |
| `tmem_slot` | 4 | `tcgen05.alloc` writes a `u32`, and `tmem::alloc_block`'s contract says `SharedArray<u32, 1, 4>` |
| `clc_queue` | `ClcQueue::ALIGNMENT` = 16 | the response is a `.b128` store |

### Ordering used to be a correctness argument; now it is only a cost

`layernorm.rs` carried the general form of this out loud:

> The partials first, because `Tile::BYTES` is the only offset in this plan a
> vector's 128-byte alignment is promised at; the barrier behind it needs eight.

That is a correctness argument about **ordering**, which the kernel had to make
because nothing else would. Under a cursor it is not one: a vector reserved
behind a barrier gets padded to 128 rather than landing misaligned. The order
now decides how many bytes the plan spends, not whether the vector is legal.
The `each_reservation_aligns_to_its_own_rule` and groupnorm-reversed cases in
the test module are that property written down — the same two reservations in
either order are both legal and differ by 120 bytes of padding.

This is also why `vec` takes the strong 128-byte rule rather than the weaker
`E::BYTES` one a `get`/`set`-only vector could live with. The weak rule would
make a plan's correctness depend on its order again, to save at most 127 bytes.

## The rejected alternative: a const-generic walk *is* a value-parameterized total

The wanted design was one walk and no `reserve`: every reservation typed, the
total falling out of it, and no byte arithmetic anywhere. That is not reachable,
and the obstruction is a language limit rather than a design one.

A `const fn` over *values* — `experiments`' rung table asks for
`shared_plan(block_n, block_k, stages)` at runtime, for rungs no kernel
implements — cannot instantiate `SharedTileRing<Bf16, R, C, S, N>`, because a
const parameter cannot be a function argument. So a host-side plan that has to
answer for shapes no kernel instantiates has to spell its reservations in bytes.
`SharedPlan::reserve` is what that costs: the value form spells the same
reservations in bytes, and the two are joined by one assert.

The conclusion that stays in the source is the narrow one: prefer a typed
reservation, and `reserve` is the escape hatch for the case the type system
cannot reach. Kernels whose plan parameters are module constants — every kernel
in `examples/` — need no value form and no assert at all, and there the total
genuinely *is* the walk.

## Why `reserve` branches, and why the flag is not `base.is_null()`

The device and the host want *different* pointer arithmetic and there is no
operation that serves both:

- `wrapping_add` const-evaluates off a sizing cursor's provenance-free base and
  **does not compile for the device** — `rustc_codegen_cuda` rejects
  `std::intrinsics::arith_offset` as unsupported. This is the error the method
  was written twice to fix.
- `add` compiles and is what every kernel's hand-written walk used, but const
  eval will not let a pointer leave an allocation, and a sizing cursor has none.

So the sizing cursor does not offset its base at all. The pointer it hands back
is inert either way, and the offset it is *not* carrying lives in
`SharedPlan::bytes`, which is the only thing that path reads.

`attached` is a `bool` field rather than a const parameter because a const
parameter would travel into every kernel's own plan struct. It is a flag rather
than `self.base.is_null()` because only one of those folds: a literal
constant-folds, and a null check folds only if LLVM can prove `get_raw()`
returns a non-null pointer, which it cannot.

## Cost, as measured

`SharedPlan` is a base, an offset and a mode flag, `Copy`, and every method is
`#[inline(always)]` arithmetic on a compile-time constant offset. Threading it
by value rather than as a `&mut` builder is what is supposed to keep it free.
`scripts/modal-run regcount` diffed row by row against `main`, over 231 register
rows and 40 opcode-census rows:

- **All 184 `device-tests` kernels are unchanged**, and so is every counter of
  the opcode census for every `gemm_*` kernel in both kernel crates — no
  instruction moved anywhere.
- `gemm_cg2_staged_x8x4`, the kernel `examples/` ships and `experiments/` keeps
  as its control arm, is unchanged in both crates (80 and 96), as are `gemm_cg2`
  (166), `gemm_cg2_staged` (42), `flash_forward` (168), `softmax_rows` (32) and
  `groupnorm_tile` (168).
- Six moved, in both directions:

  | kernel | before | after |
  | --- | --- | --- |
  | `layernorm_rows` (both crates) | 47 | 50 |
  | `gemm_cg2_staged_x8x4_2x` | 106 | 110 |
  | `gemm_cg2_staged_x8x4_hot` | 93 | 104 |
  | `gemm_cg2_dry` | 20 | 19 |
  | `gemm_ws_staged` | 44 | 43 |
  | `gemm_ws_staged_x8` | 94 | 90 |

  No spill and no stack frame moved with any of them.

Two of those cross a *register-term* occupancy step by `modal_app.py`'s
`_ctas_by_registers`, and neither crosses a real one, because another term binds
in both: `layernorm_rows` goes 10 → 9 CTAs by registers and is 6 by shared
memory either way, and `gemm_cg2_staged_x8x4_hot` goes 5 → 4 and is 2 by tensor
memory either way.

### The mode branch folds, measured rather than argued

Two side-effect-free pointer values under an `if` is the shape LLVM if-converts
to a `select`, so a branch that survived would appear as a `selp` in the PTX;
`modal_app.py`'s opcode census counts them. `gemm_cg2_idle` runs the whole
eight-reservation walk and then does nothing, so an unfolded flag would give it
eight `selp`. It has **zero**, at an unchanged 17 registers. The cursor adds no
instruction, and the six moved counts above are not it.

What that does **not** establish is which of the six they *are*. The movement is
bidirectional, the instruction counts are identical, and the two arms of
`attach_staged` that did not move sit beside the two that did — which reads as
ptxas allocating rather than as a live value this type added. The claim that was
checked is: no kernel's residency moved, no instruction did, and the branch this
type introduces is gone by codegen.

## The plan totals the tests pin

The test module walks the five plans the repo's kernels declare, each against
the literal its launch carries. These are the numbers that set residency, so a
change to `reserve`'s alignment rules moves them here before it moves them on a
B200. The shapes are copied from the kernels rather than imported — the library
cannot see a kernel crate — so a kernel that changes shape makes the test stale
rather than red.

| plan | bytes |
| --- | --- |
| `examples/src/gemm.rs` (BLOCK_M 128, HALF_N 128, BLOCK_K 64, STAGES 3) | 98,364 |
| the same plus the staged epilogue's `[32, 64]` ×4 ring | 114,816 |
| `experiments/src/gemm.rs` (the above plus the work queue) | 98,392 |
| `experiments/src/gemm_ws.rs` (STAGES 4, work queue) | 131,176 |
| `gemm_ws` plus the staged epilogue ring | 147,584 |
| `examples/src/softmax.rs` | 32,776 |
| `examples/src/layernorm.rs` (`layernorm_rows`) | 33,288 |
| `groupnorm_tile` | 32,792 |
| `groupnorm_tile` with the vector reserved last | 32,912 |
| `examples/src/flash_forward.rs` | 147,532 |

The work queue's 16-byte alignment absorbs the staging word's four, which is why
the experiments GEMM's total is unchanged from the hand-written walk it
replaced. `clc_queue` reserves its response and its barrier separately rather
than as one 24-byte run because that is what lets the whole walk
const-evaluate — and it costs nothing: 16 at 16 then 8 at 8 is `ClcQueue::BYTES`
on the nose.
