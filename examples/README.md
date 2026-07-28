# examples

Four kernels written the way we want them to read, and a header on each saying
whether it runs, whether it only compiles, or whether it is aspirational.

The value of this crate is the **diff** between what the kernels want to say and
what `kittens` can express. An aspirational example is not a placeholder — it is
a statement of a missing API in the only terms that matter, which is what a
kernel author has to type. This file collects that diff across all four.

Its own workspace, like `device-tests`, so `cargo` at the repo root never sees
it. (The root `Cargo.toml` also needs `autoexamples = false`: `examples/` is one
of cargo's own target directory names, and `exclude` does not stop target
auto-discovery, so without it `examples/src/main.rs` gets compiled as an example
target *of the library*.)

## Status

| Kernel | Status | Blocked on |
| --- | --- | --- |
| [`gemm`](src/gemm.rs) | **runs** — exact against a CPU reference | — (and no gap worked around in-file any more) |
| [`softmax`](src/softmax.rs) | **runs** — within 2⁻⁸ of a CPU reference | — |
| [`flash_forward`](src/flash_forward.rs) | **compiles** — no launcher yet | — |
| [`layernorm`](src/layernorm.rs) | **compiles** — both kernels, no launcher yet | — |

Two of the four **run** rather than merely compile, which is a strictly stronger
claim and the one worth holding the rest to: a launcher, a CPU reference, and an
exit code.

`softmax`'s blocked-on column is empty, and it is the first example to have
gone the whole way — five gaps, then one, then none. **#9** was the last of
them, and closing it took nothing but the store: `SharedTile::tma_store` walks
the same boxes `tma_load` does, and `tma_store_commit` / `tma_store_wait::<0>()`
are the completion side.

`modal run modal_app.py::bench` is that claim with a clock on it (#40): both
kernels that run, at several sizes each, every one checked against the same CPU
reference *before* it is timed — there is no path through `bench.rs` that prints
a throughput figure for a launch it did not verify. The metric follows what the
kernel is bound by, TFLOP/s for the GEMM and GB/s for the memory-bound softmax,
and the sizes are picked to cross a regime rather than to be large, so what the
table shows is the shape of a curve and not one headline number. The shape of
that curve is what produced **#47**, the first performance finding this project
got from measuring rather than from reading ThunderKittens — and it is written
up under the register ladder below, because that is where the cause turned out
to be.

Nothing on the remaining lists is arithmetic, and nothing on them is a mover.
#5 and #6 between them closed every elementwise op and every reduction the four
kernels asked for and **#38** closed the scalar-operand forms they left behind;
#21 and #25 closed both halves of the shared ↔ register path, #22 closed the
composition on top of them, in both directions and against both memories, and
**#13** closed the vector shape of shared memory. **#23 and #7 closed together**
— flash could not name its `[32, 64]` band, and the unsatisfied bound on that
band was what kept `make_causal` off the error list, so shipping either alone
would have left a confusing state. **#31** closed the in-place half of the map
mechanism, the last arithmetic entry on any of these lists. **#11** closed the
last mover, the one with no engine behind it: `global::load_rows`/`store_rows`
address a plain pitched buffer from the fragment layout's own coordinates,
which is what took the GEMM's epilogue from twelve lines of index math to one
call. Those two landed together and between them took `flash_forward` from
aspirational to compiling. **#3** was the last structural gap — layernorm's
block-scope statistic — and with it closed **no kernel here is blocked on
missing API any more**. All four compile against the real library, and what
separates the two that *run* from the two that do not is a launcher and a CPU
reference. What is still open is on the other list, the one that came from
writing kernels rather than from ThunderKittens' feature set: cluster-scope
semaphore arrival, a cluster geometry for multicast, a work-item map that knows
about clusters, and the `expect_tx` byte accounting. None of them blocks a
kernel in this directory.

`layernorm` was the first example to be **split** by a landed issue rather than
promoted whole: #13 took `layernorm_rows` out of the gate and left
`groupnorm_tile` behind it, because its statistic spans four warps and #13
supplied only the storage half of what that needs. #3 supplied the other half
and the split is over — both kernels are in the default build and both get a
real `sm_100a` compile. The `layernorm` cargo feature is **gone**, not left
empty: a feature that gates nothing only makes `gaps` print a blank section.

No arithmetic entry is open any more. `RegTile::add_assign` was the last one
and **#31** landed it, along with an `_assign` twin for every other map on all
three register families. What #31 measured on the way is more interesting than
the API it shipped, and § "in-place versus by-value" below is rewritten around
it.

```sh
cargo oxide build kittens-examples --arch sm_100a   # the default set: gemm, softmax, both layernorms
cargo oxide run kittens-examples                    # and run the ones with launchers, on a B200
cargo check --features flash                        # read flash's gap list
```

From the repo root: `modal run modal_app.py::build` for the first,
`modal run modal_app.py::examples` for the second, and
`modal run modal_app.py::gaps` for the third — it prints the gap list and never
fails, since an empty one is a finding rather than an error.

One kernel is still behind a feature, and it is `flash_forward`, which is there
for want of a launcher rather than for want of API. Everything else is in the
default build and genuinely compiles. Turning the feature on prints whatever
API is still missing as compiler errors, at the call sites that want it.
Verified: historically every error was `unresolved import`, `no method named`,
or an unsatisfied `FragmentLayout` bound — nothing in these files ever failed
for a reason other than the API not existing.

That list is read off `modal run modal_app.py::gaps`, which checks each feature
on its own so an error belongs to a known kernel. Both entries below were read
off a run of it, not off these files:

- **`flash_forward`** — **empty**, which is the interesting outcome for a gap
  list. It stays behind its feature only for want of a launcher and a CPU
  reference; nothing in it reaches past the library any more. The list was
  three errors until #23 landed, two until #11 and #31 landed together, and how
  the third went is worth keeping: `make_causal_at` (#7) was wanted all along
  and never appeared as its own error, because it is called on the `[32, 64]`
  band whose `FragmentLayout` bound already failed, and an unsatisfied bound on
  the receiver suppresses method resolution on it. Counting errors would have
  said #7 had landed.
- **`layernorm`** — **gone**, and that is the stronger outcome than empty. Its
  one error was `sync::block_reduce_sum` (#3); with the reduction in the
  library `groupnorm_tile` left the gate, the feature had nothing left to hold
  back, and it was deleted rather than kept as a section that prints nothing.
  A kernel in the default build is checked by `build`'s real `sm_100a` codegen,
  which is a stronger claim than a `cargo check` that finds no errors — a
  post-monomorphization `const { assert!(..) }` is invisible to the second and
  not to the first, and `SharedVec<F32, 4>`'s box assert is exactly one of
  those.

Nothing named `scale`, `shift` or `rsqrt` is in either list any more (#38),
nothing named `add_assign` (#31), and nothing named `FragmentLayout` or
`make_causal` (#23 + #7). That last pair is
worth keeping as a note on how to read this section: `make_causal_at` never
appeared as its own error, because it was called on the `[32, 64]` band whose
layout bound failed first and an unsatisfied bound on the receiver suppresses
method resolution on it. Counting errors would have said #7 was already
closed. Closing #23 is what made #7 surface.

`softmax` **runs** too, and what its check can claim is different from the
GEMM's in a way worth writing down. A softmax has an `exp2` and a divide in it,
so `==` is not available; instead each row's 128 inputs are a permutation of
the multiples of 1/8 below 16, which makes each row's 128 outputs distinct with
the closest pair 9% apart, against a 2⁻⁸ tolerance. Exactness is not the only
way to make a failure unambiguous — separating the right answer from every
wrong one by forty times the noise floor also works.

Which permutation a row gets is the part worth reading the code for (#56). The
seed's row term is taken mod 128, so on its own it repeats every 128 rows —
exactly one CTA band — and the check could not see a kernel that loaded one
band and stored another. No choice of row multiplier fixes that: the affine
maps mod 128 are a 2-group of exponent 128, so every row-affine seed repeats
inside a band. The band therefore picks the *column stride* instead, from a
cycle of 63, and 63 divides no power of two — so no band displacement this
grid's arithmetic can produce is invisible. [`softmax::permutation`](src/softmax.rs)
argues it and states what is left over.

`gemm` **runs**, and is the first numerical result this library has produced.
`gemm::check` launches it over `[512, 256] x [256, 256]ᵀ` on a B200 and compares
every element against a CPU reference. The operands are integers in `[-3, 3]`
and `[-10, 10]`, so every product and every partial sum is exact in bf16 and fp32
alike and the comparison is `==` — a mismatch is a wrong coordinate, a wrong
stride or a wrong operand half, and never a rounding artifact to argue about.
`main` runs every kernel that has a launcher and exits non-zero on a wrong
number; off a GPU it prints the status table and stops.

Getting there cost one real bug, recorded here because the aspirational-vs-
compiling distinction is exactly what missed it. The kernel compiled, and hung.
Both CTAs' TMA loads have to complete on the *leader's* stage barrier, and the
kernel mapped that barrier by hand — but a plain `cp.async.bulk.tensor`
completes on a barrier in the **issuing** CTA's own shared memory, so the
peer's 24 KiB never reached the count the leader had charged for 48 KiB. §8
below used to say this kernel "cannot use" `tma_load_2d_multicast_cg2`, on the
grounds that a 2-CTA UMMA replicates nothing. That is true and beside the
point: the multicast form's barrier operand is `.shared::cluster`, which is the
only way one CTA may name another's barrier. With the CTA's own bit as the
whole mask it delivers to exactly one CTA and completes on the leader — no
replication, right address space. `Semaphore::multicast_alias` was already in
the library, filed under the opposite problem.

---

## What the four kernels collectively demand

### Already filed

**#5 — generic map and the standard elementwise set. Landed.** The map
mechanism and every vector/vector and broadcast form the kernels asked for:
`exp2`, elementwise `mul`, `sub_row`, `mul_row`, `div_row`, `mul_col`,
`add_col`, `RegVec::rsqrt`.

> Note the asymmetry the examples exposed: `RegVec` already had `exp2`, `max`,
> `sub`, `add_assign`, `mul_assign`. `RegTile` had none of them. Every op the
> vector has, the tile wants — which is exactly the argument for the mechanism
> rather than another round of hand-written methods. **#31** closed the last of
> it: the in-place forms are generated from the same `op_methods!` tables, so
> all three families carry the same names.

**#38 — scalar operands. Landed.** What #5 listed and did not ship, and the
last arithmetic on any of these lists: `scale`/`shift`/`clamp_min`/`clamp_max`
on `RegTile`, `RegVec` and `ColVec`, plus a free `rsqrt(f32)` for a statistic
that never becomes a vector (`groupnorm_tile`'s variance).

The reason it was missed is worth keeping, because it decided the fix.
`UnaryOp::apply` is an associated function on a unit struct, so `scale(k)` has
**nowhere to put `k`** — the mechanism could not express it, which is a hole in
the trait shape rather than a missed line item. Treating the scalar as the
*second operand* of the existing `BinaryOp` set closes it:
`scalar_map::<Op>(k)` is one method per family, `scale` is `Mul` and `shift` is
`Add`, and `Div`/`Sub`/`Max`/`Min` against a constant are reachable with no new
op at all.

#### in-place versus by-value

**#31 — in-place maps. Landed.** Every by-value map now has an `_assign` twin
taking `&mut self`, on `RegTile`, `RegVec` and `ColVec`, generated from the
same tables as the by-value names. `RegTile::scale_rows` — hand-written since
#5 precisely to avoid the by-value `mul_row` — is a one-line wrapper over
`row_map_assign::<Mul>` now, and measures identically to the loop it replaced.

The issue was filed on the claim that `row_map::<Mul>` costs 168 → 255
registers/thread at 128 columns where `scale_rows` does not. That reproduces
exactly, and the in-place form closes it (`softmax_probe_*`, `regcount`):

| | 32 columns | 128 columns |
| --- | --- | --- |
| `scale_rows`, hand-written | 64 regs | 168 regs |
| `out = out.row_map::<Mul>(f)` | 64 regs | **255 regs** |
| `out.mul_row_assign(f)` | 64 regs | 168 regs |
| `out = out.add(p)` | 64 regs | 168 regs, 3072 B stack |
| `out.add_assign(p)` | 64 regs | 168 regs, 2560 B stack |

Two results, pulling opposite ways. The rescale is worth 87 registers/thread
in place — that is #31's premise, measured, and the reason `scale_rows` may
now be deleted. The *accumulate* `flash_forward` was blocked on is worth no
registers at all, because that call site rebinds a dead input; what it buys is
512 bytes of stack and a line that reads like what it does.

And the framing both #31 and #38 were argued under does not survive the
control. #38 read the variable as *peak liveness*, on the strength of an
uncommitted probe where `out_acc.add(block.scale(k))` spilled 84 bytes and the
rebinding `block = block.scale(k)` spilled none. That form is in the tree now
(`scalar_map_probe_*_chained`) and reproduces — 108 bytes here. But so is the
form one step further along the same axis, `out_acc.scale(k).add(block.scale(k))`,
where *nothing* is rebound and which peak liveness says should be worst of all:

| 128 columns | regs | spill | stack |
| --- | --- | --- | --- |
| `bin_map::<Mul>(splat(k))` | 255 | 124 B | 2648 |
| `scale(k)`, rebound | 255 | 60 B | 2624 |
| `add(block.scale(k))`, chained | 255 | 108 B | 2672 |
| in place, through `set` | 252 | — | 2560 |
| `scale_assign(k)` (#31) | 252 | — | 2560 |
| `out.scale(k).add(block.scale(k))` | **168** | — | 2560 |

It is the cheapest form in the table by 84 registers and the only one under
the cliff. Liveness does not order these; what does is how many whole bands
have to be **materialized between statements**. One expression fuses into a
single pass over the band and materializes none. The in-place forms
materialize none either, but make one pass per op. Every rebinding form
materializes one band per statement, and that is what spills.

So the rule for a kernel author is *say the whole step in one expression where
you can, and write it in place where you cannot* — which is exactly the
accumulator case, where the input is the output and no single expression
exists. That is what #31 shipped, and the reason it is worth having is
narrower and better founded than the reason it was filed.

> Read on before taking that rule away with you. #60 swept it over fourteen
> shapes and found the register table did not order them; #63 and #47 then put
> a clock on it and found the register table does not order *time* either, in
> either direction — and that where a band lives matters far more than how the
> step is spelled. All three are below; the advice that stands is § "the two
> timed results disagree", which reads #63 and #47 against each other.

Two caveats that belong with the numbers. The 128-column register column sits
on the cliff for four of the six forms above, so at 252–255 the spill bytes
carry the signal and the register count carries almost none; the 168 is well
clear of it and is the one to trust. And every number in this section is from
`regcount`, which was confirmed deterministic while #31 was measured — the
same tree twice gave the same 45 kernels to the byte. `regcount --determinism`
is that control as a flag now, so it is a thing you run rather than a thing you
remember (#60).

#### the same question, swept — #60

Everything above is two widths, because until #60 every width was a
hand-written probe. `device-tests`' `ladder!` generates one per (shape,
spelling), and the sweep says the two-point reading was right about less than
it looked. Registers, with spill store bytes, from `regcount`:

| per thread | shape | fused | assign | open-coded | rebound | all in place |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | `[32, 16]` | 32 | 32 | 32 | 32 | 32 |
| 32 | `[32, 32]` | 48 | 48 | 48 | 48 | 48 |
| 32 | `[16, 64]` | 56 | 56 | 56 | 55 | 56 |
| 48 | `[32, 48]` | 72 | 72 | 72 | 72 | 72 |
| 64 | `[32, 64]` | 94 | 96 | 96 | 96 | 39 |
| 64 | `[16, 128]` | 96 | 127 | 127 | 105 | 32 |
| 96 | `[32, 96]` | 128 | 47 | 47 | 251 | 32 |
| 96 | `[48, 64]` | 128 | 40 | 40 | 202 | 32 |
| 128 | `[32, 128]` | **168** | 252 | 252 | 255, 60 B | 32 |
| 128 | `[64, 64]` | 162 | 32 | 32 | 168 | 32 |
| 192 | `[32, 192]` | 255, 900 B | 255, 1012 B | 255, 1012 B | 255, 1096 B | 251 |
| 192 | `[48, 128]` | 255, 836 B | 255, 996 B | 255, 996 B | 255, 1868 B | 168 |
| 256 | `[32, 256]` | 255, 1704 B | 255, 976 B | 255, 976 B | 255, 2672 B | 255 |
| 256 | `[64, 128]` | 255, 1352 B | 255, 1584 B | 255, 1584 B | 255, 2904 B | 96 |

The first four columns are `scalar_map_probe`'s own forms, so the `[32, 128]`
row *is* the table above it, reproduced. The fifth is new: the accumulate
written in place too, which nothing in `scalar_map_probe` does.

**In place costs nothing, at fourteen shapes.** `assign` and `open_coded` —
the API and the `get`/`set` loop it replaces — agree on every counter at every
rung, including the ones that spill a kilobyte. That is #31's headline claim,
and it is now the best-supported thing in this section.

**Rows are not columns.** `[64, 64]` and `[32, 128]` are both 128 fp32 a
thread; `assign` is 32 registers at one and 252 at the other. Nothing before
this ladder could have caught that, because nothing before it varied `M` at
all — every probe in the repo is 32 rows. A register cost measured at one
extent does not transfer to the other.

**And "one expression is the floor" does not survive.** At `[32, 96]`,
`[48, 64]` and `[64, 64]` the in-place forms are 81 to 130 registers *under*
the fused one. What is happening there is visible in the stack frames the
sweep prints beside the registers: those rungs keep a frame as large as the
form they beat, or larger (`[32, 96]` is 47 registers on 1920 bytes against
128 on 1536). They did not fit the band — ptxas left it addressable in local
memory and streamed it.

So materialization-between-statements is real and it is one-directional:
rebinding is at least as expensive as the in-place spelling of the same two
ops at every rung above 48 values a thread, and dearer than the fused one at
every rung but `[16, 64]`, where it wins by a single register. What it does not
do is order the whole table, because whether the band reaches registers at all
switches on shape, and where the two disagree the shape effect is bigger.

Which leaves the rule above narrower than it was: *say the step in one
expression where you can, and write it in place where you cannot*. It is advice
about a shape and not a law — and whether an in-place form that trades 220
registers for a local-memory band is faster is a question `ptxas` cannot answer
and a timed kernel can. One has now been run.

#### with a clock on it — #63

Everything above this line is `ptxas -v`, which reports what was **allocated**
and never what it **cost**. `modal run modal_app.py::ladder_bench` times four of
those rungs on a B200: the three where in-place appears to win by 81–130
registers, and `[32, 128]` as a control where `fused` wins on both static
counters. Same probe body, each block dumping into its own band so a grid can
run it; `regcount` prints the timed rungs beside the ladder rungs and they price
identically at all twenty. Kernel time per band element per step, `x` against
the `fused` row of the same shape:

| | regs | frame | warps/SM | 1 warp ns | ×fused | device ps | ×fused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `[32, 96]` fused | 128 | 1536 | 16 | 4.703 | 1.00 | 8.6 | 1.00 |
| `assign` | 47 | 1920 | 32 | 4.654 | 0.99 | 8.5 | 0.99 |
| `rebound` | 251 | 1536 | 8 | 4.772 | 1.01 | 10.3 | 1.20 |
| `all in place` | 32 | 1152 | 32 | **4.400** | **0.94** | **6.4** | **0.75** |
| `[48, 64]` fused | 128 | 1536 | 16 | 4.756 | 1.00 | 8.8 | 1.00 |
| `assign` | 40 | 1920 | 32 | 4.538 | 0.95 | 9.3 | 1.05 |
| `rebound` | 202 | 1536 | 8 | 4.839 | 1.02 | 10.6 | 1.19 |
| `all in place` | 32 | 1152 | 32 | **4.349** | **0.91** | **6.6** | **0.74** |
| `[64, 64]` fused | 162 | 2560 | 12 | 5.117 | 1.00 | 12.4 | 1.00 |
| `assign` | 32 | 2560 | 32 | 4.816 | 0.94 | 10.2 | 0.82 |
| `rebound` | 168 | 2560 | 12 | 5.249 | 1.03 | 10.8 | 0.88 |
| `all in place` | 32 | 1536 | 32 | **4.614** | **0.90** | **7.2** | **0.58** |
| `[32, 128]` fused | 168 | 2560 | 12 | 4.798 | 1.00 | 10.6 | 1.00 |
| `assign` | 252 | 2560 | 8 | 4.526 | 0.94 | 8.9 | 0.85 |
| `rebound` | 255 | 2624 | 8 | 4.950 | 1.03 | 11.7 | 1.10 |
| `all in place` | 32 | 1536 | 32 | **4.355** | **0.91** | **6.5** | **0.62** |

`open_coded` is omitted only because it lands within 0.1% of `assign` at every
shape, as it lands on the byte at every static counter. The noise floor is
**0.8%** — the widest round-to-round spread over all twenty rungs and both
grids, most cells 0.1–0.6% — so a ratio inside 1 ± 0.008 is not a difference
this run can see. Every rung's `t(2S)/t(S)` is within 0.010 of 2.00, so nothing
was hoisted, and every launch matched an `f64` reference to 1.4e-4 against a
0.02 tolerance before it was timed at all.

**On this probe, a streamed band is not slower.** #63 was filed on the worry
that it might be serializing the inner loop, and here it is the other way round:
the *most* streamed spelling — 32 registers, the whole band left in local
memory — is the fastest at every shape on both grids, by 6–10% per warp and
25–42% across the device. Whatever the local traffic costs, on this kernel it is
less than what the freed registers buy. Hold "on this kernel" — the section
after next is the same phenomenon costing 2.6× on a different one, and the two
together are the actual result.

**And the control comes out backwards.** `[32, 128]` was chosen because it is
the cleanest cell in the whole ladder for "fused wins": 168 registers, no spill,
against `assign`'s 252. On the clock `assign` is 6% faster per warp and 15%
faster across the device. The one shape where the counters agreed is the shape
where they were wrong.

**So the register column does not order time, in either direction.** `assign` is
47 registers at `[32, 96]` and 252 at `[32, 128]`, and it is 0.99× and 0.85×
fused respectively — the *cheaper* of the two being the one that fails to win.
The frame does not order it either. Occupancy is the one with a real mechanism
and it explains more of the device column than anything else, but not all of it:
at `[32, 128]` `assign` runs 8 warps/SM against `fused`'s 12 and is faster
anyway, and at `[48, 64]` it runs 32 against 16 and is 5% slower.

Two things do survive. `assign` and `open_coded` are indistinguishable on the
clock as well as on every static counter, which is the strongest form of #31's
claim yet. And `rebound` is the slowest spelling per warp at every shape by
1–3%, three to five times the noise floor: materialization between statements
is real, and it is the only part of the model that a clock confirms.

#### the two timed results disagree — read #47 below with #63 above

Read the next section before taking any of this to a kernel. **#47 timed a
streamed band on `softmax` and found it cost 2.6×** — the same phenomenon, the
opposite sign, on a kernel rather than a probe. Both numbers are right and the
gap between them is the useful part.

What differs is not the band. It is **what the freed registers are able to buy**.

- `softmax` is **shared-memory capped**: `cuOccupancyMaxActiveBlocksPerMultiprocessor`
  says 6 blocks an SM at 39 registers a thread and 6 at 66, because shared memory
  binds either way. Streaming the band buys it *nothing*, so the local traffic
  shows up undiluted — 2.6×.
- The ladder probe uses **no shared memory at all**, so registers are the only
  constraint and the occupancy column moves with them: 8 warps an SM at 251
  registers, 32 at 32. Streaming buys 2–4× the resident warps, and that is what
  pays for it.

So the honest statement is neither "streaming is free" nor "streaming is
expensive". It is: **a streamed band costs real time, and whether it is worth
paying depends on whether the registers it frees are the constraint that is
actually binding.** The probe's own table shows the price when the payment falls
short — at `[48, 64]` `assign` runs 32 warps an SM against `fused`'s 16 and is
still 5% *slower* across the device. Twice the warps did not quite cover it.

**So the advice.** Do not read a register count as a speed. Read it with the
occupancy beside it — `regcount` prices both kernel crates now, and
`kittens-examples`' own status table prints
`cuOccupancyMaxActiveBlocksPerMultiprocessor` per kernel (#47) — and ask the one
question that matters: *does this kernel get more warps if the band leaves the
register file?* If shared memory or the launch geometry caps it anyway, keep the
band in registers and walk the tile in narrower chunks, which is exactly what
#47 did and what bought 2.6×. If registers are the binding constraint, the
in-place spellings that let `ptxas` stream are the fastest thing measured here
and the register table's "failure" is a win.

Within a spelling choice at fixed shape, the ordering is small and stable:
fully in place is fastest, one fused expression next, rebinding last. That
ordering is worth 1–3% per warp. Which side of the streaming cliff the shape
sits on is worth 25–160%. Pick the shape first.

Two limits on how far to carry the probe's half of this. It reads `M·N/32` fp32
a thread per step and does about three flops on each, so it is a load-heavy loop
with a long serial dependence, and at one warp a B200 SM is issue- and
latency-bound rather than short of registers — visible in the per-warp column
being nearly flat, 4.35–5.25 ns across every spelling and shape. A compute-dense
inner loop could price a streamed band differently. And local-memory *traffic*
was not measured on either side: that the band is in local memory is read off
the frame and the register count, and no profiler was run. What was measured is
time.

#### the timed kernel answered it — #47

**A local-memory band costs 2.6× on a real kernel.** #47 was filed as a
latency problem: `softmax` held a flat ~51 µs floor out to 64 blocks and
saturated at 362 GB/s, and the issue read "nothing is ever in flight" off the
kernel's one blocking TMA load and one blocking TMA store. The register table
says otherwise before any device is involved, and this is the reading the
sweep above exists to make possible — `regcount` now prices the examples crate
as well as the harness:

| `softmax_rows` | regs | spill | stack |
| --- | --- | --- | --- |
| `[32, 128]` band, whole row at once | 39 | — | **1024** |
| `[32, 32]` band, row walked in four | 66 | — | 256 |

39 registers cannot hold 128 fp32, and the frame beside them is where they
were: `ptxas` never promoted the band. The register column on its own reads
that as the *cheapest* kernel in the file, which is exactly the trap the stack
frame is printed to catch.

Walking the row 32 columns at a time — the widest `[32, N]` rung above with a
zero frame — and carrying `row_max`/`row_sum` across the chunks is the whole
change. The launch geometry, the barrier protocol, and both TMA waits are
untouched, so it is a controlled swap of where the band lives:

| rows × 128 × 2 planes | blocks | before, GB/s | after, GB/s |
| --- | --- | --- | --- |
| 128 | 2 | 2.5 | 5.7 |
| 512 | 8 | 10.2 | 22.4 |
| 4096 | 64 | 81.2 | 174.8 |
| 32768 | 512 | 280.6 | 646.9 |
| 262144 | 4096 | 346.9 | 922.9 |
| 524288 | 8192 | 367.1 | **952.1** |

Every size checked before it was timed, on the same B200 in the same session.
The flat floor went with it: 52.3 µs → 23.0 µs at two blocks, against the
GEMM's own 22.3 µs at two blocks. What is left there is a fixed cost every
launch on this harness pays, not something `softmax` does.

Three controls, because a number that moves is not yet an explanation.
**Occupancy did not change**: `main` now prints
`cuOccupancyMaxActiveBlocksPerMultiprocessor` per kernel, which says 6 blocks
and 24 warps an SM, and shared memory caps it at 6 either way — neither 39 nor
66 registers a thread is the binding constraint. (The same table reads 2 blocks
for `layernorm_rows` at nearly identical shared memory, which is its 255
registers, so the query is demonstrably register-sensitive and the 6 is not a
constant.) **It is not instruction issue either**: replacing `div_row`'s 128
`div.rn.f32` a thread with one `recip` per row and a `mul_row` — four divides
instead of 128 — measured 952.5 against 952.1 GB/s, and slightly *worse* in the
middle of the sweep, so it is not in the tree. And **the check sees the column
walk the change introduced**: pinning the third pass to column 0 fails 65,408
of the check's 65,536 cells by up to a relative 1.0.

So the issue named the right symptom and the wrong cause, and the residual is
worth stating rather than rounding off. 952 GB/s is still well under HBM, it is
not latency (nothing was put in flight and it went 2.6× faster), and it is not
issue (the reciprocal control). At 24 of 64 warp slots, shared-memory-capped,
what is left points at bytes per CTA — which is a launch-geometry change and a
different issue.

**#6 — reductions. Landed.** `row_reduce`/`col_reduce`/`tile_reduce` over a
`ReduceOp` (a `BinaryOp` with an identity), with `row_max`/`row_min`/`row_sum`/`row_prod`, the `col_*` mirrors,
and `tile_max`/`tile_min`/`tile_sum`/`tile_prod` derived from them. Both halves
are in the library now: a thread's own registers, then the `shuffle_xor`
butterfly over the lanes the ownership map spreads the folded axis across —
masks 1,2 for a row (what `quad_max`/`quad_sum` already were), 4,8,16 for a
column, all five for a tile.

All of it is **warp scope**. `tile_sum` folds one warp's band; layernorm's
group-norm statistic spans four warps' bands and needs them to agree, which is
the #3 entry below — #13 supplied the storage half and #3 the fold, and the two
compose exactly the way `groupnorm_tile` wanted:
`block_reduce_sum(partials, x.tile_sum())`.

**#7 — masking. Landed**, with the correction the examples surfaced. The issue
as filed describes ThunderKittens' signature, which takes no coordinate origin;
a flash kernel masks a `[queries, keys]` band whose diagonal sits at
`query_base - key_base`, and without the origin the op is unusable for anything
but a single-block attention. What shipped is `RegTile::make_causal(lane,
diagonal, fill)` with `tril`/`triu` on the same parameter, the four fills
(`left`/`right`/`upper`/`lower`) on a signed index for the same reason, and
`mask(lane, fill, keep)` underneath them all — a select at the layout's own
`(row, column)`, in place.

`make_causal_at(lane, query_base, key_base, fill)` is the spelling flash uses,
and taking the two origins rather than their difference is not sugar: the
difference is *negative* for every band above the diagonal, both bases are
`u32` at the call site, and `query_base - key_base` written there wraps to
+4 billion and masks nothing. The host tests carry that case, along with bands
wholly above and wholly below the diagonal — the two the origin-free form gets
silently wrong.

Not landed, as the issue's own sequencing asked: `transpose`, `swap_layout`,
`copy` between layouts. All three need a second `FragmentLayout` to mean
anything, and `BaseLdtm` is still the only one in tree.

**#9 — TMA store. Landed**, and it is what promoted `softmax` to **runs**.
`SharedTile::tma_store` / `tma_store_2d` mirror the load paths box for box;
`tma_store_commit` and `tma_store_wait::<N>()` are the completion side, and
they are *not* a `Semaphore` — a load's destination is shared memory where an
mbarrier can count arriving bytes, a store's is global memory where none can,
so a bulk store completes by counting per-thread groups with no barrier and no
byte accounting. `tma_store_wait_read::<N>()` is the weaker of the two waits:
it says the engine has finished *reading* the shared tile, which is what a
pipelined epilogue needs to recycle its buffer (#15's `lcsf`) and is never
enough as a kernel's last wait.

Not landed, and tracked separately as the issue asked: the **reduction** stores
(`store_add_async` and friends). `cp.reduce.async.bulk.tensor` is absent from
cuda-oxide at the pinned revision in every form, so that one is a `ptx_asm!`
intrinsic here or an upstream contribution — and nothing in the plain store
path prejudges which.

**#11 — global ↔ register. Landed.** Flash's epilogue and the GEMM's, and the
GEMM's was the one worth reading: twelve lines of open-coded
`RegTile::coordinate` arithmetic inside a `GAP` fence, exactly the index math
this library exists to delete. Both are now
`global::store_rows(GlobalRows::from_slice(&mut c, ldc), row, column, lane,
band)` — a cursor holding the destination's leading dimension, and a mover
that walks the fragment layout's own coordinates.

Three things about it are decisions rather than transcription:

- **It lives in `global.rs`, not `ldst.rs`.** `ldst` is the swizzled-fragment
  path: both its directions share one address derivation over `ldmatrix` and
  `stmatrix`, and this shares none of it. It also cannot be called
  `store_tile`, because `flash_forward` imports `ldst::store_tile` in the same
  file and would then have two of them.
- **It is fp32 and not generic over `Element`** — a choice since #3, and a
  constraint before it. The argument was that `Element` was implemented for
  `Bf16` alone, so `GlobalRows<E>` would have had exactly one instantiation and
  it would have been the wrong one: a `RegTile` *is* fp32, and this path exists
  to move one without rounding. `F32` is that impl now, so what is left is the
  parameter, a bound and two `E::read`/`E::write` calls — and no caller wanting
  a narrow one, which is why it is still fp32.
- **It is the only mover generic over `FragmentLayout`.** `ldmatrix`,
  `stmatrix` and LDTM each fix a lane map in hardware, so every other mover in
  the crate is pinned to `BaseLdtm`. A plain `st.global` fixes nothing, so
  these two take any `L` — which is the first place a second layout would pay
  off, whenever one exists.

It also costs nothing, which was not a given at 128 columns. `regcount`'s
`global_copy_probe_*` price the movers against the loop they delete, both
directions, at both widths:

| | 32 columns | 128 columns |
| --- | --- | --- |
| open-coded, `RegTile::coordinate` per value | 56 regs, 0 B stack | 168 regs, 1040 B stack |
| `load_rows` + `store_rows` | 48 regs, 0 B stack | 44 regs, 528 B stack |

Neither spills at either width. The direction is #22's again — the API is
*cheaper* than the hand-written loop, not merely free — and the likely reason
is the one thing the movers do that a call site never bothers to: the row
address is formed once per slot and the values indexed off it, so a `[32, 128]`
band costs four multiplies by the leading dimension instead of 128.

**#13 — shared vectors. Landed**, and it is what promoted `layernorm_rows`.
`kittens::shared::SharedVec<E, N>` is one flat run of elements with its own
single-box TMA path — rank 1 or a row of a rank-2 batch, both directions — and
`ldst::load_vec`/`store_vec` are the `ColVec` broadcast on top of it.

The decision worth recording is that it is **unswizzled and does not go through
`Swizzle` at all**, rather than riding a new `SwizzleNone` mode. `ATOM_BYTES`
does two jobs in `shared.rs` — swizzle period *and* TMA box width — and a
vector wants neither: its box is `N` wide, which no mode marker can carry, and
any atom small enough to divide a short vector would split it into stacked
subtiles the engine fetches one instruction each. A 128-byte atom is the
padding the issue was filed about. Making `SwizzleNone` fit would have meant
deciding what `ATOM_BYTES` means for an unswizzled *tile*, which is #14's
question; the vector needed none of it. `SharedVec<Bf16, 32>` is 64 bytes,
where the narrowest tile that even typechecks spends 128 on its single row.

`TileBox` now has its second impl, which is what #8 and #2 were both waiting
on: `BOX_COLS = N`, `BOX_ROWS = 1`, `SWIZZLE = NONE`, so a rank-1
`GlobalLayout` finally has something to hand a box to. Rank 1 is no longer
"expressible but untried" — the `shared vector round trip` device case runs one.

**#3 — block-scope reductions. Landed**, and it is what promoted
`groupnorm_tile`. `kittens::sync::block_reduce::<Op, WARPS>` folds one
warp-uniform value per warp into one block-uniform value through a
`SharedVec<F32, WARPS>`, with `block_reduce_sum` as the `Add` wrapper — the
same relationship `scale_rows` has to `row_map_assign` (#31), and for the same
reason. Under it, the `impl Element for F32` that **#2** closed without.

The issue was filed as *scope parameterization* — a `Scope` trait swept through
every signature — and it was retired in that form before it was built. The
argument for it was "cheap at 1851 lines, expensive after #5, #6 and #7"; those
landed, the op surface tripled, and the window closed. More decisively, a
`Scope` of `{ WARPS, THREADS, rank(), sync() }` **cannot express the one thing
the only real consumer wanted**: warps cannot shuffle to each other, so a fold
across them needs *storage* between two barriers, which is not a member of that
trait and cannot be. Sweeping the library would have left `groupnorm_tile`
exactly as blocked as it was.

What the trait would still be for — a `Warp`/`Warpgroup`/`Block` distinction in
the *types* rather than in the index math — this work says nothing about, and
building the reduction turned up no new evidence for it. It also did not need
one: the collective takes `WARPS` as a bare const parameter and derives nothing
else from it. Two cautions for whoever picks it up, both older than this issue
and both still standing: `Scope::rank()` would settle **#27** (implicit `lane`)
by accident, and `Warp::sync()` beside `Block::sync()` in one trait makes two
instructions of very different cost look interchangeable at a call site that
cannot afford `__syncthreads`.

Three things about the reduction are decisions rather than transcription:

- **Its bound is `ReduceOp`, not `BinaryOp`** as the issue's signature had it.
  The warps' partials arrive in slot order, which is not an order the caller
  chose, so a fold over them has to be associative and commutative to mean
  anything — the identical argument that keeps `Sub` and `Div` out of
  `row_reduce`. The identity is also what lets one loop serve every width.
- **The input must be warp-uniform, and it does not fold across lanes first.**
  Its input is a `tile_sum` result, which is already the whole warp's; folding
  again would count all 32 lanes' copies. That choice is not reversible by
  taste — under `Add` the two readings differ by a factor of 32 and the wrong
  one returns a number, so it is stated as a safety condition and a per-lane
  value has to go through `warp_reduce` at the call site.
- **Two barriers, and the second is the reusable part.** `groupnorm_tile` calls
  it twice on one scratch — mean, then variance — with nothing between the calls
  but arithmetic. The collective syncs *after* its own read as well as before
  it, so the variance pass cannot overwrite a slot the mean pass has not folded
  yet, and the caller owes no barrier. The alternative spelling (sync before the
  write instead of after the read) costs exactly the same two barriers and makes
  the rule a precondition on the caller rather than a postcondition the function
  establishes; this one was chosen so that back-to-back calls are self-sufficient
  by construction.

What it does **not** widen: `WARPS` has to be a multiple of four. `SharedVec`'s
`BOX_OK` wants a whole number of the TMA's 16-byte lines, so four fp32 — 16
bytes exactly, the minimum legal box — is the narrowest scratch that
constructs, and a 1- or 2-warp block cannot take this reduction at all. That
rule is a statement about a *TMA box*, and a scratch vector is the one use of
the type that never meets the engine; whether the assert belongs on `from_raw`
or on the four TMA methods is a live question this issue did not answer.

> The note that decided the shape, kept because it is the argument. A finding
> from writing #6: **`Scope` as filed cannot express the block reduction.** A
> block-scope fold needs somewhere for each warp's partial to live between the
> two barriers. `groupnorm_tile` guessed at that with a bare `*mut f32` carved
> out of its own shared plan, and the library cannot allocate one on its
> behalf: the shared plan belongs to the kernel. So the signature was either
> "take a scratch pointer" — not scope parameterization at all — or "take #13's
> `SharedVec`".
>
> **Settled by #13**: the second one. A `SharedVec` is a type the signature can
> name, its scalar access is per *element* rather than per packed word so warp
> `w` writing index `w` cannot read-modify-write a neighbour's value, and it is
> still the *kernel's* allocation — `from_raw` takes a base pointer like every
> shared type here — so nothing about the shared plan moved into the library.

**#8 — global layout.** ~~Not reachable from a kernel at all~~ — **closed**, and
it is why this crate now has a launcher at all. `kittens::global::GlobalLayout<E,
RANK>` describes a buffer by extents and byte strides at any rank, packed or with
a leading dimension, and `tensor_map::<Tile>()` reads the box, the swizzle and
the data type off the `SharedTile` the map is paired with — the agreement
`encode_bf16_panels` had by being one hardcoded builder, now stated as a type.
`gemm::check` builds both operands' maps from it and differs only in extents.

What #8 does *not* close: `GlobalLayout` is bf16-only, and since #3 the reason
has moved one trait along. `Element` has an fp32 impl now, but a tensor map
also needs `TensorMapElement` — the `CUtensorMapDataType` the driver reads the
buffer as — and that is `Bf16` alone, so an fp32 buffer still cannot be
described. It is one associated constant
(`CU_TENSOR_MAP_DATA_TYPE_FLOAT32`), left undone because nothing in tree TMAs
an fp32 buffer: a block reduction's scratch is written by `set` and read by
`get`, and never meets the engine. Two of the three things that entry used to
say have since gone: `TileBox` has a second impl (#13), so a layout has
something to hand a non-tile box to, and rank 1 is no longer untried — the
`shared vector round trip` device case builds one and moves a vector through
it.

**#12 — MMA coverage.** Not demanded. `mma_abt`, `mma_ab` and `mma_walk_cg2`
covered every multiply in all four kernels. Worth recording as a negative
result: the MMA layer is the part of this library that is finished.

That negative result is what shaped #12 when it landed. Nothing here wanted
`AtB`/`AtBt`, so nothing here would have caught them being wrong, and an
unused walk that is wrong is worse than an absent one. The work is therefore
mostly *harness*: `device-tests` now runs all four operand orders over one
pair of logical matrices staged four ways, with a host blind-spot sweep that
fails if the reference cannot see a dropped K plane, a permuted chunk or a
confused MN subtile, and an untransposed control required to disagree. The
one thing this file's kernels did ask for is the `mm_*` half: `flash_forward`
passed `false` twice for "start this band fresh", and that is now
`mm_abt`/`mm_ab` — the invariant in the entry point instead of in an
argument.

---

### Came from writing kernels, not from ThunderKittens' feature list

The most valuable output here. Ordered by how badly it hurts. Items 1–6 were
filed as **#21**, **#22**, **#25**, **#23** and **#24** (which covers both
cluster-scope entries); **1, 2, 3, 4 and 5 have since shipped**. The numbers are noted
inline and the prose is kept as written, because it is the argument rather than
the ticket.

#### 1. ~~There is no shared → register path at all~~ — **closed by #21**

This was the item this section was written for: `ldst` was store-only, so any
kernel whose input is not an MMA operand could not start, and softmax was
blocked before its first line.

`ldst::load_fragment` is now the `ldmatrix` half, sharing the store side's
address derivation (`ldst::fragment_address`, host-tested) and validated
against silicon by the `ldmatrix map` device case. `cuda_device::wmma::
ldmatrix_x2` lowers fine for `sm_100a` — the note that `ldmatrix` "lives only in
the Hopper `wmma` path" was about where the function is filed, not about what it
compiles to.

What remained of this item was shape, not direction — and item 3 below closed
the other half of that: `chunk_writer` spans stacked subtiles, so the path
exists at every width the library can describe. Item 2 closed the last of it:
`load_tile`/`store_tile` move a whole band, so nothing about the shared ↔
register path is a loop the kernel writes any more.

#### 2. ~~The movers are per-`[16, 16]`-block, so every kernel writes the same loop~~ — **closed by #22**

`TmemTile::fragment_tile` returned a `[16, 16]` `Fragment` and
`ldst::store_fragment` took one, so a kernel that wanted a `[32, 128]` band
wrote a four-deep block-composition loop to assemble it — `gemm.rs` did, flash
would have, `softmax.rs` wrote it *twice*, once per direction, and the device
harness kept its own copy in the test crate. The composition is a property of
the layout, and `reg.rs`'s own test
(`fragment_blocks_tile_the_bigger_shapes`) stated it as an invariant that no
function implemented.

All four directions landed together: `TmemTile::tile(row, column)` and
`TmemTile::store_tile(row, column, tile)` over TMEM, `ldst::load_tile(chunks,
row, column, lane)` and `ldst::store_tile(chunks, row, column, lane, tile)`
over a swizzled shared tile, each composing the single-block mover it is built
on out of the same two placement helpers — so a band cannot mean one thing in
TMEM and another in shared memory. The harness' `drain_band`/`stage_band` are
gone, replaced by the library calls, which is where the TMEM side's device
coverage comes from; the shared side got a `band round trip` case of its own,
and `softmax` runs on both directions of it.

The composition is free where it was measured to be dangerous (#5's `row_map`
cliff at 128 columns): every TMEM drain kernel in `device-tests` reports the
same register count as its hand-written loop, and the shared-side band is
*cheaper* composed — 36 registers against 52 for the loop `softmax.rs` used to
write (`modal run modal_app.py::regcount`).

#### 3. ~~`SwizzledChunks` cannot span stacked subtiles~~ — **closed by #25**

`SharedTile::chunk_writer` const-asserted a one-subtile tile, so only tiles 64
bf16 columns wide had a register ↔ shared path at all — both directions, since
`load_fragment` addresses through the same cursor. Softmax's `[128, 128]` tile
could be neither read nor written back.

The cursor now walks subtiles the way `tma_load` does, and for the same reason:
a stacked subtile is `SUBTILE_BYTES = rows * 128` further along, so subtile `i`
row `r` is the tile's 128-byte row `i * rows + r` and the swizzle — which keys
off *physical* address bits `[9:7]` — takes that row's phase. Chunk indices
count across the whole logical row, so `fragment_address` needed no change at
all. Checked against silicon at width by the `swizzle/stmatrix/ldmatrix … wide`
device cases, and by a 4-row tile whose second subtile starts mid-period —
which is the only shape where an absolute phase and a per-subtile one differ.

#### 4. ~~The `RegTile` shape set is closed by the library~~ — **closed by #23**

`BaseLdtm` implemented `FragmentLayout` for `(16,16)`, `(32,32)` and `(32,128)`
and nothing else, because each shape was a line of `base_ldtm_shapes!` *inside*
`src/reg.rs`. Flash wants a `[32, 64]` score band and could not add one.

The issue's cheap option — export the macro — is **not available**, and the
orphan rules are what decide it: `impl FragmentLayout<32, 64> for BaseLdtm`
from another crate is E0117 whatever macro writes it, since `FragmentLayout`
takes only *const* parameters and so no local type can ever appear in the impl
header. Confirmed by compiling it, not by reading the reference.

What landed is the issue's third option, in its trait-projection form.
`RowLayout::Slots<T>` is generic in the element, so a tile's storage is a row
array *of* a column array and `FragmentLayout` is a **blanket impl** over
`RowLayout<M> + ColLayout<N>` — no `(M, N)` needs an impl, no array length is
computed from `M` and `N`, and `generic_const_exprs` stays out of it. The shape
set is the product of the two extent lists, which now run to every multiple of
16 up to 512: 1024 shapes out of 64 impls, and every shape that fits in a
thread's registers at all, since `M * N / 32` is already 256 at the corner.

The blanket impl is also what makes the trait genuinely open: a *layout*
defined out of tree gets `FragmentLayout` for free the moment it has both
halves, which is the one thing orphan rules do allow.

#### 5. ~~No cluster-scope TMEM allocation~~ (#24) — **closed by #46**

`tmem::alloc_block` is `tcgen05.alloc.cta_group::1`. A `cg2` accumulator is one
allocation spanning the CTA pair, and this section used to say that only the
leader may issue it, with the peer reading the leader's staging word over
DSMEM. That was the bug, not the fact: all three `cta_group::2` allocator
instructions want *one full warp in each peer CTA*, the collective writes an
address into each CTA's own staging word, and issuing them from rank 0 alone
hung the kernel's second launch (#40).

Shipped as `tmem::alloc_cluster` / `tmem::dealloc_cluster` — `gemm.rs`'s own
fixed body, moved rather than rewritten, since it is the version that ran on
silicon.

#### 6. No cluster-scope semaphore arrival (#24)

A 2-CTA MMA consumes four tiles staged by two CTAs, so the issuer needs one
barrier that says *the whole stage is present* — the peer's TMA has to complete
on the leader's copy. mbarrier addresses are cluster-mappable and this is the
standard pair-wide producer handoff, but `Semaphore` is CTA-scoped by
construction and says nothing about it. `Semaphore::multicast_alias` is the one
cluster-aware thing in `sync.rs`, and it solves the opposite problem (one
barrier per CTA, one transfer).

Note also that `Semaphore::expect_tx` lowers to `mbarrier.arrive.expect_tx
… .shared::cta`, so it cannot charge a remote barrier even if handed one;
upstream's `mbarrier_arrive_cluster` takes a remote address but carries no
transaction bytes. The shape of the fix is a design question, not a
transcription.

Wanted: `Semaphore::at_rank(rank)`, and a decision about the byte accounting.

#### 7. `pipeline::run` cannot schedule a cluster

The work-item map is `blockIdx.x` strided by `gridDim.x`, so the two CTAs of a
cluster get *different* items. The GEMM therefore does not use the persistent
scaffold at all — it is one cluster per output tile with the K loop as its
pipeline. `prototype::lcf` predates clusters in ThunderKittens too, so this is
not a porting oversight; the scaffold needs a way to say who a work item
belongs to, and `blockIdx.x` is not it.

This entry used to name #3's `Scope` as the missing piece. #3 shipped a block
reduction instead and no `Scope`, so what is wanted here is unfiled: a cluster
rank in the work-item map, which is a `%cluster_ctarank` and a divisor, not a
trait. Nothing about the block reduction bears on it.

Related, and cheaper: #15's `lcsf` was filed as depending on #9, which has now
landed with the wait it needs — `tma_store_wait_read` releases the shared buffer
as soon as the engine has read it, without blocking on global visibility, which
is exactly what lets an epilogue overlap the next tile's first K loads. The
GEMM shows the second reason to want it: with the store folded into `finish`, a
persistent GEMM cannot overlap those today.

#### 8. Multicast has no geometry to live in

`tma_load_2d_multicast_cg2`, `commit_multicast_cg2` and `mma_walk_cg2` all
hardcode the 2-CTA pair (GAPS §2.4). Multicast starts paying, as *replication*,
at cluster ≥ 4 (2×2: `A` broadcast along the N axis, `B` along the M axis),
which the `_cg2` suffix rules out. Generalizing the cluster mask is filed
nowhere.

This item said the GEMM "cannot use the load", since under a 2-CTA UMMA both
operands are already split and nothing is replicated. Running the kernel
corrected that, and the correction is the more useful fact: the GEMM uses
`tma_load_2d_multicast_cg2` with a **one-CTA mask**, replicating nothing, purely
because the multicast form's barrier operand lives in `.shared::cluster` and the
plain form's does not. A pair-wide producer handoff is impossible without it.
The `multicast` in the name describes the delivery; what a cluster kernel
actually needs from it is the address space. Filed as a correction on #24.

#### 9. Smaller things, each one line of API

- **`expect_tx` byte accounting is hand-summed.** Every producer writes
  `(ATile::BYTES + BTile::BYTES) as u32` and has to keep it in step with the
  loads it issued. A tile knows its own size; `Semaphore::expect_tiles` or a
  charge returned by `tma_load` would make the two impossible to disagree.
- **Coordinate-dependent ops need `lane` passed in.** Settled by #27:
  implicit for ops that execute, explicit for pure coordinate queries. The
  masks (#7) take it explicitly, on that rule — `device-tests` and `reg.rs`
  build their expectations by calling the map across all 32 lanes on the
  **host**, where `warp::lane_id()` does not exist, and that is what makes the
  mask's own test non-vacuous rather than a restatement of the kernel.

---

## What the examples confirm already works

Worth recording, because the promotion of an example from aspirational to
compiling is the thing that proves an issue is finished, and half of this
library is already at that bar:

- **The MMA layer.** `mma_abt`, `mma_ab` and `mma_walk_cg2` covered every
  multiply in four kernels, in the layouts they wanted, with no gaps. Each
  builds its own instruction descriptor from the walk it issues and the
  operand element, so a kernel names only its accumulator band's `MmaShape`
  and cannot pair a walk with transpose bits that disagree (#30). #12 has
  since closed the operand-order square (`mma_atb`, `mma_atbt`) and given
  every walk an `mm_*` twin that starts the accumulator fresh, which is the
  form both of `flash_forward`'s MMAs now take. The one field still stated by
  a caller is `mma_walk_cg2`'s element (`mma_walk_cg2::<Bf16, CHUNKS>`), and
  it stays that way: since #12 the element routes tcgen05's `KIND` as well as
  the operand format, so a walk that carried its element back would stop being
  layout-only.
- **TMEM segment carving.** Flash splits one allocation into its `[128, 64]`
  scores and `[128, 128]` output with `TmemTile::split_columns`, so no kernel
  here adds a column offset to a bare TMEM address (#28).
- **The pipeline primitives.** `SharedTileRing` + `SemaphoreRing` express the
  GEMM's K pipeline exactly, including the subtle part: `free` is released by
  the MMA's own commit, so the accumulator instruction rather than a thread is
  what proves the operand has been read. The `index → (stage, parity)`
  arithmetic never appears in the kernel.
- **`online_rescale`.** The one genuinely subtle piece of flash attention is a
  single call.
- **The swizzle and fragment layers.** No kernel here spells a swizzle phase, a
  chunk index, a subtile stride or a descriptor field — at any tile width,
  since #25. That is the library working.
