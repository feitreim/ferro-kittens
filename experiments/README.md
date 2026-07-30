# experiments

**The measurement record.** Every sweep, every ablation, every rung that lost,
and the reasoning that picked the one `examples/` ships. It was written as
`examples/README.md` while `examples/` was doing three jobs at once; it moved
here with the sweeps it describes, unedited apart from the paths in this header
and the six links that had to follow the two crates apart.

Read it as a notebook and not as documentation. The sections are in the order
they were measured, later ones correct earlier ones by name, and a figure with
an issue number beside it belongs to the kernel that existed at that issue —
several of which no longer do. §7 says so at each point where it matters.

Four kernels were written the way we want them to read, and a header on each
said whether it runs or whether it only compiles. All four compile as of #3;
two run.

The value of that crate was the **diff** between what the kernels wanted to say
and what `kittens` could express. An aspirational example — one naming API that
did not exist yet — is not a placeholder but a statement of a missing API in the
only terms that matter, which is what a kernel author has to type. That diff is
now empty, and this file is the record of what it produced while it was not.

Its own workspace, like `device-tests` and `examples`, so `cargo` at the repo
root never sees it. (The root `Cargo.toml` also needs `autoexamples = false`:
`examples/` is one of cargo's own target directory names, and `exclude` does not
stop target auto-discovery, so without it `examples/src/main.rs` gets compiled
as an example target *of the library*.)

## Status

| Kernel | Status | Blocked on |
| --- | --- | --- |
| [`gemm`](src/gemm.rs) | **runs** — exact against a CPU reference, `==` on a bf16 `C` against a reference rounded the same way (#108) | — (and no gap worked around in-file any more) |
| [`softmax`](../examples/src/softmax.rs) | **runs** — within 2⁻⁸ of a CPU reference | — |
| [`layernorm`](../examples/src/layernorm.rs) | **runs** — `layernorm_rows` within 2⁻⁷ of a CPU reference; `groupnorm_tile` compiles, no launcher | — |
| [`flash_forward`](../examples/src/flash_forward.rs) | **compiles** — no launcher yet | — |

All four are in the default build and this crate has **no cargo features left**.
That is not tidying: `cargo oxide build kittens-examples --arch sm_100a` — which
is `modal run modal_app.py::build`, and CI tier 2 — compiles the *default*
feature set, so a gated kernel was never codegened by the thing that codegens
this crate. `flash_forward` is the example with the most library surface in it
(both MMA layouts, the swizzled `P` staging tile, `online_rescale`, masking,
both memory ends), and until #3 it was the one example nothing could catch a
regression in. The gate was standing in for a claim a real build makes better.

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
writing kernels rather than from ThunderKittens' feature set: a cluster
geometry for multicast. The `expect_tx` byte accounting was the other and is
closed by #29 — every producer's charge is now derived from the loads it
issued. None of them blocks a kernel in this directory. The work-item map that
knows about clusters is closed by #51, and the GEMM runs on it — §7 below is
worth reading for the half that is *not* the API, since picking the persistent
grid's size wrong cost a factor of two and picking it right needed a residency
the driver cannot be asked for.

`layernorm` was the first example to be **split** by a landed issue rather than
promoted whole: #13 took `layernorm_rows` out of the gate and left
`groupnorm_tile` behind it, because its statistic spans four warps and #13
supplied only the storage half of what that needs. #3 supplied the other half
and the split is over — both kernels are in the default build and both get a
real `sm_100a` compile. The `layernorm` cargo feature went with it, and so did
`flash`: see the status table above for why a feature gating a kernel that
compiles is worse than no feature at all.

No arithmetic entry is open any more. `RegTile::add_assign` was the last one
and **#31** landed it, along with an `_assign` twin for every other map on all
three register families. What #31 measured on the way is more interesting than
the API it shipped, and § "in-place versus by-value" below is rewritten around
it.

```sh
cargo oxide build kittens-examples --arch sm_100a      # all four kernels
cargo oxide run kittens-examples                       # and run the ones with launchers, on a B200
cargo oxide build kittens-experiments --arch sm_100a   # every rung and probe below
cargo oxide run kittens-experiments -- check           # and check the ones that compute a GEMM
```

From the repo root: `scripts/modal-run build` for the two builds and
`scripts/modal-run examples` for the two runs — that entry point invokes both
binaries, so the correctness gate is the one it always was.

## The gap lists, and why there is no longer a command for them

`modal run modal_app.py::gaps` is **retired** (#3). It turned each aspirational
kernel's cargo feature on and printed the resulting compiler errors, which is
the right instrument while a kernel names API that does not exist: the missing
surface *is* the error list, at the call sites that want it, read off a compiler
rather than off the last person's memory. It is gone because **every list
reached empty and the features are gone with them** — and an empty section is
not a finding, it is a command with nothing to say.

What replaced it is stronger rather than weaker. All four kernels are in the
default build, so `build` codegens each one for real `sm_100a`, and that catches
what a `cargo check` cannot: a post-monomorphization `const { assert!(..) }` is
invisible to a type-check and fires only at codegen. `SharedVec<F32, 4>`'s box
assert was exactly one of those.

The two lists as they stood at the end, both read off the last run of `gaps`
before it was retired, and kept because *how* each emptied is the useful part:

- **`flash_forward`** — **empty**, which was the interesting outcome for a gap
  list. Nothing in it reaches past the library; it wants a launcher and a CPU
  reference. The list was three errors until #23 landed, two until #11 and #31
  landed together, and how the third went is worth keeping: `make_causal_at`
  (#7) was wanted all along and never appeared as its own error, because it is
  called on the `[32, 64]` band whose `FragmentLayout` bound already failed, and
  an unsatisfied bound on the receiver suppresses method resolution on it.
  Counting errors would have said #7 had landed.
- **`layernorm`** — one error, `sync::block_reduce_sum` (#3), in
  `groupnorm_tile` only; `layernorm_rows` had left the gate with #13. #3 closed
  it, and the feature went with the kernel.

If a new aspirational kernel is ever written, the instrument is worth rebuilding
exactly as it was — a feature per kernel, checked on its own so an error belongs
to a known kernel, non-zero exit reported rather than raised. What is not worth
keeping is the gate on a kernel that compiles, because a gated kernel is one the
`sm_100a` build never sees.

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
grid's arithmetic can produce is invisible. [`softmax::permutation`](../examples/src/softmax.rs)
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
replication, right address space. The primitive was already in the library,
filed under the opposite problem; §6 below is where it acquired a name that
says what it does.

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
occupancy beside it — `regcount` prices all three kernel crates now, and
`kittens-examples`' own status table prints
`cuOccupancyMaxActiveBlocksPerMultiprocessor` per kernel (#47) — and ask the one
question that matters: *does this kernel get more warps if the band leaves the
register file?* If shared memory or the launch geometry caps it anyway, keep the
band in registers and walk the tile in narrower chunks, which is exactly what
#47 did and what bought 2.6×. If registers are the binding constraint, the
in-place spellings that let `ptxas` stream are the fastest thing measured here
and the register table's "failure" is a win.

#### and a zero in that column is two different things (#70)

Every `flash_forward` figure in this subsection is quoted at the 147536 B its
shared plan was before #125 handed it to `kittens::plan::SharedPlan`, which
reserves the `tcgen05.alloc` staging word as the four-byte `u32` it is: the
plan is 147532 B now. Four bytes out of 144 KiB decides none of what follows,
and the measurements are left at the sizes they were taken at.

`flash_forward` printed **0 blocks/SM** at 147536 B, which reads like a plan
too big for the SM and was not. Dynamic shared memory over 48 KiB is opt-in per
function — `cuFuncSetAttribute` with
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` — and
`cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 0 for a function nobody
opted in for in exactly the same way it answers 0 for tiles that do not fit.
Same function, same 147536 B, one `kittens::launch::admit_shared_plan` call
between them:

| | max dynamic shared | blocks/SM |
| --- | ---: | ---: |
| before | 49152 B | **0** |
| after | 147536 B | **1** |

The device's opt-in ceiling is 232448 B, so the plan was never within 84 KiB of
being too large. `gemm` is over 48 KiB too and launches because
`#[launch_contract]` issues the same opt-in from the prepared-launch path —
`cluster_launch` was never what saved it.

The second half of that measurement is the more useful one. Swept on a freshly
loaded function per probe, `flash_forward` answers 1 block/SM at 147536 B, at
73792, at 32800 and at **0** — and 1 at block widths of 32, 64, 128 and 256,
where `softmax_rows` goes 28/14/7/3 and `layernorm_rows` 8/4/2/1. A ceiling
flat in both shared memory and warp count is neither of those resources; it is
something held once per CTA. `flash_forward` is the only queryable entry here
that allocates TMEM. So **shrinking the K/V rings would buy nothing**, which is
the opposite of what the 0 invited, and the register-versus-occupancy advice
above applies to this kernel not at all until that per-CTA ceiling is named.

#### and the per-CTA ceiling is tcgen05 itself (#74)

That last sentence named a suspect on a correlation over a sample of one, and
said so. The control it wanted is now `device-tests`' `tmem occupancy ladder`:
a nine-register kernel whose whole content is one `tcgen05.alloc`, against the
byte-identical kernel with the allocator deleted. Same 32-byte shared plan,
same block sync, same store; the only thing that varies between rungs is the
allocator's operand. Blocks/SM on a B200 at zero dynamic shared:

| rung | columns | regs | cluster | 32 | 64 | 128 | 256 threads |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| no tcgen05 | — | 10 | none | 32 | 32 | 16 | 8 |
| `alloc` | 32 | 9 | none | 1 | 1 | 1 | 1 |
| `alloc` | 64 | 10 | none | 1 | 1 | 1 | 1 |
| `alloc` | 128 | 10 | none | 1 | 1 | 1 | 1 |
| `alloc` | 192 | 10 | none | 1 | 1 | 1 | 1 |
| `alloc` | 256 | 10 | none | 1 | 1 | 1 | 1 |
| `alloc` | 512 | 10 | none | 1 | 1 | 1 | 1 |

**TMEM confirmed.** The control sits at the hardware's own warp ceiling — 2048
threads an SM, so 32/32/16/8 — because at 10 registers and 32 bytes nothing
else is binding. Add one `tcgen05.alloc` and it is 1, at *every* block width
and at *every* legal column count.

**That last sentence is a fact about the query, and the next section is where
it stops being a fact about the SM.** Read on before acting on it: the reading
this table was given — that a CTA touching the allocator is charged the SM's
entire tensor memory, so the allocator's smallest unit costs what all 512
columns cost — has since been measured directly and is false.

Two things that table rules out, which a flat ladder alone could not. The 192
rung takes its column count as a **kernel argument**, so there is no constant
for the driver to have read: it is pricing the allocator, not a number `ptxas`
recorded. And `required_cluster_dimensions` is `None` on every rung, so tcgen05
implies no `cta_group::2` shape and the query is not quietly answering about a
launch geometry the kernel never declares.

`flash_forward` confirms it directly: cut from 192 columns to 32 and
re-queried, it still answers 1 block/SM.

So the 1 is not a defect in that kernel. What it also is not, and what the two
paragraphs that used to stand here concluded, is the SM's actual residency.

#### and the 1 is the query, not the SM (#78)

The reading above — every tcgen05 kernel pinned at 1 CTA/SM, warps per CTA the
only remaining lever — was drawn from a driver's answer, and #51 immediately
contradicted it by measuring `gemm` losing **2.07×** to a one-cluster-per-SM
grid cap. A cap cannot cost anything if it is already the ceiling. So the query
had to be checked against the hardware rather than trusted, and the instrument
for that is not another query and not a throughput curve: `%smid` names the SM
a CTA is running on and `%globaltimer` is a device-wide nanosecond clock, so a
CTA can simply **write down where it ran and when**. `device-tests`'
`tmem residency census` sweeps those intervals per SM and takes the most that
were ever open at once.

Peak CTAs counted on one SM, B200, 4736 one-warp CTAs over 148 SMs, 100 µs
each, against 233472 B of shared memory per SM:

| rung | columns | shared B | budget | resident | holding | driver says |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no tcgen05 | — | 32 | — | 32 | **32** | 32 |
| `alloc` | 32 | 32 | 16 | 17 | **16** | 1 |
| `alloc` | 64 | 32 | 8 | 9 | **8** | 1 |
| `alloc` | 128 | 32 | 4 | 5 | **4** | 1 |
| `alloc` | 256 | 32 | 2 | 3 | **2** | 1 |
| `alloc` | 512 | 32 | 1 | 2 | **1** | 1 |
| `alloc` 512 then free | (512) | 32 | — | 27 | **25** | 1 |
| `alloc_cluster` none | — | 32 | — | 8 | **8** | 8 |
| `alloc_cluster` | 128 | 32 | 4 | 5 | **4** | 1 |
| `alloc_cluster` | 512 | 32 | 1 | 2 | **1** | 1 |
| flash's plan, no tcgen05 | — | 147536 | 1 | 1 | **1** | 1 |
| `alloc` + half that plan | 256 | 73768 | 2 | 3 | **2** | 1 |
| **`flash_forward`'s envelope** | 256 | 147536 | 1 | 1 | **1** | 1 |
| **`gemm`'s envelope (cg2)** | 128 | 73792 | 3 | 3 | **3** | 1 |

`holding` counts CTAs whose columns were outstanding at the same instant, and
`budget` is `min(512 / columns, shared per SM / plan)` — the two per-CTA
resources an SM divides, whichever is tighter. **Holding equals budget at every
one of the fourteen rungs.** That is the whole model, and it is exact.

Two of those rows are the answer to #78, and they do not agree with each other's
headline:

- **`gemm`'s envelope counts 3**, which is precisely the residency #83 bisected
  out of a grid cap on a clock. Two instruments with nothing in common agreeing
  on one integer is the strongest thing in this section. Its 128 columns would
  allow 4; its 73792 B of shared memory allows 3; the smaller wins.
- **`flash_forward`'s envelope counts 1, so its 1 CTA/SM is REAL** — and
  tcgen05 is not what causes it. Its 256 columns allow 2. Its 147536 B shared
  plan allows 1, *with no allocator in the kernel at all* (the row above it).
  Shared memory is the only thing capping that kernel.

Three things fall out of the table that no column sweep alone could give:

- **`cta_group::2` is charged identically.** `alloc_cluster` at 128 columns
  admits four CTAs an SM, exactly as `alloc_block` at 128 does. The pair does
  not split one SM's tensor memory between its ranks; each rank pays its full
  count against its own SM. That was the leading hypothesis for why `gemm` and
  `flash_forward` might genuinely differ, and it is refuted.
- **The query is reacting to the instruction, not to a held resource.** The
  allocate-and-release rung takes all 512 columns and gives them straight back
  before doing anything, and still runs 25 CTAs an SM — while the query goes on
  answering 1 for it, because a `tcgen05.alloc` is present in the code.
- **The queries are otherwise accurate**, which is what made the 1 easy to
  believe: 32 predicted and 32 counted for the plain control, 8 and 8 for the
  cluster control. And `cuOccupancyMaxActiveClusters` **can** describe a
  `#[cluster_launch]` kernel — the honest `cluster` placeholder in the occupancy
  column below, and #51's note that the GEMM's residency "cannot currently be
  answered", both predate anyone calling it.

`resident` runs exactly one above `holding` wherever tensor memory is what
binds. That extra CTA is admitted to the SM and parked inside `tcgen05.alloc`,
which blocks until the columns free up: it occupies a slot and makes no
progress. Neither the occupancy query nor a grid-cap throughput sweep can tell
that apart from not being resident at all, which is why the census timestamps
entry and allocation separately. Where shared memory binds instead, `resident`
and `holding` are equal and the wait column collapses to a fraction of a
microsecond — nobody is queuing, because the CTA that would have queued was
never admitted.

##### what this leaves `flash_forward`, which is not what #74 or #70 concluded

Both of the two levers those issues considered were assessed through the query,
and the query was pinned at 1 by the allocator in every one of those readings.
Corrected:

- **Shrinking the TMEM allocation is a real lever in general** — halving columns
  to the next legal power of two doubles the CTAs tensor memory holds, for
  nothing but arithmetic — and it is **worth nothing to `flash_forward` today**,
  because shared memory already caps it below what its columns allow. #74 got
  the right advice for this kernel from an argument that does not hold.
- **Shrinking the shared plan is the lever**, and #70 ruled it out. 147536 B
  admits one CTA an SM. Two would need the plan under **116736 B** — half of the
  SM's 233472 — which is a K/V ring stage or a `PTile` away. That threshold is
  arithmetic and *not* measured: the census brackets it at 73768 B (which counts
  2) and 147536 B (which counts 1) and does not locate it between them. The
  measured claim is the weaker and sufficient one: the plan alone sets this
  kernel's residency, with no allocator anywhere in the kernel.
- Warps per CTA is still worth pricing — `flash_forward` runs 4 against its own
  `max_threads_per_block` of 512 — but it was never the only thing left, and the
  argument that it was rested on a query that a clock has now contradicted twice
  (#83, and the table above).

None of that is a change anyone should make blind: `flash_forward` still has no
launcher and no CPU reference, so a decomposition that shrinks the ring cannot
be checked for correctness, let alone timed. That prerequisite is the follow-up,
not this table.

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
launch on this harness pays, not something `softmax` does. **(Wrong — see
#76 below. Twelve of those 23 µs were this kernel's own `exp2`, and the same
launch reaches 10.9 µs.)**

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
middle of the sweep, so it is not in the tree. **(This control is the one #76
overturns, and it is worth understanding rather than deleting: it is a correct
measurement of a kernel that the `exp2` polynomial had already made
compute-bound. The same swap, once the polynomial is gone, is 3153 → 4414 GB/s.
A control is only valid under the bind it was taken in.)** And **the check sees
the column walk the change introduced**: pinning the third pass to column 0
fails 65,408 of the check's 65,536 cells by up to a relative 1.0.

So the issue named the right symptom and the wrong cause, and the residual is
worth stating rather than rounding off. 952 GB/s is still well under HBM, it is
not latency (nothing was put in flight and it went 2.6× faster), and it is not
issue (the reciprocal control). At 24 of 64 warp slots, shared-memory-capped,
what is left points at bytes per CTA — which is a launch-geometry change and a
different issue. **(It was not bytes per CTA either. It was instruction issue
after all, in the one place the reciprocal control could not see: #76.)**

#### the same defect with the opposite sign, and the rung the ladder got wrong — #67

`layernorm_rows` was the other end of it: **255 registers, 16 bytes of spill and
a 1552-byte frame**, and the occupancy query above reading **2 blocks and 8
warps an SM** where `softmax` reads 6 and 24 at 33,344 against 32,800 bytes of
shared memory. Same band shape, opposite failure — softmax's 39 registers were
too few to hold `[32, 128]`, layernorm's 255 were too many to fit two CTAs on an
SM, and *neither* kernel had the band in registers.

The diagnosis is the occupancy query and not the 255. Shared memory admits 6
CTAs at 33,344 bytes; the driver says 2; `255 × 128 = 32,640` against a 65,536
register file is 2.008. Registers were binding, and the same query says they
stopped: after the change, **6 blocks and 24 warps at byte-identical shared
memory**.

It also has a **launcher, a CPU reference and a `bench.rs` row** now, which it
did not before — that had to come first, because "faster" and "wrong faster"
are the same number otherwise. The seed is not softmax's and could not be: a
permutation seed gives every row the same multiset, so the row statistics are
the tile statistics and `groupnorm_tile` would compute layernorm's answer
exactly. `layernorm::value` carries that argument and the row-dependent
distribution shape that closes it.

**The ladder narrowed the search and then got the rung wrong**, which is the
part worth carrying forward. `[32, 32]` has a zero frame in all five spellings
of the probe above, and it is what #47 chose for `softmax`. This kernel at
`CHUNK = 32` gets **40 registers on a 128-byte frame** — the probe carries no
`ColVec` parameters and no statistics across chunks, and the extra live state
moves the cliff one rung down:

| `CHUNK` | regs | spill | frame | blocks/SM | GB/s at 8192 blocks |
| --- | --- | --- | --- | --- | --- |
| 128 — the whole band | 255 | 16 B | 1552 | 2 | 317 |
| 64 | 108 | 0 | 272 | 6 | — |
| 32 — softmax's rung | **40** | 0 | 128 | 6 | 3776 |
| **16** | 47 | 0 | **0** | 6 | **5539** |

`CHUNK = 32` is seven registers cheaper and **1.46× slower**, at identical
occupancy. That is the third time the register column has ordered time
backwards in this file, and the clearest: the two rungs differ only in whether
`ptxas` kept the band, and the one that did not is the one with fewer
registers.

| rows × 128 | blocks | before, ms | after, ms | before, GB/s | after, GB/s |
| --- | --- | --- | --- | --- | --- |
| 256 | 2 | 0.0324 | 0.0084 | 4.1 | 15.6 |
| 1024 | 8 | 0.0328 | 0.0083 | 16.0 | 63.3 |
| 8192 | 64 | 0.0340 | 0.0084 | 123.3 | 496.5 |
| 65536 | 512 | 0.1206 | 0.0135 | 278.3 | 2490.7 |
| 524288 | 4096 | 0.8571 | 0.0549 | 313.2 | 4891.3 |
| 1048576 | 8192 | 1.6926 | 0.0969 | 317.2 | **5538.9** |

**17.5×**, every size checked before it was timed, and the *old* kernel checked
against the same reference at all six — so the check is not tuned to the
rewrite. Sizes are twice `softmax`'s rows against one plane rather than two, so
the two tables share block counts and bytes moved and can be read against each
other.

Two things this corrects above. The flat floor at the small end is **8.4 µs**
here, so #47's reading of its own 23 µs as "a fixed cost every launch on this
harness pays" does not survive — whatever that 23 µs is, it is not the launch.
And `softmax` is now the slow kernel in the file by 5.8× at equal bytes and
equal blocks; a one-constant probe puts it at `CHUNK = 16` on a 128-byte frame
and 1185 against 950 GB/s. That probe is **not** shipped — softmax's header
argues for 32 at length, and rewriting it wants its own issue and its own
controls — but it says the mechanism is not peculiar to `layernorm`.

#### the rung was a quarter of it, and the polynomial was the rest — #76

That issue got filed, and the probe was the smaller half. `softmax_rows` at
`CHUNK = 16` is worth 950 → 1178 GB/s and the gap to `layernorm` was 5.8×, so
**the rung had to be measured and then set aside**, which is the part of #76
worth carrying forward: a one-constant fix that closes a fifth of a gap is a
fix and not an explanation.

The rest was found by ablation — the same kernel, the same launch geometry, one
thing removed at a time, every row at `CHUNK = 16` and 8192 blocks in one
session. The last two rows do not compute a softmax and exist to be timed:

| at `CHUNK = 16` | regs | frame | GB/s | what it removes |
| --- | --- | --- | --- | --- |
| `exp2` polynomial, `div_row` | 50 | 128 | 1178 | — |
| `exp2_hw`, `div_row` | 48 | 0 | 3153 | the FMA polynomial |
| **`exp2_hw`, `recip` + `mul_row`** | **32** | **0** | **4414** | 124 of 128 divides |
| no transcendental, no divide | 32 | 0 | 6042 | *both* `exp2` calls |
| one pass, `load_tile` → `store_tile` | 31 | 0 | 6324 | the two extra passes |

Three findings, and the register column orders none of them:

- **[`exp2_approx`] is 2.7× on this kernel.** A clamp, a shift-trick split and
  a degree-3 minimax polynomial, evaluated twice per element over 128 elements
  a thread, against one `ex2.approx.f32`. `reg.rs` says "the measurement does
  not favour" the SFU on the evidence that it takes `softmax_probe_128` from
  168 registers to 255 with spill. That is true **of registers**; here it is
  32 registers and a zero frame either way. Fourth time in this repository the
  register column has ordered time backwards (#47, #63, #67, #76).
- **The divide is 1.4×, and #47 measured it at nothing.** See the annotation
  in that section: the control was valid, under a bind that no longer holds.
- **The three passes over shared memory cost 4.5%** — the thing #47 proposed
  to fix, and the one thing that was never worth fixing.

| rows × 128 × 2 planes | blocks | before, GB/s | after, GB/s |
| --- | --- | --- | --- |
| 128 | 2 | 5.2 | 12.0 |
| 512 | 8 | 20.4 | 48.9 |
| 4096 | 64 | 157.9 | 354.2 |
| 32768 | 512 | 632.4 | 2004.9 |
| 262144 | 4096 | 917.5 | 3880.0 |
| 524288 | 8192 | 950.3 | **4389.6** |

**4.6×**, 66 registers on a 256-byte frame → **32 on a zero frame**, occupancy
unmoved at 6 blocks and 24 warps, every size checked before it was timed and
`layernorm_rows` untouched beside it at 5483 → 5497 GB/s as the control. The
5.8× is now 1.25×, and that residual is not a defect: it is the two
`ex2.approx` this kernel owes its definition and `layernorm` does not, priced
at exactly that in the ablation's fourth row.

The floor, finally: 25.4 µs → **10.9 µs** at two blocks, against the copy's
7.0 µs. #47 read its 23 µs as the launch and #67 disproved that with
`layernorm`'s 8.4 µs; the ablation says what it actually was.

Two forms measured and **not** shipped, both faster:
`exp2(x - peak - log2(total))` folds the normalizer into the exponent and needs
no divide at all — 4390/4415/4462 GB/s over three rounds against `recip`'s
4363/4414, inside each other's spread, and it would put `log2_approx`'s error
inside the exponent for nothing. Parking the numerator in the tile is one
`exp2` per element and **4670 GB/s**, the fastest correct form here, and it
takes the check's worst relative error from 1.97e-3 to **3.44e-3 against a
3.91e-3 tolerance**. 5.8% for 12% of the headroom is the wrong trade, and it
turns #47's prose worry about double rounding into a number.

The positive controls, both built as kernels and launched over the check's
65,536 cells — halving `CHUNK` doubles the chunks a row is walked in, so the
walk's errors are what the check has to see:

| deliberate error | predicted | device |
| --- | --- | --- |
| third pass reads the first chunk for every chunk | 56,960 | **56,960** |
| second pass sums the first chunk twice | 65,536 | **65,536** |

The first is exact only because the model accounts for two things a table
would have missed: the third pass reads and writes the *same* tile, so chunk 0
is overwritten before chunks 1..7 re-read it (worth 384 cells), and the
device's bf16 intermediate moves cells across a tolerance an `f64` host model
keeps inside it (the last 64). The second was picked as the *weakest* error a
chunk walk can commit and is not weak: [`softmax::permutation`](../examples/src/softmax.rs)
spreads a chunk across the whole ladder at an odd stride rather than giving it
16 adjacent values, so the smallest denominator shift over the check's 512 rows
is 4.3% — eleven times the tolerance.

[`exp2_approx`]: ../src/reg.rs

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

And one thing it moved on the way. `WARPS` was briefly forced to a multiple of
four: `SharedVec::BOX_OK` wants a whole number of the TMA's 16-byte lines, so
four fp32 — 16 bytes exactly, as predicted, the minimum legal box — was the
narrowest scratch that would *construct*, and a 1- or 2-warp block could not
take the reduction at all. But that rule is a statement about a **TMA box**, and
a block reduction's scratch is the one use of `SharedVec` that never becomes
one: it is written by `set`, read by `get`, and never handed to a descriptor. So
`BOX_OK` and `LENGTH_OK` moved off `from_raw` and onto the four transfer
methods, where they still fire at codegen for every caller they genuinely bind —
verified by hand, and the error names the instantiation:
`SharedVec::<Bf16, 4>::BOX_OK` failed *while instantiating*
`SharedVec::<Bf16, 4>::tma_store`. Nothing was relying on the construction-time
placement; the host side already rejects the same shapes when a tensor map is
built, with a better message, in `check_driver_requirements`.

The lesson is narrower than the fix. An assert on a constructor reads as a
statement about the type, and this one was a statement about four of its
methods — `SharedVec` was saying "I am a thing the TMA can deliver" when what it
is, is a run of elements that *may* be delivered.

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
cluster-scope entries); **1, 2, 3, 4, 5 and 6 have since shipped**. The numbers are noted
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

#### 6. ~~No cluster-scope semaphore arrival~~ (#24) — **closed by #50**

A 2-CTA MMA consumes four tiles staged by two CTAs, so the issuer needs one
barrier that says *the whole stage is present* — the peer's TMA has to complete
on the leader's copy. `Semaphore` was CTA-scoped by construction and said
nothing about it, and the GEMM got there through a multicast load with a
degenerate mask.

Shipped as `Semaphore::at_rank(rank)`, which `mapa`s the barrier into a peer
CTA and returns a `ClusterSemaphore` — a barrier that may be arrived at or
handed to an engine, and may not be waited on or charged. The byte accounting
question is answered rather than dodged, and the answer was forced:
`mbarrier.arrive.expect_tx` is `.shared::cta` and
`mbarrier.arrive … .shared::cluster` carries no transaction count, so a CTA can
charge exactly one barrier, its own. **The waiter therefore charges the whole
stage** — its own charge times the number of ranks, since a cluster stage is
symmetric by construction. `sync.rs`'s module docs carry the argument, including
why the per-rank alternative that would have kept the sum local (and pleased
#29) is not worth a barrier and a spinning thread per rank per stage.

The locality #29 wanted came back anyway, and without the second barrier: under
#29 each rank derives its own half of the stage from the loads it issued, and
the leader scales its half by `RANKS`. That is the same `RANKS *` this section
promised, now with nothing hand-written on either side of it.

The load half is `SharedTile::tma_load_2d_arriving_at`, which owns the
own-bit mask that used to sit open-coded in `gemm.rs`;
`tma_load_2d_multicast_cg2` keeps the mask and is now the replication form
alone (#49).

#### 7. ~~`pipeline::run` cannot schedule a cluster~~ (#51) — **closed, and the GEMM runs on it**

The work-item map was `blockIdx.x` strided by `gridDim.x`, so the two CTAs of a
cluster got *different* items. `prototype::lcf` predates clusters in
ThunderKittens too, so this was not a porting oversight; the scaffold needed a
way to say who a work item belongs to, and `blockIdx.x` was not it.

Shipped as `%clusterid` strided by `%nclusterid`, plus `Job::RANKS`. The map is
a strict generalization and not a second scheduler: a launch that declares no
cluster runs clusters of one CTA, where `%clusterid` *is* `%ctaid` and
`%nclusterid` *is* `%nctaid`, so a block-scope job provably gets back the loop
it had rather than a parallel code path. `RANKS` (default 1) says whether the
job's barriers are shared across the pair and so whether the item boundary has
to be `barrier.cluster` rather than `bar.sync` — which is what per-item barrier
re-initialization needs across a cluster, since otherwise a leader re-arms a
barrier its peer is still filling. The barrier *count* is unchanged.

This entry used to name #3's `Scope` as the missing piece, and #3 shipped a
block reduction and no `Scope`. It turns out nothing was owed: the rank half is
`cluster::block_rank()`, a `%cluster_ctarank` read with no storage, no
collective and no barrier in it — precisely what #6 found `Scope` *could* not
supply for the block-reduction half. The two halves separate cleanly, and
wrapping a special register in a trait would add a name without adding a fact.
No rank API was added.

`gemm.rs` is now a `Tile: Job` — one output tile per item, `RANKS = 2`, TMEM
allocated once outside the loop. It is the first caller this scaffold has ever
had, which is most of what it buys: `pipeline::run` was untested by anything
before, and is now on the `::examples` gate.

##### The cap is the whole performance story, and two guesses at it were wrong

The persistent grid runs `MAX_CLUSTERS` pairs and walks the rest as items. That
number is the only thing the port changes — `lcf` at one item per cluster *is*
the launch-per-tile kernel, same rings, same barriers, drained at the same
points — and getting it wrong costs a factor of two.

Min ms over 30 timed launches, B200, 148 SMs. The cap is in clusters; a cluster
is 2 CTAs. **Rows marked `=` have fewer tiles than the cap, so the grid is
byte-identical to the launch-per-tile grid and each cluster runs exactly one
item — a free control on the scaffold's own cost.** Two independent
launch-per-tile runs are shown because the cross-run spread is a few percent and
the conclusion has to survive it.

| shape | per tile (run 1) | per tile (run 4) | cap 74 | cap 148 | **cap 222** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x128x256 | 0.0224 | 0.0239 | `=` 0.0223 | `=` 0.0234 | `=` 0.0231 |
| 512x256x256 | 0.0225 | 0.0239 | `=` 0.0225 | `=` 0.0251 | `=` 0.0258 |
| 1024x1024x1024 | 0.0283 | 0.0285 | `=` 0.0268 | `=` 0.0284 | `=` 0.0292 |
| 2048x2048x2048 | 0.0451 | 0.0481 | 0.0593 | `=` 0.0478 | `=` 0.0492 |
| 4096x4096x4096 | 0.1891 | 0.1900 | 0.2808 | 0.2059 | 0.1944 |
| 8192x8192x8192 | **1.0169** | **1.0204** | **2.1036** | **1.2130** | **1.0217** |

TFLOP/s at 8192³: **1081 / 1077** launch per tile, **523** at cap 74, **906** at
cap 148, **1076** at cap 222.

At the last cap it is a dead heat — 0.13% at 8192³, and 4096³'s 2.3% is exactly
the offset that run's own `=` control rows carry. It is **not faster**. What the
sweep bought is the reason it is not slower, and that reason is not in the
scaffold.

##### The residency, measured on a clock because nothing could be asked

Picking the cap needs to know how many CTAs of this kernel an SM holds, and the
repo cannot query it: `cuOccupancyMaxActiveBlocksPerMultiprocessor` takes a
block shape and no cluster, which is why `main.rs` prints `cluster` in the
GEMM's occupancy row — the honest absence #70 built that column to preserve.

So it came off the clock by bisection, and the answer is **3 CTAs per SM**,
which is what 228 KiB / 72 KiB of shared memory admits. The route matters as
much as the number:

- **Cap 74 (1 CTA/SM) costs 2.07×.** That alone refutes extrapolating #77's 1
  CTA/SM here — if the device could only ever hold 148 CTAs, a 148-CTA grid and
  a 4096-CTA grid would *be* the same schedule and could not differ at all.
- **Cap 148 (2/SM) recovers most of it and not all**, 973 → 1688 tiles/ms:
  sublinear, still climbing.
- **Cap 222 (3/SM) draws level and the curve stops.**

This is the evidence **#78** asked for in its step one, which is whether the
GEMM is pinned at 1 CTA/SM like `flash_forward`. It is not, by a factor of
three, and the method is worth as much as the result: for a `#[cluster_launch]`
kernel a timing sweep over the persistent grid's cap is a residency probe.

##### and the 3 has a mechanism, from a second instrument (#78)

Two things above have since been sharpened rather than corrected. The sweep was
**not** the only probe available: `cuOccupancyMaxActiveClusters` takes the
cluster shape the block query has no argument for, and `device-tests`'
`tmem residency census` counts CTAs outright — every CTA records its `%smid` and
timestamps both ends of its TMEM allocation off `%globaltimer`, and the host
sweeps those intervals for the most ever open at once on one SM.

Run at this kernel's own envelope — `cta_group::2`, 128 columns, 73792 B of
shared — the census counts **3**, agreeing with the clock exactly. And it says
which resource that 3 is: the 128 columns admit **4** CTAs an SM on their own,
73792 B of shared memory admits **3**, and the smaller binds. **Shared memory is
what caps this kernel and tensor memory is not**, which is what the 228 KiB /
72 KiB arithmetic above guessed and now has a direct measurement behind it.

The census also priced `alloc_cluster` against `alloc_block` at equal column
counts and found them **identical** — 4 CTAs an SM at 128 columns either way, 1
at 512. A `cta_group::2` allocation is not split across the pair; each rank pays
its full count against its own SM. So nothing about this kernel's residency is a
cluster effect, and #78's leading hypothesis for why the GEMM and
`flash_forward` differ is refuted. What actually differs is their shared plans:
73792 B here against `flash_forward`'s 147536 B, which admits one.

##### What is still on the table

`lcf` folds the epilogue into the item, so `store_rows` and the next tile's
first K loads are separated by a boundary that drains the pipeline. A dead heat
is therefore the *ceiling* for this shape, not a disappointment: the persistent
grid saves launches and the drain gives it straight back. **#15's `lcsf`** is
what would let the two cross, and `tma_store_wait_read` (#9, landed) is the wait
it needs — it releases the shared buffer as soon as the engine has read it,
without blocking on global visibility.

> **Both halves of that were tried and neither is worth anything here.** The
> crossing costs no wait and no shared buffer at all — the accumulator survives
> the boundary in tensor memory — and letting the two cross measures −5.4% to
> +1.2% over two sessions. `tma_store_wait_read` would decouple only the global
> write, which a probe prices at 0–1.2%. See the `lcsf` section at the end of
> §7.

##### and the drain is roughly a third of 8192³, priced — #86

The paragraph above was written from the shape of the code. It is right, and it
is the largest single term: **the item boundary costs 28–37 µs per output tile,
which is 27–36% of the 8192³ launch** depending on how the fit is taken — see
the point-selection table below, which is where all the honest uncertainty in
this section lives. Deepen `K` so the
boundaries amortize and the same kernel, unchanged in every constant, does
**1369.9 TFLOP/s — 60.9% of dense peak** — against 1069.6 at 8192³.

#86 asked three things: run 16384³ against a **~1171 TFLOP/s** prediction from
wave quantization alone, name what binds the kernel with a measurement that
would have moved had it been something else, and say what the flat ~22–28 µs
floor at the small end is. Four tables answer them, every row through the same
verify-then-time path, all in one session on one B200 (driver 580.95.05,
1965 MHz, 148 SMs), with a second session quoted wherever the spread matters.

**1. 16384³ lands on the prediction, and the curve has flattened.** Wave
efficiency is `tiles / (222 · waves)`; the last column is the measured rate
divided by it, which is what the prediction claims is constant.

| shape | tiles | waves | wave eff. | min ms | TFLOP/s | ÷ efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048³ | 128 | 1 | 57.7% | 0.0472 | 363.7 | 631 |
| 4096³ | 512 | 3 | 76.9% | 0.1975 | 695.9 | 905 |
| 8192³ | 2048 | 10 | 92.3% | 1.0280 | 1069.6 | 1159 |
| **16384³** | 8192 | 37 | **99.7%** | **7.4638** | **1178.5** | **1182** |

**Confirmed, with the caveat spelled out.** 1182 against a predicted 1171 is
inside the noise, and the climb — 631, 905, 1159 — stops. But 16384³ is the
least reproducible row in this file: a second full run of the same tree gives
**7.8727 ms and 1117.3 TFLOP/s**, 5.5% away, where 8192³ reproduces to 0.02%
(1069.8 against 1069.6) and every aspect-ratio row below to under 1%. Its
median-to-minimum gap is 11% against 1% elsewhere. So the honest statement is
**1117–1179 across two runs, straddling the prediction**, and anyone reading a
single figure off this row should know it moves.

What that settles is narrower than it looks. Ragged waves *were* the story from
2048³ to 8192³ — 42% of 2048³ is empty cluster slots — and by 8192³ only 7.7%
is left, which is why 4× the work buys ~5–10%. It does **not** follow that 1171
is the kernel's ceiling, and the next table is the direct refutation.

**2. It is not a ceiling. Hold everything but the reduction depth and the same
kernel climbs past it.** `M` and `N` are 8192 in every row, so all four have the
same 2048 tiles, the same 10 waves, the same 92.3% wave efficiency, the same
grid of 444 blocks, the same 268 MB of `C` written, and the same operand bytes
*per flop* — arithmetic intensity is a property of the `[256, 128]` pair tile,
not of the shape. Only the arithmetic between one item boundary and the next
moves.

| K | K blocks per item | min ms | TFLOP/s | % of peak |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 0.4497 | 152.8 | 6.8% |
| 2048 | 32 | 0.5404 | 508.6 | 22.6% |
| 8192 | 128 | 1.0237 | 1074.0 | 47.7% |
| **32768** | 512 | 3.2105 | **1369.9** | **60.9%** |

Milliseconds against `K` is a line whose intercept is a cost that does not scale
with the arithmetic. Dividing that intercept by 10 — the items on the critical
path, since 50 clusters walk 10 and 172 walk 9 — gives a cost per output tile.
**Which points the fit uses moves the answer, so all of them are shown**, least
squares on the raw minimum milliseconds rather than on the rounded rates above:

| points | steady state | fixed | per tile | share at K=8192 |
| --- | ---: | ---: | ---: | ---: |
| K = 8192, 32768 | 1508 | 0.2948 ms | 29.5 µs | 28.8% |
| K = 2048, 8192, 32768 | 1534 | 0.3370 ms | 33.7 µs | 32.9% |
| all four | 1552 | 0.3654 ms | 36.5 µs | 35.7% |

The other session runs 2–4% under all nine of those. So across both sessions and
every selection: **the boundary is 28–37 µs per output tile and 27–36% of the
8192³ launch, and the steady state is 1480–1550 TFLOP/s.** Round to that; the
5.5% cross-run spread on 16384³ is what this benchmark's precision looks like.

**The two deepest points are the preferred fit, and the reason is not the one
that first suggests itself.** A two-point fit has no residuals, so it checks
nothing, and the obvious defence — that `K = 512` is 8 blocks against
`STAGES = 3` and the pipeline barely fills — **is wrong, and the residuals say
so**. Under the four-point fit they run `+40.0, −2.1, −50.0, +12.0 µs`, and the
marginal cost of a K block between adjacent rows is **0.378, 0.503,
0.569 µs** — reproduced as `+34.1, −1.1, −43.3, +10.4` and `0.419, 0.522, 0.580`
in the other session. A pipeline that fails to fill makes *shallow* loops dear
per block. These get dearer as they deepen. The curve is convex, and a line
through convex data has an intercept above the deep-end tangent, which is
exactly the ordering in the table.

**What is convex about it is a variable this sweep does not hold fixed, and that
is a limitation of the experiment rather than a subtlety.** `K` sets the operand
footprint: `A` and `B` are 8 MiB each at `K = 512` and **512 MiB each at
`K = 32768`**, so the sweep crosses L2 somewhere between the second and third
rows. The shallow rows are cheap per K block partly because the whole of both
operands is L2-resident for the entire launch, which the deep rows cannot be. So
the deepest two points are preferred because `K = 8192` is the regime in
question and they straddle it in the same cache regime — not because the shallow
rows fill badly. They also give the *smallest* fixed cost of any selection, so
the preferred number is the conservative one.

The part that needs no fit at all is the measured **1369.9, sitting 28% above
8192³ and 17% above the 1171** that was offered as where the curve would stop.

Nothing else in this sweep can be that intercept. It is not launch overhead
(#67, and `layernorm` reaches 9.2 µs on this harness in this same run), not
ragged waves (92.3% in all four rows), not occupancy (3 CTAs/SM, #83 and #84,
and `K` does not touch the shared plan), and not `C` traffic (268 MB in all
four).

**What it is, structurally:** at a boundary the pair drains. `done.wait`, then
`sync_threads`, then LDTM of a `[32, 128]` fp32 band into 128 registers a
thread, then `store_rows` — and `store_rows` is one `st.global.f32` per owned
value, 128 of them a thread. Under `BaseLdtm` a single one of those stores has
the warp's 32 lanes writing **eight different rows** of `C`, four four-byte
words to each, which is 8 sector transactions for 128 useful bytes. Only then
does `pipeline::run` re-arm the barriers and the next item issue its first TMA
loads, and only then, a full memory latency later, can the MMA start. Nothing in
that chain overlaps with anything. That is the shape `lcsf` changes; the
scattered fp32 store is a second, separable thing and neither is in this issue's
scope.

**3. The small-size floor is one item boundary.** 256x128x256 is a single
cluster running exactly one item, and it costs **23.4 µs** against `layernorm`'s
9.2 µs on the same harness in the same run. 512x256x256 is four clusters running
one item each: 23.3 µs. It is the same ~30 µs the depth sweep priced, at 1/222
of the contention — not a launch cost, and not a floor peculiar to small
problems, but the per-tile cost showing up undivided because there is only one
tile to divide it by. The check that makes this more than a coincidence of
magnitudes is 2048³: 128 clusters at one item each, `K = 2048`, **47.2 µs**,
against the depth sweep's 54.0 µs per item at the same `K` with 222 clusters
contending. Two shapes that share nothing but their reduction depth land within
13% of each other on a per-item basis.

**4. The traversal is worth 23%, and #86's L2 arithmetic is optimistic.** The
issue reads 12.7 TB/s of operand traffic as "~48× reuse over 268 MB of unique
input". That is the *aggregate* ratio over a whole launch. What L2 can actually
capture is the reuse available inside one wave, and that is a property of the
aspect ratio: the 222 clusters of a wave sit on `ceil(222 / tiles_n)` rows of
tiles, span `min(222 · 128, N)` columns, and march through `K` together, so at
one `K` block they collectively touch `(rows_of_A + columns_of_B) · 128` bytes.
Five shapes, every one of them 2048 tiles, 10 waves, 92.3% efficiency, 444
blocks, 1.1 TFLOP, 12.9 GB of operands requested and 268 MB of `C` written —
identical runs but for `M : N`:

| shape | tiles_n | distinct per K block | reuse | min ms | TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32768 x 2048 x 8192 | 16 | 0.72 MB | 15.1× | 0.9785 | **1123.7** |
| 16384 x 4096 x 8192 | 32 | 0.75 MB | 14.5× | 1.0005 | 1099.0 |
| 8192 x 8192 x 8192 | 64 | 1.18 MB | 9.2× | 1.0246 | 1073.1 |
| 4096 x 16384 x 8192 | 128 | 2.16 MB | 5.0× | 1.0710 | 1026.6 |
| 2048 x 32768 x 8192 | 256 | 3.67 MB | 3.0× | 1.2007 | **915.7** |

**23%, monotone, on nothing but the walk** — and reproduced row for row in the
other session (1112.6 / 1093.4 / 1066.4 / 1014.4 / 910.4, every row within 1%).
The first and last are transposes of each other with identical 570 MB total
footprints, so this is not about how much data there is; it is about which
operand a row-major item map re-reads. The reuse column is what the map leaves
L2 to capture — **3× to 15×, never the 48× the aggregate suggests** — and the
throughput follows it. That is #89, and this is the number it is worth.

It is **not** HBM saturation. `layernorm` moves 5497 GB/s of real streaming
traffic on this device in this same run, and the worst row above is *inferred*
to pull ~3.8 TB/s against that. Nor is the response a cliff, which is what
saturation looks like; it is proportional, which is what an exposed miss latency
looks like against a pipeline `STAGES = 3` deep. Note also that both the
`distinct` and `reuse` columns are arithmetic on the item map rather than
counters — see below.

**Subtracting the boundary gives the steady-state rate per shape**, which is
the number a traversal change would move. Taking the intercept off each row,
over the range of intercepts the point selections give and both sessions:

| shape | steady TFLOP/s |
| --- | ---: |
| 32768 x 2048 x 8192 | **1540 – 1790** |
| 8192 x 8192 x 8192 | 1500 – 1670 |
| 2048 x 32768 x 8192 | 1210 – 1320 |

So **this tile shape, with its boundary overlapped and its walk L2-aware, is
worth roughly 1550–1800 TFLOP/s — 69% to 80% of dense bf16 peak — with no change
to the tile, the stages or the traversal granularity.** The range is wide
because the two uncertainties compound in the same direction: a larger fixed
cost implies a larger recovered rate.

**It is a derivation, not a measurement, and it stacks two assumptions** — that
the line is the right model, which its own residuals say it is only roughly, and
that the intercept is shape-independent, which it should be since all three rows
write the same `C` over the same item count. **Nothing has run at any of these
rates.** The fastest this kernel has been measured at is the 1369.9 TFLOP/s in
the depth table, and that is the number to hold anyone to. What the derivation
is for is ordering the levers, and it is the reason #87's tile sweep is the
*last* one here rather than the first.

**What could not be measured, and it matters which.** The reuse and traffic
columns above are *arithmetic on the item map*, not counters. Nsight Compute
2025.3.0 is in the image and refuses to start: `RmProfilingAdminOnly` is 0, `ncu`
knows `gb100`, `libnvperf_target.so` is installed, and the injected driver set
has no `libnvidia-pcc.so` — the performance-counter library — which is not in
the 580.95.05 package either. So there is no measured L2 hit rate or DRAM byte
count here, and `modal_app.py::profile` is checked in with that written down. It
is worth being plain that the substitute is not strictly weaker: a counter
attributes, an intervention that holds nine things fixed and moves one
establishes cause, and the three tables above are all interventions.

**The bind at 8192³, in order.** **27–36% is the item boundary**, and `lcsf`
(#15) is what would let the epilogue and the next tile's first loads cross.
**6–7%** more is operand traffic this square shape's row-major walk does not
leave in L2 — **23% at the worst aspect ratio, and that one is measured
end to end rather than fitted** — and an L2-aware order (#89) is what would
recover it. Under both sits a steady state of roughly **1550–1800 TFLOP/s** at
this tile's fixed 85.3 FLOP/byte, and the only lever on *that* is the tile,
which is #87.

> **The second of those has since been recovered, and the estimate here was
> conservative.** #89 replaced the row-major item map with a grouped one and
> measured **8.4%** at the square shape against the 6–7% predicted, and **27.6%**
> at the worst aspect ratio against 23%. See "and the walk was the 23%,
> recovered" below.

The order is the finding. #86 framed the question as bandwidth versus latency on
the operand stream and asked which of HBM and L2 to attack; the answer is that
**the largest term is neither** — it is an epilogue that the pipeline stops for,
on a kernel whose steady state is already at 67% of dense peak. The tile sweep
that was queued behind this answer should stay queued behind #15 and #89 too.

> #89 has since landed, and it moved the kernel this sweep was fitted on. **The
> fit below should be read as describing the pre-#89 kernel**, and `gemm-depth`
> re-run on the post-#89 tree is what re-establishes the boundary's share and the
> steady state. The ordering of #15 and #87 depends on that re-fit and not on the
> numbers as they stand here.

##### and a third of that boundary was one instruction — #91

The section above priced the boundary and named its parts: the accumulator
drain, `LDTM`, and `store_rows`' scattered scalar fp32 stores. The third part
was the cheapest to remove.

`BaseLdtm` puts a thread's four values of a 16-column block at column offsets
`{0, 1, 8, 9}` — **two adjacent pairs**. So every even/odd value pair is one
`st.global.v2.f32`, and a `[128, 128]` accumulator band drops from 512 scalar
stores a lane to 256 paired ones. The pairing is stated on the layout
(`ColLayout::CONTIGUOUS_VALUES`, default `1`) rather than read off `BaseLdtm`'s
arithmetic in the mover, because #23 opened the shape set and a layout written
later must not silently inherit stores that assume this formula.

**The prediction this makes is specific, and it is what makes the result
evidence rather than a throughput delta:** if the change hit the fixed per-tile
cost and nothing else, then in the depth sweep's fit **the intercept moves and
the slope does not.** Refitted by least squares on raw minimum milliseconds,
over the same three point selections §7 reports, both trees measured in the same
session pair on one B200:

| points | fixed ms before → after | per tile µs | steady TFLOP/s before → after |
| --- | ---: | ---: | ---: |
| K = 8192, 32768 | 0.2778 → **0.1783** | 27.8 → **17.8** | 1481 → 1515 |
| K = 2048, 8192, 32768 | 0.3299 → **0.1845** | 33.0 → **18.4** | 1511 → 1519 |
| all four | 0.3598 → **0.1981** | 36.0 → **19.8** | 1530 → **1527** |

**The intercept falls 36–45%; the slope moves −0.2% on the four-point fit.**
The boundary's share of the 8192³ launch goes from 27–35% to **20–22%**.

Two things corroborate it without any fit at all. The measured rows fall in
exactly the order a fixed cost predicts — the shallower the reduction, the more
of it the boundary was:

| shape | before ms | after ms | before TFLOP/s | after TFLOP/s | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x128x256 | 0.0238 | 0.0171 | — | — | −28% time |
| 1024³ | 0.0280 | 0.0219 | 76.6 | 98.0 | +28% |
| 2048³ | 0.0470 | 0.0371 | 365.5 | 463.6 | +27% |
| 4096³ | 0.1919 | 0.1534 | 716.3 | 895.9 | +25% |
| 8192³ | 1.0214 | 0.8987 | 1076.4 | **1223.4** | +14% |
| 16384³ | 7.1955 | 6.8098 | 1222.4 | **1291.7** | +6% |

And §7's item 3 — the small-size floor *is* one item boundary — moves with it:
256x128x256 is **23.8 µs before and 17.1 µs after**, a 6.7 µs drop against the
10 µs the fit says a tile's boundary lost. One cluster running one item, with
nothing to amortize and no fit involved.

Why it is worth more than the halved instruction count suggests: under
`BaseLdtm` a single scalar store has a warp's 32 lanes writing eight rows of
`C`, four four-byte words each at columns `0, 2, 4, 6` — **8 sectors carrying 16
useful bytes each.** The pair store touches *the same eight sectors* and fills
them. So it halves the instructions and doubles the sector utilization, which is
why the effect is bigger than "half as many stores" would buy.

**The register column went the other way, and did not decide anything.**
`gemm_cg2` goes from **40 to 167 registers, with zero spill and an unchanged
528-byte frame**, which still admits 3 CTAs/SM — 65536 registers an SM over 128
threads and 3 CTAs leaves 170. That is the fifth time here the register count
has ordered time backwards (#47, #63, #67, #76). The obvious cause — the inline
asm's `clobber("memory")` blocking LLVM from sinking the LDTM drain into the
store loop, which is #63's "register cost is liveness" — **was measured and
refuted**: deleting both clobbers gives byte-identical counts. What in the asm
lowering costs the registers is not established. Worth recording because the
headroom is now thin: 167 against 170, with shared memory already capping
residency at 3, so both resources are at the limit.

**The alternative was priced and not built.** TMEM → shared → TMA store (#9) is
the route `softmax` already uses, and `tma_store_wait_read` releases the shared
buffer as soon as the engine has read it, which is what an overlapping epilogue
needs. It costs shared memory this kernel does not have. The residency census in
`device-tests` measures **3 CTAs/SM at 73792 B a CTA against 233472 B an SM** —
221376 used, leaving **4032 B a CTA** of headroom. One warp's `[32, 128]` fp32
band is 16384 B and the whole tile is 65536 B, so the staging buffer takes
residency to **2 CTAs/SM, or 1**. Against a steady state already at 67% of dense
peak, cutting concurrency by a third to recover a term now worth 20–22% is a
losing trade — and it was already close *before* this change, at 27–36%.

**The premise under that last sentence has since been measured, and it is
wrong.** "Cutting concurrency by a third" is a third of the *residency*, and a
third of the residency turns out to be 14–16% of the throughput, not 33% — see
#98 below, which prices the occupancy step on a dead allocation. The trade is
not a losing one on those grounds. It is decided instead by how much of the
boundary `lcsf` hides, and the threshold is about 70%.

The one version that could pay reuses an existing ring slot (an `ARing` stage is
16384 B, exactly one band) so the plan does not grow. That serializes the next
item's first loads against the store unless the schedule changes with it, which
makes it a scheduling change entangled with #15 and #88 rather than an epilogue
change. It also needs plumbing that does not exist: `TensorMapElement` is
`Bf16`-only, there is no fp32 `SharedTile` swizzle, and `stmatrix` is b16, so
getting fp32 registers into a swizzled shared tile is plain indexed stores plus
a proxy fence. Its own issue, queued behind #15 and #89 — not this one.
##### and the denominator says ~69% of cuBLASLt where the ratio is stable — #92

Everything above divides by a spec sheet. `bench` now divides by a **tuned
library on the same device, at the same shape, in the same process**, and at the
two largest sizes this kernel runs at **0.69 of cuBLASLt**, reproducing to about
one part in a hundred. Below 4096³ the comparison is not stable enough to quote,
for a reason that is itself the second finding here.

The baseline is `extern "C"` from the examples crate, not upstream's separate C
binary, and that choice is measurement rather than tidiness: it goes through the
same `bench::time`, the same stream, the same five discarded warm-ups and the
same minimum of thirty, so what is left different is the code between the two
CUDA events. Every row was compared element by element against the **same CPU
reference** — `gemm::check_c`, the function and not a copy of it — before it was
timed, because the way a baseline goes wrong is by computing a *different* GEMM
quickly.

Two complete sweeps of this tree, in separate sessions on one B200 (driver
580.95.05), both after #94:

| shape | ours min ms | theirs min ms | ours TFLOP/s | theirs TFLOP/s | ours/theirs | at median | run 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x128x256 | 0.0169 | 0.0127 | 1.0 | 1.3 | 0.751 | 0.669 | 0.454 |
| 512x256x256 | 0.0184 | 0.0110 | 3.7 | 6.1 | 0.598 | 0.653 | 0.484 |
| 1024³ | 0.0230 | 0.0134 | 93.3 | 160.5 | 0.581 | 0.590 | 0.480 |
| 2048³ | 0.0376 | 0.0236 | 457.3 | 729.4 | 0.627 | 0.623 | 0.564 |
| 4096³ | 0.1532 | 0.0878 | 896.8 | 1565.8 | 0.573 | 0.585 | 0.574 |
| 8192³ | 0.9032 | 0.6220 | 1217.4 | **1767.8** | **0.689** | 0.687 | 0.686 |
| 16384³ | 6.8477 | 4.7511 | 1284.5 | **1851.4** | **0.694** | 0.696 | 0.685 |

bf16 operands, fp32 `C`, `CUBLAS_COMPUTE_32F`, cuBLASLt 130000. `bench` prints
the heuristic's pick per row so the baseline is reproducible: algorithm `id=66`
at every size in both sessions, no split-K, and **no workspace used at all**
despite being offered 32 MiB.

**A ratio is *less* reproducible than either number in it, which is the opposite
of the intuition.** The instinct — and it was the stated reason for running
twice — is that run-to-run drift moves both sides together and cancels. It does
not, because here the variance is almost entirely on the *baseline* side, so the
ratio inherits all of it and adds our own on top. The split is sharp and it is
by size:

| shape | ours, run to run | theirs, run to run | ratio, run to run |
| --- | ---: | ---: | ---: |
| 256x128x256 | 3% | **77%** | 0.751 → 0.454 |
| 512x256x256 | 11% | **41%** | 0.598 → 0.484 |
| 1024³ | 10% | **33%** | 0.581 → 0.480 |
| 2048³ | 5% | **17%** | 0.627 → 0.564 |
| 4096³ | 0.6% | 0.3% | 0.573 → 0.574 |
| 8192³ | 1.5% | 2.0% | 0.689 → 0.686 |
| 16384³ | 0.9% | 2.2% | 0.694 → 0.685 |

So **only the bottom three rows carry a quotable ratio**: 0.57 at 4096³ and 0.69
at both sizes above it, each stable to about 1%. The top four are dominated by
cuBLASLt's own variance — its 256x128x256 figure nearly doubles between sessions
— and `0.751` in the table above is noise, not this kernel's best row. Anyone
reading a ratio here to three digits should stop at two, and below 4096³ should
not read one at all. This is why the min *and* the median are both printed, and
it is the entire justification for the second sweep.

**cuBLASLt reaches 1767.8–1892.4 TFLOP/s — 79% to 84% of dense bf16 peak — and
that is the calibration this project did not have.** It is an independent,
measured check on the derivation two sections up, which subtracted the fitted
item-boundary cost and claimed this tile "boundary overlapped and walk L2-aware"
would be worth roughly **1550–1800 TFLOP/s**, with the honest caveat that it
stacked two assumptions and that nothing had ever run at those rates. Something
has now, on this part: the band was **not optimistic — if anything conservative**.

**It also prices the target, and the target deserves re-reading.** #86 states
the goal as **1.7–1.9 PFLOP/s at bf16 before precision changes**. cuBLASLt
measures **1.77–1.89 PFLOP/s** — *inside that band*. So the written target is
not "most of the way to the library"; it is **level with the library**, and
reaching it means matching a vendor GEMM rather than approaching one. That is a
materially harder goal than "48% of peak → 80% of peak" sounds, and it is worth
knowing before anyone plans around it. A target above ~1.9 PFLOP/s — 2.0 and up
has been mentioned — is **above everything cuBLASLt achieved here**, so it is
not a matter of closing a known gap; it would require beating the vendor on its
own part, and nothing measured so far suggests where that would come from.

**The ratio improves with size, and the terms already named predict it.** 0.573
at 4096³ against 0.689 and 0.694 above it, reproduced in run 2 (0.574 / 0.686 /
0.685). Ragged waves go 76.9% → 92.3% → 99.7% across those rows and the per-tile
item boundary — a fixed cost, so a `1/K` share of the run — falls by 4×.
cuBLASLt pays neither. The deficit is largest where the two known fixed costs
are largest, which is #86's finding seen from outside.

It does not follow that #15 and #89 close it: at 8192³ they would have to be
worth the whole 1217 → 1768, and the steady state #90 derived for *this tile at
this shape* was 1500–1670, under what the library actually achieves. So there is
plausibly a residue that is the tile itself (#87). **This does not reorder the
levers** — a term worth 27–36% is still the one to take first — and no kernel
constant moved for this comparison; it is a denominator, not a change.

**Where this flatters us, stated plainly.** Both sides read **byte-identical
operands** — a plain packed bf16 `[m, k]` and `[n, k]`, K contiguous. Contrary
to what #92 anticipated, there is *no* rearranged input on our side to discount:
`gemm.rs` does not use `encode_bf16_panels`, and the TMA tensor map it does need
is a descriptor built once outside the clock, exactly as cuBLASLt's layouts and
heuristic are. What is genuinely uneven is **generality**: ours computes this
one form — both operands K-major, `α = 1`, `β = 0`, no epilogue,
`m % 256 == n % 128 == k % 64 == 0` — while cuBLASLt takes any of it and picks
per shape. A like-for-like rate against a library that is also general reads in
our favour, so 0.69 is an **upper bound** on how well this kernel would stand in
for one. cuBLASLt is also handed a 32 MiB workspace (allocated before the
warm-up, outside the clock) that ours does not have and, per the algorithm
lines, does not spend.

The baseline is attached to `gemm` only, not to `gemm-footprint` or
`gemm-depth`: those exist to compare the kernel against *itself* with one
variable moved, and each extra row re-stages its operands on the host, which at
these sizes is the long pole. Whether cuBLASLt also loses 23% to aspect ratio is
a good question and **a one-word change** — `baseline: CUBLASLT` on that case —
for whoever wants it.

**Linking worked**, which #92 asked to have recorded either way. `cargo oxide`
shells out to plain `cargo` and only injects a codegen backend through
`RUSTFLAGS`, so an ordinary `build.rs` emitting
`cargo:rustc-link-lib=dylib=cublasLt` is honoured with nothing special — the
same mechanism `cuda-bindings` upstream uses for the driver. No fallback to a
separate binary was needed. The feature is off by default so that a crate anyone
can build does not require a devel CUDA toolkit to link, and
`examples/cublaslt_abi.c` asserts the hand-transcribed enum values and struct
offsets against the real headers under `-fsyntax-only` in the CPU gate, so a
wrong constant fails there rather than on a B200.

##### and the hardware's own scheduler, which does not pay for itself (#88)

Blackwell has a work-stealing scheduler in silicon. `clusterlaunchcontrol.try_cancel`
launches a full grid — one cluster per output tile — and a cluster that finishes
*cancels* one the scheduler has not launched yet and runs its tile instead.
`pipeline::run_stealing` is that, behind the same `Job` the static stride uses,
and it exists for a reason that is not speed: **the grid becomes the problem's
own tile count, so `SMS` and `CTAS_PER_SM` have nothing left to do.** #84
established that the second of those is measured and underivable, so a persistent
grid picked from it is right for a B200 and cannot be known to be right for
anything else.

The ceiling was stated before it was built, and it is the ragged last wave the
capped grid idles through. Min ms over 30 timed launches, all three schedulers on
**one shared plan** — the static path carries the 24 unused bytes of the work
queue too, so what follows compares schedules and not residencies. `predicted` is
`1 - tiles/(waves x 222)` and is not fitted to anything:

| shape | tiles | waves | wave eff | predicted | static | clc | measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048³ | 128 | 1 | 57.7% | 42.3% | 0.0361 | 0.0362 | −0.3% |
| 4096³ | 512 | 3 | 76.9% | 23.1% | 0.1517 | **0.1497** | **+1.3%** |
| 8192³ | 2048 | 10 | 92.3% | 7.7% | 0.9075 | **0.8974** | **+1.1%** |
| 16384³ | 8192 | 37 | 99.7% | 0.3% | 6.9275 | **6.7643** | **+2.4%** |

**The predicted gain does not arrive.** Where the model says 23% the measurement
says 1.3%; where it says 7.7% it says 1.1%.

The wave-efficiency arithmetic is not wrong, and it is worth being precise about
what it does say. At 4096³ the ragged last wave really is **23% of the grid
idling** — 512 tiles over three waves of 222 leaves 154 clusters with nothing to
do. It is simply not 23% of the *time*, because the clusters that idle are not
the ones on the critical path: the launch ends when the last busy cluster
finishes its tile, and an idle neighbour does not make that cluster faster. The
model is right about a term, and the term is small. #90 and #94 already measured
the one that is not — the **item boundary**, at 18–30 µs per output tile and
20–36% of the launch — and a dynamic schedule removes no item boundary at all.
It only changes which cluster pays them.

**Prefetching the steal is a regression, and #88 predicted the opposite.** The
response arrives on an mbarrier, so the request can be issued before the current
item's MMA and harvested after — the latency hides, and "a steal on the critical
path would be a regression" is what the issue expected. Both forms were built and
timed. The prefetched one was **slower at all four sizes, by 8.5% at 16384³**
(7.3399 ms against the 6.7643 in the table above), which is far outside the 0.6%
this file's own numbers repeat to, and it has been **deleted**: the `clc` column
above is the form that issues its request after the item it will replace.

The plausible mechanism is that prefetching makes every cluster claim its next
tile *before* finishing its current one, so the order tiles are handed out in
stops tracking the order clusters actually become free — which is the tail
problem CLC is for, reintroduced by the optimization meant to hide its latency.
That is a hypothesis with one measurement behind it, not a finding, and what
would test it is recording per-cluster item counts and comparing their spread.

What is left is real but small: the steal is 1.1–2.4% ahead of the static stride
at the three large sizes, which is at the edge of what this file can resolve. **So CLC's case here is portability and not throughput**, and it should
be adopted on that basis or not at all.

`device-tests`' **`clc work stealing`** is what says the mechanism does what the
scheduler assumes, asked of the instruction rather than of a GEMM built on it.
Over a 4096-cluster grid: **2222 clusters ran and 1874 were cancelled and never
launched — 4096 exactly, each cluster once.** Both ranks of every cluster were
told the same thing, and every stolen coordinate was cluster-aligned. Each rank
waits on a deadline rather than on the barrier, so the failure this case exists
to catch reports itself instead of hanging — which matters, because the two
questions it settles both fail silently. It also settles the polarity on its
own evidence: `is_canceled == 1` means *work is available*.

**A correction to what this section originally claimed.** It said
`cuda_device::clc` documents that polarity two ways, with the module-level usage
example inverted against the function's own doc. That was true of a revision and
is not true of ours. Upstream fixed it in `61f4a563` on 2026-06-29 — along with
the flow diagram and all three `clc_query_get_first_ctaid_*` docs, which were
wrong the same way — and then replaced the hand-written module doc with a
generated one, so the pinned `20a56163` ships a thirteen-line `clc.rs` with no
usage example to be wrong. The claim traced to a **stale sibling cargo
checkout**: `~/.cargo/git/checkouts/cuda-oxide-…/4514af2` is v0.2.1 from
2026-06-10, and of the seven checkouts now in that directory it is the only one
still carrying the bug. The file cited as settling it, `crates/mir-lower/…/clc.rs`,
exists at no upstream revision at all.

Nothing here needs changing — `!= 0` was and is right, and the probe above is
why we know it rather than the doc. But the mechanism is worth naming, because
it is #65's failure mode with a new surface: **a `~/.cargo` checkout is not a
citation.** Revisions accumulate there and cargo does not remove the old ones,
so reading one proves nothing about what the build compiles against without
`git log -1` on that specific directory. An upstream claim in this repo should
name the revision it was read at.

##### and one CTA an SM is worth 14–16%, not a third — the premise, measured (#98)

The TMEM → shared → TMA store epilogue (#9) that #91's section priced and did
not build was rejected on an arithmetic: it costs an occupancy
step, a step is `MAX_CLUSTERS` 222 → 148, and *"cutting concurrency by a third
to recover a term now worth 20–22% is a losing trade."* The arithmetic is right.
**A third of the residency is not a third of the throughput**, and nothing in
this repo had asked which. Residency buys latency hiding, and a pipeline that
already covers its own latencies does not need the third CTA. #15 turns on the
difference: at 33% its straight route loses and it becomes a scheduling change,
at 10% it is an epilogue change and worth a lot.

**The measurement is a dead allocation.** `gemm_cg2`'s declared shared plan
grows by bytes that no code reads or writes — 8192 to cross 3 → 2, 49152 to
cross 2 → 1 — and `CTAS_PER_SM` moves with it so the persistent grid asks for
the residency the device will actually give it. The work, the traversal, the
instruction mix and the register count are identical either side: **the diff is
four integer literals and no code at all.** A real staging buffer moves the
epilogue and the residency at once, which is precisely why the paragraph above
could not separate its cost from its benefit and fell back on arithmetic. The
dead tail has all of the cost and none of the benefit, which is the control.

**There is nothing here for a compiler to delete,** which is worth stating
because a silently-eliminated allocation would measure nothing and look like a
null result. The bytes are not an array `ptxas` could see and discard — dynamic
shared memory is a *launch parameter*, written in `#[launch_contract]` and
handed to the driver, so the only thing that could refuse to charge for it is
the driver. It charges, and the census is what says so.

**The step happened, and it was counted rather than queried.** #77's
`cuOccupancyMaxActiveBlocksPerMultiprocessor` answers 1 for anything containing
a `tcgen05.alloc`, so it cannot see this at all — it reads 1 at all three
envelopes. `device-tests`' `tmem residency census` counts instead: every CTA
records its `%smid` and timestamps both ends off `%globaltimer`, and the host
sweeps for the most ever open at once on one SM. Three rungs were added at the
exact envelopes the experiment declares:

| plan | counted holders | resident | driver's query |
| --- | ---: | ---: | ---: |
| 73792 B (#84's rung) | 3 | 3 | 1 |
| **73816 B — the plan today** | **3** | 3 | 1 |
| **82008 B — +8192 dead** | **2** | 2 | 1 |
| **122968 B — +49152 dead** | **1** | 1 | 1 |

`resident` equals `holding` at every one, so no CTA is parked in a blocking
allocator inflating the count, and the census's own `verdict` *asserts* that
holders equal what the per-CTA resources divide to — the case passing is the
assertion. Residency visibly moved. It is also the first count of the plan
`gemm_cg2` actually declares: the 24 bytes the CLC work queue added sit on the
same side of the step as 73792, which the repo had assumed and never checked.

**The prediction, stated before the run:** 10–16% for the first step and a cliff
of 35–45% for the second, reasoning from §7's old cap sweep and from #91 having
made the epilogue cheaper, so there is less left for a third CTA to hide. Both
came out about right, and the ways they did not are worth naming: the first step
overshot the band on raw times at 16384³ (−18.3%, against a predicted ceiling of
16%) and lands at its top edge once drift is divided out, and the second step
was inside its band at 8192³ and 16384³ but well under it at 4096³ (−25.1%).
The cliff was predicted and it is there; its depth is size-dependent in a way
the prediction did not anticipate.

Min ms over 30 timed launches, three separate B200 containers in one session,
every row checked against the CPU reference before it was timed:

| shape | 3 CTAs/SM | 2 CTAs/SM | 1 CTA/SM | TFLOP/s, 3 → 2 → 1 |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1527 | 0.1727 | 0.2347 | 899.8 → 795.8 → 585.5 |
| 8192³ | 0.8965 | 1.0668 | 1.8694 | 1226.5 → 1030.6 → 588.2 |
| 16384³ | 6.7735 | 8.2907 | 13.7760 | 1298.6 → 1061.0 → 638.5 |

**cuBLASLt is the drift control and it moves the answer.** It runs untouched in
the same container at every rung, and it was not identical across them — 1789.5,
1739.5 and 1771.7 TFLOP/s at 8192³, a 2.9% spread — so part of each raw delta is
the session and not the step. Dividing it out gives the ratio to the library,
and the size dependence of the first step largely collapses:

| shape | ours/cuBLASLt at 3 → 2 → 1 | **3 → 2** | **2 → 1** |
| --- | ---: | ---: | ---: |
| 4096³ | 0.595 → 0.505 → 0.378 | **−15.1%** | **−25.1%** |
| 8192³ | 0.685 → 0.592 → 0.332 | **−13.6%** | **−43.9%** |
| 16384³ | 0.703 → 0.590 → 0.343 | **−16.1%** | **−41.9%** |

**One CTA an SM is worth 14–16% of this kernel's throughput at the step that
matters.** It is not worth 33%.

**The two steps are not the same price, and that is the second finding.** The
cost is strongly convex: the third CTA is worth a seventh, the second is worth
25–44%. It is a cliff, not a linear cost, which is what #98 asked for
because the two predict differently for #87 — a tile change spending the same
budget can afford the first step and must not take the second.

The far end says why, and it is a structural fact about this kernel rather than
a benchmark row. **At one CTA per SM the throughput is ~585–640 TFLOP/s at
4096³, 8192³ and 16384³ alike** — flat, where the 3-CTA curve climbs from 900 to
1299 across the same sizes. So a lone CTA sustains about 590 TFLOP/s and
*everything above that is overlap between CTAs*: rather more than half of what
this kernel achieves is one CTA's stall covered by another's work. §7's old cap
sweep got 523 TFLOP/s at cap 74 on a much earlier tree, which is the same
plateau from a different direction.

**The static grid is not what is doing it.** Resizing `MAX_CLUSTERS` with the
residency also changes the ragged last wave — in the *favourable* direction here
(8192³ goes from 92.3% to 98.8% wave efficiency), so if anything this understates
the cost. The `clc` scheduler is the control that has no cap at all: its grid is
the tile count either way and only the device's capacity changed. It agrees —
0.8900 → 1.0729 at 8192³, **−17.0%** against the static path's −16.1%.

This also supersedes §7's cap sweep as an answer to this question. That sweep
shrank the *grid* and left the capacity alone, so it could not distinguish "two
CTAs an SM" from "a grid that happens to place two on most SMs" — and CTAs are
not obliged to distribute evenly. The dead tail makes the third CTA impossible
rather than merely unrequested, and the census is what confirms it.

**What this does to #15.** The item boundary is 20–22% of the 8192³ launch after
#91. If `lcsf` hides a fraction `f` of it and pays one occupancy step costing
`c`, the trade breaks even at `0.21f ≥ c` — so at the measured 13.6–16.1% band,
**`lcsf` has to hide roughly two thirds to three quarters of the item boundary
(f ≈ 0.65–0.76) to be worth an occupancy step.** Hide all of it and 8192³ goes
to about 1330 TFLOP/s and 0.74 of cuBLASLt; hide half and it is a 4% *loss*.

So the verdict is neither the rejection nor a green light. **#94 was wrong about
the price — it is a seventh, not a third — and that is enough to put the
straight route back on the table**, where the arithmetic as written had removed
it. It is not enough to build it on. What decides #15 is now a different and
more answerable question than "what does a CTA cost": it is *how much of the
boundary does `lcsf` actually hide*, and ~70% is the number it has to beat. That
is worth measuring on a prototype before the plumbing (`TensorMapElement` is
`Bf16`-only, no fp32 `SharedTile` swizzle, `stmatrix` is b16) gets built.

> **That prototype has since been built and it hides nothing — `f ≈ 0`, against
> the 0.65 asked for here.** The framing above is right that this is the
> question; two things it assumes are not. There is **no occupancy step to
> clear** any more, because #87 made tensor memory the binding half of the
> residency formula. And the store stage needs **no staging buffer and none of
> the plumbing**, because an undrained accumulator survives the item boundary in
> tensor memory — so `lcsf` is a reordering inside `Job::work` and
> `pipeline::run` is untouched. Measured over two sessions it is −5.4% to +1.2%
> at five sizes, and a probe that holds the epilogue's instructions fixed while
> deleting its HBM traffic puts the write-bound part at 0–1.2%. See "and the
> store stage was free to build and worth nothing" below.

##### and the walk was the 23%, recovered — #89

`pipeline::run` and `run_stealing` answer *which item comes next*. Neither says
what an item **is**, and for a GEMM that second map is the one the memory system
sees. It was `(item / tiles_n, item % tiles_n)` — row-major over the output —
and it is now `pipeline::grouped(item, tiles_m, tiles_n, GROUP)`, which walks
`GROUP` tile-rows down a column before moving right. `GROUP = 1` *is* the
row-major map, so the control below is a parameter value and not a second code
path, and the whole change to the kernel is one line of `Job::work`.

**What made this worth doing was the aspect-ratio sweep above and not #86's
traffic arithmetic.** #86 read 12.7 TB/s of operand traffic as "~48x reuse over
268 MB of unique input"; that is the aggregate over a whole launch, and what L2
can capture is the reuse available inside one *wave*, which the table above
measured at **3x-15x**. The 48x should not be carried anywhere. What should is
that five shapes identical in flops, tiles, waves, grid, `C` bytes and arithmetic
intensity differed by 23% on nothing but `M : N`, with the best and worst rows
exact transposes of each other.

**Predictions, stated before the run.** The best width would be 8 — minimizing a
wave's operand footprint `256r + 128c` subject to `r * c = 222` gives `r ~ 10.5`.
The square gain would be 6-7%, per the decomposition above. The aspect spread
would **collapse**, with `2048 x 32768` gaining most and `32768 x 2048`, already
walking a near-square wave, slightly *regressing*. The width would cost no
registers. And, from the L2 story: no gain where the operands are small enough
not to press on memory, the **largest gain at the largest `K`**.

Four of the five were right. The fifth was wrong in a way worth more than the
four, and it is table 2.

**1. The width, at 8192³, both schedulers.** Min ms over 30 timed launches, every
row checked against the CPU reference first. `reuse` is `wave_reuse` — arithmetic
on the item map, not a counter — and it reproduces the published column above for
`group 1` at all five aspect ratios to the printed digit.

| group | wave | distinct | reuse | static ms | static TF/s | clc ms | clc TF/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 x 64 | 1.18 MB | 9.2x | 0.9068 | 1212.5 | 0.8942 | 1229.6 |
| 2 | 4 x 64 | 1.18 MB | 9.2x | 0.8949 | 1228.6 | 0.8858 | 1241.3 |
| 4 | 4 x 56 | 1.05 MB | 10.4x | 0.8611 | 1276.8 | 0.8630 | 1274.1 |
| **8** | 8 x 28 | 0.72 MB | 15.1x | **0.8363** | **1314.8** | **0.8301** | **1324.5** |
| 16 | 16 x 14 | 0.75 MB | 14.5x | 0.8465 | 1298.9 | 0.8453 | 1300.7 |
| 32 | 32 x 7 | 1.16 MB | 9.4x | 0.8966 | 1226.3 | 0.8932 | 1231.0 |

**8, under both schedulers, as predicted, and the losers are the argument.** The
curve is not monotone in the width — it turns over at 8, and 32 is back where 1
was — which is what says the mechanism is the wave's *shape* and not "more
grouping is better". `group 2` is the sharpest row: it has the same 4 x 64 wave
and the same 9.2x as row-major, because at `tiles_n = 64` two tile-rows of a
group still take 128 items and 222 clusters spill into the next group either
way — and it measures within 1.3% of row-major, which is this benchmark's own
spread. A width that does not move the reuse does not move the clock.

**2. The mechanism, and the prediction that was wrong.** `M` and `N` are 8192 in
every row, so tiles, waves, grid, `C` bytes and the wave's own working set are
identical and only the operand footprint moves. Static, `group 1` against
`group 8`, as `min ms / TFLOP/s`:

| shape | operand | group 1 | group 8 | gain |
| --- | ---: | ---: | ---: | ---: |
| 8192x8192x512 | 16 MiB | 0.2710 / 253.6 | 0.2716 / 253.1 | **−0.2%** |
| 8192x8192x2048 | 64 MiB | 0.3725 / 738.0 | 0.3710 / 740.8 | **+0.4%** |
| 8192x8192x8192 | 256 MiB | 0.9068 / 1212.5 | 0.8363 / 1314.8 | **+8.4%** |
| 8192x8192x32768 | 1024 MiB | 3.0459 / 1443.9 | 3.1135 / 1412.6 | **−2.2%** |

**The gain is a band, not a ramp.** The shallow end was predicted and arrives: at
`K = 512` the traversal is worth nothing, which is the control the aspect table
cannot supply — it says the 8.4% is not a fixed cost the change happens to
remove, because the same change against the same tiles, waves and grid does
nothing when the operand stream is trivial. **The deep end was predicted
backwards.** `K = 32768` was supposed to gain most and instead loses 2.2%.

The reason, stated as the hypothesis it is: **the traversal only pays where
operand delivery is on the critical path, and neither end of this sweep is
there.** At `K = 512` the launch is the item boundary — 0.271 ms for ten items
against the 18-30 µs per tile priced above — and the operand stream is 348 GB/s
on a device where `layernorm` reaches 5497. At `K = 32768` the kernel is at
1443.9 TFLOP/s, inside the derived steady state of 1480-1550, so it is bound by
the pipeline and not by memory; the traffic the swizzle removes was not being
waited on, and what is left is its cost. That cost is real and this table is
where it shows: a grouped walk makes consecutive items write `C` down a column
rather than across a row, which is worse locality on the one stream the
traversal does not help.

**#98's occupancy sweep is independent evidence for that reading**, and it is
worth more than a better argument for it. The section above measures a lone CTA
at a flat **~585–640 TFLOP/s across 4096³, 8192³ and 16384³** — so more than half
of what this kernel achieves at any size is one CTA's stall covered by another's
work, and *overlap between CTAs is the dominant mechanism everywhere*. A row
already at 1443.9 TFLOP/s has that resource close to spent: there is little stall
left uncovered for cheaper operand delivery to remove. Two measurements taken for
unrelated reasons point the same way at `K = 32768`.

So the L2-residency framing that generated the prediction gets `K = 512` right
for roughly the wrong reason — it is not that 16 MiB fits L2, it is that 16 MiB
is not enough traffic to wait on — and gets `K = 32768` wrong outright. **The
honest generalization is narrower than the one #89 argues for**: a traversal
change is worth something in the regime where operands are the bottleneck, and
this kernel is in that regime at its benchmark sizes and out of it at `K` far
past them.

**3. The aspect ratio, which is the result.** The five shapes above, static, and
the prediction was that the spread collapses:

| shape | tiles_n | reuse 1 → 8 | group 1 | group 8 | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32768x2048x8192 | 16 | 15.1x → 13.9x | 0.8416 / 1306.5 | 0.8503 / 1293.0 | **−1.0%** |
| 16384x4096x8192 | 32 | 14.5x → 15.1x | 0.8643 / 1272.1 | 0.8460 / 1299.7 | **+2.2%** |
| 8192x8192x8192 | 64 | 9.2x → 15.1x | 0.9068 / 1212.5 | 0.8363 / 1314.8 | **+8.4%** |
| 4096x16384x8192 | 128 | 5.0x → 15.1x | 0.9607 / 1144.5 | 0.8316 / 1322.1 | **+15.5%** |
| 2048x32768x8192 | 256 | 3.0x → 15.1x | 1.0561 / 1041.1 | 0.8277 / **1328.4** | **+27.6%** |

**Confirmed, including its sign on the row that was supposed to lose.** Row-major
spans 1041.1 → 1306.5 TFLOP/s, a 25.5% spread; grouped spans 1293.0 → 1328.4, a
**2.7%** spread. Every row's gain is ordered by the reuse it recovered, and
`32768 x 2048` — the one shape whose row-major walk was already near-optimal, and
whose reuse the swizzle *lowers* from 15.1x to 13.9x — is the one row that gets
worse. A prediction that only ever says "faster" is not falsifiable; this one
named the row that would regress, and that row regressed.

The five `group 1` numbers here are all above the ones measured for #86 (1306.5
against 1123.7, and so on down) because #91's paired fp32 store landed in
between. The *spread* is the comparable quantity and it is unchanged: 22.7%
there, 25.5% here.

**4. Before and after, at the sizes #89 names, under both schedulers, against
cuBLASLt on the same device minutes apart.**

| shape | group | static ms | static TF/s | clc ms | clc TF/s | cuBLASLt ms | ours/theirs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192³ | 1 | 0.9068 | 1212.5 | 0.8942 | 1229.6 | 0.6201 | 0.684 |
| 8192³ | **8** | **0.8363** | **1314.8** | **0.8301** | **1324.5** | 0.6201 | **0.742** |
| 16384³ | 1 | 6.9461 | 1266.3 | 6.6635 | 1320.0 | 4.7560 | 0.685 |
| 16384³ | **8** | **5.9990** | **1466.3** | **5.9933** | **1467.7** | 4.7560 | **0.793** |

**0.684 → 0.742 of cuBLASLt at 8192³, and 0.685 → 0.793 at 16384³.** The two
`group 1` ratios reproduce #92's 0.689 and 0.694 to within this file's spread,
which is what says the control is the kernel that was there before. 1466.3
TFLOP/s is the fastest this kernel has been measured at — past the 1369.9 of the
depth sweep above — and 65% of dense bf16 peak.

**16384³ gains more than 8192³ — 13.6% against 8.4% — and that is not noise.** A
square `M : N` is not a square *wave*. 16384³ has a 64 x 128 tile grid, so 222
clusters walked row-major span two tile-rows and all 128 columns: a 2 x 128 wave
at **5.0x** reuse, which is exactly the `4096 x 16384` row of table 3, and that
row gained 15.5%. The two agree. It is worth being careful here anyway, because
16384³ is the least reproducible row in this repo by this file's own statements —
but a 13.6% gain measured against its own control in the same run, landing where
a shape with the same reuse independently landed, is outside the 5.5% cross-run
spread that caveat is about.

**It composes with CLC rather than replacing it**, which was the requirement. The
swizzle is applied to the *item*, and both item sources hand out a permutation of
`0..tiles`, so `grouped` being a bijection makes it correct under either by
construction. Both schedulers pick the same width, both gain, and the gain is the
same size — 8.4% static and 7.7% under CLC at 8192³. One thing the table says
that is worth noticing: **most of what CLC was winning, the swizzle was already
winning.** At 16384³ CLC was 4.2% ahead of the static stride row-major (6.9461 →
6.6635) and is 0.1% ahead of it grouped (5.9990 → 5.9933). #97 concluded CLC's
case was portability and not throughput; this narrows it further.

**It costs no registers.** `ptxas -v -arch=sm_100a` reports **168 registers, no
spills, 528 B stack frame, on both entry points, before and after** — measured on
this tree and on `c802627` through the same `modal_app.py::regcount`. The shared
plan is untouched at 73 816 B, so residency is the same 3 CTAs/SM. Two integer
divisions per output tile against a 20-30 µs item boundary is not a cost this
harness can see, and the cliff at 170 registers was not approached.

**That is worth more than a null result, because of what the section above
measured.** #98 priced one CTA an SM at **14–16%**, with a 25–44% cliff behind
it. This traversal spends no shared memory, no registers and no TMEM, so it pays
none of that: the 8.4% and 27.6% above are not net of an occupancy step, and
there is no budget in which they have to be argued for. **It is currently the
only lever this repo has measured that is free of the resource cliff.** Both of
the ones still on the table have to buy their gains out of a budget it did not
touch — #15's `lcsf` has to hide ~70% of the item boundary to clear one step, and
#87's tile change can afford the first step and must not take the second. A free
lever and a lever with a 14–16% entry fee are not comparable at equal nominal
size, and should not be ranked as though they were.

**Exactness, and where the map can be silently wrong.** A wrong item map is a
tile computed twice and a tile computed by nobody — the same failure seen from
either end, both silent on the device, both reaching the host only as a wrong
`C`. So `pipeline::grouped` is held two ways. In `src/pipeline.rs` a host test
asserts it is a **bijection** over every tile grid up to 17 x 17 at every width up
to 19, which is what covers the short last group a width that does not divide
`tiles_m` leaves; and `modal_app.py::examples` runs the GEMM at widths `[1, 3, 6]`
under both schedulers at both correctness sizes — **not powers of two,
deliberately**, because every `M` this project runs is one, so a swept width of 8
or 16 always divides `tiles_m` and the short-group branch would never execute.
All twelve are exact.

**What could not be measured: the L2 hit rate.** #89's acceptance asks for it
either side, and it is the right demand — a throughput win with an unchanged hit
rate would mean the cause was something else. **There is no hit rate on this
harness.** Nsight Compute is in the image and cannot start: the injected driver
set has no `libnvidia-pcc.so`, which is not in the 580.95.05 package either, and
`modal_app.py::profile` is checked in saying so. That is the same wall #86 and
#90 hit, and nothing here moved it. The substitute is stated rather than dropped:
the `reuse` column is arithmetic on the item map, calibrated by reproducing the
published column above exactly, and the causal claim rests on two *interventions*
instead of a counter — table 2, where the effect vanishes when the operand stream
is not the bottleneck, and table 3, where the effect tracks predicted reuse
across five shapes that hold nine other things fixed. A counter attributes; an
intervention that moves one variable establishes cause. It remains true that a
hit rate would have been worth having.

**What this leaves.** The decomposition above put 6-7% on traversal at the square
shape and 23% at the worst aspect ratio; the measurements are 8.4% and 27.6%, so
the estimate was conservative in both places. `GROUP` is a measurement at the
`[256, 128]` pair tile that **#87 will move**, which is why it is a parameter
with a sweep behind it rather than a swizzle written into the map.

**And this section invalidates the fit that ranks everything queued behind it.**
The item boundary's 20–22% and the 1521 TFLOP/s steady state both come from the
K-depth sweep, which was measured on the row-major kernel. This section moved
that kernel — 16384³ grouped is **1466.3**, close enough to the old *slope* that
the fit now describes a kernel which no longer exists. So "there is 20–22% of
item boundary left to win" is **what was true pre-#89 and is not currently a
supported claim**; `gemm-depth` has to be re-run on this tree before #15 or #87
is ranked off it. That is stated here rather than left for someone to trip over,
because the whole apparatus for choosing the next lever is downstream of it.

What does survive is the *shape* of #98's constraint: whatever the boundary
re-fits to, `lcsf` has to clear an occupancy step worth 14–16% to be worth
building, and this traversal had no such threshold to clear — which is most of
why it went first and landed.

**One coincidence worth being suspicious of, and it is not a finding.** The
`K = 32768` row regressed under grouping at **1443.9 TFLOP/s**, and 16384³
grouped lands at **1466.3**. Two shapes with almost nothing in common — a deep
reduction over a small output against a large square — sitting within 2% of each
other is *consistent with* a ceiling that is not operand traffic, which would be
a much larger finding than this section's, since it would mean neither a bigger
tile (#87) nor a store stage (#15) touches what is actually binding. **Two points
measured once each is not a ceiling and this does not claim one.** What would
test it is the re-fit above plus a third shape reaching the same number by a
third route; if `gemm-depth` re-fits to a steady state near 1470 rather than
1521, that is the same suspicion arriving from a fourth. A reader coming to that
re-fit should arrive already suspicious rather than surprised.

One cross-check worth recording, since the two sections were measured for
unrelated reasons in different containers: **#98's 3-CTA control and this
section's `group 1` control are the same kernel, and they agree.** 0.8965 ms /
1226.5 TFLOP/s / 0.685 of cuBLASLt there, against 0.9068 / 1212.5 / 0.684 here —
1.1% apart on raw time and identical on the ratio. Two independently staged
baselines landing on the same number is what makes both sets of deltas readable
against each other.

##### and the re-fit that ranks the rest, on the post-#102 kernel — #87

The section above ends by invalidating the fit everything queued behind it was
ranked off, and asks for `gemm-depth` to be re-run before #15 or #87 is ordered.
This is that re-run, on `7ef5685`, one B200, every row checked against the CPU
reference before it was timed.

| K | K blocks per item | min ms | median | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 0.2731 | 0.2782 | 251.6 |
| 2048 | 32 | 0.3700 | 0.3729 | 742.9 |
| 8192 | 128 | 0.8296 | 0.8333 | 1325.4 |
| 32768 | 512 | 3.0590 | 3.2067 | **1437.8** |

8192³ reproduces #102's 1314.8 to 0.8%, which is what says this is the same
kernel. Least squares on raw minimum milliseconds, **all three point selections
stated** because #90 was sent back once for leaving that out:

| points | fixed ms | per tile | steady TFLOP/s | share of 8192³ |
| --- | ---: | ---: | ---: | ---: |
| K = 8192, 32768 (preferred) | 0.0865 | **8.6 µs** | **1480** | **10.4%** |
| K = 2048, 8192, 32768 | 0.1516 | 15.2 µs | 1518 | 18.3% |
| all four | 0.1835 | 18.3 µs | 1538 | 22.1% |

The marginal cost of a K block per item runs **0.404, 0.479, 0.581 µs** across
the three intervals, against #90's 0.378, 0.503, 0.569 — still convex, so the
two deepest points are still preferred for the reason #90 gives, and the
L2-crossing caveat on the shallow rows is unchanged.

**Three things, and one is a problem with the instrument.**

**1. The steady state did not move.** 1480–1538 against #91's 1515–1527. #102
bought 8.4% at `K = 8192` and lost 2.2% at `K = 32768`, so it moved the middle
of this sweep and not its asymptote.

**2. The ceiling suspicion the section above raised got stronger.** Four numbers
by three unrelated routes now sit in **1437–1480**: this run's `K = 32768`
(1437.8), #102's `K = 32768` (1443.9 row-major, 1412.6 grouped), #102's 16384³
grouped (1466.3), and this fit's preferred asymptote (1480). Pre-#102 the
deepest measured row sat 10.6% under the fitted asymptote (1369.9 against 1515);
it now sits **2.9%** under it. A curve closing on its own asymptote is what a
real ceiling looks like.

> **It was not a ceiling, and the section below is what refutes it.** #87's
> wider pair tile measures **1718.5 TFLOP/s** at `K = 32768` and 1622.5 at
> 16384³, straight through the 1437–1480 band that four numbers by three routes
> had converged on. What that band was is a property of the **`[256, 128]`
> tile's arithmetic intensity**, not of the device and not of this kernel's
> structure — so the convergence was real and the inference from it was wrong.
> Three shapes agreeing is evidence that one *shared* term binds them, and it
> does not say which term; every one of those three shapes had the same tile.

**3. The fit is now a worse instrument for the item boundary than it was, and
that is not a detail.** The boundary reads 10.4% to 22.1% depending on point
selection — a 2.1× spread, where #90's 28.8%→35.7% was already enough to send it
back. The cause is mechanical: the intercept is *everything that does not scale
with `K`*, and #102 put a K-dependent, non-monotone locality term into that
bucket. Rotating the `K = 8192` point down toward the line is most of why the
two-point intercept halved. **So the 8.6 µs is not evidence that the epilogue
got cheaper — nothing touched the epilogue.** The honest statement is 8.6–18.3
µs per tile, and this fit cannot narrow it further.

##### and the pair tile was worth 21%, against a prediction that it would lose — #87

**The prediction, written down before the sweep ran, was that #87's headline
rung would cost 9–13%.** The reasoning: `[256, 256] @ STAGES = 3` buys 1.5×
arithmetic intensity for one occupancy step, #98 priced that step at 14–16%, and
a kernel sitting 2.9% under its own fitted asymptote at three shapes is not
waiting on operand delivery. The boundary-count half — a wider tile halves the
tiles, so halves the item boundaries — was argued to be worth 2.5–5.5%, since
#91's evidence says roughly half the boundary is the store loop and the store
loop is proportional to `C` bytes, which a wider tile does not reduce.

**It measured +11.6% and +21.6%, and the arithmetic above is wrong in a way the
control identifies.** Min ms over 30 timed launches, static schedule, `GROUP =
8`, one B200, every row checked element-by-element against the CPU reference
before it was timed. `vs #102` is against `[256, 128] s3`, the kernel that
shipped before this sweep.

| rung | shared B | CTA/SM | shape | tiles | wave eff | reuse | min ms | TFLOP/s | vs #102 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [256,128] s3 | 73816 | 3 | 8192³ | 2048 | 92.3% | 15.1× | 0.8420 | 1305.9 | — |
| [256,128] s2 | 49224 | **4** | 8192³ | 2048 | 98.8% | 16.8× | 1.3018 | 844.6 | **−35.3%** |
| [256,128] s4 | 98408 | 2 | 8192³ | 2048 | 98.8% | 12.7× | 0.9117 | 1206.0 | −7.7% |
| [256,256] s2 | 65608 | 2 | 8192³ | 1024 | 98.8% | 11.0× | 0.8555 | 1285.2 | −1.6% |
| **[256,256] s3** | 98392 | 2 | 8192³ | 1024 | 98.8% | 11.0× | **0.7546** | **1457.1** | **+11.6%** |
| [256,128] s3 | 73816 | 3 | 16384³ | 8192 | 99.7% | 15.1× | 6.5948 | 1333.8 | — |
| [256,128] s2 | 49224 | **4** | 16384³ | 8192 | 98.8% | 16.8× | 9.4160 | 934.2 | **−30.0%** |
| [256,128] s4 | 98408 | 2 | 16384³ | 8192 | 98.8% | 12.7× | 6.4605 | 1361.5 | +2.1% |
| [256,256] s2 | 65608 | 2 | 16384³ | 4096 | 98.8% | 11.0× | 5.8492 | 1503.8 | +12.7% |
| **[256,256] s3** | 98392 | 2 | 16384³ | 4096 | 98.8% | 11.0× | **5.4214** | **1622.5** | **+21.6%** |

**1622.5 TFLOP/s is the fastest this kernel has ever been measured at** — past
#102's 1466.3, and 72% of dense bf16 peak.

**The control is what makes this readable, and it is the row #87's own table
does not have.** `[256, 128] @ STAGES = 4` has the *same* shared plan as the
winner (98408 B against 98392) and therefore the same two CTAs an SM, at
unchanged pair tile, unchanged intensity and unchanged tile count. So it prices
the occupancy step on its own, and the difference between it and the winner is
the tile and nothing else:

| | 8192³ | 16384³ |
| --- | ---: | ---: |
| occupancy step 3 → 2, at unchanged tile ([256,128] s4 vs s3) | **−7.7%** | **+2.1%** |
| the tile, at fixed residency and shared plan ([256,256] s3 vs [256,128] s4) | **+20.8%** | **+19.2%** |
| net | +11.6% | +21.6% |

Two findings fall out, and neither is #87's argument.

**A fourth pipeline stage very nearly pays for the CTA it costs.** #98 priced
the 3 → 2 step at **13.6–16.1%** on bytes no code touched. Here the identical
step, paid for with a live fourth stage, costs **7.7%** at 8192³ and **gains
2.1%** at 16384³. Subtracting, the fourth stage is worth roughly 14–18% — about
what the CTA it displaces was worth. That is a much more direct answer to "does
depth substitute for residency" than anything here has had, and it says: nearly
exactly, at this shape.

**And the reverse trade is a catastrophe.** `[256, 128] @ STAGES = 2` steps
residency *up*, to four CTAs an SM — the only rung here that does — and it is
the worst row in the table by a wide margin, **−35.3% and −30.0%**. It also has
the *best* wave efficiency and the *best* reuse of any rung. So a fourth CTA
cannot cover what a third pipeline stage was covering, and residency and
pipeline depth are not interchangeable currency even though both are bought with
shared memory. This is the rung most likely to have been skipped on the grounds
that "more occupancy is better", and it is the largest single number in the
table.

**Which of the two tile mechanisms, and the traversal is the evidence.** #89's
`GROUP` was measured at `[256, 128]` and a wider tile halves `tiles_n`, so it was
re-swept rather than carried across. At 8192³:

| group | [256,128] s3 reuse | TFLOP/s | [256,256] s3 reuse | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.2× | 1210.5 | 8.0× | 1420.3 |
| 4 | 10.4× | 1270.0 | 7.4× | 1425.1 |
| 8 | 15.1× | **1307.1** | 11.0× | 1455.2 |
| 16 | 14.5× | 1287.7 | 11.4× | **1455.9** |

**The traversal is worth 8.0% at the old tile and 2.5% at the new one.** 8 and
16 are a dead heat at the new tile — 0.05% apart — so `GROUP` stays 8 and the
choice stopped mattering much. That is the mechanism evidence: the lever #102
pulled acts on operand delivery, and the wider tile has absorbed most of what it
was recovering. Note also that the winner's *wave reuse is lower* than the
control's (11.0× against 12.7×) and it is 20% faster, so this is not L2 capture
— it is the tile's own arithmetic intensity, which is a different quantity and
the one #87 named.

**The `[256, 256] @ STAGES = 3` rung is the standout for the reason #87 gives,
and that reason is now measured.** Tensor memory caps the tile at two CTAs an SM
whatever the depth, so the third stage costs shared memory that residency was
not going to use. Against `STAGES = 2` at the same tile and the same counted
residency it is worth **+13.4%** at 8192³ and **+7.9%** at 16384³ — a free
third stage, and it is not a small thing.

##### which mechanism, settled by re-running the same fit on the new tile

The two mechanisms scale differently with `K`, which is what separates them:
**arithmetic intensity is a property of the steady state and moves the fit's
slope**, while **boundary count is a fixed per-launch cost and moves its
intercept**. So `gemm-depth` was run again on the new tile, same B200, same
verify-then-time path, and fitted the same three ways.

`[256, 256] s3` at `M = N = 8192`: 0.2447 / 0.3422 / 0.7526 / **2.5592 ms** at
`K` of 512 / 2048 / 8192 / 32768 — that is **280.8 / 803.2 / 1460.9 /
1718.5 TFLOP/s**, and the last is **76.4% of dense bf16 peak**. Items on the
critical path are 7 rather than 10, because 1024 tiles over 148 clusters is not
2048 over 222.

| points | steady TFLOP/s, `[256,128] s3` → `[256,256] s3` | fixed ms | per tile | share of 8192³ |
| --- | ---: | ---: | ---: | ---: |
| K = 8192, 32768 | 1480 → **1826** | 0.0865 → **0.1504** | 8.6 → **21.5 µs** | 10.4% → 20.0% |
| K = 2048, 8192, 32768 | 1518 → **1850** | 0.1516 → 0.1779 | 15.2 → 25.4 µs | 18.3% → 23.6% |
| all four | 1538 → **1862** | 0.1835 → 0.1903 | 18.3 → 27.2 µs | 22.1% → 25.3% |

**The slope moved 21–23% and the intercept went the wrong way.** That is the
answer, and it is unambiguous:

- **Arithmetic intensity is the entire mechanism.** +21.1% to +23.4% on the
  steady state, against the +19.2% to +20.8% the `[256, 128] s4` control
  measured for the tile net of occupancy. Two instruments that share no
  arithmetic agree, so #87's stated argument — the one the issue leads with — is
  right and is worth all of the win.
- **Boundary count is not a mechanism here, and it has the wrong sign.** The
  fixed term rose from 0.0865 to 0.1504 ms, **+74%**. Halving the tiles did not
  halve the boundaries on the critical path, for a reason the pre-registration
  missed: the occupancy step halves the *clusters* too, so items per cluster
  fall only 10 → 7, while a boundary that drains twice as many accumulator
  columns costs more than twice as much. Two and a half times the cost over
  seven tenths the count is a fixed term that grows.

So the argument advanced before the run — that boundary count was the *stronger*
of the two mechanisms and intensity would be worth nothing against a ceiling —
was wrong on both halves, and wrong in opposite directions. Recording it because
it is the most informative row here: the sweep's design was right and its
prediction was not, and the design is what made the prediction falsifiable.

It also explains the size dependence with no free parameters. At 8192³ the
K-proportional part improves 23.4% (0.7431 → 0.6022 ms) and the fixed part
worsens by 0.064 ms, netting **+10.2% predicted against +11.6% measured**. At
16384³ there is four times as much arithmetic to amortize the same fixed
regression over, so the net moves toward the slope — +21.6%, against a slope
gain of 23.4%. The two sizes are the same decomposition seen at two amortization
ratios.

**One number to be careful with.** The fitted steady state is now 1826–1862
TFLOP/s, which *brackets cuBLASLt's measured* 1741–1829. Nothing has run at
1826; the fastest measured is **1718.5**, and that is the figure to hold anyone
to. The derivation stacks the same two assumptions §7 has flagged twice.

##### the TMEM half of the residency formula, binding for the first time

Every rung above was *counted*, not predicted, by `device-tests`' `tmem
residency census` at the four envelopes the sweep declares — and `shared per SM`
is queried from the driver rather than written down, because a rung's residency
is a floor division by it.

| rung | columns | shared B | budget | resident | holding | wait µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [256,128] s2 | 128 | 49224 | 4 | 4 | 4 | 0.9 |
| [256,128] s4 | 128 | 98408 | 2 | 2 | 2 | 0.9 |
| **[256,256] s2** | 256 | 65608 | **2** | **3** | **2** | **100.0** |
| [256,256] s3 (gemm) | 256 | 98392 | 2 | 2 | 2 | 0.9 |

**Predicted and counted agree at all four, and the third row is the one worth
the run.** It is the first rung in this repo where `resident` and `holding`
differ, and the first where the `512 / columns` half of
`min(512 / columns, shared per SM / plan)` is what binds: 65608 B admits three
CTAs and 256 accumulator columns hold two. `src/tmem.rs` predicted exactly this
— a CTA admitted and parked inside a blocking `tcgen05.alloc` — and the **wait
column is the direct evidence**, 100.0 µs against 0.9 µs on every shared-bound
rung. That CTA occupies a slot for the whole hold and makes no progress, which
is also why the rung underperforms: its "extra" residency is a stalled warp.

##### what it cost, which was no registers and some generality

`ptxas -v -arch=sm_100a`, through `modal_app.py::regcount`, on every entry point
in the sweep: **168 registers, 0 spill store, 0 spill load, 528 B stack frame** —
identical across all six, and identical to what `gemm_cg2` measured before any
of this. The register column, which #47, #63, #67, #76 and #94 have each ordered
time backwards, has nothing to say here at all.

That is not automatic and it is the one piece of real work `[256, 256]` costs
that #87 does not mention. A warp's band of a 256-column accumulator is
`RegTile<32, 256, BaseLdtm>` — **256 fp32 a thread**, past the 255-register
architectural file before any of the kernel's own live state, and `regcount`'s
own ladder in the same run says a `[32, 192]` band already spills in four of the
five spellings. So the epilogue drains in two sequential 128-column bands
(`DRAIN_N`), which holds peak liveness at exactly what it was. At `BLOCK_N = 128`
the loop is one iteration and folds away, which is why the control rung's
codegen is byte-identical.

The occupancy step also **bought 87 registers of headroom**. #91 recorded the
old kernel as "thin: 167 against 170, with shared memory already capping
residency at 3, so both resources are at the limit". At two CTAs an SM the step
is at 255, and `regcount`'s gate now reports 168 against 255. The cliff #100 was
built to watch is no longer anywhere near.

**What it did cost is generality.** A launch must now have `n % 256 == 0` where
`n % 128 == 0` used to do, so this kernel computes a strictly narrower set of
shapes than it did. #92 already named generality as the axis on which a
like-for-like rate against a general library flatters us, and this moves further
along it. `GEMM_SIZES`' smallest row moved from `256x128x256` to `256x256x256`
for the same reason — it is still the one-cluster-one-item shape §7's item 3 is
about, but its microseconds are not comparable across the change, because the
item it is one of has twice the columns.

##### and the denominator

cuBLASLt, same device, same container, minutes apart, checked against the same
`gemm::check_c`:

| shape | cuBLASLt ms | theirs TFLOP/s | #102/theirs | **#87/theirs** |
| --- | ---: | ---: | ---: | ---: |
| 8192³ | 0.6233 | 1764.0 | 0.740 | **0.826** |
| 16384³ | 4.8035 | 1831.2 | 0.728 | **0.886** |

The two `#102` ratios reproduce that section's 0.742 and 0.793 — the second is
1.5 points off, which is inside what this file's own spread on 16384³ looks
like, and it is the control saying the container is not the story.

**0.886 of a tuned vendor GEMM is the closest this project has been**, against
the 0.573–0.694 band it sat in through #92 and 0.742–0.793 through #102.

##### and the whole size sweep, where the small end pays for it

The rungs above are the two sizes every table here is quoted at. The full
`bench --case gemm` sweep on the shipped kernel says the trade is not uniform,
and the direction is worth stating before anyone reads 0.886 as the headline.
*(#119 moved what `bench --case gemm` launches, from the register drain to
`staged84`. Every row below is the register drain's; a sweep run today is a
different kernel and belongs beside this table rather than in it.)*

| shape | ours ms | ours TFLOP/s | theirs TFLOP/s | ours/theirs | was (#91/#92) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256x256 | 0.0222 | 1.5 | 4.3 | 0.354 | — |
| 512x256x256 | 0.0220 | 3.1 | 8.8 | 0.348 | 0.598 |
| 1024³ | 0.0269 | 79.9 | 219.3 | 0.364 | 0.581 |
| 2048³ | 0.0340 | **505.5** | 837.6 | 0.604 | 0.627 |
| 4096³ | 0.1439 | **954.9** | 1609.2 | 0.593 | 0.573 |
| 8192³ | 0.7610 | **1444.8** | 1741.4 | **0.830** | 0.689 |
| 16384³ | 5.4873 | **1603.0** | 1828.8 | **0.877** | 0.694 |

**Above 2048³ every row improves; at and below 1024³ they get worse.** 1024³
goes 98.0 → 79.9 TFLOP/s against #91's measurement, an 18% regression, and that
is exactly what the fit predicts: a wider tile costs more per boundary and a
1024³ launch is 16 tiles over 148 clusters, which is one item each and nothing
to amortize it over. The wider tile is a trade of small-problem throughput for
large-problem throughput, and it is a good trade at the sizes this project is
aimed at and a bad one below 2048³.

The bottom two ratios are also the reproducible ones and the top three are not —
#92's finding that cuBLASLt's own variance dominates below 4096³ has not changed,
and 0.354 should be read as "small, unstable, and worse" rather than to three
digits.

##### and the store stage was free to build and worth nothing — #15

Every section above ends by pointing at `lcsf`. #90 called the item boundary
"the shape `lcsf` changes"; #94 priced a staging buffer and rejected it; #98
re-priced the occupancy step at 14–16% and set the bar at "hide ~70% of the
boundary"; #87 then took the residency to 2 CTAs an SM for a reason that has
nothing to do with shared memory. This is that measurement, and the answer is
**no** by a margin that does not depend on which fit or which session.

**The premise everything was ranked on is gone, and not because the arithmetic
was wrong.** The brief for this work re-derived the shared budget as
`233472 / 2 = 116736` B a CTA against a 98392 B plan, leaving **18344 B a CTA
already paid for and unused**. That is right. What it does not say is that the
step it was buying no longer exists: since #87 `CTAS_PER_SM` is **2 because 256
accumulator columns are half of tensor memory**, and `min(512/columns,
shared per SM/plan)` is `min(2, 2)`. So shared memory up to 18344 B a CTA is not
"free" in the sense of being pre-paid — it is *invisible*, because the tensor
memory half binds first and would go on binding. Above that the plan falls to
one CTA an SM, which is #98's 25–44% cliff.

**And 16384 B is not "a band" in any unit the epilogue is made of.** The
`[32, 128]` fp32 band the arithmetic names is *one warp's* share of one drain
step. A CTA's per-item output is `128 × 256` fp32 = **131072 B**, so 16384 B is
an eighth of it, and four warps sharing one buffer serialise. The largest
staging that fits is a `[128, 32]` strip — **eight single-buffered
fill → fence → TMA-store → wait-read round trips an item**, since double
buffering wants 32768 B and does not fit. That is the shape the "18344 B is
enough" reading actually buys, and it is worth knowing before costing it.

**None of which mattered, because the store stage does not need a staging buffer
at all.** #15 files `lcsf` as a shape `pipeline::run` would have to grow, and
#94 scoped it at 250–400 lines of missing plumbing — `TensorMapElement` is
`Bf16`-only, there is no fp32 `SharedTile` swizzle, `stmatrix` is b16. What
stands between item `i`'s epilogue and item `i + 1`'s first loads is none of
that. It is that `drain` is the last thing `work` does. **The pair's accumulator
lives in tensor memory allocated once outside the item loop, and the item
boundary re-arms mbarriers and touches no tensor memory** — so an undrained
accumulator survives it intact, and deferring the drain by one item is a
reordering of phases inside `Job::work`. `src/pipeline.rs` is **unmodified**:
the pending item is job state, and `lcf`'s scaffold already admits an `lcsf`
job. The whole epilogue, the LDTM *and* the scattered fp32 stores, then runs
while the next item's first `STAGES` loads are in flight, for **no shared
memory, no deferred registers, no fp32 TMA path and no occupancy step**.

That is `Lcsf` in `examples/src/gemm.rs`, and it is the cheapest honest
prototype there is: the thing #94 said had to be built to find out is not on the
path to finding out.

**It does need a synchronisation the boundary does not supply, and that is a
finding on its own.** The MMA is `cta_group::2`, issued by the leader alone and
writing *both* ranks' halves of the accumulator. Under the fused epilogue the
pair is separated by `run`'s cluster boundary. Here the peer's drain of item `i`
and the leader's MMA for item `i + 1` are inside one `work`, and a `bar.sync`
orders one CTA — the leader would overwrite tensor memory the peer is still
reading, silently, as a wrong `C`. So `lcsf` does not want an extra barrier at
the boundary; it wants the boundary's **scope** moved inside the item. The
`bar.sync` before the drain is load-bearing too, and for an unrelated reason:
`LDTM` is warp-collective and the producer branch leaves warp 0's lane 0
somewhere its other 31 lanes are not.

**The prediction, written into the sweep before it ran:** +4% to +11% at 8192³
and 16384³, **nothing at 1024³**, monotone between — the boundary is a fixed
per-tile cost, so what `lcsf` moves is amortized against the arithmetic between
one boundary and the next. And 168 registers with no spill, because nothing is
held across the boundary.

**The register half was right and the throughput half was wrong.** Two sessions,
one B200 each, min ms over 30 timed launches, static schedule, `GROUP = 8`,
every row checked element-by-element against the CPU reference before it was
timed. Identical grid, identical 98392 B shared plan, identical 256 accumulator
columns, identical 2 CTAs an SM — **the diff is where one function is called
from**:

| shape | tiles | items/cluster | lcf ms | lcsf ms | vs lcf | session 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024³ | 16 | 1.0 | 0.0288 | 0.0304 | **−5.4%** | −2.2% |
| 2048³ | 64 | 1.0 | 0.0366 | 0.0372 | −1.6% | −2.8% |
| 4096³ | 256 | 1.7 | 0.1447 | 0.1490 | −2.9% | −1.4% |
| 8192³ | 1024 | 6.9 | 0.7506 | 0.7521 | **−0.2%** | −0.0% |
| 16384³ | 4096 | 27.7 | 5.3869 | 5.4421 | **−1.0%** | **+1.2%** |

**Across two sessions the deferred epilogue is −5.4% to +1.2% and never reliably
positive.** The two headline rows are −0.2%/−0.0% and −1.0%/+1.2%: zero, inside
this file's own spread. The 16384³ sign flip is the row §7 has twice called its
least reproducible, and the *control* is what moved (5.3869 → 5.4972, 2.0%)
while the treatment held to 0.2% — which is the cross-run spread behaving
exactly as documented rather than a result.

**The 1024³ row is the pre-registered falsifier and it did what it was supposed
to.** Sixteen tiles over 148 clusters is one item each, so every epilogue is the
*last* epilogue, `Lcsf::finish` pays it un-hidden, and the extra cluster
rendezvous is pure cost. It loses most, at both sessions. So the cost model is
right and it is the **benefit that is absent**: `f ≈ 0`, against the `f ≈ 0.65`
#98 set as the bar for a route that also had to buy an occupancy step. This one
had no step to buy and still did not clear zero.

**What was left open, and the probe that closes it.** The deferred drain hides
`{LDTM + global stores}` behind one pipeline fill. #15's TMEM → shared → TMA
route would hide the same LDTM behind the same window — tensor memory has to be
free before the item's first MMA either way — and *additionally* decouple the
shared → global write. So exactly one question survived: **is the epilogue
bandwidth-bound?** `Epilogue::HotStore` answers it by holding the epilogue's
instructions fixed and moving only where they land: every item stores to the
cluster's own first tile, so the same LDTM and the same count of
`st.global.v2.f32` rewrite 38 MB that stays in L2 instead of streaming 268 MB or
1 GB. **It computes a wrong `C` on purpose, is excluded from the correctness
gate, and its number is an upper bound rather than a throughput** — the one
deliberate exception to rule 1 in this file, stated in the launch's own label.

| shape | lcf ms | hot ms | hot TFLOP/s | write-bound part |
| --- | ---: | ---: | ---: | ---: |
| 1024³ | 0.0268 | 0.0270 | 79.5 | −0.8% |
| 2048³ | 0.0342 | 0.0340 | 505.1 | 0.6% |
| 4096³ | 0.1437 | 0.1437 | 956.6 | 0.0% |
| 8192³ | 0.7517 | 0.7502 | 1465.6 | **0.2%** |
| 16384³ | 5.4972 | 5.4293 | 1620.1 | **1.2%** |

**Deleting a gigabyte of streaming HBM writes is worth nothing.** The epilogue
is issue-bound, not bandwidth-bound: the threads pay for the LDTM and for
issuing the stores, and not for where the stores land. #91 already halved the
instructions and doubled the sector fill and was worth 14–28%; what is left
after it is not a memory-system term, and a staging buffer whose entire benefit
is decoupling the global write has **≤1.2%** to recover at any size measured.

**So #15 is answered in both of its forms, and the second answer did not need
the 250–400 lines.** The cheap form hides the half that could be hidden and
measures zero; the expensive form's only remaining advantage over the cheap form
is worth at most 1.2%. Neither is worth building on this kernel.

**Why, structurally, and this is the part that generalizes.** The overlap window
is bounded by `STAGES` of loads and cannot be widened. The accumulator is
**single-buffered in tensor memory**, so the drain must complete before the
item's first MMA; a second accumulator is 512 columns, which is the whole of
tensor memory and one CTA an SM. Meanwhile #98 measured a lone CTA at a flat
~585–640 TFLOP/s against 900–1299 for three, which says *more than half of what
this kernel achieves is already one CTA's stall covered by another's work* — the
neighbouring CTA's K loop is what the epilogue's issue slots are already
overlapping with. `lcsf` proposes to spend a resource that inter-CTA overlap has
already spent. That reading is #102's too, at `K = 32768`, and it is now three
measurements taken for unrelated reasons pointing the same way.

**The denominator either side, same device, same container, minutes apart:**

| shape | cuBLASLt TFLOP/s | lcf/theirs | lcsf/theirs |
| --- | ---: | ---: | ---: |
| 8192³ (session 1) | 1769.7 | **0.828** | 0.826 |
| 16384³ (session 1) | 1854.5 | **0.881** | 0.872 |
| 8192³ (session 2) | 1774.4 | **0.824** | 0.824 |
| 16384³ (session 2) | 1870.7 | 0.855 | 0.866 |

The `lcf` control reproduces #87's published **0.826 and 0.886** to within this
file's spread, which is what says the container is not the story. Nothing here
moves the shipped kernel, and 0.826/0.886 stands.

**Registers and residency, both unchanged and both checked rather than
asserted.** `ptxas -v -arch=sm_100a` through `modal_app.py::regcount` reports
**168 registers, 0 spill store, 0 spill load, 528 B stack frame** on
`gemm_cg2_lcsf`, byte-identical to `gemm_cg2`, `gemm_cg2_clc` and all four sweep
rungs — the pre-registered prediction, and the direct evidence that the deferral
holds nothing across the boundary. The occupancy-step gate reads 168 against
255. Residency is **2 CTAs an SM, counted**: #87's census row counted 2 at
exactly this envelope (256 columns, 98392 B, `wait 0.9 µs`), and no byte of the
shared plan and no accumulator column moved, so the count carries by identity
rather than by extrapolation. **No new census rung was run, and that is the
reason** — a rung at an envelope already counted would have spent a B200 to
reproduce a number.

**What is kept, and why the losing attempt stays in the tree.** `gemm` shipped
`lcf` until #119 and ships `staged84` now — which is still `lcf`'s *placement*,
so nothing in this section's conclusion moved. `Lcsf`, `HotStore`, the
`Epilogue` axis and
`bench --case epilogue` stay, and `Lcsf` stays on the correctness gate at both
shapes and all three traversal widths — including `4096x4096x256`, where 256
tiles over 148 clusters means every cluster carries a pending accumulator across
boundaries and one past the loop, which is the way this change fails silently.
The reason to keep it runnable is that the constraint that decided it is a
property of **this tile**: the overlap window is one pipeline fill, and #87 moved
`STAGES` and `BLOCK_N` once already. A tile whose fill is longer, or a kernel
whose epilogue is genuinely latency-exposed, would re-run this sweep rather than
inherit its answer.

**One thing this does not claim.** `lcsf` is a bad trade *for a GEMM whose
accumulator is single-buffered in tensor memory and whose epilogue is
issue-bound*. #15's original motivation — a training kernel with a
producer-consumer epilogue — is a different kernel, and nothing here prices it.
What is now established is the method: the store stage costs one reordering to
try, so the next kernel that wants one should measure it before anyone scopes a
tensor map.

##### and `C` is bf16 now, on both sides of the ratio — #108

`gemm_cg2` accumulated in fp32 and *wrote* fp32 through #107. `bf16` in, fp32
accumulate, `bf16` out is the ordinary training-GEMM signature, so it was the
output that was the unusual half here, and this is the change that moves it.
**The accumulator does not move**: the accumulator is still 256 fp32 columns of
tensor memory, the drained band is still 128 fp32 a thread, and the single
`cvt.rn.bf16x2.f32` lives inside the store.

**The cuBLASLt baseline moved in the same commit and had to.** At 8192³ an fp32
`C` is 268 MB and a bf16 `C` is 134 MB. Timing our kernel writing half as much
against a baseline still writing twice as much, and reporting the difference as
ours, is the way this change goes wrong — so `cublaslt.rs`'s output layout is
`CUDA_R_16BF` and both columns below were re-measured in one container.
`CUBLAS_COMPUTE_32F` and the `CUDA_R_16BF` operands are untouched; the library
accumulates in fp32 exactly as we do. That module's own doc used to say
upstream's C file "uses a 2-byte output and is not the same measurement", which
had the asymmetry backwards — the baseline had been bent to match *us*.

**Which earlier rows survive this, because most do not.** Every `ours/theirs`
ratio in §7 above — #92's 0.573–0.694, #102's 0.742/0.793, #87's 0.826/0.886,
#107's 0.824–0.881 — is a *pair* of fp32-output measurements, as is every
absolute millisecond and every TFLOP/s figure. None is comparable to a number
below, and rescaling one is not available either, since the two sides did not
have to lose the same amount. What survives, being a property of the tile rather
than of the output: the counted residency, the register count and its zero
spill, `GROUP = 8`, the 98392 B shared plan, wave efficiency, `wave_reuse`, and
the whole structural argument about the item boundary. #87's rungs stay
comparable to *each other* across this change and not to their published
figures.

**The predictions, written into this file before the run.** They are in the
commit that added the section with these tables empty, which is the only way a
pre-registration in a repo is worth anything.

1. **bf16 `C` is worth roughly nothing, and might cost.** #107's `HotStore`
   deleted a gigabyte of streaming writes for 0–1.2%, so the epilogue is
   issue-bound and halving its bytes cannot buy more than that. And it does not
   reduce the instruction count: under `BaseLdtm` the contiguous run is 2 (#94's
   `CONTIGUOUS_VALUES`; the columns are `{0,1}` then `{8,9}`, eight apart), so a
   bf16 pair is a **4-byte store where an fp32 pair was 8 — the same number of
   stores, half as wide** — plus a `cvt` that was not there before. Predicted
   **−1% to +1.5%** at 8192³ and 16384³.
2. **The ratio moves against us, slightly.** cuBLASLt gets the cheaper output
   too and sits closer to the machine, so if the write is worth anything it is
   worth at least as much to them. Predicted **0 to −3 points** off #107's 0.828
   and 0.881.
3. **168 registers, 0 spill, 528 B frame, unchanged.** The `cvt` takes two fp32
   and yields one packed word, so peak liveness cannot rise.
4. **2 CTAs an SM, unchanged, and not for a shared-memory reason.** Since #87
   the binding half of `min(512 / columns, shared per SM / plan)` is tensor
   memory — `512 / 256 = 2` — and a narrower `C` moves neither term. To be
   *counted* rather than assumed.
5. **Exactness survives with no tolerance at all**, and the worst relative error
   of the output against the exact fp32 reference is **2⁻⁸ ≈ 3.9e-3**.

And a sixth, for a probe this change adds rather than for the change itself:
`Epilogue::DoubleDrain` runs the epilogue twice per item and prices it directly
instead of as a share of a fitted boundary. Predicted **5–12 µs a tile at
8192³**, so 5–12% of that launch.

**1. Both columns, both output elements, four B200 containers in one session.**
`fp32` is `a68c390` — the commit before this one, with `cublaslt.rs` still at
`CUDA_R_32F` — run back to back with the branch through the same
`bench --case gemm`. Every row on both sides is checked against the same CPU
reference before it is timed, and cuBLASLt's heuristic returned a **byte-identical
algorithm at every size in both runs** (`id=66`, same tile, `splitk=1`, no
workspace used), so nothing below is a different kernel being compared.

| shape | ours fp32 | ours bf16 | ours | theirs fp32 | theirs bf16 | theirs | fp32 ratio | bf16 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256x256 | 0.0228 | 0.0244 | −6.6% | 0.0076 | 0.0113 | −32.7% | 0.336 | 0.463 |
| 512x256x256 | 0.0231 | 0.0247 | −6.5% | 0.0075 | 0.0111 | −32.4% | 0.325 | 0.449 |
| 1024³ | 0.0268 | 0.0297 | −9.8% | 0.0101 | 0.0140 | −27.9% | 0.378 | 0.473 |
| 2048³ | 0.0341 | 0.0359 | −5.0% | 0.0210 | 0.0229 | −8.3% | 0.615 | 0.638 |
| 4096³ | 0.1426 | 0.1441 | −1.0% | 0.0859 | 0.0898 | −4.3% | 0.603 | 0.623 |
| **8192³** | 0.7489 | **0.7344** | **+2.0%** | 0.6163 | 0.5964 | +3.3% | 0.823 | **0.812** |
| **16384³** | 5.4025 | **5.2326** | **+3.2%** | 4.7637 | 4.5619 | +4.4% | 0.882 | **0.872** |

In TFLOP/s the two large rows are **1468.1 → 1497.1** and **1628.2 → 1681.0**
for us, against **1784.0 → 1843.5** and **1846.5 → 1928.2** for cuBLASLt.

**And a single pair of containers is not enough to read those two rows, which
is the first thing to say about them.** The branch was measured three times at
8192³ across this session's containers — 0.7344, 0.7456 and 0.7440 ms — where
the fp32 control was measured once, at 0.7489. So the gain at 8192³ is
**+0.4% to +2.0%, about +1.1% on the mean**, against a cross-container spread on
this row of 1.5%. At 16384³ the three bf16 measurements are 5.2326, 5.5193 and
5.4765 against the control's 5.4025 — they **straddle it**, which is what §7 has
twice said about 16384³ and is not new information about this change.

| | ours fp32 | ours bf16, three containers | theirs fp32 | theirs bf16, three containers |
| --- | ---: | --- | ---: | --- |
| 8192³ | 0.7489 | 0.7344 / 0.7456 / 0.7440 | 0.6163 | 0.5964 / 0.6154 / 0.6137 |
| 16384³ | 5.4025 | 5.2326 / 5.5193 / 5.4765 | 4.7637 | 4.5619 / 4.6969 / 4.7119 |

**So prediction 1 is confirmed and prediction 2 is confirmed.** bf16 `C` is
worth about **+1% at 8192³ and nothing resolvable at 16384³**, inside the
predicted −1% to +1.5%; and the ratio to cuBLASLt is **0.812–0.825 against the
control's 0.823** at 8192³ and **0.851–0.872 against 0.882** at 16384³ — flat to
two points down, inside the predicted 0 to −3. The library gains at least as
much from the cheaper output as we do, which is exactly why moving only our side
would have been a fabricated win: read against the *published* fp32 ratios of
0.826 and 0.886, our unmoved-baseline numbers would have looked like +3.6% and
+4.9%.

**The small end is the loud part of the table and none of it is readable.**
Ours loses 5–10% at and below 2048³ and cuBLASLt loses 28–33% at and below
1024³ — with, again, an identical algorithm pick. #92 measured cuBLASLt's own
run-to-run variance at **33% to 77%** below 2048³ and ours at 5–11%, and this
session has one measurement per size down there. Every one of those deltas is
inside the variance already documented for its own side. The `bf16 ratio` column
at the top of the table therefore says *nothing good about us*: 0.336 → 0.463 at
256³ is the baseline getting worse, not the kernel getting better, and it is the
clearest example in this file of why a ratio whose denominator moved must not be
read as a numerator result.

What is worth a second look, if anyone wants it, is that our own small-size loss
is at least the right *shape* for the mechanism: a `cvt` per pair is pure added
issue, and a launch that is one item per cluster has no operand stream for the
halved bytes to pay it back out of. Four rows in a row lean that way. It is not
established by this session and would want its own.

**The fp32 path is kept in the library and not in the kernel, and that is a
choice worth stating.** `GlobalRows<E>` carries the element, `store_rows` and
`load_rows` are generic over it, and `flash_forward` and `device-tests` still
write fp32 through exactly the code they did — so the losing side of this change
is reachable, tested and on the CPU gate. What is *not* kept is an fp32 entry
point in `gemm.rs`. That is not one line: the kernel signature, the device
buffer, the launcher's closure type and the reference the check compares against
all key off the element, so a permanent in-tree A/B is 80–120 lines and a fork
in `run`, for a configuration this issue decided against. The comparison it
would buy is the one table 1 already has, obtained the cheaper way — run the
parent commit back to back — and `a68c390` does not stop being available.

**2. The accuracy change, and the tolerance that did not move.**

The output rounds where it did not before, so the obvious expectation is that
the exactness check acquires a tolerance. **It does not, and that is worth being
precise about rather than glossing.** The comparison is still `==`, on the
16-bit words, at every element of `C`, for the kernel and for cuBLASLt alike.

What made that available is that **the rounding is put in the reference**.
`gemm::check_c` computes the exact fp32 dot product as it always did and then
applies `to_bf16` — the same round-to-nearest-even, ties included, that
`cvt.rn.bf16x2.f32` applies — and compares words. That works because the sum
itself is still exact: every operand is exact in bf16 and every partial sum
stays under 2²⁴, so the fp32 value arriving at the `cvt` is the same integer
whatever order it was summed in, and rounding a known integer once is a
deterministic function of it.

| | before #108 | after |
| --- | --- | --- |
| comparison | `==` on fp32 | `==` on bf16 words |
| tolerance | none | **none** |
| worst relative error against the exact fp32 reference | 0 | **3.86e-3** |

3.86e-3 is 2⁻⁸, it is what was predicted, and it is a property of bf16 and of
the magnitudes this reference produces rather than of the kernel — which is why
it is *reported* by `check_c` and not asserted against a bound. It is printed on
every checked size by `modal_app.py::examples`.

**The alternative was considered and is strictly weaker.** Widening the observed
bf16 back to fp32 and comparing against the unrounded reference within a
tolerance would have needed a tolerance of at least 2⁻⁸ — and a tolerance wide
enough to admit correct rounding admits everything smaller than it too, so a
wrong tile that happened to land close would pass. Rounding the reference keeps
the gate exact.

**What the check is now blind to, stated plainly.** Two fp32 accumulators
differing by less than half an ulp of bf16 round to the same word, so an error
under roughly 0.2% of an element's magnitude is invisible where it was not
before. Every failure this gate exists for — a wrong coordinate, a wrong stride,
a dropped or doubled tile, a wrong operand half, a mis-walked K — moves an
element by far more than that or leaves it at zero, so none of them got harder
to see. The case it can no longer *promise* to catch at every element is a
kernel that accumulated in bf16 rather than fp32; at these reduction depths it
would still fail most elements, but that is an argument and not a guarantee, and
it is the price of the output format.

One thing the shared check buys that is new: cuBLASLt goes through the same
`check_c`, so the baseline is now held to rounding **once, from an fp32
accumulator**. A library that rounded a partial sum would fail there rather than
be timed.

**3. Registers, spill and residency — two of the three as predicted.**

`ptxas -v -arch=sm_100a` through `modal_app.py::regcount`, on the whole examples
artifact:

| entry point | regs | spill st | spill ld | frame |
| --- | ---: | ---: | ---: | ---: |
| `gemm_cg2` | **166** | 0 | 0 | 528 B |
| `gemm_cg2_clc` | 168 | 0 | 0 | 528 B |
| `gemm_cg2_lcsf` | **172** | 0 | 0 | 528 B |
| `gemm_cg2_hot` | 168 | 0 | 0 | 528 B |
| `gemm_cg2_2x` | 168 | 0 | 0 | 528 B |
| `gemm_cg2_2s` | **255** | 0 | 0 | 528 B |
| `gemm_256x256_s2` | 166 | 0 | 0 | 528 B |
| `gemm_256x128_s2` / `s3` / `s4` | 168 | 0 | 0 | 528 B |

**Prediction 3 said 168 and unchanged, and it is 166.** The direction that
mattered held — zero spill, and the occupancy gate reads 166 against a ceiling
of 255 with 89 registers of headroom — but the count moved, in both directions
across the entry points: the shipped kernel is two lower, `lcsf` is four higher.
Six times now the register column has failed to say what it was expected to say
here (#47, #63, #67, #76, #94), and this is the seventh; nothing in the TFLOP/s
column follows it.

A plausible reading of the −2, offered as a reading: the fp32 pair store is
inline PTX carrying `clobber("memory")`, and the bf16 pair store is a plain
4-byte word — `Element::write_pair`'s override needs no asm at all, because two
bf16 *are* one packed word. So the store loop lost a scheduling barrier. That
would also be a candidate mechanism for the +1% in table 1, and neither claim is
established here.

**`gemm_cg2_2s` at 255 is a finding about the probe and it qualifies table 4.**
Holding a band live across two store loops raises peak liveness, which is #63's
"register cost is liveness" arriving exactly where that rule predicts it. It
does not spill, and 255 still admits the two CTAs an SM the kernel is sized for,
so the probe's residency is the kernel's. But a measurement taken at the
architectural ceiling may be paying for the pressure as well as for the stores,
which biases its number **up** — so table 4's split is read as a bound and not
as a partition.

**Residency is 2 CTAs an SM and it is counted, not assumed.** #87's
`tmem residency census` counted 2 at exactly this envelope — 256 accumulator
columns, 98392 B of shared, `wait 0.9 µs` — and this change moves neither term
of `min(512 / columns, shared per SM / plan)`: not one accumulator column and
not one byte of the shared plan. The binding half is still tensor memory,
`512 / 256 = 2`, which is why a smaller `C` could not have moved it. **No new
census rung was run, and that is deliberate** — a rung at an envelope already
counted spends a B200 to reproduce an integer. Prediction 4 confirmed.

**4. Is `stmatrix` reachable now, and what would it be worth.**

**Reachable: yes, and there is no plumbing left to build.** #94 scoped this
route at 250–400 lines on three missing pieces — *"`TensorMapElement` is
`Bf16`-only, there is no fp32 `SharedTile` swizzle, `stmatrix` is b16"* — and
all three were the same fact, which was that `C` was fp32. With a bf16 `C` the
library already holds every piece, and none of them is new work:

| what the route needs | what exists |
| --- | --- |
| fp32 registers → swizzled bf16 shared tile | `kittens::ldst::store_tile`, `stmatrix.m8n8.x2` per `[16, 16]` block |
| the fp32 → bf16 rounding | `Element::pack`, one `cvt.rn.bf16x2.f32` per pair — the same count the direct store already pays |
| shared → global | `SharedTile::tma_store_2d`, `tma_store_commit`, `tma_store_wait_read` (#9) |
| the descriptor | `GlobalLayout::<Bf16, 2>::tensor_map` |

So #107's rejection of the staging route really was conditional on a constraint
this change removes, and the estimate it rejected is now zero.

**Worth: bounded at about 6% of 8192³, and the bound is the interesting part.**
`Epilogue::DoubleDrain` (`2x`) runs the whole epilogue twice per item and
`Epilogue::DoubleStore` (`2s`) runs the LDTM once and the stores twice, both
aiming the extra pass at the cluster's home tile so its bytes stay in L2. Both
compute a wrong `C` on purpose and are excluded from the gate. `2x - lcf` is one
epilogue, `2s - lcf` is one store loop, `2x - 2s` is the LDTM — no fit, and
nothing deleted for a dead-code pass to have an opinion about. Per item on the
critical path:

| shape | lcf ms | 2x ms | 2s ms | epilogue µs | stores µs | LDTM µs | epilogue % of launch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024³ | 0.0268 | 0.0376 | 0.0341 | 10.78 | 7.26 | 3.52 | 40.2% |
| 2048³ | 0.0338 | 0.0464 | 0.0413 | 12.58 | 7.52 | 5.06 | 37.2% |
| 4096³ | 0.1420 | 0.1928 | 0.1744 | 25.39 | 16.18 | 9.22 | 35.8% |
| **8192³** | 0.7440 | 0.8941 | 0.8357 | **21.44** | **13.11** | 8.33 | **20.2%** |
| **16384³** | 5.4765 | 6.2259 | 6.0314 | **26.76** | **19.82** | 6.95 | **13.7%** |

The `2x` column was run twice in two containers and gives 20.74 and 21.44 µs at
8192³ — 3.4% apart, which is this benchmark's own spread.

**Prediction 6 was 5–12 µs a tile at 8192³ and the answer is 21.4. Refuted, by
about a factor of two**, and the way it is wrong is the useful part. 21.4 µs sits
*above* the whole item boundary as #104 fits it (8.6–18.3 µs) — and the epilogue
is only part of that boundary. Both numbers can be right only if the epilogue is
**partly overlapped in the real kernel and not at all in the probe**: the second
epilogue has no loads in flight to hide behind, so `2x - lcf` is the epilogue's
*serial* cost and the fit sees only its *exposed* residue. The pre-registration
argued this bias would run the other way, that a marginal epilogue would be
cheaper than the first. It is dearer.

**Which makes the ceiling arithmetic explicit.** `stmatrix` keeps the LDTM
untouched and halves the stores — 128 pair stores a thread an item become 64
warp-collective `stmatrix.x2`, at the same `cvt` count, with the global write
handed to the TMA engine. So its ceiling is half the `stores` column and none of
the `LDTM` one: **6.6 µs a tile at 8192³ and 9.9 at 16384³, which is 6.2% and
5.1% of those launches.** Under it, `gemm_cg2_2s`'s 255 registers say part of
that 13.11 µs may be register pressure rather than stores, so 6.2% is an upper
bound on an upper bound.

**Against that sits a cost the earlier sections already measured.** At two CTAs
an SM the shared budget is `233472 / 2 = 116736` B against a 98392 B plan, so
**18344 B a CTA is free without moving residency** — and a bf16 `[128, 64]`
strip is 16384 B and fits, where the fp32 version of the same argument (#107)
only reached a `[128, 32]`. A CTA's per-item output is `128 × 256` bf16 =
65536 B, so that is **four single-buffered fill → fence → TMA-store → wait-read
round trips an item**, against eight for fp32. Double buffering wants 32768 B
and still does not fit. The drain would therefore serialize into four phases
inside a window that must close before the item's first MMA, because the
accumulator is single-buffered in tensor memory and a second one is 512 columns
— the whole of it, and one CTA an SM.

**So the direct answer, in one line: reachable with no missing plumbing, worth
at most ~6% at 8192³ and ~5% at 16384³, and it has to buy that out of four
serialized TMA round trips it did not previously have.** That is a real lever
and the largest one currently identified — #102's traversal was 8.4% and is
spent, #87's tile was 11.6–21.6% and is spent — but it is an upper bound with a
known cost against it, and the honest statement is that it is now a
**build-and-measure** question rather than a plumbing one.

**Two things argue it will deliver less than the ceiling, and they should be
read before anyone starts.** `Epilogue::Deferred` moves the entire epilogue
behind a pipeline fill and `Epilogue::HotStore` deletes its HBM traffic, and
both are worth nothing now that there is half as much traffic to delete — both
re-run in each of this session's two epilogue containers, so both columns below
carry their own spread:

| probe | what it moves | at 8192³ | at 16384³ |
| --- | --- | ---: | ---: |
| `lcsf` | *when* the epilogue runs | +1.0% / +1.7% | +1.1% / −0.5% |
| `hot` | *where* its bytes land | +0.5% / −0.0% | −1.9% / −1.0% |
| `2x` | *how many* epilogues run | **−20.2%** | **−13.7%** |
| `2s` | how many *store loops* run | **−12.3%** | **−10.1%** |

Neither of the top two is a store-count experiment, so neither refutes the 6% —
but between them they say the epilogue's *placement* and its *bytes* are both
worth nothing here, and what is left for `stmatrix` is the narrow claim that its
**issue count** is worth something. The bottom two are the first evidence in this
file that it might be: adding epilogue work costs 20% of the launch and adding
half of it costs 12%, so epilogue work is plainly not free, whatever moving it
around fails to buy. (`lcsf` also stops being reliably negative here where #107
had it at zero — +1.0/+1.7 at 8192³ against that section's −0.2/−0.0 — which is
inside both sections' spread and is not a reason to ship it.)

##### and the item boundary was the epilogue, all of it — the ablation

Every section above prices a *part* of this kernel with one of two instruments,
and since #108 they have disagreed. The `gemm-depth` fit takes the intercept of
ms against `K` and puts the item boundary at 8.6–18.3 µs a tile (#104); that
intercept is everything not scaling with `K`, so it sees only what the pipeline
is **exposed** to. `2x` and `2s` run a second epilogue inside an item and price
it at 21.4 µs, split 13.1 stores and 8.3 LDTM — *more than the whole fitted
boundary*, and it can only be, because a second epilogue has nothing in flight
to hide behind and so measures the epilogue's **serial** cost.

#109 read that gap as "most of the epilogue is already overlapped with
something, and which part is exposed is the question this turns on". This asks
it a third way — take a phase **out** of the shipped kernel and measure what the
launch stops costing — and the answer is that **none of it is overlapped**.

**The instrument is a cube, not a ladder.** An item has three phases — the
operand traffic, the multiply, the epilogue — and a rung is any subset, so the
corners of `{loads} × {mma} × {drain}` are the entry points and every edge is
one phase's cost *in one context*. That matters because a ladder that peels
phases off the end prices the epilogue in a kernel that still has its multiply,
and the multiply in a kernel that has already lost its epilogue, and cannot say
which of those attributions is the overlap. Pricing a phase at more than one
corner separates **serial** from **exposed** without assuming either: if the
epilogue costs the same with the multiply beside it and without, it is
overlapped with nothing, whatever a fit says.

Every rung launches on the shipped geometry — `[256, 256] @ STAGES = 3`, static
schedule, `GROUP = 8`, 98 392 B, 2 CTAs an SM, the same grid and item count —
and on the **register** drain, which is what this ladder still decomposes after
#119 moved the default onto `staged84` (same geometry, 114 816 B) —
and `M = N = 8192` throughout, so tiles, waves, `C` bytes and the wave's working
set are identical everywhere and only the arithmetic between two boundaries
moves. **Only `whole` computes a GEMM**; it is checked element-by-element before
it is timed and the rest are `UNCHECKED` in their own labels, the same exception
`hot`, `2x` and `2s` carry.

**What says each rung measures what it names**, which is #101's standard, and
the named hazard is that a deleted drain lets a compiler delete the MMA that fed
it. An opcode census over `kittens_examples.ptx`, per entry function, is the
direct check:

| entry point | `tcgen05.mma` | `tcgen05.ld` | `cp.async.bulk.tensor` | `mbarrier.arrive` | `st.global` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemm_cg2` | 4 | 4 | 2 | 0 | 5 |
| `gemm_cg2_no_drain` | **4** | **0** | 2 | 0 | **0** |
| `gemm_cg2_no_mma` | **0** | 4 | 2 | 0 | 5 |
| `gemm_cg2_loads` | 0 | 0 | 2 | 0 | 0 |
| `gemm_cg2_dry` | 0 | 0 | **0** | **1** | 0 |
| `gemm_cg2_idle` | 0 | 0 | 0 | 0 | 0 |

`no drain` keeps all four chained MMA chunks and loses every LDTM, store and
`cvt`; `no mma` is its mirror; `dry` is the only rung with a bare
`mbarrier.arrive` and the only one with no TMA. Each removes exactly what it
names. Three things corroborate without the PTX: the MMA's commit is what
`done.wait` waits for, so deleting it gives a kernel that hangs rather than a
fast one; `no drain` and `loads` differ only in the MMA and differ by 26 µs a
tile; and `regcount` prices every entry point in the same run. The same table
also reproduces #108's probes from a direction that section did not have — `2x`
carries 8 LDTM and 10 stores, `2s` carries 4 and 10 — which is what those two
claim to be.

**1. The ladder.** Min ms over 30 timed launches, one B200, `M = N = 8192`:

| rung | what it runs | K=512 | K=2048 | K=8192 | K=32768 |
| --- | --- | ---: | ---: | ---: | ---: |
| `whole` | loads + mma + drain | 0.2320 | 0.3308 | 0.7374 | 2.5725 |
| `no drain` | loads + mma | 0.0699 | 0.1494 | **0.5944** | 2.3895 |
| `no mma` | loads + drain | 0.2832 | 0.2743 | 0.5564 | 1.8228 |
| `loads` | loads | 0.0429 | 0.1079 | 0.4122 | 1.7011 |
| `dry` | barriers only | 0.0232 | 0.0523 | 0.1680 | 0.6306 |
| `idle` | an empty item | 0.0123 | 0.0122 | 0.0122 | 0.0122 |

A second session, run in its own container on the four rungs it reached before
it was stopped, reproduces every one of them: 0.7417 / 0.6001 / 0.5596 / 0.4169
at `K = 8192` against 0.7374 / 0.5944 / 0.5564 / 0.4122 — **within 1% on all
four**, which is what makes the ratio below a measurement rather than a session.

**2. Every phase, in microseconds per output tile on the critical path**, as
what the launch stops paying when the phase goes:

| edge | K=512 | K=2048 | K=8192 | K=32768 |
| --- | ---: | ---: | ---: | ---: |
| the epilogue, with the mma | 23.15 | 25.91 | **20.43** | 26.14 |
| the epilogue, without it | 34.33 | 23.78 | **20.59** | 17.39 |
| the mma, with the epilogue | −7.32 | 8.06 | 25.87 | 107.09 |
| the mma, without it | 3.86 | 5.93 | 26.03 | 98.35 |
| the operand traffic, no mma | 2.81 | 7.94 | 34.88 | 152.93 |
| the barrier protocol | 1.57 | 5.74 | 22.27 | 88.34 |
| the floor — an empty item | 1.75 | 1.74 | 1.74 | 1.74 |

As a share of the launch each was taken out of, at `K = 8192`: the epilogue is
**19.4%**, the multiply 24.6%, the traffic 33.1%.

**3. The epilogue, with the multiply beside it and without it** — the pair the
whole design exists to produce, since if the epilogue were serial in the shipped
kernel the two would differ by whatever the multiply was covering:

| shape | with the mma, µs | without it, µs | ratio | hidden |
| --- | ---: | ---: | ---: | ---: |
| 8192x8192x512 | 23.15 | 34.33 | 1.48× | 33% |
| 8192x8192x2048 | 25.91 | 23.78 | 0.92× | −9% |
| **8192x8192x8192** | **20.43** | **20.59** | **1.01×** | **1%** |
| 8192x8192x32768 | 26.14 | 17.39 | 0.67× | −50% |

**At the benchmark's own shape the epilogue costs the same whether the multiply
is there or not, to 1%** — and the same as #108's `2x` measured *serially*
(21.4 µs), by a route sharing none of this one's arithmetic. Three numbers for
one quantity, two of them designed to differ, and they agree.

The two ends of that table are not noise and are worth reading. At `K = 512` the
epilogue is 70% of the launch and removing the multiply makes it *dearer*: with
no arithmetic to space them, 148 clusters run their epilogues at once and
contend. At `K = 32768` it goes the other way, and the honest reading is that
`whole − no drain` at a deep reduction is picking up an L2 term the two rungs do
not share rather than a hiding term — the operand footprint is 1 GiB there and
`no drain` streams it with nothing else in the kernel. **The 8192³ row is the
one to hold, and it is the row every other table in this file is quoted at.**

**4. A fit per rung, which is the decomposition of the fixed cost itself.**
Least squares on raw minimum ms against `K`, all three point selections, the
intercept over the 7 items a critical-path cluster walks:

| rung | 2-point | 3-point | all four | steady TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| `whole` | 18.0 µs | 22.9 | 25.0 | 1798 – 1840 |
| **`no drain`** | **−0.6 µs** | **−0.2** | **1.8** | 1838 – 1853 |
| `no mma` | 19.2 µs | 22.5 | 28.4 | — |
| `loads` | −2.5 µs | −0.8 | 0.5 | — |
| `dry` | 2.0 µs | 2.0 | 2.0 | — |
| `idle` | 1.7 µs | 1.7 | 1.7 | — |

**The `whole` row is `gemm-depth` re-fitted on the bf16 kernel, and it is
18.0–25.0 µs a tile** against #104's 21.5–27.2 on the fp32 one — the re-measure
#109 asked for, and it moves the boundary down by about 2 µs and not more.

**The `no drain` row is the finding.** A kernel with no epilogue has **no fixed
per-tile cost at all**, at every point selection, and its steady state is the
whole kernel's. Differencing the fits along the cube's edges says the same thing
from the other side: the epilogue carries 18.5–23.1 µs a tile of *fixed* cost —
all of it — while the multiply carries −1.2 to +0.5 and the traffic −4.5 to
−1.4, which are zeros with the fit's own noise on them.

**Three findings, and the first retires a decomposition this file has carried
since #86.**

**1. The item boundary is the epilogue, and the rest of it is free.** #90
described the boundary structurally — *"`done.wait`, then `sync_threads`, then
LDTM … then `store_rows`. Only then does `pipeline::run` re-arm the barriers and
the next item issue its first TMA loads, and only then, a full memory latency
later, can the MMA start. Nothing in that chain overlaps with anything."* That
names the right members and gets the weights wrong. Delete the LDTM and the
stores, leave every barrier, boundary and refill exactly where it was, and the
intercept goes to zero. **The barrier re-arm, the two cluster boundaries and the
pipeline refill are all inside the `K`-proportional term** — which is another
way of saying the neighbouring CTA covers them, #98's mechanism seen from a
third direction. The boundary is not a chain of five things to shave one at a
time. It is one thing.

**2. It is exposed, not hidden, and that reconciles #108 against #104 by
making #104 the broken instrument.** #109's reading — the epilogue is *partly*
overlapped and the fit sees its residue — is refuted. Nothing overlaps it, and
the fitted intercept was never a residue: it was the epilogue, attributed to a
"boundary" whose four other members cost nothing.

**3. Which makes `stmatrix`'s ceiling real rather than an overestimate, and
that was the question #109 left open.** That section bounds the route at half
the `stores` column and none of the `LDTM` one — 6.6 µs a tile at 8192³, 6.2% of
the launch — and adds *"if the store half is mostly hidden rather than exposed,
6.2% is an overestimate, and saying so is worth more than the ladder's
headline"*. **It is not hidden.** The epilogue is exposed at 1.01×, so its store
half is exposed too and 6.2% stands. That does not make it a good trade — it is
still an upper bound with four serialized TMA round trips to buy it out of — it
removes the one way it could have been a bad one for free.

**And what it says about the target, which is why this was run.** The framing
this work was set against: the fitted steady state is 1826–1862 TFLOP/s,
cuBLASLt at 16384³ measures 1851, and those being the same number means the
asymptotic rate already matches, so parity is a matter of driving the fixed
per-tile cost to zero. **The asymptote is not an artifact of point selection,
and there is now a version of it that needs no fit at all.** `no drain` *is* the
kernel with its fixed cost driven to zero, and it does not need a line drawn
through it:

| | ours, `whole` | ours, `no drain` | cuBLASLt |
| --- | ---: | ---: | ---: |
| 8192³, min ms | 0.7374 | **0.5944** | 0.6082 |
| 8192³, TFLOP/s | 1491 | **1850** | 1808 |
| `K = 32768`, TFLOP/s | 1710 | **1841** | — |
| % of dense bf16 peak | 66% | **82%** | 80% |

**The kernel with its epilogue removed is 2.3% faster than cuBLASLt's whole
kernel**, measured in the same container minutes apart, flat across a 4× change
in reduction depth, with no extrapolation. The fitted asymptote (1798–1840) and
the measured rung (1841–1850) agree, which is what says the fit was reading a
real ceiling and not a point selection.

**So the gap to cuBLASLt at 8192³ is the epilogue, quantitatively.** Ours is
0.7374 ms, theirs 0.6082, and the epilogue is 0.1430 — **111% of the entire
distance between us**. That is not a proposal to delete it: cuBLASLt writes its
`C` too, so the honest statement is that **the whole of the remaining distance
is the difference between our epilogue and theirs**, and ours costs at least as
much as the gap. Every other term this file has ranked — the traversal (#102,
spent), the tile (#87, spent), the scheduler (#97, ~1%), the store placement
(#107, zero), the epilogue's bytes (#107, ≤1.2%) — lives inside the part of the
kernel that already runs faster than the library does.

**What this cannot show, and one of them is a hole rather than a caveat.**
Three corners of the cube are not built: **every rung that runs the epilogue
with the operand loads switched off fails to return.** Two were launched at
8192x8192x512, produced nothing, and were stopped by hand; their neighbours run
(`dry` is loads-off drain-off and finishes in 23 µs, `no mma` is loads-on
drain-on and finishes at every depth), so the property is `loads = 0` and
`drain = 1` in both MMA states. The first explanation offered for it — an MMA
against never-written shared memory — is refuted by the second hang, which
issues no MMA. **What hangs is not established**, and `Ablation::at` in
`examples/src/gemm.rs` carries the observation rather than a mechanism. What it
costs: the epilogue keeps both contexts the question turns on, so nothing above
depends on the missing corners, but **the operand traffic's exposed cost is not
measured here** — its 33.1% is taken in a kernel with no arithmetic to hide it
and is a serial number. That is the one row of table 2 that should not be
ranked against the others.

The decomposition is also **context-dependent** by construction: an edge
attributes to its phase everything the launch stops paying, including whatever
that phase made other work wait for. That is why each phase is priced at more
than one corner and the numbers are printed rather than averaged. And no rung
separates the epilogue's LDTM from its stores — #108's `2s` is what does that,
and this sweep deliberately does not duplicate it.

**Registers and residency.** `ptxas -v` through `modal_app.py::regcount`:
`gemm_cg2` is **166 registers, 0 spill, 528 B frame — byte-identical to
#109's**, across two structural refactors (the phases became const parameters,
and so did `BLOCK_K`), which is the check that the shipped kernel did not move.
The rungs without an epilogue are **20–28 registers on a zero frame**, so the
epilogue is the whole of both — the eighth time in this file the register column
has been unable to order the times, since residency is tensor-memory-bound at
two CTAs an SM whatever the count. Residency is unchanged and uncounted for the
same reason #108 gives: no accumulator column and no byte of the shared plan
moved, so #87's census row carries by identity.

##### and `BLOCK_K`, which is not a free axis and was already right

`BLOCK_K` was 64 from this file's first commit and no issue had moved it. Two
facts about it turned up before a single rung was built, and both are worth more
than the sweep:

**It cannot be swept as a walk width.** `SharedTile::k_walk` carries
`const { assert!(C * E::BYTES == S::ATOM_BYTES) }` — *"a linear K-major walk
needs K to span exactly one swizzle atom"* — and `Swizzle128B` is the only mode
in tree, so **64 is what a walk *is*** at bf16. A stage that wants more K holds
several atoms and walks each in turn, which is what `Tile::multiply` does now
(`SharedTile::subtile` and `from_raw` were both already public, so this needed
nothing from `src/`), and the descriptor never sees it: a tensor map's box is
`[R, SUBTILE_COLS]` whatever the tile's `C`, so `BLOCK_K` does not reach it.
Going *down* is closed outright — a 32-wide stage needs a `Swizzle64B` that does
not exist.

**And it does not move arithmetic intensity.** A tile reads `(M + N) · K` bytes
to do `2 · M · N · K` flops however K is blocked, so the mechanism #87 found —
the whole of what the pair tile was worth — is not on this axis at all. What
`BLOCK_K` moves is the number of stage barriers, `expect_tx` charges and loop
iterations an item pays, and how coarsely the ring recycles.

**Which makes the sweep a factorization of a fixed budget rather than an axis.**
A stage is `512 · BLOCK_K` bytes at this pair tile, so two CTAs an SM cap
`BLOCK_K · STAGES` at 228 and **every extra atom in a stage is a stage given
back**. `[256, 256] @ k128 s2` — the shipped kernel's 192 K in flight arriving as
two barriers instead of three — is 131 144 B, one CTA an SM, and #98's cliff:
computed, not built.

So the rungs that matter are the ones holding the bytes fixed. Min ms over 30
timed launches, static schedule, `GROUP = 8`, every row checked against the CPU
reference first, `vs #102` against `[256, 128] k64 s3`:

| rung | K in flight | shared B | CTA/SM | 8192³ TFLOP/s | 16384³ TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[256,256] k64 s1` | 64 | 32 824 | 2 | 821.3 | 894.1 |
| **`[256,256] k128 s1`** | **128** | **65 592** | 2 | **1135.1** | **1273.0** |
| **`[256,256] k64 s2`** | **128** | **65 608** | 2 | **1326.4** | **1527.0** |
| `[256,256] k64 s3` — shipped | 192 | 98 392 | 2 | **1479.5** | **1642.7** |
| `[256,256] k128 s2` | 256 | 131 144 | 1 | *not built* | *not built* |

**The predictions, written into the source before the rungs ran, and two of the
four are wrong.**

1. `k64 s1` is a catastrophe — predicted −35% to −55% against the shipped
   kernel. **Confirmed: −44.5% and −45.6%.** One stage is no pipeline.
2. `k128 s1` loses to `k64 s2` at equal bytes, by most of what the load/MMA
   overlap is worth — predicted −20% to −45%. **Refuted in magnitude: −14.4%
   and −16.6%**, the right sign and half the size.
3. `k128 s1` beats `k64 s1` — *this is the `BLOCK_K` measurement*, at fixed
   depth — predicted +5% to +20% from a halved per-K-block fixed cost.
   **Refuted upward, and it is the informative one: +38.2% and +42.4%.**
4. Nothing beats the shipped kernel. **Confirmed**, and by a lot.

**The prediction that was wrong is wrong about the mechanism, and that is the
finding.** +38% is far too large to be barrier issue cost — the `dry` rung of the
ablation above prices the *entire* barrier protocol, walked once per K block, at
22 µs a tile against a 105 µs launch, and halving the count of those cannot buy
38%. What doubling `BLOCK_K` at one stage actually halves is **the number of
times the pipeline is exposed to a load latency**: at `STAGES = 1` the producer
cannot refill until the MMA has released the one buffer, so every K block pays a
full round trip, and a block twice as wide amortizes that same round trip over
twice the arithmetic. `BLOCK_K` is not a fixed-cost lever at all. It is a
latency-amortization lever, and it competes for the same bytes as the pipeline
depth that is a better one.

**Which is the design rule the sweep bought, and it is the pair at 65 KiB that
states it.** `k128 s1` and `k64 s2` are 16 bytes apart in shared memory, at
identical residency, identical tiles, waves, wave reuse and K in flight, and
differ in exactly one thing: one barrier over two atoms against two barriers
over one atom each. **Two shallow stages beat one deep one by 14–17%.** At a
fixed shared budget, spend it on stages and not on stage width — which is why
`BLOCK_K = 64`, the narrowest a walk admits, is the right value here rather than
merely the inherited one. **`BLOCK_K` is swept and the answer is that it was
already right**, for a reason nobody in this file had stated.

**It costs 78 registers and they decide nothing.** `ptxas -v` reads **246 for
`gemm_256x256_k128_s1` against 168 for `gemm_256x256_k64_s1`**, zero spill
either way, and the wider one is 38% *faster*; residency is tensor-memory-bound
at two CTAs an SM at both, so the count cannot cost a CTA. Eighth time in this
file (#47, #63, #67, #76, #94, #100, #109).

**The traversal, re-swept because a new rung is a new item map**, and `GROUP` is
unmoved: at the shipped rung 1450.3 / 1454.2 / **1484.3** / 1482.4 TFLOP/s at
widths 1 / 4 / 8 / 16. Eight and sixteen are 0.1% apart, exactly as #87 found.

**And the denominator, same device, same container, minutes apart:** cuBLASLt
1786.9 TFLOP/s at 8192³ and 1864.0 at 16384³, putting the shipped kernel at
**0.828 and 0.881** — reproducing #108's 0.812–0.825 and 0.851–0.872 at the top
of their bands, which is the control saying the container is not the story.
Nothing in this section moves the shipped kernel: it is the same 166 registers,
the same 98 392 B plan and the same two CTAs an SM, and the two new entry points
are rungs beside it. *(The rung is still shipped. The epilogue on it is not —
#119 moved that to `staged84`, at 80 registers and 114 816 B on the same two
CTAs an SM. Every figure in this section is the register drain's.)*

##### and the epilogue itself, staged through shared memory — #15

#114 left one lever and priced it exactly. The epilogue is the *whole* of the
item boundary, it is **exposed rather than hidden** (`whole − no drain` = 20.43
µs a tile at 8192³ against #108's 21.4 µs measured serially by an unrelated
route, a ratio of 1.01), and the epilogue-free kernel beats cuBLASLt outright —
1850 TFLOP/s against 1808 in the same container. So the epilogue is **111% of
the entire distance to the library** and cutting it pays at close to full value.
#109 split it 13.1 µs of stores against 8.3 µs of LDTM.

`Tile::drain_staged` attacks the store half: TMEM → registers → `stmatrix` into
a per-warp `[32, 64]` bf16 tile → plain 16-byte stores out of it, against the
scattered 4-byte register stores `Tile::drain` issues.
`kittens::ldst::store_tile` and `kittens::global::store_shared_rows` (#113)
already existed; nothing in `src/` moved.

**The arithmetic the issue was scoped from is wrong, and the correction is the
first result.** Per warp band of `[32, 256]` bf16 = 16 384 B, per thread:

| | today, `store_rows` | staged |
| --- | ---: | ---: |
| `st.global` | **128** × 4 B, on 8 discontiguous 16 B runs | **32** × 16 B, on 4 contiguous 128 B runs |
| `stmatrix` | 0 | 64 × 256 B a warp |
| `ld.shared` | 0 | 32 × 16 B |
| `cvt.rn.bf16x2.f32` | 128 | 128 |
| **total memory issue** | **128** | **128** |

The scoping table read 96 and got there by leaving out the `ld.shared` half of
`store_shared_rows`' chunk copy — that mover is deliberately a load *and* a
store rather than one opaque snippet, and both are instructions. **Total issue
does not fall.** The whole hypothesis is the *shape* of the global write: 4×
fewer store instructions, each landing on full 128-byte lines instead of eight
half-filled 32-byte sectors, so the band's global writes go from 1024 half-full
sector transactions to 512 full ones. It is strictly more total work — a whole
extra pass through shared memory — and wins only if that recoalescing is worth
more than 64 `stmatrix` plus 32 `ld.shared`.

**The budget is the tight part and it decided the shape.** At 2 CTAs an SM a CTA
gets 116 736 B and the shipped plan spends 98 392, leaving 18 344. Four warps ×
`[32, 64]` bf16 is 16 384 B and fits; `[·, 128]` is 32 768 and does not, and
`SharedTile::WIDTH_OK` wants a whole swizzle subtile so **64 columns is the
narrowest bf16 tile `Swizzle128B` admits** — the floor and the ceiling meet at
one width. `STAGES` was not available to buy the difference: 3 → 2 is −11.8% /
−7.3% at *unchanged* residency (the `[256,256] s2` row above), which is pipeline
depth and not an occupancy step.

Per *warp* rather than per CTA, which is what keeps the barrier count at zero:
`stmatrix` is `.sync.aligned` and therefore a convergence point for the warp
that issues it, and `store_shared_rows` is cooperative rather than collective,
so a warp writing and reading back its own 4096 B needs no `bar.sync` — only the
`bar.warp.sync` that separates one pass's read from the next pass's write. A
row-subrange of a swizzled tile *is* a swizzled tile (the period is 8 rows and
the XOR is over the row index), which is what lets four of them be carved out of
one run. **And there is no `fence.proxy.async.shared::cta`, deliberately**: that
fence orders a generic-proxy write against an *async*-proxy read, which is what
`epilogue::StoreRing`'s TMA path needs. Both ends here are generic. Which is
also why `StoreRing` was the wrong tool — its `commit` issues a TMA store and
its `acquire` waits on a bulk-copy read group, and this path needs neither.

**Min ms over 30 timed launches, static schedule, `GROUP = 8`, one container.
`lcf` and `staged` both compute the GEMM and both were checked
element-by-element against the CPU reference before they were timed.**

| shape | `lcf` ms | `lcf` TFLOP/s | `staged` ms | `staged` TFLOP/s | vs `lcf` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 0.1413 | 972.8 | 0.1308 | **1050.9** | **+8.0%** |
| 8192³ | 0.7355 | 1494.9 | 0.7057 | **1558.0** | **+4.2%** |
| 16384³ | 5.2947 | 1661.3 | 5.1966 | **1692.7** | **+1.9%** |

**And the epilogue itself, by #114's own subtraction — the launch with the drain
minus the launch without it, at the *same* envelope in each arm.** This is the
measurement the change is really about; the throughput above is what is left of
it after amortization.

| shape | `lcf` µs/tile | `staged` µs/tile | change | share of the launch |
| --- | ---: | ---: | ---: | --- |
| 4096³ | 26.82 | 21.73 | **−19.0%** | 38.0% → 33.2% |
| 8192³ | 20.14 | 16.20 | **−19.6%** | 19.2% → 16.1% |
| 16384³ | 21.60 | 16.34 | **−24.4%** | 11.4% → 8.8% |

The 20.14 at 8192³ re-measures #114's 20.43 to within 1.4%, in a different
container, which is the control saying the instrument is the instrument.

**Was the 6.2% met? No — 4.2% of it was, and the ceiling was derived from a
premise this change does not meet.** 6.2% is half the store half: 13.1 µs of
stores → 6.55 µs saved against a 105 µs tile at 8192³. The measured saving is
3.94 µs, which is **60% of that ceiling and 30% of the whole store half**. The
shortfall is exactly the correction above: the ceiling assumed the stores were
*halved*, and total memory issue does not move at all — only the global half
does. Getting 60% of a ceiling that assumed twice the reduction is the reading,
and it is the one the corrected table predicts.

**The envelope is not the story, and there is a control for it.**
`gemm_cg2_staged_no_drain` declares the staged 114 816 B and runs the
epilogue-free kernel, so it differs from `gemm_cg2_no_drain` in 16 424 bytes
nothing touches: **−0.4% / −0.4% / +1.0%**, which is noise. `device-tests`'
`tmem residency census` **counts 2 CTAs an SM at 114 816 B**, the same integer
it counts at 98 392 — as `min(512 / columns, shared per SM / plan)` predicts,
since at 256 accumulator columns the tensor-memory term binds and shared memory
had 1920 B still to give.

**Against the library, same device, same container:** cuBLASLt 1638.1 / 1819.3 /
1895.4 TFLOP/s, taking the kernel from **0.594 → 0.642**, **0.822 → 0.856** and
**0.877 → 0.893**.

**It costs −124 registers, which is the surprise.** `ptxas -v` reads **42 for
`gemm_cg2_staged` against 166 for `gemm_cg2`**, zero spill either way, on a
smaller frame (256 B against 528). The mechanism is that the band never has to
exist: `TmemTile::tile` and `ldst::store_tile` both walk `[16, 16]` blocks, so
the compiler fuses them and only one `Fragment` — eight values — is live at a
time, where `store_rows` walks the whole band slot-major and forces all 128 fp32
of it into registers at once. The register column has ordered time backwards
eight times in this file (#47, #63, #67, #76, #87, #94, #100, #109); this is the
ninth occasion it says nothing, since residency here is tensor-memory-bound and
the count was never the binding resource.

**Counted in the PTX**, because #114's dead-code hazard applies to any epilogue
rung. `regcount` now carries an opcode census per entry function, so this is a
standing check rather than a paragraph: `gemm_cg2_staged` is the only `gemm_*`
kernel carrying `stmatrix`, `ld.shared.v4` and `st.global.v4` at all, and
`gemm_cg2_staged_no_drain` is opcode-identical to `gemm_cg2_no_drain` — zero
LDTM, zero stores, zero `cvt`, with all four `tcgen05.mma` still there. (The
counts are *static* instructions, so a rolled loop shows as one; the census says
which opcodes a kernel contains, and `Tile::drain_staged` carries the dynamic
arithmetic.)

##### the same epilogue on `gemm_ws`, which is the control that says *why* it wins

The obvious reading of the table above is that the epilogue was simply in the
way — #114 measured it as fully exposed, so anything cheaper on the critical
path shows up. `gemm_ws` is the kernel that separates that from the recoalescing
claim: its epilogue is **already deferred one item and already on warps of its
own**, with the producer never stopping. If staging wins only because the
epilogue was blocking, it should be worth nothing there.

It is worth roughly what it is worth in `gemm_cg2` (separate container, so the
`ws` column moves against §7's other tables; every row checked first):

| shape | `ws` ms | `ws staged` ms | `staged` TFLOP/s | vs `ws` |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1375 | 0.1321 | **1040.4** | **+4.1%** |
| 8192³ | 0.7902 | 0.7713 | **1425.5** | **+2.5%** |
| 16384³ | 5.8878 | 5.7144 | **1539.3** | **+3.0%** |

**So the win is the store shape and not the placement.** Four warps still doing
the same work, still off the critical path, still one item behind, get 2.5–4.1%
from issuing a quarter as many global stores on four times the run length. That
also says the two levers compose rather than substitute: `ws` is +3.0% at 16384³
where `gemm_cg2` is +1.9%, and `ws staged` moves 0.804 → 0.829 of cuBLASLt.

Registers there go **168 → 44**, zero spill, by the same fusion; residency is
unmoved and could not move, since 512 accumulator columns fixed `gemm_ws` at one
CTA an SM before shared memory was consulted and 147 584 B is well inside the
233 472 an SM has.

**Neither kernel's default was changed *in this section*.** `gemm_cg2` and
`gemm_ws` still shipped the register epilogue here and `staged` was a rung
beside them, on the same axis `lcf`/`lcsf` sit on — which is what makes every
number above an A/B and keeps the whole of §7 quotable against one shipped
kernel. On this evidence it is the rung to ship, at both entry points, and that
is a change worth making on its own rather than folded into the measurement
that motivates it. **#119 is that change**, taken after #117 and #118 had
sharpened *which* rung; see "the defaults move" at the end of this section.

##### and the LDTM half, which was the wait and not the issue — #117

#116 cut the store half by ~20% and left the load half exactly as it found it,
which made it the largest identified lever with the epilogue still ~100% of the
gap to the library. Three levers were scoped. **One of them does not exist, one
of them is worth nothing, and one of them is worth more than anything measured
in this file since the port.**

**`.pack::16b` is not a lever — but the register count was the wrong reason, and
#147 closed it properly.** The scoping read it as folding the fp32→bf16 convert
into the load: eliminating the `cvt` and halving the registers reaching
`stmatrix`. At `20a56163f258e09f2c51e4c27ae4e4ff17582443`,
`intrinsics/abi-v1.toml` gives `tcgen05_ld_16x256b_x8_pack16` the result type
`[u32; 32]`, **the same 32 registers as `_x8_raw`**, and
`intrinsics/generated-reference.md` validates it as
`tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 <register-list:32>`.
`.pack::16b` is the load-side twin of `tcgen05.st`'s `.unpack::16b` — it moves
**16-bit-typed** tensor memory, pairing adjacent columns' half-words. Against an
fp32 accumulator it is not a rounding mode, it is the wrong instruction.

That last sentence is the sound half. **"The register count does not fall, so
nothing is packed" is not**, and it answered a question nobody had asked: equal
register counts are consistent with equal *bits* moved, which is exactly why no
convert was performed — it does not establish that the `cvt` instructions
survive. Register count alone is the reasoning this file has been burned by at
#47, #63, #67, #76 and #81, and this was a sixth. #147 asked the two questions
separately and got two answers, both from the census rather than from the ABI:

- **Does it remove the `cvt`? Yes.** `gemm_sol_m512_pack16` censuses
  `cvt.rn.bf16x2` at **0** against the shipped drain's 8, with `stmatrix`,
  `ld.shared.v4`, `st.global.v4` and `st.global.b32` all unmoved. So the convert
  *is* folded away, and the ABI table could never have said otherwise.
- **Is it usable? No, and not for a values reason.** The arm **faults**: one
  launch raised `Xid 13, Out Of Range Address` on every SM and returned
  `DriverError(700)`. Read as 16-bit-typed, the addressing the qualifier implies
  does not land inside a `[128, N]` fp32 allocation at all.
- **Would removing the `cvt` have helped anyway? No.** #147's `nocvt` rung is the
  shipped drain with the same LDTM count, the same wait count, the same eight
  `stmatrix` a band and `cvt` at zero — and it is **0.0 to −1.4% slower**, in two
  round-robin passes at both shapes. The convert is free. The 69% of the drain
  that `twice shared − twice global` prices is the **`stmatrix`** pass and the
  write-after-write a doubled one owes its own staging tile, not the convert
  beside it.

So the conclusion stands and the argument for it is now an instruction census and
a timed oracle. (The rounding question it was originally flagged for is moot
twice over: `check_c` compares bf16 words with `==` and **no tolerance at all**,
so a lever that cost a mantissa bit fails the gate rather than passing a loose
one.)

**The `.x8` hazard is stale history, not a live warning.**
`crates/cuda-device/src/tcgen05.rs` says `tcgen05_ld_16x256b_x4/x8/x16/x32` were
removed as broken — "stored to SMEM instead of returning registers" — while
`generated/tcgen05.rs` exposes `_x4_raw`, `_x8_raw`, `_x16_raw`, `_x32_raw` at
every shape. The removed intrinsics took a shared-memory *pointer*; every
generated variant at this pin is `status = "active"` with an array **result**,
and `_x8_pure` and `_x8_raw` share one LLVM intrinsic
(`llvm.nvvm.tcgen05.ld.16x256b.x8`) and one validated encoding. Checked against
the checkout at `20a5616`, `git rev-parse HEAD` confirmed — not against the
stale `4514af2` in the same cargo directory.

**What no document settles is the register *order*, and that got a case.**
`TmemTile::tile_x8` gets four `[16, 16]` blocks out of one instruction by
asserting the list is repeat-major — repeat `r` is what a `.x1` at `column + 8r`
would have returned. Upstream's own evidence table marks the intrinsic
`runtime: unexecuted`. `device-tests`' **`ldtm x8 map`** is the assertion: it is
`sttm round trip` with one line changed, same seed in, same `observed[i] == i`
out, so a wide load whose registers arrive in any other order returns a
permutation and the host names both the coordinate that should own each value
and the one that wrote it. It passes on a B200 — `16384 registers survived TMEM
unchanged`, the same line and the same count as the `.x1` case beside it — and
39 of 39 device cases pass. **Repeat-major is what the silicon does**, and it is
now a standing check rather than an inference.

**Counted in the PTX, and the two levers land exactly where they were aimed.**
Every other column is identical across all four rungs — the global half is
untouched, which is what makes this an ablation of the *other* half rather than
a second pass at #116's.

| kernel | ldtm | stmatrix | mma | tma | ld.sh.v4 | st.g.v4 | st.g.b32 | cvt.bf16x2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_cg2_staged` | 8 | 4 | 4 | 2 | 1 | 1 | 4 | 8 |
| `gemm_cg2_staged_x8` | **2** | 4 | 4 | 2 | 1 | 1 | 4 | 8 |
| `gemm_cg2_staged_x4` | 8 | **2** | 4 | 2 | 1 | 1 | 4 | 8 |
| `gemm_cg2_staged_x8x4` | **2** | **2** | 4 | 2 | 1 | 1 | 4 | 8 |

**Min ms over 30 timed launches, static schedule, `GROUP = 8`, one container.
All four rows declare the same 114 816 B — unlike #116's A/B, which had a 16 424
byte envelope change to price first — so one `no drain` control serves all four
and there is no envelope table here. Every row computes the GEMM and was checked
element-by-element before it was timed.**

| shape | `staged` ms | `staged8` ms | `staged4` ms | `staged84` ms | `8` vs | `4` vs | `84` vs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 0.1283 | **0.1038** | 0.1291 | 0.1043 | **+23.6%** | −0.6% | +23.1% |
| 8192³ | 0.7164 | 0.6599 | 0.7186 | **0.6585** | +8.6% | −0.3% | **+8.8%** |
| 16384³ | 5.3677 | 5.1804 | 5.4263 | **5.1061** | +3.6% | −1.1% | **+5.1%** |

| shape | `staged` TF/s | `staged8` TF/s | `staged4` TF/s | `staged84` TF/s |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 1071.3 | **1324.4** | 1064.7 | 1318.3 |
| 8192³ | 1534.8 | 1666.2 | 1530.1 | **1669.8** |
| 16384³ | 1638.7 | 1697.9 | 1621.0 | **1722.7** |

**And the epilogue itself, by #114's subtraction, which is the measurement this
is really about.**

| shape | `staged` µs/tile | `staged8` µs/tile | `staged4` µs/tile | `staged84` µs/tile | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 21.18 | 8.93 | 21.58 | 9.17 | **−57.9%** |
| 8192³ | 14.96 | 6.89 | 15.27 | 6.68 | **−55.3%** |
| 16384³ | 15.15 | 8.47 | 17.25 | 5.81 | **−61.7%** |

**#109's split is confirmed by an independent route, four issues later.** #109
put the epilogue at 13.1 µs of stores against 8.3 µs of LDTM at 8192³. #116 cut
the store half and said it did not touch the load half; this section removes the
load half and the arithmetic closes: `staged − staged8` at 8192³ is
**14.96 − 6.89 = 8.07 µs a tile**, against #109's 8.3. Two different
instruments, two different containers, four issues apart, 2.8% apart. The
epilogue after #116 was 14.96 µs and **54% of it was LDTM**.

**The mechanism is the wait, not the issue — which is why `.x4` is worth
nothing.** `TmemTile::fragment` waits after *each* of its two `.x1` loads,
because the registers a load waits on are its return value, so the shipped drain
never has two LDTM in flight and pays the full tensor-memory latency per four
values: a `[32, 64]` band is 16 loads and 16 fully exposed latencies. `.x8` is 2
and 2. **`stmatrix.x4` halves an instruction count and buys nothing** —
−0.6% / −0.3% / −1.1%, a null result with a consistent sign — which is the same
lesson #116 learned from the other direction: total issue is not what the
epilogue is paying for. This file has now measured that twice, once per half.

**The two compose only where liveness is the binding term, and the register
column is why.** `ptxas -v`, zero spill everywhere, frame 256 B throughout:

| kernel | registers | spill |
| --- | ---: | ---: |
| `gemm_cg2_staged` | 42 | 0 |
| `gemm_cg2_staged_x4` | 40 | 0 |
| `gemm_cg2_staged_x8` | **94** | 0 |
| `gemm_cg2_staged_x8x4` | **80** | 0 |

`.x8` costs +52 registers, which is exactly the 32 f32 that arrive at once where
the `.x1` path let the compiler fuse one eight-value `Fragment` through to the
store. It does **not** spill, and 94 is far inside the step — which for this
kernel is 255 and not the 168 quoted elsewhere in this file, since #87 gave up
the third CTA and two CTAs at 128 threads admit the whole register file —
residency here is tensor-memory-bound at 256 accumulator columns and the count
was never the binding resource, which is the tenth occasion this file has said
so. What `.x4` then buys is liveness rather than issue: it consumes all four of
a fragment's matrices at once, taking the composed rung from 94 back to 80, and
that is the whole of why `staged84` beats `staged8` at 16384³ (+5.1% against
+3.6%) while being a wash at 8192³ and marginally behind at 4096³. A lever worth
nothing alone is worth something in composition, and the register column is what
says which.

**Against the library, same device, same container:** cuBLASLt 1655.7 / 1768.5 /
1869.1 TFLOP/s.

| shape | `staged`/theirs | `staged8`/theirs | `staged4`/theirs | `staged84`/theirs |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.647 | **0.800** | 0.643 | 0.796 |
| 8192³ | 0.868 | 0.942 | 0.865 | **0.944** |
| 16384³ | 0.877 | 0.908 | 0.867 | **0.922** |

**0.856 → 0.944 at 8192³ and 0.893 → 0.922 at 16384³**, reading #116's ratios
across containers and this section's within one. It is the largest single step
in this table since the kernel was ported.

**Neither kernel's default was changed *in this section***, on #116's own
precedent: `staged84` is a rung beside `staged`, every number above is an A/B,
and shipping it is a change worth making on its own rather than folded into the
measurement that motivates it. **#119 made it** — `staged84` is `gemm_cg2`'s
default now, on exactly the evidence in this table. The same two widths
are untried on `gemm_ws`, whose staged rung carries twice the LDTM (16 against
8) for the same reason its bands are twice as wide. *(The second half of that
sentence is wrong and #118 measured why — the bands are the same width and the
doubling is two call sites in the PTX. See below.)*

##### the same two widths on `gemm_ws`, and the floor underneath them — #118

`gemm_ws` had never had `.x8`, so on instruction count it was the largest lever
left in the repo. It is worth a separate section because the *mechanism* #117
established makes a prediction about this kernel that the instruction count does
not: if the win is the **wait** — `TmemTile::tile` waits after each `.x1`, so a
`[32, 64]` band pays 16 fully exposed tensor-memory latencies against `.x8`'s 2
— then removing a latency somebody is already covering must be worth *less*.
`gemm_ws` covers it: the drain is deferred one item, sits on four warps of its
own, and the producer never stops.

**One container, min of 30 timed launches, static schedule, `GROUP = 8`, every
row element-by-element exact against the CPU reference before it was timed
(`check_c` compares bf16 words with `==` and no tolerance). All four staged rows
declare the same 147 584 B, so one `no drain` control serves them all; `ws s4`
is the register epilogue this kernel shipped through #119, at 131 176 B.**

| shape | `ws s4` ms | `staged` ms | `staged8` ms | `staged4` ms | `staged84` ms | `8` vs | `4` vs | `84` vs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 0.1382 | 0.1340 | **0.1158** | 0.1324 | 0.1171 | **+15.7%** | +1.2% | +14.5% |
| 8192³ | 0.7881 | 0.7649 | 0.7201 | 0.7648 | **0.7199** | +6.2% | +0.0% | +6.2% |
| 16384³ | 5.7904 | 5.6645 | **5.5015** | 5.6604 | 5.5028 | +3.0% | +0.1% | +2.9% |

| shape | `staged` TF/s | `staged8` TF/s | `staged4` TF/s | `staged84` TF/s | best vs `ws s4` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 1025.8 | **1187.1** | 1038.4 | 1174.1 | **+19.4%** |
| 8192³ | 1437.5 | 1526.8 | 1437.7 | **1527.2** | **+9.5%** |
| 16384³ | 1552.8 | **1598.9** | 1554.0 | 1598.5 | **+5.2%** |

**The prediction holds.** `.x8` is worth +15.7% / +6.2% / +3.0% here against
+23.6% / +8.6% / +3.6% on `gemm_cg2` — 67%, 72% and 83% of it — and the
shortfall is the cover this design point provides. `.x4` is the same
clean null it was there (+1.2% / +0.0% / +0.1%), and **the two do not compose**:
`staged8` and `staged84` are within 1.1% of each other and trade places between
sessions, where on `gemm_cg2` composition was worth a further 1.5 points at
16384³.

**The register column says why they do not compose, and it says it in advance.**
`ptxas -v`, zero spill everywhere, frame 256 B in every staged rung:

| registers | `staged` | `staged8` | `staged4` | `staged84` |
| --- | ---: | ---: | ---: | ---: |
| `gemm_cg2_staged*` | 42 | 94 | 40 | **80** |
| `gemm_ws_staged*` | 44 | 94 | 44 | **92** |

`.x8` costs the same +50 either side. On `gemm_cg2`, `.x4` bought 14 of those
registers back and that recovery *was* the composition gain; here it buys 2, and
buys no time to go with them. Nothing was ever near a ceiling — at six warps the
binding sub-partition holds two, so 255 a thread is reachable, and the largest
count in the file is 94.

**What the epilogue costs, by #114's `whole − no drain`, in µs a tile over the
items the busiest cluster walks.** Two controls, one per envelope; they are
opcode-identical PTX at 28 registers apiece and differ in 16 408 declared bytes
no instruction touches, which is why the staged columns all subtract the staged
one.

| shape | `ws s4` | `staged` | `staged8` | `staged4` | `staged84` | LDTM half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 8.90 | 8.65 | 4.10 | 8.24 | 4.42 | 4.55 |
| 8192³ | 8.62 | 6.97 | 3.77 | 6.96 | 3.76 | **3.20** |
| 16384³ | 9.06 | 6.76 | 3.85 | 6.69 | 3.87 | 2.91 |

**"Twice the LDTM" is true of the PTX and false of the work.** `regcount`'s
opcode census does read 16 `tcgen05.ld` for `gemm_ws_staged` against
`gemm_cg2_staged`'s 8 — and 8 against 4 for the register rungs, 8 `stmatrix`
against 4, 16 `cvt.rn.bf16x2` against 8, *every* column doubled, which is the
tell. The cause is that `gemm_ws` emits its epilogue at **two call sites**,
`Ws::work` and `Ws::finish`, where `gemm_cg2`'s fused job has one; `finish`
drains the last item once per cluster over a whole launch. The band is
`RegTile<32, 64>` in both files, so the *dynamic* LDTM per tile is identical —
and the subtraction says this kernel's LDTM half costs **3.20 µs a tile at
8192³ against `gemm_cg2`'s 8.07**, 2.5× cheaper rather than twice as dear.
The same holds for the whole epilogue: 8.62 µs exposed here against #114's
20.43, and 6.97 staged against #117's 14.96. **That 40-47% is what warp
specialization and the accumulator ping-pong buy, and it is the first time it
has been a number.**

**The floor, which is the finding.** `gemm_ws_no_drain` is the shipped kernel
with the epilogue deleted — the probe #114 ran on `gemm_cg2`, where it reached
1850 TFLOP/s against cuBLASLt's 1808 and established that that kernel's whole
gap to the library *was* its epilogue.

| shape | `ws` no drain ms | TFLOP/s | of cuBLASLt | vs `gemm`'s `staged84` |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1026 | 1339.2 | 0.829 | +3.0% |
| 8192³ | 0.6674 | 1647.4 | 0.923 | **−2.2%** |
| 16384³ | 5.2828 | 1665.1 | 0.888 | **−3.0%** |

**At 8192³ and 16384³ `gemm_ws` loses to `gemm`'s complete launch while running
no epilogue at all.** So the answer to "does `gemm_ws` now beat `gemm`" is no,
and the more useful answer is that it could not have: with each design point at
its best it is **−9.7% / −9.4% / −6.9%**, and deleting this kernel's epilogue
outright recovers 12.7 / 7.2 / 3.9 of those points. Epilogue work could reach
three quarters of the 8192³ gap and 57% of the 16384³ one *if it were free*.
#112's 7.3% at 16384³ closes to 6.9% after both kernels have spent everything
#116 and #117 found.

**Against the library, same device, same container:** cuBLASLt 1616.5 / 1784.9 /
1876.0 TFLOP/s. The `gemm 84` column reproduces #117's 0.796 / 0.944 / 0.922
from another session, which is the drift control.

| shape | `gemm` | `gemm 84` | `ws s4` | `ws staged` | `ws staged84` | `ws` floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 0.595 | 0.804 | 0.615 | 0.635 | **0.726** | 0.829 |
| 8192³ | 0.831 | 0.944 | 0.782 | 0.805 | **0.856** | 0.923 |
| 16384³ | 0.871 | 0.915 | 0.810 | 0.828 | **0.852** | 0.888 |

**Residency is counted, not queried**, because
`cuOccupancyMaxActiveBlocksPerMultiprocessor` returns 1 for anything containing
a `tcgen05.alloc` whatever the truth is (#77). `device-tests`' residency census
gained two rungs at `gemm_ws`'s own envelopes and counts **1 resident, 1
holding** at both 131 176 B and 147 584 B, against a budget of 1 from
`min(512 / 512, 233472 / plan)`. That also closes the gap the `cg2 512` rung
left: at a 32 B plan it counts 1 holding but **2 resident**, with a second CTA
parked 100.9 µs inside a blocking allocator; at the real plan the second CTA is
never admitted and the wait falls to 0.9 µs.

**Neither kernel's default was changed *in this section***, on #116's and
#117's precedent. The rung to ship on `gemm_ws` is `staged8` — `.x4` is a null
here, alone and in composition, and a lever that buys 2 registers is not worth
a second entry point's worth of surface if one has to be chosen. **#119 shipped
it**, and shipped `staged84` on `gemm_cg2`, which is the one place the two
files are deliberately different.

**What is left, and it is not the epilogue.** #112 attributed this kernel's loss
to the peer CTA hiding `gemm_cg2`'s epilogue; #114 refuted that by measuring
that epilogue as fully exposed. `gemm_ws.rs` then argued the opposite — that an
SM holding one CTA has nothing to hide *its* epilogue behind — and this refutes
that too, by deleting the epilogue and losing anyway. Priced against the library
(the only denominator that crosses containers) the two epilogue-free kernels are
about 10% apart at 8192³: 1.02 of cuBLASLt for `gemm_cg2`, 0.923 for `gemm_ws`.
The deficit is in the multiply and the operand stream at one CTA an SM, and what
has never been measured is what an SM running one CTA of six warps does to the K
pipeline against two CTAs of four. That is where the next probe belongs, and
`kittens::epilogue::StoreRing` (#111) no longer has "about 7% to find" on this
kernel — it has at most 3.0%.

##### the defaults move, and they move to different rungs — #119

The three sections above each end by saying the rung was worth shipping and
that shipping it belonged in its own change. This is that change, and it is the
whole of it: **`gemm_cg2`'s default epilogue becomes `staged84` and
`gemm_ws`'s becomes `staged8`.** No rung was added, none was deleted, nothing
was retuned; every entry point that existed before still launches and is still
on the correctness gate.

**They ship different rungs, and that is a measurement rather than an
oversight.** #117's and #118's register column, reproduced by
`scripts/modal-run regcount` on the tree that ships — zero spill everywhere,
frame 256 B in every staged rung:

| kernel | `staged` | `staged8` | `staged4` | `staged84` |
| --- | ---: | ---: | ---: | ---: |
| `gemm_cg2_staged*` | 42 | 94 | 40 | **80** |
| `gemm_ws_staged*` | 44 | 94 | 44 | 92 |

`.x8` costs +52 and +50, identically. What differs is what `.x4` hands back:
**14 registers on `gemm_cg2` and 2 on `gemm_ws`**, and that recovery *is* the
composition gain — there the pair is +5.1% at 16384³ against `.x8` alone's
+3.6%, here `staged8` and `staged84` land inside 1.1% and trade places between
sessions. So the composed rung wins where the recovery is real and the tie goes
to the rung with less surface where it is not. Tidying the two files onto one
choice would be discarding exactly what the two sections above measured.

**The confirming run — one container, min of 30 timed launches, every row
checked element-by-element first, and cuBLASLt re-measured beside them rather
than quoted** (#109 would have published a false +3.6% had it not noticed its
own commit moved the baseline). cuBLASLt is 0.0836 / 0.6275 / 4.8020 ms here,
which is 1645.0 / 1752.2 / 1831.8 TFLOP/s.

| shape | `gemm` was (`lcf`) | `gemm` ships (`staged84`) | gain | of cuBLASLt |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1432 | **0.1044** | **+37.2%** | 0.583 → **0.801** |
| 8192³ | 0.7584 | **0.6684** | **+13.5%** | 0.827 → **0.939** |
| 16384³ | 5.5157 | **5.0132** | **+10.0%** | 0.871 → **0.958** |

| shape | `gemm_ws` was (`ws s4`) | `gemm_ws` ships (`staged8`) | gain | of cuBLASLt |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1395 | **0.1166** | **+19.6%** | 0.599 → **0.717** |
| 8192³ | 0.7947 | **0.7310** | **+8.7%** | 0.790 → **0.858** |
| 16384³ | 5.9157 | **5.5663** | **+6.3%** | 0.812 → **0.863** |

in milliseconds. The `gemm` column composes #116 and #117 in one step, which is
why it is larger than either section's headline. Against the published ratios:
#117's 0.796 / 0.944 / 0.922 for `staged84` reproduces at 0.801 / 0.939 /
0.958, and #118's 0.726 / 0.856 / 0.852 for `ws staged8` at 0.717 / 0.858 /
0.863. The 16384³ row is the one that moved, and the re-measured control is
what says why: the library ran at 1831.8 TFLOP/s here against #117's 1869.1,
and our own launch was 1.8% quicker. **0.958 of cuBLASLt is the closest this
project has been.**

`staged8` against `staged84` on `gemm_ws` in this session: 0.1166/0.1154,
0.7310/0.7299, 5.5663/**5.5850** — `staged84` marginally ahead at the two small
sizes and behind at the largest, the largest gap being 1.0%. That is the third
session to land inside the tie and the second to order it differently, which is
the evidence the shipped rung was chosen on.

**What moved with the default, since a constant describing the shipped launch
has to.** `gemm`'s launch declares **114 816 B** where it declared 98 392 and
`gemm_ws`'s declares **147 584 B** where it declared 131 176; `main.rs`'s
envelope table reports those, and the register plans are still asserted and are
now what the *control* arms declare. `CTAS_PER_SM` does **not** move on either
kernel and could not: residency is `min(512 / columns, shared per SM / plan)`
and the tensor-memory term binds at both envelopes — 2 an SM for `gemm` at 256
accumulator columns, 1 for `gemm_ws` at 512 — which `device-tests`' census has
already counted at all four.

The occupancy-step gate gained the shipped kernel as a third watched entry
point, because that is the row that would go red first: `.x8` is the largest
single liveness step any epilogue rung in this tree has taken. It is green, and
the room is not the 168 this repo quoted for years — #87 gave up the third CTA,
and two CTAs at 128 threads admits the whole register file.

| kernel | threads | wants CTAs | regs | ceiling | headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemm_cg2` | 128 | 2 | 166 | 255 | 89 |
| `gemm_cg2_staged` | 128 | 2 | 42 | 255 | 213 |
| `gemm_cg2_staged_x8x4` | 128 | 2 | **80** | 255 | 175 |

The opcode census is unchanged rung for rung: `gemm_cg2_staged_x8x4` still
reads 2 `tcgen05.ld`, 2 `stmatrix`, 4 `tcgen05.mma`, 1 `ld.shared.v4`, 1
`st.global.v4` and 8 `cvt.rn.bf16x2`, and `gemm_ws_staged_x8` still reads its
doubled counts from the two call sites `Ws::work` and `Ws::finish`.

**What deliberately did not move.** The ablation cube, `bench --case epilogue`
and `bench --case staged` are all built at `lcf` and stay there: they decompose
the *register* drain, #114's 20.43 µs a tile is the figure they reproduce, and
a ladder that moved with the default would stop being comparable to every
earlier run of itself. `lcf` is also the only epilogue with a `clc` entry
point, so the scheduler A/B keeps it. And **table 1 of `ws_bench` now names
`lcf` on both sides instead of taking `gemm`'s default** — its one variable is
where the overlap comes from, and a default moving under it would have made it
two. It still reads −4.6% and −6.8% at the two large sizes, reproducing #112.

**Every `bench --case gemm*` row published before this change is against a
different kernel.** That includes the `gemm-depth` fit whose intercept ranks
everything in the ablation section and the whole-size sweep above. Those rows
are kept and labelled rather than re-run: re-fitting `gemm-depth` on `staged84`
is a measurement of its own and this change is not it.

**Correctness, before any of the above was quoted.** `scripts/modal-run
examples` passes every rung of both kernels at both check sizes and all three
traversal widths, under both schedulers wherever a rung has both — `==` on bf16
words with no tolerance at all, so a lever costing a mantissa bit fails here
rather than places. The pass line now names which rung ships, which is the
cheapest place for that fact to live:

```
512x256x256 exact on staged8, staged4 and staged84 at groups [1, 3, 6] (staged84 ships, same 114816 B)
4096x4096x256 exact on ws staged, ws staged8, ws staged4, ws staged84 (147584 B, ws staged8 ships)
```

##### and the whole ranking, re-derived in two sessions — the ceiling is an 8192³ fact

Everything above this line that orders the remaining levers was assembled out of
three containers. #114 measured an epilogue-free `gemm` at 1850 TFLOP/s against
cuBLASLt's 1808 — a **ratio of 1.02**, and a ceiling rather than a result, since
a kernel with no epilogue cannot be moved by epilogue work. #116 and #117 then
took the epilogue from 20.43 µs a tile to ~6.9. `staged84` sits at 0.944, so on
that arithmetic the residual is worth about 7.7%, removing it would put the
kernel *above* the library, and **the gap is still 100% epilogue**.

Every number in that chain was taken somewhere else, and the 1.02 predates both
of the changes that moved the thing it is a ceiling on. This re-runs the
controls beside the arms — `bench residual`, two containers, five sizes — and
the conclusion survives at one size and weakens at the others. **1.02 is a fact
about 8192³.** It is 0.98 at 16384³ and 0.98 at 4096³, and below 2048³ there is
no denominator stable enough to quote.

**1. The ceiling, the shipped rung and the library, all in one container, twice.**
Min ms over 30 timed launches, static schedule, `GROUP = 8`, the shipped
`[256, 256] @ k64 s3` rung; `s84nd` is `gemm_cg2_staged_no_drain` at 114 816 B
and `lcfnd` is #114's own `gemm_cg2_no_drain` at 98 392, neither of which
computes a GEMM. Both sessions, `session 1 / session 2`:

| shape | `s84`/theirs | `s8`/theirs | `s84nd`/theirs | `lcfnd`/theirs |
| --- | ---: | ---: | ---: | ---: |
| 1024³ | 0.564 / 0.650 | — / 0.639 | 0.793 / 0.868 | 0.793 / 0.894 |
| 2048³ | 0.728 / 0.829 | — / 0.834 | 0.916 / **1.038** | 0.893 / **1.046** |
| 4096³ | 0.791 / 0.811 | 0.795 / 0.815 | 0.955 / 0.988 | 0.944 / 0.977 |
| **8192³** | **0.940 / 0.939** | 0.941 / 0.944 | **1.024 / 1.018** | **1.019 / 1.017** |
| 16384³ | 0.937 / 0.932 | 0.949 / 0.927 | **0.993 / 0.984** | 0.979 / 0.983 |

**At 8192³ the framing holds and holds twice.** The epilogue-free kernel is
1.017–1.024 of cuBLASLt, which reproduces #114's 1.02 in two fresh containers
after both of the changes that could have invalidated it. The gap is 6.5% of our
launch (0.6549 against 0.6149); the epilogue is 7.2–7.4 µs a tile over the 7
items a critical-path cluster walks, which is **127–136% of it**. Nothing else
needs to be found at this size: the whole distance to the library is still the
difference between our epilogue and theirs.

**At 16384³ the ceiling is below parity, and that is new.** 0.984 and 0.993 —
the epilogue-free kernel *loses* by 1–2%, where at 8192³ it wins by 2%. Taking
the epilogue out of that gap leaves 10–23% of it standing, which is 0.7–1.6% of
the launch. It is small, and the honest reading of the pair is not "a new term
appeared" but **"the two asymptotes are the same number, and which one is 1%
ahead depends on the size"** — which is what #114's own fit said (1798–1862
against the library's 1851) before a single measured rung was quoted for it.

**At 4096³ it is 0.977–0.988**, so the epilogue is 82–95% of that gap and the
rest is the wave: 256 tiles over 148 clusters is two waves at 86%, a mechanism
that has nothing to do with anything above and is reported separately for that
reason. `s84`/theirs is 0.79–0.82 there against 0.94 at 8192³, and the
twelve-point difference is **not** an epilogue that got dearer — the epilogue
costs 8.7–9.0 µs a tile at 4096³ against 7.2–7.4 at 8192³, a fifth more, over a
launch six times shorter.

**2. And the denominator is not one number, which is why the rows above stop at
4096³.** cuBLASLt timed exactly as our own rungs are — same harness, same clock,
same `min` of 30 — with a second independent call in the same session (new
buffers, new handle, new heuristic query) and the two sessions against each
other:

| shape | max/min in one call | call to call, one session | session to session |
| --- | ---: | ---: | ---: |
| 1024³ | 1.26 / 1.84 | 1.10 / 1.08 | +16% |
| 2048³ | 1.10 / 1.16 | **1.26 / 1.23** | +15% |
| 4096³ | 1.08 / 1.09 | 1.01 / 1.00 | +2.6% |
| 8192³ | **1.01 / 1.01** | **1.00 / 1.00** | −1.1% |
| 16384³ | 1.15 / 1.19 | 1.00 / 1.00 | −1.0% |

**Its *minimum* moves 23–26% between two calls at 2048³** and 0.0–0.3% at
8192³. So a ratio at 8192³ is good to a tenth of a point and a ratio at 2048³ is
not a measurement at all — which is what the 1.038 in table 1 is, and why it is
printed rather than celebrated. The 1.15–1.19 max/min at 16384³ is the same
warning in a different place: the library's *median* there is 5.42 ms against a
4.71 minimum, so `min` is doing real work in that row, and every row of this
file quotes `min` on both sides for exactly that reason.

##### what is inside the residual epilogue, by doubling rather than deleting

#117 named the LDTM half and removed most of it. What is left is three things
and nothing had priced them apart. **Subtraction cannot do it**: a rung that
keeps the LDTM and drops the stores is a rung whose result a dead-code pass
decides, which is the hazard [`Epilogue::DoubleDrain`]'s own doc states and the
reason #108 built a doubling probe rather than a subtractive one. So
`Tile::drain_staged_twice` **adds** one link of the chain at a time and prices
the added one:

| rung | the second pass runs | difference from the rung above |
| --- | --- | --- |
| `staged84` | — | — |
| `s84 2g` | `store_shared_rows` | `ld.shared` + `st.global.v4` |
| `s84 2m` | `stmatrix` + `store_shared_rows` | the `cvt` + `stmatrix` pass and its `bar.warp.sync` |
| `s84 2x` | LDTM + `stmatrix` + `store_shared_rows` | the LDTM, issue and wait together |

**Counted in the PTX**, per entry function, because #114's dead-code hazard
applies to any epilogue rung and this is four of them. Every column that should
be held is held, and every column that should step, steps by exactly one pass:

| kernel | ldtm | stmatrix | ld.sh.v4 | st.g.v4 | st.g.b32 | cvt.bf16x2 | bar.warp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_cg2_staged_x8x4` | 2 | 2 | 1 | 1 | 4 | 8 | 1 |
| `gemm_cg2_staged_x8x4_2g` | 2 | 2 | **2** | **2** | **8** | 8 | 1 |
| `gemm_cg2_staged_x8x4_2m` | 2 | **4** | 2 | 2 | 8 | **16** | **2** |
| `gemm_cg2_staged_x8x4_2x` | **4** | 4 | 2 | 2 | 8 | 16 | 2 |
| `gemm_cg2_staged_x8x4_hot` | 2 | 2 | 1 | 1 | 4 | 8 | 1 |

The `ldtm` step from 2 to 4 is worth naming on its own. `tcgen05_ld_..._x8_pure`
is spelled *pure*, and the second load reads a tensor memory nothing wrote in
between, so a common-subexpression pass would have been entitled to fold it away
and take the whole third rung with it. It is there. (`_hot` is opcode-identical
to the rung it probes, which is what makes it a bandwidth measurement rather
than an instruction one, and the `2m` row shows why the second column is called
`cvt` + `stmatrix` and not `stmatrix`: the `cvt` doubles with it, because
`Element::pack` is inside the `stmatrix` pass.)

**Microseconds a tile on the critical path, both sessions.** Each column is what
the *added* pass costs, so these are **serial** costs, with the second pass's
stores aimed at the cluster's own first tile so its bytes stay in L2 — #108's
arrangement. `exposed` is #114's `whole − no drain` on the same rung in the same
table:

| shape | `ld.shared`+`st.global` | `cvt`+`stmatrix` | LDTM | serial total | exposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 2.99 / 3.46 | **4.48 / 4.08** | 0.53 / 0.43 | 8.00 / 7.97 | 9.01 / 8.72 |
| **8192³** | 1.96 / 2.32 | **3.84 / 4.13** | **0.09 / −0.04** | 5.89 / 6.41 | 7.37 / 7.21 |
| 16384³ | 0.45 / 4.22 | 3.15 / 2.89 | 4.01 / −2.69 | 7.60 / 4.42 | 7.39 / 5.10 |

**Four findings, and the first is the ranking.**

**1. The largest thing left is the `cvt` + `stmatrix` pass, at 3.8–4.5 µs a
tile.** It is 55–64% of the serial ladder, it is the same ~4 µs at 4096³ and at
8192³ and in both sessions, and it is the one component with no lever pointed at
it. `.x4` already halved the `stmatrix` *count* and bought −0.3% (#117);
`.pack::16b` is the wrong instruction against an fp32 accumulator (#117); and a
wider staging tile is closed by the shared budget, since `[·, 128]` is 32 768 B
against the 18 344 the plan leaves, and 64 columns is already the narrowest bf16
tile `Swizzle128B` admits — #116's floor and ceiling still meet at one width.
The `cvt` inside it is not optional at all: a bf16 `C` has to be rounded
somewhere, and `store_rows` pays the same 128 of them per band.

**2. The LDTM half is exhausted.** −0.04 to +0.53 µs a tile at the two sizes the
instrument is trustworthy at: after `.x8` a *second whole LDTM pass* is free.
#117 said the cost was the wait and not the issue, and this is that statement
from the other side — with the waits gone there is nothing left to remove, and
the two `tcgen05_load_wait`s a band no longer have exposed latency in them worth
attacking.

**3. The ladder accounts for 80–91% of the exposed epilogue** (7.97 against
8.72, 5.89 against 7.37, 6.41 against 7.21). Two readings are available and this
sweep does not separate them: a serial-to-exposed ratio near 1 is #114's result
reproduced on the staged rung — the epilogue is still overlapped with nothing —
and the 9–20% shortfall is either that overlap or a cold cost the first pass
pays and the doubled pass does not. It is quoted as a residual and not
attributed.

**4. And the 16384³ row is not a measurement**, which is the finding that
reaches backwards. The two sessions put the LDTM half at +4.01 and −2.69 µs a
tile. Table 6 of the sweep is what explains it, and it is a control this file has
never taken: **the same subtraction, twice in one container**, on rungs that were
timed minutes apart.

| shape | `s84`, table 1 / table 3 | `s no drain`, table 1 / table 3 | µs/tile | spread |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 0.1052 / 0.1043 | 0.0864 / 0.0869 | 9.41 → 8.72 | **−7%** |
| 8192³ | 0.6549 / 0.6539 | 0.6040 / 0.6034 | 7.28 → 7.21 | **−1%** |
| 16384³ | 5.0518 / 5.0292 | 4.7857 / **4.8865** | 9.50 → 5.10 | **−46%** |

Session 1 gives +0.4%, +5% and +39% for the same three rows. **`whole − no
drain` reproduces to 1–5% at 8192³, to 0.4–7% at 4096³ and to 39–46% at
16384³** — because at 16384³ it is a 0.2 ms difference between two 5 ms launches
divided over 28 items, and launches that reproduce to 2% leave ±40% on that
quotient. Every 16384³ epilogue figure in the sections above — #116's 16.34,
#117's 5.81, #118's 2.91 — is inside that band and none of them carries it. The
8192³ column is the one to hold, which is what every table here has said for a
different reason each time.

##### the global stores are not waiting on HBM, and that retires half a lever

`s84 hot` is `staged84` with every global store aimed at the cluster's own first
tile: opcode-identical, same LDTM, same `stmatrix`, same `ld.shared`, the same
32 × 16 B stores, and the only thing that moves is that 148 clusters rewrite one
tile each instead of streaming 134 MB or 512 MB of `C`.

| shape | `staged84` ms | `s84 hot` ms | hot vs |
| --- | ---: | ---: | ---: |
| 4096³ | 0.1049 / 0.1043 | 0.1361 / 0.1060 | −22.9% / −1.5% |
| 8192³ | 0.6621 / 0.6539 | 0.6627 / 0.6564 | **−0.1% / −0.4%** |
| 16384³ | 5.0832 / 5.0292 | 4.9935 / 5.1489 | +1.8% / −2.3% |

**Deleting the epilogue's entire HBM traffic is worth between −2.3% and +1.8%,
which is nothing.** After #116 made the writes full sector transactions they are
issue- and latency-bound and not bandwidth-bound. That is a negative result with
a consequence: the argument for #15's original route — hand the global write to a
TMA engine so the stores stop pressing on memory — has no pressure to relieve.
What a TMA store could still buy is the *issue*, which the ladder above prices at
2.0–2.3 µs a tile at 8192³, and that is a different and smaller claim than the
one `kittens::epilogue::StoreRing` was scoped against. (#107 ran this probe on
the register drain and reached the same verdict against a different store shape;
this is it re-run after the shape changed, which is what that section asked for.)

##### and the shared round trip is worth three times what #116 measured

#116's A/B ran both arms at `.x1` and won +4.2% at 8192³. #117 then took ~8 µs a
tile of exposed tensor-memory latency out of the staged arm **only**, so the
published comparison prices two changes at once and the left-hand side of it —
the register drain at `.x8` — was never built. `gemm_cg2_lcf8` is that arm:
`TmemTile::tile_x8` where `TmemTile::tile` was, the same 98 392 B, the same
`store_rows`, the same bytes into `C`, on the correctness gate.

| shape | `lcf` ms | `lcf8` ms | `lcf8` vs `lcf` | `staged` vs `lcf` | `staged8` vs `lcf8` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 0.1495 / 0.1399 | 0.1413 / 0.1404 | +5.8% / −0.4% | +15.4% / +7.7% | **+34.9% / +34.7%** |
| 8192³ | 0.7519 / 0.7440 | 0.7442 / 0.7379 | +1.0% / +0.8% | +3.9% / +3.7% | **+12.6% / +12.8%** |
| 16384³ | 5.5699 / 5.4199 | 5.5600 / 5.4780 | +0.2% / −1.1% | +4.2% / +1.5% | **+10.9% / +8.5%** |

**`.x8` on the register drain is a clean null** — −1.1% to +5.8%, centred on
nothing — against +23.6% / +8.6% / +3.6% on the staged path. This is #119's
mechanism a third time, and here the lever is worth *zero* rather than two
thirds: `store_rows` issues 128 scattered four-byte stores a thread per band and
they were already covering every LDTM latency the drain paid. `ptxas` says the
same thing from the register side — `gemm_cg2_lcf8` is **166 registers on a
528 B frame, byte-identical to `gemm_cg2`**, where `.x8` on the staged path cost
+52 (42 → 94). The band was already all in registers, so no liveness moved, and
no latency was exposed to remove.

**Which makes the round trip worth +12.7% at 8192³, not +3.9%.** Reproduced to
0.2 points across two containers. #116's decision is confirmed and its own
number understated it by 3×: the recoalescing only gets paid once the drain has
stopped waiting sixteen times a band, so the two levers are strongly
super-additive rather than independent. The general form is worth carrying,
because it is a prediction rather than a summary: **any change that removes
exposed latency from this epilogue makes the store's *shape* worth more than it
last measured.** A TMA store would be the next test of it, and on that basis the
2.0–2.3 µs a tile the ladder prices is a lower bound rather than an estimate.

**`staged8` and `staged84` trade places on `gemm` too, and #117's composition
gain does not reproduce.** Four measurements of each across two containers:
0.795 / 0.941 / 0.949 and 0.819 / 0.940 / 0.933 for `staged8`, against
0.793 / 0.939 / 0.936 and 0.818 / 0.940 / 0.936 for `staged84`. They are within
0.5% at 4096³ and 8192³ and swap sign at 16384³, which is where #117 measured
`staged84` +1.5 points ahead — a 16384³ figure, and therefore one the
reproducibility table above already puts inside its own noise. #118 found
exactly this on `gemm_ws` ("within 1.1% of each other and trade places between
sessions") and it is why `gemm_ws` ships `staged8`; **the same is now true on
`gemm`, whose default #119 set to `staged84` on the strength of that 16384³
row**. Nothing here says the shipped rung is the wrong one — it is a tie, and a
tie is not a regression — but the composition is not a win to quote, and if the
tie is ever broken again it should be broken at 8192³ where the instrument
works.

##### the ranking that comes out of it, and the three things it could not price

At 8192³, where the denominator is stable to a tenth of a point and the
subtraction to 1–5%:

| what | µs/tile | of the residual epilogue | if it went to zero | the lever |
| --- | ---: | ---: | ---: | --- |
| `cvt` + `stmatrix` | 3.8–4.1 | 55–64% | ~+5% | **none identified** |
| `ld.shared` + `st.global` | 2.0–2.3 | 31–36% | ~+2.7% | a TMA store out of the staging tile |
| LDTM after `.x8` | ~0 | 0% | 0 | spent by #117 |
| unattributed | 0.8–1.5 | 9–20% | — | — |

So the next thing worth doing is the **TMA store**, at a ceiling of about 2.7%
of the launch. `kittens::epilogue::StoreRing` (#111) exists; what it needs here
is the `fence.proxy.async.shared::cta` the current path deliberately does not
have — both ends are generic today — a staging tile that is CTA-visible rather
than warp-private, and a `tma_store_wait` in the item boundary. Call it 150–250
lines in `gemm.rs` and nothing in `src/`. It is worth doing **knowing that the
bandwidth argument for it is dead**: table 4 says there is no HBM pressure, so
what it buys is 64 instructions a thread a band and their latency, and the
round-trip result above says that number may come out larger than 2.7% once it
is measured rather than extrapolated.

It is a small lever, and that is the finding. **The largest single term in the
gap has no lever pointed at it at all**, and saying so is worth more than
ranking the ones that do.

**What could not be established, in order of what it costs.**

**1. What the 4 µs is.** The ladder prices `cvt` + `stmatrix` + `bar.warp.sync`
as one pass because they *are* one pass, and no probe here splits them. A rung
doubling the `cvt` without the `stmatrix` has nothing consuming its result and
is exactly the dead-code case doubling exists to avoid; a rung doubling the
`stmatrix` without the `cvt` writes the same bf16 twice and may well be folded.
The candidates are four — the 128 `cvt` a thread, the `stmatrix` issue, bank
behaviour in the swizzled write, and the warp barrier — and the count argument
is the weakest of them, since #117 already halved the `stmatrix` count for
−0.3%. Until it is split, "the largest term has no lever" is a statement about
the search and not about the hardware.

**2. Anything at 16384³.** The subtraction carries ±40% there, so the
decomposition, the `hot` probe and the epilogue's share of that gap all inherit
it. The one 16384³ claim this sweep does make is the ceiling being *below*
parity, and it makes it because it does not depend on the subtraction at all:
two launches against the library, reproduced in two sessions at 0.984 and 0.993.

**3. The tile.** #87 chose `[256, 256] s3` on arithmetic intensity against a
budget that has since moved — registers 166 → 80, the shared plan 98 392 →
114 816 against the 116 736 a CTA gets — and #105 is open on `[256, 256]`
regressing 1024³ by 18% and requiring `n % 256`. Nothing here re-swept it.
`bench tile` is the instrument and it was not run: the epilogue's decomposition
was the question, and a tile sweep is another container's worth of work. It is
**unranked** rather than ranked low, and at 4096³ and below — where the ratio is
0.79–0.82 and the wave is 86% full — it is the obvious place to look next after
the small-size denominator is stable enough to look with.

**Residency is unchanged and is not re-counted.** All four new staged probes
declare the same 114 816 B as `staged84`, and `gemm_cg2_lcf8` declares the same
98 392 as `gemm_cg2`, so no envelope moved. `ptxas -v` reads 80 / 87 / 125 / 102
for `staged84` / `2g` / `2m` / `2x` and **166 on a 528 B frame for `lcf8`**,
zero spill everywhere, every one of them inside the step that
`min(512 / 256, 233 472 / plan)` already binds at two CTAs an SM. That is the
eleventh occasion in this file the register column has been unable to order the
times.

##### and the 16384³ instrument was never valid, for a reason that is arithmetic — #122

#121's reproducibility control is the first thing in this file to measure the
*measurement*, and it came back 39–46% at 16384³ against 1–5% at 8192³. Three
published figures at that size — #116's 16.34 µs a tile, #117's 5.81, #118's
2.91 — are inside that band. This is what that band is, and it is not a
degradation: the same arithmetic that produces it at 16384³ produces the 1% at
8192³, so the instrument did not stop working there. **It never worked there.**

**Every epilogue figure in the sections above is a difference between two whole
launches**, and a difference is only as precise as its arms divided by its own
share of them. Call that share `s`; the amplification is `1/s`, and the
difference's relative error is the arms' relative error over `s` whatever the
estimator, the sample count or the interleave. #121's own table 6 is a
three-point instance of it, reconstructed here from its own printed numbers with
nothing measured:

| shape | epilogue / launch | `1/s` | the two arms repeat to | predicted | #121 observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096³ | 17.9% | 5.6 | 0.86% + 0.58% | 8.1% | 7% |
| 8192³ | 7.8% | 12.9 | 0.15% + 0.10% | 3.2% | 1% |
| **16384³** | **5.3%** | **19.0** | **0.45% + 2.11%** | **49%** | **46%** |

**The launches were never the problem.** They repeat to 0.1–2.1% at every size
in that table, which is a *better* instrument than most of this file assumes.
What fails is the quotient: the epilogue is a per-output-tile cost and the
launch is per tile times `K`, so `s` falls as `1/K` by construction, and at
16384³ a 2% arm error is a 40% epilogue error before anybody reads it. That is
the whole finding, it is the cheapest of the four candidates, and it means the
16384³ rows were arithmetic rather than a hardware effect.

**Three things it is not, each checked rather than assumed.** *Sampling* is not
it: `min` of 30 is the same estimator at every size and the arms' own
reproducibility is 0.10% at 8192³, so more samples have nothing to fix at the
size where the quotient works. *The wave* is not it: 8192² and 16384² are both
98.8% full over 148 clusters (#87), and in any case every comparison in this
file holds the grid, the tile and the traversal fixed between its arms, so wave
quantization is common to both and cancels. And **adding rather than deleting
does not fix it either** — `Tile::drain_staged_twice` retires the dead-code
hazard, which is a real and different hazard, but one added link is a *smaller*
share of the launch than the whole epilogue is, so the ladder is more amplified
and not less. #121's own 16384³ ladder row says so: +4.01 and −2.69 µs a tile
for the same quantity in two sessions.

**But there is a second term, and keeping the launch order is what shows it.**
`bench` sorted its thirty samples and threw the order away, so a wide
distribution and a device slowing down inside the call looked identical.
[`Timings::drift`] is the second fifteen launches over the first fifteen, in
launch order, and [`Timings::spread`] is the same call's `max/min`. Six arms
over two containers:

| shape | call to call | in-call `max/min` | **drift** |
| --- | ---: | ---: | ---: |
| 8192² × k1024 | 0.39–1.46% | 4.0–7.5% | −1.90 … −0.25% |
| 8192² × k2048 | 0.15–1.04% | 4.1–9.4% | −3.40 … −0.61% |
| 8192³ | 0.26–0.58% | 1.3–2.5% | −0.40 … −0.18% |
| 16384² × k1024 | 0.24–1.15% | 0.9–7.2% | −1.01 … +0.56% |
| **16384³** | **0.99–3.12%** | **18.4–22.6%** | **+6.85 … +11.33%** |

**At 16384³ and nowhere else, the device is 7–11% slower by the end of the call
than at the start**, in all six arms and both containers, and that drift is
essentially the whole of the 18–23% in-call spread — everywhere else the drift
is *negative*, which is a warm-up finishing rather than a clock dropping. It is
not the problem size: `16384² × k1024` has the same 4096 tiles, the same 28
items a critical-path cluster walks, the same 512 MiB of `C` and the same wave,
and drifts −1.0 to +0.6%. It is the *duration* — thirty launches of 5.1 ms is
154 ms of continuous 1.1 PFLOP/s against 16 ms at the shortened depth. So a
16384³ `min` is a boost-clock number taken from the first launch or two of a
call whose last launches run 10% slower, and how far into the boost corner it
lands is what moves it 1–3% between calls where 8192³ moves 0.3%. This is the
mechanism under #121's table 2 as well: cuBLASLt's own `max/min` is 1.15–1.19 at
16384³ and 1.01 at 8192³, measured through this same harness.

##### so measure the same geometry at a shorter reduction — `bench repro`

`K` is the one axis the epilogue's cost does not sit on. Holding `M` and `N`
holds the tile grid, the waves, the items a cluster walks, the `C` traffic and
the epilogue's *total* cost; moving `K` moves only the launch it is a fraction
of. So the fix is not a better estimator — no estimator survives a 60×
amplifier — it is to take the difference where `s` is 11% instead of 2%. The
arms are interleaved round-robin rather than run arm-by-arm, so each pair is
adjacent in time and a drift enters both sides of it; four whole measurements
of each arm give a spread rather than a point, and the table prints the spread
the amplification *predicts* beside the one it got.

The probe is `s84 2g` — `staged84` with a second `store_shared_rows` per band
and nothing deleted, so the difference is one `ld.shared` + `st.global.v4` pass,
priced by addition. Two containers, `session 1 / session 2`, µs a tile over the
items a critical-path cluster walks:

| shape | items | µs/tile | `s` | `1/s` | observed | predicted | **session to session** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192² × k1024 | 7 | **2.34 / 2.34** | 10.8 / 11.0% | 9.2 | 17 / 17% | 16 / 18% | **0.0%** |
| 8192² × k2048 | 7 | 2.42 / 2.61 | 7.8 / 8.6% | 12.2 | 13 / 4% | 19 / 9% | 7.9% |
| 8192³ | 7 | 2.07 / 2.29 | 2.2 / 2.5% | 42.8 | 33 / 18% | 41 / 30% | 10.6% |
| 16384² × k1024 | 28 | **2.11 / 2.12** | 10.8 / 10.8% | 9.3 | 5 / 8% | 12 / 13% | **0.5%** |
| **16384³** | 28 | 3.01 / 3.82 | 1.6 / 2.1% | **54.1** | **167 / 111%** | 235 / 177% | **26.9%** |

**`predicted` tracks `observed` across a twenty-fold range and bounds it from
above in nine rows of ten**, which is the diagnosis stated as a prediction
rather than as an explanation: the error on these differences is the arms'
error over `s`, and the residual — predicted running above observed — is the
round-robin pairing cancelling the drift the propagation assumed was adverse.

**And the two shortened rows reproduce to 0.0% and 0.5% across two
containers**, against 10.6% and 26.9% for the same quantity at full depth. The
16384² number nobody could measure is **2.11 µs a tile**; #121's ladder put it
at 0.45 and 4.22 in two sessions, and 2.11 is inside neither and between both.
The instrument is calibrated by the 8192² column: 2.34 µs a tile lands at the
top of the 2.0–2.3 that #121's `ld.shared` + `st.global` row already carries.

**The µs/tile is flat in `K`, which is what makes the substitution legal** —
2.34 / 2.42 / 2.07 and 2.34 / 2.61 / 2.29 across an eightfold range of depth,
with the two deep readings carrying ±18–33% of their own. That is the check
this design has to pass and it is also its residual risk: shortening `K` puts
`A` and `B` inside L2 where they were 128 MiB each, and the K sweep bounds that
confound to the deep rows' own error bar rather than to 1%. The flatness is
real to about a tenth; it is not established to a hundredth.

##### which orders `staged8` and `staged84` for the first time, and the sign is the geometry's

#120 set `gemm`'s default to `staged84` on a 16384³ row where it led by ~1.5
points, and #121 called that row a tie because ~1.5 points is inside its noise.
Both are right about the row. The shortened rows can see it. Four paired ratios
per cell, `session 1 / session 2`, above 1.000 being `staged84` ahead:

| shape | `s8`/`s84` | lowest–highest, both sessions | `1/s` | µs/tile | orders them |
| --- | ---: | --- | ---: | ---: | --- |
| 8192² × k1024 | 0.9852 / 0.9758 | 0.9724 – 0.9902 | 41–67 | −0.32 / −0.51 | **`staged8`, 1.5–2.4%** |
| 8192² × k2048 | 0.9827 / 0.9864 | 0.9813 – 0.9882 | 58–74 | −0.54 / −0.41 | **`staged8`, 1.4–1.7%** |
| 8192³ | 0.9965 / 1.0002 | 0.9944 – 1.0033 | 284–5041 | −0.33 / +0.02 | no — straddles 1.000 |
| 16384² × k1024 | 1.0104 / 1.0038 | 0.9971 – 1.0116 | 96–263 | **+0.20 / +0.07** | **`staged84`, 0.4–1.0%** |
| **16384³** | 0.9926 / 0.9874 | 0.9796 – 1.0207 | 80–135 | −1.36 / −2.26 | no — both ranges straddle |

**They are not tied, and which one wins is a property of the output geometry.**
At 8192² `staged8` is ahead by 1.4–2.4% of the launch in four independent cells,
none of whose ranges reach 1.000; at 16384² `staged84` is ahead by 0.4–1.0%,
same sign in both containers, though session 2's range does touch it.

**The full-depth rows do not merely fail to order them — they point the wrong
way by an order of magnitude.** 16384³ centres on `staged8` ahead by 1.4–2.3 µs
a tile where the same geometry shortened says `staged84` ahead by 0.07–0.20,
and both of its ranges straddle 1.000 so neither reading is a result. That is
`1/s` of 80–135 doing exactly what it is advertised to do: this comparison is a
0.7–1.3% difference between two 5 ms launches, which is a harder measurement
than the epilogue subtraction and not an easier one, and it is the row #120's
default was set on.

**So #120's default is right, for a reason its own row could not carry**, and
#121's advice — break the tie at 8192³ where the instrument works — would have
broken it the *other way* and shipped the rung that loses at the size the
default was chosen for. The correction is not "use 8192³"; it is **use the
geometry the decision is about, at a depth where the difference is a tenth of
the launch and not a hundredth.** Nothing here asks for the default to move: a
rung that wins at 16384² and loses at 8192² is a shape question, and `gemm_ws`
shipping `staged8` (#118) beside `gemm` shipping `staged84` is now two answers
to two questions rather than the same coin landing twice.

**What this does not do.** It never measures `whole − no drain`, because the
epilogue-free entry points are not on `bench`'s side of the wall and the
subtraction is what is under investigation. But the same substitution applies to
it and is worth stating with a number: the whole epilogue is ~7.3 µs a tile, so
at `16384² × k1024` it is ~0.20 ms of a 0.55 ms launch — `s` of 37% and `1/s` of
2.7, which against arms that repeat to 0.5–1.2% is **±5% where the full-depth
version is ±46%**. That is the row that would replace #116's 16.34, #117's 5.81
and #118's 2.91 rather than merely retire them, and it is a change in
`gemm.rs`'s sweeps and not in the harness. Until it is taken, the honest status
of every 16384³ epilogue total in this file is *unmeasured*, which is a weaker
claim than wrong and a much weaker one than the numbers currently read as.

**And one thing every 16384³ absolute in this file now carries.** The drift
table says a `min` at that size is a boost-clock reading from the first launches
of a call whose last launches are 7–11% slower. The TFLOP/s, the ratios against
cuBLASLt and the ms are all still comparable *to each other* — both sides of
every ratio are timed the same way in the same call — but none of them is a
sustained-clock number, and the 1178.5 / 1117.3 TFLOP/s [`GEMM_SIZES`] reports
for two runs of this tree is that 5.5% seen from the other end.

##### the TMA store, measured — it loses, and the CTA-wide ring is the better of the two — #123

Two sections ago left exactly one lever with a number on it: `ld.shared` +
`st.global` at 2.0–2.3 µs a tile, 31–36% of the residual epilogue, **~+2.7% of
the launch** if it went away. This is that change, built two ways and measured
in four containers. **It does not win anywhere the instrument can see it, and
the warp-scope form loses by 1.0–1.7%.**

Two entry points, both on the correctness gate, both at `staged84`'s own
114 816 B and the same two CTAs an SM, both keeping its `.x8` LDTM, its `.x4`
`stmatrix` and its staging tiles. What changes is 32 `ld.shared.v4` + 32
`st.global.v4` a thread a band becoming **one `cp.async.bulk.tensor`**:

- **`s84 tma`** keeps the staging tile per warp, so the ring is warp-scope —
  `fence.proxy.async.shared::cta` on every lane, `bar.warp.sync`, lane 0
  issues — and the item pays **no block barrier**, exactly as #116's design
  intended. Four `[32, 64]` boxes a band, one per warp.
- **`s84 tmac`** reads the same 16 384 B as one `[128, 64]` tile through
  `kittens::epilogue::StoreRing` at the CTA scope #111 built it at: **one**
  `[128, 64]` box a band, and eight `bar.sync` an item instead of none.

**The full-depth A/B cannot see it, and #122 is why.** At 8192³ this comparison
is a few tenths of a percent between two 0.65 ms launches — `1/share` of
**458–709**, which is a *harder* measurement than `whole − no drain` and not an
easier one. Three containers of it read `+0.2 / −0.1`, `+0.4 / +0.3` and
`−0.2 / +0.4` percent for the two arms: noise with no sign.

**So it was taken where the difference is a tenth of the launch.** #122's
substitution — hold `M` and `N`, shorten `K`, interleave the arms round-robin,
four whole measurements each — applied to a *ratio* rather than to a
subtraction. `staged84` over each arm, so **above 1.000 is the TMA arm ahead**;
median of four paired ratios and the full range, `session 1 / session 2`:

| shape | `1/share` | `s84`/`s84 tma` | range, both sessions | `s84`/`s84 tmac` | range, both sessions |
| --- | ---: | ---: | --- | ---: | --- |
| 8192² × k1024 | 61–191 | 0.9885 / 0.9835 | **0.9743 – 0.9952** | 0.9948 / 0.9980 | **0.9924 – 0.9987** |
| 16384² × k1024 | 85–427 | 0.9882 / 0.9899 | **0.9853 – 0.9913** | 0.9977 / 1.0061 | 0.9971 – 1.0069 |
| 8192³ | 294–709 | 0.9978 / 1.0018 | 0.9969 – 1.0023 | 1.0014 / 1.0034 | 0.9984 – 1.0062 |

**`s84 tma` loses, and it is not close to a tie.** Four independent cells across
two geometries and two containers, and not one of the four ranges reaches
1.000: it is behind by **1.0–1.7%**. **`s84 tmac` is a tie to a small loss** —
0.2–0.5% behind at 8192², and the two containers disagree in sign at 16384²
(0.9977 against 1.0061, with non-overlapping ranges), which is a shape this
table is built to report rather than to average away. The 8192³ anchor orders
nothing in either arm, as its `1/share` says it cannot.

Against the library at 8192³ the ratio does not move at all: 0.943–0.949 for
`staged84`, 0.941–0.948 for `tma`, 0.946 for `tmac`.

**Where the 2.7% went, by addition rather than by argument.** `s84t 2g` is
`s84 tma` with a second TMA store per band — fence, barrier, issue and commit
included — the exact twin of the `s84 2g` probe that priced `ld.shared` +
`st.global`. Both aim the extra pass at the cluster's own tile so its bytes stay
in L2. Microseconds per output tile at 8192³, four containers:

| | µs/tile |
| --- | ---: |
| `ld.shared` + `st.global` | 2.33 / 2.41 / 2.56 / 2.58 |
| the TMA hop | **1.57 / 1.90 / 1.91 / 1.99** |
| the difference the doubling probe attributes to the change | 0.42–0.75 |

**Even the doubling probe never promised 2.7%.** 2.7% was the value of that term
going to *zero*, and the engine does not take it to zero — it takes it to about
three quarters of itself, worth 0.42–0.75 µs a tile, 6–11% of a ~7 µs epilogue,
under 1% of the launch at its most generous. That is the whole of the ceiling
this change ever had.

**And the paired table says the realised value is below zero, which the doubling
probe is structurally unable to see.** A *doubled* store is an extra one, issued
and left in flight beside work that continues; a *replacing* store is on the
critical path of the next band. That difference is the depth: at `IN_FLIGHT = 0`
the ring's `acquire` is `cp.async.bulk.wait_group.read` at zero groups, so band
`k + 1`'s `stmatrix` cannot start until the engine has finished *reading* band
`k`'s buffer — four times an item — where `store_shared_rows` retires its stores
into the memory pipeline and blocks on nothing. The kernel trades 64 pipelined
instructions a thread for one instruction and one exposed engine latency, and
the exposed latency is worth more than the instructions.

**Depth 1 is not a choice and the arithmetic is the constraint.** The staged
envelope is 114 816 B against the 116 736 a CTA gets at two per SM: **1920 B of
headroom**, and a second set of `[32, 64]` buffers is 16 384. `STAGES` 3 → 2
does not buy it either (−11.8% / −7.3% at unchanged residency, above). So the
one shape this change was allowed to take is the one with no overlap in it.

**The block barrier was the obvious suspect and it is the wrong way round.**
Before running this, the live hypothesis was that `StoreRing`'s CTA-collective
`acquire`/`commit` would cost the epilogue the barriers #116 was built to avoid.
The barriers are real — the opcode census reads `bar.sync` 2 / 5 and
`bar.warp.sync` 3 / 0 for `s84 tma` / `s84 tmac` against `staged84`'s 2 / 1 —
and **the arm that pays them wins**, in all four cells of the paired table, by
0.6–1.6%. One `[128, 64]` box a band beats four `[32, 64]` boxes by more than
eight `bar.sync` an item cost. The engine's per-instruction overhead is the term
that matters here and the barrier is not.

**So `StoreRing` was usable as built, and it is also the faster of the two.**
`s84 tmac` uses the type unmodified. What it could not express was the *layout* —
a warp-private staging tile — so the type gained a `Scope` parameter (`Cta`, the
default and the behaviour it shipped with, and `Warp`): 40 lines, no byte moved,
changing only which barrier `converge` is and which thread `issuing` is. Every
argument inside the type survived untouched, because the proxy fence, the two
waits and their order are about proxies and group counts and none of them
mentions how many threads are in the barrier. The parameter earned its keep by
producing the arm that lost; on this evidence a kernel should reach for `Cta`.
`device-tests` gained `store ring warp` and `store ring warp phased` for it.

**One hardware fact fell out of the phased case and is worth recording.** `gemm`
puts its staging run at a shared plan rounded to 128 bytes and not to 1024, so
every staging tile starts mid-swizzle-period. `SharedTile::chunk_writer` folds
that phase in; whether the *TMA engine* derives the same phase from the same
address had only ever been checked on the load side (`swizzle roundtrip short`).
`store ring warp phased` is the store side of it — the whole ring pushed one
128-byte row along — and it passes, so the engine reads the phase off the
absolute address in both directions of travel. That is why no alignment padding
was needed and why both TMA rungs declare `staged84`'s envelope to the byte.

**Residency and registers unchanged.** All three new entry points declare the
same 114 816 B, so `gemm_cg2_staged_no_drain` serves their subtractions as it
does `staged84`'s. `ptxas -v` reads 98 / 94 / 96 for `s84 tma` / `s84 tmac` /
`s84t 2g` against `staged84`'s 96, zero spill, 256 B frame — all inside the step
`min(512 / 256, 233 472 / plan)` already binds at two CTAs an SM. The census
confirms both arms deleted what they claim: `ld.shared.v4` and `st.global.v4`
both go 1 → **0**, and `cp.async.bulk.tensor` goes 2 → 3, the third being the
store.

**What this leaves the ranking.** Both terms #121 could name now have their
answer, and neither was spendable:

| what | µs/tile at 8192³ | share | the lever, after this |
| --- | ---: | ---: | --- |
| `cvt` + `stmatrix` | 3.3–4.1 | 47–64% | **none identified**, unchanged |
| `ld.shared` + `st.global` | 2.3–2.6 | 35–37% | a TMA store: **measured, −1.7% to +0.6%, spent** |
| LDTM after `.x8` | ~0 | 0% | spent by #117 |

`staged84` still ships and nothing here asks it to move.

**The one thing this does not close, with the arithmetic it would need.** The
loss is attributed to depth 1 by mechanism and not by measurement, and the
measurement is buildable inside the same 16 384 B: a **depth-2 warp ring of
`[16, 64]` buffers is 2 × 2048 = 4096 B a warp**, which is exactly what a
`[32, 64]` buffer costs today, so the envelope does not move and the residency
question does not reopen. A warp would drain its 32 rows as two 16-row halves
alternating buffers, and the second half's `stmatrix` would overlap the first
half's store — which is the only thing standing between the doubling probe's
0.42–0.75 µs a tile and the launch. What it needs is a 16-row band out of
`TmemTile::tile_x8`, whose `.x8` shape is 32, so it is a question about the LDTM
width and not about shared memory. Until that is built, "depth 1 is why" is a
mechanism consistent with every number here and not a measured one.

##### and the small end of `gemm_sol` is one wave short, which no tiling fixes — #138

`gemm_sol`'s ratio against a live cuBLASLt FP16 baseline read 0.795 at 4096³,
0.873 at 8192³ and 0.946 at 16384³, and the obvious reading of that shape is
tile quantization. `bench sol` and `bench sol-small` are the two tables that
test it. The arithmetic first, because it is stated before anything is launched
and every row below is a test of it:

both shared plans declare more than half of the **233472 B** an SM divides, so
residency is **one CTA an SM**, and a `cta_group::2` MMA needs its pair
co-resident — **74 clusters** on 148 SMs. A launch is one cluster per output
tile and takes `ceil(tiles / 74)` waves of them, so
`tiles / (waves · 74)` is the fraction of it not idling:

| shape | entry | tiles | waves | wave eff |
| --- | --- | ---: | ---: | ---: |
| 4096³ | `[256, 256]` | 256 | 4 | **0.865** |
| 4096³ | `[512, 256]` | 128 | 2 | **0.865** |
| 4096³ | `[256, 128]` | 512 | 7 | 0.988 |
| 8192³ | `[256, 256]` | 1024 | 14 | 0.988 |
| 8192³ | `[512, 256]` | 512 | 7 | 0.988 |

So 4096³ and 8192³ are not the same problem: 8192³ is already wave-perfect and
13.5% of 4096³'s *grid* is idle. The two 4096³ entries quantizing **identically**
is what makes that pair a controlled comparison of everything else.

**The model is right, measured directly.** Sweeping `m` in tile steps at
`n = k = 4096` walks the efficiency up and down a sawtooth with nothing else
moving:

| shape | tiles | waves | wave eff | TFLOP/s | ÷ wave eff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096x4096x4096 | 256 | 4 | 0.865 | 1351.0 | 1562 |
| 4608x4096x4096 | 288 | 4 | 0.973 | 1469.5 | 1510 |
| 5120x4096x4096 | 320 | 5 | 0.865 | 1354.4 | 1566 |
| 5632x4096x4096 | 352 | 5 | 0.951 | 1463.6 | 1539 |

The raw column swings 8.7% and the corrected one is flat to ±2%. **CLC work
stealing is running throughout**, which settles what it is for: it removes the
*static* share and cannot remove the integral one, and the sawtooth is the
integral one surviving it in full.

**And it buys nothing against the baseline, because cuBLASLt pays it too.** The
heuristic prints its own wave count beside the algorithm it chose, and at these
four shapes it is **3.46 / 3.89 / 4.32 / 4.76** — our 256/288/320/352 tiles over
74 clusters, to three digits, on `tile=23` at every row from 2048³ up. cuBLASLt's
rate divided by the *same* efficiency is 1955 / 1964 / 2018 / 2015, equally flat.
Two implementations on the same tile with the same sawtooth: **quantization
divides out of the ratio exactly**, and 0.795 at 4096³ is a rate deficit and not
a tiling one. What is left is #144's per-tile constant and K-loop cadence.

**Every attempt to buy the quantization back loses.** `[256, 128]` is the tile
that reaches 0.988 at 4096³ and it is built, checked exact, and slower:

| shape | `[256, 256]` | `[256, 128]` | narrow/wide | wave eff, wide → narrow |
| --- | ---: | ---: | ---: | --- |
| 1024³ | 124.7 | 160.9 | **1.29** | 0.216 → 0.432 |
| 1536³ | 364.7 | 464.1 | **1.27** | 0.486 → 0.973 |
| 2048³ | 751.9 | 697.2 | 0.93 | 0.865 → 0.865 |
| 3072³ | 1283.2 | 1034.2 | 0.81 | 0.973 → 0.973 |
| 4096³ | 1343.0 | 1079.4 | 0.80 | 0.865 → 0.988 |
| 8192³ | 1682.6 | 1062.1 | 0.63 | 0.988 → 0.988 |

The two rows where efficiency does not move are the price of the tile alone: at
equal quantization the narrow tile costs 7% at 2048³, 19% at 3072³ and **37% at
8192³**, which is 1.5× the operand traffic per flop (85.3 against 128 flops per
operand byte) and twice as many per-tile constants to pay it on. At 4096³ that
price is larger than the 14% of quantization it is buying, and the tile loses
20%.

`[512, 256]` at 4096³ is the other direction and is a **dead heat**: 1323.2
against 1343.0, 0.985, inside a 4% spread. Its 0.75× operand traffic per flop —
worth 1.12× at 8192³ where quantization is equal — is cancelled at 4096³ by
half as many tiles to amortize a per-tile constant over. #138's crossover at
8192 is therefore right, and it is right for a reason.

**Split-K is arithmetically dead here, and the vendor agrees.** Splitting K in
two at 4096³ takes 256 work units to 512 and the efficiency to 0.988, worth
about 8% of the launch. It also doubles the number of epilogues, and #144 prices
the epilogue at **19.7% of that launch** — so the cheapest possible reduction,
one that adds no traffic at all, still pays 19.7% for an 8% return. Stream-K,
splitting only enough tiles to level the tail, splits at most 74 of 256 and so
pays ~5.7% for the same 8%: a ~2% return inside rows that repeat to 3–6%, for a
k-range per work unit, a partial-tile MMA, a fixup ordering and an fp32
workspace that the exact-output gate makes mandatory. The reduction instruction
being reachable does not change either figure. cuBLASLt reports
`splitk=1 reduction=0 workspace=0 B` at every shape in both tables.

**The N band is worth nothing measurable.** `group` was a rule inside the kernel
keyed on `tiles_m`; it is a launch parameter now and swept over `{1, 2, 4, 8,
16}`. At 8192³ on `[512, 256]`: 1880.3, 1877.7, 1876.2, 1878.7, 1877.6 TFLOP/s —
**flat to 0.2%**, which extends #138's `G=4` against `G=8` to the whole ladder.
At 4096³ on `[256, 256]` the range is 1311 to 1367 against rows whose own
launches spread 1.3–3.6%, so nothing there is quotable and the default is
unchanged.

**What did move: a third rung below 2048.** The narrow tile wins exactly where
halving `N` doubles the efficiency, which happens only while both tile counts
fit the same wave count — at or below **half a wave of wide tiles**. Both
1.27–1.29× rows are there and both losing rows are past it, and the ladder was
taken **twice round-robin** because a 1024³ launch is 17 µs and spreads 13–21%:
the two passes put the ratio at 1.273/1.273 at 1024³, 1.273/1.278 at 1536³,
0.927/0.934 at 2048³ and 0.806/0.811 at 3072³. `select_variant` is that
sentence. It lifts 1024³ from 0.567 to 0.731 of cuBLASLt and 1536³ from 0.597 to
0.760, and leaves every shape at and above 2048³ on the entry it already had.

**The ratio was never monotonic.** 1024³ is 0.567, 2048³ is 0.846, 4096³ is
0.795, 8192³ is 0.868 — 2048³ beats 4096³ because cuBLASLt is *also* one wave
short there (0.86 waves, 906.8 TFLOP/s, 40% of dense peak). Three points made a
trend that a fourth removes.

**One shape did not return.** `4096x4096x1024` on `[256, 256]` printed nothing
for 1200 s and `scripts/modal-run`'s watchdog stopped the container. It is not in
either table and was not chased; `4096x4096x2048` runs and is (1181.9 wide,
1004.4 narrow). A `k` shorter than `m` is otherwise unexercised by this kernel,
so whether that is a shape contract this port does not hold or a one-off is open.

##### the kernel the port is a port of, unported, on the same clock — and it does not reach cuBLASLt either

`gemm_sol` (#138) is a port of NVLabs' canonical `gemm_sol_final`, and it
measured **0.806 / 0.873 / 0.946** of cuBLASLt at 4096³ / 8192³ / 16384³ with no
way to read that: either the port dropped something the original has, or the
original stands there too. Nobody had timed the original on a B200. This is that
number, and **most of the gap is not the port's**.

The control arm is `experiments/src/gemm_sol_upstream.rs`: upstream's
`kernels.rs` at `b099f64c1a32869b74be99f4f88242fb68655b51`, copied without a
character changed and `include!`d as upstream `include!`s it, launched on
upstream's own grid through upstream's own `cuTensorMapEncodeTiled` descriptors,
under upstream's own variant selector. What is *not* upstream's is everything
outside the two CUDA events: `bench::time`'s five warm-ups and minimum of
thirty, and `gemm_sol`'s own `stage_f16` and `check_output` — the same functions,
not copies — so both kernels read byte-identical operands and both pass the same
exact-BF16 comparison before either is timed. `modal_app.py::upstream_bench` is
one container, so one device, one cuBLASLt, one day.

The revision above is the one the copy was taken at and the port was measured
against, and it is no longer the pin — `20a5616` is. Nothing here moves with it:
`gemm_sol_final/` is byte-identical at both revisions, and upstream's
`kernels.rs` and the vendored copy all three hash to
`6746e517eca19fdcc01cb0d5003e924bb638f5ee42e8c333d457a0d6f334d6e9`. The
provenance stays at `b099f64` because that is where it was read.

**Each arm at the variant it should be judged at.** cuBLASLt is the mean of its
four readings per shape; it repeats to 0.03% at 16384³, 0.15% at 8192³ and
**1.3% at 4096³**, which is the error bar the 4096³ row carries.

| | 4096³ M256xN256 | 8192³ M512xN256 | 16384³ M512xN256 |
| --- | ---: | ---: | ---: |
| cuBLASLt FP16 | 1635.9 | 2146.0 | 2158.8 |
| **upstream, unported** | 1435.0 | 1955.7 | 2086.2 |
| **our port** | 1322.7 | 1871.5 | 2041.4 |
| upstream / cuBLASLt | 0.877 | 0.911 | 0.966 |
| port / cuBLASLt | 0.809 | 0.872 | 0.946 |
| port / upstream | 0.922 | 0.957 | 0.979 |

**So the attribution, in TFLOP/s of shortfall against cuBLASLt:**

| | 4096³ | 8192³ | 16384³ |
| --- | ---: | ---: | ---: |
| total shortfall | 313.2 | 274.5 | 117.4 |
| upstream's own | 200.9 — **64%** | 190.3 — **69%** | 72.6 — **62%** |
| the port's | 112.3 — 36% | 84.2 — 31% | 44.8 — 38% |

Two thirds of it is where `gemm_sol_final` itself lands on this B200, and it
lands short of cuBLASLt at every one of the three sizes — furthest at the small
end, exactly as the port does. Whatever closes 4096³ is not a port defect and
was never going to be found by reading the port against the paper.

**And the row that does not flatter the port's control arm.** Upstream's shipped
selector takes M256xN256 up to 8192³; #138 moved that crossover to M512xN256
and this is the price of not having:

| 8192³, upstream's own policy | TFLOP/s | of cuBLASLt |
| --- | ---: | ---: |
| upstream, M256xN256 as shipped | 1800.1 | 0.839 |
| upstream, M512xN256 forced | 1955.7 | 0.911 |
| our port, M512xN256 | 1871.5 | 0.872 |

At 8192³ **the port beats the kernel it is a port of**, by 4.0%, entirely on the
crossover. At 4096³ upstream's policy is the better one — M256 1435.0 against
M512 1407.7 — and both agree at 16384³ to 0.01%, which is the auto selector and
the forced entry landing on the same kernel.

**Getting there needed two toolchain workarounds, and neither is the
vendoring's.** `modal_app.py::upstream_ptx` builds upstream's crate from a clean
clone, in its own workspace, at its own profile, and it does not assemble:

1. `cuda_device::tcgen05::stmatrix_m8n8_x2` reaches the NVPTX back end as
   `llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3`, is selected by nothing, and
   comes out as a call to an `.extern .func` that does not exist — *"line 10;
   fatal: Parsing error near '.nvvm'"*. `kittens::ldst` had already written this
   workaround for itself at `b099f64`, and the vendored copy binds the name to
   it: same signature, same instruction, upstream's file untouched. **The defect
   survives the move to `20a5616`** — same extern, same line, same `ptxas`
   message — so the workaround is not removable at this pin, and `upstream_ptx`
   is the check that would say when it is.
2. `opt -O2` turns upstream's four-way stage selects into **sixteen** lookup
   tables emitted as `.global` arrays of `.shared` addresses, which PTX forbids
   outright. `CUDA_OXIDE_OPT` pointed at an `opt` wrapper carrying
   `-switch-to-lookup=false` removes all sixteen and the whole bundle assembles.

Sixteen in upstream's own build and sixteen in ours, so this is the toolchain and
not the copy — and it is worth knowing that **`cargo oxide build` passes on both
of them**: it emits the PTX and never assembles it, so `build` gained a `ptxas`
step for this crate rather than trusting the one it had.

The second workaround is global to the compilation, so it is measured rather than
assumed: the port is timed in the same container without the flag and with it,
and reads **1320.7 / 1871.4 / 2041.0** against **1322.7 / 1871.5 / 2041.4** —
0.15%, 0.006% and 0.02% apart. The flag does not move the port, and the port's
own rows reproduce #138's published 1289.0 / 1867.7 / 2045.5 to 2.5% / 0.2% /
0.2% across two sessions and two days.

##### and the small entry's K loop is its feed's duty cycle, not its cadence — #144, #146

#144 left `[256, 256]`'s K loop at **75.8% of tensor-core peak** against
`[512, 256]`'s **99.4%**, on the same K-loop code, and listed two things it could
not separate. `bench sol-ablate` separates them, and the separation kills three
hypotheses before it names the term.

**Every arm is laddered in `K`.** That is the methodological point and it is not
cosmetic: an arm's *launch* fuses its per-K-block cadence with a per-tile constant
it carries for its own reasons, and at 64 k blocks one microsecond of constant is
5.7 points of "of peak". The un-laddered launches said the MMA warp reached only
88.7% of peak; laddered, it reaches 100.0%. One of those two numbers is a
measurement of the K loop and the other is not.

At 4096² on `[256, 256]`, µs a K block against **0.2759 at peak**:

| arm | µs a K block | of peak | what it says |
| --- | ---: | ---: | --- |
| `feed only` | 0.2636 | 104.6% | the feed alone beats the tensor core by 4.6% |
| `mma only` | 0.2742 | **100.6%** | no barrier wait, no TMA, no drain |
| `issue only` | 0.2758 | **100.0%** | the same, with the wait back |
| `no drain` | 0.3433 | 80.4% | the feed in situ costs **24.5%** |
| `whole` | 0.3503 | 78.8% | the drain costs the rate another 2.0% |
| `runtime stage` | 0.3615 | 76.3% | the spelling before this PR |
| **upstream** | **0.3352** | **82.3%** | the same algorithm, unported |

**The answer is one number per entry.** `feed only` is the operand pipeline's own
time for a K block; peak MMA is the tensor core's. Their ratio is the fraction of
the tensor core's busy time the feed has to be running:

| entry | feed alone | MMA at peak | **the feed's duty** | feed in situ |
| --- | ---: | ---: | ---: | ---: |
| `[256, 256]` | 0.2636 µs | 0.2759 µs | **95.5%** | **+24.5%** |
| `[512, 256]` | 0.3075 µs | 0.5518 µs | **55.7%** | **+0.0%** |

At 95.5% duty there is no slack for the TMA's writes and the tensor core's
operand reads to stay out of each other's way, and the cost of them not doing so
is the whole deficit. At 55.7% there is 44% of slack and it is free. That 1.72×
in duty is the 1.33× in operand bytes per flop between a `[256, 256]` cluster tile
and a `[512, 256]` one — #146's "0.75× the operand traffic per flop" read from the
other end — so **the term is the tile's arithmetic intensity, acting through the
feed's duty cycle rather than through raw bandwidth.** At `[512, 256]` nothing in
the K loop moves at all: feed, drain, barriers and ring index are 0.5557, 0.5572,
0.5570 and 0.5547 µs, every one of them 99.0–99.5% of peak.

**Three hypotheses died on the way, and each was cheap to kill.**

*Not enough arithmetic per barrier round trip* — the leading one, and the one the
second `A` ring at `[512, 256]` suggests — is dead twice. On arithmetic: both
entries execute exactly **one** `load.wait` per K block, so the large entry buys a
second MMA chain per *the same* round-trip count and additionally pays two `free`
commits where the small entry pays one; a fixed cost `C` of 88 ns from the small
entry's row would force the large one to 86% of peak against a measured 99.4%. And
on measurement: `mma only` is `issue only` with the MMA warp's `load.wait` dropped
and nothing else, and the two agree to **0.6%, with the arm that keeps the wait
nominally faster**. The barrier round trip is zero. #144's unseparated row is
closed.

*A `tcgen05` issue-rate ceiling* is dead: `issue only` is 100.0% of peak and
`mma only` 100.6%. One warp can saturate the tensor core at this tile.

*The producer's instruction count* is dead, and this one needed an experiment
rather than an argument. A rank's `B` half-panel is `128 x 64` — the same shape as
`A`, which already arrives in one TMA — and it arrives in two 64-row boxes only
because one 64-row tensor map served all three entries. Building it per entry
takes the producer from **3 instructions a K block to 2** at `[256, 256]` and from
4 to 3 at `[512, 256]`, at byte-for-byte identical traffic. It moves nothing:
0.998 and 1.014 on the launch over two passes, 0.3484 against 0.3503 µs on the
rate, and **1.008 and 0.981 on `feed only` itself**, where the feed is alone and
has nothing to hide behind. So the feed's ceiling is bytes and not instructions,
and #144's conclusion that the feed is not an instruction-rate problem survives —
though the comparison it drew for it (the feed's ceiling against the loop's
*in-situ* rate, which is low because the loop is slow) could not have failed. The
comparison that can fail is ceiling against what peak MMA demands, and that is the
duty-cycle table.

**What did pay is a port defect with upstream's own comment as its spec.** The MMA
warp indexed both operand rings and both stage barriers off
`global_k = sequence * k_blocks + k`, a runtime value. But `k_blocks` is a multiple
of `STAGES` — every entry's shape contract is `k % (STAGES · BLOCK_K) == 0` — so
`global_k % STAGES == k % STAGES`, and the four positions of the four-way unroll
are always stages 0, 1, 2, 3. Handing each position its stage as a **const** folds
both barrier addresses to literals, both operand descriptors out of the loop, the
accumulate predicate to a constant at three of the four positions, and the phase
parity to one register a turn rather than four. The PTX shows all four:
`add.s64 %rd199, %rd200, 131080` where the offset used to be computed, one
`%r158` carrying the parity across all four slots, and `mov.b32 %r130, 1` where
the predicate used to be a `setp`.

Upstream states the same fact and spells it differently — `#[unroll(4)]` over a
loop-local `k_idx & 3` with a `match` on it, and the comment *"`k_iters % 4 == 0`,
so the producer's global stage and this local stage agree at every tile boundary.
Keeping this expression loop-local lets the unroll pass fold each stage match."*
That attribute exists at our pin and **nothing in this repo uses it**; it is
rewritten only inside a `#[kernel]` or `#[device_function]` body, so a const
parameter is the spelling reachable from a plain `impl` method, and it does not
depend on an unroll pass firing.

| 4096³, `[256, 256]` | runtime stage | const stage | ratio |
| --- | ---: | ---: | ---: |
| container A, pass 1 | 1281.7 | 1333.8 | **1.041** |
| container A, pass 2 | 1274.5 | 1329.7 | **1.043** |
| container B, pass 1 | 1289.0 | 1327.2 | **1.030** |
| container B, pass 2 | 1300.7 | 1328.9 | **1.022** |
| K-block rate | 0.3615 µs | 0.3503 µs | 76.3% → **78.8%** of peak |

**+2.2% to +4.3%, two containers, four passes** — quoted as a range because each
row's own launches spread 2.6–5.7% and a ratio inherits that. At 8192³ on
`[512, 256]` it is 1.001–1.003, which is the predicted null: that K loop had
nothing to give.

**Its gain is on the inter-stage critical path, which is neither of the two places
it was expected.** `mma only` against `runtime mma` is 1.018 / 1.001 / 1.005 /
0.998 — null across four passes — so it is not `tcgen05` issue. And it is not the
`mbarrier.try_wait.parity` spin body either, which an earlier reading of this
claimed: the PTX says that body went from **7 instructions to 8**, because folding
the stage to a literal made LLVM re-materialize `mov.b64` of the dynamic-shared
symbol plus `add.s64` inside the loop instead of keeping a register live — and the
kernel got faster anyway. What is left is the scalar work **between one `load.wait`
returning and the next being entered**: pre-fold each stage computed its own parity
(`add.s32`, `bfe.u32`), its own ring offsets and its own two operand descriptors on
that path; post-fold the offsets are literals, the descriptors are hoisted out of
the K loop, the accumulate predicate is a constant at three of four positions, and
the parity is computed once a turn rather than four times. That path exists only
when the waits exist, which is exactly why `whole` gains 4% and `mma only` gains
nothing — it reconciles all three rows. Stated as the measurement, and corrected
once when the PTX contradicted the first reading.

**Against the two bars, in one container.** cuBLASLt 1707.7 and upstream 1453.5 at
4096³; ours 1294.9 before and 1328.1 after (means of the two passes):

| | of cuBLASLt | of upstream |
| --- | ---: | ---: |
| before | 0.758 | 0.891 |
| after | **0.778** | **0.917** |

So the port's K-loop share of the deficit was 6.0 points of peak and is 3.5, and
**the fold closed 42% of it.** What remains splits two ways: our drain-free K loop
(80.4%) is still 1.9 points behind upstream's drain-*inclusive* one (82.3%), so
roughly two points of the residual is in the feed's interaction rather than in the
epilogue, and it is not the ring index, not the barrier, not the issue rate and
not the producer's instruction count — all four are now measured at zero.

**The residual against upstream is neither the feed nor the issue stream**, and a
static diff settled it for no container time. Upstream's producer charges
`(A_TILE_BYTES + B_TILE_BYTES) * 2` per K block per cluster — **64 KB, identical to
ours** — as three `cp.async.bulk.tensor` per rank under a `self_mask`, one `A` and
two 64-row `B` panels, under its own comment *"the fixed host descriptor exposes a
64x64 B panel"*. Same bytes, same instructions, same mask, same four stages, and
therefore the same 95.5% duty. And `_print_mma_stream` counts non-`mma`
instructions between the first and last `tcgen05.mma`: **8.8 per issue before the
fold, 7.5 for upstream, 2.9 after** — we are 2.6× leaner than upstream and still
3.5 points behind on the rate. So **static instruction count between MMA issues is
not the binding constraint.** It does *not* foreclose the barrier-address lever,
which an earlier reading of this claimed it did: that census counts each
instruction once, a spin loop costs its body times its iterations, and no static
count between issues can see the iterations. Counted directly, the poll body is
**7 instructions pre-fold, 7 for upstream, 8 shipped** — ours needs an `add.s64`
upstream does not, because upstream gives each mbarrier its own static `.shared`
symbol (`__shared_mem_27..29`) where ours are const offsets into one dynamic
allocation. **The lever's sign is positive, and the pre-fold
spelling is not the counter-example it looks like.** That spelling had upstream's
7-instruction poll body and was 4% slower — but the fold changed *two* things with
opposite signs: it added one instruction to the poll body and removed many from the
inter-stage path. It is a two-variable change and therefore not a controlled
experiment on poll-body length at all. Removing the per-poll `add.s64` is the
single-variable version, and it is **additive with the fold rather than opposed to
it**.

The magnitude is worth stating in advance because it is large enough to matter. The
MMA warp's time in the poll loop is `whole − issue only` = 0.3503 − 0.2758 =
**0.0745 µs a K block, 21% of it**. If that loop is issue-bound, one instruction of
eight is an eighth of it — **0.0093 µs, 2.7% of the K-block rate** — which would
take the K loop from 78.8% to about 81.0% against upstream's 82.3% and close most of
the residual. The named failure mode: if `mbarrier.try_wait.parity` is instead
latency-bound on its shared read, the body's instruction count is a small fraction
of each iteration and the win is much smaller. That is the prediction and its
falsifier, and it is what a container should be spent against.

**Feasibility is settled and the cost is structural rather than budgetary.**
Upstream declares `static mut TMA_BAR0: Barrier = Barrier::UNINIT;` — eight of
them — *inside* its `#[kernel]` body, and cuda-oxide at the pin places those in
`.shared` as `__shared_mem_27..29`. So the mechanism exists and is proven in this
tree. Two costs, neither of them residency: the barriers are ~104 B moving from the
dynamic allocation to static shared out of the same 233472 B, so `dynamic_shared`
in each launch contract drops by that much and nothing crosses an occupancy step;
and **it is unproven from where this kernel would need it.** Upstream declares its
statics inside the `#[kernel]`, while `small_body` and `large_body` are deliberately
*outside* `#[cuda_module]` so the shipped kernels and the arms can be one text —
and whether a `static mut Barrier` in a plain `fn` still lands in `.shared` is the
macro's business, not the type's. If it does not, the statics have to be declared
per `#[kernel]` and threaded in, which is the design cost. Either way it is a
**library-level** change — `src/sync.rs`'s `Semaphore` takes a pointer derived from
a dynamic base and the `*_offset` consts are that base's layout, which #137 made
one walk — so it pays every kernel with a pipeline and does not belong in this PR. Filed as **#150** — and **closed**: `ptxas` folds the add, and all three poll bodies are three instructions in SASS. See the SASS section below.

One thing the poll body says that is new: **five of its seven or eight
instructions are not addressing at all** — `selp.b32`, `and.b32`, `setp.ne.b32`,
`not.pred`, `bra`, a predicate turned into an `i32`, masked, turned back into a
predicate and inverted, because `mbarrier_try_wait_parity` returns a Rust `bool`
and the loop is `while !…`. A tight spin is `try_wait` then `@!p bra`, which is
two. Upstream pays the same five, so this does not explain the gap against it — but
it is five of eight instructions in every mbarrier spin in the repo. Filed as
**#151** — and **closed**: the SASS step below was built, and `ptxas` folds the whole
round trip. Both were PTX counts of instructions the machine never runs.

**What this does not separate.** Why the feed in situ costs 24.5% at 95.5% duty
and 0.0% at 55.7% is a duty-cycle argument, not a mechanism: **bank conflicts
between the TMA's writes and `tcgen05`'s operand reads are not distinguished from
saturation of the shared-memory write port, and no arm in this file can distinguish
them** — every arm removes a phase, so each removes both candidates at once, and
bytes are the only dial. The arm that would separate them changes the *access
pattern* at constant bytes and constant instruction count: `A` and `B` staged at
swizzles or pitches that collide differently while the TMA still moves 64 KB a K
block. Bank conflicts would move; a saturated write port could not. Likewise the
3.5 points still behind upstream is latency or ordering, and the instrument for
*that* is a per-warp `%globaltimer` span probe, whose own perturbation would have
to be measured against `whole` before any of its numbers counted. The per-tile constants the ladders fit are two-point fits of
small differences between large numbers, so only the large one — the drain's 2.9 µs
a tile, which reproduces #147's independently measured 2.82 — is quoted.

**The consequence for tiling, which is somebody else's to act on.** The term is the
tile's operand bytes per flop, so the lever is the tile, and #146 swept it and
found `[512, 256]` at 4096³ a dead heat (0.985) because it halves the tile count
and so doubles the per-tile constant each tile must amortize. This file prices that
constant at 3.4 µs a tile, of which the drain is 2.9. If the drain falls, the
crossover should move below 8192 — the entry that is already at 99.3% of peak in
its K loop would then be the right one at 4096³ too.

##### and the PTX this repo counts is not always the machine's — a SASS check, and what it does and does not disturb

Nearly every conclusion in the GEMM work above is a **PTX** count. `regcount`'s
opcode census is PTX. The censuses that validate every ablation arm are PTX. The
between-MMA instruction table that foreclosed the instruction-count family is PTX,
and the `#[unroll]` stage fold was both diagnosed and verified in PTX. `ptxas` is a
real optimizer sitting between all of that and the machine, and until #148 nothing
here had ever checked one against the other.

`modal_app.py::_print_sass_loops` is that check, and it is general rather than a
one-off: point it at a kernel and it prices every tight loop the machine actually
runs. It is CPU-only — `nvdisasm -c` over the cubin `build` already assembles — and
**it finds loops by backward branch rather than by mnemonic**, so it needs nobody to
know how a Blackwell mbarrier wait is spelled in SASS, which is the part nobody here
should be guessing at. `insns` is instructions per iteration, which is the figure a
PTX loop-body count is comparable to.

**Its first use killed two levers it was built for.** #150 and #151 counted the
`mbarrier.try_wait.parity` spin body in PTX: eight instructions, of which one was an
`add.s64` we paid and upstream did not (our barriers are offsets into one dynamic
allocation, upstream's are static `.shared` symbols) and five were a predicate
round-tripped through an `i32` because the intrinsic returns a Rust `bool`. In SASS
the whole loop is:

```
YIELD ;
SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R5+URZ], R4 ;
@!P0 BRA `(.L_x_1845) ;
```

| | PTX poll body | **SASS poll body** |
| --- | ---: | ---: |
| upstream `gemm_sol_clc_multicast_4_stage_pipeline` | 7 | **3** |
| `gemm_sol_m256_runtime` | 7 | **3** |
| `gemm_sol_m256` as shipped | 8 | **3** |

`ptxas` folds the predicate round-trip to nothing — `TRYWAIT P0` straight to
`@!P0 BRA` — and holds the barrier address in a register across the loop
(`[R5+URZ]`), so the `add.s64` and the symbol rematerialization are both gone.
**All three poll bodies are the same three instructions**, the `YIELD` is `ptxas`
*adding* a spin hint rather than overhead, and #150's advance prediction (2.7% of
the K-block rate) is dead — not through its named falsifier but because the
instruction it was about does not exist. Both issues closed, no container spent.

**It also confirmed the one claim that mattered.** #148 attributes the stage fold's
+2.2–4.3% to the inter-stage critical path and explicitly *not* to the poll body,
after an earlier reading had it the other way round. The shipped and pre-fold spin
loops are **identical at 3 instructions**, so the fold's win cannot have come from
the poll body at any level; and the two kernels' loop tables differ in exactly one
row — 10 instructions an iteration where the pre-fold spelling has **14** — in a
loop that is not the poll loop. That is the fold, at the machine level, in the place
#148 said it was.

**What this disturbs, and what it does not.** Being concrete matters more here than
being alarmed, and most of what is above survives for a structural reason: **it is
comparative between two of our own kernels through one toolchain**, and a uniform
`ptxas` transform applies to both sides of such a comparison.

Safe, and why:

- **Every ablation arm's census.** `feed only` showing 0 in the `mma` column is a
  presence-or-absence claim about a whole phase, and `ptxas` cannot invent a
  `tcgen05.mma` that the PTX does not contain. Removal claims are robust in a way
  cost claims are not.
- **`regcount`'s registers, spills, stack frames and the occupancy gate.** Those
  come from `ptxas -v`, which is already the far side of the optimizer.
- **#125's `selp` count.** That was a claim that a flag folded *at PTX level*, i.e.
  that LLVM had already done it. Learning that `ptxas` folds more only makes that
  check conservative: zero in PTX is still zero in SASS.
- **#148's between-MMA table read as a negative.** The conclusion drawn from it was
  that we are leaner than upstream and *still* slower, so instruction count between
  MMA issues is not the binding constraint. If `ptxas` folds even more of both
  sides, that negative gets stronger rather than weaker.

Not safe:

- **Any absolute "this costs N instructions" from a PTX count.** #150 and #151 were
  exactly that, and both were zero. The ratios in #148's between-MMA table (2.9
  against 7.5 per issue) should be read as directional, not as magnitudes: if
  `ptxas` folds five of eight instructions in a spin body, it plausibly folds a good
  deal of the scalar chain between MMA issues too, and nobody has counted that in
  SASS.
- **PTX counts across *different* code shapes**, where the two sides may not receive
  the same transform. Comparing our kernel to upstream's is the case to be careful
  with; comparing our kernel to our own kernel one const apart is the case that is
  fine.

The pattern worth keeping: **a PTX count is a hypothesis, and a timed arm or SASS is
the evidence.** The epilogue work reached the same place from the other direction —
its census showed `.pack::16b` removing `cvt.rn.bf16x2` outright, 8 to 0, and a
legal `nocvt` control that removed the same converts measured 0.0–1.4% *slower*, so
the convert was never the cost the count implied. Two independent instruments, one
conclusion about the instrument set.

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

- **`expect_tx` byte accounting is hand-summed.** *Closed by #29.* Every
  producer wrote `(ATile::BYTES + BTile::BYTES) as u32` and had to keep it in
  step with the loads it issued. Both shapes above were on the table and the
  second one landed: `tma_load` returns a `TransactionBytes`, `expect_tx` takes
  nothing else, and the total is the sum of the calls that were made.
  `expect_tiles(&[…])` would have been the same fact stated twice — a list can
  omit a tile exactly as a sum can omit a term. The cluster case is that total
  times the ranks staging into the barrier (`across_ranks`), which is the
  interface #50 promised it. `src/sync.rs`'s module docs carry the argument.
- **Coordinate-dependent ops need `lane` passed in.** Settled by #27:
  implicit for ops that execute, explicit for pure coordinate queries. The
  masks (#7) take it explicitly, on that rule — `device-tests` and `reg.rs`
  build their expectations by calling the map across all 32 lanes on the
  **host**, where `warp::lane_id()` does not exist, and that is what makes the
  mask's own test non-vacuous rather than a restatement of the kernel.

---

## What the examples confirm already works

Worth recording, because the promotion of an example from aspirational to
compiling is the thing that proved an issue finished, and every kernel here has
now made that promotion:

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
