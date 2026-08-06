//! # GEMM, warp-specialized — the same `C = A·Bᵀ` at one CTA per SM
//!
//! **Status: runs.** [`check`] launches it against the same CPU reference
//! [`crate::gemm::check`] uses, on a B200, with the same exact `==` on bf16
//! words. It is a **variant of [`crate::gemm`] and not a replacement** — that
//! kernel is untouched, and this file exists so the two can be A/B'd.
//!
//! ## What is different, and it is one thing
//!
//! Both kernels compute the same `[256, 256]` pair tile with the same
//! `cta_group::2` UMMA over the same four-chunk `BLOCK_K = 64` stages, and
//! both drain the accumulator through the same staged epilogue — TMEM →
//! registers → `stmatrix` into a per-warp `[32, 64]` shared tile → 16-byte
//! stores. **The epilogue is deliberately the same shape on both sides**; the
//! one thing that differs is the instruction width, and #118 measured why.
//!
//! [`SHIPPED_ENTRY`] is `staged8` here and
//! [`crate::gemm::SHIPPED_EPILOGUE`] is `staged84` there, because `.x4` buys
//! 14 registers on that design point and 2 on this one. See "The widths, and
//! the floor under them" below, and do not tidy the two into one choice.
//!
//! **Table 1 of [`compare`] still puts both kernels on the register drain**,
//! which is what makes it a measurement of the occupancy/specialization
//! structure and nothing else; table 5 is where each side gets its best rung.
//!
//! What moves is *where the overlap comes from*.
//!
//! ## Since #15 the epilogue is a ladder, and it is a control
//!
//! [`Entry::Staged`] is [`Tile::drain_staged`] in place of [`Tile::drain`]:
//! `stmatrix` into a per-warp `[32, 64]` shared tile and 16-byte stores out of
//! it, which is the epilogue [`crate::gemm`] gained 2–8% from. It is here
//! because this kernel is the one that can say *why* that worked. There the
//! epilogue is the critical path — #114 measured it as fully exposed — and here
//! it is already deferred one item and already on warps of its own, so a gain
//! that survives the move is the store's shape and not its placement.
//!
//! **It survives: +4.1% / +2.5% / +3.0% at 4096³ / 8192³ / 16384³**, moving
//! this kernel from 0.804 to 0.829 of cuBLASLt at the largest size. Registers
//! go 168 → 44 with no spill, and residency cannot move — 512 accumulator
//! columns fixed it at one CTA an SM before shared memory was consulted, and
//! 147 584 B is well inside the 233 472 an SM has.
//!
//! #118 hangs #117's two instruction widths off the same rung —
//! [`Entry::StagedX8`], [`Entry::StagedX4`], [`Entry::StagedX8X4`] — plus the
//! two `no drain` controls the ladder is subtracted from. All of it is
//! [`drain`], one const on one job, and "The widths, and the floor under them"
//! below is what it measured. `experiments/README.md` §7 has both kernels' tables.
//!
//! [`crate::gemm`] gets it **across CTAs**. It is 114816 B of shared memory on
//! the epilogue it ships (98392 on the register drain) and 256 accumulator
//! columns, which is two CTAs an SM either way, so one CTA's epilogue runs
//! against another CTA's MMA. That is what forces its whole budget: 1920 B of
//! shared headroom at the shipped envelope, and an occupancy step at 168 that
//! its register drain sat two registers under.
//!
//! This kernel gets it **inside one CTA**, from warp specialization plus a
//! double-buffered accumulator — which is NVIDIA's `gemm_sol_final`
//! (cuda-oxide `20a5616`,
//! `crates/rustc-codegen-cuda/examples/gemm_sol_final/`), reproduced in this
//! repo's idiom. Six warps per CTA:
//!
//! | warp | role |
//! | ---: | --- |
//! | 0–3 | the epilogue — LDTM the accumulator, store `C` |
//! | 4 | the TMA producer |
//! | 5 | the pair-UMMA issuer (leader rank only) |
//!
//! and **two accumulator stages** in tensor memory ([`ACCUM_STAGES`]), so the
//! MMA warp fills stage `s` while the epilogue warps drain stage `1 - s` from
//! the item before.
//!
//! ## Tensor memory picks the residency, it is not imposed
//!
//! Residency is `min(512 / accumulator_columns, shared per SM / plan)` (#84).
//! Two stages of [`BLOCK_N`] columns is **512 columns — the whole SM's tensor
//! memory** — so `512 / 512 = 1` and one CTA per SM falls out of the
//! double-buffer rather than being chosen. [`kittens::tmem::TmemTile`] already
//! says this: [`kittens::tmem::TmemTile::columns_right`] is the ping-pong.
//!
//! Having given up the second CTA, the kernel then spends the budget the second
//! CTA was holding. The shared plan is [`SHARED_BYTES`] against the 233 472 B
//! an SM divides, so shared memory lands on the same 1 and neither resource is
//! the odd one out. It is worth noting that **`shared_plan(STAGES)` here is
//! byte-for-byte [`crate::gemm`]'s `[256, 256] @ STAGES = 4` rung** — the one
//! its `RUNGS` table computes and refuses to build, because at four warps that
//! plan buys one CTA an SM and nothing to do with it. At six warps and two
//! accumulator stages there is something to do with it, which is the whole
//! argument of this file in one sentence.
//!
//! ## Registers stop binding
//!
//! The register file is 65536 per SM over four sub-partitions — 16384 each,
//! granted per warp in units of 256. [`crate::gemm`] runs 2 CTAs × 4 warps = 8
//! warps an SM, 2 per sub-partition, 8192 registers each: 256 a thread, which
//! is why 168 is a step and why that kernel sits at 166 with two registers of
//! headroom. Here it is 1 CTA × 6 warps = 6 warps an SM, so the *binding*
//! sub-partition still holds 2 and **255 registers a thread is reachable**.
//!
//! **`regcount` says 168 and no spill on the register drain**, against
//! [`crate::gemm`]'s 166. So the headroom is real and *nothing here asked for
//! it*: that epilogue holds the same `RegTile<32, 128>` band it always did,
//! because 256 columns in one band is 256 fp32 a thread and past the
//! architecture's 255 at any occupancy — [`DRAIN_N`] is a hardware limit, not
//! an occupancy one. This is an argument that turned out to be inert, and it
//! is recorded that way rather than deleted, because "registers stop binding"
//! was one of the two consequences this design point was reached for.
//!
//! The epilogue this file now ships is not near it either, from the other
//! direction: `staged8` is **94 registers and no spill**, because a `[32, 64]`
//! band and a block-at-a-time `stmatrix` never materialize what
//! [`kittens::global::store_rows`]' slot-major walk forced live.
//!
//! ## Why [`kittens::pipeline`] did not have to change
//!
//! [`kittens::pipeline::Job::work`]'s doc says role dispatch lives in the job,
//! and it does: [`run`](kittens::pipeline::run),
//! [`run_stealing`](kittens::pipeline::run_stealing),
//! [`ClcQueue`](kittens::pipeline::ClcQueue) and
//! [`grouped`](kittens::pipeline::grouped) are used **unmodified**, and the
//! whole rewrite is inside [`Ws::work`]. Two things are worth recording about
//! how that lands, because neither was obvious.
//!
//! **The item boundary is what lets the pending accumulator be safe.** The
//! epilogue warps drain item `i - 1` while the MMA warp fills item `i`, and
//! nothing inside the item orders those against each other — they touch
//! disjoint TMEM columns, so nothing has to. What *does* need ordering is item
//! `i + 1`'s MMA against item `i - 1`'s drain, since those share a stage, and
//! [`kittens::pipeline::run`]'s per-item `cluster_sync` (with its
//! `tcgen05_fence_before_thread_sync` in front) is exactly that rendezvous, at
//! exactly the right scope. This is [`crate::gemm`]'s `Lcsf` finding again —
//! `lcsf` wants the boundary's *scope* moved inside the item — and here the
//! scaffold's own boundary supplies it for free.
//!
//! **The one thing the trait cannot express is a warp-asynchronous item
//! stream.** The reference has no per-item rendezvous at all: its TMA warp
//! publishes `(tile_m, tile_n)` through a shared `TILE_INFO` word and a
//! `TILE_READY` mbarrier, and its three warp groups sit at *different* items
//! simultaneously, bounded only by the two accumulator stages. A [`Job`] is
//! entered by every thread with one `item`, so the warps here are always on the
//! same item and the epilogue can lag the MMA by exactly one. That is a real
//! difference in the *slack* the two designs have — one item against two — and
//! it is the part of `gemm_sol_final` this port does not reproduce. It is not a
//! bug in the trait; it is the price of the trait's guarantee that barriers can
//! be re-armed per item. #132 took that argument up into [`Job`]'s own doc,
//! where a reader of the scaffold meets it without having to find this file.
//!
//! Which brings the second consequence, and it is in our favour: because every
//! item re-arms its own barrier set behind the boundary, **every item starts at
//! ring stage 0 and phase 0**. The reference needs `k % 256 == 0` so that
//! `k_iters % 4 == 0` and its producer's global stage index agrees with its
//! consumer's local one across a tile boundary. We need no such thing. This
//! kernel's shape contract is `m % 256 == 0`, `n % 256 == 0`, `k % 64 == 0` —
//! the reference's on `n`, weaker on `k`, and the item boundary is what buys
//! the difference.
//!
//! ## What this file deliberately did not change, and has since
//!
//! The reference's epilogue is `stmatrix` into a 64 KiB shared staging buffer
//! and 64-bit vectorized stores out of it. **That was not here at #112**: this
//! kernel kept [`crate::gemm`]'s register → global epilogue exactly, because
//! otherwise the measurement would move the occupancy/specialization structure
//! and the epilogue in one change and no row of the table would say which one
//! paid.
//!
//! It is here now. #116 built the staged shape, #118 built #117's two widths on
//! top of it, and #119 made `staged8` [`SHIPPED_ENTRY`] — each measured on its
//! own, which is what the discipline above bought. The register drain stays as
//! [`Entry::S4`] and table 1 of [`compare`] still runs on it, so #112's
//! comparison is still a comparison of one variable.
//!
//! The epilogue needed no modification to run at six warps, which is worth
//! stating because it was the open question: warps 0–3 cover the CTA's 128
//! accumulator rows at 32 rows each exactly as they did at four warps, and
//! warps 4 and 5 simply do not enter it. [`kittens::global::store_rows`] is not
//! warp-collective — its own docs say so — so a warp that does not call it
//! leaves nothing ill-formed.
//!
//! ## What it measured, and it **loses**
//!
//! `cargo oxide run kittens-experiments -- ws`, which is
//! `scripts/modal-run ws_bench`, runs this kernel and [`crate::gemm`] and
//! cuBLASLt in one container at 4096³, 8192³ and 16384³ — the three sizes the
//! reference publishes, and no smaller, because below 4096³ cuBLASLt's own
//! run-to-run spread reaches 77% and no ratio there is quotable. The control is
//! re-measured in the same session rather than quoted from
//! `experiments/README.md` §7: this change does not touch [`crate::gemm`], but #98
//! found 2.9% of drift between containers and #109 came within a paragraph of
//! publishing a false +3.6% against a baseline that had moved under it.
//!
//! The question is narrow: **does giving up the second CTA for warp
//! specialization win, lose, or wash on a B200 with our epilogue?** One B200
//! session, min of 30 timed launches, every row checked element-by-element
//! first:
//!
//! | shape | `gemm` | this, 4 stages | this, 6 stages | vs `gemm` (best) |
//! | --- | ---: | ---: | ---: | ---: |
//! | 4096³ | 951.9 | 988.0 | 1002.3 | **+5.3%** |
//! | 8192³ | 1492.9 | 1390.1 | 1451.5 | **−2.8%** |
//! | 16384³ | 1669.1 | 1536.5 | 1547.6 | **−7.3%** |
//!
//! in TFLOP/s, and against cuBLASLt in the same container (1569.8 / 1828.9 /
//! 1904.6 TFLOP/s) that is 0.606 → 0.629, 0.816 → 0.760 and 0.876 → 0.807.
//! CLC is worth a further +0.6% / +1.0% / +1.2% on top of the static schedule
//! and does not change any sign.
//!
//! **So: a small win where the launch is short and a clear loss where it is
//! not, and the loss grows with size.** The 4096³ row is not a wave-efficiency
//! artifact — both kernels tile `C` identically and both run at 86.5% wave
//! efficiency there, [`crate::gemm`] as two waves of 148 clusters and this as
//! four waves of 74 — so the crossover is real and it is about what the SM is
//! doing rather than about how the grid quantizes.
//!
//! ### The mechanism the numbers point at, which is the epilogue — **wrong,
//! and #118 is what says so**
//!
//! The rest of this section is kept as written because the correction is worth
//! more than the paragraph. Its conclusion was that the missing 7% is the
//! epilogue, on the grounds that an SM holding one CTA has nothing else to hide
//! one behind. **It is not the epilogue.** #118 removed the epilogue entirely —
//! [`kernels::gemm_ws_no_drain`], one launch parameter and one const apart from
//! the register drain — and the epilogue-free launch still runs at **0.888 of
//! cuBLASLt at 16384³**, where #114's identical probe on [`crate::gemm`] ran at
//! **1.02**. It is also **3.0% slower than `gemm`'s complete `staged84`
//! launch**. A free epilogue would not close this gap, so the gap was never
//! the epilogue. See "The widths, and the floor under them".
//!
//! Two facts narrow it. First, `regcount` says **168 registers and no spill**,
//! against [`crate::gemm`]'s 166 — so the whole "255 registers are reachable"
//! argument above is *true and bought nothing*, because nothing in this kernel
//! wanted them. The register headroom is real and unused. Second, six pipeline
//! stages instead of four is worth +1.4% / +4.4% / +0.7% and costs nothing any
//! resource can see, which says the K pipeline was mildly starved and is not
//! where the missing 7% is.
//!
//! What is left is that **an SM holding one CTA has nothing else to hide the
//! epilogue behind.** Under [`crate::gemm`]'s two CTAs an SM, one CTA's LDTM
//! and its 256 scattered pair stores run against the *other* CTA's MMA, and
//! that overlap costs no barrier and no slack. Here the same work is hidden
//! only by the one item of slack the accumulator ping-pong provides, and every
//! item ends in a `done.wait` that stalls the whole SM rather than one of two
//! CTAs on it. #108's probes priced this epilogue at 21.4 µs a tile serial,
//! 20.2% of an 8192³ launch — which is more than enough to be the whole of the
//! gap, and is the largest thing on the SM that the second CTA used to cover.
//!
//! That is a statement about *this* epilogue and not about warp specialization,
//! and it is exactly why the scope discipline was worth keeping: the reference
//! spends its surplus shared memory on `stmatrix` into a staging buffer and
//! hands the global write to 64-bit vectorized stores, which is the change this
//! PR deliberately does not make. The follow-up — `kittens::epilogue::StoreRing`
//! (#111) plus the shared→global mover being built beside it — now has a
//! sharpened question rather than a general hope: it has to find about 7% at
//! 16384³ before this design point is even level, and if it does, it will have
//! found it on the kernel that has the shared memory to spend.
//!
//! The reference's 1524 / 1868 / 2162 TFLOP/s are a B300 against an
//! FP16-in/FP16-out cuBLASLt; ours is a B200 at bf16 where cuBLASLt does
//! 1.57–1.90 PFLOP/s in the session above. Their numbers were never a target
//! this had a right to expect, and the deliverable was the sign of the delta.
//! The sign is negative, and this file stays in the tree because
//! `experiments/README.md` §7 keeps losers on purpose.
//!
//! ## The widths, and the floor under them — #118
//!
//! #117 took [`crate::gemm`]'s staged epilogue to `.x8` LDTM and
//! `stmatrix.x4` and found +23.1% / +8.8% / +5.1%, on a mechanism it stated
//! precisely: [`TmemTile::tile`] waits after **each** `.x1`, because the
//! registers it waits on *are* the load's return value, so the drain never has
//! two loads in flight and a `[32, 64]` band pays 16 fully exposed
//! tensor-memory latencies. `.x8` is 2 and 2. **The win is the wait and not
//! the issue** — `stmatrix.x4` halves an instruction count on its own and is
//! worth −0.6% to −1.1%, which is how that was established.
//!
//! That mechanism makes a prediction about *this* kernel, and it is not the
//! obvious one. A latency nobody is covering is worth removing; a latency that
//! is already covered is not. Here the drain is deferred one item, sits on
//! warps of its own, and the producer never stops — so if the prediction holds,
//! `.x8` should be worth **less** here than there, and the amount by which is a
//! measurement of how much cover this design point actually provides.
//!
//! One session, min of 30 timed launches, every row element-by-element exact
//! before it was timed:
//!
//! | shape | `ws s4` | `staged` | `staged8` | `staged4` | `staged84` |
//! | --- | ---: | ---: | ---: | ---: | ---: |
//! | 4096³ | 0.1382 | 0.1340 | **0.1158** | 0.1324 | 0.1171 |
//! | 8192³ | 0.7881 | 0.7649 | 0.7201 | 0.7648 | **0.7199** |
//! | 16384³ | 5.7904 | 5.6645 | **5.5015** | 5.6604 | 5.5028 |
//!
//! in milliseconds. Against `staged`, `.x8` is **+15.7% / +6.2% / +3.0%** and
//! `.x4` is **+1.2% / +0.0% / +0.1%** — a clean null, exactly as in
//! [`crate::gemm`]. Composed, **+14.5% / +6.2% / +2.9%**: the two do *not*
//! add here, and `staged8` and `staged84` are indistinguishable, trading places
//! between sessions. Against the register epilogue this kernel shipped through
//! #119, the best rung is **+18.1% / +9.5% / +5.2%**, taking it from 0.615 to
//! 0.726, 0.782 to 0.856 and 0.810 to 0.852 of cuBLASLt — and it is `staged8`
//! that ships, because a tie between it and `staged84` goes to the rung whose
//! `.x4` bought nothing here.
//!
//! **The prediction holds, and the register column is why `.x4` differs.** In
//! [`crate::gemm`] `.x8` cost +52 registers (42 → 94) and `.x4` bought 14 of
//! them back (94 → 80), which is the whole of why that kernel's composition
//! beat `.x8` alone at 16384³. Here `.x8` costs the same +50 (44 → 94) and
//! `.x4` recovers **2** (94 → 92) — so it buys no liveness, and it buys no
//! time. Zero spill in every rung, and 255 registers a thread is reachable at
//! six warps, so nothing here was ever near a ceiling.
//!
//! ### What the epilogue costs, and it is not twice the LDTM
//!
//! By #114's `whole − no drain`, over the items the busiest cluster walks, in
//! µs a tile — with one control per envelope, and the two controls
//! opcode-identical PTX at 28 registers apiece:
//!
//! | shape | `ws s4` | `staged` | `staged8` | `staged84` | LDTM half |
//! | --- | ---: | ---: | ---: | ---: | ---: |
//! | 4096³ | 8.90 | 8.65 | 4.10 | 4.42 | 4.55 |
//! | 8192³ | 8.62 | 6.97 | 3.77 | 3.76 | **3.20** |
//! | 16384³ | 9.06 | 6.76 | 3.85 | 3.87 | 2.91 |
//!
//! At 8192³ [`crate::gemm`]'s register epilogue is exposed at 20.43 µs a tile
//! (#114) and its staged one at 14.96, of which 8.07 was LDTM (#117). **The
//! same epilogue on this kernel is exposed at 8.62, 6.97 and 3.20** — 42%,
//! 47% and 40% of the cost, on the same instrument, for the same instructions.
//! That is what warp specialization and the accumulator ping-pong buy, stated
//! as a number for the first time, and it is why `.x8` is worth two thirds to
//! five sixths here of what it was worth there.
//!
//! **So "`gemm_ws`'s staged rung carries twice the LDTM (16 against 8)" is true
//! of the PTX and false of the work.** `regcount`'s opcode census does read 16
//! `tcgen05.ld` against `gemm_cg2_staged`'s 8 — and 8 against 4 for the
//! register rungs, 8 `stmatrix` against 4, 16 `cvt` against 8, every column
//! doubled. The cause is that this kernel emits its epilogue at **two call
//! sites**, [`Ws::work`] and [`Ws::finish`], where [`crate::gemm`]'s fused job
//! has one; `finish` runs once per cluster over a whole launch and once per
//! *item* never. A band is `RegTile<32, 64>` in both files, so the dynamic
//! LDTM per tile is identical, and the measurement above says this kernel's
//! LDTM half is not twice as expensive but **2.5× cheaper**.
//!
//! ### The floor, which is the finding
//!
//! [`kernels::gemm_ws_no_drain`] is this kernel with the epilogue deleted — one
//! const and one launch parameter from [`Entry::S4`] — and it is the probe
//! #114 ran on [`crate::gemm`] to conclude that the item boundary **is** the
//! epilogue and is fully exposed. There it reached 1850 TFLOP/s against
//! cuBLASLt's 1808, past parity. Here:
//!
//! | shape | no drain ms | TFLOP/s | of cuBLASLt | vs `gemm`'s `staged84` |
//! | --- | ---: | ---: | ---: | ---: |
//! | 4096³ | 0.1026 | 1339.2 | 0.829 | +3.0% |
//! | 8192³ | 0.6674 | 1647.4 | 0.923 | **−2.2%** |
//! | 16384³ | 5.2828 | 1665.1 | 0.888 | **−3.0%** |
//!
//! **At the two large sizes this kernel loses to `gemm`'s complete launch with
//! no epilogue at all.** Its best rung loses by 9.7% / 9.4% / 6.9%; deleting
//! the epilogue outright recovers 12.7 / 7.2 / 3.9 points of that and still
//! ends 2.2% and 3.0% behind at the two large sizes. So epilogue work could
//! reach three quarters of the 8192³ gap and only 57% of the 16384³ one even if
//! it were free — and #112's 7.3% closes to 6.9% after both kernels have spent
//! everything #116 and #117 found. What is left is the multiply and the operand
//! stream at one CTA an SM.
//!
//! Priced against the library, which is the only denominator that crosses
//! containers: at 8192³ #114's epilogue-free `gemm_cg2` ran at **1.02** of
//! cuBLASLt and this epilogue-free `gemm_ws` runs at **0.923**, so the two
//! epilogue-free kernels are about **10% apart** and the whole of the deficit
//! lives there. That is a ratio-of-ratios across two sessions and wants its own
//! container before it is quoted to three digits, but the sign and the order of
//! magnitude are not in doubt: they are the same 7–10% #112 has been carrying,
//! now measured with the epilogue taken out of both sides.
//!
//! Two things follow. The first is that `kittens::epilogue::StoreRing` (#111)
//! and the shared→global TMA mover no longer have "about 7% to find" here —
//! they have at most 3.0%, because that is all the epilogue still costs, and
//! the sharpened question has been sharpened out of existence on this kernel.
//! The second is that #112's own hypothesis is now refuted twice: #114 refuted
//! "the peer CTA hides `gemm`'s epilogue" by measuring that epilogue as fully
//! exposed, and this refutes "the epilogue is what `gemm_ws` cannot hide" by
//! deleting it and losing anyway. What has never been measured is what an SM
//! running one CTA of six warps does to the *K pipeline* against two CTAs of
//! four, and that is where the next probe belongs.

use cuda_device::cluster;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract};

use crate::bench::{Baseline, Shape, Timings, time};
use crate::gemm::{Scheduler, a_value, b_value, check_c, stage};
use std::error::Error;

use kittens::epilogue;
use kittens::global::{GlobalRows, store_rows};
use kittens::mma::{commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, ClcQueue, Job};
use kittens::plan::SharedPlan;
use kittens::reg::{BaseLdtm, RegTile};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{lane, warp_id};

/// Rows of `C` one CTA owns; the pair covers `2 * BLOCK_M`, which is the `M`
/// the widest `cta_group::2` instruction descriptor names.
const BLOCK_M: usize = 128;
/// Columns of `C` the pair computes, and this CTA's columns per accumulator
/// **stage**. Fixed at 256 rather than swept: [`ACCUM_STAGES`] stages of it
/// have to be the 512 columns an SM has, which is the whole design point.
const BLOCK_N: usize = 256;
/// This CTA's half of `B`.
const HALF_N: usize = BLOCK_N / 2;
/// K per pipeline stage: one 128-byte swizzle atom of bf16, the only width
/// [`SharedTile::k_walk`] accepts, and four chained K=16 MMA chunks.
const BLOCK_K: usize = 64;
/// Chained MMAs per stage.
const CHUNKS: usize = BLOCK_K / 16;
/// Pipeline depth over K in the shipped entry point — the reference's four.
const STAGES: usize = 4;
/// Accumulator stages in tensor memory: the MMA warp fills one while the
/// epilogue warps drain the other. `ACCUM_STAGES * BLOCK_N` must be the 512
/// columns an SM has, which is asserted below.
const ACCUM_STAGES: u32 = 2;
/// Columns [`alloc_cluster`] is asked for — both stages at once, because a CTA
/// allocates tensor memory exactly once and a stage is a column offset into
/// that allocation rather than an allocation of its own.
const ACCUM_COLUMNS: usize = ACCUM_STAGES as usize * BLOCK_N;

/// Warps that run the epilogue: 0..[`EPILOGUE_WARPS`], one per 32 accumulator
/// rows, which is the `[32, N]` band [`kittens::tmem::TmemTile::tile`] drains.
const EPILOGUE_WARPS: u32 = (BLOCK_M / 32) as u32;
/// The warp that issues this CTA's TMA loads.
const TMA_WARP: u32 = EPILOGUE_WARPS;
/// The warp that issues the pair's UMMA — in the leader rank only, since a
/// `cta_group::2` MMA is one instruction driving both CTAs' operands.
const MMA_WARP: u32 = EPILOGUE_WARPS + 1;
/// Warps per CTA, and the whole of what "warp-specialized" costs in threads.
const WARPS: u32 = MMA_WARP + 1;
/// Threads the launch declares.
pub const THREADS: u32 = 32 * WARPS;

/// Accumulator columns one warp drains in a single band.
///
/// [`BaseLdtm::WIDEST_BAND`] carries the derivation, and the register headroom
/// this design buys does not move it: it is the architecture's 255 registers
/// against one fp32 a thread per column, at any occupancy. A 256-column stage
/// therefore drains in two bands, exactly as [`crate::gemm`] does.
const DRAIN_N: usize = BaseLdtm::WIDEST_BAND;
/// Accumulator columns one warp drains in a single band of the **staged**
/// epilogue — see [`crate::gemm`]'s `STAGE_N`, which this is a copy of and for
/// the same two reasons: `SharedTile::WIDTH_OK` wants a whole swizzle subtile,
/// and 64 columns is the narrowest bf16 tile `Swizzle128B` admits.
///
/// The budget argument is *not* the same one, and it is much slacker here. This
/// kernel is one CTA an SM by tensor memory alone — [`ACCUM_COLUMNS`] is all
/// 512 an SM has — so it has the whole 233 472 B to spend and
/// [`SHARED_BYTES`] spends 131 176. What decides the width is therefore the
/// register file and the tile shape, not the plan.
const STAGE_N: usize = 64;
/// One epilogue warp's staging tile — [`crate::gemm`]'s `StageTile`, at this
/// kernel's four epilogue warps. Its band is 64 fp32 a thread against
/// [`Band`]'s 128.
type StageTile = SharedTile<Bf16, { BaseLdtm::WARP_ROWS }, STAGE_N, Swizzle128B>;
/// CTAs in the cluster, and the multiplier on a stage's transaction charge.
const RANKS: u32 = 2;
/// The CTA mask naming every half of the pair.
const PAIR: u16 = ((1u32 << RANKS) - 1) as u16;
/// The rank that owns the pair's MMA and its stage barriers.
const LEADER: u32 = 0;

type ATile = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
type BTile = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing<const STAGES: usize> = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing<const STAGES: usize> = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
/// One accumulator stage: this CTA's 128 TMEM lanes by [`BLOCK_N`] fp32
/// columns. Two of these are the SM's whole tensor memory.
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;
/// One warp's band of it, drained [`DRAIN_N`] columns at a time.
type Band = RegTile<32, DRAIN_N, BaseLdtm>;

/// The staging run, as a ring of one [`StageTile`] per epilogue warp — the
/// per-warp offset is [`SharedTileRing::tile`]'s arithmetic rather than a
/// written-down `warp_id * StageTile::BYTES`.
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, { EPILOGUE_WARPS as usize }>;

/// Everything the launch's dynamic shared memory holds before the staging run,
/// in declaration order — the same walk [`crate::gemm`] has, because the
/// pipeline this kernel drives is the same pipeline.
struct Shared<const STAGES: usize> {
    a_ring: ARing<STAGES>,
    b_ring: BRing<STAGES>,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    tmem_slot: *mut u32,
    queue: ClcQueue,
    /// The cursor past the queue: [`SharedPlan::bytes`] of it is
    /// [`shared_plan`], and it is where [`staged`] picks up.
    plan: SharedPlan,
}

/// The plan, as one walk — [`SharedPlan::attach`] for the handles,
/// [`SharedPlan::sizing`] for the envelope.
#[inline(always)]
const fn shared<const STAGES: usize>(at: SharedPlan) -> Shared<STAGES> {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (done, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    let (queue, at) = at.clc_queue();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        done,
        tmem_slot,
        queue,
        plan: at,
    }
}

/// The staging run on the end of it, 128-byte aligned by
/// [`SharedPlan::tile_ring`].
#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, { EPILOGUE_WARPS as usize }>()
}

/// [`shared`]'s walk as a `const fn` over *values*, because a const parameter
/// cannot be a function argument and the host needs this number outside any
/// monomorphization — `#[launch_contract]` takes a literal and [`Entry::shared`]
/// answers for a depth chosen at runtime.
///
/// [`crate::gemm`]'s `shared_cursor` states the same limit at more length, and
/// [`kernels::attach`] carries the one assert that joins the two.
const fn shared_cursor(stages: usize) -> SharedPlan {
    let at = SharedPlan::sizing();
    let (_, at) = at.reserve(BLOCK_M * BLOCK_K * 2 * stages, SharedPlan::TILE_ALIGN);
    let (_, at) = at.reserve(HALF_N * BLOCK_K * 2 * stages, SharedPlan::TILE_ALIGN);
    let (_, at) = at.barriers(2 * stages + 1);
    let (_, at) = at.tmem_slot();
    let (_, at) = at.clc_queue();
    at
}

/// Dynamic shared memory a `stages`-deep launch asks for: the two operand
/// rings and the scratch tail.
pub const fn shared_plan(stages: usize) -> usize {
    shared_cursor(stages).bytes()
}

/// Dynamic shared memory a **register-drain** launch must provide —
/// [`Entry::S4`], [`Entry::S6`] at its own depth, and [`Entry::NoDrain`].
///
/// **Not the shipped envelope since #119.** [`SHIPPED_ENTRY`] is `staged8` and
/// declares [`STAGED_SHARED_BYTES`]; this is the plan the staged one is laid on
/// top of, byte for byte, which is what keeps an epilogue A/B a comparison of
/// drains and 16 408 declared bytes and nothing else.
pub const SHARED_BYTES: usize = shared_plan(STAGES);

/// [`shared_plan`] with a staging tile per epilogue warp on the end, 128-byte
/// aligned — the envelope [`Entry::Staged`] declares.
///
/// See [`crate::gemm`]'s `staged_plan`, whose arithmetic this is. What differs
/// is that nothing here is close to a limit: this kernel is one CTA an SM
/// because [`ACCUM_COLUMNS`] is the SM's entire tensor memory, so the 16 384 B
/// come out of 102 296 B of slack rather than out of 18 344.
pub const fn staged_plan(stages: usize) -> usize {
    staged(shared_cursor(stages)).1.bytes()
}

/// The staged entry points' envelope, and the literal their contracts repeat
/// — since #119 **the envelope the shipped launch declares**, since
/// [`SHIPPED_ENTRY`] is one of the four rungs that carry staging tiles.
///
/// All four declare it, and so does their `no drain` control: #117's two
/// instruction widths change what the epilogue issues and not what it
/// occupies.
pub const STAGED_SHARED_BYTES: usize = staged_plan(STAGES);

/// `#[launch_contract]` takes literals, so the envelope is written twice and
/// this is what keeps the two in step — the same join [`crate::gemm`]'s
/// `SHARED_BYTES` assert makes, and for the same reason: past 48 KiB the plan
/// needs the opt-in the prepared-launch path issues (#70).
///
/// The tensor-memory line is the one that matters to the design. Two
/// accumulator stages of [`BLOCK_N`] columns is all 512 an SM has; a third
/// stage or a wider `N` would be inexpressible rather than slow.
const _: () = {
    assert!(THREADS == 192);
    assert!(SHARED_BYTES == 131_176);
    assert!(STAGED_SHARED_BYTES == 147_584 && STAGED_SHARED_BYTES <= 233_472);
    assert!(shared_plan(6) == 196_744);
    assert!(ACCUM_COLUMNS == 512);
    assert!(BLOCK_N % DRAIN_N == 0);
};

/// SMs on the device this project targets and measures on — a B200.
const SMS: u32 = 148;
/// CTAs of this kernel one SM holds. **Predicted from
/// `min(512 / columns, shared per SM / plan)` and confirmed by counting**, not
/// queried: `cuOccupancyMaxActiveBlocksPerMultiprocessor` returns 1 for any
/// kernel containing a `tcgen05.alloc` (#77) whatever the true figure is, so it
/// would agree here by accident.
///
/// Both terms give 1. [`ACCUM_COLUMNS`] is 512, so `512 / 512 = 1`; and
/// [`SHARED_BYTES`] is 131 176 against the 233 472 B an SM divides, so
/// `233472 / 131176 = 1` as well.
///
/// **Counted, on the instrument that counts rather than asks, and since #118 at
/// this kernel's own envelopes.** `device-tests`' `tmem residency census`
/// carries a `cg2 512` rung — a `cta_group::2` launch holding all 512 columns
/// across the counted interval — and at a 32 B shared plan it reads **1
/// holding, 2 resident**, against a driver prediction of 1.0 that is worth
/// nothing here for #77's reason. The gap between resident and holding is the
/// extra CTA `src/tmem.rs` describes: admitted to the SM and parked inside a
/// blocking `tcgen05.alloc` — 100.9 µs of it, at that rung.
///
/// The `ws envelope (cg2 512)` and `ws staged envelope` rungs are the same
/// launch at 131 176 B and 147 584 B, and they close that gap: **1 resident, 1
/// holding at both**, with the worst allocator wait down to 0.9 µs. The second
/// CTA is never admitted at all once the real plan is declared, which is the
/// prediction this paragraph used to make and now cites.
const CTAS_PER_SM: u32 = 1;
/// Clusters the persistent grid launches at most. A tuning constant and not a
/// correctness one — [`pipeline::run`] walks every item whatever the grid is —
/// and the constant [`Scheduler::Stealing`] exists to delete.
const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / RANKS;
const _: () = assert!(MAX_CLUSTERS == 74);

/// [`pipeline::grouped`]'s width in tile-rows, carried across from
/// [`crate::gemm`]'s measured value.
///
/// **This is inherited and not measured here**, and that is a stated
/// limitation rather than a claim. The width works by shaping a *wave*'s
/// operand footprint, and this kernel's wave is 74 clusters where
/// [`crate::gemm`]'s is 148 — so the value that won there is not the value that
/// should win here, and a sweep is the obvious follow-up. Carrying 8 keeps the
/// A/B a comparison of kernels rather than of traversals, which is the more
/// important property for one table.
const GROUP: u32 = 8;

/// One output tile of `C`, as the persistent grid's work item — the same
/// fields [`crate::gemm`]'s job carries, plus the second accumulator stage.
#[derive(Clone, Copy)]
struct Tile<const STAGES: usize> {
    a_ring: ARing<STAGES>,
    b_ring: BRing<STAGES>,
    /// Filled by the TMA, drained by the MMA. In the leader's copy the whole
    /// pair's four tiles complete on one barrier.
    load: SemaphoreRing<STAGES>,
    /// Released by the MMA's own commit, in both CTAs.
    free: SemaphoreRing<STAGES>,
    /// The item's accumulator complete, multicast by the MMA. Waited by
    /// **every** thread at the end of the item, which is what lets the next
    /// item's epilogue read it with no barrier of its own.
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    /// Stage 0 of the accumulator; stage 1 is [`Self::accumulator`]'s
    /// `columns_right`.
    accumulator: Accumulator,
    c: GlobalRows<Bf16>,
    tiles_m: u32,
    tiles_n: u32,
    group: u32,
    k_blocks: u32,
    rank: u32,
    warp_id: u32,
    lane: u32,
}

impl<const STAGES: usize> Tile<STAGES> {
    /// The accumulator stage `index` names — the ping-pong, and the whole of
    /// what [`ACCUM_STAGES`] buys.
    #[inline(always)]
    fn accumulator(&self, index: u32) -> Accumulator {
        if index % ACCUM_STAGES == 0 {
            self.accumulator
        } else {
            self.accumulator.columns_right(BLOCK_N as u32)
        }
    }

    /// Issue this rank's half of every K block, charging the leader's stage
    /// barrier for the whole pair.
    ///
    /// Unlike [`crate::gemm`]'s producer this issues the *entire* walk in one
    /// call and never yields to an epilogue, because it is not the warp that
    /// runs one: that is the point of a dedicated TMA warp, and it is the
    /// difference between this and `Lcsf`, which had to split its walk at
    /// `FILL` so warp 0 could go and store.
    ///
    /// # Safety
    ///
    /// One thread of the TMA warp, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
        unsafe {
            let a_row = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_row = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = 0u32;
            while k < self.k_blocks {
                self.free.wait_recycled(k);
                let stage = self.load.sem(k).at_rank(LEADER);
                let column = (BLOCK_K as u32 * k) as i32;
                let a_bytes = self
                    .a_ring
                    .tile(k)
                    .tma_load_2d_arriving_at(self.a_map, column, a_row, stage);
                let b_bytes = self
                    .b_ring
                    .tile(k)
                    .tma_load_2d_arriving_at(self.b_map, column, b_row, stage);
                if self.rank == LEADER {
                    self.load
                        .sem(k)
                        .expect_tx((a_bytes + b_bytes).across_ranks(RANKS));
                }
                k += 1;
            }
        }
    }

    /// Chain the whole K walk into `stage` of the accumulator and publish it.
    ///
    /// # Safety
    ///
    /// One thread of the MMA warp of the leader rank, with `stage` holding
    /// nothing any epilogue is still reading — which the item boundary two
    /// items back is what guarantees.
    #[inline(always)]
    unsafe fn multiply(&self, stage: u32) {
        unsafe {
            let accumulator = self.accumulator(stage);
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                mma_walk_cg2::<Bf16, CHUNKS, _, _>(
                    accumulator,
                    self.a_ring.tile(k).k_walk(),
                    self.b_ring.tile(k).k_walk(),
                    k > 0,
                );
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// This warp's band of `stage`, stored to the tile `item` names —
    /// [`crate::gemm::Tile::drain`] unchanged, on an epilogue warp.
    ///
    /// # Safety
    ///
    /// Every lane of a warp below [`EPILOGUE_WARPS`], with `stage` holding
    /// `item`'s completed accumulator and nothing in flight that will
    /// overwrite it.
    #[inline(always)]
    unsafe fn drain(&self, item: u32, stage: u32) {
        unsafe {
            let accumulator = self.accumulator(stage);
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
            let row_base =
                2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * self.warp_id;
            let column_base = BLOCK_N as u32 * tile_n;
            let mut column = 0u32;
            while column < BLOCK_N as u32 {
                let band: Band = accumulator.tile(32 * self.warp_id, column);
                store_rows(self.c, row_base, column_base + column, self.lane, band);
                column += DRAIN_N as u32;
            }
        }
    }

    /// [`Self::drain`] **staged through shared memory** — `stmatrix` into this
    /// warp's own [`StageTile`] and 16-byte stores out of it.
    ///
    /// [`crate::gemm::Tile::drain_staged`] is where the instruction arithmetic
    /// and the fence argument are written down; this is that epilogue on an
    /// epilogue warp, and the reason it is worth a second kernel is that the
    /// two kernels expose it differently. In [`crate::gemm`] the epilogue is on
    /// the critical path — #114 measured `whole − no drain` at 1.01× the same
    /// epilogue's serial cost — and here it is on warps of its own, one item
    /// behind, with the producer never stopping. **So this is the control for
    /// what the staged shape is actually buying**: if the win in `gemm_cg2` is
    /// recoalescing global writes it should survive being moved off the
    /// critical path, and if it is only that the epilogue was in the way, it
    /// should not.
    ///
    /// # `WIDE` and `X4` are #117's two instruction widths, one per half
    ///
    /// [`kittens::epilogue::Drain::staged`] is where both instructions are
    /// derived; this is the same two levers on a kernel that exposes the
    /// epilogue differently, which is the whole reason they are worth measuring
    /// twice.
    ///
    /// **`WIDE` is the LDTM half**, and 16 issues and 16 exposed tensor-memory
    /// latencies a `[32, 64]` band against 2 and 2. **The win #117 measured was
    /// the wait and not the issue**, and this kernel is where that claim gets
    /// its second reading: here the epilogue is already one item behind and
    /// already on warps of its own, so a latency the producer is covering ought
    /// to cost less to begin with.
    ///
    /// **`X4` is the `stmatrix` half**, and half the issues at the same 32
    /// addresses. In [`crate::gemm`] it was a clean null alone (−0.6% to −1.1%)
    /// and bought 14 registers back in composition.
    ///
    /// Neither touches the global half — `store_shared_rows` issues the same
    /// 32 × 16 B stores on the same four contiguous 128 B runs whatever these
    /// are set to — and neither moves a byte of the plan, so all four
    /// combinations declare [`STAGED_SHARED_BYTES`] and share one `no drain`
    /// control.
    ///
    /// # Safety
    ///
    /// As [`Self::drain`], and the launch must declare
    /// [`STAGED_SHARED_BYTES`] — which is what makes [`Self::stage_tile`]'s
    /// address one this launch owns.
    #[inline(always)]
    unsafe fn drain_staged<const WIDE: bool, const X4: bool>(&self, item: u32, stage: u32) {
        unsafe {
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, self.group);
            epilogue::Drain::<WIDE, X4>::staged(
                self.accumulator(stage),
                self.warp_id,
                self.stage_tile(),
                self.c,
                2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank,
                BLOCK_N as u32 * tile_n,
                self.lane,
            );
        }
    }

    /// This epilogue warp's 4096 B of the staging run.
    ///
    /// Carved here rather than handed down from `attach` so that a launch at
    /// the *shipped* envelope never forms the address at all: the only caller
    /// is [`Self::drain_staged`], which is behind a `const DRAIN` code that a
    /// non-staged entry point does not name.
    ///
    /// # Safety
    ///
    /// The launch must declare [`STAGED_SHARED_BYTES`], and `warp_id` must be
    /// below [`EPILOGUE_WARPS`].
    ///
    /// The run's offset is `Self`'s own `STAGES` since #125, where it used to
    /// be a module-level `STAGE_OFFSET` derived from the shipped depth. Every
    /// instantiation that reaches here is at that depth — `Ws::<6, ..>` is
    /// `drain::REGISTER` and names no staged arm — so the two agree everywhere
    /// they are both defined, and only this one is defined everywhere.
    #[inline(always)]
    unsafe fn stage_tile(&self) -> StageTile {
        unsafe {
            staged(shared::<STAGES>(SharedPlan::attach()).plan)
                .0
                .tile(self.warp_id)
        }
    }
}

/// Which epilogue a launch's drain warps run — [`Ws`]'s const-generic
/// selector, and the only thing an entry point in this file varies once the
/// pipeline depth is fixed.
///
/// One code rather than a `bool` per lever. [`crate::gemm`] spells the same
/// choice as three of them (`DRAIN`, `WIDE`, `X4`) because it has three; a
/// fourth arrives here — the register epilogue this kernel shipped through
/// #119 — and a launch site reading `Ws::<STAGES, true, false, false, true>`
/// says nothing about which rung it is. These names are what the tables print.
///
/// `DRAIN` is a monomorphization constant, so exactly one arm of
/// [`Ws::epilogue`] survives into any entry point. That is what keeps
/// [`Tile::stage_tile`]'s address — which lives past [`SHARED_BYTES`] — out of
/// a launch that declares only [`SHARED_BYTES`], and it is the same guarantee
/// the `const STAGED` bool gave before there were four rungs to name.
mod drain {
    /// [`super::Tile::drain`] — a `RegTile<32, 128>` band straight to global,
    /// the epilogue this kernel shipped through #119 and the one #112 measured.
    pub const REGISTER: u8 = 0;
    /// [`super::Tile::drain_staged`] at `.x1` LDTM and `stmatrix.x2` — #116's
    /// rung, and the control every width below is quoted against.
    pub const STAGED: u8 = 1;
    /// The same with the LDTM half at `.x8` — #117's first width.
    pub const STAGED_X8: u8 = 2;
    /// The same with the `stmatrix` half at `.x4` — #117's second.
    pub const STAGED_X4: u8 = 3;
    /// Both widths at once — the composition rung.
    pub const STAGED_X8_X4: u8 = 4;
    /// **No epilogue at all.** The accumulator is filled and never read, so
    /// this computes a wrong `C` on purpose and is never checked; it is the
    /// far end of #114's `whole − no drain` subtraction and nothing else.
    pub const REMOVED: u8 = 5;
}

/// The warp-specialized job: one output tile per item, with the epilogue one
/// item behind the MMA and on warps of its own.
///
/// The deferral is [`crate::gemm`]'s `Lcsf` — the accumulator lives outside the
/// item loop, so an undrained one survives the item boundary — and the warp
/// split is what makes the deferral pay. Under `Lcsf` the drain sits *between*
/// the producer's fill prefix and the rest of its walk, because warp 0 is both
/// the producer and a quarter of the epilogue; here the producer never stops.
///
/// [`drain`] picks which epilogue the drain warps run, and it is a const
/// parameter of the *same* job rather than a second job per rung, for
/// [`crate::gemm`]'s reason: a second spelling of this ordering argument is a
/// place for the arms to drift, and the argument is the hard part. Every arm
/// but [`drain::REMOVED`] computes the GEMM and is on the correctness gate.
#[derive(Clone, Copy)]
struct Ws<const STAGES: usize, const DRAIN: u8> {
    tile: Tile<STAGES>,
    /// The item whose accumulator is still in tensor memory, or [`Self::NONE`].
    pending: u32,
    /// Items this **cluster** has run, which is what selects the accumulator
    /// stage — and it cannot be the item index.
    ///
    /// [`pipeline::run`] strides items by [`MAX_CLUSTERS`], which is 74 and
    /// even, so `item % 2` is *constant* for a given cluster and would put
    /// every item in the same stage; [`pipeline::run_stealing`] hands out
    /// whatever the hardware cancelled, which has no parity at all. The
    /// ping-pong is a property of the sequence a cluster walks, so it counts
    /// the sequence.
    sequence: u32,
}

impl<const STAGES: usize, const DRAIN: u8> Ws<STAGES, DRAIN> {
    /// No accumulator owed. `u32::MAX` is not a reachable item: the tile grid
    /// is `tiles_m * tiles_n` and both are `u32`.
    const NONE: u32 = u32::MAX;

    /// The epilogue this rung runs over `item`'s accumulator in `stage` — the
    /// single place [`drain`]'s code is turned back into a call, so the two
    /// sites that drain (the item loop and [`Self::finish`]) cannot disagree
    /// about which rung they are.
    ///
    /// # Safety
    ///
    /// As [`Tile::drain`] and [`Tile::drain_staged`]: every lane of a warp
    /// below [`EPILOGUE_WARPS`], with `stage` holding `item`'s completed
    /// accumulator, and a launch declaring the envelope `DRAIN` implies.
    #[inline(always)]
    unsafe fn epilogue(&self, item: u32, stage: u32) {
        const { assert!(DRAIN <= drain::REMOVED) };
        unsafe {
            match DRAIN {
                drain::REGISTER => self.tile.drain(item, stage),
                drain::STAGED => self.tile.drain_staged::<false, false>(item, stage),
                drain::STAGED_X8 => self.tile.drain_staged::<true, false>(item, stage),
                drain::STAGED_X4 => self.tile.drain_staged::<false, true>(item, stage),
                drain::STAGED_X8_X4 => self.tile.drain_staged::<true, true>(item, stage),
                _ => {}
            }
        }
    }

    #[inline(always)]
    fn new(tile: Tile<STAGES>) -> Self {
        Self {
            tile,
            pending: Self::NONE,
            sequence: 0,
        }
    }

    /// The stage [`Self::pending`] sits in — the one the current item is *not*
    /// using.
    ///
    /// `(sequence + 1) % 2` reads as the next stage and is also the previous
    /// one, which is what makes the same expression right in [`Self::work`]
    /// (where `sequence` is the current item's) and in [`Self::finish`] (where
    /// it has already been advanced past the last).
    #[inline(always)]
    fn pending_stage(&self) -> u32 {
        (self.sequence + 1) % ACCUM_STAGES
    }
}

impl<const STAGES: usize, const DRAIN: u8> Job for Ws<STAGES, DRAIN> {
    /// The pair shares one barrier set — the peer aims its TMA at the leader's
    /// stage barrier and the leader's MMA arrives in the peer's `free` and
    /// `done` — so the item boundary that re-arms them is the cluster's.
    const RANKS: u32 = crate::gemm_ws::RANKS;

    /// Every barrier takes exactly one arrival: the leader's stage barrier from
    /// the TMA transaction count, `free` and `done` from the MMA commit.
    ///
    /// # Safety
    ///
    /// As [`Semaphore::init`]; [`pipeline::run`] owns the thread and the
    /// ordering.
    #[inline(always)]
    unsafe fn init(&self, _item: u32) {
        unsafe {
            self.tile.load.init_all(1);
            self.tile.free.init_all(1);
            self.tile.done.init(1);
        }
    }

    /// # Safety
    ///
    /// As [`Semaphore::inval`].
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe {
            self.tile.load.inval_all();
            self.tile.free.inval_all();
            self.tile.done.inval();
        }
    }

    /// One item, with the three roles side by side.
    ///
    /// The ordering argument, in full, because every way it can be wrong is
    /// silent:
    ///
    /// - The epilogue warps read stage `1 - s` (item `i - 1`); the MMA warp
    ///   writes stage `s` (item `i`). **Disjoint columns**, so nothing inside
    ///   the item orders them and nothing has to — that is what the second
    ///   accumulator stage is for.
    /// - Item `i + 1`'s MMA writes stage `1 - s`, which item `i - 1`'s epilogue
    ///   was reading. Those are separated by [`pipeline::run`]'s item boundary
    ///   — a `cluster_sync`, so it covers the *peer's* epilogue too, which a
    ///   `bar.sync` would not — with the harness'
    ///   `tcgen05_fence_before_thread_sync` in front of it retiring the LDTM.
    /// - The epilogue of item `i` needs item `i`'s MMA complete. It gets that
    ///   from the `done.wait` **every thread** takes at the end of this
    ///   function, one item earlier, plus that same boundary. So the drain at
    ///   the top of the next item takes no barrier of its own, and `done` can
    ///   be one semaphore re-armed per item rather than a ring.
    ///
    /// # Safety
    ///
    /// Every thread of both CTAs must enter with the same `item`, which is what
    /// [`pipeline::run`]'s cluster-strided map gives, and the maps must cover
    /// the tile it names.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let tile = self.tile;
            let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, tile.group);
            let stage = self.sequence % ACCUM_STAGES;

            if tile.warp_id == TMA_WARP && tile.lane == 0 {
                tile.produce(tile_m, tile_n);
            }
            if tile.rank == LEADER && tile.warp_id == MMA_WARP && tile.lane == 0 {
                tile.multiply(stage);
            }
            if tile.warp_id < EPILOGUE_WARPS && self.pending != Self::NONE {
                self.epilogue(self.pending, self.pending_stage());
            }

            // Every thread, including the epilogue warps that have just
            // finished the previous item. This is the only rendezvous inside
            // the item, and it is what makes the *next* item's drain free.
            tile.done.wait(0);
            self.pending = item;
            self.sequence += 1;
        }
    }

    /// Store the last item's accumulator, which no later item is coming to
    /// overlap — the drain the epilogue warps still owe once the item loop has
    /// run out of items to hide it behind.
    ///
    /// A cluster that ran no items owes nothing, which is the static-schedule
    /// case where [`MAX_CLUSTERS`] exceeds the tile count.
    ///
    /// # Safety
    ///
    /// As [`Self::epilogue`], and [`pipeline::run`]'s: every thread of the CTA,
    /// after the item loop and before `release`.
    #[inline(always)]
    unsafe fn finish(&mut self) {
        unsafe {
            if self.pending != Self::NONE && self.tile.warp_id < EPILOGUE_WARPS {
                self.epilogue(self.pending, self.pending_stage());
            }
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The item and the work queue, laid over the one shared plan every entry
    /// point launches with. Everything here spans items — the rings, the
    /// barriers, the operand maps, and the pair's TMEM allocation, whose
    /// `alloc_cluster` is a whole-cluster collective with a `cluster_sync` in
    /// it and must not sit inside anybody's item loop.
    ///
    /// # Safety
    ///
    /// The launch geometry's, and the operands': both maps must describe live
    /// buffers covering `k_blocks * BLOCK_K` along K and the full extent the
    /// item loop walks, and `c` must hold `ldc` columns for every row of it.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attach<const STAGES: usize>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        c: &mut DisjointSlice<u16>,
    ) -> (Tile<STAGES>, ClcQueue) {
        // The join between the two spellings of this depth's plan — the walk
        // and the value-parameterized arithmetic the host needs. Fired at
        // codegen, which is the only place the ring byte counts are known, and
        // the reason `cargo check` cannot stand in for `scripts/modal-run
        // build`. The queue's `ALIGNMENT` used to be a second assert here and
        // is `SharedPlan::clc_queue`'s since #125.
        const {
            assert!(shared::<STAGES>(SharedPlan::sizing()).plan.bytes() == shared_plan(STAGES));
        };
        unsafe {
            let shared = shared::<STAGES>(SharedPlan::attach());

            let tile = Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                done: shared.done,
                a_map,
                b_map,
                // Both stages in one allocation. The allocator's unit is the
                // CTA, so a second stage is a column offset and never a second
                // `tcgen05.alloc`.
                accumulator: Accumulator::from_raw(alloc_cluster::<ACCUM_COLUMNS>(
                    shared.tmem_slot,
                )),
                c: GlobalRows::<Bf16>::from_slice(c, ldc as usize),
                tiles_m,
                tiles_n,
                group,
                k_blocks,
                rank: cluster::block_rank(),
                warp_id: warp_id(),
                lane: lane(),
            };
            (tile, shared.queue)
        }
    }

    /// Give the pair's accumulator back.
    ///
    /// The `cluster_sync` is not decoration and the reference carries the same
    /// one with a comment on it: without a whole-cluster rendezvous before
    /// exit, a fast CTA can leave while its partner is still driving a
    /// cross-CTA operation, which faults as CUDA_EXCEPTION_17 rather than
    /// producing a wrong number. Here it also covers the cluster that got no
    /// items at all and still owes a deallocation in step with its peer.
    ///
    /// # Safety
    ///
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired — which is `finish` having run.
    #[inline(always)]
    unsafe fn release<const STAGES: usize>(tile: &Tile<STAGES>) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster::<ACCUM_COLUMNS>(tile.accumulator.raw());
        }
    }

    /// `C[m, n] = Σₖ A[m, k] · B[n, k]`, one `[256, 256]` output tile per work
    /// item, six warps to a CTA and one CTA to an SM.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus: the grid must be a whole number of clusters and
    /// `tiles_m * tiles_n` the item count they are to cover.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_176,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::REGISTER }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// The same kernel on the hardware's schedule — one cluster per output
    /// tile, and a cluster that finishes cancels one the scheduler has not
    /// launched yet.
    ///
    /// It ought to matter more here than for [`crate::gemm`]: a wave is 74
    /// clusters where that kernel's is 148, so the ragged last wave is a
    /// coarser quantization of the same tile grid and the static stride has
    /// more to lose. **Measured, it is +0.6% / +1.0% / +1.2% at 4096³ / 8192³ /
    /// 16384³** — the same order #97 found for [`crate::gemm`], and it changes
    /// no sign in the table this file is for.
    ///
    /// # Safety
    ///
    /// `attach`'s, plus [`pipeline::run_stealing`]'s: the grid is exactly
    /// `RANKS` × the tile count, one-dimensional, on sm_100a.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_176,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_clc(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, queue) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::REGISTER }>::new(tile);
            pipeline::run_stealing(&mut job, queue);
            release(&job.tile);
        }
    }

    /// The same kernel **six** pipeline stages deep — 196 744 B.
    ///
    /// It is here because the depth is *free* and that is a consequence of the
    /// design worth measuring rather than asserting. Tensor memory has already
    /// fixed residency at one CTA an SM, so shared memory has 102 296 B doing
    /// nothing at four stages; the reference spends its own surplus on a 64 KiB
    /// `stmatrix` staging buffer, and with the register epilogue kept there is
    /// nothing to spend it on but pipeline. Under [`crate::gemm`]'s two CTAs an
    /// SM the same two stages would be an occupancy step and #98 priced that at
    /// 25–44%; here they cost nothing that any resource can see.
    ///
    /// **Measured: +1.4% at 4096³, +4.4% at 8192³, +0.7% at 16384³.** Free
    /// bytes bought a real if small gain, which says the four-stage pipeline
    /// was mildly starved — and, more usefully, that starvation is *not* where
    /// this design point's 7% loss at 16384³ is hiding.
    ///
    /// # Safety
    ///
    /// As [`gemm_ws`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_744,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_s6(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<6>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<6, { drain::REGISTER }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_ws`] with its epilogue **staged through shared memory** — #15,
    /// and the control for what the staged shape buys.
    ///
    /// Identical to [`gemm_ws`] in grid, tensor memory, warp split, operand
    /// maps, item map, schedule and pipeline depth, and in every byte of the
    /// plan `attach` lays out; it declares 16 408 more of them for the four
    /// epilogue warps' staging tiles. **Residency does not move and cannot**:
    /// this kernel is one CTA an SM because [`ACCUM_COLUMNS`] is the SM's
    /// whole tensor memory, and 147 584 B is still well inside the 233 472 an
    /// SM has.
    ///
    /// What makes it worth a kernel rather than a footnote is that the
    /// epilogue is *already* off the critical path here — deferred one item
    /// and on warps of its own — where in [`crate::gemm`] it is the critical
    /// path. See [`Tile::drain_staged`].
    ///
    /// # Safety
    ///
    /// [`gemm_ws`]'s, at [`STAGED_SHARED_BYTES`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 147_584,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_staged(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::STAGED }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_ws_staged`] with the LDTM half at `.x8` — #117's first width, on
    /// the kernel it was never tried on.
    ///
    /// `tcgen05.ld.16x256b.x8` returns 32 f32 a thread where the `.x1` this
    /// crate has always issued returns 4, so a `[32, 64]` staged band is 2
    /// loads and 2 waits instead of 16 and 16. Same bytes out of tensor
    /// memory, same `stmatrix`, same stores, same 147 584 B — so
    /// [`gemm_ws_staged_no_drain`] is its control exactly as much as it is
    /// [`gemm_ws_staged`]'s.
    ///
    /// # Safety
    ///
    /// [`gemm_ws_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 147_584,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_staged_x8(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::STAGED_X8 }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// [`gemm_ws_staged`] with the `stmatrix` half at `.x4` — #117's second
    /// width, alone.
    ///
    /// A [`kittens::reg::Fragment`] is four `8x8` b16 matrices and `.x2` names
    /// two, so the shipped staged path issues two `stmatrix` per `[16, 16]`
    /// block where this issues one. The addresses are the same 32; only the
    /// lane grouping that supplies them changes.
    ///
    /// # Safety
    ///
    /// [`gemm_ws_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 147_584,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_staged_x4(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::STAGED_X4 }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// Both of #117's widths at once — the composition rung, and the only one
    /// that can say whether they add on this design point.
    ///
    /// # Safety
    ///
    /// [`gemm_ws_staged`]'s.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 147_584,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_staged_x8x4(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::STAGED_X8_X4 }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_ws_staged`] with the epilogue
    /// removed, and the one control all four staged widths subtract from.
    ///
    /// [`gemm_ws_no_drain`] would not do. It declares 131 176 B where this
    /// declares 147 584, and the point of the subtraction is that everything
    /// but the drain is held. Since #117's widths change what the epilogue
    /// issues and not what it occupies, one control at this envelope serves
    /// `staged`, `staged8`, `staged4` and `staged84` alike — the clean
    /// ablation #116 could not have, because it had a 16 408-byte envelope
    /// change to price first.
    ///
    /// # Safety
    ///
    /// [`gemm_ws_staged`]'s, less the epilogue: it writes no `C` at all.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 147_584,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_staged_no_drain(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::REMOVED }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }

    /// **A deliberately wrong GEMM** — [`gemm_ws`] with the epilogue removed,
    /// at the *register drain's* 131 176 B envelope.
    ///
    /// This is the register epilogue's own control, and it exists because #112
    /// never established why this kernel lost. It attributed the loss to the
    /// peer CTA hiding [`crate::gemm`]'s epilogue; #114 refuted that by
    /// measuring `gemm_cg2`'s epilogue as **fully exposed** (`whole − no
    /// drain` at 1.01× its serial cost). What nobody has measured is how
    /// exposed *this* kernel's epilogue is, with the drain deferred one item
    /// and on warps of its own — and that is one subtraction, at the envelope
    /// [`Entry::S4`] actually declares.
    ///
    /// # Safety
    ///
    /// [`gemm_ws`]'s, less the epilogue: it writes no `C` at all.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 131_176,
        dynamic_shared_alignment = 128
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ws_no_drain(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        group: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            let (tile, _) =
                attach::<STAGES>(a_map, b_map, tiles_m, tiles_n, group, k_blocks, ldc, &mut c);
            let mut job = Ws::<STAGES, { drain::REMOVED }>::new(tile);
            pipeline::run(&mut job, tiles_m * tiles_n);
            release(&job.tile);
        }
    }
}

/// Which entry point a plan launches.
///
/// Two axes and they are deliberately not crossed: the pipeline depth
/// ([`Entry::S4`], [`Entry::S6`]) and the epilogue ([`drain`]'s rungs). Every
/// epilogue rung is at four stages, so an epilogue A/B moves one variable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Entry {
    /// `gemm_ws` / `gemm_ws_clc` — four stages, the reference's depth, and the
    /// register epilogue this kernel shipped through #119.
    ///
    /// Still the control, and still the only entry with a
    /// [`Scheduler::Stealing`] twin: it is what table 1 of [`compare`] holds
    /// on **both** sides, since that table's one variable is where the overlap
    /// comes from and an epilogue moving with it would be two.
    S4,
    /// `gemm_ws_s6` — six, static only.
    S6,
    /// `gemm_ws_staged` — four stages with the epilogue staged through shared
    /// memory (#15/#116), static only. Same depth as [`Entry::S4`], so the
    /// pair is an epilogue comparison and nothing else.
    Staged,
    /// `gemm_ws_staged_x8` — [`Entry::Staged`] with the LDTM half at `.x8`,
    /// and **[`SHIPPED_ENTRY`] since #119**.
    StagedX8,
    /// `gemm_ws_staged_x4` — [`Entry::Staged`] with `stmatrix` at `.x4`.
    StagedX4,
    /// `gemm_ws_staged_x8x4` — both of #117's widths, the composition rung.
    ///
    /// It is what [`crate::gemm`] ships and what this kernel does **not**,
    /// which is the one place the two files disagree on purpose. There `.x4`
    /// hands back 14 of the 52 registers `.x8` costs and that recovery is the
    /// composition gain; here it hands back 2, `staged8` and `staged84` land
    /// within 1.1% and trade places between sessions, and the rung with less
    /// to go wrong wins the tie.
    StagedX8X4,
    /// `gemm_ws_staged_no_drain` — **not a GEMM**: the staged envelope with no
    /// epilogue at all, and the far end of the four staged rungs' subtraction.
    StagedNoDrain,
    /// `gemm_ws_no_drain` — **not a GEMM**: the register-drain envelope with no
    /// epilogue, which is [`Entry::S4`]'s own control.
    NoDrain,
}

/// The entry point this file ships — `staged8`, #116's staged drain with
/// #117's `.x8` LDTM and **without** `.x4`.
///
/// **It was [`Entry::S4`] through #119.** #118 measured the best staged rung at
/// **+19.4% / +9.5% / +5.2%** over the register epilogue at 4096³ / 8192³ /
/// 16384³; measured default against default in one container with cuBLASLt
/// re-measured in it, that reproduces at **+19.6% / +8.7% / +6.3%** and
/// **0.599 → 0.717, 0.790 → 0.858 and 0.812 → 0.863 of cuBLASLt**. Every rung
/// passed the same element-by-element `==` on bf16 words at both check sizes
/// and all three traversal widths.
///
/// **`.x4` is left off deliberately**, which is where this file and
/// [`crate::gemm::SHIPPED_EPILOGUE`] part company. On that kernel `.x4`
/// recovers 14 registers (94 → 80) and the recovery *is* the composition gain;
/// here it recovers 2 (94 → 92) and buys no time with them, so `staged8` and
/// `staged84` are within 1.1% and trade places between sessions. Tidying the
/// two files onto one choice would be discarding the measurement that
/// separates them.
///
/// Residency does not move with it — [`ACCUM_COLUMNS`] fixed [`CTAS_PER_SM`]
/// at 1 before shared memory was consulted, and `device-tests`' census counts
/// 1 resident and 1 holding at both 131 176 B and 147 584 B.
pub const SHIPPED_ENTRY: Entry = Entry::StagedX8;

impl Entry {
    /// Pipeline depth, which is the only thing besides the epilogue an entry
    /// varies.
    pub fn stages(self) -> usize {
        match self {
            Entry::S6 => 6,
            _ => STAGES,
        }
    }

    /// Dynamic shared memory its launch declares — the staging tiles are the
    /// one thing that is not a function of the depth, and all four widths plus
    /// their control declare the same number.
    pub fn shared(self) -> usize {
        match self {
            Entry::Staged
            | Entry::StagedX8
            | Entry::StagedX4
            | Entry::StagedX8X4
            | Entry::StagedNoDrain => staged_plan(self.stages()),
            _ => shared_plan(self.stages()),
        }
    }

    /// Whether a launch on this entry computes the GEMM. The two `no drain`
    /// rungs do not, and each says so wherever it appears.
    pub fn exact(self) -> bool {
        !matches!(self, Entry::StagedNoDrain | Entry::NoDrain)
    }

    /// What the tables call it.
    pub fn name(self) -> String {
        match self {
            Entry::S4 | Entry::S6 => format!("ws s{}", self.stages()),
            Entry::Staged => "ws staged".to_string(),
            Entry::StagedX8 => "ws staged8".to_string(),
            Entry::StagedX4 => "ws staged4".to_string(),
            Entry::StagedX8X4 => "ws staged84".to_string(),
            Entry::StagedNoDrain => "ws staged no drain".to_string(),
            Entry::NoDrain => "ws no drain".to_string(),
        }
    }

    /// [`Entry::name`] without the kernel prefix, for tables whose every
    /// column is this kernel and where `ws ` in each of eight headings is
    /// three characters of nothing said eight times.
    pub fn short(self) -> String {
        self.name().trim_start_matches("ws ").to_string()
    }
}

/// How a launch takes its work.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub scheduler: Scheduler,
    pub group: u32,
    pub entry: Entry,
}

impl Plan {
    /// The register-drain kernel at the reference's depth, walked at the
    /// measured [`GROUP`] — **not the kernel as it ships**, which is
    /// [`SHIPPED_ENTRY`].
    ///
    /// It is what table 1 of [`compare`] holds on this side of its A/B, and
    /// the only entry that answers to both schedulers, so it stays the plan a
    /// caller gets by naming nothing.
    pub fn new(scheduler: Scheduler) -> Self {
        Plan {
            scheduler,
            group: GROUP,
            entry: Entry::S4,
        }
    }
}

/// Tiles of `C` a `[m, n]` output has along each axis.
fn tile_grid(m: usize, n: usize) -> (u32, u32) {
    ((m / (2 * BLOCK_M)) as u32, (n / BLOCK_N) as u32)
}

/// Tiles of `C` a `[m, n]` output has, which is the item count.
fn tiles(m: usize, n: usize) -> u32 {
    let (rows, columns) = tile_grid(m, n);
    rows * columns
}

/// Blocks a launch asks for. The static grid is capped at [`MAX_CLUSTERS`]; the
/// stealing grid is one cluster per tile, since CLC caps it by leaving clusters
/// unlaunched — one branch reads a measured constant and the other reads the
/// problem.
fn grid_for(scheduler: Scheduler, m: usize, n: usize) -> u32 {
    let clusters = match scheduler {
        Scheduler::Static => tiles(m, n).min(MAX_CLUSTERS),
        Scheduler::Stealing => tiles(m, n),
    };
    RANKS * clusters
}

/// The `then` a run that is only being checked passes.
fn nothing_after(
    _: &cuda_core::CudaStream,
    _: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Launch `[m, k] · [n, k]ᵀ`, compare every element of `C` against the CPU
/// reference, and only then hand the launch to `then`.
///
/// The order is the design and it is [`crate::gemm::run`]'s: `then` — the
/// clock — is unreachable from a launch whose output was wrong, so no
/// throughput figure can be printed for a kernel that did not compute. The
/// reference itself is shared with [`crate::gemm`] rather than copied, through
/// `a_value` / `b_value` / `check_c`, which is what makes "the two kernels are
/// checked against the same thing" a property of the code.
fn run<T>(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    m: usize,
    n: usize,
    k: usize,
    plan: Plan,
    then: impl FnOnce(
        &cuda_core::CudaStream,
        &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>>,
) -> Result<(String, T), Box<dyn Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    use kittens::global::GlobalLayout;

    // The shape contract, and the kernel bounds-checks none of it. `k % 64`
    // rather than the reference's `k % 256`: barriers are re-armed per item, so
    // every item starts at ring stage 0 and no tile boundary has to land on a
    // pipeline cycle.
    if m % (2 * BLOCK_M) != 0 || n % BLOCK_N != 0 || k % BLOCK_K != 0 {
        return Err(format!(
            "{m}x{n}x{k} does not divide the {}x{BLOCK_N}x{BLOCK_K} tiling",
            2 * BLOCK_M
        )
        .into());
    }
    if plan.group == 0 {
        return Err("a traversal group width of 0 has no tiles in it".into());
    }

    let stream = context.default_stream();
    // SAFETY: the artifact is this crate's own and the contracts declare the
    // ABI compiled.
    let module = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::from_host(&stream, &stage(m, k, a_value))?;
    let b = DeviceBuffer::from_host(&stream, &stage(n, k, b_value))?;
    // SAFETY: both buffers outlive every launch consuming their maps below.
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<Bf16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<Bf16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BTile>(&stream)?;

    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let blocks = grid_for(plan.scheduler, m, n);
    let (tiles_m, tiles_n) = tile_grid(m, n);
    let k_blocks = (k / BLOCK_K) as u32;
    let config = LaunchConfig1D::new(blocks, THREADS, plan.entry.shared() as u32);

    let (stream_ref, module_ref) = (&stream, &module);
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    // SAFETY (every arm): both maps describe live buffers covering the walk the
    // grid takes, and `c` holds `n` columns for every row of it. The stealing
    // entry additionally takes a grid of exactly one cluster per tile, which is
    // what `grid_for` gives it.
    macro_rules! launcher {
        ($prepare:ident, $launch:ident) => {{
            let prepared = module_ref.$prepare(config)?;
            let launch = move |c: &mut DeviceBuffer<u16>| -> Result<(), Box<dyn Error>> {
                unsafe {
                    module_ref.$launch(
                        stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, plan.group,
                        k_blocks, n as u32, c,
                    )?
                };
                Ok(())
            };
            Box::new(launch) as Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>>
        }};
    }
    let launch_once = match (plan.entry, plan.scheduler) {
        (Entry::S4, Scheduler::Static) => launcher!(prepare_gemm_ws, gemm_ws),
        (Entry::S4, Scheduler::Stealing) => launcher!(prepare_gemm_ws_clc, gemm_ws_clc),
        (Entry::S6, Scheduler::Static) => launcher!(prepare_gemm_ws_s6, gemm_ws_s6),
        (Entry::Staged, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_staged, gemm_ws_staged)
        }
        (Entry::StagedX8, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_staged_x8, gemm_ws_staged_x8)
        }
        (Entry::StagedX4, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_staged_x4, gemm_ws_staged_x4)
        }
        (Entry::StagedX8X4, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_staged_x8x4, gemm_ws_staged_x8x4)
        }
        (Entry::StagedNoDrain, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_staged_no_drain, gemm_ws_staged_no_drain)
        }
        (Entry::NoDrain, Scheduler::Static) => {
            launcher!(prepare_gemm_ws_no_drain, gemm_ws_no_drain)
        }
        // A depth comparison under a moving scheduler would be two variables.
        (entry, scheduler) => {
            return Err(format!(
                "{entry:?} has no entry point on the {} schedule",
                scheduler.name()
            )
            .into());
        }
    };
    launch_once(&mut c)?;
    // Rule 1 of `crate::bench`, and the one exception to it is stated in the
    // label rather than hidden: the two `no drain` rungs write no `C` at all,
    // so checking them would fail by construction. Every rung that claims to
    // be a GEMM — every schedule, both depths, and all four epilogue widths —
    // goes through the element-by-element `==` before a clock can reach it.
    let label = if plan.entry.exact() {
        let worst = check_c(&c.to_host_vec(&stream)?, m, n, k)?;
        format!("{m}x{n}x{k} exact, worst |rel| {worst:.2e} against the fp32 reference")
    } else {
        format!(
            "{m}x{n}x{k} UNCHECKED ({} is not a GEMM)",
            plan.entry.name()
        )
    };

    let after = then(&stream, &mut || launch_once(&mut c))?;
    Ok((label, after))
}

/// A small correctness size — four tiles, so both axes of the item map move.
const M: usize = 512;
const N: usize = 512;
const K: usize = 256;

/// The size whose only job is to give every cluster **more than one work
/// item**, which is where this kernel's own failure modes live.
///
/// [`M`]x[`N`] is four tiles over 74 clusters, so it never enters
/// [`pipeline::run`]'s loop twice and would pass against a kernel with no
/// deferred epilogue and no accumulator ping-pong at all. Everything warp
/// specialization adds needs a second item to happen: a stage the MMA
/// overwrites while an epilogue warp is still reading it, a `pending` dropped
/// at the end of the loop, a stage index taken from the item rather than from
/// the sequence. 256 tiles over 74 clusters is three or four items for every
/// cluster with a ragged tail, at a `K` that wraps the ring.
const ITEMS_M: usize = 4096;
const ITEMS_N: usize = 4096;
const ITEMS_K: usize = 256;

/// Traversal widths the correctness run walks, chosen to break
/// [`pipeline::grouped`]'s short last group rather than to be fast — the same
/// three [`crate::gemm::check`] uses and for the same reason.
const CHECK_GROUPS: [u32; 3] = [1, 3, 6];

/// The four staged rungs, and the order every table prints them in.
///
/// [`Entry::Staged`] is first because it is the control: #116 measured it and
/// every column after it is a delta against that row rather than against the
/// register epilogue.
const STAGED_ENTRIES: [Entry; 4] = [
    Entry::Staged,
    Entry::StagedX8,
    Entry::StagedX4,
    Entry::StagedX8X4,
];

/// The correctness run: two sizes, three traversals, both schedulers, both
/// depths, checked and nothing timed.
pub fn check(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<String, Box<dyn Error>> {
    let mut notes = Vec::new();
    for (m, n, k) in [(M, N, K), (ITEMS_M, ITEMS_N, ITEMS_K)] {
        let mut rounding = None;
        for scheduler in [Scheduler::Static, Scheduler::Stealing] {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler,
                    group,
                    entry: Entry::S4,
                };
                let (label, _) = run(context, m, n, k, plan, nothing_after)?;
                rounding.get_or_insert(label);
            }
            let clusters = grid_for(scheduler, m, n) / RANKS;
            notes.push(format!(
                "{m}x{n}x{k} exact on {} at groups {CHECK_GROUPS:?} ({} tiles over {clusters} \
                 clusters, one deferred accumulator per cluster in flight)",
                scheduler.name(),
                tiles(m, n)
            ));
        }
        // The deeper rung under the same gate: a ring depth is a different
        // `wait_recycled` parity and a different producer/consumer distance,
        // and both go wrong as a wrong `C` rather than as a fault.
        for group in CHECK_GROUPS {
            let plan = Plan {
                scheduler: Scheduler::Static,
                group,
                entry: Entry::S6,
            };
            run(context, m, n, k, plan, nothing_after)?;
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} ({} B)",
            Entry::S6.name(),
            Entry::S6.shared()
        ));
        // #15's staged epilogue and #117's two instruction widths on it, under
        // the same traversals and the same `==`. It is the same gate
        // `crate::gemm`'s staged rungs are on and for the same reasons — a
        // swizzled tile addressed through two derivations, four warp-private
        // tiles carved out of one run, each reused four times with only a
        // `bar.warp.sync` between a read and the next write — plus one this
        // kernel adds: `.x8` returns 32 f32 in a single instruction and the
        // order they arrive in is silicon's rather than the ISA text's, which
        // `kittens::tmem::interleave_x8` asserts is repeat-major and nothing
        // but a wrong `C` would report.
        for entry in STAGED_ENTRIES {
            for group in CHECK_GROUPS {
                let plan = Plan {
                    scheduler: Scheduler::Static,
                    group,
                    entry,
                };
                run(context, m, n, k, plan, nothing_after)?;
            }
        }
        notes.push(format!(
            "{m}x{n}x{k} exact on {} ({} B, {} ships)",
            STAGED_ENTRIES.map(|entry| entry.name()).join(", "),
            Entry::Staged.shared(),
            SHIPPED_ENTRY.name()
        ));
        if let Some(label) = rounding {
            notes.push(label);
        }
    }
    Ok(notes.join(", "))
}

/// Sizes the comparison runs at, and no smaller.
///
/// Below 4096³ cuBLASLt's own run-to-run spread reaches 77%, so a ratio there
/// says more about the library's launch path than about either kernel. These
/// are also the three sizes the reference publishes, which is the only thing
/// making its numbers and these numbers even nominally about the same question.
const SIZES: [Shape; 3] = [
    Shape {
        m: 4096,
        n: 4096,
        k: 4096,
    },
    Shape {
        m: 8192,
        n: 8192,
        k: 8192,
    },
    Shape {
        m: 16384,
        n: 16384,
        k: 16384,
    },
];

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// Waves the static grid takes, and how full the last one is.
fn wave_efficiency(m: usize, n: usize) -> (u32, f64) {
    let tiles = tiles(m, n);
    let waves = tiles.div_ceil(MAX_CLUSTERS);
    (waves, tiles as f64 / (waves * MAX_CLUSTERS) as f64)
}

/// Items the busiest cluster walks — the divisor that turns a launch-level
/// difference into a per-tile one, and [`crate::gemm`]'s
/// `items_on_critical_path` at this kernel's grid.
///
/// A persistent cluster runs `ceil(items / clusters)` items back to back, so a
/// per-item cost appears in the launch that many times and no more.
fn items_on_critical_path(m: usize, n: usize) -> f64 {
    let clusters = grid_for(Scheduler::Static, m, n) / RANKS;
    tiles(m, n).div_ceil(clusters) as f64
}

fn timed(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
    plan: Plan,
) -> Result<Timings, Box<dyn Error>> {
    eprintln!(
        "{shape} on {} / {}: staging and checking",
        plan.entry.name(),
        plan.scheduler.name()
    );
    run(context, shape.m, shape.n, shape.k, plan, time).map(|(_, timings)| timings)
}

/// One epilogue rung's minimum, at the shipped depth on the static schedule —
/// so `entry` is the only thing an epilogue table moves.
fn rung(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    shape: Shape,
    entry: Entry,
) -> Result<f64, Box<dyn Error>> {
    let plan = Plan {
        scheduler: Scheduler::Static,
        group: GROUP,
        entry,
    };
    Ok(timed(context, shape, plan)?.min())
}

/// The A/B — `cargo oxide run kittens-experiments -- ws`, which is
/// `scripts/modal-run ws_bench`.
///
/// **Both controls are re-measured in this session, not quoted.** Nothing here
/// changes [`crate::gemm`], so its numbers ought to be the ones
/// `experiments/README.md` §7 already carries; #98 found 2.9% of drift between
/// containers and #109 came within one paragraph of publishing a false +3.6%
/// against a baseline that had moved under it. A control that is not moving has
/// to be one measured beside the thing it controls — which since #117 means two
/// of them, the shipped `gemm` and its best rung, because "does this design
/// point beat the other one" is now a question about two moving kernels.
///
/// Six tables: the two designs, the two free levers, the epilogue ladder, what
/// the epilogue costs by subtraction, this kernel's best against `gemm`'s best,
/// and the library.
pub fn compare(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<Baseline>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "gemm warp specialization — min ms over 30 timed launches, every row checked\n\
         element-by-element against the same CPU reference before it was timed.\n\
         `gemm` in table 1 is experiments/src/gemm.rs on its REGISTER drain (`lcf`), named\n\
         rather than defaulted since #119: [256,256] at 3 stages, 4 warps, 98392 B, 2 CTAs\n\
         an SM, 256 accumulator columns. `ws` is this file on its own register drain at 6\n\
         warps, {} B, 1 CTA an SM, 512 accumulator columns in two ping-ponged stages.\n\
         In table 1 one variable moves — where the overlap comes from — and the epilogue is\n\
         deliberately identical, so that delta is the occupancy/specialization structure and\n\
         nothing else. Tables 3 and 4 then move the epilogue and nothing else, and table 5\n\
         puts each design point's best rung against the other's — which since #119 is also\n\
         each one's shipped rung, `staged84` there and `staged8` here.",
        SHARED_BYTES
    );

    println!("\n1. the two designs, at the three sizes the reference publishes");
    println!(
        "{:<18}{:>8}{:>8}{:>10}{:>11}{:>12}{:>11}{:>12}{:>10}",
        "shape",
        "ws tiles",
        "waves",
        "wave eff",
        "gemm ms",
        "gemm TF/s",
        "ws ms",
        "ws TF/s",
        "ws vs gemm"
    );
    let mut measured = Vec::new();
    for shape in SIZES {
        eprintln!("{shape} on gemm lcf (the control): staging and checking");
        // Named rather than taken from `gemm::bench`, and #119 is why: that
        // function launches `gemm`'s shipped epilogue, which is now `staged84`.
        // Table 1's whole claim is that the epilogue is identical on both
        // sides, so this arm has to be the register drain by name — the best
        // rungs meet in table 5, where each side is allowed to move.
        let control = crate::gemm::bench_with(context, shape, crate::gemm::Epilogue::Fused)?.min();
        let ours = timed(context, shape, Plan::new(Scheduler::Static))?.min();
        let (waves, efficiency) = wave_efficiency(shape.m, shape.n);
        println!(
            "{:<18}{:>8}{:>8}{:>9.1}%{:>11.4}{:>12.1}{:>11.4}{:>12.1}{:>10}",
            shape,
            tiles(shape.m, shape.n),
            waves,
            100.0 * efficiency,
            control,
            tflops(shape, control),
            ours,
            tflops(shape, ours),
            format!("{:+.1}%", 100.0 * (control / ours - 1.0))
        );
        measured.push((shape, control, ours));
    }

    println!(
        "\n2. the two levers this design point unlocks, both free in resources.\n\
         `clc` is the hardware schedule: a wave is 74 clusters here against gemm's 148, so\n\
         the ragged last wave is a coarser quantization of the same tile grid.\n\
         `s6` is six pipeline stages instead of four — 196744 B, which tensor memory has\n\
         already made free, since 512 accumulator columns fixed residency at 1 CTA/SM\n\
         before shared memory was consulted at all."
    );
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "shape", "ws ms", "ws clc ms", "vs static", "ws s6 ms", "s6 TF/s", "vs s4"
    );
    for &(shape, _, ours) in &measured {
        let stealing = timed(
            context,
            shape,
            Plan {
                scheduler: Scheduler::Stealing,
                group: GROUP,
                entry: Entry::S4,
            },
        )?
        .min();
        let deep = timed(
            context,
            shape,
            Plan {
                scheduler: Scheduler::Static,
                group: GROUP,
                entry: Entry::S6,
            },
        )?
        .min();
        println!(
            "{:<18}{:>12.4}{:>12.4}{:>12}{:>12.4}{:>12.1}{:>12}",
            shape,
            ours,
            stealing,
            format!("{:+.1}%", 100.0 * (ours / stealing - 1.0)),
            deep,
            tflops(shape, deep),
            format!("{:+.1}%", 100.0 * (ours / deep - 1.0))
        );
    }

    println!(
        "\n3. the epilogue ladder — #116's staged shape and #117's two instruction widths,\n\
         on the kernel that exposes the epilogue differently. In `gemm_cg2` the epilogue IS\n\
         the critical path: #114 measured `whole - no drain` at 1.01x the same epilogue's\n\
         serial cost. Here it is already deferred one item and already on warps of its own,\n\
         with the producer never stopping — so this is not a repeat of #117. It asks whether\n\
         a lever whose mechanism is a *latency the drain never overlaps* is worth anything\n\
         where something else is already covering that latency.\n\
         `ws s4`     registers -> global: 4 B a thread on 8 discontiguous 16 B runs\n\
         `staged`    .x1 LDTM (16 loads, 16 waits a band), stmatrix .x2, 16 B stores\n\
         `staged8`   .x8 LDTM ( 2 loads,  2 waits a band), stmatrix .x2\n\
         `staged4`   .x1 LDTM,                             stmatrix .x4\n\
         `staged84`  .x8 LDTM,                             stmatrix .x4\n\
         All four staged rows declare the same {STAGED_SHARED_BYTES} B and differ only in how\n\
         many instructions carry the same bytes to the same addresses; `ws s4` declares\n\
         {SHARED_BYTES} B. Residency is 1 CTA an SM in every row and can be nothing else:\n\
         {ACCUM_COLUMNS} accumulator columns are an SM's entire tensor memory."
    );
    print!("{:<18}{:>12}", "shape", "ws s4 ms");
    for entry in STAGED_ENTRIES {
        print!("{:>13}", format!("{} ms", entry.short()));
    }
    println!();
    let mut ladder = Vec::new();
    for &(shape, _, ours) in &measured {
        let mut arms = Vec::new();
        for entry in STAGED_ENTRIES {
            arms.push(rung(context, shape, entry)?);
        }
        print!("{:<18}{:>12.4}", shape, ours);
        for &arm in &arms {
            print!("{:>13.4}", arm);
        }
        println!();
        ladder.push((shape, ours, arms));
    }

    println!("\n   the same rows as throughput, and as a delta against `staged`");
    print!("{:<18}", "shape");
    for entry in STAGED_ENTRIES {
        print!("{:>15}", format!("{} TF/s", entry.short()));
    }
    for entry in STAGED_ENTRIES.into_iter().skip(1) {
        print!("{:>13}", format!("{} vs", entry.short()));
    }
    println!("{:>14}", "84 vs ws s4");
    for (shape, ours, arms) in &ladder {
        print!("{:<18}", shape);
        for &arm in arms {
            print!("{:>15.1}", tflops(*shape, arm));
        }
        for &arm in arms.iter().skip(1) {
            print!("{:>13}", format!("{:+.1}%", 100.0 * (arms[0] / arm - 1.0)));
        }
        println!("{:>14}", format!("{:+.1}%", 100.0 * (ours / arms[3] - 1.0)));
    }

    println!(
        "\n4. the epilogue-free floor — this kernel with no drain at all, which is what any\n\
         epilogue here is being subtracted from and, on its own, the most useful row in this\n\
         file. #114 ran the same probe on `gemm_cg2` and got 1850 TF/s against the library's\n\
         1808: THAT kernel is past parity once its epilogue is gone, so its whole gap was\n\
         the epilogue. Two controls because there are two envelopes — {SHARED_BYTES} B for the\n\
         register rung and {STAGED_SHARED_BYTES} B for all four staged ones — and they are\n\
         opcode-identical PTX at 28 registers apiece, differing in 16408 declared bytes that\n\
         no instruction touches. `envelope` is what those bytes cost, and it is the noise\n\
         floor of table 4b. Neither computes a `C` and neither is checked."
    );
    println!(
        "{:<18}{:>16}{:>13}{:>20}{:>13}{:>12}",
        "shape", "no drain ms", "TF/s", "staged no drain ms", "TF/s", "envelope"
    );
    let mut floors = Vec::new();
    for &(shape, _, _) in &measured {
        let bare = rung(context, shape, Entry::NoDrain)?;
        let staged_bare = rung(context, shape, Entry::StagedNoDrain)?;
        println!(
            "{:<18}{:>16.4}{:>13.1}{:>20.4}{:>13.1}{:>12}",
            shape,
            bare,
            tflops(shape, bare),
            staged_bare,
            tflops(shape, staged_bare),
            format!("{:+.1}%", 100.0 * (staged_bare / bare - 1.0))
        );
        floors.push((bare, staged_bare));
    }

    println!(
        "\n4b. what the epilogue COSTS in each arm, by #114's subtraction: the launch with the\n\
         drain minus the launch without it, over the items the busiest cluster walks. The\n\
         staged columns all subtract the staged control, so the four widths are on one\n\
         footing whatever the `envelope` row above says.\n\
         `LDTM half` is `staged - staged8` — the same subtraction #117 used to close #109's\n\
         8.3 us estimate to 8.07 us on `gemm_cg2`, taken here from the other container, and\n\
         the number that says whether this kernel really carries twice the LDTM."
    );
    print!("{:<18}{:>15}", "shape", "ws s4 us/tile");
    for entry in STAGED_ENTRIES {
        print!("{:>14}", format!("{} us", entry.short()));
    }
    println!("{:>12}", "LDTM half");
    for ((shape, ours, arms), &(bare, staged_bare)) in ladder.iter().zip(&floors) {
        let per_tile =
            |milliseconds: f64| milliseconds * 1e3 / items_on_critical_path(shape.m, shape.n);
        print!("{:<18}{:>15.2}", shape, per_tile(ours - bare));
        let costs: Vec<f64> = arms
            .iter()
            .map(|&arm| per_tile(arm - staged_bare))
            .collect();
        for &cost in &costs {
            print!("{:>14.2}", cost);
        }
        println!("{:>12.2}", costs[0] - costs[1]);
    }

    println!(
        "\n5. against `gemm`'s own best rung, measured in this container. #112 asked whether\n\
         giving up the second CTA pays with both kernels on the register epilogue, and the\n\
         answer was -7.3% at 16384^3 with no mechanism established. Both kernels have since\n\
         gained an epilogue, so this is that question again with each design point at its\n\
         best — and beside it the floor from table 4, which is this kernel with a FREE\n\
         epilogue. If `ws floor` still loses to `gemm 84`, no epilogue work can close the\n\
         gap and the gap was never the epilogue."
    );
    println!(
        "{:<18}{:>12}{:>13}{:>14}{:>16}{:>14}{:>12}{:>15}",
        "shape",
        "gemm ms",
        "gemm 84 ms",
        "gemm 84 TF/s",
        "ws staged84 ms",
        "ws 84 vs 84",
        "ws floor ms",
        "floor vs 84"
    );
    let mut best = Vec::new();
    for ((&(shape, control, _), (_, _, arms)), &(bare, _)) in
        measured.iter().zip(&ladder).zip(&floors)
    {
        eprintln!("{shape} on gemm staged84 (the control): staging and checking");
        let theirs =
            crate::gemm::bench_with(context, shape, crate::gemm::Epilogue::StagedWideX4)?.min();
        println!(
            "{:<18}{:>12.4}{:>13.4}{:>14.1}{:>16.4}{:>14}{:>12.4}{:>15}",
            shape,
            control,
            theirs,
            tflops(shape, theirs),
            arms[3],
            format!("{:+.1}%", 100.0 * (theirs / arms[3] - 1.0)),
            bare,
            format!("{:+.1}%", 100.0 * (theirs / bare - 1.0))
        );
        best.push(theirs);
    }

    println!(
        "\n6. against cuBLASLt on the same device in the same container — the denominator,\n\
         and the drift control that says how much of every delta above is the session.\n\
         `ws floor` is the epilogue-free kernel, the column #114's 1850-against-1808 is on."
    );
    println!(
        "{:<18}{:>13}{:>13}{:>9}{:>9}{:>9}{:>11}{:>12}{:>10}",
        "shape",
        "cuBLASLt ms",
        "theirs TF/s",
        "gemm",
        "gemm 84",
        "ws s4",
        "ws staged",
        "ws staged84",
        "ws floor"
    );
    for (((&(shape, control, ours), (_, _, arms)), &gemm_best), &(bare, _)) in
        measured.iter().zip(&ladder).zip(&best).zip(&floors)
    {
        let Some(baseline) = baseline else {
            println!(
                "no cuBLASLt column: built without --features cublas. modal_app.py::ws_bench\n\
                 turns it on, and a ratio is the point of this table."
            );
            break;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        println!(
            "{:<18}{:>13.4}{:>13.1}{:>9.3}{:>9.3}{:>9.3}{:>11.3}{:>12.3}{:>10.3}",
            shape,
            theirs,
            tflops(shape, theirs),
            theirs / control,
            theirs / gemm_best,
            theirs / ours,
            theirs / arms[0],
            theirs / arms[3],
            theirs / bare
        );
    }
    Ok(())
}

/// The K = 3072 geometries oxide-train#80's model_shapes rows put this design
/// point's question at, and no smaller a question — `scripts/modal-run
/// ws_bench --case shallow` (#188).
///
/// Every number this file held before it was at K = M = N, which is the regime
/// where the overlap warp specialization buys is worth the least: #114's cube
/// put the shipped kernel's epilogue at ~20 µs a tile *fixed*, so its share of
/// an item falls as K grows, and −7.3% at 16384³ (#112) priced the solo MMA
/// chain and said nothing about the drain. At K = 3072 the exposed drain is
/// 15–25% of an item downstream and the shipped design decays to 0.75–0.85 of
/// cuBLASLt; this is the first table where both facts are in frame at once.
///
/// Five arms, each the smallest set that can settle the question:
/// `gemm staged84` is the shipped 2-CTA design at its best rung, `ws staged8`
/// this kernel at its own, `ws s6` the depth the freed shared memory affords,
/// `ws no drain` the floor — **not a GEMM**, and the kill criterion: a floor
/// already losing to `gemm staged84` means no epilogue placement can save the
/// design point — and cuBLASLt the denominator, all in one container.
pub fn shallow(
    context: &std::sync::Arc<cuda_core::CudaContext>,
    baseline: Option<Baseline>,
) -> Result<(), Box<dyn Error>> {
    const SHALLOW: [Shape; 4] = [
        // oxide-train's gate_up fwd, qkv fwd and o_proj fwd geometries, then
        // 8192³ — the deep-K control every prior gemm_ws table is quotable at.
        Shape {
            m: 6144,
            n: 8192,
            k: 3072,
        },
        Shape {
            m: 24576,
            n: 9216,
            k: 3072,
        },
        Shape {
            m: 24576,
            n: 3072,
            k: 3072,
        },
        Shape {
            m: 8192,
            n: 8192,
            k: 8192,
        },
    ];
    println!(
        "gemm_ws at shallow K (#188) — min ms over 30 timed launches, every arm but the\n\
         floor checked element-by-element against the CPU reference before it was timed.\n\
         `gemm 84` is experiments/src/gemm.rs on staged84: 2 CTAs an SM, 256 accumulator\n\
         columns, the epilogue exposed (#114: 1.01x). `ws` rungs are this file: 1 CTA an\n\
         SM, 512 columns in two ping-ponged stages, the epilogue deferred one item on\n\
         warps of its own. `ws floor` computes no C and is the most this design could\n\
         reach with a free epilogue."
    );
    println!(
        "{:<18}{:>8}{:>12}{:>12}{:>12}{:>12}{:>13}",
        "shape", "tiles", "gemm 84 ms", "ws 8 ms", "ws s6 ms", "ws floor", "ws 8 vs 84"
    );
    let mut rows = Vec::new();
    for shape in SHALLOW {
        eprintln!("{shape} on gemm staged84 (the control): staging and checking");
        let control =
            crate::gemm::bench_with(context, shape, crate::gemm::Epilogue::StagedWideX4)?.min();
        let ours = rung(context, shape, SHIPPED_ENTRY)?;
        let deep = rung(context, shape, Entry::S6)?;
        let floor = rung(context, shape, Entry::NoDrain)?;
        println!(
            "{:<18}{:>8}{:>12.4}{:>12.4}{:>12.4}{:>12.4}{:>13}",
            shape,
            tiles(shape.m, shape.n),
            control,
            ours,
            deep,
            floor,
            format!("{:+.1}%", 100.0 * (control / ours - 1.0))
        );
        rows.push((shape, control, ours, deep, floor));
    }
    println!("\nagainst cuBLASLt in the same container — the ratio oxide-train#80 counts in");
    println!(
        "{:<18}{:>13}{:>13}{:>9}{:>9}{:>9}{:>10}",
        "shape", "cuBLASLt ms", "theirs TF/s", "gemm 84", "ws 8", "ws s6", "ws floor"
    );
    for &(shape, control, ours, deep, floor) in &rows {
        let Some(baseline) = baseline else {
            println!(
                "no cuBLASLt column: built without --features cublas, and a ratio is the\n\
                 point of this table."
            );
            break;
        };
        eprintln!("{shape}: staging and checking {}", baseline.name);
        let theirs = (baseline.bench)(context, shape)?.0.min();
        println!(
            "{:<18}{:>13.4}{:>13.1}{:>9.3}{:>9.3}{:>9.3}{:>10.3}",
            shape,
            theirs,
            tflops(shape, theirs),
            theirs / control,
            theirs / ours,
            theirs / deep,
            theirs / floor
        );
    }
    Ok(())
}
