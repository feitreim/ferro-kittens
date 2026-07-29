//! What `gemm_sol`'s time is made of — issue #138's three candidate costs, plus
//! the one that is arithmetic rather than a candidate.
//!
//! The port is 0.795 of cuBLASLt at 4096³, 0.873 at 8192³ and 0.946 at 16384³,
//! and the PR names three suspects: TMA feed cadence, MMA issue cadence, and
//! epilogue overlap. This file separates them, and it puts a fourth in front of
//! them because the fourth is free to derive and the other three are not.
//!
//! # The fourth: wave quantization
//!
//! The shared plan is 196 864 B for M256 and 229 632 B for M512, both above
//! half of a Blackwell SM's 233 472 B, so **one CTA per SM** — and a cluster is
//! two CTAs, so a wave on this 148-SM B200 is **74 clusters**. The launch is
//! exactly one cluster per output tile, so the ceiling any cadence is measured
//! against is
//!
//! ```text
//! clusters / (74 * ceil(clusters / 74))
//! ```
//!
//! At 4096³ with the M256 entry that is 256 clusters over four waves of 74 —
//! **0.865**, with the last wave 46% full. At 8192³ with M512 it is 512 over
//! seven — 0.988. The two shapes with a gap are therefore not the same problem,
//! and a number quoted against peak at 4096³ that does not divide by 0.865 is
//! attributing a scheduling loss to a cadence.
//!
//! [`sawtooth`] measures that rather than asserting it: `N` moves in tile
//! steps, the cluster count moves with it, and predicted efficiency runs
//! sawtooth against a work total that only rises. A shape with *more* work that
//! finishes in the same wall time is wave quantization, visible without a
//! counter.
//!
//! # The three: the ablation arms
//!
//! Each arm is [`crate::gemm_sol`]'s own device body at a different `ABLATE`
//! const, so no arm can drift from the kernel it decomposes — the shipped
//! kernels are the same text at `WHOLE`.
//!
//! | arm | producer | MMA warp | epilogue warps |
//! | --- | --- | --- | --- |
//! | `whole` | TMA | `tcgen05.mma` | LDTM → `stmatrix` → `st.global` |
//! | `no drain` | TMA | `tcgen05.mma` | handshake only |
//! | `issue only` | arrives on `load`, no TMA | `tcgen05.mma` | handshake only |
//! | `feed only` | TMA | waits and commits, no MMA | handshake only |
//! | `twice global` | TMA | `tcgen05.mma` | + a second `ld.shared` + `st.global` |
//! | `twice shared` | TMA | `tcgen05.mma` | + a second `cvt` + `stmatrix` too |
//! | `twice all` | TMA | `tcgen05.mma` | + a second LDTM: the whole drain twice |
//! | `paired` | TMA | `tcgen05.mma` | 2 LDTM in flight a band, one wait |
//! | `wide` | TMA | `tcgen05.mma` | 128-column bands, 4 in flight, one wait |
//!
//! The first four are #144's and are `ABLATE` values; the last five are this
//! file's, and the `twice` three are `ABLATE` while `paired` and `wide` are a
//! second dial, `DRAIN`. `whole` is the whole kernel at `DRAIN_PER_ISSUE`, which
//! is the drain that shipped through #144 and the one the ladder below
//! decomposes; `SHIPPED_DRAIN` is `DRAIN_PAIRED` since this file, so `shipped`
//! and `whole` now differ by that one rung and the `drains` table is where the
//! difference is measured.
//!
//! **Every barrier survives every arm.** The `load`/`free` ring, the `full`/
//! `empty` accumulator handshake, the CLC cursor and the `ready` publication
//! are untouched, so an arm removes one *phase* and never the pipeline that
//! phase runs in. That is what makes `whole − no drain` the epilogue's cost
//! *including its overlap failure* rather than its cost in isolation.
//!
//! **These compute a wrong `C` on purpose** and are on no correctness gate,
//! like the third of this crate's entry points that already do. `regcount`'s
//! opcode census is the gate that matters here instead: `feed only` must show
//! zero in the `mma` column and the whole kernel's `tma` count, `issue only`
//! the reverse, and `no drain` zero across `ldtm`, `stmatrix` and the store
//! columns while keeping both of the others. An arm whose census does not read
//! that way did not remove what it names, and its number is void.
//!
//! The doubling arms are held to the same standard from the other side: a rung
//! that says it repeats a pass must show that pass's columns **doubled** and
//! every other column unmoved. And the two drains must show the thing they
//! change, which is why `regcount` gained an `ldtm.wait` column — batching moves
//! no other count in the table, so without it a batched drain and an unbatched
//! one census identically and neither arm could be gated at all.
//!
//! The census says they do. `regcount` on this tree, `mbar.arrive` 6 on every
//! M256 arm and 8 on every M512 one:
//!
//! ```text
//! kernel                     mma  ldtm  ldtm.wait  tma  stmatrix  ld.sh.v4  st.g.v4  st.g.b32  cvt  regs  frame
//! gemm_sol_m256_whole         16     2          2    3         2         1        1         4    8    96    256
//! gemm_sol_m256_nodrain       16     0          0    3         0         0        0         0    0    33      0
//! gemm_sol_m256_issue         16     0          0    0         0         0        0         0    0    32      0
//! gemm_sol_m256_feed           0     0          0    3         0         0        0         0    0    31      0
//! gemm_sol_m256_twice_global  16     2          2    3         2         2        2         8    8   114    256
//! gemm_sol_m256_twice_shared  16     2          2    3         4         2        2         8   16   137    256
//! gemm_sol_m256_twice_all     16     4          4    3         4         2        2         8   16   114    256
//! gemm_sol_m256_paired        16     2          1    3         2         1        1         4    8    96    256
//! gemm_sol_m256_wide          16     4          1    3         2         1        1         4    8   176    512
//! gemm_sol_m512_whole         32     2          2    4         2         1        1         4    8    94    256
//! gemm_sol_m512_paired        32     2          1    4         2         1        1         4    8   112    256
//! gemm_sol_m512_wide          32     4          1    4         2         1        1         4    8   168    512
//! ```
//!
//! Three things to read off it. `paired` is the shipped drain with **one wait
//! where there were two** and every other column identical, including registers
//! and frame at M256. `wide` is four issues behind one wait, and it is the only
//! rung that costs: +80 registers and a 512-byte frame where the others carry
//! 256. And the `twice` ladder doubles exactly the columns it names — the
//! compiler did not common up the second LDTM, which it would have been free to
//! do if it modelled `tcgen05.ld` as pure.
//!
//! **`wide`'s first spelling was void and is worth recording.** Written as a
//! loop over an `[_; ISSUES]` array of arrivals, it censused as **one** `ldtm`
//! with a **1024-byte** frame: LLVM had declined to unroll at four issues, so
//! the array needed a runtime index and went to local memory. Four loads in
//! flight through L1 is not the thing the arm names.
//! [`kittens::tmem::TmemTile::tile_x8_batched`] names its issues at literal
//! indices for that reason.
//!
//! `mbar.arrive` is 6 on every M256 arm and 8 on every M512 arm, which is the
//! claim that no arm dropped a barrier — and `gemm_sol_m256`/`gemm_sol_m512`
//! are opcode-identical between the two crates, so putting the dial on the
//! shipped body moved no instruction in the shipped kernel.
//!
//! Registers fall from 96 to 31-33 on the M256 arms. That cannot buy an
//! ablation occupancy it did not earn: at 192 threads even 104 registers admits
//! three CTAs, and both shared plans admit exactly one, so shared memory binds
//! residency in every arm and the register column is inert here.
//!
//! # What it found
//!
//! One B200 container, 5 warm-up and min of 30 per row. `whole` reproduced to
//! 1.5% across the four tables that measure it.
//!
//! The depth ladder is a straight line in `K` at both variants — the M512 one
//! to within 0.5% over an eightfold range — so the launch is exactly a per-tile
//! constant plus a per-`K`-block rate, and both fall out of it:
//!
//! | | per k block | at peak | of peak | per tile, fixed |
//! | --- | ---: | ---: | ---: | ---: |
//! | M256, m=n=4096 | 0.364 µs | 0.2759 µs | **75.8%** | **3.25 µs** |
//! | M512, m=n=8192 | 0.555 µs | 0.5518 µs | **99.4%** | **12.7 µs** |
//!
//! Fed back, the fit reproduces both launches to under 1%: `4 × (3.25 + 64 ×
//! 0.364) = 106.2 µs` against 105.5 measured, and `7 × (12.7 + 128 × 0.555) =
//! 586 µs` against 586.0.
//!
//! **The M512 K loop already runs at 99.4% of tensor-core peak.** Nothing in
//! its steady state is worth another look. Its whole deficit is the 12.7 µs it
//! pays per tile, and the ablation says what that is: the drain is 9.99 µs of
//! it. The M256 K loop is a different problem — it runs at 75.8%, and that is a
//! cadence rather than a constant.
//!
//! Subtracting the arms gives an additive budget for each launch, in
//! milliseconds and as a share of the whole:
//!
//! | term | 4096³ M256 | 8192³ M512 |
//! | --- | ---: | ---: |
//! | arithmetic at peak | 0.0611 (58.1%) | 0.4889 (83.4%) |
//! | wave quantization | 0.0095 (9.1%) | 0.0059 (1.0%) |
//! | epilogue, `whole − no drain` | **0.0207 (19.7%)** | **0.0699 (11.9%)** |
//! | TMA feed in situ, `no drain − issue only` | 0.0056 (5.3%) | 0.0082 (1.4%) |
//! | MMA issue cadence, the rest of `issue only` | 0.0083 (7.9%) | 0.0131 (2.2%) |
//! | measured | 0.1052 | 0.5860 |
//!
//! **The epilogue is the largest single loss at both shapes**, and at 8192³ it
//! is 11.9 of the 13.2 points between the launch and peak — the other two
//! suspects together are 3.6%. At 4096³ it is still the largest, but it is no
//! longer alone: quantization, MMA issue cadence and the feed are 9.1%, 7.9%
//! and 5.3%, and a fix that took only the epilogue would leave 22% on the
//! table there against 3.6% at 8192³.
//!
//! The drain costs the same *per byte of `C`* in both variants — 5.18 µs for
//! 131 072 B at M256 and 9.99 µs for 262 144 B at M512, which is 25.3 and
//! 26.2 GB/s per cluster. 74 clusters of that is 1.9 TB/s of stores against a
//! part that does 8, so the drain is not store-bandwidth-bound and not
//! variant-specific: it is the `tcgen05.ld` → `stmatrix` → `st.global` chain
//! itself, running at a fixed rate in four warps.
//!
//! Re-taken in this file's container the budget is 0.1056 and 0.5866 ms whole
//! against 0.0841 and 0.5162 no drain — an epilogue of **20.5%** and **12.0%**,
//! reproducing #144's 19.7 and 11.9 to within a point. The `shipped` row, which
//! is the same device body compiled in `examples/` rather than here, reads
//! 0.1060 and 0.5859: **the crate boundary is worth 0.4% and 0.1%**, which is
//! what licenses every A/B below to compare an `examples/` kernel with an
//! `experiments/` one.
//!
//! **The feed is not the problem at either shape.** `feed only` moves the
//! launch's whole operand traffic — 1.07 GB at 4096³, 6.44 GB at 8192³ — in
//! 57% and 49% of the launch, which is 18.0 and 22.4 TB/s against 11.0 and 10.2
//! in situ. Those rates are far above the part's 8 TB/s of HBM, so most of it
//! is L2, which is what the `A` matrix fitting in L2 at these sizes predicts.
//!
//! # The drain, split three ways — and the overlap term separated
//!
//! #144 left "the epilogue's own cost from its failure to overlap" unseparated
//! and said it needed a fifth arm. It needs three, and they are a **doubling
//! ladder**: each rung repeats one more pass of the drain per band, with the
//! extra global stores aimed at the cluster's own first output tile so they stay
//! in L2 and the probe prices instructions rather than a second HBM stream —
//! `gemm.rs`' `drain_staged_twice` and #123's `2g` probes, asked of this kernel.
//!
//! An extra pass has nothing left to hide behind, so it is paid **serially by
//! construction**. That makes `twice all − whole` the drain's own occupancy cost
//! `D`, against which `E = whole − no drain` is what the launch actually pays.
//! One container, µs per tile per cluster:
//!
//! | | 4096³ M256 | 8192³ M512 |
//! | --- | ---: | ---: |
//! | `ld.shared` + `st.global` | 1.07 (38%) | 3.40 (40%) |
//! | `cvt` + `stmatrix` | 1.75 (62%) | 6.00 (70%) |
//! | **LDTM and its wait** | **0.00 (0%)** | **−0.88 (−10%)** |
//! | `D`, the whole chain serially | 2.82 | 8.53 |
//! | `E`, what the launch pays | **5.39 (191% of `D`)** | **10.08 (118% of `D`)** |
//!
//! **`E > D` at both shapes, so none of the drain is hidden.** Not "incompletely
//! overlapped" — *negatively* overlapped: the first drain costs the launch more
//! than a whole extra one does, which is only possible if the drain slows the
//! phase it runs beside. At M256, where the accumulator *is* double-buffered
//! across tiles and the structure permits overlap, it is nearly twice as bad
//! (191%). So the epilogue's problem is not that some of it fails to overlap; it
//! is that all of it is exposed and it takes something from the rest of the
//! kernel besides.
//!
//! **And the LDTM half is free.** Doubling the `tcgen05.ld` count and its waits
//! — census `ldtm` 2 → 4, `ldtm.wait` 2 → 4 — costs **nothing** at 4096³ and
//! reads slightly negative at 8192³. That is the opposite of what #117 found on
//! `gemm`, and both are true: #117 took eight issues and eight waits a band down
//! to one and one and was worth +23.6/+8.6/+3.6%, and from *there* the remaining
//! waits are already hidden by the other three warps. The lever #117 opened is
//! closed. `DRAIN_PAIRED` takes the last two waits a band to one and is worth
//! +0.9% at 8192³ and nothing at 4096³, which is what a term measured at zero
//! predicts.
//!
//! **What is left is the register→shared half.** 60-70% of `D`, and by
//! subtraction 5.13 of 8.53 µs at M512. It is not bandwidth: 262 144 B of `C`
//! into shared per cluster per tile in ~6 µs is 3.5 TB/s device-wide against an
//! SM's ~230 GB/s of shared write bandwidth, or under 10% of it. It is not bank
//! conflicts either — `Swizzle128B` maps the eight rows of a `stmatrix.m8n8`
//! matrix onto eight distinct chunks whose banks tile all 32 exactly once, at
//! both staging pitches. What is left is **latency in four warps**, and the
//! epilogue cannot have more than four: TMEM lane ownership gives a warp the 32
//! lanes of its own sub-partition, so `EPILOGUE_WARPS = BLOCK_M / 32` is the
//! hardware's arithmetic and not a tuning knob.
//!
//! # What this does not separate
//!
//! - **`cvt` + `stmatrix` from a write-after-write on its own staging tile.**
//!   Rung 2's second `stmatrix` pass lands on the words rung 1 just wrote, so
//!   6.00 µs is an *upper* bound on that pass. The lower bound is the same
//!   number by subtraction — `D` minus the two clean rungs is 5.13 — and the
//!   conclusion is the same either way.
//! - **`wide`'s loss from `wide`'s stack frame.** It is the only rung that moves
//!   registers and frame, and this file cannot say how much of its −1.1% at
//!   4096³ is the wider band and how much is the 512 bytes.
//! - **Barrier round-trip latency from tensor-core issue rate.** `issue only`
//!   keeps the whole `load`/`free` handshake, so the 24.2% the M256 K loop is
//!   short of peak is those two fused. The suggestive number is that M512
//!   issues two MMA chains per handshake and loses 0.6% where M256 issues one
//!   and loses 10.5% on the same arm — but that is one observation across two
//!   variants, not a separated term.
//! - **`feed only`'s ceiling from what the feed can do under contention.** That
//!   arm has no MMA reading the same shared memory, so 18-22 TB/s is an upper
//!   bound on the feed and not its in-situ rate. The in-situ number is the
//!   `no drain − issue only` row, which is the one quoted in the budget.
//! - **The 4096³ sawtooth's residual.** `corrected` is flat to 0.6% at M512 and
//!   drifts about 6% downward with `N` at M256, and that drift is not separated
//!   from the growing L2 footprint of `B`.
//! - **cuBLASLt's own schedule.** The `of ceiling` column divides our ratio by
//!   *our* wave quantization and the library has its own, unknown, so that
//!   column is not a residual. It reads 0.878 at both shapes, which the budget
//!   above says is a coincidence of two different budgets and not a constant.
//! - **Our drain from upstream's.** `gemm_sol_final` runs the same
//!   `tcgen05.ld` → `stmatrix` → `st.global` algorithm, so some of `E` is a cost
//!   the reference pays too, and nothing here says how much. #145's ratios bound
//!   it without a measurement: the port is 4.3% behind upstream at 8192³ and
//!   7.8% at 4096³, so **at most 4.3 of the 12.0 points and at most 7.8 of the
//!   20.5 can be port overhead** — the rest is a cost upstream pays as well,
//!   unless the port is *ahead* of upstream somewhere else. The arm that would
//!   settle it is upstream's own device body at `NO_DRAIN`, which wants #145's
//!   harness in this crate.
//!
//! # What to try next, and the arithmetic that says so
//!
//! Both remaining routes are concurrency, because that is what the ladder left.
//!
//! - **A second epilogue warpgroup.** Four warps cannot fill the shared-memory
//!   pipeline and TMEM lane ownership forbids a fifth *within* a warpgroup — but
//!   warp 4 is warpgroup 1's warp 0 and owns the same 32 lanes as warp 0, so
//!   eight epilogue warps can split the accumulator's **columns** where they
//!   cannot split its rows. It is free in the resource that binds: eight warps
//!   staging `[32, 128]` is 65 536 B at M256 against four staging `[32, 256]`,
//!   and eight staging `[32, 64]` is 32 768 B at M512 against four staging
//!   `[32, 128]` — **both plans byte-identical**, so the launch contract's
//!   shared figure and the `no drain` control both survive. It costs 320 threads
//!   rather than 192, which at 112 registers is still inside the register file.
//! - **Overlapping the drain at M512 needs TMEM this variant does not have.**
//!   `ACCUM_COLUMNS` is 512, which is the whole of a CTA's tensor memory, and
//!   M512 spends all of it on one output tile — so tile `s + 1`'s MMA cannot
//!   begin until tile `s`'s drain has freed a half, which is exactly what the
//!   `empty` handshake says. The two ways out both cost elsewhere: sequencing
//!   the halves' K loops instead of interleaving them re-reads `B` (operand
//!   traffic per tile 768·K → 1024·K, +33%), and an `N`-128 accumulator buys
//!   four buffers at twice the MMA issue count. M256 already has the buffering
//!   and `E/D = 191%` there says buffering is not sufficient.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract};

use kittens::global::GlobalLayout;
use kittens::shared::F16;

use crate::bench::{Shape, Timings, time};
use crate::gemm_sol::{
    ATile, BPanel, DRAIN_PAIRED, DRAIN_PER_ISSUE, DRAIN_WIDE, FEED_ONLY, ISSUE_ONLY, K_TILE,
    N_TILE, NO_DRAIN, THREADS, TWICE_ALL, TWICE_GLOBAL, TWICE_SHARED, Variant, WHOLE, grid_for,
    m256_body, m512_body,
};

/// SMs on the B200 this file's arithmetic is for, and CTAs one of them holds at
/// either variant's shared plan. A cluster is two CTAs, so [`WAVE`] clusters
/// are resident at once and everything below follows from that.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 1;
const RANKS: u32 = 2;
/// Clusters resident at once — 74.
pub const WAVE: u32 = SMS * CTAS_PER_SM / RANKS;

const _: () = assert!(WAVE == 74);

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The shipped kernel's own device body, compiled **in this crate**.
    ///
    /// Every other arm here is this text at another dial, so the A/B never
    /// crosses a crate boundary. `Arm::Shipped` measures the `examples/` copy
    /// beside it, which is that boundary asked as a question instead.
    ///
    /// # Safety
    ///
    /// The maps and output must cover the stated tile grid and K depth, and
    /// the launch must be one two-CTA cluster per output tile — as
    /// [`crate::gemm_sol::kernels::gemm_sol_m256`]. Every arm but `whole`,
    /// `paired` and `wide` computes a wrong `C` on purpose.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_whole(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<WHOLE, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// TMA and every barrier, with the MMA and the drain gone.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_feed(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<FEED_ONLY, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// The MMA and every barrier, with no global traffic at all.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_issue(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<ISSUE_ONLY, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// The whole kernel except the drain.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_nodrain(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<NO_DRAIN, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// The drain's `ld.shared` + `st.global` half run twice a band.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_global(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<TWICE_GLOBAL, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_twice_global`] with the `cvt` + `stmatrix` pass doubled too.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_shared(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<TWICE_SHARED, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// The whole drain twice, LDTM included — the drain priced serially.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_twice_all(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<TWICE_ALL, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// The whole kernel with the drain's two LDTM issues a band in flight before
    /// one wait.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_paired(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<WHOLE, DRAIN_PAIRED>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c)
        }
    }

    /// The whole kernel draining 128-column bands, four LDTM issues behind one
    /// wait.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_wide(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m256_body::<WHOLE, DRAIN_WIDE>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_whole`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_whole(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<WHOLE, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_feed`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_feed(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<FEED_ONLY, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_issue`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_issue(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<ISSUE_ONLY, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_nodrain`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_nodrain(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<NO_DRAIN, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_twice_global`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_global(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<TWICE_GLOBAL, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_twice_shared`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_shared(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<TWICE_SHARED, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_twice_all`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_twice_all(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<TWICE_ALL, DRAIN_PER_ISSUE>(
                a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c,
            )
        }
    }

    /// [`gemm_sol_m256_paired`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_paired(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<WHOLE, DRAIN_PAIRED>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_wide`] at the M512 entry.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m512_whole`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 229_632,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m512_wide(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            m512_body::<WHOLE, DRAIN_WIDE>(a_map, b_map, tiles_m, tiles_n, k_blocks, ldc, &mut c)
        }
    }
}

/// One arm: a phase of the item removed, a pass of the drain doubled, or the
/// drain rebuilt.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The kernel `examples/` ships, compiled in *that* crate.
    Shipped,
    /// The same device body at the same dials, compiled in this one.
    Whole,
    NoDrain,
    IssueOnly,
    FeedOnly,
    TwiceGlobal,
    TwiceShared,
    TwiceAll,
    Paired,
    Wide,
}

impl Arm {
    /// The budget #144 published: every phase of the item, one at a time.
    const BUDGET: [Arm; 4] = [Arm::Whole, Arm::NoDrain, Arm::IssueOnly, Arm::FeedOnly];
    /// The doubling ladder. Each rung adds one pass of the drain to the rung
    /// below, so the differences price the chain a third at a time, and the top
    /// rung against `whole` prices a whole extra drain — serially, by
    /// construction.
    const LADDER: [Arm; 4] = [
        Arm::Whole,
        Arm::TwiceGlobal,
        Arm::TwiceShared,
        Arm::TwiceAll,
    ];
    /// The drains, all at [`crate::gemm_sol::WHOLE`] and all on the same shared
    /// plan, so one `no drain` control serves every one of them.
    const DRAINS: [Arm; 3] = [Arm::Whole, Arm::Paired, Arm::Wide];

    fn name(self) -> &'static str {
        match self {
            Arm::Shipped => "shipped",
            Arm::Whole => "whole",
            Arm::NoDrain => "no drain",
            Arm::IssueOnly => "issue only",
            Arm::FeedOnly => "feed only",
            Arm::TwiceGlobal => "twice global",
            Arm::TwiceShared => "twice shared",
            Arm::TwiceAll => "twice all",
            Arm::Paired => "paired",
            Arm::Wide => "wide",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Arm::Shipped => "the kernel examples/ ships, from that crate",
            Arm::Whole => "the same body, this crate",
            Arm::NoDrain => "TMA + MMA + every barrier; the drain removed",
            Arm::IssueOnly => "MMA + every barrier; no global traffic at all",
            Arm::FeedOnly => "TMA + every barrier; no MMA, no drain",
            Arm::TwiceGlobal => "+ a second ld.shared + st.global pass a band",
            Arm::TwiceShared => "+ a second cvt + stmatrix pass as well",
            Arm::TwiceAll => "+ a second LDTM: the whole drain twice",
            Arm::Paired => "64-column bands, 2 LDTM in flight, 1 wait",
            Arm::Wide => "128-column bands, 4 LDTM in flight, 1 wait",
        }
    }
}

/// Clusters the shape launches, and the fraction of the waves they fill.
///
/// The launch is one cluster per output tile and the device holds [`WAVE`] of
/// them, so this is the whole of it: no residency model, no measurement.
pub fn quantization(shape: Shape, variant: Variant) -> (u32, u32, f64) {
    let clusters = grid_for(shape, variant) / RANKS;
    let waves = clusters.div_ceil(WAVE);
    (clusters, waves, clusters as f64 / (waves * WAVE) as f64)
}

fn legal(shape: Shape, variant: Variant) -> Result<(), Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    if !m.is_multiple_of(variant.m_tile()) || !n.is_multiple_of(N_TILE) || !k.is_multiple_of(256) {
        return Err(format!("{shape} violates {}'s tile contract", variant.name()).into());
    }
    Ok(())
}

/// Time one arm at one shape.
///
/// Operands are zeroed rather than staged, exactly as [`crate::gemm_sol::bench`]
/// times the shipped kernel: nothing here is checked, three of the four arms
/// cannot be, and tensor cores do not take data-dependent time.
pub fn measure(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
    arm: Arm,
) -> Result<Timings, Box<dyn Error>> {
    legal(shape, variant)?;
    let Shape { m, n, k } = shape;
    let stream = context.default_stream();
    let shipped = unsafe { crate::gemm_sol::kernels::load(context)? };
    let ablated = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::<u16>::zeroed(&stream, m * k)?;
    let b = DeviceBuffer::<u16>::zeroed(&stream, n * k)?;
    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;

    let config = LaunchConfig1D::new(
        grid_for(shape, variant),
        THREADS,
        variant.shared_bytes() as u32,
    );
    let (stream_ref, a_ptr, b_ptr) = (&stream, a_map.as_ptr(), b_map.as_ptr());
    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let k_blocks = (k / K_TILE) as u32;
    let ldc = n as u32;

    // One spelling per arm, and the argument list written once: the arms differ
    // in the entry point they call and in nothing else, which is the claim the
    // table rests on.
    macro_rules! launcher {
        ($module:expr, $prepare:ident, $call:ident) => {{
            let module = $module;
            let prepared = module.$prepare(config)?;
            let launch: Box<dyn Fn(&mut DeviceBuffer<u16>) -> Result<(), Box<dyn Error>>> =
                Box::new(move |output| {
                    unsafe {
                        module.$call(
                            stream_ref, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, ldc,
                            output,
                        )?
                    };
                    Ok(())
                });
            launch
        }};
    }

    let launch_once = match (variant, arm) {
        (Variant::M256xN256, Arm::Shipped) => {
            launcher!(&shipped, prepare_gemm_sol_m256, gemm_sol_m256)
        }
        (Variant::M256xN256, Arm::Whole) => {
            launcher!(&ablated, prepare_gemm_sol_m256_whole, gemm_sol_m256_whole)
        }
        (Variant::M256xN256, Arm::NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_nodrain,
                gemm_sol_m256_nodrain
            )
        }
        (Variant::M256xN256, Arm::IssueOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m256_issue, gemm_sol_m256_issue)
        }
        (Variant::M256xN256, Arm::FeedOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m256_feed, gemm_sol_m256_feed)
        }
        (Variant::M256xN256, Arm::TwiceGlobal) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_global,
                gemm_sol_m256_twice_global
            )
        }
        (Variant::M256xN256, Arm::TwiceShared) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_shared,
                gemm_sol_m256_twice_shared
            )
        }
        (Variant::M256xN256, Arm::TwiceAll) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m256_twice_all,
                gemm_sol_m256_twice_all
            )
        }
        (Variant::M256xN256, Arm::Paired) => {
            launcher!(&ablated, prepare_gemm_sol_m256_paired, gemm_sol_m256_paired)
        }
        (Variant::M256xN256, Arm::Wide) => {
            launcher!(&ablated, prepare_gemm_sol_m256_wide, gemm_sol_m256_wide)
        }
        (Variant::M512xN256, Arm::Shipped) => {
            launcher!(&shipped, prepare_gemm_sol_m512, gemm_sol_m512)
        }
        (Variant::M512xN256, Arm::Whole) => {
            launcher!(&ablated, prepare_gemm_sol_m512_whole, gemm_sol_m512_whole)
        }
        (Variant::M512xN256, Arm::NoDrain) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_nodrain,
                gemm_sol_m512_nodrain
            )
        }
        (Variant::M512xN256, Arm::IssueOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m512_issue, gemm_sol_m512_issue)
        }
        (Variant::M512xN256, Arm::FeedOnly) => {
            launcher!(&ablated, prepare_gemm_sol_m512_feed, gemm_sol_m512_feed)
        }
        (Variant::M512xN256, Arm::TwiceGlobal) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_global,
                gemm_sol_m512_twice_global
            )
        }
        (Variant::M512xN256, Arm::TwiceShared) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_shared,
                gemm_sol_m512_twice_shared
            )
        }
        (Variant::M512xN256, Arm::TwiceAll) => {
            launcher!(
                &ablated,
                prepare_gemm_sol_m512_twice_all,
                gemm_sol_m512_twice_all
            )
        }
        (Variant::M512xN256, Arm::Paired) => {
            launcher!(&ablated, prepare_gemm_sol_m512_paired, gemm_sol_m512_paired)
        }
        (Variant::M512xN256, Arm::Wide) => {
            launcher!(&ablated, prepare_gemm_sol_m512_wide, gemm_sol_m512_wide)
        }
    };

    launch_once(&mut c)?;
    stream.synchronize()?;
    let mut launch = || launch_once(&mut c);
    time(&stream, &mut launch)
}

/// The arms that compute a *right* `C`, and the exact BF16 gate they are on.
///
/// [`Arm::Paired`] and [`Arm::Wide`] rebuild the drain, so "it is faster" is
/// worth nothing until they are known to write what the shipped drain writes —
/// and a wrong register order is exactly how a batched LDTM breaks. This is
/// [`crate::gemm_sol::check`]'s own reference and its own `==` comparison, at
/// the same check size, asked of the ablated module's entry points; `Arm::Whole`
/// rides along so a failure can be attributed to the drain rather than to this
/// harness.
///
/// Every other arm removes or doubles a phase and cannot be checked at all.
pub fn check(context: &Arc<CudaContext>) -> Result<String, Box<dyn Error>> {
    let (m, n, k) = (1024usize, 1024usize, 512usize);
    let stream = context.default_stream();
    let ablated = unsafe { kernels::load(context)? };

    let a = DeviceBuffer::from_host(
        &stream,
        &crate::gemm_sol::stage_f16(m, k, |row, depth| crate::gemm_sol::a_value(row, depth)),
    )?;
    let b = DeviceBuffer::from_host(
        &stream,
        &crate::gemm_sol::stage_f16(n, k, |row, depth| crate::gemm_sol::b_value(row, depth)),
    )?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;
    let (tiles_m, tiles_n, k_blocks) = ((m / 256) as u32, (n / 128) as u32, (k / K_TILE) as u32);

    let mut notes = Vec::new();
    for variant in [Variant::M256xN256, Variant::M512xN256] {
        let shape = Shape { m, n, k };
        let config = LaunchConfig1D::new(
            grid_for(shape, variant),
            THREADS,
            variant.shared_bytes() as u32,
        );
        for arm in Arm::DRAINS {
            let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;
            macro_rules! launch {
                ($prepare:ident, $call:ident) => {{
                    let prepared = ablated.$prepare(config)?;
                    unsafe {
                        ablated.$call(
                            &stream,
                            &prepared,
                            a_map.as_ptr(),
                            b_map.as_ptr(),
                            tiles_m,
                            tiles_n,
                            k_blocks,
                            n as u32,
                            &mut c,
                        )?
                    };
                }};
            }
            match (variant, arm) {
                (Variant::M256xN256, Arm::Whole) => {
                    launch!(prepare_gemm_sol_m256_whole, gemm_sol_m256_whole)
                }
                (Variant::M256xN256, Arm::Paired) => {
                    launch!(prepare_gemm_sol_m256_paired, gemm_sol_m256_paired)
                }
                (Variant::M256xN256, _) => {
                    launch!(prepare_gemm_sol_m256_wide, gemm_sol_m256_wide)
                }
                (Variant::M512xN256, Arm::Whole) => {
                    launch!(prepare_gemm_sol_m512_whole, gemm_sol_m512_whole)
                }
                (Variant::M512xN256, Arm::Paired) => {
                    launch!(prepare_gemm_sol_m512_paired, gemm_sol_m512_paired)
                }
                (Variant::M512xN256, _) => {
                    launch!(prepare_gemm_sol_m512_wide, gemm_sol_m512_wide)
                }
            }
            stream.synchronize()?;
            let worst = crate::gemm_sol::check_output(&c.to_host_vec(&stream)?, m, n, k)
                .map_err(|error| format!("{} {}: {error}", variant.name(), arm.name()))?;
            notes.push(format!(
                "{} {} exact over {} outputs, worst |rel| {worst:.2e}",
                variant.name(),
                arm.name(),
                m * n,
            ));
        }
    }
    Ok(notes.join("; "))
}

const PEAK: f64 = 2250.0;

/// Whole passes the drain comparison takes. Two, because the difference between
/// two drains is a few percent and one pass of it is not a number.
const PASSES: usize = 2;

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// The two shapes the PR has a gap at, with the entry each one ships.
const HEADLINE: [(Shape, Variant); 2] = [
    (
        Shape {
            m: 4096,
            n: 4096,
            k: 4096,
        },
        Variant::M256xN256,
    ),
    (
        Shape {
            m: 8192,
            n: 8192,
            k: 8192,
        },
        Variant::M512xN256,
    ),
];

/// `N` in tile steps at fixed `M` and `K`, so the cluster count sweeps across
/// the wave boundary and predicted efficiency runs sawtooth.
///
/// The rows are chosen so consecutive ones straddle a boundary in both
/// directions: at M256 and `M = K = 4096` a tile of `N` is 16 clusters, so 74
/// does not divide the step and the fill walks 0.76 → 0.86 → 0.97 → 0.82 →
/// 1.00. If the measured rate follows that curve rather than the work, the
/// 4096³ deficit is the schedule and not a cadence — and the strongest single
/// row is the one where **more** work finishes in the same time.
const SAWTOOTH_M256: [usize; 6] = [3584, 4096, 4608, 4864, 5120, 9472];
const SAWTOOTH_M512: [usize; 5] = [7168, 8192, 8704, 9216, 9472];

/// `K` at fixed `M` and `N`: the tile grid, the waves, the `C` traffic and the
/// epilogue's total cost all hold, and only the amount of arithmetic each tile
/// amortizes its fixed cost over moves. A rate that climbs with `K` here is a
/// per-tile cost — prologue, drain, CLC query — and not a steady-state cadence.
const DEPTHS: [usize; 4] = [2048, 4096, 8192, 16384];

fn quantization_table() {
    println!(
        "\nwave quantization, derived and not measured — one cluster per output tile,\n\
         {WAVE} clusters resident (148 SMs, one CTA each at either shared plan, two\n\
         CTAs a cluster). `ceiling` is the most of peak the shape can reach however\n\
         perfect the cadence is."
    );
    println!(
        "  {:<18}{:>10}{:>10}{:>8}{:>12}{:>14}",
        "shape", "variant", "clusters", "waves", "last wave", "ceiling"
    );
    for (shape, variant) in HEADLINE {
        let (clusters, waves, efficiency) = quantization(shape, variant);
        let last = clusters - WAVE * (waves - 1);
        println!(
            "  {:<18}{:>10}{:>10}{:>8}{:>11.0}%{:>13.3}",
            shape.to_string(),
            variant.name(),
            clusters,
            waves,
            100.0 * last as f64 / WAVE as f64,
            efficiency,
        );
    }
}

/// Bytes of `C` one cluster writes at one output tile — the drain's whole job,
/// and what makes the two variants' drains comparable per byte.
fn output_bytes(variant: Variant) -> f64 {
    (variant.m_tile() * N_TILE * 2) as f64
}

fn arm_table(
    context: &Arc<CudaContext>,
    shape: Shape,
    variant: Variant,
    arms: &[Arm],
) -> Result<Vec<f64>, Box<dyn Error>> {
    let (clusters, waves, ceiling) = quantization(shape, variant);
    println!(
        "\n  {shape} — {}, {clusters} clusters over {waves} waves, ceiling {ceiling:.3}",
        variant.name()
    );
    println!(
        "  {:<14}{:>11}{:>11}{:>12}{:>11}{:>11}  {}",
        "arm", "min ms", "median ms", "TFLOP/s", "of peak", "of whole", "what it runs"
    );
    let mut minima = Vec::with_capacity(arms.len());
    let mut whole = f64::NAN;
    for &arm in arms {
        let timings = measure(context, shape, variant, arm)?;
        if arm == Arm::Whole {
            whole = timings.min();
        }
        let rate = tflops(shape, timings.min());
        println!(
            "  {:<14}{:>11.4}{:>11.4}{:>12.1}{:>10.1}%{:>11.2}  {}",
            arm.name(),
            timings.min(),
            timings.median(),
            rate,
            100.0 * rate / PEAK,
            whole / timings.min(),
            arm.about(),
        );
        minima.push(timings.min());
    }
    Ok(minima)
}

fn ablation_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe ablation arms — one phase of the item removed per row, every barrier kept.\n\
         `of whole` is this arm's throughput over the whole kernel's, so a row near 1.00\n\
         is a phase that was already free and a row far above it is a phase that is not.\n\
         Three of the four compute no `C`; see this module's docs and the opcode census.\n\
         `of peak` is only meaningful on the `whole` row — an arm that skipped the\n\
         arithmetic still gets credited with it, which is why `feed only` reads above 100%.\n\
         The `shipped` row is the same device body compiled in `examples/` instead of here,\n\
         so the gap between it and `whole` is what the crate boundary is worth."
    );
    for (shape, variant) in HEADLINE {
        let arms: Vec<Arm> = std::iter::once(Arm::Shipped).chain(Arm::BUDGET).collect();
        arm_table(context, shape, variant, &arms)?;
    }
    Ok(())
}

/// The doubling ladder, and the two numbers `whole − no drain` cannot separate.
///
/// Each rung repeats one more pass of the drain per band, so a rung minus the
/// one below is that pass's own cost — and it is paid **serially**, because an
/// extra pass has nothing left to hide behind. That makes `twice all − whole` a
/// measurement of the drain's occupancy cost `D`, against which
/// `E = whole − no drain` is what the launch actually pays for it. `E ≈ D` is a
/// drain that fails to overlap at all; `E < D` is one partly hidden; `E > D` is
/// a drain that also slows the phase it runs beside.
fn chain_table(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe doubling ladder — one more pass of the drain per rung, aimed at the\n\
         cluster's own first tile so the extra stores stay in L2. `per tile` is the\n\
         launch divided by the tiles a critical-path cluster walks, and `Δ below` is\n\
         this rung's per-tile cost minus the previous rung's: the pass it added.\n\
         Every rung computes a wrong `C` and is on no correctness gate."
    );
    for (shape, variant) in HEADLINE {
        let (_, waves, _) = quantization(shape, variant);
        let per_tile = |milliseconds: f64| 1e3 * milliseconds / waves as f64;
        let minima = arm_table(context, shape, variant, &Arm::LADDER)?;
        let no_drain = measure(context, shape, variant, Arm::NoDrain)?.min();

        let (whole, twice_all) = (minima[0], minima[3]);
        let exposed = per_tile(whole - no_drain);
        let serial = per_tile(twice_all - whole);
        let halves = [
            ("ld.shared + st.global", per_tile(minima[1] - minima[0])),
            ("cvt + stmatrix", per_tile(minima[2] - minima[1])),
            ("tcgen05.ld and its wait", per_tile(minima[3] - minima[2])),
        ];
        println!(
            "\n  the drain at {shape}, in µs per tile per cluster ({:.0} B of `C`):",
            output_bytes(variant)
        );
        for (name, cost) in halves {
            println!(
                "  {:<26}{cost:>8.2}{:>9.0}%  of the serial drain",
                name,
                100.0 * cost / serial
            );
        }
        println!(
            "  {:<26}{serial:>8.2}           serial, `twice all − whole`",
            "the whole chain"
        );
        println!(
            "  {:<26}{exposed:>8.2}{:>9.0}%  of the serial drain — `whole − no drain`",
            "what the launch pays",
            100.0 * exposed / serial
        );
        println!(
            "  {:<26}{:>8.2}{:>9.0}%  of the serial drain",
            "hidden behind the MMA",
            serial - exposed,
            100.0 * (serial - exposed) / serial
        );
    }
    Ok(())
}

/// The drains, against one another and against the same `no drain` control.
///
/// All three declare the same shared plan and differ only in how many LDTM
/// issues a band puts in flight before its one wait, so the difference is the
/// wait structure and the band's register liveness and nothing else.
fn drain_table(context: &Arc<CudaContext>) -> Result<Arm, Box<dyn Error>> {
    println!(
        "\nthe drains — the same epilogue with the LDTM waits batched. One `no drain`\n\
         control serves all three: the shared plan, every barrier and every store is\n\
         identical, and only the drain's issue structure moves. `epilogue` is\n\
         `arm − no drain` in µs per tile per cluster, which is the term this whole\n\
         file exists to shrink."
    );
    let baseline = crate::bench::CUBLASLT_F16;
    let mut against_whole = [1.0f64; Arm::DRAINS.len()];
    for (shape, variant) in HEADLINE {
        let (_, waves, _) = quantization(shape, variant);
        let theirs = match baseline {
            Some(baseline) => Some((baseline.bench)(context, shape)?.0.min()),
            None => None,
        };
        // Two whole passes, the arms round-robin inside each. #122's lesson and
        // #146's: a few percent between two launches of the same kernel is not
        // quotable from one pass, and the range between the passes is the only
        // honest error bar this harness has.
        for pass in 1..=PASSES {
            let minima = arm_table(context, shape, variant, &Arm::DRAINS)?;
            let no_drain = measure(context, shape, variant, Arm::NoDrain)?.min();
            for (product, &min) in against_whole.iter_mut().zip(minima.iter()) {
                *product *= (minima[0] / min).powf(1.0 / PASSES as f64);
            }
            println!(
                "\n  pass {pass} of {PASSES}\n  {:<14}{:>11}{:>12}{:>13}{:>13}{:>14}",
                "drain", "min ms", "epilogue", "of launch", "vs whole", "vs cuBLASLt"
            );
            for (&arm, &min) in Arm::DRAINS.iter().zip(minima.iter()) {
                let epilogue = 1e3 * (min - no_drain) / waves as f64;
                println!(
                    "  {:<14}{min:>11.4}{epilogue:>10.2} µs{:>12.1}%{:>13.3}{:>14}",
                    arm.name(),
                    100.0 * (min - no_drain) / min,
                    minima[0] / min,
                    match theirs {
                        Some(theirs) => format!("{:.3}", theirs / min),
                        None => "—".to_string(),
                    },
                );
            }
            println!(
                "  {:<14}{no_drain:>11.4}       the control every row above subtracts",
                "no drain"
            );
        }
    }
    let best = Arm::DRAINS
        .iter()
        .zip(against_whole.iter())
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(&arm, _)| arm)
        .unwrap_or(Arm::Whole);
    println!(
        "\n  fastest over both shapes: `{}`, at {:.3} against `whole` on the product of\n\
         the two ratios. The depth ladder below is run on it as well as on `whole`, so\n\
         the per-tile constant can be read off both.",
        best.name(),
        against_whole[Arm::DRAINS.iter().position(|&arm| arm == best).unwrap()],
    );
    Ok(best)
}

fn sawtooth(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe wave sawtooth — `N` in tile steps, `M` and `K` held. `predicted` is the\n\
         derived ceiling above; `corrected` is the measured rate divided by it, which is\n\
         what the kernel does per resident cluster and should be flat if quantization is\n\
         the whole of the shape dependence."
    );
    for (base, columns, variant) in [
        (
            Shape {
                m: 4096,
                n: 0,
                k: 4096,
            },
            &SAWTOOTH_M256[..],
            Variant::M256xN256,
        ),
        (
            Shape {
                m: 8192,
                n: 0,
                k: 8192,
            },
            &SAWTOOTH_M512[..],
            Variant::M512xN256,
        ),
    ] {
        println!("\n  {} at m={}, k={}", variant.name(), base.m, base.k);
        println!(
            "  {:<18}{:>10}{:>7}{:>12}{:>11}{:>12}{:>12}",
            "shape", "clusters", "waves", "predicted", "min ms", "TFLOP/s", "corrected"
        );
        for &n in columns {
            let shape = Shape { n, ..base };
            let (clusters, waves, predicted) = quantization(shape, variant);
            let timings = measure(context, shape, variant, Arm::Whole)?;
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<18}{:>10}{:>7}{:>12.3}{:>11.4}{:>12.1}{:>12.1}",
                shape.to_string(),
                clusters,
                waves,
                predicted,
                timings.min(),
                rate,
                rate / predicted,
            );
        }
    }
    Ok(())
}

fn depth(context: &Arc<CudaContext>, arm: Arm) -> Result<(), Box<dyn Error>> {
    println!(
        "\nthe depth ladder on `{}` — `K` alone, so the tile grid and the waves are fixed\n\
         and only the arithmetic each tile amortizes over moves. `per tile` is the launch's\n\
         milliseconds divided by the items a critical-path cluster walks, and a straight\n\
         line through it in `k blocks` reads off the per-tile constant directly.",
        arm.name()
    );
    for (base, variant) in HEADLINE {
        println!("\n  {} at m={}, n={}", variant.name(), base.m, base.n);
        println!(
            "  {:<18}{:>9}{:>10}{:>11}{:>12}{:>11}{:>13}",
            "shape", "k blocks", "waves", "min ms", "TFLOP/s", "of peak", "ms per tile"
        );
        for k in DEPTHS {
            let shape = Shape { k, ..base };
            let (_, waves, _) = quantization(shape, variant);
            let timings = measure(context, shape, variant, arm)?;
            let rate = tflops(shape, timings.min());
            println!(
                "  {:<18}{:>9}{:>10}{:>11.4}{:>12.1}{:>10.1}%{:>13.5}",
                shape.to_string(),
                k / K_TILE,
                waves,
                timings.min(),
                rate,
                100.0 * rate / PEAK,
                timings.min() / waves as f64,
            );
        }
    }
    Ok(())
}

fn denominator(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let Some(baseline) = crate::bench::CUBLASLT_F16 else {
        println!("\nno cuBLASLt column: this build has no `cublas` feature.");
        return Ok(());
    };
    println!(
        "\nthe denominator, in this container — {} ({}). `of ceiling` is our ratio\n\
         divided by the derived wave-quantization ceiling: what would be left to explain\n\
         if the schedule were the only loss.",
        baseline.name,
        (baseline.about)()
    );
    println!(
        "  {:<18}{:>12}{:>13}{:>13}{:>13}",
        "shape", "ours TF/s", "theirs TF/s", "ours/theirs", "of ceiling"
    );
    for (shape, variant) in HEADLINE {
        let (_, _, ceiling) = quantization(shape, variant);
        let ours = measure(context, shape, variant, Arm::Shipped)?;
        let (theirs, _) = (baseline.bench)(context, shape)?;
        let ratio = theirs.min() / ours.min();
        println!(
            "  {:<18}{:>12.1}{:>13.1}{:>13.3}{:>13.3}",
            shape.to_string(),
            tflops(shape, ours.min()),
            tflops(shape, theirs.min()),
            ratio,
            ratio / ceiling,
        );
    }
    Ok(())
}

/// Everything above, in one container — `bench sol`.
pub fn decompose(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    quantization_table();
    denominator(context)?;
    ablation_table(context)?;
    chain_table(context)?;
    let best = drain_table(context)?;
    sawtooth(context)?;
    depth(context, Arm::Whole)?;
    if best != Arm::Whole {
        depth(context, best)?;
    }
    println!(
        "\nmost arms here are checked against no CPU reference and most cannot be.\n\
         `modal_app.py::regcount`'s opcode census is what says an arm removed — or\n\
         doubled — the phase it names; a row whose census disagrees is void and not a\n\
         finding. The `paired` and `wide` drains are the exception: they compute the\n\
         same `C` the shipped kernel does, and `examples`' exact BF16 gate covers them\n\
         through the same device body."
    );
    Ok(())
}
