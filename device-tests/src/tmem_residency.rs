//! How many tcgen05 CTAs an SM *actually* holds at once — counted, not
//! queried, and not inferred from a throughput curve (#78).
//!
//! #77 established with a good control that
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor` returns **1** for any kernel
//! containing a `tcgen05.alloc`: at every legal column count, at every block
//! width, against 32/32/16/8 for the byte-identical kernel with the allocator
//! deleted. That is a fact about the query and nothing here contradicts it.
//!
//! Whether the query's 1 *describes hardware residency* is a different claim,
//! and #51 is the first evidence either way. Capping `gemm`'s grid at one
//! cluster per SM cost **2.07×** at 8192³ — impossible if the SM could hold
//! only one, because then the surplus would simply queue and the cap would be
//! free. So the query has been shown once not to predict residency, and
//! `flash_forward`'s 1 stops being established.
//!
//! # The instrument
//!
//! A throughput sweep answers this indirectly: cap the grid at 1, 2, 3 … CTAs
//! an SM and see where the curve flattens. It needs the work to be
//! latency-bound for co-residency to buy anything, it needs a calibration run
//! to know the detector is not blind, and — the part no sweep can fix — a flat
//! curve cannot distinguish *not resident* from *resident and parked inside a
//! blocking `tcgen05.alloc`*.
//!
//! Blackwell hands out a better instrument. `%smid` names the SM a CTA is
//! running on and `%globaltimer` is a device-wide nanosecond clock, so a CTA
//! can simply **write down where it ran and when**. Each census CTA:
//!
//! 1. reads `%smid` and timestamps `entered`,
//! 2. allocates `columns` of tensor memory and timestamps `allocated`,
//! 3. spins on the clock until `hold_ns` have passed, timestamping `left`,
//! 4. gives the columns back and publishes the four numbers.
//!
//! The host then sweeps those intervals per SM and takes the maximum number
//! that were open at once. That is residency, counted directly. Two intervals
//! are counted, and the pair is the point:
//!
//! | interval | what its overlap counts |
//! | --- | --- |
//! | `[entered, left]` | CTAs the SM held a slot for, whatever they were doing |
//! | `[allocated, left]` | CTAs holding tensor memory at the same instant |
//!
//! A CTA parked in the allocator waiting for a peer to release columns has a
//! long first interval and a short second one. `resident` above `holding` is
//! therefore *serialization*, and `resident` at 1 is a genuine admission
//! ceiling. Nothing else in this repo separates those.
//!
//! The hold is a *wall-clock* spin rather than a fixed amount of work, so a
//! CTA occupies its SM for the same interval no matter how many co-residents
//! it is competing with. Overlap then counts residency and not contention.
//!
//! # Why the measurement is a lower bound, and why that is the safe direction
//!
//! `entered` is read after the CTA is already scheduled and `left` before it
//! deallocates and exits, so the recorded window sits strictly inside the true
//! residency window. The census can therefore undercount at a wave boundary
//! and can never overcount. Every number below is a floor on what the hardware
//! did, which is the direction a claim of "residency is at least N" needs.
//! With [`HOLD_NS`] of hold against sub-microsecond prologues, the gap is
//! noise.
//!
//! # The rungs, and what each one is for
//!
//! The column sweep is the arithmetic hypothesis: an SM has 512 columns of
//! tensor memory, so if the allocator is what binds, `512 / columns` CTAs fit —
//! 32 columns admits 16, 128 admits 4 (`gemm`'s allocation), 256 admits 2
//! (`flash_forward`'s), and 512 admits one. That last rung is the census's own
//! **positive control**: one CTA an SM is arithmetically forced there, so a
//! detector that cannot read a 1 at 512 columns cannot be believed when it
//! reads a 1 anywhere else.
//!
//! [`kernels::residency_census_free`](crate::kernels::residency_census_free)
//! takes all 512 columns and gives them straight back *before* the interval
//! that gets counted. The kernel still contains a `tcgen05.alloc`, so the query
//! still answers 1 for it. If the 1 were a static reservation made when the CTA
//! is admitted, that rung would be pinned at 1 like the 512 rung; if it is the
//! dynamic cost of *holding* columns, it is free like the control. A column
//! sweep alone cannot tell those apart.
//!
//! And three `cta_group::2` rungs, which #77's control never covered. #78 names
//! the leading hypothesis for why `gemm` and `flash_forward` might genuinely
//! differ: that `alloc_cluster` splits the SM's tensor memory across the pair
//! rather than each rank claiming all of it. The pair's two CTAs land on two
//! different SMs, so counting overlap per `%smid` needs no special case — only
//! a grid that is a multiple of the cluster size. **It is refuted**: a rank is
//! charged its full column count against its own SM, so `alloc_cluster` at 128
//! columns admits four CTAs an SM exactly as `alloc_block` at 128 does.
//!
//! # Tensor memory is one of two resources, and for these kernels the loose one
//!
//! The rungs above vary columns against a token shared plan, which isolates the
//! allocator and answers the question #78 asks. It is not by itself an answer
//! about any real kernel, because shared memory is the *other* per-CTA resource
//! an SM divides and both tcgen05 kernels in this repo carry a large plan. The
//! last four rungs put the two together, and residency comes out at
//! `min(512 / columns, shared per SM / plan)` — exactly, at every rung.
//!
//! That is what settles #78, and it settles it in two directions at once:
//!
//! - **`gemm`'s envelope counts 3.** #83 bisected that kernel's grid cap on a
//!   clock and got 3. Two instruments with nothing in common — a throughput
//!   curve on a real GEMM, and timestamps from a nine-register probe — agreeing
//!   on one integer is worth more than either alone. Its columns allow 4 and its
//!   shared plan allows 3.
//! - **`flash_forward`'s envelope counts 1, so its 1 CTA/SM is real** — and
//!   `tcgen05` is not the cause. Its columns allow 2; its 147536 B plan allows
//!   1 with no allocator in the kernel at all. #70 tried to price that plan by
//!   querying occupancy at four sizes, got 1 every time, and concluded shared
//!   memory was not the lever. Every one of those four answers was the allocator
//!   pinning the query. It is the only lever that kernel has.
//!
//! # And four rungs that locate the shared-memory step itself (#85)
//!
//! Every rung above declares an envelope some kernel has. [`THRESHOLD_PLANS`]
//! declares four nobody has, 1 KiB apart around half of the SM's 233472 B,
//! because the number #85 has to design `flash_forward`'s ring against is the
//! plan at which a second CTA stops being admitted — and what this census had
//! established about it was only that it lies between 73768 B (counts 2) and
//! 147536 B (counts 1). [`located_step`] narrows that to an interval and
//! prints it. Those rungs are the one place here where a count is reported
//! rather than asserted, since a step that is not where the arithmetic puts it
//! is the finding.
//!
//! # And three rungs that exist to be a control for something else (#98)
//!
//! `gemm`'s plan and the same plan with 8192 and 49152 dead bytes on the end
//! count **3, 2 and 1**. Nothing in this file needs those numbers; they are
//! here because `experiments/README.md` prices what an occupancy step costs that
//! kernel's *throughput*, by declaring exactly those envelopes and touching
//! none of the added bytes. That experiment's whole claim is that the only
//! thing it moved was residency, and a claim about residency in this repo is
//! counted rather than queried.

use std::error::Error;
use std::fmt::Write as _;

use cuda_core::{CudaFunction, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_host::CudaKernel;

use kittens::launch::admit_shared_plan;

use crate::{CENSUS_FIELDS, kernels};
use kittens::watchdog::ReadBack;

/// Threads a census CTA launches with. One warp is all the allocator needs and
/// all the probe does, so blocks per SM *is* warps per SM and no register or
/// warp-slot ceiling can be confused for the allocator's.
const CENSUS_THREADS: u32 = 32;

/// The census CTA's whole dynamic shared plan: the TMEM staging word, padded
/// to the alignment `DynamicSharedArray::<u8, 128>` asks for. Well under the
/// 48 KiB opt-in threshold, so #70's persistent-mutation trap has nothing to
/// act on here.
const CENSUS_SHARED: u32 = 32;

/// CTAs launched per SM. 32 is Blackwell's own resident-CTA ceiling, so the
/// control has room to show its real number rather than the grid's, and every
/// allocating rung has 32× the headroom it could possibly use.
const CTAS_PER_SM: u32 = 32;

/// Nanoseconds each CTA holds its SM. Four orders of magnitude above the
/// clock's granularity and above any prologue, so the intervals are long
/// enough that overlap is unambiguous — and short enough that the worst rung
/// (one CTA an SM, so 32 waves) is a few milliseconds.
const HOLD_NS: u64 = 100_000;

/// `flash_forward`'s dynamic shared plan, spelled here rather than imported
/// because `device-tests` does not depend on the examples crate.
///
/// Tensor memory is not the only per-CTA resource an SM divides, and this is
/// the size of the other one for the kernel #78 is about. #70 tried to rule
/// shared memory out for `flash_forward` and got 1 block/SM at 147536 B, at
/// 73792, at 32800 and at zero — but every one of those answers came from the
/// occupancy query, which this file shows is pinned at 1 by the mere presence
/// of a `tcgen05.alloc` regardless of any plan. So shared memory was never
/// actually excluded, and the rungs that carry this number are what excludes it
/// or does not.
const FLASH_SHARED: u32 = 147_536;

/// `gemm`'s dynamic shared plan, spelled here for the same reason.
///
/// #83 bisected that kernel's grid cap on a clock and found its residency to be
/// **3 CTAs an SM**. Its 128 columns admit four, so if 3 is right then shared
/// memory is the tighter of the two and this is the number that sets it. The
/// rung carrying it cross-checks #83's result on an instrument that has nothing
/// in common with a throughput curve — different kernel, different signal,
/// same envelope.
const GEMM_SHARED: u32 = 73_792;

/// `gemm_cg2`'s plan as it stands today — [`GEMM_SHARED`] plus the 24 bytes the
/// CLC work queue added, which every scheduler now pays whether it reads the
/// queue or not.
///
/// Carried separately because 73792 and 73816 are on the same side of the same
/// step and it is worth saying so with a count rather than with arithmetic:
/// three of either fit in an SM's 233472 B.
const GEMM_PLAN: u32 = 73_816;

/// [`GEMM_PLAN`] with a dead 8192 B on the end — #98's 3 → 2 step.
///
/// Nothing reads these bytes. They exist to move residency and nothing else,
/// which is what makes the throughput either side of them a price for the
/// occupancy step rather than for an epilogue. 4033 B would cross the step;
/// 8192 clears it with room to spare, and since the bytes are never touched
/// there is no cost to overshooting.
const GEMM_PLAN_TWO: u32 = GEMM_PLAN + 8_192;

/// The same, sized to cross the 2 → 1 step. A linear cost and a cliff are
/// different answers, so both steps get counted.
const GEMM_PLAN_ONE: u32 = GEMM_PLAN + 49_152;

/// #87's four built rungs, as the shared plans their launches declare.
///
/// Spelled here rather than imported for the same reason [`GEMM_SHARED`] is —
/// `device-tests` does not depend on the examples crate — and they are the
/// arithmetic `experiments/src/gemm.rs`'s `shared_plan` states: two operand rings
/// `stages` deep over a `[128, 64]` `A` tile and a `[block_n / 2, 64]` `B`
/// tile, bf16, plus the barrier and work-queue tail.
///
/// The last two are why this file gained a 256-column cluster rung. 65 608 B
/// admits three CTAs an SM and 256 accumulator columns admit two, so that is
/// the first envelope here where **tensor memory** and not shared memory is
/// what caps a kernel.
const TILE_N128_S2: u32 = 49_224;
const TILE_N128_S4: u32 = 98_408;
const TILE_N256_S2: u32 = 65_608;
const TILE_N256_S3: u32 = 98_392;

/// The shipped rung with a **staged epilogue** on it (#15): [`TILE_N256_S3`]
/// rounded up to a 128-byte tile base, plus one `[32, 64]` bf16 staging tile
/// per warp — `98 392 → 98 432 + 4 · 4096`.
///
/// It is here because the staged epilogue's whole case rests on the claim that
/// those 16 424 bytes are free, and "free" is an integer this file counts
/// rather than a `min` this file computes. 512 / 256 columns says two and
/// 233 472 / 114 816 says two, so the prediction is that nothing moves; the
/// row is worth running precisely because a step down would make the A/B a
/// residency comparison instead of an epilogue one.
const TILE_N256_S3_STAGED: u32 = 114_816;

/// `gemm_ws`'s shipped plan — two operand rings four stages deep over the same
/// `[128, 64]` `A` and `[128, 64]` `B` tiles, plus the barrier and work-queue
/// tail. Spelled here for [`GEMM_SHARED`]'s reason.
///
/// The rung carrying it is the one envelope in this file where **tensor
/// memory alone** decides: that kernel holds two 256-column accumulator stages,
/// which is all 512 columns an SM has, so `512 / 512 = 1` before shared memory
/// is consulted at all. The `cg2 512` rung above counts 1 holding and **2
/// resident** at a 32 B plan — the second CTA admitted to the SM and parked
/// inside a blocking `tcgen05.alloc`. At 131 176 B the second CTA cannot be
/// admitted, so the prediction that separates these two rows is that resident
/// falls to 1 as well.
const WS_SHARED: u32 = 131_176;

/// [`WS_SHARED`] with `gemm_ws`'s four per-warp staging tiles on it (#116),
/// which is the envelope every epilogue rung of that kernel declares — #117's
/// two instruction widths move no byte of the plan.
///
/// Its whole A/B rests on those 16 408 bytes being free, and free is an integer
/// this file counts rather than a `min` this file computes.
const WS_SHARED_STAGED: u32 = 147_584;

/// Shared plans that bracket the **two-CTAs-an-SM step** (#85), 1 KiB apart
/// around half of the SM's 233472 B.
///
/// #85 needs a plan under 116736 B for `flash_forward` to reach two CTAs an SM,
/// and 116736 is arithmetic: half the SM, from a census that only ever
/// bracketed the step between 73768 B (counts 2) and 147536 B (counts 1). Two
/// things could move it and neither is visible from the host — the driver
/// reserves a slice of an SM's shared memory per CTA, and it rounds a plan up
/// to an allocation granularity — and both push the step *down*. So the ladder
/// is three rungs at and under the arithmetic value and one above it, which is
/// the positive control: the step has to be somewhere, and a ladder that counts
/// 2 at every rung has located nothing.
///
/// They carry no allocator. Tensor memory would admit 2 at 256 columns and 4 at
/// 128, so an allocating rung here would be counting the `min` of two terms at
/// the exact point where one of them is what is being located.
const THRESHOLD_PLANS: [(&str, u32); 4] = [
    ("shared half - 2 KiB", HALF_SHARED_PER_SM - 2048),
    ("shared half - 1 KiB", HALF_SHARED_PER_SM - 1024),
    ("shared half (#85)", HALF_SHARED_PER_SM),
    ("shared half + 1 KiB", HALF_SHARED_PER_SM + 1024),
];

/// Half of a B200's 233472 B of shared memory an SM, which is where #85 puts
/// the two-CTA threshold by arithmetic. Written down rather than derived from
/// the device's own attribute because [`THRESHOLD_PLANS`] has to be a *fixed*
/// ladder — a rung that moved with the device would not be the number #85
/// designs against — and [`located_step`] says so when the device disagrees.
const HALF_SHARED_PER_SM: u32 = 116_736;

/// How far the achieved hold may sit under [`HOLD_NS`] before the harness
/// calls the spin broken rather than the residency low. A CTA that exited on
/// [`CENSUS_SPIN_GUARD`](crate::CENSUS_SPIN_GUARD) instead of on the clock
/// would report a short interval, and short intervals overlap less — which
/// would look exactly like low residency. This is the check that stops a
/// hoisted `%globaltimer` from being read as a finding.
const HOLD_FLOOR: f64 = 0.9;

/// One rung: what it is called, how many columns it holds, and whether it is a
/// `cta_group::2` launch.
struct Rung {
    name: &'static str,
    /// The PTX entry point, so the driver can be asked about the *same*
    /// function the census counts. Taken from the `CudaKernel` marker
    /// `#[kernel]` generates rather than from a `format!`, so a kernel renamed
    /// on the device side stops compiling here instead of failing to load on a
    /// GPU that costs money.
    entry: &'static str,
    columns: Option<u32>,
    ranks: u32,
    /// Whether the columns are held across the counted interval. False for the
    /// allocate-and-release rung, whose whole point is that they are not.
    holds: bool,
    /// Dynamic shared memory the launch declares. The census kernel writes four
    /// bytes of it whatever this says; the number is here because shared memory
    /// is the *other* per-CTA resource an SM divides, and a rung can be used to
    /// ask which of the two binds first.
    shared: u32,
    /// Whether this rung is here to *locate* the shared-memory step
    /// ([`THRESHOLD_PLANS`]) rather than to confirm the model at an envelope
    /// some kernel declares. What it counts is the finding, so it is reported
    /// and not asserted against: [`Measured::budget`] is `None` for these.
    bisects: bool,
}

/// What one rung's census came to.
struct Measured {
    name: &'static str,
    columns: Option<u32>,
    ranks: u32,
    holds: bool,
    shared: u32,
    bisects: bool,
    /// CTAs an SM the driver predicts for this same loaded function.
    predicted: f64,
    /// SMs that ran at least one CTA of this rung.
    sms: usize,
    /// Peak CTAs on one SM whose `[entered, left]` windows were open at once.
    resident: usize,
    /// Peak CTAs on one SM holding tensor memory at once.
    holding: usize,
    /// Longest any CTA sat between `entered` and `allocated` — time inside a
    /// blocking allocator, in microseconds.
    worst_wait_us: f64,
    /// Shortest hold any CTA achieved against the [`HOLD_NS`] it asked for.
    shortest_hold_us: f64,
}

impl Measured {
    /// CTAs this rung's *per-CTA resources* admit on one SM: the SM's 512
    /// columns of tensor memory divided by the allocation, and its shared
    /// memory divided by the plan, whichever is tighter.
    ///
    /// `None` where neither resource is doing anything — a rung holding no
    /// columns and declaring a token shared plan is bounded only by warp slots
    /// and by the grid, which are not what this file is about and not stable
    /// enough to assert — and `None` for a rung that bisects the step, whose
    /// whole job is to say where this arithmetic stops holding.
    fn budget(&self, shared_per_sm: u32) -> Option<usize> {
        if self.bisects {
            return None;
        }
        let tmem = match (self.holds, self.columns) {
            (true, Some(columns)) => Some(512 / columns as usize),
            _ => None,
        };
        // A token staging word is not a shared-memory constraint; anything at
        // the >48 KiB opt-in scale is.
        let shared = (self.shared > 48 * 1024).then(|| (shared_per_sm / self.shared) as usize);
        match (tmem, shared) {
            (Some(tmem), Some(shared)) => Some(tmem.min(shared)),
            (only @ Some(_), None) | (None, only @ Some(_)) => only,
            (None, None) => None,
        }
    }
}

fn rungs() -> Vec<Rung> {
    macro_rules! rung {
        ($name:literal, $kernel:ident, $columns:expr, $ranks:literal, $holds:expr) => {
            rung!($name, $kernel, $columns, $ranks, $holds, CENSUS_SHARED)
        };
        ($name:literal, $kernel:ident, $columns:expr, $ranks:literal, $holds:expr, $shared:expr) => {
            Rung {
                name: $name,
                entry: <kernels::${concat(__, $kernel, _CudaKernel)} as CudaKernel>::PTX_NAME,
                columns: $columns,
                ranks: $ranks,
                holds: $holds,
                shared: $shared,
                bisects: false,
            }
        };
    }
    vec![
        rung!("no tcgen05", residency_census_none, None, 1, false),
        rung!("alloc 32", residency_census_32, Some(32), 1, true),
        rung!("alloc 64", residency_census_64, Some(64), 1, true),
        rung!("alloc 128 (gemm)", residency_census_128, Some(128), 1, true),
        rung!(
            "alloc 256 (flash)",
            residency_census_256,
            Some(256),
            1,
            true
        ),
        rung!("alloc 512", residency_census_512, Some(512), 1, true),
        rung!(
            "alloc 512, released",
            residency_census_free,
            Some(512),
            1,
            false
        ),
        rung!("cg2 none", residency_census_cluster_none, None, 2, false),
        rung!(
            "cg2 128 (gemm)",
            residency_census_cluster_128,
            Some(128),
            2,
            true
        ),
        rung!("cg2 512", residency_census_cluster_512, Some(512), 2, true),
        // flash_forward's own envelope, which is the question #78 actually
        // asks. Its 256 columns admit two CTAs an SM by the arithmetic above,
        // but tensor memory is not the only per-CTA resource an SM divides and
        // `flash_forward` carries FLASH_SHARED bytes of the other one. These
        // three separate the two ceilings: a control at flash's shared plan
        // with no allocator at all, the allocation at half that plan where
        // tensor memory should still be the tighter of the two, and both
        // together at the real thing.
        rung!(
            "flash shared, no tmem",
            residency_census_none,
            None,
            1,
            false,
            FLASH_SHARED
        ),
        rung!(
            "alloc 256 + half",
            residency_census_256,
            Some(256),
            1,
            true,
            FLASH_SHARED / 2
        ),
        rung!(
            "alloc 256 + flash shared",
            residency_census_256,
            Some(256),
            1,
            true,
            FLASH_SHARED
        ),
        // gemm's own envelope, against #83's clock: cta_group::2, 128 columns,
        // 73792 B. Tensor memory admits four there and shared memory three, so
        // this rung predicts 3 — which is the number #83 bisected a grid cap to
        // find. Two instruments with nothing in common agreeing on one integer
        // is worth more than either alone.
        rung!(
            "gemm envelope (cg2)",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            GEMM_SHARED
        ),
        // #98: what one CTA an SM is worth to that kernel. The question is a
        // throughput one and it is asked in `experiments/README.md` by growing
        // `gemm_cg2`'s declared plan with bytes no code touches; these three
        // rungs are what says the growth moved residency, on the instrument
        // that counts rather than the query that is pinned at 1 (#77). The
        // dead bytes are a *launch parameter*, so there is nothing for a
        // compiler to eliminate and no work either side of them differs.
        rung!(
            "gemm plan today",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            GEMM_PLAN
        ),
        rung!(
            "gemm plan + 8192 dead",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            GEMM_PLAN_TWO
        ),
        rung!(
            "gemm plan + 49152 dead",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            GEMM_PLAN_ONE
        ),
        // #87's tile and depth sweep, at the envelopes its four built rungs
        // actually declare. The sweep prints a *predicted* residency from the
        // same `min(512 / columns, shared per SM / plan)` these rows count, and
        // a throughput table whose occupancy column was predicted rather than
        // counted is a table that cannot tell a tile effect from a residency
        // one.
        //
        // The last two are the reason this block exists. `[256, 256]` allocates
        // 256 columns, which admit **two** CTAs an SM — and at two stages its
        // 65 608 B plan admits three, so it is the first envelope in this repo
        // where tensor memory is the tighter resource rather than shared
        // memory. `budget` already computes the `min` of the two; this is the
        // first rung where the two arguments differ in that direction, so it is
        // also the first time that half of the formula is on trial.
        rung!(
            "#87 [256,128] s2",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            TILE_N128_S2
        ),
        rung!(
            "#87 [256,128] s4",
            residency_census_cluster_128,
            Some(128),
            2,
            true,
            TILE_N128_S4
        ),
        rung!(
            "#87 [256,256] s2",
            residency_census_cluster_256,
            Some(256),
            2,
            true,
            TILE_N256_S2
        ),
        // The plan `gemm_cg2` declares since #87 — the shipped envelope, and
        // the row that has to be right for the persistent grid to be sized
        // correctly. The three `gemm plan` rows above it are the envelope it
        // shipped through #102, kept because #98's occupancy prices are quoted
        // against them.
        rung!(
            "#87 [256,256] s3 (gemm)",
            residency_census_cluster_256,
            Some(256),
            2,
            true,
            TILE_N256_S3
        ),
        // #15's staged epilogue: the same rung carrying four per-warp staging
        // tiles. See `TILE_N256_S3_STAGED` — the prediction is that it counts
        // the same 2, and the A/B above it is only an epilogue comparison if
        // it does.
        rung!(
            "#15 [256,256] s3 staged",
            residency_census_cluster_256,
            Some(256),
            2,
            true,
            TILE_N256_S3_STAGED
        ),
        // `gemm_ws`'s own two envelopes (#112, #116, #117). The `cg2 512` rung
        // near the top of this list holds the same 512 columns at a token
        // shared plan and counts 1 holding against 2 resident; these two say
        // what the real plan does to the second of those, and they are the
        // only rows in this file where the tensor-memory term binds on its own.
        rung!(
            "ws envelope (cg2 512)",
            residency_census_cluster_512,
            Some(512),
            2,
            true,
            WS_SHARED
        ),
        rung!(
            "ws staged envelope",
            residency_census_cluster_512,
            Some(512),
            2,
            true,
            WS_SHARED_STAGED
        ),
    ]
    .into_iter()
    .chain(THRESHOLD_PLANS.map(|(name, shared)| Rung {
        name,
        entry: <kernels::__residency_census_none_CudaKernel as CudaKernel>::PTX_NAME,
        columns: None,
        ranks: 1,
        holds: false,
        shared,
        bisects: true,
    }))
    .collect()
}

/// What the *driver* predicts for this rung, in CTAs per SM, so the table can
/// put its answer beside the counted one for the identical loaded function.
///
/// A plain `#[kernel]` gets `cuOccupancyMaxActiveBlocksPerMultiprocessor`,
/// which is #77's query and already known to answer 1 for anything holding an
/// allocator. A `#[cluster_launch]` kernel gets **`max_active_clusters`**,
/// which is the query this repo has been saying it does not have: `main.rs`
/// prints `cluster` in the GEMM's occupancy row and #51 recorded that the
/// residency question "cannot currently be answered for this kernel". It can.
/// `cuOccupancyMaxActiveClusters` takes the cluster shape the block query has
/// no argument for, and it reports clusters *device-wide*, so it is divided
/// back out into CTAs an SM here to sit in the same column as everything else.
fn predicted(
    function: &CudaFunction,
    rung: &Rung,
    blocks: u32,
    sms: u32,
) -> Result<f64, Box<dyn Error>> {
    let block_dim = (CENSUS_THREADS, 1, 1);
    if rung.ranks == 1 {
        let blocks_per_sm =
            function.max_active_blocks_per_multiprocessor(CENSUS_THREADS, rung.shared)?;
        return Ok(blocks_per_sm as f64);
    }
    let clusters =
        function.max_active_clusters((blocks, 1, 1), block_dim, rung.shared, (rung.ranks, 1, 1))?;
    Ok((clusters * rung.ranks) as f64 / sms as f64)
}

/// Launch one rung over the whole device and hand back its raw census rows.
fn census(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    rung: &Rung,
    blocks: u32,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut out = DeviceBuffer::<u64>::zeroed(stream, blocks as usize * CENSUS_FIELDS)?;
    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (CENSUS_THREADS, 1, 1),
        shared_mem_bytes: rung.shared,
    };
    // Safety: every entry point below is launched at the one warp its own doc
    // names, and each CTA writes only the `CENSUS_FIELDS` words at its own
    // `blockIdx.x`, so no two blocks of the grid touch the same element. The
    // census body writes four bytes of the shared plan whatever the plan's
    // declared size, and the plans above 48 KiB have been admitted by the
    // caller. The `cg2` rungs carry `#[cluster_launch(2, 1, 1)]`, which routes
    // their generated launcher through `cuLaunchKernelEx` — the grid is a
    // multiple of two, as that path requires.
    unsafe {
        match (rung.ranks, rung.columns, rung.holds) {
            (1, None, _) => module.residency_census_none(stream, config, HOLD_NS, &mut out)?,
            (1, Some(32), true) => module.residency_census_32(stream, config, HOLD_NS, &mut out)?,
            (1, Some(64), true) => module.residency_census_64(stream, config, HOLD_NS, &mut out)?,
            (1, Some(128), true) => {
                module.residency_census_128(stream, config, HOLD_NS, &mut out)?
            }
            (1, Some(256), true) => {
                module.residency_census_256(stream, config, HOLD_NS, &mut out)?
            }
            (1, Some(512), true) => {
                module.residency_census_512(stream, config, HOLD_NS, &mut out)?
            }
            (1, Some(512), false) => {
                module.residency_census_free(stream, config, HOLD_NS, &mut out)?
            }
            (2, None, _) => {
                module.residency_census_cluster_none(stream, config, HOLD_NS, &mut out)?
            }
            (2, Some(128), true) => {
                module.residency_census_cluster_128(stream, config, HOLD_NS, &mut out)?
            }
            (2, Some(256), true) => {
                module.residency_census_cluster_256(stream, config, HOLD_NS, &mut out)?
            }
            (2, Some(512), true) => {
                module.residency_census_cluster_512(stream, config, HOLD_NS, &mut out)?
            }
            (ranks, columns, holds) => {
                return Err(format!(
                    "no census kernel for {ranks} rank(s) at {columns:?} columns, holds={holds}"
                )
                .into());
            }
        }
    }
    Ok(out.read_back(stream)?)
}

/// One device attribute, by the same raw-`sys` route `ladder_bench` uses.
fn device_attribute(
    context: &cuda_core::CudaContext,
    attribute: cuda_core::sys::CUdevice_attribute,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    let mut value: i32 = 0;
    let status =
        unsafe { cuda_core::sys::cuDeviceGetAttribute(&mut value, attribute, context.cu_device()) };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuDeviceGetAttribute({name}) failed with driver status {status}").into(),
        );
    }
    Ok(value as u32)
}

/// Peak number of simultaneously open intervals, by a sweep over their
/// endpoints.
///
/// Closes sort before opens at an equal timestamp, so two intervals that merely
/// touch — one CTA's slot handed straight to the next — are not counted as
/// overlapping. That is the conservative direction, and it is the same
/// direction the probe's own timestamps already err in.
fn peak_overlap(intervals: &[(u64, u64)]) -> usize {
    let mut events: Vec<(u64, i32)> = Vec::with_capacity(2 * intervals.len());
    for &(start, end) in intervals {
        events.push((start, 1));
        events.push((end, -1));
    }
    events.sort_unstable_by_key(|&(time, delta)| (time, -delta));
    let mut open = 0i32;
    let mut peak = 0i32;
    for (_, delta) in events {
        open += delta;
        peak = peak.max(open);
    }
    peak as usize
}

/// Reduce one rung's raw rows into the row the table prints.
fn tally(rung: &Rung, predicted: f64, rows: &[u64]) -> Result<Measured, Box<dyn Error>> {
    // `%smid` is bounded by the device's SM count, so a plain vector indexed by
    // it is the whole grouping structure this needs.
    let mut by_sm: Vec<Vec<(u64, u64, u64)>> = Vec::new();
    let mut worst_wait = 0u64;
    let mut shortest_hold = u64::MAX;
    for row in rows.chunks_exact(CENSUS_FIELDS) {
        let (sm, entered, allocated, left) = (row[0] as usize, row[1], row[2], row[3]);
        if left < allocated || allocated < entered {
            return Err(format!(
                "{}: SM {sm} reported times out of order ({entered}, {allocated}, {left}) — \
                 %globaltimer is not monotonic across this launch and no overlap computed from \
                 it means anything",
                rung.name
            )
            .into());
        }
        if by_sm.len() <= sm {
            by_sm.resize(sm + 1, Vec::new());
        }
        by_sm[sm].push((entered, allocated, left));
        worst_wait = worst_wait.max(allocated - entered);
        shortest_hold = shortest_hold.min(left - allocated);
    }

    let mut resident = 0usize;
    let mut holding = 0usize;
    let mut sms = 0usize;
    for windows in &by_sm {
        if windows.is_empty() {
            continue;
        }
        sms += 1;
        let lifetimes: Vec<(u64, u64)> = windows.iter().map(|&(e, _, l)| (e, l)).collect();
        let holds: Vec<(u64, u64)> = windows.iter().map(|&(_, a, l)| (a, l)).collect();
        resident = resident.max(peak_overlap(&lifetimes));
        holding = holding.max(peak_overlap(&holds));
    }

    Ok(Measured {
        name: rung.name,
        columns: rung.columns,
        ranks: rung.ranks,
        holds: rung.holds,
        shared: rung.shared,
        bisects: rung.bisects,
        predicted,
        sms,
        resident,
        holding,
        worst_wait_us: worst_wait as f64 / 1e3,
        shortest_hold_us: shortest_hold as f64 / 1e3,
    })
}

/// The claims this case holds.
///
/// The measured shape is exact — concurrent holders came out at **`512 /
/// columns` at every allocating rung, on the nose**, block-scope and
/// `cta_group::2` alike — so it is asserted as an equality rather than reported
/// as a trend. An exact law is the kind worth breaking loudly:
///
/// - **Every rung's spin actually ran on the clock.** A CTA that fell out on
///   the iteration guard reports a short interval, and short intervals overlap
///   less — indistinguishable from low residency. This is the check that stops
///   a hoisted `%globaltimer` from being read as a finding.
/// - **The control is above one.** Without it, every rung reading 1 is
///   satisfied by an SM that admits one CTA of anything, and the census would
///   be attributing to tcgen05 something it had not shown tcgen05 causes —
///   #77's own rule, carried over.
/// - **Concurrent holders are exactly what the CTA's per-CTA resources pay
///   for**, `512 / columns` against shared memory per SM over the plan,
///   whichever is tighter. Above it is impossible and means the census is
///   counting something other than concurrent holders; below it means the
///   driver has started reserving one of the two more coarsely than asked, and
///   every "residency is the column arithmetic" statement in `src/tmem.rs` and
///   `experiments/README.md` is stale. The 512-column rung is the same assertion
///   doing double duty as a positive control: one CTA an SM is arithmetically
///   forced there, so a detector that cannot read a real 1 cannot be trusted
///   when it reads a 2.
/// - **Releasing the columns releases the SM.** The allocate-and-release rung
///   still contains a `tcgen05.alloc` and the query still answers 1 for it. If
///   it ever falls to 1 in the census too, the reservation has become static at
///   admission and the mechanism described throughout this file is wrong.
fn verdict(measured: &[Measured], shared_per_sm: u32) -> Result<String, Box<dyn Error>> {
    let floor = HOLD_FLOOR * HOLD_NS as f64 / 1e3;
    if let Some(row) = measured.iter().find(|row| row.shortest_hold_us < floor) {
        return Err(format!(
            "{}: a CTA held for only {:.1} us of the {:.1} us it asked for, so its spin ended on \
             the iteration guard rather than on the clock. Short intervals overlap less, which \
             reads exactly like low residency — no number in this table is usable",
            row.name, row.shortest_hold_us, floor
        )
        .into());
    }

    if let Some((row, budget)) = measured
        .iter()
        .filter_map(|row| row.budget(shared_per_sm).map(|budget| (row, budget)))
        .find(|(row, budget)| row.holding != *budget)
    {
        return Err(format!(
            "{} held {} CTAs at once on one SM where {} columns of tensor memory and a {} B \
             shared plan pay for exactly {budget}. {}",
            row.name,
            row.holding,
            row.columns.unwrap_or(0),
            row.shared,
            if row.holding > budget {
                "That is more than the hardware has, so the census is counting something other \
                 than concurrent holders"
            } else {
                "Residency is no longer what the per-CTA resources divide to, and every \
                 statement to that effect in src/tmem.rs and experiments/README.md is stale"
            }
        )
        .into());
    }

    let control = measured
        .iter()
        .find(|row| row.columns.is_none() && row.ranks == 1)
        .ok_or("no control rung in the census")?;
    if control.resident <= 1 {
        return Err(format!(
            "the control is resident {} CTA(s) an SM, so this device is not admitting more than \
             one CTA of anything and the census cannot attribute a 1 to tcgen05",
            control.resident
        )
        .into());
    }

    let released = measured
        .iter()
        .find(|row| row.ranks == 1 && !row.holds && row.columns == Some(512))
        .ok_or("no allocate-and-release rung in the census")?;
    if released.holding <= 1 {
        return Err(format!(
            "the rung that allocates 512 columns and gives them straight back is still only {} \
             CTA(s) an SM. The cost of tcgen05 would then be a static reservation taken when the \
             CTA is admitted rather than the price of holding columns, which is the opposite of \
             what this file argues everywhere",
            released.holding
        )
        .into());
    }

    let find = |columns, shared| {
        measured
            .iter()
            .find(move |row| row.ranks == 1 && row.columns == columns && row.shared == shared)
    };
    let bare = find(Some(256), CENSUS_SHARED).ok_or("no 256-column rung in the census")?;
    let envelope =
        find(Some(256), FLASH_SHARED).ok_or("no 256-column rung at flash's shared plan")?;
    let plan_only = find(None, FLASH_SHARED).ok_or("no control rung at flash's shared plan")?;

    let tcgen05 = format!(
        "tcgen05 does not cost an SM its whole tensor memory: at flash_forward's 256 columns and \
         a token shared plan an SM holds {} CTAs, resident {}, where the driver predicts {:.0}. \
         Holders are what the per-CTA resources divide to at every rung, cta_group::2 included.",
        bare.holding, bare.resident, bare.predicted
    );
    let flash = if envelope.holding > 1 {
        format!(
            "And flash_forward's 1 CTA/SM is the QUERY: at its own {} B shared plan *and* its 256 \
             columns together, an SM still holds {}.",
            envelope.shared, envelope.holding
        )
    } else {
        format!(
            "But flash_forward's 1 CTA/SM is REAL and tcgen05 is not what causes it: its own {} B \
             shared plan admits {} CTA an SM with no allocator in the kernel at all, and the two \
             together are {}. Shared memory is the binding resource — the one #70 tried to rule \
             out and could not, because the query it asked was pinned at 1 by the allocator.",
            plan_only.shared, plan_only.holding, envelope.holding
        )
    };
    Ok(format!("{tcgen05} {flash} {}", located_step(measured)))
}

/// Where the two-CTAs-an-SM shared-memory step landed, from the
/// [`THRESHOLD_PLANS`] rungs — #85 step 2.
///
/// Reported and not asserted, and that is the point of the rungs: the number
/// #85 has to design against is the one this locates, so a rung of the ladder
/// counting something other than the arithmetic is the finding rather than a
/// failure. What it *cannot* do is say nothing — a ladder whose every rung
/// counts 2, or whose every rung counts 1, has located no step and says so.
fn located_step(measured: &[Measured]) -> String {
    let plans = |wanted: bool| {
        measured
            .iter()
            .filter(move |row| row.bisects && (row.resident >= 2) == wanted)
            .map(|row| row.shared)
    };
    let (fits, capped) = (plans(true).max(), plans(false).min());
    let arithmetic = HALF_SHARED_PER_SM;
    match (fits, capped) {
        (Some(fits), Some(capped)) => format!(
            "The two-CTA shared step (#85) is in [{fits}, {capped}): {fits} B counts 2 and \
             {capped} B counts 1, so a plan reaching two CTAs an SM has to come in at or under \
             {fits} B — {} the {arithmetic} B half of the SM #85 does the arithmetic at. \
             flash_forward's 147536 B is {} B over it.",
            if fits >= arithmetic {
                "which is"
            } else {
                "under"
            },
            FLASH_SHARED - fits
        ),
        _ => format!(
            "The step-bisecting rungs located nothing: every one of them counted {}. They are \
             spaced around {arithmetic} B, half of a B200's 233472 B an SM, so on a device with \
             another amount of shared memory they bracket the wrong number and #85's threshold \
             is unmeasured here.",
            if fits.is_some() { "2 or more" } else { "1" }
        ),
    }
}

fn table(measured: &[Measured], blocks: u32, sms: u32, shared_per_sm: u32) -> String {
    let mut table = String::new();
    let _ = write!(
        table,
        "\n  tcgen05 residency census (#78) — peak CTAs counted on one SM by\n  \
         %smid and %globaltimer, {blocks} CTAs of one warp over {sms} SMs, {} us each,\n  \
         against {shared_per_sm} B of shared memory an SM:\n  \
         {:<26}{:>8}{:>9}{:>6}{:>8}{:>9}{:>9}{:>8}{:>9}",
        HOLD_NS / 1000,
        "rung",
        "columns",
        "shared B",
        "rank",
        "budget",
        "resident",
        "holding",
        "driver",
        "wait us",
    );
    for row in measured {
        let columns = match row.columns {
            Some(columns) if row.holds => columns.to_string(),
            Some(columns) => format!("({columns})"),
            None => "—".to_string(),
        };
        let budget = match row.budget(shared_per_sm) {
            Some(budget) => budget.to_string(),
            None => "—".to_string(),
        };
        let _ = write!(
            table,
            "\n  {:<26}{columns:>8}{:>9}{:>6}{budget:>8}{:>9}{:>9}{:>8.1}{:>9.1}",
            row.name,
            row.shared,
            row.ranks,
            row.resident,
            row.holding,
            row.predicted,
            row.worst_wait_us,
        );
    }
    let _ = write!(
        table,
        "\n  budget is what the CTA's per-CTA resources pay for: 512/columns of tensor\n  \
         memory against shared memory per SM over the plan, whichever is tighter. resident\n  \
         counts [entered, left] windows open at once, holding counts [allocated, left] —\n  \
         resident above holding is a CTA parked in a blocking tcgen05.alloc, which is\n  \
         residency the occupancy query and a throughput curve both read as absence. driver\n  \
         is what that same loaded function answers: max_active_blocks_per_multiprocessor at\n  \
         one rank, max_active_clusters divided back into CTAs an SM at two."
    );
    table
}

pub fn check(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let context = stream.context();
    let sms = context.multiprocessor_count()?;
    let shared_per_sm = device_attribute(
        context,
        cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    )?;
    let blocks = CTAS_PER_SM * sms;
    let loaded = module.as_cuda_module();
    let measured: Vec<Measured> = rungs()
        .iter()
        .map(|rung| {
            let function = loaded.load_function(rung.entry)?;
            // The >48 KiB opt-in, before either the query or the launch. It is
            // a persistent mutation of the loaded function (#70), which is a
            // trap for a *sweep of queries* at descending sizes — every later
            // one answers about the admitted ceiling instead of the size asked
            // for. It is not a trap here: every query below passes its own
            // byte count as an argument, and every launch declares its own
            // plan, so no probe reads a number a previous probe set.
            admit_shared_plan(&function, rung.shared)?;
            let predicted = predicted(&function, rung, blocks, sms)?;
            let rows = census(stream, module, rung, blocks)?;
            tally(rung, predicted, &rows)
        })
        .collect::<Result<_, _>>()?;

    let seen = measured.iter().map(|row| row.sms).max().unwrap_or(0);
    let table = table(&measured, blocks, sms, shared_per_sm);
    // The table goes out on both paths. A census that moved is exactly when
    // the numbers are worth reading, and an error that swallowed them would
    // send the next person back to the B200 to get them again.
    match verdict(&measured, shared_per_sm) {
        Ok(note) => Ok(format!("{seen} SMs ran CTAs{table}\n  {note}")),
        Err(error) => Err(format!("{error}{table}").into()),
    }
}
