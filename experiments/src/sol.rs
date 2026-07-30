//! `bench sol` — the small-shape end of [`crate::gemm_sol`], and the arithmetic
//! it is measured against.
//!
//! The port's ratio against cuBLASLt **grows monotonically with the problem**:
//! 0.795 at 4096³, 0.873 at 8192³, 0.946 at 16384³. That signature does not
//! point at the inner loop, which is the one thing 16384³ says is nearly right.
//! It points at the three terms that only exist at the small end — how many
//! output tiles there are, how many waves of residency they take, and how much
//! of the last wave is idle — so this sweep prints those *before* it launches
//! anything, and every table below is a test of a prediction the header already
//! made.
//!
//! # The prediction
//!
//! A cluster is two CTAs and owns one output tile. Both entries declare more
//! than half an SM's shared memory, so residency is one CTA per SM and the
//! device holds `SMs / 2` clusters at once. CLC work stealing hands the tiles
//! out dynamically, which removes the *static* imbalance and cannot remove the
//! integer one: `ceil(tiles / resident)` waves is a floor no scheduler beats,
//! and `tiles / (waves · resident)` is the fraction of the launch that is doing
//! work.
//!
//! On 148 SMs that is 74 resident clusters, and at 4096³ with the `[256, 256]`
//! tile it is 256 tiles, 3.46 waves' worth over 4 waves, and **0.865**. The
//! same shape on the `[512, 256]` tile is 128 tiles over 2 waves — the same
//! 0.865, which is the coincidence that makes the pair a controlled comparison:
//! two entries, one quantization number, and 0.75× the operand traffic on one
//! side.
//!
//! At 8192³ both entries are at 0.988, so **quantization predicts nothing about
//! the 8192³ gap** and whatever explains it is not this.
//!
//! # What each table is for
//!
//! 1. **entry against shape** — the three entries at four sizes. `[512, 256]`
//!    and `[256, 256]` differ *only* in operand traffic per flop at 8192³, and
//!    at 4096³ they are equal on quantization too. `[256, 128]` is the other
//!    direction: it quadruples the tile count of a square problem against the
//!    wide entry's, buying 0.865 → 0.988 at 4096³ and paying 1.5× the operand
//!    traffic per flop for it.
//! 2. **the wave ladder** — `n = k = 4096` with `m` climbing in steps of 512,
//!    which walks the wave efficiency up and down a sawtooth the header
//!    predicts exactly. If measured throughput tracks it, quantization is the
//!    term; if the curve is flat, it is not, and no tiling change will help.
//! 3. **the N band** — `group`, which was a rule inside the kernel keyed on
//!    `tiles_m` and is a launch parameter since this sweep. The rule gives 4096³
//!    a band of 2 and 8192³ a band of 8, and only the second of those was ever
//!    measured (#138 found 4 against 8 neutral at 8192³).
//! 4. **a shape whose `k` is not its `m`** — one row, last, because the tables
//!    above are worth more than it is and a launch that does not return takes
//!    everything after it with it. `4096x4096x1024` wedged a container in the
//!    first run of this sweep and is not in it; `4096x4096x2048` is.
//!
//! # What is deliberately not here
//!
//! The per-K-block rate and per-tile constant each entry decomposes into, and
//! the epilogue's share of a launch. Those are #144's ablation ladder, which
//! measures them directly rather than inferring them from a depth column, and
//! this sweep would only restate them less well.

use std::error::Error;
use std::sync::Arc;

use cuda_core::CudaContext;

use crate::bench::{Baseline, ITERATIONS, Shape, WARMUP};
use crate::gemm_sol::{self, Plan, Variant};

/// The shipped entry first, so every `vs row 1` column has the control as its
/// denominator.
const VARIANTS: [Variant; 3] = [Variant::M256xN256, Variant::M256xN128, Variant::M512xN256];

/// Two CTAs to a cluster, which is the unit residency is counted in.
const RANKS: u32 = 2;

fn tflops(shape: Shape, milliseconds: f64) -> f64 {
    2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 / (milliseconds / 1e3) / 1e12
}

/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`, queried rather
/// than written down: residency is a floor division by it, so a figure that is
/// only nearly right moves an entry across an occupancy step.
fn shared_per_sm(context: &Arc<CudaContext>) -> Result<usize, Box<dyn Error>> {
    let mut bytes = 0i32;
    // SAFETY: the attribute is an `int` and `context` names a live device.
    let status = unsafe {
        cuda_core::sys::cuDeviceGetAttribute(
            &mut bytes,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            context.cu_device(),
        )
    };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) = {status}"
        )
        .into());
    }
    Ok(bytes as usize)
}

/// How many clusters the device holds at once, and the shared plan that says so.
#[derive(Clone, Copy)]
struct Residency {
    sms: u32,
    per_sm: usize,
}

impl Residency {
    fn ctas_per_sm(self, variant: Variant) -> u32 {
        (self.per_sm / variant.shared_bytes()).max(1) as u32
    }

    fn clusters(self, variant: Variant) -> u32 {
        (self.sms * self.ctas_per_sm(variant) / RANKS).max(1)
    }
}

/// Tiles, waves, and the fraction of the launch that is not idling — the whole
/// quantization model, in one place, so every table below divides by the same
/// number.
fn waves(shape: Shape, variant: Variant, residency: Residency) -> (u32, u32, f64) {
    let tiles = gemm_sol::clusters(shape, variant);
    let resident = residency.clusters(variant);
    let waves = tiles.div_ceil(resident);
    (tiles, waves, tiles as f64 / (waves * resident) as f64)
}

/// Flops per operand byte for the entry's tile: `M·N/(M+N)`, halved back out of
/// the two bytes an FP16 operand costs. It is a property of the tile alone —
/// `BLOCK_K` cancels — and it is the only thing that separates the two entries
/// at a shape where they quantize the same.
fn intensity(variant: Variant) -> f64 {
    let m = variant.m_tile() as f64;
    let n = variant.n_tile() as f64;
    m * n / (m + n)
}

/// A cuBLASLt row, taken once per shape and reused by every table that quotes
/// the same shape.
struct Denominators<'a> {
    baseline: Option<Baseline>,
    context: &'a Arc<CudaContext>,
    taken: Vec<(Shape, f64, String)>,
}

impl Denominators<'_> {
    fn at(&mut self, shape: Shape) -> Option<f64> {
        let baseline = self.baseline?;
        if let Some(row) = self
            .taken
            .iter()
            .find(|row| row.0.m == shape.m && row.0.n == shape.n && row.0.k == shape.k)
        {
            return Some(row.1);
        }
        eprintln!("{shape} on {}: staging and checking", baseline.name);
        match (baseline.bench)(self.context, shape) {
            Ok((timings, algorithm)) => {
                self.taken.push((shape, timings.min(), algorithm));
                Some(timings.min())
            }
            Err(error) => {
                println!("FAIL  {} at {shape}: {error}", baseline.name);
                None
            }
        }
    }

    fn algorithms(&self) {
        let Some(baseline) = self.baseline else {
            return;
        };
        println!(
            "\nthe algorithm {}'s heuristic chose at each shape, so the baseline can be\n\
             reproduced — and so `splitk` can be read off it rather than guessed at:",
            baseline.name
        );
        for (shape, _, algorithm) in &self.taken {
            println!("  {shape:<18}{algorithm}");
        }
    }
}

/// One timed row: check at the gate size, time at `shape`, print the plan's own
/// wave arithmetic beside what it measured.
#[allow(clippy::too_many_arguments)]
fn row(
    context: &Arc<CudaContext>,
    label: &str,
    shape: Shape,
    plan: Plan,
    residency: Residency,
    denominators: &mut Denominators<'_>,
    reference: Option<f64>,
) -> Option<f64> {
    eprintln!("{shape} on {label}: staging and checking");
    let timings = match gemm_sol::bench_plan(context, shape, plan) {
        Ok(timings) => timings,
        Err(error) => {
            println!("{label:<14}{shape:<18}FAIL  {error}");
            return None;
        }
    };
    let (tiles, count, efficiency) = waves(shape, plan.variant, residency);
    let ours = tflops(shape, timings.min());
    let theirs = denominators.at(shape).map(|ms| tflops(shape, ms));
    println!(
        "{label:<14}{shape:<18}{tiles:>7}{count:>7}{efficiency:>10.3}{:>11.4}{ours:>11.1}{:>11.1}{:>11.3}{:>10.3}{:>9.1}%",
        timings.min(),
        tflops(shape, timings.median()),
        theirs.map(|theirs| ours / theirs).unwrap_or(f64::NAN),
        reference.map(|reference| ours / reference).unwrap_or(1.0),
        100.0 * timings.spread(),
    );
    Some(ours)
}

/// The sub-4096 ladder, which is its own entry point because it is the one table
/// here whose rows are short enough that the *measurement* and not the kernel is
/// the binding uncertainty: a 1024³ launch is 17 µs and its thirty samples
/// spread 13–15%, so a difference between two rungs is quoted only if it
/// survives being taken twice.
pub fn sweep_small(
    context: &Arc<CudaContext>,
    baseline: Option<Baseline>,
) -> Result<(), Box<dyn Error>> {
    let residency = Residency {
        sms: context.multiprocessor_count()?,
        per_sm: shared_per_sm(context)?,
    };
    let mut denominators = Denominators {
        baseline,
        context,
        taken: Vec::new(),
    };

    println!(
        "gemm-sol below 4096^3 — the `[256, 128]` entry against the shipped `[256, 256]`,\n\
         each pass a complete measurement of both, taken twice round-robin so the\n\
         difference is quoted against its own repeatability rather than against thirty\n\
         samples of one launch. every row checked exact at 1024x1024x512 first.\n\
         at these sizes the wide tile does not fill a single wave, so `wave eff` is the\n\
         term and the narrow tile is what buys it: 0.216 -> 0.432 at 1024^3 and\n\
         0.486 -> 0.973 at 1536^3, against equal efficiency at 2048^3 and 3072^3 —\n\
         which is what makes those two rows the control for operand traffic alone."
    );
    for pass in 1..=2 {
        println!("\npass {pass}");
        heading("entry");
        for size in [1024usize, 1536, 2048, 3072] {
            let shape = Shape {
                m: size,
                n: size,
                k: size,
            };
            let mut reference = None;
            for variant in [Variant::M256xN256, Variant::M256xN128] {
                let plan = Plan {
                    variant,
                    group: gemm_sol::default_group(size),
                };
                let measured = row(
                    context,
                    variant.name(),
                    shape,
                    plan,
                    residency,
                    &mut denominators,
                    reference,
                );
                reference = reference.or(measured);
            }
        }
    }
    denominators.algorithms();
    Ok(())
}

/// A `k` ladder walked downward, one row at a time, announced before each launch —
/// `bench sol-k`.
///
/// #146 found `4096x4096x1024` on `[256, 256]` printing nothing for 1200 s until
/// `scripts/modal-run`'s watchdog stopped the container, while `4096x4096x2048`
/// runs. A `k` shorter than `m` is otherwise unexercised by this kernel, so the
/// question is where the boundary is and what is on the far side of it.
///
/// It is its own entry point, and it announces each row on `stderr` *before*
/// launching it, because the instrument has to survive the thing it measures: a
/// launch that does not return takes the process with it, and the last line
/// printed is then the whole of the answer. Every row is a complete
/// stage-check-time cycle at the shipped plan, so a row that prints is a row that
/// computed the right `C`.
pub fn sweep_k(
    context: &Arc<CudaContext>,
    baseline: Option<Baseline>,
) -> Result<(), Box<dyn Error>> {
    let residency = Residency {
        sms: context.multiprocessor_count()?,
        per_sm: shared_per_sm(context)?,
    };
    let mut denominators = Denominators {
        baseline,
        context,
        taken: Vec::new(),
    };
    println!(
        "gemm-sol with `k` below `m`, descending. every row is announced on stderr before it\n\
         is launched, so if the run stops the last announced row is the one that did not\n\
         return. the shape contract is `k % 256 == 0` and `k >= 256`, so every row below is\n\
         inside it, and a row that does not return is a defect rather than a misuse."
    );
    heading("entry");
    for k in [2048usize, 1792, 1536, 1280, 1024, 768, 512, 256] {
        let shape = Shape {
            m: 4096,
            n: 4096,
            k,
        };
        row(
            context,
            "M256xN256",
            shape,
            Plan {
                variant: Variant::M256xN256,
                group: gemm_sol::default_group(4096),
            },
            residency,
            &mut denominators,
            None,
        );
    }
    denominators.algorithms();
    Ok(())
}

fn heading(first: &str) {
    println!(
        "{first:<14}{:<18}{:>7}{:>7}{:>10}{:>11}{:>11}{:>11}{:>10}{:>10}",
        "shape",
        "tiles",
        "waves",
        "wave eff",
        "min ms",
        "TFLOP/s",
        "vs cuBLASLt",
        "vs row 1",
        "spread"
    );
}

pub fn sweep(context: &Arc<CudaContext>, baseline: Option<Baseline>) -> Result<(), Box<dyn Error>> {
    let residency = Residency {
        sms: context.multiprocessor_count()?,
        per_sm: shared_per_sm(context)?,
    };
    let mut denominators = Denominators {
        baseline,
        context,
        taken: Vec::new(),
    };

    println!(
        "gemm-sol at the small end — min ms over {ITERATIONS} timed launches after {WARMUP}\n\
         warm-ups, every row checked exact at 1024x1024x512 on its own plan before it is\n\
         timed. `wave eff` is tiles / (ceil(tiles / resident) * resident) and is computed\n\
         before the launch, not fitted to it."
    );

    println!("\n0. what an entry costs before anything is launched");
    println!(
        "{:<14}{:>12}{:>10}{:>12}{:>12}",
        "entry", "shared B", "CTA/SM", "clusters", "flops/byte"
    );
    for variant in VARIANTS {
        println!(
            "{:<14}{:>12}{:>10}{:>12}{:>12.1}",
            variant.name(),
            variant.shared_bytes(),
            residency.ctas_per_sm(variant),
            residency.clusters(variant),
            intensity(variant),
        );
    }
    println!(
        "{} SMs and {} B of shared memory an SM divides. `clusters` is the residency the\n\
         wave arithmetic below divides by; `flops/byte` is M*N/(M+N) for the entry's tile,\n\
         which is the only axis the two entries differ on at a shape where they quantize\n\
         the same.",
        residency.sms, residency.per_sm
    );

    println!(
        "\n1. entry against shape. at 8192^3 both entries are near 1.00 wave efficiency, so\n\
         a difference there is operand traffic and not quantization. at 4096^3 they are\n\
         *equal* on quantization too, which makes that pair the controlled comparison."
    );
    heading("entry");
    let sizes = [1024usize, 2048, 4096, 8192];
    for size in sizes {
        let shape = Shape {
            m: size,
            n: size,
            k: size,
        };
        let mut reference = None;
        for variant in VARIANTS {
            let plan = Plan {
                variant,
                group: gemm_sol::default_group(shape.m),
            };
            let measured = row(
                context,
                variant.name(),
                shape,
                plan,
                residency,
                &mut denominators,
                reference,
            );
            reference = reference.or(measured);
        }
    }

    println!(
        "\n2. the wave ladder: n = k = 4096, m climbing by 512. wave efficiency sawtooths\n\
         between 0.86 and 0.99 across these rows with nothing else moving, so this is the\n\
         quantization model asked directly. `vs row 1` is against 4096^3, whose efficiency\n\
         is the lowest in the column."
    );
    heading("entry");
    let mut reference = None;
    for m in [4096usize, 4608, 5120, 5632] {
        let shape = Shape {
            m,
            n: 4096,
            k: 4096,
        };
        let plan = Plan {
            variant: Variant::M256xN256,
            group: gemm_sol::default_group(4096),
        };
        let measured = row(
            context,
            "M256xN256",
            shape,
            plan,
            residency,
            &mut denominators,
            reference,
        );
        reference = reference.or(measured);
    }

    println!(
        "\n3. the N band. `group` is how many tile-columns of N the traversal walks before\n\
         it steps in M, and the rule it replaced gave 4096^3 a band of 2 and 8192^3 a band\n\
         of 8 off `tiles_m <= 16`. the 8192^3 half of that rule has been measured; the\n\
         4096^3 half has not."
    );
    heading("group");
    for (size, variant) in [
        (4096usize, Variant::M256xN256),
        (4096, Variant::M256xN128),
        (8192, Variant::M512xN256),
    ] {
        let shape = Shape {
            m: size,
            n: size,
            k: size,
        };
        let mut reference = None;
        for group in [1u32, 2, 4, 8, 16] {
            let plan = Plan { variant, group };
            let measured = row(
                context,
                &format!("{} @ G={group}", variant.name()),
                shape,
                plan,
                residency,
                &mut denominators,
                reference,
            );
            reference = reference.or(measured);
        }
    }

    println!(
        "\n4. a `k` that is not `m`, last in the file on purpose. wave efficiency is the\n\
         4096^3 row's exactly, so this only asks whether halving the K loop's depth moves\n\
         the entry ranking — a shallower K amortizes the per-tile constant over half as\n\
         many k-blocks, and the narrow entry has twice as many tiles to pay it on."
    );
    heading("entry");
    let mut reference = None;
    for variant in [Variant::M256xN256, Variant::M256xN128] {
        let shape = Shape {
            m: 4096,
            n: 4096,
            k: 2048,
        };
        let plan = Plan {
            variant,
            group: gemm_sol::default_group(4096),
        };
        let measured = row(
            context,
            variant.name(),
            shape,
            plan,
            residency,
            &mut denominators,
            reference,
        );
        reference = reference.or(measured);
    }

    denominators.algorithms();
    println!(
        "\n`spread` is max/min - 1 within one call, printed because a ratio between two rows\n\
         inherits both rows' repeatability divided by its own size."
    );
    Ok(())
}
