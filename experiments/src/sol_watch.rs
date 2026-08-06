//! Where every warp was when the launch stopped making progress — `bench sol-watch`.
//!
//! #146 found `4096x4096x1024` on `[256, 256]` printing nothing for 1200 s, #148
//! bracketed it between `k = 1280` and `k = 1024`, and #149 listed four candidate
//! mechanisms and could separate none of them. All three cost a container per
//! attempt and none of them could say *where* the kernel was, because a launch
//! that does not return takes the process with it: the watchdog stops the
//! container and the position of all six warps is lost.
//!
//! This is the instrument that ends that. Both arms are
//! [`crate::gemm_sol::small_body`] at [`WATCH_DEEP`] — the shipped device body, with
//! every spin replaced by [`kittens::sync::Semaphore::wait_before`] and a
//! four-word mark written past the end of `C` at every loop head and on every
//! deadline. So the launch *terminates*, and one container gives, per CTA and per
//! warp, which loop it was in, the `sequence` and ring index it was at, the
//! parity it was awaiting, and how many items the producer had published before
//! it left.
//!
//! **The two arms differ in one const and nothing else.** `one deep` is
//! [`WATCH_ONE_DEEP`]: the item handoff at a depth of one — a single `ready`
//! barrier and a single `info` cell, which is what the port carried — and `four
//! deep` is the shipped depth. Same body, same barriers, same shared plan, same
//! `C` to compare against. That is what makes the pair a test of the depth rather
//! than a story about it.
//!
//! # What it found
//!
//! One container, four `k` rungs, both depths, every row's `C` compared against
//! the fp32 reference over all 16 777 216 outputs:
//!
//! | shape | one deep | four deep |
//! | --- | --- | --- |
//! | 4096x4096x1280 | exact | exact |
//! | 4096x4096x1024 | exact | exact |
//! | **4096x4096x512** | **11 885 748 of 16 777 216 outputs wrong** | exact |
//! | **4096x4096x256** | **11 896 908 of 16 777 216 outputs wrong** | exact |
//!
//! **The wrong outputs are zeros** — `C[0, 1536] = 0, want -1042` — so they are
//! not miscomputed, they are **never written**. Whole output tiles are missing,
//! which is the overwrite branch of the defect seen directly: the producer
//! republished into the one `info` cell before the epilogue read it, the epilogue
//! read a later item's coordinates, and the item it skipped was left at the
//! buffer's zeros. 71% of `C` is a large number for that because a cluster runs
//! three or four items and losing one loses a quarter of its work.
//!
//! **No CTA stalled at any rung**, which is the honest caveat: the instrument
//! perturbs the race it measures. The marks are stores, they cost the roles a
//! little, and under them the one-deep arm takes the *overwrite* branch where the
//! shipped kernel took the *never-flips-again* branch two rungs higher. The two
//! are one defect — [`kittens::sync::handoff::depth_needed`] enumerates both from
//! the same state space — but this arm did not reproduce the hang itself, and
//! nothing here says which branch `4096x4096x1024` took before the fix.
//!
//! The report also says what a launch's warp population actually is, which is
//! worth knowing: at 4096² only **888** of the 3072 warps the grid asks for ever
//! wrote a mark, and 888 is 148 CTAs × 6 warps. CLC cancelled the other 364
//! clusters before they launched, so a "512-cluster launch" is 74 clusters running
//! three or four items each.
//!
//! # Reading a row
//!
//! Every warp's last mark is `exited` in a healthy row. A row that stalls prints
//! the marks instead, and the sites are the whole diagnosis: `STALL ready` is the
//! item handoff, `STALL free` and `STALL load` the operand ring, `STALL clc` the
//! hardware's work queue, `STALL full` and `STALL empty` the accumulator
//! handshake — one site per candidate in #149's list.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, cuda_module, kernel, launch_contract};

use kittens::global::GlobalLayout;
use kittens::shared::F16;

use crate::bench::Shape;
use crate::gemm_sol::{
    ACCUM_COLUMNS, ATile, B_BOX, BLOCK_N, BPanel, HALF_N, ONE_WARPGROUP, REPORT_FIELDS,
    REPORT_ROWS, REPORT_SLOTS, SHIPPED_DRAIN, SMALL_RINGS_END, SMALL_SHARED_BYTES, WATCH_DEEP,
    WATCH_ONE_DEEP, WHOLE, a_value, b_value, check_output, default_group, site_name, small_body,
    stage_f16, threads,
};

/// Depth one stage carries, per [`crate::sol_ablate`].
const K_TILE: usize = 64;
/// Warps a block has. Six roles are dispatched over
/// the watch kernels' 192 threads — they are pinned to one epilogue warpgroup,
/// like every other probe here; the report reserves eight slots.
const WARPS: u32 = threads(ONE_WARPGROUP) / 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The shipped `[256, 256]` entry with a deadline on every spin.
    ///
    /// # Safety
    ///
    /// As [`crate::gemm_sol::kernels::gemm_sol_m256`], and `c` must hold
    /// [`REPORT_ROWS`] rows of `ldc` past the output for the report.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_watch(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        group: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_DEEP,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }

    /// [`gemm_sol_m256_watch`] with the item handoff one deep, which is the
    /// spelling #149 is about.
    ///
    /// # Safety
    ///
    /// As [`gemm_sol_m256_watch`]. Computes a wrong `C` wherever it stalls.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196_864,
        dynamic_shared_alignment = 128
    )]
    pub unsafe fn gemm_sol_m256_watch_one_deep(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        group: u32,
        ldc: u32,
        mut c: DisjointSlice<u16>,
    ) {
        unsafe {
            small_body::<
                BLOCK_N,
                ACCUM_COLUMNS,
                HALF_N,
                B_BOX,
                BLOCK_N,
                SMALL_RINGS_END,
                WHOLE,
                true,
                SHIPPED_DRAIN,
                WATCH_ONE_DEEP,
                ONE_WARPGROUP,
                128,
                4,
            >(a_map, b_map, tiles_m, tiles_n, k_blocks, group, ldc, &mut c)
        }
    }
}

/// The handoff depth an arm was built at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    One,
    Four,
}

impl Depth {
    pub const BOTH: [Depth; 2] = [Depth::One, Depth::Four];

    fn name(self) -> &'static str {
        match self {
            Depth::One => "one deep",
            Depth::Four => "four deep",
        }
    }
}

/// One warp's last mark.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Mark {
    site: u32,
    sequence: u32,
    index: u32,
    parity: u32,
}

impl Mark {
    fn stalled(self) -> bool {
        self.site >= crate::gemm_sol::SITE_READY
    }
}

fn role(warp: u32) -> &'static str {
    match warp {
        4 => "tma",
        5 => "mma",
        _ => "epilogue",
    }
}

/// Launch one arm at `shape` and print what its warps were doing when it ended.
///
/// Operands are staged and `C` is checked, because an arm that does not stall
/// must compute the right answer and that is the whole of the control: the four-
/// deep arm is the shipped handoff, so a wrong `C` from it would mean the
/// instrument itself moved something.
pub fn watch(context: &Arc<CudaContext>, shape: Shape, depth: Depth) -> Result<(), Box<dyn Error>> {
    let Shape { m, n, k } = shape;
    let (tile_m, tile_n) = (256usize, BLOCK_N);
    if !m.is_multiple_of(tile_m) || !n.is_multiple_of(tile_n) || !k.is_multiple_of(4 * K_TILE) {
        return Err(format!("{shape} is outside the [256, 256] entry's contract").into());
    }
    let clusters = (m / tile_m * (n / tile_n)) as u32;
    let ctas = 2 * clusters;
    if (ctas * REPORT_SLOTS * REPORT_FIELDS) as usize > REPORT_ROWS * n {
        return Err(
            format!("{shape}'s {ctas} CTAs need more than {REPORT_ROWS} report rows").into(),
        );
    }

    eprintln!("  {shape} {}: staging and launching", depth.name());
    let stream = context.default_stream();
    let module = unsafe { kernels::load(context)? };
    let a = DeviceBuffer::from_host(&stream, &stage_f16(m, k, a_value))?;
    let b = DeviceBuffer::from_host(&stream, &stage_f16(n, k, b_value))?;
    let (a_layout, b_layout) = unsafe {
        (
            GlobalLayout::<F16, 2>::packed(a.cu_deviceptr(), [k, m]),
            GlobalLayout::<F16, 2>::packed(b.cu_deviceptr(), [k, n]),
        )
    };
    let a_map = a_layout.tensor_map::<ATile>(&stream)?;
    let b_map = b_layout.tensor_map::<BPanel>(&stream)?;
    // The report rows are why this allocation is not `m * n`.
    let mut c = DeviceBuffer::<u16>::zeroed(&stream, m * n + REPORT_ROWS * n)?;

    let config = LaunchConfig1D::new(ctas, threads(ONE_WARPGROUP), SMALL_SHARED_BYTES as u32);
    let (tiles_m, tiles_n) = ((m / 256) as u32, (n / tile_n) as u32);
    let (k_blocks, group, ldc) = ((k / K_TILE) as u32, default_group(m), n as u32);
    let (a_ptr, b_ptr) = (a_map.as_ptr(), b_map.as_ptr());
    match depth {
        Depth::One => {
            let prepared = module.prepare_gemm_sol_m256_watch_one_deep(config)?;
            unsafe {
                module.gemm_sol_m256_watch_one_deep(
                    &stream, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group, ldc,
                    &mut c,
                )?
            };
        }
        Depth::Four => {
            let prepared = module.prepare_gemm_sol_m256_watch(config)?;
            unsafe {
                module.gemm_sol_m256_watch(
                    &stream, &prepared, a_ptr, b_ptr, tiles_m, tiles_n, k_blocks, group, ldc,
                    &mut c,
                )?
            };
        }
    }
    stream.synchronize()?;

    let host = c.to_host_vec(&stream)?;
    let report = &host[m * n..];
    let marks: Vec<Vec<Mark>> = (0..ctas)
        .map(|cta| {
            (0..WARPS)
                .map(|warp| {
                    let at = (REPORT_FIELDS * (REPORT_SLOTS * cta + warp)) as usize;
                    let word = |field: usize| report[at + field] as u32;
                    Mark {
                        site: word(0),
                        sequence: word(1),
                        index: word(2),
                        parity: word(3),
                    }
                })
                .collect()
        })
        .collect();

    let stalled: Vec<u32> = (0..ctas)
        .filter(|&cta| marks[cta as usize].iter().any(|mark| mark.stalled()))
        .collect();
    println!(
        "\n  {shape}, {} — {} of {ctas} CTAs stalled",
        depth.name(),
        stalled.len()
    );
    let mut census: Vec<(u32, usize)> = Vec::new();
    for mark in marks.iter().flatten() {
        match census.iter_mut().find(|(site, _)| *site == mark.site) {
            Some((_, count)) => *count += 1,
            None => census.push((mark.site, 1)),
        }
    }
    census.sort_unstable();
    for (site, count) in census {
        println!("    {count:>5} warps last at  {}", site_name(site));
    }
    // Four CTAs is two clusters, which is what it takes to see that the two ranks
    // of a pair stall the same way. Printing all 512 would bury the finding.
    for &cta in stalled.iter().take(4) {
        println!("    cta {cta} (rank {})", cta % 2);
        for (warp, mark) in marks[cta as usize].iter().enumerate() {
            println!(
                "      warp {warp} {:<9} {:<18} sequence {:<4} index {:<4} parity {}",
                role(warp as u32),
                site_name(mark.site),
                mark.sequence,
                mark.index,
                mark.parity
            );
        }
    }
    if stalled.is_empty() {
        match check_output(&host[..m * n], m, n, k) {
            Ok(worst) => println!(
                "    C exact over {} BF16 outputs, worst |rel| {worst:.2e}",
                m * n
            ),
            Err(error) => println!("    C WRONG: {error}"),
        }
    }
    Ok(())
}

/// Both depths at the shape #149 is about, and at the rung above it as a control.
///
/// `1280` returns on the shipped entry and `1024` did not, so running both under
/// the instrument says whether the difference is a stall that moved or a stall
/// that appeared.
pub fn sweep(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    println!(
        "the item handoff at two depths, with a deadline on every spin. every warp's last\n\
         mark is read back out of the rows of `C` past the output, so a launch that does not\n\
         make progress still says where all six warps were. `one deep` is the spelling the\n\
         port carried; `four deep` is the depth the contract needs — see `ITEMS`."
    );
    for k in [1280usize, 1024, 512, 256] {
        for depth in Depth::BOTH {
            let shape = Shape {
                m: 4096,
                n: 4096,
                k,
            };
            if let Err(error) = watch(context, shape, depth) {
                println!("\n  {shape}, {}: FAILED — {error}", depth.name());
            }
        }
    }
    Ok(())
}
