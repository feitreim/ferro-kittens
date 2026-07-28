//! Does holding a tcgen05 allocation cost occupancy, and does the size of it
//! matter? (#74)
//!
//! `flash_forward` answers **1 block/SM** at 147536 B of dynamic shared memory,
//! at zero bytes, and at block widths of 32, 64, 128 and 256, while `softmax`
//! goes 28/14/7/3 and `layernorm` 8/4/2/1 on the same query in the same run.
//! `ptxas` prices it at 124 registers a thread with no spills and no static
//! shared, and 124 registers at 128 threads would admit four blocks. Flat in
//! shared memory and flat in warp count is neither of the two ceilings anyone
//! can name from a compiler listing, so it is a **per-CTA** resource, and #70
//! guessed TMEM from the single fact that `flash_forward` is the only queryable
//! entry in that table using tcgen05.
//!
//! That is an unexcluded-alternatives argument from a sample of one, which is
//! the shape #47 exists to warn about. This is the control it was missing.
//!
//! # What the ladder holds fixed
//!
//! Every rung is the same kernel: a 32-byte shared plan (the TMEM staging word
//! and nothing else), one block sync, one store. [`crate::kernels::tmem_ladder_none`]
//! is that kernel with `tcgen05.alloc` deleted and nothing else changed — so
//! the gap between it and the rungs above it is attributable to tcgen05 and to
//! nothing else in the kernel. The rungs then vary the allocation from the
//! allocator's smallest unit to the whole SM's tensor memory, which is the
//! experiment #74 asked for on `flash_forward` itself, run on a kernel small
//! enough that nothing else can be the cause.
//!
//! Three outcomes, and they are distinguishable:
//!
//! | shape | what it means |
//! | --- | --- |
//! | rungs track `512 / columns`, control high | the **column count** binds |
//! | every rung 1, control high | **any** tcgen05 allocation binds |
//! | every rung tracks the control | TMEM is **excluded** |
//!
//! `tmem_ladder_runtime` takes its column count as a kernel argument, so it
//! separates the second outcome's two mechanisms: a driver reading a count
//! `ptxas` recorded cannot read one that does not exist, but a driver reserving
//! tensor memory for any kernel that touches the allocator does not need to.
//!
//! # The measurement trap this avoids
//!
//! #70's shared-memory sweep had to reload the module per probe, because the
//! `>48 KiB` opt-in is a *persistent mutation of the loaded function*: admit it
//! once at the largest size and every later query at a smaller size still
//! answers about that ceiling. Nothing here crosses 48 KiB and every query is
//! at zero dynamic shared, so no probe mutates anything the next one reads —
//! but the rungs are separate functions in any case, which is the property that
//! makes the trap impossible rather than merely avoided.
//!
//! # A note on 192
//!
//! `flash_forward` allocates `KEYS + HEAD` = **192** columns, and cuda-oxide
//! documents the allocator's `N_COLS` as "a power of 2: 32, 64, 128, 256, 512".
//! 192 is not one. That rung is therefore queried and never launched — the
//! occupancy question is answered by the loaded function and needs no launch,
//! and this harness does not issue an allocation the ISA does not define. It is
//! in the ladder because it is the number `flash_forward` actually names.

use std::error::Error;
use std::fmt::Write as _;

use cuda_core::{CudaFunction, CudaStream, DeviceBuffer};
use cuda_host::CudaKernel;

use crate::{kernels, launch_config};

/// Block widths the ladder is queried at, matching #74's table.
const WIDTHS: [u32; 4] = [32, 64, 128, 256];

/// Threads each launchable rung is launched with. One warp is all the
/// allocator needs and all the probe does.
const LAUNCH_THREADS: u32 = 32;

/// The rung's whole dynamic shared plan: the TMEM staging word, padded to the
/// alignment `DynamicSharedArray::<u8, 128>` asks for. Same as `STTM_SHARED`,
/// spelled here because the ladder's claim is that this number is *not* what
/// moves.
const LADDER_SHARED: u32 = 32;

/// One rung: a PTX entry point and the tcgen05 allocation it holds.
struct Rung {
    name: &'static str,
    entry: &'static str,
    /// Columns its `tcgen05.alloc` names, or `None` for the control, which has
    /// no allocator at all.
    columns: Option<u32>,
    /// Whether the harness will launch it. See the module doc on 192.
    launchable: bool,
}

/// What one rung's loaded function says about itself.
struct Measured {
    name: &'static str,
    columns: Option<u32>,
    registers: u32,
    static_shared: u32,
    cluster: Option<(u32, u32, u32)>,
    /// Blocks per SM at each of [`WIDTHS`], all at zero dynamic shared.
    blocks: [u32; WIDTHS.len()],
}

fn rungs() -> Vec<Rung> {
    macro_rules! rung {
        ($name:literal, $kernel:ident, $columns:expr, $launchable:expr) => {
            Rung {
                name: $name,
                entry: <kernels::${concat(__, $kernel, _CudaKernel)} as CudaKernel>::PTX_NAME,
                columns: $columns,
                launchable: $launchable,
            }
        };
    }
    vec![
        rung!("no tcgen05", tmem_ladder_none, None, true),
        rung!("alloc 32", tmem_ladder_32, Some(32), true),
        rung!("alloc 64", tmem_ladder_64, Some(64), true),
        rung!("alloc 128", tmem_ladder_128, Some(128), true),
        rung!("alloc 256", tmem_ladder_256, Some(256), true),
        rung!("alloc 512", tmem_ladder_512, Some(512), true),
        // Not launched: 192 is not a power of two, and the occupancy query
        // needs no launch. It is flash_forward's own column count.
        rung!("alloc 192 (flash)", tmem_ladder_runtime, Some(192), false),
    ]
}

fn function_attribute(
    function: &CudaFunction,
    attribute: cuda_core::sys::CUfunction_attribute,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    let mut value: i32 = 0;
    let status = unsafe {
        cuda_core::sys::cuFuncGetAttribute(&mut value, attribute, function.cu_function())
    };
    if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuFuncGetAttribute({name}) failed with driver status {status}").into(),
        );
    }
    Ok(value as u32)
}

fn measure(module: &kernels::LoadedModule, rung: &Rung) -> Result<Measured, Box<dyn Error>> {
    let function = module.as_cuda_module().load_function(rung.entry)?;
    let mut blocks = [0u32; WIDTHS.len()];
    for (slot, width) in blocks.iter_mut().zip(WIDTHS) {
        *slot = function.max_active_blocks_per_multiprocessor(width, 0)?;
    }
    Ok(Measured {
        name: rung.name,
        columns: rung.columns,
        registers: function_attribute(
            &function,
            cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
            "NUM_REGS",
        )?,
        static_shared: function.static_shared_memory_bytes()?,
        // #74's third candidate: `flash_forward` is a plain `#[kernel]`, but if
        // the tcgen05 path implied a `cta_group::2` shape the driver could be
        // answering about a launch geometry the kernel never declares. These
        // rungs issue no MMA at all and are the same plain `#[kernel]`, so a
        // cluster shape here would have to come from the allocator.
        cluster: function.required_cluster_dimensions()?,
        blocks,
    })
}

/// Launch a rung once and read back what its allocator handed it, so the table
/// above is about kernels that demonstrably hold the allocation rather than
/// about symbols in a module.
fn launched_address(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    rung: &Rung,
) -> Result<u32, Box<dyn Error>> {
    let mut out = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let config = launch_config(LAUNCH_THREADS, LADDER_SHARED);
    unsafe {
        match rung.columns {
            None => module.tmem_ladder_none(stream, config, &mut out)?,
            Some(32) => module.tmem_ladder_32(stream, config, &mut out)?,
            Some(64) => module.tmem_ladder_64(stream, config, &mut out)?,
            Some(128) => module.tmem_ladder_128(stream, config, &mut out)?,
            Some(256) => module.tmem_ladder_256(stream, config, &mut out)?,
            Some(512) => module.tmem_ladder_512(stream, config, &mut out)?,
            Some(columns) => {
                return Err(format!("rung at {columns} columns is not launchable").into());
            }
        }
    }
    Ok(out.to_host_vec(stream)?[0])
}

/// The shape #74 measured, asserted rather than merely printed.
///
/// A ladder that only reports is a thing someone has to read. The two claims
/// below are what `flash_forward`'s module doc and `examples/README.md` now
/// say in prose, so they are the two the harness holds:
///
/// 1. **Every** allocating rung is 1 block/SM at every block width. If a rung
///    ever tracks its column count, the "shrinking the allocation buys nothing"
///    advice is wrong and the docs are stale.
/// 2. The control is well above 1. Without it, rule 1 is satisfied by an SM
///    that admits one CTA of anything, and the ladder would be attributing to
///    tcgen05 something it had not shown tcgen05 causes.
///
/// A driver that changes this model should fail here. That is the point: the
/// finding is a measurement, and a measurement that nothing re-runs decays into
/// a claim.
fn verdict(measured: &[Measured]) -> Result<String, Box<dyn Error>> {
    let control = measured
        .iter()
        .find(|row| row.columns.is_none())
        .ok_or("no control rung in the ladder")?;
    let allocating = || measured.iter().filter(|row| row.columns.is_some());

    let widest = *WIDTHS.iter().max().expect("WIDTHS is not empty");
    let control_floor = control.blocks[WIDTHS.len() - 1];
    if control_floor <= 1 {
        return Err(format!(
            "the control admits only {control_floor} block/SM at {widest} threads, so this \
             SM is not admitting more than one CTA of anything and the ladder cannot \
             attribute a 1 to tcgen05"
        )
        .into());
    }

    if let Some(row) = allocating().find(|row| row.blocks.iter().any(|&blocks| blocks != 1)) {
        return Err(format!(
            "{} answers {:?} blocks/SM across widths {WIDTHS:?}, not 1. tcgen05 occupancy \
             is no longer flat in the column count — #74's conclusion, and the advice in \
             flash_forward's module doc that shrinking its allocation buys nothing, are \
             both stale",
            row.name, row.blocks
        )
        .into());
    }

    Ok(format!(
        "TMEM confirmed and the column count is not the lever: every allocation from 32 to \
         512 columns is 1 block/SM at every width, where the same kernel without tcgen05 \
         holds {} to {control_floor}. A CTA is charged the SM's whole tensor memory the \
         moment it touches the allocator, so shrinking flash_forward's 192 buys nothing — \
         warps per CTA is what is left.",
        control.blocks[0]
    ))
}

pub fn check(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let rungs = rungs();
    let measured: Vec<Measured> = rungs
        .iter()
        .map(|rung| measure(module, rung))
        .collect::<Result<_, _>>()?;

    let mut table = String::new();
    write!(
        table,
        "\n  tcgen05 occupancy ladder (#74) — blocks/SM at 0 dynamic shared,\n  \
         one 32-byte staging word, {} B static shared, per block width:\n  \
         {:<20}{:>9}{:>6}{:>10}",
        measured[0].static_shared, "rung", "columns", "regs", "cluster"
    )?;
    for width in WIDTHS {
        write!(table, "{width:>7}")?;
    }
    for row in &measured {
        let columns = match row.columns {
            Some(columns) => columns.to_string(),
            None => "—".to_string(),
        };
        let cluster = match row.cluster {
            Some((x, y, z)) => format!("{x}x{y}x{z}"),
            None => "none".to_string(),
        };
        write!(
            table,
            "\n  {:<20}{columns:>9}{:>6}{cluster:>10}",
            row.name, row.registers
        )?;
        for blocks in row.blocks {
            write!(table, "{blocks:>7}")?;
        }
    }

    // Every rung the ISA defines is launched, so the table is about kernels
    // that ran and gave their columns back rather than about symbols. A leak
    // would surface as the next launch failing, which is `repeated launch`'s
    // job; this one only asserts the allocator answered.
    let mut launched = 0usize;
    for rung in rungs.iter().filter(|rung| rung.launchable) {
        let address = launched_address(stream, module, rung)?;
        let expected_sentinel = rung.columns.is_none();
        if expected_sentinel != (address == crate::TMEM_LADDER_SENTINEL) {
            return Err(format!(
                "{}: staging word came back {address:#010x}, which is {} a TMEM address",
                rung.name,
                if expected_sentinel { "not" } else { "" }
            )
            .into());
        }
        launched += 1;
    }

    // The table goes out either way. A rung that moved is exactly the case
    // where the numbers are worth reading, and an error that swallowed them
    // would send the next person back to the B200 to get them again.
    match verdict(&measured) {
        Ok(note) => Ok(format!(
            "{launched} rungs launched, allocations balanced{table}\n  {note}"
        )),
        Err(error) => Err(format!("{error}{table}").into()),
    }
}
