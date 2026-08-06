//! Does an MMA read a segment `tcgen05.st` wrote into it? (#177)
//!
//! `sttm round trip`, `ldtm x8 map` and `sttm restage` establish that
//! `TmemTile::store_tile` is the exact inverse of the drain, at the right
//! columns, against values the store path never computed. Every one of them
//! reads the segment back with **another LDTM**, and that is not the use STTM
//! has in a kernel: an in-place accumulator rescale hands the stored segment to
//! an **MMA**, and no case here has ever done that.
//!
//! What that cost is a shipped kernel. oxide-train#68's flash forward carries a
//! `RegTile<32, 128>` — 128 registers per thread and 1136 B of local depot,
//! resident for the whole key stream — because its conditional correction has to
//! rescale `O = P·V` and the only way to write TMEM was to hold the accumulator
//! in registers instead. The rescale is a read-modify-write of the segment; what
//! stopped it was that nothing said whether the MMA that follows sees the write.
//!
//! # The instrument
//!
//! [`kernels::mma_rescale`], one launch a row. `mm_abt` writes the segment, each
//! warp drains its own quadrant and stores it back scaled by [`RESCALE`], and a
//! second `mm_abt` with `accumulate = true` takes the same segment. Both MMAs
//! multiply the same operands, so every cell of the result is a multiple of the
//! identity `A·Bᵀ` already carries — `COLUMNS * row + column`, hardware-delivered
//! — and the host reads the multiple rather than a pass/fail:
//!
//! | multiple | what it says |
//! | ---: | --- |
//! | `RESCALE + 1` | the MMA accumulated onto the stored band. The claim. |
//! | `2` | the MMA accumulated onto the band **before** the rescale |
//! | `RESCALE` | the store landed and the second MMA did not |
//! | `1` | neither |
//!
//! `2` is the failure this exists to exclude, and it is also what the `no store`
//! row is *required* to produce: a case that cannot distinguish the two readings
//! would pass whatever the hardware did, so the control that says it can comes
//! before every row that rests on it.
//!
//! # The axes
//!
//! - **Which fences.** [`kernels::wg_handoff`]'s three bits — `store_wait`, then
//!   the `before` and `after` halves of the tcgen05 thread-sync fence — the same
//!   ladder [`crate::tmem_warpgroup`] runs, so the two families' answers are
//!   comparable layer for layer. That family's consumer is an LDTM read; this
//!   one's is a tensor core, which publishes and consumes through a different
//!   path, and `TmemTile::store_fragment`'s contract names both.
//! - **Who issues.** The `issuer` warp's own quadrant is the hand-off where one
//!   warp both stored and issued; the other three quadrants are the arrangement
//!   a kernel really has, where the MMA issuer is not the warp that owns the
//!   lanes. Every row measures both at once and the table counts them apart, and
//!   the issuer moves between rows so that an answer that depended on *which*
//!   quadrant is the issuer's would show up as a failure that moves with it.
//! - **Whether the wait is there.** Dropping `store_wait` is the negative
//!   control. It is a race rather than a contract, so it is reported and gates
//!   nothing — and if it comes back clean, that is not evidence the wait is
//!   unnecessary, only that this race resolved in order.

use std::error::Error;
use std::fmt::Write as _;

use cuda_core::CudaStream;
use cuda_core::DeviceBuffer;
use cuda_device::tma::TmaDescriptor;

use kittens::global::encode_bf16_panels;
use kittens::reg::{BaseLdtm, RegTile};

use crate::{
    COLUMNS, DEPTH, PROBE_SHARED, RESCALE, ROWS, TILE, accumulator_value, dump_index, kernels,
    launch_config, probe_a, probe_b,
};

/// The band each warp drains, stores back and is dumped from.
type Band = RegTile<32, COLUMNS, BaseLdtm>;
/// Warps a launch has, one TMEM quadrant each.
const WARPS: usize = ROWS / 32;
/// Floats the whole dump holds.
const DUMP: usize = WARPS * 32 * COLUMNS;

/// Every fence layer [`kernels::wg_handoff`] knows.
const FENCED: u32 = 7;

/// One launch: what it varies, and whether the case's verdict rests on it.
struct Row {
    name: &'static str,
    /// Whether the rescale is stored back at all. False is the control whose
    /// answer is the stale reading, so that the readings are known to differ.
    store: bool,
    fences: u32,
    /// The warp whose lane 0 issues the second MMA.
    issuer: u32,
    gated: bool,
}

/// The result of one row's launch, or the reason there isn't one.
struct Measured {
    name: &'static str,
    gated: bool,
    outcome: Outcome,
}

enum Outcome {
    /// Every value was the multiple of its cell the row names.
    Clean,
    /// `wrong` of [`DUMP`] values were not, `own` of them in the issuer's own
    /// quadrant, with the multiple every wrong one carried if there was one.
    Wrong {
        wrong: usize,
        own: usize,
        multiple: Option<f32>,
        detail: String,
    },
    /// The launch itself failed.
    Faulted(String),
    /// A row before this one took the context down.
    Unrun,
}

/// The table, in the order it is launched.
///
/// The control goes first for the usual reason and one more: it is the row that
/// says a stale accumulator reads differently from a rescaled one, and every row
/// under it is an assertion about which of the two came back.
fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "control: no store, the stale reading",
            store: false,
            fences: FENCED,
            issuer: 0,
            gated: true,
        },
        Row {
            name: "both fences, issuer stored too",
            store: true,
            fences: FENCED,
            issuer: 0,
            gated: true,
        },
        Row {
            name: "both fences, issuer is warp 2",
            store: true,
            fences: FENCED,
            issuer: 2,
            gated: true,
        },
        Row {
            name: "neither fence (what ships)",
            store: true,
            fences: 0b001,
            issuer: 0,
            gated: true,
        },
        Row {
            name: "neither fence, issuer is warp 2",
            store: true,
            fences: 0b001,
            issuer: 2,
            gated: true,
        },
        Row {
            name: "no fence::after_thread_sync",
            store: true,
            fences: 0b011,
            issuer: 2,
            gated: true,
        },
        Row {
            name: "no fence::before_thread_sync",
            store: true,
            fences: 0b101,
            issuer: 2,
            gated: true,
        },
        Row {
            name: "no store_wait",
            store: true,
            fences: 0b110,
            issuer: 2,
            gated: false,
        },
    ]
}

/// The multiple of the cell identity a row's dump must carry everywhere.
fn multiple(row: &Row) -> f32 {
    if row.store { RESCALE + 1.0 } else { 2.0 }
}

/// Launch one row and bring its dump back.
fn measure(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    row: &Row,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut out = DeviceBuffer::<f32>::zeroed(stream, DUMP)?;
    let config = launch_config(ROWS as u32, PROBE_SHARED as u32);
    unsafe {
        if row.store {
            module.mma_rescale(
                stream, config, a_map, b_map, row.fences, row.issuer, &mut out,
            )
        } else {
            module.mma_rescale_stale(
                stream, config, a_map, b_map, row.fences, row.issuer, &mut out,
            )
        }
    }?;
    Ok(out.to_host_vec(stream)?)
}

/// What the dump says, against the multiple the row names.
///
/// The multiple is the diagnosis worth having: a wrong value that is a *clean*
/// multiple of its own cell names the term that is missing, and the two terms
/// are the store and the MMA. A row whose wrong values share no single multiple
/// is a different failure — a misaddressed store, a lost lane — and the detail
/// lines say so by naming the cell.
fn diff(row: &Row, observed: &[f32]) -> Outcome {
    let want = multiple(row);
    let (slots, values) = (Band::SLOTS, Band::VALUES);
    let (mut wrong, mut own) = (0usize, 0usize);
    let mut multiples: Vec<f32> = Vec::new();
    let mut detail = String::new();
    for warp in 0..WARPS {
        for lane in 0..32u32 {
            for slot in 0..slots {
                for value in 0..values {
                    let (band_row, column) = Band::coordinate(lane, slot, value);
                    let cell = accumulator_value(32 * warp + band_row as usize, column as usize);
                    let got = observed[dump_index(warp, lane, slot, value, slots, values)];
                    if got == want * cell {
                        continue;
                    }
                    wrong += 1;
                    own += usize::from(warp as u32 == row.issuer);
                    // The zero cell is a multiple of nothing, so it is counted
                    // wrong and never allowed to name a term.
                    if cell != 0.0 && !multiples.contains(&(got / cell)) {
                        multiples.push(got / cell);
                    }
                    if wrong <= 6 {
                        let _ = write!(
                            detail,
                            "\n    warp {warp} lane {lane} slot {slot} value {value}: \
                             cell ({}, {column}) is {cell}, wanted {want}× and read {got}",
                            32 * warp + band_row as usize
                        );
                    }
                }
            }
        }
    }
    if wrong == 0 {
        return Outcome::Clean;
    }
    Outcome::Wrong {
        wrong,
        own,
        multiple: match multiples.as_slice() {
            [single] => Some(*single),
            _ => None,
        },
        detail,
    }
}

/// What a clean multiple means, said in terms of the term that is missing
/// rather than as a number.
fn reading(multiple: f32) -> String {
    let term = if multiple == 2.0 {
        "the MMA accumulated onto the band as it stood before the rescale"
    } else if multiple == RESCALE {
        "the store landed and the second MMA did not"
    } else if multiple == 1.0 {
        "neither the store nor the second MMA reached the segment"
    } else {
        return format!("every wrong value is {multiple}× its cell, which names no term");
    };
    format!("every wrong value is {multiple}× its cell: {term}")
}

fn table(measured: &[Measured]) -> String {
    let mut table = String::new();
    let _ = write!(
        table,
        "\n  an MMA accumulating onto a segment `tcgen05.st` wrote (#177) — both \
         MMAs\n  multiply the same operands, so a correct segment is \
         {}× its own cell\n  identity and the accumulator the store never \
         reached is 2×:\n  {:<38}{:>8}{:>8}{:>9}   {}",
        RESCALE + 1.0,
        "row",
        "gated",
        "wrong",
        "issuer's",
        "of the dump"
    );
    for row in measured {
        let (wrong, own, note) = match &row.outcome {
            Outcome::Clean => (
                "0".to_string(),
                "0".to_string(),
                "every cell at its multiple".to_string(),
            ),
            Outcome::Wrong {
                wrong,
                own,
                multiple: Some(single),
                ..
            } => (wrong.to_string(), own.to_string(), reading(*single)),
            Outcome::Wrong { wrong, own, .. } => (
                wrong.to_string(),
                own.to_string(),
                "no single multiple".to_string(),
            ),
            Outcome::Faulted(error) => ("—".to_string(), "—".to_string(), error.clone()),
            Outcome::Unrun => (
                "—".to_string(),
                "—".to_string(),
                "not run: the context was gone".to_string(),
            ),
        };
        let _ = write!(
            table,
            "\n  {:<38}{:>8}{wrong:>8}{own:>9}   {note}",
            row.name,
            if row.gated { "yes" } else { "report" },
        );
        // The ungated row's *content* is its finding, and it would otherwise be
        // measured and thrown away.
        if let (false, Outcome::Wrong { detail, .. }) = (row.gated, &row.outcome) {
            let _ = write!(table, "{detail}");
        }
    }
    let _ = write!(
        table,
        "\n  the `issuer's` column is the wrong values in the quadrant the MMA's \
         own warp\n  stored; the rest are quadrants stored by a warp that did \
         not issue.\n  {DUMP} floats a row, {WARPS} warps by 32 lanes by \
         {COLUMNS}."
    );
    table
}

pub fn check(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let a = DeviceBuffer::from_host(stream, &probe_a())?;
    let b = DeviceBuffer::from_host(stream, &probe_b())?;
    let a_map = unsafe { encode_bf16_panels::<ROWS, DEPTH>(stream, a.cu_deviceptr(), ROWS, 1)? };
    let b_map = unsafe { encode_bf16_panels::<TILE, DEPTH>(stream, b.cu_deviceptr(), TILE, 2)? };

    let rows = rows();
    let mut measured = Vec::with_capacity(rows.len());
    let mut gone = false;
    for row in &rows {
        let outcome = if gone {
            Outcome::Unrun
        } else {
            match measure(stream, module, row, a_map.as_ptr(), b_map.as_ptr()) {
                Ok(observed) => diff(row, &observed),
                Err(error) => {
                    // A tcgen05 fault is sticky on the context, so every row
                    // after this one would report the same error and none of
                    // them would mean anything.
                    gone = true;
                    Outcome::Faulted(error.to_string())
                }
            }
        };
        measured.push(Measured {
            name: row.name,
            gated: row.gated,
            outcome,
        });
    }

    let table = table(&measured);
    let mut failures = String::new();
    for row in &measured {
        let complaint = match (&row.outcome, row.gated) {
            (Outcome::Clean, _) | (_, false) => continue,
            (
                Outcome::Wrong {
                    wrong,
                    own,
                    multiple,
                    detail,
                },
                true,
            ) => format!(
                "{}: {wrong} values wrong, {own} of them in the issuer's own quadrant{}{detail}",
                row.name,
                match multiple {
                    Some(single) => format!(" — {}", reading(*single)),
                    None => String::new(),
                }
            ),
            (Outcome::Faulted(error), true) => format!("{}: the launch failed: {error}", row.name),
            (Outcome::Unrun, true) => format!("{}: never ran", row.name),
        };
        let _ = write!(failures, "\n  {complaint}");
    }

    if failures.is_empty() {
        Ok(format!(
            "an MMA reads the segment `tcgen05.st` wrote{table}"
        ))
    } else {
        Err(format!(
            "an MMA's view of a stored segment is not what it is written to be:{failures}{table}"
        )
        .into())
    }
}
