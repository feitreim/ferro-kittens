//! What has to sit between an `stmatrix` and the `ldmatrix` that reads it (#136)
//!
//! [`load_fragment`](kittens::ldst::load_fragment)'s safety contract asks the
//! caller for `fence.proxy.async.shared::cta` before an `ldmatrix` reads a tile
//! `stmatrix` wrote. Both instructions are **generic-proxy** accesses, and the
//! rule [`publish_to_async_proxy`](kittens::shared::publish_to_async_proxy)
//! states is that a generic-write/generic-read pair owes no proxy fence at all
//! — what it owes, when the writing and the reading threads differ, is a
//! *barrier*. So the contract either names a redundant instruction or is
//! standing in for an obligation it does not state, and no caller in this tree
//! pairs the two directions on one tile, so neither reading has ever run.
//!
//! # The instrument
//!
//! [`kernels::stmatrix_handoff`], one launch a row. Warp 0 fills a `[64, 64]`
//! tile twice — from a *stale* generation of identities, published with a fence
//! and a barrier so every warp is known to hold it, and then from the *fresh*
//! one — and a reader warp reads the whole tile back with `ldmatrix`. The two
//! generations are the same sixteen blocks carrying identities
//! [`HANDOFF_ROTATE`] columns apart, so what comes back names itself:
//!
//! | what a value is | what it says |
//! | --- | --- |
//! | its own identity | the read saw the store. The claim. |
//! | the stale identity | it did not — this row's synchronization is not enough |
//! | another position's | a misaddressed block, which is not a fence question |
//! | no identity at all | nothing was ever written there |
//!
//! The third line is why the rotation is 37 rather than something round:
//! `ldmatrix` lands on whole `[16, 16]` blocks, so no misaddressed read can
//! produce a displacement that is not a multiple of 16, and a stale value can
//! never be mistaken for one.
//!
//! # The axes
//!
//! - **Which layers.** [`HANDOFF_FENCE`] and [`HANDOFF_BARRIER`], each present
//!   or absent — the proxy fence the contract demands and the `bar.sync` the
//!   rule says is what is really owed.
//! - **Who reads.** The writing warp itself, where the two instructions are one
//!   warp's own program order; or another warp, where they are not. The
//!   contract makes no distinction and the rule turns on exactly this.
//!
//! A row with no barrier and a reader that is not the writer is a race and not
//! a contract — reported, gating nothing, for the reason `sttm into mma`'s
//! dropped-wait row is.
//!
//! # The control
//!
//! [`kernels::stmatrix_handoff_early`] runs the read *before* the fresh store,
//! with a barrier making that the order that happens. Its dump is required to be
//! the stale generation everywhere. A case whose two generations were
//! indistinguishable to the host would pass whatever the hardware did, so the
//! row that says they are not comes before every row that rests on it — and
//! unlike a dropped-wait control it is an ordering rather than a race, so it
//! cannot decline to fire.

use std::error::Error;
use std::fmt::Write as _;

use cuda_core::CudaStream;

use kittens::reg::Fragment;

use crate::{
    HANDOFF_BARRIER, HANDOFF_FENCE, HANDOFF_ROTATE, HANDOFF_THREADS, HANDOFF_WRITER, TILE,
    decode_cell, dump_index, handoff_cell, kernels, launch_config, tile_shared,
};
use kittens::watchdog::{self, ReadBack};

/// `[16, 16]` blocks the reader reads back, and the `warp` slot of
/// [`dump_index`] — one band a block, as `ldmatrix map` dumps.
const BLOCKS: usize = (TILE / 16) * (TILE / 16);
/// Values the whole dump holds.
const DUMP: usize = BLOCKS * 32 * Fragment::SLOTS * Fragment::VALUES;

/// One launch: what it varies, and whether the case's verdict rests on it.
struct Row {
    name: &'static str,
    /// Whether the read runs before the fresh store. True is the control, whose
    /// answer is the stale generation, so that the two are known to differ.
    early: bool,
    sync: u32,
    /// The warp that reads. [`HANDOFF_WRITER`] is the same-warp hand-off.
    reader: u32,
    gated: bool,
}

/// What one value of a dump turned out to be, in terms of the generation it
/// belongs to rather than as a number.
#[derive(PartialEq)]
enum Reading {
    /// The identity of the position it was read from, in the fresh generation.
    Fresh,
    /// That position's identity in the generation before the store.
    Stale,
    /// A real identity, belonging to some other position.
    Astray(usize, usize),
    /// No identity at all.
    Nothing,
}

/// The result of one row's launch, or the reason there isn't one.
struct Measured {
    name: &'static str,
    gated: bool,
    outcome: Outcome,
}

enum Outcome {
    /// Every value was the generation the row names.
    Clean,
    /// `wrong` of [`DUMP`] values were not, `stale` of them being the other
    /// generation rather than an addressing failure.
    Wrong {
        wrong: usize,
        stale: usize,
        detail: String,
    },
    /// The launch itself failed.
    Faulted(String),
}

/// The table, in the order it is launched.
///
/// The control goes first because every row under it is an assertion about
/// which of two generations came back, and it is the row that says they read
/// differently.
fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "control: the read runs first",
            early: true,
            sync: HANDOFF_FENCE | HANDOFF_BARRIER,
            reader: HANDOFF_WRITER + 1,
            gated: true,
        },
        Row {
            name: "same warp, nothing between them",
            early: false,
            sync: 0,
            reader: HANDOFF_WRITER,
            gated: true,
        },
        Row {
            name: "same warp, the proxy fence only",
            early: false,
            sync: HANDOFF_FENCE,
            reader: HANDOFF_WRITER,
            gated: true,
        },
        Row {
            name: "same warp, bar.sync only",
            early: false,
            sync: HANDOFF_BARRIER,
            reader: HANDOFF_WRITER,
            gated: true,
        },
        Row {
            name: "same warp, both (the contract)",
            early: false,
            sync: HANDOFF_FENCE | HANDOFF_BARRIER,
            reader: HANDOFF_WRITER,
            gated: true,
        },
        Row {
            name: "cross warp, bar.sync only",
            early: false,
            sync: HANDOFF_BARRIER,
            reader: HANDOFF_WRITER + 1,
            gated: true,
        },
        Row {
            name: "cross warp, both (the contract)",
            early: false,
            sync: HANDOFF_FENCE | HANDOFF_BARRIER,
            reader: HANDOFF_WRITER + 1,
            gated: true,
        },
        Row {
            name: "cross warp, nothing between them",
            early: false,
            sync: 0,
            reader: HANDOFF_WRITER + 1,
            gated: false,
        },
        Row {
            name: "cross warp, the proxy fence only",
            early: false,
            sync: HANDOFF_FENCE,
            reader: HANDOFF_WRITER + 1,
            gated: false,
        },
    ]
}

/// Launch one row and bring its dump back.
fn measure(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    row: &Row,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut out = watchdog::cleared::<f32>(stream, DUMP)?;
    let config = launch_config(HANDOFF_THREADS, tile_shared::<TILE, TILE>());
    unsafe {
        if row.early {
            module.stmatrix_handoff_early(stream, config, row.sync, row.reader, &mut out)
        } else {
            module.stmatrix_handoff(stream, config, row.sync, row.reader, &mut out)
        }
    }?;
    Ok(out.read_back(stream)?)
}

/// Which generation the value at one position belongs to.
fn read(row: usize, column: usize, got: f32) -> Reading {
    if got == handoff_cell(row, column, 0) {
        return Reading::Fresh;
    }
    if got == handoff_cell(row, column, HANDOFF_ROTATE) {
        return Reading::Stale;
    }
    match decode_cell(got) {
        Some((row, column)) => Reading::Astray(row, column),
        None => Reading::Nothing,
    }
}

/// What the dump says, against the generation the row names.
///
/// The counts are kept apart because they are different findings: a stale value
/// is an ordering the row did not have, and anything else is an address, which
/// no fence would have fixed.
fn diff(row: &Row, observed: &[f32]) -> Outcome {
    let want = if row.early {
        Reading::Stale
    } else {
        Reading::Fresh
    };
    let (slots, values) = (Fragment::SLOTS, Fragment::VALUES);
    let (mut wrong, mut stale) = (0usize, 0usize);
    let mut detail = String::new();
    for row_block in 0..TILE / 16 {
        for column_block in 0..TILE / 16 {
            let block = row_block * (TILE / 16) + column_block;
            for lane in 0..32u32 {
                for slot in 0..slots {
                    for value in 0..values {
                        let (tile_row, tile_column) = Fragment::coordinate(lane, slot, value);
                        let (tile_row, tile_column) = (
                            16 * row_block + tile_row as usize,
                            16 * column_block + tile_column as usize,
                        );
                        let got = observed[dump_index(block, lane, slot, value, slots, values)];
                        let reading = read(tile_row, tile_column, got);
                        if reading == want {
                            continue;
                        }
                        wrong += 1;
                        stale += usize::from(reading == Reading::Stale);
                        if wrong <= 6 {
                            let _ = write!(
                                detail,
                                "\n    block ({row_block}, {column_block}) lane {lane} slot \
                                 {slot} value {value}: ({tile_row}, {tile_column}) read {}",
                                match reading {
                                    Reading::Fresh => "the fresh generation".to_string(),
                                    Reading::Stale => "the stale generation".to_string(),
                                    Reading::Astray(row, column) =>
                                        format!("({row}, {column})'s identity"),
                                    Reading::Nothing => format!("{got}, which names no position"),
                                }
                            );
                        }
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
        stale,
        detail,
    }
}

/// What a row's counts mean, said as the term that is missing.
fn reading(row: &Row, wrong: usize, stale: usize) -> String {
    if stale == wrong {
        let missing = match row.sync {
            0 => "and nothing was asked to order them",
            HANDOFF_FENCE => "past the proxy fence alone",
            HANDOFF_BARRIER => "past bar.sync alone",
            _ => "past both the proxy fence and bar.sync",
        };
        return format!("every wrong value is the stale generation, {missing}");
    }
    if stale == 0 {
        return "no wrong value is the stale generation: these are addresses, not ordering"
            .to_string();
    }
    format!("{stale} stale and {} at some other position", wrong - stale)
}

fn table(measured: &[Measured], rows: &[Row]) -> String {
    let mut table = String::new();
    let _ = write!(
        table,
        "\n  what orders an `stmatrix` before the `ldmatrix` that reads it \
         (#136) — the\n  tile is written twice, and a value names the \
         generation that reached it:\n  {:<36}{:>8}{:>8}{:>8}   {}",
        "row", "gated", "wrong", "stale", "of the dump"
    );
    for (row, measured) in rows.iter().zip(measured) {
        let (wrong, stale, note) = match &measured.outcome {
            Outcome::Clean => (
                "0".to_string(),
                "0".to_string(),
                if row.early {
                    "every value the stale generation".to_string()
                } else {
                    "every value the fresh generation".to_string()
                },
            ),
            Outcome::Wrong { wrong, stale, .. } => (
                wrong.to_string(),
                stale.to_string(),
                reading(row, *wrong, *stale),
            ),
            Outcome::Faulted(error) => ("—".to_string(), "—".to_string(), error.clone()),
        };
        let _ = write!(
            table,
            "\n  {:<36}{:>8}{wrong:>8}{stale:>8}   {note}",
            measured.name,
            if measured.gated { "yes" } else { "report" },
        );
        // The ungated rows' *content* is their finding, and it would otherwise
        // be measured and thrown away.
        if let (false, Outcome::Wrong { detail, .. }) = (measured.gated, &measured.outcome) {
            let _ = write!(table, "{detail}");
        }
    }
    let _ = write!(
        table,
        "\n  {DUMP} values a row, {BLOCKS} blocks by 32 lanes; the stale \
         generation is\n  each position's identity {HANDOFF_ROTATE} columns \
         along, which no misaddressed\n  block can produce."
    );
    table
}

pub fn check(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let rows = rows();
    let mut measured = Vec::with_capacity(rows.len());
    for row in &rows {
        let outcome = match measure(stream, module, row) {
            Ok(observed) => diff(row, &observed),
            Err(error) => Outcome::Faulted(error.to_string()),
        };
        measured.push(Measured {
            name: row.name,
            gated: row.gated,
            outcome,
        });
    }

    let table = table(&measured, &rows);
    let mut failures = String::new();
    for (row, measured) in rows.iter().zip(&measured) {
        let complaint = match (&measured.outcome, measured.gated) {
            (Outcome::Clean, _) | (_, false) => continue,
            (
                Outcome::Wrong {
                    wrong,
                    stale,
                    detail,
                },
                true,
            ) => format!(
                "{}: {wrong} values wrong — {}{detail}",
                measured.name,
                reading(row, *wrong, *stale)
            ),
            (Outcome::Faulted(error), true) => {
                format!("{}: the launch failed: {error}", measured.name)
            }
        };
        let _ = write!(failures, "\n  {complaint}");
    }

    if failures.is_empty() {
        Ok(format!("`ldmatrix` reads what `stmatrix` wrote{table}"))
    } else {
        Err(format!(
            "an `ldmatrix` read of an `stmatrix`-written tile is not what it is \
             written to be:{failures}{table}"
        )
        .into())
    }
}
