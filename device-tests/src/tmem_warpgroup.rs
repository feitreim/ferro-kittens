//! May a warp of a *second* warpgroup read the TMEM lanes its opposite number
//! in the first warpgroup owns? (oxide-train#94)
//!
//! Everything else this library says about tensor memory is stated as warp
//! **ownership**: `TmemTile::tile`, `tile_x8`, `store_fragment` and their
//! relatives all carry the clause "all 32 lanes of the warp *owning* TMEM rows
//! `row..row + M` call this together", and every kernel in either repository
//! launches one warpgroup, whose four warps take the four quadrants of a 128-row
//! accumulator and never look at each other's. The clause has therefore never
//! been tested, because nothing has ever been in a position to violate it.
//!
//! oxide-train#94 is what puts something in that position. Its flash backward is
//! **pass-bound**: the phase probe on that issue prices the register pass at
//! 38–43% of a key-tile visit and the MMA issue at 42–49%, with every wait in the
//! kernel under 100 ticks out of 4 300. Its remedy 3 is to split the pass across
//! two warpgroups, each taking half the score band's columns, and it rests on one
//! sentence: *"the TMEM sub-partition a warp reads is `warp_id % 4`, so both
//! warpgroups address the same lanes and only the columns differ."* That reading
//! of the ISA is plausible and it is not this library's contract, and the gap
//! between the two is a kernel that either works or returns silently wrong
//! gradients.
//!
//! # The instrument
//!
//! One accumulator, [`WG_THREADS`] threads — two full warpgroups, the first
//! launch in this harness wider than one. Warpgroup 0 seeds all 128 columns of
//! its own quadrants with [`wg_cell`], whose value at `(lane, column)` is the
//! exact integer `lane * 512 + column + 1`. Warpgroup 1 then reads, and the host
//! decodes every value back to the cell it names.
//!
//! That identity is the whole instrument, and it is deliberately not
//! [`dump_index`]'s. `dump_index` names the *thread* that wrote a register,
//! which is the right identity for a round trip inside one warp and useless
//! here: the claim under test is that one warp reads what another wrote, so a
//! wrong answer has to say *which cell it reached*. Under [`wg_cell`] it does. A
//! read that lands in the wrong quadrant comes back naming that quadrant's
//! lanes, and the table below has a column for the offset rather than a column
//! for "wrong".
//!
//! # The rows, and what each one is for
//!
//! Four axes, one row per point that is worth a launch:
//!
//! - **Who reads.** Warpgroup 0 alone is the control that says the seed landed
//!   and the reader map is the shipped one. Warpgroup 1 alone is #94's minimum.
//!   Both at once is the concurrent case, which a read-only sharing story has to
//!   survive and which nothing here has ever run.
//! - **Which quadrant.** `shift = 0` is the opposite number. `shift = 1` is a
//!   quadrant no reading of the ISA gives that warp — the control that says
//!   whether the restriction is real at all, because a warp that can reach *any*
//!   quadrant was never constrained to its own and the contract is a different
//!   shape than either party thought. `warp 0 reaches lane 32` asks the same
//!   thing inside *one* warpgroup, so the answer cannot be read as something
//!   about the warpgroup boundary when it is something about the warp.
//! - **Which fences.** [`kernels::wg_handoff`]'s three bits, run as a ladder.
//!   `TmemTile::store_fragment`'s own documentation says the
//!   `tcgen05.fence::before_thread_sync` / `::after_thread_sync` pair around a
//!   cross-warp hand-off is **open** — the library uses neither on any path and
//!   does not claim they are unnecessary — so the way to close it is to take each
//!   layer away and read what survives.
//! - **Which shape.** The `[32, 16]` chunk the backward's register pass reads
//!   through `TmemTile::tile`, the `[32, 128]` band it drains through
//!   `tile_x8`, and the `[32, 64]` half the *forward* both reads and stores back
//!   through `rescale_half`. Toy shapes were not on the table: #94's kernel is
//!   written against those three and a `.x1` answer is not a `.x8` answer.
//!
//! Every row's expectation is a whole buffer, zero where nothing should have
//! been written, so a row also fails if a warpgroup that was told not to read
//! read anyway.
//!
//! # Why a wrong answer is a row and not a hang
//!
//! Reading a quadrant the ISA does not give a warp is undefined, and undefined
//! includes a fault. Nothing in a `tcgen05.ld` can be given a deadline the way
//! #190's multicast waits were — the load either retires or the launch is gone —
//! so the bound here is placed differently: **one launch per row**, the
//! undefined rows last, and a launch that faults is recorded as that row's
//! result with the rows after it marked unrun. The case still reports, and it
//! reports which row took the context down. That is also why it is registered
//! next to `repeated launch` rather than beside the other TMEM cases.

use std::error::Error;
use std::fmt::Write as _;

use cuda_core::CudaStream;

use kittens::reg::BaseLdtm;

use crate::{
    STTM_SHARED, WG_CHUNK, WG_CHUNKS, WG_COLUMNS, WG_DUMP, WG_MARK, WG_MMA_SHARED, WG_THREADS,
    WG_WARPS, kernels, launch_config, wg_cell, wg_decode, wg_index,
};
use kittens::watchdog::{self, ReadBack};

/// Warps a launch has, over both warpgroups.
const WARPS: u32 = WG_THREADS / 32;
/// Floats the whole dump holds.
const DUMP: usize = WARPS as usize * 32 * WG_DUMP;
/// Every fence layer [`kernels::wg_handoff`] knows: `store_wait`, then the
/// `before` and `after` halves of the tcgen05 thread-sync fence.
const FENCED: u32 = 7;
/// The offset stands in for "this value names no cell at all", which is not a
/// distance and must not collapse into one when the offsets are counted.
const NO_CELL: (i32, i32) = (i32::MIN, i32::MIN);

/// Which library path a row reads the accumulator through, and how the host
/// therefore builds its expectation.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// `TmemTile::tile` at `[32, WG_CHUNK]`, the score-band chunk. Under
    /// `split` each warpgroup takes half the chunks of one 64-column band —
    /// same lanes, disjoint columns, which is #94's remedy 3 exactly.
    Chunk { split: bool },
    /// `TmemTile::tile_x8` at `[32, WG_COLUMNS]`, the backward drain.
    Drain,
    /// Warpgroup 1 reads the upper `[32, 64]` half, stores a marked cell back
    /// into it, and warpgroup 0 reads both halves — the forward's
    /// `rescale_half`, across the warpgroup boundary.
    StoreBack,
    /// [`Shape::Drain`] over an accumulator the MMA wrote rather than
    /// `tcgen05.st`.
    Mma,
    /// Warp 0 alone, reading the `[16, 16]` block at lane 0 and then the one
    /// at lane 32, which is warp 1's — the same reach past a warp's own
    /// quadrant as `cross quadrant`, inside one warpgroup.
    PastQuadrant,
}

impl Shape {
    /// Shared bytes the row's kernel launches with.
    const fn shared(self) -> u32 {
        match self {
            Shape::Mma => WG_MMA_SHARED as u32,
            _ => STTM_SHARED as u32,
        }
    }
}

/// One launch: what it varies, and whether the case's verdict rests on it.
struct Row {
    name: &'static str,
    shape: Shape,
    /// Bit `g` set means warpgroup `g` reads.
    readers: u32,
    /// Quadrants the second warpgroup's read is rotated by. See the module doc.
    shift: u32,
    fences: u32,
    /// Whether a wrong value here fails the case. False for the rows whose
    /// subject is undefined behaviour: what they do is worth writing down and
    /// is not a thing to hold hardware to.
    gated: bool,
}

/// The result of one row's launch, or the reason there isn't one.
struct Measured {
    name: &'static str,
    gated: bool,
    outcome: Outcome,
}

enum Outcome {
    /// Every float in the dump was the cell the shape names.
    Clean,
    /// `wrong` of [`DUMP`] floats were not, with the offset every wrong one
    /// came from if there was exactly one such offset.
    Wrong {
        wrong: usize,
        offset: Option<(i32, i32)>,
        detail: String,
    },
    /// The launch itself failed.
    Faulted(String),
    /// A row before this one took the context down.
    Unrun,
}

/// The table, in the order it is launched.
///
/// Order is load-bearing twice. The control comes first, because nothing after
/// it means anything if the seed did not land. `cross quadrant` comes last,
/// because it is the one row that asks the hardware to do something no document
/// defines and is therefore the one row that can end the process.
fn rows() -> Vec<Row> {
    let chunk = Shape::Chunk { split: false };
    vec![
        Row {
            name: "control: warpgroup 0, own quadrant",
            shape: chunk,
            readers: 0b01,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "warpgroup 1, opposite number",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "both warpgroups, same lanes",
            shape: chunk,
            readers: 0b11,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "no fence::after_thread_sync",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: 0b011,
            gated: true,
        },
        Row {
            name: "no fence::before_thread_sync",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: 0b101,
            gated: true,
        },
        Row {
            name: "neither fence (what ships)",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: 0b001,
            gated: true,
        },
        Row {
            name: "no store_wait",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: 0b110,
            gated: false,
        },
        Row {
            name: "sync_threads alone",
            shape: chunk,
            readers: 0b10,
            shift: 0,
            fences: 0b000,
            gated: false,
        },
        Row {
            name: "#94's split: half the columns each",
            shape: Shape::Chunk { split: true },
            readers: 0b11,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "backward drain, tile_x8 [32, 128]",
            shape: Shape::Drain,
            readers: 0b11,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "forward's rescale_half, both ways",
            shape: Shape::StoreBack,
            readers: 0b11,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "MMA-written accumulator",
            shape: Shape::Mma,
            readers: 0b11,
            shift: 0,
            fences: FENCED,
            gated: true,
        },
        Row {
            name: "cross quadrant (undefined)",
            shape: chunk,
            readers: 0b10,
            shift: 1,
            fences: FENCED,
            gated: false,
        },
        Row {
            name: "warp 0 reaches lane 32 (undefined)",
            shape: Shape::PastQuadrant,
            readers: 0b01,
            shift: 0,
            fences: FENCED,
            gated: false,
        },
    ]
}

/// `RegTile<M, N, BaseLdtm>`'s slot and value counts, which the host needs to
/// walk a dump and which are `M / 8` and `N / 4` by that layout's definition.
const fn extent(rows: u32, columns: u32) -> (usize, usize) {
    ((rows / 8) as usize, (columns / 4) as usize)
}

/// The whole dump a row should produce: [`wg_cell`] wherever a band was read,
/// zero everywhere else.
///
/// This mirrors the kernel's own walk rather than deriving from it, which is
/// the point — a host that asked the device where it read could not catch a
/// device that read the wrong place.
fn expected(row: &Row) -> Vec<f32> {
    let mut want = vec![0.0f32; DUMP];
    let mut band = |warp: u32, quadrant: u32, base: usize, first: u32, rows: u32, columns: u32| {
        let (slots, values) = extent(rows, columns);
        for lane in 0..32 {
            for slot in 0..slots {
                for value in 0..values {
                    want[wg_index(warp, lane, base, slot, value, values)] = wg_cell(
                        32 * quadrant + BaseLdtm::row(lane, slot),
                        first + BaseLdtm::column(lane, value),
                    );
                }
            }
        }
    };

    match row.shape {
        Shape::Chunk { split } => {
            for group in 0..2 {
                if row.readers & (1 << group) == 0 {
                    continue;
                }
                let (first, chunks) = if split {
                    (group * WG_CHUNK * WG_CHUNKS / 2, WG_CHUNKS / 2)
                } else {
                    (0, WG_CHUNKS)
                };
                for warp in group * WG_WARPS..(group + 1) * WG_WARPS {
                    let quadrant = (warp + row.shift * group) % WG_WARPS;
                    for chunk in 0..chunks {
                        let column = first + WG_CHUNK * chunk;
                        band(
                            warp,
                            quadrant,
                            (WG_CHUNK * chunk) as usize,
                            column,
                            32,
                            WG_CHUNK,
                        );
                    }
                }
            }
        }
        Shape::Drain | Shape::Mma => {
            for group in 0..2 {
                if row.readers & (1 << group) == 0 {
                    continue;
                }
                for warp in group * WG_WARPS..(group + 1) * WG_WARPS {
                    let quadrant = (warp + row.shift * group) % WG_WARPS;
                    band(warp, quadrant, 0, 0, 32, WG_COLUMNS as u32);
                }
            }
        }
        Shape::StoreBack => {
            let half = WG_COLUMNS as u32 / 2;
            for warp in WG_WARPS..2 * WG_WARPS {
                // Warpgroup 1's own read of the upper half, before it writes.
                band(warp, warp % WG_WARPS, 0, half, 32, half);
            }
            for warp in 0..WG_WARPS {
                // The lower half, which nothing should have touched, and the
                // upper one, which warpgroup 1 marked.
                band(warp, warp, 0, 0, 32, half);
                band(warp, warp, half as usize, WG_MARK + half, 32, half);
            }
        }
        // Warp 0's own block, then the one at lane 32 — the second expectation
        // is the reading this row exists to test rather than one warp 0 is
        // entitled to.
        Shape::PastQuadrant => {
            band(0, 0, 0, 0, 16, WG_CHUNK);
            band(0, 1, 8, 0, 16, WG_CHUNK);
        }
    }
    want
}

/// Launch one row and bring its dump back.
fn measure(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    row: &Row,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut out = watchdog::cleared::<f32>(stream, DUMP)?;
    let config = launch_config(WG_THREADS, row.shape.shared());
    unsafe {
        match row.shape {
            Shape::Chunk { split } => module.warpgroup_chunk(
                stream,
                config,
                row.readers,
                row.shift,
                row.fences,
                u32::from(split),
                &mut out,
            )?,
            Shape::Drain => module.warpgroup_drain(
                stream,
                config,
                row.readers,
                row.shift,
                row.fences,
                &mut out,
            )?,
            Shape::StoreBack => {
                module.warpgroup_store_back(stream, config, row.fences, &mut out)?
            }
            Shape::Mma => {
                module.warpgroup_mma(stream, config, row.readers, row.fences, &mut out)?
            }
            Shape::PastQuadrant => module.warpgroup_past_quadrant(stream, config, &mut out)?,
        }
    }
    Ok(out.read_back(stream)?)
}

/// What the dump says, against what the shape names.
///
/// The offset is the diagnosis worth having: if every wrong value decodes to a
/// cell a fixed distance away, the read did not fail, it *landed somewhere
/// else*, and where it landed is the answer to the quadrant question. A row
/// whose wrong values have no single offset is a different failure and the
/// detail lines say so.
fn diff(want: &[f32], observed: &[f32]) -> Outcome {
    let mut wrong = 0usize;
    let mut offsets: Vec<(i32, i32)> = Vec::new();
    let mut detail = String::new();
    for (index, (&got, &expect)) in observed.iter().zip(want).enumerate() {
        if got == expect {
            continue;
        }
        wrong += 1;
        if let (Some(reached), Some(named)) = (wg_decode(got), wg_decode(expect)) {
            let offset = (
                reached.0 as i32 - named.0 as i32,
                reached.1 as i32 - named.1 as i32,
            );
            if !offsets.contains(&offset) {
                offsets.push(offset);
            }
        } else if !offsets.contains(&NO_CELL) {
            offsets.push(NO_CELL);
        }
        if wrong <= 6 {
            let (warp, rest) = (index / (32 * WG_DUMP), index % (32 * WG_DUMP));
            let (lane, slot) = (rest / WG_DUMP, rest % WG_DUMP);
            let show = |value: f32| match wg_decode(value) {
                Some((l, c)) => format!("lane {l} column {c}"),
                None if value == 0.0 => "nothing written".to_string(),
                None => format!("{value}, which names no cell"),
            };
            let _ = write!(
                detail,
                "\n    warp {warp} lane {lane} slot {slot}: wanted {}, read {}",
                show(expect),
                show(got)
            );
        }
    }
    if wrong == 0 {
        return Outcome::Clean;
    }
    let offset = match offsets.as_slice() {
        [single] if *single != NO_CELL => Some(*single),
        _ => None,
    };
    Outcome::Wrong {
        wrong,
        offset,
        detail,
    }
}

fn table(measured: &[Measured]) -> String {
    let mut table = String::new();
    let _ = write!(
        table,
        "\n  two warpgroups over one tcgen05 accumulator (oxide-train#94) — \
         warpgroup 0\n  seeds every lane with an integer naming its own cell, \
         and the reader's\n  values are decoded back to the cells they reached:\
         \n  {:<38}{:>8}{:>8}   {}",
        "row", "gated", "wrong", "of the dump"
    );
    for row in measured {
        let (wrong, note) = match &row.outcome {
            Outcome::Clean => ("0".to_string(), "every cell as seeded".to_string()),
            Outcome::Wrong {
                wrong,
                offset: Some((lane, column)),
                ..
            } => (
                wrong.to_string(),
                format!("all from lane {lane:+}, column {column:+}"),
            ),
            Outcome::Wrong { wrong, .. } => (wrong.to_string(), "no single offset".to_string()),
            Outcome::Faulted(error) => ("—".to_string(), error.clone()),
            Outcome::Unrun => ("—".to_string(), "not run: the context was gone".to_string()),
        };
        let _ = write!(
            table,
            "\n  {:<38}{:>8}{wrong:>8}   {note}",
            row.name,
            if row.gated { "yes" } else { "report" },
        );
        // The ungated rows are the ones whose *content* is the finding, and
        // theirs is the diagnosis a gated row would have carried into the error
        // string. Print it here or it is measured and thrown away.
        if let (false, Outcome::Wrong { detail, .. }) = (row.gated, &row.outcome) {
            let _ = write!(table, "{detail}");
        }
    }
    let _ = write!(
        table,
        "\n  gated rows are the contract; the others ask the hardware for \
         something no\n  document defines, and what they answer is written down \
         rather than required.\n  {DUMP} floats a row, {WARPS} warps by 32 lanes \
         by {WG_DUMP}."
    );
    table
}

pub fn check(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<String, Box<dyn Error>> {
    let rows = rows();
    let mut measured = Vec::with_capacity(rows.len());
    let mut gone = false;
    for row in &rows {
        let outcome = if gone {
            Outcome::Unrun
        } else {
            match measure(stream, module, row) {
                Ok(observed) => diff(&expected(row), &observed),
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
            (Outcome::Wrong { wrong, detail, .. }, true) => {
                format!("{}: {wrong} floats wrong{detail}", row.name)
            }
            (Outcome::Faulted(error), true) => format!("{}: the launch failed: {error}", row.name),
            (Outcome::Unrun, true) => format!("{}: never ran", row.name),
        };
        let _ = write!(failures, "\n  {complaint}");
    }

    // The table goes out on both paths, as the residency census's does: a row
    // that moved is exactly when the numbers are worth reading.
    if failures.is_empty() {
        Ok(format!(
            "a second warpgroup reads its opposite number's lanes{table}"
        ))
    } else {
        Err(format!("cross-warpgroup TMEM access is not what was assumed:{failures}{table}").into())
    }
}
