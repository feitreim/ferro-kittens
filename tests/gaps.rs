//! The landed half of `GAPS.md`, named once so it cannot rot in silence.
//!
//! `GAPS.md` is the file that says what does not exist, and its expensive
//! failure is the inverse of the one it is written to prevent: an entry still
//! reading *missing* for something that shipped, which costs the next reader a
//! re-implementation. #65 was one of those (§3.2, stale from before #6 landed),
//! #3 and #59 were two more, and no commit message contradicts any of them —
//! which is exactly why they survive.
//!
//! Absence cannot be asserted in Rust. There is no way to write "no second
//! `FragmentLayout` exists", so the half of `GAPS.md` that says *missing* stays
//! prose and stays dated. Presence can be asserted, so this file takes that
//! half: every symbol the sections below cite as landed is named here, inside a
//! generic function that is never instantiated. Nothing in it is monomorphized,
//! codegen'd or linked, which is what makes it safe to name warp shuffles and
//! `stmatrix` from a host test — the compiler checks that the names and their
//! signatures exist, and stops there.
//!
//! Deleting or renaming one of these fails `cargo test` next to the section
//! that claims it. That is the whole mechanism; it is not a harness and there
//! is nothing to register a new claim with beyond adding a line.
//!
//! Scoped to `reg`, `shared` and `ldst` — the three modules whose `GAPS.md`
//! entries #65 rewrote from *missing* to *landed*. The reduction and map
//! surfaces are also held behaviourally by `src/reg.rs`'s own tests and on a
//! B200 by `device-tests`' `reduction shuffles`; what those cannot catch is a
//! rename that leaves both sides consistent and `GAPS.md` wrong.

// Every item below is named for its signature, never called.
#![allow(dead_code)]

use kittens::reg::{Abs, Add, Exp2Approx, Fma, Max, Min, Mul};
use kittens::{
    BaseLdtm, ColLayout, ColVec, Element, ReduceOp, RegTile, RegVec, RowLayout, SharedVec,
};

/// §3.2 — `row_reduce`/`col_reduce`/`tile_reduce`, generic over a `ReduceOp`.
///
/// The entry this file was written for: §3.2 read "Missing: everything
/// column-wise, `prod`, `min`, generic `row_reduce`" for the whole of #6's
/// life. All three axes, all four foldable ops.
fn reductions_are_generic_over_reduce_op<const M: usize, const N: usize, L>(tile: RegTile<M, N, L>)
where
    L: RowLayout<M> + ColLayout<N>,
{
    let _: RegVec<M, L> = tile.row_reduce::<Add>();
    let _: ColVec<N, L> = tile.col_reduce::<Max>();
    let _: f32 = tile.tile_reduce::<Min>();
    let _: f32 = tile.tile_reduce::<Mul>();

    // The same folds one level down, on the vectors a tile reduce produces.
    let _: f32 = tile.row_reduce::<Add>().reduce::<Add>();
    let _: f32 = tile.col_reduce::<Add>().reduce::<Add>();
    let _: RegVec<M, L> = tile.row_reduce::<Add>().quad_reduce::<Max>();
    let _: ColVec<N, L> = tile.col_reduce::<Add>().column_group_reduce::<Max>();
}

/// §3.2 — the named wrappers, which are the whole of TK's reduction list.
fn every_named_reduction_tk_has<const M: usize, const N: usize, L>(tile: RegTile<M, N, L>)
where
    L: RowLayout<M> + ColLayout<N>,
{
    let _: [RegVec<M, L>; 4] = [
        tile.row_max(),
        tile.row_min(),
        tile.row_sum(),
        tile.row_prod(),
    ];
    let _: [ColVec<N, L>; 4] = [
        tile.col_max(),
        tile.col_min(),
        tile.col_sum(),
        tile.col_prod(),
    ];
    let _: [f32; 4] = [
        tile.tile_max(),
        tile.tile_min(),
        tile.tile_sum(),
        tile.tile_prod(),
    ];
}

/// §3.1 — the map entry points on `RegTile`, by-value and in place.
fn maps_reach_every_register_family<const M: usize, const N: usize, L>(
    mut tile: RegTile<M, N, L>,
    rows: RegVec<M, L>,
    cols: ColVec<N, L>,
) where
    L: RowLayout<M> + ColLayout<N>,
{
    let _ = tile.unary_map::<Exp2Approx>();
    let _ = tile.bin_map::<Add>(tile);
    let _ = tile.scalar_map::<Mul>(2.0);
    let _ = tile.ternary_map::<Fma>(tile, tile);
    let _ = tile.row_map::<Mul>(rows);
    let _ = tile.col_map::<Mul>(cols);
    tile.unary_map_assign::<Abs>();
    tile.bin_map_assign::<Add>(tile);
    tile.scalar_map_assign::<Mul>(2.0);
    tile.row_map_assign::<Mul>(rows);
    tile.col_map_assign::<Mul>(cols);
    let _ = RegTile::<M, N, L>::broadcast_row(rows);
    let _ = RegTile::<M, N, L>::broadcast_col(cols);
}

/// §1.4 — `SharedVec` as a first-class type: scalar access, both TMA
/// directions at both ranks, and the `ColVec` bridge in `ldst`.
///
/// This entry read "Absent entirely" until #65, three merged PRs after #53
/// shipped the type.
fn shared_vectors_move_and_are_read<E: Element, const N: usize, L: ColLayout<N>>(
    vec: SharedVec<E, N>,
    map: *const cuda_device::tma::TmaDescriptor,
    sem: kittens::Semaphore,
    lane: u32,
) {
    unsafe {
        let _: *mut u8 = vec.at(0);
        let _: f32 = vec.get(0);
        vec.set(0, 1.0);

        // Both loads hand back their own charge, and the only thing that takes
        // one is the barrier they complete on — #29's shape, named here
        // because a `tma_load` that went back to returning `()` would leave
        // every producer's `expect_tx` free to disagree with it again.
        sem.expect_tx(vec.tma_load(map, 0, sem) + vec.tma_load_2d(map, 0, 0, sem));
        vec.tma_store(map, 0);
        vec.tma_store_2d(map, 0, 0);

        let cols: ColVec<N, L> = kittens::ldst::load_vec(vec, lane);
        kittens::ldst::store_vec(vec, lane, cols);
    }
}

/// The one claim here cheap enough to assert rather than merely name: the
/// identities §3.2 says every fold seeds from. A `ReduceOp` whose identity is
/// wrong yields a plausible wrong number rather than a crash, which is the
/// failure mode the whole section is about.
#[test]
fn every_foldable_op_carries_a_neutral_identity() {
    fn neutral<Op: ReduceOp>(x: f32) {
        assert_eq!(Op::apply(Op::IDENTITY, x), x);
        assert_eq!(Op::apply(x, Op::IDENTITY), x);
    }
    for x in [-3.5, -1.0, 0.0, 0.5, 1.0, 7.25] {
        neutral::<Add>(x);
        neutral::<Mul>(x);
        neutral::<Max>(x);
        neutral::<Min>(x);
    }
    // `BaseLdtm` is still the only layout in tree (§1.3), so it is the one the
    // signatures above would be instantiated at if anything ever called them.
    let _ = RegTile::<32, 128, BaseLdtm>::zero();
}
