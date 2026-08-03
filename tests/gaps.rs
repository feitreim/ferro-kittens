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

/// §2.2 — the global movers are generic over the element, not fp32-only.
///
/// The 2026-07-29 audit's worst find, and the section's own failure mode again:
/// §2.2 read "It is **fp32 only**: `GlobalRows` names a `*mut f32` outright
/// rather than carrying an `Element`" for the whole of #108's life, having been
/// written as the reason a bf16 `C` was out of reach. `GlobalRows<E>` is what
/// makes the rounding happen once, in the store instruction, and naming it here
/// is what stops the entry going back to fp32 in prose.
///
/// `CONTIGUOUS_VALUES` is named beside them because the pairing (#91) is a
/// claim the *layout* makes and the movers spend; a layout that dropped it
/// would leave both of them silently scalar.
///
/// `load_cols` (#172) is the vector shape of the same path and the global half
/// of §1.4's "`RegVec` has no bridge to memory at all" — the per-column operand
/// §3.1 wants, read off one row instead of broadcast through a tile. It is
/// named here rather than beside `ldst::load_vec` because what it is generic
/// over is the *cursor*, not the element.
fn global_movers_carry_an_element<E: Element, const M: usize, const N: usize, L>(
    rows: kittens::global::GlobalRows<E>,
    lane: u32,
    tile: RegTile<M, N, L>,
) where
    L: RowLayout<M> + ColLayout<N>,
{
    unsafe {
        kittens::global::store_rows(rows, 0, 0, lane, tile);
        let _: RegTile<M, N, L> = kittens::global::load_rows(rows, 0, 0, lane);
        let _: ColVec<N, L> = kittens::global::load_cols(rows, 0, 0, lane);
        let _: usize = kittens::global::access_width(rows, 0);
        let _: bool = rows.runs_aligned(0, L::CONTIGUOUS_VALUES);
    }
}

/// §2.6 — shared → global with no engine between them (#113), and the wide
/// register→shared store the epilogue in front of it uses (#116).
///
/// `store_shared_rows` is the only mover in the crate whose *source* is shared
/// memory, and both shipping GEMMs' epilogues are made of these two calls. It
/// is named here because §2.6 says it landed and because the pair is what the
/// unwritten epilogue type of §7 would be built out of.
fn a_staged_drain_is_two_calls<E: Element<Unpacked = [f32; 2]>, const M: usize, const N: usize>(
    chunks: kittens::shared::SwizzledChunks<E>,
    tile: kittens::SharedTile<E, M, N, kittens::Swizzle128B>,
    rows: kittens::global::GlobalRows<E>,
    lane: u32,
    band: RegTile<M, N, BaseLdtm>,
) where
    BaseLdtm: RowLayout<M> + ColLayout<N>,
{
    unsafe {
        kittens::ldst::store_tile_x4(chunks, 0, 0, lane, band);
        kittens::ldst::store_fragment_x4(chunks, 0, 0, lane, kittens::Fragment::zero());
        kittens::global::store_shared_rows::<E, M, N, kittens::Swizzle128B, 32>(
            rows, 0, 0, lane, tile,
        );
    }
}

/// §2.1 — the `.x8` drain (#117), beside the `.x1` one it is eight times fewer
/// issues than.
fn the_tmem_drain_has_both_widths<const M: usize, const N: usize>(
    accumulator: kittens::TmemTile<M, N>,
) where
    BaseLdtm: RowLayout<M> + ColLayout<N>,
{
    unsafe {
        let _: RegTile<M, N, BaseLdtm> = accumulator.tile(0, 0);
        let _: RegTile<M, N, BaseLdtm> = accumulator.tile_x8(0, 0);
        let _: [kittens::Fragment; 4] = accumulator.fragments_x8(0, 0);
        accumulator.store_tile(0, 0, accumulator.tile(0, 0));
    }
}

/// §7 — the store ring's scope is in the type (#123), which is the one place in
/// the crate where a collective's scope is named rather than baked into its
/// index math.
///
/// Named here because §7's whole argument is that this is the shape the rest of
/// the crate has not taken, so a `Scope` that quietly went away would take the
/// argument with it.
fn the_store_ring_names_its_scope<SC: kittens::Scope>(
    ring: &mut kittens::StoreRing<kittens::Bf16, 128, 64, kittens::Swizzle128B, 1, SC>,
    map: *const cuda_device::tma::TmaDescriptor,
) {
    unsafe {
        let _: kittens::SharedTile<kittens::Bf16, 128, 64, kittens::Swizzle128B> = ring.acquire();
        ring.commit(map, 0, 0);
        ring.commit_2d(map, 0, 0);
        (*ring).drain();
    }
    let _: (bool, ()) = (SC::issuing(), SC::converge());
    let _: [u32; 2] = [
        kittens::StoreRing::<kittens::Bf16, 128, 64, kittens::Swizzle128B, 1, kittens::Cta>::DEPTH,
        kittens::StoreRing::<kittens::Bf16, 128, 64, kittens::Swizzle128B, 1, kittens::Warp>::DEPTH,
    ];
}

/// §7.2 — the thread identities and the proxy fence (#127), in the shape a
/// kernel makes them: a whole warp-scope store with nothing from `cuda_device`
/// in it.
///
/// The three names are the whole of the entry. `lane` is the argument every
/// function above takes, `32 * warp_id()` is the [`BaseLdtm`] band origin
/// under it, and `publish_to_async_proxy` is the fence six safety contracts in
/// the crate oblige the caller to issue — which only `StoreRing::publish` used
/// to make, privately. All five kernels opened with `use cuda_device::warp`
/// until these existed, so a rename that dropped one would put that import
/// back without failing anything else.
fn a_warp_scope_store_names_nothing_outside_kittens<E, const M: usize, const N: usize>(
    chunks: kittens::shared::SwizzledChunks<E>,
    band: RegTile<M, N, BaseLdtm>,
) where
    E: Element<Unpacked = [f32; 2]>,
    BaseLdtm: kittens::reg::FragmentLayout<M, N>,
{
    unsafe {
        kittens::ldst::store_tile(chunks, 32 * kittens::warp_id(), 0, kittens::lane(), band);
        kittens::shared::publish_to_async_proxy();
    }
}

/// §2.4 — a peer's barrier is addressable (#50) and a load can complete on it,
/// and the charge for a symmetric cluster stage is derived rather than written
/// down (#29).
fn a_cluster_stage_charges_one_barrier<E: Element, const R: usize, const C: usize>(
    tile: kittens::SharedTile<E, R, C, kittens::Swizzle128B>,
    map: *const cuda_device::tma::TmaDescriptor,
    sem: kittens::Semaphore,
) {
    unsafe {
        let peer = sem.at_rank(0);
        peer.arrive();
        let charge = tile.tma_load_2d_arriving_at(map, 0, 0, peer);
        sem.expect_tx(charge.across_ranks(2));
        let _ = tile.tma_load_2d_multicast_cg2(map, 0, 0, peer, 0b11);
    }
}

/// §3.3 — the operand-order square, and the `mm_*` twin of each walk (#12).
fn every_operand_order_has_an_entry_point<
    E: kittens::MmaElement,
    const K: usize,
    const M: usize,
    const N: usize,
>(
    tmem: u32,
    k_major: kittens::SharedTile<E, M, K, kittens::Swizzle128B>,
    mn_major: kittens::SharedTile<E, K, N, kittens::Swizzle128B>,
    shape: kittens::mma::MmaShape,
) {
    use kittens::mma::{mm_ab, mm_abt, mm_atb, mm_atbt, mma_ab, mma_abt, mma_atb, mma_atbt};
    unsafe {
        mma_abt(tmem, k_major, k_major, shape, true);
        mma_ab(tmem, k_major, mn_major, shape, true);
        mma_atb(tmem, mn_major, mn_major, shape, true);
        mma_atbt(tmem, mn_major, k_major, shape, true);
        mm_abt(tmem, k_major, k_major, shape);
        mm_ab(tmem, k_major, mn_major, shape);
        mm_atb(tmem, mn_major, mn_major, shape);
        mm_atbt(tmem, mn_major, k_major, shape);

        // The runtime-layout pair, whose walks carry their own transpose bits.
        let (a, b) = (k_major.k_walk(), mn_major.mn_walk());
        let _: [bool; 2] = [a.transposed(), b.transposed()];
        kittens::mma::mma_walk_cg2::<E, 4>(tmem, a, b, shape, true);
        kittens::mma::mm_walk_cg2::<E, 4>(tmem, a, b, shape);
    }
}

/// §4 — the scaffold, both item sources, the item map, and the rank scope
/// (#51, #88, #89). §4 read "still uncalled by anything in `examples/`" until
/// this audit; `gemm` has run on it since #83.
fn the_scaffold_has_two_item_sources<J: kittens::pipeline::Job>(
    job: &mut J,
    queue: kittens::pipeline::ClcQueue,
) {
    unsafe {
        kittens::pipeline::run(job, 64);
        kittens::pipeline::run_stealing(job, queue);
    }
    let _: (u32, u32) = kittens::pipeline::grouped(0, 4, 4, J::RANKS);
    let _: usize = kittens::pipeline::ClcQueue::BYTES;
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
