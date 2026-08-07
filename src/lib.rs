//! kittens: a ThunderKittens-style tile library for cuda-oxide, tcgen05-only.
//!
//! Kernels are written against typed shared, register and TMEM tiles with warp-
//! and warpgroup-scoped ops instead of raw intrinsics and hand-threaded index
//! math. The target is Blackwell (`sm_100a`) alone: the MMA layer is tcgen05,
//! with no wmma/wgmma backends and no arch dispatch.
//!
//! Everything here is an `#[inline(always)]` function or a `Copy` struct of
//! pointers and const generics; the crate ships no kernels and no
//! `#[cuda_module]`, so device code monomorphizes into the *calling* crate.
//!
//! A warp reading its band out of a shared tile, scaling it, writing it back:
//!
//! ```no_run
//! # use kittens::ldst::{load_tile, store_tile};
//! # use kittens::shared::{Bf16, SharedTile, Swizzle128B, publish_to_async_proxy};
//! # use kittens::{BaseLdtm, RegTile, lane, warp_id};
//! # unsafe fn kernel_body(tile: SharedTile<Bf16, 128, 64, Swizzle128B>) {
//! let (lane, row) = (lane(), 32 * warp_id());
//! let band: RegTile<32, 64, BaseLdtm> =
//!     unsafe { load_tile(tile.chunk_writer(), row, 0, lane) };
//! unsafe { store_tile(tile.chunk_writer(), row, 0, lane, band.scale(0.125)) };
//! unsafe { publish_to_async_proxy() };
//! # }
//! ```
//!
//! Design notes and measurements live in `docs/library/`.
//!
// `global`'s types are `feature = "host"`-gated, so the doc links pointing at
// them from across the crate resolve only in the host build. This suppresses
// *every* broken link off that feature, not just those — `cargo doc --features
// host` is the only real gate on them (CI.md, tier 1).
#![cfg_attr(not(feature = "host"), allow(rustdoc::broken_intra_doc_links))]
// `AccessWidth::RUNGS`, which is what makes a rung added to the access ladder a
// compile error at both drains now that they dispatch over the ladder's byte
// count and a wildcard rather than over the enum (`global`, #225). The
// toolchain is pinned to one nightly anyway — cuda-oxide's macros need it — so
// this costs nothing a consumer was not already paying.
#![feature(variant_count)]

pub mod epilogue;
pub mod global;
#[cfg(feature = "host")]
pub mod launch;
pub mod ldst;
pub mod mma;
pub mod pipeline;
pub mod plan;
pub mod reg;
pub mod shared;
pub mod sync;
pub mod tmem;

pub use epilogue::{Cta, Scope, StoreRing, Warp};
pub use plan::SharedPlan;
pub use reg::{
    BaseLdtm, BinaryOp, ColLayout, ColVec, Fragment, FragmentLayout, ReduceOp, RegTile, RegVec,
    RowLayout, TernaryOp, UnaryOp,
};
pub use shared::{
    Bf16, Element, F16, F32, MmaElement, OperandWalk, SharedCell, SharedTile, SharedTileRing,
    SharedVec, Swizzle, Swizzle128B,
};
pub use sync::{
    PhasedSemaphore, Semaphore, SemaphoreRing, TransactionBytes, block_reduce, block_reduce_sum,
};
pub use tmem::TmemTile;

/// The calling thread's lane in its warp — **the `lane` argument** every
/// warp-scope entry point in the crate takes.
///
/// Which lane holds which element of a tile is the layout's business, not the
/// caller's; the lane is only how a warp-collective op knows which thread it is
/// running on, and passing another thread's shifts a tile rather than faulting.
/// Read it once in the entry block and thread it through: against reading it
/// per op that is free at 128 columns and saves 16 registers at 32.
///
/// ```no_run
/// # use kittens::ldst::load_tile;
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # use kittens::{BaseLdtm, RegTile, lane, warp_id};
/// # unsafe fn demo(tile: SharedTile<Bf16, 128, 64, Swizzle128B>) {
/// let lane = lane();
/// let mut band: RegTile<32, 64, BaseLdtm> =
///     unsafe { load_tile(tile.chunk_writer(), 32 * warp_id(), 0, lane) };
/// band.mask(lane, f32::NEG_INFINITY, |row, column| column <= row);
/// # }
/// ```
#[inline(always)]
pub fn lane() -> u32 {
    cuda_device::warp::lane_id()
}

/// The calling warp's index in its block, `threadIdx.x / 32`.
///
/// `32 * warp_id()` is a [`BaseLdtm`] **band origin**: the layout gives a warp
/// 32 consecutive rows, so that product is the first row of this warp's band in
/// a shared tile, in a TMEM accumulator ([`TmemTile::tile`]), and in the global
/// rows a drain writes.
///
/// A *derived* value and not a hardware register, so it is the warp index only
/// for a **1-D block**; a block with a `threadIdx.y` gets the wrong answer here
/// as it would from the division written out.
///
/// ```no_run
/// # use kittens::ldst::load_tile;
/// # use kittens::shared::{Bf16, SharedTile, Swizzle128B};
/// # use kittens::{BaseLdtm, RegTile, lane, warp_id};
/// # unsafe fn demo(tile: SharedTile<Bf16, 128, 64, Swizzle128B>) {
/// let band: RegTile<32, 64, BaseLdtm> =
///     unsafe { load_tile(tile.chunk_writer(), 32 * warp_id(), 0, lane()) };
/// # }
/// ```
#[inline(always)]
pub fn warp_id() -> u32 {
    cuda_device::warp::warp_id()
}
