//! kittens: a ThunderKittens-style tile library for cuda-oxide, tcgen05-only.
//!
//! New kernels are written against typed shared/register/TMEM tiles with
//! warp- and warpgroup-scoped ops instead of raw intrinsics and hand-threaded
//! index math. The library targets Blackwell (`sm_100a`) exclusively: the MMA
//! layer is tcgen05, with no wmma/wgmma backends and no arch dispatch.
//!
//! Everything here is a plain `#[inline(always)]` function or a Copy struct
//! of pointers and const generics — the crate ships no kernels and no
//! `#[cuda_module]`. Device code monomorphizes into the *calling* crate's
//! artifact the same way `cuda-device` does, so a kernel crate pays nothing
//! for the abstraction unless ptxas says otherwise.
//!
//! Libdevice math is legal beside tcgen05 in the same pure-PTX artifact at
//! cuda-oxide b099f64. Software approximations remain where their lowering is
//! a measured kernel optimization rather than an artifact-path workaround.
//!
// `global`'s types are all `feature = "host"`, and docs across the crate link
// to them. Off that feature they are not compiled, so the links have nothing
// to resolve to; the host build is where they are checked.
#![cfg_attr(not(feature = "host"), allow(rustdoc::broken_intra_doc_links))]

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
    Bf16, Element, F32, MmaElement, OperandWalk, SharedTile, SharedTileRing, SharedVec, Swizzle,
    Swizzle128B,
};
pub use sync::{
    PhasedSemaphore, Semaphore, SemaphoreRing, TransactionBytes, block_reduce, block_reduce_sum,
};
pub use tmem::TmemTile;

/// The calling thread's lane in its warp — **the `lane` argument**.
///
/// Every warp-scope entry point in the crate takes one:
/// [`ldst::load_tile`], [`ldst::store_tile`], [`ldst::load_vec`],
/// [`global::store_rows`], [`RegTile::mask`], [`RegVec::row`], and the rest of
/// the register surface. Which lane holds which element of a
/// [`RegTile<M, N, BaseLdtm>`](RegTile) is the layout's business and not the
/// caller's; the lane is how a warp-collective op knows which thread it is
/// running on, and passing another thread's is how a tile ends up transposed
/// by 32 rows rather than faulting.
///
/// This wraps `cuda_device::warp::lane_id()` rather than re-exporting it, and
/// the reason is this paragraph: without it a kernel that calls nothing but
/// `kittens` still has to reach past the library for its first three lines,
/// and the argument's meaning is stated only in the six signatures that take
/// it (#127).
///
/// One `%laneid` read per warp is the convention — hoist it into the entry
/// block and thread it through, which `device-tests`' `lane probe` prices at
/// 0 registers at 128 columns and +16 at 32 against reading it per op. The
/// wrapper is `#[inline(always)]` over an `#[inline(never)]` lowering target,
/// so it is the same instruction either way.
#[inline(always)]
pub fn lane() -> u32 {
    cuda_device::warp::lane_id()
}

/// The calling warp's index in its block, `threadIdx.x / 32`.
///
/// `32 * warp_id()` is a [`BaseLdtm`] **band origin**: the layout gives a warp
/// 32 consecutive rows, so that product is the first row of this warp's band
/// in a shared tile, in a TMEM accumulator ([`TmemTile::tile`]), and in the
/// global rows a drain writes. It is the number every kernel in the repo
/// open-codes; the `32` is the layout's rows-per-warp, which the library does
/// not yet name (#126).
///
/// A *derived* value and not a hardware register, so it is only the warp index
/// for a **1-D block**. A block with a `threadIdx.y` gets the wrong answer
/// here as it would from the division written out.
#[inline(always)]
pub fn warp_id() -> u32 {
    cuda_device::warp::warp_id()
}
