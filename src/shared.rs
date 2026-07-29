//! Shared-memory tiles with the swizzle in the type.
//!
//! The layout scheme is the one flash-attention and GEMM kernels validated on
//! B200: a SWIZZLE_128B tile is stored as stacked 128-byte-row *subtiles* (64
//! bf16 columns each), so the swizzle phase inside each subtile equals the row
//! index — the coincidence a 64-wide panel gives for free, kept by
//! construction at every width. A `[R, 128]` bf16 operand is two stacked
//! `[R, 64]` subtiles a subtile-stride apart; `[R, 64]` operands (P/dS) are a
//! single subtile. Widths that are not a whole number of subtiles are a
//! compile error, not a differently-swizzled layout: restrict honestly
//! rather than pretend generality.
//!
//! Two facts of that layout the types keep straight:
//! - **Absolute phase.** SWIZZLE_128B XORs *physical* address bits `[9:7]`
//!   into the 16-byte-chunk index, so manual swizzled stores must fold in
//!   the tile base's own 128-byte row phase ([`SharedTile::swizzle_phase`]).
//! - **TMA loads a subtile per box.** The tensor maps built by [`crate::global`]
//!   describe 64-column boxes, so [`SharedTile::tma_load`] issues one
//!   `cp.async.bulk.tensor` per subtile, lifting the leading coordinate by 64
//!   per stack level. [`SharedTile::tma_store`] walks the same boxes the other
//!   way, and completes through a wholly different mechanism —
//!   [`tma_store_commit`] and the two waits, not a [`Semaphore`].
//!
//! [`SharedVec`] is the other shape shared memory holds, and it deliberately
//! shares none of that machinery: a vector is one flat run of elements, no
//! swizzle, one box. The reasoning is on the type.

use core::marker::PhantomData;

use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::cluster;
use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::ptx_asm;
use cuda_device::tcgen05::{
    Tcgen05ElementType, cvt_f32x2_bf16x2, tcgen05_mma_f16, tcgen05_mma_f16_cg2,
    tcgen05_mma_shared,
};
use cuda_device::tma::{
    TmaDescriptor, cp_async_bulk_commit_group, cp_async_bulk_tensor_1d_g2s,
    cp_async_bulk_tensor_1d_s2g, cp_async_bulk_tensor_2d_g2s,
    cp_async_bulk_tensor_2d_g2s_multicast_cg2, cp_async_bulk_tensor_2d_s2g,
    cp_async_bulk_tensor_3d_g2s, cp_async_bulk_tensor_3d_s2g, cp_async_bulk_wait_group,
    cp_async_bulk_wait_group_read,
};

use crate::sync::{ClusterSemaphore, Semaphore, TransactionBytes};

/// Element marker for tile types: the byte width every shape constant derives
/// from, plus how fp32 register values pack into the 32-bit words device code
/// actually moves. Device code never handles a value of the element type — a
/// tile is bytes in shared memory and packed words in registers — so the trait
/// describes the packing rather than a scalar.
pub trait Element {
    /// The fp32 values that fill one 32-bit packed word: `[f32; 2]` for a
    /// 2-byte element, `[f32; 4]` for an 8-bit one.
    ///
    /// An associated type rather than `[f32; PER_WORD]` because an array
    /// length must be a const expression of the parameters, which would need
    /// `generic_const_exprs` — the same dodge [`crate::reg::FragmentLayout`]
    /// takes for its storage. It also gives store paths a way to state their
    /// arity: the `stmatrix` path moves b16 matrices, so it bounds
    /// `E: Element<Unpacked = [f32; 2]>` and a 4-per-word element fails to
    /// typecheck there instead of packing silently wrong.
    type Unpacked: Copy;

    /// Bytes of one element in shared memory.
    const BYTES: usize;

    /// Elements per 32-bit packed word, derived from [`Self::Unpacked`] so the
    /// two cannot disagree.
    const PER_WORD: usize = size_of::<Self::Unpacked>() / size_of::<f32>();

    /// Pack one word's worth of fp32 values — the inverse of the unpack a
    /// TMEM drain does, with the first value in the low half.
    fn pack(values: Self::Unpacked) -> u32;

    /// Split one packed word back into fp32, the exact inverse of
    /// [`Self::pack`] for every element narrower than fp32 — widening loses
    /// nothing, so a load path is free of the rounding the store side has.
    fn unpack(word: u32) -> Self::Unpacked;

    /// Read one element of shared memory as the fp32 a register holds.
    ///
    /// The scalar half of the trait, beside [`Self::pack`]'s word half, and
    /// what a [`SharedVec`] is addressed with: a vector's consumers index
    /// *elements* — one column's parameter, one warp's partial — where a
    /// tile's move whole 32-bit words through `ldmatrix`. [`Self::Unpacked`]
    /// is an opaque `Copy` type with no way to pick a lane out of it, so the
    /// word form cannot serve an element index however the caller bounds it.
    ///
    /// # Safety
    ///
    /// `at` must point at a live element of `Self`, aligned to
    /// [`Self::BYTES`].
    unsafe fn read(at: *const u8) -> f32;

    /// Write one fp32 as a single element, rounding exactly as [`Self::pack`]
    /// does.
    ///
    /// A byte pointer rather than a word so a narrow element's neighbours are
    /// untouched: two lanes writing adjacent elements of a vector must not
    /// read-modify-write one word and lose each other's value.
    ///
    /// # Safety
    ///
    /// As [`Self::read`], and writable.
    unsafe fn write(at: *mut u8, value: f32);

    /// Write two *adjacent* elements in one memory access — what
    /// [`crate::reg::ColLayout::CONTIGUOUS_VALUES`] is spent on, and the only
    /// thing about a fragment mover that depends on the element.
    ///
    /// The default is the two scalar writes it replaces, so an element that
    /// has no wider spelling is correct without stating one. Both elements
    /// here do have one and they are different in kind: a 2-byte element packs
    /// the pair into a single word ([`Self::pack`]), where fp32 needs a vector
    /// instruction to move two.
    ///
    /// **The overrides are global-memory instructions**, which is what the
    /// contract below says and the reason there is no shared-memory caller: a
    /// `SharedVec`'s neighbouring elements belong to different lanes, and this
    /// pairs values one lane owns.
    ///
    /// # Safety
    ///
    /// `at` must be aligned to `2 * BYTES` and name two writable elements of a
    /// global buffer.
    #[inline(always)]
    unsafe fn write_pair(at: *mut u8, first: f32, second: f32) {
        unsafe {
            Self::write(at, first);
            Self::write(at.add(Self::BYTES), second);
        }
    }

    /// The read direction of [`Self::write_pair`], under the same contract.
    ///
    /// # Safety
    ///
    /// As [`Self::write_pair`], reading instead of writing.
    #[inline(always)]
    unsafe fn read_pair(at: *const u8) -> (f32, f32) {
        unsafe { (Self::read(at), Self::read(at.add(Self::BYTES))) }
    }
}

/// `collector::a::discard` — the `COLLECTOR_A` selector every walk here
/// issues under, and tcgen05's own default: nothing in this crate reuses an
/// `A` operand across instructions, so no chain has a collector buffer to
/// keep alive.
const COLLECTOR_DISCARD: u32 = 0;

/// Element types a tcgen05 MMA accepts as an operand.
///
/// Split from [`Element`] because operand kind is an MMA property and a tile
/// that only moves bytes never reaches an MMA — the same split, for the same
/// reason, as [`crate::reg::RowLayout`] out of [`crate::reg::FragmentLayout`].
///
/// The element is also where the *instruction* is selected, not just the
/// operand format: [`Self::mma`] and [`Self::mma_cg2`] are the whole of
/// [`crate::mma`]'s routing to silicon (#12). That is deliberate — it is what
/// makes a new operand type a new impl of this trait rather than a new MMA
/// layer — and it is why the routing is a *method* and not
/// `tcgen05_mma_shared::<{ Self::MMA_KIND }, ..>` at the call site: Rust
/// cannot pass an associated const of a type *parameter* as a const-generic
/// argument without `generic_const_exprs`. An impl writing `Self::MMA_KIND`
/// for itself is legal and is what keeps the const and the instruction from
/// drifting apart.
pub trait MmaElement: Element {
    /// The operand kind tcgen05's `KIND` const selects: 0 = f16, 1 = tf32,
    /// 2 = f8f6f4, 3 = i8. A `u32` to match the const-generic parameter of
    /// `tcgen05_mma_shared`/`tcgen05_mma_tensor` exactly, so an impl's
    /// [`Self::mma`] is a substitution and not a cast.
    const MMA_KIND: u32;

    /// The instruction descriptor's `atype`/`btype` field, which settles the
    /// operand's format *within* its [`Self::MMA_KIND`] — f16 and bf16 share
    /// kind 0 and are told apart only here. Reading the same bits under the
    /// wrong one produces a full accumulator of wrong numbers, so it belongs
    /// to the element rather than to whoever assembles the descriptor.
    const ELEMENT_TYPE: Tcgen05ElementType;

    /// One `cta_group::1` MMA of this element's kind: accumulate
    /// `a_desc · b_desc` into the TMEM accumulator at `tmem` under
    /// `instruction`, adding to what is already there when `enable_d`.
    ///
    /// # Safety
    ///
    /// Exactly one thread issues this. `tmem` must name an allocation the
    /// instruction descriptor's shape fits, both operand descriptors must
    /// describe committed shared memory of `Self`, and the descriptor's
    /// `atype`/`btype` must be [`Self::ELEMENT_TYPE`] — reading the bytes
    /// under another element's format faults nothing and computes garbage.
    unsafe fn mma(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool);

    /// [`Self::mma`] under `cta_group::2`: one instruction from the leader
    /// CTA drives the pair's shared accumulator, each CTA supplying its own
    /// operand halves at the same shared offsets.
    ///
    /// # Safety
    ///
    /// As [`Self::mma`], from the leader CTA's issuing thread only, with the
    /// cluster's peer holding its halves at those offsets.
    unsafe fn mma_cg2(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool);
}

/// IEEE fp16 staged operands.
///
/// This is the other tcgen05 kind-0 format beside [`Bf16`]. Keeping it as its
/// own element makes the tensor map, shared tile, operand descriptor, and MMA
/// instruction agree on FP16 without a format flag at any call site.
pub struct F16;

impl Element for F16 {
    type Unpacked = [f32; 2];
    const BYTES: usize = 2;

    #[inline(always)]
    fn pack(values: [f32; 2]) -> u32 {
        cvt_f16x2_f32(values[0], values[1])
    }

    #[inline(always)]
    fn unpack(word: u32) -> [f32; 2] {
        [f16_to_f32(word as u16), f16_to_f32((word >> 16) as u16)]
    }

    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f16_to_f32(unsafe { *(at as *const u16) })
    }

    #[inline(always)]
    unsafe fn write(at: *mut u8, value: f32) {
        unsafe { *(at as *mut u16) = Self::pack([value, value]) as u16 }
    }

    #[inline(always)]
    unsafe fn write_pair(at: *mut u8, first: f32, second: f32) {
        unsafe { *(at as *mut u32) = Self::pack([first, second]) }
    }

    #[inline(always)]
    unsafe fn read_pair(at: *const u8) -> (f32, f32) {
        let [first, second] = Self::unpack(unsafe { *(at as *const u32) });
        (first, second)
    }
}

impl MmaElement for F16 {
    const MMA_KIND: u32 = 0;
    const ELEMENT_TYPE: Tcgen05ElementType = Tcgen05ElementType::F16;

    #[inline(always)]
    unsafe fn mma(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool) {
        unsafe { tcgen05_mma_f16(tmem, a_desc, b_desc, instruction, enable_d) }
    }

    #[inline(always)]
    unsafe fn mma_cg2(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool) {
        unsafe { tcgen05_mma_f16_cg2(tmem, a_desc, b_desc, instruction, enable_d) }
    }
}

/// Expand one IEEE fp16 bit pattern to fp32 with integer bit operations.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits as u32) >> 10) & 0x1f;
    let fraction = (bits as u32) & 0x03ff;
    let widened = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((127 - 14 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + (127 - 15)) << 23) | (fraction << 13)
    };
    f32::from_bits(widened)
}

/// bf16 — the original staged-operand element used by the training kernels.
pub struct Bf16;

impl Element for Bf16 {
    type Unpacked = [f32; 2];
    const BYTES: usize = 2;

    #[inline(always)]
    fn pack(values: [f32; 2]) -> u32 {
        cvt_f32x2_bf16x2(values[0], values[1])
    }

    /// bf16 is fp32's leading 16 bits, so widening is a shift and no
    /// instruction — `cuda-device` exposes conversions in the narrowing
    /// direction only, and there is nothing here for one to do. Unlike
    /// [`Self::pack`] this is ordinary bit math, so it holds on the host too.
    #[inline(always)]
    fn unpack(word: u32) -> [f32; 2] {
        [
            f32::from_bits(word << 16),
            f32::from_bits(word & 0xffff_0000),
        ]
    }

    /// The same widening shift as [`Self::unpack`], off a 2-byte load — and
    /// host-testable for the same reason.
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f32::from_bits((unsafe { *(at as *const u16) } as u32) << 16)
    }

    /// [`Self::pack`]'s low half, so a value stored one at a time and a value
    /// stored two at a time round identically. Device-only, as `pack` is.
    #[inline(always)]
    unsafe fn write(at: *mut u8, value: f32) {
        unsafe { *(at as *mut u16) = cvt_f32x2_bf16x2(value, value) as u16 }
    }

    /// Two adjacent bf16 **are** one packed word, so the pair is a plain
    /// 4-byte store of [`Self::pack`] and needs no vector instruction: one
    /// `cvt.rn.bf16x2.f32` and one `st.global.u32` where the fp32 element
    /// needs a `st.global.v2.f32` carrying twice the bytes.
    #[inline(always)]
    unsafe fn write_pair(at: *mut u8, first: f32, second: f32) {
        unsafe { *(at as *mut u32) = Self::pack([first, second]) }
    }

    #[inline(always)]
    unsafe fn read_pair(at: *const u8) -> (f32, f32) {
        let [first, second] = Self::unpack(unsafe { *(at as *const u32) });
        (first, second)
    }
}

impl MmaElement for Bf16 {
    /// bf16 rides the f16 operand kind; bf16-vs-f16 is settled by the
    /// instruction descriptor's element-type field, not by `KIND`.
    const MMA_KIND: u32 = 0;
    const ELEMENT_TYPE: Tcgen05ElementType = Tcgen05ElementType::BF16;

    #[inline(always)]
    unsafe fn mma(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool) {
        unsafe {
            tcgen05_mma_shared::<{ Self::MMA_KIND }, 1, COLLECTOR_DISCARD>(
                tmem,
                a_desc,
                b_desc,
                instruction,
                enable_d,
            )
        }
    }

    #[inline(always)]
    unsafe fn mma_cg2(tmem: u32, a_desc: u64, b_desc: u64, instruction: u32, enable_d: bool) {
        unsafe {
            tcgen05_mma_shared::<{ Self::MMA_KIND }, 2, COLLECTOR_DISCARD>(
                tmem,
                a_desc,
                b_desc,
                instruction,
                enable_d,
            )
        }
    }
}

/// fp32 — the element a *statistic* is held at, not one an MMA reads.
///
/// The register side of this crate is fp32 throughout, so this is the element
/// whose [`Element::pack`] and [`Element::unpack`] are the identity and whose
/// [`Element::read`]/[`Element::write`] are a plain load and store. It carries
/// no rounding, which is exactly why a [`SharedVec<F32, N>`] is what a block
/// reduction stages its per-warp partials in
/// ([`crate::sync::block_reduce`]): a partial that went through shared memory
/// as bf16 would come back with eight bits of the sum gone, and the second of
/// two chained statistics would inherit the error of the first.
///
/// Deliberately not an [`MmaElement`]: tcgen05 reads fp32 operands as tf32,
/// which is a different element with a different mantissa, and giving this one
/// an `MMA_KIND` would let a `[R, C]` fp32 tile be staged as an operand it is
/// not the bits of.
pub struct F32;

impl Element for F32 {
    /// One value a word, where bf16's is two — the packed word *is* the value.
    /// The `Element<Unpacked = [f32; 2]>` bound `ldst`'s `stmatrix`/`ldmatrix`
    /// paths carry is what stops this element reaching them, since a b16
    /// matrix instruction has nothing to do with a 4-byte element.
    type Unpacked = [f32; 1];
    const BYTES: usize = 4;

    #[inline(always)]
    fn pack(values: [f32; 1]) -> u32 {
        values[0].to_bits()
    }

    #[inline(always)]
    fn unpack(word: u32) -> [f32; 1] {
        [f32::from_bits(word)]
    }

    /// A plain 4-byte load. Unlike [`Bf16::read`] there is no widening to do,
    /// and unlike [`Bf16::write`] the store side is exact — so both halves hold
    /// on the host and the pair is host-testable end to end.
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        unsafe { *(at as *const f32) }
    }

    #[inline(always)]
    unsafe fn write(at: *mut u8, value: f32) {
        unsafe { *(at as *mut f32) = value }
    }

    /// `st.global.v2.f32` — the two fp32 at `at` and `at + 4` in one
    /// instruction.
    ///
    /// Inline PTX for the reason [`crate::ldst::stmatrix_m8n8_x2`] is: the
    /// instruction has to be *asked for*. Widening a pair of adjacent stores is
    /// a transformation ptxas may only make when the address is provably
    /// aligned, and an address built from a runtime leading dimension never is
    /// — so the caller that has actually checked
    /// ([`crate::global::GlobalRows::runs_aligned`]) is the only one in a
    /// position to spell it. Where [`Bf16::write_pair`] moves a pair in four
    /// bytes, this one moves it in eight; the instruction count is the same and
    /// that is the whole of what a narrower `C` changes here (#108).
    #[inline(always)]
    unsafe fn write_pair(at: *mut u8, first: f32, second: f32) {
        unsafe {
            ptx_asm!(
                "st.global.v2.f32 [%0], {%1, %2};",
                in("l") at as u64,
                in("f") first,
                in("f") second,
                clobber("memory"),
            );
        }
    }

    /// `ld.global.v2.f32` — the read direction of [`Self::write_pair`].
    #[inline(always)]
    unsafe fn read_pair(at: *const u8) -> (f32, f32) {
        unsafe {
            let first: f32;
            let second: f32;
            ptx_asm!(
                "ld.global.v2.f32 {%0, %1}, [%2];",
                out("=f") first,
                out("=f") second,
                in("l") at as u64,
                clobber("memory"),
            );
            (first, second)
        }
    }
}

/// Swizzle mode marker. Only `SWIZZLE_128B` is implemented: it is the only
/// mode the validated kernels use, and the subtile scheme depends on its
/// 128-byte atom.
pub trait Swizzle {
    /// Bytes of one swizzle atom — the physical row width of a subtile.
    const ATOM_BYTES: usize;
    /// The mode's encoding in a tcgen05 shared-memory operand descriptor.
    const DESCRIPTOR_MODE: u8;
}

/// 128-byte swizzle: 16-byte chunks XOR physical address bits `[9:7]`.
pub struct Swizzle128B;
impl Swizzle for Swizzle128B {
    const ATOM_BYTES: usize = 128;
    const DESCRIPTOR_MODE: u8 = 2;
}

/// A `[R, C]` shared-memory tile of `E` elements under swizzle `S`, stored
/// as `C / (ATOM_BYTES / E::BYTES)` stacked `[R, subtile]` panels. The handle
/// is a base pointer plus compile-time shape — Copy, register-resident, and
/// free once inlined.
pub struct SharedTile<E: Element, const R: usize, const C: usize, S: Swizzle> {
    base: *mut u8,
    _marker: PhantomData<(E, S)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle> Clone for SharedTile<E, R, C, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle> Copy for SharedTile<E, R, C, S> {}

impl<E: Element, const R: usize, const C: usize, S: Swizzle> SharedTile<E, R, C, S> {
    /// Columns of one subtile (64 for bf16 under SWIZZLE_128B).
    pub const SUBTILE_COLS: usize = S::ATOM_BYTES / E::BYTES;
    /// Stacked subtiles in this tile.
    pub const SUBTILES: usize = C / Self::SUBTILE_COLS;
    /// Bytes of one `[R, SUBTILE_COLS]` subtile.
    pub const SUBTILE_BYTES: usize = R * S::ATOM_BYTES;
    /// Bytes of the whole tile — the shared-memory footprint, and what a TMA
    /// load of it charges.
    pub const BYTES: usize = R * C * E::BYTES;

    /// What a TMA load of the whole tile charges its barrier, as the receipt
    /// [`Semaphore::expect_tx`] takes: one box per subtile, `R` rows each.
    /// Equal to [`Self::BYTES`], counted the way the load loop issues it.
    ///
    /// Crate-private, because a kernel able to name a charge without issuing
    /// the load is back to writing the number down.
    pub(crate) const CHARGE: TransactionBytes =
        TransactionBytes::new(Self::SUBTILES * R * S::ATOM_BYTES);

    const WIDTH_OK: () = assert!(
        C.is_multiple_of(S::ATOM_BYTES / E::BYTES),
        "tile width must be a whole number of swizzle subtiles"
    );

    /// Wrap a raw shared-memory base (a `DynamicSharedArray` offset or a
    /// `SharedArray` static).
    ///
    /// # Safety
    ///
    /// `base` must point to at least `Self::BYTES` bytes of shared memory,
    /// 128-byte aligned, living as long as every use of the tile.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut u8) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::WIDTH_OK;
        Self {
            base,
            _marker: PhantomData,
        }
    }

    /// The tile's base address.
    #[inline(always)]
    pub const fn base(self) -> *mut u8 {
        self.base
    }

    /// Base address of stacked subtile `i`.
    ///
    /// # Safety
    ///
    /// `i < Self::SUBTILES`.
    #[inline(always)]
    pub unsafe fn subtile(self, i: usize) -> *mut u8 {
        unsafe { self.base.add(i * Self::SUBTILE_BYTES) }
    }

    /// TMA the tile from a [`crate::global`] panel map: one box per subtile,
    /// the leading (column) coordinate lifted by `SUBTILE_COLS` per stack
    /// level, `row` selecting the global row range and `plane` the panel.
    /// Completion lands on `sem`, and the call hands back the charge for it —
    /// once per tile, however many boxes — to be summed into that barrier's
    /// [`Semaphore::expect_tx`].
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches
    /// `[R, SUBTILE_COLS]`, and `sem` must be an initialized TMA barrier.
    #[inline(always)]
    pub unsafe fn tma_load(
        self,
        map: *const TmaDescriptor,
        row: i32,
        plane: i32,
        sem: Semaphore,
    ) -> TransactionBytes {
        unsafe { self.tma_load_at::<R>(map, 0, row, plane, sem) }
    }

    /// [`Self::tma_load`] landing at subtile row `dst_row` instead of the top —
    /// how a tile taller than the map's box gets built out of several global
    /// row ranges (the backward kernels stack two adjacent 64-row tiles into
    /// one 128-row operand, `dst_row = 0` then `dst_row = 64`).
    ///
    /// The stacking is layout-free precisely because a box height is a whole
    /// number of swizzle periods (8 rows of 128 bytes): landing a box at row
    /// 64 of a subtile reproduces exactly the swizzle rows 64.. of one tall
    /// tile would have had.
    ///
    /// `BOX_ROWS` is the *map's* box height and so cannot be read off this
    /// tile — a 128-row operand built this way is paired with a 64-row map —
    /// which is why it is a parameter rather than `R`. It is a const one
    /// because the charge handed back is derived from it: this call brings in
    /// `SUBTILES` boxes of `BOX_ROWS` rows, not a whole [`Self::BYTES`], and
    /// that used to be a sentence asking the caller to do the multiplication.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load`], plus `map`'s box being `BOX_ROWS` tall,
    /// `dst_row + BOX_ROWS <= R`, and `dst_row` a multiple of the 8-row
    /// swizzle period.
    #[inline(always)]
    pub unsafe fn tma_load_at<const BOX_ROWS: usize>(
        self,
        map: *const TmaDescriptor,
        dst_row: usize,
        row: i32,
        plane: i32,
        sem: Semaphore,
    ) -> TransactionBytes {
        const {
            assert!(
                BOX_ROWS <= R,
                "a box taller than the tile cannot land in it"
            )
        };
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_3d_g2s(
                    self.subtile(i).add(dst_row * S::ATOM_BYTES),
                    map,
                    (i * Self::SUBTILE_COLS) as i32,
                    row,
                    plane,
                    sem.raw(),
                );
                i += 1;
            }
        }
        TransactionBytes::new(Self::SUBTILES * BOX_ROWS * S::ATOM_BYTES)
    }

    /// TMA the tile from a 2d tensor map: one box per subtile, the leading
    /// coordinate lifted by `SUBTILE_COLS` per stack level. A K-major operand
    /// (`[R, K]`, one subtile per swizzle atom of K) is a single box at
    /// `(k, row)`; an MN-major operand (`[K, MN]`) is one box per 64-wide MN
    /// subtile at `(mn + 64 * i, k)` — the coordinate order the map's fast
    /// axis dictates, which is why both coordinates are the caller's.
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches
    /// `[R, SUBTILE_COLS]`, and `sem` must be an initialized TMA barrier.
    #[inline(always)]
    pub unsafe fn tma_load_2d(
        self,
        map: *const TmaDescriptor,
        leading: i32,
        minor: i32,
        sem: Semaphore,
    ) -> TransactionBytes {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_g2s(
                    self.subtile(i),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                    sem.raw(),
                );
                i += 1;
            }
        }
        Self::CHARGE
    }

    /// [`Self::tma_load_2d`] completing on another CTA's barrier: the boxes
    /// land in *this* CTA's tile as always, and the transaction bytes are
    /// counted at `sem` — any rank's copy of a stage barrier, named by
    /// [`Semaphore::at_rank`].
    ///
    /// It is a cta_group::2 multicast whose mask is the calling CTA's own bit
    /// and nothing else, which looks like a contradiction and is not. Nothing
    /// is replicated: one bit, one destination. What the multicast form
    /// supplies that the plain one cannot is the `.shared::cluster` barrier
    /// operand. A plain `cp.async.bulk.tensor` completes on a barrier in the
    /// *issuing* CTA's own shared memory, so a peer staging its half of a
    /// cluster operand has no way to say "my bytes, the leader's barrier", and
    /// the leader waits forever for a count it charged and nobody paid. That
    /// deadlock is the whole reason this exists.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load_2d_multicast_cg2`], with `cta_mask` this CTA's own
    /// bit.
    #[inline(always)]
    pub unsafe fn tma_load_2d_arriving_at(
        self,
        map: *const TmaDescriptor,
        leading: i32,
        minor: i32,
        sem: ClusterSemaphore,
    ) -> TransactionBytes {
        unsafe {
            self.tma_load_2d_at_arriving_at::<R>(map, 0, leading, minor, sem)
        }
    }

    /// A partial-height [`Self::tma_load_2d_arriving_at`] landing at `dst_row`.
    /// This builds a taller cluster operand from several adjacent tensor-map
    /// boxes while preserving typed transaction accounting.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load_2d_arriving_at`], plus the map box being
    /// `BOX_ROWS` tall, `dst_row + BOX_ROWS <= R`, and `dst_row` being aligned
    /// to the swizzle's eight-row period.
    #[inline(always)]
    pub unsafe fn tma_load_2d_at_arriving_at<const BOX_ROWS: usize>(
        self,
        map: *const TmaDescriptor,
        dst_row: usize,
        leading: i32,
        minor: i32,
        sem: ClusterSemaphore,
    ) -> TransactionBytes {
        const {
            assert!(BOX_ROWS <= R, "a box taller than the tile cannot land in it");
        }
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    self.subtile(i).add(dst_row * S::ATOM_BYTES),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                    sem.raw(),
                    1 << cluster::block_rank(),
                );
                i += 1;
            }
        }
        TransactionBytes::new(Self::SUBTILES * BOX_ROWS * S::ATOM_BYTES)
    }

    /// [`Self::tma_load_2d`] as a cta_group::2 multicast: every box lands in
    /// the CTAs of `cta_mask`, completing on the cluster-addressed barrier
    /// behind `sem`. The replication form — one fetch delivered to several
    /// CTAs — which is why the mask is the caller's; a load that replicates
    /// nothing and only wants the barrier's address space is
    /// [`Self::tma_load_2d_arriving_at`].
    ///
    /// The charge handed back is one tile — what a single destination
    /// receives, which is the whole of it for the own-bit mask that
    /// [`Self::tma_load_2d_arriving_at`] passes and the only mask any kernel
    /// here uses. A caller replicating into several CTAs owns the question of
    /// whether the one barrier it named sees that once or once per
    /// destination, and #49 is where that gets answered against hardware
    /// rather than guessed at here.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load_2d`], and the block must run as a cluster whose
    /// every masked CTA holds this tile's shared range and can receive it.
    #[inline(always)]
    pub unsafe fn tma_load_2d_multicast_cg2(
        self,
        map: *const TmaDescriptor,
        leading: i32,
        minor: i32,
        sem: ClusterSemaphore,
        cta_mask: u16,
    ) -> TransactionBytes {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    self.subtile(i),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                    sem.raw(),
                    cta_mask,
                );
                i += 1;
            }
        }
        Self::CHARGE
    }

    /// TMA the tile out to a [`crate::global`] panel map: one box per subtile,
    /// the same coordinates [`Self::tma_load`] reads them from, the bytes
    /// going the other way.
    ///
    /// Completion does not land on a [`Semaphore`], and the difference is not
    /// cosmetic. A load's destination is shared memory, where an mbarrier can
    /// live and count the arriving bytes; a store's destination is global
    /// memory, where none can. So a bulk store completes through
    /// [`tma_store_commit`] and [`tma_store_wait`] — the issuing thread's
    /// outstanding stores committed as one group, waited on by counting
    /// groups, with no barrier to arrive on and no byte accounting anywhere.
    /// The obligations that buys the caller are in the safety section, and
    /// they outlive the call.
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches
    /// `[R, SUBTILE_COLS]`. Beyond the call itself:
    ///
    /// - The tile must stay allocated and unwritten until the issuing thread
    ///   has waited on the group covering this store —
    ///   [`tma_store_wait_read`] to overwrite it, [`tma_store_wait`] for the
    ///   bytes to be readable in global memory. Nothing else republishes the
    ///   shared memory: dropping the handle, a `sync_threads`, or the kernel
    ///   ending are all silent here.
    /// - If the tile's contents arrived through the generic proxy — anything
    ///   written by `stmatrix` or a plain store, including
    ///   [`crate::ldst::store_fragment`], which states the same obligation for
    ///   an MMA reading the tile — the caller owes a
    ///   [`publish_to_async_proxy`] before this call. The TMA engine
    ///   reads through the async proxy and would otherwise be free to see
    ///   stale bytes. A tile that got here by [`Self::tma_load`] needs no
    ///   fence: that is the async proxy on both sides, ordered by the load's
    ///   own barrier.
    #[inline(always)]
    pub unsafe fn tma_store(self, map: *const TmaDescriptor, row: i32, plane: i32) {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_3d_s2g(
                    self.subtile(i),
                    map,
                    (i * Self::SUBTILE_COLS) as i32,
                    row,
                    plane,
                );
                i += 1;
            }
        }
    }

    /// [`Self::tma_store`] against a 2-D tensor map, taking its coordinates in
    /// the order [`Self::tma_load_2d`] takes them.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_store`], every clause.
    #[inline(always)]
    pub unsafe fn tma_store_2d(self, map: *const TmaDescriptor, leading: i32, minor: i32) {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_s2g(
                    self.subtile(i),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                );
                i += 1;
            }
        }
    }

    /// The tile base's absolute position in the 8-row swizzle period.
    /// SWIZZLE_128B XORs *physical* address bits `[9:7]` into the chunk index,
    /// so a tile whose base is not 1024-byte aligned starts mid-period and
    /// every manual swizzled store must fold this phase in.
    #[inline(always)]
    pub fn swizzle_phase(self) -> usize {
        (self.base as usize >> 7) & 7
    }

    /// Address of 16-byte chunk `chunk` of row `row`, with the swizzle applied
    /// exactly as the TMA engine would have — the store-side twin of a
    /// swizzled TMA load. Chunks are counted across the tile's whole logical
    /// row, so chunk `8 * i + c` is chunk `c` of stacked subtile `i`. For
    /// store loops, hoist the phase once with [`Self::chunk_writer`].
    ///
    /// # Safety
    ///
    /// `row < R` and `chunk < C * E::BYTES / 16`.
    #[inline(always)]
    pub unsafe fn swizzled_chunk(self, row: usize, chunk: usize) -> *mut u8 {
        unsafe { self.chunk_writer().at(row, chunk) }
    }

    /// The tile's swizzled-access handle with the base's absolute phase and
    /// the subtile stride captured once — hoist it outside a fragment loop
    /// exactly like the hand-written kernels hoisted their `p_phase`
    /// variables.
    #[inline(always)]
    pub fn chunk_writer(self) -> SwizzledChunks<E> {
        SwizzledChunks {
            base: self.base,
            rows: R,
            chunks: C * E::BYTES / 16,
            phase: self.swizzle_phase(),
            _marker: PhantomData,
        }
    }

    /// tcgen05 shared-memory operand descriptor for the K-major operand at
    /// `byte_offset` into the tile — same encoding as gemm's: 16-byte leading
    /// offset (the second core matrix sits eight bf16 columns along the row),
    /// 1024-byte stride, the swizzle mode in bits `[63:61]`. Pure bit math on
    /// the address; the MMA that consumes it carries the safety obligations.
    #[inline(always)]
    pub fn operand_descriptor(self, byte_offset: usize) -> u64 {
        self.descriptor(byte_offset, 16)
    }

    #[inline(always)]
    fn descriptor(self, byte_offset: usize, leading_bytes: u32) -> u64 {
        encode_descriptor(
            self.base as u64 + byte_offset as u64,
            leading_bytes,
            S::DESCRIPTOR_MODE,
        )
    }

    /// This tile as a K-major [`OperandWalk`]: one K=16 chunk every 32 bytes
    /// along the swizzled rows, 16-byte leading offset. Restricted to tiles
    /// whose K spans exactly one swizzle atom per row (gemm's `[128, 64]`
    /// stage) — a linear step cannot cross stacked subtiles.
    #[inline(always)]
    pub fn k_walk(self) -> OperandWalk {
        const {
            assert!(
                C * E::BYTES == S::ATOM_BYTES,
                "a linear K-major walk needs K to span exactly one swizzle atom"
            )
        };
        OperandWalk {
            base: self.base,
            chunk_step: 16 * E::BYTES,
            leading_bytes: 16,
            mode: S::DESCRIPTOR_MODE,
            transpose: false,
        }
    }

    /// This tile as an MN-major [`OperandWalk`] (the walk carries the
    /// transpose bit): one K=16 chunk every 16 rows, and the leading offset
    /// jumps to the stacked subtile holding MN columns 64..128
    /// ([`Self::SUBTILE_BYTES`] — not a step along the row).
    #[inline(always)]
    pub fn mn_walk(self) -> OperandWalk {
        const { assert!(R.is_multiple_of(16), "MN-major chunks are 16 rows each") };
        OperandWalk {
            base: self.base,
            chunk_step: 16 * S::ATOM_BYTES,
            leading_bytes: Self::SUBTILE_BYTES as u32,
            mode: S::DESCRIPTOR_MODE,
            transpose: true,
        }
    }
}

/// Make this thread's ordinary writes to CTA-shared memory visible to the
/// async proxy — `fence.proxy.async.shared::cta`, and **the fence half of
/// every "the caller owes a fence" contract in the crate.**
///
/// Shared memory is written two ways and read two ways. `stmatrix`, a plain
/// store, [`crate::ldst::store_tile`], [`crate::ldst::store_fragment`] and an
/// mbarrier's `init` all go through the *generic* proxy; the TMA engine and a
/// tcgen05 MMA read through the *async* one. The two proxies are not coherent
/// on their own and no barrier makes them so: `sync_threads` orders threads,
/// not proxies, so a `bar.sync` between the write and the engine's read is a
/// fence that was never issued.
///
/// This orders the writes of **the thread that executes it**, which is why it
/// is not a collective and why every thread that wrote a row has to call it.
/// A barrier afterwards is what carries that ordering to whichever thread
/// issues the store or the MMA — the pairing
/// [`crate::epilogue::StoreRing`] performs internally and three kernels write
/// out as `publish_to_async_proxy(); sync_threads();`.
///
/// Both directions of "which proxy" have to be true for the fence to be owed,
/// and the two cases where it is not are worth stating because they look
/// alike:
///
/// - A tile that arrived by [`SharedTile::tma_load`] and leaves by
///   [`SharedTile::tma_store`] is the async proxy on both sides, ordered by
///   the load's own barrier.
/// - A staged value written and read by ordinary threads —
///   [`crate::sync::block_reduce`]'s scratch — is the generic proxy on both
///   sides, ordered by its barriers.
///
/// It was private until #127: the crate stated the obligation in six safety
/// contracts and exported nothing that discharged it, so three kernels
/// imported `cuda_device::barrier` for the one instruction.
///
/// # Safety
///
/// A memory fence has no precondition of its own. What it does *not* do is
/// the hazard: it orders this thread's writes only, and it is not a barrier.
#[inline(always)]
pub unsafe fn publish_to_async_proxy() {
    unsafe { fence_proxy_async_shared_cta() }
}

/// Commit this thread's outstanding bulk stores as one group.
///
/// Groups are per *thread* and age in issue order: everything issued since the
/// last commit becomes the youngest group, and every earlier group's index
/// goes up by one. A tile's store is one instruction per stacked subtile, so
/// committing per tile is what makes "wait for that tile" expressible at all —
/// the waits below count groups and cannot name an instruction.
#[inline(always)]
pub fn tma_store_commit() {
    cp_async_bulk_commit_group();
}

/// Wait until at most `N` of this thread's committed store groups are still in
/// flight, *complete* meaning the bytes are in global memory and visible to
/// anything that reads them there — a following kernel, the host, another CTA.
///
/// `N = 0` drains every group this thread has committed and is the last thing
/// a kernel that wrote its result owes; nothing else makes a bulk store
/// visible, and a kernel that simply ends has not waited.
///
/// A pipelined epilogue wants [`tma_store_wait_read`] instead: reusing the
/// shared tile needs only the engine's *reads* to be done, and blocking on
/// global visibility to recycle a buffer serializes the overlap the pipeline
/// exists for.
///
/// `N` is a const parameter because the instruction's group count is an
/// immediate — a runtime depth is not a thing the hardware can be asked for.
#[inline(always)]
pub fn tma_store_wait<const N: u32>() {
    cp_async_bulk_wait_group(N);
}

/// Wait until at most `N` of this thread's committed store groups still have
/// their *source reads* outstanding: the shared tiles behind the older groups
/// are free to overwrite, while their bytes may still be on the way to global
/// memory.
///
/// The cheaper of the two waits, and the one a pipelined epilogue is written
/// around — store stage `i`, recycle its buffer for stage `i + 1`, and let the
/// global writes retire behind the next tile's work. It says nothing about
/// what any other thread, CTA or the host can see, so it is never the last
/// wait in a kernel: a result that has only been read out of shared memory has
/// not arrived anywhere.
#[inline(always)]
pub fn tma_store_wait_read<const N: u32>() {
    cp_async_bulk_wait_group_read(N);
}

#[inline(always)]
fn encode_descriptor(address: u64, leading_bytes: u32, mode: u8) -> u64 {
    const STRIDE_BYTES: u32 = 1024;
    ((address >> 4) & 0x3fff)
        | ((((leading_bytes >> 4) & 0x3fff) as u64) << 16)
        | ((((STRIDE_BYTES >> 4) & 0x3fff) as u64) << 32)
        | (1u64 << 46)
        | ((mode as u64) << 61)
}

/// One MMA operand's chunk walk with the layout erased to values: the chunk
/// step and descriptor leading offset that distinguish a K-major from an
/// MN-major operand, as runtime data instead of a type. This exists for
/// kernels that select the layout at runtime (gemm's `transposed` launch
/// parameter): a value-level walk keeps the issue loop *single* — one
/// select feeding one chain — matching the hand-written kernel's schedule,
/// where a typed two-arm branch would duplicate the MMA chain per layout
/// and hand ptxas a different instruction stream to allocate against.
#[derive(Clone, Copy)]
pub struct OperandWalk {
    base: *mut u8,
    chunk_step: usize,
    leading_bytes: u32,
    mode: u8,
    transpose: bool,
}

impl OperandWalk {
    /// Operand descriptor for K chunk `chunk`.
    #[inline(always)]
    pub fn chunk_descriptor(self, chunk: usize) -> u64 {
        encode_descriptor(
            self.base as u64 + (chunk * self.chunk_step) as u64,
            self.leading_bytes,
            self.mode,
        )
    }

    /// Whether the MMA must read this operand transposed — an MN-major walk
    /// supplies K along rows, which is the same fact as its chunk step and
    /// leading offset. Carried here so the instruction descriptor is built
    /// from the walk it is issued with rather than beside it.
    #[inline(always)]
    pub fn transposed(self) -> bool {
        self.transpose
    }
}

/// A tile's swizzled cursor: the tile as one flat stack of 128-byte rows,
/// eight 16-byte chunks each, chunk index XORed with the row's own position in
/// the swizzle period — `(row + phase) & 7`, `phase` being the tile base's
/// absolute position in it (captured once at construction).
///
/// Stacked subtiles need no term of their own because they *are* further rows
/// of that stack: subtile `i` starts [`SharedTile::SUBTILE_BYTES`] =
/// `rows * 128` bytes along, and SWIZZLE_128B keys off *physical* address bits
/// `[9:7]`, so subtile `i`'s row `r` is the tile's 128-byte row `i * rows + r`
/// and takes that row's phase. A logical chunk index therefore splits into
/// `(subtile, chunk)` and the two row terms simply add; the phase is never
/// re-derived per subtile, and no relation between `rows` and the 8-row
/// swizzle period is assumed.
///
/// The chunk arithmetic is element-independent — a 16-byte chunk is 16 bytes
/// whatever it holds — but the cursor still carries its tile's `E` so a store
/// through it packs to the element the tile is actually made of, without the
/// caller naming it a second time.
pub struct SwizzledChunks<E: Element> {
    base: *mut u8,
    rows: usize,
    chunks: usize,
    phase: usize,
    _marker: PhantomData<E>,
}

impl<E: Element> Clone for SwizzledChunks<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element> Copy for SwizzledChunks<E> {}

impl<E: Element> SwizzledChunks<E> {
    /// 16-byte chunks in one logical row of the tile — eight per stacked
    /// subtile, and the exclusive bound on [`Self::at`]'s chunk index.
    #[inline(always)]
    pub fn chunks(self) -> usize {
        self.chunks
    }

    /// Address of chunk `chunk` of row `row`, counting chunks across the whole
    /// logical row: chunk `8 * i + c` is chunk `c` of stacked subtile `i`.
    ///
    /// # Safety
    ///
    /// `row` inside the tile, `chunk < self.chunks()`.
    #[inline(always)]
    pub unsafe fn at(self, row: usize, chunk: usize) -> *mut u8 {
        let (subtile, chunk) = (chunk / 8, chunk % 8);
        let row = subtile * self.rows + row;
        unsafe {
            self.base
                .add(row * 128 + (chunk ^ ((row + self.phase) & 7)) * 16)
        }
    }
}

/// `N` elements of `E` in shared memory, contiguous and **unswizzled** — the
/// shape a parameter vector, a per-column statistic or a set of per-warp
/// partials has. Like [`SharedTile`] the handle is a base pointer plus a
/// compile-time length, and like it the shared plan belongs to the kernel.
///
/// # Why this does not go through [`Swizzle`]
///
/// Swizzling exists to keep a *2-D* tile's 16-byte chunks off the same banks
/// when successive rows are read together; the XOR is over the row index, and
/// a vector has one row for the phase to come from. So the layout question is
/// not "which mode" but "an atom is the wrong unit here at all", and the two
/// jobs [`Swizzle::ATOM_BYTES`] does make that concrete. It is the swizzle
/// period *and* the width of a TMA box, and an unswizzled vector wants
/// neither: its box is `N` wide, a number no mode marker can carry, and any
/// atom small enough to divide a short vector splits it into stacked subtiles
/// the engine would then fetch one instruction each. A 128-byte atom is worse
/// still — the narrowest bf16 tile it admits is 64 columns, which is the
/// padding per row this type exists to avoid.
///
/// Adding a `SwizzleNone` mode is therefore not the same job and is left to
/// #14, which owns *tiles* under narrower and absent swizzles: an unswizzled
/// `[R, C]` staging tile does need a `Swizzle` impl, and it needs `ATOM_BYTES`
/// to stop meaning "box width" first. A vector needs none of that, and
/// borrowing the mode marker to get one would have decided #14's question by
/// accident.
///
/// The TMA still delivers it: the box is `[N]` (or `[N, 1]` at higher rank)
/// with `CU_TENSOR_MAP_SWIZZLE_NONE`, which the engine writes contiguously —
/// the same bytes [`Self::at`] addresses. That agreement is what the
/// [`crate::global::TileBox`] impl states.
///
/// # The engine is optional
///
/// A vector need not be TMA'd at all — [`crate::sync::block_reduce`]'s scratch
/// is written by [`Self::set`] and read by [`Self::get`] and never touches a
/// descriptor — so the shape rules an unswizzled box obeys are enforced by the
/// four transfer methods rather than by [`Self::from_raw`]. `N` may be any
/// length as a handle; it must make a legal box only where one is built. See
/// `Self::TMA_OK`.
pub struct SharedVec<E: Element, const N: usize> {
    base: *mut u8,
    _marker: PhantomData<E>,
}

impl<E: Element, const N: usize> Clone for SharedVec<E, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const N: usize> Copy for SharedVec<E, N> {}

impl<E: Element, const N: usize> SharedVec<E, N> {
    /// Bytes of the whole vector — what a TMA load of it charges, and the
    /// stride to the next object in a kernel's shared plan.
    pub const BYTES: usize = N * E::BYTES;

    /// What a TMA load of the vector charges its barrier: one box, the whole
    /// of it. Crate-private for the reason [`SharedTile::CHARGE`] is.
    pub(crate) const CHARGE: TransactionBytes = TransactionBytes::new(Self::BYTES);

    /// An unswizzled TMA box's innermost dimension is measured in bytes and
    /// must be a whole number of 16-byte lines; there is no swizzle atom to
    /// round it up to, so the length itself has to be legal.
    const BOX_OK: () = assert!(
        Self::BYTES.is_multiple_of(16),
        "an unswizzled vector's bytes must be a multiple of the TMA's 16-byte line"
    );

    /// A box dimension is a `u32` field capped at 256 by the descriptor
    /// format, and a vector is a single box by construction.
    const LENGTH_OK: () = assert!(N <= 256, "a TMA box dimension is at most 256 elements");

    /// Both box rules, forced at each of the four calls that hand the vector to
    /// the engine — and nowhere else.
    ///
    /// They used to be forced by [`Self::from_raw`], which read as "a
    /// `SharedVec` is a thing the TMA can deliver" and is a stronger claim than
    /// either assert makes. Both are statements about a *descriptor's box*, and
    /// [`crate::sync::block_reduce`]'s scratch is the one use of this type that
    /// never meets a descriptor: a four-warp block's partials are 16 bytes, a
    /// two-warp block's are 8, and only the first could be constructed under
    /// the old placement. So the rules moved to the calls they are about,
    /// where they still fire at codegen for every caller they genuinely bind,
    /// and a vector that is only ever written by [`Self::set`] and read by
    /// [`Self::get`] pays nothing for a box it does not have.
    ///
    /// The host side is unchanged and is the other half of this: building a
    /// tensor map runs `GlobalLayout::check_driver_requirements`, which rejects
    /// the same shapes at map-construction time with a message naming the field
    /// and the byte count.
    const TMA_OK: () = {
        #[allow(clippy::let_unit_value)]
        let _ = (Self::BOX_OK, Self::LENGTH_OK);
    };

    /// Wrap a raw shared-memory base (a `DynamicSharedArray` offset or a
    /// `SharedArray` static).
    ///
    /// Asserts nothing about the length: see `Self::TMA_OK` for why the box
    /// rules live on the transfers rather than here.
    ///
    /// # Safety
    ///
    /// `base` must point to at least `Self::BYTES` bytes of shared memory,
    /// living as long as every use of the vector. A vector this crate's TMA
    /// paths will touch must also be 128-byte aligned, which is the engine's
    /// destination alignment; one used only through [`Self::get`] and
    /// [`Self::set`] needs no more than `E`'s own alignment.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut u8) -> Self {
        Self {
            base,
            _marker: PhantomData,
        }
    }

    /// The vector's base address.
    #[inline(always)]
    pub const fn base(self) -> *mut u8 {
        self.base
    }

    /// Address of element `index` — a flat stride, with no phase to fold in.
    ///
    /// # Safety
    ///
    /// `index < N`.
    #[inline(always)]
    pub const unsafe fn at(self, index: usize) -> *mut u8 {
        unsafe { self.base.add(index * E::BYTES) }
    }

    /// Element `index` widened to the fp32 a register holds.
    ///
    /// # Safety
    ///
    /// `index < N`, and the bytes must already be visible to the generic
    /// proxy — a TMA load needs its barrier waited on, another thread's
    /// [`Self::set`] needs a barrier between the write and this read.
    #[inline(always)]
    pub unsafe fn get(self, index: usize) -> f32 {
        unsafe { E::read(self.at(index)) }
    }

    /// Write one fp32 into element `index`, rounding to `E`.
    ///
    /// One element, not one word, so lanes writing neighbouring indices do not
    /// clobber each other — which is the whole point for a block reduction,
    /// where warp `w` owns index `w` and nothing else.
    ///
    /// # Safety
    ///
    /// `index < N`, no other thread writes the same index concurrently, and
    /// the caller owes a [`publish_to_async_proxy`] before the TMA engine or an
    /// MMA reads the vector.
    #[inline(always)]
    pub unsafe fn set(self, index: usize, value: f32) {
        unsafe { E::write(self.at(index), value) }
    }

    /// TMA the vector from a rank-1 [`crate::global::GlobalLayout`]: one box,
    /// `N` elements wide, starting at element `start`. Completion lands on
    /// `sem`, and the call hands back the charge for it, to be summed into
    /// that barrier's [`Semaphore::expect_tx`].
    ///
    /// One instruction, unlike [`SharedTile::tma_load`]'s one per subtile —
    /// an unswizzled box has no atom to be cut into.
    ///
    /// This is one of the four calls `Self::TMA_OK` binds: `N` must make a
    /// legal box here, where it need not to merely exist.
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches `[N]`,
    /// `sem` must be an initialized TMA barrier, and the vector must be
    /// 128-byte aligned.
    #[inline(always)]
    pub unsafe fn tma_load(
        self,
        map: *const TmaDescriptor,
        start: i32,
        sem: Semaphore,
    ) -> TransactionBytes {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TMA_OK;
        unsafe { cp_async_bulk_tensor_1d_g2s(self.base, map, start, sem.raw()) }
        Self::CHARGE
    }

    /// [`Self::tma_load`] against a rank-2 map — one *row* of a `[N, rows]`
    /// buffer, which is how a batch of parameter vectors is stored. The box is
    /// `[N, 1]`, so `minor` selects the row and nothing about the shared side
    /// changes.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load`], for a box shape of `[N, 1]`.
    #[inline(always)]
    pub unsafe fn tma_load_2d(
        self,
        map: *const TmaDescriptor,
        start: i32,
        minor: i32,
        sem: Semaphore,
    ) -> TransactionBytes {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TMA_OK;
        unsafe { cp_async_bulk_tensor_2d_g2s(self.base, map, start, minor, sem.raw()) }
        Self::CHARGE
    }

    /// TMA the vector out to a rank-1 map, the coordinate read the way
    /// [`Self::tma_load`] reads it.
    ///
    /// # Safety
    ///
    /// As [`SharedTile::tma_store`], every clause — completion is
    /// [`tma_store_commit`] plus a wait and never a [`Semaphore`], the vector
    /// must stay unwritten until that wait, and contents that arrived through
    /// the generic proxy ([`Self::set`]) owe a [`publish_to_async_proxy`]
    /// first.
    #[inline(always)]
    pub unsafe fn tma_store(self, map: *const TmaDescriptor, start: i32) {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TMA_OK;
        unsafe { cp_async_bulk_tensor_1d_s2g(self.base, map, start) }
    }

    /// [`Self::tma_store`] against the rank-2 map [`Self::tma_load_2d`] reads.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_store`], for a box shape of `[N, 1]`.
    #[inline(always)]
    pub unsafe fn tma_store_2d(self, map: *const TmaDescriptor, start: i32, minor: i32) {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TMA_OK;
        unsafe { cp_async_bulk_tensor_2d_s2g(self.base, map, start, minor) }
    }
}

/// One typed value in shared memory, accessed volatile for cross-warp
/// mailboxes whose visibility is ordered by a [`Semaphore`].
#[derive(Clone, Copy)]
pub struct SharedCell<T: Copy> {
    base: *mut T,
}

impl<T: Copy> SharedCell<T> {
    pub const BYTES: usize = size_of::<T>();
    pub const ALIGNMENT: usize = align_of::<T>();

    /// Attach to suitably aligned shared storage for one `T`.
    ///
    /// # Safety
    ///
    /// `base` must name exclusive shared storage for a `T` for the lifetime of
    /// the returned handle.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        Self { base: base.cast() }
    }

    /// Publish one value through a volatile shared-memory store.
    ///
    /// # Safety
    ///
    /// The caller must order readers with a barrier or semaphore.
    #[inline(always)]
    pub unsafe fn write(self, value: T) {
        unsafe { self.base.write_volatile(value) }
    }

    /// Read the currently published value through a volatile shared load.
    ///
    /// # Safety
    ///
    /// The caller must have observed the synchronization event publishing it.
    #[inline(always)]
    pub unsafe fn read(self) -> T {
        unsafe { self.base.read_volatile() }
    }
}

/// `N` same-shaped tiles backing a pipeline ring: tile `i` lives in stage
/// `i % N`. The parity arithmetic for the matching barriers lives in
/// [`crate::sync::SemaphoreRing`].
pub struct SharedTileRing<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> {
    base: *mut u8,
    _marker: PhantomData<(E, S)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> Clone
    for SharedTileRing<E, R, C, S, N>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> Copy
    for SharedTileRing<E, R, C, S, N>
{
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize>
    SharedTileRing<E, R, C, S, N>
{
    /// Bytes of the whole ring.
    pub const BYTES: usize = N * SharedTile::<E, R, C, S>::BYTES;

    /// Wrap `N` consecutive tiles' worth of shared memory.
    ///
    /// # Safety
    ///
    /// Same contract as [`SharedTile::from_raw`], for `Self::BYTES` bytes.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        Self {
            base,
            _marker: PhantomData,
        }
    }

    /// The tile of stage `index % N`.
    #[inline(always)]
    pub fn tile(self, index: u32) -> SharedTile<E, R, C, S> {
        unsafe {
            SharedTile::from_raw(
                self.base
                    .add((index as usize % N) * SharedTile::<E, R, C, S>::BYTES),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Panel = SharedTile<Bf16, 64, 128, Swizzle128B>;
    type PTile = SharedTile<Bf16, 64, 64, Swizzle128B>;
    type Paired = SharedTile<Bf16, 128, 128, Swizzle128B>;

    #[test]
    fn f16_matches_the_tcgen05_operand_tables() {
        assert_eq!(F16::BYTES, 2);
        assert_eq!(F16::MMA_KIND, 0);
        assert_eq!(F16::ELEMENT_TYPE, Tcgen05ElementType::F16);
        assert_eq!(F16::PER_WORD, 2);
    }

    #[test]
    fn f16_unpack_handles_normal_subnormal_and_special_values() {
        let [one, minus_two] = F16::unpack(0xc000_3c00);
        assert_eq!(one, 1.0);
        assert_eq!(minus_two, -2.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn bf16_matches_the_tcgen05_operand_tables() {
        assert_eq!(Bf16::BYTES, 2);
        // KIND 0 is f16, which bf16 shares (see MmaElement::MMA_KIND).
        assert_eq!(Bf16::MMA_KIND, 0);
        // Derived from Unpacked = [f32; 2] rather than written down twice.
        assert_eq!(Bf16::PER_WORD, 2);
        // `Bf16::pack` is `cvt_f32x2_bf16x2`, a device intrinsic whose host
        // body is `unreachable!()` — its bit layout is only checkable on a GPU.
        // `Bf16::unpack` is not: widening needs no instruction (below).
    }

    #[test]
    fn bf16_unpack_widens_both_halves_exactly() {
        // The low half first, matching `pack`'s "first value in the low half".
        assert_eq!(Bf16::unpack(0x4000_3f80), [1.0, 2.0]);
        // Widening is exact, so every fp32 whose low 16 bits are zero — which
        // is every value a bf16 tile can hold — comes back bit for bit.
        for high in [0x0000u32, 0x3f80, 0xbf80, 0x7f7f, 0x4049] {
            for low in [0x0000u32, 0x3f80, 0xc180, 0x0080] {
                let word = (high << 16) | low;
                let [first, second] = Bf16::unpack(word);
                assert_eq!(first.to_bits(), low << 16);
                assert_eq!(second.to_bits(), high << 16);
            }
        }
    }

    #[test]
    fn f32_packs_one_value_a_word_and_rounds_nothing() {
        assert_eq!(F32::BYTES, 4);
        // One, where bf16's is two — the number `SharedVec::at`'s stride and
        // every `Element` consumer that assumed a pair has to survive.
        assert_eq!(F32::PER_WORD, 1);
        // Unlike bf16's, *both* halves of this element are ordinary bit math,
        // so the round trip is checkable here rather than only on a GPU. It is
        // the identity on the bits, which is the property a partial staged
        // through shared memory needs: nothing of the sum is lost on the way.
        for value in [0.0f32, 1.0, -1.0, 1e-30, 3.402_823_5e38, 585.0, 0.1] {
            assert_eq!(F32::unpack(F32::pack([value])), [value]);
            assert_eq!(F32::pack([value]), value.to_bits());
        }
        // Through memory, the pair `SharedVec::get`/`set` is made of.
        let mut storage = [0u32; 4];
        let at = storage.as_mut_ptr() as *mut u8;
        for (index, value) in [1.0f32, -0.5, 1e8, 0.1].iter().enumerate() {
            unsafe { F32::write(at.add(4 * index), *value) };
            assert_eq!(unsafe { F32::read(at.add(4 * index)) }, *value);
        }
        // Neighbours untouched: four elements, four distinct words.
        assert_eq!(storage[0], 1.0f32.to_bits());
        assert_eq!(storage[3], 0.1f32.to_bits());
    }

    #[test]
    fn the_partials_vector_is_the_smallest_legal_box() {
        // `SharedVec<F32, 4>` — a block reduction's four per-warp partials — is
        // 16 bytes *exactly*, which is the TMA's line and so the smallest box
        // `BOX_OK` admits. It passes by zero margin, which is worth an
        // assertion rather than luck.
        assert_eq!(SharedVec::<F32, 4>::BYTES, 16);
        assert_eq!(SharedVec::<F32, 8>::BYTES, 32);
        // Constructing it forces nothing, which is the point of moving the box
        // rules onto the transfers: `from_raw` is a pointer and a shape, and
        // the shapes an unswizzled *box* may have are a fact about the four
        // calls that build one. So the same length that must be a whole number
        // of 16-byte lines to be TMA'd is free to be a block's scratch.
        let partials = [0.0f32; 4];
        let vec = unsafe { SharedVec::<F32, 4>::from_raw(partials.as_ptr() as *mut u8) };
        for index in 0..4usize {
            assert_eq!(
                unsafe { vec.at(index) } as usize - partials.as_ptr() as usize,
                index * 4
            );
        }
    }

    #[test]
    fn a_scratch_vector_shorter_than_a_box_still_constructs() {
        // One and two warps: 4 and 8 bytes, neither a whole 16-byte line, and
        // both illegal as a TMA box. Under the old placement of `BOX_OK` on
        // `from_raw` these failed at codegen and a 1- or 2-warp block simply
        // could not hold a block reduction's partials — an arbitrary
        // restriction, since that scratch never becomes a box. The addressing
        // is the same flat stride at every length.
        let one = [7.0f32];
        let vec = unsafe { SharedVec::<F32, 1>::from_raw(one.as_ptr() as *mut u8) };
        assert_eq!(unsafe { vec.get(0) }, 7.0);
        assert_eq!(SharedVec::<F32, 1>::BYTES, 4);

        let two = [1.0f32, 2.0];
        let vec = unsafe { SharedVec::<F32, 2>::from_raw(two.as_ptr() as *mut u8) };
        assert_eq!(unsafe { vec.get(0) }, 1.0);
        assert_eq!(unsafe { vec.get(1) }, 2.0);
        assert_eq!(SharedVec::<F32, 2>::BYTES, 8);

        // And a bf16 vector of 4, the shape `global`'s own test uses to show
        // `check_driver_requirements` rejecting an 8-byte box: illegal as a
        // box, legal as a handle, and the two facts now live in the two places
        // they belong to.
        //
        // The other direction is not expressible here — `tma_store` on this
        // same vector is a *compile* error, not a failing assertion, so no test
        // can call it. Checked by hand against this exact shape, and the error
        // names the instantiation: "evaluation of `SharedVec::<Bf16, 4>::
        // BOX_OK` failed ... while instantiating `SharedVec::<Bf16, 4>::
        // tma_store`". Making it a permanent case needs `trybuild`, which the
        // crate does not depend on.
        assert_eq!(SharedVec::<Bf16, 4>::BYTES, 8);
        let narrow = [0u16; 4];
        let _ = unsafe { SharedVec::<Bf16, 4>::from_raw(narrow.as_ptr() as *mut u8) };
    }

    #[test]
    fn shape_math_matches_the_flash_layout() {
        // The constants tcgen05.rs derives by hand: TILE_BYTES / SUBTILE_BYTES.
        assert_eq!(Panel::SUBTILES, 2);
        assert_eq!(Panel::SUBTILE_BYTES, 64 * 128);
        assert_eq!(Panel::BYTES, 64 * 128 * 2);
        assert_eq!(PTile::SUBTILES, 1);
        assert_eq!(PTile::BYTES, 64 * 64 * 2);
        // The paired backward operand: [128, 64] subtiles a TILE_BYTES apart.
        assert_eq!(Paired::SUBTILE_BYTES, 128 * 128);
    }

    #[test]
    fn swizzled_chunk_folds_the_absolute_base_phase() {
        // Base at an odd 128-byte row phase, like a P subtile mid-plan:
        // phase = (base >> 7) & 7. Pointer math only — never dereferenced.
        let base = 0x1080usize;
        let tile = unsafe { PTile::from_raw(base as *mut u8) };
        assert_eq!(tile.swizzle_phase(), (base >> 7) & 7);
        // tcgen05.rs formula: base + row*128 + (chunk ^ ((row + phase) & 7))*16.
        let phase = tile.swizzle_phase();
        for row in [0usize, 2, 7, 63] {
            for chunk in 0usize..8 {
                let expected = base + row * 128 + ((chunk ^ ((row + phase) & 7)) * 16);
                assert_eq!(
                    unsafe { tile.swizzled_chunk(row, chunk) } as usize,
                    expected
                );
            }
        }
    }

    #[test]
    fn swizzled_chunks_permute_each_row_within_itself() {
        // The property the device harness checks against the TMA engine: over
        // a whole tile the chunk map is a bijection onto the tile's 16-byte
        // slots, and no row's chunks escape that row's 128 bytes. Swept over
        // every base phase, since the XOR folds in the base's own position in
        // the 8-row swizzle period.
        for phase in 0..8usize {
            let base = 0x2000 + phase * 128;
            let tile = unsafe { PTile::from_raw(base as *mut u8) };
            let chunks = tile.chunk_writer();
            let mut seen = vec![false; 64 * 8];
            for row in 0..64usize {
                for chunk in 0..8usize {
                    let offset = unsafe { chunks.at(row, chunk) } as usize - base;
                    assert_eq!(offset / 128, row, "phase {phase} row {row} chunk {chunk}");
                    assert!(
                        !seen[offset / 16],
                        "phase {phase} slot {} twice",
                        offset / 16
                    );
                    seen[offset / 16] = true;
                }
            }
            assert!(seen.iter().all(|&slot| slot));
        }
    }

    #[test]
    fn a_wide_tiles_chunks_walk_the_stacked_subtiles() {
        // Chunk 8*i + c is chunk c of subtile i: the same one-subtile formula
        // applied at the subtile's own base, which is what a reader who knows
        // `tma_load` expects. R = 64 here, a whole number of swizzle periods,
        // so subtile 1 lands back at subtile 0's phase.
        let base = 0x2000usize;
        let tile = unsafe { Panel::from_raw(base as *mut u8) };
        let chunks = tile.chunk_writer();
        assert_eq!(chunks.chunks(), 16);
        let phase = tile.swizzle_phase();
        for subtile in 0..Panel::SUBTILES {
            for row in [0usize, 1, 7, 63] {
                for chunk in 0usize..8 {
                    let expected = base
                        + subtile * Panel::SUBTILE_BYTES
                        + row * 128
                        + (chunk ^ ((row + phase) & 7)) * 16;
                    assert_eq!(
                        unsafe { chunks.at(row, 8 * subtile + chunk) } as usize,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn a_subtiles_phase_is_its_absolute_row_not_its_own_row() {
        // The part of the derivation that only shows up when the subtile
        // height is *not* a whole number of swizzle periods: SWIZZLE_128B XORs
        // physical address bits [9:7], so subtile 1 of a 4-row tile begins 4
        // 128-byte rows down and starts at phase 4, not at phase 0. Written
        // down because at every shape the crate ships (R = 64, 128) the two
        // readings coincide and a wrong one would never be noticed.
        type Short = SharedTile<Bf16, 4, 128, Swizzle128B>;
        let base = 0x2000usize;
        let chunks = unsafe { Short::from_raw(base as *mut u8) }.chunk_writer();
        for row in 0..4usize {
            for chunk in 0usize..8 {
                let expected =
                    base + Short::SUBTILE_BYTES + row * 128 + (chunk ^ ((row + 4) & 7)) * 16;
                assert_eq!(unsafe { chunks.at(row, 8 + chunk) } as usize, expected);
            }
        }
    }

    #[test]
    fn chunks_of_a_wide_tile_are_injective_over_the_whole_tile() {
        // A wrong subtile stride collides or escapes here immediately. Note
        // what this cannot see: any phase is a permutation of a row's eight
        // chunks, so a wrong phase stays a bijection — that one is pinned by
        // the two tests above and, against the TMA engine itself, by the
        // device harness. Swept over every base phase and over a subtile
        // height that is not a multiple of the swizzle period.
        fn injective<const R: usize, const C: usize>(base: usize) {
            type Tile<const R: usize, const C: usize> = SharedTile<Bf16, R, C, Swizzle128B>;
            let chunks = unsafe { Tile::<R, C>::from_raw(base as *mut u8) }.chunk_writer();
            let mut seen = vec![false; Tile::<R, C>::BYTES / 16];
            for row in 0..R {
                for chunk in 0..chunks.chunks() {
                    let offset = unsafe { chunks.at(row, chunk) } as usize - base;
                    assert!(
                        offset < Tile::<R, C>::BYTES,
                        "row {row} chunk {chunk} escaped"
                    );
                    assert!(!seen[offset / 16], "slot {} twice", offset / 16);
                    seen[offset / 16] = true;
                }
            }
            assert!(seen.iter().all(|&slot| slot));
        }
        for phase in 0..8usize {
            injective::<128, 128>(0x10000 + phase * 128);
            injective::<64, 256>(0x10000 + phase * 128);
            injective::<4, 128>(0x10000 + phase * 128);
        }
    }

    #[test]
    fn subtiles_stack_by_subtile_bytes() {
        // What tma_load's per-box destination arithmetic rests on: stacked
        // subtiles are contiguous, one SUBTILE_BYTES apart, covering the tile.
        let base = 0x8000usize;
        let tile = unsafe { Panel::from_raw(base as *mut u8) };
        for i in 0..Panel::SUBTILES {
            assert_eq!(
                unsafe { tile.subtile(i) } as usize,
                base + i * Panel::SUBTILE_BYTES
            );
        }
        assert_eq!(Panel::SUBTILES * Panel::SUBTILE_BYTES, Panel::BYTES);
    }

    #[test]
    fn operand_descriptor_encodes_like_gemm() {
        // Same encoding smem_descriptor() produced: address bits, 16-byte
        // leading offset, 1024-byte stride, mode 2 in bits [63:61].
        let base = 0x4000usize;
        let tile = unsafe { PTile::from_raw(base as *mut u8) };
        let descriptor = tile.operand_descriptor(32);
        let address = ((base as u64 + 32) >> 4) & 0x3fff;
        let expected = address | (1u64 << 16) | (64u64 << 32) | (1u64 << 46) | (2u64 << 61);
        assert_eq!(descriptor, expected);
    }

    #[test]
    fn walks_reproduce_gemm_consume_stage_descriptors() {
        // gemm's consume_stage built build_smem_descriptor(smem + offset,
        // leading, 1024, 2) with (offset, leading) = (chunk * 32, 16) for
        // K-major and (chunk * 16 * 128, SUBTILE_BYTES = 8192) for MN-major.
        type KStage = SharedTile<Bf16, 128, 64, Swizzle128B>;
        type MnStage = SharedTile<Bf16, 64, 128, Swizzle128B>;
        assert_eq!(MnStage::SUBTILE_BYTES, 8192);
        let base = 0x4000u64;
        let expected = |offset: u64, leading: u64| {
            (((base + offset) >> 4) & 0x3fff)
                | (((leading >> 4) & 0x3fff) << 16)
                | (64u64 << 32)
                | (1u64 << 46)
                | (2u64 << 61)
        };
        let k = unsafe { KStage::from_raw(base as *mut u8) }.k_walk();
        let mn = unsafe { MnStage::from_raw(base as *mut u8) }.mn_walk();
        for chunk in 0..4usize {
            assert_eq!(k.chunk_descriptor(chunk), expected(chunk as u64 * 32, 16));
            assert_eq!(
                mn.chunk_descriptor(chunk),
                expected(chunk as u64 * 2048, 8192)
            );
        }
    }

    type Params = SharedVec<Bf16, 128>;

    #[test]
    fn a_vectors_elements_are_flat_and_disjoint() {
        // The whole of the layout: element i at i * BYTES, no phase, no atom.
        // A collision or an escape here is a wrong stride, and there is
        // nothing else the addressing could get wrong.
        let base = 0x3000usize;
        let vec = unsafe { Params::from_raw(base as *mut u8) };
        let mut seen = [false; 128];
        for index in 0..128usize {
            let offset = unsafe { vec.at(index) } as usize - base;
            assert_eq!(offset, index * Bf16::BYTES);
            assert!(offset < Params::BYTES, "element {index} escaped");
            assert!(!seen[offset / Bf16::BYTES], "slot {offset} twice");
            seen[offset / Bf16::BYTES] = true;
        }
        assert!(seen.iter().all(|&slot| slot));
        assert_eq!(Params::BYTES, 128 * 2);
    }

    #[test]
    fn a_vector_costs_no_padding_where_a_tile_would() {
        // The reason this is not a one-row SharedTile. A 32-column bf16 tile
        // is not even expressible (WIDTH_OK wants a whole 64-column subtile),
        // and the narrowest one that is spends a 128-byte atom on its single
        // row; the vector spends its elements and nothing else.
        assert_eq!(SharedVec::<Bf16, 32>::BYTES, 64);
        assert_eq!(SharedTile::<Bf16, 1, 64, Swizzle128B>::BYTES, 128);
        assert_eq!(SharedVec::<Bf16, 64>::BYTES, 128);
    }

    #[test]
    fn reading_a_vector_widens_the_element_at_that_index() {
        // Bf16::read against real bytes — the one half of the scalar pair that
        // holds on the host, since widening needs no instruction. Each element
        // names its own index, so a wrong stride reads a neighbour.
        let elements: Vec<u16> = (0..128u16).map(|index| 0x4000 | index).collect();
        let vec = unsafe { Params::from_raw(elements.as_ptr() as *mut u8) };
        for index in 0..128usize {
            let value = unsafe { vec.get(index) };
            assert_eq!(value.to_bits(), (0x4000u32 | index as u32) << 16);
        }
    }

    #[test]
    fn a_tiles_charge_is_the_boxes_its_load_issues() {
        // `CHARGE` counts what the load loop actually does — one box per
        // subtile, `R` rows of one swizzle atom each — and `BYTES` counts the
        // tile. They are the same number for every shape, and that identity is
        // the whole reason a derived charge can replace a hand-written one. A
        // width that stopped being a whole number of subtiles would break it,
        // and `WIDTH_OK` rejects those first.
        assert_eq!(Panel::CHARGE.bytes(), Panel::BYTES as u32);
        assert_eq!(PTile::CHARGE.bytes(), PTile::BYTES as u32);
        assert_eq!(Paired::CHARGE.bytes(), Paired::BYTES as u32);
        assert_eq!(Params::CHARGE.bytes(), Params::BYTES as u32);
        // A partial load — `tma_load_at::<BOX_ROWS>` — charges its boxes and
        // not the tile, which is the case the old prose asked callers to do by
        // hand. Half the rows, half the bytes.
        let half = || TransactionBytes::new(Paired::SUBTILES * 64 * Swizzle128B::ATOM_BYTES);
        assert_eq!(half().bytes(), Paired::BYTES as u32 / 2);
        assert_eq!((half() + half()).bytes(), Paired::CHARGE.bytes());
    }

    #[test]
    fn every_producers_derived_charge_is_the_sum_it_used_to_write_down() {
        // One case per `expect_tx` in `examples/`, with the tile types the
        // kernel declares and the literal byte count its hand-sum came to.
        // Both sides are spelled out: a derivation that agreed with a hand-sum
        // only because both were computed from the same expression would be
        // testing nothing.
        type GemmA = SharedTile<Bf16, 128, 64, Swizzle128B>;
        type GemmB = SharedTile<Bf16, 64, 64, Swizzle128B>;
        // gemm — `RANKS * (ATile::BYTES + BTile::BYTES)`, the one charge that
        // covers bytes the charging CTA does not issue.
        let stage = GemmA::CHARGE + GemmB::CHARGE;
        assert_eq!(stage.bytes(), 24576);
        assert_eq!(stage.across_ranks(2).bytes(), 49152);

        type Q = SharedTile<Bf16, 128, 128, Swizzle128B>;
        type KV = SharedTile<Bf16, 64, 128, Swizzle128B>;
        // flash_forward — one `QTile::BYTES` up front, then
        // `KTile::BYTES + VTile::BYTES` per key block.
        assert_eq!(Q::CHARGE.bytes(), 32768);
        assert_eq!((KV::CHARGE + KV::CHARGE).bytes(), 32768);

        // softmax, and groupnorm's second kernel — a lone `Tile::BYTES`.
        type Rows = SharedTile<Bf16, 128, 128, Swizzle128B>;
        assert_eq!(Rows::CHARGE.bytes(), 32768);

        // layernorm — `Tile::BYTES + 2 * Parameters::BYTES`, the producer
        // whose sum mixes a tile with two vectors and so was the easiest to
        // get wrong: a missed `2 *` under-charges by 256 bytes and the tile is
        // read while beta is still in flight.
        type Gamma = SharedVec<Bf16, 128>;
        assert_eq!(Gamma::CHARGE.bytes(), 256);
        assert_eq!(
            (Rows::CHARGE + Gamma::CHARGE + Gamma::CHARGE).bytes(),
            33280
        );
    }

    #[test]
    fn ring_stages_wrap() {
        let base = 0x2000usize as *mut u8;
        let ring = unsafe { SharedTileRing::<Bf16, 64, 128, Swizzle128B, 3>::attach(base) };
        assert_eq!(ring.tile(0).base(), base);
        assert_eq!(ring.tile(4).base() as usize, 0x2000 + Panel::BYTES);
        assert_eq!(ring.tile(5).base() as usize, 0x2000 + 2 * Panel::BYTES);
        assert_eq!(ring.tile(6).base(), base);
    }
}
