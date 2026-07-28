//! Global memory: the TMA-descriptor path, and the direct one.
//!
//! # Through a descriptor
//!
//! The device side of a TMA operand is just a `*const TmaDescriptor` kernel
//! parameter; what the type system can hold is the *host* side. A
//! [`GlobalLayout`] is everything `cuTensorMapEncodeTiled` needs about the
//! buffer — base address, per-dimension extents, per-dimension byte strides,
//! dimension 0 the contiguous one — and nothing about the transfer. The box
//! comes from the [`crate::shared::SharedTile`] the map is paired with, so
//! descriptor and tile agree by construction rather than by convention: one
//! box per subtile, [`crate::shared::SharedTile::SUBTILE_COLS`] wide and `R`
//! rows tall, in the tile's own swizzle mode and element type. A layout paired
//! with the wrong element does not typecheck.
//!
//! Only the *rank* is a type parameter. Extents and strides are runtime
//! values because every buffer a kernel here maps has a runtime shape — a
//! GEMM's `K`, a batch count — while the rank decides the arity of the arrays
//! the driver call takes, which is exactly what a const generic is for. A
//! per-dimension compile-time/runtime marker (ThunderKittens' `gl` has one)
//! would buy constant folding in address arithmetic that happens entirely
//! inside the TMA engine, and would need array lengths computed from the
//! markers — `generic_const_exprs`, which this crate avoids.
//!
//! Host-only (`feature = "host"`): `cuTensorMapEncodeTiled` lives in
//! cuda-core, and the device crates never see it.
//!
//! # Without one
//!
//! [`GlobalRows`], [`load_rows`] and [`store_rows`] are the other path —
//! ThunderKittens' `global_to_register.cuh`, and the reason a descriptor is
//! not the only way out of a kernel. An epilogue holding fp32 in registers has
//! no fp32 tile to stage it in ([`crate::shared::Element`] is bf16-only), so
//! through shared memory its choices are to round or to widen; and a small
//! irregular operand — a bias row, a lookup table, a ragged tail — is not
//! worth a descriptor built on the host in the first place.
//!
//! There is no engine here and nothing asynchronous: a thread computes the
//! address of each value it owns from the fragment layout's own
//! `(lane, slot, value) -> (row, column)` map and stores it. Which is why the
//! movers are the only ones in the crate generic over [`FragmentLayout`]
//! rather than pinned to [`BaseLdtm`](crate::reg::BaseLdtm) — `ldmatrix`,
//! `stmatrix` and LDTM each fix a lane map in hardware, and a plain
//! `st.global` fixes nothing.
//!
//! # Two values at a time
//!
//! A map that gives a thread *adjacent* columns has told the mover something:
//! those two accesses are one. [`crate::reg::ColLayout::CONTIGUOUS_VALUES`] is
//! that claim,
//! and [`store_rows`]/[`load_rows`] spend it on `st.global.v2.f32` and
//! `ld.global.v2.f32` — half the memory instructions, and, because the two
//! words of a pair sat in the same 32-byte sector either way, the same
//! transactions carrying twice the useful bytes (#91).
//!
//! It is stated on the layout rather than read off `BaseLdtm`'s arithmetic
//! because the shape set is open (#23): a layout written later gets the
//! default `1` and scalar accesses, which is wrong about nothing, until it
//! claims better. What the mover adds is the half the layout cannot know — a
//! *vector* access must be aligned to its own width, and a cursor's base,
//! stride and column origin are runtime numbers. So the pairing is checked
//! once per call ([`GlobalRows::runs_aligned`]) rather than promised in a
//! safety contract that an odd leading dimension would quietly break.

use crate::reg::{FragmentLayout, RegTile};
use cuda_device::{DisjointSlice, ptx_asm};

/// A row-major window of global memory: a base address and the elements per
/// row that separate one row from the next.
///
/// The device-side counterpart of [`GlobalLayout`], and deliberately much less
/// than one. A descriptor describes a whole tensor because the TMA engine
/// bounds-checks against it; this describes only what address arithmetic
/// needs, because the arithmetic is the calling thread's and nothing checks
/// it. Extents are therefore absent rather than forgotten — see
/// [`store_rows`]' safety contract for what the caller owes instead.
///
/// **fp32 only**, and since #3 that is a choice rather than a constraint. The
/// argument used to be that [`crate::shared::Element`] was implemented for
/// `Bf16` alone, so `GlobalRows<E>` would have had exactly one instantiation
/// and it would have been the wrong one — a register tile *is* fp32, and this
/// path exists to move it without rounding. [`crate::shared::F32`] is that
/// impl now, so what remains is the parameter, a bound, and two
/// `E::read`/`E::write` calls here and in [`load_rows`]/[`store_rows`]. Nobody
/// has asked for a narrow one yet, and a mover with one caller and two element
/// types is generality bought before it is wanted.
#[derive(Clone, Copy)]
pub struct GlobalRows {
    base: *mut f32,
    stride: usize,
}

impl GlobalRows {
    /// Wrap a base address and a row stride in elements — a matrix's leading
    /// dimension, which is `columns` for a packed buffer and larger for a
    /// window into one.
    ///
    /// # Safety
    ///
    /// `base` must be a device address, aligned for `f32`, of a live buffer
    /// that outlives every use of the cursor. Nothing here dereferences it;
    /// the movers' contracts say which elements they touch.
    ///
    /// A read-only source — a `&[f32]` kernel parameter — may be wrapped by
    /// casting its pointer, provided [`load_rows`] is the only thing issued on
    /// the cursor: a write through a pointer derived from a shared reference
    /// is undefined however this type spells it.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f32, stride: usize) -> Self {
        Self { base, stride }
    }

    /// The same, from the [`DisjointSlice`] a kernel's output parameter
    /// arrives as.
    ///
    /// # Safety
    ///
    /// As [`Self::from_raw`], and the cursor is `Copy` where the slice is
    /// borrowed: it carries none of `DisjointSlice`'s uniqueness proof, so
    /// the disjointness of what the threads write is the mover's contract to
    /// keep and no longer the slice's.
    #[inline(always)]
    pub unsafe fn from_slice<Space>(
        slice: &mut DisjointSlice<'_, f32, Space>,
        stride: usize,
    ) -> Self {
        unsafe { Self::from_raw(slice.as_mut_ptr(), stride) }
    }

    /// Elements per row — the leading dimension the cursor was built with.
    #[inline(always)]
    pub const fn stride(self) -> usize {
        self.stride
    }

    /// Index of `(row, column)` from the cursor's base, in elements.
    ///
    /// The whole of the address map, split out from [`Self::at`] so it is a
    /// pure function a host test can call — the same split
    /// [`crate::ldst::fragment_address`] makes on the shared side, and for the
    /// same reason.
    #[inline(always)]
    pub const fn index(self, row: u32, column: u32) -> usize {
        row as usize * self.stride + column as usize
    }

    /// Address of `(row, column)`.
    ///
    /// # Safety
    ///
    /// `(row, column)` must be inside the buffer `base` names.
    #[inline(always)]
    pub const unsafe fn at(self, row: u32, column: u32) -> *mut f32 {
        unsafe { self.base.add(self.index(row, column)) }
    }

    /// Whether a `run`-wide fp32 vector access is legally aligned at *every*
    /// `(row, column + run * k)` this cursor names.
    ///
    /// A vector access must be aligned to its whole width, which is the one
    /// thing a fragment layout cannot promise: it knows `run` divides the
    /// column it hands out, and nothing about where the buffer starts or how
    /// far apart its rows are. Three terms have to agree, and only these
    /// three, because `run * k` is aligned by construction:
    ///
    /// - the base address, so element 0 of row 0 is;
    /// - the column origin the band lands at, which shifts every row equally;
    /// - the row stride, so that every *other* row is too.
    ///
    /// A cursor over a packed buffer at an even leading dimension passes for
    /// pairs; one at an odd `ldc` does not, and gets scalar accesses instead
    /// of a misaligned-address fault.
    #[inline(always)]
    pub fn runs_aligned(self, column: u32, run: usize) -> bool {
        let width = run * size_of::<f32>();
        (self.base as usize + size_of::<f32>() * column as usize).is_multiple_of(width)
            && (size_of::<f32>() * self.stride).is_multiple_of(width)
    }
}

/// `st.global.v2.f32` — the two fp32 at `dest` and `dest + 1` in one
/// instruction.
///
/// Inline PTX for the reason [`crate::ldst::stmatrix_m8n8_x2`] is: the
/// instruction has to be *asked for*. Widening a pair of adjacent stores is a
/// transformation ptxas may only make when the address is provably aligned,
/// and an address built from a runtime leading dimension never is — so the
/// caller that has actually checked ([`GlobalRows::runs_aligned`]) is the only
/// one in a position to spell it.
///
/// # Safety
///
/// `dest` must be 8-byte aligned and name two writable fp32 of a global
/// buffer.
#[inline(always)]
unsafe fn store_pair(dest: *mut f32, first: f32, second: f32) {
    unsafe {
        ptx_asm!(
            "st.global.v2.f32 [%0], {%1, %2};",
            in("l") dest as u64,
            in("f") first,
            in("f") second,
            clobber("memory"),
        );
    }
}

/// `ld.global.v2.f32` — the read direction of [`store_pair`], under the same
/// contract.
///
/// # Safety
///
/// `src` must be 8-byte aligned and name two readable fp32 of a global buffer.
#[inline(always)]
unsafe fn load_pair(src: *const f32) -> (f32, f32) {
    unsafe {
        let first: f32;
        let second: f32;
        ptx_asm!(
            "ld.global.v2.f32 {%0, %1}, [%2];",
            out("=f") first,
            out("=f") second,
            in("l") src as u64,
            clobber("memory"),
        );
        (first, second)
    }
}

/// Write a whole `[M, N]` register tile to `(row, column)` of a row-major
/// global buffer, each thread storing the values its fragment layout gives it.
///
/// The global twin of [`crate::ldst::store_tile`], and the direct answer to an
/// fp32 epilogue: no shared tile, no descriptor, no packing step, and the
/// destination's leading dimension is a runtime number the cursor carries. One
/// `st.global.f32` per owned value, addressed by
/// `L::row_of`/`L::col_of` — the same coordinates
/// [`RegTile::coordinate`] reports, so a kernel that used to open-code this
/// loop against them gets the identical stores.
///
/// A thread's row address is formed once per slot and its values indexed off
/// it, which is what makes the inner loop a run of offsets from one register
/// rather than a multiply each. Under [`BaseLdtm`](crate::reg::BaseLdtm) the
/// four values of a 16-column block sit at column offsets `{0, 1, 8, 9}` — two
/// adjacent pairs, which the layout states as
/// `CONTIGUOUS_VALUES = 2` and this mover spends on one `st.global.v2.f32` per
/// pair when the cursor is aligned for it (#91). A `[128, 128]` accumulator
/// band is 512 stores a lane scalar and 256 paired, over the same eight
/// 32-byte sectors a warp touched either way.
///
/// The choice is made once per call, not once per value: the two spellings are
/// separate unrolled loops and the alignment test picks between them, so
/// neither path carries a branch.
///
/// Nothing here is warp-collective, which is the one way this mover differs
/// from every other one in the crate: `stmatrix` is one instruction fed by 16
/// lanes' addresses, and this is 32 threads each storing their own values. So
/// a lane that does not call it leaves its own values unwritten rather than
/// making an instruction ill-formed — the rectangle is covered when all 32
/// lanes call, and short by exactly one lane's share when one does not.
///
/// # Safety
///
/// The rectangle `row..row + M` by `column..column + N` must lie inside the
/// buffer `dest` names, at `dest.stride()` elements per row.
///
/// The threads write disjoint elements exactly when `L`'s map is injective
/// across the warp, which is a property of the layout and not of this loop —
/// `BaseLdtm`'s is
/// (`base_ldtm_covers_each_coordinate_once`). A layout that replicated a
/// coordinate would have every holder write the same value, so the store
/// stays idempotent, as [`crate::ldst::store_vec`]'s does; what it would not
/// be is a single writer per element.
#[inline(always)]
pub unsafe fn store_rows<const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    dest: GlobalRows,
    row: u32,
    column: u32,
    lane: u32,
    tile: RegTile<M, N, L>,
) {
    unsafe {
        if pairs_are_one_access::<M, N, L>(dest, column) {
            store_rows_in_runs::<2, M, N, L>(dest, row, column, lane, tile)
        } else {
            store_rows_in_runs::<1, M, N, L>(dest, row, column, lane, tile)
        }
    }
}

/// Whether this cursor and this layout together make two consecutive values
/// one memory access — the whole of the decision [`store_rows`] and
/// [`load_rows`] take, so they take it the same way.
///
/// `2` divides the run rather than equalling it: a layout claiming runs of
/// four contains two aligned pairs, and one claiming three contains none.
#[inline(always)]
fn pairs_are_one_access<const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    rows: GlobalRows,
    column: u32,
) -> bool {
    const {
        assert!(
            L::CONTIGUOUS_VALUES > 0 && L::VALUES % L::CONTIGUOUS_VALUES == 0,
            "a layout's contiguous run must divide the values it hands out"
        )
    };
    L::CONTIGUOUS_VALUES % 2 == 0 && rows.runs_aligned(column, 2)
}

/// [`store_rows`] at one access width: `RUN` values per instruction, from a
/// `RUN`-aligned value index. A const parameter and not an argument because
/// the width decides which instruction the loop is made of, and a loop that
/// asked per value would have spent more on the question than on the answer.
///
/// # Safety
///
/// As [`store_rows`], and `RUN > 1` additionally requires
/// [`pairs_are_one_access`].
#[inline(always)]
unsafe fn store_rows_in_runs<
    const RUN: usize,
    const M: usize,
    const N: usize,
    L: FragmentLayout<M, N>,
>(
    dest: GlobalRows,
    row: u32,
    column: u32,
    lane: u32,
    tile: RegTile<M, N, L>,
) {
    unsafe {
        let mut slot = 0usize;
        while slot < L::SLOTS {
            let start = dest.at(row + L::row_of(lane, slot), column);
            let mut value = 0usize;
            while value < L::VALUES {
                let at = start.add(L::col_of(lane, value) as usize);
                match RUN {
                    2 => store_pair(at, tile.get(slot, value), tile.get(slot, value + 1)),
                    _ => at.write(tile.get(slot, value)),
                }
                value += RUN;
            }
            slot += 1;
        }
    }
}

/// Read an `[M, N]` rectangle at `(row, column)` of a row-major global buffer
/// into registers — the inverse of [`store_rows`], over the same addresses.
///
/// What a small or irregular operand takes to reach a kernel that has no
/// reason to build a descriptor for it. A tile this arrives in is an ordinary
/// [`RegTile`]: it can be masked, reduced or mapped, but it is *not* an MMA
/// operand — those come from shared memory, and staging one still means a TMA
/// or a [`crate::ldst::store_tile`].
///
/// The pairing is the same: the addresses are the same addresses, so a layout
/// whose values are adjacent reads them with one `ld.global.v2.f32` under the
/// same alignment test [`store_rows`] applies (#91).
///
/// # Safety
///
/// As [`store_rows`], reading instead of writing: the rectangle must lie
/// inside the buffer, and no other thread may be writing it.
#[inline(always)]
pub unsafe fn load_rows<const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    src: GlobalRows,
    row: u32,
    column: u32,
    lane: u32,
) -> RegTile<M, N, L> {
    unsafe {
        if pairs_are_one_access::<M, N, L>(src, column) {
            load_rows_in_runs::<2, M, N, L>(src, row, column, lane)
        } else {
            load_rows_in_runs::<1, M, N, L>(src, row, column, lane)
        }
    }
}

/// [`load_rows`] at one access width; see [`store_rows_in_runs`].
///
/// # Safety
///
/// As [`load_rows`], and `RUN > 1` additionally requires
/// [`pairs_are_one_access`].
#[inline(always)]
unsafe fn load_rows_in_runs<
    const RUN: usize,
    const M: usize,
    const N: usize,
    L: FragmentLayout<M, N>,
>(
    src: GlobalRows,
    row: u32,
    column: u32,
    lane: u32,
) -> RegTile<M, N, L> {
    unsafe {
        let mut tile = RegTile::<M, N, L>::zero();
        let mut slot = 0usize;
        while slot < L::SLOTS {
            let start = src.at(row + L::row_of(lane, slot), column);
            let mut value = 0usize;
            while value < L::VALUES {
                let at = start.add(L::col_of(lane, value) as usize);
                match RUN {
                    2 => {
                        let (first, second) = load_pair(at);
                        tile.set(slot, value, first);
                        tile.set(slot, value + 1, second);
                    }
                    _ => tile.set(slot, value, at.read()),
                }
                value += RUN;
            }
            slot += 1;
        }
        tile
    }
}

#[cfg(test)]
mod rows_tests {
    use super::*;
    use crate::reg::{BaseLdtm, ColLayout, RowLayout};

    /// A device address is never dereferenced host-side; `index` is pure
    /// arithmetic on it, which is the whole of the map.
    const BASE: *mut f32 = 0x7f00_0000 as *mut f32;

    fn cursor(stride: usize) -> GlobalRows {
        unsafe { GlobalRows::from_raw(BASE, stride) }
    }

    /// Every index a `[M, N]` band's threads form, in dump order.
    fn band_indices<const M: usize, const N: usize>(
        rows: GlobalRows,
        row: u32,
        column: u32,
    ) -> Vec<usize>
    where
        BaseLdtm: FragmentLayout<M, N>,
    {
        let mut indices = Vec::new();
        for lane in 0..32u32 {
            for slot in 0..<BaseLdtm as RowLayout<M>>::SLOTS {
                for value in 0..<BaseLdtm as ColLayout<N>>::VALUES {
                    let (r, c) = RegTile::<M, N, BaseLdtm>::coordinate(lane, slot, value);
                    indices.push(rows.index(row + r, column + c));
                }
            }
        }
        indices
    }

    #[test]
    fn a_row_is_the_leading_dimension_apart_from_the_next() {
        let rows = cursor(1024);
        assert_eq!(rows.index(0, 0), 0);
        assert_eq!(rows.index(0, 7), 7);
        assert_eq!(rows.index(1, 0), 1024);
        assert_eq!(rows.index(3, 5), 3 * 1024 + 5);
        // A packed buffer is the same thing with the stride at the width.
        assert_eq!(cursor(64).index(3, 5), 3 * 64 + 5);
    }

    /// The property the store rests on: a warp's threads between them name
    /// every element of the destination rectangle exactly once, so the stores
    /// are disjoint and the rectangle is fully covered.
    #[test]
    fn a_bands_threads_cover_its_rectangle_exactly_once() {
        const STRIDE: usize = 320;
        let rows = cursor(STRIDE);
        for (row, column) in [(0u32, 0u32), (32, 128), (96, 64)] {
            let mut seen = band_indices::<32, 128>(rows, row, column);
            seen.sort_unstable();
            let mut expected: Vec<usize> = (row..row + 32)
                .flat_map(|r| (column..column + 128).map(move |c| rows.index(r, c)))
                .collect();
            expected.sort_unstable();
            assert_eq!(seen, expected, "band at ({row}, {column})");
        }
    }

    /// The stride is what separates a window from a packed buffer, and it is
    /// the one thing an epilogue gets wrong silently: with `stride == N` a
    /// band's rows are contiguous and every wrong-stride bug hides. At a
    /// wider stride the gaps are the columns this band does not own.
    #[test]
    fn a_wider_stride_leaves_the_columns_between_bands_untouched() {
        const STRIDE: usize = 256;
        let rows = cursor(STRIDE);
        let touched: std::collections::HashSet<usize> =
            band_indices::<32, 128>(rows, 0, 128).into_iter().collect();
        assert_eq!(touched.len(), 32 * 128);
        for row in 0..32 {
            for column in 0..128 {
                assert!(!touched.contains(&(row * STRIDE + column)), "left half");
                assert!(touched.contains(&(row * STRIDE + 128 + column)), "right");
            }
        }
    }

    /// The three terms a vector access needs aligned, one at a time. Each of
    /// the failures below is a real cursor a caller could build, and each
    /// costs the pairing rather than faulting on it (#91).
    #[test]
    fn a_pair_needs_the_base_the_stride_and_the_column_to_agree() {
        // 0x7f00_0000 is 8-byte aligned, and an even stride keeps every row so.
        assert!(cursor(1024).runs_aligned(0, 2));
        assert!(cursor(1024).runs_aligned(128, 2));
        // An odd leading dimension flips the parity every row.
        assert!(!cursor(1023).runs_aligned(0, 2));
        // An odd column origin shifts every row by one element.
        assert!(!cursor(1024).runs_aligned(129, 2));
        // A window starting mid-pair, which is what a `from_raw` on an
        // interior address gives.
        let odd_base = unsafe { GlobalRows::from_raw(BASE.add(1), 1024) };
        assert!(!odd_base.runs_aligned(0, 2));
        assert!(odd_base.runs_aligned(1, 2));
        // A run of one is every address, and is what a layout that claims
        // nothing gets.
        assert!(cursor(1023).runs_aligned(129, 1));
    }

    /// Each thread's values within a slot are a run of offsets from one row
    /// address — the form the mover's inner loop is written in, and the reason
    /// the row multiply happens once per slot rather than once per value.
    #[test]
    fn values_of_a_slot_are_offsets_from_that_slots_row() {
        let rows = cursor(1024);
        for lane in 0..32u32 {
            for slot in 0..<BaseLdtm as RowLayout<32>>::SLOTS {
                let start = rows.index(7 + BaseLdtm::row(lane, slot), 64);
                for value in 0..<BaseLdtm as ColLayout<64>>::VALUES {
                    let (r, c) = RegTile::<32, 64, BaseLdtm>::coordinate(lane, slot, value);
                    assert_eq!(rows.index(7 + r, 64 + c), start + c as usize);
                }
            }
        }
    }
}

#[cfg(feature = "host")]
pub use host::{GlobalLayout, PanelMap, TensorMap, TensorMapElement, TileBox, encode_bf16_panels};

#[cfg(feature = "host")]
mod host {
    use std::error::Error;
    use std::marker::PhantomData;
    use std::mem::MaybeUninit;

    use cuda_core::sys::{CUtensorMapDataType, CUtensorMapSwizzle};
    use cuda_core::{CudaStream, DeviceBuffer};
    use cuda_device::tma::TmaDescriptor;

    use crate::shared::{Bf16, Element, SharedTile, SharedVec, Swizzle, Swizzle128B};

    /// An [`Element`] the TMA engine can name in a tensor map.
    ///
    /// Separate from `Element` because the data-type enum is a cuda-core
    /// value and `Element` is device-side, and separate from
    /// [`crate::shared::MmaElement`] for the same reason that trait is
    /// separate: a buffer the TMA moves need never reach an MMA.
    pub trait TensorMapElement: Element {
        /// The `CUtensorMapDataType` the driver reads this element as.
        const DATA_TYPE: CUtensorMapDataType;
    }

    impl TensorMapElement for Bf16 {
        const DATA_TYPE: CUtensorMapDataType =
            cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }

    /// The box a shared destination wants delivered, as the map has to state
    /// it.
    ///
    /// Implemented by the shared-memory types themselves, so the numbers in a
    /// descriptor are the destination's own constants and cannot be restated
    /// wrong at a call site. The two impls answer the same question
    /// differently, which is the point of the trait rather than a wart:
    ///
    /// - [`SharedTile`] — the box is one *subtile*, not one tile.
    ///   [`SharedTile::tma_load`] issues one `cp.async.bulk.tensor` per stacked
    ///   subtile, lifting the leading coordinate by `SUBTILE_COLS` each time,
    ///   and the box is swizzled because an MMA operand must be.
    /// - [`SharedVec`] — the box is the whole vector, one row tall and
    ///   unswizzled, delivered by a single instruction. A swizzle atom is the
    ///   wrong unit for a one-row run of elements; see [`SharedVec`] for why
    ///   that is a layout decision and not a missing `Swizzle` mode.
    pub trait TileBox {
        /// The element the tile is made of, which is also the map's.
        type Element: TensorMapElement;
        /// Elements along the map's contiguous dimension — one swizzle atom.
        const BOX_COLS: usize;
        /// Rows of the tile, the box's second dimension.
        const BOX_ROWS: usize;
        /// The tile's swizzle, as the driver's enum.
        const SWIZZLE: CUtensorMapSwizzle;
    }

    impl<E: TensorMapElement, const R: usize, const C: usize, S: Swizzle> TileBox
        for SharedTile<E, R, C, S>
    {
        type Element = E;
        const BOX_COLS: usize = Self::SUBTILE_COLS;
        const BOX_ROWS: usize = R;
        const SWIZZLE: CUtensorMapSwizzle = swizzle_mode(S::ATOM_BYTES);
    }

    /// A vector is one unswizzled box `N` elements wide and one row tall. The
    /// row count is what lets a rank-1 layout describe it at all
    /// (`GlobalLayout::descriptor_shape` asserts exactly that), and the
    /// width is `N` itself rather than a constant of a swizzle mode — the
    /// difference from the tile impl, and the reason `Swizzle` is not in the
    /// picture.
    impl<E: TensorMapElement, const N: usize> TileBox for SharedVec<E, N> {
        type Element = E;
        const BOX_COLS: usize = N;
        const BOX_ROWS: usize = 1;
        const SWIZZLE: CUtensorMapSwizzle =
            cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE;
    }

    /// A swizzle atom's width is the mode, so the two cannot be stated apart.
    const fn swizzle_mode(atom_bytes: usize) -> CUtensorMapSwizzle {
        match atom_bytes {
            32 => cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_32B,
            64 => cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
            128 => cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            _ => panic!("a swizzle atom is 32, 64 or 128 bytes"),
        }
    }

    /// A `RANK`-dimensional global buffer of `E`, dimension 0 contiguous.
    ///
    /// The dimension order is the driver's, not a matrix reader's: dimension 0
    /// varies fastest, so a row-major `[rows, columns]` matrix is
    /// `extents = [columns, rows]`. It is also the order the TMA coordinates
    /// in [`SharedTile::tma_load_2d`] are given in, which is why the two are
    /// not reversed here to read more naturally.
    ///
    /// Does not borrow the buffer: the constructors are `unsafe` and the
    /// caller promises the allocation outlives every launch consuming a map
    /// built from it.
    pub struct GlobalLayout<E: Element, const RANK: usize> {
        base: u64,
        extents: [usize; RANK],
        /// Byte stride of each dimension, `[0]` being `E::BYTES`. The driver
        /// takes only dimensions `1..`, but carrying the full array keeps the
        /// length a plain `RANK` instead of `RANK - 1`, which would need
        /// `generic_const_exprs`.
        strides: [u64; RANK],
        _marker: PhantomData<E>,
    }

    impl<E: Element, const RANK: usize> Clone for GlobalLayout<E, RANK> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: Element, const RANK: usize> Copy for GlobalLayout<E, RANK> {}

    impl<E: Element, const RANK: usize> GlobalLayout<E, RANK> {
        /// A densely packed buffer: dimension 0 contiguous, every higher
        /// dimension the product of the extents below it.
        ///
        /// # Safety
        ///
        /// `base` must be the device address of a live buffer of at least
        /// `extents.iter().product()` elements of `E`, staying allocated at
        /// that address for every launch that consumes a map built from it.
        pub unsafe fn packed(base: u64, extents: [usize; RANK]) -> Self {
            let mut strides = [E::BYTES as u64; RANK];
            let mut dimension = 1;
            while dimension < RANK {
                strides[dimension] = strides[dimension - 1] * extents[dimension - 1] as u64;
                dimension += 1;
            }
            unsafe { Self::from_byte_strides(base, extents, strides) }
        }

        /// A buffer whose dimensions are spaced by `element_strides` rather
        /// than by their extents — a matrix with a leading dimension wider
        /// than its columns, or a slice of a larger tensor. `element_strides[0]`
        /// must be 1: the TMA engine reads dimension 0 contiguously.
        ///
        /// # Safety
        ///
        /// As [`Self::packed`], for a buffer spanning
        /// `Σ (extents[i] - 1) * element_strides[i] + 1` elements.
        pub unsafe fn strided(
            base: u64,
            extents: [usize; RANK],
            element_strides: [usize; RANK],
        ) -> Self {
            assert!(
                element_strides[0] == 1,
                "the TMA engine reads dimension 0 contiguously"
            );
            let mut strides = [0u64; RANK];
            let mut dimension = 0;
            while dimension < RANK {
                strides[dimension] = element_strides[dimension] as u64 * E::BYTES as u64;
                dimension += 1;
            }
            unsafe { Self::from_byte_strides(base, extents, strides) }
        }

        unsafe fn from_byte_strides(
            base: u64,
            extents: [usize; RANK],
            strides: [u64; RANK],
        ) -> Self {
            const {
                assert!(
                    RANK >= 1 && RANK <= 5,
                    "cuTensorMapEncodeTiled describes rank 1 through 5"
                )
            };
            Self {
                base,
                extents,
                strides,
                _marker: PhantomData,
            }
        }

        /// The `globalDim` and `boxDim` arrays for a map delivering `T`.
        ///
        /// Split out from [`GlobalLayout::tensor_map`] because it is the whole
        /// of the shape agreement and the only part of a descriptor that can
        /// be read back: `CUtensorMap` is 128 opaque bytes.
        fn descriptor_shape<T: TileBox>(&self) -> ([u64; RANK], [u32; RANK]) {
            const {
                assert!(
                    RANK > 1 || T::BOX_ROWS == 1,
                    "a rank-1 layout has no dimension for a tile's rows"
                )
            };
            let mut extents = [0u64; RANK];
            let mut dimension = 0;
            while dimension < RANK {
                extents[dimension] = self.extents[dimension] as u64;
                dimension += 1;
            }
            // Dimensions past the tile's own two select *which* box, so the
            // TMA moves one of them per instruction.
            let mut shape = [1u32; RANK];
            shape[0] = T::BOX_COLS as u32;
            if RANK > 1 {
                shape[1] = T::BOX_ROWS as u32;
            }
            (extents, shape)
        }
    }

    impl<E: TensorMapElement, const RANK: usize> GlobalLayout<E, RANK> {
        /// Encode and upload the tensor map that loads `T` out of this buffer.
        ///
        /// Every field the caller would otherwise state — data type, box
        /// shape, swizzle mode — comes from `T`, so the only way to build a
        /// descriptor that disagrees with the tile it feeds is to pair the
        /// layout with a different tile type than the kernel loads.
        pub fn tensor_map<T: TileBox<Element = E>>(
            &self,
            stream: &CudaStream,
        ) -> Result<TensorMap, Box<dyn Error>> {
            use cuda_core::sys::{
                CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
                CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
                cuTensorMapEncodeTiled, cudaError_enum_CUDA_SUCCESS,
            };

            let (extents, box_shape) = self.descriptor_shape::<T>();
            self.check_driver_requirements(&extents, &box_shape)?;

            let element_strides = [1u32; RANK];
            let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
            let status = unsafe {
                cuTensorMapEncodeTiled(
                    tensor_map.as_mut_ptr(),
                    E::DATA_TYPE,
                    RANK as u32,
                    self.base as *mut std::ffi::c_void,
                    extents.as_ptr(),
                    // Dimension 0's stride is the element size and is implied.
                    self.strides[1..].as_ptr(),
                    box_shape.as_ptr(),
                    element_strides.as_ptr(),
                    CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
                    T::SWIZZLE,
                    CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
                )
            };
            if status != cudaError_enum_CUDA_SUCCESS {
                return Err(format!(
                    "cuTensorMapEncodeTiled(rank {RANK}, extents {extents:?}, box {box_shape:?}) \
                     failed: {status:?}"
                )
                .into());
            }
            let tensor_map = unsafe { tensor_map.assume_init() };
            Ok(TensorMap {
                descriptor: DeviceBuffer::from_host(stream, &tensor_map.opaque)?,
            })
        }

        /// The alignment and size rules `cuTensorMapEncodeTiled` enforces,
        /// checked here so a violation names the field rather than arriving as
        /// a bare `CUDA_ERROR_INVALID_VALUE`.
        fn check_driver_requirements(
            &self,
            extents: &[u64; RANK],
            box_shape: &[u32; RANK],
        ) -> Result<(), Box<dyn Error>> {
            const ALIGNMENT: u64 = 16;
            if !self.base.is_multiple_of(ALIGNMENT) {
                return Err(
                    format!("tensor map base {:#x} is not 16-byte aligned", self.base).into(),
                );
            }
            for (dimension, stride) in self.strides.iter().enumerate().skip(1) {
                if !stride.is_multiple_of(ALIGNMENT) {
                    return Err(format!(
                        "tensor map stride {stride} of dimension {dimension} is not a multiple of 16 bytes"
                    )
                    .into());
                }
            }
            // The engine moves the contiguous dimension in 16-byte lines, so a
            // box whose innermost width is not a whole number of them has no
            // legal transfer — the rule a swizzled box satisfies for free (its
            // width *is* an atom) and an unswizzled one has to be checked for.
            let innermost = box_shape[0] as u64 * E::BYTES as u64;
            if !innermost.is_multiple_of(ALIGNMENT) {
                return Err(format!(
                    "box dimension 0 is {} bytes wide, not a multiple of 16",
                    innermost
                )
                .into());
            }
            for (dimension, (&extent, &box_extent)) in
                extents.iter().zip(box_shape.iter()).enumerate()
            {
                if extent < box_extent as u64 {
                    return Err(format!(
                        "dimension {dimension} has extent {extent}, smaller than its box {box_extent}"
                    )
                    .into());
                }
                if box_extent > 256 {
                    return Err(format!(
                        "dimension {dimension} has box {box_extent}, over the TMA maximum of 256"
                    )
                    .into());
                }
            }
            Ok(())
        }
    }

    /// A device-resident TMA tensor map: the 128 opaque bytes a kernel's
    /// `*const TmaDescriptor` parameter points at.
    pub struct TensorMap {
        descriptor: DeviceBuffer<u64>,
    }

    impl TensorMap {
        /// The pointer kernels take as their TMA parameter.
        pub fn as_ptr(&self) -> *const TmaDescriptor {
            self.descriptor.cu_deviceptr() as *const TmaDescriptor
        }
    }

    /// The 3-D bf16 panel map, under the name it had before layouts were a
    /// type. [`encode_bf16_panels`] is its constructor.
    pub type PanelMap = TensorMap;

    /// Encode a SWIZZLE_128B tensor map loading `[R, 64]` bf16 subtiles of a
    /// `SharedTile<Bf16, R, C, Swizzle128B>` from one `[rows, C]` panel of a
    /// packed `[planes, rows, C]` staging buffer. The kernel selects the
    /// panel via the third coordinate, the row range via the second, and the
    /// stacked subtile (columns `64*i..64*(i+1)`) via the first — which
    /// [`SharedTile::tma_load`] walks automatically.
    ///
    /// `rows` must be a whole number of `R`-row boxes: the kernel steps its
    /// row coordinate by `R`, so a ragged tail would arrive silently
    /// zero-filled rather than as a short tile. That is this staging layout's
    /// convention and not a rule of [`GlobalLayout`], which leaves edge boxes
    /// to the TMA engine's out-of-bounds fill.
    ///
    /// # Safety
    ///
    /// `base` must be the device address of a live buffer of at least
    /// `planes * rows * C` bf16 elements, staying allocated at that address
    /// for every launch that consumes the returned map.
    pub unsafe fn encode_bf16_panels<const R: usize, const C: usize>(
        stream: &CudaStream,
        base: u64,
        rows: usize,
        planes: usize,
    ) -> Result<PanelMap, Box<dyn Error>> {
        assert!(rows.is_multiple_of(R));
        let layout = unsafe { GlobalLayout::<Bf16, 3>::packed(base, [C, rows, planes]) };
        layout.tensor_map::<SharedTile<Bf16, R, C, Swizzle128B>>(stream)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        type Panel = SharedTile<Bf16, 64, 128, Swizzle128B>;
        type Square = SharedTile<Bf16, 64, 64, Swizzle128B>;

        /// A device address is never dereferenced host-side, and the driver
        /// call is the one thing these tests cannot reach.
        const BASE: u64 = 0x7f00_0000;

        #[test]
        fn packed_strides_are_the_products_below_each_dimension() {
            let layout = unsafe { GlobalLayout::<Bf16, 3>::packed(BASE, [128, 256, 4]) };
            assert_eq!(layout.strides, [2, 128 * 2, 256 * 128 * 2]);
        }

        #[test]
        fn strided_spaces_dimensions_by_their_leading_dimension() {
            // A [rows, columns] matrix inside a wider allocation: the row
            // stride is the pitch, not the column count.
            let layout = unsafe { GlobalLayout::<Bf16, 2>::strided(BASE, [128, 64], [1, 1024]) };
            assert_eq!(layout.strides, [2, 1024 * 2]);
        }

        #[test]
        #[should_panic(expected = "dimension 0 contiguously")]
        fn strided_rejects_a_gap_along_the_contiguous_dimension() {
            let _ = unsafe { GlobalLayout::<Bf16, 2>::strided(BASE, [128, 64], [2, 256]) };
        }

        #[test]
        fn the_box_is_one_subtile_of_the_tile_it_is_paired_with() {
            // What encode_bf16_panels built by hand: [SUBTILE_COLS, R, 1].
            let layout = unsafe { GlobalLayout::<Bf16, 3>::packed(BASE, [128, 256, 4]) };
            let (extents, shape) = layout.descriptor_shape::<Panel>();
            assert_eq!(extents, [128, 256, 4]);
            assert_eq!(shape, [Panel::SUBTILE_COLS as u32, 64, 1]);
            // A wider tile does not widen the box: the extra columns are
            // stacked subtiles the kernel fetches one instruction each.
            let (_, square) = layout.descriptor_shape::<Square>();
            assert_eq!(square, shape);
        }

        #[test]
        fn a_two_dimensional_layout_drops_the_plane_coordinate() {
            let layout = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE, [64, 1024]) };
            let (extents, shape) =
                layout.descriptor_shape::<SharedTile<Bf16, 128, 64, Swizzle128B>>();
            assert_eq!(extents, [64, 1024]);
            assert_eq!(shape, [64, 128]);
        }

        #[test]
        fn the_encoding_matches_what_encode_bf16_panels_wrote_by_hand() {
            // The pre-GlobalLayout builder's arrays, for R = C = 64,
            // rows = 64, planes = 1 — the shape every device test uses.
            let layout = unsafe { GlobalLayout::<Bf16, 3>::packed(BASE, [64, 64, 1]) };
            let (extents, shape) = layout.descriptor_shape::<Square>();
            assert_eq!(extents, [64, 64, 1]);
            assert_eq!(layout.strides[1..], [64 * 2, 64 * 64 * 2]);
            assert_eq!(shape, [64, 64, 1]);
        }

        #[test]
        fn a_vectors_box_is_its_whole_length_at_one_row() {
            // The shape agreement `SharedVec::tma_load` rests on: one box, `N`
            // wide, one instruction. Rank 1 is the shape a bare parameter
            // vector has, and `descriptor_shape`'s assert admits it only
            // because BOX_ROWS is 1.
            let layout = unsafe { GlobalLayout::<Bf16, 1>::packed(BASE, [128]) };
            let (extents, shape) = layout.descriptor_shape::<SharedVec<Bf16, 128>>();
            assert_eq!(extents, [128]);
            assert_eq!(shape, [128]);
            // The same vector out of a `[128, rows]` batch: the row is
            // selected by a coordinate, so the box gains a dimension of 1 and
            // the shared side is unchanged.
            let batched = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE, [128, 64]) };
            let (extents, shape) = batched.descriptor_shape::<SharedVec<Bf16, 128>>();
            assert_eq!(extents, [128, 64]);
            assert_eq!(shape, [128, 1]);
        }

        #[test]
        fn a_vectors_box_is_unswizzled_where_a_tiles_is_not() {
            // The decision, as the descriptor states it. A tile's swizzle also
            // caps its box at one atom, which is exactly what a vector must
            // not inherit: 128 bf16 in one box, not two boxes of 64.
            assert_eq!(
                <SharedVec<Bf16, 128> as TileBox>::SWIZZLE,
                cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE
            );
            assert_eq!(
                <Panel as TileBox>::SWIZZLE,
                cuda_core::sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B
            );
            assert_eq!(<SharedVec<Bf16, 128> as TileBox>::BOX_COLS, 128);
            assert_eq!(<Panel as TileBox>::BOX_COLS, 64);
        }

        #[test]
        fn an_unswizzled_box_must_still_be_a_whole_number_of_16_byte_lines() {
            // A swizzled box satisfies this by being an atom wide; a vector's
            // is its own length, so it is the one box shape that can violate
            // it. 8 bf16 is 16 bytes and legal, 4 is not.
            let layout = unsafe { GlobalLayout::<Bf16, 1>::packed(BASE, [64]) };
            let (extents, legal) = layout.descriptor_shape::<SharedVec<Bf16, 8>>();
            assert!(layout.check_driver_requirements(&extents, &legal).is_ok());
            let (_, narrow) = layout.descriptor_shape::<SharedVec<Bf16, 4>>();
            let error = layout
                .check_driver_requirements(&extents, &narrow)
                .unwrap_err()
                .to_string();
            assert!(error.contains("box dimension 0 is 8 bytes"), "{error}");
        }

        #[test]
        fn driver_requirements_name_the_field_that_violates_them() {
            let layout = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE, [64, 1024]) };
            let (extents, shape) = layout.descriptor_shape::<Square>();
            assert!(layout.check_driver_requirements(&extents, &shape).is_ok());

            // 8 bf16 columns is a 16-byte row stride, which is legal; 4 is not.
            let narrow = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE, [4, 1024]) };
            let (narrow_extents, _) = narrow.descriptor_shape::<Square>();
            let error = narrow
                .check_driver_requirements(&narrow_extents, &shape)
                .unwrap_err()
                .to_string();
            assert!(error.contains("dimension 1"), "{error}");
            assert!(error.contains("multiple of 16"), "{error}");

            let misaligned = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE + 2, [64, 1024]) };
            let error = misaligned
                .check_driver_requirements(&extents, &shape)
                .unwrap_err()
                .to_string();
            assert!(error.contains("16-byte aligned"), "{error}");

            // A buffer shorter than one box cannot deliver one.
            let short = unsafe { GlobalLayout::<Bf16, 2>::packed(BASE, [64, 16]) };
            let (short_extents, _) = short.descriptor_shape::<Square>();
            let error = short
                .check_driver_requirements(&short_extents, &shape)
                .unwrap_err()
                .to_string();
            assert!(error.contains("smaller than its box"), "{error}");
        }
    }
}
