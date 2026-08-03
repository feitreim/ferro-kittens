//! Global memory: the TMA-descriptor path, and the direct one.
//!
//! [`GlobalLayout`] is the host side of a TMA operand — base address, extents,
//! byte strides, dimension 0 contiguous — and [`GlobalLayout::tensor_map`]
//! encodes one for a given [`crate::shared::SharedTile`], so the descriptor and
//! the tile it feeds agree by construction. Host-only (`feature = "host"`).
//!
//! [`GlobalRows`], [`load_rows`], [`store_rows`] and [`store_shared_rows`] are
//! the direct path: ordinary loads and stores against a row-major window at a
//! runtime leading dimension, no engine and nothing asynchronous. It is what an
//! fp32 epilogue, a small irregular operand, or a band staged through shared
//! memory takes to reach global memory.
//!
//! Design notes and measurements: `docs/library/global.md`.

use core::marker::PhantomData;

use crate::reg::{ColLayout, ColVec, FragmentLayout, RegTile};
use crate::shared::{Element, SharedTile, Swizzle};
use cuda_device::DisjointSlice;
use cuda_device::ptx_asm;

/// A row-major window of global memory: a base address, the elements per row
/// that separate one row from the next, and the element those are.
///
/// The device-side counterpart of [`GlobalLayout`], carrying only what address
/// arithmetic needs. There are no extents and nothing bounds-checks a cursor —
/// see [`store_rows`]' safety contract for what the caller owes instead.
///
/// `E` is the *destination's* element, not the register tile's: a tile is fp32
/// either way, and an `E` narrower than fp32 rounds in the store instruction.
///
/// ```
/// use kittens::global::GlobalRows;
/// use kittens::shared::F32;
///
/// // A window into a buffer whose rows are 1024 elements apart.
/// let c = 0x7f00_0000usize as *mut u8;
/// let rows = unsafe { GlobalRows::<F32>::from_raw(c, 1024) };
/// assert_eq!(rows.index(3, 5), 3 * 1024 + 5);
/// ```
pub struct GlobalRows<E: Element> {
    base: *mut u8,
    stride: usize,
    _element: PhantomData<E>,
}

impl<E: Element> Clone for GlobalRows<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element> Copy for GlobalRows<E> {}

impl<E: Element> GlobalRows<E> {
    /// Wrap a base address and a row stride in elements — a matrix's leading
    /// dimension, which is `columns` for a packed buffer and larger for a
    /// window into one.
    ///
    /// # Safety
    ///
    /// - `base` is a device address of a live buffer that outlives every use of
    ///   the cursor, and is aligned for `E`.
    /// - A cursor built from a *shared* reference's pointer — a `&[f32]` kernel
    ///   parameter — may only be read from ([`load_rows`]).
    ///
    /// Nothing here dereferences `base`; the movers' contracts say which
    /// elements they touch.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut u8, stride: usize) -> Self {
        Self {
            base,
            stride,
            _element: PhantomData,
        }
    }

    /// The same, from the [`DisjointSlice`] a kernel's output parameter
    /// arrives as.
    ///
    /// `T` is whatever storage word the launch declared — `f32` for an fp32
    /// output, `u16` for a bf16 one, since the device crate has no bf16 scalar.
    /// Only its *width* has to agree with `E`, which a const assert checks.
    ///
    /// ```no_run
    /// use kittens::global::GlobalRows;
    /// use kittens::shared::Bf16;
    /// use cuda_device::DisjointSlice;
    ///
    /// # unsafe fn epilogue(c: &mut DisjointSlice<'_, u16>, ldc: usize) {
    /// // A bf16 output arrives as `u16` storage words; the cursor names it bf16.
    /// let dest = unsafe { GlobalRows::<Bf16>::from_slice(c, ldc) };
    /// # let _ = dest;
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// - As [`Self::from_raw`].
    /// - The cursor is `Copy` and carries none of `DisjointSlice`'s uniqueness
    ///   proof, so keeping the threads' writes disjoint becomes the mover's
    ///   contract rather than the slice's.
    #[inline(always)]
    pub unsafe fn from_slice<T, Space>(
        slice: &mut DisjointSlice<'_, T, Space>,
        stride: usize,
    ) -> Self {
        const {
            assert!(
                size_of::<T>() == E::BYTES,
                "a cursor's element must be as wide as the slice's storage word"
            )
        };
        unsafe { Self::from_raw(slice.as_mut_ptr().cast(), stride) }
    }

    /// Elements per row — the leading dimension the cursor was built with.
    #[inline(always)]
    pub const fn stride(self) -> usize {
        self.stride
    }

    /// Index of `(row, column)` from the cursor's base, in elements.
    ///
    /// The whole of the address map, split out from [`Self::at`] so it is a
    /// pure function a host test can call.
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
    pub const unsafe fn at(self, row: u32, column: u32) -> *mut u8 {
        unsafe { self.base.add(E::BYTES * self.index(row, column)) }
    }

    /// Whether a `run`-element vector access is legally aligned at *every*
    /// `(row, column + run * k)` this cursor names.
    ///
    /// A vector access must be aligned to its whole width, so three terms have
    /// to agree — and only these three, because `run * k` is aligned by
    /// construction:
    ///
    /// - the base address, so element 0 of row 0 is;
    /// - the column origin the band lands at, which shifts every row equally;
    /// - the row stride, so that every *other* row is too.
    ///
    /// A packed buffer at an even leading dimension passes for pairs; an odd
    /// `ldc` does not, and gets scalar accesses instead of a misaligned-address
    /// fault. The answer does not depend on `E`: a narrower element halves the
    /// offset a column costs *and* the width to align to.
    ///
    /// ```
    /// use kittens::global::GlobalRows;
    /// use kittens::shared::F32;
    ///
    /// let base = 0x7f00_0000usize as *mut u8;
    /// assert!(unsafe { GlobalRows::<F32>::from_raw(base, 1024) }.runs_aligned(128, 2));
    /// // An odd leading dimension flips the parity every row.
    /// assert!(!unsafe { GlobalRows::<F32>::from_raw(base, 1023) }.runs_aligned(128, 2));
    /// ```
    #[inline(always)]
    pub fn runs_aligned(self, column: u32, run: usize) -> bool {
        let width = run * E::BYTES;
        (self.base as usize + E::BYTES * column as usize).is_multiple_of(width)
            && (E::BYTES * self.stride).is_multiple_of(width)
    }
}

/// Write a whole `[M, N]` register tile to `(row, column)` of a row-major
/// global buffer, each thread storing the values its fragment layout gives it.
///
/// The global twin of [`crate::ldst::store_tile`], and the direct answer to a
/// register epilogue: no shared tile, no descriptor, and the destination's
/// leading dimension is a runtime number the cursor carries. One store per
/// owned value, at the coordinates [`RegTile::coordinate`] reports.
///
/// The tile is fp32 whatever `E` is. An `E` narrower than fp32 rounds here, in
/// [`Element::write`] or [`Element::write_pair`], and that is the only place it
/// does — which is why the element is the cursor's and not the tile's.
///
/// Two values a layout declares adjacent
/// ([`crate::reg::ColLayout::CONTIGUOUS_VALUES`]) become one access when the
/// cursor is aligned for it, halving the store count. The test is
/// [`GlobalRows::runs_aligned`], taken once per call rather than per value, so
/// neither path carries a branch.
///
/// Not warp-collective: this is 32 threads each storing their own values, so a
/// lane that does not call it leaves its own values unwritten rather than
/// making an instruction ill-formed.
///
/// ```no_run
/// use kittens::global::{GlobalRows, store_rows};
/// use kittens::reg::{BaseLdtm, RegTile};
/// use kittens::shared::Bf16;
///
/// # unsafe fn epilogue(c: *mut u8, ldc: usize, lane: u32) {
/// let accumulator = RegTile::<32, 128, BaseLdtm>::zero();
/// let dest = unsafe { GlobalRows::<Bf16>::from_raw(c, ldc) };
/// // Rounds fp32 registers to bf16 in the store instruction itself.
/// unsafe { store_rows(dest, 0, 128, lane, accumulator) };
/// # }
/// ```
///
/// # Safety
///
/// - The rectangle `row..row + M` by `column..column + N` lies inside the
///   buffer `dest` names, at `dest.stride()` elements per row.
/// - `L`'s map is injective across the warp, or the threads do not write
///   disjoint elements. `BaseLdtm`'s is
///   (`base_ldtm_covers_each_coordinate_once`); a layout that replicated a
///   coordinate would keep the store idempotent but lose the single writer.
#[inline(always)]
pub unsafe fn store_rows<E: Element, const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    dest: GlobalRows<E>,
    row: u32,
    column: u32,
    lane: u32,
    tile: RegTile<M, N, L>,
) {
    unsafe {
        if pairs_are_one_access::<E, N, L>(dest, column) {
            store_rows_in_runs::<2, E, M, N, L>(dest, row, column, lane, tile)
        } else {
            store_rows_in_runs::<1, E, M, N, L>(dest, row, column, lane, tile)
        }
    }
}

/// Whether this cursor and this layout together make two consecutive values
/// one memory access — the whole of the decision [`store_rows`], [`load_rows`]
/// and [`load_cols`] take, so they take it the same way.
///
/// Bounded on [`ColLayout`] and not [`FragmentLayout`] because the pairing is
/// a claim about columns alone; that is what lets the vector mover share it.
#[inline(always)]
fn pairs_are_one_access<E: Element, const N: usize, L: ColLayout<N>>(
    rows: GlobalRows<E>,
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
/// `RUN`-aligned value index. A const parameter because the width decides which
/// instruction the loop is made of.
///
/// # Safety
///
/// As [`store_rows`], and `RUN > 1` additionally requires
/// [`pairs_are_one_access`].
#[inline(always)]
unsafe fn store_rows_in_runs<
    const RUN: usize,
    E: Element,
    const M: usize,
    const N: usize,
    L: FragmentLayout<M, N>,
>(
    dest: GlobalRows<E>,
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
                let at = start.add(E::BYTES * L::col_of(lane, value) as usize);
                match RUN {
                    2 => E::write_pair(at, tile.get(slot, value), tile.get(slot, value + 1)),
                    _ => E::write(at, tile.get(slot, value)),
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
/// reason to build a descriptor for it. The tile it arrives in is an ordinary
/// [`RegTile`]: it can be masked, reduced or mapped, but it is *not* an MMA
/// operand — those come from shared memory, and staging one still means a TMA
/// or a [`crate::ldst::store_tile`].
///
/// The addresses are [`store_rows`]' addresses, so the same pairing applies
/// under the same alignment test.
///
/// ```no_run
/// use kittens::global::{GlobalRows, load_rows};
/// use kittens::reg::{BaseLdtm, RegTile};
/// use kittens::shared::F32;
///
/// # unsafe fn bias(table: *mut u8, ld: usize, lane: u32) -> RegTile<16, 64, BaseLdtm> {
/// let src = unsafe { GlobalRows::<F32>::from_raw(table, ld) };
/// unsafe { load_rows(src, 0, 0, lane) }
/// # }
/// ```
///
/// # Safety
///
/// - As [`store_rows`], reading instead of writing: the rectangle lies inside
///   the buffer `src` names.
/// - No other thread is writing that rectangle.
#[inline(always)]
pub unsafe fn load_rows<E: Element, const M: usize, const N: usize, L: FragmentLayout<M, N>>(
    src: GlobalRows<E>,
    row: u32,
    column: u32,
    lane: u32,
) -> RegTile<M, N, L> {
    unsafe {
        if pairs_are_one_access::<E, N, L>(src, column) {
            load_rows_in_runs::<2, E, M, N, L>(src, row, column, lane)
        } else {
            load_rows_in_runs::<1, E, M, N, L>(src, row, column, lane)
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
    E: Element,
    const M: usize,
    const N: usize,
    L: FragmentLayout<M, N>,
>(
    src: GlobalRows<E>,
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
                let at = start.add(E::BYTES * L::col_of(lane, value) as usize);
                match RUN {
                    2 => {
                        let (first, second) = E::read_pair(at);
                        tile.set(slot, value, first);
                        tile.set(slot, value + 1, second);
                    }
                    _ => tile.set(slot, value, E::read(at)),
                }
                value += RUN;
            }
            slot += 1;
        }
        tile
    }
}

/// Read `N` consecutive elements of one global row as the per-column operand a
/// `col_map` takes.
///
/// The global twin of [`crate::ldst::load_vec`], which reads the same
/// [`ColVec`] out of a [`crate::shared::SharedVec`]. A parameter vector — a
/// norm's weight, a bias — is one row of memory and `N` values that every row
/// of the tile it multiplies shares, so this is what it takes to reach the
/// registers that multiply by it when its length is a runtime number and a
/// `SharedVec<E, N>` is therefore not available to stage it in.
///
/// `L::VALUES` registers, where reading the same vector through [`load_rows`]
/// against a stride-zero cursor costs `L::SLOTS * L::VALUES` and holds every
/// value once per row the thread owns. Addresses, and the
/// [`crate::reg::ColLayout::CONTIGUOUS_VALUES`] pairing spent on them, are
/// [`load_rows`]' own for the single row `row`.
///
/// Not warp-collective: 32 threads each read the columns their layout gives
/// them, and a lane that does not call it leaves its own values unset.
///
/// ```no_run
/// use kittens::global::{GlobalRows, load_cols};
/// use kittens::reg::{BaseLdtm, ColVec};
/// use kittens::shared::F32;
///
/// # unsafe fn weights(gamma: *mut u8, column: u32, lane: u32) -> ColVec<64, BaseLdtm> {
/// // A `[dim]` parameter vector: one row, and no stride to speak of.
/// let src = unsafe { GlobalRows::<F32>::from_raw(gamma, 0) };
/// unsafe { load_cols(src, 0, column, lane) }
/// # }
/// ```
///
/// # Safety
///
/// - `column..column + N` of row `row` lies inside the buffer `src` names.
/// - No other thread is writing those elements.
#[inline(always)]
pub unsafe fn load_cols<E: Element, const N: usize, L: ColLayout<N>>(
    src: GlobalRows<E>,
    row: u32,
    column: u32,
    lane: u32,
) -> ColVec<N, L> {
    unsafe {
        if pairs_are_one_access::<E, N, L>(src, column) {
            load_cols_in_runs::<2, E, N, L>(src, row, column, lane)
        } else {
            load_cols_in_runs::<1, E, N, L>(src, row, column, lane)
        }
    }
}

/// [`load_cols`] at one access width; the single-row form of
/// [`load_rows_in_runs`].
///
/// # Safety
///
/// As [`load_cols`], and `RUN > 1` additionally requires
/// [`pairs_are_one_access`].
#[inline(always)]
unsafe fn load_cols_in_runs<const RUN: usize, E: Element, const N: usize, L: ColLayout<N>>(
    src: GlobalRows<E>,
    row: u32,
    column: u32,
    lane: u32,
) -> ColVec<N, L> {
    unsafe {
        let mut cols = ColVec::<N, L>::splat(0.0);
        let start = src.at(row, column);
        let mut value = 0usize;
        while value < L::VALUES {
            let at = start.add(E::BYTES * L::col_of(lane, value) as usize);
            match RUN {
                2 => {
                    let (first, second) = E::read_pair(at);
                    cols.set(value, first);
                    cols.set(value + 1, second);
                }
                _ => cols.set(value, E::read(at)),
            }
            value += RUN;
        }
        cols
    }
}

/// The unit both ends of a staged drain agree on: the largest run contiguous in
/// shared *and* in global memory, and the widest access PTX has
/// (`st.global.v4.b32`).
const CHUNK_BYTES: usize = 16;

/// Copy a whole `[R, C]` shared tile out to the `(row, column)` rectangle of a
/// row-major global buffer, `THREADS` threads splitting its chunks between
/// them.
///
/// The other half of the epilogue [`crate::ldst::store_tile`] starts: a band
/// that reached shared memory through `stmatrix` leaves it through this, in
/// accesses up to 16 bytes wide whose addresses across a warp are one
/// contiguous run. No descriptor, no engine, and no proxy fence.
///
/// Cooperative and not collective. `THREADS` threads share the tile's chunks by
/// a flat index, and there is no barrier, no fence and no wait inside. A
/// warp-scope caller passes `(lane, THREADS = 32)`; the four epilogue warps of
/// a CTA pass `(threadIdx.x, 128)`, which is what makes a warp's 32 accesses
/// land on consecutive chunks of one destination row.
///
/// [`access_width`] picks the widest access this cursor and column origin
/// admit, once per call, so an odd `ldc` gets narrower stores rather than a
/// misaligned-address fault. Nothing here rounds: a tile is already `E`.
///
/// ```no_run
/// use kittens::global::{GlobalRows, store_shared_rows};
/// use kittens::shared::{Bf16, SharedTile, Swizzle128B};
///
/// # unsafe fn drain(staged: *mut u8, c: *mut u8, ldc: usize, thread: u32) {
/// let tile = unsafe { SharedTile::<Bf16, 64, 128, Swizzle128B>::from_raw(staged) };
/// let dest = unsafe { GlobalRows::<Bf16>::from_raw(c, ldc) };
/// // The four epilogue warps of a CTA, after the `bar.sync` that publishes
/// // the `stmatrix` writes.
/// unsafe { store_shared_rows::<_, 64, 128, _, 128>(dest, 0, 0, thread, tile) };
/// # }
/// ```
///
/// # Safety
///
/// - The rectangle `row..row + R` by `column..column + C` lies inside the
///   buffer `dest` names, at `dest.stride()` elements per row.
/// - Exactly the threads `0..THREADS` call this, each passing its own `thread`.
///   There is no barrier inside, so a thread that skips it leaves its own
///   chunks unwritten rather than hanging the block.
/// - The tile's bytes are already visible to every calling thread: a `bar.sync`
///   since the last write by another thread of the CTA, and no proxy fence.
/// - The tile is not written again until every calling thread has returned,
///   which is another barrier and also the caller's.
#[inline(always)]
pub unsafe fn store_shared_rows<
    E: Element,
    const R: usize,
    const C: usize,
    S: Swizzle,
    const THREADS: u32,
>(
    dest: GlobalRows<E>,
    row: u32,
    column: u32,
    thread: u32,
    tile: SharedTile<E, R, C, S>,
) {
    const {
        assert!(
            E::BYTES == 2 || E::BYTES == 4,
            "the access ladder's narrowest rung is two bytes, so a narrower element has none"
        )
    };
    const { assert!(THREADS > 0, "a drain needs at least one thread") };
    const {
        assert!(
            (R * C * E::BYTES / CHUNK_BYTES).is_multiple_of(THREADS as usize),
            "the drain's threads must divide the tile's 16-byte chunks between them exactly"
        )
    };
    unsafe {
        match access_width(dest, column) {
            CHUNK_BYTES => drain_in_accesses::<CHUNK_BYTES, E, R, C, S, THREADS>(
                dest, row, column, thread, tile,
            ),
            8 => drain_in_accesses::<8, E, R, C, S, THREADS>(dest, row, column, thread, tile),
            // Const-false at a 4-byte element, which is what keeps this arm
            // from being emitted for one: `access_width` cannot return 2 there.
            2 if E::BYTES == 2 => {
                drain_in_accesses::<2, E, R, C, S, THREADS>(dest, row, column, thread, tile)
            }
            _ => drain_in_accesses::<4, E, R, C, S, THREADS>(dest, row, column, thread, tile),
        }
    }
}

/// Bytes per access the ladder settles on for this cursor and column origin —
/// the whole of [`store_shared_rows`]' width decision, as a pure function.
///
/// Descends 16, 8, 4, `E::BYTES`, each rung a [`GlobalRows::runs_aligned`] at
/// the run of elements that width is worth. The bottom rung is the element and
/// not a fixed 2 bytes, so the ladder always terminates: a cursor is aligned for
/// its own element by [`GlobalRows::from_raw`]'s contract. At fp32 that rung
/// *is* the 4-byte one and 2 is never returned.
///
/// ```
/// use kittens::global::{GlobalRows, access_width};
/// use kittens::shared::Bf16;
///
/// // A bf16 `C` at a leading dimension of 208, one column origin per rung.
/// let ldc = |column| unsafe {
///     access_width(GlobalRows::<Bf16>::from_raw(0x7f00_0000usize as *mut u8, 208), column)
/// };
/// assert_eq!(ldc(64), 16);
/// assert_eq!(ldc(68), 8);
/// assert_eq!(ldc(65), 2);
/// ```
pub fn access_width<E: Element>(dest: GlobalRows<E>, column: u32) -> usize {
    if dest.runs_aligned(column, CHUNK_BYTES / E::BYTES) {
        CHUNK_BYTES
    } else if dest.runs_aligned(column, 8 / E::BYTES) {
        8
    } else if E::BYTES == 2 && !dest.runs_aligned(column, 2) {
        2
    } else {
        4
    }
}

/// The tile `(row, chunk)` that flat work item `item` names, in a tile of
/// `per_row` chunks per *logical* row.
///
/// Chunks are counted across the whole logical row, so this is already the
/// index [`crate::shared::SwizzledChunks::at`] takes and a column past the
/// first stacked subtile needs nothing extra. Split out from the loop so a host
/// test can ask whether the split is a partition.
const fn drain_item(item: usize, per_row: usize) -> (usize, usize) {
    (item / per_row, item % per_row)
}

/// [`store_shared_rows`] at one access width, in bytes. A const parameter for
/// [`store_rows_in_runs`]' reason: the width decides which instruction the loop
/// is made of.
///
/// # Safety
///
/// As [`store_shared_rows`], and `WIDTH` must be what [`access_width`] returned
/// for this cursor and column.
#[inline(always)]
unsafe fn drain_in_accesses<
    const WIDTH: usize,
    E: Element,
    const R: usize,
    const C: usize,
    S: Swizzle,
    const THREADS: u32,
>(
    dest: GlobalRows<E>,
    row: u32,
    column: u32,
    thread: u32,
    tile: SharedTile<E, R, C, S>,
) {
    unsafe {
        let chunks = tile.chunk_writer();
        let per_row = C * E::BYTES / CHUNK_BYTES;
        let chunk_columns = CHUNK_BYTES / E::BYTES;
        let mut item = thread as usize;
        while item < R * per_row {
            let (tile_row, chunk) = drain_item(item, per_row);
            let from = chunks.at(tile_row, chunk);
            let to = dest.at(
                row + tile_row as u32,
                column + (chunk * chunk_columns) as u32,
            );
            let mut byte = 0usize;
            while byte < CHUNK_BYTES {
                copy_bytes::<WIDTH>(from.add(byte), to.add(byte));
                byte += WIDTH;
            }
            item += THREADS as usize;
        }
    }
}

/// Move `WIDTH` bytes from shared memory to global memory, unchanged.
///
/// A byte copy and not a value move: the tile already holds `E`, so there is
/// nothing to pack, unpack or round.
///
/// The two vector widths are inline PTX because the compiler may only widen
/// adjacent accesses when the address is provably aligned, which one built from
/// a runtime leading dimension never is; [`access_width`] is what has actually
/// checked. The two narrow widths are a single instruction either way.
///
/// # Safety
///
/// - `from` names `WIDTH` readable bytes of shared memory.
/// - `to` names `WIDTH` writable bytes of global memory.
/// - Both are aligned to `WIDTH`.
#[inline(always)]
unsafe fn copy_bytes<const WIDTH: usize>(from: *const u8, to: *mut u8) {
    const {
        assert!(
            WIDTH == 16 || WIDTH == 8 || WIDTH == 4 || WIDTH == 2,
            "a shared-to-global access is 2, 4, 8 or 16 bytes wide"
        )
    };
    unsafe {
        match WIDTH {
            16 => write_v4(to, read_shared_v4(from)),
            8 => write_v2(to, read_shared_v2(from)),
            4 => *(to as *mut u32) = *(from as *const u32),
            _ => *(to as *mut u16) = *(from as *const u16),
        }
    }
}

/// One whole swizzle chunk out of shared memory: `ld.shared.v4.b32`.
///
/// A load into registers rather than a snippet fused with the store, so the two
/// stay separate instructions the scheduler can put other work between.
///
/// # Safety
///
/// - `from` is a 16-byte-aligned shared-memory address with 16 readable bytes.
#[inline(always)]
unsafe fn read_shared_v4(from: *const u8) -> [u32; 4] {
    unsafe {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!(
            "{ .reg .u64 src; cvta.to.shared.u64 src, %4; ld.shared.v4.b32 {%0, %1, %2, %3}, [src]; }",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") from as u64,
            clobber("memory"),
        );
        [a, b, c, d]
    }
}

/// Half a chunk: `ld.shared.v2.b32`.
///
/// # Safety
///
/// - As [`read_shared_v4`], 8-byte aligned with 8 readable bytes.
#[inline(always)]
unsafe fn read_shared_v2(from: *const u8) -> [u32; 2] {
    unsafe {
        let (a, b): (u32, u32);
        ptx_asm!(
            "{ .reg .u64 src; cvta.to.shared.u64 src, %2; ld.shared.v2.b32 {%0, %1}, [src]; }",
            out("=r") a,
            out("=r") b,
            in("l") from as u64,
            clobber("memory"),
        );
        [a, b]
    }
}

/// 16 bytes into global memory in one access: `st.global.v4.b32`.
///
/// The instruction the whole staging round trip is for — a warp of 32 lanes
/// issuing it on consecutive chunks writes 512 contiguous bytes.
///
/// # Safety
///
/// - `to` is a 16-byte-aligned global address with 16 writable bytes.
#[inline(always)]
unsafe fn write_v4(to: *mut u8, words: [u32; 4]) {
    unsafe {
        ptx_asm!(
            "st.global.v4.b32 [%0], {%1, %2, %3, %4};",
            in("l") to as u64,
            in("r") words[0],
            in("r") words[1],
            in("r") words[2],
            in("r") words[3],
            clobber("memory"),
        );
    }
}

/// `st.global.v2.b32` — the rung below [`write_v4`].
///
/// # Safety
///
/// - As [`write_v4`], 8-byte aligned with 8 writable bytes.
#[inline(always)]
unsafe fn write_v2(to: *mut u8, words: [u32; 2]) {
    unsafe {
        ptx_asm!(
            "st.global.v2.b32 [%0], {%1, %2};",
            in("l") to as u64,
            in("r") words[0],
            in("r") words[1],
            clobber("memory"),
        );
    }
}

#[cfg(test)]
mod rows_tests {
    use super::*;
    use crate::reg::{BaseLdtm, ColLayout, RowLayout};
    use crate::shared::{Bf16, F32};

    /// A device address is never dereferenced host-side; `index` is pure
    /// arithmetic on it, which is the whole of the map.
    const BASE: *mut u8 = 0x7f00_0000 as *mut u8;

    fn cursor(stride: usize) -> GlobalRows<F32> {
        unsafe { GlobalRows::from_raw(BASE, stride) }
    }

    /// Every index a `[M, N]` band's threads form, in dump order.
    fn band_indices<const M: usize, const N: usize>(
        rows: GlobalRows<F32>,
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
    /// costs the pairing rather than faulting on it.
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
        let odd_base = unsafe { GlobalRows::<F32>::from_raw(BASE.add(4), 1024) };
        assert!(!odd_base.runs_aligned(0, 2));
        assert!(odd_base.runs_aligned(1, 2));
        // A run of one is every address, and is what a layout that claims
        // nothing gets.
        assert!(cursor(1023).runs_aligned(129, 1));
    }

    /// **The element does not move this test**, which is worth a case of its
    /// own because the obvious expectation is that it does. A narrower element
    /// halves the offset a column costs *and* halves the width a pair must be
    /// aligned to, so the two cancel and what is left is the same question at
    /// either element: an even stride and an even column origin, off a base
    /// aligned for the element it is.
    ///
    /// Every row below is the fp32 case above with the element swapped and the
    /// same answer.
    #[test]
    fn the_pairing_test_is_the_same_at_either_element() {
        let bf16 = |stride| unsafe { GlobalRows::<Bf16>::from_raw(BASE, stride) };
        for stride in [1024usize, 1023] {
            for column in [0u32, 128, 129] {
                assert_eq!(
                    bf16(stride).runs_aligned(column, 2),
                    cursor(stride).runs_aligned(column, 2),
                    "stride {stride} column {column}"
                );
            }
        }
        // A base one element into a pair, which is the interior address above
        // at each element's own width.
        let interior = |offset| unsafe { GlobalRows::<Bf16>::from_raw(BASE.add(offset), 1024) };
        assert!(!interior(2).runs_aligned(0, 2));
        assert!(interior(2).runs_aligned(1, 2));
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

#[cfg(test)]
mod drain_tests {
    use super::*;
    use crate::shared::{Bf16, F32};

    /// 16-byte aligned, as every device allocation is, so the *column* is what
    /// moves the ladder below and not the base.
    const BASE: *mut u8 = 0x7f00_0000 as *mut u8;
    /// The leading dimension the device case uses. A multiple of 8 bf16, so
    /// the stride never limits the width and each rung below is the column's
    /// doing alone.
    const PITCH: usize = 208;
    /// Chunks in one logical row of a `[64, 128]` bf16 tile, and its rows.
    const PER_ROW: usize = 128 * 2 / CHUNK_BYTES;
    const ROWS: usize = 64;
    /// Elements one chunk holds at bf16.
    const CHUNK_COLUMNS: usize = CHUNK_BYTES / 2;

    fn bf16(stride: usize) -> GlobalRows<Bf16> {
        unsafe { GlobalRows::from_raw(BASE, stride) }
    }

    /// Where work item `item` lands in the destination, in elements from the
    /// cursor's base — the drain's whole address map, replayed off its own
    /// [`drain_item`].
    fn destination(rows: GlobalRows<Bf16>, row: u32, column: u32, item: usize) -> usize {
        let (tile_row, chunk) = drain_item(item, PER_ROW);
        rows.index(
            row + tile_row as u32,
            column + (chunk * CHUNK_COLUMNS) as u32,
        )
    }

    /// The ladder, one rung per column origin. Every one of these is a cursor
    /// a caller could build, and none of them faults: a column that cannot
    /// carry a 16-byte store gets an 8-byte one, and so down to the element.
    #[test]
    fn the_ladder_takes_the_widest_access_the_cursor_admits() {
        assert_eq!(access_width(bf16(PITCH), 64), 16);
        assert_eq!(access_width(bf16(PITCH), 68), 8);
        assert_eq!(access_width(bf16(PITCH), 66), 4);
        assert_eq!(access_width(bf16(PITCH), 65), 2);
        // The stride is the other term that shifts every row at once: an odd
        // leading dimension flips the parity per row and costs everything.
        assert_eq!(access_width(bf16(PITCH + 1), 64), 2);
        // And an even stride stops the ladder wherever the row it lands on
        // stops it: 212 bf16 is 424 bytes, which carries an 8-byte access to
        // every row and a 16-byte one only to the first.
        assert_eq!(access_width(bf16(212), 64), 8);
        assert_eq!(access_width(bf16(210), 64), 4);
    }

    /// **fp32 has no bottom rung to reach**, which is not the obvious result:
    /// a cursor is aligned for its own element by contract, so a 4-byte access
    /// is legal at every column a 4-byte element names and the 2-byte rung is
    /// unreachable. That is what the `E::BYTES == 2` guard in
    /// [`store_shared_rows`] spends — the arm is not emitted at fp32 at all.
    #[test]
    fn a_four_byte_element_never_descends_below_its_own_width() {
        let f32s = |stride, column| {
            access_width(unsafe { GlobalRows::<F32>::from_raw(BASE, stride) }, column)
        };
        assert_eq!(f32s(PITCH, 64), 16);
        for column in [65u32, 66, 67, 68] {
            assert!(f32s(PITCH, column) >= 4, "column {column}");
        }
        for stride in [PITCH + 1, PITCH + 2, PITCH + 3] {
            assert!(f32s(stride, 64) >= 4, "stride {stride}");
        }
    }

    /// The property the drain rests on: the calling threads between them name
    /// every chunk of the tile exactly once, so the stores are disjoint and the
    /// rectangle is fully covered.
    #[test]
    fn the_threads_partition_the_tiles_chunks() {
        for threads in [32usize, 64, 128, 256] {
            let mut seen = Vec::new();
            let mut per_thread = Vec::new();
            for thread in 0..threads {
                let mut item = thread;
                let mut mine = 0usize;
                while item < ROWS * PER_ROW {
                    seen.push(drain_item(item, PER_ROW));
                    mine += 1;
                    item += threads;
                }
                per_thread.push(mine);
            }
            seen.sort_unstable();
            let expected: Vec<(usize, usize)> = (0..ROWS)
                .flat_map(|row| (0..PER_ROW).map(move |chunk| (row, chunk)))
                .collect();
            assert_eq!(seen, expected, "{threads} threads");
            // Equal shares, which is what the const assert buys and what keeps
            // the loop's trip count off the thread index.
            assert!(per_thread.iter().all(|&count| count == per_thread[0]));
        }
    }

    /// And the destination side of the same claim: the elements those chunks
    /// carry are exactly the rectangle, at a stride wider than the tile so a
    /// wrong leading dimension is a wrong answer rather than a benign one.
    #[test]
    fn the_chunks_cover_the_destination_rectangle_exactly_once() {
        const ROW: u32 = 8;
        const COLUMN: u32 = 64;
        let rows = bf16(PITCH);
        let mut touched = Vec::new();
        for item in 0..ROWS * PER_ROW {
            let start = destination(rows, ROW, COLUMN, item);
            touched.extend(start..start + CHUNK_COLUMNS);
        }
        touched.sort_unstable();
        let mut expected: Vec<usize> = (ROW..ROW + ROWS as u32)
            .flat_map(|row| (COLUMN..COLUMN + 128).map(move |column| rows.index(row, column)))
            .collect();
        expected.sort_unstable();
        assert_eq!(touched, expected);
    }

    /// Why the split is by chunk and not by row: consecutive threads take
    /// consecutive chunks of one destination row, so a warp's accesses are one
    /// contiguous run — which is the entire point of routing the band through
    /// shared memory instead of storing it from the fragment layout.
    #[test]
    fn consecutive_threads_take_consecutive_chunks_of_a_row() {
        let rows = bf16(PITCH);
        for item in 0..PER_ROW - 1 {
            assert_eq!(
                destination(rows, 0, 64, item + 1) - destination(rows, 0, 64, item),
                CHUNK_COLUMNS,
                "item {item}"
            );
        }
        // A warp of 32 covers two whole rows of a `[64, 128]` tile — 256 bytes
        // each, one run apiece — and the second starts a leading dimension
        // along, not a tile width.
        assert_eq!(
            destination(rows, 0, 64, PER_ROW) - destination(rows, 0, 64, 0),
            PITCH
        );
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

    use crate::shared::{Bf16, Element, F16, SharedTile, SharedVec, Swizzle, Swizzle128B};

    /// An [`Element`] the TMA engine can name in a tensor map.
    ///
    /// Separate from `Element` because the data-type enum is a cuda-core value
    /// and `Element` is device-side, and separate from
    /// [`crate::shared::MmaElement`] because a buffer the TMA moves need never
    /// reach an MMA.
    pub trait TensorMapElement: Element {
        /// The `CUtensorMapDataType` the driver reads this element as.
        const DATA_TYPE: CUtensorMapDataType;
    }

    impl TensorMapElement for Bf16 {
        const DATA_TYPE: CUtensorMapDataType =
            cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }

    impl TensorMapElement for F16 {
        const DATA_TYPE: CUtensorMapDataType =
            cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    }

    /// The box a shared destination wants delivered, as the map has to state
    /// it.
    ///
    /// Implemented by the shared-memory types themselves, so the numbers in a
    /// descriptor are the destination's own constants and cannot be restated
    /// wrong at a call site. The two impls answer differently:
    ///
    /// - [`SharedTile`] — the box is one *subtile*, not one tile, and is
    ///   swizzled. [`SharedTile::tma_load`] issues one `cp.async.bulk.tensor`
    ///   per stacked subtile, lifting the leading coordinate by `SUBTILE_COLS`
    ///   each time, so a wider tile does not widen the box.
    /// - [`SharedVec`] — the box is the whole vector, one row tall and
    ///   unswizzled, delivered by a single instruction.
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
    /// (`GlobalLayout::descriptor_shape` asserts exactly that).
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
    /// **The dimension order is the driver's, not a matrix reader's**:
    /// dimension 0 varies fastest, so a row-major `[rows, columns]` matrix is
    /// `extents = [columns, rows]`. It is also the order the TMA coordinates in
    /// [`SharedTile::tma_load_2d`] are given in.
    ///
    /// Does not borrow the buffer: the constructors are `unsafe` and the caller
    /// promises the allocation outlives every launch consuming a map built from
    /// it.
    ///
    /// ```no_run
    /// use kittens::global::GlobalLayout;
    /// use kittens::shared::Bf16;
    ///
    /// # fn layout(base: u64) -> GlobalLayout<Bf16, 2> {
    /// // A row-major [4096 rows, 1024 columns] bf16 matrix.
    /// unsafe { GlobalLayout::<Bf16, 2>::packed(base, [1024, 4096]) }
    /// # }
    /// ```
    pub struct GlobalLayout<E: Element, const RANK: usize> {
        base: u64,
        extents: [usize; RANK],
        /// Byte stride of each dimension, `[0]` being `E::BYTES`. The driver
        /// takes only dimensions `1..`; the full array keeps the length a plain
        /// `RANK` instead of a `RANK - 1` needing `generic_const_exprs`.
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
        /// - `base` is the device address of a live buffer of at least
        ///   `extents.iter().product()` elements of `E`.
        /// - It stays allocated at that address for every launch that consumes
        ///   a map built from this layout.
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
        /// than its columns, or a slice of a larger tensor.
        ///
        /// Panics unless `element_strides[0] == 1`: the TMA engine reads
        /// dimension 0 contiguously.
        ///
        /// ```no_run
        /// use kittens::global::GlobalLayout;
        /// use kittens::shared::Bf16;
        ///
        /// # fn layout(base: u64) -> GlobalLayout<Bf16, 2> {
        /// // A [128, 64] window of a buffer whose rows are 1024 elements apart.
        /// unsafe { GlobalLayout::<Bf16, 2>::strided(base, [64, 128], [1, 1024]) }
        /// # }
        /// ```
        ///
        /// # Safety
        ///
        /// - As [`Self::packed`], for a buffer spanning
        ///   `Σ (extents[i] - 1) * element_strides[i] + 1` elements.
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
        /// Split out from [`GlobalLayout::tensor_map`]: it is the whole of the
        /// shape agreement, and `CUtensorMap` itself is 128 opaque bytes a test
        /// cannot read back.
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
        /// Every field the caller would otherwise state — data type, box shape,
        /// swizzle mode — comes from `T`, so the only way to build a descriptor
        /// that disagrees with the tile it feeds is to pair the layout with a
        /// different tile type than the kernel loads.
        ///
        /// Errors name the field that broke a driver rule: a 16-byte-aligned
        /// base, dimension strides a multiple of 16 bytes, a box dimension 0 a
        /// whole number of 16-byte lines, no box wider than its extent, and no
        /// box over 256.
        ///
        /// ```no_run
        /// use kittens::global::GlobalLayout;
        /// use kittens::shared::{Bf16, SharedTile, Swizzle128B};
        ///
        /// # fn build(
        /// #     stream: &cuda_core::CudaStream,
        /// #     base: u64,
        /// # ) -> Result<(), Box<dyn std::error::Error>> {
        /// let layout = unsafe { GlobalLayout::<Bf16, 2>::packed(base, [1024, 4096]) };
        /// // The box is one [64, 64] subtile of this tile, swizzled as it is.
        /// let map = layout.tensor_map::<SharedTile<Bf16, 64, 128, Swizzle128B>>(stream)?;
        /// let parameter = map.as_ptr();
        /// # let _ = parameter;
        /// # Ok(())
        /// # }
        /// ```
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
            // legal transfer. A swizzled box satisfies this for free — its
            // width *is* an atom — and an unswizzled one has to be checked.
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

    /// The 3-D bf16 panel map [`encode_bf16_panels`] builds.
    pub type PanelMap = TensorMap;

    /// Encode a SWIZZLE_128B tensor map loading `[R, 64]` bf16 subtiles of a
    /// `SharedTile<Bf16, R, C, Swizzle128B>` from one `[rows, C]` panel of a
    /// packed `[planes, rows, C]` staging buffer. The kernel selects the
    /// panel via the third coordinate, the row range via the second, and the
    /// stacked subtile (columns `64*i..64*(i+1)`) via the first — which
    /// [`SharedTile::tma_load`] walks automatically.
    ///
    /// Panics unless `rows` is a whole number of `R`-row boxes: the kernel
    /// steps its row coordinate by `R`, so a ragged tail would arrive silently
    /// zero-filled rather than as a short tile. That is this staging layout's
    /// convention and not a rule of [`GlobalLayout`], which leaves edge boxes to
    /// the TMA engine's out-of-bounds fill.
    ///
    /// ```no_run
    /// # fn build(
    /// #     stream: &cuda_core::CudaStream,
    /// #     base: u64,
    /// # ) -> Result<(), Box<dyn std::error::Error>> {
    /// // Four [256, 128] bf16 panels, delivered as [64, 128] tiles.
    /// let map = unsafe { kittens::global::encode_bf16_panels::<64, 128>(stream, base, 256, 4)? };
    /// # let _ = map;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// - `base` is the device address of a live buffer of at least
    ///   `planes * rows * C` bf16 elements.
    /// - It stays allocated at that address for every launch that consumes the
    ///   returned map.
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
            // The box is [SUBTILE_COLS, R, 1].
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
            // R = C = 64, rows = 64, planes = 1 — the shape every device test
            // uses, with the arrays `encode_bf16_panels` names for it.
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
