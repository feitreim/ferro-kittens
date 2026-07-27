//! Global-memory layouts and their TMA tensor maps.
//!
//! The device side of a global operand is just a `*const TmaDescriptor` kernel
//! parameter; what the type system can hold is the *host* side. A
//! [`GlobalLayout`] is everything `cuTensorMapEncodeTiled` needs about the
//! buffer — base address, per-dimension extents, per-dimension byte strides,
//! dimension 0 the contiguous one — and nothing about the transfer. The box
//! comes from the [`crate::shared::SharedTile`] the map is paired with, so
//! descriptor and tile agree by construction rather than by convention: one
//! box per subtile, [`SharedTile::SUBTILE_COLS`] wide and `R` rows tall, in
//! the tile's own swizzle mode and element type. A layout paired with the
//! wrong element does not typecheck.
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

    use crate::shared::{Bf16, Element, SharedTile, Swizzle, Swizzle128B};

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

    /// The box a shared tile wants delivered, as the map has to state it.
    ///
    /// Implemented once, for [`SharedTile`], so the numbers in a descriptor
    /// are the tile's own constants and cannot be restated wrong at a call
    /// site. A tensor map's box is one *subtile*, not one tile:
    /// [`SharedTile::tma_load`] issues one `cp.async.bulk.tensor` per stacked
    /// subtile, lifting the leading coordinate by `SUBTILE_COLS` each time.
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
