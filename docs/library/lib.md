# `lib.rs` — crate root

## Libdevice math beside tcgen05

Libdevice math is legal beside tcgen05 in the same pure-PTX artifact. That was
established at cuda-oxide `b099f64`, and the citation stays at that revision: no
gate re-establishes it at the current pin, because **nothing in this tree makes
a libdevice call**.

- `reg.rs`'s `Abs` clears the sign bit rather than calling `fabsf`.
- `fmax`/`fmin` are comparison-selects.
- `Sqrt` and `rsqrt` lower to native `sqrt.rn.f32`.
- `exp2`/`log2` are FMA polynomials or the `ex2` SFU.

So a green `build` says nothing about the claim either way, and moving the hash
forward would assert it at a revision nobody checked. The software
approximations stay where they are because their lowering is a measured kernel
optimization, not a workaround for the artifact path.

## Why `lane()` wraps rather than re-exports

`lane()` wraps `cuda_device::warp::lane_id()` instead of re-exporting it. Two
reasons, both about the reader:

- Without it, a kernel that calls nothing but `kittens` still has to reach past
  the library for its first three lines.
- The `lane` argument's meaning would be stated only in the six signatures that
  take it, and nowhere a reader would look first.

The wrapper is `#[inline(always)]` over an `#[inline(never)]` lowering target,
so the two spellings are the same instruction.

## The `lane probe`

`device-tests`' `lane probe` prices hoisting one `%laneid` read into the entry
block and threading it through, against reading it per op:

| columns | registers vs. per-op read |
| ------- | ------------------------- |
| 128     | 0                         |
| 32      | +16                       |

Hoisting is therefore the convention, and it is what the source line in `lane()`
tells callers to do.

## `32` is the layout's rows-per-warp

`32 * warp_id()` is the band origin every kernel in the repo open-codes. The
`32` is `BaseLdtm`'s rows-per-warp; the library does not yet give it a name, so
the literal appears at each call site.

## Broken intra-doc links off the `host` feature

`global`'s types are all behind `feature = "host"`, and docs across the crate
link to them. Off that feature they are not compiled, so the links have nothing
to resolve to — hence the crate-level
`allow(rustdoc::broken_intra_doc_links)` gated on `not(feature = "host")`. The
host build is where those links are actually checked.
