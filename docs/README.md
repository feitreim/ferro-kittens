# Design notes and measurements

Source files carry what a *caller* needs: what a thing does, the contract it
owes, and an example of calling it. This directory carries everything else —
the measurements a design decision rests on, the alternatives that were tried
and lost, and the reasoning that would otherwise sit between a reader and the
code.

The split is a rule about audience, not about value. A number that changes what
a caller writes belongs in the source, in one line (`CHUNK = 16; 32 is 4.4x
slower`). The table that number was read off belongs here.

## Library — `docs/library/`

One file per module of `src/`, named for it.

## Kernels — `docs/kernels/`

One file per kernel in `examples/src/`, named for it.

## Related

- `README.md` — what the library is, and the register-count tooling.
- `GAPS.md` — the surface diff against ThunderKittens.
- `CI.md` — the three gate tiers and what each one can see.
- `experiments/README.md` — the GEMM ablation ladder.
