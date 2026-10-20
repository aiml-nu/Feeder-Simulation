# Examples

Runnable simulations built on top of `src/`. Two purposes, both live in this folder:

1. **Validation** — small, well-defined problems with a known closed-form, numerical, and/or
   experimental reference solution, used to check that a given `BeamFormulation` /
   `GeometricStiffness` (or `NonlinearBeamFormulation`) combination is actually correct before
   trusting it on the real flexure. See `design/architecture.md` and `derivations/` for the
   underlying models, `derivations/references.md` for the sources of the reference solutions.
2. **The feeder flexure itself** — the actual device this whole tool exists to design
   (`design/flexure_design.md`), set up as a runnable example so others can reproduce results
   without having to reconstruct the geometry/loading from scratch.

## Layout convention

One subfolder per example:

```
examples/<name>/
    README.md       # problem statement, parameters, reference solution + citation, expected result
    run.py           # (once src/ exists) builds the structure, runs it, compares to reference
```

## Current examples (scaffolded, not yet implemented — no `src/` to build against yet)

- `euler_buckling_cantilever/` — simplest possible buckling validation: single beam, cantilever,
  axial compression, compare computed critical load to the classical Euler formula (and its
  axially-compressible correction). Good first test for any new formulation/buckling model.
- `williams_toggle/` — two-beam snap-through benchmark with published experimental data. More
  demanding than the cantilever case: exercises large rotation and instability with a real
  reference to check against, including the unstable/snap-through branch.
- `feeder_flexure/` — the actual wire-feeder flexure (`design/flexure_design.md`), parametrized
  the same way `Dual_Cam_OG` was in the earlier prototype, once ported into the new structure.

Each example should be runnable against whichever formulations/buckling models are implemented
at the time, so the same example can be re-run later to compare, e.g., the linear + additive-
softening approach against the geometrically nonlinear formulation once both exist
(`design/roadmap.md`).
