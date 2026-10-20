# Example: Pure Bending of a Cantilever

Status: **implemented and validated** (`pure_bending_cantilever.ipynb`) — the first example built
against real `src/flexsim` code, per `design/roadmap.md`.

## Problem

Cantilever of length `L`, bending stiffness `EI`, fixed at one end, loaded at the free end by a
pure moment `M` (no applied force). Bends into a circular arc of radius `R = EI/M` — exact for
arbitrarily large rotation, not just small deflections, which makes this a clean large-rotation
validation case with a genuinely closed-form answer (Jirásek et al. 2021, Section 4.1; see
`derivations/references.md`).

## What the notebook does

1. Defines the geometry (single `JirasekBeam`, cantilever support via `Frame`).
2. Renders the undeformed geometry.
3. Applies the end moment incrementally (force-controlled, via `Frame.solve`) up to a 90-degree
   tip rotation.
4. Plots the deformed shape at several load steps, straight from the element's own
   finite-difference integration grid (`Beam.deformed_shape_global`) — no separate
   shape-function interpolation.
5. Compares the final tip position/rotation against the exact circular-arc formula.
6. Sweeps the number of integration segments (`n_seg`) to show the expected convergence trend
   (Jirásek et al. 2021, Table 1).

## Result

Matches the analytical solution to ~3e-5 in tip displacement and ~6e-13 in tip rotation at 90
degrees of bending. See `design/roadmap.md` ("Status: Jirásek beam element...") for implementation
notes and known follow-up work (full closed-loop bending not yet run, frame-level load-stepping
still retry-based rather than truly adaptive).
