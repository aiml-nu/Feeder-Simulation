# Example: Parallel-Beam Flexure (Two Rigid Bodies, Two Beams, Mixed BCs)

Status: **implemented and validated** (`parallel_beam_flexure.ipynb`) — second example built
against `src/flexsim`, following `pure_bending_cantilever`.

## Problem

Two rigid bodies ("left", "right") connected by two horizontal beams ("upper", "lower")
attached above/below each body's reference point — a double-parallel-leaf-spring / parallel-
motion guide, and a direct stepping stone toward the real feed mechanism's ground + two-inputs
+ jaw topology (`design/flexure_design.md`). The two beams can have independently specified
lengths, letting the model explore the effect of beam-length tolerance mismatch between the two
beams — an explicit future use case.

Left body: fully fixed. Right body: **mixed** boundary condition — vertical displacement `w`
(`u_y`) prescribed (moved downward), horizontal force `F_x` and moment `M` prescribed to zero
(free).

## What's new in `src/flexsim`

Beams previously only connected directly at a node's own reference point. `rigid.py` adds
large-rotation-exact kinematics for a beam attaching at a fixed local offset `(dx, dy)` from a
rigid body's reference point (`anchor_dof`, `anchor_jacobian`, `anchor_jacobian_prime`), and
`Frame.add_element`/`Frame._assemble` (`frame.py`) were generalized to carry that offset through
the chain rule into the internal force and tangent stiffness. Reduces exactly to the old
point-to-point behavior at offset `(0, 0)` — the `pure_bending_cantilever` example was re-run
unchanged as a regression check after this change (identical result).

## What the notebook does

1. Defines geometry: two rigid bodies, two beams attached at `+-h/2` vertical offset each,
   independently specifiable lengths.
2. Renders the undeformed geometry.
3. Applies the mixed BC (`Frame.fix` + `Frame.load` on different DOF of the same node) and
   solves incrementally.
4. Plots deformed shapes of both beams at several load steps.
5. Validates the rigid-body-offset assembly against classical small-deflection beam theory: at
   tiny displacement, reaction force matches `k = 2*12EI/L^3` (two guided-end beams in parallel)
   to ~0.2%.
6. Sweeps beam-length mismatch and reports the resulting parasitic horizontal drift and rotation
   of the moving body — noting that a nonzero parasitic rotation exists even at zero mismatch
   (known large-deflection behavior of a simple two-beam parallel guide, not a numerical
   artifact; see notebook for the reasoning).

## Result

Small-deflection reaction force matches classical parallel-guide stiffness to ~0.2% (consistent
with the finite-difference segment-count error already characterized in the pure-bending
example). Length-mismatch sweep produces a smooth, monotonic trend in parasitic drift/rotation
as expected.
