# Example: Parallel-Beam Flexure with Contact and a Load Path

Status: **implemented and validated** (`parallel_beam_flexure_contact.ipynb`) — third example,
building on `parallel_beam_flexure`.

## Problem

Same two-rigid-body, two-beam geometry as `parallel_beam_flexure`, extended with:

1. **Contact** against a flat rigid surface `0.1` below the right body's reference point, using
   two contact points on the right body (so contact can constrain rotation, not just position) —
   directly mirroring the original prototype's `Rigid_Contact` two-point active-set contact
   (`derivations/contact_friction_model.md`) rather than a compliant/penalty law.
2. **A load path**: the *left* body's `w` is driven down 0.2, then back up to 0 (not a single
   monotonic ramp), with `F_x = M = 0` otherwise free there. The right body is free except where
   it contacts the surface.

Two contact-point configurations are compared: **configuration 1** (points symmetric about the
body's reference point, `dx = -0.15, +0.15`) and **configuration 2** (both points on the same
side, `dx = 0.05, 0.25`).

## What's new in `src/flexsim`

- **`contact.py`**: `two_point_G` reparametrizes a rigid body's DOF as `[w_point1, u_center,
  w_point2]` (the two contact points' own physical vertical displacement) instead of `[u, w,
  phi]`, generalizing the old code's symmetric-only `G` matrix to arbitrary, independent point
  offsets. `FlatSurfaceContactPoint` tracks each point's penetration/active state against a flat
  surface (active-set/hard contact). `PenaltyFlatContact` implements a second, compliant contact
  law instead: a one-sided linear spring (`F = -k * penetration` when penetrating, `0`
  otherwise), attached directly to an *ordinary* node via the existing anchor-offset machinery —
  no `G`-matrix reparametrization needed, since a point force doesn't require redefining the
  body's DOF basis. Tangent validated against finite differences.
- **`frame.py`**: `Node` gained a `G` matrix (identity by default); `_assemble` composes it with
  the existing anchor-offset transform (`J = H(phi) @ G`) so beams can attach normally to a
  contact-reparametrized body, with a second-derivative correction term generalized accordingly —
  validated against finite differences before use. `Frame.add_contact_node`/`add_flat_contact`
  build the active-set kind of node; `Frame.add_penalty_contact` attaches a penalty contact point
  to any ordinary node, folded into `_assemble` the same way as a one-ended beam.
  `Frame.solve_path_with_contact` runs a multi-leg load path (warm-started across leg boundaries,
  including through the down-then-up reversal) with an active-set contact check after every
  converged step, re-solving the step if any point's status changed.
- Both reduce exactly to prior behavior when unused (`G` = identity, no contacts) — the first two
  examples were re-run unchanged as a regression check after each change.

## Active-set vs. penalty contact

Added after a design discussion: is active-set (hard, exact, no tunable stiffness, matches the
original prototype) the right choice, or should contact be compliant/penalty (the more common
default in general FEA, and needed to represent real local contact stiffness — e.g. a 3D-printed
PLA jaw is much softer than aluminum and would show measurable penetration and an uneven
clamping-load split between contact points)? Both are now implemented, and the notebook
cross-checks them directly: with a stiff-enough penalty spring (`k = 1e4` here), the penalty
result matches the active-set solution to ~5 significant figures — good evidence the two
formulations describe the same underlying physics. A softer-`k` sweep then shows the PLA-like
regime: real, non-negligible penetration, and a load split between the two contact points that
becomes more uneven as the surface gets stiffer (relative to the beams), converging toward the
same single-point-dominant behavior already seen with active-set. Setting `k` to a physically
meaningful value (rather than an arbitrary one) is flagged as future work — Hertzian line-contact
theory (line-on-flat contact stiffness from material modulus and edge radius) is a natural
starting point once real geometry/material parameters are available.

## A well-posedness issue found (and how it's handled)

Prescribing *only* `w` at the left body (leaving `F_x = M = 0` free there too, mirroring how the
right body's mixed BC worked in `parallel_beam_flexure`) leaves the whole mechanism with no
absolute restraint on net horizontal translation or rotation until contact engages — confirmed by
the assembled stiffness being singular on the free DOF before any grounding is added. The left
body is treated like a vertical slide instead: `u` and `phi` are also fixed (at 0), with only `w`
driven — the natural reading of "enforce vertical displacement" for an input actuator, and it
removes the singularity while keeping `w` as the only *driven* quantity.

## What the notebook does

1. Builds both contact configurations from a shared `build_frame` function.
2. Renders initial geometry, including the two contact points and the flat surface.
3. Solves the down-then-up load path for both configurations.
4. Plots each contact point's penetration against the driving displacement — showing that
   configuration 2 reliably engages only its farther point, and configuration 1 (nominally
   symmetric) still favors one point over the other within this stroke, due to the same
   parasitic-rotation effect already characterized in `parallel_beam_flexure`.
5. Plots deformed shapes at several points along the path.
6. Confirms the mechanism returns to (very near) its undeformed state once the load path
   completes and both contact points release.

## Result

Both configurations solve over the full 200-step down-then-up path without incident. Neither
configuration achieves simultaneous two-point contact within a 0.2 stroke against a 0.1
clearance — a genuine finding, not a bug: the same large-deflection parasitic rotation that
produces the small "no rotation isn't quite true" effect in `parallel_beam_flexure` means even
nominally symmetric contact points don't touch down perfectly together. A deeper stroke would
eventually bring configuration 1's second point into contact too, since its penetration is
monotonically closing the gap.
