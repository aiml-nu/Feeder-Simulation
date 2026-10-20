# Example: Williams Toggle

Status: scaffolded, not yet implemented (no `src/` package to build against yet).

## Problem

Two symmetric, rigidly-connected straight beams whose axes each deviate from horizontal by a
small angle `ψ`, pinned/clamped at the outer ends, joined at a common center point loaded by a
vertical force `P` (or, equivalently, driven by a prescribed vertical displacement at the center
point — see below). See Jirásek et al. (2021), Section 4.2, Figure 4
(`derivations/references.md`).

By symmetry, only one beam needs modeling: clamped at the outer end, with the inner end
constrained to zero rotation and zero horizontal displacement but free to slide vertically
(displacement `w_G/2` where `w_G` is the full toggle's center-point displacement).

## Reference data

Original source: Williams, F. W. (1964), *QJ Mech Appl Math* 17(4):451-469
(`derivations/references.md`) — experimental load-displacement data for two physical toggles,
plus Williams' own approximate analytical solution.

Geometry/material (from Williams' tests, as reported in Jirásek et al. 2021):

- `L = 12.94 in` (beam length, each half)
- `EA = 1.885e6 lb`
- `EI = 9.27e3 lb·in²`
- **Toggle 1**: `ψ = 0.0247 rad` — monotonic load-displacement curve (no snap-through).
- **Toggle 2**: `ψ = 0.02985 rad` — snap-through: load-displacement curve has a local max
  followed by a local min, with an unstable branch between them under load control (stable under
  displacement control).

## What to check

Displacement control is the natural choice here (per the paper: "if the load control is replaced
by direct displacement control, which is perfectly legitimate here, the model has no global
unknowns" — i.e. sweep the center-point vertical displacement `w_G` and read off the resulting
force, which also sidesteps the load-control instability on the descending branch). This maps
directly onto how this project already drives simulations (prescribed input displacements,
`design/flexure_design.md`), so this example is a good structural match for the eventual
time-stepping code, not just a formulation check.

1. Sweep `w_G` from 0 up through the full range Williams tested (~0.7 in based on the paper's
   plots), compute the resulting force `P` at each step.
2. Compare against:
   - Williams' analytical curve
   - Jirásek et al.'s numerical curve (their Figure 5A)
   - The raw experimental data points
   for both toggle geometries (`ψ = 0.0247` and `ψ = 0.02985`).
3. **This is the sharpest available test of whichever nonlinear formulation is implemented** —
   it requires correctly capturing large rotation, the coupling between axial and bending
   behavior, and (for toggle 2) an actual instability/snap-through, all in a minimal 1-DOF-per-
   half-symmetry model. If a formulation gets this right, it's a strong signal it'll behave
   correctly on the real flexure.
4. Integration-refinement check (if using the Jirásek finite-difference element): the paper notes
   10 segments already gives error comparable to experimental scatter, 20 is close to fully
   converged — worth reproducing that convergence trend here too, not just the final curve.

## Notes

- This example does *not* need the jaw contact model (`derivations/contact_friction_model.md`)
  — it's a pure two-beam mechanism, no contact/friction involved. Good for isolating formulation
  correctness from contact-logic correctness.
- Local coordinate convention (per the paper): with the half-beam's local end-2 displacement
  components `u_2 = -w_G sin(ψ)`, `w_2 = w_G cos(ψ)` (for the symmetric half), the applied force
  is recovered as `P = -X_21 sin(ψ) + Z_21 cos(ψ)` from the computed end forces — worth
  cross-checking sign conventions against this project's own local/global force conventions
  (`derivations/rigid_body_kinematics.md`) when implementing.
