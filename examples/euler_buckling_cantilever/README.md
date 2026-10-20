# Example: Euler Buckling of a Cantilever

Status: scaffolded, not yet implemented (no `src/` package to build against yet).

## Problem

Single straight beam, length `L`, fixed at one end (all 3 DOF constrained), loaded at the free
end by an axial compressive force `P` directed along the beam's initial axis. Find the critical
load at which the straight, uniformly-compressed configuration becomes unstable.

## Reference solution

Classical Euler formula, assuming axial incompressibility (buckling length `L_b = 2L` for a
cantilever):

```
P_E = EI π² / (4 L²)
```

Jirásek et al. (2021), Appendix B (see `derivations/references.md`) derive the correction for an
axially *compressible* column (finite `EA`), via the second variation of potential energy:

```
P_cr = (EA/2) * (1 - sqrt(1 - 4 EI π² / (EA L²)))   ≈  P_E * (1 + P_E/EA)   for P_E << EA
```

The paper's numerical studies use dimensionless `L = 1`, `EI = 1`, with `EA L² / EI = 100` and
`10,000` (i.e. `EA = 100` or `10,000`) — both comfortably in the `P_cr << EA` slender-column
regime relevant here. Good starting parameter choices for a first test.

## What to check

1. **Linear formulation + additive softening** (`k_stiff - P*k_soft`,
   `derivations/geometric_softening.md`): incrementally increase `P`, watch for the global
   stiffness matrix (restricted to free DOF) becoming singular / smallest eigenvalue crossing
   zero near `P_E` (or `P_cr` if axial compressibility is included in the check). Since this
   project's beams are cantilevered off a rigid body with the other end free-ish, may need a
   minimal 2-rigid-body + 1-beam `Structure` to set this up, or a slight variation matching this
   project's actual BC pattern.
2. **Nonlinear formulation** (once implemented, `design/architecture.md`): track the smallest
   eigenvalue of the tangent stiffness matrix as load increases; onset of instability is where
   it crosses zero (Jirásek et al. 2021, Section 4.3 describes exactly this diagnostic).
3. Compare both against `P_E` and `P_cr` above, and against each other — they should agree
   closely in this slender regime, and any discrepancy is worth understanding rather than
   dismissing.

## Parameters to use (proposed, matching the paper's regime)

- `L = 1`, `EI = 1`, `EA = 100` (dimensionless) as a first pass.
- Real flexure-relevant dimensions (`design/flexure_design.md`) as a second pass, once locked in
  — the actual `L/t` slenderness of the printed beams should be checked against this same
  formula as a sanity bound on expected buckling loads.
