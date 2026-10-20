# Geometric (Axial-Load) Softening Term

Working notes. Derives `_k_soft` (`functions.py`) and how it's used to capture buckling /
axial-softening effects — `k_beam = k_stiff - P * k_soft`, applied incrementally in
`Beam.update()` where `P` is the beam's own axial force from the previous solve.

Source cited in the code: Martin, H.C., *"Large Deflection and Stability Analysis by the
Direct Stiffness Method,"* NASA TR 32-931, Aug 1966.

## What this term physically represents

An axially loaded beam's transverse (bending) stiffness changes with the axial load: a beam
under compression is *less* resistant to transverse/rotational deflection (softer — this is
what eventually leads to buckling as `P` approaches the critical load), and a beam under tension
is stiffer. This is the same physical effect captured by "P-Delta" analysis in structural
engineering, "stress stiffening" in FEA, or the "geometric stiffness matrix" `K_g` in matrix
structural analysis — same idea, different derivation routes.

## Derivation route (consistent geometric stiffness matrix)

The rigorous derivation comes from including the second-order (nonlinear) term in strain energy
due to axial force acting through the *rotation* of the deformed beam axis. For an axially loaded
beam-column, adding an extra term to potential energy:

```
U_geo = 1/2 ∫ P (dv/dx)^2 dx
```

on top of the usual bending strain energy, then differentiating with the same cubic Hermite
shape functions used for `k_bend` (see `beam_stiffness_euler_bernoulli.md`) gives the classic
**consistent geometric stiffness matrix** (Przemieniecki, *Theory of Matrix Structural Analysis*,
and equivalently derivable from the exact beam-column stability functions, which is the route
Martin's NASA report takes). For the `[v_l, θ_l, v_r, θ_r]` DOF ordering:

```
K_g = (P / 30L) [[ 36,   3L,  -36,   3L ],
                  [ 3L,  4L²,  -3L,  -L²],
                  [-36,  -3L,   36,  -3L],
                  [ 3L,  -L²,  -3L,  4L²]]
```

Dividing out `P` gives `k_soft`'s nonzero submatrix. Numerically verified against `_k_soft(L)`
in `functions.py`:

```
k_soft[v,θ,v,θ] == (1/30L) [[36, 3L, -36, 3L], [3L, 4L², -3L, -L²], [-36,-3L,36,-3L], [3L,-L²,-3L,4L²]]
```

confirmed to match exactly (see verification note at the end of this file). No axial-axial
softening term (row/col 0 and 3 are zero) — this formulation only captures the *P-affects-bending*
coupling, not axial stiffness change itself.

## Sign convention

`P = F_beam_xyθ[0]`, the beam's local axial force at the left end, with the code's sign
convention (`_stress` computes axial stress as `-F[0]/(wt)`, positive = tension → `F[0]` positive
means compression). So `k_stiff - P*k_soft` **reduces** transverse/rotational stiffness when
`F[0] > 0` (compression), consistent with softening toward buckling. Worth double-checking this
sign against a known Euler buckling load case as a unit test (see `design/roadmap.md`).

## Relation to "P-Delta"

Classic structural-engineering P-Delta is usually implemented differently: iteratively apply
fictitious lateral forces (or moments) proportional to `P × drift` at each node, updated each
iteration/load step, rather than modifying the stiffness matrix directly. It's an approximate,
iterative way of capturing the same second-order effect, and converges to something similar for
small deflections but isn't derived from the same consistent shape-function assumptions. The
NASA-report approach used here is the more rigorous, closed-form alternative — it's applied once
per increment (`P` from the previous converged step, per the "requires incremental application"
note in `functions.py`) rather than iterated to convergence within a step, so it is itself a
linearized/incremental approximation of the true nonlinear response, not exact for large steps.

## Update: reframing as geometrically nonlinear time-stepping, not a bolted-on matrix

After reading Jirásek, La Malfa Ribolla & Horák (2021) (`derivations/references.md`), the better
framing for "P-Delta" here is: it's not really a special stiffness-matrix correction, it's
**incremental time-stepping using a formulation that is geometrically nonlinear on its own** —
apply load in increments, re-solve equilibrium on the *current deformed* geometry each time, and
the second-order (P-Delta-like) softening emerges naturally rather than needing to be added in
as a separate term. That sidesteps the exact worry raised above (does a hand-derived softening
matrix stay compatible with whatever beam formulation it's paired with) — a fully nonlinear
formulation doesn't need a paired softening matrix at all, because large-rotation/large-
displacement effects are already in its internal force and tangent stiffness.

This doesn't invalidate the derivation above — `k_stiff - P*k_soft` is still a legitimate, cheap,
closed-form way to add first-order geometric softening to a small-displacement linear beam, and
worth keeping as the fast option for early-stage design iteration. But for buckling and large
jaw displacement specifically (i.e. the regime this project actually cares about), the plan is
to move to a properly nonlinear formulation (Jirásek et al. 2021, candidate
`NonlinearBeamFormulation` in `design/architecture.md`) driven by an incremental/Newton-
iteration solve, rather than trying to get the additive-matrix approach to agree with it after
the fact. See `design/architecture.md` ("Geometrically nonlinear track") for the interface
sketch, and `examples/euler_buckling_cantilever/` / `examples/williams_toggle/` for validation
targets either approach should be checked against.

## Known limitations / open questions

- **Incremental linearization**: using `P` from the *previous* step's solve, not solved
  self-consistently within the current step. For rapidly changing axial load (e.g., right at
  buckling onset) this could introduce error or instability — worth checking convergence with
  step-size (`eps` in `Dual_Cam_OG.simulate`).
- **No captured axial-axial softening** — a beam right at its buckling load doesn't get *any*
  softer axially in this formulation, only transversely. Should be fine for slender-beam bifurcation
  buckling but worth confirming this is the dominant failure mode of interest, vs. other geometric
  nonlinearities (e.g., large in-plane rotation of the whole flexure, not just per-beam P-Delta).
- **Geometrically nonlinear formulation not yet implemented** — per the update above, this is now
  the preferred long-term approach for buckling/large-displacement behavior rather than "P-Delta"
  as a separate matrix. Flagged in `design/roadmap.md`; once implemented, comparing its results
  against this additive-softening approach on a shared test case would be a good validation and
  a natural conference figure.
- No large-rotation update of the beam's local frame (co-rotational formulation) — beam axis
  orientation `R_beam` is fixed at initialization from `x_init_global`, not updated as the
  structure deforms. Fine for small deflections; questionable once inputs move the jaw by
  amounts comparable to beam length. Worth quantifying.
