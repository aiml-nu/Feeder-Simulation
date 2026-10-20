# Euler-Bernoulli Planar Beam Stiffness Matrix

Working notes. Derives the 6x6 stiffness matrix used in `_k_stiff` (`functions.py`), for the
default beam formulation. This is the baseline against which alternative formulations
(Timoshenko, etc.) will be compared — see `design/architecture.md`.

## Setup

Planar 2-node beam element, local axes: x along the beam, y transverse, θ = rotation about z
(out of plane). Each node has 3 DOF: `[u_x, u_y, θ]`. Full local DOF vector:

```
d = [u_x_l, u_y_l, θ_l, u_x_r, u_y_r, θ_r]
```

Cross-section: rectangular, width `w` (into the page / out-of-plane), thickness `t` (in-plane
bending direction). Length `L`, modulus `E`.

```
A = w t                (axial area)
I = w t^3 / 12          (second moment of area about bending axis)
```

Two uncoupled sub-problems in the local frame: axial (bar) and transverse+rotation (beam
bending). They only couple once rotated into the global frame via `R_beam`.

## Axial (bar) stiffness

Linear (1D) shape functions for axial displacement, strain energy `U = 1/2 ∫ EA (du/dx)^2 dx`
gives the standard 2x2 bar stiffness, embedded into the `[u_x_l, u_x_r]` slots:

```
k_axial = (EA/L) [[ 1, -1],
                   [-1,  1]]
```

## Bending (Euler-Bernoulli) stiffness

Cubic Hermite shape functions for transverse displacement `v(x)` interpolate `[v_l, θ_l, v_r,
θ_r]` such that `v` and `dv/dx` are continuous at the nodes (this is the same Hermite basis used
in `_hermite` for plotting deflected shape). Strain energy from bending curvature,
`U = 1/2 ∫ EI (d²v/dx²)^2 dx`, gives the standard 4x4 result:

```
k_bend = (EI/L^3) [[ 12,   6L,  -12,   6L ],
                    [ 6L,  4L²,  -6L,  2L² ],
                    [-12,  -6L,   12,  -6L ],
                    [ 6L,  2L²,  -6L,  4L² ]]
```

ordered `[v_l, θ_l, v_r, θ_r]`.

## Assembly into 6x6

Interleave axial and bending blocks into the `[u_x_l, u_y_l, θ_l, u_x_r, u_y_r, θ_r]` ordering:

```
k_stiff =
[[ AE/L,   0,       0,     -AE/L,   0,       0    ],
 [ 0,      12EI/L³, 6EI/L², 0,     -12EI/L³, 6EI/L²],
 [ 0,      6EI/L²,  4EI/L,  0,     -6EI/L²,  2EI/L ],
 [-AE/L,   0,       0,      AE/L,   0,       0    ],
 [ 0,     -12EI/L³,-6EI/L², 0,      12EI/L³,-6EI/L²],
 [ 0,      6EI/L²,  2EI/L,  0,     -6EI/L²,  4EI/L ]]
```

This matches `_k_stiff(w, t, L, E)` exactly. Verified symmetric, rank 3 (3 rigid-body modes:
translation x, translation y, rotation — correct for an unconstrained 2-node planar beam).

## Local -> global

Force-displacement relation is local: `F_beam = k_stiff @ u_beam`. Global assembly uses
`R_beam` (block-diagonal rotation, local->global) and `H_beam` (rigid-body-to-anchor-point
compatibility, see `rigid_body_kinematics.md`):

```
k_rigid = H.T @ R @ k_stiff @ R.T @ H
```

which is what `Structure.solve()` assembles per beam into the global stiffness matrix.

## Assumptions / limits worth flagging

- Small-strain, small-rotation linear elasticity (geometric nonlinearity handled separately via
  the softening term, not baked into this matrix).
- Plane sections remain plane and perpendicular to neutral axis (no shear deformation) — this is
  the Euler-Bernoulli assumption. For the thin flexures here (`t` on the order of the beam
  length is *not* tiny — need to check slenderness ratio `L/t`), Timoshenko shear correction may
  matter. This is the concrete swap-target formulation for `design/architecture.md`.
- Rectangular, prismatic, homogeneous cross-section. FDM/AFSD printed material may have
  anisotropic modulus (layer direction) — not currently captured; `E` is a single scalar.
- No coupling between axial and bending in the *linear* matrix (that coupling only appears in
  the geometric softening term below).
