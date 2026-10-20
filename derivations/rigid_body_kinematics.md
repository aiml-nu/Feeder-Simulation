# Rigid Body Kinematics, Assembly, and the Mixed-BC Solve

Working notes covering `_H_point`/`_H_beam`, `_R_point`/`_R_beam`, global stiffness assembly,
and the known/unknown (mixed force-displacement) boundary condition solve in
`Structure.solve()`. This machinery is formulation-agnostic — it only needs a local beam
stiffness matrix `k_beam`, so it should carry over unchanged when swapping beam formulations
(see `design/architecture.md`).

## Rigid body -> anchor point compatibility (`H`)

A rigid body has reference-point DOF `[u_x, u_y, θ]` (small rotation `θ`). A beam anchors to the
rigid body at some fixed local offset `d = [dx, dy]` from that reference point. Small-rotation
rigid-body kinematics gives the displacement of the anchor point:

```
u_anchor_x = u_x - dy * θ
u_anchor_y = u_y + dx * θ
θ_anchor   = θ
```

(this is `u_rigid + θ ẑ × d` in 2D). In matrix form this is exactly `_H_point(d)`:

```
H = [[1, 0, -dy],
     [0, 1,  dx],
     [0, 0,  1 ]]
```

`_H_beam` is block-diagonal, stacking `H_point(d_l)` and `H_point(d_r)` for the beam's two
anchors, mapping the two rigid bodies' 6 reference DOF to the beam's 6 anchor DOF:
`x_anchor_global = H @ x_rigid_global`.

## Local -> global rotation (`R`)

Straightforward: `_R_point(θ)` is the standard 2D rotation matrix (embedded in a 3x3 so it acts
on `[x, y, θ]` without touching `θ`, since the local rotation DOF doesn't rotate). `_R_beam` is
block-diagonal for the two beam ends, using the beam's fixed initial orientation angle
(`self.angle = atan2(diff_y, diff_x)`) — see the "no co-rotation" caveat in
`geometric_softening.md`.

## Assembling one beam into the global stiffness matrix

Beam-local stiffness `k_beam` (from whichever formulation) relates local forces/displacements:
`F_beam = k_beam @ u_beam`. To place this into the global system in terms of each rigid body's
reference-point DOF, need two transformations: local frame -> global frame (`R`), and beam
anchor point -> rigid body reference point (`H`). Virtual work invariance (`F_rigid^T u_rigid` must
equal `F_beam^T u_beam` for any compatible virtual displacement) requires the *contragredient*
transform on the force side, giving:

```
k_rigid = H.T @ R @ k_beam @ R.T @ H
```

This 6x6 (for the beam's two connected rigid bodies) is then scattered into the appropriate
3x3 blocks of the global `k` matrix, accumulating contributions from every beam touching each
rigid body — standard direct stiffness assembly.

## Mixed force/displacement boundary conditions

Every DOF in the problem is *either* displacement-prescribed (`u` known, reaction `F` unknown)
*or* force-prescribed (`F` known — usually 0 for an unloaded free DOF, or an applied load; `u`
unknown). Which one applies is set per-DOF by `rigid.constraints` (`True` = displacement
prescribed).

Let `S_u = diag(constraints)` (1 at displacement-prescribed DOFs) and `S_F = I - S_u`
(1 at force-prescribed DOFs); these are complementary orthogonal projectors
(`S_u S_F = 0`, `S_u^2 = S_u`, `S_F^2 = S_F`). Full global state:

```
u = S_u u_in + S_F u_unknown        (u_in = prescribed displacement)
F = S_F F_in + S_u F_unknown        (F_in = prescribed/applied force)
```

Pack the two unknown quantities into a single vector `z = S_u F_unknown + S_F u_unknown` — valid
because the projectors are orthogonal, so `z` has no ambiguity between the two unknown types at
any given index. Substituting into `F = K u` and using `S_u^2=S_u`, `S_F^2=S_F`, `S_u S_F=0`:

```
S_F F_in + S_u z = K (S_u u_in + S_F z)
(S_u - K S_F) z = K S_u u_in - S_F F_in
```

which, after multiplying through by -1, is exactly the code's

```
LHS_mat = K @ S_F - S_u
RHS_vec = (S_F - K @ S_u) @ (S_F F_in + S_u u_in)
z = solve(LHS_mat, RHS_vec)
```

(worth re-deriving by hand once to confirm the sign — did so here and it checks out exactly,
including the `RHS_mat @ (...)` expansion using the same projector-orthogonality identities.)
Recovering physical quantities is then just re-selecting by projector:

```
F_out = S_F F_in + S_u z     (known forces + solved reactions)
u_out = S_u u_in + S_F z     (known displacements + solved free displacements)
```

This is a clean, reusable trick for any per-DOF mixed Dirichlet/Neumann BC problem — no
resizing/reindexing of the system needed, and constraints can change dynamically at each
contact-check step (the `Rigid_Contact.check()` machinery toggles entries of `constraints`
between solves), which is exactly why it's used here instead of the more typical
partition-and-eliminate approach.

## Coupling to `Rigid_Contact` DOF (`G` matrix)

`Rigid_Contact` bodies use a different DOF set (`[u_x_top, u_y, u_x_bottom]` instead of
`[u_x, u_y, θ]`) since displacement/force *at the two contact points* is more directly useful
than at a centroid. `Structure.solve()` swaps this into the standard `[x,y,θ]` basis via a
per-rigid-body block-diagonal `G` matrix before assembly (`k_G = G.T @ k @ G`), then swaps back
after solving. Derivation of `G` itself belongs with the contact model —
see `contact_friction_model.md`.
