# Jaw Contact and Friction Model

Working notes on `Rigid_Contact` (`elements.py`): the `xyx` DOF parameterization used for the
jaw, the `G` matrix, and the contact/stick-slip logic in `check()`. This corresponds to the
updated flexure design (`design/flexure_design.md`) where the jaw is in distributed contact with
the wire along its whole length, not just a single point-force as in the older `Dual_Cam_OG`
simulate loop.

## Why a different DOF set for the jaw

Every other rigid body uses `[u_x, u_y, θ]` about its centroid. The jaw instead uses
`[u_x_top, u_y, u_x_bottom]` — the physical x-displacement at two points spaced `L` apart
(top and bottom of the contact line, `±L/2` from centroid), plus the shared y-displacement.
This is a deliberate choice: normal-direction contact/penetration checking and stick-slip
friction both need to reference the physical displacement/force *at the contact points*, not an
abstracted centroid + rotation.

## `G`: xyx <-> xyθ kinematic map

Small-rotation rigid body kinematics at offset `dy` from centroid (same relation as
`_H_point`, see `rigid_body_kinematics.md`):

```
u_x_top    = u_x_c - (L/2) θ
u_x_bottom = u_x_c + (L/2) θ
```

Solving for the centroid quantities:

```
u_x_c = (u_x_top + u_x_bottom) / 2
θ     = (u_x_bottom - u_x_top) / L
```

which is exactly `_G(L)`:

```
G = [[ 1/2,  0,  1/2 ],
     [  0,   1,   0  ],
     [-1/L,  0,  1/L ]]
```

so `u_xyθ = G @ u_xyx`. Confirmed by direct substitution.

## ⚠ Force transform looks inconsistent — flagging for follow-up

`Rigid_Contact.update()` does `F_out_global_xyθ = G @ F_out_global_xyx`, reusing the same `G` as
for displacements. That's **not** generally correct. Since `u_xyx` are physical point
displacements/rotations, the physical resultant force and moment at the centroid from two point
forces `F_x_top`, `F_x_bottom` (at `dy = ±L/2`) and `F_y` should be plain statics:

```
F_x_c = F_x_top + F_x_bottom
F_y_c = F_y
M_c   = (L/2)(F_x_bottom - F_x_top)
```

Checked numerically (`L=2.5`, `F_top=3, F_y=-1.2, F_bottom=7`): true resultant is
`[10.0, -1.2, 5.0]`. `G @ F_xyx` gives `[5.0, -1.2, 1.6]` — wrong. The energy/virtual-work
consistent transform is `G⁻ᵀ`, not `G`: `G⁻ᵀ @ F_xyx` gives `[10.0, -1.2, 5.0]`, matching statics
exactly. (`G` is not orthogonal, so `G⁻ᵀ ≠ G` — confirmed numerically.)

This only affects `F_out_global_xyθ` on `Rigid_Contact` (used for e.g. `rigid.plot()`'s rotation
and any downstream reporting/plotting of jaw resultant force/moment) — it does **not** affect
`Structure.solve()` itself, which correctly uses the energy-consistent `k_G = G.T @ k @ G` for
the stiffness change of basis, nor does it affect the `check()` contact/friction logic, which
operates directly on `F_out_global_xyx` (the physical point forces) rather than the `xyθ`
version. Recommend fixing to `G.T @ np.linalg.inv(G).T @ ...`, i.e. precomputing `G_inv_T` once,
before this is relied on for reported results. Added to `design/roadmap.md` as a near-term
correctness fix.

## Contact activation (penetration check)

Two independent contact points (top, bottom), each starts unconstrained (`contact_top`/
`contact_bottom = False`). Per step, `check()`:

- If not yet in contact and displacement has penetrated the wire surface
  (`u_x < u_contact`, i.e. more negative than the surface offset — wire pushes back), activate
  contact: constrain that DOF (`constraints[i] = True`) and set the prescribed displacement to
  exactly the contact surface (`u_in_global_xyx[i] = u_contact`). This is a hard (non-penetrating)
  contact model — no compliance/elastic contact stiffness, the point snaps to the surface once
  penetration is detected.
- If already in contact, check the reaction force sign: a contact constraint should only ever
  produce a *pushing* (non-negative, by this code's sign convention) reaction. If it goes
  negative — physically the surface would have to be pulling the jaw back, i.e. sticking to a
  surface it should be able to separate from — release the constraint.

This is a fairly standard active-set contact method (constraint activation/deactivation each
step based on penetration/force sign), reasonable for quasi-static time-stepping but worth
watching for chattering (repeated activate/deactivate in consecutive steps) as a numerical
concern — see `design/roadmap.md`.

## Stick-slip friction (tangential, along the wire)

Once in contact at either point, tangential ("along the wire") friction uses a Coulomb-like
model:

```
normal   = F_x_top + F_x_bottom          (total normal/perpendicular reaction)
tangent  = F_y                            (force along the wire)
friction = max(mu * normal, 0)            (static friction capacity, clipped ≥ 0)
```

- **Sticking**: if `|tangent| < min(friction, F_AFSD)`, the jaw is treated as sticking to the
  wire — constrain the tangential DOF (`constraints[1] = True`) and hold the current
  displacement (`u_in_global_xyx[1] = u_out_global_xyx[1]`, i.e. "freeze in place," not a fixed
  value — a standard trick for holding a DOF at its just-computed value once it transitions to
  constrained).
- **Slipping**: otherwise, release the displacement constraint and instead apply a *prescribed
  force* equal to the (signed) lesser of the AFSD feed-force requirement and the friction
  capacity: `F_in_global_xyx[1] = sign(tangent) * min(F_AFSD, friction)`.
- **Not in contact**: friction/tangential DOF fully released, zeroed.

Note the `F_AFSD` cap: this isn't pure Coulomb friction — it caps the tangential force delivered
to the wire at the force actually required for the AFSD (additive friction stir deposition)
feed process, which only makes physical sense if `F_AFSD` represents a *process* force limit
rather than a friction limit. There's a `# TODO`-style comment in the code
(`"Get rid of F_AFSD here, just do friction and record y displacements when above F_AFSD
externally"`) suggesting this coupling was already flagged as something to clean up — worth
revisiting when this model is ported into the new element/formulation structure, since mixing a
process-force constraint into what should be a purely kinematic/frictional contact law makes the
element harder to reuse for other flexure/contact problems.
