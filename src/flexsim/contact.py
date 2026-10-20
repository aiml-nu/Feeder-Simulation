"""
Two-point contact-DOF kinematics and flat-surface active-set (hard) contact.

Mirrors the approach used in the original prototype's `Rigid_Contact`
(`derivations/contact_friction_model.md`): rather than tracking a rigid
body's own reference-point DOF `[u, w, phi]` directly, a body that needs
contact checking is instead parameterized by the physical displacement *at
its two contact points*, since contact activation/deactivation and (later)
friction both need to reference physical point quantities, not an abstracted
center + rotation.

The old code's jaw used `[u_x_top, u_y, u_x_bottom]` (two points differing
in *y*, contact/penetration checked in *x*). Here the geometry is rotated
90 degrees: two contact points differing in *x* (horizontal offsets `dx1`,
`dx2` from the body's own reference point, both at the same initial height
as the reference point), penetration checked in *z* against a flat
horizontal surface. So the raw DOF here are `[w1, u_c, w2]` -- the vertical
displacement at each contact point, plus the shared horizontal displacement
(the two points don't need distinct horizontal DOF for this purpose, same
rationale as the old code's shared `u_y`).

Unlike the old code (which only ever used symmetric points at `+-L/2`), the
two contact points here can sit at arbitrary, independent offsets `dx1` and
`dx2` -- needed to compare the user's two requested configurations (points
symmetric about the body's reference point vs. both on the same side of it).
`two_point_G` below is the corresponding generalization of the old `G`
matrix.

Small-rotation relation between the two representations (same order of
approximation as the old `_H_point`/`G`, i.e. exact for the same rigid-body
kinematics used everywhere else in this package would require `sin`/`cos`,
but this reparametrization is treated as fixed/linear for tractability --
see note in `design/roadmap.md`):

    w1 = w_c + dx1 * phi
    w2 = w_c + dx2 * phi
    u_c = u_c

Solving for the standard DOF given the two point displacements:

    phi = (w2 - w1) / (dx2 - dx1)
    w_c = w1 - dx1 * phi
    u_c = u_c
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def two_point_G(dx1: float, dx2: float) -> np.ndarray:
    """3x3 matrix mapping contact-point DOF q_con = [w1, u_c, w2] to standard
    rigid-body DOF q_std = [u_c, w_c, phi]: q_std = G @ q_con.

    Requires dx1 != dx2 (the two contact points must be at distinct
    horizontal offsets -- this is what makes the two-point reparametrization
    invertible, whether or not the points are symmetric about the body's
    reference point)."""
    if dx1 == dx2:
        raise ValueError("contact points must be at distinct offsets (dx1 != dx2)")
    D = dx2 - dx1
    return np.array([
        [0.0,      1.0, 0.0     ],
        [dx2 / D,  0.0, -dx1 / D],
        [-1.0 / D, 0.0, 1.0 / D ],
    ])


@dataclass
class FlatSurfaceContactPoint:
    """One contact point of a two-point contact node, checked against a flat
    horizontal surface at global height `surface_z`.

    `raw_index` is which of the node's 3 raw DOF (0 or 2, in the `[w1, u_c,
    w2]` convention above) is this point's own vertical displacement --
    `Frame`'s existing per-DOF `fix`/`load` mechanism is reused directly on
    that raw index for activating/releasing the contact constraint (exactly
    as in the old code's active-set `check()`), rather than introducing a
    separate Lagrange-multiplier or penalty formulation.
    """
    node_idx: int
    raw_index: int          # 0 or 2 -- which raw DOF of the node is this point's w
    z0: float                # initial global z of this contact point
    surface_z: float         # global z of the flat surface
    active: bool = False

    def contact_target(self) -> float:
        """The prescribed displacement value once this point is pinned to
        the surface (constant regardless of load path progress, since the
        surface doesn't move)."""
        return self.surface_z - self.z0

    def penetration(self, w_current: float) -> float:
        """Positive when the point has penetrated the surface (i.e. should
        be -- or already is -- in contact)."""
        z_current = self.z0 + w_current
        return self.surface_z - z_current


@dataclass
class PenaltyFlatContact:
    """A one-sided (push-only) linear elastic normal contact law against a
    flat horizontal surface, evaluated at a single anchor point (the same
    anchor-point convention used by beams -- `rigid.anchor_dof`/
    `anchor_jacobian`). Unlike `FlatSurfaceContactPoint`, this attaches
    directly to an ordinary `Node` (no `G`-matrix DOF reparametrization
    needed): it behaves exactly like a beam that only has one end, so
    `Frame._assemble` can fold it in with the same `H(phi) @ G` chain rule
    already used for offset beam attachment.

    Energy: `U = 0.5 k delta**2` for penetration `delta > 0`, else `U = 0`
    (delta = surface_z - z_actual, positive when penetrating). Internal
    force conjugate to the anchor's own `w`: `F_w = dU/dw = -k*delta` when
    active, else 0; tangent `dF_w/dw = k` when active, else 0 -- a standard
    penalty/compliant contact law, continuously valued (no activate/release
    decision, unlike `FlatSurfaceContactPoint`'s active-set check), at the
    cost of some real, nonzero penetration and an arbitrary stiffness `k` to
    choose. `k` here is a lumped point stiffness; a physically motivated
    value (e.g. from Hertzian line-contact theory, given a material modulus
    and edge radius) is a natural refinement once real geometry/material
    parameters are on hand -- see `design/roadmap.md`.
    """
    z0: float          # initial global z of this anchor point
    surface_z: float   # global z of the flat surface
    k: float           # contact penalty stiffness

    def compute(self, dof_anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (f_anchor (3,), K_anchor (3x3), penetration) at this
        anchor point's current displacement state `dof_anchor = [u, w, phi]`."""
        w = dof_anchor[1]
        z_actual = self.z0 + w
        delta = self.surface_z - z_actual
        if delta > 0:
            f_w = -self.k * delta
            k_w = self.k
        else:
            f_w = 0.0
            k_w = 0.0
        f_anchor = np.array([0.0, f_w, 0.0])
        K_anchor = np.diag([0.0, k_w, 0.0])
        return f_anchor, K_anchor, delta
