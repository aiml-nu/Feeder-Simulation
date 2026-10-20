"""
Rigid-body offset kinematics: lets a beam attach to a rigid body at a fixed
local offset (dx, dy) from the body's own reference-point DOF, rather than
only at the reference point itself.

This is the "H_point" idea from `derivations/rigid_body_kinematics.md`, but
exact for large rotation (matching the Jirasek beam's own large-rotation
formulation) rather than the old code's linearized small-rotation H matrix.

A rigid body has 3 global DOF at its reference point: [u, w, phi] (same
convention as everywhere else in this package: u = horizontal displacement,
w = vertical displacement, phi = rotation). A point rigidly attached to that
body at initial local offset (dx, dy) -- measured in the body's own initial,
unrotated orientation -- moves to:

    anchor_pos(t) = body_pos0 + [u, w] + R(phi) @ [dx, dy]
    anchor_pos(0) = body_pos0 + [dx, dy]                    (phi = 0 there)

so the anchor's *displacement* (what beam elements consume) is

    anchor_u   = u + dx*(cos(phi) - 1) - dy*sin(phi)
    anchor_w   = w + dx*sin(phi)       + dy*(cos(phi) - 1)
    anchor_phi = phi

(the anchor's rotation equals the body's rotation -- a cross-section
rigidly attached to the body rotates by exactly the body's rotation).

This reduces to the identity map when (dx, dy) = (0, 0), so beams attached
directly at a body's own reference point are unaffected -- existing examples
(single point-to-point beams) keep working unchanged.
"""

from __future__ import annotations

import numpy as np


def anchor_dof(body_dof: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Map a rigid body's own [u, w, phi] to the displacement DOF of a point
    rigidly attached at initial local offset (dx, dy)."""
    u, w, phi = body_dof
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        u + dx * (c - 1.0) - dy * s,
        w + dx * s + dy * (c - 1.0),
        phi,
    ])


def anchor_jacobian(phi: float, dx: float, dy: float) -> np.ndarray:
    """d(anchor_dof)/d(body_dof), 3x3. Reduces to identity at (dx, dy) = (0, 0)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [1.0, 0.0, -dx * s - dy * c],
        [0.0, 1.0,  dx * c - dy * s],
        [0.0, 0.0,  1.0],
    ])


def anchor_jacobian_prime(phi: float, dx: float, dy: float) -> np.ndarray:
    """d(anchor_jacobian)/d(phi), 3x3. Needed for the extra Newton tangent-
    stiffness term from differentiating H itself (same pattern as the
    Jirasek beam's own T/T_prime corotational transform). Zero at
    (dx, dy) = (0, 0)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [0.0, 0.0, -dx * c + dy * s],
        [0.0, 0.0, -dx * s - dy * c],
        [0.0, 0.0,  0.0],
    ])


def anchor_initial_position(x0: float, z0: float, dx: float, dy: float) -> tuple[float, float]:
    """Initial global position of a point rigidly attached to a body whose
    own reference point starts at (x0, z0)."""
    return x0 + dx, z0 + dy
