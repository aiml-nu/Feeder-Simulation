"""
Finite-rotation utilities, written for JAX autodiff.

Every 6-DOF entity (beam node or rigid body) carries a *total* rotation vector
psi, with current rotation R = exp(psi). This is singular only at |psi| = pi,
far beyond anything a flexure node sees here.

Both maps below are written as smooth functions of theta^2 with Taylor
branches near zero, so first AND second derivatives (needed for the autodiff
Hessian/tangent stiffness) are finite at psi = 0. A naive implementation
(arccos of the trace, or sin(theta)/theta with theta = |psi|) has an infinite
derivative there.
"""
from __future__ import annotations

import jax.numpy as jnp


def skew(v):
    return jnp.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])


def _safe(x2, small, series, exact):
    """Double-where trick: evaluate `exact` only on a safe argument so its
    gradient never produces NaN in the unused branch."""
    is_small = x2 < small
    x2_safe = jnp.where(is_small, small, x2)
    return jnp.where(is_small, series(x2), exact(x2_safe))


def expm(psi):
    """Rodrigues: rotation vector -> rotation matrix."""
    t2 = jnp.dot(psi, psi)
    A = _safe(t2, 1e-6,
              lambda x: 1.0 - x / 6.0 + x * x / 120.0,
              lambda x: jnp.sin(jnp.sqrt(x)) / jnp.sqrt(x))
    B = _safe(t2, 1e-6,
              lambda x: 0.5 - x / 24.0 + x * x / 720.0,
              lambda x: (1.0 - jnp.cos(jnp.sqrt(x))) / x)
    K = skew(psi)
    return jnp.eye(3) + A * K + B * (K @ K)


def logv(R):
    """Rotation matrix -> rotation vector, valid for |theta| < 90 deg.

    Used only for *deformational* (corotated, local) rotations of a beam
    element, which are small. Uses v = vee(R - R^T)/2 = sin(theta) n and the
    smooth factor asin(s)/s, s = |v|.
    """
    v = 0.5 * jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s2 = jnp.dot(v, v)
    f = _safe(s2, 1e-6,
              lambda x: 1.0 + x / 6.0 + 3.0 * x * x / 40.0,
              lambda x: jnp.arcsin(jnp.sqrt(jnp.minimum(x, 1.0))) / jnp.sqrt(x))
    return f * v


def normalize(v):
    return v / jnp.sqrt(jnp.dot(v, v))
