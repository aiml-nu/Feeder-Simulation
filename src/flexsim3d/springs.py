"""
Elastic bushings: 6-DOF linear springs between a point on one entity (A) and
the reference point of another (B). Used to give the printed mounting blocks
(where leaf stacks root into the jaw adapter, input bodies and housing) a
finite stiffness instead of treating them as perfectly rigid.

Energy (isotropic, measured in A's rotated frame, so it is objective under
rigid motion of the pair):
    d   = R_A^T (x_B - x_A_point)                   relative translation
    th  = log(R_A^T R_B)                            relative rotation
    U   = 1/2 kt |d|^2 + 1/2 kr |th|^2
Residual and tangent are the autodiff gradient and Hessian.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .rotation import expm, logv


def _energy(q, d):
    qA, qB = q[:6], q[6:]
    RA = expm(qA[3:])
    RB = expm(qB[3:])
    xA = d["XA"] + qA[:3] + RA @ d["oA"]
    xB = d["XB"] + qB[:3]
    dl = RA.T @ (xB - xA)
    th = logv(RA.T @ RB)
    return 0.5 * d["kt"] * jnp.dot(dl, dl) + 0.5 * d["kr"] * jnp.dot(th, th)


_res_v = jax.jit(jax.vmap(jax.grad(_energy), in_axes=(0, 0)))
_jac_v = jax.jit(jax.vmap(jax.hessian(_energy), in_axes=(0, 0)))


class Bushings:
    def __init__(self, model, pairs, kt, kr):
        """pairs: list of (entity A, entity B); B's reference point is the
        attachment point (a point rigidly fixed to A in the reference state)."""
        n = len(pairs)
        self.dofmap = np.array([np.concatenate([6 * a + np.arange(6), 6 * b + np.arange(6)]) for a, b in pairs], int)
        XA = np.array([model.X[a] for a, _ in pairs])
        XB = np.array([model.X[b] for _, b in pairs])
        self.D = dict(XA=jnp.asarray(XA), XB=jnp.asarray(XB), oA=jnp.asarray(XB - XA),
                      kt=jnp.full((n,), float(kt)), kr=jnp.full((n,), float(kr)))
        self.pairs = pairs

    def evaluate(self, Q, Qp):
        return _res_v(Q, self.D), _jac_v(Q, self.D)

    def forces(self, q):
        """Bushing force/moment on B, in A's frame: (n, 6)."""
        r = np.asarray(_res_v(jnp.asarray(q[self.dofmap]), self.D))
        return r[:, 6:]
