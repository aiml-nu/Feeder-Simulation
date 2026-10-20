"""
Corotational 3D beam element (2 nodes, 6 DOF per node) for geometrically
nonlinear analysis, incl. buckling and post-buckling.

Formulation (Crisfield 1990; Battini & Pacoste 2002 for the 3D frame):

1. Each element end belongs to a 6-DOF "owner" entity (a beam node or a rigid
   body). End position  p = X + u + R(psi) @ o   (o = offset from the owner's
   reference point; zero for ordinary beam nodes). End triad  T = R(psi) @ E0,
   where E0 is the element's own initial frame [e1 e2 e3] (e1 along the chord).
2. A corotated element frame Er is built from the current chord e1 and the
   average of the two end triads' second axes (objective under rigid motion).
3. Local deformational rotations theta_a = log(Er^T Ta), theta_b = log(Er^T Tb)
   are small even when the element rotates a lot, so a *linear* local strain
   energy is accurate; all large-rotation geometry lives in step 2.
4. Local energy = axial (with the cubic-interpolation "bowing" term, which
   gives the consistent geometric stiffness and makes few-element buckling
   loads accurate) + torsion + Timoshenko bending about both axes.

The global internal force and tangent stiffness are the exact gradient and
Hessian of this energy, obtained with JAX autodiff (so the tangent is always
consistent with the residual -> quadratic Newton convergence).

Section axes convention: e2 and e3 are the cross-section axes. EI3 resists
bending that deflects the beam along e2; EI2 resists deflection along e3.
For a leaf flexure we use e2 = thickness direction, so EI3 = E w t^3 / 12.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .rotation import expm, logv, normalize

# element parameter vector layout
P_EA, P_GJ, P_EI2, P_EI3, P_PHI2, P_PHI3, P_L0 = range(7)
N_PARAM = 7


@dataclass
class Section:
    """Cross-section + material. Units: mm, N, MPa."""
    E: float
    nu: float
    A: float
    I2: float          # second moment about e2 (deflection along e3)
    I3: float          # second moment about e3 (deflection along e2)
    J: float           # torsion constant
    kappa: float = 5.0 / 6.0
    c2: float = 0.0    # extreme-fibre distance along e3 (for stress)
    c3: float = 0.0    # extreme-fibre distance along e2 (for stress)

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @staticmethod
    def rectangle(E, nu, t, w):
        """t along e2 (thin), w along e3 (wide)."""
        a, b = max(t, w), min(t, w)
        J = a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - (b / a) ** 4 / 12.0))
        return Section(E=E, nu=nu, A=t * w, I2=t * w**3 / 12.0, I3=w * t**3 / 12.0,
                       J=J, kappa=5.0 / 6.0, c2=w / 2.0, c3=t / 2.0)

    @staticmethod
    def circle(E, nu, d):
        A = np.pi * d**2 / 4.0
        I = np.pi * d**4 / 64.0
        return Section(E=E, nu=nu, A=A, I2=I, I3=I, J=2.0 * I, kappa=0.9,
                       c2=d / 2.0, c3=d / 2.0)

    def params(self, L0):
        G = self.G
        phi2 = 12.0 * self.E * self.I2 / (self.kappa * G * self.A * L0**2)
        phi3 = 12.0 * self.E * self.I3 / (self.kappa * G * self.A * L0**2)
        return np.array([self.E * self.A, G * self.J, self.E * self.I2, self.E * self.I3,
                         phi2, phi3, L0])


def _end_state(q, X, o, E0):
    R = expm(q[3:6])
    p = X + q[0:3] + R @ o
    T = R @ E0
    return p, T


def local_deformation(qa, qb, Xa, Xb, oa, ob, E0, prm):
    """Return (l, theta_a, theta_b) in the corotated frame."""
    pa, Ta = _end_state(qa, Xa, oa, E0)
    pb, Tb = _end_state(qb, Xb, ob, E0)
    d = pb - pa
    l = jnp.sqrt(jnp.dot(d, d))
    e1 = d / l
    qv = 0.5 * (Ta[:, 1] + Tb[:, 1])
    e2 = normalize(qv - jnp.dot(qv, e1) * e1)
    e3 = jnp.cross(e1, e2)
    Er = jnp.stack([e1, e2, e3], axis=1)
    tha = logv(Er.T @ Ta)
    thb = logv(Er.T @ Tb)
    return l, tha, thb


def _bend_energy(EI, phi, L0, a, b):
    c = EI / (L0 * (1.0 + phi))
    return 0.5 * c * ((4.0 + phi) * (a * a + b * b) + 2.0 * (2.0 - phi) * a * b)


def element_energy(qa, qb, Xa, Xb, oa, ob, E0, prm):
    L0 = prm[P_L0]
    l, tha, thb = local_deformation(qa, qb, Xa, Xb, oa, ob, E0, prm)
    bow = (2.0 * tha[1] ** 2 - tha[1] * thb[1] + 2.0 * thb[1] ** 2
           + 2.0 * tha[2] ** 2 - tha[2] * thb[2] + 2.0 * thb[2] ** 2) / 30.0
    eps = (l - L0) / L0 + bow
    U = 0.5 * prm[P_EA] * L0 * eps * eps
    U += 0.5 * prm[P_GJ] / L0 * (thb[0] - tha[0]) ** 2
    U += _bend_energy(prm[P_EI3], prm[P_PHI3], L0, tha[2], thb[2])
    U += _bend_energy(prm[P_EI2], prm[P_PHI2], L0, tha[1], thb[1])
    return U


def element_section_forces(qa, qb, Xa, Xb, oa, ob, E0, prm):
    """Local resultants: [N, T, M2a, M3a, M2b, M3b] (N > 0 tension)."""
    L0 = prm[P_L0]
    l, tha, thb = local_deformation(qa, qb, Xa, Xb, oa, ob, E0, prm)
    bow = (2.0 * tha[1] ** 2 - tha[1] * thb[1] + 2.0 * thb[1] ** 2
           + 2.0 * tha[2] ** 2 - tha[2] * thb[2] + 2.0 * thb[2] ** 2) / 30.0
    N = prm[P_EA] * ((l - L0) / L0 + bow)
    T = prm[P_GJ] / L0 * (thb[0] - tha[0])

    def mom(EI, phi, a, b):
        c = EI / (L0 * (1.0 + phi))
        return c * ((4.0 + phi) * a + (2.0 - phi) * b), c * ((4.0 + phi) * b + (2.0 - phi) * a)

    M3a, M3b = mom(prm[P_EI3], prm[P_PHI3], tha[2], thb[2])
    M2a, M2b = mom(prm[P_EI2], prm[P_PHI2], tha[1], thb[1])
    return jnp.array([N, T, M2a, M3a, M2b, M3b])


def frame_from_axis(e1, e2_hint):
    """Orthonormal frame [e1 e2 e3] with e2 as close as possible to e2_hint."""
    e1 = np.asarray(e1, float)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.asarray(e2_hint, float)
    e2 = e2 - e2.dot(e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)
