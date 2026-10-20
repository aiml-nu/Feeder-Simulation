"""
Reusable small studies on a single jaw module (no wire).
"""
from __future__ import annotations

import numpy as np

from .feeder import Feeder, FeederParams
from .solver import solve_step


def build_module(**over):
    p = FeederParams(n_modules=1)
    for k, v in over.items():
        setattr(p, k, type(getattr(p, k))(v))
    return p, Feeder(p, with_wire=False)


def appendix_estimate(p):
    """Eqs. A.1-A.6: clamp stiffness per jaw, as written and with cos^2(phi/2)."""
    E, w, t, L1, L2, N = p.E_flex, p.w, p.t, p.L1, p.L2, p.N_leaves
    A, I = w * t, w * t**3 / 12
    th = np.deg2rad(p.theta_deg)
    ks = N * E * A / L1
    ka = N * (E * A / L2 * np.cos(th) ** 2 + 12 * E * I / L2**3 * np.sin(th) ** 2)
    kf = 2 * ks * ka / (ks + ka)
    phi_full = np.deg2rad(p.V_angle_deg)
    return dict(written=2 * kf * np.cos(phi_full / 2), corrected=2 * kf * np.cos(phi_full / 2) ** 2)


def leaf_stress_components(F, ef):
    """Per-leaf max |stress| components [MPa] from element section forces
    ef = [N, T, M2a, M3a, M2b, M3b]:
      axial      |N|/A
      thin       |M3| c3/I3   (bending in the leaf's thin, intended direction)
      wide       |M2| c2/I2   (bending about the stiff axis: 'abnormal' direction)
      torsion    |T| t/J      (max shear, thin rectangle)
    """
    s = F.leaf_section
    le = F.leaf_elems
    ax = np.abs(ef[:, 0]) / s.A
    thin = np.maximum(np.abs(ef[:, 3]), np.abs(ef[:, 5])) * s.c3 / s.I3
    wide = np.maximum(np.abs(ef[:, 2]), np.abs(ef[:, 4])) * s.c2 / s.I2
    tor = np.abs(ef[:, 1]) * (2 * s.c3) / s.J
    return {k: v[le].max(axis=1) for k, v in dict(axial=ax, thin=thin, wide=wide, torsion=tor).items()}


def clamp_sweep(F, n=40, d_max=0.4, verbose=False):
    """Radially displace jaw 0 outward (the wire's reaction direction) with the
    inputs held; return rows (d, clamp force, max P/Pcr, share>0.95, max stress per component)."""
    m = F.m
    mod = F.modules[0]
    jaw, er = mod["jaw"], mod["er"]
    q = np.zeros(m.ndof)
    fixed = {6 * mod["inputs"][lev] + 2: 0.0 for lev in (+1, -1)}
    rows = []
    for d in np.linspace(0, d_max, n + 1)[1:]:
        pres = dict(fixed)
        pres[6 * jaw + 0] = d * er[0]
        res = solve_step(m, q, pres, q_prev=q, maxit=40)
        if not res.converged:
            if verbose:
                print("no convergence at", d)
            break
        q = res.q
        r, _ = m.assemble(q, q, with_K=False)
        ef = m.element_forces(q)
        util = np.maximum(-ef[:, 0][F.leaf_elems].mean(1), 0) / F.leaf_Pcr
        sc = leaf_stress_components(F, ef)
        rows.append((d, r[6 * jaw + 0] * er[0], util.max(), np.mean(util > 0.95),
                     sc["axial"].max(), sc["thin"].max(), sc["wide"].max(), sc["torsion"].max()))
    return np.array(rows), q
