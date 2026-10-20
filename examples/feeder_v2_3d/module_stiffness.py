"""
Single jaw module (no wire): clamp force vs radial jaw interference, and
feed-direction (axial) force vs jaw travel while clamped, from the nonlinear
3D model. Compared against the Appendix A linear stiffness estimate.

    python examples/feeder_v2_3d/module_stiffness.py [--set E_flex=3000]
"""
import argparse, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d.feeder import Feeder, FeederParams
from flexsim3d.solver import solve_step
from flexsim3d import plots  # noqa: F401  (rcParams / palette)
from flexsim3d.plots import INK, INK2, MUTED

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--out", default=str(ROOT / "examples/feeder_v2_3d/results/module_stiffness"))
a = ap.parse_args()


def build(over):
    p = FeederParams(n_modules=1)
    for kv in over:
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(v))
    F = Feeder(p, with_wire=False)
    return p, F


def appendix_estimate(p):
    """Eqs. A.1-A.6 (paper as written and corrected)."""
    E, w, t, L1, L2, N = p.E_flex, p.w, p.t, p.L1, p.L2, p.N_leaves
    A, I = w * t, w * t**3 / 12
    th = np.deg2rad(p.theta_deg)
    ks = N * E * A / L1
    ka = N * (E * A / L2 * np.cos(th) ** 2 + 12 * E * I / L2**3 * np.sin(th) ** 2)
    kf = 2 * ks * ka / (ks + ka)
    phi_full = np.deg2rad(p.V_angle_deg)
    return dict(written=2 * kf * np.cos(phi_full / 2), corrected=2 * kf * np.cos(phi_full / 2) ** 2)


def sweep(F, p, n=40, d_max=0.4):
    m = F.m
    mod = F.modules[0]
    jaw = mod["jaw"]
    er = mod["er"]
    q = np.zeros(m.ndof)
    inputs_fixed = {6 * mod["inputs"][lev] + 2: 0.0 for lev in (+1, -1)}
    out = []
    for d in np.linspace(0, d_max, n + 1)[1:]:
        pres = dict(inputs_fixed)
        # the wire pushes the jaw OUTWARD relative to where the inputs put it
        # (= interference): displace the jaw radially outward, inputs held
        pres[6 * jaw + 0] = d * er[0]
        res = solve_step(m, q, pres, q_prev=q, maxit=40)
        if not res.converged:
            print("no convergence at", d); break
        q = res.q
        r, _ = m.assemble(q, q, with_K=False)
        ef = m.element_forces(q)
        util = np.maximum(-ef[:, 0][F.leaf_elems].mean(1), 0) / F.leaf_Pcr
        out.append((d, r[6 * jaw + 0] * er[0], util.max(), np.mean(util > 0.95)))
    return np.array(out), q


p, F = build(a.set)
est = appendix_estimate(p)
res, qc = sweep(F, p)
k_lin = res[0, 1] / res[0, 0]
print(f"3D model initial clamp stiffness: {k_lin:.0f} N/mm per jaw;  A.6 as written {est['written']:.0f}, corrected {est['corrected']:.0f}")
print(f"clamp force at 0.2 mm interference: {np.interp(0.2, res[:, 0], res[:, 1]):.0f} N/jaw "
      f"(A.6 as written: {0.2 * est['written']:.0f}, corrected: {0.2 * est['corrected']:.0f})")
if (res[:, 2] > 0.95).any():
    i_b = np.argmax(res[:, 2] > 0.95)
    print(f"first leaves reach 0.95 P_cr at {res[i_b, 0]:.3f} mm, {res[i_b, 1]:.0f} N")
print("max leaf P/Pcr at end:", res[-1, 2], " share of leaves >0.95:", res[-1, 3])

fig, ax = plt.subplots(figsize=(7.5, 4.8))
dd = np.linspace(0, res[-1, 0], 50)
ax.plot(dd, est["written"] * dd, color=MUTED, ls="--", lw=1.5, label=f"Eq. A.6 as written ({est['written']:.0f} N/mm)")
ax.plot(dd, est["corrected"] * dd, color=INK2, ls=":", lw=1.8, label=f"A.6 with cos²(φ/2) ({est['corrected']:.0f} N/mm)")
ax.plot(res[:, 0], res[:, 1], color="#2a78d6", lw=2.4, label="3D nonlinear model")
ax.axvline(0.2, color=GRID if (GRID := "#c9c8c2") else None, lw=1)
ax.text(0.203, ax.get_ylim()[1] * 0.04, "design interference 0.2 mm", color=INK2, fontsize=9)
ax.set_xlabel("Radial jaw interference [mm]")
ax.set_ylabel("Clamp force per jaw [N]")
ax.set_title(f"One jaw module, inputs held (E = {p.E_flex:.0f} MPa)")
ax.legend(loc="upper left")
out = pathlib.Path(a.out + (f"_E{int(p.E_flex)}" if p.E_flex != 1000 else "") + ".png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
np.savez(out.with_suffix(".npz"), res=res, **{k: v for k, v in est.items()})
print("saved", out)
