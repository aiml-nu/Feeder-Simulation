"""
Validation of the wire-in-tube contact: a 250 mm wire, pushed at its top end
into a straight tube (radial clearance r = 0.05 mm) against a fixed bottom end.
The wire buckles into a helix against the wall; friction makes the force that
reaches the bottom smaller than the force applied at the top.

Analytical reference (weightless rod, fully developed helix; Lubinski 1962,
Mitchell 1986): the wall contact force per unit length is
    w = r F^2 / (4 EI)
and with Coulomb friction dF/ds = -mu w, so
    F_bottom = F_top / (1 + mu r L F_top / (4 EI)).

    python examples/feeder_v2_3d/tube_validation.py
"""
import pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d import Model, Section, solve_step, plots
from flexsim3d.solver import solve_stabilized
from flexsim3d.contact import WireTubeContact

E, nu, d, L, Ri, mu = 70000.0, 0.33, 0.9, 250.0, 0.5, 0.3
h = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
sec = Section.circle(E, nu, d)
EI = E * sec.I3
r = Ri - d / 2

m = Model()
n_el = int(round(L / h))
top = m.add_entity([0, 0, 0.0], "node", "top")
bot = m.add_entity([0, 0, -L], "node", "bottom")
lam, a = 30.0, 0.04        # helical imperfection ~ clearance (spool cast), tapered to zero at the ends
# long-wavelength bow (spool cast) + short helix: covers both the low modes
# (Euler load ~0.4 N over 250 mm) and helix-like wavelengths
imp = (1.0, lambda s: a * np.array([np.sin(np.pi * s) + 0.25 * np.sin(np.pi * s) * np.cos(2 * np.pi * s * L / lam),
                                    0.5 * np.sin(2 * np.pi * s) + 0.25 * np.sin(np.pi * s) * np.sin(2 * np.pi * s * L / lam),
                                    0.0]) / 1.25)
g = m.add_member(top, [0, 0, 0], bot, [0, 0, -L], sec, (1, 0, 0), n_el, imp, "wire")
nodes = list(g.nodes)
z = np.array([m.X[k][2] for k in nodes])
hh = np.abs(np.diff(z))
ltrib = np.concatenate([[hh[0] / 2], 0.5 * (hh[:-1] + hh[1:]), [hh[-1] / 2]])
m.fix(top, [0, 1, 5])
m.fix(bot, [0, 1, 2, 5])
tube = m.add_term(WireTubeContact(m, nodes, ltrib, z_top=0.5, length=L + 1, inner_radius=Ri,
                                  rw=d / 2, mu=mu))
m.finalize()
eta_lat = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
du_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.003   # ~0.5 N per step, like the feeder
if eta_lat > 0:     # same always-on lateral regularization as the feeder model
    _, K0 = m.assemble(np.zeros(m.ndof), np.zeros(m.ndof))
    c = np.zeros(m.ndof)
    for k in nodes:
        for comp in (0, 1, 3, 4):
            if 6 * k + comp not in m.fixed:
                c[6 * k + comp] = eta_lat * abs(K0[6 * k + comp, 6 * k + comp])
    m.stab_base = c

q = np.zeros(m.ndof)
out = []
t0 = time.time()
stab_energy, n_stab = 0.0, 0
u, du = 0.0, 2e-4
while u < 1.2:
    u_new = u + du
    res = solve_step(m, q, {6 * top + 2: -u_new}, q_prev=q, maxit=30)
    if not res.converged and du < 1e-4:
        res, e = solve_stabilized(m, q, {6 * top + 2: -u_new}, q, eta=1e-4)
        stab_energy += e
        n_stab += res.converged
    if not res.converged:
        du /= 2
        if du < 1e-7:
            print("stalled at", u); break
        continue
    if getattr(m, "stab_base", None) is not None:
        stab_energy += float(m.stab_base @ (res.q - q) ** 2)
    m.commit(res.q, q)
    q, u = res.q, u_new
    du = min(du * 1.5, du_max)
    rr, _ = m.assemble(q, q, with_K=False)
    Ft, Fb = -rr[6 * top + 2], rr[6 * bot + 2]
    info = tube.info(q, q)  # after commit
    out.append((u, Ft, Fb, (info["N"] > 0).mean()))
    if len(out) % 50 == 0:
        print(f"  u={u:.3f}  F_top={Ft:6.1f}  F_bot={Fb:6.1f}  contact={(info['N'] > 0).mean():.2f}  t={time.time() - t0:.0f}s", flush=True)
out = np.array(out)
work = np.trapezoid(out[:, 1], out[:, 0]) if len(out) > 1 else 0.0
print(f"{len(out)} steps, {time.time() - t0:.0f} s, stabilized steps {n_stab}, "
      f"stabilization energy {stab_energy:.3g} N.mm vs work {work:.3g} N.mm")
Ft, Fb = out[:, 1], out[:, 2]
mitchell = Ft / (1 + mu * r * L * Ft / (4 * EI))
for F in (25, 50, 100, 150, 200):
    if Ft.max() >= F:
        i = np.argmin(abs(Ft - F))
        print(f"F_top {Ft[i]:6.1f} N: F_bottom {Fb[i]:6.1f} N (model), {mitchell[i]:6.1f} N (Mitchell), "
              f"loss {100 * (1 - Fb[i] / Ft[i]):4.1f}% vs {100 * (1 - mitchell[i] / Ft[i]):4.1f}%, "
              f"wall contact on {100 * out[i, 3]:.0f}% of nodes")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ft, 100 * (1 - Fb / Ft), color="#2a78d6", label="3D model")
ax.plot(Ft, 100 * (1 - mitchell / Ft), color=plots.INK2, ls="--", label="Helical buckling theory (Mitchell)")
ax.set(xlabel="Force applied at top [N]", ylabel="Force lost to wall friction [%]",
       title=f"250 mm wire in tube, clearance {r * 1000:.0f} µm, μ = {mu}")
base_E = 0.0
ax.legend(loc="upper left")
outp = ROOT / f"examples/feeder_v2_3d/results/tube_validation_eta{eta_lat:g}.png"
fig.savefig(outp)
np.save(outp.with_suffix(".npy"), out)
print("saved", outp)
