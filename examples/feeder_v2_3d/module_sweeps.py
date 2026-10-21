"""
Fast single-module sweeps (no wire): clamp force vs radial interference for
  (a) mounting-block stiffness k_mount, (b) leaf imperfection amplitude.
A jaw gap g simply shifts the curve: interference = 0.2 mm design - g.

    python examples/feeder_v2_3d/module_sweeps.py
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d import plots
from flexsim3d.plots import INK, INK2, MUTED
from flexsim3d.studies import build_module, clamp_sweep
from flexsim3d.solver import lowest_eigs

out = ROOT / "examples/feeder_v2_3d/results"
res = {}
ks = [0.0, 3e4, 1e4, 3e3, 1e3]
for k in ks:
    p, F = build_module(k_mount=k)
    rows, _ = clamp_sweep(F, n=40, d_max=0.4)
    res[k] = rows
    f = lambda d: np.interp(d, rows[:, 0], rows[:, 1])
    print(f"k_mount={k:7.0f}: k_lin={rows[0, 1] / rows[0, 0]:6.0f} N/mm  F(0.1)={f(0.1):5.0f}  F(0.2)={f(0.2):5.0f}  peak={rows[:, 1].max():5.0f} N at {rows[np.argmax(rows[:, 1]), 0]:.2f} mm", flush=True)
imps = [0.0, 0.005, 0.02, 0.05, 0.1]
ires = {}
for a in imps:
    p, F = build_module(imperfection=a, imp_sign="same")
    rows, q = clamp_sweep(F, n=40, d_max=0.4)
    mod = F.modules[0]
    pres = {6 * mod["inputs"][1] + 2: 0.0, 6 * mod["inputs"][-1] + 2: 0.0, 6 * mod["jaw"]: rows[-1, 0] * mod["er"][0]}
    ires[a] = (rows, lowest_eigs(F.m, q, pres, k=1)[0][0])
    print(f"imperfection {a * 1000:4.0f} um: peak {rows[:, 1].max():5.0f} N, reached {rows[-1, 0]:.2f} mm, min eig {ires[a][1]:+.3f}", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
cols = plots.SEQ(np.linspace(1.0, 0.35, len(ks)))
for c, k in zip(cols, ks):
    r = res[k]
    axs[0].plot(r[:, 0], r[:, 1], color=c if k else INK, lw=2.2, label="rigid mounts" if not k else f"{k:,.0f} N/mm")
axs[0].axvline(0.2, color=MUTED, lw=1)
axs[0].text(0.2, 15, " gap 0", color=INK2, fontsize=9)
axs[0].axvline(0.1, color=MUTED, lw=1, ls=":")
axs[0].text(0.1, 15, " gap 0.1 mm", color=INK2, fontsize=9)
axs[0].set(xlabel="Radial jaw interference [mm]  (design 0.2 mm minus jaw gap)",
           ylabel="Clamp force per jaw [N]", title="Mounting-block stiffness")
axs[0].legend(loc="upper left", fontsize=9.5)
cols = plots.SEQ(np.linspace(0.35, 1.0, len(imps)))
for c, a in zip(cols, imps):
    r, ev = ires[a]
    axs[1].plot(r[:, 0], r[:, 1], color=c, lw=2.2, ls="-" if ev > 0 else "--",
                label=f"{a * 1000:.0f} µm" + ("" if ev > 0 else " (unstable branch)"))
axs[1].set(xlabel="Radial jaw interference [mm]", ylabel="Clamp force per jaw [N]",
           title="Leaf initial bow (all the same way)")
axs[1].legend(loc="upper left", fontsize=9.5)
fig.savefig(out / "module_sweeps.png")
np.savez(out / "module_sweeps.npz", **{f"k{k:g}": res[k] for k in ks}, **{f"imp{a:g}": ires[a][0] for a in imps})
print("saved", out / "module_sweeps.png")
