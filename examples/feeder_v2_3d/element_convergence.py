"""
Mesh convergence: elements per leaf vs. single-module clamp response.

    python examples/feeder_v2_3d/element_convergence.py
"""
import pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d import plots
from flexsim3d.studies import build_module, clamp_sweep

out = ROOT / "examples/feeder_v2_3d/results"
out.mkdir(parents=True, exist_ok=True)
counts = [int(c) for c in sys.argv[1:]] or [2, 3, 4, 6, 8, 10, 12, 16]
res = {}
for n in counts:
    t0 = time.time()
    p, F = build_module(n_el_leaf=n)
    rows, _ = clamp_sweep(F, n=30, d_max=0.3)
    res[n] = rows
    f02 = np.interp(0.2, rows[:, 0], rows[:, 1])
    print(f"n_el={n}: {F.m.ndof:5d} DOF, reached {rows[-1, 0]:.2f} mm, F(0.2)={f02:6.1f} N, "
          f"F(end)={rows[-1, 1]:6.1f} N, max thin-bend {rows[:, 5].max():5.1f} MPa, "
          f"max wide-bend {rows[:, 6].max():5.2f} MPa  ({time.time() - t0:.0f} s)")

ref = res[counts[-1]]
cols = plots.SEQ(np.linspace(0.3, 1.0, len(counts)))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for c, n in zip(cols, counts):
    r = res[n]
    axs[0].plot(r[:, 0], r[:, 1], color=c, label=f"{n} el/leaf")
    axs[1].plot(r[:, 0], r[:, 5], color=c, label=f"{n} el/leaf")
axs[0].set(xlabel="Radial jaw interference [mm]", ylabel="Clamp force per jaw [N]", title="Clamp force")
axs[1].set(xlabel="Radial jaw interference [mm]", ylabel="Max thin-direction bending stress [MPa]",
           title="Peak leaf bending stress")
axs[0].legend(loc="upper left")
fig.savefig(out / "element_convergence.png")
np.savez(out / "element_convergence.npz", **{f"n{n}": res[n] for n in counts})
print("saved", out / "element_convergence.png")
