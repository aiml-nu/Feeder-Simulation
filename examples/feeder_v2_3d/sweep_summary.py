"""
Summarize the jaw-gap and mount-stiffness sweeps (full feeder runs).

    python examples/feeder_v2_3d/sweep_summary.py [results_dir]
expects <results_dir>/{gap0,gap0.05,gap0.1,gap0.15,gap0.2,mount10000,mount3000}.npz
(default results_dir: examples/feeder_v2_3d/results/cad); missing runs are skipped.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d import plots
from flexsim3d.plots import INK, INK2, MUTED

R = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples/feeder_v2_3d/results/cad"
MEASURED = 65.0          # N, paper (force measured at the nozzle)


def steady(name, window=180.0):
    d = np.load(R / f"{name}.npz", allow_pickle=False)
    psi, ram = d["psi"], d["ramp"]
    m = (ram >= 1) & (psi >= psi.max() - window)
    feed = d["feed_force"][m]
    N = d["N"][m].sum(axis=2)                       # per jaw
    clamp = np.percentile(N[N > 1], 90) if (N > 1).any() else 0.0
    on = d["N"][m] > 1e-3
    slip = (on & d["slip"][m].astype(bool)).sum() / max(on.sum(), 1)
    out = dict(name=name, mean=feed.mean(), lo=feed.min(), hi=feed.max(), clamp=clamp, slip=slip,
               util=d["leaf_util"][m].max(), travel=-d["wire_uz_bot"][m][-1])
    if "jaw_push" in d.files:
        out["push"] = d["jaw_push"][m].mean()
    return out


runs = {}
for n in ["gap0", "gap0.05", "gap0.1", "gap0.15", "gap0.2", "mount10000", "mount3000", "tube325_gap0", "tube325_gap0.1"]:
    if (R / f"{n}.npz").exists():
        runs[n] = steady(n)
        r = runs[n]
        print(f"{n:15s} nozzle {r['lo']:5.1f} / {r['mean']:5.1f} / {r['hi']:5.1f} N   clamp/jaw {r['clamp']:5.0f} N   "
              f"slip share {100 * r['slip']:4.1f}%   max P/Pcr {r['util']:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)


def panel(ax, names, labels, title, xlabel):
    xs = np.arange(len(names))
    for x, n in zip(xs, names):
        if n not in runs:
            continue
        r = runs[n]
        ax.bar(x, r["mean"], width=0.55, color="#2a78d6")
        ax.plot([x, x], [r["lo"], r["hi"]], color=INK, lw=1.6)
        ax.plot([x - 0.08, x + 0.08], [r["lo"]] * 2, color=INK, lw=1.6)
        ax.plot([x - 0.08, x + 0.08], [r["hi"]] * 2, color=INK, lw=1.6)
        ax.text(x, r["hi"] + 3, f"{r['mean']:.0f} N\nclamp {r['clamp']:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.axhline(MEASURED, color="#eb6834", lw=1.6, ls="--")
    ax.text(-0.45, MEASURED + 2, "measured 65 N", color="#eb6834", ha="left", fontsize=9, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xticks(xs, labels)
    ax.grid(axis="x", visible=False)
    ax.set(title=title, xlabel=xlabel)


ax = axs[0]
gaps = [(0.0, "gap0"), (0.05, "gap0.05"), (0.1, "gap0.1"), (0.15, "gap0.15"), (0.2, "gap0.2")]
gaps = [(g, n) for g, n in gaps if n in runs]
gx = np.array([g for g, _ in gaps])
mean = np.array([runs[n]["mean"] for _, n in gaps])
ax.fill_between(gx, [runs[n]["lo"] for _, n in gaps], [runs[n]["hi"] for _, n in gaps], color="#2a78d6", alpha=0.15, lw=0)
ax.plot(gx, mean, color="#2a78d6", marker="o", ms=7)
for g, n in gaps:
    ax.annotate(f"clamp {runs[n]['clamp']:.0f} N/jaw", (g, runs[n]["hi"]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8.5, color=INK2)
ax.axhline(MEASURED, color="#eb6834", lw=1.6, ls="--")
ax.text(gx.max(), MEASURED + 2, "measured 65 N", color="#eb6834", ha="right", fontsize=9)
ax.set(title="Jaw-to-wire gap", xlabel="Radial gap at assembly [mm] (CAD = 0)")
ax.set_xticks(gx)
panel(axs[1], ["gap0", "mount10000", "mount3000"], ["rigid", "10,000 N/mm", "3,000 N/mm"],
      "Printed mount stiffness", "Stiffness per stack-end block")
axs[0].set_ylabel("Steady feed force at nozzle [N]\n(mean; band / whiskers: min-max)")
ymax = max([r["hi"] for r in runs.values()] + [MEASURED]) * 1.3
axs[0].set_ylim(0, ymax)
fig.savefig(R / "sweep_summary.png")
json.dump({k: {kk: float(vv) if not isinstance(vv, str) else vv for kk, vv in v.items()} for k, v in runs.items()},
          open(R / "sweep_summary.json", "w"), indent=1)
print("saved", R / "sweep_summary.png")
