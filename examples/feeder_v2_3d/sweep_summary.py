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
NO_CONTACT = 0.175       # mm: from this gap on the jaw stroke never reaches the wire (E-independent; seen in runs)


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
def _num(stem, pre):
    try:
        return float(stem[len(pre):])
    except ValueError:
        return None


names = sorted([f.stem for f in R.glob("gap*.npz") if _num(f.stem, "gap") is not None], key=lambda n: _num(n, "gap"))
names += sorted([f.stem for f in R.glob("mount*.npz") if _num(f.stem, "mount") is not None], key=lambda n: -_num(n, "mount"))
names += sorted(f.stem for f in R.glob("tube*_gap*.npz"))
for n in names:
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
gaps = [(_num(n, "gap"), n) for n in names if n.startswith("gap")]
gx = np.array([g for g, _ in gaps])
mean = np.array([runs[n]["mean"] for _, n in gaps])
ax.fill_between(gx, [runs[n]["lo"] for _, n in gaps], [runs[n]["hi"] for _, n in gaps], color="#2a78d6", alpha=0.15, lw=0)
ax.plot(gx, mean, color="#2a78d6", marker="o", ms=7)
for i, (g, n) in enumerate(gaps):     # alternate sides so neighbouring labels do not collide
    ax.annotate(f"{runs[n]['mean']:.0f} N\nclamp {runs[n]['clamp']:.0f} N/jaw", (g, runs[n]["mean"]), textcoords="offset points",
                xytext=(8, 4) if i % 2 == 0 else (-8, -6), ha="left" if i % 2 == 0 else "right",
                va="bottom" if i % 2 == 0 else "top", fontsize=8.5, color=INK2)
ax.axhline(MEASURED, color="#eb6834", lw=1.6, ls="--")
ax.text(gx.min(), MEASURED - 4, "measured 65 N", color="#eb6834", ha="left", va="top", fontsize=9)
E = json.loads(str(np.load(R / f"{gaps[0][1]}.npz")["params"]))["E_flex"]
ax.set(title=f"Jaw-to-wire gap (E = {E / 1000:g} GPa)", xlabel="Radial gap at assembly [mm] (CAD = 0)")
ax.set_xticks(np.round(np.arange(0, 0.2 + 1e-9, 0.05), 3))
ax.set_xlim(-0.012, 0.215)
if NO_CONTACT:
    ax.axvspan(NO_CONTACT, 0.215, color=MUTED, alpha=0.12, lw=0)
    ax.text(NO_CONTACT + 0.004, 0.5, "jaws never\nreach the wire", fontsize=8.5, color=INK2, va="center", transform=ax.get_xaxis_transform())
mounts = [n for n in names if n.startswith("mount")]
panel(axs[1], ["gap0"] + mounts, ["rigid"] + [f"{_num(n, 'mount'):,.0f} N/mm" for n in mounts],
      "Printed mount stiffness", "Stiffness per stack-end block")
axs[0].set_ylabel("Steady feed force at nozzle [N]\n(mean; band / whiskers: min-max)")
ymax = max([r["hi"] for r in runs.values()] + [MEASURED]) * 1.3
axs[0].set_ylim(0, ymax)
fig.savefig(R / "sweep_summary.png")
json.dump({k: {kk: float(vv) if not isinstance(vv, str) else vv for kk, vv in v.items()} for k, v in runs.items()},
          open(R / "sweep_summary.json", "w"), indent=1)
print("saved", R / "sweep_summary.png")
