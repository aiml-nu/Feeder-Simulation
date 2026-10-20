"""
Slide-ready plots from a feeder results .npz (see examples/feeder_v2_3d/run_feeder.py).

Colors: jaw pairs use the first three categorical slots (blue, orange, aqua),
which are validated for colour-vision deficiency as a set; line-load maps use
a single-hue sequential blue ramp.
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False, "lines.linewidth": 2.0,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    d["params"] = json.loads(str(d["params"]))
    return d


def _cam(d):
    """Cam angle for the cam-driven part of the run (drop the assembly ramp)."""
    mask = d["ramp"] >= 1.0 - 1e-12
    first = np.argmax(mask)
    mask[first] = True
    return mask, d["psi"][mask]


def pair_of(k):
    return k % 3


def _phase_bands(ax, d, pair=0):
    """Light background bands marking when `pair` is clamped (contacting)."""
    p = d["params"]
    S = p["psi_c"] + p["psi_p"] + p["psi_u"]
    per = S + p["psi_r"]
    ph = 60.0 * pair
    lo, hi = ax.get_xlim()
    k0 = int(np.floor((lo - ph) / per)) - 1
    for k in range(k0, k0 + int((hi - lo) / per) + 3):
        a = ph + k * per
        ax.axvspan(a + p["psi_c"], a + p["psi_c"] + p["psi_p"], color=PAIR_COLORS[pair], alpha=0.07, lw=0)


def fig_forces(d, path=None):
    m, psi = _cam(d)
    N = d["N"][m]           # (steps, jaws, wire nodes)
    ftz = d["ftz"][m]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax = axs[0]
    ax.plot(psi, d["feed_force"][m], color=INK)
    ax.set_ylabel("Feed force [N]")
    ax.set_title("Feed force carried by the wire (bottom spring)")
    ax = axs[1]
    for pr in range(3):
        ax.plot(psi, N[:, [pr, pr + 3]].sum(axis=2).mean(axis=1), color=PAIR_COLORS[pr], label=f"Pair {pr + 1}")
    ax.set_ylabel("Clamp force per jaw [N]")
    ax.set_title("Normal (clamping) force")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax = axs[2]
    for pr in range(3):
        ax.plot(psi, -ftz[:, [pr, pr + 3]].sum(axis=2).mean(axis=1), color=PAIR_COLORS[pr], label=f"Pair {pr + 1}")
    mu = d["params"]["mu"]
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("Downward push per jaw [N]")
    ax.set_title(f"Friction on the wire (+ feeding, − resisting), μ = {mu}")
    ax.set_xlabel("Cam angle [deg]")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    fig.align_ylabels(axs)
    if path:
        fig.savefig(path)
    return fig


def line_load_grid(d, jaw, ngrid=181):
    """Line load q [N/mm] on a fixed grid along the jaw (0 = bottom, L = top)."""
    m, psi = _cam(d)
    L = d["params"]["jaw_length"]
    hw = d["params"]["wire_h"]
    s = d["s_on_jaw"][m][:, jaw, :]
    q = d["N"][m][:, jaw, :] / hw
    grid = np.linspace(0, L, ngrid)
    Q = np.array([np.interp(grid, s[i], q[i], left=0, right=0) for i in range(len(psi))])
    return psi, grid, Q


def fig_line_load(d, path=None, jaws=(0, 1, 2)):
    fig, axs = plt.subplots(len(jaws), 1, figsize=(10, 2.6 * len(jaws) + 0.6), sharex=True)
    axs = np.atleast_1d(axs)
    vmax = 0
    grids = []
    for j in jaws:
        g = line_load_grid(d, j)
        grids.append(g)
        vmax = max(vmax, np.percentile(g[2], 99.5))
    for ax, j, (psi, grid, Q) in zip(axs, jaws, grids):
        im = ax.pcolormesh(psi, grid, Q.T, cmap=SEQ, vmin=0, vmax=vmax, shading="nearest", rasterized=True)
        ax.set_ylabel("Along jaw [mm]\n(0 = bottom)")
        ax.set_title(f"Jaw {j + 1} (pair {pair_of(j) + 1}): contact line load", loc="left")
        ax.grid(False)
    axs[-1].set_xlabel("Cam angle [deg]")
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.015)
    cb.set_label("Line load [N/mm]")
    if path:
        fig.savefig(path)
    return fig


def fig_profiles(d, jaw=0, n=6, path=None):
    """Line-load profiles along the jaw at evenly spaced angles within one clamped interval."""
    p = d["params"]
    psi, grid, Q = line_load_grid(d, jaw)
    tot = Q.sum(axis=1)
    on = tot > 0.05 * tot.max()
    # pick the last complete contact interval
    idx = np.nonzero(on)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    seg = np.split(idx, breaks + 1)
    seg = seg[-2] if len(seg) > 1 else seg[-1]
    picks = seg[np.linspace(0, len(seg) - 1, n).astype(int)]
    cm = SEQ(np.linspace(0.35, 1.0, n))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for c, i in zip(cm, picks):
        ax.plot(Q[i], grid, color=c, label=f"{psi[i]:.0f}°")
    ax.set_xlabel("Line load [N/mm]")
    ax.set_ylabel("Position along jaw [mm] (0 = bottom)")
    ax.set_title(f"Jaw {jaw + 1}: load distribution through one grip")
    ax.legend(title="Cam angle", loc="best")
    if path:
        fig.savefig(path)
    return fig


def fig_buckling(d, module=0, path=None):
    m, psi = _cam(d)
    U = d["leaf_util"][m]
    kinds = [("angled", 1, "Angled, top"), ("angled", -1, "Angled, bottom"),
             ("straight", 1, "Straight, top"), ("straight", -1, "Straight, bottom")]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (kind, lev, lab), c in zip(kinds, cols):
        sel = (d["leaf_kind"] == kind) & (d["leaf_level"] == lev) & (d["leaf_module"] == module)
        ax.plot(psi, U[:, sel].max(axis=1), color=c, label=lab)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    quiet = psi[np.argmin(U.max(axis=1))]
    ax.text(quiet, 0.985, "fixed-fixed Euler load", color=INK2, va="top", ha="center", fontsize=9)
    ax.set_ylabel("Max leaf compression / P_cr")
    ax.set_xlabel("Cam angle [deg]")
    ax.set_title(f"Module {module + 1}: most-compressed leaf per stack")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    if path:
        fig.savefig(path)
    return fig


def fig_slip(d, path=None):
    """Fraction of each pair's contact length that is slipping (|slip| > 3 eps)."""
    m, psi = _cam(d)
    eps = d["params"]["eps_slip"]
    N, sl = d["N"][m], d["slip"][m]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for pr in range(3):
        on = N[:, [pr, pr + 3]] > 1e-3
        slipping = on & (sl[:, [pr, pr + 3]] > 3 * eps)
        frac = slipping.sum(axis=(1, 2)) / np.maximum(on.sum(axis=(1, 2)), 1)
        ax.plot(psi, frac * 100, color=PAIR_COLORS[pr], label=f"Pair {pr + 1}")
    ax.set_ylabel("Contact slipping [%]")
    ax.set_xlabel("Cam angle [deg]")
    ax.set_ylim(-3, 103)
    ax.set_title("Jaw slip (share of in-contact length sliding on the wire)")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    if path:
        fig.savefig(path)
    return fig
