"""
Cam follower forces and Hertz contact stress on the printed face cams.

    python examples/feeder_v2_3d/cam_loads.py results/cad/tube325_gap0.npz results/cad/tube325_gap0.1.npz \
        --set D_bearing=10 b_contact=4 n_bearings=1 n_stackups=2 E_cam=1000

Writes cam_force.png, cam_stress.png and cam_loads.json next to the first .npz
(or --outdir). Steady state = the last 360 deg of cam rotation, jaw group 1.
See src/flexsim3d/cam_contact.py for the stack, sign convention and formulas.
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flexsim3d import plots
from flexsim3d.plots import INK, INK2, MUTED
from flexsim3d.cam_contact import CamStack, SURFACES, input_forces, follower_forces, surface_stress
from flexsim3d.feeder import FeederParams

LOWER, UPPER = "#2a78d6", "#eb6834"       # follower colour; solid = face-up cam, dashed = face-down cam

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--set", nargs="*", default=[], help="CamStack overrides name=value")
ap.add_argument("--pair", type=int, default=0, help="jaw group (0-2)")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

cs = CamStack.from_dict(dict(kv.split("=", 1) for kv in a.set))
out = pathlib.Path(a.outdir) if a.outdir else pathlib.Path(a.npz[0]).parent
out.mkdir(parents=True, exist_ok=True)


def label_of(d, path):
    g = d["params"].get("clearance", 0.0)
    tube = d["params"].get("tube_length", 0.0)
    return f"jaw gap {g:g} mm" + (f", {tube:g} mm tube" if tube else ", no tube")


cases = []
for i, f in enumerate(a.npz):
    d = plots.load(f)
    psi, fz, cam = input_forces(d)
    psi, fz = psi[cam], fz[cam]
    keep = psi >= psi.max() - 360.0
    psi, fz = psi[keep], fz[keep]
    ff = follower_forces(fz, n_modules=fz.shape[1], n_stackups=cs.n_stackups)
    P = FeederParams.from_dict(d["params"])
    st = surface_stress(psi, ff, cs, P, pair=a.pair)
    lab = a.labels[i] if a.labels and i < len(a.labels) else label_of(d, f)
    cases.append(dict(label=lab, psi=psi, ff=ff[:, a.pair], st=st, d=d))
    print(f"\n{lab}  ({f})")
    print(f"  input force per jaw [N]: top {fz[:, :, 0].min():.1f}..{fz[:, :, 0].max():.1f}, "
          f"bottom {fz[:, :, 1].min():.1f}..{fz[:, :, 1].max():.1f}")
    for name, s in st.items():
        j = int(np.argmax(s["p0"]))
        print(f"  {name:30s} peak bearing load {s['Fn'].max():6.1f} N   p_max {s['p0'][j]:5.1f} MPa   "
              f"interlayer shear {s['tau_xz'][j]:4.1f} MPa at {1000 * s['z_tau_xz'][j]:4.0f} um   "
              f"half-width a {1000 * s['a'][j]:4.0f} um")

x0 = lambda c: c["psi"] - c["psi"].min()


def bands(ax, c):
    p = c["d"]["params"]
    S = p["psi_c"] + p["psi_p"] + p["psi_u"]
    per = S + p["psi_r"]
    ph = 60.0 * a.pair - c["psi"].min()
    for k in range(-2, 6):
        s0 = ph + k * per
        ax.axvspan(s0, s0 + p["psi_c"] + p["psi_p"] + p["psi_u"], color=MUTED, alpha=0.08, lw=0)
    ax.set_xlim(0, 360)


# ---------------------------------------------------------------- forces
n = len(cases)
fig, axs = plt.subplots(1, n, figsize=(5.6 * n, 4.2), sharey=True, squeeze=False)
for ax, c in zip(axs[0], cases):
    bands(ax, c)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(x0(c), c["ff"][:, 0], color=UPPER, label="Upper follower (top input bodies)")
    ax.plot(x0(c), c["ff"][:, 1], color=LOWER, label="Lower follower (bottom input bodies)")
    ax.set(title=c["label"], xlabel="Cam angle over the last revolution [deg]")
    ax.set_xticks(np.arange(0, 361, 60))
axs[0, 0].set_ylabel(f"Vertical force on follower, per stackup [N]\n(+ up; jaw group {a.pair + 1}, {cs.n_stackups} stackups)")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9.5)
fig.text(0.5, -0.09, "+ : carried by the face-up cam below the follower.  − : by the face-down cam above it.  "
         "Grey bands: this jaw group is clamping, pushing or releasing (not reset).", fontsize=8.5, color=INK2, ha="center")
fig.savefig(out / "cam_force.png")
plt.close(fig)

# ---------------------------------------------------------------- stress
fig, axs = plt.subplots(1, n, figsize=(5.6 * n, 4.2), sharey=True, squeeze=False)
ymax = max(s["p0"].max() for c in cases for s in c["st"].values()) * 1.15
for ax, c in zip(axs[0], cases):
    bands(ax, c)
    for (name, lev, sgn) in SURFACES:
        s = c["st"][name]
        ax.plot(x0(c), s["p0"], color=UPPER if lev > 0 else LOWER, ls="-" if sgn > 0 else "--", label=name)
        j = int(np.argmax(s["p0"]))
        if s["p0"][j] > 0.5 * ymax:
            ax.annotate(f"{s['p0'][j]:.0f}", (x0(c)[j], s["p0"][j]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, color=INK2)
    ax.set(title=c["label"], xlabel="Cam angle over the last revolution [deg]", ylim=(0, ymax))
    ax.set_xticks(np.arange(0, 361, 60))
    if ax is axs[0, -1]:
        sec = ax.secondary_yaxis("right", functions=(lambda p: 0.25 * p, lambda t: 4 * t))
        sec.set_ylabel("Interlayer shear  τ_xz,max = 0.25 p_max [MPa]", color=INK2)
axs[0, 0].set_ylabel("Peak contact pressure p_max [MPa]")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9.5)
slope = f", track radius {cs.cam_radius:g} mm" if cs.cam_radius > 0 else ", ramp slope ignored"
pre = f", preload {cs.preload:g} N" if cs.preload > 0 else ""
fig.text(0.5, -0.15, f"Hertz line contact, steel bearing Ø{cs.D_bearing:g} × {cs.b_contact:g} mm on a flat printed face "
         f"(E = {cs.E_cam:g} MPa, ν = {cs.nu_cam:g}); {cs.n_bearings} bearing(s) per surface, {cs.n_stackups} stackups{slope}{pre}.\n"
         "τ_xz acts on planes parallel to the cam face (the print layers if printed face-up), at depth 0.5 a.",
         fontsize=8.5, color=INK2, ha="center")
fig.savefig(out / "cam_stress.png")
plt.close(fig)

summary = dict(cam_stack=cs.to_dict(), E_star=cs.E_star, cases=[])
for c in cases:
    e = dict(label=c["label"], follower_force_per_stackup=dict(
        upper=[float(c["ff"][:, 0].min()), float(c["ff"][:, 0].max())],
        lower=[float(c["ff"][:, 1].min()), float(c["ff"][:, 1].max())]), surfaces={})
    for name, s in c["st"].items():
        j = int(np.argmax(s["p0"]))
        e["surfaces"][name] = dict(F_bearing_max=float(s["Fn"].max()), p_max=float(s["p0"][j]),
                                   tau_xz_max=float(s["tau_xz"][j]), depth_tau_xz_um=float(1000 * s["z_tau_xz"][j]),
                                   half_width_um=float(1000 * s["a"][j]))
    summary["cases"].append(e)
json.dump(summary, open(out / "cam_loads.json", "w"), indent=1)
print("\nwrote", out / "cam_force.png", out / "cam_stress.png")
