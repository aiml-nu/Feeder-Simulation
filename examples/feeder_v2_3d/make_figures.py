"""
Build slide figures (PNG) and the interactive 3D viewer (single HTML file)
from a results .npz.

    python examples/feeder_v2_3d/make_figures.py results/base.npz [--last-rev]
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
from flexsim3d import plots
from flexsim3d.viewer_export import export

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--outdir", default=None)
ap.add_argument("--viewer-frames", type=int, default=240)
ap.add_argument("--viewer-revs", type=float, default=0.0, help="show only the last N cam revolutions (0 = whole run)")
a = ap.parse_args()

npz = pathlib.Path(a.npz)
out = pathlib.Path(a.outdir) if a.outdir else npz.parent / (npz.stem + "_figs")
out.mkdir(parents=True, exist_ok=True)
d = plots.load(npz)
plots.fig_forces(d, out / "forces.png")
plots.fig_line_load(d, out / "line_load_map.png")
plots.fig_profiles(d, jaw=0, path=out / "line_load_profiles_jaw1.png")
plots.fig_buckling(d, module=0, path=out / "leaf_buckling_module1.png")
plots.fig_slip(d, out / "slip.png")
if "leaf_sig_wide" in d:
    plots.fig_stress_components(d, out / "stress_components.png")
plots.fig_tube(d, out / "tube_force.png")

# viewer: frames from the last N revolutions
qpsi = d["q_psi"]
sel = np.arange(len(qpsi)) if a.viewer_revs <= 0 else np.nonzero(qpsi >= qpsi.max() - 360.0 * a.viewer_revs)[0]
data = export(npz, out / "viewer_data.json", frames=sel, max_frames=a.viewer_frames)
# explicit UTF-8: Windows' default (cp1252) would turn "×" into "Ã—" when the template is read
tpl = (ROOT / "src/flexsim3d/viewer_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*__DATA__*/", (out / "viewer_data.json").read_text(encoding="utf-8"))
if any(m in html for m in ("Ã", "Â°", "â€")):
    raise SystemExit("viewer text looks double-encoded (UTF-8 read as cp1252); not writing it")
prm = json.loads(str(np.load(npz)["params"])) if "params" in np.load(npz).files else {}
tag = [f"gap {prm['clearance']:g} mm"] if prm.get("clearance") else []
tag += [f"mounts {prm['k_mount']:g} N/mm"] if prm.get("k_mount") else []
if tag:
    html = html.replace("<title>Dauber Feeder Viewer</title>", f"<title>Feeder Viewer, {', '.join(tag)}</title>")
# standalone file for local use (the published artifact gets its own document skeleton)
(out / "feeder_viewer.html").write_text(
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
    + html + "\n</body></html>\n", encoding="utf-8")
(out / "feeder_viewer_artifact.html").write_text(html, encoding="utf-8")
print("wrote", out)
