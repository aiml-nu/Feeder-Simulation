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
ap.add_argument("--viewer-frames", type=int, default=180)
ap.add_argument("--viewer-revs", type=float, default=1.0, help="show the last N cam revolutions in the viewer")
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

# viewer: frames from the last N revolutions
qpsi = d["q_psi"]
sel = np.nonzero(qpsi >= qpsi.max() - 360.0 * a.viewer_revs)[0]
data = export(npz, out / "viewer_data.json", frames=sel, max_frames=a.viewer_frames)
tpl = (ROOT / "src/flexsim3d/viewer_template.html").read_text()
html = tpl.replace("/*__DATA__*/", (out / "viewer_data.json").read_text())
# standalone file for local use (the published artifact gets its own document skeleton)
(out / "feeder_viewer.html").write_text(
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
    + html + "\n</body></html>\n", encoding="utf-8")
(out / "feeder_viewer_artifact.html").write_text(html, encoding="utf-8")
print("wrote", out)
