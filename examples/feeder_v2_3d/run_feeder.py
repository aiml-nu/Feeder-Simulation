"""
Run the Dauber V2 feeder simulation over several cam revolutions.

    python examples/feeder_v2_3d/run_feeder.py --revs 3 --dpsi 1 --out results/base.npz
    python examples/feeder_v2_3d/run_feeder.py --set E_flex=3000 k_feed=200 --out results/stiff.npz

Any FeederParams field can be overridden with --set name=value.
Output: an .npz with per-step histories (see flexsim3d.sim.Recorder) plus
state vectors every --frame-every steps for the 3D viewer, and the params as JSON.
"""
import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from flexsim3d.feeder import Feeder, FeederParams
from flexsim3d.sim import run_cycle, Recorder

ap = argparse.ArgumentParser()
ap.add_argument("--revs", type=float, default=3.0, help="cam revolutions (1 rev = 2 feed cycles)")
ap.add_argument("--dpsi", type=float, default=1.0, help="cam step [deg]")
ap.add_argument("--frame-every", type=int, default=2, help="store full state every N steps")
ap.add_argument("--eig-every", type=int, default=0, help="lowest tangent eigenvalue every N steps (slow)")
ap.add_argument("--set", nargs="*", default=[], help="FeederParams overrides name=value")
ap.add_argument("--out", default=str(ROOT / "examples/feeder_v2_3d/results/base.npz"))
a = ap.parse_args()

p = FeederParams()
for kv in a.set:
    k, v = kv.split("=")
    setattr(p, k, type(getattr(p, k))(v))

t0 = time.time()
F = Feeder(p)
print(f"model: {F.m.n_ent} entities, {F.m.ndof} DOF, {F.m.n_el} elements, {len(F.leaves)} leaves")
rec = Recorder(F, store_q_every=a.frame_every)
q, rec = run_cycle(F, psi_end=360.0 * a.revs, dpsi=a.dpsi, n_ramp=10, rec=rec, eig_every=a.eig_every)
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
arr = rec.arrays()
np.savez_compressed(out, params=json.dumps(p.to_dict()), wire_z=F.wire_z,
                    leaf_Pcr=F.leaf_Pcr, leaf_kind=F.leaf_kind, leaf_module=F.leaf_module,
                    leaf_level=F.leaf_level, leaf_arm=F.leaf_arm, leaf_i=F.leaf_i,
                    phases=np.array([mod["phase"] for mod in F.modules]), **arr)
print(f"saved {out}  ({time.time() - t0:.0f} s)")
