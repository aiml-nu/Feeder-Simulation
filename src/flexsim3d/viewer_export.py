"""
Export feeder results to compact JSON for the three.js viewer.

Per frame we store displacements (quantized int16) of every leaf's element
end points, the wire nodes, jaw/input poses, and contact line loads, so a
few hundred frames fit in a few MB.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from .feeder import Feeder, FeederParams


def _rot(psi):
    th = np.linalg.norm(psi)
    K = np.array([[0, -psi[2], psi[1]], [psi[2], 0, -psi[0]], [-psi[1], psi[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export(npz, out_json, frames=None, max_frames=200):
    d = dict(np.load(npz, allow_pickle=False))
    p = FeederParams(**json.loads(str(d["params"])))
    F = Feeder(p)
    m = F.m
    Q = d["q_frames"].astype(np.float64)
    psi = d["q_psi"]
    if frames is None:
        frames = np.arange(len(psi))
    frames = np.asarray(frames)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]

    # leaf polyline points: element start points + last end point
    ea, eb = m.ea, m.eb
    oa, ob = np.asarray(m.oa), np.asarray(m.ob)
    le = F.leaf_elems
    nL, nE = le.shape
    owners = np.zeros((nL, nE + 1), int)
    offs = np.zeros((nL, nE + 1, 3))
    owners[:, :nE] = ea[le]
    offs[:, :nE] = oa[le]
    owners[:, nE] = eb[le[:, -1]]
    offs[:, nE] = ob[le[:, -1]]
    ref = m.Xarr[owners] + offs                         # (nL, nE+1, 3)
    width_dir = np.array([np.asarray(m.E0)[g.elements[0]][:, 2] for g in F.leaves])

    wire = np.array(F.wire_nodes)
    wire_ref = m.Xarr[wire]
    jaws = [mod["jaw"] for mod in F.modules]
    inputs = [mod["inputs"][lev] for mod in F.modules for lev in (+1, -1)]
    bodies = jaws + inputs

    # map recorder rows to frames (q stored every k rows)
    every = int(round(len(d["psi"]) / max(len(psi), 1))) or 1
    row_of = {}
    for i, ps in enumerate(d["psi"]):
        row_of.setdefault(round(float(ps), 6), i)

    leaf_disp, wire_disp, body_pose, loads, feed, util, wz = [], [], [], [], [], [], []
    hw = p.wire_h
    for f in frames:
        q = Q[f].reshape(-1, 6)
        Rs = {e: _rot(q[e, 3:]) for e in set(owners.ravel()) if m.entity_kind[e] != "node"}
        pos = m.Xarr[owners] + q[owners, :3]
        for e, R in Rs.items():
            sel = owners == e
            pos[sel] += offs[sel] @ R.T
        node = np.array([m.entity_kind[e] == "node" for e in owners.ravel()]).reshape(owners.shape)
        pos[node] += offs[node]
        leaf_disp.append(pos - ref)
        wire_disp.append(q[wire, :3])
        body_pose.append(q[bodies])
        r = row_of.get(round(float(psi[f]), 6), min(f * every, len(d["psi"]) - 1))
        loads.append(d["N"][r] / hw)                       # (jaws, wire nodes) N/mm
        feed.append(d["feed_force"][r])
        util.append(d["leaf_util"][r])
        wz.append(d["wire_uz_bot"][r])

    leaf_disp = np.array(leaf_disp)
    scale = max(np.abs(leaf_disp).max(), np.abs(np.array(wire_disp)).max(), 1e-9) / 32000.0
    qmax = max(np.array(loads).max(), 1e-9)
    data = dict(
        params=p.to_dict(),
        n_frames=len(frames), psi=[float(psi[f]) for f in frames], feed=[float(x) for x in feed],
        nL=nL, nP=nE + 1,
        leaf_ref=_b64(ref.astype(np.float32)), leaf_wdir=_b64(width_dir.astype(np.float32)),
        leaf_kind=[1 if k == "angled" else 0 for k in F.leaf_kind],
        leaf_module=F.leaf_module.tolist(), leaf_level=F.leaf_level.tolist(),
        disp_scale=scale,
        leaf_disp=_b64(np.round(leaf_disp / scale).astype(np.int16)),
        wire_ref=_b64(wire_ref.astype(np.float32)),
        wire_disp=_b64(np.round(np.array(wire_disp) / scale).astype(np.int16)),
        body_ref=_b64(m.Xarr[bodies].astype(np.float32)),
        body_pose=_b64(np.array(body_pose).astype(np.float32)),
        n_jaws=len(jaws),
        jaw_alpha=[float(mod["alpha"]) for mod in F.modules],
        jaw_edge_r=p.d_wire / 2 + p.clearance,
        loads=_b64(np.round(np.array(loads) / qmax * 65535).astype(np.uint16)),
        load_max=float(qmax),
        util_peak=float(np.nanmax(d["leaf_util"])),
        leaf_util=_b64(np.clip(np.round(np.array(util) / 1.5 * 255), 0, 255).astype(np.uint8)),
        wire_travel=[float(x) for x in wz],
        # full-run history for the timeline chart
        hist_psi=[float(x) for x in d["psi"][d["ramp"] >= 1]],
        hist_feed=[float(x) for x in d["feed_force"][d["ramp"] >= 1]],
        jaw_half_depth=p.r_apex,
        input_geom=dict(s_out=p.s0 + p.L2 * np.cos(p.theta), d_input=p.d_input,
                        span=(p.N_leaves - 1) * p.pitch + p.t),
    )
    with open(out_json, "w") as fh:
        json.dump(data, fh)
    return data
