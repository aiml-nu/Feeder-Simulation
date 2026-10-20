"""
Export feeder results for the three.js viewer.

All per-frame arrays are quantized (int16/uint16), delta-encoded along the frame
axis, concatenated, gzip-compressed and base64-encoded into one JSON document,
so a full revolution at 8 elements per leaf (incl. per-leaf stresses) fits in
a single self-contained HTML file. The browser inflates it with
DecompressionStream('gzip').
"""
from __future__ import annotations

import base64
import gzip
import json

import numpy as np

from .feeder import Feeder, FeederParams


def _rot(psi):
    th = np.linalg.norm(psi)
    K = np.array([[0, -psi[2], psi[1]], [psi[2], 0, -psi[0]], [-psi[1], psi[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


class _Pack:
    def __init__(self):
        self.parts, self.meta, self.off = [], {}, 0

    def add(self, name, arr, kind, scale=None, delta=False):
        """kind: 'f32' raw float32; 'i16' signed quantized (value = int*scale);
        'u16' unsigned quantized in [0, scale]."""
        arr = np.asarray(arr)
        if arr.size == 0:
            arr = arr.astype(np.float32)
            scale = scale or 1.0
        if kind == "f32":
            b = arr.astype("<f4")
        elif kind == "i16":
            scale = scale or max(np.abs(arr).max() if arr.size else 0.0, 1e-12) / 32000.0
            b = np.round(arr / scale).astype("<i2")
        elif kind == "u16":
            scale = scale or max(arr.max() if arr.size else 0.0, 1e-12)
            b = np.round(np.clip(arr / scale, 0, 1) * 65535).astype("<u2")
        else:
            raise ValueError(kind)
        if delta:   # along axis 0 (frames); integer wrap-around is fine
            b = b.copy()
            b[1:] = (b[1:].astype(np.int64) - b[:-1].astype(np.int64)).astype(b.dtype)
        raw = np.ascontiguousarray(b).tobytes()
        pad = (-self.off) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.off += pad
        self.meta[name] = dict(kind=kind, shape=list(arr.shape), offset=self.off, nbytes=len(raw),
                               scale=float(scale) if scale is not None else None, delta=delta)
        self.parts.append(raw)
        self.off += len(raw)

    def blob(self):
        return base64.b64encode(gzip.compress(b"".join(self.parts), 9)).decode()


def export(npz, out_json, frames=None, max_frames=120):
    d = dict(np.load(npz, allow_pickle=False))
    p = FeederParams(**json.loads(str(d["params"])))
    F = Feeder(p)
    m = F.m
    Q = d["q_frames"].astype(np.float64)
    psi = d["q_psi"]
    frames = np.arange(len(psi)) if frames is None else np.asarray(frames)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    nF = len(frames)

    # ---- leaf geometry: points at element ends, owner entities, offsets
    ea, eb = m.ea, m.eb
    oa, ob = np.asarray(m.oa), np.asarray(m.ob)
    E0 = np.asarray(m.E0)
    le = F.leaf_elems
    nL, nE = le.shape
    owners = np.zeros((nL, nE + 1), int)
    offs = np.zeros((nL, nE + 1, 3))
    owners[:, :nE] = ea[le]
    offs[:, :nE] = oa[le]
    owners[:, nE] = eb[le[:, -1]]
    offs[:, nE] = ob[le[:, -1]]
    ref = m.Xarr[owners] + offs
    tan0 = E0[le][:, :, :, 0]                      # (nL, nE, 3) element axis at reference
    wdir = E0[le[:, 0]][:, :, 2]                   # (nL, 3) width direction
    tdir = E0[le[:, 0]][:, :, 1]                   # (nL, 3) thickness direction
    rigid = np.array([k != "node" for k in m.entity_kind])

    # ---- per-frame arrays
    sec = F.leaf_section
    rows_psi = d["psi"]
    row_of = {round(float(x), 6): i for i, x in enumerate(rows_psi)}
    wire = np.array(F.wire_nodes)
    jaws = [mod["jaw"] for mod in F.modules]
    inputs = [mod["inputs"][lev] for mod in F.modules for lev in (+1, -1)]
    bodies = jaws + inputs
    L_disp, L_rot, S_ax, S_tor, S_thin, S_wide = [], [], [], [], [], []
    W_disp, B_pose, J_load, T_load, W_N, feed, push, util, wz = [], [], [], [], [], [], [], [], []
    for f in frames:
        q = Q[f]
        Qe = q.reshape(-1, 6)
        pos = m.Xarr[owners] + Qe[owners, :3]
        for e in np.unique(owners[rigid[owners]]):
            sel = owners == e
            pos[sel] += offs[sel] @ _rot(Qe[e, 3:]).T
        pos[~rigid[owners]] += offs[~rigid[owners]]
        L_disp.append(pos - ref)
        L_rot.append(Qe[owners, 3:])
        ef = m.element_forces(q)[le]               # (nL, nE, 6): N, T, M2a, M3a, M2b, M3b
        S_ax.append(ef[:, :, 0] / sec.A)
        S_tor.append(ef[:, :, 1] * (2 * sec.c3) / sec.J)
        # internal bending moment at nodes: start = -Ma, end = +Mb, interior averaged
        for (ia, ib, I, c, out) in ((2, 4, sec.I2, sec.c2, S_wide), (3, 5, sec.I3, sec.c3, S_thin)):
            mn = np.zeros((nL, nE + 1))
            mn[:, 0] = -ef[:, 0, ia]
            mn[:, nE] = ef[:, -1, ib]
            mn[:, 1:nE] = 0.5 * (ef[:, :-1, ib] - ef[:, 1:, ia])
            out.append(mn * c / I)
        W_disp.append(Qe[wire, :3])
        B_pose.append(Qe[bodies])
        r = row_of.get(round(float(psi[f]), 6), min(len(rows_psi) - 1, int(f * len(rows_psi) / len(psi))))
        J_load.append(d["N"][r] / np.asarray(F.wire_ltrib)[F.jaw_wire_idx][None, :])
        T_load.append(d["tube_N"][r] / np.asarray(F.wire_ltrib)[F.tube_wire_idx] if "tube_N" in d
                      else np.zeros(len(F.tube_wire_idx)))
        W_N.append(d["wire_N"][r] if "wire_N" in d else np.zeros(len(F.wire_elems)))
        feed.append(d["feed_force"][r]); wz.append(d["wire_uz_bot"][r])
        push.append(d["jaw_push"][r] if "jaw_push" in d else np.nan)
        util.append(d["leaf_util"][r])

    pk = _Pack()
    pk.add("leaf_ref", ref, "f32")
    pk.add("leaf_tan0", tan0, "f32")
    pk.add("leaf_wdir", wdir, "f32")
    pk.add("leaf_tdir", tdir, "f32")
    pk.add("leaf_disp", np.array(L_disp), "i16", delta=True)
    pk.add("leaf_rot", np.array(L_rot), "i16", delta=True)
    for name, arr in (("s_axial", S_ax), ("s_torsion", S_tor), ("s_thin", S_thin), ("s_wide", S_wide)):
        pk.add(name, np.array(arr), "i16", delta=True)
    pk.add("leaf_util", np.clip(np.array(util), 0, 1.5), "u16", scale=1.5)
    pk.add("wire_ref", m.Xarr[wire], "f32")
    pk.add("wire_disp", np.array(W_disp), "i16", delta=True)
    pk.add("body_ref", m.Xarr[bodies], "f32")
    pk.add("body_pose", np.array(B_pose), "f32")
    pk.add("jaw_load", np.array(J_load), "u16")
    pk.add("tube_load", np.array(T_load), "u16")
    pk.add("wire_N", np.array(W_N), "i16")

    data = dict(
        params=p.to_dict(), n_frames=nF, psi=[float(psi[f]) for f in frames],
        feed=[float(x) for x in feed], push=[float(x) for x in push], wire_travel=[float(x) for x in wz],
        nL=nL, nE=nE, n_wire=len(wire), jaw_wire_idx=F.jaw_wire_idx.tolist(),
        tube_wire_idx=F.tube_wire_idx.tolist(), z_entry=float(getattr(F, "z_entry", 0.0)),
        wire_elem_z=[float(0.5 * (m.Xarr[m.ea[e]][2] + m.Xarr[m.eb[e]][2])) for e in F.wire_elems],
        leaf_kind=[1 if k == "angled" else 0 for k in F.leaf_kind],
        leaf_module=F.leaf_module.tolist(), leaf_level=F.leaf_level.tolist(),
        leaf_arm=F.leaf_arm.tolist(), leaf_i=F.leaf_i.tolist(),
        leaf_length=[float(g.length) for g in F.leaves], leaf_Pcr=F.leaf_Pcr.tolist(),
        n_jaws=len(jaws), jaw_alpha=[float(mod["alpha"]) for mod in F.modules],
        jaw_edge_r=p.d_wire / 2 + p.clearance,
        util_peak=float(np.nanmax(d["leaf_util"])),
        hist_psi=[float(x) for x in d["psi"][d["ramp"] >= 1]],
        hist_feed=[float(x) for x in d["feed_force"][d["ramp"] >= 1]],
        hist_push=[float(x) for x in d["jaw_push"][d["ramp"] >= 1]] if "jaw_push" in d else [],
        blobs=pk.meta, bin=pk.blob(),
    )
    with open(out_json, "w") as fh:
        json.dump(data, fh)
    return data
