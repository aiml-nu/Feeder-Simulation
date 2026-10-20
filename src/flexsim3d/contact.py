"""
Wire-to-jaw line contact with regularized Coulomb friction.

Each contact item pairs one wire node (a beam node of the wire, 6 DOF) with one
jaw (a rigid body, 6 DOF). The jaw's contact surface is its straight edge: a
line through the jaw-attached point p0 with tangent t (along the jaw length)
and inward normal n (pointing from the jaw toward the wire axis). Both rotate
with the jaw.

Normal law (penalty, per unit length):
    q_n = k_line * pen,   pen = r_wire - (c - p0).n     (smoothed with softplus)
    nodal force N = q_n * l_trib * window(s)
k_line is a *linearized line-contact stiffness* [N/mm per mm of contact
length]. For an Al wire on a steel edge, Hertz line contact gives roughly
1e4 N/mm^2 at the loads here (see feeder.hertz_line_stiffness); the jaw's
mounting compliance would lower it.

Friction (Coulomb with elastic stick and memory, i.e. a return mapping): each
pair stores the tangential force from the last converged step. The trial force
is that force (projected onto the current tangent plane) minus kt*l times the
slip increment since that step; it is capped at mu N by a smoothed max.
Sticking contacts therefore hold a sustained tangential load without creep,
and the jaws slip when the feed load exceeds the grip. Drivers must call
Model.commit() after every converged increment to update the history.

The force N n + f_t acts on the wire at the contact point c - r n; the
opposite force acts on the jaw at the same point. Moments use small-increment
rotation differences (rotations here are well below 1 degree).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .rotation import expm, skew


def _coulomb(fprev, ds_t, proj, N, d):
    """Coulomb friction with elastic stick (return mapping, smoothed).
    fprev: tangential force at the last converged step (global, history);
    proj(v): projection onto the current tangent space;
    trial f = proj(fprev) - kt*l*ds_t, limited to |f| <= mu N with a smooth
    max so the Jacobian stays continuous. N = 0 gives f = 0 (history cleared)."""
    ftr = proj(fprev) - d["kt"] * d["ltrib"] * ds_t
    a = jnp.sqrt(jnp.dot(ftr, ftr) + 1e-24)
    g = d["mu"] * N
    dl = d["fsmooth"]
    smax = 0.5 * (a + g + jnp.sqrt((a - g) ** 2 + dl * dl))
    return ftr * (g / smax)


def _kinematics(q, qp, d):
    """Contact geometry + forces for one wire-node/jaw pair."""
    qw, qJ = q[:6], q[6:]
    qwp, qJp = qp[:6], qp[6:]
    Xw, XJ, o, t0, n0 = d["Xw"], d["XJ"], d["o"], d["t0"], d["n0"]
    c = Xw + qw[:3]
    RJ = expm(qJ[3:])
    xJ = XJ + qJ[:3]
    p0 = xJ + RJ @ o
    t = RJ @ t0
    n = RJ @ n0
    rel = c - p0
    s = jnp.dot(rel, t)
    dist = jnp.dot(rel, n)
    pen = d["rw"] - dist
    delta = d["smooth"]
    # C1 smoothing: zero force out of contact, quadratic over [0, delta], linear beyond
    pen_s = jnp.where(pen <= 0.0, 0.0,
                      jnp.where(pen < delta, pen * pen / (2.0 * delta), pen - 0.5 * delta))
    ramp = d["ramp"]
    def smoothstep(x):
        x = jnp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    # C1 taper over `ramp` centred on each jaw end
    win = smoothstep(s / ramp + 0.5) * smoothstep((d["Lj"] - s) / ramp + 0.5)
    N = d["k"] * d["ltrib"] * pen_s * win
    pc = c - d["rw"] * n
    # slip increment (wire material point minus jaw material point)
    dw = (qw[:3] - qwp[:3]) + jnp.cross(qw[3:] - qwp[3:], -d["rw"] * n)
    dJ = (qJ[:3] - qJp[:3]) + jnp.cross(qJ[3:] - qJp[3:], pc - xJ)
    ds = dw - dJ
    ds_t = ds - jnp.dot(ds, n) * n
    ft = _coulomb(d["fprev"], ds_t, lambda v: v - jnp.dot(v, n) * n, N, d)
    return dict(c=c, n=n, t=t, s=s, pen=pen, N=N, ft=ft, pc=pc, xJ=xJ, ds_t=ds_t, win=win)


def _residual(q, qp, d):
    k = _kinematics(q, qp, d)
    Fw = k["N"] * k["n"] + k["ft"]
    Mw = jnp.cross(-d["rw"] * k["n"], k["ft"])
    FJ = -Fw
    MJ = jnp.cross(k["pc"] - k["xJ"], FJ)
    # internal-force convention: residual = -(external force on the DOF)
    return -jnp.concatenate([Fw, Mw, FJ, MJ])


_res_v = jax.jit(jax.vmap(_residual, in_axes=(0, 0, 0)))
_jac_v = jax.jit(jax.vmap(jax.jacfwd(_residual), in_axes=(0, 0, 0)))


@jax.jit
def _info_v(Q, Qp, D):
    def f(q, qp, d):
        k = _kinematics(q, qp, d)
        return jnp.concatenate([jnp.array([k["N"], k["s"], k["pen"], k["win"]]), k["ft"],
                                k["n"], k["ds_t"]])
    return jax.vmap(f)(Q, Qp, D)


class WireJawContact:
    """Vectorized set of wire-node / jaw contact pairs."""

    def __init__(self, model, wire_nodes, wire_ltrib, jaws, k_line, mu, rw,
                 kt=None, fsmooth=1e-3, smooth=2e-5, ramp=0.25):
        """jaws: list of dicts {ent, p0 (global point on edge, bottom end),
        t0 (unit, along jaw), n0 (unit inward normal), L (edge length)}."""
        rows, data = [], {k: [] for k in ["Xw", "XJ", "o", "t0", "n0", "ltrib", "Lj"]}
        self.pair_wire, self.pair_jaw = [], []
        for ji, J in enumerate(jaws):
            for wi, (wn, lt) in enumerate(zip(wire_nodes, wire_ltrib)):
                Xw = model.X[wn]
                # every wire node is paired with every jaw: the wire slides
                # axially, so nodes enter and leave the jaw span (window())
                rows.append(np.concatenate([6 * wn + np.arange(6), 6 * J["ent"] + np.arange(6)]))
                data["Xw"].append(Xw)
                data["XJ"].append(model.X[J["ent"]])
                data["o"].append(np.asarray(J["p0"]) - model.X[J["ent"]])
                data["t0"].append(J["t0"]); data["n0"].append(J["n0"])
                data["ltrib"].append(lt); data["Lj"].append(J["L"])
                self.pair_wire.append(wi); self.pair_jaw.append(ji)
        self.dofmap = np.array(rows, int)
        n = len(rows)
        self.D = {k: jnp.asarray(np.array(v, float)) for k, v in data.items()}
        for name, val in dict(k=k_line, mu=mu, rw=rw, smooth=smooth, ramp=ramp,
                              kt=k_line if kt is None else kt, fsmooth=fsmooth).items():
            self.D[name] = jnp.full((n,), float(val))
        self.D["fprev"] = jnp.zeros((n, 3))
        self.pair_wire = np.array(self.pair_wire)
        self.pair_jaw = np.array(self.pair_jaw)
        self.n_jaws = len(jaws)
        self.n_wire = len(wire_nodes)

    def set_param(self, name, value):
        self.D[name] = jnp.full(self.D[name].shape, float(value))

    def evaluate(self, Q, Qp):
        return _res_v(Q, Qp, self.D), _jac_v(Q, Qp, self.D)

    def commit(self, q, q_prev):
        """Store the converged friction forces as history for the next step."""
        self.D["fprev"] = jnp.asarray(self.info(q, q_prev)["ft"])

    def info(self, q, q_prev):
        """Per-pair arrays: N, s, pen, window, ft(3), n(3), ds_t(3)."""
        dm = self.dofmap
        A = np.asarray(_info_v(jnp.asarray(q[dm]), jnp.asarray(q_prev[dm]), self.D))
        return dict(N=A[:, 0], s=A[:, 1], pen=A[:, 2], win=A[:, 3], ft=A[:, 4:7],
                    n=A[:, 7:10], ds_t=A[:, 10:13])


# --------------------------------------------------------------------- tube
def _tube_kin(q, qp, d):
    """Wire node inside a rigid tube. The tube axis runs along -z from z_top
    with centreline offset  x_c(s) = ecc * (a2 s^2 + a3 s^3) * dir_xy,
    s = (z_top - z)/L in [0, 1] (straight tube: ecc = 0)."""
    c = d["Xw"] + q[:3]
    s = jnp.clip((d["ztop"] - c[2]) / d["L"], 0.0, 1.0)
    off = d["ecc"] * (d["a2"] * s * s + d["a3"] * s * s * s)
    centre = jnp.array([off * d["ex"], off * d["ey"]])
    dxy = c[:2] - centre
    r = jnp.sqrt(jnp.dot(dxy, dxy) + 1e-12)
    nout = jnp.array([dxy[0] / r, dxy[1] / r, 0.0])
    pen = r + d["rw"] - d["Ri"]
    delta = d["smooth"]
    pen_s = jnp.where(pen <= 0.0, 0.0,
                      jnp.where(pen < delta, pen * pen / (2.0 * delta), pen - 0.5 * delta))

    def smoothstep(x):
        x = jnp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    win = smoothstep((d["ztop"] - c[2]) / d["ramp"] + 0.5)      # 0 above the tube entry
    N = d["k"] * d["ltrib"] * pen_s * win
    # slip of the wire surface point at the wall (tube is fixed)
    ds = (q[:3] - qp[:3]) + jnp.cross(q[3:] - qp[3:], d["rw"] * nout)
    ds_t_full = ds - jnp.dot(ds, nout) * nout
    # axial-only option: friction resists sliding along the tube axis only; the
    # circumferential (helix-rolling) direction is frictionless. This is what the
    # force-transmission theory assumes and it keeps Newton well conditioned.
    ez = jnp.array([0.0, 0.0, 1.0])
    ds_t = jnp.where(d["axial"] > 0.5, ds[2] * ez, ds_t_full)
    proj = lambda v: jnp.where(d["axial"] > 0.5, v[2] * ez, v - jnp.dot(v, nout) * nout)
    ft = _coulomb(d["fprev"], ds_t, proj, N, d)
    return dict(N=N, nout=nout, ft=ft, pen=pen, ds_t=ds_t, r=r)


def _tube_res(q, qp, d):
    k = _tube_kin(q, qp, d)
    F = -k["N"] * k["nout"] + k["ft"]
    M = jnp.cross(d["rw"] * k["nout"], k["ft"])
    return -jnp.concatenate([F, M])


_tube_res_v = jax.jit(jax.vmap(_tube_res, in_axes=(0, 0, 0)))
_tube_jac_v = jax.jit(jax.vmap(jax.jacfwd(_tube_res), in_axes=(0, 0, 0)))


@jax.jit
def _tube_info_v(Q, Qp, D):
    def f(q, qp, d):
        k = _tube_kin(q, qp, d)
        return jnp.concatenate([jnp.array([k["N"], k["pen"], k["r"]]), k["ft"], k["ds_t"]])
    return jax.vmap(f)(Q, Qp, D)


class WireTubeContact:
    """Penalty + regularized-Coulomb contact of wire nodes with a rigid tube."""

    def __init__(self, model, wire_nodes, wire_ltrib, z_top, length, inner_radius, rw,
                 k=1e4, mu=0.3, kt=None, fsmooth=1e-3, smooth=2e-5, ramp=0.5,
                 ecc=0.0, shape=(0.0, 0.0), direction=(1.0, 0.0), axial_only=True):
        n = len(wire_nodes)
        self.dofmap = np.array([6 * w + np.arange(6) for w in wire_nodes], int)
        dr = np.asarray(direction, float) / np.linalg.norm(direction)
        self.D = dict(Xw=jnp.asarray(np.array([model.X[w] for w in wire_nodes])),
                      ltrib=jnp.asarray(np.asarray(wire_ltrib, float)))
        for name, val in dict(ztop=z_top, L=length, Ri=inner_radius, rw=rw, k=k, mu=mu,
                              kt=k if kt is None else kt, fsmooth=fsmooth,
                              smooth=smooth, ramp=ramp, ecc=ecc, a2=shape[0], a3=shape[1],
                              ex=dr[0], ey=dr[1], axial=float(axial_only)).items():
            self.D[name] = jnp.full((n,), float(val))
        self.D["fprev"] = jnp.zeros((n, 3))
        self.n = n

    def set_param(self, name, value):
        self.D[name] = jnp.full(self.D[name].shape, float(value))

    def evaluate(self, Q, Qp):
        return _tube_res_v(Q, Qp, self.D), _tube_jac_v(Q, Qp, self.D)

    def commit(self, q, q_prev):
        self.D["fprev"] = jnp.asarray(self.info(q, q_prev)["ft"])

    def info(self, q, q_prev):
        dm = self.dofmap
        A = np.asarray(_tube_info_v(jnp.asarray(q[dm]), jnp.asarray(q_prev[dm]), self.D))
        return dict(N=A[:, 0], pen=A[:, 1], r=A[:, 2], ft=A[:, 3:6], ds_t=A[:, 6:9])
