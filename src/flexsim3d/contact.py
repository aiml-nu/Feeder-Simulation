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

Friction (regularized Coulomb, history-dependent): slip increment of the wire
surface material point relative to the jaw material point since the last
converged step, projected onto the contact tangent plane:
    f_t = -mu N  ds_t / sqrt(|ds_t|^2 + eps^2)
|ds_t| << eps behaves as stick (stiffness mu N / eps); |ds_t| >> eps is full
slip at mu N. This is what lets the jaws slip when the feed load exceeds the
grip, rather than a hard stick/slip switch that chatters.

The force N n + f_t acts on the wire at the contact point c - r n; the
opposite force acts on the jaw at the same point. Moments use small-increment
rotation differences (rotations here are well below 1 degree).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .rotation import expm, skew


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
    nrm = jnp.sqrt(jnp.dot(ds_t, ds_t) + d["eps"] ** 2)
    ft = -d["mu"] * N * ds_t / nrm
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
                 eps=1e-4, smooth=2e-5, ramp=0.25):
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
        for name, val in dict(k=k_line, mu=mu, rw=rw, eps=eps, smooth=smooth, ramp=ramp).items():
            self.D[name] = jnp.full((n,), float(val))
        self.pair_wire = np.array(self.pair_wire)
        self.pair_jaw = np.array(self.pair_jaw)
        self.n_jaws = len(jaws)
        self.n_wire = len(wire_nodes)

    def set_param(self, name, value):
        self.D[name] = jnp.full(self.D[name].shape, float(value))

    def evaluate(self, Q, Qp):
        return _res_v(Q, Qp, self.D), _jac_v(Q, Qp, self.D)

    def info(self, q, q_prev):
        """Per-pair arrays: N, s, pen, window, ft(3), n(3), ds_t(3)."""
        dm = self.dofmap
        A = np.asarray(_info_v(jnp.asarray(q[dm]), jnp.asarray(q_prev[dm]), self.D))
        return dict(N=A[:, 0], s=A[:, 1], pen=A[:, 2], win=A[:, 3], ft=A[:, 4:7],
                    n=A[:, 7:10], ds_t=A[:, 10:13])
