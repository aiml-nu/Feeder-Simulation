"""
Quasi-static cam-cycle driver + result recording for the feeder.
"""
from __future__ import annotations

import time

import numpy as np

from .solver import solve_step, lowest_eigs


class Recorder:
    def __init__(self, F, store_q_every=1):
        self.F = F
        self.every = store_q_every
        self.rows = []          # dict per step
        self.q_frames, self.q_psi = [], []
        self.stab_log = []      # (psi, energy dissipated by stabilization) for stabilized steps

    def record(self, psi, q, q_prev, res, ramp, eig=None):
        F, m = self.F, self.F.m
        info = F.contact.info(q, q_prev)
        nj, nw = F.contact.n_jaws, F.contact.n_wire
        N = info["N"].reshape(nj, nw)
        ftz = info["ft"][:, 2].reshape(nj, nw)
        slip = np.linalg.norm(info["ds_t"], axis=1).reshape(nj, nw)
        ef = m.element_forces(q)
        s = F.leaf_section
        Nel = ef[:, 0]
        sig = (np.abs(Nel) / s.A
               + np.maximum(np.abs(ef[:, 3]), np.abs(ef[:, 5])) * s.c3 / s.I3
               + np.maximum(np.abs(ef[:, 2]), np.abs(ef[:, 4])) * s.c2 / s.I2)
        le = F.leaf_elems
        leaf_N = Nel[le].mean(axis=1)                 # axial force per leaf
        leaf_sig = sig[le].max(axis=1)
        Q = q.reshape(-1, 6)
        row = dict(
            psi=psi, ramp=ramp, iters=res.iters, converged=res.converged, loose=res.loose, rnorm=res.rnorm, stabilized=getattr(res, 'stabilized', False),
            wire_uz_bot=Q[F.wire_bot, 2], wire_uz_top=Q[F.wire_top, 2],
            feed_force=-F.p.k_feed * Q[F.wire_bot, 2],          # N, upward on wire
            N=N, ftz=ftz, slip=slip,
            s_on_jaw=info["s"].reshape(nj, nw),
            jaw_q=np.array([Q[mod["jaw"]] for mod in F.modules]),
            inp_q=np.array([[Q[mod["inputs"][+1]], Q[mod["inputs"][-1]]] for mod in F.modules]),
            leaf_N=leaf_N, leaf_util=np.maximum(-leaf_N, 0) / F.leaf_Pcr, leaf_sig=leaf_sig,
            eig=eig,
        )
        self.rows.append(row)
        if len(self.rows) % self.every == 0:
            self.q_frames.append(q.astype(np.float32))
            self.q_psi.append(psi)

    def arrays(self):
        out = {}
        for k in self.rows[0]:
            vals = [r[k] for r in self.rows]
            if k == "eig":
                vals = [np.nan if v is None else v for v in vals]
            out[k] = np.array(vals)
        out["q_frames"] = np.array(self.q_frames)
        out["q_psi"] = np.array(self.q_psi)
        out["stab_log"] = np.array(self.stab_log).reshape(-1, 2)
        return out


def run_cycle(F, psi_end, dpsi=1.0, n_ramp=10, psi_start=0.0, rec=None, verbose=True,
              eig_every=0, q0=None, max_depth=4, predictor=True, stab_eta=1e-3):
    """Ramp inputs from the as-printed neutral to their psi_start positions,
    then rotate the cam to psi_end in increments of dpsi (adaptive halving on
    non-convergence). Returns (q, recorder)."""
    m = F.m
    rec = rec or Recorder(F)
    q = np.zeros(m.ndof) if q0 is None else q0.copy()
    t0 = time.time()

    last = {"dq": None, "dpsi": None}
    node_mask = np.repeat(np.array([k == "node" for k in m.entity_kind], float), 6)
    for dof in m.fixed:
        node_mask[dof] = 0.0
    stab_log = rec.stab_log

    def advance(q, psi_a, psi_b, ramp_a, ramp_b, depth=0):
        pres = F.prescribed(psi_b, ramp_b)
        guess = q
        if predictor and last["dq"] is not None and depth == 0:
            guess = q + last["dq"]          # secant predictor from previous increment
        res = solve_step(m, guess, pres, q_prev=q, maxit=25)
        if not res.converged and guess is not q:
            res = solve_step(m, q, pres, q_prev=q, maxit=25)
        if res.converged:
            return res
        if depth >= max_depth and stab_eta and getattr(m, "stab", None) is None:
            # local instability (a leaf snapping between buckled shapes): retry this
            # increment with viscous stabilization on beam-node DOFs, c_i = eta*K_ii
            _, K = m.assemble(q, q)
            c = stab_eta * np.abs(K.diagonal()) * node_mask
            m.stab = c
            try:
                sub = 8
                qq = q
                for i in range(1, sub + 1):
                    pa = psi_a + (psi_b - psi_a) * (i - 1) / sub
                    pb = psi_a + (psi_b - psi_a) * i / sub
                    rr = solve_step(m, qq, F.prescribed(pb, ramp_a + (ramp_b - ramp_a) * i / sub),
                                    q_prev=qq, maxit=40)
                    if not rr.converged:
                        break
                    stab_log.append((pb, float(c @ (rr.q - qq) ** 2)))
                    qq = rr.q
            finally:
                m.stab = None
            if rr.converged:
                rr.stabilized = True
                return rr
        if depth >= max_depth:
            np.savez("failed_state.npz", q=q, psi_a=psi_a, psi_b=psi_b, ramp_a=ramp_a, ramp_b=ramp_b)
            raise RuntimeError(f"no convergence at psi={psi_b:.4f} ramp={ramp_b:.3f}: {res}")
        pm, rm = 0.5 * (psi_a + psi_b), 0.5 * (ramp_a + ramp_b)
        r1 = advance(q, psi_a, pm, ramp_a, rm, depth + 1)
        return advance(r1.q, pm, psi_b, rm, ramp_b, depth + 1)

    def step(q, psi_a, psi_b, ra, rb):
        res = advance(q, psi_a, psi_b, ra, rb)
        last["dq"] = res.q - q
        eig = None
        if eig_every and len(rec.rows) % eig_every == 0:
            eig = lowest_eigs(m, res.q, F.prescribed(psi_b, rb), k=1, q_prev=q)[0][0]
        rec.record(psi_b, res.q, q, res, rb, eig)
        return res

    for i in range(1, n_ramp + 1) if q0 is None else []:
        res = step(q, psi_start, psi_start, (i - 1) / n_ramp, i / n_ramp)
        q = res.q
        if verbose:
            print(f"ramp {i}/{n_ramp}: iters {res.iters}, feed {rec.rows[-1]['feed_force']:.2f} N")
    psis = np.arange(psi_start + dpsi, psi_end + 1e-9, dpsi)
    for j, psi in enumerate(psis):
        res = step(q, psi - dpsi, psi, 1.0, 1.0)
        q = res.q
        if verbose and (j % 10 == 0 or j == len(psis) - 1):
            r = rec.rows[-1]
            print(f"psi {psi:7.2f}  it {res.iters:2d}  feed {r['feed_force']:7.2f} N  "
                  f"wire dz {r['wire_uz_bot']:+.4f}  sumN {r['N'].sum():8.1f}  "
                  f"max util {r['leaf_util'].max():.2f}  t={time.time() - t0:.0f}s")
    return q, rec
