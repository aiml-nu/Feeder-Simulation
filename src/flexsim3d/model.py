"""
Model container + global assembly.

Everything with DOFs is an *entity* with 6 DOFs  q = [ux, uy, uz, psi_x, psi_y, psi_z]
about a reference point X: beam nodes and rigid bodies are treated identically.
A rigid body is simply an entity that beam elements attach to at a non-zero
offset (see beam.py), so no constraint equations are needed for beam-to-rigid
connections. "Ground" is an entity whose 6 DOFs are all fixed.

Extra physics (contact, friction, ...) plug in as *terms*: objects that expose
  - dofmap: (n_items, m) int array of global DOF indices
  - evaluate(Qloc, Qloc_prev) -> (r (n_items, m), J (n_items, m, m))
contributing r to the global residual (internal-force convention:
residual = internal forces - external forces) and J to the tangent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .beam import N_PARAM, Section, element_energy, element_section_forces, frame_from_axis

jax.config.update("jax_enable_x64", True)


def _energy_flat(qab, Xa, Xb, oa, ob, E0, prm):
    return element_energy(qab[:6], qab[6:], Xa, Xb, oa, ob, E0, prm)


_grad = jax.grad(_energy_flat)
_hess = jax.hessian(_energy_flat)


@jax.jit
def _beam_grad_hess(Qab, Xa, Xb, oa, ob, E0, prm):
    g = jax.vmap(_grad)(Qab, Xa, Xb, oa, ob, E0, prm)
    H = jax.vmap(_hess)(Qab, Xa, Xb, oa, ob, E0, prm)
    return g, H


@jax.jit
def _beam_grad(Qab, Xa, Xb, oa, ob, E0, prm):
    return jax.vmap(_grad)(Qab, Xa, Xb, oa, ob, E0, prm)


@jax.jit
def _beam_energy(Qab, Xa, Xb, oa, ob, E0, prm):
    return jax.vmap(_energy_flat)(Qab, Xa, Xb, oa, ob, E0, prm)


@jax.jit
def _beam_forces(Qab, Xa, Xb, oa, ob, E0, prm):
    f = lambda q, *a: element_section_forces(q[:6], q[6:], *a)
    return jax.vmap(f)(Qab, Xa, Xb, oa, ob, E0, prm)


@dataclass
class BeamGroup:
    """Bookkeeping for one physical member (e.g. one leaf) made of n elements."""
    name: str
    elements: np.ndarray       # element indices
    nodes: np.ndarray          # entity indices along the member, incl. both ends' owners
    section: Section
    length: float
    tags: dict = field(default_factory=dict)


class Model:
    def __init__(self):
        self.X = []                 # entity reference positions
        self.entity_kind = []       # "node" | "rigid" | "ground"
        self.entity_name = []
        # beam element arrays (lists until finalize)
        self._ea, self._eb, self._oa, self._ob, self._E0, self._prm = [], [], [], [], [], []
        self.groups: list[BeamGroup] = []
        self.terms = []
        self.fixed = {}             # dof -> value
        self.springs = []           # (dof, k)
        self._finalized = False

    # ------------------------------------------------------------------ build
    def add_entity(self, X, kind="node", name=""):
        self.X.append(np.asarray(X, float))
        self.entity_kind.append(kind)
        self.entity_name.append(name)
        return len(self.X) - 1

    def add_rigid(self, X, name=""):
        return self.add_entity(X, "rigid", name)

    def add_ground(self, name="ground"):
        g = self.add_entity(np.zeros(3), "ground", name)
        for k in range(6):
            self.fixed[6 * g + k] = 0.0
        return g

    def dof(self, ent, k):
        return 6 * ent + k

    def fix(self, ent, comps=range(6), value=0.0):
        for k in comps:
            self.fixed[6 * ent + k] = value

    def add_spring(self, ent, comp, k):
        self.springs.append((6 * ent + comp, float(k)))

    def attach_point(self, ent, P):
        """Offset of global point P from entity ent's reference point."""
        return np.asarray(P, float) - self.X[ent]

    def add_member(self, a, Pa, b, Pb, section: Section, e2_hint, n_el,
                   imperfection=None, name="", tags=None):
        """Straight member from point Pa (rigidly attached to entity a) to Pb
        (attached to entity b), split into n_el elements with n_el-1 new nodes.

        imperfection: optional (amplitude, shape(s)) displacing interior nodes
        along e2 by amplitude*shape(s), s in [0, 1]. Elements are stress-free in
        the imperfect geometry.
        """
        Pa, Pb = np.asarray(Pa, float), np.asarray(Pb, float)
        L = np.linalg.norm(Pb - Pa)
        F = frame_from_axis(Pb - Pa, e2_hint)
        s = np.linspace(0.0, 1.0, n_el + 1)
        pts = Pa[None, :] + s[:, None] * (Pb - Pa)[None, :]
        if imperfection is not None:
            amp, shape = imperfection
            vals = np.array([shape(si) for si in s], float)
            if vals.ndim == 2:      # shape returns a global 3-vector offset per point
                pts = pts + amp * vals
            else:                   # scalar shape along the member's e2 axis
                pts = pts + amp * vals[:, None] * F[:, 1][None, :]
        owners = [a]
        offs = [self.attach_point(a, pts[0])]
        for i in range(1, n_el):
            nid = self.add_entity(pts[i], "node", f"{name}/n{i}")
            owners.append(nid)
            offs.append(np.zeros(3))
        owners.append(b)
        offs.append(self.attach_point(b, pts[-1]))
        elems = []
        for i in range(n_el):
            Ei = frame_from_axis(pts[i + 1] - pts[i], F[:, 1])
            L0 = np.linalg.norm(pts[i + 1] - pts[i])
            self._ea.append(owners[i]); self._eb.append(owners[i + 1])
            self._oa.append(offs[i] if i == 0 else np.zeros(3))
            self._ob.append(offs[i + 1] if i == n_el - 1 else np.zeros(3))
            self._E0.append(Ei)
            self._prm.append(section.params(L0))
            elems.append(len(self._ea) - 1)
        g = BeamGroup(name, np.array(elems), np.array(owners), section, L, tags or {})
        self.groups.append(g)
        return g

    def add_term(self, term):
        self.terms.append(term)
        return term

    # --------------------------------------------------------------- finalize
    def finalize(self):
        self.n_ent = len(self.X)
        self.ndof = 6 * self.n_ent
        self.Xarr = np.array(self.X)
        self.ea = np.array(self._ea, int)
        self.eb = np.array(self._eb, int)
        self.oa = jnp.asarray(np.array(self._oa))
        self.ob = jnp.asarray(np.array(self._ob))
        self.E0 = jnp.asarray(np.array(self._E0))
        self.prm = jnp.asarray(np.array(self._prm).reshape(-1, N_PARAM))
        self.Xa = jnp.asarray(self.Xarr[self.ea])
        self.Xb = jnp.asarray(self.Xarr[self.eb])
        self.edofs = np.concatenate([6 * self.ea[:, None] + np.arange(6)[None, :],
                                     6 * self.eb[:, None] + np.arange(6)[None, :]], axis=1)
        self.n_el = len(self.ea)
        self._finalized = True
        return self

    # --------------------------------------------------------------- assembly
    def _gather(self, q, dofmap):
        return q[dofmap]

    def _build_pattern(self):
        """Fixed sparsity pattern: map every COO contribution to its CSR slot once,
        so each assembly is a single bincount."""
        n = self.ndof
        rows = [np.repeat(self.edofs, 12, axis=1).ravel()]
        cols = [np.tile(self.edofs, (1, 12)).ravel()]
        for dof, k in self.springs:
            rows.append(np.array([dof])); cols.append(np.array([dof]))
        for term in self.terms:
            dm = term.dofmap
            mm = dm.shape[1]
            rows.append(np.repeat(dm, mm, axis=1).ravel())
            cols.append(np.tile(dm, (1, mm)).ravel())
        rows = np.concatenate(rows).astype(np.int64)
        cols = np.concatenate(cols).astype(np.int64)
        keys = rows * n + cols
        ukeys, inv = np.unique(keys, return_inverse=True)
        urow, ucol = ukeys // n, ukeys % n
        indptr = np.zeros(n + 1, np.int64)
        np.add.at(indptr, urow + 1, 1)
        self._pat = dict(inv=inv, nnz=len(ukeys), indices=ucol, indptr=np.cumsum(indptr),
                         key=self._pattern_key())

    def _pattern_key(self):
        return (tuple(self.springs),
                tuple(hash(np.ascontiguousarray(t.dofmap).tobytes()) for t in self.terms))

    def assemble(self, q, q_prev, with_K=True):
        """Return residual r (internal - external) and sparse (CSR) tangent K."""
        n = self.ndof
        Qab = jnp.asarray(q[self.edofs])
        if with_K:
            g, H = _beam_grad_hess(Qab, self.Xa, self.Xb, self.oa, self.ob, self.E0, self.prm)
            vals = [np.asarray(H).ravel()]
        else:
            g = _beam_grad(Qab, self.Xa, self.Xb, self.oa, self.ob, self.E0, self.prm)
        r = np.bincount(self.edofs.ravel(), weights=np.asarray(g).ravel(), minlength=n)
        for dof, k in self.springs:
            r[dof] += k * q[dof]
            if with_K:
                vals.append(np.array([k]))
        for term in self.terms:
            dm = term.dofmap
            rt, Jt = term.evaluate(jnp.asarray(q[dm]), jnp.asarray(q_prev[dm]))
            r += np.bincount(dm.ravel(), weights=np.asarray(rt).ravel(), minlength=n)
            if with_K:
                vals.append(np.asarray(Jt).ravel())
        stab = getattr(self, "stab", None)
        base = getattr(self, "stab_base", None)
        if base is not None:
            stab = base if stab is None else stab + base
        if stab is not None:     # viscous stabilization (only active when the driver sets it)
            r += stab * (q - q_prev)
        if not with_K:
            return r, None
        p = getattr(self, "_pat", None)
        if p is None or p["key"] != self._pattern_key():
            self._build_pattern()
            p = self._pat
        data = np.bincount(p["inv"], weights=np.concatenate(vals), minlength=p["nnz"])
        K = sp.csr_matrix((data, p["indices"], p["indptr"]), shape=(n, n))
        if stab is not None:
            K = K + sp.diags(stab, format="csr")
        return r, K

    def internal_force(self, q, terms=()):
        """Elastic internal force dU/dq from the beams, grounded springs and the
        given history-free terms (e.g. bushings). At equilibrium this equals the
        external force, so at a prescribed DOF it is the reaction."""
        Qab = jnp.asarray(q[self.edofs])
        g = _beam_grad(Qab, self.Xa, self.Xb, self.oa, self.ob, self.E0, self.prm)
        r = np.bincount(self.edofs.ravel(), weights=np.asarray(g, dtype=float).ravel(), minlength=self.ndof)
        for dof, k in self.springs:
            r[dof] += k * q[dof]
        for term in terms:
            dm = term.dofmap
            rt, _ = term.evaluate(jnp.asarray(q[dm]), jnp.asarray(q[dm]))
            r += np.bincount(dm.ravel(), weights=np.asarray(rt).ravel(), minlength=self.ndof)
        return r

    def commit(self, q, q_prev):
        """Call after every converged increment that becomes the next start
        state: history-dependent terms (friction) store their state."""
        for t in self.terms:
            if hasattr(t, "commit"):
                t.commit(q, q_prev)

    # ------------------------------------------------------------ post-proc
    def element_forces(self, q):
        Qab = jnp.asarray(q[self.edofs])
        return np.asarray(_beam_forces(Qab, self.Xa, self.Xb, self.oa, self.ob, self.E0, self.prm))

    def strain_energy(self, q):
        Qab = jnp.asarray(q[self.edofs])
        return float(np.sum(_beam_energy(Qab, self.Xa, self.Xb, self.oa, self.ob, self.E0, self.prm)))

    def node_positions(self, q):
        Q = q.reshape(-1, 6)
        return self.Xarr + Q[:, :3]
