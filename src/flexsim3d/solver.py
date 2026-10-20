"""
Quasi-static Newton-Raphson solver with prescribed DOFs and history.

Each call to `solve_step` finds equilibrium for one load/displacement
increment. `q_prev` is the last converged state; history-dependent terms
(friction) measure slip relative to it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla


class NewtonResult:
    def __init__(self, q, converged, iters, rnorm, fscale):
        self.q, self.converged, self.iters, self.rnorm, self.fscale = q, converged, iters, rnorm, fscale
        self.loose = False

    def __repr__(self):
        return f"NewtonResult(converged={self.converged}, iters={self.iters}, |r|={self.rnorm:.2e})"


def _line_search(model, q, r, free, dq, q_prev, max_tries=6):
    """Energy-type line search (Crisfield): find alpha with dq . r(q + alpha dq) ~ 0.

    The merit dq.r is the directional derivative of the total potential, so it
    is unit-consistent (forces x displacements + moments x rotations) unlike
    ||r||, which stalls near limit points where force and moment residuals
    trade off. Friction makes the system non-conservative, but dq.r is still
    a good 1D merit along the Newton direction.
    """
    def trial(a):
        qt = q.copy()
        qt[free] += a * dq
        rt, _ = model.assemble(qt, q_prev, with_K=False)
        return qt, rt, None, float(dq @ rt[free])

    s0 = float(dq @ r[free])
    qt, rt, Kt, s1 = trial(1.0)
    if s0 >= 0.0 or abs(s1) <= 0.5 * abs(s0):
        return qt, None, None, 1.0
    best = (abs(s1), qt, rt, Kt, 1.0)
    a_lo, s_lo, a_hi, s_hi = 0.0, s0, 1.0, s1
    if s1 < 0.0:  # still descending at full step: allow a modest extrapolation
        a_hi, (qt, rt, Kt, s_hi) = 2.0, trial(2.0)
        if abs(s_hi) < best[0]:
            best = (abs(s_hi), qt, rt, Kt, 2.0)
        if s_hi < 0.0:
            return best[1], best[2], best[3], best[4]
    for _ in range(max_tries):
        a = a_hi - s_hi * (a_hi - a_lo) / (s_hi - s_lo)       # regula falsi
        a = float(np.clip(a, a_lo + 0.05 * (a_hi - a_lo), a_hi - 0.05 * (a_hi - a_lo)))
        qt, rt, Kt, s = trial(a)
        if abs(s) < best[0]:
            best = (abs(s), qt, rt, Kt, a)
        if abs(s) <= 0.5 * abs(s0):
            break
        if (s < 0.0) == (s_lo < 0.0):
            a_lo, s_lo = a, s
        else:
            a_hi, s_hi = a, s
    return best[1], best[2], best[3], best[4]


def _linsolve(K, b):
    """Sparse LU tuned for our nearly-symmetric tangents: symmetric-mode
    ordering on A+A^T with diagonal pivoting is ~6x faster than the default.
    Falls back to the general solver if the result is inaccurate."""
    try:
        lu = spla.splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.01,
                       options=dict(SymmetricMode=True))
        x = lu.solve(b)
        if np.all(np.isfinite(x)) and np.linalg.norm(K @ x - b) <= 1e-8 * max(np.linalg.norm(b), 1e-300):
            return x
    except RuntimeError:
        pass
    return spla.spsolve(K.tocsc(), b)


def solve_step(model, q_guess, prescribed=None, q_prev=None, rtol=1e-8, atol=1e-8,
               maxit=40, verbose=False, max_step=None, loose_rtol=1e-6):
    """Newton solve. prescribed: dict dof -> value (on top of model.fixed).

    Convergence: ||r_free||_inf <= atol + rtol * fscale, where fscale is the
    largest reaction/internal force magnitude (a single fixed reference scale
    per solve, see roadmap lesson #1 from the 2D tool).
    max_step: optional cap on the largest translational Newton update (mm),
    a cheap trust region that helps when contact switches on/off.
    """
    n = model.ndof
    q = np.array(q_guess, float, copy=True)
    if q_prev is None:
        q_prev = q.copy()
    fixed = dict(model.fixed)
    if prescribed:
        fixed.update(prescribed)
    fdofs = np.array(sorted(fixed.keys()), int)
    q[fdofs] = np.array([fixed[d] for d in fdofs])
    mask = np.ones(n, bool)
    mask[fdofs] = False
    free = np.nonzero(mask)[0]

    r, K = model.assemble(q, q_prev)
    fscale = max(1.0, np.abs(r).max())
    rn = np.abs(r[free]).max()
    it = 0
    while it < maxit:
        if rn <= atol + rtol * fscale:
            return NewtonResult(q, True, it, rn, fscale)
        Kff = K[free][:, free]
        try:
            dq = _linsolve(Kff, -r[free])
        except Exception:
            return NewtonResult(q, False, it, rn, fscale)
        if not np.all(np.isfinite(dq)):
            return NewtonResult(q, False, it, rn, fscale)
        if max_step is not None:
            trans = dq.reshape(-1)  # crude: scale whole step by its largest entry
            mx = np.abs(trans).max()
            if mx > max_step:
                dq *= max_step / mx
        q, r, K, alpha = _line_search(model, q, r, free, dq, q_prev)
        if K is None:
            r, K = model.assemble(q, q_prev)
        fscale = max(fscale, np.abs(r).max())
        rn = np.abs(r[free]).max()
        it += 1
        if verbose:
            print(f"  it {it:2d}  |r|={rn:.3e}  alpha={alpha:.3g}  |dq|={np.abs(dq).max():.2e}")
        if np.abs(dq).max() * alpha < 1e-13:
            break
    res = NewtonResult(q, rn <= atol + rtol * fscale, it, rn, fscale)
    if not res.converged and loose_rtol and rn <= loose_rtol * fscale:
        res.converged, res.loose = True, True   # stalled at an engineering-negligible residual
    return res


def reactions(model, q, q_prev=None):
    """Residual at fixed DOFs = reaction forces (internal-force sign convention:
    r = f_int - f_ext, so the force the support applies is -r... we return r,
    i.e. the force the structure exerts on the support)."""
    r, _ = model.assemble(q, q if q_prev is None else q_prev, with_K=False)
    return r


def lowest_eigs(model, q, prescribed=None, k=3, q_prev=None):
    """Smallest eigenvalues of the free-DOF tangent stiffness. A value
    crossing zero signals a stability limit (buckling / bifurcation)."""
    n = model.ndof
    fixed = dict(model.fixed)
    if prescribed:
        fixed.update(prescribed)
    mask = np.ones(n, bool)
    mask[list(fixed.keys())] = False
    free = np.nonzero(mask)[0]
    _, K = model.assemble(q, q if q_prev is None else q_prev)
    Kff = K[free][:, free]
    Ks = 0.5 * (Kff + Kff.T)
    if Ks.shape[0] <= 3000:
        vals, vecs = np.linalg.eigh(Ks.toarray())
        return vals[:k], vecs[:, :k], free
    # large models: eigenvalues nearest a slightly negative shift; ask for
    # several so a stiff positive mode near zero cannot hide a negative one
    kk = max(k, 8)
    vals, vecs = spla.eigsh(Ks, k=kk, sigma=-1e-6 * abs(Ks.diagonal()).max(), which="LM")
    order = np.argsort(vals)[:k]
    return vals[order], vecs[:, order], free


def solve_stabilized(model, q, prescribed, q_prev, eta=1e-4, mask=None, maxit=40):
    """Newton solve with viscous stabilization r += c (q - q_prev), c_i = eta*|K_ii|
    (Abaqus *STABILIZE-style). Lets a quasi-static solve pass a snap (a local
    loss of stability) by spreading the jump over several increments.
    Returns (result, dissipated energy estimate sum c_i dq_i^2)."""
    _, K = model.assemble(q, q_prev)
    c = eta * np.abs(K.diagonal())
    if mask is not None:
        c = c * mask
    model.stab = c
    try:
        res = solve_step(model, q, prescribed, q_prev=q_prev, maxit=maxit)
    finally:
        model.stab = None
    return res, float(c @ (res.q - q_prev) ** 2)
