"""
Geometrically nonlinear planar beam element, after:

    Jirasek, M., La Malfa Ribolla, E., Horak, M. "Efficient finite difference
    formulation of a geometrically nonlinear beam element." International
    Journal for Numerical Methods in Engineering, 122(23):7013-7053, 2021.
    https://doi.org/10.1002/nme.6820

See derivations/references.md for citation details and design/architecture.md
for how this fits into the rest of the tool.

Equation numbers in comments refer to the paper. Variable names follow the
paper's notation where practical: X_ab, Z_ab, M_ab are the left-end
(support-side) force/moment components acting on the beam, expressed in a
corotational frame attached to the beam's left end; u, w, phi are centerline
displacement/rotation fields.

This is intentionally the *only* beam formulation implemented right now
(design/architecture.md originally sketched a formal BeamFormulation
interface so multiple formulations could be swapped in; that abstraction is
deferred until there's a second formulation to actually swap against -- no
point guessing its shape from one data point).

Validated (examples/pure_bending_cantilever) against the closed-form circular-
arc solution for pure end-moment bending, up to a full 90-degree tip
rotation: matches to ~3e-5 in displacement, ~6e-13 in rotation. Tangent
stiffness cross-checked against central-difference numerical differentiation
of the internal force.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Local (element-level) solve: shooting method + Newton-Raphson
# ---------------------------------------------------------------------------

def n_segments_from_thickness(L: float, t: float, spacing_factor: float = 2.0) -> int:
    """
    Choose the number of finite-difference integration segments as a multiplier
    on section thickness, e.g. spacing_factor=2 means "one integration point
    per two thickness-widths of arc length" -- finer than that doesn't buy much
    accuracy for a beam this slender, coarser starts to matter (see Jirasek
    et al. 2021, Table 1, for how error drops with segment count).
    """
    n = int(np.ceil(L / (spacing_factor * t)))
    return max(n, 4)  # floor so very short/thick beams still get a few points


def _primal_pass(X_ab: float, Z_ab: float, M_ab: float,
                  EA: float, EI: float, L: float, n_seg: int) -> dict:
    """
    Forward-marching finite-difference integration of the fundamental
    equations (37)-(39), using the explicit half-step scheme of eqs (59)-(65).
    Returns the full grid-point history (needed both for the Newton
    iteration's residual and for plotting the deformed shape).
    """
    dx = L / n_seg
    x = np.linspace(0.0, L, n_seg + 1)
    phi = np.zeros(n_seg + 1)
    u = np.zeros(n_seg + 1)
    w = np.zeros(n_seg + 1)
    phi_half = np.zeros(n_seg + 1)  # phi_half[i] = phi at (i-1/2), used for step i
    N_half = np.zeros(n_seg + 1)

    for i in range(1, n_seg + 1):
        M_im1 = -M_ab + X_ab * w[i - 1] - Z_ab * (x[i - 1] + u[i - 1])        # (59)
        ph = phi[i - 1] + M_im1 * dx / (2.0 * EI)                              # (60)
        Nh = -X_ab * np.cos(ph) + Z_ab * np.sin(ph)                            # (61)
        u[i] = u[i - 1] + ((1.0 + Nh / EA) * np.cos(ph) - 1.0) * dx            # (62)
        w[i] = w[i - 1] - (1.0 + Nh / EA) * np.sin(ph) * dx                    # (63)
        M_i = -M_ab + X_ab * w[i] - Z_ab * (x[i] + u[i])                       # (64)
        phi[i] = ph + M_i * dx / (2.0 * EI)                                    # (65)
        phi_half[i] = ph
        N_half[i] = Nh

    return dict(x=x, phi=phi, u=u, w=w, phi_half=phi_half, N_half=N_half)


def _linearized_pass(base: dict, dX_ab: float, dZ_ab: float, dM_ab: float,
                      X_ab: float, Z_ab: float,
                      EA: float, EI: float, L: float, n_seg: int) -> np.ndarray:
    """
    One column of the local Jacobian G = d(u_b,w_b,phi_b)/d(X_ab,Z_ab,M_ab),
    obtained by running the *linearized* recursion (eqs 73-77) for a unit
    perturbation of one of the three left-end force/moment components,
    holding the other two at zero. This is exact algorithmic differentiation
    of the primal recursion, not a finite-difference bump.
    """
    dx = L / n_seg
    x, phi, u, w = base["x"], base["phi"], base["u"], base["w"]
    phi_half, N_half = base["phi_half"], base["N_half"]

    dphi = np.zeros(n_seg + 1)
    du = np.zeros(n_seg + 1)
    dw = np.zeros(n_seg + 1)

    for i in range(1, n_seg + 1):
        dM_im1 = (-dM_ab + dX_ab * w[i - 1] + X_ab * dw[i - 1]
                  - dZ_ab * (x[i - 1] + u[i - 1]) - Z_ab * du[i - 1])           # (73)
        dph = dphi[i - 1] + dM_im1 * dx / (2.0 * EI)
        ph, Nh = phi_half[i], N_half[i]
        dNh = (-dX_ab * np.cos(ph) + X_ab * np.sin(ph) * dph
               + dZ_ab * np.sin(ph) + Z_ab * np.cos(ph) * dph)                 # (74)
        du[i] = (du[i - 1] + (dNh / EA) * np.cos(ph) * dx
                 - (1.0 + Nh / EA) * np.sin(ph) * dph * dx)                    # (75)
        dw[i] = (dw[i - 1] - (dNh / EA) * np.sin(ph) * dx
                 - (1.0 + Nh / EA) * np.cos(ph) * dph * dx)                    # (76)
        dM_i = (-dM_ab + dX_ab * w[i] + X_ab * dw[i]
                - dZ_ab * (x[i] + u[i]) - Z_ab * du[i])                        # (77)
        dphi[i] = dph + dM_i * dx / (2.0 * EI)

    return np.array([du[-1], dw[-1], dphi[-1]])


def local_jacobian(base: dict, X_ab: float, Z_ab: float,
                    EA: float, EI: float, L: float, n_seg: int) -> np.ndarray:
    """3x3 Jacobian G = d(u_b,w_b,phi_b) / d(X_ab,Z_ab,M_ab), columns in that order."""
    col_X = _linearized_pass(base, 1.0, 0.0, 0.0, X_ab, Z_ab, EA, EI, L, n_seg)
    col_Z = _linearized_pass(base, 0.0, 1.0, 0.0, X_ab, Z_ab, EA, EI, L, n_seg)
    col_M = _linearized_pass(base, 0.0, 0.0, 1.0, X_ab, Z_ab, EA, EI, L, n_seg)
    return np.column_stack([col_X, col_Z, col_M])


def _damped_newton(u_b_target: np.ndarray, EA: float, EI: float, L: float, n_seg: int,
                    f_ab_start: np.ndarray, tol: float, max_iter: int
                    ) -> tuple[bool, np.ndarray, dict, float]:
    """
    One damped Newton-Raphson attempt at eq (69), with a backtracking line
    search (halve the step until the residual actually decreases). Does not
    raise on failure -- returns (converged, f_ab, base, residual_norm) so the
    caller (`solve_local_end_forces`) can fall back to sub-stepping instead.
    """
    f_ab = np.array(f_ab_start, dtype=float)
    base = _primal_pass(f_ab[0], f_ab[1], f_ab[2], EA, EI, L, n_seg)
    g = np.array([base["u"][-1], base["w"][-1], base["phi"][-1]])
    residual = u_b_target - g
    res_norm = np.linalg.norm(residual)

    for _ in range(max_iter):
        if res_norm < tol:
            return True, f_ab, base, res_norm
        try:
            G = local_jacobian(base, f_ab[0], f_ab[1], EA, EI, L, n_seg)
            df = np.linalg.solve(G, residual)
        except np.linalg.LinAlgError:
            return False, f_ab, base, res_norm

        step = 1.0
        improved = False
        for _ in range(30):
            f_try = f_ab + step * df
            base_try = _primal_pass(f_try[0], f_try[1], f_try[2], EA, EI, L, n_seg)
            g_try = np.array([base_try["u"][-1], base_try["w"][-1], base_try["phi"][-1]])
            residual_try = u_b_target - g_try
            res_try = np.linalg.norm(residual_try)
            if res_try < res_norm:
                improved = True
                break
            if step < 1e-6:
                break
            step *= 0.5

        if not improved:
            return False, f_ab, base, res_norm

        f_ab, base, residual, res_norm = f_try, base_try, residual_try, res_try

    return False, f_ab, base, res_norm


def solve_local_end_forces(u_b_target: np.ndarray, EA: float, EI: float, L: float,
                            n_seg: int, f_ab_guess: np.ndarray | None = None,
                            tol: float = 1e-10, max_iter: int = 60,
                            max_substeps: int = 64) -> tuple[np.ndarray, dict]:
    """
    Solve eq (69) for the left-end forces f_ab=[X_ab,Z_ab,M_ab] that reproduce
    a prescribed right-end state u_b_target=[u_b,w_b,phi_b] (relative to the
    left end, in the left end's corotational frame).

    For realistic beam slenderness, the local Newton map is badly scaled (the
    moment-vs-rotation entries of the Jacobian can be many orders of
    magnitude larger than the force-vs-displacement ones), so a single
    Newton attempt -- even damped with a line search -- can stall well short
    of convergence if `u_b_target` is far from the current state (e.g. a
    single large jump in prescribed rotation). When that happens, fall back
    to sub-stepping: linearly ramp from the current state's own right-end
    values to `u_b_target` in progressively more increments (doubling each
    retry) and re-run Newton at each increment, warm-started from the
    previous one. This is the same fix used at the frame level for large
    load steps, just applied automatically at the element level so callers
    don't have to reason about how big a "safe" displacement increment is.
    """
    f_ab_start = np.zeros(3) if f_ab_guess is None else np.array(f_ab_guess, dtype=float)

    converged, f_ab, base, res_norm = _damped_newton(
        u_b_target, EA, EI, L, n_seg, f_ab_start, tol, max_iter
    )
    if converged:
        return f_ab, base

    base_start = _primal_pass(f_ab_start[0], f_ab_start[1], f_ab_start[2], EA, EI, L, n_seg)
    g_start = np.array([base_start["u"][-1], base_start["w"][-1], base_start["phi"][-1]])

    n_sub = 4
    while n_sub <= max_substeps:
        f_ab = f_ab_start
        ok = True
        for k in range(1, n_sub + 1):
            target_k = g_start + (u_b_target - g_start) * (k / n_sub)
            converged, f_ab, base, res_norm = _damped_newton(
                target_k, EA, EI, L, n_seg, f_ab, tol, max_iter
            )
            if not converged:
                ok = False
                break
        if ok:
            return f_ab, base
        n_sub *= 2

    raise RuntimeError(
        f"Local Newton iteration did not converge even after sub-stepping "
        f"into {max_substeps} increments (final residual norm {res_norm:.3e}); "
        f"check n_seg or whether u_b_target is physically reachable."
    )


# ---------------------------------------------------------------------------
# Element: global coordinates, assembly interface, tangent stiffness
# ---------------------------------------------------------------------------

@dataclass
class ElementResult:
    fG_ab: np.ndarray   # (3,) global force/moment applied to the beam at end A
    fG_ba: np.ndarray   # (3,) global force/moment applied to the beam at end B
    K: np.ndarray        # (6,6) tangent stiffness, DOF order [uA,wA,phiA,uB,wB,phiB]
    f_ab_local: np.ndarray  # converged local left-end force/moment (for warm-starting, diagnostics)
    shape_local: dict    # grid-point arrays (x,u,w,phi in the corotational frame) for plotting


class JirasekBeam:
    """
    One planar beam element between two joints (A, B), each with 3 global DOF
    [u, w, phi]. Geometry is fixed at construction (initial, undeformed
    coordinates); `compute()` takes the *current total* global displacement
    state of both end joints and returns internal forces + tangent stiffness.

    Section is a rectangular w(idth) x t(hickness) cross-section by default;
    pass EA/EI directly if a different section is needed.
    """

    def __init__(self, xa0: float, za0: float, xb0: float, zb0: float,
                 E: float, width: float, thickness: float,
                 n_seg: int | None = None, spacing_factor: float = 2.0):
        self.xa0, self.za0 = xa0, za0
        self.xb0, self.zb0 = xb0, zb0
        self.L0 = float(np.hypot(xb0 - xa0, zb0 - za0))
        self.alpha0 = float(np.arctan2(zb0 - za0, xb0 - xa0))
        self.E = E
        self.width = width
        self.thickness = thickness
        self.EA = E * width * thickness
        self.EI = E * width * thickness ** 3 / 12.0
        self.n_seg = n_seg or n_segments_from_thickness(self.L0, thickness, spacing_factor)
        self.f_ab = np.zeros(3)  # warm-start state, carried between calls

    def _T(self, phiG_a: float) -> np.ndarray:
        """eq (98): rotation from global to the left-end corotational frame."""
        a = self.alpha0 - phiG_a
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s, 0.0],
                          [-s, c, 0.0],
                          [0.0, 0.0, 1.0]])

    def _T_prime(self, phiG_a: float) -> np.ndarray:
        """eq (104): d T / d phiG_a."""
        a = self.alpha0 - phiG_a
        c, s = np.cos(a), np.sin(a)
        return np.array([[s, -c, 0.0],
                          [c, s, 0.0],
                          [0.0, 0.0, 0.0]])

    def _l(self, phiG_a: float) -> np.ndarray:
        """eq (98): rigid-body-motion offset term."""
        return self.L0 * np.array([np.cos(phiG_a) - 1.0, np.sin(phiG_a), 0.0])

    def _l_prime(self, phiG_a: float) -> np.ndarray:
        """eq (105): d l / d phiG_a."""
        return self.L0 * np.array([-np.sin(phiG_a), np.cos(phiG_a), 0.0])

    def compute(self, dofA: np.ndarray, dofB: np.ndarray) -> ElementResult:
        dofA = np.asarray(dofA, dtype=float)
        dofB = np.asarray(dofB, dtype=float)
        phiG_a = dofA[2]

        T = self._T(phiG_a)
        rel = dofB - dofA  # (duG_b - duG_a) as a 3-vector, third component = phiG_b - phiG_a
        u_b_target = T @ rel + self._l(phiG_a)

        f_ab, base = solve_local_end_forces(
            u_b_target, self.EA, self.EI, self.L0, self.n_seg, f_ab_guess=self.f_ab
        )
        self.f_ab = f_ab  # warm start for next call (e.g. next Newton iteration / load step)

        X_ab, Z_ab, M_ab = f_ab
        u_b, w_b = u_b_target[0], u_b_target[1]

        fG_ab = T.T @ f_ab                                                     # (97)
        M_ba = -M_ab + X_ab * w_b - Z_ab * (self.L0 + u_b)                      # (95)/(108)
        fG_ba = np.array([-fG_ab[0], -fG_ab[1], M_ba])                          # (93)-(94), (95)

        G = local_jacobian(base, X_ab, Z_ab, self.EA, self.EI, self.L0, self.n_seg)
        Tprime = self._T_prime(phiG_a)
        lprime = self._l_prime(phiG_a)
        vecC = Tprime @ rel + lprime

        # Need G^{-1} @ T and G^{-1} @ vecC (eq 107). Solve linear systems
        # rather than forming G^{-1} explicitly -- better conditioned, and
        # falls back to a least-squares solve if G is (numerically) singular
        # at this particular configuration, rather than crashing the whole
        # Newton iteration over what should only ever affect the tangent
        # stiffness accuracy, not the already-converged internal forces.
        rhs = np.column_stack([T, vecC])  # (3,4): solve both at once
        try:
            sol = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(G, rhs, rcond=None)
        Ginv_T, Ginv_vecC = sol[:, :3], sol[:, 3]

        A = T.T @ Ginv_T                                                        # (107), translational block
        b = T.T @ Ginv_vecC + Tprime.T @ f_ab                                   # (107), phiG_a-specific term

        K = np.zeros((6, 6))
        K[0:3, 0:3] = -A
        K[0:3, 2] += b
        K[0:3, 3:6] = A
        K[3, :] = -K[0, :]
        K[4, :] = -K[1, :]
        # sixth row/column via symmetry + eq (110)
        uGa, wGa = dofA[0], dofA[1]
        uGb, wGb = dofB[0], dofB[1]
        k16, k26, k36 = K[0, 5], K[1, 5], K[2, 5]
        k66 = ((self.L0 * np.sin(self.alpha0) + (wGb - wGa)) * k16
               - (self.L0 * np.cos(self.alpha0) + (uGb - uGa)) * k26
               - k36)
        K[5, 0:5] = K[0:5, 5]
        K[5, 5] = k66

        return ElementResult(fG_ab=fG_ab, fG_ba=fG_ba, K=K, f_ab_local=f_ab,
                              shape_local=base)

    def deformed_shape_global(self, dofA: np.ndarray, result: ElementResult) -> np.ndarray:
        """
        Global (x, z) coordinates of every integration grid point in the
        current deformed configuration, straight from the shooting method's
        own grid -- no separate shape-function interpolation needed.
        Returns an array of shape (n_seg+1, 2).
        """
        dofA = np.asarray(dofA, dtype=float)
        uGa, wGa, phiGa = dofA
        # Local -> global rotation block. Reuse T() rather than re-deriving the
        # angle/sign convention separately: fG_ab = T.T @ f_ab already fixes
        # "T.T rotates local vectors into global", so the same block rotates
        # local *positions* into global too.
        T = self._T(phiGa)
        Rg = T.T[0:2, 0:2]
        x_local = result.shape_local["x"] + result.shape_local["u"]
        z_local = result.shape_local["w"]
        local_xy = np.vstack([x_local, z_local])  # (2, n_seg+1)
        global_xy = Rg @ local_xy + np.array([[self.xa0 + uGa], [self.za0 + wGa]])
        return global_xy.T
