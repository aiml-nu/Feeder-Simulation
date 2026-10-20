"""
Minimal 2D nonlinear frame solver: joints (3 global DOF each) connected by
beam elements (currently only JirasekBeam, design/architecture.md's
"geometrically nonlinear track"), solved by incremental load-stepping with
Newton-Raphson equilibrium iteration.

This deliberately stays small and single-purpose for now -- it is *not* the
generalized rigid-body/contact `Structure` described in design/architecture.md
(that will eventually need to absorb this, plus RigidContact, plus multiple
formulations). It exists to drive the validation examples in `examples/`
(euler_buckling_cantilever, williams_toggle, pure_bending_cantilever) with the
least amount of machinery needed to do that correctly.

Boundary conditions are per-DOF, either:
  - prescribed displacement (the DOF's `u` is held at a target value, ramped
    linearly over the load steps; the corresponding reaction force is
    whatever the solve produces), or
  - prescribed force (the DOF's external force is held at a target value,
    ramped the same way; the corresponding displacement is whatever the
    solve produces).
This is the same "every DOF is either displacement- or force-controlled"
idea used by the linear mixed-BC solve in `derivations/rigid_body_kinematics.md`,
just resolved by Newton iteration each step instead of one linear solve,
since the element relation is no longer linear.

Validated (examples/pure_bending_cantilever) against the closed-form circular-
arc solution for a cantilever under a pure end moment, up to a full 90-degree
tip rotation: matches to ~3e-5 in displacement, ~6e-13 in rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .rigid import anchor_dof, anchor_jacobian, anchor_jacobian_prime
from .contact import FlatSurfaceContactPoint, PenaltyFlatContact, two_point_G


@dataclass
class Node:
    """A rigid body, represented by 3 raw global DOF. For an ordinary node
    these are the reference-point DOF `[u, w, phi]` directly (`G` = identity).
    A *contact* node (`Frame.add_contact_node`) instead uses the physical
    displacement at its two contact points, `[w1, u_c, w2]`, converted to
    standard `[u, w, phi]` via `G` (`contact.two_point_G`) whenever a beam
    needs to attach to it -- see `contact.py` for the rationale (mirrors the
    original prototype's `Rigid_Contact` two-point DOF parameterization).

    Beams may attach either directly at the reference point (offset (0, 0),
    the common case) or at a fixed local offset via `Frame.add_element`'s
    `offset_a`/`offset_b` (`rigid.py`) -- e.g. two beams attaching above/below
    a body's center."""
    x0: float
    z0: float
    fixed: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))
    u_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    F_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    G: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class StepResult:
    load_factor: float
    u: np.ndarray            # (3*n_nodes,) converged global displacement vector
    F_int: np.ndarray         # (3*n_nodes,) internal force vector (== reactions at fixed DOF)
    n_newton_iters: int


class Frame:
    def __init__(self):
        self.nodes: list[Node] = []
        # (node_a_idx, node_b_idx, beam, offset_a, offset_b); offsets are
        # (dx, dy) of the beam's attachment point relative to the node's own
        # reference point, in the node's initial (unrotated) orientation.
        self.elements: list[tuple[int, int, object, tuple, tuple]] = []
        # (node_idx, offset, contact_obj); a penalty contact behaves like a
        # beam with only one end, attached at a fixed offset from node_idx.
        self.penalty_contacts: list[tuple[int, tuple, PenaltyFlatContact]] = []

    def add_node(self, x: float, z: float) -> int:
        self.nodes.append(Node(x0=x, z0=z))
        return len(self.nodes) - 1

    def add_contact_node(self, x: float, z: float, dx1: float, dx2: float) -> int:
        """A rigid body parameterized by two contact points at horizontal
        offsets `dx1`, `dx2` from its reference point `(x, z)` (both at the
        same height as the reference point). Raw DOF are `[w1, u_c, w2]`
        (`contact.py`); use `Frame.add_flat_contact` to attach an actual
        contact surface to one of the two raw indices (0 or 2), and
        `Frame.add_element`'s offsets still work normally for beams attaching
        elsewhere on this same body (converted through `G` automatically)."""
        self.nodes.append(Node(x0=x, z0=z, G=two_point_G(dx1, dx2)))
        return len(self.nodes) - 1

    def add_flat_contact(self, node_idx: int, raw_index: int, dx: float,
                          surface_z: float) -> FlatSurfaceContactPoint:
        """Register a flat-surface contact point at horizontal offset `dx`
        from `node_idx`'s reference point (dy=0), tied to raw DOF index
        `raw_index` (0 or 2 -- must match how the node's `G` was built via
        `add_contact_node`). Returns the `FlatSurfaceContactPoint`, whose
        `.active` flag is toggled by `solve_path_with_contact`'s active-set
        loop; pass the same object in that call's `contacts` list."""
        node = self.nodes[node_idx]
        z0_point = node.z0  # dy = 0 for contact points, so initial height == node's own
        return FlatSurfaceContactPoint(node_idx=node_idx, raw_index=raw_index,
                                        z0=z0_point, surface_z=surface_z)

    def add_penalty_contact(self, node_idx: int, offset: tuple, surface_z: float,
                             k: float) -> PenaltyFlatContact:
        """Attach a one-sided linear penalty contact point (`contact.py`) at
        a fixed local offset from `node_idx`'s reference point -- an
        ordinary node works fine here (no `G`-matrix reparametrization
        needed, unlike `add_flat_contact`'s active-set points), since this
        behaves like a beam with only one end. Registered automatically in
        `_assemble`; no separate solve-time bookkeeping needed since the
        force/stiffness is a continuous function of the current state."""
        dx, dy = offset
        node = self.nodes[node_idx]
        z0_point = node.z0 + dy  # initial global z of the offset anchor point
        contact = PenaltyFlatContact(z0=z0_point, surface_z=surface_z, k=k)
        self.penalty_contacts.append((node_idx, tuple(offset), contact))
        return contact

    def add_element(self, node_a: int, node_b: int, beam,
                     offset_a: tuple = (0.0, 0.0), offset_b: tuple = (0.0, 0.0)) -> None:
        """Connect `beam` between two nodes (rigid-body reference points).

        `offset_a`/`offset_b` let the beam attach at a fixed local offset
        from each node's reference point rather than the reference point
        itself -- e.g. a rigid body with two beams attached above/below its
        center (`rigid.py`). Defaults to (0, 0), i.e. direct point-to-point
        attachment, matching all previous behavior.

        `beam`'s own initial endpoint coordinates (xa0, za0, xb0, zb0) must
        already be set to the *anchor's* initial position (node position +
        offset), not the node's own reference position -- see
        `rigid.anchor_initial_position`.
        """
        self.elements.append((node_a, node_b, beam, tuple(offset_a), tuple(offset_b)))

    def fix(self, node_idx: int, dof: int, target: float = 0.0) -> None:
        """Prescribe a displacement (relative to initial position) for one DOF (0=u,1=w,2=phi)."""
        self.nodes[node_idx].fixed[dof] = True
        self.nodes[node_idx].u_target[dof] = target

    def load(self, node_idx: int, dof: int, target: float) -> None:
        """Prescribe an external force/moment target for one DOF (0=Fx,1=Fz,2=M)."""
        self.nodes[node_idx].fixed[dof] = False
        self.nodes[node_idx].F_target[dof] = target

    def ndof(self) -> int:
        return 3 * len(self.nodes)

    def _fixed_mask(self) -> np.ndarray:
        return np.concatenate([n.fixed for n in self.nodes])

    def _u_target_vec(self) -> np.ndarray:
        return np.concatenate([n.u_target for n in self.nodes])

    def _F_target_vec(self) -> np.ndarray:
        return np.concatenate([n.F_target for n in self.nodes])

    def _assemble(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (F_int, K) for the current global displacement vector u.

        For each element, the beam itself only knows about its own two
        *anchor* points, expressed in each node's *standard* `[u, w, phi]`
        DOF. Two things can separate a node's raw DOF (what `u` actually
        stores, 3 entries per node) from those standard anchor-point DOF:

          1. A fixed local offset (`rigid.anchor_dof`) if the beam attaches
             away from the node's own reference point.
          2. The node's own `G` matrix (`contact.two_point_G`), for a
             contact node whose raw DOF are two contact points' physical
             displacement rather than `[u, w, phi]` directly.

        Composing both: q_std = G @ q_raw, anchor_dof = h(q_std, offset), so
        the full Jacobian from raw DOF to anchor DOF is `J = H(phi_std) @ G`
        (`H` = `rigid.anchor_jacobian`; `G` = identity for an ordinary node,
        so this reduces exactly to the plain offset case). Virtual work
        (`F_raw = J^T @ f_anchor`) and its Newton tangent then pick up an
        extra correction term from `dJ/d(q_raw)`, weighted by `G`'s row for
        `phi` (`G[2, :]`) since that's the only route by which `q_raw`
        affects `phi_std` and hence `H`. Validated against finite
        differences during development (see `contact.py`'s tests); reduces
        to the old `(dH/dphi)^T @ f` correction exactly when `G` = identity.
        """
        ndof = self.ndof()
        F_int = np.zeros(ndof)
        K = np.zeros((ndof, ndof))
        for node_a, node_b, beam, offset_a, offset_b in self.elements:
            sa, sb = 3 * node_a, 3 * node_b
            nodeA, nodeB = self.nodes[node_a], self.nodes[node_b]
            q_rawA = u[sa:sa + 3]
            q_rawB = u[sb:sb + 3]
            GA, GB = nodeA.G, nodeB.G
            q_stdA = GA @ q_rawA
            q_stdB = GB @ q_rawB
            phiA, phiB = q_stdA[2], q_stdB[2]

            dofA = anchor_dof(q_stdA, *offset_a)
            dofB = anchor_dof(q_stdB, *offset_b)
            HA = anchor_jacobian(phiA, *offset_a)
            HB = anchor_jacobian(phiB, *offset_b)
            JA = HA @ GA
            JB = HB @ GB

            result = beam.compute(dofA, dofB)
            fG_ab, fG_ba = result.fG_ab, result.fG_ba
            K_anchor = result.K
            K_AA = K_anchor[0:3, 0:3]
            K_AB = K_anchor[0:3, 3:6]
            K_BA = K_anchor[3:6, 0:3]
            K_BB = K_anchor[3:6, 3:6]

            F_bodyA = JA.T @ fG_ab
            F_bodyB = JB.T @ fG_ba

            K_AA_body = JA.T @ K_AA @ JA
            K_BB_body = JB.T @ K_BB @ JB
            HAprime = anchor_jacobian_prime(phiA, *offset_a)
            HBprime = anchor_jacobian_prime(phiB, *offset_b)
            K_AA_body += np.outer(GA.T @ HAprime.T @ fG_ab, GA[2, :])
            K_BB_body += np.outer(GB.T @ HBprime.T @ fG_ba, GB[2, :])
            K_AB_body = JA.T @ K_AB @ JB
            K_BA_body = JB.T @ K_BA @ JA

            F_int[sa:sa + 3] += F_bodyA
            F_int[sb:sb + 3] += F_bodyB
            K[sa:sa + 3, sa:sa + 3] += K_AA_body
            K[sa:sa + 3, sb:sb + 3] += K_AB_body
            K[sb:sb + 3, sa:sa + 3] += K_BA_body
            K[sb:sb + 3, sb:sb + 3] += K_BB_body

        # Penalty contacts: identical chain rule, one-ended (no coupling to
        # a second node) -- behaves exactly like a beam's single-end
        # contribution.
        for node_idx, offset, contact in self.penalty_contacts:
            s = 3 * node_idx
            node = self.nodes[node_idx]
            q_raw = u[s:s + 3]
            Gn = node.G
            q_std = Gn @ q_raw
            phi = q_std[2]

            dof_anchor = anchor_dof(q_std, *offset)
            H = anchor_jacobian(phi, *offset)
            J = H @ Gn

            f_anchor, K_anchor, _ = contact.compute(dof_anchor)

            F_body = J.T @ f_anchor
            K_body = J.T @ K_anchor @ J
            Hprime = anchor_jacobian_prime(phi, *offset)
            K_body += np.outer(Gn.T @ Hprime.T @ f_anchor, Gn[2, :])

            F_int[s:s + 3] += F_body
            K[s:s + 3, s:s + 3] += K_body
        return F_int, K

    def _save_warm_starts(self):
        return [beam.f_ab.copy() for _, _, beam, _, _ in self.elements]

    def _restore_warm_starts(self, saved) -> None:
        for (_, _, beam, _, _), f_ab in zip(self.elements, saved):
            beam.f_ab = f_ab.copy()

    def _reference_force_scale(self, free: np.ndarray) -> float:
        """
        A single, fixed force/moment scale for the *whole* solve, used to
        turn the residual into a relative (dimensionless) convergence check.

        Per-step or per-component scaling (e.g. norm of the current step's
        F_ext) breaks down for DOF whose target is exactly zero (a pure
        moment load with zero prescribed force at the same node, which is
        the common case) -- any nonzero residual there looks "large" relative
        to a zero target even when it is physically tiny, and looks
        arbitrarily "small" once the *other* free DOF's forces happen to be
        large. Using the full (final, unfactored) applied load's norm avoids
        both failure modes. Falls back to a characteristic stiffness*unit
        scale for purely displacement-controlled problems where F_target is
        identically zero everywhere.
        """
        F_full = self._F_target_vec()
        scale = np.linalg.norm(F_full[free])
        if scale > 0:
            return scale
        # No applied force anywhere (pure displacement control) -- fall back
        # to the diagonal of K evaluated at the zero state as a characteristic
        # stiffness scale, so the tolerance still means something.
        _, K0 = self._assemble(np.zeros(self.ndof()))
        diag = np.abs(np.diag(K0)[free])
        return max(diag.max() if diag.size else 0.0, 1e-300)

    def solve(self, n_steps: int = 20, tol: float = 1e-6, max_newton: int = 30,
              verbose: bool = False, max_step_doublings: int = 8) -> list[StepResult]:
        """
        Public entry point: try `_solve_fixed_steps` with `n_steps` load
        increments; if any increment fails to converge (the outer Newton
        hits the same "correct descent direction, needs a much smaller step"
        conditioning issue documented in derivations/geometric_softening.md
        and design/architecture.md for the local element solve), retry the
        whole solve with 2x as many, smaller increments, up to
        `max_step_doublings` retries. Simpler than a true adaptive/recursive
        step-halving scheme and re-does earlier steps on failure, but load
        steps are cheap (see design/roadmap.md performance note) and this
        keeps the core loop simple.
        """
        n = n_steps
        last_err = None
        for _ in range(max_step_doublings + 1):
            try:
                return self._solve_fixed_steps(n, tol, max_newton, verbose)
            except RuntimeError as e:
                last_err = e
                n *= 2
        raise RuntimeError(
            f"Frame solve did not converge even after doubling load steps up "
            f"to {n} (from {n_steps}). Last error: {last_err}"
        )

    def _newton_equilibrium(self, u: np.ndarray, F_ext: np.ndarray, free: np.ndarray,
                             ref_scale: float, tol: float, max_newton: int,
                             verbose: bool = False, step_label: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Core Newton-Raphson equilibrium iteration for one load/displacement
        increment, given `u` with its fixed DOF already set to their target
        value for this step and its free DOF holding a warm-start guess
        (typically the previous step's converged solution). Returns the
        converged `(u, F_int, K, n_iter)`.

        Each JirasekBeam keeps a persistent `f_ab` warm-start (the converged
        local left-end force from its last `compute()` call), used to seed
        the next local Newton solve. That's essential for performance/
        robustness across load steps, but it's a mutable side effect of
        `compute()` -- and the line search below calls `compute()` (via
        `_assemble`) speculatively for trial displacements that may get
        rejected. Left unguarded, a rejected trial would leave every beam's
        warm-start state pointing at a force consistent with a *different*
        (discarded) configuration, contaminating the next real solve. So:
        explicitly snapshot each beam's warm-start state before every trial
        and restore it on rejection, exactly like restoring `u` itself.

        `tol` is relative to `ref_scale`, a single fixed force/moment scale
        for the whole solve (see `_reference_force_scale`), not the current
        step's load.
        """
        F_int, K = self._assemble(u)
        accepted_warm = self._save_warm_starts()
        residual = F_ext - F_int
        res_norm = np.linalg.norm(residual[free]) / ref_scale if free.any() else 0.0

        n_iter = 0
        for n_iter in range(1, max_newton + 1):
            if verbose:
                print(f"  {step_label} iter {n_iter:2d}  |residual_free| = {res_norm:.3e}")
            if res_norm < tol:
                break
            K_ff = K[np.ix_(free, free)]
            du_free = np.linalg.solve(K_ff, residual[free])

            # Backtracking line search on the outer (global equilibrium)
            # Newton step -- same rationale as the local element solve.
            # Every trial re-runs `_assemble`, which calls `compute()` on
            # each beam and would otherwise clobber its warm-start state
            # even for a rejected trial, so restore that state before
            # each attempt (see `_save_warm_starts`/`_restore_warm_starts`).
            dstep = 1.0
            for _ in range(30):
                self._restore_warm_starts(accepted_warm)
                u_try = u.copy()
                u_try[free] += dstep * du_free
                try:
                    F_int_try, K_try = self._assemble(u_try)
                    res_try = np.linalg.norm((F_ext - F_int_try)[free]) / ref_scale
                except (RuntimeError, np.linalg.LinAlgError):
                    res_try = np.inf
                if res_try < res_norm or dstep < 1e-6:
                    break
                dstep *= 0.5

            u, F_int, K, residual, res_norm = (
                u_try, F_int_try, K_try, F_ext - F_int_try, res_try
            )
            accepted_warm = self._save_warm_starts()
        else:
            raise RuntimeError(
                f"Frame Newton iteration did not converge in {max_newton} "
                f"iterations ({step_label}) (final |residual_free| = {res_norm:.3e})."
            )
        return u, F_int, K, n_iter

    def _solve_fixed_steps(self, n_steps: int, tol: float, max_newton: int,
                            verbose: bool) -> list[StepResult]:
        """Single monotonic ramp from the undeformed state (u=0) to the
        currently-set fix/load targets, over `n_steps` increments. See
        `_newton_equilibrium` for the per-step solve."""
        ndof = self.ndof()
        fixed = self._fixed_mask()
        free = ~fixed
        u_target = self._u_target_vec()
        F_target = self._F_target_vec()
        ref_scale = self._reference_force_scale(free)

        u = np.zeros(ndof)
        history: list[StepResult] = []

        for step in range(1, n_steps + 1):
            factor = step / n_steps
            u[fixed] = u_target[fixed] * factor
            F_ext = F_target * factor
            u, F_int, K, n_iter = self._newton_equilibrium(
                u, F_ext, free, ref_scale, tol, max_newton, verbose,
                step_label=f"step {step:3d}")
            history.append(StepResult(load_factor=factor, u=u.copy(),
                                       F_int=F_int.copy(), n_newton_iters=n_iter))
        return history

    def solve_path(self, leg_targets: list[dict], n_steps_per_leg: int = 100,
                   tol: float = 1e-6, max_newton: int = 25, max_step_doublings: int = 6,
                   verbose: bool = False) -> list[StepResult]:
        """
        Ramp prescribed-displacement targets through a sequence of legs (e.g.
        a multi-step actuation schedule), continuing warm-started across leg
        boundaries -- the same continuation idea as `solve_path_with_contact`,
        but without the active-set contact-checking loop, for problems whose
        only contact (if any) is penalty-based (`PenaltyFlatContact`), which
        is already folded into `_assemble` and needs no special-casing here.

        `leg_targets` is a list of `{(node_idx, dof): target_value}` dicts,
        one per leg; each updates those `Frame.fix` targets before ramping
        to them from wherever the previous leg ended.
        """
        ndof = self.ndof()
        u = np.zeros(ndof)
        ref_scale = self._reference_force_scale(~self._fixed_mask())
        history: list[StepResult] = []

        for leg in leg_targets:
            for (node_idx, dof), target in leg.items():
                self.fix(node_idx, dof, target)
            u_target_leg = self._u_target_vec().copy()
            fixed_at_leg_start = self._fixed_mask().copy()
            u_leg_start = u.copy()

            n_steps = n_steps_per_leg
            last_err = None
            for attempt in range(max_step_doublings + 1):
                try:
                    leg_history, u_end = self._solve_leg_plain(
                        u_leg_start, fixed_at_leg_start, u_target_leg, n_steps,
                        tol, max_newton, ref_scale, verbose)
                    break
                except RuntimeError as e:
                    last_err = e
                    n_steps *= 2
            else:
                raise RuntimeError(
                    f"solve_path: leg did not converge even after doubling to "
                    f"{n_steps} steps. Last error: {last_err}"
                )
            history += leg_history
            u = u_end
        return history

    def _solve_leg_plain(self, u_start: np.ndarray, fixed_at_leg_start: np.ndarray,
                          u_target_leg: np.ndarray, n_steps: int, tol: float,
                          max_newton: int, ref_scale: float,
                          verbose: bool) -> tuple[list[StepResult], np.ndarray]:
        """One leg of `solve_path`: ramp the DOF fixed at this leg's start
        linearly from their leg-start value to the leg's target; the fixed
        mask itself does not change mid-leg (no active-set contact here)."""
        u = u_start.copy()
        leg_history: list[StepResult] = []
        free = ~fixed_at_leg_start

        for step in range(1, n_steps + 1):
            factor = step / n_steps
            u_step = u.copy()
            u_step[fixed_at_leg_start] = (
                u_start[fixed_at_leg_start]
                + (u_target_leg[fixed_at_leg_start] - u_start[fixed_at_leg_start]) * factor
            )
            F_ext = self._F_target_vec() * factor
            u, F_int, K, n_iter = self._newton_equilibrium(
                u_step, F_ext, free, ref_scale, tol, max_newton, verbose,
                step_label=f"leg step {step:3d}/{n_steps}")
            leg_history.append(StepResult(load_factor=factor, u=u.copy(),
                                           F_int=F_int.copy(), n_newton_iters=n_iter))
        return leg_history, u

    def solve_path_with_contact(self, leg_targets: list[dict],
                                 contacts: list[FlatSurfaceContactPoint],
                                 n_steps_per_leg: int = 100, tol: float = 1e-7,
                                 max_newton: int = 25, max_step_doublings: int = 6,
                                 max_contact_iters: int = 5,
                                 verbose: bool = False) -> list[StepResult]:
        """
        Ramp prescribed-displacement targets through a sequence of legs
        (e.g. down, then back up), each leg continuing warm-started -- both
        the displacement solution and every beam's persistent local force
        state -- from the end of the previous leg. Necessary for a load
        sequence that reverses direction, where restarting from the
        undeformed state each leg would lose the load-path history and could
        converge to a different equilibrium branch, especially with a
        path-dependent, one-sided constraint like contact in the mix.

        `leg_targets` is a list of `{(node_idx, dof): target_value}` dicts,
        one per leg; each updates those `Frame.fix` targets (via `self.fix`)
        before ramping to them.

        `contacts` is a list of `FlatSurfaceContactPoint` (`contact.py`).
        After every converged step, each contact point is checked
        active-set style (mirroring the original prototype's
        `Rigid_Contact.check()`, `derivations/contact_friction_model.md`):
        if inactive and penetrating, pin it to the surface (`Frame.fix`);
        if active and its reaction has gone non-push, release it
        (`Frame.load`). If any point's state changed, the step is re-solved
        with the updated boundary conditions (up to `max_contact_iters`
        times) before moving on -- a small fixed-point iteration on top of
        Newton, since one point activating/releasing can occasionally flip
        another.
        """
        ndof = self.ndof()
        u = np.zeros(ndof)
        ref_scale = self._reference_force_scale(~self._fixed_mask())
        history: list[StepResult] = []

        for leg in leg_targets:
            for (node_idx, dof), target in leg.items():
                self.fix(node_idx, dof, target)

            n_steps = n_steps_per_leg
            last_err = None
            for attempt in range(max_step_doublings + 1):
                try:
                    leg_history, u_end = self._solve_leg_with_contact(
                        u, contacts, n_steps, tol, max_newton, max_contact_iters,
                        ref_scale, verbose)
                    break
                except RuntimeError as e:
                    last_err = e
                    n_steps *= 2
            else:
                raise RuntimeError(
                    f"solve_path_with_contact: leg did not converge even after "
                    f"doubling to {n_steps} steps. Last error: {last_err}"
                )
            history += leg_history
            u = u_end
        return history

    def _solve_leg_with_contact(self, u_start: np.ndarray,
                                 contacts: list[FlatSurfaceContactPoint],
                                 n_steps: int, tol: float, max_newton: int,
                                 max_contact_iters: int, ref_scale: float,
                                 verbose: bool) -> tuple[list[StepResult], np.ndarray]:
        """One leg of `solve_path_with_contact`. DOF that were already fixed
        at the *start* of this leg (the leg's own driving displacement,
        e.g. the input body's prescribed w) ramp linearly from their
        leg-start value to the leg's target. DOF that become fixed mid-leg
        because a contact point just activated are *not* ramped -- they
        snap directly to the (constant) contact surface target, same as the
        original prototype's active-set contact."""
        fixed_at_leg_start = self._fixed_mask().copy()
        u_target_leg = self._u_target_vec().copy()

        u = u_start.copy()
        leg_history: list[StepResult] = []

        for step in range(1, n_steps + 1):
            factor = step / n_steps
            F_ext = self._F_target_vec() * factor

            for contact_iter in range(max_contact_iters):
                fixed_now = self._fixed_mask()
                target_now = self._u_target_vec()
                u_step = u.copy()
                for idx in np.nonzero(fixed_now)[0]:
                    if fixed_at_leg_start[idx]:
                        # driving DOF for this leg: ramp from leg-start value
                        u_step[idx] = u_start[idx] + (u_target_leg[idx] - u_start[idx]) * factor
                    else:
                        # newly-active contact DOF: snap to the (constant) target
                        u_step[idx] = target_now[idx]
                free_now = ~fixed_now

                u_solved, F_int, K, n_iter = self._newton_equilibrium(
                    u_step, F_ext, free_now, ref_scale, tol, max_newton, verbose,
                    step_label=f"step {step:3d}/{n_steps} contact-iter {contact_iter}")

                changed = self._update_contacts(u_solved, F_int, contacts)
                u = u_solved
                if not changed:
                    break

            leg_history.append(StepResult(load_factor=factor, u=u.copy(),
                                           F_int=F_int.copy(), n_newton_iters=n_iter))
        return leg_history, u

    def _update_contacts(self, u: np.ndarray, F_int: np.ndarray,
                          contacts: list[FlatSurfaceContactPoint]) -> bool:
        """Active-set check for each contact point, mirroring the original
        prototype's `Rigid_Contact.check()`: activate on penetration, release
        once the reaction would have to pull rather than push. Returns
        whether any point's state changed."""
        changed = False
        for c in contacts:
            idx = 3 * c.node_idx + c.raw_index
            if not c.active:
                if c.penetration(u[idx]) > 0:
                    self.fix(c.node_idx, c.raw_index, c.contact_target())
                    c.active = True
                    changed = True
            else:
                if F_int[idx] <= 0:
                    self.load(c.node_idx, c.raw_index, 0.0)
                    c.active = False
                    changed = True
        return changed
