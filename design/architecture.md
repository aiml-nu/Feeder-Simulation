# Architecture

Working notes on how `src/` should be laid out and why. Goal: keep the planar rigid-body +
beam-element solver reusable across flexure designs, and make the beam formulation (Euler-
Bernoulli vs. Timoshenko, etc.) and buckling/geometric-stiffness model (NASA TR softening term
vs. P-Delta, etc.) both swappable without touching assembly/solve code.

This builds directly on `elements.py` / `functions.py` / `flexures.py` from the earlier
prototype (see `derivations/` for the math those files implement) — the goal here is
restructuring, not rewriting the underlying formulas.

## Proposed layout

```
src/flexsim/
    elements/
        rigid.py          # RigidBody base, RigidRegular, RigidContact
        beam.py           # Beam (formulation-agnostic; holds refs to a formulation + a buckling model)
        structure.py       # Structure: assembly, mixed-BC solve, contact-check loop driver
    formulations/
        base.py            # BeamFormulation interface
        euler_bernoulli.py # current default (from functions.py _k_stiff, _stress)
        timoshenko.py       # planned
    buckling/
        base.py             # GeometricStiffness interface
        nasa_softening.py   # current default (from functions.py _k_soft)
        p_delta.py          # planned, for cross-validation against nasa_softening
    contact/
        jaw_contact.py       # RigidContact-specific G matrix + stick-slip friction (from Rigid_Contact.check())
    flexures/
        dual_cam.py           # concrete topology(ies) — parametric flexure geometry builders
    sim/
        loading.py             # input displacement/force schedules over time
        timestepper.py          # generic time-stepping loop: apply loads, solve, check contact, record
    viz/
        plotting.py              # Structure.plot / Beam.plot / animation (matplotloom)
```

Two things drive this split: (1) `Structure`'s assembly and mixed-BC solve
(`derivations/rigid_body_kinematics.md`) don't care what's inside a beam's stiffness matrix, so
they shouldn't live next to formulation-specific code; (2) elastic formulation and buckling
model are logically independent choices (you could pair Euler-Bernoulli with either softening
method, or eventually Timoshenko with either), so they get separate interfaces rather than one
combined "beam type" enum.

## Formulation interface

```python
class BeamFormulation(Protocol):
    def stiffness_matrix(self, w: float, t: float, L: float, E: float) -> np.ndarray: ...
    def stress(self, F_beam: np.ndarray, w: float, t: float) -> np.ndarray: ...
    # shape functions for plotting deflected shape (_hermite today) probably also belongs here,
    # since Timoshenko shape functions differ from Euler-Bernoulli's cubic Hermite.
    def deflected_shape(self, u_beam: np.ndarray, L: float, n: int) -> tuple[np.ndarray, np.ndarray]: ...
```

`EulerBernoulli` wraps today's `_k_stiff` / `_stress` / `_hermite`
(`derivations/beam_stiffness_euler_bernoulli.md`). Adding Timoshenko later means implementing
this interface once, with no changes to `Beam`, `Structure`, or any flexure topology file.

## Buckling / geometric-stiffness interface (linear-formulation track)

```python
class GeometricStiffness(Protocol):
    def softening_matrix(self, L: float) -> np.ndarray: ...
```

`NasaSoftening` wraps today's `_k_soft` (`derivations/geometric_softening.md`). `Beam.update()`
computes `k_beam = formulation.stiffness_matrix(...) - beam.P * buckling.softening_matrix(...)`
the same way it does today, just via the two injected objects instead of module-level functions.
This stays useful as a cheap, closed-form option for a linear beam formulation (fast, good for
optimization loops over many design iterations), but see below for why it's no longer the
primary plan for capturing buckling/large-displacement behavior.

## Geometrically nonlinear track (preferred, per Jirásek et al. 2021)

Revisiting the "P-Delta" idea after reading Jirásek, La Malfa Ribolla & Horák (2021) — see
`derivations/references.md` — reframes it usefully: **P-Delta is really just incremental,
geometrically nonlinear time-stepping**, not fundamentally a special stiffness-matrix
correction. Apply the load in increments; at each increment, solve for equilibrium using the
beam's *current, deformed* geometry (updated orientation, updated internal moment arms); repeat.
Any second-order/P-Delta-like softening then falls out naturally from re-evaluating equilibrium
on the deformed shape, rather than needing an extra matrix bolted onto a linear formulation and
hoping it's compatible (the concern that motivated this rethink — see
`derivations/geometric_softening.md`).

This points toward a second, more general `BeamFormulation` variant that doesn't return a fixed
linear stiffness matrix at all:

```python
class NonlinearBeamFormulation(Protocol):
    def internal_force(self, state: BeamState) -> np.ndarray: ...      # possibly nonlinear in u
    def tangent_stiffness(self, state: BeamState) -> np.ndarray: ...    # d(internal_force)/d(u)
```

A linear `EulerBernoulli` formulation is the trivial special case (`internal_force = k @ u`,
`tangent_stiffness = k`, constant). `Jirasek2021` would implement this properly: internal force
and moment at any beam section come from integrating the paper's finite-difference scheme along
the beam (a shooting-method-style local iteration per element, at the element level — the number
of *global* DOF per beam stays at 6, same as today), giving forces and a consistent tangent
stiffness that already account for large rotation and axial-bending coupling. This is a genuine
alternative to bolting a softening matrix onto a small-displacement element, and it's the
formulation this repo cites for actual large-displacement flexure behavior (many of these
flexure beams likely see rotations that aren't "small" in the linear-theory sense once the jaw
is displaced — worth quantifying, `design/roadmap.md`).

Driving this from `Structure`/`sim/timestepper.py` means each load increment is no longer a
single linear solve (as today's `Structure.solve()` does): it needs a **Newton-Raphson (or
similar) equilibrium iteration** within the increment, re-evaluating `internal_force` and
`tangent_stiffness` at the current guess until residual (applied load minus internal force) is
small. The existing mixed-BC projector trick (`derivations/rigid_body_kinematics.md`) still
applies unchanged *inside* each Newton iteration — it only assumed a linear `F = K u` relation
locally at the current iterate, which a tangent-stiffness linearization provides regardless of
formulation. So `Structure`'s assembly/BC machinery doesn't need to change; what changes is that
`solve()` becomes an iterative loop (call it `solve_increment()`) instead of one-shot, and the
existing per-step `Beam.update()` "use `P` from the previous step" trick
(`derivations/geometric_softening.md`) becomes exact Newton iteration to convergence within each
step, rather than a single explicit update.

Both tracks (linear + additive softening, vs. fully nonlinear) are worth keeping available:
linear is cheap and fine early in design iteration where small-deflection behavior dominates;
nonlinear is the one to trust for anything near buckling, snap-through, or large jaw
displacement, and for the validation cases in `examples/` (Euler buckling, Williams toggle) that
this repo's buckling handling should be checked against either way.

## Contact and time-stepping

`RigidContact`'s `xyx` DOF and stick-slip friction (`derivations/contact_friction_model.md`) stay
separate from the generic `RigidBody`/`elements/rigid.py` — contact is a property of *how a
specific rigid body interacts with its environment*, not a core kinematic concept, so new contact
types (e.g. a different jaw geometry, or contact for the wire itself) can be added without
touching `RigidRegular`.

Today's `Dual_Cam_OG.simulate()` bakes a specific loading schedule (preload + cyclic top/bottom
displacement) directly into the flexure class. Proposed split: `flexures/dual_cam.py` only builds
geometry (rigid bodies + beams + `Structure`), and `sim/timestepper.py` takes a `Structure` plus a
`loading.py`-defined schedule (per-`RigidBody` prescribed displacement/force vs. time) and runs
the step loop (apply BCs -> `structure.check_contact()` -> `structure.solve()` -> record). This
lets the same time-stepper drive different flexure topologies, and different loading scenarios
drive the same topology, independently.

## What stays as-is

The mixed-BC solve trick in `Structure.solve()` (`derivations/rigid_body_kinematics.md`) is
already fully formulation-agnostic and doesn't need restructuring, just relocating into
`elements/structure.py`.

## Not yet decided

- Exact package/module name (`flexsim` used above as a placeholder).
- Whether formulation + buckling model are per-`Beam` (allowing mixed formulations in one
  structure — useful for validation, e.g. compare Timoshenko vs. Euler-Bernoulli beams
  side-by-side in one model) or a single global setting per `Structure`. Leaning per-`Beam` for
  flexibility, defaulted at `Structure`/flexure-builder level for convenience.
- Testing strategy (unit tests per formulation against closed-form solutions — e.g. cantilever
  tip deflection, Euler buckling load — belongs in `design/roadmap.md` as a concrete task).
- Whether `NonlinearBeamFormulation` and `BeamFormulation` (linear) should actually be the same
  interface (with linear as a trivial case that skips Newton iteration entirely, e.g.
  `tangent_stiffness` constant and `internal_force` linear lets a generic solver converge in one
  iteration) rather than two separate interfaces/solve paths. Leaning toward unifying once the
  Jirásek-style formulation is actually implemented and the interface has been exercised for
  real, rather than guessing the right abstraction now.
- How much of `Structure.solve()`'s mixed-BC machinery needs to change to support per-increment
  Newton iteration vs. how much can be reused as-is inside each iteration — sketched above,
  needs to be proven out in code.
