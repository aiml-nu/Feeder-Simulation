# Roadmap

Working notes, rough priority order. Target: physical prototype + this tool presented together
at a conference in ~3 months (mid-October 2026).

## Status: Jirásek beam element + nonlinear frame solver (done, first pass)

`src/flexsim/formulations/jirasek.py` (element) and `src/flexsim/frame.py` (multi-element solver)
are implemented and validated against the closed-form circular-arc solution for pure end-moment
bending, up to a full 90-degree tip rotation (`examples/pure_bending_cantilever/`) — matches to
~3e-5 in displacement, ~6e-13 in rotation, and the tangent stiffness matches central-difference
numerical differentiation of the internal force. Three non-obvious bugs came up building the
Newton solver, worth remembering when this gets folded into the general `Structure`
(`design/architecture.md`):

1. **Relative, not absolute, convergence tolerance, and a single fixed reference scale.** Beam
   stiffness scale (`EA`, `EI`) can easily be ~1e-7 for a slender flexure, so an absolute residual
   tolerance can end up *looser* than the applied load itself, "converging" at zero deflection
   without taking a Newton step. Worse, normalizing by the *current step's* force magnitude breaks
   down for any DOF whose target is exactly zero (e.g. a pure-moment load with zero prescribed
   force at the same node) — that DOF's residual always looks "large" relative to its own zero
   target. Fix: normalize by a single reference scale for the whole solve (the full, final applied
   load's norm, or a characteristic stiffness for pure displacement-controlled problems).
2. **Warm-start state contamination during line search.** `JirasekBeam` keeps a persistent
   `f_ab` warm-start (last converged local left-end force) to seed the next local Newton solve —
   necessary for performance. But the outer Newton loop's backtracking line search calls
   `compute()` speculatively on trial displacements that may get rejected, and `compute()` mutates
   that persistent state every time, including for rejected trials. Left unguarded, this
   contaminates the next real solve with a warm start from a discarded configuration. Fix:
   explicitly snapshot/restore each beam's warm-start state around every trial evaluation, same as
   restoring the displacement vector itself.
3. **Full Newton steps can badly overshoot for realistic slenderness**, at both the local
   (element) and global (frame) level — the moment-vs-rotation tangent stiffness entries can be
   orders of magnitude larger than the force-vs-displacement ones, so a step that's a good local
   descent direction can still land far from the true nonlinear solution. A plain backtracking
   line search sometimes gets stuck taking many tiny, barely-improving steps rather than
   converging. Fix (local level): automatic sub-stepping — ramp the target incrementally and
   re-run Newton at each increment, warm-started from the previous one, doubling the increment
   count on failure. Fix (frame level, currently cruder): retry the whole solve with 2x as many,
   smaller load steps on failure.

**Update:** extended with rigid-body offset kinematics (`src/flexsim/rigid.py`) so beams can
attach at a fixed local offset from a node's reference point instead of only at the point
itself — needed for `examples/parallel_beam_flexure/` (two rigid bodies, two beams attached
above/below each body's center, mixed boundary conditions). Uses the same large-rotation-exact
chain-rule pattern as the Jirasek element's own corotational transform: `H(phi)^T @ K @ H`, plus
a correction term `dH/dphi^T @ f` in the diagonal block for the rotation column. Reduces exactly
to the old point-to-point assembly at offset `(0, 0)` (`pure_bending_cantilever` re-run
unchanged as a regression check). New assembly validated against classical small-deflection
parallel-guide stiffness `k = 2*12EI/L^3` (~0.2% match) before trusting the nonlinear/mismatch
results. See `examples/parallel_beam_flexure/README.md`.

**Update 2:** added active-set contact (`src/flexsim/contact.py`) and multi-leg load paths
(`Frame.solve_path_with_contact`, `src/flexsim/frame.py`) for
`examples/parallel_beam_flexure_contact/`. Contact reuses the original prototype's two-point
`Rigid_Contact` idea (`derivations/contact_friction_model.md`) rather than a penalty law:
a contacted rigid body's DOF are reparametrized as `[w_point1, u_center, w_point2]`
(`contact.two_point_G`, generalized beyond the old code's symmetric-only points to arbitrary
independent offsets), so each point's contact status can be toggled with the existing
`Frame.fix`/`Frame.load` machinery directly — no separate constraint/Lagrange-multiplier
machinery needed. `Node.G` composes with the existing anchor-offset transform
(`J = H(phi) @ G`) so beams attach normally; validated against finite differences. The load-path
generalization (`solve_path_with_contact`, `_solve_leg_with_contact`) ramps prescribed-displacement
targets through a sequence of legs (e.g. down then back up), continuing warm-started across the
reversal, and re-solves a step if the active-set contact check flips any point's state.
Both previous examples were re-run unchanged as regression checks after these changes.

Also surfaced (and documented, `examples/parallel_beam_flexure_contact/README.md`) a
well-posedness gotcha: prescribing only `w` at a driving body (leaving `F_x`/`M` free there too)
can leave a mechanism with no absolute restraint on net translation/rotation until contact
engages, a genuine singular-stiffness condition, not a bug — worth remembering as a checklist item
before assembling similar test problems later (ground the driving body's other DOF unless contact
or something else is providing that restraint from the start).

**Update 3:** added a second, compliant/penalty contact law (`contact.PenaltyFlatContact`,
`Frame.add_penalty_contact`) alongside the active-set contact from Update 2, following a
discussion of the tradeoffs. Active-set is exact (no tunable stiffness, no artificial
penetration) but doesn't scale as gracefully to many simultaneous contact points and needs a
combinatorial activate/release decision (chattering risk, already flagged for the old code).
Penalty is the more common default in general FEA, avoids the combinatorial part entirely, and
is needed to represent *real* local contact compliance — e.g. a 3D-printed PLA jaw is much softer
than aluminum and should show measurable penetration and an uneven clamping-load split between
contact points, not just a rigid on/off constraint. `PenaltyFlatContact` attaches to an *ordinary*
node (no `G`-matrix reparametrization needed) via the same anchor-offset chain rule already used
for beams, so it required no new machinery in `Frame._assemble` beyond what Update 2 already
built. Cross-validated against the active-set result: with the penalty stiffness large relative
to the beams' own stiffness, results match to ~5 significant figures
(`examples/parallel_beam_flexure_contact/`).

**Follow-up**: the penalty stiffness `k` is currently an arbitrary lumped value, tuned only for
"stiff enough" vs. "PLA-soft" demonstration purposes. Setting it to a physically meaningful value
once real geometry/material parameters exist — **Hertzian line-contact theory** (line-on-flat
contact stiffness from the two materials' effective modulus and the contact geometry's radius of
curvature) is the natural starting point — is flagged as future work, along with eventually
moving beyond two lumped contact points to a true distributed-contact model along the actual
wire/jaw contact line, once this needs to inform real part design (`design/flexure_design.md`).

**Update 4:** built `examples/feeder_flexure/` — the full V1 mechanism (ground, two independent
top/bottom inputs, jaw, `N` horizontal + `N` angled beams per side, `M_contact` penalty contact
points along the jaw), assembled entirely from machinery already validated in Updates 1-3 (no
`src/flexsim` changes needed). Confirms the generalized anchor-offset chain rule and
`Frame.solve_path` compose cleanly into a larger, more coupled (8-beam, 4-contact-point) mechanism.
One new finding: at a more "realistic" flexure thickness (`0.01`), the angled beams' slenderness
(`L/t` ~150-220) made the local per-element Newton solve (`jirasek.solve_local_end_forces`) so
ill-conditioned it became extremely slow/occasionally non-convergent for the actuation strokes
used here — confirmed by directly profiling `solve_path` at different `thickness` values.
`thickness=0.05` (`L/t` ~30-45) fixed both convergence and speed; this is a solver-robustness
choice, not yet a physically-tuned one. The 4-leg actuation sequence (both inputs up, top down,
both down, bottom up) solves in ~12s, and the final-state contact force distribution across the
jaw's contact points is uneven — the same parasitic-rotation effect from Updates 2-3, now visible
as a genuine clamping-force gradient along the jaw, which was the original motivation for adding
the penalty contact law. Animated via **matplotloom** per request
(`examples/feeder_flexure/feeder_flexure_actuation.mp4`). See
`examples/feeder_flexure/README.md` for full detail and open items (Hertzian `k_contact` tuning,
friction/feed force, stress/buckling margin reporting — all still open).

**Follow-up work**, not yet done:
- The frame-level retry-the-whole-solve-on-failure approach (point 3) is wasteful — it re-does
  every previously-converged step. A true adaptive scheme (subdivide only the failing increment,
  like the local element solve already does) would be faster, worth it once this is driving
  anything bigger than a two-node validation example.
- `examples/pure_bending_cantilever/` currently starts directly at a large, hand-picked step
  count (`n_steps=3000` for a 90-degree target) rather than letting the adaptive-doubling logic
  find a good count from a small starting guess, purely so the notebook runs in a reasonable time
  — that logic path is real but slow when it has to double many times from scratch (see point
  above). Revisit once frame-level sub-stepping is properly adaptive.
- Full closed-loop bending (`M = 2*pi*EI/L`, the paper's own benchmark) not yet run end-to-end
  here — only up to a 90-degree quarter turn. Should work with enough load steps, just untested,
  and would take a while with the current retry-based stepping.
- `NonlinearBeamFormulation` interface (`design/architecture.md`) not yet extracted — only one
  formulation exists, so the interface is still speculative; extract once there's a second
  formulation (Euler-Bernoulli, or a genuine comparison case) to actually swap against.
- No jaw contact / `Rigid_Contact` integration with this new nonlinear track yet — still only
  the old linear `elements.py` has contact logic (`derivations/contact_friction_model.md`).
- `williams_toggle` and `euler_buckling_cantilever` examples are still scaffolding only
  (`examples/`) — pure bending was the first one actually implemented.

## Near-term correctness fixes

- **`Rigid_Contact` force transform bug** — `G @ F_out_global_xyx` should be the energy-consistent
  `G⁻ᵀ @ F_out_global_xyx` for recovering centroid force/moment resultants. Confirmed numerically
  wrong as-is (see `derivations/contact_friction_model.md`). Doesn't affect the solve itself, but
  affects any reported/plotted jaw resultant force. Fix before trusting jaw force plots.
- **`F_AFSD` coupling into friction model** — currently caps tangential contact force at the AFSD
  process force requirement inside what should be a purely kinematic Coulomb friction law
  (flagged by a `# TODO`-ish comment already in the old code). Decide: split into (a) a pure
  friction/contact law and (b) a separate process-force check applied externally, per the
  original comment's suggestion.
- **Buckling sign convention** — `k_stiff - P * k_soft` softens under compression per the
  derivation, but worth a concrete unit test (see below) rather than trusting the derivation
  alone.

## Restructuring (this session's design docs cover the "what"; this is the "when")

1. Stand up `src/flexsim/` package skeleton per `design/architecture.md`.
2. Port `elements.py` into `elements/{rigid,beam,structure}.py`, keeping behavior identical —
   this is a refactor, not a rewrite, so a regression check against the old code's output on a
   fixed test case is worth doing before/after.
3. Extract `BeamFormulation` interface, implement `EulerBernoulli` as the port of current
   `_k_stiff`/`_stress`/`_hermite`.
4. Extract `GeometricStiffness` interface, implement `NasaSoftening` as the port of `_k_soft`.
5. Separate flexure topology (`flexures/dual_cam.py`) from loading schedule (`sim/loading.py`)
   and time-stepping loop (`sim/timestepper.py`).
6. Update jaw contact to the new distributed-contact design (`design/flexure_design.md`) if the
   old `Dual_Cam_OG` geometry doesn't already match it.

## Formulation / buckling extensions

- **Timoshenko beam formulation** — add as a second linear `BeamFormulation` once the interface
  is settled. Worth first checking slenderness (`L/t`) of the actual flexure beams to see how
  much this would even matter quantitatively before investing time.
- **Geometrically nonlinear beam formulation (Jirásek et al. 2021)** — now the preferred approach
  for buckling and large-displacement behavior, superseding the earlier plan to implement
  "P-Delta" as a second additive softening matrix (see `derivations/geometric_softening.md`
  update and `design/architecture.md`, "Geometrically nonlinear track"). Concretely: implement
  the `NonlinearBeamFormulation` interface (internal force + tangent stiffness from the paper's
  finite-difference/shooting-method element-level scheme), and drive it via incremental load
  application with Newton-Raphson equilibrium iteration per step ("generalized geometrically
  nonlinear time-stepping") rather than the current one-shot linear solve. This is a bigger
  lift than the additive-matrix approach but removes the open compatibility question between a
  hand-derived softening matrix and whatever formulation it's paired with, and directly
  supports large-rotation flexure behavior and post-buckling/snap-through.
- **Co-rotational frame** — beams currently don't update their local orientation (`R_beam`) as
  the structure deforms (noted in `derivations/geometric_softening.md`). Likely moot once the
  Jirásek-style nonlinear formulation is in place (it doesn't rely on a fixed local frame the
  same way), but worth quantifying for the interim linear-formulation track.

## Validation / testing

- Closed-form checks: cantilever beam tip deflection under end load, Euler critical buckling
  load for a pinned-pinned/cantilever column (should be recoverable from the softening term at
  the point where the solve becomes singular/ill-conditioned as `P` approaches `P_critical`, or
  directly from a nonlinear formulation's tangent stiffness eigenvalues going to zero — see
  Jirásek et al. 2021, Section 4.3 / Appendix B for the axially-compressible correction).
  Scaffolded as `examples/euler_buckling_cantilever/`.
- **Williams toggle** (`examples/williams_toggle/`) — two-beam snap-through benchmark with
  published experimental data (Williams 1964) and both an approximate analytical solution and
  independent numerical results (Jirásek et al. 2021, Section 4.2) to compare against. Good
  minimal test of large-rotation + instability behavior once a nonlinear formulation exists;
  more demanding than the Euler buckling case since it includes post-buckling/snap-through, not
  just onset of instability.
- Regression test: current `Dual_Cam_OG` notebook results (`flexure_3.ipynb`) as a baseline to
  compare against once ported into the new structure.
- Mesh/element-count convergence: since each physical flexure beam is modeled as a *single*
  element (per the "single beam elements, not a bunch of plate elements" goal stated for this
  project), worth confirming this is adequate — e.g. compare against a multi-element (or full
  plate/FEA) reference for one representative beam to bound the single-element approximation
  error, especially under combined bending + buckling.

## Time-stepping / contact

- Check for contact chattering (repeated activate/deactivate across consecutive steps) — noted
  as a numerical concern in `derivations/contact_friction_model.md`.
- Decide convergence criteria / step-size sensitivity for the incremental buckling update (`P`
  from previous step, not solved self-consistently within a step).

## Documentation / conference prep

- Keep `derivations/references.md` current as new formulations/models are added or considered.
- Fill in `design/flexure_design.md` open items once current prototype parameters are locked in.
- Add the actual flexure figure into `design/figures/` (only had it inline in chat this
  session).
- Once formulation/buckling swapping is implemented, a comparison plot (Euler-Bernoulli vs.
  Timoshenko; NASA softening vs. P-Delta) would make a strong conference figure — flag as a
  deliverable, not just an internal check.
