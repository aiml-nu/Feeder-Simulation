# Example: Feeder Flexure V1

Status: **implemented and validated** (`feeder_flexure.ipynb`) — fourth `src/flexsim` example,
the actual mechanism this tool exists to design, built entirely from pieces validated in the
first three examples with no changes to `src/flexsim` itself.

## Geometry

A fixed ground body; two independently-actuated input rigid bodies (top and bottom); a jaw rigid
body. `N` horizontal beams connect ground to each input; `N` angled beams connect each input to
the jaw (top beams angled down by `theta`, bottom beams angled up by the same angle, closing the
gap down to a small vertical separation `d_jaw_gap` at the jaw). `M_contact` penalty contact
points sit along the jaw's right side against a flat surface (standing in for the wire).

Every "a distance" from the verbal spec is a named, adjustable parameter: `N`, `d_stack` (vertical
spacing between the `N` parallel beams in a group), `d_offset` (horizontal offset from a
horizontal beam's input-side anchor to the corresponding angled beam's input-side anchor),
`theta_deg`, `d_jaw_gap`, `Z_hub`/`L_horiz` (input hub height / horizontal beam length),
`d_contact_gap`/`d_contact_spacing`/`M_contact` (jaw contact point placement), `contact_clearance`/
`k_contact` (penalty contact surface offset and stiffness). `L_angled` and the jaw's horizontal
position `x_jaw` are *derived*, not independent — the angled beams' vertical rise over `d_offset`
must exactly close the gap between the input hubs down to `d_jaw_gap`, so the geometry is always
self-consistent for any other parameter choice.

## What's new in `src/flexsim`

Nothing — this example is a pure application of machinery already built and validated in the
first three examples: `Frame.solve_path` (plain multi-leg load path, from
`parallel_beam_flexure_contact`), `Frame.add_penalty_contact` (compliant one-sided spring contact,
added alongside that same example), and the rigid-body anchor-offset kinematics (`rigid.py`, from
`pure_bending_cantilever`/`parallel_beam_flexure`). This was itself a useful check: the
generalized anchor-offset chain rule and multi-leg solver composed cleanly into an 8-beam,
4-contact-point mechanism with no further changes needed.

## A slenderness gotcha (found and fixed here)

The first attempt used `thickness=0.01` (closer to a "real" flexure), which gives the angled
beams (`L_angled` ~2.2) a slenderness `L/t` of 150-220. At that slenderness, the local
per-element Newton solve (`jirasek.solve_local_end_forces`) is so ill-conditioned that it becomes
extremely slow to converge for the actuation strokes used below, and can fail outright even with
its own adaptive sub-stepping. Profiling directly (timing `solve_path` on the same geometry at
different `thickness`) confirmed this: `thickness=0.05` (`L/t` ~30-45) converges the full 4-leg
sequence in ~12s; `thickness=0.01` did not converge within a much longer budget. `thickness` is
still a free parameter, worth revisiting once real material/cross-section parameters are
available — this default is chosen for solver robustness, not physical accuracy.

## Actuation sequence

The tunable sequence requested: (1) both inputs move up, (2) top moves back down to zero (bottom
stationary), (3) both move down together, (4) bottom moves back up to zero (top stationary).
Simulated once, not repeated, via `Frame.solve_path` with 4 legs. Each input's `w` DOF is driven
directly; `F_x` and `M` stay free at both inputs throughout (the same mixed-BC convention used in
`parallel_beam_flexure`/`parallel_beam_flexure_contact`).

## What the notebook does

1. Builds the parametric geometry (`build_feeder`), prints derived quantities (`L_angled`,
   `x_jaw`).
2. Renders the initial (undeformed) geometry, including the contact points and surface.
3. Solves the 4-leg actuation sequence, plots driven input displacement and jaw response over the
   sequence.
4. Reports the final-state contact force and penetration at each of the jaw's contact points —
   tying back to the "distribution of clamping stress" motivation from the penalty-contact
   discussion in `parallel_beam_flexure_contact`: a compliant contact law lets each point carry a
   different load depending on its own penetration, which a rigid/active-set constraint can't
   show.
5. Animates the deformed shape at every solved step using **matplotloom**
   (`feeder_flexure_actuation.mp4`), as requested.

## Result

The full sequence solves cleanly (8 beams, 4 penalty contact points, 60 total steps in ~12s).
The final-state contact-force distribution across the jaw's contact points is uneven — the same
parasitic-rotation effect already characterized in `parallel_beam_flexure`/
`parallel_beam_flexure_contact`, now visible as an actual clamping-force gradient along the jaw.

## Open items / future work

- `k_contact` is still an arbitrary placeholder stiffness, as in `parallel_beam_flexure_contact`.
  Hertzian line-contact theory (line-on-flat contact stiffness from material modulus and edge
  radius) remains the natural next step for a physically meaningful value, especially for a
  softer 3D-printed jaw material.
- No friction/tangential (feed) force yet — `derivations/contact_friction_model.md`'s stick-slip
  model is not wired in; this example only evaluates the normal (clamping) direction.
- No stress/buckling margin reporting yet (`derivations/beam_stiffness_euler_bernoulli.md`'s
  `_stress` output, and a buckling check on the most heavily axially-loaded beams) — flagged in
  the original scaffold as an eventual "what to check" item, still open.
- `thickness` chosen for solver robustness (see above), not tuned against a real cross-section.
