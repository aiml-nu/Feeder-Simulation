# Current Flexure Design

Working notes describing the flexure topology as of the updated design (figure shared 2026-07-16
— top half only, "Single Flexure" + "Simplified Representation"). This supersedes the geometry in
the old `Dual_Cam_OG` class but keeps the same underlying rigid-body/beam topology.

*(TODO: save the figure itself into `design/figures/` and reference it here — it was shared
inline in chat, not as a file, so isn't in the repo yet.)*

## Rigid bodies

- **Ground** — leftmost body, spans the full height, fully fixed (`constraints = [True, True,
  True]`). Equivalent to `R1` in `Dual_Cam_OG`.
- **Two independent inputs** — one above centerline, one below (mirror images about the wire's
  horizontal symmetry axis). Each moves in `y` only (`constraints = [False, True, False]` per the
  old design, i.e. `y` prescribed, `x`/`θ` free) — these are the actuated DOF that open/close the
  jaw. Equivalent to `R2`/`R3`.
- **Jaw** — rightmost body, a `Rigid_Contact` in distributed contact with the wire along its
  whole length (`d3`/`L` in the old parameterization), rather than a single point-force contact.
  Sees a normal (perpendicular, compressive) force from clamping plus a tangential (along-wire)
  force from the AFSD feed process — see `derivations/contact_friction_model.md`. Equivalent to
  `R4`.

## Beams (two stages per input, per figure labels)

- **`N1` beams** — connect ground to each input body. Roughly horizontal (parallel to the wire
  axis) in the figure. `N1` of these per input side (4 shown). Equivalent to the old
  `N_gi` "ground-input" beams.
- **`N2` beams** — connect each input body to the jaw. Angled, converging toward the jaw. `N2` of
  these per input side. Equivalent to the old `N_ji` "jaw-input" beams.

So each half (top or bottom) of the flexure is a two-stage compliant mechanism: ground -> input
(via `N1` parallel beams, giving the input body a controlled, mostly-vertical compliance) ->
jaw (via `N2` angled beams, converting input vertical motion into jaw clamping displacement +
mechanical advantage). Mirrored top/bottom about the wire centerline, both stages feed into the
same jaw body.

## "Simplified Representation" — reduced kinematic model

The lower half of the figure collapses each `N1`/`N2` beam group into a single equivalent rigid
link mechanism for intuition: a link `L1` (ground -> pinned joint) in line with the symmetry axis,
and a link `L2` (pinned joint -> jaw, at angle `θ` to the symmetry axis) delivering force `F` to
the jaw along the symmetry line. This is a standard technique for reasoning about a compliant
mechanism's kinematics (motion amplification, mechanical advantage) via an equivalent rigid
linkage, useful for parameter studies (how `θ`, `L1`, `L2` trade off stiffness vs. stroke vs.
force) before committing to the full multi-beam FE model. Worth formalizing as a quick
closed-form check against the full simulation (`design/roadmap.md`).

## Loading / contact picture

Unlike the simplified single-point-force `F` shown in the figure (drawn as a single compressive
arrow for clarity), the actual jaw interaction is:

- **Normal/perpendicular** — distributed contact along the jaw length against the wire, active
  once penetration is detected at the top and/or bottom contact point independently
  (`Rigid_Contact.check()`), not necessarily engaging uniformly.
- **Tangential/along-wire** — friction force from the AFSD process pushing the wire through,
  capped by both Coulomb friction capacity (`mu * normal`) and the AFSD process force
  requirement (`F_AFSD`) — see `derivations/contact_friction_model.md` for the stick-slip logic
  and an open question about that force cap's physical meaning.

## Differences from `Dual_Cam_OG`

- Same overall 4-body / 2-stage-per-side topology, but the jaw's contact model is the
  distributed-length `Rigid_Contact` (top + bottom contact points, independent activation)
  rather than treated as a simple point load in the old `simulate()` loop.
- Old code hard-coded loading (preload + cyclic top/bottom sweep) into the flexure class itself;
  new structure should separate topology from loading schedule (`design/architecture.md`,
  "Contact and time-stepping").
- Parametrization (`N1`, `N2`, lengths, angles, gaps, thicknesses) should carry over largely
  unchanged from `Dual_Cam_OG.__init__`'s geometry-construction logic, just re-homed into
  `flexures/dual_cam.py` per the proposed layout.

## Open items to confirm

- Exact values / ranges for `θ`, `L1`, `L2`, `N1`, `N2` for the current physical prototype — not
  yet captured here, pull from CAD or current notebook parameters when available.
- Whether the two inputs are driven by identical schedules (symmetric) or intentionally
  asymmetric in some operating modes (e.g. one input over-driven to bias the jaw) — matters for
  how `sim/loading.py` should be structured.
