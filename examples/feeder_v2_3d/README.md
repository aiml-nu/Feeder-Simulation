# Example: Feeder V2 (3D, all six jaws, flexible wire)

Status: **implemented, validated element, first results.** Quasi-static,
geometrically nonlinear simulation of the complete V2 feeder: 6 jaw modules,
each with two leaf stacks in a 60° V per level, a flexible wire held by
line contact with Coulomb friction, and a spring at the wire's bottom end so
the feed load builds until the jaws slip.

Package: `src/flexsim3d/` (new; the 2D `src/flexsim/` is untouched).
Formulation notes: `derivations/corotational_beam_3d.md`.

## Run it

```bash
pip install -r requirements-3d.txt             # numpy, scipy, jax, matplotlib
python tests/test_flexsim3d_validation.py      # 13 closed-form checks, ~30 s
python examples/feeder_v2_3d/run_feeder.py --revs 3 --out examples/feeder_v2_3d/results/base.npz
python examples/feeder_v2_3d/make_figures.py examples/feeder_v2_3d/results/base.npz
```

Override any parameter: `--set E_flex=3000 mu=0.25 k_feed=200`. A 3-revolution
run (1080 cam steps, 13.7k DOF) takes ~15 min on 2 cores; the first ~20 s are
JAX compilation.

`make_figures.py` writes PNGs for slides and `feeder_viewer.html`, a
self-contained interactive 3D viewer (open in a browser; needs internet for
three.js from the CDN).

## Model

| Part | Representation |
|---|---|
| Leaves (720) | 4 corotational 3D beam elements each (Timoshenko + bowing term), both bending axes + torsion + axial; 50 µm initial bow, random sign |
| Jaws, input bodies | rigid bodies (6 DOF); leaves attach at offsets, no constraint equations |
| Cams | input-body z prescribed at the cam-bearing point; other 5 DOF free (no side load from cam slope) |
| Wire | 0.9 mm Al beam, 0.5 mm elements, 5 mm below / 3 mm above the jaws |
| Wire BCs | top: lateral guide; bottom: lateral guide + twist + **axial spring `k_feed`** |
| Jaw-wire contact | each wire node vs each jaw edge line, penalty `k_line` (Hertz line contact ≈ 1e4 N/mm²), C1-smoothed |
| Friction | regularized Coulomb, slip measured per increment, `mu`=0.2, `eps_slip`=0.1 µm |

Cam schedule: jaw path of Fig. A.3 (clamp, push, release, reset), converted to
input motions with the rigid-link inverse kinematics (A.13/A.14), φ = 30° half
angle. Pairs (k, k+3) in phase; the three pairs 60° of cam apart.

### Assumed dimensions (not in the paper; all `FeederParams` fields)

`pitch` = 1.5 mm leaf spacing, `r_apex` = 4.5 mm, `s0` = 2 mm, `d_input` = 4 mm,
`imperfection` = 0.05 mm. Check against CAD.

## Validation (`tests/test_flexsim3d_validation.py`)

Linear tip load in both planes (Timoshenko), torsion, 90° pure bending (exact
arc), Euler buckling (cantilever and fixed-fixed; 0.05% error at 4 el/leaf),
post-buckled elastica (20.3° vs 20.0°), parallel-guide stiffness incl. a stack
rotated 30°. The tangent stiffness matches finite differences to 1e-8.

## First results (3 cam revolutions, μ = 0.2)

| | E = 1 GPa (`base`) | E = 3 GPa (`E3000`, k_feed = 300) |
|---|---|---|
| Steady feed force (min / mean / max) | 94 / 101 / 114 N | 280 / 302 / 342 N |
| Clamp force per jaw, full grip | ~356 N | ~1060 N |
| Worst leaf compression / P_cr | 0.98 | 0.98 |
| Peak leaf stress (axial + bending) | 20 MPa | 60 MPa |

- **Clamp force is buckling-limited.** In the single-module study
  (`module_stiffness.py`) the initial clamp stiffness is 2317 N/mm per jaw,
  matching A.6 with cos²(φ/2) (A.6 as written gives 2675). The curve rolls
  over near the 0.2 mm design interference and plateaus at about 440 N
  (E = 1 GPa) as the most-compressed leaves reach P_cr. Since P_cr ∝ E, every
  force scales with E.
- **The top level is the critical stack.** Top angled and straight leaves sit at
  0.93–0.97 P_cr through every grip; the bottom ones at ~0.65.
- **Load shifts toward the top of the jaw** as the feed load builds during a
  grip, and the bottom of the jaw unloads (`line_load_map.png`,
  `line_load_profiles_jaw1.png`).
- **Slip happens at handover, not mid-push.** In steady state the pushing jaws
  stick. Each pair slips as it releases, and the wire gives back about as much
  as the push gained, which sets the maximum feed force (`slip.png`).

## Numerical notes

- Newton with an energy-type (Crisfield) line search. A plain ||r|| line
  search stalls near buckling.
- Individual leaves can snap between buckled shapes (a genuine local
  instability, detected as the lowest tangent eigenvalue crossing zero). The
  driver retries such increments with tiny viscous stabilization on leaf-node
  DOFs (Abaqus `*STABILIZE` style) and logs the dissipated energy
  (`stab_log` in the .npz; ~1e-5 N·mm when it has triggered, negligible).
- With a 5 µm imperfection, leaves can sit on the unstable straight branch
  above P_cr (P/P_cr up to 1.22). 50 µm (FDM-realistic) keeps them on the
  stable buckled branch. Keep the imperfection realistic.
- `--eig-every N` records the lowest eigenvalue of the symmetrized tangent as a
  stability monitor. During sliding friction the tangent is non-symmetric, so a
  negative value sampled exactly at a handover slip (one of 55 samples in
  `E3000`, at 491°) is not by itself an instability. The neighbouring stored
  state is positive with and without friction.

## Next steps

- 250 mm wire in a tube: add a wire-vs-cylinder contact term (same pattern as
  `contact.WireJawContact`) and lengthen `wire_below`.
- Real cam profiles from CAD in place of the idealized piecewise-linear schedule.
- Jaw holder / blade compliance (currently a rigid jaw + Hertz contact).
