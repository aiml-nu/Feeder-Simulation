# Example: Feeder V2 (3D, all six jaws, flexible wire, wire-in-tube)

Quasi-static, geometrically nonlinear simulation of the complete V2 feeder:
- 6 jaw modules, each with two leaf stacks in a 60° V per level
- a flexible wire gripped by line contact with Coulomb friction
- 325 mm of that wire buckling inside the 1 mm ID spindle tube
- a spring at the nozzle, so the feed load builds until the jaws slip

Package: `src/flexsim3d/` (the 2D `src/flexsim/` is untouched).
Formulation and validation notes: `derivations/corotational_beam_3d.md`.

## Run it

```bash
pip install -r requirements-3d.txt                 # numpy, scipy, jax, matplotlib
python tests/test_flexsim3d_validation.py          # 13 closed-form checks, ~30 s
python examples/feeder_v2_3d/run_feeder.py --revs 2 --set k_feed=300 --out examples/feeder_v2_3d/results/tube325.npz
python examples/feeder_v2_3d/make_figures.py examples/feeder_v2_3d/results/tube325.npz
```

- Override any `FeederParams` field with `--set`, e.g. `--set tube_length=0 clearance=0.1 k_mount=10000`.
- Two cam revolutions reach steady state with `k_feed=300`. That takes ~40 min
  on 2 cores with the 325 mm tube, ~15–20 min without.
- `make_figures.py` writes slide PNGs and `feeder_viewer.html`, a
  self-contained interactive viewer. Open it in a current browser; it needs
  internet to load three.js. It plays back the whole run and has a leaf
  inspector (click any leaf) and a wire-in-tube panel.

Studies:

| Script | What it does | Time |
|---|---|---|
| `module_stiffness.py` | single module: clamp force vs. interference vs. Appendix A | ~15 s |
| `element_convergence.py` | elements per leaf (2–16) | ~2 min |
| `tube_validation.py` | wire-in-tube friction loss vs. helical buckling theory | ~1–2 min |
| `module_sweeps.py` | single module: mount stiffness and leaf-bow sweeps | ~1 min |
| `sweep_summary.py` | summary of the full-feeder gap / mount sweeps | seconds |

## Model

| Part | Representation |
|---|---|
| Leaves (720) | **8** corotational 3D beam elements each (Timoshenko + bowing term): both bending axes, torsion, axial; 50 µm initial bow (first-mode shape, all the same way) |
| Jaws, input bodies | rigid bodies (6 DOF); leaves attach at offsets, no constraint equations |
| Printed mounts (optional) | `k_mount` > 0: every stack end roots in a block joined to its body by a 6-DOF bushing (96 blocks) |
| Jaw gap | `clearance`: radial gap between each jaw edge and the wire at the as-printed neutral (CAD = 0) |
| Cams | input-body z prescribed at the cam-bearing point; other 5 DOF free (no side load from cam slope) |
| Wire | 0.9 mm Al beam: 0.5 mm elements through the jaws, 1 mm in the tube; 40 µm bow + helix imperfection in the tube (spool cast) |
| Wire BCs | top: lateral guide; nozzle: lateral + twist + axial spring `k_feed` |
| Jaw-wire contact | each wire node vs. each jaw edge line, penalty `k_line` (Hertz line contact ≈ 1e4 N/mm²), C1-smoothed |
| Tube | rigid straight cylinder, 325 mm, ID 1.0 mm (50 µm radial clearance), penalty + friction `mu_tube` = 0.3; `tube_ecc` bends it into the orbital flexure's static shape |
| Friction | Coulomb with elastic stick and memory (return mapping), jaws `mu` = 0.2 |

Cam schedule: the jaw path of Fig. A.3 (clamp, push, release, reset),
converted to input motions with the rigid-link inverse kinematics
(A.13/A.14), φ = 30° half-angle. Pairs (k, k+3) are in phase; the three pairs
are 60° of cam rotation apart.

Assumed dimensions (not in the paper; all `FeederParams` fields): `pitch`
1.5 mm, `r_apex` 4.5 mm, `s0` 2 mm, `d_input` 4 mm.

## Results (E = 1 GPa, μ = 0.2, k_feed = 300 N/mm, steady state = last 180°)

Baseline with the 325 mm tube (`tube325`):
- force at the nozzle: 94 / 98 / 108 N (min / mean / max)
- tube loss: 1.8 N (1.7%) on average, 5.8 N at most, with wall contact on 75% of the in-tube wire
- worst leaf P/P_cr: 0.95 (top stacks) and 0.72 (bottom)
- peak leaf stresses: thin bending 12.8 MPa, axial 3.4, wide bending 1.7, torsion 0.2 MPa

### What could explain the measured 65 N? (`sweep_summary.png`, no tube)

| Case | Feed force at nozzle (min / mean / max) | Clamp per jaw (full grip) |
|---|---|---|
| CAD (rigid mounts, no gap) | 88 / 102 / 123 N | 348 N |
| Jaw gap 0.05 mm | 71 / 92 / 111 N | 274 N |
| Jaw gap 0.10 mm | 64 / 75 / 90 N | 165 N |
| Jaw gap 0.15 mm | 21 / 28 / 41 N | 53 N |
| Jaw gap 0.20 mm | 0 (the jaws never touch) | 0 |
| Mounts 10,000 N/mm | 86 / 102 / 121 N | 292 N |
| Mounts 3,000 N/mm | 63 / 77 / 92 N | 192 N |

- **A jaw gap acts strongly and very nonlinearly.** The design interference is
  only δc/2 = 0.2 mm, so a gap takes it away one-for-one. 65 N mean
  corresponds to a gap of about 0.11 mm on every jaw, which is plausible for
  the press-fit. A gap also shortens each grip window: the jaws touch later
  and let go sooner, and slip rises from 41% to 69% of the contact time at
  0.1 mm.
- **Mount compliance barely matters until it is very soft.** 10,000 N/mm
  (estimated for solid printed blocks) cuts clamp force by 16% and leaves
  feed force unchanged. Feed force here is limited by stroke and handover
  slip, not friction capacity (348 N × 0.2 × 4 jaws ≈ 280 N available vs.
  102 N delivered). Reaching 65 N needs roughly 2,000 N/mm or softer, which
  would mean much softer prints than estimated.
- **Gap is therefore the more likely single explanation.** To tell them apart
  on the machine: a gap shows up as lost wire advance per revolution and a
  clamp that engages late in the cam cycle; soft mounts show up as lower clamp
  force with normal timing.
- **The tube remains a ~2% effect**, and leaf bow trims the peak clamp force by
  a few percent (`module_sweeps.png`).

Earlier findings that still hold:
- **Clamp force is buckling-limited.** One module (`module_stiffness.png`):
  initial stiffness 2317 N/mm per jaw, matching A.6 with cos²(φ/2). A.6 as
  written gives 2675. The peak of about 385 N sits at the 0.2 mm design
  interference.
- **The top level is the critical stack.**
- **Bending about the leaf's stiff axis (the V over-constraint) is about 15% of
  the thin-direction bending** (`stress_components.png`).
- **Load shifts toward the top of the jaw** during a grip.
- **Each pair slips at handover, not mid-push.**

## Numerical notes

- Newton with an energy-type (Crisfield) line search; a ‖r‖ line search stalls
  near buckling.
- Local snaps (a leaf flipping between buckled shapes, the wire re-arranging
  its helix) are passed with tiny viscous stabilization (Abaqus `*STABILIZE`
  style). The dissipated energy is logged per step (`stab_energy`).
- The in-tube wire's lateral DOFs carry an always-on viscous term η = 1e-4 ×
  K_ii. It removes the zero-stiffness helix-rolling mode that otherwise costs
  20+ Newton iterations per step. It was validated against helical buckling
  theory at feeder-like load rates (`tube_validation.png`). η = 1e-3 delays
  helix formation; keep η ≤ 1e-4.
- **A leaf imperfection is required.** With straight leaves the imposed motions
  barely excite the first buckling mode, so Newton follows the unstable
  straight branch. Leaves then reach 1.7 P_cr with negative tangent
  eigenvalues, overstating clamp force by about 20%; below ~20 µm the solve
  stalls or does the same. From 20 µm up the path is stable, and the bow
  amplitude (print straightness) is then a physical parameter: 20 → 100 µm
  lowers the peak clamp force about 410 → 360 N. The default is 50 µm,
  first-mode shape, every leaf bowed the same way (`imp_sign`). Likewise, a
  near-straight wire never buckles into the tube wall.
- Mesh: 8 elements per leaf is converged (8–16 agree within 1%);
  4 elements is ~15% too stiff post-buckling (`element_convergence.png`).
- `--eig-every N` records the lowest eigenvalue of the symmetrized tangent.
  During sliding friction the tangent is non-symmetric, so an isolated negative
  sample at a handover is not by itself an instability.

## Next steps

- Bent or orbiting tube (`tube_ecc`), wire cast curvature, nozzle friction.
- Real cam profiles from CAD in place of the idealized piecewise-linear schedule.
- Jaw holder / blade compliance (currently a rigid jaw + Hertz contact).
