# Example: Feeder V2 (3D, all six jaws, flexible wire, wire-in-tube)

Quasi-static, geometrically nonlinear simulation of the complete V2 feeder:
- 6 jaw modules, each with two leaf stacks in a 60° V per level
- a flexible wire gripped by line contact with Coulomb friction
- 250 mm of that wire buckling inside the 1 mm ID spindle tube
- a spring at the nozzle, so the feed load builds until the jaws slip

Package: `src/flexsim3d/` (the 2D `src/flexsim/` is untouched).
Formulation and validation notes: `derivations/corotational_beam_3d.md`.

## Run it

```bash
pip install -r requirements-3d.txt                 # numpy, scipy, jax, matplotlib
python tests/test_flexsim3d_validation.py          # 13 closed-form checks, ~30 s
python examples/feeder_v2_3d/run_feeder.py --revs 3 --set k_feed=300 --out examples/feeder_v2_3d/results/tube.npz
python examples/feeder_v2_3d/make_figures.py examples/feeder_v2_3d/results/tube.npz
```

- Override any `FeederParams` field with `--set`, e.g. `--set tube_length=0 E_flex=3000 mu=0.25`.
- A 3-revolution run with the tube (32k DOF, 1080 cam steps) takes ~45 min on
  2 cores; without the tube, ~25 min.
- `make_figures.py` writes slide PNGs and `feeder_viewer.html`, a
  self-contained interactive viewer. Open it in a current browser; it needs
  internet to load three.js. The viewer has a leaf inspector (click any leaf)
  and a wire-in-tube panel.

Studies:

| Script | What it does | Time |
|---|---|---|
| `module_stiffness.py` | single module: clamp force vs. interference vs. Appendix A | ~15 s |
| `element_convergence.py` | elements per leaf (2–16) | ~2 min |
| `tube_validation.py` | wire-in-tube friction loss vs. helical buckling theory | ~1–2 min |

## Model

| Part | Representation |
|---|---|
| Leaves (720) | **8** corotational 3D beam elements each (Timoshenko + bowing term): both bending axes, torsion, axial; 50 µm initial bow, random sign |
| Jaws, input bodies | rigid bodies (6 DOF); leaves attach at offsets, no constraint equations |
| Cams | input-body z prescribed at the cam-bearing point; other 5 DOF free (no side load from cam slope) |
| Wire | 0.9 mm Al beam: 0.5 mm elements through the jaws, 1 mm in the tube; 40 µm bow + helix imperfection in the tube (spool cast) |
| Wire BCs | top: lateral guide; nozzle: lateral + twist + axial spring `k_feed` |
| Jaw-wire contact | each wire node vs. each jaw edge line, penalty `k_line` (Hertz line contact ≈ 1e4 N/mm²), C1-smoothed |
| Tube | rigid straight cylinder, ID 1.0 mm (50 µm radial clearance), penalty + friction `mu_tube` = 0.3; `tube_ecc` bends it into the orbital flexure's static shape |
| Friction | Coulomb with elastic stick and memory (return mapping), jaws `mu` = 0.2 |

Cam schedule: the jaw path of Fig. A.3 (clamp, push, release, reset),
converted to input motions with the rigid-link inverse kinematics
(A.13/A.14), φ = 30° half-angle. Pairs (k, k+3) are in phase; the three pairs
are 60° of cam rotation apart.

Assumed dimensions (not in the paper; all `FeederParams` fields): `pitch`
1.5 mm, `r_apex` 4.5 mm, `s0` 2 mm, `d_input` 4 mm.

## Results (E = 1 GPa, 3 cam revolutions, k_feed = 300 N/mm)

| | No tube (`notube`) | 250 mm tube (`tube`) |
|---|---|---|
| Force at nozzle, steady state (min / mean / max) | 88 / 102 / 124 N | 95 / 100 / 112 N |
| Force lost in the tube (mean / max) | – | 1.8 N (1.7%) / 4.9 N |
| Wall contact (share of in-tube length) | – | 82% |
| Worst leaf P / P_cr | 0.96 | 0.96 |

- **Clamp force is buckling-limited.** One module (`module_stiffness.png`):
  initial stiffness 2317 N/mm per jaw, matching A.6 with cos²(φ/2). A.6 as
  written gives 2675. The curve peaks at about 410 N at 0.21 mm interference,
  right at the 0.2 mm design value, and then softens as leaves buckle. P_cr ∝ E,
  so forces scale with E (v1 run at E = 3 GPa: about 3×).
- **The top level is the critical stack.** Top leaves run at 0.93–0.96 P_cr
  through every grip; bottom leaves at ~0.65.
- **Stiff-axis ("wide") bending from the V over-constraint is real but
  secondary** (`stress_components.png`): peak 1.9–2.1 MPa, vs. 13–14 MPa
  thin-direction bending, 3.5 MPa axial and 0.3 MPa torsion in the top stacks.
- **Load shifts toward the top of the jaw** as the feed load builds during a grip.
- **Slip happens at handover, not mid-push.** Each pair slips as it releases,
  which caps the feed force at ~100–120 N.
- **The tube costs only ~2–5%.** The wire buckles helically against the wall,
  but with 50 µm clearance the wall load is only ~0.05 N/mm, consistent with
  w = rF²/(4EI). Wall friction also smooths the force at the nozzle. A straight
  tube does not explain the gap to the measured 65 N. Still open: tube
  bending/orbit, wire cast, the nozzle (softened wire swelling, PCD friction),
  μ, and the cam profiles.

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
- Keep imperfections realistic. With a 5 µm leaf bow, leaves can sit on the
  unstable straight branch above P_cr; a near-straight wire never buckles into
  the tube wall at all.
- Mesh: 8 elements per leaf is converged (8–16 agree within 1%);
  4 elements is ~15% too stiff post-buckling (`element_convergence.png`).
- `--eig-every N` records the lowest eigenvalue of the symmetrized tangent.
  During sliding friction the tangent is non-symmetric, so an isolated negative
  sample at a handover is not by itself an instability.

## Next steps

- Bent or orbiting tube (`tube_ecc`), wire cast curvature, nozzle friction.
- Real cam profiles from CAD in place of the idealized piecewise-linear schedule.
- Jaw holder / blade compliance (currently a rigid jaw + Hertz contact).
