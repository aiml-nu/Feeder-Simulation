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
python examples/feeder_v2_3d/run_feeder.py --revs 2 --set k_feed=300 --out examples/feeder_v2_3d/results/cad/tube325_gap0.npz
python examples/feeder_v2_3d/make_figures.py examples/feeder_v2_3d/results/cad/tube325_gap0.npz
```

### Parameter change -> run -> viewer (manual workflow)

1. **Pick parameters.** Every geometry, material, contact and solver input is
   a field of `FeederParams` in `src/flexsim3d/feeder.py` (with units and a
   comment). Either edit the default there, or override per run with
   `--set name=value ...` (no spaces around `=`), e.g.
   `--set clearance=0.1 tube_length=0 k_mount=10000 E=1200`.
   The values actually used are stored in the output `.npz` (`params`), so a
   result always carries its own inputs.
2. **Run.** `run_feeder.py --revs 2 --set ... --out <name>.npz`.
   Progress prints one line per 10° of cam. Useful flags: `--revs` (cam
   revolutions; 2 reaches steady state with `k_feed=300`), `--dpsi` (cam step,
   default 1°), `--eig-every N` (stability check). `tube_length=0` drops the
   tube and is ~2× faster.
3. **Figures + viewer.** `make_figures.py <name>.npz` writes
   `<name>_figs/`: slide PNGs and `feeder_viewer.html`. Open that file in a
   browser (it needs internet for three.js). `--viewer-frames` sets how many
   frames are embedded (default 240 over the whole run).
4. **Compare runs.** `sweep_summary.py [results_dir]` tabulates and plots the
   gap / mount sweep (it skips runs that are missing).

### How a run starts (rest -> in action)

The model starts in the as-printed, unloaded state: all leaves at their
CAD shape (plus the 50 µm bow), all input bodies at z = 0, and each jaw edge
`clearance` away from the wire. There is a separate **ramp** before the cam
turns: over `n_ramp` = 10 load increments the input bodies are moved
linearly from 0 to their cam positions at ψ = 0 (different for each pair,
because the pairs are 60° of cam apart). Only then does the cam advance in
1° steps. The ramp increments are stored with `ramp < 1` and are excluded
from the steady-state statistics. Because the problem is quasi-static and
path-dependent only through friction, the ramp just has to be gentle enough
for Newton; it has no physical duration.

### Viewer controls

- **View** buttons: cutaway / full / wire + jaws / tube; M1–M6 show or hide modules.
- **Leaf colour**: the field (compression / buckling load P/P_cr, combined stress,
  axial, thin-direction bending, wide-direction bending, torsion), the min and
  max colours, and the value mapped to the max colour ("Run maximum" resets it
  to the largest value in the run). The legend follows.
- **Contact load** checkbox: shows or hides the line-load markers on the wire,
  the tube-contact colouring and their legend.
- Click a leaf for the **leaf inspector** (axial force, both bending moments
  and torsion along the leaf, through the run).

Studies:

| Script | What it does | Time |
|---|---|---|
| `module_stiffness.py` | single module: clamp force vs. interference vs. Appendix A | ~15 s |
| `element_convergence.py` | elements per leaf (2–16) | ~2 min |
| `tube_validation.py` | wire-in-tube friction loss vs. helical buckling theory | ~1–2 min |
| `module_sweeps.py` | single module: mount stiffness and leaf-bow sweeps | ~1 min |
| `sweep_summary.py [dir]` | summary of the full-feeder gap / mount sweeps | seconds |
| `cam_loads.py run.npz [...] --set ...` | cam follower forces and Hertz contact stress on the printed cams | ~30 s per run |

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

Dimensions from CAD (`FeederParams`):
- `pitch` = 1.27 mm: vertical leaf spacing within a stack.
- `stack_gap` = 4.18 mm: vertical gap between the anchors of the lowest
  angled top leaf and the highest angled bottom leaf, centred on the 45 mm jaw.
- `d_input` = 9.6 mm: input body length along the arm (end of the straight
  leaves to start of the angled leaves).
- `root_to_jaw_face` = 19.14 mm: plan (top-view) distance from the jaw-side
  root of the angled leaves to the middle of the jaw's flat face, same for
  both arms. With `r_apex` = 4.5 mm this puts the roots s_root = 15.5 mm out
  along each arm from the V apex.
- Still assumed: `r_apex` 4.5 mm (V apex radius from the wire axis). Older
  parameter sets used `s0` directly; set `root_to_jaw_face=0` to get that back.

## Results (CAD geometry, E = 1 GPa, μ = 0.2, k_feed = 300 N/mm, steady state = last 180°)

Result files are in `results/cad/` (`results/*.npz` from before the CAD update
used the assumed dimensions and are kept for reference).

With the 325 mm tube (`tube325_gap0`, `tube325_gap0.1`; one viewer each):

| | CAD, no gap | Jaw gap 0.1 mm |
|---|---|---|
| Force at nozzle (min / mean / max) | 90 / 94 / 103 N | 65 / 69 / 76 N |
| Clamp per jaw (full grip) | 336 N | 164 N |
| Tube friction loss (mean / max) | 1.6 N (1.6%) / 5.3 N | 0.6 N (0.9%) / 2.6 N |
| Worst leaf P/P_cr, top / bottom stacks | 0.97 / 0.78 | 0.72 / 0.39 |
| Peak thin-bending / axial / wide-bending / torsion stress | 14.1 / 3.5 / 1.8 / 0.3 MPa | 4.4 / 2.6 / 1.7 / 0.1 MPa |
| Contact time spent slipping | 40% | 62% |

The tube's own friction loss is small in both cases. The tube still lowers the
0.1 mm-gap mean from 74 to 69 N because the buckled wire column adds axial
compliance: part of each jaw stroke goes into compressing the helix instead
of the nozzle spring.

### What could explain the measured 65 N? (`results/cad/sweep_summary.png`, no tube)

| Case | Feed force at nozzle (min / mean / max) | Clamp per jaw (full grip) |
|---|---|---|
| CAD (rigid mounts, no gap) | 84 / 97 / 117 N | 336 N |
| Jaw gap 0.05 mm | 68 / 89 / 108 N | 270 N |
| Jaw gap 0.10 mm | 63 / 74 / 89 N | 163 N |
| Jaw gap 0.15 mm | 21 / 27 / 40 N | 52 N |
| Jaw gap 0.20 mm | 0 (the jaws never touch) | 0 |
| Mounts 10,000 N/mm | 82 / 98 / 118 N | 286 N |
| Mounts 3,000 N/mm | 62 / 76 / 91 N | 190 N |

- **A jaw gap acts strongly and very nonlinearly.** The design interference is
  only δc/2 = 0.2 mm, so a gap takes it away one-for-one. With the tube,
  0.1 mm gives 69 N against the measured 65 N, so the measurement is
  consistent with a press-fit gap of roughly 0.1 mm on every jaw. A gap also
  shortens each grip window: the jaws touch later and let go sooner, and
  slip rises from about 40% to 60–70% of the contact time.
- **Mount compliance barely matters until it is very soft.** 10,000 N/mm
  (estimated for solid printed blocks) cuts clamp force by 15% and leaves
  feed force unchanged. Feed force here is limited by stroke and handover
  slip, not friction capacity (336 N × 0.2 × 4 jaws ≈ 270 N available vs.
  ~95 N delivered). Reaching 65 N needs softer than 3,000 N/mm, which would
  mean much softer prints than estimated.
- **Gap is therefore the more likely single explanation.** To tell them apart
  on the machine: a gap shows up as lost wire advance per revolution and a
  clamp that engages late in the cam cycle; soft mounts show up as lower clamp
  force with normal timing.
- **A gap also protects the leaves**: at 0.1 mm the top stacks stay at
  0.72 P_cr and peak bending stress drops from 14 to 4 MPa, so the clamp is no
  longer buckling-limited. Controlling the press-fit tolerance trades feed
  force against leaf margin directly.

Earlier findings that still hold:
- **Clamp force is buckling-limited at zero gap.** One module
  (`module_stiffness.png`): initial stiffness 2287 N/mm per jaw with the CAD
  geometry (2317 with the old assumed one), matching A.6 with cos²(φ/2). A.6
  as written gives 2675. It reaches 353 N at the 0.2 mm design interference.
- **The top level is the critical stack.**
- **Bending about the leaf's stiff axis (the V over-constraint) is about 15% of
  the thin-direction bending** (`stress_components.png`).
- **Load shifts toward the top of the jaw** during a grip.
- **Each pair slips at handover, not mid-push.**

### Cam loads and contact stress on the printed cams (`results/cad/cam_force.png`, `cam_stress.png`)

The inputs are driven by prescribed z, so the force the cam must supply is the
reaction at that DOF (`Feeder.input_forces`; recorded per step as `inp_fz` in
new runs, recomputed from the stored frames for older ones). Each input body's
other five DOFs balance to within 0.001 N, which checks the recomputation.

Stack per stackup, bottom to top: outer bearing | bottom cam (face up) |
lower follower (one bearing down, one up) | middle-lower cam (face down) +
middle-upper cam (face up) | upper follower | upper cam (face down) | outer
bearing. Assumed: bottom input bodies ride on the lower follower, top input
bodies on the upper follower. Follower force per stackup = (sum over the two
jaws of the group) / `n_stackups`; that force then sets the stress through
Hertz, so the stress scales with √(force/N), not force/N.

```bash
python examples/feeder_v2_3d/cam_loads.py results/cad/tube325_gap0.npz results/cad/tube325_gap0.1.npz \
    --set D_bearing=10 b_contact=4 n_bearings=1 n_stackups=2 E_cam=1000 cam_radius=0 preload=0
```

`CamStack` fields (`src/flexsim3d/cam_contact.py`): bearing OD and contact
length, bearings per surface, stackups per jaw group, cam and bearing E/ν,
the mean track radius on the face cam (> 0 adds the ramp slope:
F_n = F_z / cos α) and a stack preload.

Per jaw group the top inputs need up to 50 N down (gap 0) or 36 N (0.1 mm gap)
while clamping; the bottom inputs need 41 N / 22 N up. With **placeholder
bearings (Ø10 × 4 mm, 1 per surface, 2 stackups, E_cam = 1 GPa)**, the upper
(face-down) cam peaks at p_max ≈ 30 MPa (gap 0) / 26 MPa (0.1 mm), with
contact half-width ≈ 0.25 mm. The shear on planes parallel to the face
(the print layers, if printed face-up) peaks at 0.25 p_max ≈ 6–7.5 MPa about
0.12 mm deep, i.e. at the first layer interface or two.

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
  lowers the peak clamp force about 376 → 330 N (CAD geometry). The default is 50 µm,
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
