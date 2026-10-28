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
python examples/feeder_v2_3d/run_feeder.py --revs 2 --set k_feed=300 clearance=0.15 --out examples/feeder_v2_3d/results/e3000/tube325_gap0.15.npz
python examples/feeder_v2_3d/make_figures.py examples/feeder_v2_3d/results/e3000/tube325_gap0.15.npz
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
   revolutions; with `k_feed=300`, 2 reach steady state when there is a jaw gap,
   zero gap at 3 GPa needs 3), `--dpsi` (cam step,
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
- **M1 iso**: an isometric-style view of module 1 on its own (other modules off).
- **Legend**: drag it anywhere in the 3D view; double-click puts it back. The
  position is remembered per browser.
- **Save** (export row): renders the last 180° of the run with the current camera,
  colours, visible modules, deformation scale and legend, and saves it.
  - Format: **MP4** (H.264; set "Loop until stopped" in PowerPoint for a GIF-like
    loop; VP9 in .mp4 if the browser has no H.264 encoder, .webm as a last
    resort) or **GIF** (256 colours per frame; files are several times larger, and
    frame delays round to 10 ms, so 15 frames/s plays at about 14).
  - Width: as on screen, or 960–3840 px (capped at the GPU's limit, at most 4096).
    Shape: as on screen, 16:9, 4:3 or 1:1. A different shape widens or narrows the
    view sideways; the vertical framing stays as on screen. Text and legend scale
    with the frame height, and the legend keeps its relative position.
  - The selected-leaf highlight is left out unless "highlight selected leaf" is ticked.
  - Frame spacing is that of the stored frames (about 3° of cam per frame);
    `make_figures.py --viewer-revs 0.5` puts all stored frames into the last 180°.
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
| Cams | input-body z prescribed at the cam-bearing point; other 5 DOF free (no side load from cam slope). Optional: `k_cam` > 0 puts a z spring (the cam stack's compliance) between the cam profile and each input body; `input_scale` scales all input motions |
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

## Results (CAD geometry, E = 3 GPa, μ = 0.2, k_feed = 300 N/mm, steady state = last 180°)

**E = 3 GPa (HT-PLA-GF datasheet).** Runs before 2026-10-28 used 1 GPa in
error; they are kept in `results/cad/` for reference only. Current results:
`results/e3000/`.

### What explains the measured 65 N? (`results/e3000/sweep_summary.png`, no tube)

| Case | Feed force at nozzle (min / mean / max) | Clamp per jaw (full grip) | Worst leaf P/P_cr |
|---|---|---|---|
| CAD (rigid mounts, no gap) | 266 / 284 / 320 N | 1008 N | 0.97 |
| Jaw gap 0.10 mm | 189 / 205 / 227 N | 487 N | 0.71 |
| Jaw gap 0.125 mm | 127 / 141 / 162 N | 321 N | 0.48 |
| Jaw gap 0.14 mm | 88 / 99 / 120 N | 221 N | 0.34 |
| Jaw gap 0.15 mm | 62 / 72 / 92 N | 155 N | 0.24 |
| Jaw gap ≥ 0.175 mm | 0: the jaw stroke never reaches the wire | 0 | – |
| Mounts 30,000 N/mm | 250 / 275 / 308 N | 856 N | 0.91 |
| Mounts 10,000 N/mm | 198 / 221 / 247 N (still rising ~4% per half rev at 2 revs) | 598 N | 0.78 |

With the 325 mm tube at 0.15 mm (`tube325_gap0.15`): **63 / 66 / 74 N against
the measured 65 N**, clamp 155 N per jaw, tube friction loss 0.3 N (0.5%).
With the tube at zero gap (`tube325_gap0`, 3 revs) the force is still rising
(mean 237 → 255 N over the last two half revolutions; the buckled wire adds
compliance, so it builds more slowly). Clamp is 1002 N per jaw and the tube loss
2.7%, so it should settle a little below the no-tube 284 N; it needs about 6
revolutions to confirm.

- **A jaw gap of about 0.15 mm explains the measurement.** The design
  interference is δc/2 = 0.2 mm and the free jaw stroke runs out between 0.15
  and 0.175 mm, so a gap takes interference away one-for-one and the curve is
  steep near the end: 0.1 → 0.15 mm cuts the feed force from 205 to 72 N.
  The gap also shortens each grip window (the jaws touch later and let go
  sooner), and slip rises from 37% to 74% of the contact time.
- **Mount compliance is not the explanation.** A solid printed root block at
  3 GPa is EA/L ≈ 3000 × 110 / 5–10 ≈ 3–7 × 10⁴ N/mm. At 30,000 N/mm the clamp
  force drops 15% and the feed force barely moves; even at 10,000 N/mm (3–7×
  softer than estimated) the feed force is still above 200 N. Feed force at zero gap is
  limited by stroke and handover slip, not friction capacity.
- **Cam-stack compliance acts like extra gap.** With `k_cam` (a spring per
  input body between the cam profile and the body: Hertz contacts, follower
  bearings and pin, printed cams, frame), at 0.15 mm gap, no tube:

  | Cam stack stiffness per input | Feed force (min / mean / max) | Clamp per jaw | Max input stroke lost |
  |---|---|---|---|
  | rigid | 62 / 72 / 92 N | 155 N | 0 |
  | 3,000 N/mm | 55 / 65 / 85 N | 139 N | 15 µm |
  | 1,000 N/mm | 44 / 54 / 73 N | 111 N | 38 µm |

  The lost input stroke δ shows up at the jaw as about 0.31 δ of lost radial
  interference (tan θ / cos φ with θ = 15°, φ = 30°), the same as that much more
  gap. Unlike a uniform cut in the input motions (`input_scale`), the loss is
  proportional to the cam force, so it bites only while clamping. One measured
  feed force cannot separate gap from stack compliance: 65 N fits 0.15 mm with
  a rigid stack or roughly 0.14 mm with ~3,000 N/mm. Measuring the jaw gap on
  the assembled feeder (feeler or pin gauge) would separate them.
- **The gap protects the leaves.** At 0.15 mm the worst leaf sits at 0.25 P_cr
  and peak thin-direction bending is 13 MPa; at zero gap the top stacks are at
  0.97 P_cr (buckling-limited) with 42 MPa. Compare these with the material's
  strength and fatigue data for the print settings used.
- **Design point.** The tolerance on the press-fit sets the operating point
  directly: every 0.025 mm of gap near 0.15 mm is worth about 30–40 N of feed
  force, and closing the gap trades leaf and cam margin for feed force.

| Stress / load (last 180°) | Gap 0 (no tube) | Gap 0.15 mm, 325 mm tube |
|---|---|---|
| Peak thin-bending / axial / wide-bending / torsion stress | 41.9 / 10.5 / 5.2 / 1.0 MPa | 13.3 / 2.7 / 5.2 / 0.2 MPa |
| Worst leaf P/P_cr, top / bottom stacks | 0.97 / 0.78 | 0.25 / 0.13 |
| Contact time spent slipping | 37% | 66% |

Single module (`module_stiffness_E3000.png`): initial clamp stiffness
6862 N/mm per jaw, matching A.6 with cos²(φ/2) (6951; A.6 as written gives
8026); 1058 N at the 0.2 mm design interference. Other findings that still hold:
the top level is the critical stack, load shifts toward the top of the jaw
during a grip, and each pair slips at handover, not mid-push.

### Cam loads and contact stress on the printed cams (`results/e3000/cam_force.png`, `cam_stress.png`)

The inputs are driven by prescribed z, so the force the cam must supply is the
reaction at that DOF (`Feeder.input_forces`; recorded per step as `inp_fz` in
new runs, recomputed from the stored frames for older ones). Each input body's
other five DOFs balance to within 0.001 N, which checks the recomputation.

Stack per stackup, bottom to top: outer bearing | bottom cam (face up) |
lower follower (one bearing down, one up) | middle-lower cam (face down) +
middle-upper cam (face up) | upper follower | upper cam (face down) | outer
bearing. The bottom input bodies ride on the lower follower (between the bottom
and middle-lower cams), the top input bodies on the upper follower. Follower
force per stackup = (sum over the two jaws of the group) / `n_stackups`. That
force sets the stress through Hertz, so stress scales with √(force/N), not
force/N.

```bash
python examples/feeder_v2_3d/cam_loads.py results/e3000/gap0.npz results/e3000/tube325_gap0.15.npz
```

`CamStack` defaults (`src/flexsim3d/cam_contact.py`; override with
`--set name=value`): 3 mm ID bearings, Ø13 × 5 mm; 1 bearing per cam face
per stackup; 2 stackups per jaw group; cam E = 3 GPa (same HT-PLA-GF as the
flexures). The stack preload is the assembly displacement, which is already the
ramp at the start of every run, so `preload` = 0.

| | Gap 0 (design) | Gap 0.15 mm (matches 65 N) |
|---|---|---|
| Upper follower, max down force per stackup → upper cam (face down) | 150 N | 47 N |
| Lower follower, max up force per stackup → bottom cam (face up) | 121 N | 33 N |
| Middle cams (reset / handover only) | 22 N | 22 N |
| Peak contact pressure p_max, upper / bottom cam | 70 / 63 MPa | 39 / 33 MPa |
| Interlayer shear 0.25 p_max (planes parallel to the face), upper cam | 17.6 MPa at 136 µm | 9.8 MPa at 76 µm |
| Contact half-width a, upper cam | 0.27 mm | 0.15 mm |

- The face-down **upper cam is the most loaded surface**. The middle cams only
  see the ~22 N left over from the assembly displacement.
- **The prototype that did not delaminate ran at about 10 MPa of interlayer
  shear** (the 0.15 mm case), 0.05–0.1 mm below the surface. Closing the gap
  to the design value would raise it to about 18 MPa.
- **Ramp slope is left out** (`cam_radius` = 0). On the 85 mm track the steepest
  ramp is 1.6°, so the normal force rises by only 0.04%. The side force it adds
  (≈ 2.8% of the vertical load) is not modelled, because the sim prescribes z
  only; quantifying it needs the cams in the model.

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
  lowers the peak clamp force by about 12% (376 → 330 N at 1 GPa; forces
  scale with E). The default is 50 µm,
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
