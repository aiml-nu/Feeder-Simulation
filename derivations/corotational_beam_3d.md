# 3D corotational beam, rigid bodies, contact (flexsim3d)

Implementation: `src/flexsim3d/{rotation,beam,model,contact,solver,sim}.py`.

## DOFs and kinematics

Every entity (beam node or rigid body) has q = [u (3), ψ (3)] about a reference
point X, with total rotation vector ψ and R = exp(ψ) (Rodrigues). A beam end
attached to entity e at offset o (measured in the reference configuration) sits
at p = X + u + R o, and its triad is T = R E0, where E0 is the element's
initial frame [e1 e2 e3] with e1 along the chord. A rigid body is simply an
entity with offsets, so beam-to-rigid joints need no constraint equations.

`exp` and `log` are written as smooth functions of θ² with series branches near
zero (double-where trick), so autodiff Hessians are finite at ψ = 0.

## Element (Crisfield 1990; Battini & Pacoste 2002)

Corotated frame Er: e1 = (pb − pa)/l; e2 = the average of the end triads'
second axes, orthogonalized against e1; e3 = e1 × e2. Local deformational
rotations: θa = log(Erᵀ Ta), θb = log(Erᵀ Tb), small even under large rigid
rotation.

Strain energy:

    ε  = (l − L0)/L0 + Σ_{i=2,3} (2θai² − θai θbi + 2θbi²)/30        (bowing term)
    U  = ½ EA L0 ε² + ½ GJ/L0 (θb1 − θa1)²
       + Σ_{i=2,3} ½ EIi/(L0(1+Φi)) [(4+Φi)(θai² + θbi²) + 2(2−Φi) θai θbi]
    Φi = 12 EIi / (κ G A L0²)                                           (Timoshenko)

The bowing term is the average axial strain from a cubic deflection shape. It
supplies the consistent geometric stiffness, so a 4-element leaf gets the
fixed-fixed Euler load to 0.02%.

Internal force = ∂U/∂q and tangent = ∂²U/∂q², both from JAX autodiff (vmapped
over elements), so the tangent is exactly consistent.

Section: leaves use e2 = thickness direction, so EI3 = E w t³/12 governs
thin-direction bending and buckling. Rectangular torsion constant from Roark.

## Contact and friction

For each (wire node, jaw) pair the jaw edge is a line p0 + s t with inward
normal n (both rotate with the jaw). Penetration g = r_wire − (c − p0)·n, and
the line load is q = k_line · g_s, with g_s a C1 smoothing of max(g, 0):
quadratic over 0 < g < δ = 2e-5 mm. A C1 smoothstep window tapers the load
over ±0.125 mm at the jaw ends. Nodal force N = q · l_trib.

k_line comes from Hertz line contact (Johnson):
δ = 2q/(πE*) (2 ln(4R/b) − 1), b = √(4qR/(πE*)). For Al on steel at q ≈ 5–20 N/mm
this gives q/δ ≈ 0.9–1.05e4 N/mm².

Friction is Coulomb with elastic stick and memory (a return mapping). Each pair
stores its tangential force f_prev from the last converged step. The trial force
is

    f_tr = P_t(f_prev) − k_t l Δs_t        (Δs_t = tangential slip increment since that step)

and it is limited to |f| ≤ μN with a smooth max:

    f = f_tr · μN / smax(|f_tr|, μN),   smax(a, b) = ½(a + b + √((a − b)² + δ²))

so sticking contacts hold a sustained load without creep and N = 0 clears the
history. `Model.commit()` stores f after every converged increment. (The first
version used a memoryless regularization f = −μN Δs/√(|Δs|² + ε²). That lets
a stuck contact creep by about ε per step and under-mobilizes friction where
sliding is slow.)

## Wire in the tube

Each wire node below the tube entry is tested against a rigid cylinder of inner
radius R_i. The penetration is g = |c_xy − centre(z)| + r_w − R_i, with the same
C1 penalty law, a smoothstep at the entry, and the same Coulomb law with memory
(circumferential and axial). The centreline can be offset as
e(a₂s² + a₃s³), with s the normalized depth: e.g. the clamped-pinned static
shape of the orbital flexure. The default is straight.

Helical post-buckling is a sequence of snaps, so the in-tube wire's lateral DOFs
(u_x, u_y, θ_x, θ_y) carry an always-on viscous term η K_ii (q − q_prev) with
η = 1e-4. Axial DOFs are excluded, so it adds no axial drag. Validation
(`examples/feeder_v2_3d/tube_validation.py`, steps of about 0.5 N as in the
feeder) against the weightless-rod helical buckling solution (Lubinski 1962;
Mitchell 1986):

    w = r F² / (4 EI),   dF/ds = −μ w   ⇒   F_bottom = F_top / (1 + μ r L F_top / (4 EI))

gives a loss of 3.4% vs 4.0% at 100 N, 5.8% vs 5.9% at 150 N and 8.1% vs 7.7% at
200 N. η = 1e-3 delays helix formation to about 120 N, so keep η ≤ 1e-4.

## Compliant printed mounts

With `k_mount > 0`, every stack end (the jaw adapter, both sides of each input
body, and the housing) roots its 15 leaves into a small rigid block. The block
joins its body through a 6-DOF bushing (`springs.Bushings`):

    d  = R_Aᵀ (x_B − x_A,point),   θ = log(R_Aᵀ R_B)
    U  = ½ k_t |d|² + ½ k_r |θ|²,   k_r = k_t · (span/√12)²

That is, a uniform elastic interface over the stack height, measured in the
body frame so it is objective under rigid motion. 96 blocks add 576 DOFs.
Order of magnitude for a printed root block: E·A/L ≈ 1000 MPa × (22 × 5) mm² /
5–10 mm ≈ 1–2 × 10⁴ N/mm. The swept range is 3 × 10³ to 3 × 10⁴ N/mm.

## Leaf imperfection

A geometric imperfection is required, not optional. With perfectly straight
leaves the imposed motions barely excite the first (symmetric) buckling mode,
and Newton follows the unstable straight branch. In the single-module test the
leaves reach 1.72 P_cr with several negative tangent eigenvalues, and clamp
force is overstated by about 20%. Below about 20 µm the same happens or the
solve stalls. From 20 µm up the path is stable, and the peak clamp force falls
from about 376 N (20 µm) to 355 N (50 µm) to 330 N (100 µm) with the CAD
geometry: the imperfection is
a real physical parameter (print straightness), not just a numerical device.
Default: 50 µm, first fixed-fixed mode shape, every leaf bowed the same way
(`imp_sign = "same"`; `"random"` gives seeded random signs).

## Mesh

8 elements per leaf is converged: 8, 10, 12 and 16 agree within 1% for clamp
force and peak stress through post-buckling
(`examples/feeder_v2_3d/element_convergence.py`). 4 elements matches before
buckling but is about 15% too stiff after it and under-predicts peak bending
stress by about a third.

## Solver

Load-stepped Newton–Raphson with prescribed DOFs eliminated. Convergence:
‖r_free‖∞ ≤ 1e-8 (1 + max|r|). Line search: regula falsi on s(α) = Δq·r(q + αΔq),
the directional derivative of the potential. This is unit-consistent, unlike
‖r‖, which stalls near buckling. If a step fails, it is halved once; a
sub-step that still fails is retried with viscous stabilization
c_i = 1e-4 K_ii on beam-node DOFs (wire axial DOFs excluded) before halving
further. This handles local snaps, i.e. a leaf flipping between buckled shapes
or the wire re-arranging its helix. The dissipated energy is logged per step
(`stab_energy`, `stab_log`).

## Buckling reporting

P_cr per leaf = 4π²EI3/L² / (1 + P_e/(κGA)) (fixed-fixed, no sway, with shear
correction). Utilization = max(−N, 0)/P_cr. Leaves at ≈1.0 are buckled and
carry no more axial load, which caps the clamp force. True stability is
checked separately via the tangent's lowest eigenvalue (`--eig-every`).
