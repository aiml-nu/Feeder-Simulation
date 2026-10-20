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

Friction: the slip increment of the wire-surface material point relative to the
jaw material point since the last converged step, Δs_t (tangential part), gives

    f_t = −μ N Δs_t / √(|Δs_t|² + ε²)

This is stick for |Δs_t| ≪ ε and full slip at μN otherwise. It is
non-conservative, so its Jacobian (jacfwd) makes the tangent mildly
unsymmetric. The solver uses SuperLU in symmetric mode with a fallback.

## Solver

Load-stepped Newton–Raphson with prescribed DOFs eliminated. Convergence:
‖r_free‖∞ ≤ 1e-8 (1 + max|r|). Line search: regula falsi on s(α) = Δq·r(q + αΔq),
the directional derivative of the potential. This is unit-consistent, unlike
‖r‖, which stalls near buckling. Steps that fail are halved recursively. A
local snap (lowest eigenvalue < 0) is passed with viscous stabilization
c_i = 1e-3 K_ii on leaf-node DOFs over 8 sub-steps, with the dissipated energy
logged.

## Buckling reporting

P_cr per leaf = 4π²EI3/L² / (1 + P_e/(κGA)) (fixed-fixed, no sway, with shear
correction). Utilization = max(−N, 0)/P_cr. Leaves at ≈1.0 are buckled and
carry no more axial load, which caps the clamp force. True stability is
checked separately via the tangent's lowest eigenvalue (`--eig-every`).
