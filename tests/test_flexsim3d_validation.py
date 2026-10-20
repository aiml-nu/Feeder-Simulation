"""
Closed-form validation of the corotational 3D beam (run: python tests/test_validation.py).

1. Linear cantilever, tip load in both bending planes (Timoshenko)  -> PL^3/3EI + PL/kGA
2. Large rotation: cantilever under end moment M = 2*pi*EI/L/4 ... full circle quarter -> exact arc
3. Torsion: tip twist = TL/GJ
4. Euler buckling via tangent eigenvalue: cantilever (pi^2 EI / 4L^2),
   fixed-fixed / fixed-guided-no-sway (4 pi^2 EI / L^2)
5. Post-buckling: imperfect cantilever column load-deflection vs elastica
6. Parallel guide (2 leaves to a rigid body): k = 2 * 12 EI / L^3, and the
   out-of-plane / rotated configuration gives the same with the leaf frame rotated
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from flexsim3d import Model, Section, solve_step, lowest_eigs

E, nu, t, w, L = 1000.0, 0.35, 0.8, 3.2, 24.0
sec = Section.rectangle(E, nu, t, w)
results = []


def check(name, got, want, tol):
    err = abs(got - want) / abs(want)
    ok = err < tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.6g}, want {want:.6g}, rel err {err:.2e}")


def cantilever(n_el=8, axis=(1, 0, 0), e2=(0, 0, 1), imperfection=None):
    m = Model()
    g = m.add_ground()
    tip = m.add_rigid(np.array(axis) * L, "tip")
    m.add_member(g, [0, 0, 0], tip, np.array(axis) * L, sec, e2, n_el, imperfection)
    m.finalize()
    return m, tip


# 1. linear tip loads
m, tip = cantilever()
P = 1e-4
for comp, I, name in [(2, sec.I3, "thin-direction (e2=z)"), (1, sec.I2, "wide-direction (e3=-y)")]:
    m.springs = []
    q = np.zeros(m.ndof)
    # apply load through a spring trick: use term-free external force via fixed? -> use residual offset
    class Load:
        dofmap = np.array([[6 * tip + comp]])
        def evaluate(self, Q, Qp):
            import jax.numpy as jnp
            return -P * jnp.ones((1, 1)), jnp.zeros((1, 1, 1))
    m.terms = [Load()]
    res = solve_step(m, q)
    want = P * L**3 / (3 * E * I) + P * L / (sec.kappa * sec.G * sec.A)
    check(f"linear cantilever tip deflection, {name}", res.q[6 * tip + comp], want, 2e-3)

# 3. torsion
class Torque:
    def __init__(s, T): s.T = T
    dofmap = None
    def evaluate(s, Q, Qp):
        import jax.numpy as jnp
        return -s.T * jnp.ones((1, 1)), jnp.zeros((1, 1, 1))
T = 1e-3
tq = Torque(T); tq.dofmap = np.array([[6 * tip + 3]])
m.terms = [tq]
res = solve_step(m, np.zeros(m.ndof))
check("tip twist TL/GJ", res.q[6 * tip + 3], T * L / (sec.G * sec.J), 1e-6)

# 2. large rotation pure bending: end moment rotating tip by 90 deg
EI = E * sec.I3
M_target = (np.pi / 2) * EI / L
m, tip = cantilever(n_el=16)
q = np.zeros(m.ndof)
nsteps = 10
for i in range(1, nsteps + 1):
    tq = Torque(-M_target * i / nsteps)  # moment about global y bends +x toward +z? sign checked below
    tq.dofmap = np.array([[6 * tip + 4]])
    m.terms = [tq]
    res = solve_step(m, q, q_prev=q)
    q = res.q
rot = q[6 * tip + 4]
R = EI / M_target
x_exact, z_exact = R * np.sin(L / R), R * (1 - np.cos(L / R))
check("pure bending tip rotation (90 deg)", abs(rot), np.pi / 2, 1e-6)
check("pure bending tip x", L + q[6 * tip + 0], x_exact, 2e-3)
check("pure bending tip |z|", abs(q[6 * tip + 2]), z_exact, 2e-3)

# 4. buckling: axial compressive force on tip, tangent eigenvalue -> 0 at Pcr
def buckling_load(n_el, guided):
    m, tip = cantilever(n_el=n_el)
    if guided:   # clamp tip rotations and lateral motion -> fixed-fixed with axial freedom
        m.fix(tip, [1, 2, 3, 4, 5])
    m.finalize()
    def lam_at(Pc):
        class Ax:
            dofmap = np.array([[6 * tip + 0]])
            def evaluate(s, Q, Qp):
                import jax.numpy as jnp
                return Pc * jnp.ones((1, 1)), jnp.zeros((1, 1, 1))
        m.terms = [Ax()]
        res = solve_step(m, np.zeros(m.ndof))
        vals, _, _ = lowest_eigs(m, res.q, k=1)
        return vals[0]
    # bisection on sign of smallest eigenvalue
    lo, hi = 0.0, 8 * np.pi**2 * EI / L**2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if lam_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

for n_el in (4, 6):
    check(f"cantilever buckling, {n_el} el/leaf", buckling_load(n_el, False),
          np.pi**2 * EI / (4 * L**2), 2e-2)
    check(f"fixed-fixed (axially free) buckling, {n_el} el/leaf", buckling_load(n_el, True),
          4 * np.pi**2 * EI / L**2, 3e-2)

# 5. post-buckling of imperfect cantilever: P = 1.015 Pcr -> elastica tip rotation ~ 20 deg
#    elastica: P/Pcr = (2K(k)/pi)^2 with k = sin(alpha/2)
from scipy.special import ellipk
alpha = np.deg2rad(20.0)
kk = np.sin(alpha / 2)
P_el = (2 * ellipk(kk**2) / np.pi) ** 2 * np.pi**2 * EI / (4 * L**2)
m, tip = cantilever(n_el=8, imperfection=(1e-3, lambda s: 1 - np.cos(np.pi * s / 2)))
q = np.zeros(m.ndof)
for i in range(1, 41):
    Pc = P_el * i / 40
    class Ax:
        dofmap = np.array([[6 * tip + 0]])
        def evaluate(s, Q, Qp, Pc=Pc):
            import jax.numpy as jnp
            return Pc * jnp.ones((1, 1)), jnp.zeros((1, 1, 1))
    m.terms = [Ax()]
    res = solve_step(m, q, q_prev=q, maxit=60)
    q = res.q
check("post-buckled cantilever tip rotation @ elastica load for 20 deg", np.rad2deg(abs(q[6 * tip + 4])), 20.0, 3e-2)

# 6. parallel guide in thin direction, and the same rotated 30 deg about z (V-stack style frame)
for ang in (0.0, 30.0):
    c, s_ = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    ax = np.array([c, s_, 0.0])
    m = Model()
    g = m.add_ground()
    body = m.add_rigid(ax * L)
    for zoff in (-5.0, 5.0):
        m.add_member(g, [0, 0, zoff], body, ax * L + [0, 0, zoff], sec, (0, 0, 1), 6)
    m.finalize()
    dz = 1e-4
    res = solve_step(m, np.zeros(m.ndof), prescribed={6 * body + 2: dz})
    r, _ = m.assemble(res.q, res.q, with_K=False)
    k = r[6 * body + 2] / dz
    # analytic: two fixed-guided leaves in parallel + body rotation restrained by axial stiffness pair
    check(f"parallel guide z-stiffness, stack rotated {ang:.0f} deg", k, 2 * 12 * EI / L**3, 1.5e-2)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
