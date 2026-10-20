"""
Dauber V2 wire feeder: parametric geometry, cam schedule and run driver.

Coordinates: wire along global z (up = +z, the nozzle / feed direction is -z).
Six jaw modules at azimuth alpha_k = 60 k deg. Module k:

  * jaw          rigid body; its contact edge is a straight 45 mm line at
                 radius r_wire (designed zero clearance at the as-printed state)
  * two V arms   leaf stacks in the vertical planes at alpha_k +/- 30 deg
                 (60 deg included angle), radiating from an apex on the jaw's
                 radial line at r_apex
  * per arm, per level (top / bottom):
        N angled leaves  jaw -> input body   (inclined theta, top up / bottom down)
        N straight leaves input body -> ground (horizontal)
  * two input bodies (top, bottom), each shared by both arms of the module;
    the cam drives its z at the cam-bearing point, all other DOFs free.

Jaw pairs: modules k and k+3 face each other and are driven in phase; the three
pairs are 60 deg of cam rotation (120 deg of the 180 deg cam period) apart.

Assumed dimensions not given in the paper (all parameters, see FeederParams):
leaf pitch in a stack, apex radius, gap from apex to first leaf, input-body
length along the arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .beam import Section
from .contact import WireJawContact
from .model import Model


@dataclass
class FeederParams:
    # flexure leaves (paper A.4)
    E_flex: float = 1000.0          # MPa (HT-PLA-GF datasheet value)
    nu_flex: float = 0.35
    t: float = 0.8                  # leaf thickness (bending direction) mm
    w: float = 3.2                  # leaf width mm
    L1: float = 24.0                # straight leaf length
    L2: float = 24.0                # angled leaf length
    theta_deg: float = 15.0         # angled-leaf inclination
    N_leaves: int = 15              # leaves per stack
    pitch: float = 1.5              # leaf spacing in a stack [assumed]
    n_el_leaf: int = 4              # beam elements per leaf
    imperfection: float = 0.05      # leaf initial bow amplitude, mm (seeded random sign); ~FDM print tolerance
    seed: int = 0
    # layout
    n_modules: int = 6
    V_angle_deg: float = 60.0       # included angle between the two arms of a module
    r_apex: float = 4.5             # radius of the V apex on the jaw radial line [assumed]
    s0: float = 2.0                 # apex -> first leaf attachment along arm [assumed]
    d_input: float = 4.0            # input body length along arm [assumed]
    jaw_length: float = 45.0
    level_gap: float = 0.0          # extra z gap between top and bottom stacks at the jaw
    # wire
    d_wire: float = 0.9
    E_wire: float = 70000.0         # ER5356
    nu_wire: float = 0.33
    wire_below: float = 5.0         # wire length below the jaws (to the spring BC)
    wire_above: float = 3.0         # wire length above the jaws (to the top guide)
    wire_h: float = 0.5             # wire element length
    k_feed: float = 100.0           # N/mm, spring at the wire's bottom end
    # contact / friction
    k_line: float = 1.0e4           # N/mm per mm contact length (Hertz-based)
    mu: float = 0.2
    eps_slip: float = 1e-4          # mm, friction regularization
    clearance: float = 0.0          # radial jaw-edge clearance at neutral
    # cam (paper A.4 values)
    delta_c: float = 0.4            # jaw clamp stroke (radial)
    delta_f: float = 0.8            # jaw feed stroke (axial)
    psi_c: float = 35.0
    psi_p: float = 90.0
    psi_u: float = 35.0
    psi_r: float = 20.0

    @property
    def theta(self):
        return np.deg2rad(self.theta_deg)

    @property
    def phi_half(self):
        return np.deg2rad(self.V_angle_deg / 2.0)

    def to_dict(self):
        return asdict(self)


def hertz_line_stiffness(q, R, E1, nu1, E2, nu2):
    """Secant line-contact stiffness q/delta [N/mm^2] for a cylinder of radius R
    on a flat at line load q [N/mm] (Johnson, Contact Mechanics, eq. 5.39-type
    approach estimate: delta = 2q/(pi E*) (2 ln(4R/b) - 1)).
    """
    Es = 1.0 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    b = np.sqrt(4 * q * R / (np.pi * Es))
    delta = 2 * q / (np.pi * Es) * (2 * np.log(4 * R / b) - 1)
    return q / delta


# ----------------------------------------------------------------- cam
def jaw_target(psi, p: FeederParams):
    """Ideal (rigid-link) jaw motion at cam angle psi [deg], following Fig. A.3.

    Returns (c, f): c = radial closing toward the wire (+ = clamping),
    f = axial travel in the feed direction (+ = down). One period = 180 deg.
      (2) psi=0      open (c=-dc/2), top (f=-df/2)
      (3) psi_c      closed,         f advanced df*psi_c/S
      (4) +psi_p     closed,         f advanced
      (5) +psi_u     open,           f=+df/2 (bottom)
      reset          open, back to top over psi_r
    """
    S = p.psi_c + p.psi_p + p.psi_u
    a = np.mod(psi, p.psi_c + p.psi_p + p.psi_u + p.psi_r)
    dc, df = p.delta_c, p.delta_f
    if a <= S:
        f = -df / 2 + df * a / S
        if a <= p.psi_c:
            c = -dc / 2 + dc * a / p.psi_c
        elif a <= p.psi_c + p.psi_p:
            c = dc / 2
        else:
            c = dc / 2 - dc * (a - p.psi_c - p.psi_p) / p.psi_u
    else:
        c = -dc / 2
        f = df / 2 - df * (a - S) / p.psi_r
    return c, f


def input_targets(psi, p: FeederParams):
    """Top/bottom input z displacements (+ up) from the rigid-link inverse
    kinematics (Eqs. A.11-A.14 with phi = half V angle):
        u_t - u_b = -2 c cos(phi)/tan(theta)   (inputs converging -> jaw closes)
        (u_t + u_b)/2 = -f
    """
    c, f = jaw_target(psi, p)
    k = np.cos(p.phi_half) / np.tan(p.theta)
    return -f - c * k, -f + c * k


def module_phase(k, p: FeederParams):
    """Cam phase lag [deg] of module k: pairs (k, k+3) in phase, pairs 60 deg apart."""
    return 60.0 * (k % 3)


# ------------------------------------------------------------ geometry
class Feeder:
    def __init__(self, p: FeederParams = None, with_wire=True):
        self.p = p = p or FeederParams()
        m = self.m = Model()
        rng = np.random.default_rng(p.seed)
        self.ground = m.add_ground()
        leaf = Section.rectangle(p.E_flex, p.nu_flex, p.t, p.w)
        self.leaf_section = leaf
        th = p.theta
        zhalf = p.jaw_length / 2.0
        span = (p.N_leaves - 1) * p.pitch
        # z of leaf i at the jaw, per level (top stack centred in the upper half)
        zc_top = zhalf / 2.0 + p.level_gap / 2.0
        z_jaw = {+1: zc_top + (np.arange(p.N_leaves) - (p.N_leaves - 1) / 2) * p.pitch,
                 -1: -zc_top - (np.arange(p.N_leaves) - (p.N_leaves - 1) / 2) * p.pitch}
        self.modules = []
        self.leaves = []           # BeamGroup list with tags
        bow = lambda s: 0.5 * (1 - np.cos(2 * np.pi * s))   # fixed-fixed first mode shape
        for k in range(p.n_modules):
            al = np.deg2rad(360.0 / p.n_modules * k)
            er = np.array([np.cos(al), np.sin(al), 0.0])
            et = np.array([-np.sin(al), np.cos(al), 0.0])
            apex = p.r_apex * er
            jaw = m.add_rigid(apex.copy(), f"jaw{k}")
            mod = dict(k=k, alpha=al, er=er, et=et, jaw=jaw, inputs={}, phase=module_phase(k, p))
            arms = []
            for sgn in (+1, -1):
                ph = p.phi_half * sgn
                d = np.cos(ph) * er + np.sin(ph) * et       # arm direction in plan
                arms.append(d)
            for lev in (+1, -1):              # +1 top, -1 bottom
                s_in = p.s0
                s_out = p.s0 + p.L2 * np.cos(th)
                s_g0 = s_out + p.d_input
                s_g1 = s_g0 + p.L1
                dz_ang = lev * p.L2 * np.sin(th)
                # input body reference = cam bearing point, midway between arms
                mid_s = 0.5 * (s_out + s_g0)
                Xin = apex + mid_s * np.cos(p.phi_half) * er + np.array([0, 0, lev * zc_top + dz_ang])
                inp = m.add_rigid(Xin, f"input{k}{'T' if lev > 0 else 'B'}")
                mod["inputs"][lev] = inp
                for ai, d in enumerate(arms):
                    for i in range(p.N_leaves):
                        zj = z_jaw[lev][i]
                        P_jaw = apex + s_in * d + np.array([0, 0, zj])
                        P_in = apex + s_out * d + np.array([0, 0, zj + dz_ang])
                        P_g0 = apex + s_g0 * d + np.array([0, 0, zj + dz_ang])
                        P_g1 = apex + s_g1 * d + np.array([0, 0, zj + dz_ang])
                        imp = (p.imperfection * rng.choice([-1, 1]), bow) if p.imperfection else None
                        tags = dict(module=k, level=lev, arm=ai, index=i)
                        g1 = m.add_member(jaw, P_jaw, inp, P_in, leaf, (0, 0, 1), p.n_el_leaf, imp,
                                          f"m{k}L{lev}a{ai}i{i}/angled", dict(tags, kind="angled"))
                        imp = (p.imperfection * rng.choice([-1, 1]), bow) if p.imperfection else None
                        g2 = m.add_member(inp, P_g0, self.ground, P_g1, leaf, (0, 0, 1), p.n_el_leaf, imp,
                                          f"m{k}L{lev}a{ai}i{i}/straight", dict(tags, kind="straight"))
                        self.leaves += [g1, g2]
            # jaw contact edge
            mod["edge"] = dict(ent=jaw, p0=(p.d_wire / 2 + p.clearance) * er + np.array([0, 0, -zhalf]),
                               t0=np.array([0, 0, 1.0]), n0=-er, L=p.jaw_length)
            self.modules.append(mod)

        self.wire_nodes = []
        if with_wire:
            self._build_wire()
        m.finalize()
        self._leaf_index()

    def _build_wire(self):
        p, m = self.p, self.m
        z0 = -p.jaw_length / 2 - p.wire_below
        z1 = p.jaw_length / 2 + p.wire_above
        n_el = int(round((z1 - z0) / p.wire_h))
        zs = np.linspace(z0, z1, n_el + 1)
        wsec = Section.circle(p.E_wire, p.nu_wire, p.d_wire)
        self.wire_section = wsec
        bot = m.add_entity([0, 0, z0], "node", "wire0")
        top = m.add_entity([0, 0, z1], "node", f"wire{n_el}")
        g = m.add_member(bot, [0, 0, z0], top, [0, 0, z1], wsec, (1, 0, 0), n_el, None, "wire")
        self.wire_group = g
        self.wire_nodes = list(g.nodes)
        self.wire_z = zs
        h = np.diff(zs)
        self.wire_ltrib = np.concatenate([[h[0] / 2], 0.5 * (h[:-1] + h[1:]), [h[-1] / 2]])
        self.wire_bot, self.wire_top = bot, top
        m.fix(bot, [0, 1, 5])           # guide into the spindle: lateral + twist
        m.fix(top, [0, 1])              # upper guide: lateral
        m.add_spring(bot, 2, p.k_feed)  # feed resistance (stiffness BC)
        self.contact = m.add_term(WireJawContact(
            m, self.wire_nodes, self.wire_ltrib, [mod["edge"] for mod in self.modules],
            k_line=p.k_line, mu=p.mu, rw=p.d_wire / 2, eps=p.eps_slip))

    def _leaf_index(self):
        """Arrays for fast per-leaf post-processing."""
        self.leaf_elems = np.array([g.elements for g in self.leaves])     # (n_leaves, n_el)
        self.leaf_kind = np.array([g.tags["kind"] for g in self.leaves])
        self.leaf_module = np.array([g.tags["module"] for g in self.leaves])
        self.leaf_level = np.array([g.tags["level"] for g in self.leaves])
        self.leaf_arm = np.array([g.tags["arm"] for g in self.leaves])
        self.leaf_i = np.array([g.tags["index"] for g in self.leaves])
        s = self.leaf_section
        EI_thin = s.E * s.I3
        L = np.array([g.length for g in self.leaves])
        # fixed-fixed (no sway) Euler load, thin direction, Timoshenko-corrected
        Pe = 4 * np.pi**2 * EI_thin / L**2
        self.leaf_Pcr = Pe / (1 + Pe / (s.kappa * s.G * s.A))

    # ------------------------------------------------------------ driving
    def prescribed(self, psi, ramp=1.0):
        """Input-body z DOFs at cam angle psi (deg); ramp in [0,1] scales from neutral."""
        pres = {}
        for mod in self.modules:
            ut, ub = input_targets(psi - mod["phase"], self.p)
            pres[6 * mod["inputs"][+1] + 2] = ramp * ut
            pres[6 * mod["inputs"][-1] + 2] = ramp * ub
        return pres
