"""
Cam-follower loads and Hertz contact stress on the printed face cams.

The simulation prescribes each input body's z at its cam-bearing point, so the
force the cam must apply is the reaction at that DOF (Feeder.input_forces).

Stack (one per stackup, bottom to top):
    [outer brg] bottom cam (face up) | lower follower: brg down, brg up |
    middle-lower cam (face down) + middle-upper cam (face up) |
    upper follower: brg down, brg up | upper cam (face down) [outer brg]

Assumed mapping: bottom input body -> lower follower, top input body -> upper
follower. An upward force on a follower (+) is carried by its down-facing
bearing on the face-up cam below; a downward force (-) by its up-facing bearing
on the face-down cam above. With no stack preload only one of the two touches.

Follower force per stackup = (sum over the jaws of the group at that level) / n_stackups.
Load per bearing = that / n_bearings. Stress then follows from Hertz, so it scales with
sqrt(load), not with load.

Hertz line contact, cylinder (bearing outer race, radius R = D/2, length b) on a
flat (Johnson, Contact Mechanics, ch. 4):
    P' = F_n / b,   1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2
    a  = sqrt(4 P' R / (pi E*))            contact half-width
    p0 = sqrt(P' E* / (pi R)) = 2 P' / (pi a)
    max shear (principal)       0.30 p0 at depth 0.786 a
    max tau_xz (on planes parallel to the surface, i.e. the print layers
                when the cam is printed face-up) 0.25 p0 at depth 0.5 a, x = +/-0.87 a
Rolling friction / traction and the cone (speed) mismatch of a cylinder on a
face cam are ignored; the sharp transitions of the idealized cam schedule are
taken as flat (real cams have fillets, whose curvature adds stress).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields

import numpy as np

SURFACES = [  # (name, follower level: +1 upper / -1 lower, force sign that loads it)
    ("Bottom cam (face up)", -1, +1),
    ("Middle-lower cam (face down)", -1, -1),
    ("Middle-upper cam (face up)", +1, +1),
    ("Upper cam (face down)", +1, -1),
]


@dataclass
class CamStack:
    D_bearing: float = 10.0      # follower bearing outer diameter [mm]  (placeholder)
    b_contact: float = 4.0       # contact length: bearing width, or cam track width if narrower [mm] (placeholder)
    n_bearings: int = 1          # follower bearings sharing one cam surface, per stackup
    n_stackups: int = 2          # stackups per jaw group (pair of jaws)
    E_cam: float = 1000.0        # printed cam modulus normal to the face [MPa]
    nu_cam: float = 0.35
    E_bearing: float = 200e3     # steel [MPa]
    nu_bearing: float = 0.30
    cam_radius: float = 0.0      # mean radius of the bearing track on the face cam [mm]; > 0 adds ramp slope
    preload: float = 0.0         # stack clamp preload: contact force on each follower bearing at zero cam load [N]

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items() if k in names})

    @property
    def E_star(self):
        return 1.0 / ((1 - self.nu_cam**2) / self.E_cam + (1 - self.nu_bearing**2) / self.E_bearing)

    def to_dict(self):
        return asdict(self)


def hertz_line(Fn, R, b, E_star):
    """Cylinder on flat. Returns dict of arrays: a [mm], p0, tau_max, tau_xz [MPa]."""
    Pl = np.maximum(np.asarray(Fn, float), 0.0) / b
    a = np.sqrt(4 * Pl * R / (np.pi * E_star))
    p0 = np.sqrt(Pl * E_star / (np.pi * R))
    return dict(a=a, p0=p0, tau_max=0.30 * p0, z_tau_max=0.786 * a, tau_xz=0.25 * p0, z_tau_xz=0.5 * a, P_line=Pl)


# ------------------------------------------------------------- input forces
def input_forces(d, F=None):
    """(psi, fz) with fz (n_steps, n_modules, 2) [top, bottom], N, + up.
    Uses the per-step record if the run has it, else recomputes from the stored
    frames (every few steps; float32 states are accurate to ~1e-3 N here)."""
    if "inp_fz" in d:
        return np.asarray(d["psi"]), np.asarray(d["inp_fz"]), np.asarray(d["ramp"]) >= 1 - 1e-12
    from .feeder import Feeder, FeederParams
    if F is None:
        prm = d["params"] if isinstance(d["params"], dict) else json.loads(str(d["params"]))
        F = Feeder(FeederParams.from_dict(prm))
    qf, qp = np.asarray(d["q_frames"]), np.asarray(d["q_psi"])
    fz = np.array([F.input_forces(q) for q in qf])
    # frames before the cam starts are ramp frames (psi stays 0); keep only the last of them
    cam = np.ones(len(qp), bool)
    z = np.nonzero(qp <= qp[0])[0]
    cam[z[:-1]] = False
    return qp, fz, cam


def follower_forces(fz, n_modules=6, n_stackups=2):
    """Per-stackup follower force for each jaw group: (n_steps, n_pairs, 2) [upper, lower]."""
    npair = n_modules // 2
    g = fz[:, :npair, :] + fz[:, npair:, :]          # modules k and k+3 form group k
    return g / n_stackups


def ramp_slope(psi, p, phase, level, cam_radius):
    """tan(alpha) of the cam ramp under the follower of `level` (+1 top / -1 bottom)."""
    from .feeder import input_targets
    if cam_radius <= 0:
        return np.zeros_like(psi)
    h = 0.05
    zp = np.array([input_targets(s - phase + h, p) for s in psi])
    zm = np.array([input_targets(s - phase - h, p) for s in psi])
    col = 0 if level > 0 else 1
    dzdpsi = (zp[:, col] - zm[:, col]) / np.deg2rad(2 * h)
    return dzdpsi / cam_radius


def surface_stress(psi, ff, cs: CamStack, p=None, pair=0):
    """Contact on the four printed surfaces of one stackup of jaw group `pair`.
    ff: follower forces from follower_forces. Returns {surface name: dict(F, Fn, hertz...)}."""
    out = {}
    for name, lev, sgn in SURFACES:
        Fz = ff[:, pair, 0 if lev > 0 else 1]
        tan_a = ramp_slope(psi, p, 60.0 * pair, lev, cs.cam_radius) if p is not None else np.zeros_like(psi)
        # preload P (each bearing at zero external load): the loaded side carries
        # P + F/2 until the other side lifts off at F = 2P, then F (rigid follower,
        # equal contact stiffness both sides)
        Fb = np.maximum.reduce([cs.preload + sgn * Fz / 2, sgn * Fz, np.zeros_like(Fz)]) if cs.preload > 0 \
            else np.maximum(sgn * Fz, 0.0)
        Fb = Fb / cs.n_bearings
        Fn = Fb * np.sqrt(1 + tan_a**2)
        h = hertz_line(Fn, cs.D_bearing / 2, cs.b_contact, cs.E_star)
        out[name] = dict(Fz=Fb, Fn=Fn, tan_a=tan_a, **h)
    return out
