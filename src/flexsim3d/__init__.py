"""flexsim3d: geometrically nonlinear 3D beam / rigid-body / contact simulation
for compliant mechanisms (developed for the Dauber V2 wire feeder)."""
from .beam import Section
from .model import Model
from .solver import solve_step, lowest_eigs
