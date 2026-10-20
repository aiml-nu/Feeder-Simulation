# References

Citations for the models implemented (or planned) in this tool. Grouped by topic; each entry
notes where/how it's used in this repo.

## Beam formulations

- **Jirásek, M., La Malfa Ribolla, E., Horák, M.** "Efficient finite difference formulation of a
  geometrically nonlinear beam element." *International Journal for Numerical Methods in
  Engineering*, 122(23):7013-7053, 2021. https://doi.org/10.1002/nme.6820
  Planar Bernoulli beam element valid for arbitrarily large rotations, based on Biot strain and
  the strong-form equilibrium equations (not shape-function-based). Internal forces at any
  section come from integrating a finite-difference scheme along the beam (shooting-method
  style), so one element can represent an arbitrarily long, arbitrarily curved beam accurately
  by refining the integration grid rather than the mesh. Includes closed-form validation cases
  (pure bending, Williams toggle, buckling of an axially compressible cantilever) — see
  `examples/`. Candidate for a `GeometricallyNonlinear` beam formulation
  (`design/architecture.md`), and the basis for reframing this project's buckling handling as
  incremental nonlinear time-stepping rather than a separate additive softening matrix
  (`derivations/geometric_softening.md`).

## Buckling / geometric stiffness

- **Martin, H. C.** "Large Deflection and Stability Analysis by the Direct Stiffness Method."
  NASA Technical Report 32-931, Jet Propulsion Laboratory, August 1966.
  Source of the additive geometric-softening matrix (`_k_soft` in `functions.py`,
  `derivations/geometric_softening.md`) used as the current (linear-formulation) buckling
  correction: `k_beam = k_stiff - P * k_soft`, applied incrementally using the previous step's
  axial force.
- **Przemieniecki, J. S.** *Theory of Matrix Structural Analysis*. McGraw-Hill, 1968 (Dover
  reprint available). Standard textbook derivation of the consistent geometric stiffness matrix
  for a 2-node beam-column element — used to independently verify `_k_soft` numerically
  (`derivations/geometric_softening.md` confirms exact match).

## Validation benchmarks

- **Williams, F. W.** "An approach to the non-linear behaviour of the members of a rigid jointed
  plane framework with finite deflections." *Quarterly Journal of Mechanics and Applied
  Mathematics*, 17(4):451-469, 1964. https://doi.org/10.1093/qjmam/17.4.451
  Source of the "Williams toggle" experiment/analytical solution: two symmetric, nearly
  collinear beams under a vertical load, exhibiting snap-through buckling for one of two tested
  geometries. Used by Jirásek et al. (2021, Section 4.2) as a large-displacement validation
  case with experimental data; planned as a validation example here
  (`examples/williams_toggle/`) since it exercises both large rotation and instability/snap-
  through in a minimal two-beam model — directly relevant to validating whichever
  large-displacement formulation this project adopts.
- Euler critical buckling load (pinned/cantilever column) — classical closed-form result, also
  re-derived by Jirásek et al. (2021, Appendix B) including the axially-compressible correction
  term. Simplest possible validation case for any buckling-capable formulation
  (`examples/euler_buckling_cantilever/`).

## Further reading (cited within Jirásek et al. 2021, not yet used directly here)

- **Simo, J. C., Vu-Quoc, L.** "A three-dimensional finite-strain rod model. Part II:
  Computational aspects." *Computer Methods in Applied Mechanics and Engineering*,
  58(1):79-116, 1986. The "geometrically exact" beam element family used as a comparison
  baseline in Jirásek et al. (2021, Section 4.1) — worth a look if the finite-difference
  formulation turns out to have practical drawbacks (e.g. implementation complexity, cost of the
  local element-level iteration).
