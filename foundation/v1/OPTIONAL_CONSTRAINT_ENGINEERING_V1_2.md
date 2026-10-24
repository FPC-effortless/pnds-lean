# PNDS v1.2 — mathematically optional decision and relational representation adapters

Date: 2026-10-24. **No new trained TAC-OSM variant or empirical performance claim.**
Paired repository: [TAC-OSM](https://github.com/FPC-effortless/TAC-OSM/pull/389).
Historical 17-version indexed mathematical contract remains unchanged.

## 1. Do not conflate model identities

The researched TAC-OSM is Task–Action–Counterfactual Observational State
Model: nonlinear fast/slow state, selective associative memory, causal
observations, action-conditioned readouts where supported, and real-world
outcome verification. It is **not** a mixed-integer logistical solver by
definition. External descriptions of "Tactical Optimization and Constraint-Aware
Operational State Models" are not source authority for these repositories.

The useful hypothesis is a **separate optional constraint-aware interface**:

known-at-time observations + verified physical state + trained forecast ->
admissible desired planar control (from some authorized policy) ->
conditionally constrained QP -> status, conditional proposed action.

This does not create verified actions, authorize physical control, or make
the underlying world model causal. A forecast is not the measured state.

## 2. Exact conditional sampled barrier derivation

For an ideal *single-integrator* civilian robot with constant planar
velocities on the current interval, let p'=u, o'=w, r=p-o and
h(r)=r·r-R². Suppose measured obstacle velocity w_hat has a genuine
external error bound ||w-w_hat||<=epsilon, gamma>0 and dt>0.

Cauchy-Schwarz implies that if

  2r·u >= 2r·w_hat - gamma*h(r) + 2||r||*epsilon,

then the actual derivative at the sample start obeys

  2r·(u-w)+gamma*h(r) >= 0.

If r and w evolve under CONSTANT relative velocity v=u-w through the
interval, then for every s in [0,dt],

  h(r+s*v) = (1-gamma*s)h(r)
              + s*(2r·v+gamma*h(r)) + s²||v||².

Given h(r)>=0 and gamma*dt<=1, all terms on the RHS are nonnegative.
This is a **conditional ideal-model interval** statement, not a proof
for discrete motor dynamics, acceleration, delays, collision geometry,
sensor errors without a valid epsilon, or a real robot.

PNDS/ConstraintAdapter.lean proves exact planar sample algebra and the
one-sample nonnegative barrier result for known constant relative
velocity, plus convex scalar projection optimality and KKT feasibility,
stationarity, nonnegative multiplier and complementarity.

## 3. Exact two-dimensional Euclidean constraint projection

At a known sample, let n=2r and b=2r·w_hat-gamma*h+2||r||epsilon.
Given a desired command u_d, solve

  minimize 1/2 ||u-u_d||² subject to n·u >= b.

For n·n>0 and *no actuator bounds*, the exact projection is

  u* = u_d + max(0,b-n·u_d) * n/(n·n).

Within each strict active-set region its gradient is well-defined.
At the active boundary, no globally smooth derivative is guaranteed.
This special case does **not** establish general KKT/implicit
differentiation for nonlinear constraints or discrete mixed-integer
decisions.

With independently authenticated componentwise actuator bounds
l<=u<=h, feasibility is determined by

  max_(u in box) n·u = sum_i n_i*(h_i if n_i>=0 else l_i).

If this maximum is < b, no admissible safe command exists in the box.
The PNDS Python reference enumerates candidates on the box interior,
constraint halfspace and four edges (2D convex active-set geometry)
and returns None on infeasibility. It must not output zero as a
putatively safe fallback. Float roundoff and missing physical parameters
must be treated as potential invalidation, not silently relaxed constraints.

The differentiable TAC-OSM candidate uses only the **unbounded**
closed-form halfspace layer, so it cannot be deployed as a hard
actuator-safe controller; use the bounded reference as an engineering
diagnostic, not a certified real-time system.

## 4. Hypergraph representation: structural candidate, not auto-superiority

For authenticated incidence H in {0,1}^(N×E), positive edge weights
W and strictly positive vertex and edge degrees, a normalized
hypergraph operator can be written:

  S=Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2),
  L=I-S.

This may represent genuinely observed group-level relations (e.g. common
source actor track interactions) not faithfully reducible to independent
edge messages. However its useful topology must be grounded in
observed entities, known-at-time edges and missingness; inferred
hyperedges cannot be used as factual authority. Its cost depends
on incidence sparsity and edge cardinality, not just N.

Do not replace native fast/slow/associative state with a hypergraph layer
without real-data ablations under the compulsory equal parameters,
training compute, exposure, time-cut and output-level comparison.
A hypergraph Laplacian does not itself prove learned world dynamics.

## 5. Diffusion, energy, manifold and Mamba claims

- Energy-based guidance with a finite penalty does not guarantee a hard
  inequality; even a diffusion sampler with an ideal target has sampling
  and discretization errors. No global optimality or trajectory feasibility
  is inferred from energy descent. See 2025 robotics diffusion surveys and
  ICML 2026 probabilistic guidance guarantees with explicit assumptions.
- A Riemannian geometry requires a symmetric positive-definite metric
  tensor, adequate regularity, and an actual physical/operational
  interpretation. An arbitrary threat/cost map is not automatically
  such a metric. Here it is not implemented.
- Mamba's input-conditioned affine selective recurrence and legal prefix
  operations do not make TAC-OSM's nonlinear native state scan-parallel.
  Mamba, Transformer and EBM alternatives stay as *controlled baselines or
  separately preregistered candidates*, not mandatory replacements.
- Unidentified P(y|observed a,s) is not P(y|do(a),s). CBF admissibility
  on a proposed control does not prove a causal world model or policy utility.

## 6. Nonnegotiable verification/status boundary

The new mathematics has levels:

- **S1 conditional formal theorem**: Lean proof under ideal physics,
  positive sample conditions and/or scalar QP.
- **S2 synthetic mechanism test**: Python bounded Euclidean projection,
  false zero-braking example, numerical differentiation of optional Torch
  layer, and explicit failure semantics. These are mathematical tests.
- **S3 complete model**: NOT established. No native TAC trained graph
  has been source-refined by this optional adapter.
- **S4 real physical capability**: NOT established. No original real
  trajectories, authorized actuator response logs, new controlled
  deployment, fitted neural weights or independent outcomes.

References:
- Amos & Kolter, OptNet (ICML 2017):
  https://proceedings.mlr.press/v70/amos17a.html
- Ames et al., Control Barrier Function Based Quadratic Programs (IEEE TAC 2017):
  https://doi.org/10.1109/TAC.2016.2638961
- Feng et al., Hypergraph Neural Networks (AAAI 2019):
  https://doi.org/10.1609/aaai.v33i01.33013558
- Hoshino et al., Doob h-transform guidance and probabilistic safety (ICML 2026):
  https://proceedings.mlr.press/v306/hoshino26a.html
