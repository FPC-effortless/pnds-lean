# ME108 neural full-formalization and refinement boundary — 2026-10-26

**Status: PARTIAL MACHINE-CHECKED PROOFS, NOT COMPLETE MODEL CORRECTNESS.**
TAC source: branch research/neural-refinement-2026-10-26, original trained
native model research/me108-role-trained-2026-10-26; immutable original
ME108 five-source seeds and trained source registration. Underlying mathematical
graph: PNDS/SensorGraph105.lean, RoleSpecializedFFN.lean,
RoleGuardedTraining.lean.

## Source-level theorem target

Given any source-valid finite [B,T,13] sensor values and observation masks,
ages, calendar times, station owners and causal histories, any ME108 102,713
trainable parameter vector, and IEEE FP32/FP64 execution with the *recorded*
operator semantics and safe nonfinite/overflow cases, show:

1. Forward model trace and all 6 marginal split-censored forecasts refine a
   Lean-denoted finite tensor computation graph, respecting dimensions,
   entity separation, missing-information guards and original input-only
   prefix-scan semantics.
2. Reverse-mode autograd of each objective (NLL, median point, location MSE,
   masked-current reconstruction NLL, proper interval score) refines a
   declared selection of subgradients and finite-precision conventions at all
   nondifferentiable branches (clamp/min/max, censoring, zero thresholds,
   missing-data masks, detached uncertainty location).
3. Every physical scalar arithmetic kernel and reduction has an exact
   IEEE-754 interpretation with explicit operation association, fused or
   unfused multiply-add, round-to-nearest-even, subnormals, special values,
   overflow, underflow, NaN/Inf and backend-specific transcendental errors.
   Establish a nonvacuous accumulated output/gradient error bound from
   source input and trained parameter bounds to final forecast/optimizer step.
4. For the *actual* standard normal input measure, show that the source
   censored two-sided log-normal CDF, negative log likelihood, zero-value atom,
   density Jacobian and inverse quantile coincide with the mathematical
   pushforward and have total probability one. Handle strict positive scales,
   location/scale transformations and censored median.
5. Compose these into a single source-to-theorem Lean statement with no
   unchecked source-alignment assumptions, sorry/admit, unverified external
   axioms or hidden floating-point/oracle assumptions.

No theorem in the repository currently meets all five conditions.

## What is actually proved in this branch

- PNDS.NeuralRefinement.linear_input_vjp_correct: finite-sum transpose law
  for learned affine encoder/head weights. All out/in shapes generic.
- PNDS.NeuralRefinement.linear_parameter_vjp_correct: output cotangents
  reproduce weight/bias first-order parameter variations.
- PNDS.NeuralRefinement.ffn_input_vjp_correct and the three 102→16→12/12/6
  instances: exact-real source-shaped two-affine-layer JVP/VJP adjoints for
  supplied hidden activation diagonal. The separate theorem
  tanh_derivative_real derives the exact-real scalar tanh derivative from
  sinh/cosh rather than pretending torch.tanh is IEEE exact.
- PNDS.NeuralRefinement.two_stage_affine_roundoff: conditional additive
  absolute error propagation for one two-operation affine step. **It does not
  prove the local rounding premises or source's GEMM error bound**.
- PNDS.NeuralRefinement.split_cdf_at_location: the split CDF equals one-half
  at the median if the supplied NormalLaw.cdf(0)=1/2. This is conditional.
- PNDS.CensoredMeasureRefinement.physicalDraw_measurable,
  physicalDraw_nonnegative, rawLaw_total_mass and rawLaw_no_negative_mass:
  the *actual source-shaped* piecewise-affine Gaussian residual transform,
  raw exponential map, and censoring construct a normalized probability law
  from any input probability measure, and forbid physical negative readings.
  This formally discharges pushforward **mass normalization**. It DOES NOT
  yet identify that pushforward with the numeric torch.special.ndtr/log_ndtr
  implementation or prove Gaussian base integration/inverse-CDF identities.

Both modules are checked by the exact source-pinned Lean4.14/mathlib toolchain.
The CI explicitly compiles both modules, verifies their .olean files and
prints theorem axioms. The foundation's older proofs remain unchanged.

## What is verified numerically, not proved

Actual code source branch research/neural-refinement-2026-10-26 adds
scripts/me108_neural_refinement_audit.py and a read-only original-model CI.
All five original fitted ME108 checkpoint SHA256 IDs are checked; 3 source
FFNs × 5 seeds × 2 precisions = **30 source-based neural VJP checks** against
an independent NumPy affine/tanh implementation, with real UCI501
2013–2014 FIT histories. The same audit checks source zero-censoring,
negative support and median quantiles, preserving the unopened 2016–2017
protected cohort. This is a meaningful CPU backend numerical oracle, not
cross-device IEEE equivalence or a Lean source-refinement proof.

Prior ME108 original multi-module source/checkpoint FP32/64 audits, optimizer
moment/proposal bit-exact comparisons, 1059 Python tests, and eight limited
ME108 optimizer declarations remain fully preserved. These do not disappear,
but they cannot be silently upgraded to full formal source correctness.

## Remaining blocking obligations

- FWD-01: formal tensor semantics for every ME108 matrix, normalization,
  sigmoid/softplus/SiLU, source-admission, masked prefixes, fast/slow
  arithmetic, index maps, affine scan and both pre/inclusive reads; prove
  *one end-to-end source mapping*, not separate handwritten formulas.
- AD-01: prove the exact tanh FFN source gradient bridge for all parameter
  inputs; currently analytic FFN chain transpose plus numerical confirmation.
- AD-02: complete reverse-mode differentiability/subgradient analysis
  of ALL five objectives, Bernoulli masks, all 13 optimizer blocks and all
  piecewise probability transforms. Distinguish causal read/write and
  PyTorch detach actual semantics.
- FP-01: select and formalize a certified IEEE754 library and backend
  contract; prove kernel-specific accuracy/exception properties including
  transcendentals. Inexact parallel reductions are not associative in
  floating-point; mathematical affine scan grouping may differ bitwise.
- FP-02: complete the end-to-end numerical error budget and explicit
  infinite-history count/age overflow limitations; arbitrary-long-time
  safe behavior is not presently proved.
- LAW-01: instantiate a concrete normalized standard Gaussian measure
  (the generic pushforward proof only ASSUMES its base μ is a probability).
- LAW-02: derive the raw piecewise CDF, zero atom, positive-density
  Jacobian, NLL and quantile inversion from that measure, and source-tie
  torch.special.ndtr, ndtri, log_ndtr with explicit kernel accuracy bounds.
- TRAIN-01: prove the source's actual clipped AdamW moments, rounding
  recheck and 13-block guard semantics refine RoleGuardedTraining.lean.
  A local nonpositive dot is not finite-step loss improvement.
- SOURCE-01: formally link Lean tensors/indexes and all runtime
  preprocessing guards to source using a verified compiler/verified
  code translation (not solely a test script).
- QUAL-01: empirical full prediction gates remain FAILED 0/5 on real
  exposed ME108 evaluation; correctness proofs cannot transform these
  numbers into predictive performance or safe robot control.

**No mathematical or experimental result is promoted until its own
predeclared verification gate passes. Preserve original source, trained
checkpoints, failure receipts, CPU/parameter parity and six-family registry.**
