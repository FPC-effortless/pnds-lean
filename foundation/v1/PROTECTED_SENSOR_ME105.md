# ME105 — ProtectedSensor

**Status:** exact-real causal-core and scalar-head contract; full source, neural,
IEEE and calibration proof OPEN. **Variant:** `protected_sensor` (ME105), a new
child of ME104 `median_split_sensor`. **Source:** implemented
`src/tac_osm/variants/protected_sensor_world.py`, TAC source commit
`573f81cb9f2f421d141ddb722525ecce25972edc`. **Lean:**
[PNDS/ProtectedSensor.lean](../../PNDS/ProtectedSensor.lean).
**Python:** `foundation/v1/protected_sensor_reference.py` and
`scripts/me105_graph_reference.py` are byte-identical to the pinned TAC independent
serial formulation. Eight tests against actual source pass, including FP32/FP64
outputs, states, input/weight gradients, source floors, zero-age gradients,
CDF/NLL, quantiles and chunk carryover. These numerical tests are not a Lean
proof of complete floating-point source equivalence.

## Typed input/output boundary

For a context of length `T`, the observation is
`values, observed, ages : Fin T → Fin 13 → ℝ/Bool`,
`clock : Fin T → Fin 6 → ℝ`, an owner identity, and recorded hours. The inherited
causal state is the product of 13 source sums, 13 source masses, 62 evidence
sums, one evidence count, and an 8×8 associative memory. The current ME105
Lean contract types these separately as `CoreState`. `CoreEvent` makes the
source-safe vector, per-source beta schedule, evidence basis and eligibility,
scalar alpha schedule, scalar memory write strength, memory key and memory value explicit. The feature vector is shared across all six channels and both width heads.

For event `t`, with safe source input `x_t`, evidence basis `e_t`, key `k_t`,
value `v_t`, and schedules `beta_t[i], alpha_t`, plus scalar `write_strength_t`, the specified causal step is:

```text
S'[i] = (1-beta[i]) S[i] + beta[i] * x[i]/8
C'[i] = (1-beta[i]) C[i] + beta[i]
E' = E + (eligible ? e : 0)
count' = count + (eligible ? 1 : 0)
M'[i,j] = eligible ? alpha*M[i,j]
          + write_strength * (v[i] - alpha * sum_q(M[i,q]*k[q])) * k[j]
          : M[i,j]
```

Thus an eligible memory update is `alpha*M + write_strength*(v - alpha*M*k) ⊗ k`, with row-vector
convention fixed by the displayed coordinates; an ineligible event leaves memory unchanged. Input masking is represented by
the contract's `safeSource`; the exact source rule for producing that vector,
eligibility, schedules, and encoded evidence remains to be mapped to the
source-pinned TAC implementation. `serialFold` defines chronological repeated
application. Its append/composition theorem states that folding a prefix and
then a suffix equals folding their concatenation; it does not claim arbitrary
parallel equivalence or prove that the implementation uses this step.

The width and median heads consume a typed `Fin 102 → ℝ` vector in this order:
13 source means, 13 source masses, 62 evidence means, six latest ages, and eight
inclusive memory reads. Source means use `8*S/max(C,1e-6)` and evidence means
use `E/max(count,1)`. `featureConcat` gives this layout explicitly and proves
the coordinate ranges. The concrete ME105 architecture uses a 102→6 median
head and a 102→12 asymmetric width head. Each head is affine in its learned
weights and bias, followed by the declared real `tanh` transforms. The Lean
contract models these as functions of the same 102 features; parameter-matrix
and bias correspondence is OPEN.

## Protected forecast law

For each of six output channels, the new median is
`mu = old_mu + 0.5*tanh(median_raw)`. The left and right scales are

```text
floor_scale = sqrt(floor_variance)
missing_guard = current_missing ? 0.8*base_scale : 0
factor_left/right = exp(log(4)*tanh(width_raw_left/right))
side_scale = max(floor_scale, max(missing_guard, base_scale*factor))
```

Under `floor_variance ≥ 0`, the Lean proof establishes `side_scale² ≥
floor_variance`; this preserves the complete inherited variance floor. A
positive floor gives positive side scales, and positive base scale also gives
positive side scales. The 0.8 bound is conditional on the source's
`current_missing` flag. A median correction bound of 0.5 is proved under the
explicit premise `|tanh(raw)| ≤ 1`; the actual tanh range theorem is not claimed
here. At zero-initialized heads, the median equals the parent median. The scales
equal the parent scale if the floor scale is no greater than the base scale,
the base scale is nonnegative, and both width raws are zero; the floor maximum
and missing guard are stated in the law, not silently dropped.

The parent forecast uses the inherited equal-half Gaussian split in standardized
log1p coordinates and the existing censored raw-unit transform. ME105 does not
change that distribution definition. Neither the Gaussian normalization,
CDF/quantile/NLL, zero-censoring atom, nor calibration is proved by the new
causal-core and scale lemmas. The new median head can change the reported point
median after training; the 0.5 correction bound alone makes no coverage or
accuracy guarantee.

## Parameter and training contract

The intended architecture has 100,349 trainable parameters: ME104's 99,731 plus
the 102→6 median head's 618. There are 33 learned parameter tensors as specified
for the source architecture. All parameters remain trainable. The inherited
parameters use learning rate `3e-6`; the new median and width heads use `3e-4`.
The split learning rates and trainability are training-protocol facts, not
Lean theorems. Actual instantiated counts, optimizer groups, and gradient
activity require source-level receipts.

## Proof ledger and open obligations

Lean proves the exact core coordinate equations, fold composition, feature
coordinate mapping, scale-floor lower bound, scale positivity under stated
premises, the conditional median correction bound, and zero-median-head
identity. These are exact-real specification results. `ProtectedHeads` and
`Graph.parentGraph` are interfaces; they are not existence/equivalence proofs
for the neural source.

Still OPEN: exact TAC source SHA and graph mapping; explicit affine parameter
matrices and all inherited encoder/admission/schedule functions; source-safe
mask and eligibility mapping; actual state serialization/chunk semantics;
parallel scan equivalence; original-source FP32/FP64 forward, state, loss and
backward-gradient equivalence; Gaussian split normalization and quantile/CDF
proofs; censoring and NLL/Jacobian proof; IEEE error bounds; calibration,
coverage, capability and utility. No full neural, whole-source or empirical
guarantee is claimed. A matching Python reference and tests plus reciprocal
source-SHA registration are required before changing these statuses. This
source pin is provenance only; source equivalence remains OPEN until its
independent mathematical reference and actual-source forward/backward/state
checks are recorded.

## Explicit full neural equations (additive formulation)

`PNDS/SensorGraph105.lean` replaces the opaque parent-network boundary with
finite matrix/bias parameters and explicit encoder, station embedding, source
admission, fast/slow proposals, evidence prefixes, controlled delta-rule memory,
chronological fold, pre-event and inclusive reads, original decoder, current
belief/reconstruction, FIT floor and protected median/width equations. It defines
streaming and fresh-history entry points. A shape-arithmetic theorem gives
100,349 trainable scalars, matching the actual Python model. Normal CDF/inverse
and log(2pi) are a named `NormalLaw` boundary whose identities remain OPEN;
raw quantile, zero atom, CDF and NLL formulas use that boundary explicitly.

These are full typed real-arithmetic graph **definitions**, not a proof that the
actual PyTorch network refines them. Owner/time/parameter-version validation,
all input domain premises, neural derivative and non-smooth semantics, Gaussian
measure normalization, floating-point rounding and empirical calibration remain
OPEN. Python source/reference tests and fitted audits provide numerical evidence
separately. No random Gaussian-law function is certified to be the standard
normal by compilation alone.
