# ME101 proposal — source-aligned latent sensor uncertainty

**Status:** partial scalar formulation with an implementation available for source comparison. This specification does not certify the full learned model or numerical equivalence. TAC-OSM owns the implementation and experiment record; PNDS owns this versioned formulation and its Lean/reference artifacts.

**Ancestry:** additive uncertainty head over `SHARED-ENTITY-SENSOR-ME100` (TAC source path `src/tac_osm/variants/belief_sensor_world.py`, exact TAC source commit `493032ff7210604d6123a923d0d1a786607552ea`). ME100's causal per-channel source-sum/mass trace is inherited unchanged: `S'=(1-β)S+βz`, `C'=(1-β)C+β`, with missing source cells setting `β=0`. Its existing proved identity, boundedness, and affine-composition results remain the only trace claims. ME101 does not amend or re-prove that kernel.

## Typed contract and equations

For an entity and a chronological observation window, the model receives value tensor `values ∈ ℝ[B,T,13]`, Boolean observed mask of the same shape, normalized ages in `[0,1]`, a six-coordinate context, entity IDs, and timestamps. The six-channel output is a log-location vector and a strictly positive log-scale vector. The existing ME100 encoder, causal trace, memory, retrieval, location path, and future scale are inherited. Missingness is explicit; a missing value is not interpreted as a measured zero. ME100 keeps its 97,235 parameters and 152-value (608-byte FP32) persistent state; ME101 adds 18 state-noise parameters and 6 response parameters for 97,259 total parameters, without adding a persistent tensor.

For output/current channel `c`, let `m_c ∈ {0,1}` indicate whether the current cell is observed, `a_c = expm1(age_input_c·log(257)) ∈ [0,256]` be source age in hours, `C_c = clamp(source_mass_c,0,1)` be inherited source mass/confidence, and `b0_c,b1_c,b2_c,response_c` be learned scalar parameters. Define `sp(x)=log(1+exp(x))` and

```text
v_c = (1-m_c) · (sp(b0_c) + sp(b1_c)·a_c/24)
      · (1 + sp(b2_c)·(1-C_c))
j_c = 1 + 0.25·tanh(response_c)
Xlatent_c | trace ~ Normal(anchor_c, v_c)
Ylog_c = mu_base_c + j_c·(Xlatent_c-anchor_c) + ε_c
ε_c ~ Normal(0, future_scale_c²), independent of Xlatent_c
```

The exact marginalized conditional mean is `mu_base_c`; the exact diagonal variance is `future_scale_c² + j_c²·v_c`, and the returned log-scale is its square root. `future_scale_c ≥ 0.05`. The scalar Lean module proves `v_c ≥ 0` assuming nonnegative values of the softplus transform at the three learned coefficients, with `a_c ≥ 0` and `C_c ≤ 1`; it proves `v_c=0` for an observed current cell, monotonicity in age at fixed confidence assuming the age and confidence domains and nonnegative softplus slope/multiplier, and strict positivity of the marginalized variance. Formal positivity of the actual `log(1+exp(x))` transform remains open. These statements are per-channel scalar claims. They do not imply positive definiteness for arbitrary matrix covariance sums.

The uncertainty query is read-only with respect to the inherited source trace: it reads `C` and age/mask metadata and emits a marginal variance; it does not write a new persistent state tensor or feed its result back into the causal history. The stated persistent model-state budget remains 152 FP32 values (608 bytes); age/mask metadata is accounted separately. The anticipated model parameter count is 97,259 (97,235 inherited +18 noise coefficients +6 response coefficients), independently counted against the actual implementation.

## Formal and executable artifacts

`PNDS/BeliefSensor.lean` imports `PNDS.SourceAlignedTrace` and contains the scalar exact-real definitions and theorems. Its proofs cover only the displayed scalar kernel under explicit transform-positivity premises and the read-only trace property. `foundation/v1/belief_sensor_reference.py` composes the inherited independent ME100 serial evaluator with an independently computed ME101 variance head. `tests/test_belief_sensor_source.py` checks source forward values, state, uncertainty traces, and all gradients in FP32 and FP64. These checks establish numerical refinement only for the exercised synthetic cases.

## Open obligations and evidence boundary

The reciprocal SHARED-BELIEF-SENSOR-ME101 record is appended on both repositories and pinned to the original source. Formal proof of softplus positivity remains open. The Python reference checks the implementation's per-channel response and variance formula against source forward values, inherited state, uncertainty traces, and gradients in FP32 and FP64 on synthetic inputs, but is not an IEEE error proof. Any additional dependence in the learned Jacobian, shared-channel covariance, or feedback/write path must be reflected here before claiming broader correspondence. No training, capability, calibration, causal-effect, or real-world superiority claim follows from these scalar proofs. ME100's historical score failures remain intact.

## Recorded source experiment, distinct from proof

[TAC draft PR397](https://github.com/FPC-effortless/TAC-OSM/pull/397) preserves five
original 97,259-parameter models, all frozen at3752a98e before exposed2015 scoring.
Ordinary six-channel point errors improve, but missing-data density/interval scores
regress and all five full capability gates fail. This is MIXED_TRADEOFF, not
qualified calibration or intelligence. Original clean replay and GitHub replay
40178388335 pass; actual fitted-weight source/state/gradient checks pass. Protected
2016–2017 outcomes remain closed. Full results and original model/checksum paths
are in TAC research/mathematical-engineering/me101 and research/replay-bundles/ME101.
These empirical outcomes do not strengthen the scalar Lean theorem scope.
