# ME102 — ReconstructionSensor

**Status:** PARTIAL scalar exact-real contract; whole-model/source correspondence OPEN.  
**Variant:** `reconstruction_sensor` (new ME102 variant, descended from ME101 `BeliefSensor`).  
**Formal source:** `PNDS/ReconstructionSensor.lean`.  
**TAC source pin:** `FPC-effortless/TAC-OSM@bd7fe3c08f9979d34483d27dcd048eb5cec984e3`, `src/tac_osm/variants/reconstruction_sensor_world.py`.  
**Python reference:** `foundation/v1/reconstruction_sensor_reference.py`, byte-identical to TAC's `scripts/me102_graph_reference.py` and local `scripts/me102_graph_reference.py`.

## Purpose and boundary

ReconstructionSensor adds a learned current-value reconstruction head to the ME101 uncertainty-aware forecast. It conditions the head on admitted source context, evidence context, and current-source age. The current prediction contributes to the existing six-hour pollutant forecast only when that pollutant's current value is missing. The variant retains ME101 source admission, entity ownership, causal timing, and memory updates. The current head is a diagonal conditional emission; this record makes no full-network Jacobian, covariance, calibration, or coverage claim.

All value and standard-deviation quantities below are in standardized `log1p` units. The input age is the inherited normalized age; source code converts it to hours as `age_hours = expm1(age * ln(257))` before evaluating the age floor. The six current pollutant channels share the existing output convention.

## Typed input/output and computation path

For a batch of size `B` and history length `T`, the inherited observation path consumes values `[B,T,13]`, observed mask `[B,T,13]`, ages `[B,T,13]`, clock `[B,T,6]`, owner IDs `[B]`, and elapsed hours `[B,T]`, plus optional source-memory state. The source/evidence encoder supplies a 94-wide head context:

| Component | Width | Meaning |
| --- | ---: | --- |
| admitted source mean | 13 | source sum / mass, rescaled by 8 |
| source mass | 13 | admitted source support |
| final evidence sum/count | 62 | inherited evidence summary |
| current pollutant ages | 6 | normalized ages for the six modeled channels |

A current-event embedding is recomputed from the masked current sensor event using the existing 49-to-48 encoder. The existing address projection maps that embedding to the normalized `shared_write_key`; the inclusive final memory matrix reads it as `belief_read = final.memory @ shared_write_key`. This read is appended to the inherited 94-wide context, giving 102 features. It reuses the existing encoder and address parameters and introduces no persistent state or scan feedback. The learned affine `Linear(102,12)` produces six reconstruction-location values and six scale preactivations. The location values pass through `2*tanh`; scale preactivations pass through `softplus`. The variant has 98,495 parameters. The first six current pollutant observations are the supervised targets, standardized in `log1p` space. Training-only artificial masks hide one pollutant across an artificial missing block and score its actually recorded current value with explicit Gaussian NLL supervision. The authentic future six-hour targets and natural missingness evaluation remain the existing primary task; artificial masks are not primary evaluation data.

The causal path is: observation and owner/time checks → inherited source admission, evidence, and memory updates → recomputed current event embedding → shared address projection → read from inclusive final memory → 102-wide context → 12-wide current head → masked current reconstruction loss during training and missing-current uncertainty contribution during forecasting → inherited affine future response → six-hour output. The added read is computed after the inherited scan and cannot feed back into that scan. It reuses the write address projection, strengthening supervised read/write coupling without changing the scan or adding persistent state. The current head's forecast gradient path is through the missing-current mean and variance terms; the artificial-mask NLL gradient path reaches the current head, current event encoder, shared address projection, and read memory path during training.

## Scalar equations

For channel `c`, let `a_c` be the observed current value when present and the rescaled admitted source mean otherwise. Let `u_c` and `s_c` be the two corresponding head outputs, `z_c` the inherited ME101 latent-state variance, `d_c` the missingness indicator, `j_c` the inherited affine response, `n_c` the inherited future standard deviation, `r_c` the fixed FIT-derived floor rate, and `h_c` source age in hours:

```text
qmean_c = a_c + 2 tanh(u_c)
qvar_c  = 0.0025 + softplus(s_c)^2 + z_c
floor_c = d_c * r_c * h_c / 24
future_mean_c = old_mean_c + j_c * (qmean_c - a_c) * d_c
future_var_c  = max(n_c^2 + j_c^2 * qvar_c * d_c, floor_c)
future_scale_c = sqrt(future_var_c)
```

`r_c` is nonlearned and is constructed from the mean squared 24-hour differences between actual measured FIT records for channel `c`, in the model's standardized units. It is a variance growth floor, not a guarantee of calibrated intervals or empirical coverage. Missingness `d_c` is Boolean in the source and cast to a real mask in the scalar formalization. For an observed current value (`d_c=0`), the current head does not change the inherited future mean or variance. For a missing current value, the variance is at least both the inherited noise contribution and the age-dependent floor. Under `r_c ≥ 0`, nonnegative age, and nonnegative missingness, the floor is monotone in age.

## Lean contract and proof scope

`PNDS.ReconstructionSensor` defines the scalar head, missing-current mean update, and max-floored variance. It proves nonnegativity and the fixed lower bound of `qvar` given nonnegative inherited latent-state variance; the future variance bounds; observed-current mean/variance preservation; age-floor monotonicity; and identity preservation of the inherited trace. The theorem hypotheses are explicit. These exact-real scalar results do not prove the neural encoder/head, PyTorch gradients, finite-precision behavior, Gaussian likelihood correctness in the implementation, training convergence, calibration, or coverage.

The source-pinned Lean build and Python numerical workflow are registered in `.github/workflows/reconstruction-sensor-math.yml`. The independent reference derives the 48-wide current event embedding, shared address key, inclusive memory read, and all 102 input features. The workflow requires `TAC_SOURCE_COMMIT=bd7fe3c08f9979d34483d27dcd048eb5cec984e3`, checks both byte-identical reference copies, and runs the forward/state/trace/loss/gradient comparison in FP32 and FP64. The local source-pin test and both byte comparisons passed against that immutable source commit; the GitHub workflow itself has not yet run. Reciprocal variant-math links and experiment results remain separate evidence obligations; formal and numerical correspondence alone do not certify empirical capability.

## Experiment registration and evidence boundary

The assigned comparison is the existing ME102 same-architecture protocol: three predeclared loss/floor recipes by five seeds, with matched data, parameter count, training compute, and existing splits. Recipe values, source SHA, training configuration, and outcomes belong in the TAC-OSM receipt and reciprocal manifest. Genuine natural-gap results remain the primary evaluation. Any synthetic artificial-mask metric is a training diagnostic only. This record does not infer a result from an experiment plan.
