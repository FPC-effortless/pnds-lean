# GP-UNC-001 — scalar probability-law and bounded-filter mathematics (PARTIAL)

**Epistemic status: exact-real scalar theorems in Lean, executable Python scalar checks, unqualified full-model equivalence.** No new model FIT, no publisher dataset, no sealed TEST, and no neural model superiority.

Original proposed TAC source implementation: [TAC-OSM commit 21a5ffef240024fe6d46e119d562d0db7281bdc8](https://github.com/FPC-effortless/TAC-OSM/commit/21a5ffef240024fe6d46e119d562d0db7281bdc8) (includes projected GP math reference, no-fit tests and CI-count correction). This is a PINNED CODE SNAPSHOT. Any later TAC source changes must be re-audited separately. Original experiment PR: [TAC-OSM #491](https://github.com/FPC-effortless/TAC-OSM/pull/491). Research tracker: [#492](https://github.com/FPC-effortless/TAC-OSM/issues/492).

## Lean equations and assumptions

For scalar jointly Gaussian coordinates of prior variance `p`, cross-covariance `c` and positive observed variance `v>0`:

- `Var(f* | y)=p-c²/v` is no larger than `p` for `v>0`.
- It is nonnegative only with additional covariance premise `c² ≤ pv`, i.e. a valid positive-semidefinite 2×2 covariance.
- With `c=0`, conditioning does not alter the variance.
- A scalar positive-noise information update `precision' = precision + feature²/noise` does not decrease precision, and a zero feature adds no information.

Lean file: `PNDS/GPUncertainty.lean`. Kernel/axiom audit: `Audit.lean`. Corresponding independent scalar Python code: `foundation/v1/gp_uncertainty_math.py`. Generated no-fit tests: `tests/test_gp_uncertainty_math.py`.

## Bounded inducing filter math (not yet proved in Lean)

TAC reference `src/tac_osm/gp_sparse_nofit.py` is a finite inducing-point projected process, **not** a variational sparse-GP learner. Let `Kzz = LL^T` (jitter regularized), `φ(x)=L^-1 k(Z,x)`, noise `ν>0`. The reference maintains `A = I + Σφφ^T/ν`, `b=Σφr/ν`, with `m × m` retained precision and `m` information entries. Predict `μ=φ*^T A^-1 b`; `V=k(x*,x*)−φ*^Tφ*+φ*^T A^-1φ*`. This is a source-code mathematical specification, not a proof of full-matrix positive definiteness, asymptotic calibration, effective retained information after long histories, floating-point Cholesky stability, or gradient identity.

## Exact scope and OPEN gates

- **Proved if Lean CI compiles:** the five named scalar inequalities/identities, given their mathematical premises.
- **Software-tested:** scalar Python formulations on constructed data, separate TAC no-fit 11+72+10 diagnostics.
- **OPEN:** Lean specification/proof of full projected covariance + Cholesky algorithm; differentiability/AD; masked, timed real-source neural residual extraction; trained sparse inducing parameters; source-to-reference numerical/gradient equivalence; 100k trainable/deployed parameter counts; approximate versus exact GP error bound; real-world predictive quality and calibrated probability law.
- Six named TAC family cases remain distinct; no new trained model ID is registered or implied. Do not append this mathematical component as an eighteenth validated *trained variant* in the historical 17-row registry.
- Counterfactual predictive uncertainty is not identified causal/interventional uncertainty without an explicit SCM, action support, real outcomes and valid identification assumptions.

Cross-repo science rules apply before all new empirical fits or promotion. Both PRs remain draft and unmerged until separately reviewed.


## GP-UNC-002/003 paired source update — 2026-10-27

Additional source-pinned numerical candidate (not changing the earlier source pin):
[TAC-OSM code snapshot 7d8426ad60f6981850de935a3809842f90f9db54](https://github.com/FPC-effortless/TAC-OSM/commit/7d8426ad60f6981850de935a3809842f90f9db54).
This snapshot adds `src/tac_osm/gp_rankone_nofit.py`, the offline constructed timing harness, and `src/tac_osm/gp_prequential_nofit.py`.

The scalar rank-one covariance identity is now specified in Lean:
`P' = P - (P φ)²/(ν + P φ²) = Pν/(ν+Pφ²)`, for scalar `P≥0`, `ν>0`.
The associated conditional nonnegativity and nonincrease results are in `PNDS/GPUncertainty.lean` with axiom audit in `Audit.lean`. The Python numerical reference in `foundation/v1/gp_uncertainty_math.py` checks 256 constructed scalar samples in addition to the original Gaussian-conditioning properties.

The multivariate Sherman–Morrison covariance update, any online nonstationary law, finite-precision PSD preservation, adjoints and derivative correspondence to PyTorch, and all six full neural variants remain **OPEN**. Experimental note: the TAC branch reports only constructed CPU benchmarks (4/8/16/32 inducing points) and no new original-real-data forecasting gain. The chronological delayed-label interface is a software fixture, not proof of authenticated outcome-time validity.
