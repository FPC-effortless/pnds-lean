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


## GP-UNC-004 source-time interface boundary (software-only; 2026-10-27)

New source-only TAC snapshot [`2faed40ae5cf50a199d32fd63ff66b2bba076b2a`](https://github.com/FPC-effortless/TAC-OSM/commit/2faed40ae5cf50a199d32fd63ff66b2bba076b2a) adds `src/tac_osm/gp_event_boundary_nofit.py` and an original UCI Appliances Energy integration audit. Forecast issue events exclude future truth by **exact dictionary schema**; only later separate `label_arrived` events may present truth, and adaptation is default-off. The original-source run [40485229043](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40485229043) passed 64 authentic already-exposed CAL forecast/label integrations and 64 source-anchored future-label poisoning diagnostics. It did **not** fit a GP on authentic labels; its base checkpoint is an already-trained non-neural classical ridge model.

**Formal mathematical status:** previous scalar Gaussian conditioning/rank-one theorems remain valid only under their premises. A Lean 4 model of two distinct event types, source availability, chronological noninterference, and an executable refinement from this Python event code is **OPEN**. Publisher historical row index does not certify actual operational label-availability time. Even with typed Lean event models, independent original-source hashes, provenance, actor/scorer isolation and full PyTorch computation-graph mapping remain empirical and systems obligations. No real-world model superiority follows from the software source audit.


## GP-UNC-005 first genuine trained TAC-OSM latent-input bridge (2026-10-27)

TAC source audit snapshot: [`8d32b7b1232a77c11dc6d6323e05b6aed166a27e`](https://github.com/FPC-effortless/TAC-OSM/commit/8d32b7b1232a77c11dc6d6323e05b6aed166a27e); original ME111 trained source code/checkpoints pinned at [`77c753a69f10fa13cd7fdf849edb12bc53f03e6d`](https://github.com/FPC-effortless/TAC-OSM/commit/77c753a69f10fa13cd7fdf849edb12bc53f03e6d). Exact UCI501 original FIT-source ZIP SHA `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.

[TAC neural-to-GP CI run 40500563279](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40500563279) **PASS** for five original ME111 frozen 104,663-parameter trained networks, two original 2014 FIT hour origins, 12 stations and six forecast channels: **720 original neural issue events**, 720 successful future-only source-input poisoning equivalence checks. GP features are now **original learned per-station 8×8 associative state flattened into 64D**, not merely the forecast output. The original trained point forecasts, split/asymmetric uncertainty scales, and all source/evidence/native memory state remain bit-identical under the future-only perturbation and no-fit GP pass. No model updates, residual assimilation, 2015 REPORT scoring or protected 2016–17 evaluation.

The original loader pre-parses all original FIT-period records, including future FIT measurements, and uses FIT-wide normalization; only the **post-checkpoint inference input frame** is proven not to depend on the artificially altered future source cells. This is not a proof that the model training never saw its future FIT values, or that historical source timestamps represent prospective decision-time record availability.

A new **mathematical neural-component interface proof obligation** is opened: source-equivalent 152D native ME111 carried state (specifically 64D association), projection to the GP RBF inducing feature, reference posterior updates, and the original asymmetric conditional output law must be connected by tensor-shape, IEEE/FP numerical-refinement and automatic-differentiation proofs. Scalar exact-real Gaussian and rank-one identities proven here DO NOT discharge those obligations. Using a 6-channel × 12-station independent GP filter arrangement would retain ~9,864 additional floating scalars for two fixed inducing coordinates (unqualified theoretical count, Python overhead additional). Do not assert native ultra-low-memory parity or improved interval calibration until qualified measured comparisons exist.

[Detailed original-source result receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/gp-unc-001-six-family-nofit-2026-10-27/research/mathematical-engineering/gp-unc-001/GP-UNC-005-NATIVE-ME111-RESULTS.md). Original ME111 undercoverage failures remain unmodified. Full multivariate GP/finite-precision/AD/entire graph and real-world model improvement: **OPEN**.
