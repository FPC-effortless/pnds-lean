# PNDS v1.6–v1.7: parallel state, bounded retention and strict capability limits

Date: 2026-10-25. Companion empirical results: TAC draft PR #389, research/mathematical-engineering/pnds-native-v1-1/TIMESCALE_RETENTION_RESULTS_V1_7.md. PNDS is the mathematical authority, **not** a publisher of asserted new world-model skill.

## Exact-real temporal state

Given scalar evidence h, observation basis b ∈ [-1,1], and fixed externally known gamma ∈ [0,1]:

  h'=(1-gamma)h+gamma*b.

PNDS/TimescaleEvidence.lean proves invariance h'∈[-1,1] from h∈[-1,1], and the exact difference law
  h'(u)-h'(v)=(1-gamma)(u-v),
including the absolute-value nonexpansiveness consequence. For fixed gamma and input-derived b, its scalar right-affine pair (p,q)=(1-gamma,gamma*b) composes chronologically with (pA pB, qA pB+qB) and is associative. A full 88D basis divides coordinates into four externally fixed rates 0.125,0.0625,0.015625,0.00390625 (22 coordinates each), with 88 carried scalars independent of observed sequence length. This is a native TAC temporal evidence candidate, not an SSM rebranding.

## Exact-real paired associative memory and decoder

PNDS/JointAffineTAC.lean defines the product of:
- 88 independent source-known scalar affine causal evidence transformations, and
- the learned 8x8 native source-addressed associative right-affine transformation.

It proves chronological associative product composition and that executing the paired transformation with an exact-real scan matches a fixed-size streaming event fold for **any** finite input-derived schedule. A completely arbitrary *deterministic* downstream decoder of the resulting paired state returns the same value (conditional on identical gate schedules and exact tensor semantics). This does not prove the original native nonlinear fast/slow recurrence is scan-parallel or that exact PyTorch numerical logits are bitwise-identical under reassociation.

## Controlled selective retention gate certificate

For original input-derived sigmoid gates a,b∈[0,1], fixed write scaler 1/4 and forgetting scale 1/50:

    keep=1-(1-a)/50,
    beta_eff=b/4,
    alpha_eff=keep*(1-b/16).

PNDS/RetentionBudget.lean proves:
- 49/50≤keep≤1;
- 0≤beta_eff≤1/4;
- **147/160≤alpha_eff≤1-b/16** (lower bound on *scalar multiplier only*);
- under normalized 8D keys/values and the usual delta right-affine operator norm inequality, ||M'||F≤alpha_eff||M||F+beta_eff, it conditionally proves norm(M')≤4 from norm(M)≤4.

Important caveat: The complete matrix right-affine operator also includes (I-beta_eff k kT). Its addressed-direction singular values can be smaller than alpha_eff, and a high alpha_eff **does not prove retained factual recall**. Full Lean tensor spectral-norm proof, floating source equivalence, learned task-relevant memory and world-model generalization remain OPEN.

## Original authentic development evidence

Pinned LeRobot ALOHA Mobile Cabinet original dataset revision 7a752b39f7e69de7e38aee485a6bab07528a061a, immutable parquet sha256 a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a. The 2-seed, 64-history, equal-93,324-parameter, 472-source TRAIN-window, 104-DEV-window factual experiments ran in TAC GitHub Actions on existing exposed DEV only, no new independent TEST.

- Multiscale temporal evidence: actual DEV MSE original 0.000391491/0.000382060 versus new multiscale 0.000409109/0.000404562. **Failed to improve prediction.**
- Controlled retention: actual DEV MSE original 0.000391491/0.000382060 versus controlled 0.000392125/0.000382458. **Failed to improve prediction.**
- But measured actual scalar gate ten-step retention products rose from original 0.000414791/0.0029869734 to controlled 0.6551933885/0.6406611204. Slow forgetting achieved; useful semantic long-term memory not established.
- Both PyTorch graph implementations and streaming states passed 294+162 finite-precision/gradient/4,096-step tests while keeping fixed 152 FP32 state values =608B/sample (excluding count, weights and buffers).
- Four original-source fitted checkpoint/optimizer/source-schedule artifacts preserved for each real pilot in GitHub Actions. Original 17-variant historical registry unchanged.

## Forward requirements

Do **not** treat conditional mathematical stability as world intelligence, and do **not** merge unqualified work. The new bottleneck is objectively demonstrating that memory identifies and preserves a fact that is absent from the current observation and needed for future real prediction or decision. Choose authentic delayed/partial-observation event streams with stable identity and observed future outcomes; frozen real source and untouched site/episode splits. Match trainable params, real data, split, sample schedule and full training compute. CPU elapsed time equality is not a fairness requirement. Currently neither new variant qualifies independent performance.
