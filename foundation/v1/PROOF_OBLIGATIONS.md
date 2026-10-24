# PNDS → TAC-OSM proof obligations and qualification matrix (v1)

States: **proved-in-Lean**, **proved-on-paper/assumption-dependent**, **software-tested**, **trained-exposed-DEV**, **OPEN**, **FAILED**. A source link is provenance, not proof. Never convert one state into another based on wording alone.

| Contract ID | Scope | Formal/property target | Present proof/qualification | Required future verification |
|---|---|---|---|---|
| MATH-001 | Core bounded gate | \(h'=(1-g)h+gv\) preserves interval for \(g\in[0,1]\) | Exact scalar Lean proof in `PNDS/UnifiedTACOSM.lean` (if CI passes); NOT entire native graph | Source-level PyTorch operator and gradient correspondence |
| MATH-002 | Core gated-delta | \(v-h'=(1-g)(v-h)\) for scalar target | Exact scalar Lean algebra, **not 8×8 memory theorem** | Matrix retrieval error, cross-key interference, clipping, gradient contraction |
| MATH-003 | Physical ME095 filter | two affine steps compose and reassociate | Lean real-algebra scan, conditional | Full source-vs-spec serial/parallel FP/grad equivalence; classifier inputs independent of scanned state |
| MATH-004 | Enterprise event | source EOR and normalized conditional activity probabilities total one | Lean scalar mass formula **if** activity mass=1 | Check positive time laws, censored endpoints, numeric stability, actual EOR semantics |
| MATH-005 | Physical ME094 | translated base CDF has required median | Paper equation conditional on true base median | Finite precision inversion, implicit backward, intervals, calibration |
| MATH-006 | ES-WAM motion | \(\gamma(0)=0\) at nonzero denominator | Simple exact identity | Small nonzero speeds, noisy velocity and observed stationarity no-regression; ESWAM014 FAILED |
| MATH-007 | Nonlinear native | joint Jacobian spectral/energy growth under all allowed inputs | OPEN for contraction/ISS; PNDS v1.3 conditionally proves forward-state boundedness for source-shaped fast/slow/memory, and disproves blanket strict contraction of preserved read-only memory | Finite-time stability, controlled forgetting, disturbance propagation, nonvanishing selective retention |
| MATH-008 | Core/ES-WAM memory | bounded keys/capacity/interference after arbitrary distractors | OPEN for useful capacity; PNDS v1.1 gives exact generic unprojected row-query interference and unit-key residual, NOT projected long-term retention | Counterexamples and then conditional constructive error bounds; authentic entity traces |
| MATH-009 | Temporal | valid-time and known-at-time information cut | Specific adapter software tests only | Compositional noninterference proof and actual source time validation |
| MATH-010 | Counterfactual | identified \(do(a)\) response for real actions | OPEN, no SCM or randomized/propensity identification | Valid observed action support, interventions and nonconfounding proof |
| MATH-011 | Full source-to-proof | Lean graph == exact full PyTorch tensor program with defined error budget | OPEN (some ESWAM shape/role results only) | Operator map, semantic refinement, bounded floating-point differences and full backward checks |
| MATH-012 | Experimental promotion | matched parameter/data/compute and independent source | POLICY (not neural theorem) | Actual fitted controls, original replay, honest uncertainty and downstream outcome |
| MATH-013 | Risk/decision | calibrated and actionable event risk, not only physical MAE | OPEN; ME077A exposed-DEV event detection weak | Site-independent risk labels, decision-time threshold, actual verified intervention utility |

**Historical failures must remain:** ME069 lost GRU next-event log-loss; ME073 is capability-only and early-history forecasts weak; ME074 associative state readout near-passive on workflow DEV; DIAGKEY004 strict raw-logit replay failed 2/819200 despite pass on predicted classes; ME076 lost physical persistence; ME077 improved mean DEV physical MAE but missed >5% gate; ME094 physical full qualification failed; ESWAM014 low-speed stationary no-regression gate failed. No one may suppress these rows by publishing a later positive score.

**Formal proof coverage:** All PNDS v0.2 modules remain intact. PR #1 dynamic-representation proof has failing historical CI, PR #2 finite budget is a separate passing branch, and PR #4 memory-state theorem is a separate passing draft; do not assert they are merged into this new root. Lean full graph/source equivalence remains open.

**No information loss policy:** original Git branches/commits/checkpoints/results are the source of truth. New entries append to `evidence-ledger.jsonl` with parent versions and original provenance; never rewrite an older outcome as if it disappeared. `archive-gaps.json` lists unindexed historical ranges instead of falsely declaring complete archival coverage.

### Native memory derivations (v1.1)

See [NATIVE_MATHEMATICAL_ENGINEERING_V1_1.md](NATIVE_MATHEMATICAL_ENGINEERING_V1_1.md) and [PNDS/NativeMemory.lean](../../PNDS/NativeMemory.lean). Exact read/interference and collision results apply only with the stated address/projection assumptions; whole-model MATH-007/008/011 remain OPEN. Python reference tests and original pinned Torch operator correspondence must be reported separately. No original numerical or empirical failure is reclassified.

### Native forward boundedness, not information stability (v1.3)

The [conditional state-envelope certificate](NATIVE_STABILITY_CERTIFICATE_V1_3.md) distinguishes mathematical boundedness from retention, causal predictability and Jacobian stability. Its source-shaped Lean proof assumes bounded neural candidates/gates and Frobenius projection; original Torch source tests cover finite precision and read-only preservation separately. Since read-only events keep M exactly, a universal whole-state strict contraction is impossible in norms that distinguish retained memory states. The real-world qualification and MATH-008/011 obligations remain OPEN.
