# PNDS ARCHFUSE089 — scoped exact-real memory operator contract

TAC operator source: `FPC-effortless/TAC-OSM`, commit `d21cfbc1048bd9c735ea6aeff1fbf1fb3b9b02ba` on branch `research/archfuse089-operator-falsifiers-2026-10-30`, file `research/mathematical-engineering/archfuse089/operators089.py`.

Parent native TAC: `LANGMEM086` original model file `research/mathematical-engineering/langmem086/compact_tac.py`; source input-to-output graph remains **unchanged**. LANGMEM087 negative findings and original checkpoints are preserved. Experimental status **NO_FIT**, no real-language comparison or model promotion.

## Types and equations (exact real)

Native memory has value-by-key orientation `M : value_channel → key_channel → ℝ`, key `k : key_channel → ℝ`, value `v : value_channel → ℝ`, channel retention `α : key_channel → [0,1]`, and scalar write strength `β ∈ [0,1]`.

`A[i,j]=M[i,j] α[j]`
`e[i]=v[i]-Σ_j A[i,j]k[j]`
`M'[i,j]=A[i,j]+β e[i] k[j]`

Lean module `PNDS/ArchFuse089.lean` defines channel retention and update, and states narrow propositions for zero-write behavior and unit retention. The independent Python source `foundation/v1/archfuse089_channel.py` uses plain loops to evaluate the same equations. Test `foundation/v1/test_archfuse089_channel.py` checks 50 seeded random scalar-retention correspondences, zero-write, featurewise retention, and shape fail-closed behavior.

## Explicit limits

- **Local independent Python tests: 3 PASS**. Lean compilation, explicit ArchFuse089.olean verification, no-sorry check, scoped axiom audit and CI Python reference tests: **PASS**, original GitHub Actions run #40992954959 (2026-10-30). Earlier import-order compilation failure remains preserved in Actions history.
- The entire graph's learned gates, temporal recurrent recurrence, softmax decoding, CPU finite precision, PyTorch AD gradients, causal actual-source stream, and streaming/chunk equivalence are **OPEN**.
- The unmodified original TAC remains 100,521 parameters; the candidate operators are standalone reference functions, not a 100,521-parameter replacement.
- No original dataset, optimizer, checkpoint, reward, evaluator or real next-byte label is used here.
- Reciprocal version-index canonical append and separately reviewed PNDS/TAC proof promotions remain **OPEN**.

The full TAC-OSM algorithm cannot be certified by these two small algebraic lemmas. Proven algebra is not trained predictive capability.
