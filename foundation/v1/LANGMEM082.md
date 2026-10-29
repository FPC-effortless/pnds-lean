# LANGMEM082 partial compressed read contract

Source TAC-OSM draft#624, immutable implementation dc39eaaa7f7cb3209173f7117bab7daa503f84b7.
Companion PNDS draft#79 and source-pinned reciprocal bridge `langmem082_source_bridge.json`.

Typed exact-real matrix M:Fin8×Fin8→R, key/value:Fin8→R. Source original
slow update M'=MP+epsilon v w^T. Current read M'q and externally supplied old
writer read M'w_a are separate access paths. Supplied old keys and tagged
Q_a products are evaluator storage, not the original fixed4160B tensor state.
Wrong-content event uses the same historical key, timestamp and right-side
transport, and wrong value scaled to the true value norm. It is an evaluator
intervention on observed old content, never a future-label-fed predictor.

Lean defines/proves read additivity, isolated rank-one read direction, row
permutation squared-norm invariance, ring no-future bound and exact payload
capacities240/4144 for total256/4160 minus16 metadata bytes. Python reference
adds norm-matched wrong-event construction and executable numerical checks.

Full codebook byte-argmax, original trained logits, all source graph/AD,
finite-precision error proof, exact packed implementation layout, whole
information-flow isolation and semantic content recovery are OPEN. No theorem
asserts authentic empirical capability. See TAC source protocol for original
publisher licence, frozen already-exposed DEV owners, three immutable original
checkpoints, no-fit and matched-source controls. No new fit/GPU/TEST/promotion.

The original canonical17-variant index and its source-pinned historical rows
remain immutable. This diagnostic's reciprocal record is explicit and partial;
canonical coordinated main/master registry evolution remains OPEN.
