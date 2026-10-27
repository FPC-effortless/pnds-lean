# HISTORY-QUERY016 — revised loss, partial original graph contract

TAC draft PR505, original inference source fd201207cb76b30470f64e1fc1baecf9de44560d;
revised-loss source 3d7946a3e46233fc0c09e86f60ad2c24c27a1674,
scripts/train_history_query_diverse.py::full_loss. No architectural rewrite.

Inputs per window: source owner (12 stations), normalized observations,
observed-mask, causal age and last arrays [88,13], source calendar past clocks
[8,6] and future clock[1,6]. Replay -23..64 with original
SourceTimedEventQuery / SourceTimedNonassociativeQuery from LATFUSE008. All
original encoder/state/write/read/forecast weights remain active and count99390.
The associative state is endogenous to observations; historical query input
contains clocks and owner only, never target values or oracle writer keys.
Both outputs exist before loss accesses targets and masks. Predict normalized
log1p PM25 at anchor0..56 step8, forecast13 channels anchor70.

The original encoder and memory equations remain source-defined, not re-created
by this companion. This record formalizes only the changed masked eight-query
SmoothL1 objective and unchanged Gaussian/MAE future loss. Lean definitions:
PNDS/HistoryQuery016.lean. Scalar Python reference:
foundation/v1/history_query016.py. Full original graph numerical/gradient/IEEE,
precision, source-mask and formal sufficiency equivalence remain OPEN. Lean
compilation is pending CI. No theorem or completion claimed. AD at SmoothL1
boundaries and finite-precision normalization require separate analysis.

First/warm full operator preflight is arithmetic resource evidence, not model
skill. Only authentic internal source evaluation can supply empirical results;
protected TEST remains unopened and no checkpoint promotion occurs. Preserve
LATFUSE018 original negative memory/calibration gates and PR495 negative frozen
clock-specificity results. No synthetic test grants a real-data capability claim.
