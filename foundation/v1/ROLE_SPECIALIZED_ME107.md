# Native TAC role FFNs: full typed graph, limited exact proofs

TAC source `ddf04323ea352ef5ce80d2047074d54a9a047198`, prototype branch
`research/me107-role-specialized-2026-10-26`; original source2e0869d1 remains in
history. A one-shot block-iterator validation bug was fixed before any fit by
materializing/validating the full partition; the regression remains in source.
No new model has been trained and no capability promotion is established.

`PNDS/RoleSpecializedFFN.lean` explicitly defines three learned
102→16→{12,12,6} tanh FFNs, a backbone without replaced head tensors, native scan,
inclusive current reconstruction, bounded median innovation, left/right scales,
source floors, and wrappers around the same censored raw quantile/CDF/NLL law.
Zero old-head fields are constant API carriers only, not extra learned tensors.

`foundation/v1/role_specialized_reference.py` is byte-identical to the pinned TAC
independent reference. It reconstructs FFNs with direct functional tensor ops
and the inherited chronological reference, never calling the new FFN forward
implementations. Pinned source tests compare FP32/64 full outputs, state and all
parameter gradients; other tests enforce unchanged streaming state, output role
gradient zeros and fail-closed proposed-displacement checks. These are numerical
source/reference tests, **not a proof that Lean elaboration refines PyTorch**.

Hard-assigned **output roles**, not answers or domain classes, delimit the heads:
current state12, uncertainty shape12, point innovation6. Values/weights are learned.
There is no attention, new SSM, decoder feedback into the same scan, additional
recurrent state, or evidence of a new seventh active model family.

Eleven Lean declarations compile. Exact mathematical facts include arithmetic
102713parameters; inherited scan/probability wrappers; output shapes; and generic
finite-real-vector protection of a proposed displacement with nonpositive point
dot. Standard axioms only, no added axiom/sorry/admit. The FP64 accumulation and
rounded-parameter fallback in Python are tested and not proved by that theorem.

The guard is pure and not integrated into a trained optimizer. Even its ideal
first-order dot constraint does not prove finite-step MAE descent at kinks,
calibration, other-objective improvement or generalization. Shared ancestors can
still carry conflicting gradients. The current FFNs are randomly initialized,
not bit-identical parent warm starts.

Whole-neural/source, automatic-differentiation and IEEE refinement, the normal-law
implementation interface, predictive improvement, useful memory and real-world
decision utility remain OPEN. Original ME106 failed gates, DIAGGRAD001 FP32
cross-host exceptions and all older checkpoint evidence are preserved. The
machine-readable TAC `research/ARCHITECTURE-INVARIANTS.json` tracks these limits.
