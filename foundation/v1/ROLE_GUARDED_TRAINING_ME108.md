# ME108: explicit moments, proposed displacement, and role-protected updates

TAC source95cfd29e9adb6f95eb705ad8f89f0a66d94747bc, registered before any
optimization. Forward is unchanged ME107 RoleSpecializedSensorWorld102713params;
fresh three role FFNs and allfive ME103 inherited backbones are trained together.
The current linear head is replaced; no identical-forecast warmstart is claimed.

`PNDS/RoleGuardedTraining.lean` defines raw combined-gradient AdamW moments,
bias-corrected proposal including decay/rate, then the per-block displacement
operator from ME107. Moments advance from externally clipped combined gradients
even when a weight move is projected/rejected. The unclipped point gradient is
the protection direction. Eight limited exact-real declarations compile with
standard axioms; the bound is a first-order dot statement, not convergence,
finite-step MAE descent, calibration, model gradients or generalization.

`foundation/v1/role_guarded_training_reference.py` is byte-identical to the pinned
independent executable tensor equations in TAC. It includes model-precision AdamW
arithmetic and both displacement and actual-weight rounding checks. Actual
optimizer tests compare moments/committed parameters inFP32/64, unopposed
TorchAdamW and failure-before-mutation behavior. No real TAC fits occur in these
tests; their tiny parameters are arithmetic fixtures, not neural benchmarks.

Production `propose()` is a read-only evaluator; `step()` consumes a fresh point
snapshot and commits only after all validations. Complete real weight displacement
is rechecked inFP64 after addition rounding. Fallback rejects opposing nonzero
point-coordinate moves, leaving zero-gradient coordinates free. This extra IEEE
behavior is tested, **not proved** by the exact-real Lean contract.

The fixed experiment is five seeds x4096new updates, B16, contexts8/64/256;
65536sampled examples and7163648repeated history exposures/model. Inherited
backbone lineage3072+4096updates; fresh heads4096. Allfive unchanged engineering
gates are required across allfive seeds, on already-exposed2015 only; reserved
2016–2017 numeric data stay closed. Same-source old models are unmatched historical
references, not architectural controls. No attention/SSM replacement or active
family is added. No capability claim follows from this mathematical file.

The full neural/source/AD/IEEE and normal-law refinements remain OPEN. Original
models, failed gates and source versions are preserved. Real original checkpoints
and optimizer/RNG/receipts must be Git-frozen before REPORT scoring.

Preflight revision2 sourceeb40cd1c corrects only the station/hour plan geometry
validator before the first optimizer update. The aborted zero-step launch and
original registration are preserved in TAC. The original operator/reference
files from95cfd29e are byte-identical in the corrected source; mathematical
operators, data plans, budgets and gates are unchanged. CI pins corrected source.
