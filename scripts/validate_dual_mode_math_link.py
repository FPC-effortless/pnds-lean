#!/usr/bin/env python3
"""Independent PNDS-side exact source preservation and scientific tier gate."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/"foundation/v1/dual-mode-math-link-v1-5.json").read_text())
assert m["schema"]=="pnds.tac_osm.dual_mode_evidence_v1.5"
assert m["historical_17_variant_registry_unchanged"] is True
assert m["original_aloha"]["matched_training_flops"] is False
assert m["original_aloha"]["exposed_test_fresh_skill_qualification"] is False
assert m["formal"]["full_tensor_fp_nonlinear_equivalence"]=="OPEN"
assert m["formal"]["full_17_model_source_theorems"]=="OPEN"
assert m["formal"]["source_model_real_skill"]=="UNQUALIFIED"
for k,v in m["artifact_blobs"].items():
    if not k.startswith("PNDS/"):continue
    f=root/k.removeprefix("PNDS/")
    actual=subprocess.check_output(["git","hash-object",str(f)],text=True).strip()
    if actual!=v:raise AssertionError("tracked Lean/math source diverged: "+k)
print(json.dumps({"status":"PASS_PNDS_UNPROMOTED_DUAL_MODE_PROVENANCE",
  "algebraic_proof_not_equated_to_real_skill":True},sort_keys=True))

# Reciprocal v1.7 negative real evidence and exact source blob guard.
v17=json.loads((root/"foundation/v1/temporal-retention-link-v1-7.json").read_text())
assert v17["schema"]=="pnds.tac_osm.temporal_memory_retention_v1.7"
assert v17["formal"]["full_original_fp32_neural_tensor_proof"] is False
assert v17["formal"]["all_seventeen_variant_proofs_complete"] is False
assert v17["authentic_data"]["flop_parity_certified"] is False
assert v17["authentic_data"]["independent_real_skill_qualified"] is False
assert v17["caveats"]["prior_negative_ESWAM_QUAL_evidence_unchanged"] is True
for path,blob in {
    "PNDS/TimescaleEvidence.lean":v17["formal"]["timescale_source_blob"],
    "PNDS/JointAffineTAC.lean":v17["formal"]["joint_schedule_source_blob"],
    "PNDS/RetentionBudget.lean":v17["formal"]["retention_math_source_blob"],
    "foundation/v1/BOUNDED_TEMPORAL_RETENTION_V1_7.md":v17["evidence"]["pnds_report_blob"]
}.items():
    actual=subprocess.check_output(["git","hash-object",str(root/path)],text=True).strip()
    assert actual==blob, "v1.7 mathematical source drift: "+path

