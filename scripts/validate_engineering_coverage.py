#!/usr/bin/env python3
"""Fail-closed cross-version mathematical-engineering coverage (NOT proof verification)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "foundation/v1"
LEVELS = ["S0_source_typed", "S1_conditional_lean",
          "S2_original_operator_numeric", "S3_full_graph_refinement",
          "S4_independent_real_skill"]


def verify(root=ROOT):
    registry = json.loads((root / "registry.json").read_text())
    coverage = json.loads((root / "ENGINEERING_COVERAGE_V1_1.json").read_text())
    rows = coverage["versions"]
    indexed = {v["id"]: v for v in registry["variants"]}
    if coverage.get("schema") != "pnds.tac_osm.engineering_coverage.v1.1":
        raise ValueError("coverage schema mismatch")
    if coverage.get("levels") != LEVELS:
        raise ValueError("lost source/proof/skill separation")
    ids = [v["id"] for v in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(indexed):
        raise ValueError("lost, duplicated or invented variant coverage")
    for v in rows:
        prior = indexed[v["id"]]
        if (v["source_commit"] != prior["source"]["commit_sha"] or
            v["source_pr"] != prior["source"]["pull_request"] or
            v["evidence_status"] != prior["status"] or
            v["evidence_tier"] != prior["evidence_tier"] or
            v["state_kind"] != prior["state_kind"] or
            v["math_proof_targets"] != prior["proof_obligations"]):
            raise ValueError("historical evidence/proof/source provenance changed: "+v["id"])
        if not v["graph_class"] or any(v[k] != "OPEN" for k in (
            "full_graph_lean_semantics", "complete_source_forward_backward",
            "global_nonlinear_stability", "useful_long_memory")):
            raise ValueError("unearned full mathematical certification: "+v["id"])
        if v["independent_capability_promotion"] != "UNCHANGED_HISTORICAL_EVIDENCE":
            raise ValueError("mathematical activity promoted skill: "+v["id"])
        if v["id"] != "PHYSICAL-ME095" and v["actual_proved_time_parallelism"] != "NO_WHOLE_GRAPH_SCAN_PROOF":
            raise ValueError("non-affine time scan falsely asserted")
        if v["id"] == "PHYSICAL-ME095" and v["actual_proved_time_parallelism"] != "AFFINE_ADJUNCT_ONLY_NOT_FULL_NATIVE":
            raise ValueError("affine subfilter scope broadened")
    return {"covered_historical_versions":len(rows), "full_model_certificates":0,
            "original_evidence_changed":False, "historical_completeness_claimed":False}


if __name__ == "__main__":
    print(json.dumps(verify(),sort_keys=True))
