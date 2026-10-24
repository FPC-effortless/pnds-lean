#!/usr/bin/env python3
"""Check a versioned PNDS/TAC-OSM mathematics/evidence foundation snapshot.

Stdlib-only. Deliberately OFFLINE: checks schemas and locally maintained
provenance, not whether an external GitHub branch still exists or whether a
learned PyTorch graph conforms to a theorem. Source SHA is authoritative.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT / "foundation" / "v1"
SCHEMA = "pnds.tac_osm.foundation.v1"
EVIDENCE_KINDS = {
    "synthetic_diagnostic", "trained_exposed_dev", "no_fit_audit",
    "math_candidate", "interface_only", "formal_contract",
}
REQUIRED_LANES = {
    "core", "system1", "physical", "enterprise", "temporal",
    "edge", "counterfactual", "eswam", "qualification",
}
SENTINEL_FAILURES = {"MEM-002", "SYS-001", "PHY-003", "ENT-001", "ESW-004"}
REQUIRED_PATHS = [
    "README.md", "ARCHITECTURE.md", "PROOF_OBLIGATIONS.md",
    "prior-math.json", "archive-gaps.json",
    "registry.json", "evidence-ledger.jsonl",
]


class FoundationError(ValueError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise FoundationError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path = FOUNDATION):
    for name in REQUIRED_PATHS:
        ensure((root / name).is_file(), f"missing foundation file: {name}")

    registry = read_json(root / "registry.json")
    gaps = read_json(root / "archive-gaps.json")
    prior = read_json(root / "prior-math.json")
    ensure(registry.get("schema") == SCHEMA, "unrecognized foundation schema")
    ensure(registry.get("foundation_version") == "1.0.0", "wrong foundation snapshot version")
    ensure(registry.get("as_of_date") == "2026-10-24", "unexpected version cut")
    ensure(set(registry.get("research_lanes", [])) == REQUIRED_LANES, "research lanes omitted")
    policy = registry.get("normative_policy", {})
    ensure(policy.get("real_world_primary") is True, "authentic data rule lost")
    ensure(policy.get("synthetic_promotion_forbidden") is True, "synthetic promotion prohibited")
    ensure(policy.get("live_model_parameters_min") == 95000
           and policy.get("live_model_parameters_max") == 105000,
           "100k prospective budget changed")
    ensure(policy.get("within_comparator_parameter_relative_tolerance") == 0.01,
           "matched comparator +/-1% budget changed")
    ensure(policy.get("match_training_data") and policy.get("match_training_compute"),
           "fair train-data/compute parity was weakened")
    ensure(policy.get("match_CPU_wall_clock") is False,
           "CPU wall time is not compulsory equality gate")
    obligations = set(registry.get("proof_obligations", []))
    ensure(obligations == {"MATH-" + f"{n:03d}" for n in range(1, 14)},
           "missing mathematical proof obligations")

    records = registry.get("variants")
    ensure(isinstance(records, list) and len(records) >= 17,
           "source-pinned research lines have disappeared")
    ids = [r.get("id") for r in records]
    ensure(len(set(ids)) == len(ids), "duplicate variant ID")
    ensure({r.get("lane") for r in records} == REQUIRED_LANES,
           "a research lane has no versioned source")
    recorded = set(ids)
    for row in records:
        ident = row.get("id")
        ensure(isinstance(ident, str) and ident, "missing version id")
        ensure(row.get("lane") in REQUIRED_LANES,
               f"{ident} invalid research lane")
        ensure(row.get("evidence_tier") in EVIDENCE_KINDS,
               f"{ident} unknown evidence class")
        ensure(bool(row.get("implementation")) and bool(row.get("version")),
               f"{ident} no defined version / implementation")
        ensure(bool(row.get("status")) and isinstance(row.get("limitations"), list)
               and len(row["limitations"]) > 0, f"{ident} lacks limitations")
        ensure(isinstance(row.get("supported"), list), f"{ident} missing supported claims")
        ensure(isinstance(row.get("parent_versions"), list), f"{ident} missing parents")
        ensure(set(row.get("proof_obligations", [])) <= obligations,
               f"{ident} unknown proof obligations")
        src = row.get("source") or {}
        ensure(src.get("repository") == "FPC-effortless/TAC-OSM",
               f"{ident} must point to authoritative implementation source")
        ensure(re.fullmatch(r"[0-9a-f]{40}", src.get("commit_sha", "")) is not None,
               f"{ident} source commit is not a pinned Git SHA")
        ensure(isinstance(src.get("pull_request"), int)
               and src["pull_request"] > 0, f"{ident} missing source PR")
        ensure(src.get("evidence_path") is None or
               (isinstance(src.get("evidence_path"), str)
                and not src["evidence_path"].startswith("/")),
               f"{ident} invalid source-relative evidence path")
        params = row.get("params")
        ensure(params is None or isinstance(params, int) and params > 0,
               f"{ident} invalid parameter inventory")
        ensure(ident not in set(row["parent_versions"]),
               f"{ident} has self-referential version ancestry")

    seen = {}
    source_path = root / "evidence-ledger.jsonl"
    for lineno, raw in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        ensure(bool(raw.strip()), f"blank row at line {lineno}")
        row = json.loads(raw)
        key = row.get("event_id")
        ensure(isinstance(key, str) and key and key not in seen,
               f"duplicate/missing evidence event {key} line {lineno}")
        seen[key] = row
        ensure(row.get("foundation_version") == "1.0.0",
               f"evidence version mismatches at {key}")
        ensure(row.get("variant_id") in recorded,
               f"evidence {key} refers to removed variant")
        ensure(row.get("outcome") in {"pass", "fail", "mixed", "open"},
               f"invalid outcome for {key}")
        ensure(row.get("epistemic_scope") in
               {"synthetic_only", "numerical_strict_exception", "unverified",
                "exposed_real_dev", "math_candidate", "interface_only",
                "formal_contract", "no_fit_audit"},
               f"unknown evidence claim tier {key}")
        ensure(bool(row.get("finding")) and row.get("immutable_originals") is True,
               f"missing original retained evidence {key}")
        ensure(row.get("source_of_truth") == "original_TAC_OSM_commit_pinned_in_registry",
               f"evidence {key} has no original source of truth")
        ensure(not (row["epistemic_scope"] == "synthetic_only"
                    and row.get("promotion_scope") == "independent_real_capability"),
               f"synthetic finding illegally promoted: {key}")

    ensure(SENTINEL_FAILURES <= set(seen), "historical negative findings were dropped")
    ensure(all(seen[k]["outcome"] == "fail" for k in SENTINEL_FAILURES),
           "historical failure overwritten by unsupported pass")
    ensure(len(seen) >= 23, "original version-one evidence ledger was truncated")
    ensure(gaps.get("completeness") == "PARTIAL_ACTIVE_AND_SENTINEL_INVENTORY"
           and len(gaps.get("explicitly_unindexed_ranges", [])) >= 5,
           "historical archival gaps silently omitted")
    ensure(len(prior.get("unmerged_pnds_research", [])) >= 3,
           "original PNDS formal research branches omitted")

    # No network: the external commit's actual content may still be missing or
    # altered after source transfer. Independent source-replay is separate.
    return {"foundation_version": registry["foundation_version"],
            "indexed_versions": len(records), "evidence_records": len(seen),
            "negative_sentinels": len(SENTINEL_FAILURES),
            "unindexed_ranges": len(gaps["explicitly_unindexed_ranges"]),
            "external_source_files_fetched_by_this_check": 0,
            "real_world_superiority_established_by_this_check": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=FOUNDATION)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
