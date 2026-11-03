#!/usr/bin/env python3
"""Fail-closed offline per-variant mathematical formulation/provenance validator."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "foundation/v1/variant-math-links.json"
REGISTRY = ROOT / "foundation/v1/registry.json"


class VariantMathError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VariantMathError(message)


def check(base: Path = ROOT) -> dict:
    manifest = json.loads((base / "foundation/v1/variant-math-links.json").read_text())
    registry = json.loads((base / "foundation/v1/registry.json").read_text())
    require(manifest.get("schema") == "pnds.tac_osm.variant_math_links.v1", "bad link schema")
    require(manifest.get("canonical_authority") == {
        "mathematics": "FPC-effortless/pnds-lean",
        "source_and_evidence": "FPC-effortless/TAC-OSM"}, "wrong repository roles")
    records = registry["variants"]
    links = manifest.get("variants")
    require(isinstance(links, list), "missing variant links")
    indexed = {v["id"]: v for v in records}
    ids = [v.get("id") for v in links]
    require(len(ids) == len(set(ids)) and set(ids) == set(indexed),
            "PNDS registry and reciprocal per-variant links diverged")
    require(len(links) >= 17, "original 17-variant index was truncated")
    for v in links:
        name = v["id"]
        source = indexed[name]["source"]
        tac, pnds = v.get("tac", {}), v.get("pnds", {})
        require(v.get("lane") == indexed[name]["lane"] and
                v.get("state_kind") == indexed[name]["state_kind"],
                f"{name} state/lane provenance mismatch")
        require(tac.get("repository") == "FPC-effortless/TAC-OSM" and
                tac.get("commit") == source["commit_sha"] and
                tac.get("pull_request") == source["pull_request"] and
                tac.get("evidence_path") == source.get("evidence_path"),
                f"{name} TAC original PR/SHA/evidence pin changed")
        require(re.fullmatch("[0-9a-f]{40}", tac["commit"]) is not None,
                f"{name} invalid commit pin")
        if name in {'COLLECT018-PHASE_JOINT', 'COLLECT018-PHASE_FIRST', 'COLLECT018-CONTEXT_JOINT', 'COLLECT018-CONTEXT_FIRST', 'COLLECT018-GRU_JOINT', 'COLLECT018-GRU_FIRST'}:
            require(tac['commit'] == '7b9af0d5dccc1f52c5055db850f8f8c28c3ae70c' and pnds.get('repository') == 'FPC-effortless/pnds-lean' and
                    pnds.get('pull_request') == 109 and pnds.get('commit') == 'c15d2ee51991b97bcc887461f85ed89a2fac5e00',
                    f'{name} COLLECT018 source/math authority mismatch')
            require(pnds.get('lean_path') == 'PNDS/COLLECT018.lean' and pnds.get('lean_symbol') == 'COLLECT018.TypedSequenceGraph' and
                    pnds.get('python_path') == 'foundation/v1/collect018/reference.py' and pnds.get('python_symbol') == 'whole_reference',
                    f'{name} missing source-specific C18 formulations')
        elif name in {"FLOCK013-PHASE-ADJOINT", "FLOCK013-CONTEXT-ADJOINT"}:
            require(tac["commit"] == "41af8c423eceeb2bb4f1d95ae391ee807779f603" and tac.get("pull_request") == 705 and
                    pnds.get("repository") == "FPC-effortless/pnds-lean" and pnds.get("pull_request") == 111 and
                    pnds.get("commit") == "4f0ea7f3116f752542f1df12a20bde0a5bbe91be", f"{name} source/adjoint authority mismatch")
            require(pnds.get("lean_path") == "PNDS/FLOCK013.lean" and pnds.get("lean_symbol") == "FLOCK013.candidate" and
                    pnds.get("python_path") == "foundation/v1/flock013/adjoint_reference.py" and
                    pnds.get("python_symbol") == "memory_gain_adjoint", f"{name} missing partial adjoint formulations")
        elif name in {"COLLECT016-FULL_BPTT", "COLLECT016-DETACH_128", "COLLECT016-CURRENT_ONLY", "COLLECT017-ATTENTION"}:
            expected_source = "e5813459c9337af058cdfb2d3e864c767c457053" if name == "COLLECT017-ATTENTION" else "2209e1328e3b42db4dd48cd702a21a55ffad24af"
            require(tac["commit"] == expected_source and
                    pnds.get("repository") == "FPC-effortless/pnds-lean" and
                    pnds.get("pull_request") == 107 and
                    pnds.get("commit") == "090c7f8b1d7af0c8680c5856df156d3d7c181e2e",
                    f"{name} COLLECT016 source/math authority mismatch")
            require(pnds.get("lean_path") == "PNDS/COLLECT016.lean" and
                    pnds.get("lean_symbol") == "COLLECT016.CausalSequenceGraph" and
                    pnds.get("python_path") == "foundation/v1/collect016/reference.py" and
                    pnds.get("python_symbol") == "whole_reference",
                    f"{name} missing source-specific Lean/Python paths")
        else:
            require(pnds.get("repository") == "FPC-effortless/pnds-lean" and
                    pnds.get("pull_request") == 5, f"{name} mathematical authority missing")
            require(pnds.get("lean_path") == "PNDS/TACOSMVariantContracts.lean" and
                    pnds.get("python_path") == "foundation/v1/variant_math.py" and
                    pnds.get("python_symbol") == "evaluate_variant",
                    f"{name} missing exact paired Lean and Python paths")
        require(v.get("formalization_status") == "PARTIAL_SHARED_KERNEL_NOT_WHOLE_MODEL" and
                v.get("lean_full_graph_proved") is False and
                v.get("python_source_equivalence_verified") is False and
                v.get("real_world_capability_inferred") is False,
                f"{name} unsupported mathematical or scientific proof promotion")
        require(bool(v.get("formula")) and isinstance(v.get("operator_contracts"), list),
                f"{name} no mathematical equation or operator status")
        require((base / pnds["lean_path"]).is_file() and
                (base / pnds["python_path"]).is_file(), f"{name} missing math source")
    lean = (base / "PNDS/TACOSMVariantContracts.lean").read_text(encoding="utf-8")
    py = (base / "foundation/v1/variant_math.py").read_text(encoding="utf-8")
    require("def indexedVariants" in lean and "def evaluate_kernel" in py and
            "def evaluate_variant" in py,
            "Lean/Python entrypoints removed")
    for v in links:
        needle = f'⟨"{v["id"]}", "{v["tac"]["commit"]}", .{v["kernel"]}⟩'
        require(needle in lean, f"{v['id']} missing exact Lean variant/source/kernel")
    require("sorry" not in re.sub(r"/-!.*?-/", "", lean, flags=re.S) and
            "admit" not in re.sub(r"/-!.*?-/", "", lean, flags=re.S),
            "Lean proof hole in contract module")
    # Explicit module import/verification is checked by Lean CI, not string search.
    return {"indexed_variants": len(links), "all_source_pinned": True,
            "all_have_lean_python_links": True, "complete_full_graph_proofs": 0,
            "online_cross_repository_equality_checked": False}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
