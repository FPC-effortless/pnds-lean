#!/usr/bin/env python3
"""Legacy manifest equality plus explicitly source-pinned prospective candidates.

Only the new optional prospective_candidates field is excluded from historical
snapshot equality. Every prospective row is separately compared with the peer
at an immutable metadata commit, including exact map bytes and original source
hashes. This is provenance correspondence, not Lean or trained qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = "foundation/v1/variant-math-links.json"
RAW = "https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"
PEER_REGISTRY = "research/pnds-variant-math-links.json"
REMOTE = RAW+"master/"+PEER_REGISTRY


class ReciprocalError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReciprocalError(message)


def fetch_bytes(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{"+str(length)+"}", value) is not None


def _path(base, relative):
    require(isinstance(relative, str) and relative and not relative.startswith("/"),
            "repository-relative path required")
    result = (base/relative).resolve()
    require(result.is_relative_to(base.resolve()), "candidate path escaped repository")
    return result


def verify_candidate(base, row, read_remote):
    fields = {"id", "tac_source_commit", "source_map_path", "tac_source_map_path",
              "lean_path", "python_path", "status"}
    require(isinstance(row, dict) and set(row) == fields, "candidate fields are incomplete or unexpected")
    require(row["id"] == "LANGCACHE104-DEEP-ASSOCIATIVE-CARRIED-CACHE",
            "candidate needs its own reviewed verifier")
    require(row["status"] == "OPEN" and _sha(row["tac_source_commit"], 40),
            "candidate source pin missing or unsupported proof promotion")
    map_path = _path(base, row["source_map_path"])
    pin_path = map_path.parent/"TAC_PIN.json"
    require(pin_path.is_file(), "missing immutable prospective TAC companion pin")
    pin = json.loads(pin_path.read_text())
    require(pin.get("schema") == "pnds.langcache104.tac_pin.v1" and
            _sha(pin.get("tac_metadata_commit"), 40) and
            pin.get("tac_source_commit") == row["tac_source_commit"] and
            _sha(pin.get("source_map_sha256"), 64), "invalid prospective TAC companion pin")
    metadata = pin["tac_metadata_commit"]
    peer = json.loads(read_remote(RAW+metadata+"/"+PEER_REGISTRY))
    peers = peer.get("prospective_candidates", [])
    matches = [x for x in peers if isinstance(x, dict) and x.get("id") == row["id"]]
    require(len(matches) == 1 and matches[0] == row,
            "prospective candidate differs from immutable TAC peer")
    local_map_bytes = map_path.read_bytes()
    peer_map_bytes = read_remote(RAW+metadata+"/"+row["tac_source_map_path"])
    require(local_map_bytes == peer_map_bytes and
            _digest(local_map_bytes) == pin["source_map_sha256"],
            "prospective source-map bytes or hash differ from immutable peer")
    source_map = json.loads(local_map_bytes)
    require(source_map.get("variant_id") == row["id"] and
            source_map.get("tac_commit") == row["tac_source_commit"],
            "source map variant/implementation pin mismatch")
    require(source_map.get("lean_module") == row["lean_path"] and
            source_map.get("python_reference") == row["python_path"] and
            _path(base, row["lean_path"]).is_file() and
            _path(base, row["python_path"]).is_file(),
            "candidate mathematical formulation missing or mislinked")
    math_hashes = source_map.get("mathematical_files_sha256")
    require(isinstance(math_hashes, dict) and
            row["lean_path"] in math_hashes and row["python_path"] in math_hashes,
            "mathematical formulation hash map missing")
    for relative, expected in math_hashes.items():
        mathematical_file = _path(base, relative)
        require(_sha(expected, 64) and mathematical_file.is_file() and
                _digest(mathematical_file.read_bytes()) == expected,
                "mathematical formulation hash mismatch: "+relative)
    paths, hashes = source_map.get("tac_source_paths"), source_map.get("tac_source_sha256")
    require(isinstance(paths, list) and bool(paths) and isinstance(hashes, dict) and
            len(paths) == len(set(paths)) and set(hashes) == set(paths),
            "complete original-source hash map required")
    for relative in paths:
        _path(base, relative)
        require(_sha(hashes[relative], 64), "invalid original-source SHA256")
        original = read_remote(RAW+row["tac_source_commit"]+"/"+relative)
        require(_digest(original) == hashes[relative], "original TAC source hash mismatch: "+relative)
    return {"id": row["id"], "candidate_equal_at_pinned_peer": True,
            "source_map_bytes_equal": True, "original_source_files_verified": len(paths),
            "whole_model_source_equivalence_established": False}


def verify(base=ROOT, read_remote=fetch_bytes):
    local = json.loads((base/LEGACY_PATH).read_text(encoding="utf-8"))
    remote = json.loads(read_remote(REMOTE))
    local_legacy, remote_legacy = dict(local), dict(remote)
    candidates = local_legacy.pop("prospective_candidates", [])
    remote_legacy.pop("prospective_candidates", None)
    require(local_legacy == remote_legacy, "Legacy TAC-OSM/PNDS manifest does NOT match")
    require(isinstance(candidates, list), "prospective_candidates must be an array")
    ids = [x.get("id") for x in candidates if isinstance(x, dict)]
    require(len(ids) == len(candidates) and len(ids) == len(set(ids)),
            "candidate IDs missing or duplicated")
    verified = [verify_candidate(base, row, read_remote) for row in candidates]
    result = {"legacy_manifest_equal": True,
              "cross_repository_linked_legacy_variants": len(local_legacy["variants"]),
              "prospective_candidates": verified,
              "whole_model_source_equivalence_established": False}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        verify()
    except (ReciprocalError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit("Reciprocal verification blocked: "+str(exc)) from exc
