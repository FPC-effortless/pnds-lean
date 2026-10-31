#!/usr/bin/env python3
"""CI cross-repository exact manifest parity: fetch only the peer JSON file.

This verifies equality of current main/master snapshots, NOT source/model
correspondence nor full Lean mathematical proof. Network/API failures fail
closed instead of silently weakening the two-repository invariant.
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "foundation/v1/variant-math-links.json"
REMOTE_PREFIX = "https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"


def verify(peer_ref="master"):
    # The production default stays master. Paired draft PRs must name an
    # immutable peer commit; arbitrary hosts, moving branches and URLs denied.
    if peer_ref != "master" and re.fullmatch(r"[0-9a-f]{40}", peer_ref) is None:
        raise ValueError("peer ref must be master or an immutable 40-hex commit")
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    remote_url = REMOTE_PREFIX + peer_ref + "/research/pnds-variant-math-links.json"
    with urllib.request.urlopen(remote_url, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("Reciprocal TAC-OSM version/math manifest does NOT match PNDS")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "peer_ref": peer_ref,
                      "production_snapshots_compared": peer_ref == "master",
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-ref", default="master")
    args = parser.parse_args()
    verify(args.peer_ref)
