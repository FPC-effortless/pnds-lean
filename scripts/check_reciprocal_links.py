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
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"
          "master/"
          "research/pnds-variant-math-links.json")


def verify(peer_ref="master"):
    if peer_ref != "master" and not re.fullmatch(r"[0-9a-f]{40}",peer_ref):
        raise SystemExit("draft peer-ref must be an immutable commit")
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE.replace("/master/", "/"+peer_ref+"/"), timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("Reciprocal TAC-OSM version/math manifest does NOT match PNDS")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--peer-ref",default="master")
    verify(p.parse_args().peer_ref)
