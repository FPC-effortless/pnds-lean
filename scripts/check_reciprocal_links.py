#!/usr/bin/env python3
"""CI cross-repository exact manifest parity: fetch only the peer JSON file.

This verifies equality of current main/master snapshots, NOT source/model
correspondence nor full Lean mathematical proof. Network/API failures fail
closed instead of silently weakening the two-repository invariant.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "foundation/v1/variant-math-links.json"
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"
          "master/"
          "research/pnds-variant-math-links.json")


def verify(peer_pin=None):
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    url=REMOTE;expected=None;scope="current_mainline_snapshots"
    if peer_pin is not None:
        pin=json.loads(Path(peer_pin).read_text())
        if (pin.get("schema")!="pnds.peer-manifest.v1" or
            pin.get("repository")!="FPC-effortless/TAC-OSM" or
            pin.get("path")!="research/pnds-variant-math-links.json" or
            not re.fullmatch(r"[0-9a-f]{40}",pin.get("commit","")) or
            not re.fullmatch(r"[0-9a-f]{64}",pin.get("sha256",""))):
            raise SystemExit("Invalid pinned peer manifest")
        url="https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"+pin["commit"]+"/"+pin["path"]
        expected=pin["sha256"];scope="paired_draft_commits_only"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw=response.read()
        if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
            raise SystemExit("Pinned peer manifest bytes changed")
        remote=json.loads(raw.decode("utf-8"))
    if local != remote:
        raise SystemExit("Reciprocal TAC-OSM version/math manifest does NOT match PNDS")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "cross_repository_linked_variants": len(local["variants"]),
                      "parity_scope":scope,
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--peer-pin",type=Path)
    verify(p.parse_args().peer_pin)
