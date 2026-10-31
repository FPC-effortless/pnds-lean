#!/usr/bin/env python3
"""CI cross-repository exact manifest parity: fetch only the peer JSON file.

This verifies equality of current main/master snapshots, NOT source/model
correspondence nor full Lean mathematical proof. Network/API failures fail
closed instead of silently weakening the two-repository invariant.
"""
import json
import re
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "foundation/v1/variant-math-links.json"
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"
          "master/"
          "research/pnds-variant-math-links.json")


PIN = LOCAL.parents[2] / "foundation/langcap103/TAC_PIN.json"
if PIN.exists():
    pin = json.loads(PIN.read_text())
    if pin.get("repository") != "FPC-effortless/TAC-OSM" or not re.fullmatch("[0-9a-f]{40}", pin.get("commit", "")):
        raise SystemExit("Invalid immutable TAC source peer pin")
    REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/" +
              pin["commit"] + "/research/pnds-variant-math-links.json")


def verify():
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("Reciprocal TAC-OSM version/math manifest does NOT match PNDS")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    verify()
