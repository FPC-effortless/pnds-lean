#!/usr/bin/env python3
"""CI cross-repository exact manifest parity: fetch only the peer JSON file.

By default this compares current main/master snapshots. --peer-ref compares
against an explicit immutable TAC commit for a paired draft. Neither check
establishes source/model correspondence or full Lean mathematical proof.
Network/API failures fail closed instead of weakening the invariant.
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "foundation/v1/variant-math-links.json"
PEER_ROOT = "https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/"
PEER_PATH = "research/pnds-variant-math-links.json"
REMOTE = PEER_ROOT + "master/" + PEER_PATH


def immutable_ref(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise argparse.ArgumentTypeError("--peer-ref requires a full immutable 40-hex TAC commit")
    return value.lower()


def verify(peer_ref=None):
    # Validate direct callers too; a branch/tag override is never accepted.
    peer_ref = immutable_ref(peer_ref) if peer_ref is not None else None
    remote_url = REMOTE if peer_ref is None else PEER_ROOT + peer_ref + "/" + PEER_PATH
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(remote_url, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("Reciprocal TAC-OSM version/math manifest does NOT match PNDS")
    result = {"reciprocal_manifest_equal": True,
              "cross_repository_linked_variants": len(local["variants"]),
              "whole_model_source_equivalence_established": False}
    if peer_ref is not None:
        result.update(peer_ref=peer_ref, comparison_scope="immutable_peer_commit_snapshot")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-ref", type=immutable_ref,
                        help="Full immutable TAC commit; default compares PNDS main to TAC master")
    verify(parser.parse_args().peer_ref)
