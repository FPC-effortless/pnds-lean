#!/usr/bin/env python3
"""Fail-closed audit for language theorem axioms emitted by Lean 4.

Does not trust a green compiler exit alone. Checks every #print axioms name,
duplicates, unexpected/undeclared results, absent records, and nonstandard
axioms. The checked theorem set is source-controlled in AuditLanguage.lean.
"""
from __future__ import annotations
import argparse
import pathlib
import re

ALLOWED = frozenset({"propext", "Quot.sound", "Classical.choice"})
EXPECTED_RE = re.compile(r"(?m)^#print axioms ([A-Za-z0-9_.]+)\s*$")
RECORD_RE = re.compile(
    r"(?m)^'([A-Za-z0-9_.]+)' "
    r"(?:depends on axioms: \[([^\]\r\n]*)\]|does not depend on any axioms)\s*$"
)

def verify_audit(audit_source: str, lean_stdout: str, expected_count: int) -> dict:
    expected = EXPECTED_RE.findall(audit_source)
    if not expected or len(expected) != expected_count:
        raise ValueError(f"expected {expected_count} declarations; source has {len(expected)}")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate #print axioms declaration")
    if "sorryAx" in lean_stdout or re.search(r"(?m)^.*error:.*$", lean_stdout):
        raise ValueError("Lean reported sorryAx or error")
    matches = RECORD_RE.findall(lean_stdout)
    names = [name for name, _ in matches]
    if len(matches) != expected_count or set(names) != set(expected) or len(set(names)) != len(names):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(f"incomplete or duplicate axiom audit: expected={expected_count}, "
                         f"observed={len(matches)}, missing={missing}, extra={extra}")
    for name, raw in matches:
        axioms = {a.strip() for a in raw.split(",") if a.strip()}
        unexpected = axioms - ALLOWED
        if unexpected:
            raise ValueError(f"unexpected axioms in {name}: {sorted(unexpected)}")
    return {
        "checked_theorems":len(matches),
        "allowed_axioms":sorted(ALLOWED),
        "zero_axiom_theorems":sum(not raw.strip() for _, raw in matches),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=pathlib.Path, default=pathlib.Path("AuditLanguage.lean"))
    ap.add_argument("--lean-log", type=pathlib.Path, required=True)
    ap.add_argument("--expected", type=int, default=30)
    args=ap.parse_args()
    print(verify_audit(args.audit.read_text(),args.lean_log.read_text(),args.expected))

if __name__=="__main__":
    main()
