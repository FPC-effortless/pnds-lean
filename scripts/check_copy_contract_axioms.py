"""Fail closed on missing/duplicate/unreviewed scoped theorem receipts."""
import argparse
import json
from pathlib import Path
import re

EXPECTED = {"CopyTaskContract." + n for n in (
    "lookup_empty", "lookup_singleton", "singleton_key_blind", "latest_write_wins",
    "witness_excludes_key_blind", "same_query_no_witness", "two_key_witness")}
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
LINE = re.compile(r"^'?(CopyTaskContract\.[A-Za-z0-9_]+)'? (?:depends on axioms: \[(.*?)\]|does not depend on any axioms)\s*$")


def check(text):
    if any(word in text for word in ("sorry", "error:", "panic")):
        raise ValueError("incomplete or erroneous proof output")
    seen = set()
    for line in text.splitlines():
        found = LINE.fullmatch(line.strip())
        if found is None:
            continue
        name, deps = found.groups()
        if name not in EXPECTED or name in seen:
            raise ValueError("unexpected/duplicate theorem")
        if not {d.strip() for d in (deps or "").split(",") if d.strip()} <= ALLOWED:
            raise ValueError("unreviewed theorem dependency")
        seen.add(name)
    if seen != EXPECTED:
        raise ValueError("missing theorem reports")
    return {"status": "PASS_SEVEN_SCOPED_FINITE_RECORD_AXIOM_REPORTS",
            "theorems": sorted(seen), "neural_source_IEEE_AD_proof": False}


def self_test():
    rows = [f"'{name}' does not depend on any axioms" for name in sorted(EXPECTED)]
    check("\n".join(rows))
    for bad in (rows[:-1], rows + [rows[0]], rows + ["sorryAx"],
                [r.replace("does not depend on any axioms", "depends on axioms: [Untrusted]") for r in rows],
                rows + ["'CopyTaskContract.extra' does not depend on any axioms"]):
        try:
            check("\n".join(bad))
        except ValueError:
            continue
        raise ValueError("invalid proof receipt accepted")
    return {"status": "PASS_RECEIPT_PARSER_FIXTURES", "proof_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else check(args.path.read_text())))
