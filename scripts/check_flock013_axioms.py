"""Check all six declared FLOCK013 operator theorem axiom reports; no proof inference."""
import argparse
import json
from pathlib import Path
import re

EXPECTED = {"FLOCK013."+name for name in ("zero_gain_forward", "exact_gain_increment", "legacy_zero_gain_credit", "positive_gain_credit_unchanged", "nonzero_boundary_counterexample", "masked_chain_stays_zero")}
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
LINE = re.compile(r"^'?(FLOCK013\.[A-Za-z0-9_]+)'? (?:depends on axioms: \[(.*?)\]|does not depend on any axioms)\s*$")


def check(text):
    if "sorryAx" in text:
        raise ValueError("unproved theorem axiom")
    seen = set()
    for line in text.splitlines():
        found = LINE.fullmatch(line.strip())
        if found is None:
            continue
        name, axioms = found.groups()
        if name not in EXPECTED or name in seen:
            raise ValueError("unexpected or duplicate theorem report")
        names = {x.strip() for x in (axioms or "").split(",") if x.strip()}
        if not names <= ALLOWED:
            raise ValueError("unexpected theorem axiom")
        seen.add(name)
    if seen != EXPECTED:
        raise ValueError("missing theorem reports")
    return {"status": "PASS_SCOPED_OPERATOR_AXIOM_AUDIT", "theorems": sorted(seen), "full_model_AD_IEEE_proof": False}


def self_test():
    rows = ["'"+name+"' depends on axioms: [propext, Quot.sound]" for name in sorted(EXPECTED)]
    check("\n".join(rows))
    invalid = [rows[:-1], rows+[rows[0]], [x.replace("Quot.sound", "sorryAx") for x in rows], [x.replace("Quot.sound", "untrusted") for x in rows], rows+["'FLOCK013.unlisted' does not depend on any axioms"]]
    for case in invalid:
        try:
            check("\n".join(case))
        except ValueError:
            continue
        raise ValueError("axiom parser failed to reject invalid evidence")
    return {"self_test_cases": 6, "status": "PASS", "proof_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.path is not None:
        print(json.dumps(check(args.path.read_text())))
    else:
        parser.error("supply a Lean output file or --self-test")
