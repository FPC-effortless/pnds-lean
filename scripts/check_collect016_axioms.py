#!/usr/bin/env python3
"""Audit two existing facts against their exact reviewed foundational axioms."""
from pathlib import Path
import json
import re
import sys

NAMES = ('COLLECT016.stopGradient_value', 'COLLECT016.stopGradient_declared_tangent')
APPROVED = {name: ('propext', 'Classical.choice', 'Quot.sound') for name in NAMES}
DEPENDENCIES = re.compile(
    r"(?P<quote>['`])(?P<name>[^'`]+)(?P=quote) depends on axioms: \[(?P<axioms>[^\[\]]*)\]\.?")


def check(text):
    if re.search(r"sorryax|declaration uses ['`]sorry['`]|\berror:|\bfatal:", text, re.I):
        raise ValueError('admitted goal or Lean error in scoped receipt')
    seen = {}
    for line in text.splitlines():
        if 'axiom' not in line.lower():
            continue
        match = DEPENDENCIES.fullmatch(line.strip())
        if match is None or match['name'] not in NAMES or match['name'] in seen:
            raise ValueError('unexpected or duplicated axiom receipt')
        axioms = tuple(part.strip() for part in match['axioms'].split(','))
        if len(axioms) != len(set(axioms)) or set(axioms) != set(APPROVED[match['name']]):
            raise ValueError('missing, duplicated or unreviewed axiom dependency')
        seen[match['name']] = list(axioms)
    if set(seen) != set(NAMES):
        raise ValueError('missing scoped theorem axiom receipt')
    return {'scoped_axiom_receipt': 'PASS_REVIEWED_FOUNDATIONAL_AXIOMS_ONLY',
            'theorems': list(NAMES), 'reviewed_axioms': {n: list(a) for n, a in APPROVED.items()},
            'observed_axioms': seen, 'full_graph_or_AD_proved': False}


def self_test():
    good = '\n'.join("'" + name + "' depends on axioms: [propext, Classical.choice, Quot.sound]"
                     for name in NAMES)
    assert check(good)['scoped_axiom_receipt'] == 'PASS_REVIEWED_FOUNDATIONAL_AXIOMS_ONLY'
    bad = [
        '', good.splitlines()[0],
        good.replace('propext, Classical.choice, Quot.sound', 'propext, Classical.choice, customAssumption', 1),
        good + '\nerror: test', good + '\nsorryAx',
        good + '\n' + good.splitlines()[0],
        good.replace('Quot.sound]', 'Quot.sound]; depends on axioms: [custom]', 1),
        good + "\n'Other.theorem' depends on axioms [custom]",
        good.replace('propext, Classical.choice, Quot.sound', 'propext, Quot.sound', 1),
        good.replace('propext, Classical.choice, Quot.sound', 'propext, Classical.choice, Quot.sound, propext', 1),
        good.replace('propext, Classical.choice, Quot.sound', '', 1),
        good.replace(' depends on axioms: [propext, Classical.choice, Quot.sound]',
                     ' does not depend on any axioms', 1),
    ]
    for text in bad:
        try:
            check(text)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed scoped axiom receipt accepted')
    return {'self_test': 'PASS', 'positive_cases': 1, 'rejection_cases': len(bad), 'Lean_run': False}


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        print(json.dumps(self_test()))
    elif len(sys.argv) == 2:
        print(json.dumps(check(Path(sys.argv[1]).read_text())))
    else:
        raise SystemExit('usage: check_collect016_axioms.py LOG | --self-test')
