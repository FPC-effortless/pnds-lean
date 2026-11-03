#!/usr/bin/env python3
"""Validate only two named COLLECT016 empty-axiom receipts, not a model proof."""
from pathlib import Path
import json
import re
import sys

NAMES = ('COLLECT016.stopGradient_value', 'COLLECT016.stopGradient_declared_tangent')
EMPTY = re.compile(r"(?P<quote>['`])(?P<name>[^'`]+)(?P=quote) does not depend on any axioms\.?")


def check(text):
    if re.search(r"sorryax|declaration uses ['`]sorry['`]|\berror:|\bfatal:", text, re.I):
        raise ValueError('admitted goal or Lean error in scoped receipt')
    seen = set()
    for line in text.splitlines():
        if 'axiom' not in line.lower():
            continue
        match = EMPTY.fullmatch(line.strip())
        if match is None or match['name'] not in NAMES or match['name'] in seen:
            raise ValueError('unexpected, duplicated or nonempty axiom dependency')
        seen.add(match['name'])
    if seen != set(NAMES):
        raise ValueError('missing scoped theorem axiom receipt')
    return {'scoped_axiom_receipt': 'PASS', 'theorems': list(NAMES),
            'expected_axioms': [], 'full_graph_or_AD_proved': False}


def self_test():
    good = '\n'.join("'" + name + "' does not depend on any axioms" for name in NAMES)
    assert check(good)['scoped_axiom_receipt'] == 'PASS'
    bad = [
        '', good.splitlines()[0],
        good.replace('does not depend on any axioms', 'depends on axioms [customAssumption]', 1),
        good + '\nerror: test', good + '\nsorryAx',
        good + '\n' + good.splitlines()[0],
        good.replace('does not depend on any axioms',
                     'does not depend on any axioms; depends on axioms [custom]', 1),
        good + "\n'Other.theorem' depends on axioms [custom]",
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
