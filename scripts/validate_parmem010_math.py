#!/usr/bin/env python3
"""Validate the prospective source-pinned record without promoting legacy proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def check(tac_root=None):
    contract=json.loads((ROOT/'foundation/v1/PAR-MEM-010-FORMAL-CONTRACT.json').read_text())
    links=json.loads((ROOT/'foundation/v1/variant-math-links.json').read_text())
    assert contract in links['prospective_studies'], 'missing reciprocal study registration'
    assert contract['source']['commit']=='63ea0aa67904fa946b9baa69d2e4b0cc23dc1c9a'
    assert contract['complete_formal_verification'] is False
    assert contract['training_done'] is False and contract['real_world_capability_inferred'] is False
    assert len(contract['initial_open_obligations'])>=6
    assert len(contract['open_obligations'])>=5
    extension=contract['proof_extension']
    assert extension['torch_runtime_formally_refined'] is False
    assert extension['local_kernel_and_transcendental_accuracy_formally_proved'] is False
    assert extension['full_graph_scalar_compiler']=='PROVED'
    assert extension['all_history_jvp_and_vjp']=='PROVED_AT_REGULAR_POINTS'
    for path,digest in extension['reference_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path in [*contract['lean'].values(),contract['python']['complete_forward_reference'],
                 contract['python']['exact_operator_reference'],contract['python']['exact_tests']]:
        assert (ROOT/path).is_file(),path
    reference=ROOT/contract['python']['complete_forward_reference']
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==contract['python']['forward_reference_sha256']
    theorem_count=0
    for path in contract['lean'].values():
        if not path.startswith('PNDS/'):
            continue
        contents=(ROOT/path).read_text()
        # These files do not contain nested comments. The kernel audit is a
        # separate required execution check; this lexical guard cannot replace it.
        stripped=re.sub(r'/\-.*?\-/','',contents,flags=re.S)
        stripped=re.sub(r'--[^\n]*','',stripped)
        assert not re.search(r'\bsorry\b|^\s*(?:axiom\b|admit\s*$)',stripped,re.M),path
        theorem_count+=len(re.findall(r'^(?:@\[[^\n]*\]\s*)?theorem\s+',stripped,flags=re.M))
    if tac_root is not None:
        assert links==json.loads((tac_root/'research/pnds-variant-math-links.json').read_text())
        assert contract==json.loads((tac_root/'research/parallel-six/PAR-MEM-010-FORMAL-CONTRACT.json').read_text())
        for path,expected in contract['source']['sha256'].items():
            assert hashlib.sha256((tac_root/path).read_bytes()).hexdigest()==expected,path
        receipt=json.loads((tac_root/contract['python']['source_correspondence_receipt']).read_text())
        assert receipt['source_commit']==contract['source']['commit']
        assert receipt['reference_sha256']==contract['python']['forward_reference_sha256']
        assert receipt['optimizer_steps']==0 and receipt['real_data_evaluation'] is False
        assert len(receipt['checks'])>=439
    return dict(study=contract['id'],source_pinned=True,declared_theorems=theorem_count,
                local_reciprocal_files_checked=tac_root is not None,
                kernel_audit_performed=False,complete_formal_verification=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tac-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.tac_root),sort_keys=True))
