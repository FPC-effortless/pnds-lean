#!/usr/bin/env python3
"""Execute Lean's axiom printer for every PAR-MEM-010 theorem; fail closed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'propext','Classical.choice','Quot.sound'}
MODULES=['SourceAlignedMemory','SourceAlignedVerification','SourceAlignedModel',
         'SourceAlignedScan','SourceAlignedExecution','SourceAlignedDifferentiation',
         'SourceAlignedRoundoff','SourceAlignedPrograms']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    expected=[]
    for module in MODULES:
        contents=(ROOT/f'PNDS/{module}.lean').read_text()
        expected += [f'PNDS.{module}.{name}' for name in re.findall(r'^theorem\s+(\w+)',contents,re.M)]
    source=(ROOT/'AuditSourceAligned.lean').read_text()
    listed=re.findall(r'^#print axioms (\S+)',source,re.M)
    assert listed==expected,'audit omits/reorders a theorem; regenerate explicit audit file'
    result=subprocess.run(['lake','env','lean','AuditSourceAligned.lean'],cwd=ROOT,
                          text=True,capture_output=True)
    print(result.stdout,end='')
    print(result.stderr,end='')
    assert result.returncode==0,'Lean axiom audit did not compile'
    assert 'sorry' not in result.stdout.lower(),'proof placeholder in audit'
    found={}
    for name,axioms in re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]",result.stdout):
        names={a.strip() for a in axioms.split(',') if a.strip()}
        assert names<=ALLOWED,(name,names-ALLOWED)
        found[name]=sorted(names)
    for name in re.findall(r"'([^']+)' does not depend on any axioms",result.stdout):
        found[name]=[]
    assert set(found)==set(expected),('incomplete axiom output',set(expected)-set(found))
    report=dict(schema='parmem010.lean_axiom_audit.v1',status='PASS',
                theorem_count=len(expected),allowed_foundational_axioms=sorted(ALLOWED),
                declarations=found,ieee_or_torch_refinement_proved=False,
                toolchain=(ROOT/'lean-toolchain').read_text().strip(),
                proof_sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                    for p in [*[f'PNDS/{m}.lean' for m in MODULES],
                              'AuditSourceAligned.lean','lake-manifest.json']})
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
