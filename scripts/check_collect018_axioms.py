"""Strict scoped audit; exact expected foundational sets, never a full graph proof."""
import json,re,sys
from pathlib import Path
EXPECTED={'COLLECT018.value_path_independent':{'propext','Classical.choice','Quot.sound'},
          'COLLECT018.first_loss_increment':{'propext','Classical.choice','Quot.sound'},
          'COLLECT018.first_loss_not_smaller':{'propext','Classical.choice','Quot.sound'}}
def check(text):
    if re.search(r'sorry|\berror:|\bfatal:',text,re.I):raise ValueError('Lean error/proof hole in scoped audit')
    seen={}
    for line in text.splitlines():
        if 'axiom' not in line:continue
        empty=re.fullmatch(r"['`]([^'`]+)['`] does not depend on any axioms\.?",line.strip())
        normal=re.fullmatch(r"['`]([^'`]+)['`] depends on axioms: \[([^\]]*)\]\.?",line.strip())
        if empty:name=empty[1];items=[]
        elif normal:name=normal[1];items=[x.strip() for x in normal[2].split(',') if x.strip()]
        else:raise ValueError('unrecognized scoped receipt')
        if name not in EXPECTED or name in seen or len(items)!=len(set(items)) or set(items)!=EXPECTED[name]:raise ValueError('missing/extra/duplicated/unreviewed axiom set')
        seen[name]=items
    if set(seen)!=set(EXPECTED):raise ValueError('missing theorem receipt')
    return dict(status='PASS_EXACT_SCOPED_AXIOM_SETS',observed=seen,full_graph_proved=False)
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]).read_text()),sort_keys=True))
