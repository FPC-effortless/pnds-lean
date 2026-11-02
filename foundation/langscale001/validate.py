"""Local content pins; reciprocal remote identities are checked separately."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
m=json.loads((ROOT/'foundation/langscale001/SOURCE_MAP.json').read_text())
for path,expected in m['pnds_files_sha256'].items():
 assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
assert len(m['variants'])==4
assert m['optimizer_updates']==0 and m['global_ieee_equivalence']=='OPEN'
assert 'sorry' not in (ROOT/'PNDS/LANGSCALE001.lean').read_text()
for v in m['variants']:
 assert v['nonlexical_state_floats']==8*(v['c']+4*v['k']+2*v['v']*v['k'])
print('PASS_LOCAL_CONTENT_PINS_NOT_A_LEAN_BUILD_OR_TRAINING_QUALIFICATION')
