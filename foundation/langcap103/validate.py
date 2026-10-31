"""Validate the new source/equation map without promoting finite tests to proof."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate():
    data = json.loads((ROOT/'foundation/langcap103/SOURCE_MAP.json').read_text())
    links = json.loads((ROOT/'foundation/v1/variant-math-links.json').read_text())
    if data['schema'] != 'pnds.langcap103.graph-map.v1' or len(data['variants']) != 6:
        raise ValueError('six exact source configurations required')
    idx = links['prospective_graph_index']
    if idx['source_map_sha256'] != hashlib.sha256((ROOT/'foundation/langcap103/SOURCE_MAP.json').read_bytes()).hexdigest():
        raise ValueError('source-map digest mismatch')
    if set(idx['ids']) != {v['id'] for v in data['variants']} or len(set(idx['ids'])) != 6:
        raise ValueError('prospective registry identity mismatch')
    for name, expected in data['mathematical_files_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('equation hash mismatch: '+name)
    for v in data['variants']:
        if len(v['source_commit']) != 40 or any(v[k] is not False for k in
             ('lean_full_graph_proved','python_global_source_equivalence_proved','trained_capability_established')):
            raise ValueError('missing source pin or unsupported promotion')
    print(json.dumps({'historical_entries_preserved':len(links['variants']),
        'prospective_source_configurations':6,'whole_graph_proved':False,'training_authorized':False}))


if __name__ == '__main__': validate()
