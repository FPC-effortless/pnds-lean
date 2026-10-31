#!/usr/bin/env python3
"""Verify the prospective objective snapshot; no model/data execution or fitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def check():
    link=json.loads((ROOT/'foundation/v1/langcontrol098-source-link.json').read_text())
    row=json.loads((ROOT/'foundation/v1/langcontrol098-registry-append.json').read_text())
    registry=json.loads((ROOT/'foundation/v1/variant-math-links.json').read_text())
    matches=[r for r in registry['prospective_training_candidates'] if r.get('id')=='LANGCONTROL098']
    if matches != [row]: raise ValueError('missing, duplicate or different prospective registry row')
    source=link['tac']['source_commit']
    if not re.fullmatch('[0-9a-f]{40}',source) or source != row['tac']['commit']:
        raise ValueError('exact reciprocal source commit required')
    if link['pnds']['proof_commit'] != row['pnds']['commit']:
        raise ValueError('mathematical source pins differ')
    if (link['capability_claim'] or link['promotion_permitted']
            or row['lean_full_graph_proved'] or row['python_source_equivalence_verified']
            or row['real_world_capability_inferred'] or row['inference_graph_changed']):
        raise ValueError('unsupported proof/capability or changed inference graph')
    hashes=link['tac']['source_sha256']
    if set(hashes)!=set(link['tac']['source_paths']) or len(hashes)!=10:
        raise ValueError('complete exact source bundle required')
    for path,expected in hashes.items():
        if (not path.startswith('research/mathematical-engineering/')
                or '..' in Path(path).parts or not path.endswith('.py')
                or not re.fullmatch('[0-9a-f]{64}',expected)):
            raise ValueError('invalid source path/digest')
        url='https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/'+source+'/'+path
        with urllib.request.urlopen(url,timeout=30) as response:
            original=response.read()
        if hashlib.sha256(original).hexdigest()!=expected:
            raise ValueError('original source digest differs: '+path)
    return {'status':'PASS_EXACT_PROSPECTIVE_SOURCE_DIGESTS','study':'LANGCONTROL098',
            'tac_source_commit':source,'pnds_formulation_commit':row['pnds']['commit'],
            'source_files_verified':len(hashes),'model_forward_calls':0,'optimizer_updates':0,
            'canonical_main_master_reciprocal_parity':'OPEN_UNMERGED',
            'full_source_equivalence_proved':False,'learned_capability_inferred':False}


if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
