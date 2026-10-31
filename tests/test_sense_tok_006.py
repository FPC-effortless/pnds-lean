"""Partial finite operator/provenance checks, no trained or full-model claims."""
import json
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'foundation/v1'))
from sense_tok_006 import selected_readout, realized_regret, dense_byte_parameter_count


class SenseTok006Operators(unittest.TestCase):
    def test_joint_relabelling_and_fixed_key_wrong_value_differ(self):
        rng = random.Random(42)
        values = [[rng.uniform(-2, 2) for _ in range(32)] for _ in range(8)]
        permutation = list(reversed(range(8)))
        inverse = [permutation.index(i) for i in range(8)]
        relabelled = [values[i] for i in permutation]
        for q in range(8):
            self.assertEqual(selected_readout(relabelled, inverse[q]), selected_readout(values, q))
            self.assertNotEqual(selected_readout(values, q, permutation), selected_readout(values, q))
        with self.assertRaises(ValueError):
            selected_readout(values, 0, [0]*8)
        with self.assertRaises(ValueError):
            selected_readout([[math.nan]], 0)

    def test_regret_ties_and_parameter_count(self):
        self.assertEqual(realized_regret(2.2, 2.1, True), 2.2-2.1)
        self.assertEqual(realized_regret(2.2, 2.1, False), 0.)
        self.assertEqual(realized_regret(2.2, 2.2, True), 0.)
        self.assertEqual(dense_byte_parameter_count(), 100688)
        with self.assertRaises(ValueError):
            realized_regret(math.nan, 1., True)

    def test_prospective_source_record_is_explicitly_partial(self):
        data = json.loads((ROOT/'foundation/v1/variant-math-links.json').read_text())
        records = [r for r in data['prospective_studies'] if r['id'] == 'SENSE-TOK-006-PREPARATION']
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(len(record['tac']['commit']), 40)
        self.assertEqual(record['tac']['pull_request'], 675)
        self.assertFalse(record['lean_full_graph_proved'])
        self.assertFalse(record['python_source_equivalence_verified'])
        self.assertFalse(record['training_authorized'])
        self.assertTrue((ROOT/record['pnds']['lean_path']).is_file())
        self.assertTrue((ROOT/record['pnds']['python_path']).is_file())


if __name__ == '__main__':
    unittest.main()
