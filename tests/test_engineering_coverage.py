import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_engineering_coverage import verify


class EngineeringCoverageTests(unittest.TestCase):
    def test_original_variants_not_promoted(self):
        v=verify()
        self.assertEqual(v["covered_historical_versions"],17)
        self.assertEqual(v["full_model_certificates"],0)

    def test_invented_whole_model_proof_fails(self):
        base=ROOT/"foundation/v1"
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for p in ("registry.json","ENGINEERING_COVERAGE_V1_1.json"):
                (root/p).write_bytes((base/p).read_bytes())
            p=root/"ENGINEERING_COVERAGE_V1_1.json"
            data=json.loads(p.read_text())
            data["versions"][0]["full_graph_lean_semantics"]="PASS"
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify(root)

    def test_original_negative_evidence_cannot_be_erased(self):
        base=ROOT/"foundation/v1"
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for p in ("registry.json","ENGINEERING_COVERAGE_V1_1.json"):
                (root/p).write_bytes((base/p).read_bytes())
            p=root/"ENGINEERING_COVERAGE_V1_1.json"
            data=json.loads(p.read_text())
            key=next(v for v in data["versions"] if v["id"]=="ESWAM-ESWAM014")
            key["evidence_status"]="pass"
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify(root)
