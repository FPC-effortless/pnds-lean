"""Regression tests for PNDS v1 provenance and nonpromotion contract."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copytree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_foundation import FOUNDATION, FoundationError, validate


class FoundationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "v1"
        copytree(FOUNDATION, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def edit_registry(self, change):
        path = self.root / "registry.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        change(original)
        path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    def edit_ledger(self, change):
        path = self.root / "evidence-ledger.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        change(records)
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_current_versioned_research_validates(self):
        result = validate(self.root)
        self.assertGreaterEqual(result["indexed_versions"], 17)
        self.assertGreaterEqual(result["evidence_records"], 23)
        self.assertFalse(result["real_world_superiority_established_by_this_check"])

    def test_synthetic_promotion_cannot_be_declared_real(self):
        def corrupt(rows):
            first = next(r for r in rows if r["epistemic_scope"] == "synthetic_only")
            first["promotion_scope"] = "independent_real_capability"
        self.edit_ledger(corrupt)
        with self.assertRaises(FoundationError):
            validate(self.root)

    def test_failure_evidence_cannot_be_edited_into_success(self):
        def corrupt(rows):
            next(r for r in rows if r["event_id"] == "SYS-001")["outcome"] = "pass"
        self.edit_ledger(corrupt)
        with self.assertRaises(FoundationError):
            validate(self.root)

    def test_cannot_drop_research_lane(self):
        self.edit_registry(lambda r: r.__setitem__("variants", [
            v for v in r["variants"] if v["lane"] != "enterprise"
        ]))
        with self.assertRaises(FoundationError):
            validate(self.root)

    def test_every_source_version_must_have_commit_pin(self):
        def corrupt(r):
            r["variants"][0]["source"]["commit_sha"] = "main"
        self.edit_registry(corrupt)
        with self.assertRaises(FoundationError):
            validate(self.root)

    def test_training_budget_preserves_actual_parity_not_cpu_runtime(self):
        def corrupt(r):
            r["normative_policy"]["match_CPU_wall_clock"] = True
        self.edit_registry(corrupt)
        with self.assertRaises(FoundationError):
            validate(self.root)

    def test_historical_gaps_cannot_be_silently_erased(self):
        path = self.root / "archive-gaps.json"
        data = json.loads(path.read_text())
        data["completeness"] = "COMPLETE_ALL_HISTORY"
        path.write_text(json.dumps(data))
        with self.assertRaises(FoundationError):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
