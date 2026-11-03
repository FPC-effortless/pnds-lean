"""Constructed mutation fixtures for fail-closed reciprocal provenance checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_reciprocal_links import (LEGACY_PATH, PEER_REGISTRY, RAW, REMOTE,
                                            ReciprocalError, verify)


class CandidatePinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source, self.metadata = "a"*40, "b"*40
        self.row = {"id": "LANGCACHE104-DEEP-ASSOCIATIVE-CARRIED-CACHE",
                    "tac_source_commit": self.source,
                    "source_map_path": "foundation/langcache104/SOURCE_MAP.json",
                    "tac_source_map_path": "research/langcache104/MATH_SOURCE_MAP.json",
                    "lean_path": "PNDS/LANGCACHE104.lean",
                    "python_path": "foundation/langcache104/math_reference.py", "status": "OPEN"}
        self.legacy = {"schema": "legacy-fixture", "variants": [{"id": "preserved-old"}],
                       "untouched_legacy_field": {"value": 104}}
        self.local = dict(self.legacy, prospective_candidates=[self.row])
        self.map = {"variant_id": self.row["id"], "tac_commit": self.source,
                    "lean_module": self.row["lean_path"], "python_reference": self.row["python_path"],
                    "tac_source_paths": ["research/langcache104/model.py"],
                    "tac_source_sha256": {"research/langcache104/model.py": hashlib.sha256(b"source bytes").hexdigest()}}
        self.write(LEGACY_PATH, self.local)
        lean_bytes = self.write(self.row["lean_path"], "scoped Lean fixture")
        python_bytes = self.write(self.row["python_path"], "independent Python fixture")
        self.map["mathematical_files_sha256"] = {
            self.row["lean_path"]: hashlib.sha256(lean_bytes).hexdigest(),
            self.row["python_path"]: hashlib.sha256(python_bytes).hexdigest()}
        map_bytes = self.write(self.row["source_map_path"], self.map)
        self.pin_path = "foundation/langcache104/TAC_PIN.json"
        self.write(self.pin_path, {"schema": "pnds.langcache104.tac_pin.v1",
            "tac_metadata_commit": self.metadata, "tac_source_commit": self.source,
            "source_map_sha256": hashlib.sha256(map_bytes).hexdigest()})
        self.remote = {REMOTE: json.dumps(self.legacy).encode(),
            RAW+self.metadata+"/"+PEER_REGISTRY: json.dumps({"prospective_candidates": [self.row]}).encode(),
            RAW+self.metadata+"/"+self.row["tac_source_map_path"]: map_bytes,
            RAW+self.source+"/research/langcache104/model.py": b"source bytes"}

    def write(self, path, value):
        target = self.base/path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2)+"\n").encode() if isinstance(value, dict) else value.encode()
        target.write_bytes(raw)
        return raw

    def check(self):
        return verify(self.base, lambda url: self.remote[url])

    def test_matching_pinned_candidate_and_original_source(self):
        result = self.check()
        self.assertTrue(result["legacy_manifest_equal"])
        self.assertEqual(result["prospective_candidates"][0]["original_source_files_verified"], 1)
        self.assertNotIn("reciprocal_manifest_equal", result)

    def test_missing_pin_blocks(self):
        (self.base/self.pin_path).unlink()
        with self.assertRaisesRegex(ReciprocalError, "missing immutable"):
            self.check()

    def test_unknown_legacy_field_is_not_ignored(self):
        changed = deepcopy(self.legacy)
        changed["untouched_legacy_field"]["value"] = 105
        self.remote[REMOTE] = json.dumps(changed).encode()
        with self.assertRaisesRegex(ReciprocalError, "Legacy"):
            self.check()

    def test_candidate_peer_mismatch_blocks(self):
        changed = dict(self.row, status="PROVED")
        self.remote[RAW+self.metadata+"/"+PEER_REGISTRY] = json.dumps({"prospective_candidates": [changed]}).encode()
        with self.assertRaisesRegex(ReciprocalError, "candidate differs"):
            self.check()

    def test_source_map_byte_mismatch_blocks(self):
        self.remote[RAW+self.metadata+"/"+self.row["tac_source_map_path"]] += b" "
        with self.assertRaisesRegex(ReciprocalError, "source-map bytes"):
            self.check()

    def test_original_source_mutation_blocks(self):
        self.remote[RAW+self.source+"/research/langcache104/model.py"] = b"changed source bytes"
        with self.assertRaisesRegex(ReciprocalError, "source hash mismatch"):
            self.check()

    def test_mathematical_formulation_mutation_blocks(self):
        self.write(self.row["python_path"], "changed independent reference")
        with self.assertRaisesRegex(ReciprocalError, "mathematical formulation hash"):
            self.check()


if __name__ == "__main__":
    unittest.main()
