"""Native OCR dispatch stays required; legacy fixtures isolate only their unit."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_reciprocal_links as dispatch


class NativeOCRDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def no_remote(self, url):
        self.fail("unexpected remote access: " + url)

    def test_missing_pin_blocks_full_native_entry(self):
        with self.assertRaisesRegex(FileNotFoundError, "IMMUTABLE_PEER_PIN"):
            dispatch.verify(self.base, self.no_remote)

    def test_strict_contract_failure_cannot_be_skipped(self):
        with patch.object(dispatch, "verify_ocr004", side_effect=ValueError("strict OCR failure")) as strict:
            with self.assertRaisesRegex(ValueError, "strict OCR failure"):
                dispatch.verify(self.base, self.no_remote)
        strict.assert_called_once_with(self.base, self.no_remote)

    def test_actual_contract_rejects_moving_pin_before_legacy_or_network(self):
        native = self.base / "foundation/tac_ocr004"
        native.mkdir(parents=True)
        original = dispatch.ROOT / "foundation/tac_ocr004"
        shutil.copyfile(original / "verify_immutable_peer.py", native / "verify_immutable_peer.py")
        pin = json.loads((original / "IMMUTABLE_PEER_PIN.json").read_text())
        pin["metadata_commit"] = "main"
        (native / "IMMUTABLE_PEER_PIN.json").write_text(json.dumps(pin))
        with self.assertRaisesRegex(ValueError, "immutable commit"):
            dispatch.verify(self.base, self.no_remote)

    def test_scoped_success_cannot_waive_legacy_failure(self):
        path = self.base / dispatch.LEGACY_PATH
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"variants": [], "prospective_candidates": []}))
        with patch.object(dispatch, "verify_ocr004", return_value={"scope": "CONTROL_ONLY"}) as strict:
            with self.assertRaisesRegex(dispatch.ReciprocalError, "Legacy"):
                dispatch.verify(self.base, lambda url: b'{"variants": ["changed"]}')
        self.assertEqual(strict.call_count, 1)


if __name__ == "__main__":
    unittest.main()
