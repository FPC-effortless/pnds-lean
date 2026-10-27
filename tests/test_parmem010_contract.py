import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('parmem010_gate',ROOT/'scripts/validate_parmem010_math.py')
gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FormalContractGates(unittest.TestCase):
    def copy(self,base):
        for path in ['PNDS','foundation/v1','tests']:
            shutil.copytree(ROOT/path,base/path,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(ROOT/'AuditSourceAligned.lean',base/'AuditSourceAligned.lean')

    def test_record_is_explicitly_incomplete_and_source_pinned(self):
        result=gate.check()
        self.assertFalse(result['complete_formal_verification'])
        self.assertGreaterEqual(result['declared_theorems'],47)

    def test_changed_independent_equations_fail_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); self.copy(base)
            with (base/'foundation/v1/source_aligned_model_math.py').open('a') as f:
                f.write('\n# changed after receipt\n')
            with patch.object(gate,'ROOT',base),self.assertRaises(AssertionError):
                gate.check()

    def test_missing_specification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); self.copy(base)
            (base/'PNDS/SourceAlignedModel.lean').unlink()
            with patch.object(gate,'ROOT',base),self.assertRaises(AssertionError):
                gate.check()

    def test_proof_placeholder_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); self.copy(base)
            with (base/'PNDS/SourceAlignedVerification.lean').open('a') as f:
                f.write('\ntheorem fake : False := by sorry\n')
            with patch.object(gate,'ROOT',base),self.assertRaises(AssertionError):
                gate.check()


if __name__=='__main__': unittest.main()
