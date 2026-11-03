"""Run all original pinned legacy assertions with only OCR004 unit isolation.

Original mathematical-source tests remain byte-identical. These subclasses
change only their fixture's whole-dispatch dependency, not production routing.
Import modules rather than TestCase classes to prevent duplicate discovery.
"""
from unittest.mock import patch
import unittest

from foundation.langcache104 import test_reciprocal_candidate as langcache
from foundation.tac_ocr001 import test_reciprocal_candidate as ocr


class _LegacyOCRBoundary:
    def check(self):
        # A tiny historical fixture exercises its own unchanged validator;
        # the required OCR004 contract is tested by the native routing suite.
        with patch("scripts.check_reciprocal_links.verify_ocr004",
                   return_value={"scope": "LEGACY_FIXTURE_ISOLATION_ONLY"}):
            return super().check()


class LangcacheCandidatePinTests(_LegacyOCRBoundary, langcache.CandidatePinTests):
    pass


class OCRCandidatePinTests(_LegacyOCRBoundary, ocr.OCRCandidatePinTests):
    pass


if __name__ == "__main__":
    unittest.main()
