"""Constructed source-shape checks; not trained-source or IEEE proof."""
import importlib.util
from pathlib import Path
import unittest

FILE=Path(__file__).resolve().parents[1]/"foundation/v1/langdecode064_capped_position.py"
spec=importlib.util.spec_from_file_location("langdecode064_capped_position",FILE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CappedPositionReference(unittest.TestCase):
    def test_cached_equals_uncached(self):
        cached=module.build_cached()
        self.assertEqual(len(cached),7)
        for t in list(range(0,130))+[256,512,8192,65536,1_000_000]:
            self.assertEqual(module.cached_capped(t,cached),module.source_capped(t))
        self.assertEqual(cached,module.source_capped(127))

    def test_boundary_and_invalid_shapes(self):
        self.assertNotEqual(module.phi_position(126),module.phi_position(127))
        self.assertEqual(module.source_capped(128),module.source_capped(127))
        with self.assertRaises(ValueError):
            module.cached_capped(129,(1.0,))
        with self.assertRaises(ValueError):
            module.cached_capped(-1,module.build_cached())
        with self.assertRaises(ValueError):
            module.phi_position(1.0)

    def test_uncapped_prior_features_not_clipped(self):
        for t in (0,1,7,31,64,126,127):
            self.assertEqual(module.source_capped(t),module.phi_position(t))
