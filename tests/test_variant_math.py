"""Numerical and mutation gates for PNDS dual-language variant references."""
from __future__ import annotations
import json
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "foundation/v1"))
from validate_variant_math import check, VariantMathError
from variant_math import (
    gated, affine_run, affine_compose, projected_gated_delta,
    quadratic_motion_gate, source_eor_mass, translated_cdf, bitemporal_eligible,
    window_readout, distribution_mass, edge_abs_error, causal_identified,
    independent_real_basis, KERNELS, evaluate_kernel,
)


class MathematicalKernels(unittest.TestCase):
    def test_gate_interval_error_identity(self):
        rng = random.Random(17)
        for _ in range(200):
            old, target = rng.uniform(-1, 1), rng.uniform(-1, 1)
            g = rng.random()
            value = gated(old, target, g)
            self.assertTrue(-1.000000000001 <= value <= 1.000000000001)
            self.assertAlmostEqual(target-value, (1-g)*(target-old), places=12)
        with self.assertRaises(ValueError):
            gated(0, 1, 1.01)

    def test_affine_compose_matches_serial_and_associative(self):
        rng = random.Random(42)
        for _ in range(150):
            a = (rng.random(), rng.uniform(-1, 1))
            b = (rng.random(), rng.uniform(-1, 1))
            c = (rng.random(), rng.uniform(-1, 1))
            s = rng.uniform(-1, 1)
            ab = affine_compose(b, a)
            self.assertAlmostEqual(affine_run(*ab, s),
                                   affine_run(*b, affine_run(*a, s)), places=12)
            left = affine_compose(c, affine_compose(b, a))
            right = affine_compose(affine_compose(c, b), a)
            for l, r in zip(left, right):
                self.assertAlmostEqual(l, r, places=12)

    def test_delta_projection(self):
        m = [[3.0, 0.0], [0.0, 4.0]]
        r = projected_gated_delta(m, [1, 0], [1, 2], .9, .5, 2)
        self.assertLessEqual(math.hypot(*(x for row in r for x in row)), 2 + 1e-12)
        with self.assertRaises(ValueError):
            projected_gated_delta(m, [1], [1, 2], .9, .5, 2)

    def test_eor_mass_and_median_assumptions(self):
        end, acts = source_eor_mass(.65, [.2, .3, .5])
        self.assertAlmostEqual(end + sum(acts), 1)
        self.assertAlmostEqual(translated_cdf(
            lambda x: 1/(1+math.exp(-x)), 0, 3, 3), .5)
        with self.assertRaises(ValueError):
            source_eor_mass(.5, [.1, .1])

    def test_temporal_actor_window_and_motion(self):
        self.assertTrue(bitemporal_eligible(1, 9, 2, 10, 5, 5))
        self.assertFalse(bitemporal_eligible(1, 9, 2, 10, 5, 1))
        self.assertAlmostEqual(window_readout([.25, .75], [2, 6]), 5)
        self.assertEqual(quadratic_motion_gate(0, .25), 0)
        self.assertLess(quadratic_motion_gate(.01, .25), 1)
        with self.assertRaises(ValueError):
            quadratic_motion_gate(1, 0)

    def test_remaining_interfaces_are_limited_not_causal_proofs(self):
        self.assertEqual(distribution_mass([.2, .8]), 1)
        self.assertEqual(edge_abs_error(1, .99), abs(1-.99))
        self.assertFalse(causal_identified(False))
        self.assertTrue(causal_identified(True))
        self.assertFalse(independent_real_basis("synthetic_diagnostic"))
        self.assertEqual(len(KERNELS), 12)
        self.assertEqual(evaluate_kernel("motion", speed=0.0, c=.25), 0)
        with self.assertRaises(ValueError):
            evaluate_kernel("missing")


class VariantProvenanceGates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "foundation/v1").mkdir(parents=True)
        (self.root / "PNDS").mkdir()
        for f in ["foundation/v1/variant-math-links.json",
                  "foundation/v1/registry.json",
                  "foundation/v1/variant_math.py",
                  "PNDS/TACOSMVariantContracts.lean"]:
            shutil.copyfile(ROOT / f, self.root / f)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, fn):
        p = self.root / "foundation/v1/variant-math-links.json"
        data = json.loads(p.read_text())
        fn(data)
        p.write_text(json.dumps(data))

    def test_17_original_links_and_open_gates(self):
        r = check(self.root)
        self.assertEqual(r["indexed_variants"], 17)
        self.assertEqual(r["complete_full_graph_proofs"], 0)

    def test_missing_link_fails(self):
        self.change(lambda d: d["variants"].pop())
        with self.assertRaises(VariantMathError):
            check(self.root)

    def test_commit_mutation_fails(self):
        self.change(lambda d: d["variants"][0]["tac"].update(commit="f"*40))
        with self.assertRaises(VariantMathError):
            check(self.root)

    def test_false_mathematical_completion_fails(self):
        self.change(lambda d: d["variants"][0].update(lean_full_graph_proved=True))
        with self.assertRaises(VariantMathError):
            check(self.root)

    def test_missing_python_reference_fails(self):
        (self.root / "foundation/v1/variant_math.py").unlink()
        with self.assertRaises(VariantMathError):
            check(self.root)

    def test_math_kernel_link_mutation_fails(self):
        self.change(lambda d: d["variants"][0].update(kernel="event"))
        with self.assertRaises(VariantMathError):
            check(self.root)


if __name__ == "__main__":
    unittest.main()
