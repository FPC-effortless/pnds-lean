"""Constructed PNDS operator checks; no trained source/real-world performance."""
from pathlib import Path
import importlib.util
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "foundation/v1/langdecode063_shared_upper.py"
spec = importlib.util.spec_from_file_location("langdecode063_shared_upper", PATH)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class SharedUpperReference(unittest.TestCase):
    def test_one_projection_exact_same_value_and_order(self):
        rng = random.Random(63063)
        for i in range(12):
            z = tuple(rng.uniform(-2,2) for _ in range(280))
            h = tuple(rng.uniform(-2,2) for _ in range(48))
            calls = [0]
            def project(vec):
                calls[0] += 1
                return tuple(.125 * vec[j] for j in range(48))
            ops=mod.UpperOps(
                project=project,
                gate_logits=lambda u,p: tuple(.1*a + .05*b for a,b in zip(u,p)),
                output=lambda x: tuple(.02*x[j%48] for j in range(280)),
                forget_logits=lambda u: tuple(.1*x for x in u),
                offer_logits=lambda u: tuple(-.3*x for x in u),
                residual_gain=-2.0,
            )
            baseline=mod.unfused(z,h,ops)
            self.assertEqual(calls[0],2)
            calls[0]=0
            optimized=mod.fused(z,h,ops)
            self.assertEqual(calls[0],1)
            self.assertEqual(baseline,optimized)
            self.assertEqual(len(optimized[0]),280)
            self.assertEqual(len(optimized[1]),48)

    def test_impure_projection_is_not_certified(self):
        z=(0.0,)*280
        h=(0.0,)*48
        calls=[0]
        def bad_project(_):
            calls[0]+=1
            return (float(calls[0]),)*48
        ops=mod.UpperOps(
            project=bad_project,
            gate_logits=lambda u,p: (0.0,)*48,
            output=lambda u: (0.0,)*280,
            forget_logits=lambda u: (0.0,)*48,
            offer_logits=lambda u: (0.0,)*48,
            residual_gain=-2.0,
        )
        with self.assertRaises(ValueError):
            mod.unfused(z,h,ops)
        self.assertEqual(calls[0],2)

    def test_reject_mismatched_shape(self):
        ops=mod.UpperOps(lambda z:(0.0,)*48,lambda u,p:(0.0,)*48,
                         lambda u:(0.0,)*280,lambda u:(0.0,)*48,
                         lambda u:(0.0,)*48,-2.0)
        with self.assertRaises(ValueError):
            mod.fused((0.0,)*279,(0.0,)*48,ops)
