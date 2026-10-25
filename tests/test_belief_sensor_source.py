"""Source-pinned FP32/FP64 numerical refinement for ME101."""
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TAC = Path(os.environ.get("TAC_SOURCE_ROOT", "/tmp/tac-evidence")).resolve()
sys.path[:0] = [str(TAC), str(TAC / "src"), str(ROOT / "foundation/v1")]

from tac_osm.variants.belief_sensor_world import BeliefSensorWorld

spec = importlib.util.spec_from_file_location(
    "pnds_belief_sensor_reference", ROOT / "foundation/v1/belief_sensor_reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class BeliefSensorSourceTest(unittest.TestCase):
    def test_source_pin_when_requested(self):
        expected = os.environ.get("TAC_SOURCE_COMMIT")
        if not expected:
            self.skipTest("TAC_SOURCE_COMMIT is unset; run in the pinned workflow")
        actual = subprocess.check_output(
            ["git", "-C", str(TAC), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(actual, expected)

    def test_full_forward_state_and_all_gradients(self):
        torch.set_num_threads(1)
        for dtype, tol in [(torch.float32, 2e-5), (torch.float64, 1e-9)]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(101)
                model = BeliefSensorWorld().to(dtype)
                with torch.no_grad():
                    model.readout[2].weight[:6].normal_(0, 0.01)
                values = torch.randn(2, 23, 13, dtype=dtype)
                observed = torch.rand(2, 23, 13) > 0.3
                # Include both an observed and missing current channel, at ages
                # 0h and 256h, to exercise the two boundary cases.
                observed[0, -1, 0] = True
                observed[1, -1, 0] = False
                ages = torch.rand_like(values)
                ages[0, -1, 0] = 0
                ages[1, -1, 0] = 1
                clock = torch.randn(2, 23, 6, dtype=dtype).tanh()
                owners = torch.tensor([2, 9])
                hours = torch.arange(23)[None].expand(2, -1)

                pred, state, trace = model(
                    values, observed, ages, clock, owners, hours, return_trace=True
                )
                got = reference.serial(model, values, observed, ages, clock, owners)
                expected = (
                    pred.log_location,
                    pred.log_scale,
                    state.source_sum,
                    state.source_mass,
                    state.evidence_sum,
                    state.memory,
                )
                for actual, independent in zip(expected, got[:6]):
                    torch.testing.assert_close(
                        actual, independent, rtol=tol, atol=tol * 0.1
                    )
                for name, independent in got[6].items():
                    torch.testing.assert_close(
                        trace[name], independent, rtol=tol, atol=tol * 0.1
                    )
                self.assertEqual(got[6]["latent_state_variance"][0, 0].detach().item(), 0.0)
                self.assertGreater(got[6]["latent_state_variance"][1, 0].detach().item(), 0.0)

                model_grads = torch.autograd.grad(
                    pred.log_location.square().sum() + pred.log_scale.square().sum(),
                    tuple(model.parameters()),
                )
                ref_grads = torch.autograd.grad(
                    got[0].square().sum() + got[1].square().sum(),
                    tuple(model.parameters()),
                )
                for actual, independent in zip(model_grads, ref_grads):
                    self.assertIsNotNone(actual)
                    self.assertIsNotNone(independent)
                    torch.testing.assert_close(
                        actual, independent, rtol=tol * 10, atol=tol
                    )


if __name__ == "__main__":
    unittest.main()
