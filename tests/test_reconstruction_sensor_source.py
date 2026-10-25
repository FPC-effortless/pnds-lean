"""Source-pinned FP32/FP64 numerical refinement for ME102."""
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

from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld

spec = importlib.util.spec_from_file_location(
    "pnds_reconstruction_sensor_reference",
    ROOT / "foundation/v1/reconstruction_sensor_reference.py",
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReconstructionSensorSourceTest(unittest.TestCase):
    def test_source_pin_when_requested(self):
        expected = os.environ.get("TAC_SOURCE_COMMIT")
        if not expected:
            self.skipTest("TAC_SOURCE_COMMIT is unset; run in the pinned workflow")
        actual = subprocess.check_output(
            ["git", "-C", str(TAC), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(actual, expected)

    def test_full_forward_state_and_all_used_gradients(self):
        torch.set_num_threads(1)
        for dtype, tol in [(torch.float32, 3e-5), (torch.float64, 2e-9)]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(102)
                model = ReconstructionSensorWorld(
                    floor_rate=torch.tensor([0.01, 0.03, 0.05, 0.07, 0.09, 0.11])
                ).to(dtype)
                self.assertEqual(model.current_head.in_features, 102)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 98495)
                with torch.no_grad():
                    model.readout[2].weight[:6].normal_(0, 0.01)
                    model.current_head.weight.normal_(0, 0.01)
                values = torch.randn(2, 23, 13, dtype=dtype)
                observed = torch.rand(2, 23, 13) > 0.3
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

                targets = torch.randn(2, 6, dtype=dtype)
                target_mask = torch.rand(2, 6) > 0.4
                target_mask[0, 0] = True
                source_loss = model.reconstruction_loss(trace, targets, target_mask)
                ref_loss = reference.reconstruction_loss(got[6], targets, target_mask)
                torch.testing.assert_close(source_loss, ref_loss, rtol=tol, atol=tol * 0.1)

                source_objective = (
                    pred.log_location.square().sum()
                    + pred.log_scale.square().sum()
                    + 0.03 * source_loss
                )
                reference_objective = (
                    got[0].square().sum() + got[1].square().sum() + 0.03 * ref_loss
                )
                parameters = tuple(model.parameters())
                source_grads = torch.autograd.grad(
                    source_objective, parameters, allow_unused=True
                )
                reference_grads = torch.autograd.grad(
                    reference_objective, parameters, allow_unused=True
                )
                for source_grad, reference_grad in zip(source_grads, reference_grads):
                    self.assertEqual(source_grad is None, reference_grad is None)
                    if source_grad is not None:
                        torch.testing.assert_close(
                            source_grad, reference_grad,
                            rtol=tol * 10, atol=tol,
                        )

                # Observed current channels leave inherited mean/variance alone.
                torch.testing.assert_close(
                    trace["current_anchor"][0, 0], values[0, -1, 0]
                )
                torch.testing.assert_close(
                    trace["total_variance"][0, 0], trace["future_variance"][0, 0]
                )
                self.assertGreater(trace["floor_variance"][1, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
