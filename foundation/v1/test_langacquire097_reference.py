"""Constructed mathematical references, not real-model capability evidence."""
import unittest

import numpy as np

from langacquire097_reference import (
    affine_run, masked_cross_entropy, scalar_query_loss, scalar_sequence,
)


class ObjectiveReferenceTests(unittest.TestCase):
    def test_known_binary_cross_entropy_and_gradient(self):
        loss, gradient = masked_cross_entropy([[0., 0.]], [0], [True])
        self.assertAlmostEqual(loss, np.log(2.))
        np.testing.assert_array_equal(gradient, [[-.5, .5]])

    def test_inactive_direct_losses_are_irrelevant_not_removed_inputs(self):
        logits = np.array([[100., -100.], [.4, -.2], [2., 1.]])
        first, gradient = masked_cross_entropy(logits, [0, 1, 0], [False, True, False])
        logits[[0, 2]] = [[-9000., 9000.], [-4., 5.]]
        changed, _ = masked_cross_entropy(logits, [1, 1, 1], [False, True, False])
        self.assertEqual(first, changed)
        np.testing.assert_array_equal(gradient[[0, 2]], np.zeros((2, 2)))

    def test_masked_logit_derivative_matches_finite_differences(self):
        rng = np.random.default_rng(97001)
        logits = rng.normal(size=(5, 4))
        targets = np.array([0, 3, 2, 1, 0])
        mask = np.array([False, False, True, True, True])
        _, gradient = masked_cross_entropy(logits, targets, mask)
        eps = 1e-6
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += eps
            minus[index] -= eps
            finite = (masked_cross_entropy(plus, targets, mask)[0]
                      - masked_cross_entropy(minus, targets, mask)[0]) / (2 * eps)
            self.assertAlmostEqual(gradient[index], finite, delta=3e-9)

    def test_serial_acquisition_delay_query_chunk_law(self):
        rng = np.random.default_rng(97002)
        matrix, input_map = rng.normal(size=(3, 3)) / 4, rng.normal(size=(3, 2))
        initial, observed = rng.normal(size=3), rng.normal(size=(12, 2))
        complete = affine_run(matrix, input_map, initial, observed)
        for split in (0, 1, 7, 12):
            acquired = affine_run(matrix, input_map, initial, observed[:split])
            streamed = affine_run(matrix, input_map, acquired, observed[split:])
            np.testing.assert_array_equal(streamed, complete)

    def test_acquisition_parameter_path_reaches_query_only_loss(self):
        theta, acquisition, query = .7, [.3, -.2, .8, .6], [-.4, .2]
        target = [0, 1]
        _, derivative, receipt = scalar_query_loss(theta, acquisition, query, target)
        eps = 1e-6
        finite = (scalar_query_loss(theta + eps, acquisition, query, target)[0]
                  - scalar_query_loss(theta - eps, acquisition, query, target)[0]) / (2 * eps)
        self.assertAlmostEqual(derivative, finite, delta=3e-9)
        self.assertNotEqual(receipt["acquired_parameter_derivative"], 0.)
        # Deliberately detached negative control: the query state is numerically
        # identical but its gradient omits the acquisition path.
        q, detached_dq = scalar_sequence(theta, query, receipt["acquired_state"], 0.)
        _, gradient = masked_cross_entropy(np.stack((q, -q), 1), target, [True, True])
        detached = float(np.sum((gradient[:, 0] - gradient[:, 1]) * detached_dq))
        self.assertGreater(abs(derivative - detached), .01)

    def test_all_active_loss_matches_unmasked_average(self):
        logits = np.array([[1., 2., 3.], [.3, -.2, .1]])
        loss, _ = masked_cross_entropy(logits, [2, 0], [True, True])
        expected = np.mean(np.log(np.exp(logits).sum(1)) - logits[[0, 1], [2, 0]])
        self.assertAlmostEqual(loss, expected)

    def test_empty_non_boolean_and_illegal_target_masks_rejected(self):
        for targets, mask in (([0, 1], [False, False]), ([0, 1], [1, 1]),
                              ([0, 3], [True, True]), ([0., 1.], [True, True])):
            with self.subTest(targets=targets, mask=mask), self.assertRaises(ValueError):
                masked_cross_entropy(np.zeros((2, 2)), targets, mask)

    def test_unstable_or_malformed_affine_inputs_rejected(self):
        with self.assertRaises(ValueError):
            affine_run(np.eye(2), np.ones((2, 1)), np.ones(3), np.ones((4, 1)))
        with self.assertRaises(ValueError):
            masked_cross_entropy([[np.nan, 0.]], [0], [True])


if __name__ == "__main__":
    unittest.main()
