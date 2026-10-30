"""No-fit Python mathematical tests. No model scoring or source qualification."""
from archfuse089_channel import channel_delta, retained
import random
import unittest


class ChannelDeltaReference(unittest.TestCase):
    def test_scalar_equivalence(self):
        rng = random.Random(89)
        for _ in range(50):
            m = [[rng.uniform(-1, 1) for j in range(8)] for i in range(8)]
            k = [rng.uniform(-1, 1) for j in range(8)]
            v = [rng.uniform(-1, 1) for i in range(8)]
            alpha = [.997] * 8
            beta = .3
            target = channel_delta(m, k, v, alpha, beta)
            base = [[.997 * z for z in row] for row in m]
            expected = [[base[i][j] + beta * (v[i] - sum(base[i][l] * k[l]
                        for l in range(8))) * k[j] for j in range(8)] for i in range(8)]
            for i in range(8):
                for j in range(8):
                    self.assertAlmostEqual(target[i][j], expected[i][j], places=13)

    def test_beta_zero_and_channel_independence(self):
        m = [[1, 2], [3, 4]]
        self.assertEqual(channel_delta(m, [1, 0], [5, 6], [1, 0], 0),
                         retained(m, [1, 0]))
        self.assertEqual(retained(m, [1, 0]), [[1, 0], [3, 0]])

    def test_invalid_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            channel_delta([[1, 2]], [1], [1], [1], .3)
        with self.assertRaises(ValueError):
            channel_delta([[1]], [1], [1], [1.2], .3)


if __name__ == "__main__":
    unittest.main()
