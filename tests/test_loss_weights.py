"""Contract tests for estimating_hank_nn.loss_weights."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimating_hank_nn.loss_weights import ConstantWeights, ReLoBRaLo


class TestConstantWeights(unittest.TestCase):
    def test_initial_and_passthrough(self):
        scheme = ConstantWeights(weights=[1.0, 10.0])
        weights = scheme.initial(2)
        self.assertEqual(weights, [1.0, 10.0])
        out, state = scheme.update(weights, [0.5, 0.1], 3, 100, None)
        self.assertEqual(out, weights)
        self.assertIsNone(state)

    def test_default_is_ones(self):
        self.assertEqual(ConstantWeights().initial(3), [1.0, 1.0, 1.0])


class TestReLoBRaLo(unittest.TestCase):
    def test_invariants_over_updates(self):
        scheme = ReLoBRaLo()
        n = 3
        weights, state = scheme.initial(n), None
        losses = [1.0, 0.5, 2.0]
        for i in range(20):
            losses = [l * f for l, f in zip(losses, (0.9, 0.99, 0.5))]
            weights, state = scheme.update(weights, losses, i, warmup_iter=100, state=state)
            self.assertEqual(len(weights), n)
            self.assertTrue(all(w > 0 for w in weights))
            self.assertAlmostEqual(sum(weights), n, places=6)

    def test_warmup_reanchors_and_resets(self):
        scheme = ReLoBRaLo()
        weights, state = scheme.initial(2), None
        for i in range(5):
            weights, state = scheme.update(weights, [1.0, 3.0], i, warmup_iter=5, state=state)
        weights, state = scheme.update(weights, [10.0, 0.1], 5, warmup_iter=5, state=state)
        self.assertEqual(weights, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
