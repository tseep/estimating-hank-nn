"""Contract tests for estimating_hank_nn.helpers."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimating_hank_nn.helpers import (TruncatedDistribution, add_measurement_error,
                                        ergodic_sigma, fisher_burmeister)


class TestTruncatedDistribution(unittest.TestCase):
    def setUp(self):
        self.t = TruncatedDistribution(torch.distributions.Normal(2.0, 0.1), 1.8, 2.1)

    def test_prob_integrates_to_one(self):
        x = torch.linspace(1.8, 2.1, 20_001)
        integral = torch.trapezoid(self.t.prob(x), x)
        self.assertAlmostEqual(float(integral), 1.0, places=4)

    def test_log_prob_outside_bounds_is_minus_inf(self):
        self.assertEqual(float(self.t.log_prob(torch.tensor(1.7))), float("-inf"))
        self.assertEqual(float(self.t.prob(torch.tensor(2.2))), 0.0)

    def test_sample_respects_bounds(self):
        draws = self.t.sample((1000,))
        self.assertTrue(((draws >= 1.8) & (draws <= 2.1)).all())

    def test_sample_finite_when_cdf_underflows_at_bound(self):
        # A bound many sigmas from the mean underflows cdf(low) to exactly 0 in float32, where
        # the naive inverse CDF returns -inf; the correct draw at that edge is the bound itself.
        t = TruncatedDistribution(torch.distributions.Normal(100.0, 5.0), 70.0, 120.0)
        self.assertEqual(float(t.base_dist.cdf(torch.tensor(70.0))), 0.0)
        torch.manual_seed(1234)
        draws = t.sample((500_000,))
        self.assertTrue(torch.isfinite(draws).all())
        self.assertTrue(((draws >= 70.0) & (draws <= 120.0)).all())


class TestHelpers(unittest.TestCase):
    def test_ergodic_sigma_matches_paper_calibration(self):
        self.assertAlmostEqual(ergodic_sigma(0.966, 0.1292710331048684), 0.5, places=10)

    def test_fisher_burmeister_complementarity_roots(self):
        zero = torch.tensor(0.0)
        self.assertAlmostEqual(float(fisher_burmeister(zero, torch.tensor(3.0))), 0.0)
        self.assertAlmostEqual(float(fisher_burmeister(torch.tensor(2.0), zero)), 0.0)
        self.assertGreater(float(fisher_burmeister(torch.tensor(1.0), torch.tensor(1.0))), 0.0)
        self.assertLess(float(fisher_burmeister(torch.tensor(-1.0), zero)), 0.0)

    def test_add_measurement_error_deterministic_with_given_shocks(self):
        data = {"a": torch.tensor([1.0, 2.0, 3.0, 4.0]), "b": torch.tensor([0.0, 1.0, 0.0, 1.0])}
        shocks = {"a": torch.ones(4), "b": torch.ones(4)}
        me1, R1 = add_measurement_error(data, error_share=0.1, me_shocks=shocks)
        me2, R2 = add_measurement_error(data, error_share=0.1, me_shocks=shocks)
        self.assertTrue(torch.equal(R1, R2))
        self.assertEqual(tuple(R1.shape), (2, 2))
        for key in data:
            self.assertTrue(torch.equal(me1[key], me2[key]))
        self.assertTrue(torch.allclose(me1["a"], data["a"] + torch.sqrt(R1[0, 0])))


if __name__ == "__main__":
    unittest.main()
