"""Contract tests for estimating_hank_nn.rwmh."""

import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimating_hank_nn.rwmh import rwmh
from estimating_hank_nn.structures import Priors

_silent = lambda *args, **kwargs: None


class NaNNet(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(float("nan"))


class ZeroNet(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(0.0)


class NaNAfterStartNet(torch.nn.Module):
    """Finite at the start vector, NaN for every proposal."""

    def __init__(self, start):
        super().__init__()
        self.start = start

    def forward(self, x):
        return torch.tensor(0.0) if torch.equal(x, self.start) else torch.tensor(float("nan"))


class SpyNet(torch.nn.Module):
    """Records every parameter vector it is evaluated at."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return torch.tensor(0.0)


class TestRWMH(unittest.TestCase):
    def setUp(self):
        self.priors = Priors({"a": 0.5, "b": 1.0}, {"a": torch.distributions.Normal(0.5, 0.1)})
        self.start = torch.tensor([0.5, 1.0])
        self.cov = torch.eye(1) * 0.01

    def test_nonfinite_initial_posterior_asserts(self):
        pf = types.SimpleNamespace(network=NaNNet())
        with self.assertRaises(AssertionError):
            rwmh(pf, self.priors, steps=30, param_start=self.start,
                 cov_estimable=self.cov, seed=0, print_fn=_silent)

    def test_nan_surrogate_rejects_every_proposal(self):
        pf = types.SimpleNamespace(network=NaNAfterStartNet(self.start))
        draws, plls = rwmh(pf, self.priors, steps=30, param_start=self.start,
                           cov_estimable=self.cov, seed=0, print_fn=_silent)
        self.assertEqual(tuple(draws.shape), (30, 1))
        self.assertTrue(torch.equal(draws, torch.full((30, 1), 0.5)))

    def test_fixed_slots_stay_fixed_at_network_input(self):
        spy = SpyNet()
        pf = types.SimpleNamespace(network=spy)
        rwmh(pf, self.priors, steps=50, param_start=self.start,
             cov_estimable=self.cov, seed=0, print_fn=_silent)
        xs = torch.stack(spy.inputs)
        self.assertTrue((xs[:, 1] == 1.0).all())
        self.assertGreater(float(xs[:, 0].std()), 0.0)

    def test_chain_moves_and_only_estimable_slots_are_proposed(self):
        pf = types.SimpleNamespace(network=ZeroNet())
        draws, plls = rwmh(pf, self.priors, steps=200, param_start=self.start,
                           cov_estimable=self.cov, seed=0, print_fn=_silent)
        self.assertEqual(tuple(draws.shape), (200, 1))
        self.assertGreater(float(draws.std()), 0.0)
        self.assertTrue(torch.isfinite(plls).all())


if __name__ == "__main__":
    unittest.main()
