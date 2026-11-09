"""Contract tests for estimating_hank_nn.particle_filter."""

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimating_hank_nn.networks import NormalizeLayer, ScaleLayer
from estimating_hank_nn.particle_filter import ParticleFilter
from estimating_hank_nn.structures import Parameters


class ConstModel:
    """Minimal PF-contract model whose observable is identically zero."""

    def __init__(self):
        self.par = Parameters({"x": 1.0})
        self.P = None

    def to(self, device):
        return self

    def prepare(self, par_draw, batch, device="cpu"):
        self.P = batch

    def forecast(self, shocks=None):
        return {"y": torch.zeros(self.P, 1)}

    def resample(self, idx):
        pass


class NoisyModel(ConstModel):
    """Observable driven by the torch RNG stream, for CRN tests."""

    def forecast(self, shocks=None):
        return {"y": torch.randn(self.P, 1) * 0.1}


class TwoObsModel(ConstModel):
    """Two constant observables, optionally returned in reversed dict order."""

    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def forecast(self, shocks=None):
        out = {"a": torch.zeros(self.P, 1), "b": torch.ones(self.P, 1)}
        return dict(reversed(out.items())) if self.reverse else out


def make_pf(model, sim=5, r=0.04):
    data = {"y": torch.zeros(1, sim)}
    return ParticleFilter(model, data, torch.eye(1) * r, seed=0)


class TestLogProb(unittest.TestCase):
    def test_matches_torch_multivariate_normal(self):
        torch.manual_seed(0)
        R = torch.diag(torch.tensor([0.04, 0.09, 0.01]))
        L = torch.linalg.cholesky(R)
        error = torch.randn(3, 50)
        ours = ParticleFilter.log_prob(error, L)
        reference = torch.distributions.MultivariateNormal(torch.zeros(3), R).log_prob(error.T)
        self.assertTrue(torch.allclose(ours, reference, atol=1e-5))


class TestKitagawa(unittest.TestCase):
    def test_concentrated_weights_select_one_index(self):
        w = torch.tensor([0.0, 0.0, 1.0, 0.0])
        idx = ParticleFilter.kitagawa(w, 4, aux=0.4532)
        self.assertTrue((idx == 2).all())

    def test_uniform_weights_deterministic_offset(self):
        w = torch.full((4,), 0.25)
        idx = ParticleFilter.kitagawa(w, 4, aux=0.0)
        self.assertTrue(torch.equal(idx, torch.arange(4)))


class TestFilter(unittest.TestCase):
    def test_constant_state_analytic_likelihood(self):
        sim, r = 5, 0.04
        pf = make_pf(ConstModel(), sim=sim, r=r)
        loglik, _ = pf.filter(P=32, burn=2, sim=sim)
        per_step = -0.5 * (math.log(2 * math.pi) + math.log(r))
        self.assertAlmostEqual(float(loglik), sim * per_step, places=4)

    def test_likelihood_independent_of_forecast_key_order(self):
        data = {"a": torch.zeros(1, 3), "b": torch.zeros(1, 3)}
        R = torch.diag(torch.tensor([0.04, 1.0]))
        ll_fwd, _ = ParticleFilter(TwoObsModel(reverse=False), data, R, seed=0).filter(P=8, burn=1, sim=3)
        ll_rev, _ = ParticleFilter(TwoObsModel(reverse=True), data, R, seed=0).filter(P=8, burn=1, sim=3)
        self.assertEqual(float(ll_fwd), float(ll_rev))

    def test_crn_seed_reproduces_likelihood(self):
        pf = make_pf(NoisyModel())
        ll1, _ = pf.filter(P=16, burn=1, sim=5, crn_seed=7)
        ll2, _ = pf.filter(P=16, burn=1, sim=5, crn_seed=7)
        ll3, _ = pf.filter(P=16, burn=1, sim=5, crn_seed=8)
        self.assertEqual(float(ll1), float(ll2))
        self.assertNotEqual(float(ll1), float(ll3))


class TestSurrogate(unittest.TestCase):
    def make_pf_with_dataset(self, ll):
        pf = ParticleFilter(types.SimpleNamespace(to=lambda d: None), {"y": torch.zeros(1, 2)},
                            torch.eye(1), seed=0)
        pf.dataset = {"par": torch.randn(len(ll), 3), "log_likelihood": torch.tensor(ll)}
        return pf

    def test_nonfinite_targets_cannot_poison_scale_bounds(self):
        pf = self.make_pf_with_dataset([-2500.0, float("nan"), float("-inf"), -2400.0])
        net = pf.make_surrogate_network(N_inputs=3, hidden=8, layers=2, normalize=False, seed=0)
        scale = net[-1]
        self.assertIsInstance(scale, ScaleLayer)
        self.assertTrue(torch.isfinite(scale.lower_bound).all())
        self.assertTrue(torch.isfinite(net(torch.zeros(4, 3))).all())

    def test_checkpoint_round_trip_with_mixed_activations(self):
        pf = self.make_pf_with_dataset([-2.0, -1.0, -3.0, -2.5])
        pf.make_surrogate_network(N_inputs=3, hidden=8, layers=3, activation=["Tanh", "ReLU", "CELU"],
                                  normalize=False, seed=0)
        x = torch.randn(16, 3)
        y = pf.network(x)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(tmp, name="pf_test")
            pf2 = ParticleFilter(types.SimpleNamespace(to=lambda d: None), {"y": torch.zeros(1, 2)},
                                 torch.eye(1), seed=0)
            pf2.load(f"{tmp}/pf_test.pt")
        acts = [type(m).__name__ for m in pf2.network
                if type(m).__name__ not in ("Linear", "NormalizeLayer", "ScaleLayer")]
        self.assertEqual(acts, ["Tanh", "ReLU", "CELU"])
        self.assertTrue(torch.equal(y, pf2.network(x)))

    def test_checkpoint_loads_scaled_network_without_training_dataset(self):
        pf = self.make_pf_with_dataset([-2500.0, -2400.0, -2300.0])
        pf.make_surrogate_network(N_inputs=3, hidden=8, layers=2, normalize=False, seed=0)
        x = torch.randn(4, 3)
        expected = pf.network(x).detach()
        pf.dataset = None
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(tmp)
            restored = ParticleFilter(types.SimpleNamespace(to=lambda d: None),
                                      {"y": torch.zeros(1, 2)}, torch.eye(1))
            restored.load(Path(tmp) / "pf.pt")
        self.assertIsNone(restored.dataset)
        self.assertIsInstance(restored.network[-1], ScaleLayer)
        self.assertFalse(restored.network.training)
        torch.testing.assert_close(restored.network(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
