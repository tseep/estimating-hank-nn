"""Contract tests for estimating_hank_nn.structures."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimating_hank_nn.structures import Parameters, Priors, Ranges, Shocks, State


class TestElement(unittest.TestCase):
    def test_str_handles_multi_element_tensors(self):
        s = State({"b": torch.ones(4, 100, 1), "s": torch.zeros(4, 100, 1)})
        self.assertIn("tensor(4, 100, 1)", str(s))

    def test_str_scalar_value(self):
        p = Parameters({"beta": 0.99})
        self.assertIn("beta: 0.99", str(p))

    def test_cat_concatenates_last_dim(self):
        s = State({"a": torch.ones(2, 3, 1), "b": torch.zeros(2, 3, 1)})
        self.assertEqual(tuple(s.cat().shape), (2, 3, 2))

    def test_expand_dim(self):
        s = State({"b": torch.ones(1, 3, 1)})
        self.assertEqual(tuple(s.expand_dim(0, 5).b.shape), (5, 3, 1))

    def test_update_replaces_values(self):
        s = State({"b": torch.zeros(2)})
        s.update({"b": torch.ones(2)})
        self.assertTrue(torch.equal(s.b, torch.ones(2)))

    def test_eq(self):
        a = State({"b": torch.ones(2)})
        b = State({"b": torch.ones(2)})
        c = State({"b": torch.zeros(2)})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestShocks(unittest.TestCase):
    def test_antithetic_shape_and_mirror(self):
        sh = Shocks({"s": torch.distributions.Normal(0.0, 1.0)})
        draw = sh.sample((4, 2, 1), antithetic=True)["s"]
        self.assertEqual(tuple(draw.shape), (4, 2, 1))
        self.assertTrue(torch.equal(draw[:2], -draw[2:]))

    def test_antithetic_odd_count_asserts(self):
        sh = Shocks({"s": torch.distributions.Normal(0.0, 1.0)})
        with self.assertRaises(AssertionError):
            sh.sample((3, 2, 1), antithetic=True)


class TestPriors(unittest.TestCase):
    def test_log_prob_pairs_values_by_key_order_with_interleaved_fixed(self):
        priors = Priors({"a": 0.5, "b": 1.0, "c": 2.0},
                        {"a": torch.distributions.Normal(0.0, 1.0),
                         "c": torch.distributions.Normal(10.0, 1.0)})
        param = torch.tensor([0.3, 99.0, 9.5])   # slot b is Fixed: its value must be ignored
        expected = (torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(0.3))
                    + torch.distributions.Normal(10.0, 1.0).log_prob(torch.tensor(9.5)))
        self.assertTrue(torch.allclose(priors.log_prob(param), expected))


class TestRanges(unittest.TestCase):
    def setUp(self):
        self.ranges = Ranges({"a": 1.0, "b": 2.0}, {"b": torch.distributions.Uniform(0.0, 1.0)})

    def test_keys_estimable_excludes_identity(self):
        self.assertEqual(self.ranges.keys_estimable(), ["b"])

    def test_low_high_tensor(self):
        self.assertTrue(torch.equal(self.ranges.low_tensor(), torch.tensor([-1.0, 0.0])))
        self.assertTrue(torch.equal(self.ranges.high_tensor(), torch.tensor([1.0, 1.0])))

    def test_sample_keeps_identity_fixed(self):
        draw = self.ranges.sample((3,))
        self.assertTrue(torch.equal(draw.a, torch.ones(3)))
        self.assertTrue(((draw.b >= 0.0) & (draw.b <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
