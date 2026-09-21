import pathlib
import sys
import unittest

import numpy as np


SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from experiments.adaptive_reward import PPACorrelationWeightLearner


class PPACorrelationWeightLearnerTest(unittest.TestCase):
    def make_learner(self, **overrides):
        params = {
            "mask_names": ["reg", "hier", "df", "wire", "port"],
            "initial_weights": [0.45, 0.2, 0.15, 0.15, 0.05],
            "learnable_masks": ["reg", "hier", "df", "wire"],
            "min_observations": 3,
            "update_rate": 1.0,
            "temperature": 0.2,
            "prior_strength": 0.0,
            "min_weight": 0.02,
            "tns_weight": 0.0,
            "wns_weight": 0.0,
        }
        params.update(overrides)
        return PPACorrelationWeightLearner(**params)

    def test_waits_for_minimum_number_of_ppa_observations(self):
        learner = self.make_learner()
        initial = learner.weights.copy()
        result = learner.observe(
            {"reg": 1, "hier": 2, "df": 3, "wire": 4, "port": 5},
            gp_hpwl=10,
        )
        self.assertFalse(result.updated)
        np.testing.assert_allclose(learner.weights, initial)

    def test_increases_weight_of_proxy_correlated_with_bad_ppa(self):
        learner = self.make_learner()
        observations = [
            ({"reg": 1, "hier": 3, "df": 1, "wire": 2, "port": 5}, 10),
            ({"reg": 2, "hier": 2, "df": 1, "wire": 3, "port": 5}, 20),
            ({"reg": 3, "hier": 1, "df": 1, "wire": 1, "port": 5}, 30),
        ]
        for costs, hpwl in observations:
            result = learner.observe(costs, gp_hpwl=hpwl)

        self.assertTrue(result.updated)
        self.assertGreater(result.weights["reg"], result.weights["hier"])
        self.assertAlmostEqual(result.weights["port"], 0.05)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0)

    def test_rejects_unknown_learnable_mask(self):
        with self.assertRaisesRegex(ValueError, "unknown learnable masks"):
            self.make_learner(learnable_masks=["timing"])

    def test_tied_rank_values_do_not_get_arbitrary_order(self):
        ranks = PPACorrelationWeightLearner._rank_with_ties([2, 1, 2, 3])
        np.testing.assert_allclose(ranks, [1.5, 0.0, 1.5, 3.0])


if __name__ == "__main__":
    unittest.main()
