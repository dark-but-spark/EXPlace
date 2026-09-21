import pathlib
import sys
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PerformanceDefaultsTest(unittest.TestCase):
    def test_quick_config_disables_expensive_artifacts(self):
        with (ROOT / "config" / "compare_quick.yaml").open() as handle:
            config = yaml.safe_load(handle)
        self.assertFalse(config["evaluate_best_reward_with_dmp"])
        self.assertFalse(config["visualize_prototype"])
        self.assertEqual(config["visualization_interval"], 0)
        self.assertTrue(config.get("compact_transition", True))
        self.assertTrue(config["worker_rollout"])
        self.assertFalse(config["mixed_precision"])
        self.assertTrue(config["mask_cache"])
        self.assertTrue(config["vectorized_occupancy"])
        self.assertTrue(config["batch_device_copy"])

    def test_long_optimization_config_enables_performance_paths(self):
        with (ROOT / "config" / "optimization_long.yaml").open() as handle:
            config = yaml.safe_load(handle)
        self.assertGreaterEqual(config["episode"], 100)
        self.assertTrue(config["compact_transition"])
        self.assertTrue(config["worker_rollout"])
        self.assertTrue(config["mask_cache"])
        self.assertTrue(config["vectorized_occupancy"])
        self.assertTrue(config["batch_device_copy"])
        self.assertFalse(config["use_dmp_for_evaluation"])


if __name__ == "__main__":
    unittest.main()
