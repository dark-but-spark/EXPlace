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
        self.assertEqual(config["worker_policy_device"], "cpu")
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

    def test_gpu_long_config_uses_cuda_policy_decode(self):
        with (ROOT / "config" / "optimization_gpu_long.yaml").open() as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["use_cuda"])
        self.assertEqual(config["worker_policy_device"], "cuda")
        self.assertTrue(config["policy_mask_fp32"])

    def test_physical_quick_config_runs_evaluation_during_smoke(self):
        with (ROOT / "config" / "compare_physical_quick.yaml").open() as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["use_dmp_for_evaluation"])
        self.assertTrue(config["use_timer_for_evaluation"])
        self.assertEqual(config["episode"] % config["rollout_batch_size"], 0)
        self.assertLessEqual(config["evaluation_interval"], config["episode"])


if __name__ == "__main__":
    unittest.main()
