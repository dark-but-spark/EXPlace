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


if __name__ == "__main__":
    unittest.main()
