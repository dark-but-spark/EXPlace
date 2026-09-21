import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np


SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from place_env.place_env import PlaceEnv


def make_env(reward_granularity):
    args = SimpleNamespace(
        grid=32,
        n_macro=2,
        ratio_x=1.0,
        ratio_y=1.0,
        ratio_sum=2.0,
        used_masks=["displacement", "pos"],
        trade_off_coeff=[1.0],
        dataflow_cutoff=0.0,
        corner_flag=False,
        visualize_flag=False,
        core_area_block=0,
        regulator_flag=False,
        reward_granularity=reward_granularity,
        use_reward_scaling=False,
        prototype_flag=False,
        soft_coefficient=1.0,
        regularity_mode="corner_regular",
        log_dir="/tmp",
    )
    env_params = {
        "macro_pos": {0: (0, 0, 1, 1), 1: (2, 2, 1, 1)},
        "node_id_to_name": [0, 1],
        "ratio_x": 1.0,
        "ratio_y": 1.0,
        "port_pos": [],
        "pin_blocking_rectangles": [],
        "macro_clusters": [],
        "node_to_net_dict": {0: [], 1: []},
        "net_info": {},
        "node_info": {},
        "port_to_net_dict": {},
        "port_info": {},
    }
    return PlaceEnv(args, env_params)


class RewardGranularityTest(unittest.TestCase):
    def test_step_mode_preserves_original_dense_reward(self):
        env = make_env("step")
        state = env.reset()
        self.assertEqual(state.dtype, np.float32)
        _, first_reward, first_done, _ = env.step(5 * env.grid + 5)
        _, second_reward, second_done, info = env.step(0)

        self.assertFalse(first_done)
        self.assertTrue(second_done)
        self.assertLess(first_reward, 0.0)
        self.assertAlmostEqual(first_reward + second_reward, info["scaled_reward"])

    def test_global_mode_returns_only_terminal_aggregate(self):
        env = make_env("global")
        env.reset()
        _, first_reward, first_done, _ = env.step(5 * env.grid + 5)
        _, terminal_reward, terminal_done, info = env.step(0)

        self.assertFalse(first_done)
        self.assertEqual(first_reward, 0.0)
        self.assertTrue(terminal_done)
        self.assertAlmostEqual(terminal_reward, info["scaled_reward"])

    def test_runtime_weight_update_changes_reward_scale(self):
        baseline_env = make_env("step")
        baseline_env.reset()
        _, baseline_reward, _, _ = baseline_env.step(5 * baseline_env.grid + 5)

        updated_env = make_env("step")
        updated_env.set_reward_weights([0.5])
        updated_env.reset()
        _, updated_reward, _, _ = updated_env.step(5 * updated_env.grid + 5)
        self.assertAlmostEqual(updated_reward, 0.5 * baseline_reward)


if __name__ == "__main__":
    unittest.main()
