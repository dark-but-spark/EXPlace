import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import PPOAgent


class WorkerPolicyTest(unittest.TestCase):
    def test_inference_only_agent_has_no_critic_or_optimizers(self):
        args = SimpleNamespace(
            device="cpu", used_masks=["proxy", "pos"], prototype_flag=False,
            actor_arch="parallel", local_summary_k=2, grid=32,
            soft_coefficient=1.0, trade_off_coeff=[1.0], lr=1e-3,
            batch_size=2, buffer_size=1, n_macro=2, mixed_precision=False,
            amp_dtype="float16",
        )
        agent = PPOAgent(args=args, inference_only=True)
        self.assertIsNone(agent.critic)
        self.assertIsNone(agent.actor_opt)
        self.assertIsNone(agent.critic_opt)
        agent.train()
        agent.eval()


if __name__ == "__main__":
    unittest.main()
