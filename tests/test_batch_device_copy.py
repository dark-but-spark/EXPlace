import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import PPOAgent


class Transition:
    def __init__(self, state, action, reward, action_log_prob, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.action_log_prob = action_log_prob
        self.done = done


def make_agent(batch_device_copy, compact_transition=True):
    args = SimpleNamespace(
        device="cpu", batch_device_copy=batch_device_copy, gamma=0.95,
        reward_granularity="step", global_reward_gamma=1.0, batch_size=2,
        buffer_size=1, epoch=1, clip_param=0.2, max_grad_norm=0.5,
        disable_tqdm=True, mixed_precision=False, amp_dtype="float16",
        compact_transition=compact_transition,
    )
    agent = PPOAgent.__new__(PPOAgent)
    agent.args = args
    agent.actor = nn.Sequential(nn.Flatten(), nn.Linear(4, 2), nn.Softmax(dim=-1))
    agent.critic = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))
    agent.actor_opt = torch.optim.SGD(agent.actor.parameters(), lr=0.01)
    agent.critic_opt = torch.optim.SGD(agent.critic.parameters(), lr=0.01)
    agent.buffer_capacity = 2
    agent.batch_size = 2
    agent.counter = 2
    agent.training_step = 0
    agent.state_rebuilder = None
    agent.mixed_precision = False
    agent.amp_dtype = torch.float32
    state = np.ones((1, 1, 2, 2), dtype=np.float32)
    agent.buffer = [
        Transition(state, 0, torch.tensor(1.0), -0.69, False),
        Transition(state, 1, torch.tensor(0.5), -0.69, True),
    ]
    return agent


class BatchDeviceCopyTest(unittest.TestCase):
    def test_staged_and_per_minibatch_paths_both_update(self):
        for enabled in (True, False):
            agent = make_agent(enabled)
            duration, actor_loss, critic_loss = agent.update()
            self.assertGreaterEqual(duration, 0.0)
            self.assertTrue(np.isfinite(actor_loss))
            self.assertTrue(np.isfinite(critic_loss))
            self.assertEqual(agent.training_step, 1)

    def test_default_fp32_path_does_not_enable_bf16_loss_fallback(self):
        agent = make_agent(True)
        self.assertFalse(agent.mixed_precision)
        self.assertEqual(agent.amp_dtype, torch.float32)

    def test_non_compact_baseline_keeps_fp32_transition_values(self):
        agent = make_agent(True, compact_transition=False)
        precise = np.array([[[[1.0001234, 2.0001234], [3.0001234, 4.0001234]]]], dtype=np.float32)
        agent.buffer[0].state = precise
        agent.buffer[1].state = precise
        batch = agent._build_state_batch().numpy()
        np.testing.assert_array_equal(batch[0], precise[0])

    def test_compact_transition_can_use_fp16_host_storage(self):
        agent = make_agent(True, compact_transition=True)
        precise = np.array([[[[1.0001234, 2.0001234], [3.0001234, 4.0001234]]]], dtype=np.float32)
        agent.buffer[0].state = precise
        agent.buffer[1].state = precise
        batch = agent._build_state_batch().numpy()
        self.assertGreater(np.max(np.abs(batch[0] - precise[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
