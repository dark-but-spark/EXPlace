"""Regression tests for the evaluation -> training mode hand-off."""

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main


class ModeRecordingAgent:
    """Fake PPOAgent that records whether rollouts see training mode."""

    def __init__(self):
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def update(self):
        return 0.0, 0.0, 0.0

    def save_model(self, *args, **kwargs):
        pass

    def set_reward_weights(self, weights):
        pass

    def set_state_rebuilder(self, fn):
        pass


class EvaluationModeTest(unittest.TestCase):
    def run_training(self, directory, async_evaluation=True,
                     wait_ready=True, evaluation_valid=True):
        config_path = ROOT / "config" / "adaptive_global.yaml"
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        config.update(
            log_dir=str(directory), design_name="superblue18", debug=False,
            episode=30, rollout_batch_size=5, evaluation_interval=10,
            reward_weight_mode="fixed", reward_granularity="global",
            async_evaluation=async_evaluation, compact_transition=False,
            worker_rollout=False, mask_cache=False, vectorized_occupancy=False,
            batch_device_copy=False, mixed_precision=False, policy_mask_fp32=False,
            use_reward_scaling=False, visualize_prototype=False,
            visualization_interval=0,
        )
        args = SimpleNamespace(**config)
        agent = ModeRecordingAgent()
        test_env = mock.Mock()
        workers = [mock.Mock() for _ in range(args.rollout_batch_size)]
        problem = mock.Mock()
        problem.last_evaluation_valid = evaluation_valid
        problem.last_evaluation_failure = "" if evaluation_valid else "synthetic_invalid"
        problem.evaluate.side_effect = lambda macro_pos: (100.0, -10.0, -1.0)
        evaluator = mock.Mock()
        writer = mock.Mock()

        modes_at_rollout = []
        inference_calls = []
        rollout_info = dict(
            reg=1, hier=1, df=1, wire=1, port=1, scaled_reward=-1,
            batch_best_idx=0,
        )

        def fake_run(*args, **kwargs):
            modes_at_rollout.append(agent.training)
            return rollout_info, None

        def fake_inference(*args, **kwargs):
            inference_calls.append(agent.training)
            agent.eval()
            return {"macro_pos": {"macro": 1}}

        def fake_evaluate(episode, macro_pos):
            return {
                "episode": episode,
                "metrics": (100.0, -10.0, -1.0),
                "valid": evaluation_valid,
                "failure_reason": "" if evaluation_valid else "synthetic_invalid",
            }

        def fake_wait(refs, **kwargs):
            if wait_ready:
                return refs, []
            return [], refs

        evaluator.evaluate.remote.side_effect = fake_evaluate

        with contextlib.ExitStack() as stack:
            patches = (
                mock.patch.object(main, "seed_torch"),
                mock.patch.object(main, "SummaryWriter", return_value=writer),
                mock.patch.object(main, "ProblemInstance", return_value=problem),
                mock.patch.object(main, "init_ray_envs", return_value=workers),
                mock.patch.object(main, "create_single_env", return_value=test_env),
                mock.patch.object(main, "PPOAgent", return_value=agent),
                mock.patch.object(main, "run", side_effect=fake_run),
                mock.patch.object(main, "inference", side_effect=fake_inference),
                mock.patch.object(main.AsyncPhysicalEvaluator, "remote",
                                  return_value=evaluator),
                mock.patch.object(main.ray, "get", side_effect=lambda value: value),
                mock.patch.object(main.ray, "wait", side_effect=fake_wait),
                mock.patch.object(main.ray, "kill"),
                mock.patch.object(main.ray, "shutdown"),
                mock.patch.object(main.th, "save"),
            )
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main(args, str(config_path))

        return modes_at_rollout, inference_calls

    def assert_all_training(self, modes, inference_calls):
        self.assertTrue(inference_calls, "inference was never called")
        self.assertEqual(modes, [True] * len(modes),
                         f"rollout batches ran without sampling: {modes}")

    def test_async_valid_evaluation_keeps_training_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            modes, calls = self.run_training(directory)
        self.assert_all_training(modes, calls)

    def test_async_pending_evaluation_keeps_training_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            modes, calls = self.run_training(directory, wait_ready=False)
        self.assert_all_training(modes, calls)

    def test_async_invalid_evaluation_keeps_training_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            modes, calls = self.run_training(directory, evaluation_valid=False)
        self.assert_all_training(modes, calls)

    def test_sync_evaluation_keeps_training_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            modes, calls = self.run_training(directory, async_evaluation=False)
        self.assert_all_training(modes, calls)


if __name__ == "__main__":
    unittest.main()
