"""Safety tests for asynchronous physical evaluation."""

import contextlib
import csv
import io
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import async_evaluator
import main


SENTINEL = object()


class StubAgent:
    def __init__(self):
        self.training = True
        self.reward_weight_calls = []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def update(self):
        return 0.0, 0.0, 0.0

    def save_model(self, *args, **kwargs):
        pass

    def set_reward_weights(self, weights):
        self.reward_weight_calls.append(list(weights))

    def set_state_rebuilder(self, fn):
        pass


class AsyncSafetyTest(unittest.TestCase):
    def run_training(self, directory, metrics=(100.0, -10.0, -1.0),
                     raise_on_get=False, wait_ready=True,
                     evaluation_valid=True, async_eval_timeout=None,
                     reward_weight_mode="fixed"):
        config_path = ROOT / "config" / "adaptive_global.yaml"
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        config.update(
            log_dir=str(directory), design_name="superblue18", debug=False,
            episode=30, rollout_batch_size=5, evaluation_interval=10,
            reward_weight_mode=reward_weight_mode,
            reward_granularity="global", async_evaluation=True,
            compact_transition=False, worker_rollout=False, mask_cache=False,
            vectorized_occupancy=False, batch_device_copy=False,
            mixed_precision=False, policy_mask_fp32=False,
            use_reward_scaling=False, visualize_prototype=False,
            visualization_interval=0,
        )
        if async_eval_timeout is not None:
            config["async_eval_timeout"] = async_eval_timeout
        args = SimpleNamespace(**config)
        agent = StubAgent()
        test_env = mock.Mock()
        workers = [mock.Mock() for _ in range(args.rollout_batch_size)]
        problem = mock.Mock()
        problem.last_evaluation_valid = evaluation_valid
        problem.last_evaluation_failure = "" if evaluation_valid else "synthetic_invalid"
        evaluator = mock.Mock()
        writer = mock.Mock()
        rollout_info = dict(
            reg=1, hier=1, df=1, wire=1, port=1, scaled_reward=-1,
            batch_best_idx=0,
        )

        def fake_run(*args, **kwargs):
            return rollout_info, None

        def fake_inference(*args, **kwargs):
            agent.eval()
            return dict(reg=1, hier=1, df=1, wire=1, port=1,
                        macro_pos={"macro": 1})

        def fake_remote(episode, macro_pos):
            if raise_on_get:
                return SENTINEL
            return {
                "episode": episode,
                "metrics": metrics,
                "valid": evaluation_valid,
                "failure_reason": "" if evaluation_valid else "synthetic_invalid",
            }

        def fake_get(value):
            if value is SENTINEL:
                raise RuntimeError("async evaluator worker died")
            return value

        def fake_wait(refs, **kwargs):
            if wait_ready:
                return refs, []
            return [], refs

        evaluator.evaluate.remote.side_effect = fake_remote

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
                mock.patch.object(main.ray, "get", side_effect=fake_get),
                mock.patch.object(main.ray, "wait", side_effect=fake_wait),
                mock.patch.object(main.ray, "kill"),
                mock.patch.object(main.ray, "shutdown"),
                mock.patch.object(main.th, "save"),
            )
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main(args, str(config_path))

        physical_path = pathlib.Path(args.log_dir) / "physical_evaluations.csv"
        with physical_path.open(newline="") as handle:
            physical_rows = list(csv.DictReader(handle))
        weights_path = pathlib.Path(args.log_dir) / "reward_weights.csv"
        if weights_path.exists():
            with weights_path.open(newline="") as handle:
                weight_rows = list(csv.DictReader(handle))
        else:
            weight_rows = []
        return physical_rows, weight_rows, agent

    def test_non_finite_async_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, weights, agent = self.run_training(
                directory,
                metrics=(float("nan"), -10.0, -1.0),
                reward_weight_mode="ppa_correlation",
            )
        self.assertTrue(rows)
        self.assertTrue(all(row["valid"] == "0" for row in rows))
        self.assertTrue(all(row["failure_reason"] == "non_finite_metrics"
                            for row in rows))
        self.assertEqual(len(weights), 1)  # episode 0 only
        self.assertEqual(agent.reward_weight_calls, [])

    def test_async_evaluator_exception_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, weights, agent = self.run_training(
                directory, raise_on_get=True,
                reward_weight_mode="ppa_correlation",
            )
        self.assertTrue(rows)
        self.assertTrue(all(row["valid"] == "0" for row in rows))
        self.assertTrue(all(
            row["failure_reason"] == "async_eval_exception:RuntimeError"
            for row in rows
        ))
        self.assertEqual(len(weights), 1)
        self.assertEqual(agent.reward_weight_calls, [])

    def test_blocking_consume_times_out_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _ = self.run_training(
                directory, wait_ready=False, async_eval_timeout=0.01,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid"], "0")
        self.assertEqual(rows[0]["failure_reason"], "async_eval_timeout")
        self.assertEqual(rows[0]["episode"], "10")

    def test_worker_import_paths_are_prepared(self):
        worker_paths = async_evaluator._WORKER_PYTHONPATH.split(os.pathsep)
        self.assertIn(str(ROOT / "src"), worker_paths)
        self.assertIn(str(ROOT / "DREAMPlace" / "install"), worker_paths)
        extra = "/tmp/explace_runtime_paths_probe"
        args = SimpleNamespace(runtime_paths=[extra])
        saved = list(sys.path)
        try:
            sys.path[:] = [path for path in sys.path
                           if os.path.abspath(path) != str(ROOT / "src")]
            async_evaluator.prepare_worker_import_paths(args)
            self.assertIn(extra, sys.path)
            self.assertIn(str(ROOT / "src"), [os.path.abspath(p) for p in sys.path])
        finally:
            sys.path[:] = saved


if __name__ == "__main__":
    unittest.main()
