"""Exercise training's evaluation branches without a GPU or DREAMPlace run."""

import contextlib
import csv
import io
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main
from utils.log_utils import save_physical_evaluation


class AdaptiveEvaluationTest(unittest.TestCase):
    def test_physical_log_migrates_legacy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "physical_evaluations.csv"
            path.write_text(
                "episode,kind,gp_hpwl,tns,wns\n"
                "100,periodic,120,-20,-5\n"
            )
            save_physical_evaluation(path, 200, "periodic", 110, -10, -4)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["valid"], "1")
            self.assertEqual(rows[1]["valid"], "1")
            self.assertEqual(rows[1]["episode"], "200")

    def run_training(self, directory, async_evaluation=False, mode="ppa_correlation",
                     granularity="global", evaluation_valid=True):
        config_path = ROOT / "config" / "adaptive_global.yaml"
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        config.update(
            log_dir=str(directory), design_name="superblue18", debug=False,
            episode=20, rollout_batch_size=5, evaluation_interval=5,
            reward_weight_mode=mode, reward_granularity=granularity,
            async_evaluation=async_evaluation, compact_transition=False,
            worker_rollout=False, mask_cache=False, vectorized_occupancy=False,
            batch_device_copy=False, mixed_precision=False, policy_mask_fp32=False,
            use_reward_scaling=False, visualize_prototype=False,
            visualization_interval=0,
        )
        args = SimpleNamespace(**config)
        agent = mock.Mock()
        agent.update.return_value = (0.0, 0.0, 0.0)
        test_env = mock.Mock()
        workers = [mock.Mock() for _ in range(args.rollout_batch_size)]
        problem = mock.Mock()
        problem.last_evaluation_valid = evaluation_valid
        problem.last_evaluation_failure = "synthetic_invalid_placement" if not evaluation_valid else ""
        evaluator = mock.Mock()
        writer = mock.Mock()
        # Correlated regularity and PPA, anti-correlated hierarchy. These are
        # inference costs; the training rollout below deliberately has different
        # costs so observing the wrong trajectory fails the weight assertions.
        trajectories = [
            dict(reg=i, hier=5-i, df=1, wire=1, port=1, macro_pos={"macro": i})
            for i in range(1, 5)
        ]
        metrics = [(100.0*i, -10.0*i, -float(i)) for i in range(1, 5)]
        problem.evaluate.side_effect = metrics
        evaluator.evaluate.remote.side_effect = [
            {
                "episode": 5*i,
                "metrics": value,
                "valid": evaluation_valid,
                "failure_reason": "synthetic_invalid_placement" if not evaluation_valid else "",
            }
            for i, value in enumerate(metrics, 1)
        ]
        rollout_info = dict(
            reg=1, hier=1, df=1, wire=1, port=1, scaled_reward=-1,
            batch_best_idx=0,
        )
        with contextlib.ExitStack() as stack:
            patches = (
                mock.patch.object(main, "seed_torch"),
                mock.patch.object(main, "SummaryWriter", return_value=writer),
                mock.patch.object(main, "ProblemInstance", return_value=problem),
                mock.patch.object(main, "init_ray_envs", return_value=workers),
                mock.patch.object(main, "create_single_env", return_value=test_env),
                mock.patch.object(main, "PPOAgent", return_value=agent),
                mock.patch.object(main, "run", return_value=(rollout_info, None)),
                mock.patch.object(main, "inference", side_effect=trajectories),
                mock.patch.object(main.AsyncPhysicalEvaluator, "remote", return_value=evaluator),
                mock.patch.object(main.ray, "get", side_effect=lambda value: value),
                mock.patch.object(main.ray, "wait", side_effect=lambda refs, **kw: (refs, [])),
                mock.patch.object(main.ray, "kill"),
                mock.patch.object(main.ray, "shutdown"),
                mock.patch.object(main.th, "save"),
            )
            for patch in patches:
                stack.enter_context(patch)
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main(args, str(config_path))

        weights_path = pathlib.Path(args.log_dir) / "reward_weights.csv"
        if weights_path.exists():
            with weights_path.open() as handle:
                weight_rows = list(csv.DictReader(handle))
        else:
            weight_rows = []
        with (pathlib.Path(args.log_dir) / "physical_evaluations.csv").open() as handle:
            physical_rows = list(csv.DictReader(handle))
        return SimpleNamespace(
            args=args, agent=agent, test_env=test_env, workers=workers,
            writer=writer, weight_rows=weight_rows, physical_rows=physical_rows,
            output=output.getvalue(),
        )

    def assert_adaptation(self, result):
        rows = result.weight_rows
        self.assertEqual([int(row["episode"]) for row in rows], [0, 5, 10, 15, 20])
        self.assertEqual([int(row["updated"]) for row in rows], [0, 0, 0, 0, 1])
        self.assertEqual(len(result.physical_rows), 4)
        self.assertEqual(result.output.count("Adaptive reward weights:"), 4)
        initial = [float(rows[0][f"weight_{name}"]) for name in result.args.used_masks[:-1]]
        final = [float(rows[-1][f"weight_{name}"]) for name in result.args.used_masks[:-1]]
        self.assertFalse(np.allclose(initial, final))
        self.assertGreater(final[0], initial[0])
        self.assertLess(final[1], initial[1])
        self.assertAlmostEqual(sum(final), sum(initial))
        self.assertAlmostEqual(final[-1], initial[-1])  # fixed port prior
        np.testing.assert_allclose(result.args.trade_off_coeff, final)
        result.agent.set_reward_weights.assert_called_with(final)
        result.test_env.set_reward_weights.assert_called_with(final)
        self.assertEqual(result.agent.set_reward_weights.call_count, 4)
        for worker in result.workers:
            worker.set_reward_weights.remote.assert_called_with(final)
            self.assertEqual(worker.set_reward_weights.remote.call_count, 4)
        self.assertIn(mock.call("reward_weight/reg", final[0], 20),
                      result.writer.add_scalar.call_args_list)

    def test_legacy_sync_updates_weights_for_step_and_global(self):
        for granularity in ("step", "global"):
            with self.subTest(granularity=granularity), tempfile.TemporaryDirectory() as directory:
                self.assert_adaptation(self.run_training(directory, granularity=granularity))

    def test_sync_and_async_observe_the_same_placement_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            sync = self.run_training(pathlib.Path(directory) / "sync")
            async_result = self.run_training(pathlib.Path(directory) / "async", async_evaluation=True)
            self.assert_adaptation(sync)
            self.assert_adaptation(async_result)
            self.assertEqual(sync.weight_rows, async_result.weight_rows)
            self.assertEqual(sync.physical_rows, async_result.physical_rows)

    def test_fixed_baseline_never_updates_weights_in_either_evaluation_mode(self):
        for async_evaluation in (False, True):
            with self.subTest(async_evaluation=async_evaluation), tempfile.TemporaryDirectory() as directory:
                result = self.run_training(directory, async_evaluation=async_evaluation, mode="fixed")
                self.assertEqual(result.weight_rows, [])
                self.assertEqual(len(result.physical_rows), 4)
                self.assertEqual(result.args.trade_off_coeff, [0.45, 0.2, 0.15, 0.15, 0.05])
                result.agent.set_reward_weights.assert_not_called()
                result.test_env.set_reward_weights.assert_not_called()
                for worker in result.workers:
                    worker.set_reward_weights.remote.assert_not_called()

    def test_invalid_sync_evaluations_are_logged_but_do_not_update(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_training(directory, evaluation_valid=False)
            self.assertEqual(len(result.physical_rows), 4)
            self.assertTrue(all(row["valid"] == "0" for row in result.physical_rows))
            self.assertTrue(all(
                row["failure_reason"] == "synthetic_invalid_placement"
                for row in result.physical_rows
            ))
            self.assertEqual(result.weight_rows[0]["episode"], "0")
            self.assertEqual(len(result.weight_rows), 1)
            result.agent.set_reward_weights.assert_not_called()

    def test_invalid_async_evaluations_are_logged_but_do_not_update(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_training(directory, async_evaluation=True,
                                        evaluation_valid=False)
            self.assertEqual(len(result.physical_rows), 4)
            self.assertTrue(all(row["valid"] == "0" for row in result.physical_rows))
            self.assertEqual(len(result.weight_rows), 1)
            result.agent.set_reward_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
