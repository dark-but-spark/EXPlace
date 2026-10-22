import csv
import importlib.util
import io
import pathlib
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "script" / "summarize_experiment.py"
SPEC = importlib.util.spec_from_file_location("summarize_experiment", MODULE_PATH)
summarize_experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_experiment)


class SummarizeExperimentTest(unittest.TestCase):
    def write_metrics(self, root, rows, timestamp="run_1"):
        path = (
            pathlib.Path(root)
            / "fixed_step"
            / "superblue18"
            / "seed_3"
            / "output"
            / "rl_logs"
            / "superblue18"
            / timestamp
            / "eval_metrics.csv"
        )
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as stream:
            fieldnames = ["episode", "tns", "wns", "gp_hpwl"]
            if any("valid" in row for row in rows):
                fieldnames.append("valid")
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_distinguishes_logged_rows_from_physical_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_metrics(
                temp_dir,
                [
                    {"episode": 10, "tns": 0, "wns": 0, "gp_hpwl": 0},
                    {"episode": 100, "tns": -20, "wns": -5, "gp_hpwl": 120},
                    {"episode": 200, "tns": -10, "wns": -6, "gp_hpwl": 100},
                    {"episode": 300, "tns": -15, "wns": -4, "gp_hpwl": 110},
                ],
            )
            result = summarize_experiment.summarize_file(
                path, "fixed_step", "superblue18", 3
            )

        self.assertEqual(result.logged_rows, 4)
        self.assertEqual(result.physical_evaluations, 3)
        self.assertEqual(result.best_gp_hpwl_episode, 200)
        self.assertEqual(result.best_tns_episode, 200)
        self.assertEqual(result.best_wns_episode, 300)
        self.assertEqual(result.final_episode, 300)
        self.assertEqual(result.gp_best_to_final_delta, 10)
        self.assertEqual(result.tns_best_to_final_delta, 5)
        self.assertAlmostEqual(result.gp_best_to_final_percent, 10)
        self.assertAlmostEqual(result.tns_best_to_final_percent, 50)

    def test_rejects_multiple_metric_files_for_one_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = [{"episode": 100, "tns": -1, "wns": -1, "gp_hpwl": 10}]
            self.write_metrics(temp_dir, rows, timestamp="run_1")
            self.write_metrics(temp_dir, rows, timestamp="run_2")
            with self.assertRaisesRegex(ValueError, "multiple eval_metrics"):
                summarize_experiment.discover_runs(temp_dir)

    def test_prefers_episode_aligned_physical_evaluation_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            eval_path = self.write_metrics(
                temp_dir,
                [{"episode": 200, "tns": 0, "wns": 0, "gp_hpwl": 0}],
            )
            physical_path = eval_path.with_name("physical_evaluations.csv")
            with physical_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["episode", "kind", "gp_hpwl", "tns", "wns"],
                )
                writer.writeheader()
                writer.writerow({
                    "episode": 100, "kind": "periodic", "gp_hpwl": 120,
                    "tns": -20, "wns": -5,
                })
                writer.writerow({
                    "episode": 200, "kind": "periodic", "gp_hpwl": 100,
                    "tns": -10, "wns": -4,
                })

            result = summarize_experiment.discover_runs(temp_dir)[0]

        self.assertEqual(result.metrics_path, str(physical_path))
        self.assertEqual(result.physical_evaluations, 2)
        self.assertEqual(result.best_gp_hpwl_episode, 200)
        self.assertEqual(result.final_episode, 200)

    def test_excludes_invalid_physical_rows_from_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_metrics(
                temp_dir,
                [
                    {"episode": 100, "tns": -20, "wns": -5, "gp_hpwl": 120,
                     "valid": "1"},
                    {"episode": 200, "tns": 100, "wns": 100, "gp_hpwl": 1,
                     "valid": "0"},
                    {"episode": 300, "tns": -10, "wns": -4, "gp_hpwl": 100,
                     "valid": "1"},
                ],
            )
            result = summarize_experiment.summarize_file(
                path, "fixed_step", "superblue18", 3
            )

        self.assertEqual(result.invalid_evaluations, 1)
        self.assertEqual(result.physical_evaluations, 2)
        self.assertEqual(result.best_gp_hpwl, 100)
        self.assertEqual(result.best_tns, -10)
        self.assertEqual(result.final_episode, 300)

    def test_printed_summary_names_independent_best_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_metrics(
                temp_dir,
                [
                    {"episode": 100, "tns": -10, "wns": -3, "gp_hpwl": 120},
                    {"episode": 200, "tns": -20, "wns": -4, "gp_hpwl": 100},
                ],
            )
            aggregate = summarize_experiment.aggregate_runs(
                summarize_experiment.discover_runs(temp_dir)
            )
            output = io.StringIO()
            summarize_experiment.print_summary(aggregate, output)

        text = output.getvalue()
        self.assertIn("mean_best_gp_hpwl", text)
        self.assertIn("mean_best_tns", text)
        self.assertIn("fixed_step,superblue18,1", text)


if __name__ == "__main__":
    unittest.main()
