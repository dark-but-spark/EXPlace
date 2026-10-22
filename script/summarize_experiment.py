#!/usr/bin/env python3
"""Summarize physical-evaluation rows from an EXPlace matrix run."""

import argparse
import csv
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_COLUMNS = {"episode", "tns", "wns", "gp_hpwl"}


@dataclass(frozen=True)
class RunMetrics:
    variant: str
    benchmark: str
    seed: int
    metrics_path: str
    logged_rows: int
    physical_evaluations: int
    invalid_evaluations: int
    final_episode: int
    best_gp_hpwl_episode: int
    best_gp_hpwl: float
    best_gp_hpwl_tns: float
    best_gp_hpwl_wns: float
    best_tns_episode: int
    best_tns: float
    best_tns_gp_hpwl: float
    best_tns_wns: float
    best_wns_episode: int
    best_wns: float
    final_gp_hpwl: float
    final_tns: float
    final_wns: float
    gp_best_to_final_delta: float
    gp_best_to_final_percent: float
    tns_best_to_final_delta: float
    tns_best_to_final_percent: float
    wns_best_to_final_delta: float
    wns_best_to_final_percent: float


def _number(row, name, path):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name!r} in {path}: {row.get(name)!r}") from error


def _episode(row, path):
    value = _number(row, "episode", path)
    if not value.is_integer():
        raise ValueError(f"non-integer episode in {path}: {value}")
    return int(value)


def _degradation_percent(best, final):
    """Return positive percent when final is farther from the zero optimum."""
    denominator = abs(best)
    if denominator == 0:
        return 0.0 if final == 0 else float("inf")
    return 100.0 * (abs(final) - denominator) / denominator


def summarize_file(path, variant, benchmark, seed):
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns in {path}: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"no metric rows in {path}")

    episodes = [_episode(row, path) for row in rows]
    if episodes != sorted(episodes) or len(episodes) != len(set(episodes)):
        raise ValueError(f"episodes are not strictly increasing in {path}")

    # The training logger writes gp_hpwl=0 on batches where DREAMPlace was not
    # called. A real HPWL is strictly positive, so this is the physical-row marker.
    def is_valid(row):
        # Older runs predate the validity column and remain usable here.
        value = row.get("valid", "1").strip().lower()
        return value not in {"0", "false", "no"}

    invalid_rows = [row for row in rows if not is_valid(row)]
    physical_rows = [
        row for row in rows
        if is_valid(row) and _number(row, "gp_hpwl", path) > 0
    ]
    if not physical_rows:
        raise ValueError(f"no physical-evaluation rows in {path}")

    best_gp = min(physical_rows, key=lambda row: _number(row, "gp_hpwl", path))
    best_tns = max(physical_rows, key=lambda row: _number(row, "tns", path))
    best_wns = max(physical_rows, key=lambda row: _number(row, "wns", path))
    final = physical_rows[-1]

    best_gp_value = _number(best_gp, "gp_hpwl", path)
    best_tns_value = _number(best_tns, "tns", path)
    best_wns_value = _number(best_wns, "wns", path)
    final_gp = _number(final, "gp_hpwl", path)
    final_tns = _number(final, "tns", path)
    final_wns = _number(final, "wns", path)

    return RunMetrics(
        variant=variant,
        benchmark=benchmark,
        seed=int(seed),
        metrics_path=str(path),
        logged_rows=len(rows),
        physical_evaluations=len(physical_rows),
        invalid_evaluations=len(invalid_rows),
        final_episode=_episode(final, path),
        best_gp_hpwl_episode=_episode(best_gp, path),
        best_gp_hpwl=best_gp_value,
        best_gp_hpwl_tns=_number(best_gp, "tns", path),
        best_gp_hpwl_wns=_number(best_gp, "wns", path),
        best_tns_episode=_episode(best_tns, path),
        best_tns=best_tns_value,
        best_tns_gp_hpwl=_number(best_tns, "gp_hpwl", path),
        best_tns_wns=_number(best_tns, "wns", path),
        best_wns_episode=_episode(best_wns, path),
        best_wns=best_wns_value,
        final_gp_hpwl=final_gp,
        final_tns=final_tns,
        final_wns=final_wns,
        gp_best_to_final_delta=final_gp - best_gp_value,
        gp_best_to_final_percent=100.0 * (final_gp - best_gp_value) / best_gp_value,
        tns_best_to_final_delta=best_tns_value - final_tns,
        tns_best_to_final_percent=_degradation_percent(best_tns_value, final_tns),
        wns_best_to_final_delta=best_wns_value - final_wns,
        wns_best_to_final_percent=_degradation_percent(best_wns_value, final_wns),
    )


def discover_runs(run_root):
    run_root = Path(run_root).resolve()
    matches = sorted(run_root.glob("*/**/seed_*/output/rl_logs/*/*/eval_metrics.csv"))
    if not matches:
        raise ValueError(f"no matrix eval_metrics.csv files under {run_root}")

    discovered = {}
    for path in matches:
        relative = path.relative_to(run_root)
        parts = relative.parts
        seed_index = next(
            (index for index, part in enumerate(parts) if part.startswith("seed_")),
            None,
        )
        if seed_index is None or seed_index < 2:
            raise ValueError(f"cannot identify variant/benchmark/seed from {path}")
        variant = parts[seed_index - 2]
        benchmark = parts[seed_index - 1]
        try:
            seed = int(parts[seed_index].removeprefix("seed_"))
        except ValueError as error:
            raise ValueError(f"invalid seed directory in {path}") from error
        key = (variant, benchmark, seed)
        if key in discovered:
            raise ValueError(
                f"multiple eval_metrics.csv files for {key}: "
                f"{discovered[key]} and {path}"
            )
        discovered[key] = path

    runs = []
    for (variant, benchmark, seed), metrics_path in sorted(discovered.items()):
        physical_path = metrics_path.with_name("physical_evaluations.csv")
        if physical_path.is_file():
            metrics_path = physical_path
        runs.append(summarize_file(metrics_path, variant, benchmark, seed))
    return runs


def aggregate_runs(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault((run.variant, run.benchmark), []).append(run)

    fields = (
        "best_gp_hpwl",
        "best_tns",
        "best_wns",
        "final_gp_hpwl",
        "final_tns",
        "final_wns",
        "gp_best_to_final_percent",
        "tns_best_to_final_percent",
        "wns_best_to_final_percent",
    )
    rows = []
    for (variant, benchmark), group in sorted(grouped.items()):
        row = {
            "variant": variant,
            "benchmark": benchmark,
            "seeds": " ".join(str(run.seed) for run in sorted(group, key=lambda run: run.seed)),
            "run_count": len(group),
            "logged_rows_per_run": ";".join(str(run.logged_rows) for run in group),
            "physical_evaluations_per_run": ";".join(
                str(run.physical_evaluations) for run in group
            ),
            "invalid_evaluations_per_run": ";".join(
                str(run.invalid_evaluations) for run in group
            ),
        }
        row.update(
            {
                f"mean_{field}": statistics.mean(getattr(run, field) for run in group)
                for field in fields
            }
        )
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, stream=sys.stdout):
    print(
        "variant,benchmark,runs,logged_rows/evaluations,"
        "invalid_evaluations,"
        "mean_best_gp_hpwl,mean_best_tns,mean_final_gp_hpwl,mean_final_tns,"
        "mean_gp_regression_pct,mean_tns_regression_pct",
        file=stream,
    )
    for row in rows:
        print(
            f"{row['variant']},{row['benchmark']},{row['run_count']},"
            f"{row['logged_rows_per_run']}/{row['physical_evaluations_per_run']},"
            f"{row['invalid_evaluations_per_run']},"
            f"{row['mean_best_gp_hpwl']:.6f},{row['mean_best_tns']:.6f},"
            f"{row['mean_final_gp_hpwl']:.6f},{row['mean_final_tns']:.6f},"
            f"{row['mean_gp_best_to_final_percent']:.6f},"
            f"{row['mean_tns_best_to_final_percent']:.6f}",
            file=stream,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize independent best-GP, best-TNS, best-WNS, and final "
            "physical evaluations from an EXPlace matrix run."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write per_seed_metrics.csv and aggregate_metrics.csv here",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        runs = discover_runs(args.run_root)
        aggregate = aggregate_runs(runs)
        if args.output_dir:
            write_csv(
                args.output_dir / "per_seed_metrics.csv",
                [asdict(run) for run in runs],
            )
            write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
        print_summary(aggregate)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
