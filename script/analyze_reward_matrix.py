#!/usr/bin/env python3
"""Compute paired step/global effects from summarize_experiment per-seed CSV."""
import argparse
import csv
import statistics
from pathlib import Path


METRICS = {
    "best_gp_hpwl": "lower",
    "final_gp_hpwl": "lower",
    "best_tns": "higher",
    "final_tns": "higher",
    "best_wns": "higher",
    "final_wns": "higher",
    "gp_best_to_final_percent": "lower",
}


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("per_seed_metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.per_seed_metrics)
    indexed = {}
    for row in rows:
        key = (row["variant"], row["benchmark"], row["seed"])
        if key in indexed:
            raise SystemExit(f"duplicate run: {key}")
        indexed[key] = row

    contrasts = [
        ("fixed_global_minus_fixed_step", {"fixed_global": 1.0, "fixed_step": -1.0}),
        ("adaptive_global_minus_adaptive_step", {"adaptive_global": 1.0, "adaptive_step": -1.0}),
        (
            "global_main_effect_mean_over_weight_mode",
            {"fixed_global": 0.5, "adaptive_global": 0.5,
             "fixed_step": -0.5, "adaptive_step": -0.5},
        ),
        ("adaptive_minus_fixed_at_step", {"adaptive_step": 1.0, "fixed_step": -1.0}),
        ("adaptive_minus_fixed_at_global", {"adaptive_global": 1.0, "fixed_global": -1.0}),
    ]
    pairs = sorted({(row["benchmark"], row["seed"]) for row in rows})
    output = []
    for contrast, weights in contrasts:
        for metric, direction in METRICS.items():
            deltas = []
            missing = []
            for benchmark, seed in pairs:
                needed = [(variant, benchmark, seed) for variant in weights]
                if not all(key in indexed for key in needed):
                    missing.append(f"{benchmark}:seed_{seed}")
                    continue
                delta = sum(
                    weight * float(indexed[(variant, benchmark, seed)][metric])
                    for variant, weight in weights.items()
                )
                deltas.append(delta)
            if not deltas:
                continue
            favorable = sum(value < 0 if direction == "lower" else value > 0 for value in deltas)
            output.append({
                "contrast": contrast,
                "metric": metric,
                "direction": direction,
                "paired_count": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "median_delta": statistics.median(deltas),
                "stdev_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "favorable_pairs": favorable,
                "missing_pairs": ";".join(missing),
            })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    for row in output:
        print(
            f"{row['contrast']},{row['metric']},n={row['paired_count']},"
            f"mean_delta={row['mean_delta']:.6f},"
            f"favorable={row['favorable_pairs']}/{row['paired_count']}"
        )


if __name__ == "__main__":
    main()
