#!/usr/bin/env python3
"""Paired analysis for the P1-2 multi-design matrix, on the post-adaptation window.

Why this script exists
----------------------
`script/summarize_experiment.py` reports `best_*` over the whole episode range.
For an `adaptive_*` arm that is misleading: the PPA-correlation weight learner
does not touch the reward until it has `reward_weight_min_observations` real
evaluations (4 by default).  Every evaluation up to and including the episode
where the first update happens is therefore computed with the *same* fixed
weights in both arms, and those rows are bit-identical across `fixed_*` and
`adaptive_*`.  The global `best_tns` frequently lands on one of them, which
makes a paired difference collapse to exactly 0 and hides the real spread.

This script therefore:
  1. derives the first update episode from `reward_weights.csv` (first row with
     `updated == 1`) instead of hard-coding it,
  2. keeps only evaluations strictly *after* that episode (the prefix rows are
     shared by construction and carry no treatment information),
  3. reports the ep1500 row as the only fully comparable "final" value,
  4. emits per-pair and aggregate CSVs so the numbers in README.md are
     reproducible.

Usage
-----
    /usr/bin/python3 experiment_records/20261109_p12_multidesign/analyze_post_adaptation.py \
        long_runs/p12_multidesign_20261109 \
        --output-dir experiment_records/20261109_p12_multidesign
"""

import argparse
import csv
import glob
import os
import statistics as st
from math import comb

VARIANTS = ("fixed_global", "adaptive_global")
DESIGNS = ("superblue1", "superblue3")
SEEDS = (3, 7, 11, 17, 23, 29)
METRICS = (("tns", True), ("gp_hpwl", False), ("wns", True))
DEFAULT_ADAPT_START = 400


def find_eval_csv(run_root, variant, design, seed):
    pattern = os.path.join(
        run_root, variant, design, f"seed_{seed}",
        "output", "rl_logs", design, "*", "physical_evaluations.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # A resumed/repeated run can leave several timestamp dirs; keep the longest.
    return max(files, key=lambda f: len(list(csv.DictReader(open(f)))))


def load_rows(run_root, variant, design, seed):
    path = find_eval_csv(run_root, variant, design, seed)
    if path is None:
        return None, None
    return list(csv.DictReader(open(path))), path


def load_adapt_start(run_root, design, seed):
    pattern = os.path.join(
        run_root, "adaptive_global", design, f"seed_{seed}",
        "output", "rl_logs", design, "*", "reward_weights.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return DEFAULT_ADAPT_START
    rows = list(csv.DictReader(open(files[0])))
    updated = [int(r["episode"]) for r in rows if r.get("updated") == "1"]
    return min(updated) if updated else DEFAULT_ADAPT_START


def windowed_best(rows, metric, adapt_start, higher_is_better):
    """Best metric strictly after the first weight update (excludes shared prefix)."""
    valid = [r for r in rows if int(r["episode"]) > adapt_start and r["valid"] == "1"]
    if not valid:
        return None, None
    pick = max if higher_is_better else min
    chosen = pick(valid, key=lambda r: float(r[metric]))
    return float(chosen[metric]), int(chosen["episode"])


def final_value(rows, metric, episode=1500):
    for r in rows:
        if int(r["episode"]) == episode and r["valid"] == "1":
            return float(r[metric])
    return None


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="matrix run dir, e.g. long_runs/p12_multidesign_20261109")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = args.run_root.rstrip("/")
    out_dir = args.output_dir or run_root
    os.makedirs(out_dir, exist_ok=True)

    rows, sources, adapt_start = {}, {}, {}
    for design in DESIGNS:
        for seed in SEEDS:
            adapt_start[(design, seed)] = load_adapt_start(run_root, design, seed)
            for variant in VARIANTS:
                data, path = load_rows(run_root, variant, design, seed)
                rows[(variant, design, seed)] = data
                sources[(variant, design, seed)] = path

    per_pair = []
    for design in DESIGNS:
        for seed in SEEDS:
            start = adapt_start[(design, seed)]
            rec = {
                "benchmark": design,
                "seed": seed,
                "first_update_episode": start,
                "window": f"episode>{start}",
                "paired": 1,
            }
            for variant in VARIANTS:
                data = rows.get((variant, design, seed))
                tag = "fixed" if variant.startswith("fixed") else "adaptive"
                if not data:
                    rec[f"{tag}_evaluations"] = 0
                    rec["paired"] = 0
                    continue
                rec[f"{tag}_evaluations"] = len(data)
                rec[f"{tag}_invalid"] = sum(1 for r in data if r["valid"] != "1")
                rec[f"{tag}_last_episode"] = int(data[-1]["episode"])
                for metric, high in METRICS:
                    value, episode = windowed_best(data, metric, start, high)
                    rec[f"{tag}_best_{metric}_post"] = value
                    rec[f"{tag}_best_{metric}_post_episode"] = episode
                    rec[f"{tag}_final_{metric}"] = final_value(data, metric)
            for metric, high in METRICS:
                for kind in ("best", "final"):
                    suffix = "_post" if kind == "best" else ""
                    f = rec.get(f"fixed_{kind}_{metric}{suffix}")
                    a = rec.get(f"adaptive_{kind}_{metric}{suffix}")
                    delta = None if (f is None or a is None) else a - f
                    rec[f"delta_{kind}_{metric}"] = delta
                    if delta is not None and f:
                        rec[f"relative_{kind}_{metric}_percent"] = delta / abs(f) * 100
            per_pair.append(rec)

    pair_path = os.path.join(out_dir, "post_adaptation_per_pair.csv")
    with open(pair_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_pair[0].keys()))
        writer.writeheader()
        writer.writerows(per_pair)

    summary = []
    for scope in list(DESIGNS) + ["ALL"]:
        selected = [r for r in per_pair
                    if r["paired"] == 1 and (scope == "ALL" or r["benchmark"] == scope)]
        for metric, high in METRICS:
            for kind in ("best", "final"):
                deltas = [r[f"delta_{kind}_{metric}"] for r in selected
                          if r.get(f"delta_{kind}_{metric}") is not None]
                if not deltas:
                    continue
                rels = [r[f"relative_{kind}_{metric}_percent"] for r in selected
                        if r.get(f"relative_{kind}_{metric}_percent") is not None]
                wins = losses = ties = 0
                for delta in deltas:
                    if delta == 0:
                        ties += 1
                    elif (delta > 0) == high:
                        wins += 1
                    else:
                        losses += 1
                summary.append({
                    "scope": scope,
                    "metric": metric,
                    "kind": kind,
                    "higher_is_better": int(high),
                    "n": len(deltas),
                    "mean_delta_adaptive_minus_fixed": st.mean(deltas),
                    "median_delta": st.median(deltas),
                    "mean_relative_percent": st.mean(rels) if rels else "",
                    "adaptive_wins": wins,
                    "fixed_wins": losses,
                    "ties": ties,
                    "sign_test_p": round(sign_test_p(wins, losses), 4),
                })

    sum_path = os.path.join(out_dir, "post_adaptation_summary.csv")
    with open(sum_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {pair_path}")
    print(f"wrote {sum_path}")
    print()
    print(f"{'scope':<11} {'metric':<8} {'kind':<6} {'n':>3} {'mean_delta':>12} "
          f"{'median':>10} {'rel%':>7} {'adapt_w':>8} {'fixed_w':>8} {'ties':>5} {'sign_p':>7}")
    for rec in summary:
        scale = 1e6 if rec["metric"] == "gp_hpwl" else 1.0
        rel = rec["mean_relative_percent"]
        print(f"{rec['scope']:<11} {rec['metric']:<8} {rec['kind']:<6} {rec['n']:>3} "
              f"{rec['mean_delta_adaptive_minus_fixed']/scale:>12.2f} "
              f"{rec['median_delta']/scale:>10.2f} "
              f"{(f'{rel:7.2f}' if rel != '' else '   n/a ')} {rec['adaptive_wins']:>8} "
              f"{rec['fixed_wins']:>8} {rec['ties']:>5} {rec['sign_test_p']:>7.3f}")


if __name__ == "__main__":
    main()
