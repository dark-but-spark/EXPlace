#!/usr/bin/env python3
"""Archive quality logs and compare each result with its same-seed baseline."""
import argparse, csv, re, shutil
from pathlib import Path
from statistics import mean, stdev

PATTERN = re.compile(r"Timing results - GP_HPWL:\s*([-+0-9.eE]+),\s*TNS:\s*([-+0-9.eE]+),\s*WNS:\s*([-+0-9.eE]+)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("quality_root", type=Path)
    p.add_argument("--performance-root", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path)
    a = p.parse_args()
    root = a.quality_root.resolve()
    archive = (a.archive_dir or root / "archive").resolve()
    archive.mkdir(parents=True, exist_ok=True)
    rows = []
    for log in sorted(root.glob("*_seed_*.log")):
        m = re.match(r"(.+)_seed_(\d+)\.log$", log.name)
        if not m: continue
        hit = PATTERN.search(log.read_text(errors="replace"))
        if not hit: continue
        variant, seed = m.group(1), int(m.group(2))
        gp, tns, wns = map(float, hit.groups())
        rows.append({"variant":variant,"benchmark":"superblue18","seed":seed,
                     "gp_hpwl":gp,"tns":tns,"wns":wns,"log":str(log)})
    keys = [(r["variant"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate variant/seed quality logs found")
    base = {(r["seed"]): r for r in rows if r["variant"] == "baseline"}
    if not base:
        raise SystemExit("baseline quality logs are required for paired comparison")
    missing_baselines = sorted({r["seed"] for r in rows} - set(base))
    if missing_baselines:
        raise SystemExit(f"missing baseline quality for seeds: {missing_baselines}")
    for r in rows:
        b = base[r["seed"]]
        r["gp_delta"] = r["gp_hpwl"] - b["gp_hpwl"]
        r["gp_pct"] = 100 * r["gp_delta"] / b["gp_hpwl"]
        r["tns_delta"] = r["tns"] - b["tns"]
        r["wns_delta"] = r["wns"] - b["wns"]
        r["quality_status"] = "baseline" if r["variant"] == "baseline" else ("improved_or_equal" if r["gp_delta"] <= 0 and r["tns_delta"] >= 0 and r["wns_delta"] >= 0 else "tradeoff_or_regression")
    fields = list(rows[0].keys())
    with (archive / "quality_per_seed.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    variants = sorted({r["variant"] for r in rows})
    summary=[]
    for v in variants:
        g=[r for r in rows if r["variant"]==v]
        summary.append({"variant":v,"seeds":len(g),"mean_gp_hpwl":mean(r["gp_hpwl"] for r in g),"mean_tns":mean(r["tns"] for r in g),"mean_wns":mean(r["wns"] for r in g),"mean_gp_pct_vs_baseline":mean(r["gp_pct"] for r in g),"mean_tns_delta_vs_baseline":mean(r["tns_delta"] for r in g),"mean_wns_delta_vs_baseline":mean(r["wns_delta"] for r in g)})
    with (archive / "quality_summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    shutil.copy2(a.performance_root / "results.csv", archive / "performance_results.csv")
    shutil.copy2(a.performance_root / "speedups.csv", archive / "performance_speedups.csv")
    (archive / "README.txt").write_text(f"quality_root={root}\nperformance_root={a.performance_root.resolve()}\nrows={len(rows)}\n")
    print(f"Archived {len(rows)} quality rows to {archive}")
    for r in summary: print(r)

if __name__ == "__main__": main()
