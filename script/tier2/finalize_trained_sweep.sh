#!/usr/bin/env bash
# Wait for the trained sweep to drain, then analyse it.
# Requested behaviour: no analysis while the round is still running.
set -uo pipefail
P=/data0/cjy_cxsj/EXPlace-test/EXPlace
OUT=$P/experiment_records/20261118_p1_1_trained_sweep
LOG=$OUT/finalize.log

echo "[finalize] waiting for the sweep to finish ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"
while pgrep -f 'spread_sampler\.sh' > /dev/null 2>&1; do sleep 60; done
while pgrep -x openroad > /dev/null 2>&1; do sleep 30; done
echo "[finalize] sweep drained ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"

/usr/bin/python3 - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import csv, io, os, sys
from itertools import combinations

out_dir = sys.argv[1]
rows = list(csv.DictReader(io.open(os.path.join(out_dir, 'labels.csv'), encoding='utf-8')))
rows = [r for r in rows if r.get('tier2_grt_tns')]
for r in rows:
    r['gp'] = float(r['tier1_gp_hpwl'])
    r['grt'] = float(r['tier2_grt_tns'])
    r['gwn'] = float(r['tier2_grt_wns']) if r['tier2_grt_wns'] else None
    r['drt'] = float(r['tier2_drt_tns']) if r['tier2_drt_tns'] else None
    r['drc'] = float(r['drc_violations']) if r['drc_violations'] else None
    r['rwl'] = float(r['rwl_um']) if r['rwl_um'] else None


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0]*len(v); i = 0
    while i < len(o):
        j = i
        while j+1 < len(o) and v[o[j+1]] == v[o[i]]:
            j += 1
        a = (i+j)/2 + 1
        for k in range(i, j+1):
            out[o[k]] = a
        i = j+1
    return out


def sp(a, b):
    n = len(a)
    if n < 3:
        return float('nan')
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = (sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb))**0.5
    return num/den if den else float('nan')


lines = []
lines.append(f"# Trained sweep (600 episodes fixed): analysis\n")
lines.append(f"samples with post-GRT labels: {len(rows)}")
lines.append("")
lines.append("| seed | halo | cab | gp_hpwl (M) | grt_tns | grt_wns | drt_tns | rWL (um) | DRC | note |")
lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for r in sorted(rows, key=lambda x: (x['halo'], x['core_area_block'], int(x['seed']))):
    lines.append("| {s} | {h} | {c} | {gp:.3f} | {grt:g} | {gwn} | {drt} | {rwl} | {drc} | {n} |".format(
        s=r['seed'], h=r['halo'], c=r['core_area_block'], gp=r['gp']/1e6,
        grt=r['grt'], gwn='' if r['gwn'] is None else r['gwn'],
        drt='' if r['drt'] is None else r['drt'],
        rwl='' if r['rwl'] is None else f"{r['rwl']:.0f}",
        drc='' if r['drc'] is None else f"{r['drc']:.0f}", n=r['note']))
lines.append("")

gp = [r['gp'] for r in rows]
grt = [r['grt'] for r in rows]
lines.append("## Correlations (raw values; negative = proxy informative)")
lines.append(f"- rho(gp_hpwl, grt_tns) = **{sp(gp, grt):+.3f}**  (n={len(rows)})")
if any(r['gwn'] is not None for r in rows):
    g = [r for r in rows if r['gwn'] is not None]
    lines.append(f"- rho(gp_hpwl, grt_wns) = {sp([r['gp'] for r in g], [r['gwn'] for r in g]):+.3f}  (n={len(g)})")
if any(r['drt'] is not None for r in rows):
    g = [r for r in rows if r['drt'] is not None]
    lines.append(f"- rho(gp_hpwl, drt_tns) = {sp([r['gp'] for r in g], [r['drt'] for r in g]):+.3f}  (n={len(g)})")
if any(r['rwl'] is not None for r in rows):
    g = [r for r in rows if r['rwl'] is not None]
    lines.append(f"- rho(gp_hpwl, rwl_um)  = {sp([r['gp'] for r in g], [r['rwl'] for r in g]):+.3f}  (n={len(g)})")
if any(r['drc'] is not None for r in rows):
    g = [r for r in rows if r['drc'] is not None]
    lines.append(f"- rho(gp_hpwl, drc)     = {sp([r['gp'] for r in g], [r['drc'] for r in g]):+.3f}  (n={len(g)})")
lines.append("")

band = [r for r in rows if r['gp'] < 1.5e7]
lines.append("## Outlier sensitivity")
lines.append(f"- gp_hpwl span: {min(gp)/1e6:.3f}M .. {max(gp)/1e6:.3f}M ({(max(gp)-min(gp))/min(gp)*100:.1f}%)")
if len(band) >= 3:
    bg = [r['gp'] for r in band]; bt = [r['grt'] for r in band]
    lines.append(f"- excluding placements above 15M: n={len(band)}, rho = **{sp(bg, bt):+.3f}**")
    lines.append(f"- in-band gp_hpwl span: {(max(bg)-min(bg))/min(bg)*100:.2f}%  "
                 f"while grt_tns spans {min(bt):g} .. {max(bt):g} "
                 f"({(max(bt)-min(bt))/abs(max(bt))*100:.0f}% of |best|)")
outliers = [r for r in rows if r['gp'] >= 1.5e7]
if outliers:
    lines.append("- outliers (>15M): " + ", ".join(
        f"seed {r['seed']} halo{r['halo']} cab{r['core_area_block']} gp={r['gp']/1e6:.2f}M tns={r['grt']:g}"
        for r in outliers))
lines.append("")

lines.append("## Per-configuration (4 seeds each)")
lines.append("| halo | cab | n | gp mean (M) | grt_tns mean | rho (4 seeds) |")
lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
groups = {}
for r in rows:
    groups.setdefault((r['halo'], r['core_area_block']), []).append(r)
for key in sorted(groups):
    g = groups[key]
    if len(g) < 3:
        continue
    lines.append("| {h} | {c} | {n} | {gp:.3f} | {t:+.2f} | {rho:+.3f} |".format(
        h=key[0], c=key[1], n=len(g), gp=sum(x['gp'] for x in g)/len(g)/1e6,
        t=sum(x['grt'] for x in g)/len(g),
        rho=sp([x['gp'] for x in g], [x['grt'] for x in g])))
lines.append("")

verdict = []
if len(band) >= 3:
    rho_band = sp([r['gp'] for r in band], [r['grt'] for r in band])
    if abs(rho_band) < 0.4:
        verdict.append(f"in-band rho = {rho_band:+.3f} < 0.4 -> P1-1 decision gate says "
                       "stop learning reward weights from this proxy")
    elif abs(rho_band) < 0.7:
        verdict.append(f"in-band rho = {rho_band:+.3f} in 0.4-0.7 -> downgrade to exploratory")
    else:
        verdict.append(f"in-band rho = {rho_band:+.3f} > 0.7 -> proxy is usable")
lines.append("## Reading")
lines += [f"- {v}" for v in verdict]
lines.append("- Any pooled correlation across episode lengths is confounded: episode count correlates")
lines.append("  with both gp_hpwl (-0.79) and post-GRT TNS (+0.68). Compare inside one length only.")
lines.append("- DRT columns are empty where the flow died in 6_report (PSM-0069); post-GRT labels there")
lines.append("  are still valid (the GRT stage completed).")

text = "\n".join(lines) + "\n"
io.open(os.path.join(out_dir, 'analysis.md'), 'w', encoding='utf-8').write(text)
print(text)
PY

echo "[finalize] wrote $OUT/analysis.md ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"
