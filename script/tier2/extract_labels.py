#!/usr/bin/env python3
"""Extract Tier-2 label fields from an OpenROAD flow run.

Emits one CSV row (no header) with:

    grt_tns,grt_wns,drt_tns,drt_wns,rwl_um,hotspots,drc_violations,area_um2

Sources, all produced by the flow itself:
  reports/.../5_global_route.rpt   tns max / wns max
  reports/.../6_finish.rpt         tns max / wns max, design area
  reports/.../congestion.rpt       congestion hotspot count
  reports/.../5_route_drc.rpt      detailed-routing violation count
  results/.../6_final.def          routed wire length (NETS section only)

rWL is summed over the DEF `NETS` section (`+ ROUTED` geometry), skipping
`SPECIALNETS` so the PDN straps do not inflate it.

Usage:
    extract_labels.py --results results/nangate45/bp_fe/EXPlace \
                      [--reports ...] [--dbu 2000] [--format csv|json]
"""

import argparse
import json
import os
import re

COORD = re.compile(r'\(\s*(\*|-?\d+)\s+(\*|-?\d+)\s*\)')


def first_metric(path, key):
    """Grab '<key> max <value>' from an OpenROAD report."""
    if not os.path.exists(path):
        return None
    pattern = re.compile(rf'^{re.escape(key)}\s+max\s+(-?[0-9.]+)\s*$')
    for line in open(path, errors='ignore'):
        hit = pattern.match(line.strip())
        if hit:
            return float(hit.group(1))
    return None


def count_violation_blocks(path):
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path, errors='ignore'):
        if line.startswith('violation type:'):
            n += 1
    return n


def total_power(path):
    """Total power (W) from the report_power table's Total row."""
    if not os.path.exists(path):
        return None
    for line in open(path, errors='ignore'):
        parts = line.split()
        if parts and parts[0] == 'Total' and len(parts) >= 5:
            try:
                return float(parts[4])
            except ValueError:
                continue
    return None


def routed_wirelength(def_path, dbu_per_um):
    """Sum Manhattan length of DEF NETS routing geometry, in um."""
    if not os.path.exists(def_path):
        return None
    total_dbu = 0.0
    in_nets = False
    for raw in open(def_path, errors='ignore'):
        line = raw.strip()
        if line.startswith('NETS ') and not line.startswith('SPECIALNETS'):
            in_nets = True
            continue
        if not in_nets:
            continue
        if line.startswith('END NETS'):
            break
        if '+ ROUTED' not in line and '+ FIXED' not in line and '+ COVER' not in line:
            continue
        points = COORD.findall(line)
        if len(points) < 2:
            continue
        last_x = last_y = None
        for xs, ys in points:
            x = None if xs == '*' else float(xs)
            y = None if ys == '*' else float(ys)
            if x is None:
                x = last_x
            if y is None:
                y = last_y
            if x is None or y is None:
                last_x, last_y = x, y
                continue
            if last_x is not None and last_y is not None:
                total_dbu += abs(x - last_x) + abs(y - last_y)
            last_x, last_y = x, y
    return round(total_dbu / dbu_per_um, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='results/<platform>/<nick>/<variant>')
    ap.add_argument('--reports', default=None)
    ap.add_argument('--dbu', type=float, default=2000.0, help='DEF DB units per micron')
    ap.add_argument('--format', choices=('csv', 'json'), default='csv')
    args = ap.parse_args()

    reports = args.reports or args.results.replace(os.sep + 'results' + os.sep,
                                                  os.sep + 'reports' + os.sep, 1)
    grt_rpt = os.path.join(reports, '5_global_route.rpt')
    fin_rpt = os.path.join(reports, '6_finish.rpt')
    cong_rpt = os.path.join(reports, 'congestion.rpt')
    drc_rpt = os.path.join(reports, '5_route_drc.rpt')
    final_def = os.path.join(args.results, '6_final.def')

    row = {
        'grt_tns': first_metric(grt_rpt, 'tns'),
        'grt_wns': first_metric(grt_rpt, 'wns'),
        'drt_tns': first_metric(fin_rpt, 'tns'),
        'drt_wns': first_metric(fin_rpt, 'wns'),
        'rwl_um': routed_wirelength(final_def, args.dbu),
        'hotspots': count_violation_blocks(cong_rpt),
        'drc_violations': count_violation_blocks(drc_rpt),
        'power_w': total_power(fin_rpt),
    }

    if args.format == 'json':
        print(json.dumps(row))
    else:
        order = ['grt_tns', 'grt_wns', 'drt_tns', 'drt_wns', 'rwl_um',
                 'hotspots', 'drc_violations', 'power_w']
        print(','.join('' if row[k] is None else str(row[k]) for k in order))


if __name__ == '__main__':
    main()
