#!/usr/bin/env python3
"""Generate perturbed macro-placement TCL files from a legal `macro_out`.

Why: RL training converges into a narrow gp_hpwl band, so sampling checkpoints or
configs cannot produce the quality spread a label-fidelity study needs. Jittering
and shuffling an existing legal macro placement does, and needs no training.

Input : a `macro_out` produced by OpenROAD-flow/run_EXPlace.tcl
        lines: place_macro -macro_name {..} -location {x y} -orientation R0
Output: macro_out_j<a>.tcl (jitter), macro_out_shuffle.tcl, manifest.csv

Coordinates are microns. Jitter is clamped into the core area so placements stay
inside the die.
"""

import argparse
import csv
import os
import random

def read_macros(path):
    """Parse place_macro lines by splitting on braces (regex was fragile here)."""
    out = []
    for line in open(path):
        s = line.strip()
        if not s.startswith('place_macro'):
            continue
        parts = s.split('}')
        if len(parts) < 3:
            continue
        name = parts[0].rsplit('{', 1)[-1].strip()
        loc = parts[1].rsplit('{', 1)[-1].split()
        if len(loc) < 2:
            continue
        out.append((name, float(loc[0]), float(loc[1]), parts[2]))
    return out


def emit(macros, path):
    with open(path, 'w') as handle:
        for name, x, y, tail in macros:
            handle.write(f'place_macro -macro_name {{{name}}} -location '
                         f'{{{x:.3f} {y:.3f}}}{tail}\n')


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro-out', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--core', default='5,5,795,595', help='x0,y0,x1,y1 in microns')
    ap.add_argument('--jitters', default='5,20,50')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    x0, y0, x1, y1 = (float(v) for v in args.core.split(','))
    os.makedirs(args.out_dir, exist_ok=True)
    base = read_macros(args.macro_out)
    if not base:
        raise SystemExit(f'no place_macro lines in {args.macro_out}')
    print(f'base macros: {len(base)}')

    manifest = os.path.join(args.out_dir, 'manifest.csv')
    rows = [('anchor', 'tcl', 'n_macros')]
    emit([(n, clamp(x, x0, x1), clamp(y, y0, y1), t) for n, x, y, t in base],
         os.path.join(args.out_dir, 'macro_out_base.tcl'))
    rows.append(('base', 'macro_out_base.tcl', len(base)))

    rng = random.Random(args.seed)
    for amp in (float(v) for v in args.jitters.split(',')):
        moved = [(n, clamp(x + rng.uniform(-amp, amp), x0, x1),
                  clamp(y + rng.uniform(-amp, amp), y0, y1), t) for n, x, y, t in base]
        name = f'macro_out_j{int(amp)}.tcl'
        emit(moved, os.path.join(args.out_dir, name))
        rows.append((f'jitter{int(amp)}', name, len(moved)))

    slots = [(clamp(x, x0, x1), clamp(y, y0, y1)) for _, x, y, _ in base]
    rng.shuffle(slots)
    shuffled = [(n, slots[i][0], slots[i][1], t) for i, (n, _, _, t) in enumerate(base)]
    emit(shuffled, os.path.join(args.out_dir, 'macro_out_shuffle.tcl'))
    rows.append(('shuffle', 'macro_out_shuffle.tcl', len(shuffled)))

    with open(manifest, 'w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    print(f'wrote {len(rows) - 1} variants into {args.out_dir}')


if __name__ == '__main__':
    main()
