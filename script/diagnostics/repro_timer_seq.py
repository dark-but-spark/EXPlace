#!/usr/bin/env python3
"""Replay a *sequence* of dumped pre-timing positions, then stop.

Single-position replays cannot reproduce a crash that needs several
`update_timing()` calls on different placements in one process (the timer is
built once in `ProblemInstance.__init__` and reused for every evaluation).

Point it at a dump directory; files are replayed in filename order.

    PYTHONPATH=<install_dbg> gdb --batch -ex run -ex 'bt 40' --args \
        /usr/bin/python3 script/repro_timer_seq.py --benchmark superblue3 \
        --dump-dir long_runs/<run>/dumps_fixed3 --gpu 7
"""

import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch as th
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([
    os.path.join(ROOT_DIR, 'src'),
    os.path.join(ROOT_DIR, 'DREAMPlace', 'install'),
    os.path.join(ROOT_DIR, 'DREAMPlace', 'install', 'dreamplace'),
])


def say(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='superblue3')
    parser.add_argument('--config', default='adaptive_global')
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()

    os.chdir(ROOT_DIR)
    with open(os.path.join(ROOT_DIR, 'config', f'{args.config}.yaml')) as handle:
        config = yaml.safe_load(handle)
    config.update(benchmark=args.benchmark, config=args.config, gpu=args.gpu,
                  use_cuda=th.cuda.is_available(), visualize_prototype=False,
                  visualization_interval=0, disable_tqdm=True,
                  use_timer_for_evaluation=True, log_dir='/tmp')
    namespace = SimpleNamespace(**config)
    namespace.device = f'cuda:{args.gpu}' if config['use_cuda'] else 'cpu'

    from problem_instance import ProblemInstance

    say('[seq] building ProblemInstance (timer built once, reused)')
    problem = ProblemInstance(namespace, args.benchmark)
    placer = problem.dmp_placer
    timing_op = placer.op_collections.timing_op
    say('[seq] ready')

    files = sorted(glob.glob(os.path.join(args.dump_dir, 'pre_timing_pos_*.npy')))
    if not files:
        raise SystemExit(f'no dumps in {args.dump_dir}')
    say(f'[seq] replaying {len(files)} positions in order')

    for index, path in enumerate(files):
        array = np.load(path)
        tensor = th.tensor(array, dtype=th.float32, device=namespace.device)
        say(f'[seq] {index}/{len(files) - 1} {os.path.basename(path)} time_unit()')
        unit = timing_op.timer.time_unit()
        say(f'[seq] {index} timing_op(pos)')
        timing_op(tensor.clone().cpu())
        say(f'[seq] {index} update_timing()  <- crash here means stateful bug')
        timing_op.timer.update_timing()
        tns = timing_op.timer.report_tns_elw(split=1) / (unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (unit * 1e15)
        say(f'[seq] {index} ok tns={tns:.4f} wns={wns:.4f}')

    say('[seq] all positions replayed without crash')


if __name__ == '__main__':
    main()
