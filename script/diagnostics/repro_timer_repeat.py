#!/usr/bin/env python3
"""Test the "cumulative timer corruption" hypothesis for the OpenTimer SIGSEGV.

The production crash never reproduces on a single replay: both crashed P12 runs
died on their 8th evaluation and this repro died on its 2nd, always at
`timing_op.timer.time_unit()` -- the first timer touch of the *next* evaluation,
i.e. after the previous `update_timing()`.  That points at repeated
`update_timing()` on the same Timer object rather than at one bad placement.

This script therefore calls the timing operator + update_timing() in a loop on
one fixed position.  If it dies at some iteration, the bug is stateful.

    gdb --batch -ex run -ex 'bt 40' --args /usr/bin/python3 \
        script/repro_timer_repeat.py --benchmark superblue3 \
        --pos <dump.npy> --gpu 7 --iters 20
"""

import argparse
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
    parser.add_argument('--pos', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--reuse-pos', action='store_true', default=True)
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

    say('[repeat] building ProblemInstance (timer constructed here)')
    problem = ProblemInstance(namespace, args.benchmark)
    placer = problem.dmp_placer
    timing_op = placer.op_collections.timing_op
    say('[repeat] ProblemInstance ready; initial update_timing() already ran in __init__')

    pos = np.load(args.pos)
    tensor = th.tensor(pos, dtype=th.float32, device=namespace.device)
    say(f'[repeat] loaded {args.pos} shape={pos.shape}; looping {args.iters} times')

    for i in range(args.iters):
        say(f'[repeat] iter {i}: time_unit()')
        unit = timing_op.timer.time_unit()
        say(f'[repeat] iter {i}: timing_op(pos)')
        timing_op(tensor.clone().cpu())
        say(f'[repeat] iter {i}: update_timing()')
        timing_op.timer.update_timing()
        tns = timing_op.timer.report_tns_elw(split=1) / (unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (unit * 1e15)
        say(f'[repeat] iter {i}: ok tns={tns:.4f} wns={wns:.4f}')

    say('[repeat] completed all iterations without crash')


if __name__ == '__main__':
    main()
