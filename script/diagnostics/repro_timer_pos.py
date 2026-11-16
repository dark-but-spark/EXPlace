#!/usr/bin/env python3
"""Replay one dumped pre-timing position through DREAMPlace's OpenTimer binding.

Used to reproduce an OpenTimer SIGSEGV that killed a training run, without
re-running the whole RL episode.  Run this under gdb to get a C++ backtrace:

    gdb --batch -ex run -ex 'bt 30' --args \
        /usr/bin/python3 script/repro_timer_pos.py \
        --benchmark superblue3 --config adaptive_global \
        --pos long_runs/<run>/_pre_timing_dumps/pre_timing_pos_007.npy

Exit code 139 (SIGSEGV) reproduces the crash; 0 means that position analyses
fine.  The dumped array is exactly `NonLinearPlace.pos[0]`, i.e. the flat
(x..., y...) vector that `evaluate_timing()` hands to the timing operator.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch as th
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
DREAMPLACE_PARENT_DIR = os.path.join(ROOT_DIR, 'DREAMPlace', 'install')
sys.path.extend([SRC_DIR, DREAMPLACE_PARENT_DIR,
                 os.path.join(DREAMPLACE_PARENT_DIR, 'dreamplace')])


def say(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--config', default='adaptive_global')
    parser.add_argument('--pos', required=True, help='.npy from EXPLACE_DUMP_PRE_TIMING')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.chdir(ROOT_DIR)
    with open(os.path.join(ROOT_DIR, 'config', f'{args.config}.yaml')) as handle:
        config = yaml.safe_load(handle)
    config['benchmark'] = args.benchmark
    config['config'] = args.config
    config['gpu'] = args.gpu
    config['use_cuda'] = th.cuda.is_available()
    config['visualize_prototype'] = False
    config['visualization_interval'] = 0
    config['disable_tqdm'] = True
    config['use_timer_for_evaluation'] = True
    config.setdefault('log_dir', '/tmp')
    namespace = SimpleNamespace(**config)
    namespace.device = f'cuda:{args.gpu}' if config['use_cuda'] else 'cpu'

    from problem_instance import ProblemInstance

    say(f'[repro] building ProblemInstance for {args.benchmark} on {namespace.device}')
    problem = ProblemInstance(namespace, args.benchmark)
    placer = problem.dmp_placer
    say('[repro] ProblemInstance ready (initial update_timing already ran)')

    pos = np.load(args.pos)
    say(f'[repro] loaded {args.pos} shape={pos.shape} dtype={pos.dtype}')
    tensor = th.tensor(pos, dtype=th.float32, device=namespace.device)

    timing_op = placer.op_collections.timing_op
    say('[repro] pushing positions into timing op')
    timing_op(tensor.clone().cpu())
    say('[repro] calling update_timing() -- last line means the crash reproduced')
    timing_op.timer.update_timing()
    unit = timing_op.timer.time_unit()
    tns = timing_op.timer.report_tns_elw(split=1) / (unit * 1e17)
    wns = timing_op.timer.report_wns(split=1) / (unit * 1e15)
    say(f'[repro] no crash: tns={tns} wns={wns}')


if __name__ == '__main__':
    main()
