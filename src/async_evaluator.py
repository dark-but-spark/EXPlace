"""Isolated asynchronous DREAMPlace/OpenTimer evaluator."""

import copy
import os
import sys

import ray


def _candidate_import_paths():
    paths = []
    module_file = globals().get('__file__')
    if module_file:
        src_dir = os.path.dirname(os.path.abspath(module_file))
        root_dir = os.path.dirname(src_dir)
        paths = [
            src_dir,
            os.path.join(root_dir, 'DREAMPlace', 'install'),
            os.path.join(root_dir, 'DREAMPlace', 'install', 'dreamplace'),
        ]
    return paths


def prepare_worker_import_paths(args):
    """Make repo-local and DREAMPlace modules importable in a Ray worker."""
    paths = list(getattr(args, 'runtime_paths', ()) or ())
    paths.extend(_candidate_import_paths())
    for path in paths:
        if path and path not in sys.path:
            sys.path.insert(0, path)


_WORKER_PYTHONPATH = os.pathsep.join(
    _candidate_import_paths() + [os.environ.get('PYTHONPATH', '')]
).strip(os.pathsep)


@ray.remote(
    num_cpus=0,
    num_gpus=0,
    runtime_env={"env_vars": {"PYTHONPATH": _WORKER_PYTHONPATH}},
)
class AsyncPhysicalEvaluator:
    """Own the physical-design state in one serialized Ray actor."""

    def __init__(self, args, benchmark):
        # A worker must never share PlaceDB/OpenTimer state with training.
        prepare_worker_import_paths(args)
        from problem_instance import ProblemInstance

        eval_args = copy.copy(args)
        eval_args.use_cuda = False
        eval_args.device = "cpu"
        self.problem = ProblemInstance(eval_args, benchmark, init=False)

    def evaluate(self, episode, macro_pos):
        frozen_pos = copy.deepcopy(macro_pos)
        metrics = self.problem.evaluate(frozen_pos)
        return {
            "episode": int(episode),
            "macro_pos": frozen_pos,
            "metrics": tuple(float(value) for value in metrics),
            "valid": bool(self.problem.last_evaluation_valid),
            "failure_reason": self.problem.last_evaluation_failure,
        }

    def save_placement(self, path):
        self.problem.save_placement(path)

    def plot(self, hpwl, path):
        self.problem.plot(float(hpwl), path)
