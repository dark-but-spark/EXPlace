"""Isolated asynchronous DREAMPlace/OpenTimer evaluator."""

import copy

import numpy as np
import ray


@ray.remote(num_gpus=0)
class AsyncPhysicalEvaluator:
    """Own the physical-design state in one serialized Ray actor."""

    def __init__(self, args, benchmark):
        # A worker must never share PlaceDB/OpenTimer state with training.
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
