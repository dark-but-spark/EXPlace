import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main


class _RemoteRollout:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def remote(self, *args):
        self.calls += 1
        return self.result


class _FakeActor:
    def state_dict(self):
        return {}


class _FakeAgent:
    def __init__(self):
        self.actor = _FakeActor()
        self.transitions = []

    def store_transition(self, transition):
        self.transitions.append(transition)


class WorkerRolloutProtocolTest(unittest.TestCase):
    def test_driver_uses_one_rpc_per_worker_and_collects_rollout(self):
        results = []
        workers = []
        for index in range(2):
            info = {"scaled_reward": float(index + 1), "macro_pos": {}}
            rollout = [({"place_idx": 0}, 3, -1.0, -0.2, True)]
            remote = _RemoteRollout((rollout, info))
            worker = SimpleNamespace(rollout=remote)
            workers.append(worker)
            results.append(remote)

        original_get = main.ray.get
        main.ray.get = lambda refs: refs
        try:
            agent = _FakeAgent()
            args = SimpleNamespace(
                trade_off_coeff=[1.0],
                corner_flag=False,
                compact_transition=True,
            )
            best, infos = main.run_worker_rollouts(workers, agent, args)
        finally:
            main.ray.get = original_get

        self.assertEqual([remote.calls for remote in results], [1, 1])
        self.assertEqual(len(agent.transitions), 2)
        self.assertEqual(best["scaled_reward"], 2.0)
        self.assertEqual(len(infos), 2)


if __name__ == "__main__":
    unittest.main()
