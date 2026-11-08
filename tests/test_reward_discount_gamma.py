import pathlib
import sys
import unittest
from types import SimpleNamespace


SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import resolve_return_gamma


class RewardDiscountGammaTest(unittest.TestCase):
    """``reward_discount_gamma`` decouples the step-reward discount from
    ``gamma``, which is also used as the PLM community resolution parameter.

    This is the un-confounded control group for the ``step + gamma=1``
    experiment: step-granularity rewards should be discountable without
    touching PLM partitioning (``gamma``).
    """

    def test_step_defaults_to_gamma(self):
        args = SimpleNamespace(gamma=0.95, reward_granularity="step")
        self.assertAlmostEqual(resolve_return_gamma(args), 0.95)

    def test_step_uses_reward_discount_gamma_when_present(self):
        args = SimpleNamespace(
            gamma=0.95,
            reward_granularity="step",
            reward_discount_gamma=1.0,
        )
        self.assertAlmostEqual(resolve_return_gamma(args), 1.0)

    def test_global_uses_global_reward_gamma_and_ignores_step_discount(self):
        args = SimpleNamespace(
            gamma=0.95,
            reward_granularity="global",
            global_reward_gamma=1.0,
            reward_discount_gamma=0.9,
        )
        self.assertAlmostEqual(resolve_return_gamma(args), 1.0)

    def test_global_defaults_to_one(self):
        args = SimpleNamespace(gamma=0.95, reward_granularity="global")
        self.assertAlmostEqual(resolve_return_gamma(args), 1.0)

    def test_default_granularity_is_step(self):
        args = SimpleNamespace(gamma=0.95)
        self.assertAlmostEqual(resolve_return_gamma(args), 0.95)

    def test_gamma_is_not_read_for_step_discount_when_override_present(self):
        # The whole point: ``gamma`` may stay at 0.95 (PLM resolution) while the
        # step discount is 1.0, without any coupling between the two.
        args = SimpleNamespace(
            gamma=0.95,
            reward_granularity="step",
            reward_discount_gamma=1.0,
        )
        self.assertEqual(args.gamma, 0.95)
        self.assertNotEqual(resolve_return_gamma(args), args.gamma)


if __name__ == "__main__":
    unittest.main()
