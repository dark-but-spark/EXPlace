import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from model.actor import Actor
from state_parsing import StateParsing


def make_parser(policy_mask_fp32):
    return StateParsing(SimpleNamespace(
        grid=4,
        used_masks=["a", "b", "pos"],
        prototype_flag=False,
        trade_off_coeff=[0.25, 0.5],
        policy_mask_fp32=policy_mask_fp32,
    ))


class PolicyMaskTest(unittest.TestCase):
    def test_fp32_decoder_matches_reference_and_avoids_float64(self):
        # [place_idx, canvas, three current masks, three look-ahead masks, sizes]
        state = torch.arange(1 + 4 * 4 + 3 * 2 * 4 * 4 + 2, dtype=torch.float32)
        state = state.view(1, -1)
        fp32 = make_parser(True).state2soft_mask(state)
        reference = make_parser(False).state2soft_mask(state)

        self.assertEqual(fp32.dtype, torch.float32)
        self.assertEqual(reference.dtype, torch.float64)
        np.testing.assert_allclose(fp32.numpy(), reference.numpy(), rtol=1e-6, atol=1e-6)

    def test_fp32_decoder_broadcasts_across_batch(self):
        parser = make_parser(True)
        state = torch.ones((3, 1 + 4 * 4 + 3 * 2 * 4 * 4 + 2), dtype=torch.float32)
        output = parser.state2soft_mask(state)
        self.assertEqual(output.shape, (3, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_weight_updates_invalidate_decoder_cache(self):
        parser = make_parser(True)
        state = torch.ones((1, 1 + 4 * 4 + 3 * 2 * 4 * 4 + 2), dtype=torch.float32)
        parser.state2soft_mask(state)
        self.assertEqual(len(parser._reward_coeff_cache), 1)
        parser.set_reward_weights([0.75, 0.25])
        self.assertEqual(parser._reward_coeff_cache, {})

    def test_fp16_logits_are_masked_in_fp32_without_overflow(self):
        actor = Actor.__new__(Actor)
        torch.nn.Module.__init__(actor)
        actor.args = SimpleNamespace(soft_coefficient=1.0, policy_mask_fp32=True)
        actor.grid = 2
        actor.softmax = torch.nn.Softmax(dim=-1)

        class Masks:
            @staticmethod
            def state2position_mask(state):
                return torch.tensor([[[0, 1], [0, 0]]], device=state.device)

            @staticmethod
            def state2soft_mask(state):
                return torch.zeros((1, 4), dtype=torch.float32, device=state.device)

        actor.state_parsing = Masks()
        logits = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
        probabilities = actor._decode_action_probs(logits, torch.zeros((1, 1)))

        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertEqual(probabilities[0, 1].item(), 0.0)
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
