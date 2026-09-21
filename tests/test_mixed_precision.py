import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from model.cnn import MyCNN


class MixedPrecisionTest(unittest.TestCase):
    def test_local_cnn_bfloat16_matches_float32_within_tolerance(self):
        args = SimpleNamespace(used_masks=["proxy", "pos"])
        model = MyCNN(args=args).eval()
        inputs = torch.randn(2, 4, 32, 32)
        with torch.no_grad():
            reference = model(inputs)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                mixed = model(inputs)
        max_error = (reference - mixed.float()).abs().max().item()
        self.assertLess(max_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
