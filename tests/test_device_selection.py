import pathlib
import os
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main


class DeviceSelectionTest(unittest.TestCase):
    def test_inherited_single_visible_gpu_maps_to_logical_zero(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False), \
             mock.patch.object(main.th.cuda, "is_available", return_value=True):
            self.assertEqual(main.resolve_device(1, True), "cuda:0")

    def test_explicit_gpu_is_not_written_to_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(main.th.cuda, "is_available", return_value=True), \
             mock.patch.object(main.th.cuda, "device_count", return_value=2):
            self.assertEqual(main.resolve_device(1, True), "cuda:1")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_cpu_fallback_does_not_require_gpu_selection(self):
        self.assertEqual(main.resolve_device(1, False), "cpu")


if __name__ == "__main__":
    unittest.main()
