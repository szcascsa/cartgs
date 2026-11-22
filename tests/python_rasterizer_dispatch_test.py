"""使用实际 Python wrapper 验证推理入口，保留训练路径及梯度语义。"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch


class RasterizerDispatchTest(unittest.TestCase):
    def setUp(self):
        self.native = ModuleType("rasterizer_dispatch_test._C")
        color = torch.zeros(3, 4, 6)
        radii = torch.ones(2, dtype=torch.int32)
        self.native.rasterize_gaussians = Mock(return_value=(
            1, 1, color, radii, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)))
        self.native.rasterize_gaussians_inference = Mock(return_value=(color, radii))
        path = Path(__file__).resolve().parents[1] / "python/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"
        spec = importlib.util.spec_from_file_location("rasterizer_dispatch_test", path,
                                                     submodule_search_locations=[str(path.parent)])
        self.wrapper = importlib.util.module_from_spec(spec)
        self.modules = patch.dict(sys.modules, {spec.name: self.wrapper, spec.name + "._C": self.native})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec.loader.exec_module(self.wrapper)
        self.settings = self.wrapper.GaussianRasterizationSettings(
            4, 6, 1., 1., torch.zeros(3), 1., torch.eye(4), torch.eye(4), 3, torch.zeros(3), False, False)
        self.inputs = dict(means3D=torch.zeros(2, 3, requires_grad=True), means2D=torch.zeros(2, 3),
                           dc=torch.zeros(2, 1, 3), shs=torch.zeros(2, 15, 3), opacities=torch.ones(2, 1),
                           scales=torch.ones(2, 3), rotations=torch.ones(2, 4))

    def test_inference_calls_dedicated_native_entry_without_autograd(self):
        rasterizer = self.wrapper.GaussianRasterizer(self.settings, inference_only=True)
        result, _ = rasterizer(**self.inputs)
        self.native.rasterize_gaussians.assert_not_called()
        self.native.rasterize_gaussians_inference.assert_called_once()
        self.assertFalse(result.requires_grad)
        args = self.native.rasterize_gaussians_inference.call_args[0]
        self.assertEqual(len(args), 20)
        for index, key in [(1, "means3D"), (3, "opacities"), (4, "scales"),
                           (5, "rotations"), (14, "dc"), (15, "shs")]:
            self.assertIs(args[index], self.inputs[key])

    def test_default_preserves_training_forward_and_autograd(self):
        result, _ = self.wrapper.GaussianRasterizer(self.settings)(**self.inputs)
        self.native.rasterize_gaussians.assert_called_once()
        self.native.rasterize_gaussians_inference.assert_not_called()
        self.assertTrue(result.requires_grad)

    def test_old_binary_fails_explicitly_without_training_fallback(self):
        del self.native.rasterize_gaussians_inference
        with self.assertRaisesRegex(RuntimeError, "Rebuild diff_gaussian_rasterization"):
            self.wrapper.GaussianRasterizer(self.settings, inference_only=True)


if __name__ == "__main__":
    unittest.main()
