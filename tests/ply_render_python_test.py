"""验证独立 PLY 的属性往返、补偿烘焙和实际评估加载路径。"""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from gaussian_model import GaussianModel
from transform_field import transformed_attributes


def point_cloud():
    model = GaussianModel(3)
    model._xyz = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)
    model._features_dc = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 1, 3))
    model._features_rest = nn.Parameter(torch.arange(180, dtype=torch.float32).reshape(4, 15, 3) / 100)
    model._opacity = nn.Parameter(torch.tensor([[-50.], [50.], [0.], [1.]]))
    model._scaling = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10 - 3)
    model._rotation = nn.Parameter(torch.tensor([[2., .1, .2, .3]] * 4))
    model.active_sh_degree = 3
    return model


class Field:
    grid = None
    maturity_compute_epsilon = 0.
    entry_maturity = 0.

    def __init__(self):
        self.grid = self

    def inside_mask(self, xyz):
        return xyz[:, 0] < .8

    def __call__(self, xyz, ratio):
        delta_rotation = xyz.new_zeros(len(xyz), 4)
        delta_rotation[:, 1] = .1
        return torch.full_like(xyz, .03), torch.full_like(xyz, .2), delta_rotation


class PlyRenderTest(unittest.TestCase):
    def test_binary_ply_roundtrip_full_selected_compensated_and_empty(self):
        model = point_cloud()
        mask = torch.tensor([1., 0., 1., 1.])
        attributes = transformed_attributes(Field(), model, mask, torch.tensor([1., 1., 0., .5]), .05)
        raw_names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
        before = {name: getattr(model, name).detach().clone() for name in raw_names}
        with tempfile.TemporaryDirectory() as temporary:
            for selection, transformed in [(None, None), (mask, None), (mask, attributes), (torch.zeros(4), None)]:
                path = Path(temporary) / "point_cloud.ply"
                count = model.save_ply(path, selection, transformed)
                saved = PlyData.read(path, mmap=False)
                self.assertFalse(saved.text)
                self.assertEqual(saved["vertex"].count, count)
                selected = torch.ones(4, dtype=torch.bool) if selection is None else selection.bool()
                loaded = GaussianModel(3)
                loaded.load_ply(path, device="cpu")
                self.assertEqual(loaded.get_xyz.shape[0], int(selected.sum()))
                expected = {
                    "get_xyz": model.get_xyz if transformed is None else transformed["xyz"],
                    "get_scaling": model.get_scaling if transformed is None else transformed["scaling"],
                    "get_rotation": model.get_rotation if transformed is None else transformed["rotation"],
                    "_opacity": model._opacity,
                    "_features_dc": model._features_dc,
                    "_features_rest": model._features_rest,
                }
                for name, value in expected.items():
                    self.assertTrue(torch.allclose(getattr(loaded, name), value[selected], atol=1e-7, rtol=1e-6), name)
                for name in raw_names:
                    self.assertTrue(torch.equal(getattr(model, name), before[name]), name)

    def test_evaluation_exports_reloads_then_supports_standalone_ply(self):
        path = ROOT / "python/run.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        main = next(node for node in module.body if isinstance(node, ast.If))
        start = next(index for index, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                     and [item.id for item in node.targets[0].elts] == ["width", "height", "fovx", "fovy"])
        end = next(index for index, node in enumerate(main.body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "render_data")
        program = compile(ast.Module(body=main.body[start:end + 1], type_ignores=[]), str(path), "exec")
        model = point_cloud()
        mask = torch.tensor([1., 0., 1., 1.])
        attrs = transformed_attributes(Field(), model, mask, torch.ones(4), .05)
        original_load = GaussianModel.load_ply
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = source / "12_shutdown" / "ply"
            ply = checkpoint / "point_cloud" / "iteration_12" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            model.save_ply(ply)
            (checkpoint / "cameras.json").write_text(json.dumps([dict(width=64, height=48, fx=50, fy=50)]))
            output = Path(temporary) / "ratio_0p05"
            output.mkdir()
            prepared = Mock(return_value={})
            selection = Mock(return_value=(mask, dict(total_gaussians=4, active_gaussians=3)))
            transform = Mock(return_value=attrs)
            namespace = dict(
                os=os, json=json, GaussianModel=GaussianModel, gaussians=GaussianModel(3), sh_degree=3,
                shutdown_dir="12_shutdown", shutdown_ply_dir=str(checkpoint), source_result_path=str(source),
                output_result_path=str(output), args=SimpleNamespace(ply=None, selector="selector.pt", selector_ratio=.05),
                selector_model=object(), transform_model=object(), selector_config={}, selector_config_path="config.yaml",
                selector_temperature=1., selector_protection_enabled=True,
                load_selector_metadata=lambda *args: {"maturity": torch.ones(4)},
                build_selector_mask=selection, transformed_attributes=transform, prepare_render_data=prepared,
                focal2fov=lambda *args: 1.,
            )
            with patch.object(GaussianModel, "load_ply", lambda self, path: original_load(self, path, device="cpu")):
                exec(program, namespace)
                selection.assert_called_once()
                transform.assert_called_once()
                exported = output / "point_cloud.ply"
                manifest = json.loads((output / "ply_manifest.json").read_text())
                self.assertEqual(manifest["active_gaussians"], 3)
                self.assertEqual(manifest["source_ply"], str(ply))
                self.assertTrue(manifest["transform_applied"])
                loaded = prepared.call_args[0][0]
                self.assertEqual(loaded.get_xyz.shape[0], 3)
                self.assertTrue(torch.allclose(loaded.get_xyz, attrs["xyz"][mask.bool()]))
                self.assertEqual(namespace["selector_stats"]["rasterizer_mode"], "inference")
                namespace.update(args=SimpleNamespace(ply=str(exported)), selector_model=None, transform_model=None)
                exec(program, namespace)
                selection.assert_called_once()
                transform.assert_called_once()
                self.assertEqual(namespace["selector_stats"]["render_gaussians"], 3)
                self.assertEqual(namespace["selector_stats"]["ply_path"], str(exported))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 和重新编译的 rasterizer")
    def test_cuda_export_reload_preserves_rendered_pixels(self):
        import numpy as np
        from renderer import prepare_render_data, render, render_prepared
        from utils import MiniCam

        model = point_cloud()
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            setattr(model, name, nn.Parameter(getattr(model, name).detach().cuda()))
        mask = torch.tensor([1., 0., 1., 1.], device="cuda")
        attrs = transformed_attributes(Field(), model, mask, torch.ones(4, device="cuda"), .05)
        background = torch.zeros(3, device="cuda")
        with tempfile.TemporaryDirectory() as temporary:
            for selection, attributes in [(None, None), (mask, None), (mask, attrs), (torch.zeros_like(mask), None)]:
                ply = Path(temporary) / "point_cloud.ply"
                model.save_ply(ply, selection, attributes)
                loaded = GaussianModel(3)
                loaded.load_ply(ply)
                data = prepare_render_data(loaded)
                for offset in (0., .1):
                    w2c = np.eye(4)
                    w2c[0, 3] = offset
                    camera = MiniCam(64, 48, 1., 1., w2c)
                    with torch.no_grad():
                        expected = render(camera, model, background, selector_mask=selection,
                                          elastic_attributes=attributes)["render"]
                    actual = render_prepared(camera, data, background, measure_time=True)
                    self.assertTrue(torch.allclose(actual["render"], expected, atol=1e-5, rtol=1e-4))
                    self.assertGreater(actual["rendering_ms"], 0)


if __name__ == "__main__":
    unittest.main()
