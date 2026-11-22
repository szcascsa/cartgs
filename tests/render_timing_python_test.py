"""验证缓存渲染的输入一致性、计时边界和可选 CUDA 像素一致性。"""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
HAS_RASTERIZER = importlib.util.find_spec("diff_gaussian_rasterization") is not None
if not HAS_RASTERIZER:
    rasterizer_stub = ModuleType("diff_gaussian_rasterization")
    rasterizer_stub.GaussianRasterizationSettings = SimpleNamespace
    rasterizer_stub.GaussianRasterizer = None
    sys.modules["diff_gaussian_rasterization"] = rasterizer_stub

import renderer


class PointCloud:
    active_sh_degree = 3

    def __init__(self, device="cpu"):
        self.get_xyz = torch.tensor(
            [[-.4, -.2, 2.], [.3, -.1, 2.5], [.1, .3, 3.], [-.2, .2, 2.2]],
            device=device, requires_grad=True,
        )
        self.get_scaling = torch.full((4, 3), .1, device=device, requires_grad=True)
        self.get_rotation = torch.tensor([[1., 0., 0., 0.]] * 4, device=device)
        self.get_opacity = torch.full((4, 1), .8, device=device, requires_grad=True)
        self.get_features_dc = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 1, 3) / 12
        self.get_features_rest = torch.zeros(4, 15, 3, device=device)


class RenderTimingTest(unittest.TestCase):
    def setUp(self):
        self.pc = PointCloud()
        self.camera = SimpleNamespace(image_height=4, image_width=6)
        self.background = torch.tensor([.1, .2, .3])

    def test_full_lod_and_transform_inputs_match_legacy_renderer(self):
        zeros_like = torch.zeros_like
        calls = []

        def cpu_zeros_like(value, **kwargs):
            kwargs.pop("device", None)
            return zeros_like(value, **kwargs)

        def capture(**inputs):
            calls.append(inputs)
            return torch.zeros(3, 4, 6), torch.ones(inputs["means3D"].shape[0], dtype=torch.int32)

        mask = torch.tensor([1., 0., 1., 0.])
        transformed = {
            "xyz": self.pc.get_xyz.detach() + .25,
            "scaling": self.pc.get_scaling.detach() * 1.2,
            "rotation": self.pc.get_rotation.roll(1, dims=1),
        }
        for selection, attributes in [(None, None), (mask, None), (mask, transformed)]:
            with self.subTest(lod=selection is not None, transform=attributes is not None):
                with patch.object(renderer, "_make_rasterizer", return_value=capture), \
                        patch.object(renderer.torch, "zeros_like", side_effect=cpu_zeros_like):
                    with torch.no_grad():
                        renderer.render(self.camera, self.pc, self.background,
                                        selector_mask=selection, elastic_attributes=attributes)
                    data = renderer.prepare_render_data(self.pc, selection, attributes)
                    renderer.render_prepared(self.camera, data, self.background)
                legacy, prepared = calls[-2:]
                self.assertEqual(legacy.keys(), prepared.keys())
                for key in legacy:
                    if legacy[key] is None:
                        self.assertIsNone(prepared[key])
                    else:
                        self.assertTrue(torch.equal(legacy[key], prepared[key]), key)
                        self.assertFalse(prepared[key].requires_grad, key)
                        self.assertTrue(prepared[key].is_contiguous(), key)

    def test_cached_inputs_are_reused_and_setup_is_outside_timing(self):
        data = renderer.prepare_render_data(self.pc, torch.tensor([1., 0., 1., 0.]))
        pointers = {key: value.data_ptr() for key, value in data["inputs"].items()
                    if value is not None}
        events = []

        def rasterize(**inputs):
            events.append("rasterize")
            self.assertFalse(torch.is_grad_enabled())
            for key, pointer in pointers.items():
                self.assertEqual(inputs[key].data_ptr(), pointer)
            return torch.zeros(3, 4, 6), torch.ones(2, dtype=torch.int32)

        def setup(*args, **kwargs):
            self.assertTrue(kwargs["inference_only"])
            events.append("setup")
            return rasterize

        times = iter([10., 10.002, 20., 20.004])

        def clock():
            events.append("clock")
            return next(times)

        with patch.object(renderer, "_make_rasterizer", side_effect=setup), \
                patch.object(renderer.torch.cuda, "synchronize", side_effect=lambda device: events.append("sync")), \
                patch.object(renderer.time, "perf_counter", side_effect=clock):
            for expected_ms in (2., 4.):
                result = renderer.render_prepared(self.camera, data, self.background, measure_time=True)
                self.assertAlmostEqual(result["rendering_ms"], expected_ms)
        self.assertEqual(events, ["setup", "sync", "clock", "rasterize", "sync", "clock"] * 2)

    def test_empty_selection_keeps_background_without_rasterizer_work(self):
        data = renderer.prepare_render_data(self.pc, torch.zeros(4))
        with patch.object(renderer, "_make_rasterizer") as factory:
            result = renderer.render_prepared(self.camera, data, self.background)
        factory.return_value.assert_not_called()
        expected = self.background[:, None, None].expand(3, 4, 6)
        self.assertTrue(torch.equal(result["render"], expected))
        self.assertIsNone(result["rendering_ms"])

    def test_evaluation_outputs_use_measured_frames_and_exclude_warmup(self):
        # Execute the actual evaluation loop without CUDA metrics or dataset dependencies.
        path = ROOT / "python/run.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        main = next(node for node in module.body if isinstance(node, ast.If))
        start = next(index for index, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == "tstamp")
        evaluation = ast.Module(body=main.body[start:], type_ignores=[])
        times = iter([1., 3.])

        def render_frame(camera, data, background, measure_time=False):
            return {"render": torch.zeros(3, 4, 6),
                    "rendering_ms": next(times) if measure_time else None}

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = Path(temporary) / "output"
            source.mkdir()
            output.mkdir()
            (source / "TrackingTime.txt").write_text("0.1\n0.2\nsummary\nsummary\nsummary\n")
            cv2 = SimpleNamespace(
                imread=lambda *args: np.zeros((4, 6, 3), dtype=np.uint8),
                cvtColor=lambda image, code: image,
                imwrite=lambda *args: True,
                IMREAD_COLOR=1, COLOR_BGR2RGB=2, COLOR_RGB2BGR=3,
            )
            render_mock = Mock(side_effect=render_frame)
            namespace = dict(
                os=os, np=np, cv2=cv2, json=json,
                args=SimpleNamespace(evaluation_frame_set=None, no_save_images=False),
                torch=SimpleNamespace(clamp=torch.clamp, from_numpy=lambda array:
                                      SimpleNamespace(to=lambda device: torch.from_numpy(array))),
                traj_est=SimpleNamespace(timestamps=[0., 1.]), gt_tstamp=[0., 1.],
                associate_frames=lambda *args: [(3, 0), (7, 1)],
                output_result_path=str(output), source_result_path=str(source),
                trange=lambda count, **kwargs: range(count), poses=[np.eye(4)] * 8,
                MiniCam=lambda *args: self.camera, width=6, height=4, fovx=1., fovy=1.,
                render_prepared=render_mock, render_data={}, background=self.background,
                gt_color_paths=["first.png", "second.png"], selector_stats=None,
                calc_psnr=lambda *args: torch.tensor(20.),
                calc_ssim=lambda *args: torch.tensor(.8),
                calc_lpips=lambda *args: torch.tensor(.2),
            )
            exec(compile(evaluation, str(path), "exec"), namespace)
            self.assertEqual(render_mock.call_count, 12)
            self.assertEqual(sum(call[1].get("measure_time", False)
                                 for call in render_mock.call_args_list), 2)
            recorded = np.loadtxt(output / "render_time.txt")
            np.testing.assert_array_equal(recorded, [[3, 1.], [7, 3.]])
            metrics = dict(line.split(": ", 1) for line in (output / "eval.txt").read_text().splitlines())
            self.assertEqual(float(metrics["rendering ms"]), 2.)
            self.assertEqual(float(metrics["rendering FPS"]), 500.)

    @unittest.skipUnless(torch.cuda.is_available() and HAS_RASTERIZER,
                         "需要 CUDA 和已编译的 diff_gaussian_rasterization")
    def test_cuda_full_lod_transform_and_empty_pixels_match(self):
        from utils import MiniCam

        pc = PointCloud("cuda")
        background = self.background.cuda()
        mask = torch.tensor([1., 0., 1., 0.], device="cuda")
        transformed = {"xyz": pc.get_xyz.detach() + .05,
                       "scaling": pc.get_scaling.detach() * 1.1,
                       "rotation": pc.get_rotation}
        cases = [(None, None), (mask, None), (mask, transformed), (torch.zeros_like(mask), None)]
        for selection, attributes in cases:
            data = renderer.prepare_render_data(pc, selection, attributes)
            for x in (0., .1):
                w2c = np.eye(4)
                w2c[0, 3] = x
                camera = MiniCam(64, 48, math.pi / 2, math.pi / 2, w2c)
                with torch.no_grad():
                    expected = renderer.render(camera, pc, background,
                                               selector_mask=selection, elastic_attributes=attributes)["render"]
                result = renderer.render_prepared(camera, data, background, measure_time=True)
                self.assertTrue(torch.equal(expected, result["render"]))
                self.assertGreater(result["rendering_ms"], 0)


if __name__ == "__main__":
    unittest.main()
