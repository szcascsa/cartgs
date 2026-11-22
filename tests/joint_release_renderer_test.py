"""优化默认入口的CPU契约测试；不编译CUDA、不把mock调度测试当图像验收。"""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from joint_release_renderer import DEFAULT_RENDER_MODE, NETWORK_OPTIONS, JointReleaseRenderer, _request
from compensation_student import StudentCompensator, load_student, save_student


class JointReleaseRequestTest(unittest.TestCase):
    def test_strict_budget_boundaries(self):
        for budget in [0, 1, 31, 32, 33, 255, 256, 257, 32768, np.int64(5000)]:
            self.assertEqual(_request(budget, None, False), (int(budget), 0., False))
        for ratio in [0., .01, .1, .3, .5, 1.]:
            self.assertEqual(_request(None, ratio, False), (0, ratio, True))
        self.assertEqual(_request(None, None, True), (0, 0., False))

    def test_invalid_requests_rejected(self):
        for request in [(None, None, False), (1, .1, False), (-1, None, False), (True, None, False),
                (1.2, None, False), (None, True, False), (None, -.1, False), (None, 1.01, False),
                (None, float('nan'), False), (None, float('inf'), False), (1, None, True),
                (None, .1, True), (None, None, 1)]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                _request(*request)

    def test_default_only_182_and_student_required(self):
        self.assertEqual(DEFAULT_RENDER_MODE, 182)
        for mode in [0, 38, 203, 301, 302, True, 182., '182']:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'mode=182'):
                JointReleaseRenderer(None, None, None, student=object(), mode=mode)
        with self.assertRaisesRegex(ValueError, 'frozen Student'):
            JointReleaseRenderer.from_checkpoint('/unused')


class JointReleaseDispatchTest(unittest.TestCase):
    def setUp(self):
        self.renderer = JointReleaseRenderer.__new__(JointReleaseRenderer)
        r = self.renderer
        r.mode, r.backend, r.total, r.rest, r.degree = 182, 'joint_release_182', 10, 45, 3
        r.handle, r._calibration = object(), None
        r.network = SimpleNamespace(_background=torch.zeros(3), _pipeline=Mock(),
            _full=torch.zeros(10), _all_rows=torch.arange(10))
        self.calibration = dict(width=97, height=65, fx=100., fy=100.)
        self.image, self.radii, self.ids = torch.zeros(3, 65, 97), torch.ones(2, dtype=torch.int32), torch.tensor([2, 4])
        r.network._pipeline._render_pose_reused.side_effect = self.local_output
        r.network._pipeline._render_pending_current.side_effect = self.local_output
        r.network._pipeline._render_full_pose.return_value = (self.image, self.radii, self.ids,
            dict(candidate_gaussians=10, active_gaussians=2))
        self.stream = patch('torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=17))
        self.projection = patch.dict(sys.modules, {'utils': SimpleNamespace(getProjectionMatrix=lambda **kw: torch.eye(4))})
        self.stream.start()
        self.projection.start()
        self.addCleanup(self.stream.stop)
        self.addCleanup(self.projection.stop)

    def local_output(self, *args):
        return self.image, self.radii, self.ids, dict(candidate_gaussians=7, active_gaussians=2,
            requested_budget=3, ratio=.5)

    def test_current_pose_and_request_recomputed(self):
        with torch.no_grad():
            result = self.renderer.render_pose(np.eye(4), self.calibration, ratio_budget=.1)
            other = np.eye(4)
            other[0, 3] = 2.
            self.renderer.render_pose(other, self.calibration, budget=33)
        calls = self.renderer.network._pipeline._render_pose_reused.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0].args[2], np.eye(4))
        np.testing.assert_array_equal(calls[1].args[2], other)
        self.assertEqual(calls[0].args[7:10], (0, .1, True))
        self.assertEqual(calls[1].args[7:10], (33, 0., False))
        self.assertEqual(result['stats']['rasterizer_mode'], 182)
        self.assertEqual(result['stats']['budget_shortfall'], 1)
        self.assertFalse(result['stats']['cache_hit'])

    def test_full_uses_native_full_pose(self):
        with torch.no_grad():
            output = self.renderer.render_pose(np.eye(4), self.calibration, full=True)
        self.renderer.network._pipeline._render_full_pose.assert_called_once()
        self.renderer.network._pipeline._render_pose_reused.assert_not_called()
        self.assertEqual(output['stats']['selection'], 'full')
        self.assertEqual(output['stats']['budget_shortfall'], 0)

    def test_gradient_enabled_rejected(self):
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, 'frozen inference only'):
            self.renderer.render_pose(np.eye(4), self.calibration, budget=1)
        self.renderer.network._pipeline._render_pose_reused.assert_not_called()

    def test_pose_and_calibration_rejected_before_native(self):
        with torch.no_grad():
            for pose in [np.zeros((3, 3)), np.full((4, 4), np.nan)]:
                with self.assertRaises(ValueError):
                    self.renderer.render_pose(pose, self.calibration, budget=1)
            for change in [dict(width=0), dict(height=1.5), dict(fx=False), dict(fy=float('inf'))]:
                with self.assertRaises(ValueError):
                    self.renderer.render_pose(np.eye(4), {**self.calibration, **change}, budget=1)
        self.renderer.network._pipeline._render_pose_reused.assert_not_called()

    def test_gpu_camera_routes_same_request(self):
        camera = SimpleNamespace(world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3), image_width=97, image_height=65, znear=.01, FoVx=1.2, FoVy=1.)
        with torch.no_grad():
            self.renderer.render(camera, budget=0)
        call = self.renderer.network._pipeline._render_pending_current.call_args.args
        self.assertEqual(call[8:11], (0, 0., False))

    def test_native_options_match_validated_release(self):
        network = SimpleNamespace(_base=SimpleNamespace(joint_schema=2, base_master=[None] * 5 + [torch.zeros(450)],
            pc=SimpleNamespace(active_sh_degree=3)), total=10, knots=[.01,.05,.1,.3,.5,.7],
            _full=torch.zeros(10), _pipeline=Mock())
        extension = Mock()
        with patch('joint_release_native._extension.extension', return_value=extension):
            self.renderer._initialize(network)
        extension.PendingRasterizer.assert_called_once_with(512, True, True, 15, 200, 64)
        network._pipeline._small_radix_finish.assert_called_once_with(True)
        network._pipeline._direct_pose.assert_called_once_with(True)
        self.assertEqual(NETWORK_OPTIONS['tensor_compensator'], 11)


class JointReleaseArtifactTest(unittest.TestCase):
    def test_native_source_hashes_and_includes(self):
        root = ROOT / 'python/joint_release_native'
        provenance = json.loads((root / 'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['default_mode'], 182)
        self.assertEqual(len(provenance['native_files']), 67)
        for name, sha in provenance['native_files'].items():
            path = root / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha, name)
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(encoding='utf-8'), re.M):
                if include in ('cuda_runtime.h', 'device_launch_parameters.h', 'stdio.h'):
                    continue
                self.assertTrue(any((directory / include).is_file() for directory in
                    [path.parent, root, root / 'raster/reference/cuda_rasterizer']), (name, include))
        self.assertNotIn('research/', (root / '_extension.py').read_text(encoding='utf-8'))

    def test_matching_student_and_wrong_fingerprint(self):
        student = StudentCompensator(True)
        student.teacher_fingerprint = 'a' * 64
        self.assertEqual(sum(p.numel() for p in student.parameters()), 3534)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'student.pt'
            save_student(path, student, metadata={})
            loaded, _ = load_student(path, expected_fingerprint='a' * 64)
            self.assertEqual(loaded.teacher_fingerprint, 'a' * 64)
            with self.assertRaisesRegex(ValueError, 'different teacher'):
                load_student(path, expected_fingerprint='b' * 64)

    def test_cli_uses_joint182_and_cpu_pose(self):
        source = (ROOT / 'python/deployment_infer.py').read_text(encoding='utf-8')
        self.assertIn('JointReleaseRenderer.from_checkpoint', source)
        self.assertIn('runtime.render_pose', source)
        self.assertNotIn('MiniCam(', source)


if __name__ == '__main__':
    unittest.main()
