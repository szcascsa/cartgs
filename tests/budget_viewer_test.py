"""Viewer协议验证，不依赖显示服务；真实CUDA相机与图像另做接入测试。"""
from pathlib import Path
import json
import sys
import struct
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from budget_viewer import ViewerHandler, ThreadingHTTPServer, validate_pose, native_frame_packet


class BudgetViewerTest(unittest.TestCase):
    def test_native_pixels_roundtrip_bitwise(self):
        image = torch.rand(3, 17, 23)
        reference = torch.rand_like(image)
        for ref in (None, reference):
            packet = native_frame_packet(image, ref, dict(stats=dict(active_gaussians=123)))
            length, = struct.unpack('<I', packet[:4])
            header = json.loads(packet[4:4+length])
            values = np.frombuffer(packet[4+length:], dtype='<f4')
            size = image.numel()
            np.testing.assert_array_equal(values[:size].reshape(17, 23, 3), image.permute(1, 2, 0).numpy())
            self.assertEqual(header['reference_bytes'], 0 if ref is None else size*4)
            if ref is not None:np.testing.assert_array_equal(values[size:].reshape(17, 23, 3), ref.permute(1, 2, 0).numpy())
    def test_rigid_pose_validation(self):
        validate_pose(np.eye(4))
        for p in [np.zeros((4, 4)), np.eye(3), np.eye(4)*2, np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):validate_pose(p)

    def test_http_routes_and_error_status(self):
        def render(payload, native=False):
            if payload.get('invalid'):raise ValueError('invalid request')
            if native:return native_frame_packet(torch.zeros(3, 2, 3), None, dict(stats=dict(active_gaussians=payload['budget'])))
            return dict(stats=dict(active_gaussians=payload['budget']), request_id=payload.get('request_id'))
        server = ThreadingHTTPServer(('127.0.0.1', 0), ViewerHandler)
        server.scene = SimpleNamespace(lock=threading.Lock(), metadata=lambda: dict(frames=[0]), render=render)
        worker = threading.Thread(target=server.serve_forever, daemon=True);worker.start()
        try:
            root = f'http://127.0.0.1:{server.server_port}'
            self.assertTrue(json.load(urlopen(root+'/health'))['ready'])
            self.assertEqual(json.load(urlopen(root+'/api/meta'))['frames'], [0])
            self.assertIn('预算', urlopen(root).read().decode())
            payload = Request(root+'/api/render', json.dumps(dict(budget=123, request_id=7)).encode(),
                              headers={'Content-Type':'application/json'})
            self.assertEqual(json.load(urlopen(payload))['stats']['active_gaussians'], 123)
            native = Request(root+'/api/render-native', b'{"budget":12}', headers={'Content-Type':'application/json'})
            native_body = urlopen(native).read()
            length, = struct.unpack('<I', native_body[:4])
            self.assertEqual(json.loads(native_body[4:4+length])['stats']['active_gaussians'], 12)
            bad = Request(root+'/api/render', b'{"invalid":true}', headers={'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as caught:urlopen(bad)
            self.assertEqual(caught.exception.code, 400)
            with self.assertRaises(HTTPError):urlopen(root+'/../../etc/passwd')
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__ == '__main__':unittest.main()
