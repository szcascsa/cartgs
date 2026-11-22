"""训练后推理的原生PI/PID预算控制；不修改训练或固定预算入口。"""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=2)
def pid_extension(build_directory=None):
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    return load(name='mycartgs_latency_pid_v1', sources=[str(root/'latency_pid_bindings.cpp')],
        extra_cflags=['-O3'], with_cuda=True, build_directory=build_directory, verbose=False)


class LatencyPIDRenderer:
    """每实例串行、单CUDA Stream；只调整请求K，最终K由原GPU候选逻辑裁剪。

    renderer须提供render(camera, packet, background)，返回(image, radii)。
    自动视角突变检测使用调用方已有的CPU camera-to-world pose，禁止为此下载GPU矩阵。
    模型、设备、分辨率或渲染器变更时应新建实例；普通移动每帧重新准备。
    """
    def __init__(self, engine, renderer, background, *, config=None, event_slots=8, build_directory=None):
        if background.device.type != 'cuda':
            raise ValueError('PID rendering requires a CUDA background and frozen inference engine')
        self.engine, self.renderer, self.background = engine, renderer, background
        self.controller = pid_extension(build_directory).AsyncPID(dict(config or {}), background.device.index, event_slots)
        self._size = None

    def render(self, camera, *, camera_to_world=None, reset=False, timestamp_s=-1.):
        if torch.is_grad_enabled():
            raise RuntimeError('LatencyPIDRenderer is inference-only; use torch.inference_mode()')
        if isinstance(camera_to_world, torch.Tensor):
            raise ValueError('Pass an existing host pose, not a Tensor that could synchronize the GPU')
        pose = [] if camera_to_world is None else [float(x) for row in camera_to_world for x in row]
        size = (int(camera.image_width), int(camera.image_height))
        if self._size is not None and size != self._size:
            raise ValueError('Create a new PID renderer when the resolution changes')
        self._size = size
        budget = self.controller.request(pose, bool(reset), float(timestamp_s))
        packet, ids, stats = self.engine.prepare(camera, budget=budget)
        token = self.controller.start(len(ids), stats['eligible_candidates'])
        try:
            image, radii = self.renderer.render(camera, packet, self.background)
        except BaseException:
            self.controller.finish(token, False)
            raise
        self.controller.finish(token, True)
        return dict(render=image, radii=radii, ids=ids, stats=stats, pid_requested_budget=budget)

    def set_target(self, target_ms):
        self.controller.set_target(float(target_ms))

    def poll(self):
        """只读取已完成的计时，不等待GPU。"""
        self.controller.poll()
        return self.controller.stats()

    def close(self):
        self.controller.close()
