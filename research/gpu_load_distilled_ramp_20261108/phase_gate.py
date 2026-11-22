"""人工负载进入渲染阶段，排空确认后才能返回准备阶段。"""
import ctypes
import json
from pathlib import Path


class RenderLoad:
    def __init__(self, library, log):
        self.library = ctypes.CDLL(str(library))
        self.library.render_load_create.argtypes = [ctypes.c_char_p]
        self.library.render_load_create.restype = ctypes.c_void_p
        self.library.render_load_resume.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.library.render_load_resume.restype = ctypes.c_long
        self.library.render_load_pause.argtypes = [ctypes.c_void_p]
        self.library.render_load_pause.restype = ctypes.c_int
        self.library.render_load_destroy.argtypes = [ctypes.c_void_p]
        self.library.render_load_error.restype = ctypes.c_char_p
        self.library.render_load_information.argtypes = [ctypes.c_void_p]
        self.library.render_load_information.restype = ctypes.c_char_p
        self.pointer = self.library.render_load_create(str(log).encode())
        if not self.pointer:
            raise RuntimeError(self.library.render_load_error().decode())
        self.information = json.loads(self.library.render_load_information(self.pointer).decode())

    def resume(self, stage, duty):
        generation = self.library.render_load_resume(self.pointer, stage, duty)
        if generation < 0:
            raise RuntimeError(self.library.render_load_error().decode())
        return generation

    def pause(self):
        if self.library.render_load_pause(self.pointer):
            raise RuntimeError(self.library.render_load_error().decode())

    def close(self):
        if self.pointer:
            self.library.render_load_destroy(self.pointer)
            self.pointer = None


def validate_phase_intervals(frames, packets):
    """每个负载包必须完全位于对应请求的渲染阶段内。"""
    by_generation = {int(row['load_generation']): row for row in frames}
    if len(by_generation) != len(frames):
        raise AssertionError('Duplicate load generation')
    for packet in packets:
        frame = by_generation[int(packet['generation'])]
        start = float(packet['submit_monotonic_s'])
        end = float(packet['drained_monotonic_s'])
        if not float(frame['render_phase_start_s']) <= start <= end <= float(frame['render_phase_end_s']):
            raise AssertionError('Artificial load overlaps a protected phase')
        if float(frame['prepare_end_s']) > start:
            raise AssertionError('Artificial load started before preparation completed')
    return dict(passed=True, requests=len(frames), load_packets=len(packets),
        artificial_load_prepare_overlap=0)
