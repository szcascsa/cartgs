"""只使用官方Torch负尺寸异常，区分ASan运行时问题与本轮原生实现。"""
import faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(8,exit=True)
import torch
print('ASAN_TORCH_EXCEPTION_BEFORE',flush=True)
try:torch.empty((-1,))
except RuntimeError as exc:print('ASAN_TORCH_EXCEPTION_CAUGHT',type(exc).__name__,flush=True)
else:raise AssertionError('Negative dimension accepted')
faulthandler.cancel_dump_traceback_later()
