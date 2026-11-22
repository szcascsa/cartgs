"""定位系统ASan预加载的导入问题，超时保留Python栈并自动退出。"""
import faulthandler,time
faulthandler.enable();faulthandler.dump_traceback_later(20,exit=True)
print('ASAN_PROBE_BEFORE_TORCH',time.time(),flush=True)
import torch
print('ASAN_PROBE_AFTER_TORCH',time.time(),flush=True)
from torch.utils.cpp_extension import load
print('ASAN_PROBE_AFTER_CPP_EXTENSION',time.time(),flush=True)
faulthandler.cancel_dump_traceback_later()
