"""冻结Student同输入Rasterizer；透射率早停独立开关，不改变输入K或属性。"""
from functools import lru_cache
import importlib.util
import os
from student_study import Renderer, HERE

MODES = {182: 'control182', 281: 'encoded_half2'}


@lru_cache(maxsize=1)
def extension():
    name = 'mycartgs_student_encoded_half2_20261115_v1'
    root, build = HERE / 'source_encoded_half2', HERE / 'build_encoded_half2'
    binary = build / (name + '.so')
    if binary.exists():
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    os.environ['MAX_JOBS'] = '2'
    os.environ['PATH'] = '/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    return load(name=name, sources=[str(root / 'graph_main.cu'), str(root / 'graph_forward.cu')],
        extra_include_paths=[str(root / 'reference/cuda_rasterizer')], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-lineinfo'], build_directory=str(build), verbose=True)


class EncodedHalf2Renderer(Renderer):
    def __init__(self, mode=281):
        assert mode in MODES
        self.mode = mode
        self.native = extension().FootprintRasterizer(512, True, True, 10, 18 + mode, 64)


if __name__ == '__main__':
    print('ENCODED_HALF2_BUILD_PASS', extension().__file__, flush=True)
