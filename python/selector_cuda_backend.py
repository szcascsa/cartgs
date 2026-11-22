"""FP32 CUDA Selector；使用当前 Python Torch ABI 编译，不加载 Native LibTorch。"""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[1]
    return load(
        name='cartgs_selector_cuda_fp32',
        sources=[str(root/'python/selector_cuda_bindings.cpp'),
                 str(root/'src/selector_cuda.cpp'), str(root/'src/selector_cuda.cu')],
        extra_include_paths=[str(root)], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--fmad=false', '-gencode=arch=compute_86,code=sm_86'],
        extra_ldflags=['-lcublas', '-lcublasLt'], verbose=False,
    )


def network(inputs, parameters):
    return extension().network(list(inputs), list(parameters))
