"""记录Python进程真实加载的CUDA/LibTorch路径，避免裸ldd误判。"""
from benchmark import *
import ctypes,hashlib,importlib.util


def main():
    current=SparseRenderer(4)
    spec=importlib.util.spec_from_file_location('identity_old_runtime',Path('/home/prmi/zwj/experiments/sparse_rasterizer_20261109/runtime.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);previous=old.SparseRenderer(2)
    torch.empty(1,device='cuda');torch.cuda.synchronize()
    paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if '/' in line and any(key in line for key in ['libcudart','libc10','libtorch','sparse_rasterizer','diff_gaussian_rasterization/_C'])})
    versions={}
    for path in paths:
        if 'libcudart' in path:
            value=ctypes.c_int();code=ctypes.CDLL(path).cudaRuntimeGetVersion(ctypes.byref(value))
            versions[path]=dict(status=code,version=value.value)
    binaries={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if 'rasterizer' in p or 'rasterization/_C' in p}
    result=dict(python=sys.version,torch=torch.__version__,torch_build_cuda=torch.version.cuda,
        loaded_paths=paths,cuda_runtime_versions=versions,binary_sha256=binaries)
    (HERE/'reports/RUNTIME_IDENTITY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
