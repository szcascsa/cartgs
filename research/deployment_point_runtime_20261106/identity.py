from common import *
from deployment_runtime import DeploymentRenderer
import subprocess


@torch.no_grad()
def main():
    s=scene();runtime=DeploymentRenderer(s.model,s.gaussians,s.config)
    names=['deployment_runtime.py','deployment_runtime_bindings.cpp','deployment_runtime_kernels.cu',
           'deployment_infer.py','DEPLOYMENT_RUNTIME.md']
    binary=Path(runtime._ext.__file__).resolve();baseline=Path(runtime._base.ext.__file__).resolve()
    assert binary==ROOT/'build/mycartgs_deployment_fp32_v1/mycartgs_deployment_fp32_v1.so'
    assert baseline==ROOT/'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so'
    maps=Path('/proc/self/maps').read_text();assert str(binary) in maps and str(baseline) in maps
    (ROOT/'reports/RUNTIME_MAPS.txt').write_text(maps)
    save(ROOT/'reports/BINARY_IDENTITY.json',dict(path=str(binary),sha256=sha(binary),
        baseline=dict(path=str(baseline),sha256=sha(baseline)),
        sources={name:sha(ROOT/'source/python'/name) for name in names},
        build_ninja=(binary.parent/'build.ninja').read_text(),torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),tf32=torch.backends.cuda.matmul.allow_tf32,
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=name,pstate,clocks.current.sm,utilization.gpu,memory.used','--format=csv'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))
    print('DEPLOYMENT_BINARY_VERIFIED',binary,flush=True)


if __name__=='__main__':main()
