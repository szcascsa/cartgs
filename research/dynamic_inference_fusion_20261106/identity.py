from common import *
import subprocess,platform
s=scene();engine=FrozenResidualInference(s.model,s.gaussians)
paths={n:sha(ROOT/'source/python'/n) for n in ['anchor_budget.py','anchor_visibility.py','budget_viewer.py','frozen_inference_cuda.py','frozen_inference_bindings.cpp','frozen_inference_kernels.cu']}
binary=Path(engine.ext.__file__).resolve();assert binary==ROOT/'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so'
old=json.loads(Path('/home/prmi/zwj/experiments/joint_view_inference_20261106/reports/BINARY_IDENTITY.json').read_text());assert sha(old['path'])==old['sha256']
maps=Path('/proc/self/maps').read_text();assert str(binary) in maps;(ROOT/'reports/RUNTIME_MAPS.txt').write_text(maps)
save(ROOT/'reports/BINARY_IDENTITY.json',dict(path=str(binary),sha256=sha(binary),baseline_binary=dict(path=old['path'],sha256=old['sha256']),source_hashes=paths,torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),device=torch.cuda.get_device_name(0),
 tf32=torch.backends.cuda.matmul.allow_tf32,build_ninja=(binary.parent/'build.ninja').read_text(),gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv'],text=True),
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))
