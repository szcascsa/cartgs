"""冻结本轮代码、binary 和数学环境；不读取任意环境变量或凭据。"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import torch

dest=Path(__file__).resolve().parent
root=dest.parents[2]
production=Path('/home/prmi/zwj/mycartgs')
sources=['src/selector_cuda.cpp','src/selector_cuda.cu','include/selector_cuda.h',
 'improvements/selector/gumbel_network.h','src/transform_field_cuda_optimizer.cpp',
 'src/transform_field_cuda_optimizer.cu','src/constrained_adam.cpp','src/gi_teacher.cpp',
 'include/selector_handoff.h','include/elastic_entry.h','python/selector.py','python/elastic_entry.py','python/elastic_budget.py']
binaries=['tests/diagnostics/cuda_selector_revised_20261015/build/network_benchmark',
 'tests/diagnostics/cuda_selector_revised_20261015/build/caller_replay',
 'tests/diagnostics/cuda_selector_revised_20261015/build/selector_parity',
 'tests/diagnostics/cuda_selector_revised_20261015/build/guard_transaction',
 'lib/libgaussian_mapper.so']
sha=lambda p:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
result=dict(torch=torch.__version__,python_torch_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
 cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
 nvcc=subprocess.check_output(['/usr/local/cuda-11.8/bin/nvcc','--version'],text=True),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,uuid','--format=csv,noheader'],text=True),
 sources={name:sha(root/name) for name in sources},
 production_sources={name:sha(production/name) for name in sources},
 binary_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in binaries})
header='improvements/selector/gumbel_network.h'
def cuda_header(path):
    text=path.read_text()
    text=re.sub(r'#ifndef CARTGS_EXPERIMENTAL_SELECTOR_CUDA.*?#endif','',text,flags=re.S)
    text=re.sub(r'//[^\n]*','',text)
    return hashlib.sha256(re.sub(r'\s+','',text).encode()).hexdigest()
result['cuda_enabled_header_sha256']={
    'isolated':cuda_header(root/header),'production':cuda_header(production/header)}
result['header_note']='Raw header differs only in comments and the disabled-CUDA-build error message; CUDA-enabled effective definitions are identical.'
assert all(result['sources'][name]==result['production_sources'][name] for name in sources if name!=header)
assert result['cuda_enabled_header_sha256']['isolated']==result['cuda_enabled_header_sha256']['production']
if (dest/'environment.json').exists():
    previous=json.loads((dest/'environment.json').read_text())
    assert previous['binary_sha256']==result['binary_sha256'], 'Native measurement binary changed'
    result['native_binaries_unchanged']=True
    if not (dest/'environment_before.json').exists():
        (dest/'environment_before.json').write_text(json.dumps(previous,indent=2)+'\n')
(dest/'environment.json').write_text(json.dumps(result,indent=2)+'\n')
print('Source and deployed implementation match; environment captured')
