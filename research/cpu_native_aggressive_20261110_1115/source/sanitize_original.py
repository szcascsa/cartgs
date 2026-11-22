"""独立ASan+UBSan扩展检查，原生正式二进制和冻结源码不变。"""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','');os.environ['MAX_JOBS']='1'
from torch.utils.cpp_extension import load
(HERE/'build_sanitize').mkdir(exist_ok=True)
source=(HERE/'source/cpu_native.cpp').read_text().replace('CpuPipelineV4','CpuPipelineSanitizedV4')
(HERE/'build_sanitize/cpu_native_sanitize.cpp').write_text(source)
module=load(name='mycartgs_cpu_aggressive_asan_v4',sources=[str(HERE/'build_sanitize/cpu_native_sanitize.cpp')],
    extra_cflags=['-O1','-g1','-mavx2','-mfma','-fno-omit-frame-pointer','-fsanitize=address,undefined','-pthread'],
    extra_ldflags=['-fsanitize=address,undefined','-pthread'],build_directory=str(HERE/'build_sanitize'),verbose=True)
import cpu_runtime
cpu_runtime.extension=lambda:module
import boundaries
boundaries.main()
