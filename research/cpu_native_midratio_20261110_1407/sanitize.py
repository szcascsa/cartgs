"""独立ASan/UBSan构建；复用已定位的系统ASan异常分支限制。"""
from pathlib import Path
import os,sys,faulthandler
faulthandler.enable();sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','');os.environ['MAX_JOBS']='1'
import torch
from torch.utils.cpp_extension import load
kind=os.environ.get('CPU_SANITIZER_KIND','address,undefined');assert kind in ['address,undefined','undefined']
label='asan' if kind=='address,undefined' else 'ubsan';build=HERE/('build_'+label);build.mkdir(exist_ok=True)
source=(HERE/'source/cpu_mid.cpp').read_text().replace('CpuPipelineMid','CpuPipelineMidSanitized'+label)
source=source.replace('NativeHandoff','NativeHandoffSanitized'+label).replace('NativeTicket','NativeTicketSanitized'+label)
header=(HERE/'source/native_handoff.h').read_text().replace('CpuPipelineMid','CpuPipelineMidSanitized'+label).replace('NativeHandoff','NativeHandoffSanitized'+label).replace('NativeTicket','NativeTicketSanitized'+label)
(build/'native_handoff.h').write_text(header);(build/'cpu_sanitized.cpp').write_text(source)
module=load(name='cpu_mid_'+label,sources=[str(build/'cpu_sanitized.cpp')],
    extra_cflags=['-O1','-g1','-mavx2','-mfma','-fno-omit-frame-pointer','-fsanitize='+kind,'-pthread','-I/usr/local/cuda/include'],
    extra_ldflags=['-fsanitize='+kind,'-pthread','-L/usr/local/cuda/lib64','-lcudart','-lc10_cuda','-ltorch_cuda'],build_directory=str(build),verbose=True)
faulthandler.dump_traceback_later(35,exit=True)
import mid_runtime
mid_runtime.extension=lambda:module
import boundaries
if kind=='address,undefined':
    code=boundaries.code;start=code.index("    for value in [float('nan')");end=code.index('    after,_,_=',start)
    code=code[:start]+code[end:];namespace=dict(__name__='sanitized_small_boundaries')
    exec(compile(code,str(HERE/'boundaries.py'),'exec'),namespace);namespace['main']()
    print('ASAN_VALID_PATHS_PASS_EXCEPTION_CASES_EXCLUDED',5,flush=True)
else:boundaries.namespace['main']()
faulthandler.cancel_dump_traceback_later()
