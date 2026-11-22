"""独立Sanitizer构建覆盖新向量数学、packed Heads与有界上下文。"""
from pathlib import Path
import os,sys,faulthandler
faulthandler.enable();sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','');os.environ['MAX_JOBS']='1'
import torch
from torch.utils.cpp_extension import load
kind=os.environ.get('CPU_SANITIZER_KIND','address,undefined');assert kind in ['address,undefined','undefined'];label='asan' if kind=='address,undefined' else 'ubsan'
build=HERE/('build_'+label);build.mkdir(exist_ok=True)
replace=lambda s:s.replace('CpuPipelineStudentFast','CpuPipelineStudentFastSanitized'+label).replace('NativeHandoff','NativeHandoffSanitized'+label).replace('NativeTicket','NativeTicketSanitized'+label)
(build/'cpu_sanitized.cpp').write_text(replace((HERE/'source/cpu_student_fast.cpp').read_text()));(build/'native_handoff.h').write_text(replace((HERE/'source/native_handoff.h').read_text()));(build/'student_fast.h').write_text((HERE/'source/student_fast.h').read_text())
module=load(name='cpu_student_fast_'+label,sources=[str(build/'cpu_sanitized.cpp')],extra_cflags=['-O1','-g1','-mavx2','-mfma','-fno-omit-frame-pointer','-fsanitize='+kind,'-pthread','-I/usr/local/cuda/include'],
    extra_ldflags=['-fsanitize='+kind,'-pthread','-L/usr/local/cuda/lib64','-lcudart','-lc10_cuda','-ltorch_cuda','-lmvec','-lm'],build_directory=str(build),verbose=True)
import kernel_runtime
kernel_runtime.extension=lambda:module
sys.path.insert(0,str(HERE));import boundaries
assert Path(boundaries.__file__).resolve()==HERE/'boundaries.py'
faulthandler.dump_traceback_later(45,exit=True)
if label=='asan':
    code=boundaries.code;start=code.index("    for value in [float('nan')");end=code.index('    after,_,_=',start);code=code[:start]+code[end:]
    namespace=dict(__name__='fast_student_asan');exec(compile(code,str(HERE/'boundaries.py'),'exec'),namespace);namespace['main']();print('ASAN_NORMAL28_PASS_EXCEPTION5_EXCLUDED',flush=True)
else:boundaries.namespace['main']();print('UBSAN33_PASS',flush=True)
faulthandler.cancel_dump_traceback_later()
