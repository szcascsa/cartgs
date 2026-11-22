"""复用同源ASan/UBSan二进制，实际覆盖Student32分支和动态K。"""
from pathlib import Path
import importlib.util,os,sys,faulthandler
import torch
faulthandler.enable();sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'cpu_native_midratio_20261110_1407';sys.path.insert(0,str(BASE))
label=os.environ.get('CPU_SANITIZER_LABEL','asan');assert label in ['asan','ubsan']
original=(BASE/'source/cpu_mid.cpp').read_text();source=(BASE/('build_'+label)/'cpu_sanitized.cpp').read_text()
source=source.replace('CpuPipelineMidSanitized'+label,'CpuPipelineMid').replace('NativeHandoffSanitized'+label,'NativeHandoff').replace('NativeTicketSanitized'+label,'NativeTicket')
assert source==original,'Sanitizer source does not match the frozen backend'
spec=importlib.util.spec_from_file_location('cpu_mid_'+label,BASE/('build_'+label)/('cpu_mid_'+label+'.so'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
import mid_runtime
mid_runtime.extension=lambda:module
sys.path.insert(0,str(HERE))
import boundaries
assert Path(boundaries.__file__).resolve()==HERE/'boundaries.py'
faulthandler.dump_traceback_later(35,exit=True)
if label=='asan':
    code=boundaries.code;start=code.index("    for value in [float('nan')");end=code.index('    after,_,_=',start);code=code[:start]+code[end:]
    namespace=dict(__name__='student_asan_boundaries');exec(compile(code,str(HERE/'boundaries.py'),'exec'),namespace);namespace['main']()
    print('STUDENT_ASAN_28_NORMAL_PATHS_PASS_5_EXCEPTIONS_EXCLUDED',flush=True)
else:boundaries.namespace['main']();print('STUDENT_UBSAN_33_PATHS_PASS',flush=True)
faulthandler.cancel_dump_traceback_later()
