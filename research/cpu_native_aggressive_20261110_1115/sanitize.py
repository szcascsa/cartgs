"""独立ASan+UBSan扩展检查，原生正式二进制和冻结源码不变。"""
from pathlib import Path
import os,sys,faulthandler
faulthandler.enable()
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','');os.environ['MAX_JOBS']='1'
from torch.utils.cpp_extension import load
print('SANITIZER_IMPORTED_CPP_EXTENSION',flush=True)
kind=os.environ.get('CPU_SANITIZER_KIND','address,undefined');assert kind in ['address,undefined','undefined']
label='asan' if kind=='address,undefined' else 'ubsan';build=HERE/('build_sanitize' if label=='asan' else 'build_ubsan');build.mkdir(exist_ok=True)
source=(HERE/'source/cpu_native.cpp').read_text().replace('CpuPipelineV4','CpuPipelineSanitizedV4'+label)
path=build/'cpu_native_sanitize.cpp'
if not path.exists() or path.read_text()!=source:path.write_text(source)
module=load(name='mycartgs_cpu_aggressive_'+label+'_v4',sources=[str(path)],
    extra_cflags=['-O1','-g1','-mavx2','-mfma','-fno-omit-frame-pointer','-fsanitize='+kind,'-pthread'],
    extra_ldflags=['-fsanitize='+kind,'-pthread'],build_directory=str(build),verbose=True)
print('SANITIZER_LOADED_NATIVE_MODULE',flush=True)
faulthandler.dump_traceback_later(20,exit=True)
import cpu_runtime
cpu_runtime.extension=lambda:module
print('SANITIZER_IMPORTING_BOUNDARIES',flush=True)
import boundaries
print('SANITIZER_RUNNING_BOUNDARIES',flush=True)
if os.environ.get('CPU_SANITIZER_VALID_ONLY')=='1':
    content=(HERE/'boundaries.py').read_text();start=content.index("    for value in [float('nan')");end=content.index('    after,_,_=',start)
    content=content[:start]+content[end:]
    namespace=dict(__name__='sanitized_valid_boundaries',__file__=str(HERE/'boundaries.py'))
    exec(compile(content,str(HERE/'boundaries.py'),'exec'),namespace);namespace['main']()
    print('SANITIZER_VALID_ONLY_SKIPPED_EXCEPTION_CASES',5,flush=True)
else:boundaries.main()
faulthandler.cancel_dump_traceback_later()
