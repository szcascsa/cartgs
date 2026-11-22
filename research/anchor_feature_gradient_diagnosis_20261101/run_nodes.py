from common import *
import shutil,json

for name in ['fixture.pt','fixture.json']:shutil.copy2(PREV/name,R/name)
checkpoint=str(Path(json.loads((R/'fixture.json').read_text())['checkpoint'])/'selector.pt')
base=json.loads((PREV/'compile_inference_gate.json').read_text())['command'];source_index=base.index(str(PREV/'inference_gate.cpp'))
flags=base[:source_index]+['-I'+str(R)]
run(flags+['-c',str(R/'diagnostic_pack.cpp'),'-o',str(R/'diagnostic_pack.o')],'compile_candidate_cpp')
T=P/'third_party/libtorch-cu118/libtorch'
run(['/usr/local/cuda-11.8/bin/nvcc','-ccbin','/usr/bin/g++-9','-std=c++17','-O3','--fmad=false','-Xcompiler=-fPIC',
 '-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(R),'-I'+str(T/'include'),'-I'+str(T/'include/torch/csrc/api/include'),
 '-gencode=arch=compute_86,code=sm_86','-c',str(R/'diagnostic_pack.cu'),'-o',str(R/'diagnostic_pack_cuda.o')],'compile_candidate_cuda')
compile_probe('node_probe')
save('runtime.json',dict(source=str(S),checkpoint=checkpoint,checkpoint_sha=sha(checkpoint),candidate={n:sha(R/n) for n in ['diagnostic_pack.cpp','diagnostic_pack.cu','diagnostic_pack.h']},
    production_default_unchanged=True,determinism_is_diagnostic_only=True,production_adam_lr=.001,production_adam_eps=1e-15))
run([str(R/'node_probe'),str(R),checkpoint],'node_probe')
save('nodes_complete.json',dict(passed=True))
print('NODES_COMPLETE',flush=True)
