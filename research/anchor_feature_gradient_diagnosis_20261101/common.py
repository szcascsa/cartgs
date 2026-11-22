from pathlib import Path
import os,json,time,subprocess,hashlib
R=Path(__file__).resolve().parent
PREV=R.parent/'residual48_cuda_optimization_20261101_1937'
S=PREV/'source';B=S/'build';P=Path('/home/prmi/zwj/mycartgs')
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],OMP_NUM_THREADS='2',
         CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
env['LD_LIBRARY_PATH']=':'.join(str(x) for x in [S/'lib',P/'third_party/libtorch-cu118/libtorch/lib',P/'third_party/ORB-SLAM3/lib',
 P/'third_party/install/opencv-cuda118-sm86-gcc9/lib',P/'third_party/install/jsoncpp-1.9.5/lib',Path('/usr/local/cuda-11.8/lib64')])
def save(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(command,label):
 start=time.monotonic()
 with (R/(label+'.log')).open('w') as f:p=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(label+'.json',dict(command=command,returncode=p.returncode,seconds=time.monotonic()-start));assert p.returncode==0,label
def compile_probe(name):
 command=json.loads((PREV/'compile_inference_gate.json').read_text())['command']
 command[command.index(str(PREV/'inference_gate.cpp'))]=str(R/(name+'.cpp'))
 command[command.index('-o')+1]=str(R/name)
 command.insert(1,'-I'+str(R));command.insert(command.index('-o'),str(R/'diagnostic_pack.o'));command.insert(command.index('-o'),str(R/'diagnostic_pack_cuda.o'))
 run(command,'compile_'+name)
