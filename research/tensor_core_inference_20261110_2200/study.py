"""只读现有Replica教师/学生；原v31和独立Tensor后端使用相同输入。"""
from pathlib import Path
import os,sys,importlib.util
HERE=Path(__file__).resolve().parent
os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
sys.dont_write_bytecode=True
sys.path.insert(0,'/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053')
from harness import *
from compensation_student import load_student
from deployment_runtime import DeploymentRenderer as OriginalRenderer
from frozen_renderer import create_renderer,create_camera_preparer
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tensor_deployment_runtime',HERE/'source/deployment_runtime.py')
tc=importlib.util.module_from_spec(spec);spec.loader.exec_module(tc)
MODEL_ROOT=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models')
PRIMARY=[.1,.2,.3,.4,.5]
CONDITIONS=[f'local{x:g}' for x in PRIMARY]+['local0.01','local0.7','local0.15','local0.45','fixed0','fixed1','fixed1000','fixed5000','full']
TUNE=['Replica__office2__rgbd','Replica__room0__rgbd','Replica__office1__mono']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def rotate(values,i):
    k=i%len(values);items=values[k:]+values[:k];return items[::-1] if i%2 else items
def engine(source,student,smode=0,cmode=0,warps=2,original=False,graph=True,comp_warps=None):
    cls=OriginalRenderer if original else tc.DeploymentRenderer
    kw=dict(student=student,prepare_graph=graph)
    if not original:kw.update(tensor_selector=smode,tensor_compensator=cmode,tensor_warps=warps,tensor_compensator_warps=comp_warps)
    return cls(source.scene.model,source.scene.gaussians,source.scene.config,**kw)
def student_for(source,slug):
    path=MODEL_ROOT/slug/'continuous_calibration/best.pt';s,payload=load_student(path,'cuda',source.fingerprint)
    assert sum(p.numel() for p in s.parameters())==3534
    return s,path,payload
def request(e,cam,condition):
    if condition=='full':p,ids=e.full();return p,ids,dict(candidate_gaussians=len(ids))
    kw=dict(ratio_budget=float(condition[5:])) if condition.startswith('local') else dict(budget=int(condition[5:]))
    return e.prepare(cam,**kw)
def difference(a,b):
    result={}
    for k,x in a['inputs'].items():
        if isinstance(x,torch.Tensor):
            y=b['inputs'][k];result[k]=float((x-y).abs().max()) if x.numel() else 0.
    return result

if __name__=='__main__':
    mod=tc.deployment_extension();print('TENSOR_BUILD_PASS',mod.__file__,sha(mod.__file__),flush=True)
