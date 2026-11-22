"""只读加载已冻结A/C/D；本轮不构建、不训练、不修改历史研究。"""
from pathlib import Path
import sys,importlib.util,json,time,math,gc,hashlib,argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
TC=Path('/home/prmi/zwj/experiments/tensor_core_inference_20261110_2200')
PREP=TC/'preparation_phase'
sys.path.insert(0,str(PREP))
import prep_study as p
torch,np=p.torch,p.np
SIZE=(2000,2000)
SLUGS=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
RATIOS=[.01,.05,.1,.3,.7]
CONDITIONS=['full']+[f'local{r:g}' for r in RATIOS]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verified(name,path,expected):
    assert sha(path)==expected,str(path)
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def configure():
    c=json.loads((TC/'reports/FROZEN_CANDIDATE.json').read_text())
    d=json.loads((PREP/'reports/FROZEN_JOINT_CANDIDATE.json').read_text())
    for root,config in [(TC,c),(PREP,d)]:
        for name,expected in config['sources'].items():assert sha(root/name)==expected,name
    cm=verified('mycartgs_tensor_deployment_v1',Path(c['binary']),c['binary_sha256'])
    dm=verified('mycartgs_tensor_preparation_v1',Path(d['binary']),d['binary_sha256'])
    p.prior.tc.deployment_extension=lambda:cm
    p.dp.deployment_extension=lambda:dm
    import deployment_runtime
    assert sha(deployment_runtime.deployment_extension().__file__)==c['baseline_sha256']
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return c,d,cm,dm


class Camera:
    def __init__(self,source,variant,dm,size=SIZE):
        self.source=source;self.size=size;self.variant=variant
        config=source.scene.calibration
        self.fx=2*math.atan(config['width']/(2*config['fx']))
        self.fy=2*math.atan(config['height']/(2*config['fy']))
        if variant=='D':
            mod=p.runtime('tc_square_camera_api',PREP/'source/preparation_camera.py')
            self.factory=mod.NativeCameraPreparer(p.MiniCam,dm)
        else:self.factory=p.create_camera_preparer(p.MiniCam,variant='R')

    def create(self,frame):
        pose=self.source.scene.views[frame]['pose']
        if self.variant=='D':return self.factory.create_from_pose(*self.size,self.fx,self.fy,pose)
        return self.factory.create(*self.size,self.fx,self.fy,np.linalg.inv(np.asarray(pose)))


def engine(source,student,variant):
    return p.make_engine(source,student,variant,frustum=1 if variant=='D' else 0,
        projection=1 if variant=='D' else 0,device=1 if variant=='D' else 0)


def frames_for(source):return p.views_for(source,'final')


def event(function):
    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    a.record();value=function();z.record();z.synchronize()
    return a.elapsed_time(z),value


def write(name,rows,root=None):p.write_csv((root or HERE/'reports')/name,rows)


def save(name,value,root=None):p.save((root or HERE/'reports')/name,value)
