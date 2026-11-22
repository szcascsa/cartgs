import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def tensors(p):return {k:v.detach()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()}
class Scene:
    def __init__(self):
        self.shared=Path('/dev/shm')/f'redundancy_audit_{os.getpid()}.bin';self.shared.write_bytes(b'')
        with self.shared.open('r+b')as f:f.truncate(4*3*480*640*4)
        self.mapping=np.memmap(self.shared,dtype=np.float32,mode='r+',shape=(4,3,480,640))
        self.log=(ROOT/'worker.log').open('a');self.proc=subprocess.Popen([str(ROOT/'audit_worker'),str(ROOT/'input_manifest.json'),str(self.shared)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
    def call(self,c):
        self.proc.stdin.write(json.dumps(c)+'\n');self.proc.stdin.flush()
        while True:
            line=self.proc.stdout.readline()
            if not line:raise RuntimeError('Worker exited: '+(ROOT/'worker.log').read_text()[-6000:])
            if line.startswith('AUDIT_RESPONSE '):return json.loads(line[15:])
    def close(self):
        result=self.call(dict(method='verify'));save(ROOT/'frozen_tensor_checks.json',result)
        self.proc.stdin.write('{"method":"quit"}\n');self.proc.stdin.flush();self.proc.wait(timeout=30);assert self.proc.returncode==0
        self.log.close();del self.mapping;self.shared.unlink()
class Metrics:
    def __init__(self,gt):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.functional.image.lpips import _normalize_tensor
        self.normalize=_normalize_tensor;self.gt=gt.cuda();self.lp=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda().eval().net
    @torch.no_grad()
    def all(self,image,indices):
        from torchmetrics.functional.image import structural_similarity_index_measure
        image=image.clamp(0,1);gt=self.gt[indices];psnr=-10*torch.log10((image-gt).square().mean((1,2,3)).clamp_min(1e-30))
        ss=structural_similarity_index_measure(image,gt,data_range=1.,reduction='none')
        a=self.lp.net(self.lp.scaling_layer(2*image-1));b=self.lp.net(self.lp.scaling_layer(2*gt-1));terms=[]
        for k in range(self.lp.L):terms.append(self.lp.lins[k]((self.normalize(a[k])-self.normalize(b[k])).square()).mean((2,3),keepdim=True))
        return torch.stack((psnr,ss,sum(terms).flatten()),1).cpu().numpy()
