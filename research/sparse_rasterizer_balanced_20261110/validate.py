"""原二进制对照，检验逐像素/radii/Tile成员与同深度顺序。"""
from pathlib import Path
import json
import sys
import time
import hashlib
HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/research/compensator_distillation_replica_all_20261108')
sys.path.insert(0,str(PRIOR))
from scene_runtime import *
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime import SparseRenderer


def bits_equal(a,b):
    return a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


@torch.no_grad()
def main():
    renderers={'control':SparseRenderer(0),'warp_expand':SparseRenderer(1),
        'short':SparseRenderer(0,True),'warp_short':SparseRenderer(1,True),'bitset':SparseRenderer(2),'hierarchical':SparseRenderer(3),'compact':SparseRenderer(4)}
    rows=[]
    def compare(label,cam,packet,bg,inspect=False):
        reference=render_prepared(cam,packet,bg,return_radii=True)
        diagnostics={}
        for name,engine in renderers.items():
            # _C的空集与外层背景特殊处理分开记录。
            if not len(packet['inputs']['means3D']):
                result=engine.render(cam,packet,bg)
                assert not result[1].numel() and torch.equal(result[0],torch.zeros_like(result[0]))
                continue
            result=engine.render(cam,packet,bg,inspect)
            passed=bits_equal(result[0],reference['render']) and bits_equal(result[1],reference['radii'])
            row=dict(case=label,variant=name,K=len(result[1]),passed=passed,
                max_image_abs=float((result[0]-reference['render']).abs().max()),radii_equal=bits_equal(result[1],reference['radii']))
            rows.append(row)
            if not passed:
                (HERE/'reports/VALIDATION_FAILURE.json').write_text(json.dumps(row,indent=2))
                raise AssertionError(row)
            if inspect:diagnostics[name]=result
        if inspect and diagnostics:
            c=diagnostics['control'];valid=c[1]>0
            for name,out in diagnostics.items():
                assert bits_equal(out[2][valid],c[2][valid]),(label,name,'xy')
                assert bits_equal(out[3][valid],c[3][valid]),(label,name,'conic')
                assert bits_equal(out[4][valid],c[4][valid]),(label,name,'depth')
                assert bits_equal(out[5],c[5]),(label,name,'bbox count')
                if name in ['control','warp_expand','short','warp_short']:
                    assert bits_equal(out[6],c[6]) and bits_equal(out[7],c[7]),(label,name,'sorted key/order')
            # Bitset每Tile按rank枚举，严格对应原排序中该Tile的Gaussian序列。
            import numpy as np
            bs=diagnostics['bitset'];mask=bs[9].cpu().numpy().astype(np.uint32);sorted_ids=bs[7].cpu().tolist()
            original_ids=c[7].cpu().tolist();ranges=c[8].cpu().tolist()
            for tile,(start,end) in enumerate(ranges):
                rank=[]
                for w,word in enumerate(mask[tile]):
                    value=int(word)
                    while value:
                        bit=(value&-value).bit_length()-1;rank.append(sorted_ids[w*32+bit]);value&=value-1
                assert rank==original_ids[start:end],(label,tile,'bitset ordering')
        print('PARITY',label,len(rows),flush=True)
    torch.manual_seed(19)
    for W,H in [(97,65),(640,480)]:
        cam=MiniCam(W,H,1.2,1.0,np.eye(4))
        for P in [0,1,31,32,33,127,257,1024,4096]:
            xyz=torch.randn(P,3,device='cuda');xyz[:,2]=xyz[:,2].abs()+1
            if P>0 and P%2:xyz[:,2]=2 # 同深度稳定性
            scales=torch.rand(P,3,device='cuda')*.2+.002
            if P>=31:
                scales[:3]=torch.tensor([1.5,.1,.6],device='cuda')
                xyz[3:5,2]=-.5
                xyz[5,0]=float('nan');scales[6,0]=float('inf')
            rotations=torch.randn(P,4,device='cuda');rotations/=rotations.norm(dim=1,keepdim=True).clamp_min(1e-6)
            inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),opacities=torch.rand(P,1,device='cuda'),
                scales=scales,rotations=rotations,dc=torch.rand(P,1,3,device='cuda'),
                shs=torch.randn(P,15,3,device='cuda')*.1,colors_precomp=None,cov3D_precomp=None)
            bg=torch.tensor([.02,.2,.6],device='cuda')
            compare(f'synthetic/{W}x{H}/{P}',cam,dict(sh_degree=3,inputs=inputs),bg,inspect=P in [31,257,1024])
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']:
        scene,source=load_scene(slug)
        for frame in [19,len(source.scene.views)//2]:
            for ratio in [.01,.05,.1,.3,.5,.7,.2]:
                for local in [False,True]:
                    case=source.sample(frame,ratio,local)
                    compare(f'{slug}/{frame}/{ratio}/{local}',case['camera'],case['packet'],source.scene.background,
                        inspect=ratio==.01 and frame==19)
        del source
        torch.cuda.empty_cache()
    result=dict(passed=True,comparisons=len(rows),rows=rows,time=time.time(),
        variants=list(renderers),pixel_bitwise=True,radii_bitwise=True,stable_tile_order_verified=True)
    (HERE/'reports/VALIDATION.json').write_text(json.dumps(result,indent=2))
    print('VALIDATION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
