"""比较相同冻结网络的Compensator 1/2 warp属性逐位一致，复用已验收画质。"""
from study import *
import gc
@torch.no_grad()
def main():
    rows=[]
    for spec in [s for s in SCENES if s['dataset_kind']=='Replica']:
        slug=spec['slug'];_,source=load_scene(slug);student,_,_=student_for(source,slug)
        old=engine(source,student,2,2,4,comp_warps=1);new=engine(source,student,2,2,4,comp_warps=2)
        frames=[int(x*(len(source.scene.views)-1)) for x in [.17,.57,.93]]
        for frame in frames:
            cam=source.camera(frame)
            for c in CONDITIONS:
                a,ids,st=request(old,cam,c);b,bids,bst=request(new,cam,c)
                assert torch.equal(ids,bids),(slug,frame,c,'ids')
                assert all(bits_equal(x,b['inputs'][k]) for k,x in a['inputs'].items() if isinstance(x,torch.Tensor)),(slug,frame,c,'attributes')
                rows.append(dict(scene=spec['id'],frame=frame,condition=c,K=len(ids),ids_and_attributes_bitwise=True))
        write_csv(HERE/'reports/BLOCK_PARITY.csv',rows);print('TC_BLOCK_PARITY',slug,len(rows),flush=True)
        del old,new,source,student,a,b;gc.collect();torch.cuda.empty_cache()
    save(HERE/'reports/BLOCK_PARITY.json',dict(passed=True,checks=len(rows),scope='与quality03完全相同的16组/3帧/14条件，s4/c1对s4/c2逐位相同；既有RGB质量证据可复用',binary_sha256=sha(tc.deployment_extension().__file__)))
if __name__=='__main__':main()
