"""保存checkpoint及原固定噪声全CPU恢复，未初始化CUDA。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);checks=[]
    for slug in ['Replica__office2__rgbd','TUM_RGBD__freiburg1_xyz__rgbd','VECtor__corner-slow__stereo']:
        source=load_scene(slug);path=ORIGINAL/'noise_snapshots'/(slug+'_seed0.pt');noise=torch.load(path,map_location='cpu',weights_only=True)
        if isinstance(noise,dict):noise=noise['noise']
        engine=Mid.from_checkpoint(source['checkpoint'],path,threads=6,selection_mode=1,comp_batch=6);source['reference'].noise.copy_(noise)
        reference=native(source,'baseline');cam=camera(source['records'][len(source['records'])//2])
        for condition in ['local0.01','local0.08','local0.7','fixed0','fixed1000','global0.1']:
            packet,ids,_=request(engine,cam,condition);expected,oldids,_=request(reference,cam,condition);assert torch.equal(ids,oldids)
            for key,value in packet['inputs'].items():
                if isinstance(value,torch.Tensor):
                    assert torch.equal(value,expected['inputs'][key]) or (key=='rotations' and torch.allclose(value,expected['inputs'][key],atol=1e-6,rtol=0))
            checks.append(dict(scene=slug,condition=condition,K=len(ids),nonrotation_bitwise=True))
        del engine,reference,source,packet,expected;gc.collect()
    assert not torch.cuda.is_initialized();write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),cuda_initialized=False))
    print('CHECKPOINT_CPU_PASS',len(checks),flush=True)

if __name__=='__main__':main()
