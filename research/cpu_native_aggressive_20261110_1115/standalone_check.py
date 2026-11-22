"""验证checkpoint＋原GPU固定键可以全CPU独立恢复，结果与内存快照相同。"""
from cpu_study import *


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);checks=[]
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']:
        source=load_scene(slug);noise_path=HERE/'noise_snapshots'/(slug+'_seed0.pt')
        noise=torch.load(noise_path,map_location='cpu',weights_only=True)
        if isinstance(noise,dict):noise=noise['noise']
        engine=CpuRenderer.from_checkpoint(source['checkpoint'],noise_path,threads=6)
        reference=CpuRenderer(source['model'],source['pc'],noise,6)
        cam=camera(source['records'][len(source['records'])//2])
        for condition in ['local0.01','local0.08','local0.7','fixed0','fixed1000','global0.1']:
            expected,ids,_=call(reference,cam,condition);actual,selected,_=call(engine,cam,condition)
            assert torch.equal(ids,selected)
            for name,value in expected['inputs'].items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,actual['inputs'][name]),(slug,condition,name)
            checks.append(dict(scene=slug,condition=condition,K=len(ids),same_ids=True,same_attributes=True))
        del engine,reference,source
    assert not torch.cuda.is_initialized()
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),cuda_initialized=False))
    print('STANDALONE_CPU_PASS',len(checks),flush=True)


if __name__=='__main__':main()
