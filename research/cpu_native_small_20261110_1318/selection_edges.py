"""真实Pipeline上的同分、资格不足、空候选和K边界，对照原排序语义。"""
from common import *
import selector,gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);source=load_scene('Replica__office2__rgbd');model=source['model'];model.soft_net.weight.zero_();model.soft_net.bias.zero_()
    n=len(source['pc'].get_xyz);base_mask=selector.slot_eligibility;records=source['records'];cam=camera(records[len(records)//2]);checks=[]
    empty=copy.copy(cam);empty.full_proj_transform=torch.zeros(4,4);empty.full_proj_transform[3,3]=-1000.
    conditions=['fixed'+str(k) for k in [0,1,2,3,7,8,9,31,32,33,63,64,65,127,128,129,511,512,513,1000,5000,n,n+1]]+[
        'local0','local0.0000001','local0.01','local0.03','local0.5','local0.999','local1','global0.0000001','global0.01','global0.1','global1']
    try:
        for noise in ['constant','discrete']:
            if noise=='constant':source['reference'].noise.zero_()
            else:
                g=torch.Generator().manual_seed(123);source['reference'].noise.copy_(torch.randint(-2,3,source['reference'].noise.shape,generator=g).float())
            for mask_name in ['all','striped','one','none']:
                mask=torch.ones(n,dtype=torch.bool) if mask_name=='all' else torch.arange(n)%5!=0 if mask_name=='striped' else torch.arange(n)==n//2 if mask_name=='one' else torch.zeros(n,dtype=torch.bool)
                selector.slot_eligibility=lambda model,opacity,mask=mask:mask
                baseline=native(source,'baseline');variants={v:native(source,v) for v in ['serial','parallel','combined']}
                for view_name,view in [('normal',cam),('empty',empty)]:
                    for condition in conditions:
                        expected,ids,stats=request(baseline,view,condition,compensation=False)
                        for variant,engine in variants.items():
                            packet,selected,info=request(engine,view,condition,compensation=False)
                            assert torch.equal(ids,selected),(noise,mask_name,view_name,condition,variant,len(ids),len(selected))
                            assert (selected[1:]>selected[:-1]).all() and info['candidate_gaussians']==stats['candidate_gaussians']
                            checks.append(dict(noise=noise,eligibility=mask_name,view=view_name,condition=condition,variant=variant,K=len(ids),M=stats['candidate_gaussians'],same_ids=True))
                del baseline,variants,engine;gc.collect();print('SELECTION_EDGES',noise,mask_name,len(checks),flush=True)
    finally:selector.slot_eligibility=base_mask
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),strict_local_K=True,global_ties_preserved=True,cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()
