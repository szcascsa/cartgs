"""真实CPU输入核对：候选、严格K、分数、属性和前缀消融；无GPU计算。"""
from cpu_study import *


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=load_scene('Replica__office2__rgbd');ref=source['reference'];model=source['model'];pc=source['pc'];rows=[]
    engines={name:CpuRenderer(model,pc,ref.noise,threads=6,selector_prefix=p,factored=f,diagnose=True)
        for name,p,f in [('native',False,False),('aggressive',True,True)]}
    for frame in [0,len(source['records'])//2,len(source['records'])-1]:
        cam=camera(source['records'][frame])
        for condition in ['local0.01','local0.05','local0.1','local0.7','fixed0','fixed1','fixed1000','global0.1','full']:
            truth,ids,stats=call(ref,cam,condition)
            for name,e in engines.items():
                tick=time.perf_counter_ns();packet,selected,info=call(e,cam,condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                if condition.startswith('local'):assert len(selected)==min(math.floor(float(condition[5:])*info['candidate_gaussians']),int(info['eligible_candidates']))
                assert len(selected)==len(ids) or condition.startswith('global'),(condition,info,len(ids))
                same=torch.equal(selected,ids);overlap=len(set(selected.tolist())&set(ids.tolist()))/max(1,len(ids));errors={}
                if same:
                    for key in ['means3D','opacities','scales','rotations','dc','shs']:
                        errors[key]=float((packet['inputs'][key]-truth['inputs'][key]).abs().max()) if len(ids) else 0.
                rows.append(dict(frame=frame,condition=condition,variant=name,N=len(pc.get_xyz),M=stats['candidate_gaussians'],K=len(selected),
                    same_ids=same,overlap=overlap,ms=elapsed,max_xyz=errors.get('means3D',-1),max_scale=errors.get('scales',-1),
                    max_rotation=errors.get('rotations',-1),max_dc=errors.get('dc',-1),workspace_bytes=e.native.workspace_bytes()))
                if condition.startswith('local') and len(ids):
                    visible,source_rows,scores,residual=e.native.diagnostics();expected=anchor_frustum_mask(model.anchors.centers,model.anchors.offset_scales,cam.full_proj_transform,ref.occupied)
                    assert torch.equal(visible,expected),'Nominal frustum changed'
                assert all(torch.isfinite(v).all() for v in packet['inputs'].values() if isinstance(v,torch.Tensor))
            print('CPU_NATIVE_SMOKE',frame,condition,rows[-1],flush=True)
    write_csv(out/'checks.csv',rows);save(out/'SUMMARY.json',dict(passed=True,checks=len(rows),unchanged_candidates=True,
        same_ids=sum(r['same_ids'] for r in rows),minimum_overlap=min(r['overlap'] for r in rows if r['K']),
        binary=extension().__file__,binary_sha256=sha(extension().__file__),cuda_initialized=torch.cuda.is_initialized()))
    print('CPU_NATIVE_SMOKE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
