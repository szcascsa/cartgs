"""同一冻结Teacher准备流程，实测每次切换视角的准备＋渲染收益。"""
from benchmark import *
import importlib.util


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('old_runtime',Path('/home/prmi/zwj/experiments/sparse_rasterizer_20261109/runtime.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    variants={'bitset_v1':old.SparseRenderer(2),'balanced':SparseRenderer(4)}
    samples=[];observations=[]
    for slug in ['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']:
        scene,source=load_scene(slug)
        cameras=[((19+137*i)%len(source.scene.views),source.camera((19+137*i)%len(source.scene.views))) for i in range(8)]
        for condition,request in [('local1pct',dict(ratio_budget=.01)),('local5pct',dict(ratio_budget=.05)),
                ('local10pct',dict(ratio_budget=.1)),('local70pct',dict(ratio_budget=.7)),
                ('local1000',dict(budget=1000)),('local5000',dict(budget=5000)),('full',{})]:
            load=external_load();observations.append(load);assert not load['active'],load
            functions={'original':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
            functions.update({n:(lambda cam,p,e=e:e.render(cam,p,source.scene.background)[0]) for n,e in variants.items()})
            for _,cam in cameras:
                if condition=='full':packet,_=source.runtime.full()
                else:packet,_,_=source.runtime.prepare(cam,**request)
                for fn in functions.values():
                    for _ in range(3):image=fn(cam,packet)
            torch.cuda.synchronize()
            for repeat in range(3):
                for i in range(40):
                    frame,cam=cameras[i%8]
                    names=list(functions)
                    if (i+repeat)%2:names.reverse()
                    else:names=names[(i+repeat)%3:]+names[:(i+repeat)%3]
                    pair=[]
                    for name in names:
                        torch.cuda.synchronize();start=time.perf_counter()
                        if condition=='full':
                            packet,ids=source.runtime.full();stats={'candidate_gaussians':len(ids)}
                        else:packet,ids,stats=source.runtime.prepare(cam,**request)
                        torch.cuda.synchronize();prepared=time.perf_counter()
                        image=functions[name](cam,packet);torch.cuda.synchronize();end=time.perf_counter()
                        samples.append(dict(scene=scene['id'],condition=condition,frame=frame,repeat=repeat,iteration=i,
                            variant=name,K=len(ids),candidate_N=stats['candidate_gaussians'],prepare_ms=(prepared-start)*1000,
                            render_ms=(end-prepared)*1000,total_ms=(end-start)*1000))
                        pair.append((ids.clone(),image))
                    assert all(torch.equal(a[0],pair[0][0]) and bits_equal(a[1],pair[0][1]) for a in pair[1:]),(slug,frame,condition)
            load=external_load();observations.append(load);assert not load['active'],load
            write_csv(args.output/'samples.csv',samples)
            (args.output/'LOAD.json').write_text(json.dumps(observations,indent=2))
            print('PIPELINE_BENCHMARK',slug,condition,flush=True)
        del source
        torch.cuda.empty_cache()
    results=[]
    for scene,condition in sorted({(r['scene'],r['condition']) for r in samples}):
        group=[r for r in samples if (r['scene'],r['condition'])==(scene,condition)]
        original=st.median(r['total_ms'] for r in group if r['variant']=='original')
        v1=st.median(r['total_ms'] for r in group if r['variant']=='bitset_v1')
        for name in ['original','bitset_v1','balanced']:
            g=[r for r in group if r['variant']==name]
            results.append(dict(scene=scene,condition=condition,variant=name,K_median=st.median(r['K'] for r in g),
                total_ms=st.median(r['total_ms'] for r in g),prepare_ms=st.median(r['prepare_ms'] for r in g),
                render_ms=st.median(r['render_ms'] for r in g),speedup=original/st.median(r['total_ms'] for r in g),
                speedup_v1=v1/st.median(r['total_ms'] for r in g)))
    (args.output/'SUMMARY.json').write_text(json.dumps(dict(passed=True,results=results,samples=len(samples),
        per_request_prepare=True,bitwise_images=True,identical_selection=True,network_changed=False,
        formal_idle_benchmark=True,observations=observations),indent=2))
    print('PIPELINE_COMPLETE',len(samples),flush=True)


def bits_equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


if __name__=='__main__':main()
