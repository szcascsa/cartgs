"""只重放既有局部选点，按冻结Master行ID统计跨视角并集；逐帧核对历史哈希。"""
from pathlib import Path
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/prmi/zwj/realLOD/saved_experiments/cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv')
SOURCE_SHA='e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891'
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def records():
    assert sha(SOURCE)==SOURCE_SHA
    return [r for r in csv.DictReader(SOURCE.open(encoding='utf-8-sig')) if r['method']=='ours_view_ratio']


def setup(items,folder):
    import torch
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend='Agg'
    launch=json.loads((Path(items[0]['source'])/'launch.json').read_text())
    command=launch['command']
    entry=next(i for i,arg in enumerate(command) if str(arg).endswith('/eval_joint_entry.py'))
    job_root=Path(command[entry]).parent
    script=job_root/'source/python/run_metrics_only.py'
    sys.path[:0]=[str(job_root/'source/python'),str(job_root/'evaluation_runtime')]
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRatioRenderer:
        def __init__(self,model,gaussians,ratio,config,temperature,protection):
            assert int(model.joint_view_conditioning_schema)==2
            self.renderer=FixedBudgetRenderer(model,gaussians,
                dict(config,temperature=temperature,protection_enabled=protection),seed=0,backend='pytorch')
        def prepare(self,camera,ratio,compensation=False):
            return self.renderer.prepare(camera,ratio_budget=ratio,view_conditioned=True,
                                         compensation=compensation,measure_time=False)

    anchor_frustum.AnchorFrustumRenderer=LocalRatioRenderer
    args=command[entry+2:]
    args.remove('--joint-local-ratio')
    args[args.index('--output_path')+1]=str(folder/'setup')
    args.append('--selector-anchor-frustum')
    sys.argv=[str(script),*args]
    prefix=script.read_text().split('    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []')[0]
    tree=ast.parse(prefix,filename=str(script))
    # 选点重放无需构建PSNR/SSIM/LPIPS模型，保留原模型加载、相机、轨迹和关联代码。
    tree.body=[node for node in tree.body if not
        (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
         ['calc_psnr','calc_ssim','calc_lpips'] for t in node.targets)) and not
        (isinstance(node,ast.ImportFrom) and (node.module or '').startswith('torchmetrics.'))]
    namespace={'__name__':'__main__','__file__':str(script)}
    exec(compile(ast.fix_missing_locations(tree),str(script),'exec'),namespace)
    current=json.loads((folder/'setup/evaluation_frames.json').read_text())
    for item in items:
        previous=json.loads((Path(item['source'])/'evaluation_frames.json').read_text())
        assert previous==current,(item['group'],item['ratio'],'评价帧或图像序列变化')
        assert len(previous['gt_indices'])==int(item['frames'])
    assert namespace['gaussians'].get_xyz.shape[0]==int(items[0]['N'])
    protected={SOURCE,script}
    selector=Path(args[args.index('--selector')+1])
    protected.update(p for p in selector.parent.rglob('*') if p.is_file())
    protected.add(Path(args[args.index('--selector-config')+1]))
    protected.update(Path(module.__file__) for module in list(sys.modules.values())
                     if getattr(module,'__file__',None) and str(module.__file__).startswith(str(job_root))
                     and Path(module.__file__).is_file())
    for item in items:
        protected.update(Path(item['source'])/name for name in
                         ['selection_hashes.json','metric_summary.json','evaluation_frames.json','launch.json','anchor_frustum_per_view.csv'])
    return namespace,{str(p):sha(p) for p in protected}


def run(group,limit):
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.18)
    items=sorted([r for r in records() if r['group']==group],key=lambda r:float(r['ratio']))
    assert [float(r['ratio']) for r in items]==RATIOS
    folder=ROOT/('smoke' if limit else 'runs')/group;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'COMPLETE.json').exists():
        print('ALREADY_COMPLETE',group,flush=True);return
    started=time.monotonic();ns,protected=setup(items,folder)
    write(folder/'INPUTS_BEFORE.json',protected)
    from utils import MiniCam
    from anchor_frustum import anchor_frustum_mask,compact_gaussians
    from anchor_budget import exact_budget_mask
    from selector import slot_eligibility
    from selector_sampling import gumbel_selection
    renderer=ns['frustum_renderer']
    hashes=[json.loads((Path(r['source'])/'selection_hashes.json').read_text()) for r in items]
    history=[list(csv.DictReader((Path(r['source'])/'anchor_frustum_per_view.csv').open())) for r in items]
    associations=ns['associations'];frames=len(associations)
    assert all(len(h)==len(v)==frames for h,v in zip(hashes,history))
    count=int(items[0]['N']);seen=np.zeros((11,count),dtype=np.bool_)
    totals=np.zeros(11,dtype=np.int64);compensation_checks=[]
    stop=min(frames,limit) if limit else frames
    with (folder/'PER_VIEW_UNION.csv').open('w',newline='') as stream,torch.no_grad():
        writer=csv.DictWriter(stream,['frame','trajectory_index','gt_index','ratio','K','new_unique','union_count','selection_sha256'])
        writer.writeheader()
        for frame,(trajectory,gt) in enumerate(associations[:stop]):
            camera=MiniCam(ns['width'],ns['height'],ns['fovx'],ns['fovy'],np.linalg.inv(ns['poses'][trajectory]))
            # 同一相机的候选池、物理属性与视角编码和Ratio无关，逐帧只计算一次。
            fixed=renderer.renderer;fixed.check_state();anchors=fixed.model.anchors
            visible=anchor_frustum_mask(anchors.centers,anchors.offset_scales,camera.full_proj_transform,fixed.occupied)
            source_rows=torch.nonzero(visible[anchors.rows]).flatten()
            compact=compact_gaussians(fixed.gaussians,source_rows)
            eligible=slot_eligibility(fixed.model,compact.get_opacity,source_rows)
            available=int(eligible.sum());population=len(source_rows)
            anchor_code=fixed.model.joint_view_code(camera) if population and available else None
            noise=fixed.noise[source_rows]
            for index,ratio in enumerate(RATIOS):
                budget=math.floor(ratio*population)
                mask=torch.zeros(population,dtype=torch.bool,device=source_rows.device)
                if available and budget>=available:
                    mask=eligible.clone()
                elif population and available and budget:
                    _,_,logits=fixed.model.forward_view_selection_features(
                        compact.get_xyz,compact.get_scaling,compact.get_features_dc,compact.get_opacity,
                        fixed.gaussians.get_xyz.new_tensor([[ratio]]),camera,
                        source_rows=source_rows,return_logits=True,anchor_code=anchor_code)
                    if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite Selector logits')
                    scores=gumbel_selection(logits,hard=False,noise=noise)
                    mask=exact_budget_mask(scores,eligible,budget)
                ids=source_rows[mask]
                selected=ids.cpu().numpy()
                digest=hashlib.sha256(selected.tobytes()).hexdigest()
                assert digest==hashes[index][frame],(group,ratio,frame,'选点ID哈希不符')
                assert len(selected)==int(float(history[index][frame]['active_gaussians']))
                assert len(selected)==min(budget,available)
                if frame in {0,frames//2,frames-1}:
                    full_data,full_ids,_=renderer.prepare(camera,ratio,compensation=True)
                    assert torch.equal(ids,full_ids),(group,ratio,frame,'补偿开关影响选点')
                    compensation_checks.append(dict(frame=frame,ratio=ratio,ids_bitwise_equal=True))
                    del full_data,full_ids
                new_unique=int((~seen[index,selected]).sum())
                seen[index,selected]=True;totals[index]+=len(selected)
                writer.writerow(dict(frame=frame,trajectory_index=int(trajectory),gt_index=int(gt),ratio=ratio,
                                     K=len(selected),new_unique=new_unique,union_count=int(seen[index].sum()),selection_sha256=digest))
                del ids
            if (frame+1)%64==0 or frame+1==stop:
                stream.flush();status=dict(group=group,frame=frame+1,frames=stop,
                    ratios=11,elapsed_seconds=time.monotonic()-started,updated_unix=time.time(),
                    union_counts=seen.sum(axis=1).tolist())
                write(folder/'STATUS.json',status);print('PROGRESS',json.dumps(status),flush=True)
    after={p:sha(p) for p in protected};assert protected==after,'源输入发生改变'
    write(folder/'INPUTS_AFTER.json',after)
    result=[]
    for index,item in enumerate(items):
        actual_mean=float(totals[index])/stop
        if not limit:assert abs(actual_mean-float(item['K_mean']))<1e-9,(group,item['ratio'],'逐视角均值不符')
        union=np.flatnonzero(seen[index]).astype(np.int64)
        assert actual_mean<=len(union)<=count
        np.savez_compressed(folder/f"UNION_IDS_{item['ratio']}.npz",ids=union)
        result.append(dict(group=group,ratio=float(item['ratio']),N=count,views=stop,
            local_union_count=len(union),local_mean_K=actual_mean,
            reference_mean_K=float(item['K_mean']),source=item['source'],
            union_ids_sha256=hashlib.sha256(union.tobytes()).hexdigest(),
            all_view_hashes_match=True))
    write(folder/('SMOKE.json' if limit else 'COMPLETE.json'),dict(complete=not bool(limit),smoke=bool(limit),
        group=group,ratios=result,compensation_checks=compensation_checks,
        source_unchanged=True,frames=stop,total_frames=frames,images_saved=0,training_steps=0,
        seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('UNION_COMPLETE' if not limit else 'SMOKE_COMPLETE',group,'frames',stop,'seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True)
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args();run(args.group,args.limit)
