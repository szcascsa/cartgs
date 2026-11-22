"""只补渲染已排名的少量视角，复用冻结局部入口并核对历史选点和PSNR。"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parent
BATCH=Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
RATIOS=[.01,.05,.1,.3]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(item, output):
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend='Agg'
    entry=BATCH/'eval_joint_entry.py'
    command=load(Path(item['result']['output'])/'launch.json')['command']
    sys.argv=list(command[command.index(str(entry)):])
    sys.argv[sys.argv.index('--output_path')+1]=str(output)
    prefix=entry.read_text().split('\nrunpy.run_path(sys.argv[0],run_name=')[0]
    assert prefix!=entry.read_text()
    exec(compile(prefix,str(entry),'exec'),{'__file__':str(entry),'__name__':'candidate_setup'})
    sys.argv=sys.argv[1:]
    script=BATCH/'source/python/run_metrics_only.py'
    text=script.read_text();marker='    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []'
    assert text.count(marker)==1
    namespace={'__name__':'__main__','__file__':str(script)}
    exec(compile(text.split(marker)[0],str(script),'exec'),namespace)
    return namespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',required=True)
    parser.add_argument('--alternative',action='store_true')
    parser.add_argument('--rgb-indices',type=int,nargs='+')
    args=parser.parse_args()
    ranking=load(ROOT/'reports/RANKING.json')
    group=next(group for group in ranking['groups'] if group['id']==args.group)
    job=next(job for job in load(BATCH/'LOD_PIPELINE.json')['jobs'] if job['id']==args.group)
    # 每组保留绝对最优帧；同时给前三名提供审核素材，原始排名不变。
    if args.rgb_indices:
        with (ROOT/'reports/ALL_FRAME_RANKING.csv').open(encoding='utf-8-sig') as stream:
            candidates=[row for row in csv.DictReader(stream)
                        if row['id']==args.group and int(row['rgb_index']) in args.rgb_indices]
        assert len(candidates)==len(args.rgb_indices)
        for row in candidates:
            for key in ['rgb_index','evaluation_ordinal','trajectory_index','N','candidates']:
                row[key]=int(row[key])
    else:
        candidates=[group['best']]
        for row in group['top']:
            if all(abs(row['rgb_index']-old['rgb_index'])>=8 for old in candidates):
                candidates.append(row)
            if len(candidates)==3:break
    label='furnished' if args.rgb_indices else ('alternatives' if args.alternative else 'best')
    folder=ROOT/'images'/label/args.group
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(job['trained_model']);protected={}
    for path in checkpoint.rglob('*'):
        if path.is_file() and (path.suffix in ['.pt','.ply'] or path.name=='cameras.json'):
            protected[str(path)]=sha(path)
    for row in candidates:
        out=folder/f"rgb_{row['rgb_index']:06d}"
        out.mkdir(exist_ok=True)
        source=Path(row['rgb_path']);shutil.copyfile(source,out/('GT_source'+source.suffix))
    records=[]
    for ratio in RATIOS:
        import cv2
        import numpy as np
        import torch
        item=next(item for item in job['evaluations'] if item['scope']=='view_ratio' and item['ratio']==ratio)
        original=Path(item['result']['output'])
        work=ROOT/'work'/args.group/f'ratio_{ratio:g}'
        namespace=setup(item,work)
        from renderer import render_prepared
        from utils import MiniCam
        expected=load(original/'selection_hashes.json')
        with (original/'per_frame_metrics.csv').open() as stream:metrics=list(csv.DictReader(stream))
        with (original/'anchor_frustum_per_view.csv').open() as stream:budgets=list(csv.DictReader(stream))
        for row in candidates:
            ordinal=row['evaluation_ordinal'];trajectory_index,rgb_index=namespace['associations'][ordinal]
            trajectory_index,rgb_index=int(trajectory_index),int(rgb_index)
            assert trajectory_index==row['trajectory_index'] and rgb_index==row['rgb_index']
            camera=MiniCam(namespace['width'],namespace['height'],namespace['fovx'],namespace['fovy'],
                           np.linalg.inv(namespace['poses'][trajectory_index]))
            with torch.no_grad():
                data,ids,stats=namespace['frustum_renderer'].prepare(camera)
                digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                assert digest==expected[ordinal],(args.group,ratio,ordinal,'选点身份不一致')
                assert len(ids)==int(budgets[ordinal]['active_gaussians'])
                image=render_prepared(camera,data,namespace['background'])['render'].clamp(0,1)
                truth=cv2.cvtColor(cv2.imread(row['rgb_path'],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                target=torch.from_numpy(truth).to('cuda').permute(2,0,1)[None]/255.
                scores={name:namespace['calc_'+name](image[None],target).item() for name in ['psnr','ssim','lpips']}
                errors={name:abs(scores[name]-float(metrics[ordinal][name])) for name in scores}
                assert errors['psnr']<2e-4 and errors['ssim']<2e-5 and errors['lpips']<2e-5,errors
                output=folder/f"rgb_{rgb_index:06d}"
                rgb=(image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(str(output/f'ratio_{int(ratio*100):02d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if ratio==.01:cv2.imwrite(str(output/'GT.png'),cv2.cvtColor(truth,cv2.COLOR_RGB2BGR))
                record=dict(id=args.group,label=label,ratio=ratio,rgb_index=rgb_index,evaluation_ordinal=ordinal,
                            trajectory_index=trajectory_index,rgb_path=row['rgb_path'],N=row['N'],
                            K=len(ids),candidates=int(stats['candidate_gaussians']),scores=scores,
                            historical_scores={name:float(metrics[ordinal][name]) for name in scores},
                            abs_errors=errors,selection_sha256=digest,selection_identical=True,
                            output=str(output/f'ratio_{int(ratio*100):02d}.png'))
                records.append(record)
                print('RENDER_VERIFIED',json.dumps(record),flush=True)
                del image,target,data,ids
        del namespace
        gc.collect();torch.cuda.empty_cache()
    assert all(sha(Path(name))==digest for name,digest in protected.items())
    receipt=dict(group=args.group,candidates=candidates,records=records,checkpoint_unchanged=True,
                 checkpoint_sha256=protected)
    (folder/'RENDER_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('GROUP_COMPLETE',args.group,len(records),flush=True)


if __name__=='__main__':
    main()
