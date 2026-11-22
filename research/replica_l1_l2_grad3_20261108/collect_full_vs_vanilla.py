"""只读当前3τ0 Full与纯Vanilla历史结果，绑定同次指标、N和来源。"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import time

R=Path('/home/prmi/zwj/realLOD/checkpoint/replica_l1_l2_grad3_20261108')
V=Path('/home/prmi/zwj/realLOD/vanilla/vanilla_all_datasets_20261031_2011')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vanilla_path(text):
    for root in ['/home/prmi/zwj/experiments/vanilla_all_datasets_20261031_2011',
                 '/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011']:
        if text.startswith(root+'/'):return V/Path(text).relative_to(root)
    path=Path(text)
    assert path.resolve().is_relative_to(V.resolve()),text
    return path


def ply_count(path):
    with path.open('rb') as stream:
        for _ in range(150):
            line=stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):return int(line.split()[-1])
            if line=='end_header':break
    raise ValueError('PLY缺少vertex数量: '+str(path))


def main():
    assert read(R/'FINISHED.json')['passed']
    protected={}
    def protect(path):
        protected[str(path)]=sha(path)
    candidates=defaultdict(list)
    source=V/'reports/PER_RUN.csv';protect(source)
    for raw in csv.DictReader(source.open()):
        if not raw['scene'].startswith('replica/') or raw['state']!='complete':continue
        _,mode,scene=raw['scene'].split('/')
        assert raw['seed']=='0' and raw['phase']=='FULL' and raw['ratio']=='1.0'
        evaluation=vanilla_path(raw['evaluation']);checkpoint=vanilla_path(raw['checkpoint'])
        row=dict(scene=scene,mode=mode,method='vanilla',repeat=int(raw['repeat']),seed=0,
                 psnr=float(raw['psnr']),ssim=float(raw['ssim']),lpips=float(raw['lpips']),
                 gaussian_count=int(raw['N']),frames=int(raw['views']),checkpoint=str(checkpoint),
                 evaluation=str(evaluation),online_steps=int(raw['online']),tail_steps=int(raw['tail']))
        for metric in ['psnr','ssim','lpips']:
            path=evaluation/(metric+'.txt');protect(path)
            values=[float(l) for l in path.read_text().splitlines() if l.strip()]
            assert len(values)==row['frames'] and all(math.isfinite(v) for v in values)
            assert abs(sum(values)/len(values)-row[metric])<1e-6
        candidates[scene,mode].append(row)
    assert len(candidates)==16 and all(len(v)==3 for v in candidates.values())
    jobs=read(R/'QUEUE.json');evaluations=read(R/'TRAJECTORY_EVALUATION.json');protect(R/'QUEUE.json');protect(R/'TRAJECTORY_EVALUATION.json')
    by_id={j['id']:j for j in evaluations['jobs']};groups=[]
    for job in jobs:
        e=by_id[job['id']];assert job['state']==e['state']=='complete'
        original=e['result'];evaluation=Path(original['output']);checkpoint=Path(job['result']['checkpoint'])
        protect(evaluation/'verified.json');protect(checkpoint/'checkpoint_experiment.json')
        assert read(checkpoint/'checkpoint_experiment.json')['l1_to_l2_tau0_multiplier']==3
        full=dict(scene=job['scene'],mode=job['mode'],method='ours',repeat=1,seed=0,
                  gaussian_count=job['result']['gaussian_count'],frames=original['evaluated_frames'],
                  psnr=original['trajectory_psnr'],ssim=original['trajectory_ssim'],lpips=original['trajectory_lpips'],
                  checkpoint=str(checkpoint),evaluation=str(evaluation),
                  online_steps=job['result']['online_steps'],tail_steps=job['result']['tail_steps'])
        with (evaluation/'per_frame_metrics.csv').open() as stream:frames=list(csv.DictReader(stream))
        protect(evaluation/'per_frame_metrics.csv');assert len(frames)==full['frames']
        for metric in ['psnr','ssim','lpips']:
            assert abs(sum(float(f[metric]) for f in frames)/len(frames)-full[metric])<1e-9
        vanilla=max(candidates[job['scene'],job['mode']],key=lambda x:(x['psnr'],-x['repeat']))
        for row in [full,vanilla]:
            cloud,=Path(row['checkpoint']).rglob('point_cloud.ply')
            assert ply_count(cloud)==row['gaussian_count']
            assert all(math.isfinite(row[m]) for m in ['psnr','ssim','lpips'])
        groups.append(dict(scene=job['scene'],mode=job['mode'],ours=full,vanilla=vanilla,
                           vanilla_candidates=candidates[job['scene'],job['mode']]))
    assert len(groups)==16
    assert protected=={path:sha(Path(path)) for path in protected}
    target=R/'figures/full_vs_vanilla/data';target.mkdir(parents=True,exist_ok=True)
    value=dict(collected_unix=time.time(),groups=groups,source_hashes=protected,
               verification=dict(groups=16,metric_records=32,vanilla_candidates=48,
                                 raw_metric_means_verified=True,ply_vertex_counts_verified=True,source_unchanged=True),
               protocols=dict(ours='本轮3τ0 Full+tail，seed0单次；无LoD',
                   vanilla='每场景/相机模式seed0三次中Full全轨迹PSNR最高重复；SSIM/LPIPS/N绑定同次',
                   metrics='各自完整轨迹均值；未作共同RGB帧重聚合；Mono沿用尺度对齐，RGB-D不做尺度对齐',
                   count='未补偿Full地图存储Gaussian总数N，不是可见点或Alpha合成点数'))
    (target/'FULL_VS_VANILLA.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print('COLLECTED',value['verification'],flush=True)


if __name__=='__main__':main()
