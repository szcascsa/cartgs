"""核查32组Full赢家，固定当前源码快照，在指定结果根目录隔离运行。"""
from pathlib import Path
import collections
import copy
import json
import math
import shutil
import time
from pipeline import sha,read,write

R=Path(__file__).resolve().parent
SAVED=Path('/home/prmi/zwj/realLOD/saved_experiments')
OLD=Path('/home/prmi/zwj/experiments/all_best_full_lod_no_frustum_20261104')


def archived(path):
    text=str(path)
    for n in ['multi_dataset_full_tail_prune1_seed0_20261103','tum_full_tail_prune_20261103']:
        text=text.replace('/home/prmi/zwj/experiments/'+n,str(SAVED/n))
    return Path(text).resolve()


def main():
    assert R==SAVED/'dual_view_li_best_full_20261107'
    assert not (R/'PREPARED.json').exists()
    assert shutil.disk_usage(R).free>5*2**30
    for n in ['configs','reports','logs','runs','evaluation','validation']: (R/n).mkdir(exist_ok=True)
    sources=read(R/'reports/SOURCE_HASHES.json')
    assert all(sha(R/'source'/n)==v for n,v in sources.items())
    input_hashes={};training={};evaluations={}
    for name in ['multi_dataset_full_tail_prune1_seed0_20261103','tum_full_tail_prune_20261103']:
        b=SAVED/name
        for n in ['QUEUE.json','TRAJECTORY_EVALUATION.json']:
            input_hashes[str(b/n)]=sha(b/n)
        for j in read(b/'QUEUE.json'):
            if j['prune']==1:
                dataset=j.get('dataset','TUM_RGBD');training[dataset,j['id']]=j
        for j in read(b/'TRAJECTORY_EVALUATION.json')['jobs']:
            if j['prune']==1:evaluations[j.get('dataset','TUM_RGBD'),j['id']]=j
    original=read(OLD/'SELECTED_BEST_FULL.json');assert len(original)==32
    input_hashes[str(OLD/'SELECTED_BEST_FULL.json')]=sha(OLD/'SELECTED_BEST_FULL.json')
    jobs=[]
    for prior in original:
        dataset,scene,mode=(prior[k] for k in ['dataset','scene','mode'])
        assert mode in (['mono','stereo'] if dataset=='VECtor' else ['mono','rgbd'])
        candidates=[j for (d,_),j in training.items() if d==dataset and j['scene']==scene and j['mode']==mode]
        assert sorted(j['repeat'] for j in candidates)==[1,2,3]
        measured={c['id']:c['result'] for c in prior['candidates']}
        for j in candidates:
            e=evaluations[dataset,j['id']]
            assert j['state']==e['state']=='complete' and j['seed']==0
            if mode!='stereo':
                assert e['result']['trajectory_psnr']==measured[j['id']]['trajectory_psnr']
            else:
                assert 'uncorrected_trajectory_psnr' in measured[j['id']]
                metric=Path(measured[j['id']]['output'])/'metric_summary.json'
                assert metric.exists() and abs(read(metric)['psnr']-measured[j['id']]['trajectory_psnr'])<1e-9
                input_hashes[str(metric)]=sha(metric)
            assert math.isfinite(measured[j['id']]['trajectory_psnr'])
        winner=max(candidates,key=lambda j:(measured[j['id']]['trajectory_psnr'],-j['repeat']))
        assert winner['id']==prior['winner'],prior['id']
        job=copy.deepcopy(prior);job.update(state='pending',evaluations=[],attempt_history=[],health_failures=0,oom_failures=0)
        full=archived(winner['result']['checkpoint']);full.relative_to(SAVED)
        assert read(full/'real_lod_state.json')['lod_steps']==0
        job['full_job']=copy.deepcopy(winner);job['full_job']['result']['checkpoint']=str(full)
        job['full_evaluation']['model']=str(archived(job['full_evaluation']['model']))
        job['full_evaluation']['data']=str(archived(job['full_evaluation']['data']))
        assert Path(job['full_evaluation']['data']).is_dir()
        job['target_steps']=winner['result']['online_steps']
        assert read(full/'real_lod_state.json')['online_full_steps']==job['target_steps']
        for k in ['camera_config','mapping_config']:
            original_config=archived(job[k]);dest=R/'configs'/dataset/scene/mode/original_config.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original_config,dest);job[k]=str(dest)
            input_hashes[str(original_config)]=sha(original_config)
        cloud,=(full/'point_cloud').glob('iteration_*/point_cloud.ply')
        for p in [cloud,*cloud.parent.glob('*.pt'),*cloud.parent.glob('*.json'),full/'selector.pt',full/'selector_config.yaml',full/'cameras.json',full/'real_lod_state.json',
                  Path(job['full_evaluation']['model'])/'CameraTrajectory_TUM.txt',Path(job['full_evaluation']['model'])/'TrackingTime.txt']:
            assert p.is_file(),p;input_hashes[str(p)]=sha(p)
        for v in read(full/'cameras.json'):
            image=Path(v['img_name'])
            if not image.is_absolute():image=Path(job['full_evaluation']['data'])/image
            assert image.is_file(),image
        jobs.append(job)
    deferred=[j for j in jobs if j['dataset']=='VECtor']
    jobs=[j for j in jobs if j['dataset']!='VECtor'];assert len(jobs)==26 and len(deferred)==6
    write(R/'DEFERRED_VECTOR_BY_USER.json',dict(reason='用户2026-11-07要求VECtor整个数据集暂时不跑，避免OOM',jobs=deferred))
    shutil.copyfile(R/'source/cfg/selector/selector_config.yaml',R/'configs/dual_view_li.yaml')
    text=(R/'configs/dual_view_li.yaml').read_text()
    for expected in ['ViewBudget.conditioning: view','ViewBudget.li_weight: 0.1','ViewBudget.enabled: 1','Selector.anchor_only_pruning: 1']:
        assert expected in text
    shutil.copytree(OLD/'evaluation_runtime',R/'evaluation_runtime',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(OLD/'configs/vector_stereo_native_maps.yml.gz',R/'configs/vector_stereo_native_maps.yml.gz')
    previous=read(Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/reports/PRODUCTION_BEFORE.json'))
    production={n:sha(Path(n)) for n in previous if Path(n).is_file()}
    for base in ['/home/prmi/zwj/realLOD/src','/home/prmi/zwj/realLOD/include','/home/prmi/zwj/realLOD/cfg','/home/prmi/zwj/realLOD/bin','/home/prmi/zwj/realLOD/lib']:
        b=Path(base)
        if b.exists():
            for p in b.rglob('*'):
                if p.is_file() and not p.is_symlink():production[str(p)]=sha(p)
    write(R/'PROTECTED_BEFORE.json',dict(inputs=input_hashes,production=production))
    write(R/'SELECTED_BEST_FULL.json',jobs)
    state=dict(phase='prepared',jobs=jobs,validated_modes=[],protocol=dict(seed=0,conditioning='both_real_view',li_weight=.1,
        train_ratios=[.01,.05,.1,.3,.5,.7],unseen_ratios=[.03,.08,.2,.4,.6],evaluation_scopes=['global','view_ratio','full'],
        objective='Full + global LoD + local LoD; one shared Adam',master_updates=True,topology_frozen=True,optimizer_reset=True,
        fine_steps='selected online Mapping count; tail excluded',saved_images=0,max_oom_retries=2,max_health_retries=2,hourly_interval_seconds=3600))
    write(R/'LOD_PIPELINE.json',state)
    write(R/'PREPARED.json',dict(passed=True,groups=26,deferred_vector_groups=6,counts=dict(collections.Counter(j['dataset']+'/'+j['mode'] for j in jobs)),
        fine_steps=sum(j['target_steps'] for j in jobs),protected_inputs=len(input_hashes),protected_production_files=len(production),created_unix=time.time()))
    lines=['# 两网络真实视角＋LI：批量LoD实验','',
        '每个数据集／场景／相机模式选择seed0三次中Full全轨迹PSNR最高一次，only anchor prune=1；本轮仅Replica 16组和TUM 10组，VECtor全部6组按用户要求暂不运行。',
        '当前本地main 26c355b及未提交兼容／推理变更独立快照构建，不覆盖生产源码、bin/lib或共享环境。',
        'Fine采用Full＋全局LoD＋局部LoD，真实view7＋mode1，LI=0.1；Master可更新、拓扑固定；Adam/RNG重置，不称完整SLAM恢复。',
        '训练ratio [.01,.05,.1,.3,.5,.7]；未见ratio [.03,.08,.2,.4,.6]仅评价。全局与局部两种全轨迹评价，另评Full；不保存图片。',
        '沿用原长尾/零选点/饱和/非有限值检查，触发后从原Full重新训练，不改参数或Loss。OOM最多额外重试2次；长尾/零选点重试策略见LOD_PIPELINE.json。',
        '每小时独立监督器核查PID/start_ticks/boot_id、实际进程、日志与CSV增量、磁盘、零图片及健康停止；控制器即时处理健康失败，较每小时更及时。',
        '保留最终模型、逐ratio/逐帧数值结果及必要复现/失败证据；仅清理本轮初始化副本及无用中间导出，不删除原Full或历史实验。','',
        '|数据集／场景／模式|Full赢家|Full PSNR|Fine步数|','|---|---|---:|---:|']
    for j in jobs:lines.append(f"|{j['id']}|{j['winner']}|{j['full_psnr']:.6f}|{j['target_steps']}|")
    (R/'reports/BATCH_PROTOCOL.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(read(R/'PREPARED.json')),flush=True)


if __name__=='__main__':main()
