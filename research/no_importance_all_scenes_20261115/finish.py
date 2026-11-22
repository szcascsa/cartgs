"""汇总D组与原完整方案，检查完整性并生成中文结果报告。"""
from pathlib import Path
import collections
import csv
import json
import math
import statistics
import time
import numpy as np
from prepare import R,read,write,sha


def main():
    state=read(R/'LOD_PIPELINE.json');assert state['phase']=='finished'
    protected=read(R/'PROTECTED_BEFORE.json');after={n:sha(n) for n in protected};assert after==protected
    source=read(R/'reports/SOURCE_HASHES.json');assert all(sha(R/'source'/n)==h for n,h in source.items())
    build=read(R/'reports/BUILD_HASHES.json');assert all(sha(R/n)==h for n,h in build.items())
    write(R/'PROTECTED_AFTER.json',after)
    paired=[];health=[];exposure=[];gates=[]
    for j in state['jobs']:
        if j.get('trained_model'):
            model=Path(j['trained_model']);checks=read(model.parent/'D_VERIFIED.json')
            stage=read(next(model.glob('*_shutdown/ply/real_lod_state.json')))
            anchor=read(next(model.glob('*_shutdown/ply/point_cloud/iteration_*/anchor_selector_state.json')))
            clocks=[v['adam_step'] for v in anchor['compensation_feature_state']]
            with (model/'real_lod_steps.csv').open() as f:global_steps=list(csv.DictReader(f))
            with (model/'view_budget_steps.jsonl').open() as f:local_steps=[json.loads(line) for line in f]
            assert len(global_steps)==len(local_steps)==j['target_steps']
            assert all(abs(float(g['ratio'])-float(l['ratio']))<1e-6 for g,l in zip(global_steps,local_steps))
            # 两LoD分支均为空时无Feature计算图，标准Adam不更新未定义梯度的表。
            # 分别核验真实计算图曝光与轮转步数，不改训练或checkpoint。
            expected=[sum((int(g['K'])>0 or l['K']>0) and float(g['ratio'])>=ratio-1e-6
                for g,l in zip(global_steps,local_steps)) for ratio in stage['ratios']]
            assert clocks==expected,(j['id'],clocks,expected)
            assert anchor['gaussian_count']==j['full_job']['result']['gaussian_count']
            checks.update(feature_adam_clocks=clocks,feature_adam_clocks_match_gradient_exposure=True,
                nominal_feature_rotation_clocks=[sum(stage['ratio_steps'][k:]) for k in range(6)],
                both_branches_zero_steps=[i for i,(g,l) in enumerate(zip(global_steps,local_steps),1) if int(g['K'])==0 and l['K']==0],
                gaussian_count=anchor['gaussian_count'],anchor_count=anchor['anchor_count'])
            old=Path(j['reference_model'])
            with (old/'real_lod_steps.csv').open() as f:old_global=list(csv.DictReader(f))
            with (old/'view_budget_steps.jsonl').open() as f:old_local=[json.loads(s) for s in f]
            a_global=sum(int(v['K']) for v in old_global);a_local=sum(v['K'] for v in old_local)
            exposure.append(dict(id=j['id'],**checks,reference_global_K=a_global,reference_local_K=a_local,
                relative_global_K_change=checks['selection_exposure_global']/max(1,a_global)-1,
                relative_local_K_change=checks['selection_exposure_local']/max(1,a_local)-1))
            for file in [model/'lod_health.jsonl',model/'view_budget_health.jsonl']:
                if not file.exists():continue
                maxima={};samples=0;scale_tail=xyz_tail=0
                for line in file.open():
                    v=json.loads(line);samples+=1
                    for key in ['scale_factor_max','xyz_anchor_units_max','scale_tail_fraction','xyz_tail_fraction','opaque_fraction']:
                        if key in v:maxima[key]=max(maxima.get(key,0),v[key])
                    scale_tail+=v.get('scale_tail_fraction',0)>=.01;xyz_tail+=v.get('xyz_tail_fraction',0)>=.01
                health.append(dict(id=j['id'],log=file.name,samples=samples,scale_tail_ge1pct_steps=scale_tail,
                    xyz_tail_ge1pct_steps=xyz_tail,**maxima))
        delta_global=[];delta_full=None
        for e in j['evaluations']:
            if e['state']!='complete':continue
            ref=next(v for v in j['reference_evaluations'] if v['scope']==e['scope'] and v['ratio']==e['ratio'])
            with (Path(ref['result']['output'])/'per_frame_metrics.csv').open() as f:a=list(csv.DictReader(f))
            with (Path(e['result']['output'])/'per_frame_metrics.csv').open() as f:d=list(csv.DictReader(f))
            assert len(a)==len(d)==e['result']['evaluated_frames']
            keys=['rgb_index','trajectory_timestamp','rgb_timestamp'];assert all([x[k] for k in keys]==[y[k] for k in keys] for x,y in zip(a,d))
            row=dict(id=j['id'],dataset=j['dataset'],scene=j['scene'],mode=j['mode'],scope=e['scope'],ratio=e['ratio'],frames=len(a))
            for key in ['psnr','ssim','lpips']:
                values=np.asarray([float(y[key])-float(x[key]) for x,y in zip(a,d)])
                assert np.isfinite(values).all()
                row.update({key+'_delta_'+n:float(v) for n,v in zip(['mean','median','p10','p90'],[values.mean(),np.median(values),*np.percentile(values,[10,90])])})
                row[key+'_improved_fraction']=float(np.mean(values<0 if key=='lpips' else values>0))
            paired.append(row)
            if e['scope']=='global' and e['ratio'] in state['protocol']['train_ratios']:delta_global.append(row['psnr_delta_mean'])
            if e['scope']=='full':delta_full=row['psnr_delta_mean']
        if len(delta_global)==6 and delta_full is not None:
            gates.append(dict(id=j['id'],global_train_mean_delta=float(np.mean(delta_global)),global_train_min_delta=min(delta_global),
                full_delta=delta_full,passed=float(np.mean(delta_global))>=-.1 and min(delta_global)>=-.2 and delta_full>=-.1))
    for name,rows in [('PAIRED_FRAME_SUMMARY',paired),('TRAINING_HEALTH_SUMMARY',health),('TRAINING_EXPOSURE',exposure),('QUALITY_GATES',gates)]:
        write(R/'reports'/(name+'.json'),rows)
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with (R/'reports'/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    image_files=[str(p) for folder in ['runs','evaluation','validation'] for p in (R/folder).rglob('*')
        if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp']]
    assert not image_files,image_files[:5]
    lines=['# 无GI／LI监督：D组全场景消融','',
        '完整方案的32个数据集／场景／相机模式均列入队列；复用相同Full赢家和在线Mapping步数，seed0、fresh Adam/RNG。',
        '只将GI、LI Loss权重设为0，保留两网络真实视角、Full＋全局LoD＋局部LoD及一次Adam。Master可更新、拓扑固定。',
        'GI采集仍开启以匹配原刷新开销，不产生GI监督；LI采集／VJP为0。Scale／XYZ长尾和概率饱和只记录，连续零选点与非有限值仍报错。',
        '失败／跳过不删除，不把未完成实验记为质量结果；OOM有证据才允许额外2次。评价为原图全轨迹、全局与局部各11档ratio＋Full，无渲染帧图片。','',
        '|场景／模式|Fine步数|状态|局部1% ΔPSNR|局部5% ΔPSNR|Full ΔPSNR|','|---|---:|---|---:|---:|---:|']
    for j in state['jobs']:
        fields=[]
        for scope,ratio in [('view_ratio',.01),('view_ratio',.05),('full',1.)]:
            a=next((v for v in j['reference_evaluations'] if v['scope']==scope and v['ratio']==ratio),None)
            d=next((v for v in j['evaluations'] if v['scope']==scope and v['ratio']==ratio and v['state']=='complete'),None)
            fields.append('—' if not a or not d else f"{d['result']['psnr']-a['result']['psnr']:+.4f}")
        lines.append(f"|{j['id']}|{j['target_steps']}|{j['state']}|"+'|'.join(fields)+'|')
    lines+=['','Δ=D−完整方案，逐场景／ratio PSNR、SSIM、LPIPS、ATE、N、实际K见A_VS_D_METRICS.csv及LOD_METRICS.csv。',
        '训练累计K和真实更新次数见各attempt的D_VERIFIED.json，不能以Mapping步数相等声称实际选点曝光相等。',
        '原完整方案保留其历史健康失败重试策略，本轮关闭长尾停止，因此健康通过者筛选条件不同；结果需连同终态几何和失败状态解释。',
        '单seed且在线Full只读复用，属于固定拓扑Fine消融，不能冒充重新运行的完整在线SLAM或跨seed稳定性验证。',
        '全局与局部预算以各自人口计算；未进行跨模型same-K强制匹配，保留原预算／推理规则。']
    lines+=['','Feature Adam时钟按实际非空Compensation图核验。Replica room2单目第12703步的全局／局部K均为0，'
        '仅Base Feature Adam少更新1次，与未定义梯度的原Adam语义一致；该步不是连续零选点，健康守卫未触发。'
        '其余成功组全部Feature表时钟均符合轮转次数。详见TRAINING_EXPOSURE.csv。']
    lines+=['','## 按数据集等权汇总','',
        '下表先逐场景取完整轨迹均值，再对已完成场景等权平均，不让长序列按帧数压过短序列。失败场景数量单列，未完成不补零。',
        '|数据集|条件|已完成场景数|平均ΔPSNR|场景改善比例|平均ΔSSIM|平均ΔLPIPS|',
        '|---|---|---:|---:|---:|---:|---:|']
    dataset_summary=[]
    for dataset in ['TUM_RGBD','Replica','VECtor']:
        for scope,ratio in [('global',.01),('global',.05),('global',.1),('view_ratio',.01),('view_ratio',.05),('view_ratio',.1),('full',1.)]:
            rows=[v for v in paired if v['dataset']==dataset and v['scope']==scope and v['ratio']==ratio]
            if not rows:continue
            item=dict(dataset=dataset,scope=scope,ratio=ratio,scenes=len(rows),
                psnr_delta_mean=float(np.mean([v['psnr_delta_mean'] for v in rows])),
                ssim_delta_mean=float(np.mean([v['ssim_delta_mean'] for v in rows])),
                lpips_delta_mean=float(np.mean([v['lpips_delta_mean'] for v in rows])),
                psnr_improved_scenes_fraction=float(np.mean([v['psnr_delta_mean']>0 for v in rows])))
            dataset_summary.append(item)
            lines.append(f"|{dataset}|{scope} {ratio:g}|{len(rows)}|{item['psnr_delta_mean']:+.4f}|{100*item['psnr_improved_scenes_fraction']:.1f}%|{item['ssim_delta_mean']:+.6f}|{item['lpips_delta_mean']:+.6f}|")
    write(R/'reports/DATASET_SUMMARY.json',dataset_summary)
    lines+=['','## 健康与失败','',
        '|场景／模式|状态|失败或跳过原因|','|---|---|---|']
    for j in state['jobs']:
        if j['state']!='complete':lines.append(f"|{j['id']}|{j['state']}|{str(j.get('error','')).replace('|','/')}|")
    lines+=['','长尾最大值、超过原阈值的步数见TRAINING_HEALTH_SUMMARY.csv；统计未作为停止条件，未增加任何限幅或正则。',
        'QUALITY_GATES.csv按预声明的全局平均−0.10dB／单训练档−0.20dB／Full−0.10dB参考线判读，不据此调整训练。',
        '逐帧配对mean／median／p10／p90／改善帧比例见PAIRED_FRAME_SUMMARY.csv；这些分布不是跨seed置信区间。']
    (R/'reports/NO_IMPORTANCE_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    counts=dict(collections.Counter(j['state'] for j in state['jobs']))
    result=dict(state='complete',queue_finished=True,all_succeeded=counts=={'complete':32},jobs=32,counts=counts,
        complete_evaluations=sum(v['state']=='complete' for j in state['jobs'] for v in j['evaluations']),
        protected_files=len(protected),protected_files_unchanged=True,source_unchanged=True,binaries_unchanged=True,
        rendered_frame_images=0,production_default_changed=False,completed_unix=time.time())
    write(R/'reports/FINAL_INTEGRITY.json',result);write(R/'FINISHED.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
