"""三组重跑结束后核验模型/指标、原29组完整性并更新中文最终报告。"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import datetime
import math
import statistics
import time
import run_batch as b
import supervisor

R=b.R;p=b.p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',default='FINAL_FAILURES_RERUN_REQUEST.json')
    args=parser.parse_args()
    state=p.read(R/'LOD_PIPELINE.json')
    assert state['phase']=='finished' and not state['parallel_workers']
    assert not supervisor.running(state)
    assert Path(args.request).name==args.request
    request=p.read(R/'reports'/args.request)
    first=request.get('first_attempt',request.get('next_attempt',7))
    last=request.get('last_attempt',first)
    old=p.read(Path(request['backup'])/'LOD_PIPELINE.json')
    current={j['id']:j for j in state['jobs']}
    for job in old['jobs']:
        if job['state']=='complete':assert current[job['id']]==job,job['id']
    old_hashes=p.read(Path(request['checkpoint_hashes']))
    assert all(p.sha(Path(n))==h for n,h in old_hashes.items())
    if request.get('correction_protected_model_hashes'):
        assert all(p.sha(Path(n))==h for n,h in p.read(Path(request['correction_protected_model_hashes'])).items())
    complete=[j for j in state['jobs'] if j['state']=='complete']
    failed=[j for j in state['jobs'] if j['state'] in ['skipped_health','skipped_oom']]
    assert len(complete)+len(failed)==32
    evaluated=0
    for job in complete:
        b.verify_training(Path(job['trained_model']).parent,job['target_steps'])
        assert len(job['evaluations'])==23
        for item in job['evaluations']:
            assert item['state']=='complete'
            values=item['result'];summary=p.read(Path(values['output'])/'metric_summary.json')
            assert summary['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
            assert all(math.isfinite(values[k]) and values[k]==summary[k]
                       for k in ['psnr','ssim','lpips','ate_rmse'])
            evaluated+=1
    with (R/'reports/LOD_METRICS.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==evaluated and all(v['evaluation_state']=='complete' for v in rows)
    before=p.read(R/'PROTECTED_BEFORE.json');assert p.read(R/'PROTECTED_AFTER.json')==before
    assert all(p.sha(Path(n))==h for values in before.values() for n,h in values.items())
    assert all(p.sha(R/n)==h for n,h in p.read(R/'reports/BUILD_HASHES.json').items())
    assert not any(path.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']
                   for folder in ['runs','evaluation','validation'] for path in (R/folder).rglob('*'))
    base={j['id']:j for name in ['dual_view_li_best_full_20261107','dual_view_li_vector_best_full_20261107']
          for j in p.read(R.parent/'baseline'/name/'LOD_PIPELINE.json')['jobs']}
    def result(job,scope,ratio):
        return next(e['result'] for e in job['evaluations'] if e['scope']==scope and abs(e['ratio']-ratio)<1e-7)
    comparisons=[]
    for dataset in ['ALL','Replica','TUM_RGBD','VECtor']:
        cohort=[j for j in complete if dataset=='ALL' or j['dataset']==dataset]
        for scope in ['global','view_ratio','full']:
            for ratio in ([1.] if scope=='full' else [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]):
                pairs=[(result(j,scope,ratio),result(base[j['id']],scope,ratio)) for j in cohort]
                record=dict(dataset=dataset,scope=scope,ratio=ratio,n=len(pairs))
                for key in ['psnr','ssim','lpips']:
                    record['baseline_'+key]=statistics.mean(z[key] for a,z in pairs)
                    record['frozen_'+key]=statistics.mean(a[key] for a,z in pairs)
                    record['delta_'+key]=statistics.mean(a[key]-z[key] for a,z in pairs)
                record['psnr_wins']=sum(a['psnr']>z['psnr'] for a,z in pairs)
                record['N_equal']=all(a['gaussian_count']==z['gaussian_count'] for a,z in pairs)
                record['actual_K_equal']=all(all(a[k]==z[k] for k in ['actual_K_mean','actual_K_min','actual_K_max'])
                                             for a,z in pairs)
                comparisons.append(record)
    targets=[]
    for name in request['targets']:
        job=current[name];record=dict(id=name,state=job['state'],health_failures=job['health_failures'],
            oom_failures=job.get('oom_failures',0),evaluations_completed=sum(e['state']=='complete' for e in job['evaluations']))
        bounds=request.get('attempt_ranges',{}).get(name,dict(first=first,last=last))
        attempt_first,attempt_last=bounds['first'],bounds['last']
        record['attempt_range']=[attempt_first,attempt_last]
        attempts=[h for h in job['attempt_history']
                  if attempt_first<=int(Path(h['output']).name.split('_')[-1])<=attempt_last]
        assert attempts and len(attempts)<=attempt_last-attempt_first+1
        record['attempts_this_round']=attempts
        if job.get('trained_model'):record['trained_model']=job['trained_model']
        targets.append(record)
    zone=datetime.timezone(datetime.timedelta(hours=8))
    fmt=lambda value:datetime.datetime.fromtimestamp(value,zone).isoformat()
    receipt=dict(audit_passed=True,all_32_groups_succeeded=len(complete)==32,queue_finished=True,
        finished_hongkong=fmt((R/'FINISHED.json').stat().st_mtime),checked_hongkong=fmt(time.time()),
        successful_groups=len(complete),failed_groups=len(failed),completed_evaluations=evaluated,
        csv_rows=len(rows),models_verified=len(complete),protected_files_verified=sum(map(len,before.values())),
        original_29_groups_unchanged=True,original_model_files_verified=len(old_hashes),images_saved=0,
        rerun_targets=targets,comparisons=comparisons,health_policy=request.get('health_policy'),
        counts_by_dataset={ds:dict(Counter(j['state'] for j in state['jobs'] if j['dataset']==ds))
                           for ds in ['Replica','TUM_RGBD','VECtor']})
    p.write(R/'reports/FINAL_AUDIT.json',receipt)
    result_name=args.request.replace('_REQUEST.json','_RESULT.json')
    p.write(R/'reports'/result_name,{k:v for k,v in receipt.items() if k!='comparisons'})
    lines=['# Master固定消融最终报告（失败组重新运行后）','',
        f'队列于香港时间{receipt["finished_hongkong"]}结束；{len(complete)}组成功，{len(failed)}组最终失败，'
        f'完成{evaluated}项全轨迹评价。','',
        '本轮各组尝试范围见下表；首次成功即停止重训。'
        '历史失败和原先29组结果保留；训练步数、'
        'seed0、网络和LI=0.1保持不变。Master六类参数在Full、全局LoD和局部LoD三路均固定。',
        '',request.get('health_policy','原健康阈值不变。'),
        '','|本次重跑组|尝试范围|最终状态|本轮尝试数|累计健康失败|新增评价|','|---|---|---|---:|---:|---:|']
    for job in targets:
        span='–'.join(map(str,job['attempt_range']))
        lines.append(f'|{job["id"]}|{span}|{job["state"]}|{len(job["attempts_this_round"])}|{job["health_failures"]}|{job["evaluations_completed"]}|')
    lines+=['','只有训练完成且冻结验证通过的模型才进入评价；停止策略见本轮协议。',
        '','|Ratio|全局LoD PSNR差值(dB)|局部LoD PSNR差值(dB)|','|---|---:|---:|']
    for ratio in [.01,.05,.1,.3,.5,.7]:
        a=next(v for v in comparisons if v['dataset']=='ALL' and v['scope']=='global' and v['ratio']==ratio)
        z=next(v for v in comparisons if v['dataset']=='ALL' and v['scope']=='view_ratio' and v['ratio']==ratio)
        lines.append(f'|{ratio:.0%}|{a["delta_psnr"]:+.4f}|{z["delta_psnr"]:+.4f}|')
    lines+=['',f'以上为{len(complete)}个成功场景/相机模式的等权配对平均，差值=冻结Master减去可更新Master。'
        '失败场景不混入质量均值；存在失败与重复尝试的幸存样本偏差。',
        '',f'原29组模型文件{len(old_hashes)}项哈希及队列记录均不变；全部成功模型、指标与'
        f'{sum(map(len,before.values()))}个原Full、baseline和生产受保护文件验证通过。未保存渲染图片。',
        '','数据：reports/LOD_METRICS.csv；逐帧：evaluation/；完整审计、逐数据集逐ratio均值：reports/FINAL_AUDIT.json。',
        '','旧终态报告与审计：'+request['backup'],
        '','生产默认未替换，未commit或push。']
    (R/'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print({k:v for k,v in receipt.items() if k not in ['comparisons','rerun_targets']},flush=True)
    print('RERUN_TARGETS',[(j['id'],j['state'],len(j['attempts_this_round'])) for j in targets],flush=True)


if __name__=='__main__':main()
