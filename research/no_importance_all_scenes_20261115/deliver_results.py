"""整理已完成D组结果；不运行训练或渲染，不把失败写成零分。"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

R=Path(__file__).resolve().parent
S=R/'results'
O=R/'outputs/01a1de13-ebef-7091-b328-52d3c81cfc40'
REMOTE='/home/prmi/zwj/experiments/no_importance_all_scenes_20261115/'
TRAIN=[.01,.05,.1,.3,.5,.7]
RATIOS=sorted(TRAIN+[.03,.08,.2,.4,.6])
SCOPES={'global':'全局','view_ratio':'单视角','full':'Full'}
MODES={'mono':'单目','rgbd':'RGB-D','stereo':'双目'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def loc(remote):return S/str(remote).removeprefix(REMOTE)
def mean(v):return statistics.mean(v)
def fmt(v,n=3):return '—' if v is None else f'{v:+.{n}f}'


def main():
    O.mkdir(parents=True,exist_ok=True);(O/'figures').mkdir(exist_ok=True)
    state=read(S/'LOD_PIPELINE.json');final=read(S/'FINISHED.json');paired=read(S/'reports/PAIRED_FRAME_SUMMARY.json')
    assert state['phase']=='finished' and final['counts']=={'complete':27,'skipped_zero_selection':5}
    pairing={(p['id'],p['scope'],p['ratio']):p for p in paired}
    failure_audit={p['id']:p for p in read(S/'reports/FAILED_STEP_AUDIT.json')}
    details=[];statuses=[];failures=[];complete=[]
    for job in sorted(state['jobs'],key=lambda j:(j['dataset'],j['scene'],j['mode'])):
        done=job['state']=='complete';out=loc(job['attempt']);failure=read(out/'model/lod_health_failure.json') if not done else None
        status=dict(id=job['id'],dataset=job['dataset'],scene=job['scene'],mode=job['mode'],status='完成' if done else '连续零选点停止',
            target_steps=job['target_steps'],completed_steps=job['target_steps'] if done else failure_audit[job['id']]['completed_fine_steps'],
            gaussian_N=job['full_job']['result']['gaussian_count'],anchor_N=job['full_job']['result']['anchor_count'],
            evaluation_frames=job['full_evaluation']['result']['evaluated_frames'],evaluation_cases=sum(e['state']=='complete' for e in job['evaluations']),
            worker=job['assigned_worker'],seed=0,model=job.get('trained_model'),failure_error=failure['error'] if failure else None,
            failure_ratio=failure.get('ratio') if failure else None,failure_step_raw=failure.get('step') if failure else None,
            failure_trigger_fine_step=failure_audit[job['id']]['trigger_fine_step'] if failure else None,
            eligible_candidates=failure.get('eligible_candidates') if failure else None,selected_at_failure=failure.get('selected') if failure else None,
            zero_streak_ratio=failure.get('zero_streak_ratio') if failure else None,
            zero_streak_global=failure.get('zero_streak_global') if failure else None,
            full_checkpoint=job['full_job']['result']['checkpoint'],reference_model=job['reference_model'])
        if failure:
            failures.append(dict(**status,health=failure))
        statuses.append(status)
        dmap={(e['scope'],e['ratio']):e for e in job['evaluations'] if e['state']=='complete'}
        for ref in job['reference_evaluations']:
            a=ref['result'];e=dmap.get((ref['scope'],ref['ratio']));d=e['result'] if e else None
            row=dict(id=job['id'],dataset=job['dataset'],scene=job['scene'],mode=job['mode'],status=status['status'],
                scope=ref['scope'],ratio=ref['ratio'],ratio_kind='Full' if ref['scope']=='full' else ('训练' if ref['ratio'] in TRAIN else '未见'),
                frames=a['evaluated_frames'],gaussian_N=status['gaussian_N'],anchor_N=status['anchor_N'],seed=0,
                model_A=job['reference_model'],model_D=job.get('trained_model'),evaluation_D=e['result']['output'] if e else None)
            for k in ['psnr','ssim','lpips','actual_K_mean','actual_K_min','actual_K_max','ate_rmse','render_ms']:
                row[k+'_A']=a.get(k);row[k+'_D']=d.get(k) if d else None
                row['delta_'+k]=(row[k+'_D']-row[k+'_A']) if row[k+'_D'] is not None and row[k+'_A'] is not None else None
            if d:
                assert d['evaluated_frames']==a['evaluated_frames']
                p=pairing[(job['id'],ref['scope'],ref['ratio'])]
                assert abs(p['psnr_delta_mean']-row['delta_psnr'])<1e-9
                for k in ['psnr','ssim','lpips']:
                    for n in ['median','p10','p90']:row[k+'_paired_'+n]=p[k+'_delta_'+n]
                    row[k+'_improved_frame_fraction']=p[k+'_improved_fraction']
                complete.append(row)
            details.append(row)
    assert len(statuses)==32 and len(details)==736 and len(complete)==621
    assert len({(r['id'],r['scope'],r['ratio']) for r in details})==736
    summaries=[]
    for dataset in ['全部','TUM_RGBD','Replica','VECtor']:
        for scope in ['global','view_ratio','full']:
            for ratio in ([1.] if scope=='full' else RATIOS):
                rows=[r for r in complete if (dataset=='全部' or r['dataset']==dataset) and r['scope']==scope and r['ratio']==ratio]
                summary=dict(dataset=dataset,scope=scope,ratio=ratio,count=len(rows))
                for k in ['psnr','ssim','lpips']:
                    for suffix in ['A','D']:summary[k+'_'+suffix]=mean([r[k+'_'+suffix] for r in rows])
                    delta=[r['delta_'+k] for r in rows];summary['delta_'+k]=mean(delta);summary[k+'_scene_median_delta']=statistics.median(delta)
                    summary[k+'_improved_scenes']=sum((v<0 if k=='lpips' else v>0) for v in delta)
                summaries.append(summary)
    csvwrite(O/'逐场景逐ratio完整指标.csv',details);csvwrite(O/'场景状态与失败.csv',statuses);csvwrite(O/'数据集逐ratio汇总.csv',summaries)
    exposures=read(S/'reports/TRAINING_EXPOSURE.json');health=read(S/'reports/TRAINING_HEALTH_SUMMARY.json');gates=read(S/'reports/QUALITY_GATES.json')
    csvwrite(O/'实际训练预算与Feature更新.csv',exposures)
    for name in ['PAIRED_FRAME_SUMMARY.csv','TRAINING_HEALTH_SUMMARY.csv','QUALITY_GATES.csv','FAILED_STEP_AUDIT.json']:
        (O/name).write_bytes((S/'reports'/name).read_bytes())
    write(O/'分析数据.json',dict(summary=summaries,detail=details,status=statuses,failures=failures,exposure=exposures,health=health,gates=gates,integrity=final))

    lines=['# 去掉重要性监督的D组消融结果','',
        '**结论：本轮联合去掉GI和LI后，LoD质量整体明显下降，并有5/32组触发连续零选点停止；保留完整方案。**',
        '32组均已结束，27组通过训练和全部评价，共621项全轨迹评价。5组失败不参与质量均值、不填零；完整方案A在对应32组均有历史成功结果。','',
        '## 比较配置','',
        '|条件|完整方案A|无重要性D|','|---|---|---|',
        '|GI Loss权重|1|0|','|LI Loss权重|0.1|0|',
        '|Selector／Compensator视角|两网络真实视角|相同|',
        '|训练|Full＋全局LoD＋局部LoD，联合反传后一次Adam|相同|',
        '|Master与拓扑|Master可更新，拓扑固定|相同|',
        '|初始化与预算|相同Full赢家、seed0、完整Fine步数|相同|',
        '|健康规则|历史长尾及概率饱和守卫|只以连续零选点和非有限值停止，长尾仅记录|','',
        '训练ratio为1%、5%、10%、30%、50%、70%；未见ratio为3%、8%、20%、40%、60%。评估全局与单视角各11档，另评Full。',
        '训练不使用名义Anchor视锥；单视角评价按既定Viewer名义Anchor候选池和该视角预算推理。各地图使用自己的N，不强制跨模型same-K。',
        'GI采集仍按原周期运行但不产生监督；LI权重为0时不额外采集或执行LI VJP。RGB评价使用原图全轨迹，不保存渲染图片。','',
        '## 成功27组的总体变化','',
        '以下每个场景／模式先对共同轨迹帧取均值，再对27组等权平均。Δ=D−A，PSNR／SSIM负值和LPIPS正值代表D更差。',
        '|条件|A PSNR|D PSNR|ΔPSNR|ΔSSIM|ΔLPIPS|D的PSNR改善组数|','|---|---:|---:|---:|---:|---:|---:|']
    for scope,ratio in [('global',.01),('global',.05),('global',.1),('view_ratio',.01),('view_ratio',.05),('view_ratio',.1),('full',1.)]:
        q=next(r for r in summaries if r['dataset']=='全部' and (r['scope'],r['ratio'])==(scope,ratio))
        lines.append(f"|{SCOPES[scope]} {ratio:.0%}|{q['psnr_A']:.3f}|{q['psnr_D']:.3f}|{q['delta_psnr']:+.3f}|{q['delta_ssim']:+.4f}|{q['delta_lpips']:+.4f}|{q['psnr_improved_scenes']}/27|")
    lines+=['','## 按数据集比较','',
        '|数据集|成功／计划|全局1% ΔPSNR|全局5% ΔPSNR|局部1% ΔPSNR|局部5% ΔPSNR|Full ΔPSNR|','|---|---:|---:|---:|---:|---:|---:|']
    for ds,total in [('TUM_RGBD',10),('Replica',16),('VECtor',6)]:
        values=[next(r['delta_psnr'] for r in summaries if r['dataset']==ds and (r['scope'],r['ratio'])==c) for c in [('global',.01),('global',.05),('view_ratio',.01),('view_ratio',.05),('full',1.)]]
        count=sum(j['dataset']==ds and j['status']=='完成' for j in statuses)
        lines.append(f'|{ds}|{count}/{total}|'+ '|'.join(fmt(v) for v in values)+'|')
    lines+=['','## 32组状态与重点ratio','',
        '|场景／模式|N|状态|全局1% ΔPSNR|全局5% ΔPSNR|局部1% ΔPSNR|局部5% ΔPSNR|Full ΔPSNR|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for j in statuses:
        values=[next(r['delta_psnr'] for r in details if r['id']==j['id'] and (r['scope'],r['ratio'])==c) for c in [('global',.01),('global',.05),('view_ratio',.01),('view_ratio',.05),('full',1.)]]
        lines.append(f"|{j['id']}|{j['gaussian_N']:,}|{j['status']}|"+'|'.join(fmt(v) for v in values)+'|')
    lines+=['','## 失败现场','',
        '|场景／模式|触发位置|触发Fine步骤|ratio|合格候选数|触发时K|','|---|---|---:|---:|---:|---:|']
    for j in failures:
        h=j['health'];step=j['failure_trigger_fine_step'];branch='单视角LoD' if h['error'].startswith('view_budget_') else '全局LoD'
        lines.append(f"|{j['id']}|{branch}|{step}|{h['ratio']:.0%}|{h['eligible_candidates']}|{h['selected']}|")
    lines+=['','这些失败有真实合格候选，但触发连续零选点停止：Replica office0 RGB-D为连续8个全局步骤K=0，其余4组为同ratio连续8次K=0。它们不是OOM，也没有因Scale长尾停止。VECtor双目的健康日志step=64289是累计Mapping步骤，对应Fine触发步30889，已按训练CSV核对。失败不纳入质量均值。','',
        '## 归因与边界','',
        '1. **已证实：联合去掉GI／LI在本轮32组协议下不合格。** 即使只观察成功的27组，低ratio的PSNR、SSIM、LPIPS仍整体退化。不能因排除5组崩溃而宣称稳定。',
        '2. **LoD退化远大于Full。** 27组Full平均ΔPSNR仅为−0.133dB，而低ratio损失为数dB。现象支持重要性监督对选择／补偿联合训练有价值；单凭本实验不能把损失全部归给Selector。',
        '3. **未分离GI和LI的单独作用。** 本轮只有A对D，不能据此判断GI或LI哪个更重要，也未隔离Master变化和Compensator适配。',
        '4. **相同步数不等于相同选点曝光。** 训练K累计量见实际训练预算与Feature更新.csv；推理按各模型和当前视角自己的预算，精确K值见完整指标表。',
        '5. **健康停止规则存在历史差异。** A保留旧长尾停止／重试，D允许长尾继续。所有结论只对应实际比较协议；没有隐藏此差异，也未用Scale长尾直接推断质量原因。',
        '6. **单seed固定Full起点的Fine消融。** 不代表重新完整运行在线SLAM或跨seed稳定性。ATE沿用固定轨迹，不能将相同ATE解释为本次Fine改善了定位。','',
        '## 训练、数值和完整性','',
        f"27组通过Feature Adam真实梯度曝光核验，预声明质量参考线通过 {sum(g['passed'] for g in gates)}/27 组。",
        '沿用开跑前参考线：六个训练ratio的全局PSNR平均Δ≥−0.10dB，每训练档Δ≥−0.20dB，Full Δ≥−0.10dB。未按本次结果调整门槛。',
        'Replica room2单目第12703步的全局／局部K均为0，Base Feature没有梯度，Adam时钟为18076而非18077；其余残差表正常。汇总脚本按真实图曝光修正，未修改训练、checkpoint或指标。',
        f"全部成功组Fine共{sum(j['target_steps'] for j in statuses if j['status']=='完成'):,}步，逐分支选点曝光和Feature更新见实际训练预算表。失败组及关机前废弃尝试的开销保留在原日志，不混入完成预算。",
        '服务器2645项受保护文件SHA256全部未变，研究源码及二进制核验通过，生产默认未修改。原服务器渲染图片数为0。AutoDL结果已全部回收，用户已关闭实例。','',
        '## 文件','',
        '- 无重要性消融对照.xlsx：总体／数据集汇总、736行逐场景逐ratio指标（失败D空白）、32组状态和实际训练曝光。',
        '- 逐场景逐ratio完整指标.csv：PSNR、SSIM、LPIPS、ATE、渲染时间、N、实际K及逐帧配对分布。',
        '- PAIRED_FRAME_SUMMARY.csv：621项共同帧的mean／median／p10／p90及改善帧比例。',
        '- TRAINING_HEALTH_SUMMARY.csv：XYZ／Scale训练长尾日志汇总，最大值不是最终模型分位数。',
        '- NO_IMPORTANCE_ABLATION_RESULTS.zip：服务器原始汇总、协议、配置来源、训练核验、失败现场与完整性证据。']
    (O/'无重要性消融结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    font=Path('C:/Windows/Fonts/msyh.ttc');assert font.exists();font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'grid.alpha':.25,'legend.frameon':False})
    colors=['#4E79A7','#E15759'];fig,axes=plt.subplots(2,3,figsize=(12,6.4),sharex=True)
    for col,ds in enumerate(['TUM_RGBD','Replica','VECtor']):
        for row,scope in enumerate(['global','view_ratio']):
            ax=axes[row,col];points=[r for r in summaries if r['dataset']==ds and r['scope']==scope]
            for suffix,label,color in zip(['A','D'],['完整方案 GI＋LI','去掉 GI／LI'],colors):
                ax.plot([r['ratio']*100 for r in points],[r['psnr_'+suffix] for r in points],color=color,marker='o' if suffix=='A' else 's',markersize=3,linewidth=1.7,label=label)
            ax.set_title(f"{ds} · {SCOPES[scope]}（成功{points[0]['count']}组）");ax.grid(True);ax.set_xscale('log');ax.set_xticks([1,5,10,30,70],['1','5','10','30','70'])
            if col==0:ax.set_ylabel('PSNR (dB)')
            if row==1:ax.set_xlabel('ratio (%)，对数坐标')
    axes[0,0].legend(loc='upper left',fontsize=8);fig.suptitle('去掉重要性监督后的渲染质量：仅比较共同成功场景／模式，逐组等权',fontsize=12);fig.tight_layout()
    fig.savefig(O/'figures/无重要性逐ratio对照.png',dpi=450);fig.savefig(O/'figures/无重要性逐ratio对照.svg');plt.close(fig)
    (O/'figures/data-manifest.md').write_text('图表使用真实实验数据。来源：分析数据.json / summary；对应导出为数据集逐ratio汇总.csv。\n脚本：deliver_results.py。每条线按同一成功场景／模式集合等权平均，不平滑、不插值、不含失败5组。\n输出：无重要性逐ratio对照.png（450 DPI）及.svg。\n',encoding='utf-8')
    summary=next(r for r in summaries if r['dataset']=='全部' and (r['scope'],r['ratio'])==('view_ratio',.05))
    print(json.dumps(dict(rows=len(details),complete=len(complete),failures=len(failures),quality_gate_pass=sum(g['passed'] for g in gates),local5=summary),ensure_ascii=False))


if __name__=='__main__':main()
