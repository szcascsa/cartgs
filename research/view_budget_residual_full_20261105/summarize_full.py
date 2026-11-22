"""三组共同帧配对汇总，残差关闭诊断单列，审计真实曝光与历史对照限制。"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
OUT=ROOT/'evaluation/reports'
TRAIN=[.01,.05,.1,.3,.5,.7];UNSEEN=[.03,.08,.2,.4,.6]
GROUPS=['baseline','view_no_li','view_li01']
NAMES={'baseline':'原联合无LI','view_no_li':'视角残差无LI','view_li01':'视角残差＋LI=0.1'}
COMPARISONS=[('baseline','view_no_li'),('baseline','view_li01'),('view_no_li','view_li01')]


def write(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def split(values):
    for label in ['all','bank','nonbank']:
        chosen=[r for r in values if label=='all' or (r['bank']=='True')==(label=='bank')]
        assert len(chosen)==dict(all=792,bank=72,nonbank=720)[label]
        yield label,chosen


def stats(values,key,item,prefix=None):
    a=np.array([float(r[key]) for r in values]);assert np.isfinite(a).all()
    prefix=key if prefix is None else prefix
    for label,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:
        item[prefix+'_'+label]=float(value)


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    with (OUT/'PER_VIEW.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==88704
    cases=defaultdict(list)
    for r in rows:cases[(r['group'],r['case'])].append(r)
    summaries=[]
    for (group,case),values in cases.items():
        for partition,chosen in split(values):
            item={k:values[0][k] for k in ['group','case','scope','ratio','ratio_split','diagnostic']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']:
                stats(chosen,key,item)
            item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in chosen);summaries.append(item)
    write('SUMMARY.csv',summaries);(OUT/'SUMMARY.json').write_text(json.dumps(summaries,indent=2))
    lookup={(r['group'],r['case'],int(r['frame'])):r for r in rows};assert len(lookup)==len(rows)
    pairs=[]
    for reference,group in COMPARISONS:
        for row in rows:
            if row['group']!=group or row['diagnostic']=='True':continue
            ref=lookup[(reference,row['case'],int(row['frame']))]
            assert row['image']==ref['image'] and row['timestamp']==ref['timestamp']
            item={k:row[k] for k in ['case','scope','ratio','ratio_split','frame','bank']}
            item.update(reference=reference,group=group)
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:
                item['delta_'+key]=float(row[key])-float(ref[key])
            item['same_selection']=row['selection_sha256']==ref['selection_sha256'];pairs.append(item)
    write('PAIRED_PER_VIEW.csv',pairs)
    paired_cases=defaultdict(list)
    for r in pairs:paired_cases[(r['reference'],r['group'],r['case'])].append(r)
    paired_summary=[]
    for (reference,group,case),values in paired_cases.items():
        for partition,chosen in split(values):
            item={k:values[0][k] for k in ['reference','group','case','scope','ratio','ratio_split']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:stats(chosen,'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in chosen]))
            item['same_K_fraction']=float(np.mean([r['delta_K']==0 for r in chosen]));paired_summary.append(item)
    write('PAIRED_SUMMARY.csv',paired_summary);(OUT/'PAIRED_SUMMARY.json').write_text(json.dumps(paired_summary,indent=2))
    index={(r['group'],r['case'],r['partition']):r for r in summaries}
    paired_index={(r['reference'],r['group'],r['case'],r['partition']):r for r in paired_summary}
    gates=[]
    for ref,group in COMPARISONS:
        d=lambda scope,r:paired_index[(ref,group,f'{scope}_{r:g}','all')]['delta_psnr_mean']
        g=dict(reference=ref,group=group,global_train_mean_delta=float(np.mean([d('global',r) for r in TRAIN])),
            global_train_min_delta=min(d('global',r) for r in TRAIN),full_delta=paired_index[(ref,group,'full','all')]['delta_psnr_mean'],
            view_train_mean_delta=float(np.mean([d('view',r) for r in TRAIN])),
            view_unseen_mean_delta=float(np.mean([d('view',r) for r in UNSEEN])),
            view_1pct_delta=d('view',.01),view_5pct_delta=d('view',.05))
        g['quality_guard_passed']=g['global_train_mean_delta']>=-.1 and g['global_train_min_delta']>=-.2 and g['full_delta']>=-.1
        gates.append(g)
    (OUT/'QUALITY_GATES.json').write_text(json.dumps(gates,indent=2))

    diagnostic=[]
    for group in GROUPS[1:]:
        for off,on in [(f'view_off_{r:g}',f'view_{r:g}') for r in [.01,.05,.1]]+[(f'points_off_{b}',f'points_{b}') for b in [1000,2000]]:
            values=[]
            for frame in range(792):
                a,b=lookup[(group,off,frame)],lookup[(group,on,frame)]
                assert a['K']==b['K'],'同checkpoint开关残差改变提交K'
                v=dict(bank=a['bank'])
                for key in ['psnr','ssim','lpips','contributing_K']:v['delta_'+key]=float(b[key])-float(a[key])
                values.append(v)
            for partition,chosen in split(values):
                item=dict(group=group,case=on,partition=partition,frames=len(chosen),same_K=True)
                for key in ['psnr','ssim','lpips','contributing_K']:stats(chosen,'delta_'+key,item)
                item['psnr_improved_fraction']=float(np.mean([v['delta_psnr']>0 for v in chosen]));diagnostic.append(item)
    write('RESIDUAL_ON_OFF.csv',diagnostic);(OUT/'RESIDUAL_ON_OFF.json').write_text(json.dumps(diagnostic,indent=2))

    training=[]
    status=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    for group in GROUPS:
        folder=PREVIOUS/'runs/full_no_li' if group=='baseline' else ROOT/'runs'/f'full_{group}'
        with (folder/'real_lod_steps.csv').open() as stream:steps=list(csv.DictReader(stream))
        local=[json.loads(s) for s in (folder/'view_budget_steps.jsonl').read_text().splitlines()]
        fine=json.loads((folder/'23560_shutdown/ply/real_lod_state.json').read_text())
        assert len(steps)==len(local)==11580 and all(v['adam_commits_this_step']==1 and not v['strict_topk'] for v in local)
        item=dict(group=group,reused=group=='baseline',seconds=status['groups'][group]['seconds'],
            full_renders=fine['full_training_renders'],global_renders=fine['lod_training_renders'],
            local_renders=fine['view_budget_training_renders'],gi_refresh_renders=fine['importance_full_renders'],
            li_captures=fine['view_budget_li_full_renders'],li_vjps=fine['view_budget_li_backwards'],
            zero_global_steps=sum(int(v['K'])==0 for v in steps),zero_local_steps=sum(v['K']==0 for v in local),
            nonfinite=fine['lod_nonfinite'],global_K_total=sum(int(v['K']) for v in steps),local_K_total=sum(v['K'] for v in local),
            mean_mapping_ms=float(np.mean([float(v['ms']) for v in steps])))
        item['late_per_ratio']=[]
        for ratio in TRAIN:
            chosen=[v for v in local[-2000:] if abs(v['ratio']-ratio)<1e-6]
            v=dict(ratio=ratio,samples=len(chosen),K_mean=float(np.mean([r['K'] for r in chosen])),
                relative_budget_error=float(np.mean([abs(r['K']-r['target_K'])/max(1.,r['target_K']) for r in chosen])))
            for key in ['photo_loss','li_loss']:v[key+'_mean']=float(np.mean([r[key] for r in chosen]))
            item['late_per_ratio'].append(v)
        for key in ['li_global_label_disagreement','li_selected_mass_fraction','li_pool_full_contribution_fraction','li_selector_grad_norm']:
            chosen=[v for v in local[-2000:] if key in v]
            if chosen:
                item[key]=dict(samples=len(chosen));stats(chosen,key,item[key])
        training.append(item)
    (ROOT/'reports/TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
    metric=lambda group,case,key='psnr',part='all':index[(group,case,part)][key+'_mean']
    lines=['# fr1_xyz：小型视角残差完整训练质量对照', '',
        '已有原联合方案无LI模型及26928条逐帧指标按用户要求只读复用。本轮只训练视角残差无LI、视角残差＋LI=0.1；两组从同一未训练Full11980开始，各11580步。没有继续36步验收模型，也没有重训或重渲染已有基线。', '',
        '本轮未得到适合替换原基线的方案。加入LI明显改善了无LI视角模型，但相对原联合基线，1%局部PSNR、SSIM和LPIPS仍退化，固定1k也下降。30%及部分未见ratio有收益，不能概括为低预算整体改善。保留研究实现和模型，生产默认不切换。', '',
        f'相对已有baseline，视角残差无LI的局部六训练档平均变化{gates[0]["view_train_mean_delta"]:+.4f}dB，1%/5%分别{gates[0]["view_1pct_delta"]:+.4f}/{gates[0]["view_5pct_delta"]:+.4f}dB；视角残差＋LI的局部均值{gates[1]["view_train_mean_delta"]:+.4f}dB，1%/5%分别{gates[1]["view_1pct_delta"]:+.4f}/{gates[1]["view_5pct_delta"]:+.4f}dB。两组全局质量筛查分别为{gates[0]["quality_guard_passed"]}/{gates[1]["quality_guard_passed"]}。这些是本场景单seed结果，不能直接推广为稳定收益。', '',
        '## 主要结果', '', '| 比较 | 局部训练六档ΔPSNR | 局部未见五档ΔPSNR | 局部1% Δ | 局部5% Δ | 全局六档Δ | Full Δ | 质量筛查 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for g in gates:
        lines.append(f'| {NAMES[g["group"]]} − {NAMES[g["reference"]]} | {g["view_train_mean_delta"]:+.4f} | {g["view_unseen_mean_delta"]:+.4f} | {g["view_1pct_delta"]:+.4f} | {g["view_5pct_delta"]:+.4f} | {g["global_train_mean_delta"]:+.4f} | {g["full_delta"]:+.4f} | {g["quality_guard_passed"]} |')
    lines+=['', '单位dB。相对已有baseline的预注册筛查：全局训练六档平均≥−0.10dB、任何训练档≥−0.20dB、Full≥−0.10dB。相邻两个新模型的筛查列仅供同阈值参考。单seed不足以证明稳定收益或统计非劣，不能用未见档正均值掩盖1%/5%下降。', '',
        '## 局部ratio PSNR', '', '| ratio | 类型 | 原联合无LI | 视角无LI | 视角＋LI | 视角＋LI−baseline | 视角＋LI−视角无LI |', '|---:|---|---:|---:|---:|---:|---:|']
    for r in sorted(TRAIN+UNSEEN):
        a,b,c=[metric(g,f'view_{r:g}') for g in GROUPS]
        lines.append(f'| {r:g} | {"训练" if r in TRAIN else "未见"} | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} |')
    for scope,title in [('view','局部'),('global','全局')]:
        lines+=['',f'## {title} SSIM / LPIPS', '', '| ratio | baseline SSIM | 视角无LI SSIM | 视角＋LI SSIM | baseline LPIPS | 视角无LI LPIPS | 视角＋LI LPIPS |', '|---:|---:|---:|---:|---:|---:|---:|']
        for r in sorted(TRAIN+UNSEEN):
            values=[metric(g,f'{scope}_{r:g}',k) for k in ['ssim','lpips'] for g in GROUPS]
            lines.append('| '+f'{r:g} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines+=['', 'SSIM越高越好，LPIPS越低越好。photo L1、Bank/non-Bank、median/p10/p90和改善视角比例均保存在SUMMARY与PAIRED_SUMMARY CSV。', '',
        '## 全局与Full PSNR', '', '| ratio | baseline | 视角无LI | 视角＋LI |', '|---:|---:|---:|---:|']
    for case,label in [(f'global_{r:g}',f'{r:g}') for r in sorted(TRAIN+UNSEEN)]+[('full','Full')]:
        lines.append('| '+label+' | '+' | '.join(f'{metric(g,case):.4f}' for g in GROUPS)+' |')
    lines+=['', '## 固定点数局部质量', '', '| 请求预算 | baseline PSNR | 视角无LI PSNR | 视角＋LI PSNR | 三组实际K均值 | 三组逐帧K相同占比 |', '|---:|---:|---:|---:|---|---:|']
    for budget in [1000,2000,3000,4000,5000,10000]:
        case=f'points_{budget}';q=[metric(g,case) for g in GROUPS];k=[metric(g,case,'K') for g in GROUPS]
        same=float(np.mean([len({lookup[(g,case,f)]['K'] for g in GROUPS})==1 for f in range(792)]))
        lines.append(f'| {budget:,} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {k[0]:.1f}/{k[1]:.1f}/{k[2]:.1f} | {same:.1%} |')
    lines+=['', '实际K=min(B,E)；10k可能候选不足且ratio超训练范围，均按原规则记录。固定K不等于同选点身份、同地图或同最终Alpha贡献数。', '',
        '## 同checkpoint关闭残差诊断', '', '| 模型 | 局部配置 | 残差开−关ΔPSNR | ΔSSIM | ΔLPIPS | 改善视角比例 |', '|---|---|---:|---:|---:|---:|']
    for d in diagnostic:
        if d['partition']!='all':continue
        lines.append(f'| {NAMES[d["group"]]} | {d["case"]} | {d["delta_psnr_mean"]:+.4f} | {d["delta_ssim_mean"]:+.5f} | {d["delta_lpips_mean"]:+.5f} | {d["psnr_improved_fraction"]:.1%} |')
    lines+=['', '这些预先声明的诊断在同模型、同相机、同K、同补偿网络下，只切换评分残差，可判断新增分支在当前checkpoint是否帮助局部选点。它不等于将整个训练历史改成无残差。', '',
        '## 训练曝光与时间', '', '| 组 | 是否复用 | Full/全局/局部 | LI采集/VJP | 全局K累计 | 局部K累计 | Mapping均值ms | 进程秒 |', '|---|---|---|---|---:|---:|---:|---:|']
    for t in training:
        lines.append(f'| {NAMES[t["group"]]} | {t["reused"]} | {t["full_renders"]}/{t["global_renders"]}/{t["local_renders"]} | {t["li_captures"]}/{t["li_vjps"]} | {t["global_K_total"]} | {t["local_K_total"]} | {t["mean_mapping_ms"]:.3f} | {t["seconds"]:.2f} |')
    lines+=['', '每组共享Adam/Master/Selector更新11580，六档各1930次；另有GI刷新Full864次。Feature Adam时钟11580/9650/7720/5790/3860/1930。固定N13298、A2790。共享GPU存在其他任务，时间只作实际运行记录，不能据此直接宣称CUDA或端到端速度差异。', '',
        '## 实现、可比性与限制', '',
        '- 原Selector11→32→32→2、Residual48 Compensator及训练目标主体保留；视角残差39→16→1新增657参数。7D相机/Anchor编码与隐藏32D均detach，输出零初始化。',
        '- 新局部Photo/Budget通过live全局logit训练原Selector及原渲染路径；LI限定VJP只更新四张残差参数，不直接修改Master、原Selector、Compensator或Feature已有梯度。三路反向后一次共享Adam。',
        '- 原61个初始状态Tensor、Master SHA、11580步相机/ratio序列相同；两个视角组新分支初态相同。Adam/RNG重置，不称完整SLAM恢复。两组新实验使用完全相同的已验收二进制。',
        '- 历史baseline使用较早无视角版本二进制，SHA有别；按用户要求复用已有模型与指标。新开关关闭的原路径已回归验证，但不能声称三组二进制相同或多步CUDA训练逐bit可重复。',
        '- 新评价显式调用view_conditioned=True；全局导出始终不调用视角残差。基线34×792行复用，本轮两个模型各39×792行新增，总88704行；计数Renderer234次颜色/radii逐bit核对。',
        '- Full、global/view共11ratio、固定点数及各自全局贡献匹配预算均评价。后者使用每组自身全局Alpha贡献数，不是跨模型严格same-K，也不保证最终Alpha贡献数相同。',
        '- 792帧含Bank72、non-Bank720；FP32且TF32关闭、颜色clamp[0,1]、逐帧PSNR平均、11×11 SSIM和AlexNet LPIPS(normalize=True)。未根据non-Bank质量选checkpoint、权重、seed或终点。',
        '- LI贡献/梯度诊断每100步与六ratio轮转重合，不能当作各ratio均匀采样。新残差在Full/global后没有已有梯度，旧pre_local余弦的0不能解释为与局部Photo正交；本轮不利用此值解释因果。',
        '- Master和Compensator共享更新，所以新模型相对历史baseline的全部变化不能只归因于视角评分。7D相机信息不提供真实遮挡，Full贡献也不是删点后边际图像价值。',
        '- 单场景单seed仅提供本协议下的方向；小幅收益需配对复验。生产默认未切换，无commit/push，不保存渲染帧图片，完整性记录见FINAL_INTEGRITY.json。', '',
        '## 本轮结果归因', '',
        '视角残差＋LI通过全局/Full质量筛查，但局部1%仍低于baseline约0.329dB，SSIM低约0.00810、LPIPS高约0.01110；5% PSNR仅高约0.0107dB，同时LPIPS高约0.00109。30% PSNR高约0.2423dB。固定1k低约0.1209dB，2k/3k分别高约0.0687/0.0335dB，其余预算没有一致改善。单seed不能判定这些小幅收益稳定。', '',
        '同checkpoint且同K开启视角残差，LI模型的1%/5%分别提高约0.1103/0.0657dB，非Bank也分别提高约0.1124/0.0644dB；说明新增分支在部分预算有实际作用，不能仅凭相对baseline的下降就断言它完全无效。但同模型固定2k开启残差反而下降约0.1410dB（SSIM下降、LPIPS上升），因此它没有在所有预算下学到有利排序。需要隔离不同ratio/预算条件和训练造成的共享状态变化。', '',
        '无LI视角模型相对baseline的1%下降约0.9355dB，而同模型残差开关只改变约0.0011dB；这支持把训练后共享Selector/Master/Compensator变化作为下一项检查，而非把全部退化归咎于推理残差。这里尚未分开固定地图、固定补偿和固定原Selector，也没有将训练差异直接当作正常数值误差。全部模型零选点步数与LoD非有限值计数均为0，当前结果没有训练崩溃证据。', '',
        '## 下一步', '',
        '优先依据同checkpoint残差开关诊断决定方向：若残差开关本身有效、但相对baseline整体退化，应继续隔离共享Selector/Master/Compensator的训练变化；若残差本身在相同K下降低质量，应优先检查LI教师真实选点质量及跨视角可学习性。只有低ratio、固定预算与全局保护同时成立，才值得配对复验后考虑接入。', '',
        f'新模型路径：`{ROOT}/runs/full_view_no_li/23560_shutdown/ply` 与 `{ROOT}/runs/full_view_li01/23560_shutdown/ply`。', '',
        '![Ratio质量](figures/VIEW_RESIDUAL_RATIO_PSNR.png)', '', '![固定预算](figures/VIEW_RESIDUAL_FIXED_BUDGET_PSNR.png)']
    (OUT/'VIEW_RESIDUAL_FULL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(gates,indent=2))


if __name__=='__main__':main()
