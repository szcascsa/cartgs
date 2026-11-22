"""汇总三个预定权重；配对图像，审计真实训练曝光，不挑选checkpoint。"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
OUT=ROOT/'evaluation/reports'
TRAIN=[.01,.05,.1,.3,.5,.7]
UNSEEN=[.03,.08,.2,.4,.6]
GROUPS=['no_li','li01','li05']
COMPARISONS=[('no_li','li01'),('no_li','li05'),('li01','li05')]


def write_csv(name,values):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def partitions(values):
    for label in ['all','bank','nonbank']:
        chosen=[r for r in values if label=='all' or (r['bank']=='True')==(label=='bank')]
        assert len(chosen)==dict(all=792,bank=72,nonbank=720)[label]
        yield label,chosen


def distribution(values,key,prefix,item):
    a=np.array([float(r[key]) for r in values]);assert np.isfinite(a).all()
    item.update({prefix+'_'+label:float(value) for label,value in [
        ('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]})


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    with (OUT/'PER_VIEW.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==80784
    cases=defaultdict(list)
    for row in rows:cases[(row['group'],row['case'])].append(row)
    summary=[]
    for (group,case),values in cases.items():
        for partition,chosen in partitions(values):
            item={key:values[0][key] for key in ['group','case','scope','ratio','ratio_split']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']:
                distribution(chosen,key,key,item)
            item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in chosen)
            summary.append(item)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));write_csv('SUMMARY.csv',summary)
    lookup={(r['group'],r['case'],int(r['frame'])):r for r in rows}
    assert len(lookup)==len(rows)
    paired=[]
    for reference,group in COMPARISONS:
        for row in rows:
            if row['group']!=group:continue
            ref=lookup[(reference,row['case'],int(row['frame']))]
            assert row['image']==ref['image'] and row['timestamp']==ref['timestamp']
            item={key:row[key] for key in ['case','scope','ratio','ratio_split','frame','bank']}
            item.update(reference=reference,group=group)
            for key in ['psnr','ssim','lpips','K','contributing_K','projected_K']:
                item['delta_'+key]=float(row[key])-float(ref[key])
            item['same_selection']=row['selection_sha256']==ref['selection_sha256'];paired.append(item)
    write_csv('PAIRED_PER_VIEW.csv',paired)
    pair_cases=defaultdict(list)
    for r in paired:pair_cases[(r['reference'],r['group'],r['case'])].append(r)
    pair_summary=[]
    for (reference,group,case),values in pair_cases.items():
        for partition,chosen in partitions(values):
            item={key:values[0][key] for key in ['reference','group','case','scope','ratio','ratio_split']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','K','contributing_K','projected_K']:
                distribution(chosen,'delta_'+key,'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in chosen]))
            item['same_K_fraction']=float(np.mean([r['delta_K']==0 for r in chosen]));pair_summary.append(item)
    (OUT/'PAIRED_SUMMARY.json').write_text(json.dumps(pair_summary,indent=2));write_csv('PAIRED_SUMMARY.csv',pair_summary)
    index={(r['group'],r['case'],r['partition']):r for r in summary}
    pair_index={(r['reference'],r['group'],r['case'],r['partition']):r for r in pair_summary}
    gates=[]
    for reference,group in COMPARISONS:
        delta=lambda scope,r:pair_index[(reference,group,f'{scope}_{r:g}','all')]['delta_psnr_mean']
        gate=dict(reference=reference,group=group,
            global_train_mean_delta=float(np.mean([delta('global',r) for r in TRAIN])),
            global_train_min_delta=min(delta('global',r) for r in TRAIN),
            full_delta=pair_index[(reference,group,'full','all')]['delta_psnr_mean'],
            view_train_mean_delta=float(np.mean([delta('view',r) for r in TRAIN])),
            view_unseen_mean_delta=float(np.mean([delta('view',r) for r in UNSEEN])),
            thresholds=dict(global_train_mean=-.1,global_train_min=-.2,full=-.1))
        gate['quality_guard_passed']=gate['global_train_mean_delta']>=-.1 and gate['global_train_min_delta']>=-.2 and gate['full_delta']>=-.1
        gate['local_mean_improved']=gate['view_train_mean_delta']>0;gates.append(gate)
    (OUT/'QUALITY_GATES.json').write_text(json.dumps(gates,indent=2))

    training=[];initial=None;keyframes=None;initial_master=None;camera_ratio_sequence=None
    for group in GROUPS:
        root=ROOT if group=='li05' else PREVIOUS
        folder=root/'runs'/f'full_{group}'
        source=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        keys=(folder/'training_keyframes.csv').read_bytes()
        master_sha=hashlib.sha256((folder/'step0_master.ply').read_bytes()).hexdigest()
        if initial is None:initial=source;keyframes=keys;initial_master=master_sha
        assert set(initial)==set(source) and len(initial)==61
        for key in source:
            a,b=initial[key],source[key]
            assert a.shape==b.shape and a.dtype==b.dtype
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
        assert master_sha==initial_master and keyframes==keys
        state=json.loads((folder/'23560_shutdown/ply/real_lod_state.json').read_text())
        anchor=json.loads((folder/'23560_shutdown/ply/point_cloud/iteration_23560/anchor_selector_state.json').read_text())
        assert state['optimizer_commits']==state['master_updates']==state['selector_updates']==11580
        clocks=[s['adam_step'] for s in anchor['compensation_feature_state']]
        assert clocks==[11580,9650,7720,5790,3860,1930]
        with (folder/'real_lod_steps.csv').open() as stream:steps=list(csv.DictReader(stream))
        local=[json.loads(line) for line in (folder/'view_budget_steps.jsonl').read_text().splitlines()]
        assert len(steps)==len(local)==11580 and state['ratio_steps']==[1930]*6
        assert all(not r['strict_topk'] and r['adam_commits_this_step']==1 and r['optimizer_committed'] for r in local)
        assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
        weight=dict(no_li=0,li01=.1,li05=.5)[group]
        assert state['view_budget_li_weight']==float(np.float32(weight))
        sequence=[(r['step'],r['keyframe_id'],r['ratio']) for r in local]
        if camera_ratio_sequence is None:camera_ratio_sequence=sequence
        assert sequence==camera_ratio_sequence
        status=json.loads((root/'reports/EXPERIMENT_STATUS.json').read_text())
        audit=dict(group=group,li_weight=weight,steps=len(steps),full_renders=state['full_training_renders'],
            global_renders=state['lod_training_renders'],local_renders=state['view_budget_training_renders'],
            importance_renders=state['importance_full_renders'],li_full_captures=state['view_budget_li_full_renders'],
            li_vjps=state['view_budget_li_backwards'],feature_adam_clocks=clocks,gaussians=anchor['gaussian_count'],
            anchors=anchor['anchor_count'],wall_seconds=status['groups'][group]['seconds'],
            global_K_total=sum(int(r['K']) for r in steps),local_K_total=sum(r['K'] for r in local),
            zero_selection_steps=sum(r['K']==0 for r in local),
            global_zero_selection_steps=sum(int(r['K'])==0 for r in steps),lod_nonfinite=state['lod_nonfinite'],
            mean_mapping_ms=float(np.mean([float(r['ms']) for r in steps])),
            median_mapping_ms=float(np.median([float(r['ms']) for r in steps])))
        for key in ['li_loss','li_global_label_disagreement','li_pool_full_contribution_fraction',
                    'li_selected_mass_fraction','li_selector_grad_norm','pre_local_selector_grad_norm','li_vs_pre_local_gradient_cosine']:
            chosen=[r for r in local if key in r]
            if chosen:
                value=dict(samples=len(chosen));distribution(chosen,key,key,value);audit[key]=value
        audit['diagnostic_ratio_samples']={f'{ratio:g}':sum('li_selector_grad_norm' in r and abs(r['ratio']-ratio)<1e-6 for r in local) for ratio in TRAIN}
        audit['late_li_diagnostics']={}
        for key in ['li_loss','li_selected_mass_fraction','li_pool_full_contribution_fraction',
                    'li_global_label_disagreement','li_selector_grad_norm','li_vs_pre_local_gradient_cosine']:
            chosen=[r for r in local[-2000:] if key in r]
            if chosen:
                value=dict(samples=len(chosen));distribution(chosen,key,key,value);audit['late_li_diagnostics'][key]=value
        audit['local_ratio_budget_error']=[]
        for ratio in TRAIN:
            chosen=[r for r in local[-2000:] if abs(r['ratio']-ratio)<1e-6]
            audit['local_ratio_budget_error'].append(dict(ratio=ratio,actual_K_mean=float(np.mean([r['K'] for r in chosen])),
                budget_relative_error=float(np.mean([abs(r['K']-r['target_K'])/max(1.,r['target_K']) for r in chosen]))))
        training.append(audit)
    (ROOT/'reports/TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
    (ROOT/'reports/INITIALIZATION_PARITY.json').write_text(json.dumps(dict(passed=True,tensors=61,bitwise=True,
        master_sha256=initial_master,camera_sequence_identical=True,matched_camera_ratio_steps=len(camera_ratio_sequence)),indent=2))

    def metric(group,case,key='psnr',partition='all'):
        return index[(group,case,partition)][key+'_mean']

    lines=['# fr1_xyz：局部LI权重0 / 0.1 / 0.5对比', '',
        '本轮只把 ViewBudget.li_weight 从0.1改为0.5，从相同未训练Full checkpoint重新训练11580步；没有接着0.1模型继续训练。复用上一轮已验收的完全相同二进制。无LI与0.1组指标只读复用，本轮仅新增0.5组渲染。', '',
        f'本次不支持将LI权重提高到0.5：相对0.1，局部训练六档均值下降{abs(gates[2]["view_train_mean_delta"]):.4f}dB，全局训练六档下降{abs(gates[2]["global_train_mean_delta"]):.4f}dB。局部1%从{metric("li01","view_0.01"):.4f}降到{metric("li05","view_0.01"):.4f}dB，5%从{metric("li01","view_0.05"):.4f}降到{metric("li05","view_0.05"):.4f}dB。0.5相对无LI未通过原定全局质量保护门槛，生产默认保持不变。单seed仍不能将下降的具体机制确认为LI梯度冲突。', '',
        '## 结论及预注册质量保护', '',
        '| 比较 | 局部训练六档ΔPSNR | 局部未见五档ΔPSNR | 全局训练六档ΔPSNR | 全局最差训练档Δ | Full Δ | 质量保护 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for g in gates:
        lines.append(f'| {g["group"]} − {g["reference"]} | {g["view_train_mean_delta"]:+.4f} | {g["view_unseen_mean_delta"]:+.4f} | {g["global_train_mean_delta"]:+.4f} | {g["global_train_min_delta"]:+.4f} | {g["full_delta"]:+.4f} | {g["quality_guard_passed"]} |')
    lines+=['', '单位dB。预注册筛查相对无LI：全局训练均值≥−0.10dB、任一训练档≥−0.20dB、Full≥−0.10dB。0.5对0.1一行也展示相同阈值，但不是另设统计检验。门槛通过只表示本次质量筛查通过，不能证明统计非劣或稳定收益。', '',
        '## 局部单视角ratio质量', '', '| ratio | 类型 | 无LI PSNR | 0.1 PSNR | 0.5 PSNR | 0.5−无LI | 0.5−0.1 | 0.5改善视角占比（对无LI） |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(TRAIN+UNSEEN):
        case=f'view_{r:g}';a,b,c=[metric(g,case) for g in GROUPS]
        fraction=pair_index[('no_li','li05',case,'all')]['psnr_improved_fraction']
        lines.append(f'| {r:g} | {"训练" if r in TRAIN else "未见"} | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} | {fraction:.1%} |')
    for scope,title in [('view','局部'),('global','全局')]:
        lines+=['',f'## {title} SSIM / LPIPS', '', '| ratio | 无LI SSIM | 0.1 SSIM | 0.5 SSIM | 无LI LPIPS | 0.1 LPIPS | 0.5 LPIPS |', '|---:|---:|---:|---:|---:|---:|---:|']
        for r in sorted(TRAIN+UNSEEN):
            values=[metric(g,f'{scope}_{r:g}',key) for key in ['ssim','lpips'] for g in GROUPS]
            lines.append('| '+f'{r:g} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines+=['', 'SSIM越高越好，LPIPS越低越好。PSNR、SSIM和LPIPS方向必须分别判断；0.7是训练ratio，不重复列作未见。', '',
        '## 全局导出与Full', '', '| ratio | 无LI PSNR | 0.1 PSNR | 0.5 PSNR | 0.5−无LI | 0.5−0.1 |', '|---:|---:|---:|---:|---:|---:|']
    for r in sorted(TRAIN+UNSEEN):
        a,b,c=[metric(g,f'global_{r:g}') for g in GROUPS]
        lines.append(f'| {r:g} | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} |')
    a,b,c=[metric(g,'full') for g in GROUPS]
    lines.append(f'| Full | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} |')
    lines+=['', '## 单视角固定点数', '', '| 请求预算 | 无LI PSNR | 0.1 PSNR | 0.5 PSNR | 0.5−无LI | 0.5−0.1 | 0.5实际K均值 | 三组K相同视角占比 |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in [1000,2000,3000,4000,5000,10000]:
        case=f'points_{budget}';a,b,c=[metric(g,case) for g in GROUPS]
        same=float(np.mean([len({int(lookup[(g,case,frame)]['K']) for g in GROUPS})==1 for frame in range(792)]))
        lines.append(f'| {budget:,} | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} | {metric("li05",case,"K"):.1f} | {same:.1%} |')
    lines+=['', '实际K=min(请求预算,合格候选数)。10k可能候选不足且网络ratio超训练范围，未扩大候选池强行补点。K相同并不代表选点身份或最终实际Alpha贡献数相同。', '',
        '## 各自全局贡献数量匹配的局部预算', '', '| 来源全局ratio | 无LI局部PSNR | 0.1局部PSNR | 0.5局部PSNR | 无LI/0.1/0.5实际K均值 |', '|---:|---:|---:|---:|---|']
    for r in [.1,.2,.3,.4,.5]:
        case=f'matched_{r:g}';q=[metric(g,case) for g in GROUPS];k=[metric(g,case,'K') for g in GROUPS]
        lines.append(f'| {r:g} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {k[0]:.1f}/{k[1]:.1f}/{k[2]:.1f} |')
    lines+=['', '每组使用自身模型该帧全局渲染的实际Alpha贡献数作局部请求预算，网络ratio=B/N。这不是三组严格same-K，亦不保证最终Alpha贡献数相同。', '',
        '## 训练曝光与开销', '', '| 组 | Full/全局/局部 | LI采集/VJP | 全局K累计 | 局部K累计 | Mapping均值ms | 完整进程秒 |', '|---|---|---|---:|---:|---:|---:|']
    for a in training:
        lines.append(f'| {a["group"]} | {a["full_renders"]}/{a["global_renders"]}/{a["local_renders"]} | {a["li_full_captures"]}/{a["li_vjps"]} | {a["global_K_total"]} | {a["local_K_total"]} | {a["mean_mapping_ms"]:.3f} | {a["wall_seconds"]:.2f} |')
    lines+=['', '每组GI刷新额外Full864次，共享Adam/Master/Selector更新各11580；ratio各1930。六Feature表Adam时钟11580/9650/7720/5790/3860/1930。最终N=13298，Anchor=2790，拓扑固定。共享GPU还有其他任务，墙钟与Mapping时间仅是实际运行记录，不是隔离的速度Benchmark。', '',
        '## 训练末期LI诊断', '', '| 组 | 最后2000步LI Loss均值 | 已选Full贡献占比 | LI Selector梯度范数 | 与LI反传前已有Selector梯度余弦 |',
        '|---|---:|---:|---:|---:|']
    for a in training[1:]:
        d=a['late_li_diagnostics']
        values=[d[key][key+'_mean'] for key in ['li_loss','li_selected_mass_fraction','li_selector_grad_norm','li_vs_pre_local_gradient_cosine']]
        lines.append('| '+a['group']+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['', 'Loss均值覆盖最后2000步，贡献/梯度诊断只有其中20个采样点，主要落在5%/30%/70%，不是全部ratio的均匀统计。余弦参考为当前LI VJP之前已累积的Selector梯度，不能直接称作与纯Photo梯度的夹角。跨模型的贡献量及教师本身随Master变化；此表只提供机制线索，不证明因果。', '',
        '## 协议、实现与验证', '',
        '- 三组从同一Full11980起点开始，最终23560；初始61个状态Tensor逐bit一致，Master文件SHA和72个训练相机序列一致。Adam/RNG重置，不称完整SLAM恢复。',
        '- Full＋全局LoD＋局部LoD同相机同ratio，原Photo/GI/软预算Loss不变，三分支反向后一次Adam。局部候选按名义Anchor AABB视锥；训练是Gumbel-ST，不改为严格Top-K。',
        '- LI为同一步未补偿Full全像素alpha×T/HW，不增加Full渲染；局部教师前floor(rN)仅用于标签。LI及教师detach，独立VJP仅更新local_emd与soft_net六张Selector参数，原Master/Feature/Compensator梯度路径保持。',
        '- 本轮只改变LI权重，不改源码、结构、学习率、ratio、终点或推理规则。复用上轮已经测试的二进制，新增12步真实checkpoint验收，验证实际加载本轮独立Mapper动态库。',
        '- 完整评价792帧：Bank72、non-Bank720；每组34配置，三组合计80784行。本轮新渲染26928行，旧53856行只读导入并核验原checkpoint和CSV SHA。逐帧指标及配对median/p10/p90、改善视角比例见CSV。',
        '- FP32且TF32关闭，颜色clamp[0,1]；逐帧PSNR再平均、TorchMetrics 11×11 SSIM、AlexNet LPIPS(normalize=True)。新模型首/中/末×34配置102次颜色/radii与普通Renderer逐bit一致。',
        '- 首次汇总将Python双精度0.1与C++保存的FP32值0.10000000149011612直接比较而触发断言；已改为严格比较对应FP32表示。失败日志保留，只重新运行汇总，未重训或重渲染，也未放宽任何数值验证。',
        '- 不保存渲染帧图片，只保存指标、日志、模型及曲线图。生产604文件、本轮二进制及评价输入的前后完整性以FINAL_INTEGRITY.json为准。', '',
        '## 可确认与尚未隔离的因素', '',
        '- 本次可以判断0.5在这一个固定协议下的结果方向；单场景单seed不能证明收益稳定。相同初态与相机不代表多步CUDA训练逐bit可重复。',
        '- 上轮无LI与历史联合复跑在Full仅差约−0.0009dB，而1%局部约−0.1806dB。这个历史差异只说明不能把小幅变化直接当作稳定收益，不能用来判定本轮任何差异“正常”。',
        '- Master共同更新，候选人口、全局/局部实际选点曝光随训练变化，不能把所有图像差异只归因于Selector排序。固定点数对照部分约束数量因素，未固定地图或Compensator。',
        '- 11D Selector没有相机输入；视角相关LI教师可能冲突。LI/GI标签分歧、LI已选贡献比例、Selector梯度范数和余弦见TRAINING_AUDIT.json。它们是稀疏诊断，不是独立因果验证。',
        '- 每100步诊断与六ratio轮转重合，除最初六步外主要覆盖0.05/0.3/0.7。不可将该诊断均值当作六ratio均值；实际Loss、预算和最终渲染评价全量记录。',
        '- 全像素贡献不是删除点后的边际质量损失；LI Loss下降或已选贡献提高，本身不能证明图像质量更好。', '',
        '## 下一步', '',
        '不根据本次单seed自动切换默认。若0.5在固定预算和低ratio同时稳定改善，下一步对无LI与0.5配对复跑；若只是同ratio收益而固定预算无收益，优先固定Master/Compensator和实际K，检查LI教师选点的真实渲染上限及跨视角标签翻转，而非继续盲目增加权重。', '',
        f'0.5最终模型：`{ROOT}/runs/full_li05/23560_shutdown/ply`。前两组保留在`{PREVIOUS}/runs/`。无commit/push，未更改生产默认。', '',
        '![Ratio质量](figures/LI_WEIGHT_RATIO_PSNR.png)', '', '![固定预算](figures/LI_WEIGHT_FIXED_BUDGET_PSNR.png)']
    (OUT/'LI_WEIGHT_05_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(gates,indent=2))


if __name__=='__main__':main()
