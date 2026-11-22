"""配对结果与真实训练预算审计；不修改模型、不挑选终点。"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'evaluation/reports'
TRAIN=[.01,.05,.1,.3,.5,.7]
UNSEEN=[.03,.08,.2,.4,.6]
rows=list(csv.DictReader((OUT/'PER_VIEW.csv').open()))
assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
groups=defaultdict(list)
for row in rows:groups[(row['group'],row['case'])].append(row)
summary=[]
for (group,case),values in groups.items():
    for partition in ['all','bank','nonbank']:
        chosen=[r for r in values if partition=='all' or (r['bank']=='True')==(partition=='bank')]
        assert len(chosen)=={'all':792,'bank':72,'nonbank':720}[partition]
        item={key:values[0][key] for key in ['group','case','scope','ratio','ratio_split']}
        item.update(partition=partition,frames=len(chosen))
        for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']:
            a=np.array([float(r[key]) for r in chosen]);assert np.isfinite(a).all()
            for label,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:item[key+'_'+label]=float(value)
        item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in chosen)
        summary.append(item)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
with (OUT/'SUMMARY.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
lookup={(r['group'],r['case'],int(r['frame'])):r for r in rows}
paired=[]
for row in rows:
    if row['group']!='li01':continue
    ref=lookup[('no_li',row['case'],int(row['frame']))]
    assert row['image']==ref['image'] and row['timestamp']==ref['timestamp']
    item={key:row[key] for key in ['case','scope','ratio','ratio_split','frame','bank']}
    for key in ['psnr','ssim','lpips','K','contributing_K','projected_K']:
        item['delta_'+key]=float(row[key])-float(ref[key])
    item['same_selection']=row['selection_sha256']==ref['selection_sha256'];paired.append(item)
with (OUT/'PAIRED_PER_VIEW.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
pair_summary=[]
for case in sorted({r['case'] for r in paired}):
    for partition in ['all','bank','nonbank']:
        chosen=[r for r in paired if r['case']==case and (partition=='all' or (r['bank']=='True')==(partition=='bank'))]
        item={key:chosen[0][key] for key in ['case','scope','ratio','ratio_split']};item.update(partition=partition,frames=len(chosen))
        for key in ['psnr','ssim','lpips','K','contributing_K','projected_K']:
            a=np.array([r['delta_'+key] for r in chosen])
            for label,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:item['delta_'+key+'_'+label]=float(value)
        item['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in chosen]))
        item['same_K_fraction']=float(np.mean([r['delta_K']==0 for r in chosen]));pair_summary.append(item)
(OUT/'PAIRED_SUMMARY.json').write_text(json.dumps(pair_summary,indent=2))
with (OUT/'PAIRED_SUMMARY.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(pair_summary[0]));writer.writeheader();writer.writerows(pair_summary)
index={(r['group'],r['case'],r['partition']):r for r in summary}
pair_index={(r['case'],r['partition']):r for r in pair_summary}
delta=lambda scope,r:pair_index[(f'{scope}_{r:g}','all')]['delta_psnr_mean']
gates={'global_train_mean_delta':float(np.mean([delta('global',r) for r in TRAIN])),
       'global_train_min_delta':min(delta('global',r) for r in TRAIN),
       'full_delta':pair_index[('full','all')]['delta_psnr_mean'],
       'view_train_mean_delta':float(np.mean([delta('view',r) for r in TRAIN])),
       'view_unseen_mean_delta':float(np.mean([delta('view',r) for r in UNSEEN])),
       'thresholds':{'global_train_mean':-.1,'global_train_min':-.2,'full':-.1}}
gates['quality_guard_passed']=gates['global_train_mean_delta']>=-.1 and gates['global_train_min_delta']>=-.2 and gates['full_delta']>=-.1
gates['local_mean_improved']=gates['view_train_mean_delta']>0
(OUT/'QUALITY_GATES.json').write_text(json.dumps(gates,indent=2))

training=[];initial=None;keyframes=None;initial_master=None
def same_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
for group in ['no_li','li01']:
    folder=ROOT/'runs'/f'full_{group}'
    source=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
    keys=(folder/'training_keyframes.csv').read_text()
    master_sha=hashlib.sha256((folder/'step0_master.ply').read_bytes()).hexdigest()
    if initial is None:initial=source;keyframes=keys;initial_master=master_sha
    assert set(initial)==set(source) and all(same_bits(initial[k],source[k]) for k in initial)
    assert master_sha==initial_master
    assert keyframes==keys
    state=json.loads((folder/'23560_shutdown/ply/real_lod_state.json').read_text())
    anchor=json.loads((folder/'23560_shutdown/ply/point_cloud/iteration_23560/anchor_selector_state.json').read_text())
    assert state['optimizer_commits']==state['master_updates']==state['selector_updates']==11580
    clocks=[s['adam_step'] for s in anchor['compensation_feature_state']];assert clocks==[11580,9650,7720,5790,3860,1930]
    steps=list(csv.DictReader((folder/'real_lod_steps.csv').open()))
    local=[json.loads(line) for line in (folder/'view_budget_steps.jsonl').read_text().splitlines()]
    assert len(steps)==len(local)==11580 and state['ratio_steps']==[1930]*6
    assert all(not r['strict_topk'] and r['adam_commits_this_step']==1 for r in local)
    audit=dict(group=group,steps=len(steps),full_renders=state['full_training_renders'],global_renders=state['lod_training_renders'],
        local_renders=state['view_budget_training_renders'],importance_renders=state['importance_full_renders'],
        li_full_captures=state['view_budget_li_full_renders'],li_vjps=state['view_budget_li_backwards'],
        feature_adam_clocks=clocks,gaussians=anchor['gaussian_count'],anchors=anchor['anchor_count'],
        global_K_total=sum(int(r['K']) for r in steps),local_K_total=sum(r['K'] for r in local),
        mean_mapping_ms=float(np.mean([float(r['ms']) for r in steps])),median_mapping_ms=float(np.median([float(r['ms']) for r in steps])))
    for key in ['li_loss','li_global_label_disagreement','li_pool_full_contribution_fraction',
                'li_selected_mass_fraction','li_selector_grad_norm','li_vs_pre_local_gradient_cosine']:
        a=[r[key] for r in local if key in r]
        if a:audit[key]=dict(samples=len(a),mean=float(np.mean(a)),median=float(np.median(a)),p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)))
    audit['diagnostic_ratio_samples']={f'{ratio:g}':sum('li_selector_grad_norm' in r and abs(r['ratio']-ratio)<1e-6 for r in local)
                                     for ratio in TRAIN}
    audit['local_ratio_budget_error']=[]
    for ratio in TRAIN:
        chosen=[r for r in local[-2000:] if abs(r['ratio']-ratio)<1e-6]
        audit['local_ratio_budget_error'].append(dict(ratio=ratio,actual_K_mean=float(np.mean([r['K'] for r in chosen])),
            budget_relative_error=float(np.mean([abs(r['K']-r['target_K'])/max(1.,r['target_K']) for r in chosen]))))
    training.append(audit)
(ROOT/'reports/TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
(ROOT/'reports/INITIALIZATION_PARITY.json').write_text(json.dumps(dict(passed=True,tensors=len(initial),
    bitwise=True,master_sha256=initial_master,camera_sequence_identical=True),indent=2))

lines=['# 局部LI对fr1_xyz联合LoD训练的消融', '',
    '本轮使用相同独立二进制配对运行no_li与li01，均从同一未训练Full checkpoint开始。LI权重固定0.1，未依据non-Bank质量调权重或选择终点。两组11580步，相机序列、初始61个模型Tensor与Master一致。', '',
    'LI是当前未补偿Full的全像素alpha×T/HW。教师仅在名义Anchor局部候选中排序，实际训练保持Gumbel-ST与软预算；LI只更新local_emd和soft_net六张Tensor。全局GI仍为原opacity×T口径。', '',
    '## 质量结论', '',
    f'训练六档局部PSNR平均变化 {gates["view_train_mean_delta"]:+.4f} dB，未见五档平均 {gates["view_unseen_mean_delta"]:+.4f} dB。全局训练六档平均 {gates["global_train_mean_delta"]:+.4f} dB，最差训练档 {gates["global_train_min_delta"]:+.4f} dB；Full变化 {gates["full_delta"]:+.4f} dB。',
    f'预注册质量保护门槛通过：{gates["quality_guard_passed"]}；局部训练档均值改善：{gates["local_mean_improved"]}。单场景单seed不能视为统计非劣证明。', '',
    '本轮观察到的收益很小且方向不一致，暂不建议切换生产默认。尤其1%局部变化与历史复跑波动处于相近量级，不能仅凭本次正均值确认稳定提升。', '',
    '| ratio | 类型 | 全局noLI | 全局LI | Δ | 局部noLI | 局部LI | Δ | 局部改善视角 |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for ratio in sorted(TRAIN+UNSEEN):
    g0,g1,v0,v1=[index[(group,f'{scope}_{ratio:g}','all')]['psnr_mean'] for scope,group in [('global','no_li'),('global','li01'),('view','no_li'),('view','li01')]]
    lines.append(f'| {ratio:g} | {"训练" if ratio in TRAIN else "未见"} | {g0:.4f} | {g1:.4f} | {g1-g0:+.4f} | {v0:.4f} | {v1:.4f} | {v1-v0:+.4f} | {pair_index[(f"view_{ratio:g}","all")]["psnr_improved_fraction"]:.1%} |')
lines+=['', '完整PSNR/SSIM/LPIPS、实际K、贡献数量、Bank/non-Bank和逐视角分布见SUMMARY.csv和PAIRED_SUMMARY.csv。局部1%/0.08的LPIPS改善，而0.03和5%的LPIPS分别变差约0.0063和0.0076；不能以PSNR正变化概括全部质量。', '',
    '## 固定点数对照', '', '| 请求预算 | 无LI PSNR | LI PSNR | ΔPSNR | 提交K均值差 |', '|---:|---:|---:|---:|---:|']
for budget in [1000,2000,3000,4000,5000,10000]:
    case=f'points_{budget}';a=index[('no_li',case,'all')];b=index[('li01',case,'all')]
    lines.append(f'| {budget:,} | {a["psnr_mean"]:.4f} | {b["psnr_mean"]:.4f} | {b["psnr_mean"]-a["psnr_mean"]:+.4f} | {b["K_mean"]-a["K_mean"]:+.1f} |')
lines+=['', '固定预算下1k/3k/4k/5k/10k只提高约0.002–0.009dB，2k下降约0.044dB；没有观察到明显的固定点数质量收益。高预算的不足及ratio外推均保留原口径，详细实际K和贡献数见SUMMARY.csv。', '',
    '## 实际训练预算及开销', '', '| 组 | Full/全局/局部渲染 | LI Full采集 | LI VJP | 全局K累计 | 局部K累计 | Mapping均值ms |',
    '|---|---|---:|---:|---:|---:|---:|']
for a in training:lines.append(f'| {a["group"]} | {a["full_renders"]}/{a["global_renders"]}/{a["local_renders"]} | {a["li_full_captures"]} | {a["li_vjps"]} | {a["global_K_total"]} | {a["local_K_total"]} | {a["mean_mapping_ms"]:.3f} |')
lines+=['', '两组Adam均11580，六表时钟11580/9650/7720/5790/3860/1930。GI刷新所需Full渲染另记864，不计为Mapping。共享GPU负载影响墙钟，不把单次时间差视作严格算子Benchmark；实际选点曝光变化也可能影响耗时和质量。', '',
    '## 实现与验证', '',
    '- 新增配置ViewBudget.li_weight，缺省0；独立示例spatial_offset3_residual48_global_view_joint_li.yaml设0.1，不改生产模板。',
    '- src/real_lod.cpp复用当前Full的CaptureScope(true)，获取真实alpha*T并归一化HW。按当前step/keyframe/topology校验，下一step清空缓存。全局刷新仍CaptureScope(false)，未替换其opacity*T。',
    '- include/local_importance.h复用预算一致排序；局部ST-mask L1标签detach。src/view_budget_training.cpp通过autograd::grad限定local_emd.*和soft_net.*六张Tensor，手工累加已有grad；Photo/Budget仍按原路径反传，之后一次共享Adam。网络新增参数为0。',
    '- 真实CUDA fused Selector VJP测试确认：Master、Compensator、全部Feature已有梯度逐bit保持；六张MLP梯度按预期累加。CPU/CUDA教师测试、Full采集、开关、小Mapping、原健康检测与全局训练回归全部通过，另有两组真实checkpoint十二步验收。',
    '- 初始61个状态Tensor逐bit一致、Master SHA一致、训练相机序列一致；最终N=13298、A=2790、拓扑固定。两组完整评价共53856行，34配置×792帧×2组，首/中/末逐配置204次Rasterizer颜色/radii逐bit一致。',
    '- 当前步LI主缓存N个FP32标量约52KiB，候选标签和排序缓冲随N有界，不跨相机/参数更新复用。Mapping参考耗时约增加12.1%，其中贡献聚合、教师排序、独立VJP与GPU负载未作Nsight细分。', '',
    '## 限制和机制', '',
    '- 11D Selector没有相机输入，局部教师可能在不同视角冲突；记录的LI/GI标签分歧和梯度余弦见TRAINING_AUDIT.json。',
    '- 末期稀疏诊断显示候选池捕获Full贡献均值约93.8%，LI与既有全局GI的标签分歧约6.46%，与此前全局分支梯度的余弦均值约0.073；分位数见审计。这些只是所采样档位的描述，不证明剩余误差的主因，也没有测量完整跨视角标签翻转率。',
    '- 每100步的额外诊断与六档轮转存在周期重合，末期主要覆盖0.05/0.3/0.7；这些诊断均值不能当作六档整体均值。其余三档只保留了初始诊断。所有ratio的实际Loss、预算及最终图像质量均完整记录，不受诊断采样影响。',
    '- Full贡献不是删除点后的边际PSNR损失，Photo仍为质量依据。Master也共同更新，不能把全部变化归因于Selector。',
    '- 同seed不等于多步训练逐bit可重复。本轮noLI对照与历史联合结果在Full均值上仅差约−0.0009dB，但1%局部约−0.1806dB；历史复现详见HISTORICAL_NO_LI_REPRODUCTION.json。没有将这种差异自动判为无害数值误差，主结论采用本轮配对，小幅收益仍需配对seed复验。',
    '- 局部同ratio的候选人口和实际K均单独记录；固定点数及全局贡献匹配预算可进一步检查数量因素。后者不保证最终实际Alpha贡献点数严格相同。',
    '- 10k局部预算不足、网络ratio外推按原Viewer处理，没有扩大候选池或强行补点。',
    '- 首轮测试因Variable Anchor实际扩充点数超过输入点数而触发了写死人口断言；已修正测试为实际Master人口，失败现场保留。训练算法未因该测试失败调整。', '',
    '## 下一步最有信息价值的验证', '',
    '优先配对复验无LI和0.1，确认1%/3%的PSNR收益是否超过同seed复跑及跨seed波动。再在固定Master/补偿网络、相同相机和固定K下比较LI教师选点与当前Selector实际渲染质量，并统计同一点跨视角标签翻转：可区分教师本身无效、11D输入无法表达视角区别和联合训练地图变化。现有证据不支持直接增加网络或继续调大LI权重。', '',
    '## 文件', '',
    '两组最终模型位于 `runs/full_no_li/23560_shutdown/ply` 和 `runs/full_li01/23560_shutdown/ply`。服务器根目录 `/home/prmi/zwj/experiments/local_importance_li_20261105`。', '',
    '![Ratio质量](figures/LI_RATIO_PSNR.png)', '', '![局部固定预算](figures/LI_FIXED_BUDGET_PSNR.png)', '',
    '本轮保留生产默认，不自动提交或推送；实现与实验协议见PROTOCOL.md，输入和生产文件哈希见最终完整性记录。']
(OUT/'LOCAL_IMPORTANCE_LI_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(gates,indent=2))
