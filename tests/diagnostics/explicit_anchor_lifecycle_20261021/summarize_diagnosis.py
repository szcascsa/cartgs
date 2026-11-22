"""从原模型测量与独立复跑证据生成中文诊断报告。"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
repo=here.parents[2]
def read(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def dump(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
original=[r for r in read(here/'frozen/original_opacity_summary.csv') if r['checkpoint']=='10000_shutdown']
contribution=read(here/'frozen/contribution_summary.csv')
quality=read(here/'frozen/low_opacity_ablation_quality.csv')
budget=read(here/'frozen/original_lod_low_opacity_selection.csv')
replay=json.loads((here/'replay/replay_summary.json').read_text())
cohort=read(here/'replay/replay_lifetime_summary.csv')
lifetimes=read(here/'replay/replay_deleted_anchor_lifetimes.csv')
survivors=read(here/'replay/replay_surviving_anchor_births.csv')
hist=read(here/'frozen/original_failed_member_histogram.csv')
gaussians=read(here/'frozen/gaussian_contributions.csv')
anchors=read(here/'frozen/original_surviving_anchors.csv')
mass_by_uid={}
for g in gaussians:
    key=(g['seed'],g['uid'])
    a=mass_by_uid.setdefault(key,dict(total_mass=0.,low_opacity_mass=0.,below_alpha_cutoff_members=0,nonzero_members=0))
    mass=float(g['alpha_T_sum']);opacity=float(g['opacity'])
    a['total_mass']+=mass;a['low_opacity_mass']+=mass*(opacity<.02)
    a['below_alpha_cutoff_members']+=int(opacity<1/255);a['nonzero_members']+=int(mass>1e-8)
for a in anchors:
    a.update(mass_by_uid[(a['seed'],a['uid'])])
    a['low_opacity_mass_fraction']=a['low_opacity_mass']/a['total_mass'] if a['total_mass'] else None
dump(here/'frozen/original_surviving_anchors_with_contribution.csv',anchors)
source_pruning=[]
for seed in (2026,2027):
    rows=[r for r in survivors if int(r['seed'])==seed]
    for group in ['real_obs_1_3','real_obs_4_plus','growth']:
        selected=[]
        for r in rows:
            n=r['real_observations_at_birth']
            name='growth' if not n else ('real_obs_1_3' if int(n)<=3 else 'real_obs_4_plus')
            if name==group:selected.append(r)
        if selected:
            n=sum(int(r['failed_members']) for r in selected)
            source_pruning.append(dict(seed=seed,group=group,anchors=len(selected),failed_members=n,
                failed_fraction=n/(len(selected)*10),mean_failed_members=n/len(selected)))
dump(here/'replay/replay_birth_support_vs_final_pruning.csv',source_pruning)

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
for i,seed in enumerate((2026,2027)):
    rows=[r for r in hist if int(r['seed'])==seed and r['checkpoint']=='10000_shutdown' and r['level']=='-1']
    ax[0,0].bar(np.arange(11)+(i-.5)*.36,[int(r['anchors']) for r in rows],width=.36,label=str(seed))
    r=replay[i]
    ax[0,1].bar(i-.2,r['final_real_obs_1_3']/r['final']*100,width=.4,color=f'C{i}',label=f'{seed}: 1–3真实观测')
    ax[0,1].bar(i+.2,r['final_growth_born']/r['final']*100,width=.4,color=f'C{i}',alpha=.4,label=f'{seed}: Gaussian增长来源')
    xs=np.array([int(v['lifetime_mapping']) for v in lifetimes if int(v['seed'])==seed]);xs.sort()
    ax[1,0].plot(xs,np.arange(1,len(xs)+1)/len(xs),label=str(seed))
    h=r['creation_count_histogram'];x=np.array(sorted(map(int,h)))
    ax[1,1].plot(x,[h[str(k)] for k in x],marker='o',label=str(seed))
ax[0,0].set(title='原模型：存活 Anchor 的逐点失败成员',xlabel='opacity < 0.02 的成员数',ylabel='Anchor 数');ax[0,0].legend()
ax[0,1].set(title='诊断复跑：最终 Anchor 的出生来源',xticks=[0,1],xticklabels=['2026','2027'],ylabel='占最终 Anchor (%)');ax[0,1].legend(fontsize=8)
ax[1,0].set(title='诊断复跑：已删除 Anchor 的寿命 CDF',xlabel='Mapping 步数',ylabel='累计比例');ax[1,0].legend()
ax[1,1].set(title='诊断复跑：空间 key 创建次数',xlabel='同一 (level,key) 创建次数',ylabel='空间 key 数',yscale='log');ax[1,1].legend()
fig.savefig(here/'diagnosis.png',dpi=180);plt.close(fig)

lines=['# 固定十点显式 Anchor 训练诊断','',
'本轮只做源码审查、原 checkpoint 只读测量和带生命周期记录的隔离复跑；未修改本地 production 算法、阈值、Loss、Selector 或 pruning 规则。未 stage、commit、push。','',
'## 结论与证据范围','',
'已经证实的主要问题是：固定十点 + 全部失败才整组删除，保留了大量逐点已失败且几乎不贡献渲染的成员；这些点还占据部分 LoD 选择预算。初始化支持稀疏、空间 key 重建和短寿命情况见下面的独立复跑。它们是结构与训练行为诊断，不能直接证明此前 Full 质量退化的因果来源。','',
'- **原训练**：fr1_xyz，seed 2026/2027，10,000 Mapping；保留原模型与原日志。','- **诊断复跑**：相同场景、seed、配置、10,000 Mapping，仅增加出生/删除/逐点 predicate 日志。在线 SLAM/Mapping 异步，插桩会影响到达时序，不能把复跑历史写成原运行的精确历史。',
'- 本地、服务器 production `/home/prmi/zwj/mycartgs`、原隔离训练 source 的 29 个受控文件 SHA 全部相同。原 source 与原模型在诊断前后 SHA 均未改变。',
'- 原 checkpoint 没保存 birth source/count、逐 UID death step 或 deleted-key 历史；`exist_since_iter_` 也未导出到 PLY。因此原运行 Q1/Q3/Q4 的精确答案不可追溯。',
'- 轨迹差异明确可见：seed2026 原/复跑最终 Anchor 为 2876/3016（+4.87%）；seed2027 为 3019/3392（+12.36%）。因此历史统计仅支持机制诊断，不是原训练逐事件重现，也不能用复跑结果代替原质量。','',
'## 1. 初始化来源','',
'“真实观测数”定义为出生时传入 `observeBlocks()` 的当前批次不同 XYZ 数，不是像素观测次数，也不是该区域最终累计观测数。`growBlocks()` 的输入来自已优化 Gaussian，单独分类，不能称为真实 SLAM 观测。','',
'|诊断复跑 seed|最终 Anchor|出生于真实观测|其中仅 1–3 真实观测|占最终全部|占观测来源|Gaussian 增长来源|','|---|---:|---:|---:|---:|---:|---:|']
for r in replay:lines.append(f"|{r['seed']}|{r['final']}|{r['final_observation_born']}|{r['final_real_obs_1_3']}|{r['real_obs_1_3_fraction_all_survivors']:.2%}|{r['real_obs_1_3_fraction_observation_survivors']:.2%}|{r['final_growth_born']}|")
lines+=['','1/2/3 个观测的最终存活 Anchor 分解：','']
for r in replay:lines.append(f"- seed {r['seed']}：{r['final_real_obs_1_3_by_count']}；按 L0/L1/L2：{r['final_real_obs_1_3_by_level']}。")
lines+=['','源码行为：少于 10 个不同来源时，重复使用来源点并对 **全部 10 个 slot** 作确定性小扰动（半径 0.025h，cell 内约束）；因此 1–3 个观测至少缺少 7–9 个独立真实位置支持。它不是复制十份完全相同 XYZ，但也不是十个独立真实观测。出生时只看当前 batch，后续已占用 cell 的观测不会补充 slot 初始化支持；图像重建优化仍会继续。',
'','最终组内低 opacity 比例与出生支持的关系（复跑，相关性不代表因果）：','','|seed|出生来源|存活 Anchor|低 opacity 成员比例|','|---|---|---:|---:|']
for r in source_pruning:lines.append(f"|{r['seed']}|{r['group']}|{r['anchors']}|{r['failed_fraction']:.2%}|")
lines+=['','## 2. Pruning 保留情况：原模型实测','',
'正式配置 `densify_min_opacity=0.02`、`prune_big_point_after_iter=-1`；因此本次逐点 predicate 是 `opacity < 0.02`，没有 screen/world size 删除条件。以下是最终状态 predicate，不能冒充任意过去 pruning 时刻的 predicate。','',
'|seed|存活 Anchor / Gaussian|含失败成员的 Anchor|至少 5 成员失败|逐点失败成员 / 比例|失败点平均 opacity|失败点渲染权重占比|','|---|---:|---:|---:|---:|---:|---:|']
for r in original:
    c=next(c for c in contribution if c['seed']==r['seed'] and c['group']=='low_opacity')
    lines.append(f"|{r['seed']}|{r['anchors']} / {r['gaussians']}|{r['anchors_with_low']}|{r['anchors_with_ge5_low']}|{r['low_points']} / {float(r['low_point_fraction']):.2%}|{float(r['low_opacity_mean']):.6g}|{float(c['total_mass_fraction']):.4%}|")
lines+=['','|失败成员数 / 10|seed2026 Anchor|seed2027 Anchor|','|---:|---:|---:|']
for n in range(11):
    counts=[next(r['anchors'] for r in hist if int(r['seed'])==s and r['checkpoint']=='10000_shutdown' and r['level']=='-1' and int(r['failed_members'])==n) for s in [2026,2027]]
    lines.append(f'|{n}|{counts[0]}|{counts[1]}|')
lines+=['','每个 Anchor 的十个 opacity、失败成员数、总/失败成员渲染权重、渲染非零成员数都保存在 `frozen/original_surviving_anchors_with_contribution.csv`；逐 Gaussian 的 UID、slot、opacity、累计权重保存在 `frozen/gaussian_contributions.csv`。','',
'**真实贡献测量**：从原先 792 个共同视图等间隔预选 64 个。在原渲染器中，以预计算颜色的 VJP 测量 `d(sum image_R)/d(color_i,R) = sum_pixels alpha_i*T_i`。颜色不影响 alpha/深度顺序；没有 image clamp，没有 optimizer step。测量前后原模型 SHA 不变，权重和图像总 alpha 的守恒误差 < 2e-4（实际约 1e-7）。这不是旧 SI 的 `opacity*T` 代理量，也不是质量增益归因指标。','']
for c in contribution:
    if c['group']=='low_opacity':lines.append(f"- seed {c['seed']}：失败点中 {float(c['zero_contribution_fraction']):.2%} 在这 64 视图累计贡献为零；累计权重总占比 {float(c['total_mass_fraction']):.4%}。")
lines+=['','临时移除所有 opacity<0.02 成员的只读敏感性测试（同 64 视图，未重训）：','','|seed|Full → 临时移除后点数|ΔPSNR (dB)|ΔSSIM|ΔLPIPS|','|---|---:|---:|---:|---:|']
for s in [2026,2027]:
    a=next(r for r in quality if int(r['seed'])==s and r['mode']=='full');b=next(r for r in quality if int(r['seed'])==s and r['mode']=='drop_low')
    lines.append(f"|{s}|{a['render_gaussians']} → {b['render_gaussians']}|{float(b['psnr'])-float(a['psnr']):+.6f}|{float(b['ssim'])-float(a['ssim']):+.7f}|{float(b['lpips'])-float(a['lpips']):+.7f}|")
lines+=['','这是删点敏感性实验，**不是 same-K 方法对比，也不是已获准的生产 pruning 修改**；渲染贡献不等同独立质量贡献。','',
'### 额外暴露的问题：不可渲染成员仍占预算','',
'`forward.cu` 定义 `alpha=min(0.99, opacity*exp(power))`，`power<=0`，并跳过 `alpha<1/255`。因此在当前 opacity 不变时，`opacity<1/255` 的成员对任何视图都无法贡献像素；TF 目前仅改 XYZ/scale/rotation，不改 opacity。这些点没有 photo 渲染梯度，Full opacity regularizer 仍可施加向下压力；不能据此断言所有后续优化中 opacity 永远不变。','',
'|seed|策略|ratio|实际选择|低于 pruning 阈值|低于 renderer 截断|截断成员比例|','|---|---|---:|---:|---:|---:|---:|']
for r in budget:
    lines.append(f"|{r['seed']}|{r['policy']}|{r['ratio']}%|{r['selected']}|{r['selected_below_pruning_threshold']}|{r['selected_below_renderer_alpha_cutoff']}|{int(r['selected_below_renderer_alpha_cutoff'])/int(r['selected']):.2%}|")
lines+=['','## 3. 同一空间 key 的重复创建','',
'原模型最终同层空间碰撞桶为零，只能说明最终快照没有同时重叠的 block，不能说明历史没有重建。UID 是新 block 的永久身份；空间身份是 `(level,kx,ky,kz)`。本次复跑没有几何/CD 迁移事件，因此下面的同 key 统计不混入几何运输。','',
'|复跑 seed|总创建|不同空间 key|创建过多次的 key|额外重建事件|单 key 最大创建次数|同时重复创建|','|---|---:|---:|---:|---:|---:|---:|']
for r in replay:lines.append(f"|{r['seed']}|{r['created']}|{r['unique_spatial_keys']}|{r['recreated_keys']}|{r['recreation_events']}|{r['maximum_creation_count']}|{r['concurrent_duplicate_creation']}|")
lines+=['','同 key 创建次数分布：','']
for r in replay:lines.append(f"- seed {r['seed']}：{r['creation_count_histogram']}。")
lines+=['','源码原因：`pruneBlocks()` 删除 block 后，`rebuildBlockLookup()` 只保留当前存活 key，没有删除墓碑/历史；以后新观测或 gradient growing 再次命中同 key，就会新建完整十点并分配新 UID。这是删除后重建，不是 UID 重复、也不是已占用 key 去重失效。逐 key 全部 UID/来源在 `replay_spatial_key_creation_counts.csv`，相邻两次创建的空窗在 `replay_recreation_events.csv`。','',
'## 4. Anchor 寿命','',
'时间单位是 Mapper 已完成 Mapping 计数（`protection.mapping_iteration`，初始建图记 0）；不是视频帧、不是 Gaussian optimizer 更新次数。在线观测批次按上一个已完成 Mapping 计数入账，精度约一个 Mapping 步。删除寿命=`death_mapping-birth_mapping`。快速死亡阈值预先固定为 ≤1/10/50/100/250/500/1000，不按结果挑阈值。','',
'|复跑 seed|创建 / 删除 / 存活|已删平均寿命|中位数|p10 / p90|已删中 ≤100 步|已删中 ≤500 步|','|---|---:|---:|---:|---:|---:|---:|']
for r in replay:
    c=next(c for c in cohort if int(c['seed'])==r['seed'] and c['source']=='all')
    lines.append(f"|{r['seed']}|{r['created']} / {r['deleted']} / {r['final']}|{float(c['lifetime_mean']):.2f}|{float(c['lifetime_median']):.1f}|{float(c['lifetime_p10']):.1f} / {float(c['lifetime_p90']):.1f}|{float(c['dead_within_100_fraction']):.2%}|{float(c['dead_within_500_fraction']):.2%}|")
lines+=['','避免期末右删失偏差：仅对出生≤9500、至少有500步观察机会的 Anchor，统计“出生后500步内删除 / 所有可观察出生”：','']
for r in replay:
    c=next(c for c in cohort if int(c['seed'])==r['seed'] and c['source']=='all')
    lines.append(f"- seed {r['seed']}：{c['followup500_early_deaths']} / {c['followup500_eligible_births']} = {float(c['followup500_early_death_fraction']):.2%}。")
lines+=['','分来源的周转情况：','','|seed|来源|创建|删除比例|已删平均寿命|已删中 ≤500 步|','|---|---|---:|---:|---:|---:|']
for c in cohort:
    if c['source']=='all':continue
    lines.append(f"|{c['seed']}|{c['source']}|{c['created']}|{float(c['deleted_fraction']):.2%}|{float(c['lifetime_mean']):.1f}|{float(c['dead_within_500_fraction']):.2%}|")
lines+=['','已删除 Anchor 的平均寿命不是所有 Anchor 的预期寿命；存活 Anchor 是右删失样本。pruning 约每500 Mapping执行，寿命会受这个检查节奏量化。细层增长来源组的高短期删除率说明 growing 存在大量周转，但不能单独判断这些增长对质量的净效应。分来源的寿命与所有预注册阈值见 `replay_lifetime_summary.csv`；逐 UID 生死时刻见 `replay_deleted_anchor_lifetimes.csv`。','',
'## 5. 问题定位与后续验证边界','',
'1. **已证实的结构代价**：十点强绑定 + 全失败才删，使大量单独失败的 slot 长期留存。这符合此前确认的规则，不是实现违反固定十点约束。','2. **已证实的选择代价**：低于 renderer 截断的 slot 被 LoD 选中，占用真实预算；same-K 数量匹配也不等于有效贡献数量匹配。',
'3. **初始化风险**：少量当前批次来源撑起完整十点，且粗细层各初始化一组；它提供几何自由度，也增加弱支持成员。复跑支持数与最终低 opacity 的相关统计不足以证明初始化是唯一原因。',
'4. **生命周期风险**：删除 lookup 历史后允许同 key 再生，可能形成创建—优化—整组删—再创建的周转；实际次数与短寿命比例已量化。增长候选和真实观测必须分开分析。',
'5. **可观测性缺口**：原训练缺逐 UID 事件，无法追溯其三项历史问题。独立诊断补齐记录，但没有把复跑的数值冒充原运行。',
'6. 当前证据不支持把问题归因到“MLP 不够大/训练步数不够”；它首先暴露的是 slot 有效性、预算利用率、支持来源和生命周期周转。也不能用这次诊断宣称已经解释所有 Full/LPIPS 差异。',
'','本轮没有更改固定十点、整组 prune、增长阈值、网络、Loss、TF、CD 或训练轮转。任何后续修复都应单独对照，并同时保留真实 N、有效贡献 N、same-K、Full 与 LoD 指标。','',
'## 6. 验证与产物','',
'- 原模型只读：opacity 全量统计；64共同视图 alpha*T VJP 与守恒检查；相同视图的低 opacity 删除敏感性；原导出 mask 的预算占用。','- 复跑：CPU/CUDA 显式十点结构测试；CPU fixture 与原版本 exact；CUDA fixture 输出 exact，仅保存的 Anchor Feature 存在最大 1.17e-10 差异，allclose(atol=2e-6, rtol=2e-5) 通过。补做原二进制 CUDA repeat，其与原 fixture 的差异为0；没有把 CUDA 参数 bitwise gate 写成通过。两个10,000 Mapping复跑的出生UID唯一、出生减删除等于最终UID集合、N=10M、无同时重复key、寿命非负、日志计数闭合。',
'- 源码与模型保护：29文件 SHA、本地 HEAD/index 核对、原 checkpoint SHA。诊断只保存在独立目录。',
'- 服务器原实验：`/home/prmi/zwj/octree-explicit-20261021`。诊断：`/home/prmi/zwj/octree-explicit-diagnostics-20261021`。',
'- 本地明细：`tests/diagnostics/explicit_anchor_lifecycle_20261021/`。其中 `frozen/` 是原模型测量，`replay/` 是带事件记录的独立复跑。','',
f"![原模型与复跑诊断]({(here/'diagnosis.png').as_posix()})",'',
f"源码依据：[初始化与补点](<{(repo/'improvements/selector/explicit_anchor_blocks.h').as_posix()}:39>)；[整组删除与 lookup 重建](<{(repo/'improvements/selector/explicit_anchor_blocks.h').as_posix()}:193>)；[逐点 pruning predicate](<{(repo/'src/gaussian_model.cpp').as_posix()}:923>)；[Full opacity regularizer](<{(repo/'src/gaussian_mapper.cpp').as_posix()}:1589>)；[renderer alpha cutoff](<{(repo/'cuda_rasterizer/forward.cu').as_posix()}:411>)；[TF 属性输出](<{(repo/'python/transform_field.py').as_posix()}:188>)。"]
(repo/'OCTREE_EXPLICIT_ANCHOR_TRAINING_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(here/'audit_summary.json').write_text(json.dumps(dict(original_opacity=original,original_contribution=contribution,
    original_quality_sensitivity=quality,replay=replay,scope='DIAGNOSIS ONLY; NO ALGORITHM CHANGE'),indent=2),encoding='utf-8')
print(repo/'OCTREE_EXPLICIT_ANCHOR_TRAINING_DIAGNOSIS.md')
