"""同批比较16+8、8+8、16共享与R3，不按评价指标调参。"""
import csv
import json
from pathlib import Path
import statistics as st

base=Path(__file__).resolve().parent
repo=base.parents[2]
out=base/'evidence/validation'
q=list(csv.DictReader((out/'quality_summary.csv').open()))
selection=list(csv.DictReader((out/'selection_statistics.csv').open()))
runtime=json.loads((out/'training_runtime.json').read_text())
plan=json.loads((out/'evaluation_plan.json').read_text())
candidate='shared16_ratio8'
references=['shared16','shared8_ratio8','r3_independent14']
train=[.01,.05,.1,.15]
unseen=[.025,.075,.125]
def mean(method,policy,ratios,metric='psnr',seed=None):
    return st.mean(float(x[metric]) for x in q if x['method']==method and x['policy']==policy and x['split']=='nonbank'
                   and float(x['ratio']) in ratios and (seed is None or int(x['seed'])==seed))
rows=[]
for reference in references:
    for policy in ['cached','same_k']:
        for group,ratios in [('train',train),('unseen',unseen),('all',train+unseen)]:
            for seed in [2026,2027,None]:
                row=dict(reference=reference,policy=policy,group=group,seed='mean' if seed is None else seed)
                for metric in ['psnr','ssim','lpips','photo']:
                    a=mean(reference,policy,ratios,metric,seed);b=mean(candidate,policy,ratios,metric,seed)
                    row['reference_'+metric]=a;row['candidate_'+metric]=b;row['delta_'+metric]=b-a
                rows.append(row)
with (base/'aggregate_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
metric_rows=[]
for reference in references:
    summary[reference]={}
    for group,ratios in [('train',train),('unseen',unseen),('all',train+unseen)]:
        x=next(x for x in rows if x['reference']==reference and x['policy']=='same_k' and x['group']==group and x['seed']=='mean')
        ds=[mean(candidate,'same_k',ratios,seed=s)-mean(reference,'same_k',ratios,seed=s) for s in [2026,2027]]
        cached=mean(candidate,'cached',ratios)-mean(reference,'cached',ratios)
        summary[reference][group]=dict(delta_psnr=x['delta_psnr'],seed_deltas=ds,cached_delta_psnr=cached,
            delta_ssim=x['delta_ssim'],delta_lpips=x['delta_lpips'],delta_photo=x['delta_photo'])
        metric_rows.append(f'| {reference} | {group} | {cached:+.4f} | {x["delta_psnr"]:+.4f} | {ds[0]:+.4f}/{ds[1]:+.4f} | {x["delta_ssim"]:+.5f} | {x["delta_lpips"]:+.5f} | {x["delta_photo"]:+.5f} |')
ratio_rows=[]
for ratio in sorted(train+unseen):
    v={m:mean(m,'same_k',[ratio]) for m in references+[candidate]}
    ratio_rows.append(f'| {ratio*100:g}% | {plan["common_k"][str(ratio)]} | {v["shared16"]:.4f} | {v["shared8_ratio8"]:.4f} | {v[candidate]:.4f} | {v["r3_independent14"]:.4f} |')
runs=[]
for name,x in runtime.items():
    a=x['anchor'];s=x['schedule']
    runs.append(f'| {name} | {a["iteration"]} | {a["gaussian_count"]} | {a["anchor_count"]} | {a["input_dim"]} | {s["full_render_count"]} | {s["lod_render_count"]} | {a["compensation_forwards"]} | {x["wall_seconds"]:.2f} |')
budgets=[]
for method in references+[candidate]:
    for ratio in sorted(train+unseen):
        a=[x for x in selection if x['model'].startswith(method+'_') and float(x['ratio'])==ratio]
        p=st.mean(float(x['p_mean']) for x in a);real=st.mean(int(x['cached_k'])/int(x['population']) for x in a)
        budgets.append(dict(method=method,ratio=ratio,mean_probability=p,expected_relative_error=p/ratio-1,actual_fraction=real))
with (base/'budget_interpolation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(budgets[0]));w.writeheader();w.writerows(budgets)
budget_rows=[f'| {x["method"]} | {x["mean_probability"]*100:.3f}% | {x["actual_fraction"]*100:.3f}% | {x["expected_relative_error"]*100:+.1f}% |' for x in budgets if x['ratio']==.025]
full={r:mean(candidate,'Full',[1.])-mean(r,'Full',[1.]) for r in references}
(base/'audit_summary.json').write_text(json.dumps(dict(summary=summary,full_delta_psnr=full,
    nonbank=len(plan['nonbank_frames']),tests_run=False,isolated_build=False),indent=2))
text=f'''# 16D共享＋8D ratio Anchor Feature质量实验

结果：16＋8＋log相对8＋8＋log的same-K non-Bank训练档差 **{summary['shared8_ratio8']['train']['delta_psnr']:+.4f} dB**、插值档 **{summary['shared8_ratio8']['unseen']['delta_psnr']:+.4f} dB**；相对原16D共享分别 **{summary['shared16']['train']['delta_psnr']:+.4f} / {summary['shared16']['unseen']['delta_psnr']:+.4f} dB**；相对同批R3分别 **{summary['r3_independent14']['train']['delta_psnr']:+.4f} / {summary['r3_independent14']['unseen']['delta_psnr']:+.4f} dB**。两seed与每档明细见下表，不能只按均值判定稳定性。

## 实际改动

每Anchor共享16D，训练ratio1/5/10/15%各存8D选择特征及14D补偿潜在特征，总计104个标量。35D输入：`共享16+ratio8+Offset3+Scale3+SH DC3+Opacity1+log(ratio)1`，共享Encoder35→32 ReLU；选择头h32→32 ReLU→2。选中K行h32加补偿潜在14D，通过46→32 ReLU→14解码原XYZ/Scale/Rotation/Opacity/SH DC补偿。8D与14D在未见ratio上均按raw ratio线性插值，范围外端点clamp；log(ratio)仍使用实际查询值。

保留8＋8版的原26D编码器列映射，将新增共享8D插入中间；其特征从零开始，新增权重采用普通Linear初始化。后续层、bias及随机时钟保留。再按原方式追加零权重log列。按构造16＋8与8＋8初始函数相同（浮点精度内），补偿初始均零；按用户要求未额外跑parity/单元套件，不把构造说明写成实测通过。

新增配置`Selector.anchor_ratio_shared_dim: 16`（旧默认8），配合`anchor_ratio_features: 1`、`anchor_ratio_scalar_input: "log"`和`compensation_method: "anchor_ratio_latent14"`。新checkpoint布局v9；C++/Python兼容34/35D输入，保留旧8＋8、16共享及R3加载路径。ratio表始终8D，不随共享宽度变为16D。

Anchor增长、同层max继承、细层零初始化、prune、成员重排、闭环加权迁移与Adam行迁移复用现有动态宽度路径。只改变特征参数化，没有增加Loss、改变采样器、调度或TF。

## 同批在线对照

| 标签 | 结构 | 补偿 |
|---|---|---|
| shared16 | 共享Anchor16＋log(ratio) | 额外ratio潜在14D，MLP解码 |
| shared8_ratio8 | 共享8＋ratio8＋log(ratio) | 同上 |
| shared16_ratio8 | 本轮共享16＋ratio8＋log(ratio) | 同上 |
| r3_independent14 | 原共享Anchor16＋log(ratio) | Gaussian/ratio直接14D系数 |

fr1_xyz、seed2026/2027，共八次真实SLAM，从头训练，同一源码/二进制；R3为同批参考，不沿用旧报告质量直接相减。Variable Anchor、当前Gaussian/Anchor交替拓扑、`anchor_only_pruning=0`、原joint、Cached Bernoulli、ratio轮转、原photo及预算Loss、TF/SI/GI关闭均保留。自然停止，不强制10000步，不按non-Bank选超参数或checkpoint。

| 运行 | Mapping | Gaussian | Anchor | 输入维数 | Full次数 | LoD次数 | 补偿次数 | 墙钟秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(runs)}

固定64个RGB评价帧，剔除八次Bank并集后共 **{len(plan['nonbank_frames'])}** 个non-Bank。各地图使用自己的最终估计位姿。训练ratio1/5/10/15%，未见2.5/7.5/12.5%；Full为独立原Master评价。

Cached生产选择与same-K诊断分别保存。same-K按`logit(p)−logit(cached_uniform)`排序，所有八个模型使用共同K与相同视图。它控制点数，不能消除在线地图、位姿与Adam次数差异。

## Same-K non-Bank PSNR

| ratio | K | 原16共享 | 8＋8＋log | 16＋8＋log | R3 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(ratio_rows)}

以下均为16＋8减参考；PSNR/SSIM越高越好，LPIPS/photo越低越好。

| 参考 | 范围 | Cached ΔPSNR | same-K ΔPSNR | same-K ΔPSNR seed2026/2027 | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Full non-Bank PSNR差：相对16共享 **{full['shared16']:+.4f} dB**、8＋8 **{full['shared8_ratio8']:+.4f} dB**、R3 **{full['r3_independent14']:+.4f} dB**。逐视图mean/median/p10/p90与改善比例保存于paired_statistics.csv。

归因限制：相对8＋8，Full本身也改善，因此不能将LoD增益全部解释为额外共享8维提高了Selector表达能力。相对R3，Full均值更高而LoD仍低，差距也不能简单归为Master总体质量不足；具体空间分布、优化次数和稀疏路径尚未完全隔离。2.5%平均预算仍不足，增加共享维度未解决插值预算泛化。

## 2.5%插值预算

| 方法 | 平均选择概率 | 实际Cached比例均值 | 期望预算相对误差 |
|---|---:|---:|---:|
{chr(10).join(budget_rows)}

## 成本和局限

16＋8比8＋8每Anchor多8个FP32标量，Encoder多256个权重；参数、梯度及Adam一二阶状态合计增加约128×Anchor数＋4096 bytes（估算，不是峰值显存）。相对纯16共享，则多四档8D特征表。补偿头宽度和14D表数量不变。所有ratio-feature模型都使用现有LibTorch输入路径；未新增CUDA融合，服务器其他GPU作业使墙钟不能当独占benchmark。

本轮观察的是在线整体效果，不是冻结同一地图的纯容量结论；两seed仍属于一个场景。初始化按构造匹配不保证训练轨迹一致，same-K也不保证相同空间覆盖。未运行单元/回归套件，因此不能宣称所有几何迁移及历史checkpoint边界均已实测。未评价1%～15%以外质量。

服务器直接同步常用项目并复用build-performance-20261023-121515构建tum_rgbd，无隔离构建、无stage/commit/push。默认仍R3，未自动替换。

本轮只重建并运行tum_rgbd；其他应用入口未重建。因Mapper头文件布局有新增字段，使用其他入口前应基于当前源码重编，不应将本轮构建记录视为全部入口已更新。

## 证据

本地：`tests/diagnostics/anchor_shared16_ratio8_20261026/`。汇总为aggregate_comparison.csv、budget_interpolation.csv、audit_summary.json；evidence/validation含逐视图四指标、共同K、选择统计、真实结构、特征分布、训练配置、Full/LoD次数及原始日志。服务器完整模型：`/home/prmi/zwj/anchor-shared16-ratio8-20261026/`。
'''
(repo/'ANCHOR_SHARED16_RATIO8_AUDIT.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2));print('FULL_DELTA',full);print('REPORT_WRITTEN')
