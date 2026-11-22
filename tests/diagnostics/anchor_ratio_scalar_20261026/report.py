"""比较不加标量、log(ratio)、raw ratio三组最终在线结果。"""
import csv
import json
from pathlib import Path
import statistics as st

base=Path(__file__).resolve().parent
repo=base.parents[2]
out=base/'evidence/validation'
quality=list(csv.DictReader((out/'quality_summary.csv').open()))
selection=list(csv.DictReader((out/'selection_statistics.csv').open()))
runtime=json.loads((out/'training_runtime.json').read_text())
plan=json.loads((out/'evaluation_plan.json').read_text())
architecture=list(csv.DictReader((out/'model_architecture.csv').open()))
columns_table=[f'| {x["model"]} | {x["selector_input"]} | {x["compensation_input"]} | {float(x["scalar_column_norm"]):.5f} |' for x in architecture]
train=[.01,.05,.1,.15]
unseen=[.025,.075,.125]
def mean(method,policy,ratios,metric='psnr',seed=None):
    return st.mean(float(x[metric]) for x in quality if x['method']==method and x['policy']==policy and
                   x['split']=='nonbank' and float(x['ratio']) in ratios and (seed is None or int(x['seed'])==seed))
comparisons=[]
for candidate,reference in [('log','none'),('raw','none'),('raw','log')]:
    for policy in ['cached','same_k']:
        for group,ratios in [('train',train),('unseen',unseen),('all',sorted(train+unseen))]:
            for seed in [2026,2027,None]:
                row=dict(candidate=candidate,reference=reference,policy=policy,group=group,seed='mean' if seed is None else seed)
                for metric in ['psnr','ssim','lpips','photo']:
                    a=mean(reference,policy,ratios,metric,seed);b=mean(candidate,policy,ratios,metric,seed)
                    row['reference_'+metric]=a;row['candidate_'+metric]=b;row['delta_'+metric]=b-a
                comparisons.append(row)
with (base/'aggregate_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
budgets=[]
for method in ['none','log','raw']:
    for ratio in sorted(train+unseen):
        s=[x for x in selection if x['model'].startswith(method+'_') and float(x['ratio'])==ratio]
        p=st.mean(float(x['p_mean']) for x in s);real=st.mean(int(x['cached_k'])/int(x['population']) for x in s)
        budgets.append(dict(method=method,ratio=ratio,mean_probability=p,expected_relative_error=p/ratio-1,
            actual_fraction=real,actual_relative_error=real/ratio-1,actual_k_seed2026=int(s[0]['cached_k']),actual_k_seed2027=int(s[1]['cached_k'])))
with (base/'budget_interpolation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(budgets[0]));w.writeheader();w.writerows(budgets)
metric_rows=[]
summary={}
for method in ['log','raw']:
    summary[method]={}
    for group,ratios in [('train',train),('unseen',unseen),('all',train+unseen)]:
        selected=next(x for x in comparisons if x['candidate']==method and x['reference']=='none' and x['policy']=='same_k' and x['group']==group and x['seed']=='mean')
        ds=[mean(method,'same_k',ratios,seed=s)-mean('none','same_k',ratios,seed=s) for s in [2026,2027]]
        cached=mean(method,'cached',ratios)-mean('none','cached',ratios)
        summary[method][group]=dict(delta_psnr=selected['delta_psnr'],seed_deltas=ds,cached_delta_psnr=cached,
            delta_ssim=selected['delta_ssim'],delta_lpips=selected['delta_lpips'],delta_photo=selected['delta_photo'])
        metric_rows.append(f'| {method} | {group} | {cached:+.4f} | {selected["delta_psnr"]:+.4f} | {ds[0]:+.4f}/{ds[1]:+.4f} | {selected["delta_ssim"]:+.5f} | {selected["delta_lpips"]:+.5f} | {selected["delta_photo"]:+.5f} |')
ratios_table=[]
for ratio in sorted(train+unseen):
    vals=[mean(m,'same_k',[ratio]) for m in ['none','log','raw']]
    ratios_table.append(f'| {ratio*100:g}% | {plan["common_k"][str(ratio)]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[1]-vals[0]:+.4f} | {vals[2]-vals[0]:+.4f} |')
budget_table=[]
for x in budgets:
    if x['ratio'] in unseen:
        budget_table.append(f'| {x["method"]} | {x["ratio"]*100:g}% | {x["mean_probability"]*100:.3f}% | {x["actual_fraction"]*100:.3f}% | {x["expected_relative_error"]*100:+.1f}% | {x["actual_k_seed2026"]}/{x["actual_k_seed2027"]} |')
runs=[]
for name,x in runtime.items():
    a=x['anchor'];s=x['schedule']
    runs.append(f'| {name} | {a["iteration"]} | {a["gaussian_count"]} | {a["anchor_count"]} | {s["full_render_count"]} | {s["lod_render_count"]} | {a["compensation_forwards"]} | {x["wall_seconds"]:.2f} |')
full={m:mean(m,'Full',[1.])-mean('none','Full',[1.]) for m in ['log','raw']}
direct=mean('raw','same_k',train+unseen)-mean('log','same_k',train+unseen)
(base/'audit_summary.json').write_text(json.dumps(dict(summary=summary,full_delta_psnr=full,
    raw_minus_log_psnr=direct,nonbank=len(plan['nonbank_frames']),tests_run=False,isolated_build=False),indent=2))
text=f'''# 8＋8＋14结构的ratio标量输入对照

相对同批无标量对照：**log(ratio)** 的same-K non-Bank训练档差 **{summary['log']['train']['delta_psnr']:+.4f} dB**，插值档 **{summary['log']['unseen']['delta_psnr']:+.4f} dB**；**raw ratio** 分别 **{summary['raw']['train']['delta_psnr']:+.4f} / {summary['raw']['unseen']['delta_psnr']:+.4f} dB**。全部七档raw−log平均 **{direct:+.4f} dB**。每档与两seed分别报告，不能只看均值。

判断：log在训练档的两个seed均改善，是本轮训练档较好的编码；未见ratio的seed结果混合，不能认定插值泛化已改善。raw插值均值略升，但2.5%仍明显下降，不能仅凭三档均值称其泛化稳定。2.5% same-K差分别为log **{mean('log','same_k',[.025])-mean('none','same_k',[.025]):+.4f} dB**、raw **{mean('raw','same_k',[.025])-mean('none','same_k',[.025]):+.4f} dB**；预算偏差减小仍未换来该档质量改善，因此预算数量不是唯一问题。

## 结构与控制

三组均保留Anchor共享8D、四档ratio8D、四档补偿潜在14D（每Anchor共96标量）：

- none：`8+8+Offset3+Scale3+SH DC3+Opacity1=26`，共享Encoder26→32。
- log：在上述26维末尾追加 `log(ratio)`，共享Encoder27→32。
- raw：只追加 `ratio`，不输入log(ratio)，共享Encoder27→32。

三组原选择头均h32→32 ReLU→2；选中K行的h32加对应Anchor的补偿潜在14D，再通过46→32 ReLU→14输出原属性补偿。14D表只用于补偿。未见ratio的8D/14D均按raw ratio线性插值，范围外端点clamp；本轮不改变插值权重、Loss、预算、Cached Bernoulli或Joint。

log/raw在创建原26D网络后扩一列：复制原26列、bias、后续网络，新增列为零；使用ScopedTorchRNG避免消耗后续初始化随机状态。三个版本初始函数按构造相同，允许浮点差异；本轮按用户要求未运行单元/回归套件，不将此表述为额外通过了数值parity测试。新增输入只增加32个网络权重。

配置：`Selector.anchor_ratio_features: 1`、`Selector.compensation_method: "anchor_ratio_latent14"`，另设`Selector.anchor_ratio_scalar_input: "none" / "log" / "raw"`。checkpoint额外buffer记录模式，C++与Python按该buffer识别27D ratio-feature模型，保留旧27D Anchor16模型加载语义。

raw输入使用比例数值0.01/0.05/0.10/0.15，log使用这些数值的自然对数；未对任一组额外缩放或调整学习率。实际checkpoint如下，新增列范数为共享Encoder末列32个权重的L2范数：

| 模型 | Selector输入维数 | 补偿头输入维数 | 新增列范数 |
|---|---:|---:|---:|
{chr(10).join(columns_table)}

## 真实在线实验

fr1_xyz，seed2026/2027，三组共六次，从头真实SLAM。仅标量模式不同。当前Variable Anchor、Gaussian/Anchor交替拓扑、`anchor_only_pruning=0`，TF/SI/GI关闭，原joint与ratio轮转均保留；自然停止，不强制10000步，不根据non-Bank选LR、步数或checkpoint。

| 运行 | Mapping | Gaussian | Anchor | Full次数 | LoD次数 | 补偿次数 | 墙钟秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(runs)}

固定64个RGB评价帧，排除六次Bank并集后共 **{len(plan['nonbank_frames'])}** 个non-Bank视图。使用各自最终估计位姿。训练ratio1/5/10/15%，未见2.5/7.5/12.5%。Cached与same-K分开；same-K按`logit(p)-logit(cached_uniform)`排序，K采用六个模型共同预算。不把Full和Selector100%混用。

## Same-K质量

| Ratio | K | none PSNR | log PSNR | raw PSNR | log−none | raw−none |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ratios_table)}

| 方法 | 范围 | Cached ΔPSNR | same-K ΔPSNR | same-K ΔPSNR seed2026/2027 | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

PSNR/SSIM越高越好，LPIPS/photo越低越好。Full non-Bank PSNR：log−none **{full['log']:+.4f} dB**，raw−none **{full['raw']:+.4f} dB**。逐视图mean/median/p10/p90/改善比例含log−none、raw−none及raw−log三组比较，保存于paired_statistics.csv。

## 插值预算是否恢复

平均概率为sum(p)/N，实际比例为Cached K/N；不能将平均概率准确等同于选点质量良好。

| 方法 | 查询ratio | 平均选择概率 | 实际选择比例均值 | 期望预算相对误差 | 实际K seed2026/2027 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(budget_table)}

## 解释边界与交付

本轮是在线整体效果，Gaussian地图、位姿、拓扑、实际训练/Adam次数仍会分叉；初始函数匹配与same-K不能消除这些混杂。不能单凭本轮确认冻结地图下的纯因果机制，也不能把一个场景两个seed当成跨场景验证。若预算恢复而same-K质量未改善，说明解决数量泛化还不足以解决排序/补偿质量；若质量改善也不能直接认定共享梯度冲突已消除。

所有ratio-feature模型都走现有LibTorch输入路径，未引入新CUDA优化。服务器存在其他GPU作业，墙钟不是独占速度benchmark。只评价1%～15%，不宣称区间外泛化。潜在表在区间外仍clamp，但log/raw标量继续变化，区间外行为未评价。

服务器常用项目直接更新并复用build-performance-20261023-121515构建tum_rgbd；无隔离构建、无测试套件、无stage/commit/push。默认仍为R3；未将任一结果设为默认。

本地证据根：`tests/diagnostics/anchor_ratio_scalar_20261026/`。`aggregate_comparison.csv`、`budget_interpolation.csv`、`audit_summary.json`为汇总；`evidence/validation/`保存逐视图四指标、选择数量、配对统计、运行配置、实际结构和特征统计。完整服务器模型：`/home/prmi/zwj/anchor-ratio-scalar-20261026/`。
'''
(repo/'ANCHOR_RATIO_SCALAR_INPUT_AUDIT.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2));print('RAW_MINUS_LOG',direct);print('FULL_DELTA',full);print('REPORT_WRITTEN')
