"""汇总8+8+14与匹配在线对照，不依据评价指标选择模型。"""
import csv
import json
from pathlib import Path
import statistics as st

base=Path(__file__).resolve().parent
repo=base.parents[2]
out=base/'evidence/validation'
quality=list(csv.DictReader((out/'quality_summary.csv').open()))
runtime=json.loads((out/'training_runtime.json').read_text())
plan=json.loads((out/'evaluation_plan.json').read_text())
candidate='anchor8_ratio8_comp14'
references=['anchor16_comp14','r3_independent14']
train=[.01,.05,.1,.15]
unseen=[.025,.075,.125]
def average(method,policy,ratios,metric='psnr',seed=None):
    return st.mean(float(x[metric]) for x in quality if x['method']==method and x['policy']==policy and
                   x['split']=='nonbank' and float(x['ratio']) in ratios and (seed is None or int(x['seed'])==seed))
rows=[]
for reference in references:
    for policy in ['cached','same_k']:
        for group,ratios in [('train',train),('unseen',unseen),('all',sorted(train+unseen))]:
            for seed in [2026,2027,None]:
                a={m:average(reference,policy,ratios,m,seed) for m in ['psnr','ssim','lpips','photo']}
                b={m:average(candidate,policy,ratios,m,seed) for m in a}
                rows.append(dict(reference=reference,policy=policy,group=group,seed='mean' if seed is None else seed,
                    **{'reference_'+m:v for m,v in a.items()},**{'candidate_'+m:v for m,v in b.items()},
                    **{'delta_'+m:b[m]-a[m] for m in a}))
with (base/'aggregate_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
mean_rows={(x['reference'],x['policy'],x['group']):x for x in rows if x['seed']=='mean'}
primary=mean_rows['anchor16_comp14','same_k','all']
seed_deltas=[average(candidate,'same_k',train+unseen,seed=s)-average(references[0],'same_k',train+unseen,seed=s) for s in [2026,2027]]
conclusion=('两个seed均优于上一轮结构' if min(seed_deltas)>0 else
            ('两个seed均未超过上一轮结构' if max(seed_deltas)<0 else '两个seed结果混合，尚无稳定改善'))
tables=[]
for r in sorted(train+unseen):
    a=average(references[0],'same_k',[r]);b=average(candidate,'same_k',[r]);c=average(references[1],'same_k',[r])
    ds=[average(candidate,'same_k',[r],seed=s)-average(references[0],'same_k',[r],seed=s) for s in [2026,2027]]
    tables.append(f'| {r*100:g}% | {plan["common_k"][str(r)]} | {a:.4f} | {b:.4f} | {c:.4f} | {b-a:+.4f} | {b-c:+.4f} | {ds[0]:+.4f}/{ds[1]:+.4f} |')
runs=[]
for name,x in runtime.items():
    a=x['anchor'];schedule=x['schedule']
    runs.append(f'| {name} | {a["iteration"]} | {a["gaussian_count"]} | {a["anchor_count"]} | {a["compensation_forwards"]} | {x["wall_seconds"]:.2f} |')
metric_rows=[]
for ref in references:
    for group in ['train','unseen','all']:
        x=mean_rows[ref,'same_k',group];c=mean_rows[ref,'cached',group]
        metric_rows.append(f'| {ref} | {group} | {c["delta_psnr"]:+.4f} | {x["delta_psnr"]:+.4f} | {x["delta_ssim"]:+.5f} | {x["delta_lpips"]:+.5f} | {x["delta_photo"]:+.5f} |')
positive={ref:sum(average(candidate,'same_k',[r],seed=s)>average(ref,'same_k',[r],seed=s) for s in [2026,2027] for r in train+unseen) for ref in references}
train_delta=mean_rows[references[0],'same_k','train']['delta_psnr']
unseen_delta=mean_rows[references[0],'same_k','unseen']['delta_psnr']
full={ref:average(candidate,'Full',[1.])-average(ref,'Full',[1.]) for ref in references}
summary=dict(conclusion=conclusion,train_delta_psnr=train_delta,unseen_delta_psnr=unseen_delta,seed_deltas_psnr=seed_deltas,
    mean_rows=list(mean_rows.values()),positive_cases=positive,full_delta_psnr=full,tests_run=False,isolated_build=False)
(base/'audit_summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# Anchor共享8D＋ratio8D＋补偿ratio14D质量实验

结论：**{conclusion}**。相对匹配的上一轮Anchor16结构，non-Bank same-K训练档平均 **{train_delta:+.4f} dB**，插值档 **{unseen_delta:+.4f} dB**，全部七档 **{primary['delta_psnr']:+.4f} dB**。14个seed×ratio病例中，优于上一轮结构{positive[references[0]]}个，优于R3 {positive[references[1]]}个。

## 实际结构

每个Anchor：共享8D一份、训练ratio各8D选择特征、各14D补偿特征，四档合计96个标量。

```
共享8 + ratio8 + Offset3 + activated Scale3 + SH DC3 + activated Opacity1
    -> 26 -> 32 ReLU（共享编码h）
选择头：h32 -> 32 ReLU -> 2
选中K行的 h32 + Anchor/ratio补偿潜在14
    -> 46 -> 32 ReLU -> 14
```

没有将ratio或log(ratio)数值拼入MLP输入。ratio仍决定预算和特征查询。未见ratio分别对8D、14D特征按raw ratio线性插值，再进入网络；超区间使用最近端点。14D潜在特征仅进入补偿头，输出依旧是XYZ3、log-scale3、quaternion4、opacity-logit1、SH DC3。

端点策略的直接限制：固定模型状态下，超过15%的两套特征均与15%一致；因没有额外ratio数值输入，Cached选择概率和补偿也相同。本轮只评价1%～15%区间内泛化，不能宣称支持20%～100%的预算泛化；Full仍是单独Master路径。

先按Cached Bernoulli选点，只有选中的K行读取14D并解码。共享8D及选择ratio8D同时接受原选择路径与补偿路径梯度，14D只接受补偿路径梯度。四档独立Parameter；未使用表梯度为None，Adam不推进。初始特征为零，补偿末层为零。26D与27D Selector初始函数不同，不能称为等价初始化。

复用现有Anchor增长、逐通道max同层继承、细层零初始化、prune、成员重排、闭环加权迁移及Adam行状态维护。C++和Python均支持新组合checkpoint；按用户要求未运行单元/回归套件，不能将代码接入表述为所有生命周期边界已测试。

实际导出的两个B checkpoint均为：Selector输入26D、补偿头输入46D、共享Feature8D、4张ratio8D表、4张补偿14D表。18张Anchor特征表全部finite且非零；8D ratio表RMS约0.0367～0.0934，14D表约0.0079～0.0197。特征不是完全不更新；非零行可能包含继承，不等于每行均获得过直接渲染梯度。

## 三组匹配在线对照

- A `anchor16_comp14`：上一轮Anchor16＋log(ratio)，27D共享输入，补偿额外ratio14D。
- B `anchor8_ratio8_comp14`：本轮8＋8＋14，无显式ratio输入。
- C `r3_independent14`：当前默认，Gaussian/ratio直接14D系数参考。

fr1_xyz真实SLAM、seed2026/2027，共六次。相同源码/二进制，配置仅补偿方法及`anchor_ratio_features`不同。保留当前joint、原loss、ratio轮转、Cached Bernoulli、Full Master、TF/SI/GI关闭，Variable Anchor及当前`anchor_only_pruning=0`。使用自然停止，未强制10000步，未按评价质量选择checkpoint或超参数。

| 运行 | Mapping iteration | Gaussian | Anchor | 补偿forward次数 | 墙钟秒 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(runs)}

复用相同64个RGB评价帧；剔除六次Bank并集后，non-Bank共 **{len(plan['nonbank_frames'])}** 帧。使用各自最终估计位姿。训练ratio为1/5/10/15%，未见ratio为2.5/7.5/12.5%。Full为无补偿Master，另行评价。

same-K排序为`logit(p)-logit(cached_uniform)`，K按六个模型共同最小Gaussian population乘ratio并受共同eligible上限约束。Cached生产选择与same-K诊断分别保存。

## Same-K non-Bank PSNR

| ratio | K | A上一轮 | B新方案 | C R3 | B−A | B−C | B−A seed2026/2027 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

## 四指标与生产选择

下表均为B减对应参考；PSNR/SSIM越高越好，LPIPS/photo越低越好。

| 参考 | 范围 | Cached ΔPSNR | same-K ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Full non-Bank PSNR：B−A **{full[references[0]]:+.4f} dB**，B−R3 **{full[references[1]]:+.4f} dB**。逐视图mean/median/p10/p90/改善比例见paired_statistics.csv。

B的Full相对A也下降，说明在线地图差异参与了结果，不能把全部LoD损失都归因于ratio特征。相对R3，Full平均接近而LoD明显较差，仍只构成本次在线稀疏路径的负面证据，并非冻结地图的因果隔离。

## 成本与解释边界

B每Anchor96标量，A为72；共享头输入从27变26，补偿头均46→32→14。B比A每Anchor多24个标量，新增状态并非免费。B不走原27D fused输入打包内核，而走现有LibTorch 26D路径；未新增CUDA算法。服务器有其他GPU作业，墙钟不是独占性能benchmark。

原默认R3保持不变。B配置为`Selector.anchor_ratio_features: 1`及`Selector.compensation_method: "anchor_ratio_latent14"`；源码与配置导出已同步服务器常用项目，复用现有build-performance-20261023-121515构建tum_rgbd，无隔离构建、无测试套件、无stage/commit/push。

在线拓扑、位姿、实际Full/LoD步数和Adam次数可能不同。same-K只控制数量，不能消除地图/训练量差异；本轮结果是在线整体效果，不是冻结地图下纯参数化因果结论。两个seed仍为一个场景，三个插值点不能保证所有连续ratio质量。完整模型保留服务器实验目录，未替换默认算法。

## 证据位置

- 本地实验：`tests/diagnostics/anchor_ratio8_compensation14_20261026/`
- `aggregate_comparison.csv`、`audit_summary.json`、`feature_statistics.csv`
- `evidence/validation/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`selection_statistics.csv`
- `evidence/validation/preregistration.json`、`training_runtime.json`、`evaluation_plan.json`
- `source_manifest.json`、`evidence/sync_receipt.json`、`evidence/source_after.json`、`evidence/validation/build.log`
- 服务器完整模型：`/home/prmi/zwj/anchor-ratio8-comp14-20261026/`
'''
(repo/'ANCHOR_RATIO8_COMPENSATION14_AUDIT.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2));print('REPORT_WRITTEN')
