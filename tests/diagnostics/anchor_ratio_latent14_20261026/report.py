"""汇总在线质量对照；只读取锁定最终 checkpoint 的结果。"""
import csv
import json
from pathlib import Path
import statistics as st

base = Path(__file__).resolve().parent
repo = base.parents[2]
out = base/'evidence/validation'
q = list(csv.DictReader((out/'quality_summary.csv').open()))
runtime = json.loads((out/'training_runtime.json').read_text())
plan = json.loads((out/'evaluation_plan.json').read_text())
methods = ['r3_independent14', 'anchor_ratio_latent14']
trained, unseen = [.01, .05, .1, .15], [.025, .075, .125]
def mean(method, policy, ratios, metric, seed=None):
    return st.mean(float(x[metric]) for x in q if x['method']==method and x['policy']==policy and
                   x['split']=='nonbank' and float(x['ratio']) in ratios and (seed is None or int(x['seed'])==seed))
groups = []
for policy in ['cached', 'same_k']:
    for label, ratios in [('train', trained), ('unseen', unseen), ('all', sorted(trained+unseen))]:
        for seed in [2026, 2027, None]:
            a = {m:mean(methods[0],policy,ratios,m,seed) for m in ['psnr','ssim','lpips','photo']}
            b = {m:mean(methods[1],policy,ratios,m,seed) for m in a}
            groups.append(dict(policy=policy,group=label,seed='mean' if seed is None else seed,
                **{'r3_'+m:v for m,v in a.items()},**{'latent_'+m:v for m,v in b.items()},
                **{'delta_'+m:b[m]-a[m] for m in a}))
with (base/'aggregate_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
table=[]
for r in sorted(trained+unseen):
    a=mean(methods[0],'same_k',[r],'psnr');b=mean(methods[1],'same_k',[r],'psnr')
    d=[mean(methods[1],'same_k',[r],'psnr',s)-mean(methods[0],'same_k',[r],'psnr',s) for s in [2026,2027]]
    table.append(f'| {r*100:g}% | {plan["common_k"][str(r)]} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {d[0]:+.4f} / {d[1]:+.4f} |')
run_table=[]
for name,item in runtime.items():
    a=item['anchor'];schedule=item['schedule']
    run_table.append(f'| {name} | {a["iteration"]} | {a["gaussian_count"]} | {a["anchor_count"]} | {item["wall_seconds"]:.2f} |')
summary={f'{x["policy"]}_{x["group"]}':x for x in groups if x['seed']=='mean'}
full_delta=mean(methods[1],'Full',[1.],'psnr')-mean(methods[0],'Full',[1.],'psnr')
positive=sum(mean(methods[1],'same_k',[r],'psnr',s)>mean(methods[0],'same_k',[r],'psnr',s) for r in trained+unseen for s in [2026,2027])
conclusion='整体改善，但需注意各档与seed差异' if summary['same_k_all']['delta_psnr']>0 else '本次未超过当前R3'
text=f'''# Anchor × Ratio 14D 潜在特征补偿实验

结论：**{conclusion}**。non-Bank same-K 平均差为 **{summary['same_k_all']['delta_psnr']:+.4f} dB**；训练档 **{summary['same_k_train']['delta_psnr']:+.4f} dB**，插值档 **{summary['same_k_unseen']['delta_psnr']:+.4f} dB**。14个 seed×ratio 对照中 {positive} 个 PSNR 上升。

## 实现

- 原 Anchor16、27D Selector 输入、32D 共享编码及选择头保持原结构；额外为每个 Anchor 的1/5/10/15%分别保存一张14D潜在特征表。
- 先按 Cached Bernoulli 选点，再为选中K行查询所属 Anchor 的潜在特征；补偿头 `(shared32 + latent14) -> 32 ReLU -> 14`。14维输出仍是 XYZ3、log-scale3、quaternion4、opacity-logit1、SH DC3。
- 未见ratio按原始ratio线性插值潜在特征，再解码；范围外用端点特征。原共享编码中仍保留 log(ratio)。这不是R3直接插值输出系数。
- 新表零初始化；头原有32列、bias和最后层复制，新增输入列保留随机非零初始化。初始零特征不改变原输出；从零补偿头训练时，特征梯度需要末层开始更新后才能传入。
- 四张独立Parameter，未使用ratio表梯度为None。特征及Adam随Anchor增长、prune、重排和闭环成员加权迁移；同层增长沿用逐通道max继承、细层零初始化。
- 配置开关：`Selector.compensation_method: "anchor_ratio_latent14"`。原全局默认R3未改；本轮未stage、commit、push。

## 真实在线训练与评价

fr1_xyz，从头SLAM，两个seed2026/2027。两臂使用同一个源码及二进制，配置仅补偿方法不同。保留当前joint、Full-only Master、原loss、ratio轮转、TF/SI/GI关闭和自然停止；不强制10000步。

当前配置为Variable Anchor1..10，`anchor_only_pruning=0`，Gaussian/Anchor交替拓扑；这与部分旧报告的Anchor-only配置不同。本轮只比较新跑的匹配R3，不拿旧报告质量直接相减。原16D Feature保留其正常joint训练，并非冻结其权重。

服务器直接更新 `/home/prmi/zwj/mycartgs`，复用 `build-performance-20261023-121515` 构建tum_rgbd；未进行隔离构建或运行测试套件。更新前差异源码和主程序已备份至实验目录backup。

| 运行 | 最终Mapping iteration | Gaussian | Anchor | 墙钟秒 |
|---|---:|---:|---:|---:|
{chr(10).join(run_table)}

相同64个RGB评价帧；剔除四次运行Bank并集后共 **{len(plan['nonbank_frames'])}** 个non-Bank。PSNR/SSIM/LPIPS/photo均保存逐视图；same-K按`logit(p)-logit(cached_uniform)`排序，K为共同最小population乘ratio并受共同eligible上限约束。Full另评无补偿Master。

## Same-K non-Bank PSNR

| Ratio | K | R3 | Anchor-ratio14 | 平均差dB | seed2026 / 2027差dB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

| 范围 | Cached差PSNR | same-K差PSNR | same-K差SSIM | same-K差LPIPS | same-K差photo |
|---|---:|---:|---:|---:|---:|
'''
for label in ['train','unseen','all']:
    x=summary['same_k_'+label];c=summary['cached_'+label]
    text+=f'| {label} | {c["delta_psnr"]:+.4f} | {x["delta_psnr"]:+.4f} | {x["delta_ssim"]:+.5f} | {x["delta_lpips"]:+.5f} | {x["delta_photo"]:+.5f} |\n'
text+=f'''
Full non-Bank PSNR差：{full_delta:+.4f} dB。SSIM/PSNR越高越好，LPIPS/photo越低越好。

## 参数与局限

每Anchor额外56个FP32参数；头增加448个权重。私有特征参数量为56M，对比R3为56N；因多个Gaussian共享同Anchor，新方案通常更省特征参数，但解码头可达的个体补偿仍受共享与非线性限制。表的参数+梯度+Adam一二阶状态估算为896M bytes，不等同于峰值显存。

本次两seed的R3表参数分别418,264 / 397,040，新方案112,056 / 129,696，减少约73.2% / 67.3%（不含头）。新增特征表均finite且RMS非零；每张表Adam实际步数为：R3的780 / 685，新方案680 / 680。自然停止及在线拓扑差异没有通过伪造optimizer步数补齐；这些优化曝光差异必须保留在解释中。

在线地图、位姿、实际更新步数可能分叉；same-K控制点数，不能消除地图差异，也不能独立证明潜在特征参数化的因果收益。两个seed来自同一场景，并非跨场景验证。服务器有其他GPU作业，本轮墙钟不能作为独占性能benchmark。

生命周期和原生checkpoint加载代码已接入，但按用户要求未跑对应单元/回归测试。实际完成在线增长、优化、保存和Python载入渲染；不能据此宣称所有闭环边界都已覆盖。三个插值ratio不构成对所有连续ratio的质量保证。

## 证据

- `tests/diagnostics/anchor_ratio_latent14_20261026/aggregate_comparison.csv`
- `evidence/validation/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`selection_statistics.csv`
- `evidence/validation/training_runtime.json`、`evaluation_plan.json`、`preregistration.json`
- `source_manifest.json`、`evidence/sync_receipt.json`、`evidence/validation/build.log`
- 服务器完整模型：`/home/prmi/zwj/anchor-ratio-latent14-20261026/`各方法_seed目录。
'''
(repo/'ANCHOR_RATIO_LATENT14_COMPENSATION_REPORT.md').write_text(text,encoding='utf-8')
(base/'audit_summary.json').write_text(json.dumps(dict(conclusion=conclusion,summary=summary,full_delta_psnr=full_delta,
    positive_seed_ratio_cases=positive,total_cases=14,tests_run=False,isolated_build=False),indent=2))
print(json.dumps(summary,indent=2));print('REPORT_WRITTEN')
