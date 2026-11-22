"""根据已独立复算的结果生成中文报告和静态质量曲线。"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
repo = root.parents[2]
validation = root/'evidence/validation'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
table = lambda p: list(csv.DictReader(p.open(encoding='utf-8',newline='')))
assert read(root/'local_delivery_checks.json')['passed']
summary = table(validation/'quality_summary.csv')
paired = table(validation/'paired_statistics.csv')
diagnostics = {r['method']:r for r in table(validation/'model_diagnostics.csv')}
plan = read(validation/'evaluation_plan.json')
by = {(r['method'],r['policy'],r['level']):r for r in summary}
pair = {(r['policy'],r['level'],r['metric']):r for r in paired}
fixed, variable = diagnostics['fixed'], diagnostics['variable']

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig, axes = plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
colors = {'fixed':'#d07920','variable':'#276ab0'}
for name,label in [('fixed','固定十点'),('variable','可变成员')]:
    points = [by[name,'cached_bernoulli',level] for level in ['1','5','10','15']]
    axes[0].plot([int(p['selected']) for p in points],[float(p['psnr']) for p in points],marker='o',color=colors[name],label=label+'：Cached Bernoulli')
    for point in points:
        axes[0].annotate(point['level']+'%',(int(point['selected']),float(point['psnr'])),xytext=(3,6),textcoords='offset points',fontsize=8)
    full = by[name,'cached_bernoulli','Full']
    axes[0].scatter(int(full['selected']),float(full['psnr']),marker='*',s=120,color=colors[name])
    axes[0].annotate(label+' Full',(int(full['selected']),float(full['psnr'])),xytext=(-10,-15),ha='right',textcoords='offset points',fontsize=8)
axes[0].set_xscale('log')
axes[0].set_xlabel('实际渲染点数（对数轴）')
axes[0].set_ylabel('PSNR（dB）')
axes[0].legend(loc='lower right',fontsize=8)
axes[0].grid(alpha=.2)
ks = plan['common_k']
deltas = [float(pair['same_k','K'+str(k),'psnr']['mean']) for k in ks]
axes[1].bar(range(4),deltas,color=['#b54b49' if d<0 else '#318760' for d in deltas],width=.58)
axes[1].set_xticks(range(4),[str(k) for k in ks])
axes[1].axhline(0,color='#444',linewidth=.8)
for i,d in enumerate(deltas):
    axes[1].text(i,d+(.025 if d>=0 else -.025),f'{d:+.3f}',ha='center',va='bottom' if d>=0 else 'top')
axes[1].set_ylim(-.36,.83)
axes[1].set_xlabel('相同绝对 K；各模型输入 ratio=K/N')
axes[1].set_ylabel('可变成员 − 固定十点：ΔPSNR（dB）')
axes[1].grid(axis='y',alpha=.2)
fig.suptitle('原 joint｜fr1_xyz｜seed2026｜64 个共同评价视图')
fig.savefig(root/'quality_curves.png',dpi=180)
plt.close(fig)

lines = [
'# 可变成员 Anchor：原 joint 重建质量测试',
'',
'**结论：减少存储点和低 opacity 成员的目标达成；当前结果不支持“整体重建质量稳定提高”。Full 四指标均退化，原 ratio 四档 LoD 均退化；相同 K 下有中等预算收益，但存在预算和指标依赖。**',
'',
'## 实验定义与固定基线',
'',
'正式对照为 `fr1_xyz / seed2026 / 10,000 Mapping`。候选使用原 joint：online 按 1%/5%/10%/15%/Full 轮转；LoD 联合优化 Gaussian、Selector、Anchor Feature 和 TF，Full 只优化 Master。尾部保留原机制的 Full 后一个轮转 joint LoD。`Training.serial_full_lod=0`、Full100=0。没有使用此前串行 joint 的可变成员模型。',
'',
'- 固定十点：复用 `/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026` 的既有模型与 Full/原 ratio 质量结果。',
'- 可变成员：本次 `/home/prmi/zwj/variable-anchor-joint-quality-20261021/variable_seed2026`，只增加可变成员生命周期；其余参数沿用固定模型的输入配置，并显式关闭 serial/Full100。',
'- 两者均 SI/GI disabled、maturity=1、27 维网络、原 Loss、Cached Bernoulli、原 TF/CD，slot 停用设置保留。没有根据质量调参数或选择 checkpoint。',
'- 沿用此前锁定的 64 个共同时间采样 RGB 视图；不是按质量选帧，也不声称是严格的 non-Bank 集合。两模型使用各自在线估计位姿。',
'',
'用户补充“复用固定实验”时，刚启动的固定重复训练已完成。该结果保留在本次目录的 `fixed_seed2026`，**不参与任何正式质量表**。后续没有追加训练；固定模型只补充新的 same-K 推理。原始协议和用户指令修订分别保存在 `training_protocol.json`、`protocol_amendment.json`。',
'',
'## Full-map 质量',
'',
'PSNR/SSIM 越高越好，LPIPS/photo 越低越好。photo 使用原生 `0.8 L1 + 0.2 (1-fast_ssim)`，不含训练 opacity 正则。',
'',
'| 方法 | Gaussian | PSNR | SSIM | LPIPS | photo |',
'|---|---:|---:|---:|---:|---:|'
]
for name,label in [('fixed','固定十点'),('variable','可变成员')]:
    r = by[name,'cached_bernoulli','Full']
    lines.append(f"| {label} | {int(r['selected']):,} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
lines += ['', 'Full 差值：PSNR **−0.2493 dB**、SSIM −0.00815、LPIPS +0.01666、photo +0.00287。PSNR 改善 28/64 视图，LPIPS 改善仅 3/64；不能将均值 PSNR 接近解释为感知质量保持不变。', '',
'## 原 ratio 的生产 Cached Bernoulli', '',
'这些是生产选择结果，点数随各自地图人口变化，不能作为相同预算的纯选择器比较。', '',
'| ratio | 方法 | 实际点数 | PSNR | SSIM | LPIPS | photo |',
'|---|---|---:|---:|---:|---:|---:|']
for level in ['1','5','10','15']:
    for name,label in [('fixed','固定十点'),('variable','可变成员')]:
        r = by[name,'cached_bernoulli',level]
        lines.append(f"| {level}% | {label} | {int(r['selected']):,} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
lines += ['', '四档 PSNR 差值分别 **−3.0418 / −2.4011 / −1.6458 / −1.4567 dB**；四个指标的均值均退化。地图显著缩小后，继续使用同一名义 ratio 会分配更少的绝对点数，这一预算变化必须与选择质量分开看。', '',
'## 相同绝对 K 的质量', '',
'预先按人口确定共同预算区间 `[ceil(0.01 × max N), floor(0.15 × min N)] = [427,1064]`，均匀取四个整数 K。每个模型以 **K/N** 同时输入 Selector 与 TF，均处于训练 ratio 的 1%～15% 数值范围内，避免外推；大多数不是直接训练过的四个 ratio，因此结果也包含 Selector/TF 的 ratio 插值行为。使用 `logit(p_effective)−logit(cached_uniform)` 加 source-row tie-break 取 exact K，仅作诊断，没有替换生产采样算法。', '',
'| K | 固定输入 ratio | 可变输入 ratio | 固定 PSNR | 可变 PSNR | ΔPSNR | PSNR 改善视图 |',
'|---|---:|---:|---:|---:|---:|---:|']
for k in ks:
    a,b = by['fixed','same_k','K'+str(k)],by['variable','same_k','K'+str(k)]
    p = pair['same_k','K'+str(k),'psnr']
    lines.append(f"| {k} | {float(a['ratio'])*100:.3f}% | {float(b['ratio'])*100:.3f}% | {float(a['psnr']):.4f} | {float(b['psnr']):.4f} | {float(p['mean']):+.4f} | {round(float(p['improved_fraction'])*64)}/64 |")
lines += ['', '| K | ΔSSIM | ΔLPIPS | Δphoto |', '|---|---:|---:|---:|']
for k in ks:
    p = [float(pair['same_k','K'+str(k),m]['mean']) for m in ['ssim','lpips','photo']]
    lines.append(f'| {k} | {p[0]:+.5f} | {p[1]:+.5f} | {p[2]:+.5f} |')
lines += ['',
'K=852 的四指标均值全部改善，PSNR/LPIPS/photo 分别改善 50/56/53 个视图。K=639 的 PSNR/photo 改善，但 LPIPS 均值略差；K=1064 的 PSNR 略升，LPIPS 退化；K=427 的 PSNR/photo 退化。应判为 **same-K MIXED**，不能只引用最高的 +0.653 dB。', '',
f'![原 joint 质量曲线]({(root/"quality_curves.png").as_posix()})', '',
'## 人口、低 opacity 与实际训练轨迹', '',
'| 项目 | 固定十点 | 可变成员 |', '|---|---:|---:|']
fields = [('gaussians','Gaussian 总数'),('anchors','Anchor 数'),('below_cutoff','opacity < 1/255 的存储点'),('below_002','opacity < 0.02 的存储点'),('inactive','窗口停用点'),('full_renders','Full Render'),('lod_renders','LoD Render'),('gaussian_updates','Master Adam 更新'),('selector_updates','Selector 正常更新'),('tf_updates','TF 正常更新'),('online_end_mapping','SLAM online 结束 Mapping')]
for key,label in fields:
    lines.append(f"| {label} | {int(fixed[key]):,} | {int(variable[key]):,} |")
lines += ['',
'存储 Gaussian 减少 **83.35%**。opacity ≥1/255 的存储点从 13,368 降至 7,099，减少 **46.90%**；这仍不是逐点真实渲染贡献统计。最终可变成员模型中，每个点当前 opacity 均 ≥0.02，四档生产选择均没有低于 renderer 截断的点。', '',
'平均成员数为 3.948；L0/L1/L2 Anchor 分别为 627/1,157/14。新模型人口、初始点覆盖、增长、剪枝和在线运行时序一起变化，不能将收益或损失单独归因于某一项。原 joint 调度规则相同，但异步 SLAM 结束时间不同，导致实际 Full/LoD 更新次数不同；本轮没有强行改调度以匹配次数。', '',
f"原生 Selector forward 的五组均值为固定 {float(fixed['native_forward_ms']):.3f} ms、可变 {float(variable['native_forward_ms']):.3f} ms，分别测各自实际地图。该差异包含人口变化，不是相同 N 下的算法加速对照。候选训练 wall time 为 68.61 秒，不作为严格性能基准。", '',
'## 验证与交付', '',
'- 18 个评价条目、1,152 条逐视图结果、36 条配对统计已在本地从原始数组重新计算；共同帧、实际选择行和 exact K 全部核验。',
'- 250 个证据文件与 57 个已验证源文件 SHA256 匹配；58 个固定评价文件直接复用且与此前本地归档 hash 一致。',
'- 两个正式模型的四档 C++/Python/导出 Cached Bernoulli mask exact 一致，评价前后模型 SHA 不变。',
'- 候选成员数、Anchor 归属、topology 交替、maturity=1、原 joint 与 Full100=0 的状态核验通过。',
'',
f'- [质量汇总 CSV]({(validation/"quality_summary.csv").as_posix()})',
f'- [逐视图 CSV]({(validation/"quality_per_view.csv").as_posix()})',
f'- [配对统计 CSV]({(validation/"paired_statistics.csv").as_posix()})',
f'- [地图与训练统计]({(validation/"model_diagnostics.csv").as_posix()})',
f'- [本地交付核验]({(root/"local_delivery_checks.json").as_posix()})', '',
'本轮仅支持这一个 scene/seed 的结论。减少低 opacity 点没有自动换来整体画质提升；可变成员方案在中等绝对预算上有收益，同时 Full 感知质量与最低预算仍有损失。本轮止于质量测试，没有依据结果调 pruning、容量、网络或 Loss，也没有 stage、commit 或 push。', '']
(repo/'VARIABLE_ANCHOR_JOINT_QUALITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('WROTE VARIABLE_ANCHOR_JOINT_QUALITY_REPORT.md and quality_curves.png')
