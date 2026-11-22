"""从已验证的原始评价结果生成中文报告和质量曲线，不修改实验配置。"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'evidence/validation'
rows=list(csv.DictReader((out/'quality_summary.csv').open(newline='')))
paired=list(csv.DictReader((out/'paired_statistics.csv').open(newline='')))
models=json.loads((out/'model_diagnostics.json').read_text())
checks=json.loads((root/'local_delivery_checks.json').read_text());assert checks['passed']
runtime=json.loads((out/'training_runtime.json').read_text())
pick=lambda method,level,policy='cached_bernoulli':next(r for r in rows if (r['method'],r['level'],r['policy'])==(method,level,policy))
pair=lambda comparison,level,metric,policy='cached_bernoulli':next(r for r in paired if (r['comparison'],r['level'],r['metric'],r['policy'])==(comparison,level,metric,policy))
link=lambda name,path:f'[{name}]({path.as_posix()})'
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for method,label in [('tf_baseline','Selector + TF'),('shared_compensation','Shared selector + compensation'),('compensation_off','New model: compensation OFF')]:
    axes[0].plot([1,5,10,15],[float(pick(method,str(k))['psnr']) for k in [1,5,10,15]],'o-',label=label)
for method,label in [('tf_baseline','Selector + TF'),('shared_compensation','Shared selector + compensation')]:
    data=[r for r in rows if r['method']==method and r['policy']=='same_k']
    axes[1].plot([int(r['selected']) for r in data],[float(r['psnr']) for r in data],'o-',label=label)
    axes[2].plot([int(r['selected']) for r in data],[float(r['lpips']) for r in data],'o-',label=label)
for ax,title,x,y in zip(axes,['Production Cached Bernoulli','Exact same-K diagnostic','Exact same-K diagnostic'],['Ratio (%)','Selected K','Selected K'],['PSNR (dB), higher better','PSNR (dB), higher better','LPIPS, lower better']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.25)
axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'quality_curves.png',dpi=170);plt.close(fig)
text=['# 共享浅层 Selector＋选后补偿：实现与质量测试',
    '**结论：结构已实现并通过功能验证；补偿头有明确作用，但本次替换 TF 后未改善整体重建质量。** 同一新模型、同一选中集合下，补偿显著改善图像；与原可变成员＋TF 基线相比，Full 和四档原 ratio 的四指标均值均退化。same-K 仅最低预算改善，其余三个预算退化。不能据此判定 Anchor Feature 无法替代六平面，也不能将功能通过表述为质量成功。',
    '## 实现',
    '输入保持 27 维：Anchor Feature 16＋局部 Offset 3＋activated Scale 3＋SH DC 3＋activated Opacity 1＋log(ratio) 1。共享编码为 `27→32 ReLU`；私有选择分支为 `32→32 ReLU→2 softmax`；私有补偿分支为 `32→32 ReLU→10`。',
    '先对 N 个候选计算共享特征和选择概率，沿用 Cached Bernoulli 得到集合 S；随后只 gather `h[S]` 并解码 K 行补偿。缓存仅在当前 forward/backward 内存活。保持旧选择网络的参数命名与布局，旧 checkpoint 继续可读；新 checkpoint 含补偿 schema 和 head 权重。',
    '补偿输出为 δXYZ、δlog-scale、δquaternion_raw；应用 `XYZ′=XYZ+δXYZ`、`scale′=scale·exp(δlog-scale)`、`q′=normalize(normalize([1,0,0,0]+δq)⊗q)`。只改变当前 LoD render 的属性，Master 继续显式存储。最后一层 weight/bias exact zero，初始化几何恒等，选择概率 exact 不变。',
    'LoD 保持 joint：更新 Master Gaussian、共享层、两个私有分支和 Anchor Feature；Full 只更新 Master。补偿自 iteration 3000 开始，与基线 TF 启用时刻相同。共享参数由原 Selector optimizer ownership 管理，一次正常 backward 后更新一次。',
    '本次关闭 TF，不创建 HexPlane 或 TF optimizer；原 HexPlane 正则随之移除。原 photo、ratio loss、Cached Bernoulli、ratio 调度、可变成员生命周期、停用功能和 SI/GI disabled 保持不变；没有新增 loss、feature、采样策略或正则。',
    '几何 Change Distill 将原 TF 的补偿后几何目标运输并入 Selector 同一次 CD 更新，同时保持原选择概率目标；event-local Adam 不改正常 Adam 状态。当前实现要求 Selector/原 TF CD epochs 相等，本次均为 5。',
    '## 固定实验协议',
    '- `fr1_xyz / seed2026 / 10000 Mapping`，只训练一次候选；评价最终 checkpoint，无质量调参、无选最好 step。',
    '- 基线复用 `/home/prmi/zwj/variable-anchor-joint-quality-20261021/variable_seed2026`，没有重训固定或串行 joint 模型。',
    '- 原 joint：online 按 1%/5%/10%/15%/Full 轮转；tail 每轮 Full 后执行一个轮转 joint LoD。`serial_full_lod=0`，Full100=0。',
    '- 使用此前锁定的 64 个共同 RGB 视图，各自在线估计位姿；不是新挑的最好视图，也不声称严格 non-Bank。',
    '- 相同 K 从共同训练 ratio 区间内确定为 75/405/734/1064。每模型输入 ratio=K/N；以 cached logit margin 排 exact-K 仅作诊断，生产仍为 Cached Bernoulli。',
    '- 补偿 OFF 诊断复用新模型权重、ratio 和逐行 exact 相同的选中集合，只在推理时关闭补偿。它不等于重新训练一个无补偿模型。',
    '## Full-map 质量',
    '| 方法 | Gaussian | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |','|---|---:|---:|---:|---:|---:|']
for name,label in [('tf_baseline','旧 Selector＋TF'),('shared_compensation','共享 Selector＋补偿')]:
    r=pick(name,'Full');text.append(f"| {label} | {r['selected']} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
text+=['Full PSNR 下降 0.2845 dB，改善仅 15/64 视图；LPIPS 退化 0.01463，改善仅 5/64。Full 推理不执行选择或补偿，差异来自联合在线训练后的 Master 状态。',
       '## 生产 Cached Bernoulli 质量',
       '| ratio | 方法 | 实际 K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |','|---|---|---:|---:|---:|---:|---:|']
for level in ['1','5','10','15']:
    for method,label in [('tf_baseline','旧 TF'),('shared_compensation','共享补偿')]:
        r=pick(method,level);text.append(f"| {level}% | {label} | {r['selected']} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
text+=['四档 ΔPSNR 为 −0.5964 / −0.3290 / −0.5430 / −0.6764 dB；实际预算不完全相同，需结合下面的 same-K 表判断。',
       '## 相同 K：新方法减旧 TF',
       '| K | 旧 PSNR | 新 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for level in ['K75','K405','K734','K1064']:
    a=pick('tf_baseline',level,'same_k');b=pick('shared_compensation',level,'same_k')
    p=pair('shared_compensation-tf_baseline',level,'psnr','same_k')
    ds=[float(b[m])-float(a[m]) for m in ['psnr','ssim','lpips','photo']]
    text.append(f"| {level[1:]} | {float(a['psnr']):.4f} | {float(b['psnr']):.4f} | {ds[0]:+.4f} | {ds[1]:+.5f} | {ds[2]:+.5f} | {ds[3]:+.5f} | {round(float(p['improved_fraction'])*64)}/64 |")
text+=['K=75 四指标均值改善，但 LPIPS 中位数退化、仅 25/64 视图改善；其余三个 K 的四指标均值均退化。结论为 **same-K MIXED，主要预算退化**。',
       '## 补偿是否真的有效：同模型、同一选中集合 ON 减 OFF',
       '| ratio | K | OFF PSNR | ON PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |','|---|---:|---:|---:|---:|---:|---:|---:|']
for level in ['1','5','10','15']:
    a=pick('compensation_off',level);b=pick('shared_compensation',level);d=[float(b[m])-float(a[m]) for m in ['psnr','ssim','lpips','photo']]
    text.append(f"| {level}% | {b['selected']} | {float(a['psnr']):.4f} | {float(b['psnr']):.4f} | {d[0]:+.4f} | {d[1]:+.5f} | {d[2]:+.5f} | {d[3]:+.5f} |")
text+=['四档的 PSNR、SSIM、photo 均为 64/64 视图改善，LPIPS 改善比例分别为 100%/76.6%/65.6%/62.5%。说明头部学习到了有效补偿，同时联合训练后的 LoD 已依赖该补偿；这不是相对旧 TF 的收益。',
       '补偿幅度也较大：1% 时 δXYZ 的逐分量 RMS 为 0.3567、最大绝对分量 2.4077；15% 时为 0.0883、2.1012（世界坐标单位，基础 h=0.1）。1% 的 δlog-scale RMS 为 0.6677。这不是几乎零输出或完全没更新，但目前证据不足以定位是共享梯度、学习率、表示还是目标造成质量差距。',
       f'![质量曲线]({(root/"quality_curves.png").as_posix()})',
       '## 训练轨迹与初步成本',
       '| 项目 | 旧 TF | 共享补偿 |','|---|---:|---:|']
for key,label in [('gaussians','Gaussian'),('anchors','Anchor'),('full_renders','Full Render'),('lod_renders','LoD Render'),('gaussian_updates','Master Adam 更新'),('selector_updates','Selector 更新'),('tf_updates','TF 更新'),('online_end_mapping','online 结束 Mapping'),('selector_parameters','Selector＋补偿参数，不含 Feature'),('anchor_feature_values','Anchor Feature 标量数'),('tf_parameters','TF 参数')]:
    text.append(f"| {label} | {models[0][key]} | {models[1][key]} |")
text+=[f"候选补偿执行 {models[1]['compensation_forwards']} 次，累计解码 3,310,927 行；本次没有发生补偿 CD（0 步）。L0/L1/L2 Anchor 分布：旧 {models[0]['anchors_by_level']}，新 {models[1]['anchors_by_level']}。",
       f"10% 下 Python 完整选择＋补偿的五组均值：旧 {models[0]['inference_ms_mean']:.3f} ms，新 {models[1]['inference_ms_mean']:.3f} ms；临时峰值 allocated 增量分别为 {models[0]['inference_peak_extra_bytes']/1048576:.3f} / {models[1]['inference_peak_extra_bytes']/1048576:.3f} MiB。包含统计/同步与各自 N/K，不是固定人口的原生 SLAM FPS 对照。TF 约 118.7 万参数被 1386 个新增 head 参数替代；此项参数节省是确定的。",
       f"候选训练 wall time {runtime['wall_seconds']:.2f} 秒；基线已存记录 68.61 秒。异步 online 结束时间和实际更新次数不同，因此不据此声称训练加速或将质量差距单独归因于网络结构。",
       '## 验证状态与边界',
       '- **IMPLEMENTED / TESTED**：共享编码、私有选择头、选后 K 行补偿、TF 关闭、原 joint/预算/lifecycle、保存加载和导出接入。',
       '- **TESTED**：CPU/CUDA 零初始化恒等、原选择概率 exact 一致、不污染 RNG；补偿梯度到达 head/trunk/Feature；真实 renderer photo 反传、Full 参数与 Adam 隔离；原 online/tail 调度；旧可变成员/Mapping/CD 回归。',
       '- **TESTED**：最终真实 checkpoint 的四档 C++/Python mask exact 一致；补偿后 XYZ/Scale/Rotation 与 PLY 导出数值一致；Master SH/Opacity 沿用显式属性；模型和源文件 hash 冻结。',
       '- **PARTIALLY VERIFIED**：补偿几何 CD 已通过合成局部几何修正、教师冻结和正常 Adam 不变测试；本次真实训练的补偿 CD 为 0，不能声称已验证真实 Loop Closure 质量。',
       '- **NOT TESTED**：多场景、多 seed、真实 Loop Closure 压力测试、原生端到端性能；没有开展学习率或容量搜索。',
       '- 补偿头使用现有 Selector Adam，LR≈0.001；原 TF MLP 使用独立且更低的 LR 时钟。此轮没有匹配两者 LR 的消融。架构、补偿参数化、优化尺度和移除 HexPlane 正则共同变化，当前结果不能证明某一个因素单独失败。',
       f"- 本地已复算 {checks['arms']} 条评价、{checks['per_view_rows']} 条逐视图记录、{checks['paired_rows']} 条配对统计；核对 {checks['files_sha_verified']} 个证据文件、{checks['source_sha_verified']} 个源文件和 {checks['reused_baseline_files']} 个基线复用文件的 SHA256。",
       '## 交付',
       '- '+link('启用配置',repo/'cfg/selector/anchor_compensation.yaml')+'：共享补偿开启、TF 关闭；未强制覆盖历史配置。',
       '- '+link('质量汇总',out/'quality_summary.csv')+' / '+link('逐视图',out/'quality_per_view.csv')+' / '+link('配对统计',out/'paired_statistics.csv'),
       '- '+link('补偿幅度',out/'compensation_statistics.csv')+' / '+link('训练与成本',out/'model_diagnostics.json')+' / '+link('功能门控',out/'correctness.json')+' / '+link('交付核验',root/'local_delivery_checks.json'),
       '- '+link('最终 Selector＋补偿权重',root/'evidence/candidate_seed2026/10000_shutdown/ply/selector.pt'),
       '完成后停止，没有调参、扩网、新损失、恢复 TF 或追加训练；没有 stage、commit、push。']
document=text[0]
for previous,current in zip(text,text[1:]):
    document+=('\n' if previous.startswith('|') and current.startswith('|') else '\n\n')+current
(repo/'SHARED_SELECTOR_COMPENSATION_REPORT.md').write_text(document+'\n',encoding='utf-8')
print('REPORT_WRITTEN')
