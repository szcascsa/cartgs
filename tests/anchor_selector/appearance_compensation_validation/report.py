"""从本地已复核的实验数据生成中文实现说明、质量对照和图表。"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'evidence/validation'
checks=json.loads((root/'local_delivery_checks.json').read_text());assert checks['passed']
rows=list(csv.DictReader((out/'quality_summary.csv').open(newline='')))
pairs=list(csv.DictReader((out/'paired_statistics.csv').open(newline='')))
models=json.loads((out/'model_diagnostics.json').read_text())
stats=list(csv.DictReader((out/'compensation_statistics.csv').open(newline='')))
plan=json.loads((out/'evaluation_plan.json').read_text())
pick=lambda m,l,p='cached_bernoulli':next(r for r in rows if (r['method'],r['level'],r['policy'])==(m,l,p))
pair=lambda c,l,m,p='cached_bernoulli':next(r for r in pairs if (r['comparison'],r['level'],r['metric'],r['policy'])==(c,l,m,p))
delta=lambda a,b,m:float(b[m])-float(a[m])
link=lambda label,path:f'[{label}]({path.as_posix()})'
levels=['1','5','10','15'];metrics=['psnr','ssim','lpips','photo']
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for method,label in [('geometry_v1','Geometry 10D'),('appearance_v2','Geometry + appearance 14D'),('appearance_off','14D model: appearance OFF')]:
    axes[0].plot([1,5,10,15],[float(pick(method,l)['psnr']) for l in levels],'o-',label=label)
for method,label in [('geometry_v1','Geometry 10D'),('appearance_v2','Geometry + appearance 14D')]:
    data=[r for r in rows if r['method']==method and r['policy']=='same_k']
    for ax,m in [(axes[1],'psnr'),(axes[2],'lpips')]:
        ax.plot([int(r['selected']) for r in data],[float(r[m]) for r in data],'o-',label=label)
for ax,title,x,y in zip(axes,['Production Cached Bernoulli','Exact same-K diagnostic','Exact same-K diagnostic'],['Ratio (%)','Selected K','Selected K'],['PSNR (dB), higher better','PSNR (dB), higher better','LPIPS, lower better']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.25)
axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'quality_curves.png',dpi=170);plt.close(fig)
count=sum(delta(pick('geometry_v1',l),pick('appearance_v2',l),'psnr')>0 and delta(pick('geometry_v1',l),pick('appearance_v2',l),'photo')<0 for l in levels)
text=['# Selector 的 Opacity 与颜色补偿：实现和质量测试',
      f'**已实现 14 维补偿并通过 14 项功能与回归门控。** 单次 `fr1_xyz / seed2026 / 10000 Mapping` 对照中，原 ratio 有 {count}/4 档同时改善 PSNR 和 photo；完整质量、相同 K 与同集合外观消融列于下方，不据单次结果推断跨场景稳定性。',
      '10%／15% 的 PSNR 分别提升 0.3549／0.5395 dB；1%／5% 分别下降 0.2315／0.0478 dB。相同 K 下三档 PSNR 提升，最低预算下降；应判为有中高预算收益、最低预算尚未改善。新增外观补偿的同集合推理作用是 PSNR／photo 四档均改善，但 5%／10%／15% 的 LPIPS 略差，不能声称所有指标都提升。',
      '## 补偿定义',
      '共享浅层和选择分支保持不变；私有补偿头从 `32→32 ReLU→10` 扩为 `32→32 ReLU→14`。先 Cached Bernoulli 选择，再仅对 K 个选中点的缓存特征解码。',
      '| 分量 | 维度 | 应用方式 |','|---|---:|---|',
      '| XYZ | 3 | `XYZ′=XYZ+δXYZ` |','| Scale | 3 | `scale′=scale·exp(δlog-scale)` |',
      '| Rotation | 4 | 原单位四元数残差与原旋转相乘并归一化 |',
      '| Opacity | 1 | `opacity′=sigmoid(master_opacity_logit+δopacity_logit)` |',
      '| 颜色基色 SH DC | 3 | `SH_DC′=master_SH_DC+δSH_DC` |',
      '颜色只补偿 RGB 基色对应的 SH DC。高阶 SH 保留原值，不增加相机或视角输入。基色 RGB 的变化是 `0.28209479·δSH_DC`；沿用原 renderer 的颜色处理，不额外 clamp。',
      'Opacity 直接读取 Master 的显式原始 logit；不通过 `logit(sigmoid(logit))` 反算，因此有限但饱和的 logit 不会被变成 Inf。ST gate 仅在补偿后乘一次：`render_opacity=ST_mask·opacity′`。',
      '全部输出最后层 weight/bias 仍 exact zero 初始化，补偿在 iteration 3000 开始。LoD 联合更新 Master、Selector、两个分支和 Anchor Feature；Full 只优化原 Master Gaussian，不执行任何补偿。',
      '正常 photo、ratio loss、27 维输入、Cached Bernoulli、原 joint online/tail 调度、可变成员生命周期、停用机制、SI/GI disabled 和 TF disabled 均保留。新增输出增加 132 个参数，FP32 权重增加 528 bytes；这不等于已测出整个训练峰值显存。',
      '剪枝、停用与 Selector 输入仍读取 Master Opacity／SH DC，补偿只影响当前 LoD 渲染；本轮没有改变候选资格或 pruning 规则。',
      '## 保存、导出与 Change Distill',
      '- 新 checkpoint schema=2，对应 14 维；旧 schema=1 继续按 10 维加载、克隆和推理，旧真实模型的原生选择及几何输出 exact 不变。不会静默把旧权重升级为 14 维。',
      '- Python 推理、直接 prepared renderer 和标准 PLY 导出均使用补偿后的 Opacity logit 与 SH DC；高阶 SH 保留原值。未选中点的完整属性在临时补偿结果中保持 exact 原值。',
      '- 几何 Change Distill 继续运输 XYZ／Scale／Rotation；额外保留补偿后的 Opacity 和 SH DC。仅 CD 阶段增加新增属性的 L1 保真项：`mean|α′student−α′teacher| + C0·mean|DC′student−DC′teacher|`，不增加正常重建 loss 或正则。',
      '- 教师冻结、Master 不被 CD 修改、正常 Adam 状态不变、event-local Adam 更新和新旧 schema 均已做功能验证。',
      '## 实验协议',
      '- 仅新训练一次 14 维候选；旧 10 维模型直接复用上一轮保存结果。两轮输入 YAML 的 SHA256 exact 相同，保持同 seed、10000 Mapping、最后 checkpoint，不根据评价调参或选步数。',
      '- 原 joint，`serial_full_lod=0`、Full100=0、TF 关闭。online 为 1%/5%/10%/15%/Full 轮转，tail 保留 Full 后一个轮转 joint LoD。',
      '- 使用原先锁定的 64 个共同 RGB 视图和各自在线估计位姿；不是按质量选帧，不声称是严格 non-Bank。',
      f"- 共同整数预算 K={plan['common_k']}，每模型输入 ratio=K/N，均在 1%～15% 内；cached logit margin exact-K 仅作诊断，生产仍 Cached Bernoulli。",
      '- 外观 OFF 只在新模型推理时忽略新增四维输出，保留同一权重、ratio、选中集合和几何补偿；其 PLY 几何和高阶 SH exact 一致。这不是重新训练一个没有外观补偿的模型。',
      '## Full 和生产 Cached Bernoulli',
      '| 档位 | 方法 | 实际 K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |','|---|---|---:|---:|---:|---:|---:|']
for level in ['Full']+levels:
    for method,label in [('geometry_v1','几何10维'),('appearance_v2','几何＋外观14维')]:
        r=pick(method,level);text.append(f"| {level if level=='Full' else level+'%'} | {label} | {r['selected']} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
text+=['Full 四指标均值改善，但 PSNR 只提高 0.0672 dB，中位数下降 0.0094 dB，仅 31/64 视图改善，不能解释为稳定的 Full 质量提升。原 ratio 的预算略有不同，因此同时提供 exact same-K 对照。']
text+=['## 相同 K：14 维减 10 维',
       '| K | 旧 PSNR | 新 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for k in plan['common_k']:
    l='K'+str(k);a=pick('geometry_v1',l,'same_k');b=pick('appearance_v2',l,'same_k')
    d=[delta(a,b,m) for m in metrics];p=pair('appearance_v2-geometry_v1',l,'psnr','same_k')
    text.append(f"| {k} | {float(a['psnr']):.4f} | {float(b['psnr']):.4f} | {d[0]:+.4f} | {d[1]:+.5f} | {d[2]:+.5f} | {d[3]:+.5f} | {round(float(p['improved_fraction'])*64)}/64 |")
text+=['## 同模型外观补偿 ON 减 OFF，几何补偿保留',
       '| ratio | K | OFF PSNR | ON PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |','|---|---:|---:|---:|---:|---:|---:|---:|']
for l in levels:
    a=pick('appearance_off',l);b=pick('appearance_v2',l);d=[delta(a,b,m) for m in metrics]
    text.append(f"| {l}% | {b['selected']} | {float(a['psnr']):.4f} | {float(b['psnr']):.4f} | {d[0]:+.4f} | {d[1]:+.5f} | {d[2]:+.5f} | {d[3]:+.5f} |")
text+=['## 外观与几何补偿幅度',
       '| ratio | 选中点原 Opacity 均值 | 补偿后 Opacity 均值 | ΔRGB RMS | δXYZ RMS：10维→14维 |','|---|---:|---:|---:|---:|']
for r in stats:
    if r['method']!='appearance_v2':continue
    old=next(v for v in stats if v['method']=='geometry_v1' and v['ratio']==r['ratio'])
    text.append(f"| {100*float(r['ratio']):g}% | {float(r['opacity_before_mean']):.4f} | {float(r['opacity_after_mean']):.4f} | {float(r['delta_rgb_rms']):.4f} | {float(old['xyz_rms']):.4f}→{float(r['xyz_rms']):.4f} |")
text+=[f'![质量曲线]({(root/"quality_curves.png").as_posix()})',
       '## 训练轨迹与验证边界',
       '| 项目 | 几何10维 | 几何＋外观14维 |','|---|---:|---:|']
for key,label in [('gaussians','Gaussian'),('anchors','Anchor'),('full_renders','Full Render'),('lod_renders','LoD Render'),('gaussian_updates','Master Adam 更新'),('selector_updates','Selector 更新'),('online_end_mapping','online 结束 Mapping'),('compensation_forwards','补偿执行次数'),('cd_compensation_steps','实际补偿 CD 步数'),('selector_parameters','网络参数，不含 Anchor Feature')]:
    text.append(f"| {label} | {models[0][key]} | {models[1][key]} |")
text+=['- **IMPLEMENTED / TESTED**：14 维选后补偿，真实 renderer 的新增通道梯度，Full 参数及 Adam 隔离，零初始化和饱和 logit 恒等，未选中属性及高阶 SH 保留，CPU/CUDA 保存加载，原生／Python／PLY／prepared renderer 一致。',
       '- **TESTED**：旧 10 维真实模型输出 exact 兼容；旧可变成员 Mapping 和 CD 回归；14 项功能门控均通过。',
       '- **PARTIALLY VERIFIED**：新增外观 CD 通过合成局部几何修正测试。本次实际补偿 CD 为 0 步，不能宣称已验证真实 Loop Closure 后的质量。',
       '- 原 joint 为异步在线训练，实际人口、位姿、online 结束时刻和 Full／LoD 更新数不同。此轮只支持一个 scene/seed 的实测比较，不是固定地图的纯网络归因实验。',
       '- 不单独归因于 Opacity 或颜色：两项本轮一起加入；外观 OFF 诊断也一起关闭。没有依据评价结果自动调 LR、增宽网络或加正则。',
       f"- 本地重算 {checks['arms']} 条质量评价、{checks['per_view_rows']} 条逐视图记录及 {checks['paired_rows']} 条配对统计；{checks['files_sha_verified']} 个证据文件、{checks['source_sha_verified']} 个源文件 SHA256 核验通过。",
       '## 交付',
       '- '+link('启用配置',repo/'cfg/selector/anchor_compensation.yaml')+'：新训练默认输出14维，TF关闭。',
       '- '+link('质量汇总',out/'quality_summary.csv')+' / '+link('逐视图',out/'quality_per_view.csv')+' / '+link('配对统计',out/'paired_statistics.csv'),
       '- '+link('补偿统计',out/'compensation_statistics.csv')+' / '+link('验证记录',out/'correctness.json')+' / '+link('交付核验',root/'local_delivery_checks.json'),
       '- '+link('最终14维模型',root/'evidence/candidate_seed2026/10000_shutdown/ply/selector.pt'),
       '没有 stage、commit、push；既有10维模型、源码快照和结果保留。完成后停止。']
document=text[0]
for before,after in zip(text,text[1:]):document+=('\n' if before.startswith('|') and after.startswith('|') else '\n\n')+after
(repo/'SELECTOR_APPEARANCE_COMPENSATION_REPORT.md').write_text(document+'\n',encoding='utf-8')
print('REPORT_WRITTEN')
