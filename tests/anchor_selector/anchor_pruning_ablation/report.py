"""生成两项拓扑策略消融的中文报告，完整保存预算、配对质量和拓扑证据。"""
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
opacity_rows=list(csv.DictReader((out/'selected_opacity.csv').open(newline='')))
plan=json.loads((out/'evaluation_plan.json').read_text());runtime=json.loads((out/'training_runtime.json').read_text())
labels=dict(baseline='旧：逐点＋Anchor pruning',anchor_only='A：仅 Anchor pruning',anchor_only_full10='B：仅 Anchor pruning＋满10才向下增长')
pick=lambda m,l,p='cached_bernoulli':next(r for r in rows if (r['method'],r['level'],r['policy'])==(m,l,p))
pair=lambda c,l,m,p='cached_bernoulli':next(r for r in pairs if (r['comparison'],r['level'],r['metric'],r['policy'])==(c,l,m,p))
link=lambda label,path:f'[{label}]({path.as_posix()})'
levels=['1','5','10','15'];metrics=['psnr','ssim','lpips','photo']
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for method,label in [('baseline','Point + Anchor prune'),('anchor_only','A: Anchor prune only'),('anchor_only_full10','B: A + full10 fine growth')]:
    axes[0].plot([1,5,10,15],[float(pick(method,l)['psnr']) for l in levels],'o-',label=label)
    data=[r for r in rows if r['method']==method and r['policy']=='same_k']
    for ax,m in [(axes[1],'psnr'),(axes[2],'lpips')]:ax.plot([int(r['selected']) for r in data],[float(r[m]) for r in data],'o-',label=label)
for ax,title,x,y in zip(axes,['Production Cached Bernoulli','Exact same-K diagnostic','Exact same-K diagnostic'],['Ratio (%)','Selected K','Selected K'],['PSNR (dB), higher better','PSNR (dB), higher better','LPIPS, lower better']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.25)
axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'quality_curves.png',dpi=170);plt.close(fig)
win=lambda right,left,policy:sum(float(pair(right+'-'+left,l,'psnr',policy)['mean'])>0 and
    float(pair(right+'-'+left,l,'photo',policy)['mean'])<0 for l in (levels if policy=='cached_bernoulli' else ['K'+str(k) for k in plan['common_k']]))
text=['# 仅 Anchor Pruning 与满10点下一层增长：质量消融',
      f"**两组实验已完成，逐 Gaussian 质量 pruning 删除数均为 0。** 相同 K 下，A 相对旧基线有 {win('anchor_only','baseline','same_k')}/4 档同时改善 PSNR 与 photo；B 相对 A 有 {win('anchor_only_full10','anchor_only','same_k')}/4 档同时改善。完整指标和人口差异列于下方，不能仅按原 ratio 均值归因。",
      '**当前判断：保留 A 作为本轮请求恢复的模式；不建议启用“满10才向下一层增长”的 B 门槛。** A 的四档原 ratio 四指标均值均改善，但 same-K 为 MIXED，不能声称选择质量在所有预算提高。B 相对 A 的四档原 ratio 和四档相同 K 四指标均值均退化，门槛本次没有带来 LoD 收益。这只是单场景、单 seed 证据。',
      '## 三组定义',
      '| 方法 | Gaussian 逐点 pruning | Anchor 窗口 pruning | 下一层 growing | 同层 growing |','|---|---|---|---|---|',
      '| 旧基线 | 原 opacity／尺寸规则 | 原规则 | 原梯度规则 | 原梯度规则 |',
      '| A | 关闭 | 原规则 | 原梯度规则 | 原梯度规则 |',
      '| B | 关闭 | 原规则 | 原梯度规则＋源 Anchor 恰好10成员 | 原梯度规则 |',
      '按用户确认，B 仅限制下一层 growing，同层 growing 不要求满10。新观测的多尺度 Anchor 初始化继续执行；因此部分细层 Anchor 可以由新观测直接创建，不能仅凭细层 Anchor 存在就判定门槛失败。',
      '保留 Gaussian clone/split 用于填充现有 Anchor，保留 Gaussian／Anchor 拓扑交替。split 仍以子点替换父点，这是密度更新事务，不是 opacity／尺寸质量 pruning；日志及计数将其分开。',
      'Anchor 删除继续使用原窗口规则：可见访问累计所有成员的正 opacity 总和 O、访问数 T；仅 `T>check_interval·0.8` 且 `O<0.005·T` 才整 Anchor 删除。窗口重置、逐 slot 统计和停用规则不变。',
      '## 实现与配置',
      '- `Selector.anchor_only_pruning=1`：关闭 Gaussian 回合按 opacity／尺寸判定的逐点 pruning；clone/split 和原 Anchor 回合保留。',
      '- `Selector.anchor_refine_requires_full=1`：只在生成下一层新 Anchor 候选时检查源 Anchor 成员数；同层候选、既有父点归属及新观测插入不变。',
      '- 本地 '+link('当前补偿配置',repo/'cfg/selector/anchor_compensation.yaml')+' 已设为 A。B 在实验配置中开启第二个开关；不根据质量结果自动切换默认方法。历史配置不写新开关时沿用旧行为，可继续回归。',
      '- 保存配置和 Anchor 状态中记录两项策略、逐点与整 Anchor 删除数；growing 日志记录 full／partial 来源及被门槛拦截数。',
      '## 锁定协议',
      '- `fr1_xyz / seed2026 / 10000 Mapping`，只新增训练 A、B 各一次；旧14维模型复用 `/home/prmi/zwj/shared-selector-appearance-20261021/candidate_seed2026`，不重训。',
      '- 27维输入、共享浅层＋私有选择分支＋14维几何／Opacity／SH DC补偿、零初始化、start3000、Adam／LR、重建和预算损失不变。',
      '- 原 joint，`serial_full_lod=0`、Full100=0，TF关闭、SI/GI disabled，maturity=1、Cached Bernoulli与停用功能保留。online五档轮转，tail为Full后一个轮转joint LoD。',
      '- 评价最终10000 checkpoint，无调参或选最好step；沿用64个共同RGB帧，各自在线估计位姿，不宣称严格non-Bank。',
      f"- 三组共同整数K={plan['common_k']}，由 `[ceil(.01·max N), floor(.15·min N)]` 均匀取四点。每组输入ratio=K/N，均在1%～15%范围；cached margin exact-K仅作诊断，不替换生产采样。",
      '## 人口、拓扑和成本',
      '| 项目 | 旧基线 | A | B |','|---|---:|---:|---:|']
for key,label in [('gaussians','Gaussian'),('anchors','Anchor'),('full_anchors','满10成员 Anchor'),('occupancy_mean','平均成员数'),('below_cutoff','Master opacity <1/255'),('below_002','Master opacity <0.02'),('inactive','窗口停用点'),('gaussian_pruned_members','逐点质量 pruning 删除成员'),('anchor_pruned_members','整 Anchor pruning 删除成员'),('clones','clone 数'),('split_parents','split 父点数'),('fine_candidates','下一层候选 Gaussian 数'),('fine_blocked_not_full','未满10拦截的下一层候选'),('fine_partial_candidates','实际放行的部分成员来源下一层候选'),('new_anchors_from_growing','梯度 growing 新 Anchor'),('native_forward_ms','原生 Selector forward ms')]:
    vals=[]
    for m in models:
        value=m[key];vals.append('旧模型未记录' if value is None else f'{value:.4f}' if isinstance(value,float) else str(value))
    text.append('| '+label+' | '+' | '.join(vals)+' |')
text+=['新旧基线的 fine 来源细分计数日志格式不同，旧模型此项记为0表示未记录，不作为门槛证据；B 的实际放行部分成员来源候选必须为0。原生forward在各自实际N上测量，不能视作相同N的算法性能比。',
       '## Full 与生产 Cached Bernoulli',
       '| 档位 | 方法 | 实际K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |','|---|---|---:|---:|---:|---:|---:|']
for l in ['Full']+levels:
    for name in labels:
        r=pick(name,l);text.append(f"| {l if l=='Full' else l+'%'} | {labels[name]} | {r['selected']} | {float(r['psnr']):.4f} | {float(r['ssim']):.5f} | {float(r['lpips']):.5f} | {float(r['photo']):.5f} |")
text+=['原ratio的K随地图人口变化，不能把更多点带来的收益直接归因于取消逐点pruning。',
       '## 相同K：三组对照',
       '| K | 比较（前减后） | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR改善视图 |','|---:|---|---:|---:|---:|---:|---:|']
for k in plan['common_k']:
    for a,b,label in [('anchor_only','baseline','A−旧'),('anchor_only_full10','baseline','B−旧'),('anchor_only_full10','anchor_only','B−A')]:
        p=[pair(a+'-'+b,'K'+str(k),metric,'same_k') for metric in metrics]
        text.append(f"| {k} | {label} | {float(p[0]['mean']):+.4f} | {float(p[1]['mean']):+.5f} | {float(p[2]['mean']):+.5f} | {float(p[3]['mean']):+.5f} | {round(float(p[0]['improved_fraction'])*64)}/64 |")
text+=['A−旧的四档 same-K ΔPSNR 为 +1.5078 / −0.2686 / −0.0226 / +0.1593 dB。B−A 为 −1.3337 / −0.5584 / −0.6848 / −0.6400 dB，PSNR 改善视图仅 1/9/10/10 个（每档64视图）；不能把 B 的 Full 小幅改善解释为 LoD 改善。',
       '## 低 Opacity 成员与预算使用',
       '关闭逐点 pruning 后，低 Opacity 存储点重新出现：A 有3934个、B有3551个 Master opacity低于1/255。停用分别覆盖1937／1860个点；其余点可能尚未满足完整可见窗口资格，不能将未观察到的访问算作零opacity。',
       '| ratio | 方法 | 实际K | 选中Master opacity<1/255 | 补偿后opacity<1/255 |','|---|---|---:|---:|---:|']
for r in opacity_rows:
    if r['method']=='baseline':continue
    text.append(f"| {float(r['ratio'])*100:g}% | {'A' if r['method']=='anchor_only' else 'B'} | {r['selected']} | {r['raw_below_cutoff']} | {r['compensated_below_cutoff']} |")
text+=['15%时补偿后仍低于截断的选中点为 A 112/1710（6.55%）、B 151/1633（9.25%）。这是该checkpoint的opacity诊断，未新增候选屏蔽或修改停用阈值，也不把它等同于逐点真实图像贡献测量。']
text+=[f'![质量曲线]({(root/"quality_curves.png").as_posix()})',
       '## 在线训练轨迹与验证边界',
       '| 项目 | 旧基线 | A | B |','|---|---:|---:|---:|']
for key,label in [('full_renders','Full Render'),('lod_renders','LoD Render'),('gaussian_updates','Master Adam更新'),('selector_updates','Selector更新'),('online_end_mapping','online结束Mapping'),('compensation_forwards','补偿前向'),('cd_compensation_steps','补偿CD步数'),('anchor_feature_values','Anchor Feature标量数'),('gaussian_parameter_bytes','显式Gaussian参数bytes')]:
    text.append('| '+label+' | '+' | '.join(str(m[key]) for m in models)+' |')
text+=[f"A/B训练wall time分别为 {runtime['anchor_only']['wall_seconds']:.2f} / {runtime['anchor_only_full10']['wall_seconds']:.2f} 秒；不是隔离性能基准。显式参数bytes不含Adam、梯度、SLAM或renderer临时显存，未测完整训练峰值显存。",
       '- **IMPLEMENTED / TESTED**：两项独立开关、保存配置、原Anchor prune规则、逐点prune关闭、clone/split保留、下一层门槛与同层不受限、原joint/Full梯度隔离。',
       '- **TESTED**：CPU/CUDA策略单测及Mapping接入，14维补偿与旧10维checkpoint兼容回归，共17项门控通过；最终真实模型C++/Python/Cached Bernoulli mask exact一致，补偿属性与PLY导出数值一致，source与模型hash冻结。',
       '- **PARTIALLY VERIFIED**：真实训练的门槛日志及无逐点prune计数通过；这里只验证一个scene/seed，未做跨场景和多seed稳定性实验。异步SLAM导致在线结束时刻、地图人口、位姿与实际优化次数不同，不能作单因素固定地图因果归因。',
       f"- 本地重算{checks['arms']}条评价、{checks['per_view_rows']}条逐视图和{checks['paired_rows']}条配对统计；{checks['files_sha_verified']}个证据文件、{checks['source_sha_verified']}个源文件及{checks['reused_baseline_files']}个旧评价文件hash核验通过。",
       '## 交付',
       '- '+link('质量汇总',out/'quality_summary.csv')+' / '+link('逐视图',out/'quality_per_view.csv')+' / '+link('配对统计',out/'paired_statistics.csv'),
       '- '+link('人口与拓扑',out/'model_diagnostics.json')+' / '+link('增长事件',out/'growth_events.csv')+' / '+link('选中点opacity',out/'selected_opacity.csv'),
       '- '+link('功能门控',out/'correctness.json')+' / '+link('交付核验',root/'local_delivery_checks.json'),
       '- '+link('A模型',root/'evidence/anchor_only_seed2026/10000_shutdown/ply/selector.pt')+' / '+link('B模型',root/'evidence/anchor_only_full10_seed2026/10000_shutdown/ply/selector.pt'),
       '完成后停止；没有额外调参、改loss或追加实验，没有stage、commit、push。']
document=text[0]
for before,after in zip(text,text[1:]):document+=('\n' if before.startswith('|') and after.startswith('|') else '\n\n')+after
(repo/'ANCHOR_ONLY_PRUNING_ABLATION_REPORT.md').write_text(document+'\n',encoding='utf-8')
print('REPORT_WRITTEN')
