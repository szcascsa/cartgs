"""从原始 online summary 生成中文统计表；不把样本均值在范围内等同验收通过。"""
import csv
import json
from pathlib import Path
import statistics
import sys

dest=Path(__file__).resolve().parent
summary=json.loads(Path(sys.argv[1]).read_text())
assert summary['complete'] and len(summary['runs']) == 24
assert len(summary['comparisons']) == 4 and all(c['repeated_runs_complete'] for c in summary['comparisons'])
report=dest.parents[2]/'CUDA_SELECTOR_IMPLEMENTATION_REPORT.md'


def score(values, digits=3):
    return f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}' if len(values)>1 else f'{values[0]:.{digits}f}'


def interval(item, digits=4):
    return f"{item['mean']:.{digits}f} ± {item['std']:.{digits}f}" if item['std'] is not None else f"{item['mean']:.{digits}f}"


text='''## 完整在线 A/B 与 legacy 自身波动

最终网络+Adam 队列：`/home/prmi/zwj/selector-cuda-20261015/results/selector_online_ab_final_20261015`。room/desk/xyz/long，每场景先三次 legacy，再三次 CUDA，均为 seed0 的独立进程重复，不是三个不同 seed。各组使用同一 binary/library/dataset/SLAM/Selector/Protection/TF/ratio/初始化策略/stop，只改 `Selector.backend`。

使用既有 online-only driver，在 SLAM shutdown 时停止，两组都没有 post-SLAM tail。冻结 fast.yaml 的完整副本和 SHA 在在线目录；不能外推所有 Guard/CD/DSD 配置组合。评估 Full 和四个 ratio，保留当前 checkpoint cached-uniform Bernoulli；原始 seed0 draw 另作 sampler parity。图像落盘关闭，渲染/指标计算和模型保存保留。

冻结配置为 Protection=1、Replay=1、ChangeDistill=1、TF backend=cuda、GradientConstraint=0、DSD=0。Guard/DSD 的数学、状态和集成测试另有记录，不能称本轮完整线上启用了它们。

下列 ± 为三个独立运行的 sample std，原始每 run 数值、min/max、CUDA-Legacy delta/std 和首次日志输入分歧在 D/online_summary.json、online_quality_comparison.csv。

### Mapping 周期与累计工作时间

| 场景 | legacy 周期 mean ms | CUDA 周期 mean ms | mean 倍率 | legacy p95 ms | CUDA p95 ms | legacy 累计周期 wall s | CUDA 累计周期 wall s |
|---|---|---|---:|---|---|---|---|
'''
comparisons=summary['comparisons']
for comp in comparisons:
    m=comp['metrics'];a,b=m['complete_mapping_mean_ms']['legacy'],m['complete_mapping_mean_ms']['cuda']
    p=m['complete_mapping_p95_ms'];t=m['accumulated_mapping_wall_ms']
    text+=f"| {comp['scene']} | {interval(a)} | {interval(b)} | {a['mean']/b['mean']:.3f}× | {interval(p['legacy'])} | {interval(p['cuda'])} | {interval({**t['legacy'],'mean':t['legacy']['mean']/1000,'std':t['legacy']['std']/1000})} | {interval({**t['cuda'],'mean':t['cuda']['mean']/1000,'std':t['cuda']['std']/1000})} |\n"
text+='''
这里的倍率是完整 Mapping **平均优化周期**，不是整个 SLAM 进程 wall 的倍率。MappingProfiler 使用原有 begin/end 范围的 host wall，不额外同步 GPU；它不是 Nsight GPU active time 或严格 device span。两组人口和调度也不同，周期倍率不能单独归因于 Selector；固定输入下的归因由网络/caller benchmark 提供。固定 SLAM 停止窗口内，更快的周期允许更多更新；累计 Mapping 工作时间不一定下降。累计值由 MappingProfiler 的 mean×cycles 得到。外层队列为独立诊断暂停过协调器，但不中断正在运行的训练子进程；外层 subprocess wall 可能含延迟回收时间，故不用于性能结论。

### 实际更新和人口

| 场景/后端 | cycles | committed GS updates | Selector updates | TF normal updates | 最终 N=eligible | compatibility Mature |
|---|---|---|---|---|---|---|
'''
for scene in ('room','desk','xyz','long'):
    for backend in ('legacy','cuda'):
        r=[r for r in summary['runs'] if r['scene']==scene and r['backend']==backend]
        if not r:continue
        vals=[score([v['shutdown'][k] for v in r],1) for k in ('mapping_cycles','selector_updates','tf_updates','gaussian_total')]
        gs=score([v['committed_updates_from_step_log']['gaussian_optimizer_step'] for v in r],1)
        mature=score([v['shutdown_population']['compatibility_mature_count'] for v in r],1)
        text+=f'| {scene}/{backend} | {vals[0]} | {gs} | {vals[1]} | {vals[2]} | {vals[3]} | {mature} |\n'
text+='''
本配置 entry=0，eligible 等于总 N。compatibility Mature 是连续 maturity 的兼容统计，不是恢复旧 Mature 分池规则。每 run 的 maturity mean/p10/p50/p90、四 ratio 的 selected count/实际 ratio、TF transformed count 和 CD/Replay counters 均保存在 JSON。

| 场景/后端 | LoD selected mean | LoD 实际 selected ratio | LoD TF transformed mean | TF 累计处理行数 |
|---|---|---|---|---|
'''
for scene in ('room','desk','xyz','long'):
    for backend in ('legacy','cuda'):
        r=[v['lod_statistics'] for v in summary['runs'] if v['scene']==scene and v['backend']==backend]
        if not r:continue
        text+=f"| {scene}/{backend} | {score([v['selected']['mean'] for v in r],1)} | {score([v['actual_ratio']['mean'] for v in r],5)} | {score([v['tf_transformed']['mean'] for v in r],1)} | {score([v['tf_transformed_total'] for v in r],0)} |\n"
text+='''
TF 累计处理行数是 normal LoD 调用的工作量，包含同一 Gaussian 多次处理，不是 unique Gaussian count。

本轮 24 次日志的 `replay_tf_updates` 均为 0，不能宣称线上覆盖了 Replay accepted-update；该路径由独立集成夹具覆盖。`cd_tf_steps` 观测范围为 0–60，各 run 原始计数保存在 JSON。

### Trajectory 与图像质量

trajectory 沿用原 evaluator 的 translation-part APE 与 rotation-part APE（后者不是角度单位）。

| 场景 | translation RMSE legacy / CUDA | rotation-part RMSE legacy / CUDA | CUDA translation 均值在 legacy min/max 内 |
|---|---|---|---|
'''
for comp in comparisons:
    m=comp['metrics'];a,b=m['translation_rmse'],m['rotation_rmse']
    text+=f"| {comp['scene']} | {interval(a['legacy'],6)} / {interval(a['cuda'],6)} | {interval(b['legacy'],6)} / {interval(b['cuda'],6)} | {'是' if a['cuda_mean_inside_observed_legacy_range'] else '否'} |\n"
text+='''
| 场景/ratio | PSNR legacy / CUDA | SSIM legacy / CUDA | LPIPS legacy / CUDA | 三项 CUDA 均值均在 legacy min/max 内 |
|---|---|---|---|---|
'''
csv_rows=[]
for comp in comparisons:
    for ratio in ('full','ratio_0p01','ratio_0p05','ratio_0p1','ratio_0p15'):
        entries=[comp['metrics'][ratio+'/'+metric] for metric in ('psnr','ssim','lpips')]
        values=[interval(v['legacy'])+' / '+interval(v['cuda']) for v in entries]
        inside=all(v['cuda_mean_inside_observed_legacy_range'] for v in entries)
        text+=f"| {comp['scene']}/{ratio} | {' | '.join(values)} | {'是' if inside else '否'} |\n"
        for metric,v in zip(('psnr','ssim','lpips'),entries):
            csv_rows.append(dict(scene=comp['scene'],ratio=ratio,metric=metric,
                **{'legacy_'+k:v['legacy'][k] for k in ('n','mean','std','min','max')},
                **{'cuda_'+k:v['cuda'][k] for k in ('n','mean','std','min','max')},
                delta=v['cuda_minus_legacy_mean'],delta_div_legacy_std=v['delta_div_legacy_std'],
                cuda_mean_inside_observed_legacy_range=v['cuda_mean_inside_observed_legacy_range'],
                all_cuda_runs_inside_observed_legacy_range=v['all_cuda_runs_inside_observed_legacy_range']))
with (dest/'online_quality_comparison.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
outside=[r for r in csv_rows if not r['cuda_mean_inside_observed_legacy_range']]
all_inside=sum(r['all_cuda_runs_inside_observed_legacy_range'] for r in csv_rows)
text+=f'''\n共 {len(csv_rows)} 项图像质量比较，其中 {len(outside)} 项 CUDA 均值超出三次 legacy 的观测 min/max；仅 {all_inside} 项满足所有 CUDA 单次结果均在 legacy min/max 内。这是原始观察范围的报告，不是人为质量容差或显著性检验。三个样本不足以证明统计等价，均值落入范围也不自动放行。\n\n'''
text+='''日志可见在线输入在 Selector 更新前就不同：room 的 legacy 自重复在首个 training step 的 N 已分别为 1989 和 1940；CUDA 对照首 step 的 keyframe/N 也不同。因此不能把在线权重/ID 轨迹差异直接当成同输入的 CUDA 数值漂移，也不能据此声称 CUDA 质量提升。固定 checkpoint、固定输入/随机数的数学和 replay 对照用于单独隔离算法一致性。

尚未记录 live Mapping 内每一步 global selected-ID trace、独立 Selector forward/backward 的逐步 p50/p95；不能用 microbenchmark 冒充这些在线观测。最终 checkpoint 的同输入 inference/mask parity 另有独立记录。

'''
with (dest/'online_checkpoint_parity.csv').open(newline='') as stream:
    checkpoints=list(csv.DictReader(stream))
assert len(checkpoints)==96
check_fields=('probability_within_tolerance','current_mask_exact','compact_ids_exact','seed0_bernoulli_exact',
              'seed0_next_draw_exact','seed0_state_exact','two_sided_near_tie_exact','global_rng_unchanged')
checkpoint_summary=dict(cases=96,passed=sum(r['passed']=='True' for r in checkpoints),
    max_abs=max(float(r['max_abs']) for r in checkpoints),
    failures={k:sum(r[k]!='True' for r in checkpoints) for k in check_fields},
    failed_cases=[r for r in checkpoints if r['passed']!='True'],production_accepted=False)
(dest/'online_checkpoint_parity_summary.json').write_text(json.dumps(checkpoint_summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text+='''### 实际在线 checkpoint 的同输入验收

24 个实际 checkpoint × 四 ratio，共 96 组，分别使用相同已训练参数、Gaussian 属性和 metadata。比较的是同一 checkpoint 的 legacy/CUDA，不是把不同在线运行的参数相互比较。

'''
text+=f"当前 cached-uniform mask、compact IDs、独立 seed0 Bernoulli、generator state/next draw、全局 CUDA RNG 不变检查均为 **96/96 exact**。概率均满足初始 float 容差，最大 abs={checkpoint_summary['max_abs']:.8e}。\n\n"
text+=f"但是包含双侧 near-tie 后仅 **{checkpoint_summary['passed']}/96 通过**。失败来自 `desk/legacy/repeat2`、N=71165 的 ratio=0.10/0.15；两个 case 存在 FP32 概率差异，不能用实际 cached uniform 未碰到阈值来放行。原始 CSV、逐层定位和摘要均保留。**strict 是候选名，不代表其已全面满足 hard-mask 门槛。**\n\n"
text+='''首次排队的 checkpoint 测试未设置 Ninja 所在 PATH，在 extension loader 阶段失败；补齐 PATH 后完成上述 96 组。失败环境日志单独保留，没有把环境错误当成数学通过。

'''
if (dest/'online_stage_probe.csv').exists():
    with (dest/'online_stage_probe.csv').open(newline='') as stream:
        probe=list(csv.DictReader(stream))
    lookup={(r['mode'],r['ratio'],r['stage']):r for r in probe}
    assert all(int(r['unequal'])==0 for r in probe if r['stage'] in ('ratio_first_op_n','same_input_final'))
    text+='''### 最终 fixed-noise 失败的逐层定位

`desk/legacy/repeat2/3465_shutdown/ply`，N=71165，FP32/TF32=false：三个属性分支无差异；最早差异出现在 `[M,1]` ratio 第一层（stage_6），而不是 final GEMM 或 softmax。

| Ratio / 边界 | max abs | RMS | p99 | 不同元素或 hard class 数 | worst global ID |
|---|---:|---:|---:|---:|---:|
'''
    for ratio in ('0.1','0.15'):
        for stage,label in (('stage_6','ratio 第一层'),('probability','probability'),('fixed_tie_hard','固定 Gumbel noise hard class')):
            r=lookup[('row_ratio',ratio,stage)]
            text+=f"| {ratio} {label} | {float(r['max_abs']):.8e} | {float(r['rms']):.8e} | {float(r['p99']):.8e} | {r['unequal']} | {r['worst_global_gaussian_id']} |\n"
    text+='''
固定 `noise=-legacy_logits` 令 legacy 两类恰好打平，按首类优先。ratio=0.15 的冻结候选实际改变 **58663 个 hard class**；首个不同 global Gaussian ID 为 1。scalar ratio 的四个 ratio 在同一 checkpoint 全部 exact，不能外推 row ratio。

诊断直接复现当前 `CUBLAS_OP_T, lda=1` 的 ratio 第一层差异；对同一组参数/输入，改用等价 `CUBLAS_OP_N, lda=32` 的单层参考调用后误差为 0。用完全相同 packed feature 调用 legacy/custom final 时误差也为 0。这把该 case 的原因定位到 one-column ratio GEMM 的转置/布局及对应 FP32 累加路径；没有 TF32 或 RNG 改动。

非转置参考仅用于单层定位，未替换冻结的实验实现，不能称已修复或继承本报告全部验收/性能结果。该候选按用户 hard-ID 硬门槛判定失败，保留为明确标注的实验，不进入 production。原始逐层证据：D/online_stage_probe.cpp、online_stage_probe.py、online_stage_probe.csv、online_stage_probe.json。

'''
content=report.read_text(encoding='utf-8');start=content.index('## 完整在线 A/B 与 legacy 自身波动');end=content.index('## 生产切换与尚未通过项')
report.write_text(content[:start]+text+content[end:],encoding='utf-8')
print(json.dumps(dict(completed=len(summary['runs']),quality_comparisons=len(csv_rows),cuda_means_outside_legacy_range=len(outside))))
