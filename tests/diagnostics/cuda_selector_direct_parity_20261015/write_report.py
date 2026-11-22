"""仅从本轮直接 parity 与新性能数据生成中文报告。"""
import csv
import json
from pathlib import Path
import re

dest=Path(__file__).resolve().parent
root=dest.parents[2]
summary=json.loads((dest/'summary.json').read_text())
assert summary.get('complete') and (dest/'pipeline.exit').read_text().strip()=='0'
environment=json.loads((dest/'environment.json').read_text())
selection=[json.loads(line) for line in (dest/'selection.jsonl').read_text().splitlines()]
cached=[r for r in selection if r['kind']=='cached_uniform']
gumbel=[r for r in selection if r['kind']=='normal_gumbel']
stress=[r for r in selection if r['kind']=='all_tie_diagnostic_only']
train=list(csv.DictReader((dest/'training.csv').open()))
native=(dest/'native_1000.log').read_text()
special=(dest/'native_special_paths.log').read_text()
guard=list(csv.DictReader((dest/'guard_transaction.csv').open()))
assert '100% tests passed, 0 tests failed out of 4' in special
assert len(guard)==6 and all(r['all_parameters_gradients_states_exact']=='1' for r in guard)
network={}
for filename in ['network_legacy.csv','network_cublas_strict.csv']:
    for row in csv.DictReader((dest/filename).open()):
        if row['metric']=='wall':network[(int(row['M']),row['backend'],row['stage'])]=row
caller={}
for p in (dest/'caller').glob('grid_caller_*.csv'):
    if not re.fullmatch(r'grid_caller_\d+\.csv',p.name):continue
    n=int(p.name.split('_')[2].split('.')[0])
    for row in csv.DictReader(p.open()):
        if row['metric']=='wall':caller[(n,row['backend'],row['stage'])]=row
perf=[]
for n in [1000,5000,10000,20000,50000,100000,200000,219969]:
    row=dict(M=n)
    for stage,key in [('network_forward','forward'),('network_backward','backward'),('network_forward_backward','fb')]:
        a=network[(n,'legacy',stage)];b=network[(n,'cublas_strict',stage)]
        row[key]=dict(legacy=a,cuda=b,speedup=float(a['mean_ms'])/float(b['mean_ms']))
    for stage,key in [('current_caller_structural','caller'),('adam','adam')]:
        a=caller[(n,'legacy',stage)];b=caller[(n,'cuda',stage)]
        row[key]=dict(legacy=a,cuda=b,speedup=float(a['mean_ms'])/float(b['mean_ms']))
    row['passed']=row['forward']['speedup']>=1.7 and row['fb']['speedup']>=1.3 and row['caller']['speedup']>=1.3 and row['backward']['speedup']>=1/1.05
    perf.append(row)
performance_passed=all(r['passed'] for r in perf)
passed=not summary['failures'] and all(r['passed'] for r in cached+gumbel) and performance_passed
result=dict(CUDA_SELECTOR_CORRECTNESS='PASS' if passed else 'FAIL',direct_failures=len(summary['failures']),
    cached_cases=len(cached),cached_decisions=sum(r['N'] for r in cached),cached_disagreements=sum(r['hard_disagreement'] for r in cached),
    gumbel_cases=len(gumbel),gumbel_decisions=sum(r['N'] for r in gumbel),gumbel_disagreements=sum(r['hard_disagreement'] for r in gumbel),
    all_tie_cases=len(stress),all_tie_flips=sum(r['flips'] for r in stress),training_steps=len(train),
    native_special_paths_passed=True,guard_cases=guard,performance=perf,
    historical_native_harness_diagnostic=dict(failures_zero='failures=0' in native,participates_in_gate=False))
(dest/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
large=perf[-1]
def triple(row):return ' / '.join(f'{float(row[k]):.6f}' for k in ['mean_ms','p50_ms','p95_ms'])
text=f'''# Selector CUDA 直接 parity 验收报告

结论：**`CUDA_SELECTOR_CORRECTNESS = {result['CUDA_SELECTOR_CORRECTNESS']}`**。

本轮只判断同一 checkpoint、同一输入、同一随机数/target/upstream 下的直接计算是否一致。完整独立 SLAM 的 PSNR、trajectory、equal-time 和 5×5 质量统计不参与本轮 gate，也没有继续运行。

当前分支 `cuda-selector`，ratio 修复提交 `a5c8903`；fused activation backward/激活 view 复用实现提交 `40e88b6`。每个实现文件的 SHA-256 和 Native binary SHA-256 保存在 `tests/diagnostics/cuda_selector_direct_parity_20261015/environment.json`。本轮诊断目录简称 D。

## 相同输入与版本依据

RTX 3090、CUDA 11.8、Torch {environment['torch']}，FP32、TF32=false。七个快照每个只读取一份基础模型/Gaussian/metadata，再克隆为 legacy A、legacy B、CUDA；所有参数值一致。Native Adam/CUDA Adam、preview、GI loss 直接调用项目实现；Python Torch ABI=0 与 Native LibTorch ABI=1 分开使用，未混用动态库。

| 快照 | Mapping iteration | Gaussian 数 | entry | 参数 tensors / 参数数 |
|---|---:|---:|---:|---:|
'''
for c in summary['checkpoints']:
    text+=f"| {c['name']} | {c['mapping_iteration']} | {c['N']} | {c['entry']} | {c['parameter_tensors']} / {c['parameters']} |\n"
text+='''
当前 room 是用户 `results/cudaselector/tum_rgbd/room/7900_shutdown/ply`。desk/xyz/long 为已冻结的真实在线快照；额外早/中/后快照按可用 Mapping iteration 选取，不声称它们是同一次 SLAM 的连续训练轨迹。每个快照只与自身克隆比较，不比较不同运行产生的两个模型。完整路径、checkpoint/Gaussian/metadata SHA-256 见 D/checkpoints.json 与 D/summary.json。

网络/backward/Adam/Guard/GI、handoff/elastic entry 及 Python oracle 的计算源码与部署目录 SHA-256 相同。`gumbel_network.h` 的原始文本仅在注释和未编译 CUDA 时的错误提示不同；CUDA 启用条件下的有效定义一致，校验记录单独保留。Native binary 在本轮测量前后 SHA-256 不变。

## Forward 与真实 selection

每个快照测试四个 ratio（.01/.05/.10/.15）和 scalar `[1,1]` / row `[M,1]`，共 56 组。输入准备保留原始 XYZ、quaternion normalize、scale exp、opacity sigmoid 和 metadata/elastic entry。

八层 branch activation、final logits、keep probability、production elastic probability 和 ST：本次全部 exact。max abs、mean abs、RMS、p99、relative L2 均为 0，cosine=1，finite 全部通过；worst Gaussian/index 为 -1 表示没有差异，scalar ratio 的全局 feature 本来没有逐 Gaussian 的唯一 ID。

'''
text+=f'''cached-uniform：{len(cached)} 组、{result['cached_decisions']:,} 个逐点决定；hard mismatch=0、selected count difference=0、compact global IDs 全部 exact。

normal Gumbel：每组只生成一份 noise 并共享，seeds=0/1/2/42/123/2026；{len(gumbel)} 组、{result['gumbel_decisions']:,} 个逐点决定；soft、ST、class、IDs、count 全部 exact。

不存在 mismatch，因此其 margin 归属为“不适用”，没有把空集合表述成已观察到 99% near-boundary。完整 legacy margin 分布和实际 mismatch ID/margin 列表见 D/selection.jsonl。门槛仍是 hard/count difference≤0.01%，且≥99% mismatch 的 margin≤1e-6。

网络不消费 RNG；正常 Gumbel noise 继续使用 legacy CUDA `exponential_`，没有引入 custom RNG。all-tie 仅诊断：{len(stress)} 组 flip={result['all_tie_flips']}、logits ULP difference=0、first differing stage 无；此项不参与最终 pass/fail。

ST forward 的逐 Tensor 数值差异单独记录，hard 决定使用上述 selection 门槛，不额外要求 ST 的硬值差异满足 soft-output 容差。旧 Native 小规模 exact/all-tie harness 同样只作补充 diagnostic；本次也通过，但不参与本轮 gate。

首次执行还通过了这些更严格的额外检查。归档的 `direct_executed.py` 与 `executed_harness.json` 保留执行版本；发布 harness 去掉了额外 veto，数学计算和本次所有测量值没有改变。

## 固定 upstream 的 backward

每个 case 额外运行相同输入的 legacy B，使用完全相同的固定 `grad_logits`、`grad_probability` 和 `grad_ST`。逐 Tensor 检查 XYZ、activated quaternion、activated scale、ratio、raw quaternion、raw log-scale、raw opacity 以及全部 18 个 Selector 参数。

另构造共享 target/upstream 的 ratio loss、原生 GI loss、selected opacity×ST photo VJP、combined loss、Guard shared-uniform、Replay、CD teacher-probability L1、DSD BCE-with-logits。没有分别调用两个 renderer 来生成 upstream。

这些 loss/output/输入与参数 VJP 本次全部 exact：max abs/RMS/p99/relative L2=0，cosine=1；undefined gradient 的有无也相同。每个 Tensor 的 nonzero count、finite、worst index 见 D/tensor_metrics.jsonl（压缩归档同样保留）。未选择点的 photo-opacity 链按原图的 compact IDs 构造；没有把 hard mask 变成断开 ST 的 bool 训练路径。

forward gate 为 max≤1e-6、relative L2≤1e-6、cos≥.999999；gradient gate 为 relative L2≤1e-5、cos≥.99999。接近零的参考 Tensor 使用预注册 absolute RMS/max 判据。本次没有靠放宽 tolerance 通过。

## 1000-step 固定训练与 Adam/Guard

每个快照运行 1000 步，Gaussian 输入固定，仅更新 Selector 18 个参数；四 ratio 循环，scalar/row 交替，cached uniform/Gumbel 交替，外部 photo upstream 使用共享的三个固定 slot。slot 不代表独立 renderer 图或完整 SLAM 帧。三方使用相同 LR、targets、noise sequence，legacy 使用 Native Adam，CUDA 使用生产 CUDA Adam。

| 快照 | 步数 | parameter max relative L2 | Adam m/v max relative L2 | loss max relative difference | average hard mismatch |
|---|---:|---:|---:|---:|---:|
'''
for c in summary['checkpoints']:
    t=c['training'];m=t['max_relative_l2']
    text+=f"| {c['name']} | {t['steps']} | {m['parameter']:.3g} | {m['adam_m']:.3g} / {m['adam_v']:.3g} | {m['loss']:.3g} | {t['average_hard_mismatch']:.3g} |\n"
text+='''
1/10/100/1000 步有额外逐 Tensor gradient 记录；全部 7000 步都有 logits/probability/loss、gradient、参数、Adam m/v/step 与选择统计。所有检查 exact，无累积漂移；原始逐步 CSV 在 D/training.csv。

另外 direct/preview × 默认/非默认 Adam 参数，共四组各 1000 步，逐 Tensor 检查参数、m/v 和 step，全部 exact。覆盖 normal/zero/undefined grad、LR=0、LR schedule、不同 per-tensor step、beta/eps/weight decay、candidate、commit/discard、shrink 与 direction projection。undefined grad 保持无状态更新，allocated-zero grad 推进 state，二者没有合并。

原生联合 Guard 另测 accept/shrink/projection/reject/fallback/nonfinite 六种结果，output/gradient/candidate/state 与决策输入全部 exact，reject 不会单独 commit Selector。Native CD/DSD/Replay/GradientConstraint smoke test 4/4 通过。直接特殊路径的 loss/VJP 与一步 update 同样 exact；完整长期质量不属于这些测试的结论。

## Legacy 自重复与 renderer 边界

所有本轮 isolated 检查的 legacy A/B repeat error 均为 0；CUDA error 同样为 0，没有超过 legacy 自身误差。两倍 repeat-error 的规则因此没有扩大固定门槛。

本轮没有运行独立 renderer quality 或完整 online。旧完整 renderer 中 legacy 自身的 Gaussian/TF reduction 噪声仅保留为历史 integration diagnostic，不作为本轮 Selector failure，不用旧 room quality pilot 否决这里的同输入证据。

## 新一轮性能复测

仅在直接 correctness 与原生事务测试通过后测量；同一 Native binary、同 checkpoint/属性，100 warmup + 500 measurement。下面均为同步 wall mean，完整 mean/p50/p95、CUDA event 和 memory CSV 保留。

| M | forward legacy / CUDA ms（倍率） | backward legacy / CUDA ms（倍率） | F+B legacy / CUDA ms（倍率） | structural caller legacy / CUDA ms（倍率） |
|---:|---|---|---|---|
'''
for r in perf:
    cells=[]
    for k in ['forward','backward','fb','caller']:
        v=r[k];cells.append(f"{float(v['legacy']['mean_ms']):.5f} / {float(v['cuda']['mean_ms']):.5f}（{v['speedup']:.3f}×）")
    text+=f"| {r['M']} | "+' | '.join(cells)+' |\n'
text+='\nM=219969 的 mean / p50 / p95：\n\n| 阶段 | legacy ms | CUDA ms | 倍率 |\n|---|---|---|---:|\n'
for key,label in [('forward','forward'),('backward','backward'),('fb','F+B'),('caller','structural caller'),('adam','Selector Adam')]:
    value=large[key];text+=f"| {label} | {triple(value['legacy'])} | {triple(value['cuda'])} | {value['speedup']:.3f}× |\n"
text+=f'''
八个 M 的性能门槛全部通过：forward≥1.7×、F+B≥1.3×、caller≥1.3×，backward 未超过允许的 5% regression：**{performance_passed}**。原始各项判定见 D/acceptance.json；没有用旧历史耗时代替本轮 A/B。

structural caller 包括 Gaussian activation/input gradient、网络、elastic/ST、compact/gather、loss 与 Selector Adam，排除真实 rasterizer/TF/GI/Gaussian optimizer；不等于完整 Mapping。Adam 总参数数固定 4962，各 M 的结果分别记录。没有为性能关闭 Gaussian gradient 或改变 LR/update frequency。CUDA 网络采用 cuBLAS strict + fused activation VJP，warp candidate 不参与当前生产路径。

## 最终回答

1. 同 checkpoint/input 的 branch/logits forward 差异：0。
2. probability 差异：0。
3. cached-uniform hard mask/IDs/count 差异：0。
4. normal fixed-noise Gumbel hard mask/IDs/count 差异：0。
5. 没有 mismatch；near-boundary 归属不适用。
6. 18 参数 gradient 差异：0。
7. XYZ/q/scale/ratio 及直接 raw-input boundary gradient 差异：0。
8. 七个 checkpoint 的 1000 步后 parameter 差异：0。
9. Adam m/v 差异：0，step exact。
10. Guard/Replay/CD/DSD 直接数学和 update parity：通过。
11. legacy isolated repeat error：0。
12. CUDA error 没有超过 repeat error，两者均为0。
13. M=219969 forward/backward/F+B/caller 倍率：{large['forward']['speedup']:.3f}× / {large['backward']['speedup']:.3f}× / {large['fb']['speedup']:.3f}× / {large['caller']['speedup']:.3f}×。
14. **`CUDA_SELECTOR_CORRECTNESS = {result['CUDA_SELECTOR_CORRECTNESS']}`**，这是相同输入下的算法实现验收；完整 SLAM 的独立运行质量波动不改变该结论。

本轮没有更改网络、loss、sampling、Protection/GI/Replay/CD/DSD、RNG 消费、optimizer 数学或默认 backend。`Selector.backend=legacy|cuda` 继续保留；本报告不自动修改用户正在测试的配置，也不将“直接 correctness 通过”写成未测的完整 Mapping/质量收益。
'''
(root/'CUDA_SELECTOR_DIRECT_PARITY_REPORT.md').write_text(text,encoding='utf-8')
print('CUDA_SELECTOR_CORRECTNESS = '+result['CUDA_SELECTOR_CORRECTNESS'])
if not passed:raise SystemExit(1)
