"""从保留的原始统计生成中文交付报告。"""
from pathlib import Path
import json

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
paired = {(r['stage'], r['branch']): r for r in json.loads((D / 'paired_summary.json').read_text())}
timings = {(r['stage'], r['case']): r for r in json.loads((D / 'timings.json').read_text())}
components = {(r['stage'], r['component']): r for r in json.loads((D / 'components.json').read_text())}
nsys = json.loads((D / 'nsys_summary.json').read_text())
attributed = json.loads((D / 'nsys_components.json').read_text())

def ms(stage, branch): return paired[stage, branch]['cpu_wall_ms']['median']
def speed(a, b, branch): return ms(a, branch) / ms(b, branch)
def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' + ''.join(
        '| ' + ' | '.join(str(x) for x in row) + ' |\n' for row in rows)
def errors(path, key, prefix=None):
    rows = json.loads((D / path).read_text())[key]['tensors']
    if prefix: rows = [r for r in rows if r['tensor'].startswith(prefix)]
    return [len(rows), f"{max(r['max_abs'] for r in rows):.6g}",
            f"{max(r['mean_abs'] for r in rows):.6g}", '通过' if all(r['allclose'] for r in rows) else '失败']

text = '''# ChangeDistill 运行时优化报告

日期：2026-10-18（延续 2026-10-17 审计与同一优化批次）。

## 结论

保留 Stage A 的保守 group membership 方案、Stage B 的有覆盖证明的 cache-hit 路径、
Stage C 的 batch 内 Selector 属性 embedding 复用。拒绝逐点 BMM transport 实验；
不修改 Stage D 输入梯度路径。没有提交 commit，没有更改正式 YAML，没有重新运行 SLAM。

'''
text += f"在同一 LC 6207、batch=8192、四个 ratio、每分支 5 epochs / 20 optimizer steps 下，\n完整事件的正反顺序复测中位数从 **{ms('baseline','both'):.3f} ms** 降至 **{ms('stage_c','both'):.3f} ms**，\n提速 **{speed('baseline','stage_c','both'):.3f}×**，耗时减少 **{100*(1-ms('stage_c','both')/ms('baseline','both')):.2f}%**。\n"
text += '''Selector 独立分支的改善可重复；Stage C 单独对完整事件的边际收益较小，
不能把所有独立分支的节省直接相加。具体限制与负面结果见下文。

## 1. 输入、源码与测量契约

开始时逐项核对原报告、`raw/source_manifest.json` 和当前生产源码：32 个文件的 SHA-256 全部一致，
`drift=[]`。此基线包含用户已有未提交修改，不等于 Git HEAD。
初始完整 Git diff、staged diff、status、HEAD 保存在 `baseline/`。
独立 `selector_epochs` / `tf_epochs` 及未设置时回退 `epochs` 的行为完整保留。

本任务相对初始工作树只改变 6 个生产文件和 1 个 cache 测试；其他 manifest 文件保持相同哈希。
详见 `integrity.json`、`production_optimization.diff` 和完整 `final_working_tree.diff`。
原审计目录、冻结源码及原始 profiling 文件没有改写。

真实输入：

| 项目 | 保持的值 |
| --- | --- |
| event | Local LC 6207 / event_6207_1 |
| Gaussian 总量 | 85,653 |
| mature / corrected mature / unaffected mature | 44,889 / 29,687 / 15,202 |
| TF in-domain / outside skipped | 29,683 / 4 |
| correction groups | 144 |
| ratios（顺序不变） | 0.01, 0.05, 0.10, 0.15 |
| batch / epochs | 8192 / Selector=5, TF=5 |
| optimizer steps | Selector=20；TF=20；完整事件=40 |

Selector 保持每 epoch 4 个 stratified batches；四个 ratio 的 loss 按原顺序累加，
每 batch 只有一次 backward / Adam step。TF 保持每 epoch 4 batches，最后一批 5107 点。
Selector anchors、TF audit 集合及其随机序列、outside_identity、AABB、loss 权重/reduction 均未改变。
所有 pre/post 和逐 epoch evaluation、untouched audit 都在 native 总时间内。

服务器：`172.19.163.58`，RTX 3090，driver 535.230.02，CUDA 11.8，sm_86。
原生 runner 链接项目 LibTorch 2.3.1 / ABI1；Python torch 2.3.1+cu118 / ABI0 仅做离线分析。
编译保持 `-O3 -DNDEBUG -std=c++17`；FP32、TF32 设置及 Selector CublasStrict / DirectFinalRelu 均不变。
环境与输入哈希见 `environment.json`、`input_sha256.json`、各阶段 `raw/source_manifest.json`。

每次 run 都从完全相同的 teacher 参数恢复 Student，新建 manager、Adam 与 event-local cache。
Gaussian 不变、normal optimizer state 不变和实际 step 数由原有 harness 断言。
先 cold run，再 2 次独立恢复的一 epoch warm-up，再测 warm runs，绝不在已训练模型上接着计时。
初始各阶段 10 次 warm；另做两个正反顺序块、每块 5 次 warm，主表采用这 10 次复测。
`monitor.json` 记录 benchmark PID 和 GPU 进程；交错复测没有其他 GPU 进程污染。
CPU 调度、GPU 时钟和 allocator/plan 状态仍会产生波动，因此同时保留初始结果和两个块的结果。

native mode0 无细粒度 hooks。总 CPU wall 在 final CUDA Event 完成后读取；
CUDA Event 测相同 current stream 的跨度，包含 CPU 发射不足导致的 stream 空闲，**不是纯 kernel time**。
teacher clone/snapshot preparation 单列 `event_preparation_ms`；主表 event wall 对应完整 coordinator finalize。
mode1 组件是独立插桩 executable 的 inclusive CPU/CUDA spans，不能跨嵌套层级相加或直接当作 native wall 分解。
Nsight Systems 的 kernel duration 另列。未改驱动权限、未绕过原审计的 ERR_NVGPUCTRPERM；
本轮没有硬件 counter，因此不宣称整个 CD compute-bound 或 memory-bound。

## 2. 原瓶颈与 Stage A

原 `transportTarget` 每次 miss 对 144 个 group 执行 `ids==g → nonzero → gather → applyGaussianTransform → index_copy`。
20 次 miss 共 2880 次 group 检查；其中 4 次是 untouched audit，实际没有任何应 transport 的点，仍扫描全部 group。
真实 event 的 `correction_group_id[N]` 已由 `buildEvent` 构造并验证无重复/交叠；-1 表示未修正。
本轮不改变 coordinator、group 构造或 Gaussian-ID 对应关系。

从原实现确认的精确语义：

```text
XYZ flag:   matmul(xyz * float_scale, matrix[0:3,0:3]) + matrix[3,0:3]
SCALE flag: log_scaling + std::log(float_scale)
ROT flag:   normalize(multiply(correction_wxyz, target_wxyz))
```

矩阵是转置存储的 row-vector 约定；quaternion correction 左乘；无 flag 的属性保持原值；
-1 点不能被附带 normalize。

### 被拒绝：逐点 BMM

`stage_a/` 将组变换打包一次，按 group ID gather，再按点 BMM；没有 `[B,G,...]` 中间量。
XYZ 最大绝对误差 9.53674e-7，log-scale/quaternion 逐位一致，所有 transported targets 通过前向容差。
native TF 中位数 207.078 ms，但 20 步后参数 p22 最大误差 3.35574e-5，部分元素不满足原
`atol=2e-5, rtol=2e-4`。首步梯度/更新通过仍不足以验收完整训练。
因此撤销该实现，没有放宽容差。BMM 与按组 GEMM 的 FP32 累加路径不同；
L1 在接近 target 的位置对微小数值差异敏感，不能仅凭前向误差小就保留。
完整失败数据在 `stage_a/raw/comparison_baseline_tf.json`。

### 保留：TargetTransport 的 host membership 方案

`src/change_distill.cpp::TargetTransport` 在 TF event 开始只复制一次 group-ID 表到 CPU。
每个 Teacher miss 将当前 batch IDs 复制到 CPU，单次 O(B) 遍历按 group 收集局部位置，
将所有位置合并为一次 H2D，再按原 group 顺序及原组内点顺序调用原 `applyGaussianTransform`。
保留原 GEMM 的形状、原 log/rotation 算式以及三次 out-of-place index_copy。
不在 GPU 上重复动态发现 144 个 group，不会修改 frozen snapshot 的别名。

这是保守替代方案，并未声称实现了完整点级 transport fusion。
它仍有 EVENT-ONCE group ID D2H、每 miss 一次 batch-ID D2H、适用 group 的 gather/transform/index_copy。
audit 全为 -1 时直接返回；空 batch、空 group、无 group、混合 flags 和非单位 identity quaternion 均验证。
group-ID CPU 表额外 685,224 bytes；当前 batch host indices/positions 规模 O(B)，没有 O(BG) tensor。

'''
text += f"Stage A 的 TF 正反顺序复测：{ms('baseline','tf'):.3f} → {ms('stage_a_grouped','tf'):.3f} ms，**{speed('baseline','stage_a_grouped','tf'):.3f}×**。\n"
text += '''真实 event 的所有点、四个 ratio 的 Teacher 输出及常规 cache-miss 批次 packed targets：
transport 最大/平均绝对误差均为 0。最终训练误差和真实 fixtures 结果另见正确性表。

## 3. Stage B：有覆盖证明的 cache-hit 路径

`FrozenTeacherTargetCache::getCached` 只执行 NoGrad + cached index_select。
`get(..., fully_cached)` 仅在调用方有证明时分流；未知 coverage 保持原 lazy missing-index 流程。
没有每次 `.all().item()`，没有从这次事件推广“所有 anchors 永远已缓存”的假设。

| 调用集合 | 使用直接 gather 的证明 |
| --- | --- |
| Selector primary 训练/后评估 | baseline evaluate 已遍历整个 primary 集合、每个 ratio |
| Selector 固定 audit anchors 后评估 | baseline anchor_pre 已遍历同一个固定集合 |
| Selector epoch0 训练 anchors | 与 baseline anchor_pre 相同 |
| 后续重采样 anchors | 只有初始 anchors 数量等于全部 unaffected 数量，才证明已覆盖全部；否则继续 lazy |
| TF primary 训练/后评估 | pre evaluation 已填充所有 in-domain primary |
| TF audit_post | audit_pre 已填充同一个固定 audit 集合 |

每个 ratio、每个 event 仍是独立 cache。generic path、partial miss、重复 IDs、重复 hit、
返回值不别名 cache、不同 event 输入和 loss/gradient routing 均有测试。

'''
text += f"Selector：{ms('baseline','selector'):.3f} → {ms('stage_b','selector'):.3f} ms，{speed('baseline','stage_b','selector'):.3f}×。\nTF 在 Stage A 后：{ms('stage_a_grouped','tf'):.3f} → {ms('stage_b','tf'):.3f} ms，{speed('stage_a_grouped','stage_b','tf'):.3f}×，小但两个复测块方向一致。\n"
text += '''Selector missing-index 检查 304→24 次，Teacher miss 仍为24；
TF missing-index 检查184→20次，Teacher miss仍为20。target gather 数和实际 Teacher 点集不变。
不把 target_cache 的 inclusive 时间与 cache_missing_indices 再相加。

## 4. Stage C：Selector 跨 ratio 复用

增加 `GumbelNetworkImpl::forwardProbabilities` → `selector_cuda::networkMany` → `spatialForward`。
同一调用中只计算一次 XYZ/rotation/scale 的六个 Linear+ReLU，ratio 两层 embedding 和
最终 packed `[B,128]→[B,2]` CublasStrict GEMM 仍各 ratio 独立执行。
`selectorProbabilities` 同时把 current Gaussian gather 与 activated scale exp 从每 ratio 移到每 batch/stratum。
legacy 路径继续使用原 forwardProbability。

这里只复用 **forward activation**。每个 ratio 的 custom Function 仍显式接入原18个参数、执行原 backward，
梯度通过这些参数边累加；不向已缓存的内部 activation 再回传一份梯度。
原 DirectFinalRelu 的路由、small-N fallback、参数顺序、逐 ratio loss 和一次总 backward/step 均保留。
没有换到 batched-GEMM candidate，没有共享 Student activation 到下一 optimizer step。
这也意味着尚未消除跨 ratio 重复的 embedding backward；本轮选择数值风险较低的 forward 复用。

真实 Selector event 中，Student forwardProbability 对应304个 ratio调用变成76个多-ratio调用，
仍输出304组 probability；六个空间 Linear 的 Student 执行数1824→456，减少1368次。
Teacher 24次 miss 的计算不变；训练 custom network backward 仍160次，optimizer step仍20。

专门测试 B=1、16、17、3800、7430、8192 × ratio_count=1、2、4：
输出、loss、输入梯度、全部参数梯度和单步 CUDA Adam 更新共762项全部逐位一致。
真实 event 重排后存在约1e-9级梯度累加误差，最终参数约1e-8级差异；均通过原容差，详见下表。

'''
text += f"Stage C 相对 B 的 Selector：{ms('stage_b','selector'):.3f} → {ms('stage_c','selector'):.3f} ms，**{speed('stage_b','stage_c','selector'):.3f}×**。\n"
text += '''完整事件的边际改善比独立 Selector 小：两个块 B→C 分别约0.16%与2.71%，
不足以宣称稳定的单独完整事件百分比；保留依据是完整 Selector 分支在两个块均改善约5%–6%。
TF 路径未因 Stage C 改变，B/C TF 时间差视为测量波动，不计作 Stage C 收益。

## 5. native 结果与内存

主表为两个正反顺序块合并的10次 warm run。单位 ms；方括号是 min–max，std 是总体标准差。

'''
rows = []
for stage, label in [('baseline','初始'), ('stage_a_grouped','A'), ('stage_b','A+B'), ('stage_c','A+B+C')]:
    for branch in ['selector','tf','both']:
        if (stage, branch) not in paired: continue
        p = paired[stage,branch]
        a,b = p['cpu_wall_ms'],p['cuda_elapsed_ms']
        rows.append([label, branch, f"{a['median']:.3f}", f"{a['min']:.3f}–{a['max']:.3f}",
                     f"{a['std']:.3f}", f"{b['median']:.3f}", '40' if branch=='both' else '20'])
text += table(['阶段','分支','CPU median','min–max','std','CUDA span median','steps'], rows)
text += '\n初始单独批次10次 warm 的 native median（完整保留，用于检验批次/时钟差异）：\n\n'
text += table(['阶段','Selector','TF','完整事件'], [[label, *[f"{timings[stage,b]['cpu_wall_ms_median']:.3f}" if (stage,b) in timings else '未测/路径未变'
          for b in ['selector','tf','both']]] for stage,label in [('baseline','初始'),('stage_a_grouped','A'),('stage_b','A+B'),('stage_c','A+B+C')]])
text += '''
不能把分支独立 benchmark 的时间直接相加作为完整事件时间：独立 TF case 使用原配对审计的固定 TF seed，
完整事件使用生产 RNG 流程；两者各自在对应 before/after 内严格相同。
主结论只比较同一种 case；原报告225.970/556.474 ms只是历史参考。

GPU allocator peak 和相对 event 开始的 additional peak，单位 MiB（不含 CUDA context/driver 的整卡占用）：

'''
text += table(['阶段','分支','peak allocated','additional peak'], [[label,b,
        f"{paired[stage,b]['peak_allocated_bytes']['median']/1048576:.3f}",
        f"{paired[stage,b]['additional_allocated_bytes']['median']/1048576:.3f}"]
        for stage,label in [('baseline','初始'),('stage_a_grouped','A'),('stage_b','A+B'),('stage_c','A+B+C')]
        for b in ['selector','tf','both'] if (stage,b) in paired])
text += '''
A/B没有增加观测到的 GPU peak；A另有约669 KiB CPU group-ID表。
C降低独立 Selector peak约15.71 MiB，完整事件peak约1.68 MiB；不是通过减少样本或loss实现。

## 6. 组件与 kernel 证据

以下为一次代表性 mode1 run 的 inclusive component，总时间不能与子项相加。
EVENT-ONCE metadata 与 per-miss transport 明确分开：

'''
rows=[]
for stage,label,comp in [('baseline','初始','tf/geometry_transport'),('stage_a_grouped','A','tf/transport_metadata'),
    ('stage_a_grouped','A','tf/geometry_transport'),('baseline','初始','selector/cache_missing_indices'),
    ('stage_b','B','selector/cache_missing_indices'),('baseline','初始','tf/cache_missing_indices'),
    ('stage_b','B','tf/cache_missing_indices'),('stage_b','B','selector/student_forward'),
    ('stage_c','C','selector/student_forward'),('stage_c','C','selector/backward')]:
    r=components[stage,comp]
    rows.append([label,comp,'EVENT-ONCE' if comp.endswith('metadata') else 'CACHE-MISS' if comp.endswith('geometry_transport') else '每调用',
                 r['calls'],f"{r['cpu_ms']:.3f}",f"{r['cuda_span_ms']:.3f}",f"{r['cpu_ms']/r['calls']:.4f}"])
text += table(['阶段','组件','频率','calls','CPU ms','CUDA span ms','CPU ms/call'],rows)
text += '\nNsight Systems：同一event、E5，各trace一个完整分支，含evaluation。\n\n'
text += table(['阶段','分支','kernel calls','sum kernel GPU ms','<10us kernels'], [[r['stage'],r['branch'],r['kernel_calls'],f"{r['gpu_kernel_ms']:.3f}",r['tiny_lt_10us']] for r in nsys])
text += '\n按NVTX的CUDA launch关联得到的代表性归因：\n\n'
selected=[r for r in attributed if r['component'].endswith('/geometry_transport') or
          r['component'].endswith('/selector_network_backward') or
          (r['stage']=='stage_b' and any(r['component'].endswith('/'+x) for x in ['hexplane_backward','tf_mlp_backward','output_backward']))]
text += table(['阶段','组件','kernel calls','kernel GPU ms'], [[r['stage'],r['component'],r['kernel_calls'],f"{r['kernel_gpu_ms']:.3f}"] for r in selected])
text += '''
每条 kernel 名称、calls、总/平均时长见各阶段 `raw/nsys_*_kernels.csv`，原始 `.nsys-rep`、SQLite及stats也保留。
归因使用CPU launch correlation，不把GPU执行时间是否落在CPU range里当作归属。
`total.backward()` 会在autograd线程发射kernel，因此其整体范围统计覆盖所有捕获线程；
custom backward内部scope按实际发射线程归因。上述均为单独benchmark进程，不包含其他GPU进程。
组件范围有嵌套，不可叠加成总GPU时间。

kernel 数量的下降和 tiny kernel 比例支持“减少 orchestration/launch/indexing 有效”；
硬件occupancy、SM利用率、带宽瓶颈依然 inconclusive。不能根据 CUDA span 减少就宣称 tensor-core 利用率提高。

## 7. 正确性、失败与验证边界

保持 `atol=2e-5, rtol=2e-4`，没有放宽。下表的“mean最大值”是各 tensor 的 mean absolute error 再取最大，
完整逐 tensor 的 max/mean/shape/bitwise/allclose 数据均在 comparison JSON，不能理解成所有元素的合并mean。

'''
rows=[]
for label,path,key,prefix in [
    ('A：packed targets','stage_a_grouped/raw/comparison_baseline_tf.json','verification','target_'),
    ('A：TF首步梯度','stage_a_grouped/raw/comparison_baseline_tf.json','verification','tf_gradient_'),
    ('A：TF首步更新','stage_a_grouped/raw/comparison_baseline_tf.json','verification','tf_updated_'),
    ('A：TF逐步loss','stage_a_grouped/raw/comparison_baseline_tf.json','verification','tf_loss_'),
    ('A：TF最终参数','stage_a_grouped/raw/comparison_baseline_tf.json','final_parameters',None),
    ('B：Selector全部校验','stage_b/raw/comparison_baseline_selector.json','verification',None),
    ('B：Selector最终参数','stage_b/raw/comparison_baseline_selector.json','final_parameters',None),
    ('B：TF最终参数vs A','stage_b/raw/comparison_stage_a_grouped_tf.json','final_parameters',None),
    ('C：Selector首步梯度','stage_c/raw/comparison_stage_b_selector.json','verification','selector_gradient_'),
    ('C：Selector首步更新','stage_c/raw/comparison_stage_b_selector.json','verification','selector_updated_'),
    ('C：Selector逐步loss','stage_c/raw/comparison_stage_b_selector.json','verification','selector_loss_'),
    ('C：Selector最终参数','stage_c/raw/comparison_stage_b_selector.json','final_parameters',None),
    ('组合：完整事件校验','stage_c/raw/comparison_baseline_both.json','verification',None),
    ('组合：完整事件最终TF参数','stage_c/raw/comparison_baseline_both.json','final_parameters',None)]:
    rows.append([label,*errors(path,key,prefix)])
text += table(['比较','tensor数','max abs最大值','mean abs最大值','结果'],rows)
text += '''
`verification.pt` 保存每次TF miss的packed target、每步loss、首步全部参数梯度和首步更新。
native warm_0另存20步后最终参数。完整事件也独立比较；Gaussian、normal Adam和steps断言均通过。
TF grid backward本身含atomicAdd，重复训练参数允许既有容差下的微小非逐位差异；不要求不切实际的全模型逐位确定性。

额外验证：

'''
fixture=json.loads((D/'fixture_comparisons.json').read_text())
text += table(['真实fixture','校验tensor数','max abs最大值','mean abs最大值','结果'],
              [[k,len(v),f"{max(x['max_abs'] for x in v):.6g}",f"{max(x['mean_abs'] for x in v):.6g}",'全部通过'] for k,v in fixture.items()])
text += '''
每个global/AABB fixture还单独执行全部点、全部ratio的直接transport比较；
local全部144组和混合flags/空组/-1/空输入/无group用例均通过。
既有 `change_distill_test.cpp` 在CPU legacy、CUDA Selector+TF两种backend重新编译执行，
涵盖local/global/AABB、outside、anchors、对应关系、split epoch、normal optimizer隔离等；
A、B、C各阶段均通过，cache扩展测试也通过。
本轮没有运行全量SLAM/轨迹质量评估；验收范围是冻结真实event和已有生命周期fixtures，不能推断长期轨迹质量收益。

只接受了有完整分支收益且通过正确性的阶段。失败的向量化源码、初始结果和comparison仍保存在`stage_a/`；
没有保留一段仅因microbenchmark快而未通过最终参数容差的代码。

## 8. Stage D 与下一步

本轮仅检查，不实现输入梯度裁剪：

| 边界 | 真实CD中的无梯度输入 | 仍然必须计算的内容 | 结论 |
| --- | --- | --- | --- |
| Selector NetworkFunction | Gaussian XYZ/rotation/scale和ratio | 两层之间的dh、所有embedding及最终层参数梯度 | 仅最外层输入dx理论可省；当前不改 |
| TF QueryFunction | xyz、ratio、AABB | 每个plane的插值权重、plane参数atomicAdd | queryGradKernel把坐标导数与plane梯度混合；需单独隔离验证 |
| TF OutputFunction | base xyz/log-scale/quaternion | residual/head梯度，含quaternion归一化链 | 可考虑跳过base输出写入，不能切断head梯度 |
| TF NeuralFunction | sampled feature需要梯度 | feature梯度用于训练HexPlane | 不可detach feature |

`ctx->needs_input_grad`或等价mask有潜力，但当前profile不能隔离这些可省子计算的完整event收益。
Selector已有backward kernel数在B/C相同；TF query/output全部kernel时间还包含必需参数梯度。
因此未花费较大工作量重写这些CUDA kernel，也没有将它们的总耗时假定为可节省量。

下一项最高价值工作是继续降低TF transport的逐组算式/整批index_copy成本，
先定位p22容差失败对应的XYZ累加差异，并设计保留原数值语义的融合或grouped transport方案。
保留的A仍有约3万个geometry kernel、显著CPU wall，不能认为这一瓶颈已彻底解决。
第二候选是Selector shared embedding backward，但必须重新验证梯度累加顺序及完整分支收益。

## 9. 文件与复现

本地证据目录：`tests/diagnostics/change_distill_optimization_20261017/`。
服务器镜像：`/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017`。
所有源码改动在当前本地仓库；服务器只在独立目录构建/运行，未覆盖正式build/lib。

| 文件/函数 | 改动 |
| --- | --- |
| src/change_distill.cpp::TargetTransport / transportTarget | event-local membership table，按原算式transport |
| include/frozen_teacher_target_cache.h::get / getCached | 已证明coverage的直接cache gather |
| src/change_distill.cpp::execute | 显式传递coverage；同batch跨ratio复用probabilities |
| improvements/selector/gumbel_network.h::forwardProbabilities | CD多ratio入口，legacy回退 |
| include/selector_cuda.h / src/selector_cuda.cu::spatialForward / forward | 复用六个空间embedding activation，保持strict final GEMM |
| src/selector_cuda.cpp::NetworkFunction / networkMany | activation共享，原参数边和逐ratio backward保留 |
| tests/frozen_teacher_target_cache_test.cpp | direct/lazy/partial/repeated-ID覆盖 |

```bash
cd /home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017
PY=/home/prmi/anaconda3/envs/zwj/bin/python
$PY stage_c/build_native.py
$PY build_verify.py stage_c
$PY run_cases.py stage_c:selector:new_selector stage_c:tf:new_tf stage_c:both:new_both
$PY compare.py baseline stage_c selector
$PY build_profile.py stage_c
$PY run_cases.py stage_c:selector:new_profile:1:1
```

原生底层命令：

```text
stage_c/benchmark_native EVENT OUTPUT BRANCH 5 8192 4 10 0 2
```

EVENT固定为原`events_supplement/local_loop/event_6207_1`。
build_commands JSON记录完整编译/链接命令；`cuda_objects.json`验证未变CUDA source的object复用，
改变的Selector CUDA object从源码重新编译，未错误复用baseline object。
Stage C最后仅清理helper未使用的参数签名；初始源码在`stage_c.tar.gz`，最终源码在`stage_c/source`，
主表正反顺序复测使用最终原生构建，最后也重建并执行完整事件correctness runner。

`profiling_outputs.tar.gz`保留全部新raw CSV/JSON、model/target/gradient PT、日志、Nsight原始输出、
各阶段源码、完整diff和构建记录。`artifact_manifest.json`记录归档哈希。
`timings.csv/json`、`paired_summary.json`、`components.csv/json`、`nsys_components.csv/json`可直接继续分析。
生产正常路径未添加逐操作synchronize；profiler hooks只存在隔离副本。

最终建议：保留A保守方案、B和C；逐点BMM已撤销；D不启用。
部署或其他机器比较时按相同输入/完整分支契约重新测量，不沿用本机百分比作为保证。
'''
deployment = D / 'deployment_20261018/DEPLOYMENT_REPORT.md'
if deployment.exists():
    text = text.replace('所有源码改动在当前本地仓库；服务器只在独立目录构建/运行，未覆盖正式build/lib。',
                        '第1–9节记录优化验证阶段，当时服务器只在独立目录构建/运行；正式部署结果见第10节。')
    text += '\n## 10. 2026-10-18 正式部署\n\n' + deployment.read_text(encoding='utf-8').split('\n', 1)[1].lstrip()
(ROOT/'CHANGE_DISTILL_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8',newline='\n')
print(ROOT/'CHANGE_DISTILL_OPTIMIZATION_REPORT.md')
