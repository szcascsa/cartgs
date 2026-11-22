"""只使用已完成的本轮验收与测量生成中文报告。"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import subprocess

D=Path(__file__).resolve().parent
root=D.parents[2]
s=json.loads((D/'final_summary.json').read_text())
c=s['correctness'];assert c['all_exact'] and c['complete'] and c['failures']==0
def ms(e,label,stage):return e['network'][label][stage]['wall']['mean_ms']
e=s['M_grid'][-1];b=ms(e,'baseline','network_backward');f=ms(e,'final','network_backward')
fb=ms(e,'baseline','network_forward_backward');ff=ms(e,'final','network_forward_backward')
cb=e['caller']['baseline']['cuda']['wall']['mean_ms'];cf=e['caller']['final']['cuda']['wall']['mean_ms']
legacyb=ms(e,'legacy','network_backward');legacyfb=ms(e,'legacy','network_forward_backward')
legacyf=ms(e,'legacy','network_forward');newf=ms(e,'final','network_forward')
legacycaller=e['caller']['final']['legacy']['wall']['mean_ms']
paired=list(csv.DictReader((D/'paired.csv').open()))
def pair(mode,n,stage):
    return next(r for r in paired if int(r['strategy'])==mode and int(r['M'])==n and r['stage']==stage)
pfb=float(pair(1,219969,'fb_wall')['mean_ms']);pff=float(pair(6,219969,'fb_wall')['mean_ms'])
before=json.loads((D/'profile_final_1/profile_summary.json').read_text())
after=json.loads((D/'profile_final_6/profile_summary.json').read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
paired_table=table(['M','原 CUDA paired F+B mean/p50/p95','新 CUDA paired F+B mean/p50/p95','speedup'],
    [[n,'/'.join(f"{float(pair(1,n,'fb_wall')[k]):.4f}" for k in ['mean_ms','p50_ms','p95_ms']),
      '/'.join(f"{float(pair(6,n,'fb_wall')[k]):.4f}" for k in ['mean_ms','p50_ms','p95_ms']),
      f"{float(pair(1,n,'fb_wall')['mean_ms'])/float(pair(6,n,'fb_wall')['mean_ms']):.3f}×"]
     for n in [1000,5000,10000,20000,50000,100000,200000,219969]])
def groups(r):
    a=defaultdict(float)
    for k in r['kernels_detail']:
        p=k['stage'];n=k['name'];v=k['duration_ms']/r['iterations']
        g=('final dFeature' if 'gemm_m128_n' in p and '_k2_nn' in p else
           'final dW' if '/gemm_m128_n2_' in p else
           'bias/ratio reduction' if '/bias_' in p else
           'branch2 dX' if 'linear_vjp_in32_out32/gemm_m32_n' in p and '_k32_nn' in p else
           'branch2 dW' if 'linear_vjp_in32_out32/gemm_m32_n32_k' in p else
           'branch1 dW' if ('linear_vjp_in3_out32/gemm_m3_n32_' in p or 'linear_vjp_in4_out32/gemm_m4_n32_' in p) else
           'XYZ/q/scale dX' if ('linear_vjp_in3_out32/gemm_m3_n' in p or 'linear_vjp_in4_out32/gemm_m4_n' in p) else
           'split + ReLU2' if 'splitRelu' in n else
           'direct final + ReLU2' if 'directFinal' in n else
           'ReLU1 / scalar ReLU' if 'reluVjp' in n else 'ratio small GEMM')
        a[g]+=v
    return a
g1=groups(before['profile_219969']);g6=groups(after['profile_219969'])
runtime_rows=[]
for label,record in [('原 CUDA',before['profile_219969']),('新 CUDA',after['profile_219969'])]:
    t=record['transfers_and_memsets']
    runtime_rows.append([label,f"{len(t)/10:.0f}",f"{sum(v.get('bytes',0) for v in t)/10:.0f}",
        f"{sum(v['end']-v['start'] for v in t)/1e7:.6f}",
        ', '.join(f"{k}={v['count']:.0f}" for k,v in record['runtime_apis'].items())])
runtime_table=table(['版本','memset/iteration','clear bytes/iteration','memset active ms','captured runtime API/iteration'],runtime_rows)
breakdown=table(['backward 阶段','原 CUDA active ms','新 CUDA active ms'],
    [[k,f'{g1.get(k,0):.6f}',f'{g6.get(k,0):.6f}'] for k in dict.fromkeys(list(g1)+list(g6))])
profiles=table(['M','原/新 kernels','原/新 active ms','原/新 device span ms'],
    [[n,f"{before['profile_'+str(n)]['kernels']:.0f} / {after['profile_'+str(n)]['kernels']:.0f}",
      f"{before['profile_'+str(n)]['active_ms']:.4f} / {after['profile_'+str(n)]['active_ms']:.4f}",
      f"{before['profile_'+str(n)]['mean_device_span_ms']:.4f} / {after['profile_'+str(n)]['mean_device_span_ms']:.4f}"]
     for n in [10000,50000,100000,219969]])
repeat=table(['独立批次','原 CUDA B','新 CUDA B','原 CUDA F+B','新 CUDA F+B'],
    [[i+2,f"{r['baseline']['network_backward']['mean_ms']:.5f}",f"{r['final']['network_backward']['mean_ms']:.5f}",
      f"{r['baseline']['network_forward_backward']['mean_ms']:.5f}",f"{r['final']['network_forward_backward']['mean_ms']:.5f}"]
     for i,r in enumerate(s['repeat_219969'])])
memory={}
for row in csv.reader((D/'final_measure/network_memory.csv').open()):
    if row[1]=='219969':memory[row[0],row[2]]=int(row[5])
mb=memory['strategy1','network_forward_backward']/2**20;mf=memory['strategy6','network_forward_backward']/2**20
boundary=list(csv.DictReader((D/'boundary_final.csv').open()));assert all(r['exact']=='1' for r in boundary)
training=json.loads((D/'candidate_6_all_final/summary.json').read_text())['checkpoints']
train_table=table(['checkpoint','M','steps','parameter / Adam m/v relative L2','average hard mismatch'],
    [[r['name'],r['N'],r['training']['steps'],'0 / 0 / 0',r['training']['average_hard_mismatch']] for r in training])
kernels=defaultdict(lambda:dict(count=0,active_ms=0.,registers=set(),shared=set()))
for k in after['profile_219969']['kernels_detail']:
    v=kernels[k['name']];v['count']+=.1;v['active_ms']+=k['duration_ms']/10
    v['registers'].add(k['registers']);v['shared'].add((k['static_shared'] or 0)+(k['dynamic_shared'] or 0))
kernel_table=table(['kernel','launch/iteration','active ms','register/thread','shared bytes/block'],
    [['`'+(name if len(name)<=160 else name[:157]+'...').replace('|','/')+'`',f"{v['count']:.0f}",f"{v['active_ms']:.6f}",'/'.join(map(str,sorted(v['registers']))),'/'.join(map(str,sorted(v['shared'])))]
     for name,v in sorted(kernels.items(),key=lambda kv:-kv[1]['active_ms'])])
candidate5=list(csv.DictReader((D/'measure_5/strategy_5.csv').open()))
def oldmean(rows,stage):return next(float(r['mean_ms']) for r in rows if r['M']=='219969' and r['stage']==stage and r['metric']=='wall')
baseline_head=json.loads((D/'baseline_git.json').read_text())['head']
implementation=subprocess.check_output(['git','-C',str(root),'log','-1','--format=%H','--','src/selector_cuda.cu'],text=True).strip()
text=f'''# CUDA Selector backward 优化报告

结论：**`CUDA_SELECTOR_CORRECTNESS = PASS`**。建议合入本轮 final 分支梯度路由与 ReLU 融合；`Selector.backend = legacy | cuda` 保留，配置默认仍为 legacy。没有修改 forward 数学或任何 loss/sampling/optimizer/Guard 规则。

本轮 M=219969、100 warmup + 500 measurements 的同步 wall：CUDA isolated backward **{b:.5f} → {f:.5f} ms**（{b/f:.3f}×，降低 {(1-f/b)*100:.2f}%）；同进程交替顺序 paired F+B **{pfb:.5f} → {pff:.5f} ms**（{pfb/pff:.3f}×）；structural caller **{cb:.5f} → {cf:.5f} ms**（{cb/cf:.3f}×）。每项before/after使用同binary。完整 Mapping 未测量，不以 caller 的收益代替 Mapping 收益。

## 1. 版本、环境与修改范围

起点 HEAD：`{baseline_head}`；分支 `cuda-selector`。基线冻结/profiler 提交 `52fc6b4`；直接路由提交 `434c838`；final+ReLU 提交 `474e8e0`；最终小 batch 修正提交 `{implementation}`。本报告随后单独提交，避免把报告自身 hash 写成循环引用。

RTX3090 / CUDA11.8.89 / GCC9 / Torch2.3.1，FP32、TF32=false、current CUDA stream。Native LibTorch ABI=1，Python Torch ABI=0，分别编译。隔离实验目录 `/home/prmi/zwj/selector-cuda-20261015`；生产 `/home/prmi/zwj/mycartgs` 的本轮训练配置、二进制和代码未覆盖。

产品改动只有 `include/selector_cuda.h`、`src/selector_cuda.cu`。thin autograd 与 forward 源码未改；网络层、参数布局/identity、checkpoint、quaternion/scale activation、Gaussian input gradients、ST、RNG、GI、Protection、Replay/CD/DSD、Adam/Guard 数学保持原样。原有未提交工作保留，未 push。运行验收使用隔离环境中冻结的其他依赖；当前工作区其他未提交修改不属于此报告的验证范围。

所有可复现源、脚本、CSV、源码/binary hash、Nsight trace、候选补丁与原始验收归档位于 `tests/diagnostics/cuda_selector_backward_20261015`（下文简称 D）。最终源 SHA-256：

```json
{json.dumps(s['source_sha256'],indent=2)}
```

## 2. 基线重新测量与 profiler 分解

改代码前，冻结 HEAD/status/hash，重新顺序测量 legacy 与 CUDA 八规模。首批 M219969 wall：legacy F/B/F+B=3.96291/2.29648/6.16905 ms，CUDA=2.09733/2.24939/4.27948 ms；不是引用旧报告数字。D/baseline 保存此批数据。最终又在同一候选 binary 内重测 legacy/原 CUDA/final，并额外重复两批；下文最终收益以这组新 A/B 为准。

初始 NVTX+Nsight M219969：43 kernels，active 2.17933 ms，device span 2.23963 ms。主要成本是 bias/ratio reduction 约0.409 ms、split+ReLU2约0.357 ms、ReLU1约0.214 ms、branch2 dX约0.346 ms；final dFeature约0.139 ms，final dW约0.143 ms，branch2/branch1 dW约0.255/0.156 ms。原实现已经有 scalar ratio reduction；没有重复实现两层大 M ratio GEMM。

最终同一 profiler executable、10 个 captured backward 的 before/after：

{breakdown}

表中 cuBLAS split-K reduction 跟随对应 GEMM 归类；bias/ratio reduction 保留 ATen tree。ratio small GEMM 约为微秒级，不是主要收益。input preparation / quaternion normalize / scale exp 在上层图，network profiler 不将其混入 MLP backward；caller 则包含这些直接相连的输入准备与 VJP。

## 3. 候选、验收与取舍

| 候选 | 正确性 | M219969 wall（B / F+B / caller） | 决策 |
|---|---|---|---|
| 原 CUDA strategy1 | frozen oracle | 见最终 A/B | 保留诊断 oracle |
| B：direct final，单独 ReLU | 七快照与7000步 exact | {oldmean(candidate5,'network_backward'):.5f} / {oldmean(candidate5,'network_forward_backward'):.5f} / 4.93702 ms（探索批） | 已单独提交，随后由 C 改善 |
| C：direct final + ReLU2 | 最终完整 gate exact | {f:.5f} / {ff:.5f} / {cf:.5f} ms | 采用 |
| warp tree narrow dX | 满足旧 tolerance，但新增梯度 relative L2≈1.17e-7 | 未继续验收 | 拒绝，基线 exact，未放宽标准 |
| ordered-FMA warp narrow dX | 七快照/7000步 exact | 2.54948 / 4.60709 / 5.26858 ms（探索批） | 拒绝，低 lane 利用率使整体回退 |
| shared tile spatial ReLU+dX | 七快照/7000步 exact，小 batch 36个输入梯度 Tensor不 exact | 独占重复 B=2.40994/2.41045 ms，C=1.99468 ms | 拒绝，增加空间未换得速度 |

空间不作为 gate：已按用户补充要求尝试保留三分支 dH 并共享 tile。候选9初始 forward 也异常变慢，存在 CPU 编译并行干扰，初始 `measure_9` 不用于性能结论；`clean9` 顺序重测仍慢，因而拒绝。所有失败候选只保留 patch/证据，不在产品 enum/hot path 中。每个通过单步的候选才进入7000步训练，再进行八规模性能/caller；warp tree 在 exact 检查失败后停止。

## 4. 最终 kernel 结构与小 batch 分派

M>16 时去掉完整 `[M,128] dFeature` 分配/GEMM。每个 branch 用 current-stream CUDA kernel 按原两项乘加顺序生成 `[M,32]` 的第二层 preactivation VJP，同时应用原 ReLU mask；不先写 `[M,128]` 再 split。final dW 继续使用原 packed forward feature 做完整128维 cuBLAS GEMM，forward 累加顺序完全没动。

本轮额外边界扫描发现 cuBLAS 对 M≤16 的 k=2 GEMM 有不同累加顺序，直接公式会新增约1e-8～1e-7差异。因此 M≤16 继续走原 FusedActivation backward；不是用 tolerance 接受差异。修正后与原 CUDA strategy1 的 {len(boundary)} 条 gradient Tensor 比较 exact，覆盖 M=0/1/2/3/7/12～25/31/32/33/63/64/65/127/128/129/257/1000/5000/10000、scalar/row。这项补充测试比较新旧 CUDA，不额外宣称所有小 batch 的 legacy bitwise 结果。

scalar ratio 仍先生成正确逐点 contribution、应用同一个第二层 mask、用原 ATen sum得到 `[1,32]`，只 backward 一次 ratio MLP。row ratio 保持逐行 generic path。没有将 `sum(dZ*W)` 改成 `sum(dZ)*W`，避免 reduction 顺序变化。`[M,32]` ratio contribution 仍存在。

branch2 dX、branch1 XYZ/q/scale dX 和全部 dW 继续 cuBLAS；narrow custom dX 的已测实现没有获得合格整体收益。db 继续 ATen `sum(0)`，没有重新引入此前已拒绝的 fused bias reduction。ReLU 的零点与 NaN 条件沿用原实现的 `h<=0 ? 0 : upstream`，未改为 `>=0` 或新的判断式。

## 5. 七 checkpoint direct parity 与1000-step

复用 `CUDA_SELECTOR_DIRECT_PARITY_REPORT.md` 的同输入 oracle，每个快照只加载一次基础数据，clone 为 legacy A/B/CUDA。四 ratio .01/.05/.10/.15 × scalar/row；全部八层 branch、logits、probability、production elastic probability、cached uniform、六 seed fixed-noise Gumbel。

最终 **{c['checks']:,} 项比较，0 failure，全部 exact**：max abs/mean abs/RMS/p99/relative L2=0，cosine=1；XYZ、activated quaternion/scale、ratio、raw quaternion/log-scale/opacity，以及全部18参数 VJP 均 exact。固定 `grad_logits/grad_probability/grad_ST`，ratio/GI/photo-opacity/combined/Guard uniform/Replay/CD/DSD 的外部 target/upstream 完全共享，没有分别运行 renderer 生成不同 upstream。

cached-uniform 56组、5,151,304个决定，mismatch=0；Gumbel 336组、30,907,824个决定，mismatch=0；compact IDs/count exact，无新增 near/far-boundary disagreement。网络不消费 RNG，保留原 CUDA generator/noise 顺序。all-tie 仅 diagnostic，不作 gate。

{train_table}

全部7000步的 logits/probability/loss、gradient、参数、Adam m/v/step、selected IDs exact；1/10/100/1000步另有逐 Tensor记录。四组 direct/preview Adam 各1000步同样 exact，覆盖 normal/zero/undefined grad、LR=0、非默认 beta/eps/decay、不同 per-tensor step、commit/discard/shrink/projection。Native Guard 六种 Gaussian/Selector/TF 联合事务 exact；Native CD/DSD/Replay/GradientConstraint 4/4通过。legacy A vs B repeat error=0，CUDA error=0；没有新的 selected-ID、gradient、parameter 或 Adam 漂移。

完整 renderer 质量/SLAM 不作为本轮 gate，也未运行在线统计实验。

## 6. 八规模性能与独立复测

以下均为本次 frozen workload、100 warmup +500 measurements，单位 ms。同步 host wall；CUDA event 的对应 mean/p50/p95保存在CSV/final_summary.json。F+B是fresh graph，B是相同saved graph VJP。没有省略 Gaussian input gradient。

{(D/'performance_tables.md').read_text(encoding='utf-8')}

M219969另外两批顺序复测（新→原，和主批原→新交替）：

{repeat}

上述顺序批次的M219969主批F+B mean仅改善{(1-ff/fb)*100:.2f}%，但另两批改善更大，故追加了同一进程、相同参数/输入/allocator环境、每轮交替`1→6 / 6→1`的paired实验。每个strategy仍有100 warmup与500 measurements；没有修改网络或VJP数学。下面是更能隔离host/进程顺序影响的结果：

{paired_table}

paired的M219969 event F+B：{float(pair(1,219969,'fb_event')['mean_ms']):.5f}→{float(pair(6,219969,'fb_event')['mean_ms']):.5f} ms；forward part event {float(pair(1,219969,'forward_part_event')['mean_ms']):.5f}→{float(pair(6,219969,'forward_part_event')['mean_ms']):.5f} ms，backward part event {float(pair(1,219969,'backward_part_event')['mean_ms']):.5f}→{float(pair(6,219969,'backward_part_event')['mean_ms']):.5f} ms。fresh-forward后的cache状态与isolated backward不同，不能将这个backward part数值当成独立Bbenchmark。paired FB trace/解析另存D/paired_fb.*、paired_profile_summary.json。

最终相对 legacy：forward {legacyf/newf:.3f}×、backward {legacyb/f:.3f}×、F+B {legacyfb/ff:.3f}×、structural caller {legacycaller/cf:.3f}×。caller包含本 fixture 的概率/ST/Protection/compact/gather、photo-opacity/ratio VJP与Selector update，**不包括 renderer/TF/Gaussian optimizer 或完整 Mapping**。本轮没有 Mapping wall speedup结论。

八规模包含 launch/host 占比较高的小 M，不能把大 M收益外推为所有 M同比例提升。F+B 的host span在重复批次中有可见波动，因此保留每批 mean/p50/p95，不用其中最低值代表最终收益。forward源码未变，其小幅计时差按噪声解释，不能称为新的 forward优化。

## 7. Nsight before/after、资源与内存

{profiles}

M219969最终 kernel 列表：

{kernel_table}

{runtime_table}

本次backward capture没有H2D/D2H/D2D copy，也没有cudaMalloc/cudaFree或显式同步API；8次memset仅清零ATen reduction的少量控制/计数空间，共32 bytes，约0.009ms。不能把这部分称为大Tensor清零瓶颈。profile外部的warmup、fixture载入、事件等待和全局同步仍存在，分别用于准备数据和统一计时。

原始trace与解析结果见 D/profile_final_1、D/profile_final_6；另有219969 row-ratio trace。表中长模板名缩写，完整demangled name见原始JSON。register/shared-memory来自 CUPTI trace。Nsight Compute明确返回 `ERR_NVGPUCTRPERM`，因此**DRAM bytes与 achieved occupancy 未实测**，没有用理论值冒充counter。Nsight kernel active/device span与未profile的 wall分开报告。runtime API、copies/memsets也在 profile_summary.json；ATen caching allocator内部CPU开销未逐次归因，不能从GPU空隙倒推出具体Tensor allocation成本。

关键中间量：

| Tensor | 原/新 | 生命周期与逻辑字节（M219969，FP32） |
|---|---|---|
| forward packed `[M,128]` | 都保留 | saved until backward，112,624,128 bytes；确保forward/final dW顺序 |
| final dFeature `[M,128]` | 原保留，新在M>16消除 | 原从final dX到最后branch；112,624,128 bytes |
| branch dPre2 `[M,32]` | 都保留 | 逐branch，28,156,032 bytes；ratio后sum |
| branch dH1 `[M,32]` | 都保留 | 当前branch两层VJP之间，28,156,032 bytes |
| scalar ratio dH `[1,32]` | 都保留 | reduction后仅128 bytes，两层小GEMM |
| input gradients `[M,3/4/3]` | 都返回 | autograd上层继续使用 |

消除dFeature自身一次写回和一次读取，逻辑量为225,248,256 bytes/调用；这仅是该workspace的Tensor流量差，不是整个backward的实测DRAM差，也未扣除分支重复读dZ的缓存行为。

F+B incremental allocated peak：原 CUDA **{mb:.2f} MiB**，新 CUDA **{mf:.2f} MiB**，减少 {mb-mf:.2f} MiB。它是消除冗余往返的副作用；显存不作为本轮目标或gate。未新增全局预分配workspace、异步跨stream事务或CUDA Graph。

## 8. 仍保留的成本与最终判断

主要收益来自取消完整 dFeature GEMM/global roundtrip，并将其按branch生成与ReLU2合并。final/各branch dW的batch reduction、精确兼容的bias/ratio sum、branch2与窄input dX仍是主要剩余成本；它们在当前已验证实现中保留原cuBLAS/ATen顺序。本轮拒绝了更改累加树的方案，也拒绝了correctness通过但完整F+B/caller变慢的方案。

建议合入本轮两个成功的backward优化和小batch回退；启用CUDA backend后自动使用新backward。是否改变产品默认backend、完整Mapping收益和用户最终在线质量测试属于独立决策。本报告不声称已稳定低于2.0ms、达到1.5～1.8ms理想目标，或所有GPU/库版本均bitwise一致。当前证据适用于本轮RTX3090/CUDA11.8/Torch环境及覆盖的输入。

最终回答：isolated backward {b:.5f}→{f:.5f}ms；paired F+B {pfb:.5f}→{pff:.5f}ms，caller {cb:.5f}→{cf:.5f}ms；correctness仍为 **PASS**，最终被采用实现没有观察到任何新增selected-ID、gradient、parameter或Adam漂移。失败候选已退出生产路径。
'''
(root/'CUDA_SELECTOR_BACKWARD_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
print('report written',len(text))
