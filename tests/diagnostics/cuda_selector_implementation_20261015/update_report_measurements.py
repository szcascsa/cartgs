"""用已归档的最终 full-gradient caller 数据更新报告，避免混用初版边界。"""
import csv
import json
from pathlib import Path

dest = Path(__file__).resolve().parent
report = dest.parents[2]/'CUDA_SELECTOR_IMPLEMENTATION_REPORT.md'


def rows(name):
    with (dest/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


caller = {(r['backend'],r['stage']): r for r in rows('caller_benchmark.csv') if r['metric']=='wall'}
latency = lambda r: ' / '.join(f'{float(r[k]):.6f}' for k in ('mean_ms','p50_ms','p95_ms'))
section = '''## Selector caller 性能

当前 room checkpoint N=109564，保存真实 continuous maturity/uniform，六类 Gaussian 属性 gather。计时包含 raw quaternion normalization、scale/opacity activation、网络/elastic mix、ST、动态 compact、opacity VJP、ratio、全部 Selector 输入及 raw Gaussian 相关 backward、Selector Adam。每次 backward 显式检查 XYZ/opacity/log-scale/raw quaternion 的 gradient 存在；没有为速度关闭这些 gradient。

这是有 Gaussian 输入梯度的结构化 caller；仍**不含真实 rasterizer、TF、GI 和 Gaussian optimizer step**，不能等同完整 Mapping。100 warmup + 500 measurement，D/caller_benchmark.csv。以下均为 wall mean/p50/p95。

| 阶段 | legacy ms | CUDA ms | mean 倍率 |
|---|---|---|---:|
'''
for stage,label in [('current_caller_structural','结构化 caller，含 Gaussian 输入梯度'),('adam','Adam'),
                    ('guard_preview_commit','Guard preview+commit'),('guard_preview_discard','Guard preview+discard'),
                    ('selection_mask_gather','ST/mask/compact/gather'),('legacy_gumbel_st','原始两类 Gumbel/ST')]:
    a,b=(caller[(backend,stage)] for backend in ('legacy','cuda'))
    section += f"| {label} | {latency(a)} | {latency(b)} | {float(a['mean_ms'])/float(b['mean_ms']):.3f}× |\n"
section += '''
Guard 此项不含历史渲染、projection solve、质量检查，不能称完整 Guard speedup。两组 Gumbel/ST 使用同一原实现、相同 logits 和重置的同一 RNG state；小差异属于测量波动，没有采样优化。外围成本低，未引入 custom RNG、CUB scan 或融合 Protection/ST。

全部八个 M 也完成 caller/ST/selection/Adam/preview 网格，每项 100+500。该组使用真实审计 Gaussian 属性和 Selector checkpoint，maturity 固定为 1、entry=0、固定 cached uniform，完整保留 Gaussian 输入 gradient；用于统一规模比较，不能代替真实 maturity 的 room fixture。各阶段 event/wall mean/p50/p95 见 D/caller_grid_benchmark.csv。

| M | legacy caller mean ms | CUDA caller mean ms | 倍率 |
|---:|---:|---:|---:|
'''
for n in (1000,5000,10000,20000,50000,100000,200000,219969):
    r=[r for r in rows(f'grid_caller_{n}.csv') if r['metric']=='wall' and r['stage']=='current_caller_structural']
    a,b=(float(next(x for x in r if x['backend']==backend)['mean_ms']) for backend in ('legacy','cuda'))
    section += f'| {n} | {a:.5f} | {b:.5f} | {a/b:.3f}× |\n'
section += '''
## Nsight、dispatch 与显存

真实审计 M=219969，10 次单独捕获的网络 F+B：

| Backend | Forward kernels / active ms | Backward kernels / active ms | 网络总 kernels |
|---|---|---|---:|
'''
network=json.loads((dest/'round2_nsight_summary.json').read_text())
for key,label in [('legacy','legacy'),('cublas','split cuBLAS'),('warp','warp'),('cublas_strict','strict cuBLAS')]:
    f,b=(network[key]['stages'][name] for name in ('network_forward','network_backward'))
    section += f"| {label} | {f['kernels_per_iteration']:g} / {f['kernel_active_ms']:.5f} | {b['kernels_per_iteration']:g} / {b['kernel_active_ms']:.5f} | {f['kernels_per_iteration']+b['kernels_per_iteration']:g} |\n"
section += '''
这不是旧审计 351-kernel caller 的同边界对照。strict forward kernel 数未减少，bias/ReLU epilogue 降低 active time；backward 增加 split/VJP launches，大 M 单独 backward 仍略慢。D/round2_nsight_summary.json 保留 kernel names、device span、active/gap、CUDA API、copy 和 launch gap。

最终有 Gaussian 输入梯度的 structural caller（N=109564，10 次捕获）：

| Scope | legacy kernels/iteration | CUDA kernels/iteration |
|---|---:|---:|
'''
nsight=json.loads((dest/'caller_nsight_summary.json').read_text())
for stage,label in [('caller_preparation','Gaussian activation preparation'),('probability_network','network/probability/elastic mix'),
                    ('selection_mask_gather','ST/mask/compact/gather'),('loss','loss'),('caller_backward','caller backward'),
                    ('adam','Adam'),('selector_caller','总计')]:
    a,b=(nsight[backend]['stages'][stage]['kernels_per_iteration'] for backend in ('legacy','cuda'))
    section += f'| {label} | {a:g} | {b:g} |\n'
section += '\n'
for backend in ('legacy','cuda'):
    v=nsight[backend]['stages']['selector_caller']
    average=lambda key:sum(s[key] for s in v['spans'])/len(v['spans'])
    section += f"{backend}：active {average('active_ms'):.5f} ms，device span {average('device_span_ms'):.5f} ms，span 内 gap {average('gap_ms'):.5f} ms；copy {v['copy_bytes_per_iteration']:g} bytes/iteration，`cudaStreamSynchronize` host API {v['synchronization_cpu_ms_per_iteration']:.6f} ms。\n\n"
section += '''legacy 每次 D2D=128 bytes、D2H=4 bytes；CUDA D2D=0、D2H=4 bytes；H2D=0。两组均有一次动态 count 的 stream synchronize。warmup 后捕获区间无 cudaMalloc/cudaFree，仍有 caching allocator 中间 Tensor 和每次 8 个 memset。不能把 nonzero 等待前序网络的 host 时间全算作 compact GPU 成本。D/caller_nsight_summary.json。

Native Kineto、M=219969：legacy 每次 `aten::addmm=9`、`relu=8`、`threshold_backward=8`、`cat=1`；CUDA 全部为 0，hot path 已接管。strict custom pack 仍占 `[M,128]`，不能称 concat workspace 已消除。D/operator_dispatch_summary.json。

网络 F+B allocator incremental peak（M=219969）：legacy **397195264 bytes**，strict **465846272 bytes**，增加约 **65.5 MiB**。这不是显存优化成功；packed、完整 dFeature 和保存激活是严格路径的代价。原始各阶段 allocated/reserved peak 见 D/round2_network_memory.csv，不能叠加各阶段 peak。

有 Gaussian 输入 gradient 的 room caller allocator 记录：

| Backend | allocated before bytes | allocated peak bytes | incremental peak bytes |
|---|---:|---:|---:|
'''
for r in rows('caller_benchmark.csv.memory.csv'):
    if r['stage']=='current_caller_structural':
        section += f"| {r['backend']} | {r['allocated_before']} | {r['allocated_peak']} | {r['incremental_peak']} |\n"
section += '\n以上包含 fixture 和既有 handle/workspace，前置分配不同；reserved peak 受同进程缓存复用影响，不是独立进程峰值。网格每阶段另存 `.memory.csv`。\n\n'
text=report.read_text(encoding='utf-8')
start,end=text.index('## Selector caller 性能'),text.index('## 固定地图质量')
text=text[:start]+section+text[end:]
text=text.replace('1.408×','1.426×').replace('1.381×','1.394×')
report.write_text(text,encoding='utf-8')
print('已更新最终 caller/NSight/显存数据，Gaussian gradient 保持启用。')
