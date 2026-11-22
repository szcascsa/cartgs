from pathlib import Path
import csv
import json
import statistics


STUDY=Path(__file__).resolve().parent
REPORTS=STUDY/'results/reports'


def read(name):
    return json.loads((REPORTS/name).read_text(encoding='utf-8'))


def main():
    identity=read('BINARY_IDENTITY.json')
    assert 'v21' in identity['binary']['path']
    integrity=read('FINAL_INTEGRITY.json')
    assert integrity['passed']
    measured={}
    csv_rows=[]
    for repeat in [1,2,3]:
        for row in read(f'BENCHMARK_{repeat}.json')['results']:
            measured.setdefault((row['group'],row['condition']),[]).append(row)
            before=row['summary']['v13'];after=row['summary']['optimized']
            csv_rows.append(dict(group=row['group'],condition=row['condition'],repeat=repeat,
                original_prepare_ms=before['prepare_ms']['median'],optimized_prepare_ms=after['prepare_ms']['median'],
                original_total_ms=before['total_ms']['median'],optimized_total_ms=after['total_ms']['median'],
                original_p90_ms=before['prepare_ms']['p90'],optimized_p90_ms=after['prepare_ms']['p90'],
                original_p99_ms=before['prepare_ms']['p99'],optimized_p99_ms=after['prepare_ms']['p99'],
                N_min=min(x['N'] for x in row['actual']),N_max=max(x['N'] for x in row['actual']),
                K_min=min(x['K'] for x in row['actual']),K_max=max(x['K'] for x in row['actual'])))
    with (STUDY/'DEPLOYMENT_PERFORMANCE.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    table=[]
    for (group,condition),rows in measured.items():
        assert len(rows)==3
        before,after=[statistics.median(x['summary'][tag]['prepare_ms']['median'] for x in rows) for tag in ['v13','optimized']]
        original_total,optimized_total=[statistics.median(x['summary'][tag]['total_ms']['median'] for x in rows) for tag in ['v13','optimized']]
        table.append(f'| {group} | {condition} | {before:.4f} | {after:.4f} | {100*(1-after/before):+.1f}% | {original_total:.4f} | {optimized_total:.4f} | {100*(1-optimized_total/original_total):+.1f}% |')
    profiles={x['label']:x for x in read('PREPARE_PROFILE_ANALYSIS.json')['summary']}
    profile_table=[]
    for group in ['tum','replica']:
        for condition in ['1000','5000','70pct']:
            a,b=[profiles[f'prepare/{tag}/{group}/{condition}'] for tag in ['v13','optimized']]
            profile_table.append(f"| {group}/{condition} | {a['cpu_launches']:g}→{b['cpu_launches']:g} | {a['kernel_nodes']:g}→{b['kernel_nodes']:g} | {a['kernel_us']:.1f}→{b['kernel_us']:.1f} | {a['syncs']:g}→{b['syncs']:g} | {a['dtoh_bytes']:g}→{b['dtoh_bytes']:g} |")
    memory_table=[]
    for group in ['tum','replica']:
        a,b=[read(f'MEMORY_{group}_{tag}.json') for tag in ['v13','optimized']]
        memory_table.append(f"| {group} | {a['final_extra_bytes']/2**20:.2f} | {b['final_extra_bytes']/2**20:.2f} | {(b['final_extra_bytes']-a['final_extra_bytes'])/2**20:+.2f} | {a['process_delta_mib']}→{b['process_delta_mib']} | {a['cold_ms']:.2f}→{b['cold_ms']:.2f} |")
    validation=read('VALIDATION.json')
    tum,replica=[read(f'TRAJECTORY_VALIDATION_{group}.json') for group in ['tum','replica']]
    boundaries=read('GRAPH_BOUNDARIES.json')
    host=read('HOST_CAPACITY_VALIDATION.json')
    workspace=read('GRAPH_WORKSPACE.json')
    workspace_lines=[]
    for row in workspace['results']:
        first=row['checkpoints'][0];last=row['checkpoints'][-1]
        workspace_lines.append(f"- {row['group']} 覆盖 {last['graph_count']} 个执行计划，四轮末尾分配量均为 {last['retained']/2**20:.2f} MiB，缓存池保留量为 {last['reserved']/2**20:.2f} MiB。")
        assert first['retained']==last['retained']
    protection=[]
    for group in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        row=integrity[group];protection.append(f"| {group} | {row['files']} | {len(row['changed'])} |")
    document='''# 部署推理四方向 CUDA 优化报告

日期为 2026-11-07。本轮以已验收 v13 为比较起点，最终候选为 v21。数值结论限于这两个冻结部署后端。正常训练和常规评价入口的完整性另行核验。

## 1. 四个方向的结果

| 方向 | 实际实现与测试 | 最终处理 |
|---|---|---|
| P0 有界 Workspace 与准备 CUDA Graph | 动态相机和请求打包、有限容量计划、设备端 N/K、独立输出直接发布、零 RNG Replay、普通与 Graph Kernel 分别编译 | 保留 |
| P1 精确选点 | K 等于全部有资格候选时跳过阈值细化，沿原顺序稳定发布所有候选 | 保留 |
| P2 按实际 K 分流 Compensator | 分别实现双 Kernel 分流和单 Kernel 内分流，逐位检查与配对测速 | 两种实现均因耗时增加而回滚 |
| P3 请求内 Anchor 共享投影 | 标记实际使用 Anchor，残差累加后共用第一层 Feature 投影，保留各 Gaussian 的 Offset 运算 | 因额外准备成本超过节省而回滚 |

P0 与 P1 已进入独立部署入口。计算仍使用 FP32，选点保持严格 K、原资格 Mask、固定 Gumbel 键、同分优先级及原点顺序。Renderer 的源码与二进制完整性核验通过。

## 2. 模型、范围和测量口径

| 实际模型 | 全图 Gaussian 数量 | Anchor 数量 | 轨迹范围 |
|---|---|---|---|
| fr1_xyz，both_li | 13298 | 2790 | 全轨迹 792 帧 |
| Replica office3 RGB-D | 49102 | 19936 | checkpoint 保存的 137 个视角 |

服务器实验根目录为 `/home/prmi/zwj/experiments/deployment_graph_adaptive_20261107`，source、build、reports、evaluation 和 logs 均在此目录内。

测量使用 RTX 3090、CUDA 11.8 编译与分析工具，以及现有 zwj 中的 PyTorch 2.0.1+cu117，驱动为 535.230.02，TF32 关闭。每种条件先预热，再测 120 个跳转视角，请求顺序和 A/B 执行先后轮换。正式版本独立重复三轮，所有轮次保留在 CSV 和原始 JSON 中。

准备计时覆盖已有相机 Tensor 进入部署入口后的视锥、候选整理、Selector、精确选点、Compensator、属性发布及控制记录读回。完整推理计时继续执行原 Rasterizer。相机矩阵构建、窗口显示和图片编码另计。

服务器存在其他 GPU 作业。部分轮次出现两版同时增加约 2 ms 的排队延迟，原始数据全部保留。表中先分别取三轮中位数的中位数，再计算耗时变化。正百分比表示耗时减少。Nsight 下的墙钟时间独立记录。

## 3. 保留实现

### P0

- 使用固定相机 Buffer 和设备请求记录。每次请求更新相机矩阵、中心、预算、ratio、aspect、near 和独立输出地址，随后重新执行全部准备工作。
- 容量按 1024、2048、4096、8192、16384、32768 和全图上界分档，再按视角模式及补偿开关区分。计划数量具有固定上限。启用补偿且请求容量上界大于 32768 时，在统一入口使用原生调度。
- Gaussian、Anchor、权重和固定选择键是模型冻结快照。视锥 Mask、ratio Feature、选择集合与网络输出每次重新计算。
- 补偿 Kernel 直接写入本次请求独立分配的属性和 ID Buffer，清除或重放执行计划后输出仍可使用。普通与 Graph 发布分别通过编译期参数实例化，普通路径消除动态地址分支。
- 捕获过程保留 LibTorch 内存池管理。Replay 检查捕获 RNG 增量为零，然后直接调用 CUDA Graph Launch。捕获资源统一创建为普通 Tensor，支持 inference_mode 与 no_grad 混用。
- 调整执行开关会清除计划。切换 checkpoint 或地图时创建新实例。同一实例按串行请求使用。

### P1

当实际 K 等于全部有资格候选时，任何有限 Selector 分数都会得到同一个保留集合。此时直接设置完整保留阈值，沿用原计数和稳定发布 Kernel，省去多轮 Radix 细化。Selector 及非有限值检查仍正常执行。普通部分选择沿用原精确算法。

## 4. 数值与边界验证

{VALIDATION}

比较覆盖 1%–70% 与未见 ratio、ratio 为 0/1、固定 0/1/1000/2000/5000/10000 点、超额预算、两网络视角模式及补偿开关。附加测试覆盖容量边界、不同矩阵 stride、near/aspect 变化、非默认 Stream、输出所有权、RNG、源模型修改隔离、NaN 报错及下一请求恢复。

像素和 radii 在上述轨迹范围内逐位一致。PSNR、SSIM 和 LPIPS 的图像输入完全相同。Replica 的范围为保存视角，完整离线轨迹评价属于另一项验证。

## 5. 真实配对测速

所有时间单位为 ms。完整推理包含相同的原 Rasterizer。

| 场景 | 条件 | 原准备 | 新准备 | 准备减少 | 原完整推理 | 新完整推理 | 完整减少 |
|---|---|---|---|---|---|---|---|
{PERFORMANCE}

各条件的实际候选 N、实际 K 范围、各轮 p90/p99 与原始样本见 DEPLOYMENT_PERFORMANCE.csv 和 BENCHMARK_1/2/3.json。mixed 在每次请求轮换固定预算与 ratio，同时跳转视角。

## 6. Nsight Systems 复核

每个条件采集 20 个已预热请求。NVTX 范围前完成同步，范围末尾包含部署内部同步，因此按时间戳归属 Graph 子节点。

| 场景与条件 | CPU Launch 前→后 | GPU Kernel 节点前→后 | 累计 GPU μs 前→后 | 同步次数前→后 | DtoH 字节前→后 |
|---|---|---|---|---|---|
{PROFILE}

Graph 的收益主要来自减少 Host 发射和重复调度。GPU 内部仍执行原来的数学步骤。原始 nsys-rep、sqlite 和逐 Kernel 明细保存在服务器 reports 中，完整 Rasterizer 保持独立计时。

## 7. 显存和冷启动

下表给出构造前至 400 次请求后的增量。Torch 列统计当前进程 Tensor 与 Workspace，进程列通过 nvidia-smi 单独读取当前 PID，包含驱动侧开销。冷启动以二进制已编译、模型加载后的首次 prepare 为起点，包含 CUDA 库初始化和计划捕获。

| 场景 | 原 Torch MiB | 新 Torch MiB | 增加 MiB | 原→新进程增量 MiB | 原→新冷启动 ms |
|---|---|---|---|---|---|
{MEMORY}

{WORKSPACE}

上述执行计划压力测试还验证了清除计划后的输出所有权。真实应用可以在模型加载后预热将使用的容量和开关组合。

## 8. 失败候选及修正

- P2 双 Kernel 方案增加了发射和无效 Block。单 Kernel 方案虽然减少一次发射，但合并两套计算占用更多资源。在 v16 匹配测试中，TUM 2000 点准备由约 0.227 ms 增至 0.258 ms，Replica 由约 0.242 ms 增至 0.273 ms。候选源码与结果保存在 stage_v14 和 stage_v16。
- P3 在 TUM 的 K=5926 条件下，组件计时约从 133.45 μs 增至 151.32 μs，Replica K=15229 约从 236.21 μs 增至 323.97 μs。低预算的附加准备成本更明显。第一层 Feature 只是网络的一部分，额外标记、投影和发射抵消了复用收益。
- P3 改变第一层的 FP32 分组顺序，出现微小属性和像素差异，已测样本最大像素绝对差约为 0.00483。9 个 TUM 条件中的最差 PSNR 变化约为 −0.0000024 dB，Replica 仅完成像素比较。速度失败后保留这些记录，质量结论限于已测样本。
- v17 暴露了 LibTorch Graph Replay 在 inference_mode 和 no_grad 混用时的内部 RNG Tensor 异常。最终实现用零 RNG 直接 Replay 与普通捕获资源解决，并通过恢复测试。
- v18 的大容量 Graph 存在空 Block 开销。v19/v20 的普通输出路径又受到动态输出地址分支影响。最终通过入口容量路由及分开编译普通/Graph 发布路径消除这些额外路径。阶段数据按版本保留。

## 9. 使用、回退与完整性

独立部署 API 为 DeploymentRenderer，默认 prepare_graph=True、saturated_selection=True。设置两项为 False 可进行保留路径对照。正常训练与常规评价入口的导入审计已完成，独立 CLI 通过 6 帧启动测试。

源码变更集中于 deployment_runtime.py、deployment_prepare.cpp、deployment_device_control.h、deployment_runtime_kernels.cu 和 deployment_radix.cu。SOURCE_MANIFEST.json、V13_TO_V21.patch 和 BINARY_IDENTITY.json 提供源码及二进制身份。

| 核验集合 | 文件数 | 变化文件数 |
|---|---|---|
{PROTECTION}

PRODUCTION 覆盖 `/home/prmi/zwj/mycartgs` 与 `/home/prmi/zwj/realLOD` 的关键源码、配置、二进制及动态库，其余三组覆盖 84 个模型输入文件。v13 基线二进制的 SHA256 为 `{BASELINE}`。

Rasterizer 二进制 SHA256 为 `{RASTERIZER}`，前后匹配。本地 HEAD 为 26c355b32bf373a75877fc4605a4e41b8de9b47f，保留原有未提交工作。本轮交付由独立实验源码、回退开关与审计清单组成。

最终部署二进制 SHA256 为 `{BINARY}`。

## 10. 局限与下一步

- 端到端速度受到共享 GPU 排队影响，当前数字应结合逐轮数据与同步等待阅读。独占设备上的实际端侧延迟仍值得单独复验。
- 大容量请求继续使用原生调度，容量超过 65536 的地图保持已有 Host 计数路径。98204 点边界测试属于合成容量验证。
- 32768 的 Graph 容量路由阈值来自当前 RTX 3090 和两个真实模型的测量，其他端侧硬件需要重新标定。
- 后续优先依据当前 Nsight 明细研究高 K Compensator 的权重访存和寄存器调度，以及低 K 精确选点的同步成本。Anchor 投影若继续研究，应先消除本轮测得的请求内准备开销，再做独立质量和速度对照。
- 当前 FrozenPreparationGraph 使用已检查的 LibTorch CUDAGraph 接口与受保护成员。升级 LibTorch/CUDA 时，应重跑编译、RNG、模式混用和数值回归。
'''
    counts=f'''| 验证 | 规模 | 结果 |
|---|---|---|
| 五类 checkpoint | {validation['cases']} 组，{validation['comparisons']} 项比较 | 全部逐位一致 |
| TUM 全轨迹 | {tum['frames']} 帧，{tum['cases']} 组，{tum['comparisons']} 项比较 | ID、属性、像素和 radii 逐位一致 |
| Replica 保存视角 | {replica['frames']} 视角，{replica['cases']} 组，{replica['comparisons']} 项比较 | ID、属性、像素和 radii 逐位一致 |
| 动态 Graph 边界 | {boundaries['comparisons']} 项比较 | 全部通过 |
| 大容量保留路径 | {host['N']} 点，{host['cases']} 组 | 全部通过 |
| 随机预算与连续 ratio | 两场景各 311 组 | 对 v13 的 ID 与属性零差异 |
| Compute Sanitizer | 两场景各 80 组 | 0 errors |'''
    replacements=dict(VALIDATION=counts,PERFORMANCE='\n'.join(table),PROFILE='\n'.join(profile_table),
        MEMORY='\n'.join(memory_table),WORKSPACE='\n'.join(workspace_lines),PROTECTION='\n'.join(protection),
        RASTERIZER=integrity['RASTERIZER']['sha256'],BINARY=identity['binary']['sha256'],BASELINE=identity['baseline']['sha256'])
    for key,value in replacements.items():document=document.replace('{'+key+'}',value)
    (STUDY/'DEPLOYMENT_FOUR_DIRECTION_OPTIMIZATION_REPORT.md').write_text(document,encoding='utf-8')
    print('REPORT_READY')


if __name__=='__main__':main()
