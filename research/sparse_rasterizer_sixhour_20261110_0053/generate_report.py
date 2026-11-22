"""从最终原始统计和验收文件生成中文技术报告，不填写未运行结果。"""
from pathlib import Path
import datetime as dt
import json
import re
import pandas as pd

HERE=Path(__file__).resolve().parent


def read_json(path):return json.loads((HERE/path).read_text(encoding='utf-8'))


def table(frame):
    def val(value):
        if isinstance(value,float):return f'{value:.4f}'
        return str(value).replace('|','/')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    protocol=read_json('reports/PROTOCOL.json');result=read_json('FINAL_VERIFICATION.json')
    prep=read_json('PREPARATION_FINAL_VERIFICATION.json');q=read_json('CAMERA_GRAPH_FINAL_VERIFICATION.json')
    async_path=HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json'
    async_passed=async_path.exists() and json.loads(async_path.read_text())['passed']
    final_variant='R' if async_passed else 'Q'
    if async_passed:q=read_json('CAMERA_ASYNC_FINAL_VERIFICATION.json')
    integrity=read_json('reports/INTEGRITY_FINAL.json');trajectory=read_json('reports/trajectory01/SUMMARY.json')
    camera=read_json('reports/async_camera_trajectory01/SUMMARY.json' if async_passed else 'reports/camera_trajectory01/SUMMARY.json');cleanup=read_json('reports/CLEANUP_MANIFEST.json')
    assert result['passed_new_numerics'] and q['passed'] and integrity['passed']
    raster=pd.read_csv(HERE/'RASTERIZER_RESULTS.csv');pipe=pd.read_csv(HERE/('FINAL_ASYNC_PIPELINE_RESULTS.csv' if async_passed else 'FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv'))
    gates=pd.read_csv(HERE/'FULL_GATES.csv');q_gates=pd.read_csv(HERE/('FINAL_ASYNC_FULL_GATES.csv' if async_passed else 'FINAL_CAMERA_GRAPH_FULL_GATES.csv'))
    profile=pd.read_csv(HERE/'GPU_PROFILE.csv');environment=pd.read_csv(HERE/'GPU_ENVIRONMENT.csv')
    scenes=read_json('reports/SCENES.json');order=[s['id'] for s in scenes]
    config=read_json('reports/FROZEN_CANDIDATE.json');work=pd.read_csv(HERE/'TILE_WORKLOADS.csv')
    gains=[]
    for condition in ['local0.01','local0.05','local0.1']:
        r=raster[(raster.variant=='G')&(raster.condition==condition)];p=pipe[(pipe.variant==final_variant)&(pipe.condition==condition)]
        gains.append(dict(Ratio=condition[5:],实际K=f'{int(r.K_min.min())}–{int(r.K_max.max())}',渲染相对A=f'{r.speedup_A.min():.2f}–{r.speedup_A.max():.2f}×',
            渲染相对B=f'{r.speedup_B.min():.2f}–{r.speedup_B.max():.2f}×',
            总延迟相对A=f'{p.speedup_A.min():.2f}–{p.speedup_A.max():.2f}×',
            总延迟相对B=f'{p.speedup_B.min():.2f}–{p.speedup_B.max():.2f}×',
            总延迟ms=f'{p.median_ms.min():.4f}–{p.median_ms.max():.4f}'))
    fullr=raster[(raster.condition=='full')&(raster.variant=='G')].set_index('scene').loc[order].reset_index()
    fullp=pipe[(pipe.condition=='full')&(pipe.variant==final_variant)].set_index('scene').loc[order].reset_index()
    rtable=fullr[['scene','N','A_median_ms','B_median_ms','median_ms','p95_ms','speedup_B','repeat_cv_pct']].copy()
    rtable.columns=['场景/模式','地图N','A ms','B ms','G ms','G p95 ms','G/B加速','重复CV%']
    ptable=fullp[['scene','A_median_ms','B_median_ms','G_median_ms','F_median_ms','median_ms','p95_ms','speedup_B']].copy()
    ptable.columns=['场景/模式','A总ms','B总ms','G总ms','F总ms','Q总ms','Q p95 ms','Q/B加速']
    scaling=pd.read_csv(HERE/'COUNT_VS_SPEED_PER_VIEW.csv');scaling=scaling[(scaling.scope=='raster')&scaling.condition.isin(['local0.01','local0.05','local0.1'])]
    q_scaling=pd.read_csv(HERE/('FINAL_ASYNC_COUNT_VS_FULL.csv' if async_passed else 'FINAL_CAMERA_GRAPH_COUNT_VS_FULL.csv'));q_scaling=q_scaling[q_scaling.condition.isin(['local0.01','local0.05','local0.1'])]
    scaling_rows=[]
    for condition in ['local0.01','local0.05','local0.1']:
        a=scaling[scaling.condition==condition];b=q_scaling[q_scaling.condition==condition]
        scaling_rows.append(dict(Ratio=condition[5:],测试视角=len(a),渲染快于Full=int((a.speedup_vs_own_full>=1).sum()),
            渲染加速中位数=a.speedup_vs_own_full.median(),渲染最小加速=a.speedup_vs_own_full.min(),
            Q总延迟快于Full=int((b.speedup_vs_own_full>=1).sum()),Q总延迟最小加速=b.speedup_vs_own_full.min()))
    footprint=pd.read_csv(HERE/'COMPENSATION_FOOTPRINT_SUMMARY.csv');footprint=footprint[footprint.condition!='full']
    footprint=footprint[['condition','paired_views','median_tile_factor','median_alpha_factor','median_latency_factor']]
    footprint.columns=['条件','相同身份对照视角','Tile实例倍数','Alpha交互倍数','渲染耗时倍数']
    lowprofiles=profile[(profile.scope=='raster')&profile.condition.isin(['local0.01','full'])][['variant','condition','kernel_count_per_frame','kernel_us_per_frame','host_kernel_launches_per_frame','graph_launches_per_frame','memcpy_calls_per_frame']].copy()
    lowprofiles.columns=['版本','条件','GPU核/帧','GPU核累计μs','Host Kernel发射/帧','Graph发射/帧','Memcpy API/帧']
    pipeline_profiles=profile[(profile.scope=='pipeline')&profile.variant.isin(['G','Q','R'])][
        ['variant','kernel_count_per_frame','kernel_us_per_frame','host_kernel_launches_per_frame','graph_launches_per_frame','stream_syncs_per_frame']].copy()
    pipeline_profiles.columns=['版本','GPU核/帧','GPU核累计μs','Host Kernel发射/帧','Graph发射/帧','Stream等待/帧']
    start=dt.datetime.fromisoformat(protocol['start_utc'].replace('Z','+00:00'));end=dt.datetime.fromtimestamp(integrity['checked_at'],dt.timezone.utc)
    elapsed=(end-start).total_seconds()/3600
    holdout=pd.read_csv(HERE/'HELDOUT_GENERALIZATION.csv');holdout=holdout[holdout.split=='heldout']
    work_low=work[work.condition=='local0.01']
    alpha_full=work[work.condition=='full']
    renderer_checks=result['new_parity_inputs'];bad_b=pd.read_csv(HERE/'BASELINE_B_NUMERICAL_DIFFERENCES.csv')
    graph_memory=pd.read_csv(HERE/('CAMERA_ASYNC_MEMORY.csv' if async_passed else 'CAMERA_GRAPH_MEMORY.csv'))
    graph_memory_text=(f"模型和原矩阵库已加载条件下，首次相机图构建{graph_memory.first_graph_camera_ms.min():.3f}–{graph_memory.first_graph_camera_ms.max():.3f}ms；"
        f"逻辑常驻Tensor最大{graph_memory.persistent_tensor_bytes.max():.0f}B，"
        f"allocator实测allocated增量最大{graph_memory.allocated_delta.max()/2**20:.3f}MiB、reserved增量最大{graph_memory.reserved_delta.max()/2**20:.3f}MiB。")
    report=f'''# mycartgs 低点数渲染：六小时 CUDA 优化研究报告

## 结论与交付状态

本轮保留已通过严格数值验证的 CUDA 渲染路径 G，以及最终相机准备组合 Q。全部 **24 个场景/相机模式**覆盖 Replica 全16组、TUM4组和VECtor4组；每项最终配置在5组调优后冻结，19组保留验证没有逐场景调参。

**Full逐场景两轮中位延迟均不退化**：G纯渲染相对A为{fullr.speedup_A.min():.2f}–{fullr.speedup_A.max():.2f}×，相对上一版B为{fullr.speedup_B.min():.2f}–{fullr.speedup_B.max():.2f}×；Q完整相机＋准备＋渲染相对B为{fullp.speedup_B.min():.2f}–{fullp.speedup_B.max():.2f}×。门限逐场景、逐轮验证，没有用平均收益掩盖反例。

{table(pd.DataFrame(gains))}

表中“相对B”是本轮新增收益；“相对A”包含上一轮已有收益，不能全部归给本轮。纯Rasterizer与完整延迟分别使用各自同一批输入的配对对照，没有跨批次相除。

正确性方面，原始A是唯一数值oracle。全轨迹渲染审计完成 **{trajectory['comparisons']:,}** 项RGB/radii逐位对照；新相机路径完成 **{camera['full_trajectory_camera_frames']:,}** 帧、**{camera['camera_tensor_comparisons']:,}** 项相机Tensor逐位对照，另对 **{camera['paired_selection_rgb_radii_inputs']:,}** 份输入核对选点身份与图像。Compute Sanitizer的最终渲染memcheck/initcheck/synccheck及相机memcheck均为0错误。源码、原模型和生产完整性保护 **{integrity['protected_files']} 项变更0项**。

本轮没有切换生产默认，没有训练、commit、push或部署。独立研究目录为 `/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053`。实际墙钟从{start.isoformat()}到{end.isoformat()}，约 **{elapsed:.2f}小时**，六小时预算截止{protocol['deadline_utc']}。没有通过重复无信息测试填满预算。

## 对照身份、评价口径和限制

- **A**：本轮开始时实际原推理二进制，与运行Viewer的原Rasterizer SHA一致：`{protocol['baseline_A']['sha256']}`。
- **B**：上一轮已验收的balanced mode4，二进制只读：`{protocol['baseline_B']['sha256']}`。本轮发现其少数Tile coverage/负零/空集边界不满足A语义，所有数值差异单列；仍保留B作为历史速度对照，未修改它。
- **C**：本轮普通CUDA路径；所有最终数值与原A一致，作为完整回退版本。
- **G**：C的有界容量执行图；缓存执行计划，不缓存任何视角、属性、Feature、可见性或图像。
- **F**：G＋固定内参投影矩阵预计算，64B；每帧仍按原方式求位姿/BMM/inverse。
- **Q**：G＋固定内参＋相机BMM/inverse执行图；每帧保持原求逆错误检查、显式输入更新和独立输出。最后推荐的是Q，F/G/C均保留独立使用方式。

所有版本同checkpoint、相机、Gaussian身份、补偿属性、背景、分辨率、SH阶数和FP32。主测试覆盖局部Ratio .01/.03/.05/.08/.1/.3/.7、全局 .01/.05/.1、固定0/1000/5000及Full；未见Ratio与固定预算均未使用专用参数。输入地图N范围为{result['input_N_range'][0]:,}–{result['input_N_range'][1]:,}。每场景使用连续运动12视角及6个跳转视角，正式准备每次重新计算CPU位姿、GPU相机、候选、Selector/Compensator；实际K与候选池按场景和视角变化，没有same-K跨场景比较。

这里的完整延迟包含相机输入构造、网络准备和渲染，不含数据集图片解码、HTTP传输、Viewer显示、存盘、在线Tracking/Mapping。测试是冻结端侧推理路径在RTX3090上的验证，不能称完整在线SLAM加速，也不能外推为所有端侧GPU的毫秒数。相机BMM/inverse执行图保持原FP32算子，没有改用数学上等价但可能舍入不同的刚体逆公式。

## 保留实现及其作用

### 1. 共享属性、像素独立链和稳定排序

每Tile的有序成员及Gaussian投影属性在共享内存装入一次，供全部像素复用。每线程处理两个独立像素，使属性读取服务两条独立T/C链；各像素内部Gaussian、Alpha和颜色累加顺序保持不变。256项短列表溢出时完整走有序目录，不截断、不丢点。此前已有的Bitset与目录机制不重复算作新增创新。

深度键直接在预处理写出，省一次独立depthKeys Kernel。相机35个标量按原stride读入共享内存，省逐帧相机contiguous拷贝。稳定排序使用512项Block整数排序和通用归并；全深度键和原输入ID共同保持同深度次序。所有规模均使用同一排序算法，没有按K/Ratio把Full绕开。

### 2. footprint任务并行

旧版低K且大footprint时，仅少量Gaussian线程承担大量Tile测试，SM并行不足。新路径先处理小footprint，将大覆盖拆为每任务256个Tile，使用GPU任务计数和固定数量workers。分派依据真实footprint，所有Ratio/Full走同一流程；位表使用整数OR，不引入浮点归约或排序重排。

Tile覆盖测试用原二进制SASS定位的FP32舍入顺序：`__fmul_rn`、`__fadd_rn`、`__fmaf_rn`固定乘法和FMA方向。这是修复旧路径的真实数值错误，不是设容差接受误差。后续直接核对原二进制完整Tile成员，而不只看PSNR。

### 3. 有界执行图和空间换时间

G按512点容量粒度预编译计划，模型加载时仅用地图容量和W/H，不预看任何视角。每帧GPU更新当前输入、相机、背景和输出指针，所有几何/SH/网络输出重新读取；padding条目只初始化为无效，不参与选择、排序有效身份或合成。图内显式清零位表/计数，避免旧Graph清零路径的initcheck问题。

计划数量最多512，共享Workspace上限1GiB、Bitset上限256MiB；容量或分辨率改变立即销毁旧地址计划并重建，单实例绑定一个device/Stream。真实成员数溢出完整枚举，Workspace超限明确报错，不静默换算法。输出每帧独立持有，不受下一帧覆盖。正式测试和4000次运动压力测试没有计划miss。

最大实测持久Workspace：B {result['max_workspace_MiB']['B']:.2f}MiB，C {result['max_workspace_MiB']['C']:.2f}MiB，G {result['max_workspace_MiB']['G']:.2f}MiB。G比B增加约{result['max_workspace_MiB']['G']-result['max_workspace_MiB']['B']:.2f}MiB。GPU输出/实际应用峰值另见MEMORY_RESULTS.csv，不能把Workspace当整个应用显存。加载时所有容量图预编译为{result['graph_precompile_ms_range'][0]:.3f}–{result['graph_precompile_ms_range'][1]:.3f}ms，单独计入startup CSV，未伪装成零启动成本。

### 4. 相机固定计算和执行图

F消除固定投影矩阵每帧CPU构造、标量写入及上传，内参/分辨率/near/far/device/Stream/dtype改变即失效，每实例只保留一个64B矩阵。Q进一步捕获原BMM和`linalg_inv_ex`，每帧更新同一4×4输入缓冲后执行。之后仍立即调用原`aten._linalg_check_errors`，保留奇异矩阵报错与必要同步；图外clone完整投影与相机中心，保留输出独立性。

Q不是缓存CPU/GPU位姿或上帧结果。测试包含同地址数据变化、近病态仿射矩阵、原错误消息逐字相同、Stream更换和所有旧输出存活。{graph_memory_text}相机图额外显存、首次建图与allocator reserved增量详见CAMERA_GRAPH_MEMORY.csv；逻辑常驻Tensor有界，PyTorch图私有池的保留块可能大于这些Tensor本身。Q相对F每帧额外节省约40–51μs，首次约29–34ms建图需约600–850帧摊销；短会话可以直接使用已验收F，不能只报热态收益而隐藏启动成本。

## Full全部24组结果

两轮各自的硬门限见FULL_GATES.csv和FINAL_CAMERA_GRAPH_FULL_GATES.csv；下表为合并两轮的中位、p95及重复波动，不能替代逐轮门限。重复CV为每个重复内中位延迟的CV，p95还包含不同视角负载差异。

{table(rtable)}

完整相机＋准备＋渲染使用最终Q独立配对批次：

{table(ptable)}

## 少点是否真的更快

同视角、同分辨率下，三个低Ratio的纯Rasterizer在全部 **{len(scaling):,}** 项视角/Ratio中快于G自己的Full；因此本轮有跨场景证据支持“减少Gaussian可提高纯渲染速度”。但减少倍数和提速倍数不成比例，也不是仅由K决定。

{table(pd.DataFrame(scaling_rows))}

Q完整延迟相对自己的Full仍有 **{q['per_view_regressions_vs_own_full']} / {len(q_scaling)}** 项反例，最小加速为{q['worst_speedup_vs_own_full']:.4f}×。原因是Full直接读取Master，而局部LoD额外支付候选、Selector和Compensator成本，不能把“纯渲染更快”写成“所有视角完整推理必然更快”。全部反例见FINAL_CAMERA_GRAPH_COUNT_VS_FULL.csv，未删点或放宽门限。

1%输入的活跃Tile比例中位为{(work_low.active_tiles/work_low.tiles).median()*100:.2f}%，并非大多数Tile为空。低K补偿Gaussian往往覆盖大范围屏幕，像素写出和Alpha链仍存在。全像素实际Alpha参与点、交互量、预处理radii可见点、Tile实例、成员p50/p95/max都在计时外采集，见TILE_WORKLOADS.csv。

为了隔离补偿footprint，用同一Selector选中身份和K，单独将属性替换成对应Master属性做归因。该对照不改变最终推理算法，也不把质量变化混入优化结果：

{table(footprint)}

例如1%时，补偿使Tile实例和Alpha交互中位约放大2.43/2.53倍，而不是Selector选了更多点。因此仅按K估计时间会低估工作量。原结果、相同身份SHA及所有144个配对视角/条件均在SAME_ID_COMPENSATION_CONTROL.csv中。

## Kernel、同步和测量方法

正式数值与墙钟来自独立配对轮次。版本运行顺序轮转并反向，逐次同步，CUDA Event另测，诊断在计时外。首批渲染/准备/组合两轮共{result['raw_rows']:,}条样本，其中纯Rasterizer {int(raster.samples.sum()):,}条；固定内参两轮{prep['raw_rows']:,}条；Q最终两轮{q['raw_rows']:,}条，其中完整准备＋渲染{int(pipe.samples.sum()):,}条，其余为准备单测。各批次分别可复现，既有未变代码的纯渲染没有重复重跑。

Nsight Systems CUDA/NVTX/Graph节点证据（20调用/案例）如下。Profiler期间的Kernel累计不能与同步墙钟或CUDA Event混用；CPU API发射/同步时间也不等于GPU执行时间。

{table(lowprofiles)}

完整输入准备的独立Profile：

{table(pipeline_profiles)}

相机执行图可能增加少量GPU拷贝核，但减少Python/Host发射和Stream等待，不能将Host调度减少写成GPU核数量必然减少。最终异步上传候选的Stream等待下降由实际API记录验证，GPU计算仍保持原数学和输出。

G比C可能多两个很短的GPU核，但Host端只需输入更新＋一次Graph发射，完整调用更快。因此没有将“减少Host Launch”误写成“GPU Kernel数一概更少”。Q保留求逆info检查同步；没有跳过finite检查、把必要检查延后到下一帧或变更数据依赖。

profile_suite01发生Xorg活动，整批耗时不用，完整重跑profile_suite02；formal03亦整轮无效，重跑formal04。所有原始失败证据保留。1Hz监测记录外部SM活动、时钟、温度、功耗和进程身份。正式轮次无外部活跃GPU样本；时钟/温度范围在GPU_ENVIRONMENT.csv。NCU计数器受权限限制，错误日志保留，未修改驱动全局权限。Kineto对Graph捕获曾发生无进展，只停止已核验PID/start_ticks/boot_id的本轮进程，Graph后改用Nsight。

## 正确性和旧B缺陷

- 最终renderer的正式两轮{renderer_checks:,}项C/G输入RGB、radii逐bit一致；每帧重新准备后IDs和图像也核对。
- 24场景完整轨迹共{sum(s['trajectory_frames'] for s in trajectory['scenes']):,}帧，Full/1%/5%/10%、C/G两路径合计{trajectory['comparisons']:,}项逐位通过。
- 343份输入直接解析原二进制sorted Tile/depth/ID缓冲，重建全部Tile位表，无缺失/多余成员；原training/inference oracle图像亦逐位相同。
- 合成边界覆盖0/1/31/32/33、255/256/257、511/512/513、1023/1024/1025、4097等K；同深度、极薄/巨大Scale、全部不可见、无Alpha、负零、NaN/Inf、预计算颜色/协方差、非整Tile分辨率、非连续Tensor和不同Stream。
- burst测试先完成全部A oracle，再连续提交候选，仅末尾同步，覆盖Workspace复用、输出独立、同地址属性值变化和255/256/257 Tile任务边界。动态扩容/分辨率变化另有独立检查，不以oracle隐式同步掩盖竞争。
- 相机Q {camera['camera_tensor_comparisons']:,}项全轨迹Tensor逐位、{camera['paired_selection_rgb_radii_inputs']:,}份图像/IDs对照；两轮正式{q['correctness_inputs']:,}份选点/相机/图像输入一致。奇异矩阵错误类型、失败对角和消息均相同。
- Compute Sanitizer日志、退出码和过程身份同时保留；renderer memcheck/initcheck/synccheck、相机memcheck全部0错误。没有因为原版存在Atomic非确定性而放行本轮差异。

旧B在本轮正式集有{len(bad_b)}行（两轮、3个独立非空输入）RGB误差，最大abs为{bad_b.max_absolute_error.max():.10f}。这些差异不是允许容差，而是按原SASS定位的Tile覆盖计算融合方向不同。本轮修复后重做正确性、两轮测速和Sanitizer，旧版本保持只读。formal01和旧冻结v1作为失败保留，不用于最终性能结论。

空集还存在A产品入口返回背景、B旧primitive返回零图的语义区别；因此fixed0新方案可能比B更慢，不能声称所有负载相对B都加速，也不能直接把该空集时延当同图像性能。全部fixed0原始数值和计时保留，新路径遵循A背景语义。

## 失败方案、剩余瓶颈与后续边界

失败候选详见FAILURE_CANDIDATES.md，包含Warp shuffle、Alpha流水线、4像素/线程、metadata重用、Tile枚举渲染融合、radix参数、预处理排序融合、64/128属性缓存、每Gaussian一个CTA以及8/4行子Tile。没有根据保留场景结果给某场景定制参数。不能把减少Kernel但实际变慢的候选并入最终路径。

剩余纯渲染热点是像素Alpha混合、Tile成员压缩和稳定归并。低Ratio中大footprint已经占据多数屏幕，不存在仅靠“跳过大量空Tile”获得任意倍数加速的前提。继续提高需要按真实Tile负载改善属性读取/可用并发，同时逐位验证FMA和合成顺序。完整输入准备仍有原求逆info同步和Selector/Compensator开销；这些检查不能为速度删除。

当前证据仅覆盖已列地图规模、轨迹和原始分辨率，不预设所有相机/地图都成立。不支持将低点数提速倍数线性外推至更大地图、不同GPU或任意投影。真正收益与其边界一起保留。

## 完整性、磁盘和复现

保护清单前后SHA见PROTECTED_BEFORE/AFTER：{integrity['protected_files']}项生产源码、配置、二进制、旧研究和原checkpoint均未改变；最终冻结源码/二进制SHA亦一致，研究目录外未写入训练配置。无未结束的本轮实验进程，外部进程不操作。没有渲染图片，只有性能曲线。

最终普通C二进制SHA：`{config['binaries']['regular']['sha256']}`；G二进制SHA：`{config['binaries']['graph']['sha256']}`。实际加载库和CUDA/LibTorch运行身份随报告留证，不以裸ldd猜测依赖。`frozen_renderer.py`只读验证并加载.so，不触发JIT；`use_graph=False`回到完整C路径。生产原入口及B文件保持原位，未部署。

本轮只清理本目录已确认可重建的object/SQLite，无损gzip长重复诊断；清单{len(cleanup['files'])}项，释放约{cleanup['reclaimed_bytes']/2**20:.2f}MiB。保留最终源码、两.so、配置/哈希、模型路径、逐样本CSV、Sanitizer、少量代表性Profile及失败证据；不复制数据集和大模型，不清理共享缓存或其他任务。

主要文件：RASTERIZER_RESULTS.csv、FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv、FULL_GATES.csv、FINAL_CAMERA_GRAPH_FULL_GATES.csv、TILE_WORKLOADS.csv、GPU_PROFILE.csv、GPU_ENVIRONMENT.csv、MEMORY_RESULTS.csv、CAMERA_GRAPH_MEMORY.csv及最终交付包。全部24组曲线入口：`figures/FINAL_SCENE_INDEX.md`，同时输出450DPI PNG和SVG，线性Ratio坐标标注实际K，无合成或平滑数据。
'''
    if async_passed:
        report=re.sub(r'(?<![A-Za-z])Q(?![A-Za-z])','R',report).replace('FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv','FINAL_ASYNC_PIPELINE_RESULTS.csv').replace('FINAL_CAMERA_GRAPH_FULL_GATES.csv','FINAL_ASYNC_FULL_GATES.csv').replace('FINAL_CAMERA_GRAPH_COUNT_VS_FULL.csv','FINAL_ASYNC_COUNT_VS_FULL.csv').replace('CAMERA_GRAPH_MEMORY.csv','CAMERA_ASYNC_MEMORY.csv')
        # Profile同时保留Q和R的原始标签，不能随主体最终方案改名。
        original_profile_table=table(pipeline_profiles)
        report=report.replace(original_profile_table.replace('Q','R'),original_profile_table)
        low=pipe[(pipe.variant=='R')&pipe.condition.isin(['local0.01','local0.05','local0.1'])]
        added_min=low.speedup_Q.min();added_max=low.speedup_Q.max()
        delta_f=(low.F_median_ms-low.median_ms)
        amortize_low=graph_memory.first_graph_camera_ms.min()/delta_f.max()
        amortize_high=graph_memory.first_graph_camera_ms.max()/delta_f.min()
        report=report.replace('相对F每帧额外节省约40–51μs，首次约29–34ms建图需约600–850帧摊销',
            f'相对F在低Ratio每帧额外节省{delta_f.min()*1000:.1f}–{delta_f.max()*1000:.1f}μs，首次建图需约{amortize_low:.0f}–{amortize_high:.0f}帧摊销')
        extra=('\n\n最终R在已验收Q基础上，将相机位姿与固定投影上传设置为`non_blocking=True`，删除H2D后冗余Host等待；'
            '复制、相机图和渲染仍在同一Stream串行依赖，原求逆info检查每帧立即执行。没有延后必要检查或缓存位姿。'
            f'Q的原始实现和两轮证据保留。R相对Q低Ratio新增加速为{added_min:.4f}–{added_max:.4f}×，不将此前CUDA/相机图收益重复计入本候选；'
            '最终门限还逐场景两轮要求R Full不慢于Q。数值、Sanitizer、全轨迹和最终Profile均单独复验。\n\n')
        extra+=(f"R与Q逐场景/条件/轮次共有{q['per_round_regressions_vs_Q']}项轻微负收益，最大耗时增加{q['worst_regression_vs_Q_pct']:.3f}%；"
            '未隐藏高Ratio/全局导出的反例，见ASYNC_VS_Q_REGRESSIONS.csv。Full门限仍独立逐项通过。\n\n')
        report=report.replace('## 对照身份、评价口径和限制',extra+'## 对照身份、评价口径和限制',1)
        report=report.replace('最后推荐的是R，F/G/C均保留独立使用方式。','最后推荐的是R，Q/F/G/C均保留独立使用方式。')
    (HERE/'LOW_POINT_RENDERING_SIX_HOUR_REPORT.md').write_text(report,encoding='utf-8')
    print('REPORT_GENERATED',len(report),elapsed)


if __name__=='__main__':main()
