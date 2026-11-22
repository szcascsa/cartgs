from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics


ROOT=Path(__file__).resolve().parents[3]
REPORTS=ROOT/'reports'


def read(name):return json.loads((REPORTS/name).read_text())


def main():
    status=read('ACCEPTED_STATUS.json');assert status['complete']
    integrity=read('FINAL_INTEGRITY.json');assert integrity['passed']
    rounds=[read(f'BENCHMARK_{i}.json') for i in [14,15,16]]
    groups=defaultdict(list)
    for run in rounds:
        for row in run['results']:groups[(row['group'],row['condition'])].append(row)
    performance=[]
    for (group,condition),rows in groups.items():
        result=dict(scene=group,condition=condition,samples_per_version=390)
        for version in ['v27','optimized']:
            for metric in ['prepare_ms','total_ms','native_ms']:
                result[version+'_'+metric]=statistics.median(row['summary'][version][metric]['median'] for row in rows)
                for percentile in ['p90','p99']:
                    result[version+'_'+metric+'_'+percentile]=statistics.median(row['summary'][version][metric][percentile] for row in rows)
        for metric in ['prepare_ms','total_ms']:
            a,b=[result[v+'_'+metric] for v in ['v27','optimized']]
            result[metric+'_saved_us']=(a-b)*1000
            result[metric+'_reduction_percent']=(1-b/a)*100
        actual=[v for row in rows for v in row['actual']]
        result.update(candidate_N_min=min(v['N'] for v in actual),candidate_N_max=max(v['N'] for v in actual),
            K_min=min(v['K'] for v in actual),K_max=max(v['K'] for v in actual))
        performance.append(result)
    with (REPORTS/'DEPLOYMENT_STATIC_PREFIX_PERFORMANCE.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(performance[0]));writer.writeheader();writer.writerows(performance)
    save=dict(passed=True,baseline='v27',candidate='v30',rounds=[14,15,16],summary=performance)
    (REPORTS/'FINAL_PERFORMANCE.json').write_text(json.dumps(save,indent=2))
    text=['# 冻结部署三项 CUDA 优化报告','',
        '本轮完成 Selector 静态前缀、可见 Anchor 视角编码与小 K Compensator 专用 Tile。数值基准为已验收 v27，独立部署候选为 v30。最终推荐以空闲 RTX 3090 的三轮配对结果为依据。','',
        '实现位于本轮独立研究目录。正常训练、常规评价、生产默认及 Gaussian Rasterizer 的完整性见后文哈希核验。','',
        '## 实际运行性能','',
        '表中采用三轮中位数的中位数，每个条件、每个版本共 390 个有效请求。相机逐次跳转，变化预算条件交替使用固定 K 和 ratio。prepare 为 Python 部署 API 调用到同步结果返回的墙钟，包含原有 C++ 调度、内存分配、GPU 工作、96 字节读回与结果打包。total 为 prepare 加原始 Rasterizer 的同步墙钟，包含其正常动态 Tile 处理。HTTP、显示、文件导出及 SLAM 不在此计时范围内。','',
        '组件微测使用 CUDA Event，整帧主测试使用墙钟。最终主测试两版本共同预热两秒，执行次序按 ABBA 轮换，并保存 GPU 时钟、温度及 CUDA PID。历史带共享负载或事件创建开销的测试已按阶段归档。','',
        '|场景|请求|v27 准备 ms|v30 准备 ms|准备减少 %|v27 总 ms|v30 总 ms|K 范围|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    labels={'tum':'TUM fr1_xyz','replica':'Replica office3'}
    for r in performance:
        text.append(f"|{labels[r['scene']]}|{r['condition']}|{r['v27_prepare_ms']:.4f}|{r['optimized_prepare_ms']:.4f}|{r['prepare_ms_reduction_percent']:.2f}|{r['v27_total_ms']:.4f}|{r['optimized_total_ms']:.4f}|{r['K_min']}–{r['K_max']}|")
    text += ['', '正值表示耗时减少。收益按实际预算分别判断，不能将 K=1000 的收益外推到所有 ratio。1% 及高 ratio 的差值通常只有数微秒，部分条件出现轻微退化，当前没有稳定的普遍加速证据。原 Rasterizer 作为计时消费者保持相同，渲染本身的加速不计作本轮网络收益。','',
        '## 三项实现','',
        '1. Selector 读取冻结地图中每点固定的 10D 物理输入，在初始化时为 32 个首层神经元预计算前十项 FMA。缓存形状为 N×32，Bias 保留在完整仿射计算的原位置。运行时从缓存 FP32 值继续逐项累加 log-ratio 与视角输入，11D 和 19D 输入均保留原始顺序。前缀版本仅为动态输入分配共享内存，旧版完整输入路径由独立开关保留。','',
        '2. 每帧完成名义 Anchor 候选构建后，把本帧 Mask 传给视角编码 Kernel。不可见 Anchor 提前退出，可见 Anchor 重算全部 7D 视角值。零号 Anchor 的无效填充读取位置显式写零，从而保留大 K 计算 Kernel 的原有地址选择方式。每帧读取新相机和新 Mask，切换视角后重新编码。','',
        '3. 小 K 使用 8 Gaussian／CTA、256 线程、每线程两个点累加器的 Register Tile。每个神经元仍按原序执行 FP32 FMA，四个 Head 保持独立权重及输出。小 K 路径从请求容量上界 768 起启用，Graph 中只作用于原 1024 桶。更小容量沿用原单点 CTA，大容量保留原 16 点 Tile。所有决定使用已有 Host 请求上界，无新增 GPU 计数读回。','',
        'Graph 计划键增加 Tile 身份，避免同一个 1024 容量桶混用两种 Kernel。改变优化开关先清空旧计划。保持严格 K、固定噪声、资格 Mask、同分规则、原始行顺序和输出存储所有权。','',
        '三项独立开关为 selector_static_prefix、visible_view、small_compensation_tile，均在冻结部署入口中默认开启。前缀每个候选点减少 320 项 FMA，但首层静态值的逻辑读取量从 10 个增加到 32 个 FP32。这使前缀收益受到带宽与缓存限制，完整请求的收益小于减少的算术比例。后半句是基于算子与读取量的解释，硬件缓存命中率尚未取得计数权限。','',
        '## 空间换时间及生命周期','',
        '|场景|Gaussian 数|Anchor 数|前缀逻辑字节|前缀 MiB|初始化 GPU ms|',
        '|---|---:|---:|---:|---:|---:|']
    for r in read('SPACE_COST.json')['results']:
        text.append(f"|{labels[r['group']]}|{r['N']}|{r['A']}|{r['prefix_bytes']}|{r['prefix_bytes']/2**20:.3f}|{r['initialization_gpu_ms']:.4f}|")
    text += ['', '|场景|关闭三开关保留 MiB|开启三开关保留 MiB|增量 MiB|开启后峰值 MiB|',
        '|---|---:|---:|---:|---:|']
    for r in read('SPACE_COST.json')['results']:
        off,on=r['instances']
        text.append(f"|{labels[r['group']]}|{off['warm_bytes']/2**20:.3f}|{on['warm_bytes']/2**20:.3f}|{(on['warm_bytes']-off['warm_bytes'])/2**20:.3f}|{on['peak_extra_bytes']/2**20:.3f}|")
    text += ['', '该表使用同一 v30 实现，将三开关关闭与开启，依次创建 1、128、767、768、1000、1024、1025、5000 预算计划，包含冻结快照与计划工作区。显存值为 PyTorch 分配量，原始 SPACE_COST.json 同时保留每次捕获的首请求墙钟。']
    text += ['', '前缀容量严格为 128N 字节。模型实例持有独立冻结快照，切换 checkpoint、参数或拓扑时重新构造实例。关闭前缀会清空 Graph 并释放前缀 Tensor。它复用固定物理输入的投影，不包含相机、ratio 查询结果或 Anchor48 残差累加结果。','',
        '视角编码复用原有工作区。新增一个小 K 执行计划身份，验收按 32 个计划的保守上界检查，实测计划数量、保留显存和重复请求稳定性见 GRAPH_WORKSPACE.json。所有结果 Tensor 独立保存，下一请求和清空计划后仍保持有效。SPACE_COST.json 另记录首请求计划创建耗时以及完整实例的逻辑保留显存和分配峰值。','',
        '## 组件与独立消融','',
        'CUDA Event 微测采用固定输入捕获执行，选择与视角组件以三视角及所测 ratio 的中位数汇总。完整 prepare 消融单独关闭渲染消费者，六种配置交错执行。它用于判断机制，完整运行收益按前面的 prepare 加 render 表格评估。','',
        '|场景|组件|v27 μs|优化 μs|', '|---|---|---:|---:|']
    component_rows=read('MICRO.json')['rows']
    for group in ['tum','replica']:
        for module,version,title in [('selector','prefix','Selector 前缀'),('view','visible','可见 Anchor 视角编码')]:
            rows=[v for v in component_rows if v['group']==group and v['module']==module]
            values={n:statistics.median(v['median_us'] for v in rows if v['version']==n) for n in ['v27',version]}
            text.append(f"|{labels[group]}|{title}|{values['v27']:.3f}|{values[version]:.3f}|")
    text += ['', '|场景与请求|v27 μs|仅前缀 μs|仅可见编码 μs|仅小 Tile μs|三项开启 μs|',
        '|---|---:|---:|---:|---:|---:|']
    ablations=[read(f'ABLATION_{i}.json') for i in [18,19,20]]
    for group in ['tum','replica']:
        for condition in ['1pct','1000','5000','70pct']:
            rows=[v for d in ablations for v in d['results'] if v['group']==group and v['condition']==condition]
            values=[statistics.median(v['summary'][n]['prepare_ms']['median'] for v in rows)*1000 for n in ['v27','prefix','visible','small_tile','optimized']]
            text.append(f"|{labels[group]} {condition}|"+'|'.join(f'{v:.3f}' for v in values)+'|')
    text += ['',
        '## 正确性验收','',
        '|验证|结果|', '|---|---|']
    for name,label in [('VALIDATION.json','五类 checkpoint 主验证'),('SMALL_SCHEMA_VALIDATION.json','小 K 与模式／视角组合'),
        ('PREFIX_EDGES.json','前缀及开关边界'),('GRAPH_BOUNDARIES.json','Graph 边界'),('HOST_CAPACITY_VALIDATION.json','大容量保留路径'),
        ('TRAJECTORY_VALIDATION_tum.json','TUM 全轨迹'),('TRAJECTORY_VALIDATION_replica.json','Replica 已保存相机')]:
        d=read(name);assert d['passed'];details=[]
        for key,title in [('frames','帧'),('cases','组'),('comparisons','项逐位比较')]:
            if key in d:details.append(str(d[key])+title)
        text.append(f"|{label}|通过，{'，'.join(details)}|")
    text += ['', '逐位比较覆盖分数、选点身份、全部 Renderer 属性、像素和 radii。图像在内存中比较，保存图片数为零。随机比率、空集、预算边界、旧 11D、非默认 Stream、源模型修改隔离、有限值异常及下一请求恢复也通过。Replica 的 137 个视角来自 checkpoint 保存相机，覆盖范围按此记录。','',
        '所有新 Kernel 均保持 FP32 运算顺序。本轮实测像素与 v27 逐位一致，因此这些配对视角的 PSNR、SSIM 和 LPIPS 输入也一致。该证明以 v27 为基准，独立部署与训练 PyTorch 实现的历史差异沿用既有报告。','',
        '最终 memcheck 与 initcheck 的 TUM、Replica 四次验收均为零错误，原始输出保存在 logs/accepted_memcheck_* 与 accepted_initcheck_*。','',
        '## Nsight 归因','',
        '|场景与请求|版本|GPU Kernel μs|Kernel 数|Host Launch 数|同步次数|DtoH 字节|',
        '|---|---|---:|---:|---:|---:|---:|']
    profile=read('PREPARE_PROFILE_ANALYSIS.json')['summary']
    for r in profile:
        _,version,group,condition=r['label'].split('/')
        text.append(f"|{labels[group]} {condition}|{version}|{r['kernel_us']:.3f}|{r['kernel_nodes']:.0f}|{r['cpu_launches']:.0f}|{r['syncs']:.0f}|{r['dtoh_bytes']:.0f}|")
    text += ['', '三项改动主要减少重复计算、无效视角编码与小 K Tile 的空闲 SM。准备阶段的 Kernel 数和必要读回以实测表格为准。Nsight 的 Kernel 时间用于归因，完整 API 加速采用未 Profiling 的三轮墙钟。Nsight Compute 硬件计数权限在前轮返回 ERR_NVGPUCTRPERM，本轮采用 Systems、CUDA Event 和 cuobjdump 资源信息。','',
        '## 淘汰候选及修正','',
        '- 单点 CTA 的四项、八项局部循环展开通过数值测试，但显著变慢，已从最终源码移除。源码与结果保存在 stage_v28。',
        '- 16 点 Tile 提前用于 512–1024 容量有收益，但 8 点双累加器在三轮配对中更快。4 点单累加器在 K≈1000 时退化，最终源码移除对应分发。结果保存在 stage_v30_tiles。',
        '- 对每个尾部填充点增加备用 Anchor 索引，产生大 K 性能退化。最终用视角 Kernel 对零号填充入口写零，恢复原计算 Kernel 的地址模式，并通过 initcheck。对照保存在 stage_v30_padding。',
        '- 首轮两版本次序轮换规则在双版本时相互抵消。正式测速采用 ABBA，并让两版本共同预热。正式墙钟移除每请求 CUDA Event 创建成本，组件 Event 微测单独进行。','',
        '## 完整性及交付','',
        '|核验对象|文件数|变化数|', '|---|---:|---:|']
    for tag in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        row=integrity[tag];text.append(f"|{tag}|{row['files']}|{len(row['changed'])}|")
    gpu=read('FINAL_GPU_AUDIT.json')
    text += ['', f"正式三轮计时期间共保留 {gpu['samples']} 次独立 GPU 进程采样，其他 CUDA PID 数为 {len(gpu['foreign'])}。详细记录见 FINAL_GPU_AUDIT.json，且每个条件还保存前后 GPU 快照。"]
    identity=read('BINARY_IDENTITY.json')
    text += ['', 'Rasterizer 二进制 SHA256 为 '+integrity['RASTERIZER']['sha256']+'。', '',
        '最终扩展位于 '+identity['binary']['path']+'，SHA256 为 '+identity['binary']['sha256']+'。', '',
        '服务器研究目录为 '+str(ROOT)+'。本地源码、阶段结果、最终性能 CSV 和源码哈希清单一并交付。Git 提交、推送和生产部署均保留用户原有状态。','',
        '下一步应把完整调用中的 Host 调度、结果 Tensor 发布与必要同步继续拆分。当前网络准备已处于亚毫秒区间，更大收益需要有测量支持的调度或数据流改动。该结论仅针对所测 RTX 3090、两张真实地图和已列请求，端侧设备应重新选择 Tile 阈值。']
    (REPORTS/'DEPLOYMENT_STATIC_PREFIX_OPTIMIZATION_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(performance,indent=2))


if __name__=='__main__':main()
