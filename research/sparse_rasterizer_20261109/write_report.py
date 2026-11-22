"""只报告已取得的并发诊断证据，空闲GPU性能留待独立复测。"""
from pathlib import Path
import json,csv,statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results/reports').exists() else HERE


def read(p):return json.loads(p.read_text())


def main():
    r=ROOT/'reports';folder=r/'concurrent_diagnostic_final'
    if not (folder/'SUMMARY.json').exists():folder=r/'concurrent_diagnostic_v2'
    d=read(folder/'SUMMARY.json')
    validation=read(r/'VALIDATION.json');lifecycle=read(r/'LIFECYCLE_VALIDATION.json')
    workloads=list(csv.DictReader((folder/'workloads.csv').open()))
    records=d['results'];index={(v['scene'],v['condition'],v['variant']):v for v in records}
    rows=[]
    for scene,condition in sorted({(v['scene'],v['condition']) for v in records}):
        g=[v for v in workloads if v['scene']==scene and v['condition']==condition]
        times={n:index[(scene,condition,n)]['wall_ms'] for n in ['original','control','warp_expand','short','warp_short','bitset']}
        row=dict(scene=scene,condition=condition,K_median=st.median(int(v['K']) for v in g),
            bbox_instances_median=st.median(int(v['tile_bbox_instances']) for v in g),
            valid_instances_median=st.median(int(v['valid_tile_instances']) for v in g),
            nonempty_short_tile_fraction=st.mean(int(v['short_nonempty_tiles'])/max(1,int(v['active_tiles'])) for v in g))
        row.update(times)
        row.update(bitset_vs_original=times['original']/times['bitset'],bitset_vs_control=times['control']/times['bitset'],
            warp_vs_control=times['control']/times['warp_expand'],short_vs_control=times['control']/times['short'],
            warp_short_vs_warp=times['warp_expand']/times['warp_short'])
        rows.append(row)
    with (r/'SPARSE_RASTERIZER_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    low=[v for v in rows if v['condition'] in ['local1pct','local5pct','local10pct']]
    lines=['# 低点数Rasterizer CUDA实现与诊断报告','','## 结论与证据范围','',
        '已在独立研究目录实现三项候选：稳定Warp展开、短Tile列表Warp混合、全局稳定深度排序与Tile Bitset。没有按Gaussian数量设置新旧路径切换门槛；三项实现对高点数也执行测试。正常训练、生产默认、Selector、Compensator和模型保持不变。','',
        f'FP32逐像素及radii通过 {validation["comparisons"]} 组逐位比较，动态形状、输出独立、预计算RGB/协方差、SH与CUDA Stream通过 {lifecycle["checks"]} 项检查。Tile成员及同深度顺序独立核验。结果相同意味着匹配输入下PSNR、SSIM、LPIPS相同，不以接近的PSNR代替逐位检查。','',
        '用户已明确选择“先完成并发负载下的诊断，正式测速稍后”。服务器其他TUM评价与Viewer进程保留；负载采样记录在LOAD.json。因此下面速度只用于候选筛选，不是空闲GPU的正式加速结论。','',
        f'并发诊断中，1%/5%/10%条件的Bitset相对原入口加速范围 {min(v["bitset_vs_original"] for v in low):.2f}–{max(v["bitset_vs_original"] for v in low):.2f}×，相对同研究入口控制组 {min(v["bitset_vs_control"] for v in low):.2f}–{max(v["bitset_vs_control"] for v in low):.2f}×。Full结果单独报告，避免用低点数收益掩盖高密度退化。','',
        '## 源码与运行边界','',
        '服务器 `/home/prmi/zwj/experiments/sparse_rasterizer_20261109/`；本地对应research/sparse_rasterizer_20261109。参考来自实际使用的global_view_joint_lod_20261105/source/cuda_rasterizer及其只读推理二进制，原二进制SHA256为27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86。','',
        '`source/sparse_inference.cu`实现独立推理入口和三个算法；`generate_forward.py`从只读参考源生成`source/forward_sparse.cu`和逐字参考展开核`reference_emit.cuh`；`runtime.py`提供显式SparseRenderer；`validate.py`、`test_lifecycle.py`、`sanitizer_smoke.py`及`benchmark.py`负责验证与诊断。','',
        '每实例复用有限个Scratch Buffer，所有投影、颜色、深度、Tile成员逐帧重算，输出Tensor独立分配。单个Binning/Bitset工作区上限256MiB，超过时显式报错，不截断结果。实例绑定device与stream；切换stream须另建实例，避免共享Workspace竞争。无训练梯度、无GI/贡献统计路径修改，不缓存上帧可见性或已补偿属性。','',
        '## 三项算法','',
        '1. 原展开由单线程串行扫描一个Gaussian的包围矩形。Warp候选让32条线程并行计算连续Tile，ballot与popcount稳定压缩有效项。每Gaussian输出仍保持原行优先顺序，末尾填入同样的sentinel，随后使用相同稳定CUB排序。','',
        '2. 短列表候选在每个Tile列表长度≤32时，用Warp Shuffle广播Gaussian数据并保持每像素原始混合算式；长列表继续执行同一核中的原循环。这是按Tile实际工作量优化，不是按地图点数屏蔽高点数。','',
        '3. Bitset候选对Gaussian中心深度排序一次；无效Gaussian置于末尾。每Tile用按深度rank排列的Bitset表示有效成员，成员判断沿用原包围矩形、conic与Opacity判据。逐位枚举成员，按与原Tile排序相同的Gaussian顺序执行原Alpha合成。消除大规模重复实例排序，以及实例总数的CPU回读；保留FP32运算、Alpha cutoff、透射率早停和同深度输入顺序。','',
        'Bitset状态大小为4×Tile数×ceil(K/32)字节，位表扫描随K增长。低K、大投影的工作负载可以避免大量重复排序；高K、密集覆盖下扫描空字和维护位表的成本可能超过节省量，不承诺普遍加速。','',
        '## 测量方法','',
        '4个Replica场景/模式：office1单目、office2 RGB-D、room0 RGB-D、room2单目。每组8个跳转视角；同一包渲染输入固定，不把Selector/Compensator变化混入。条件包括局部1/5/10/70%、固定1000/5000点和Full；每条件3次重复×40帧，版本交替执行。实际K不足固定预算时如实记录。纯渲染墙钟包含入口调用至CUDA同步，不含模型读取、选点、补偿或图片保存。','',
        '逐项消融包含：原已加载二进制、同研究入口的逐字串行展开控制、Warp展开、仅短列表、两者组合、Bitset。控制与候选统一预处理和入口；原入口对照保留真实部署成本。初轮V1控制核并非逐字参考，可能放大算法差距，已修正为V2；V1保留回溯，不用于正式结论。','',
        '当前未找到可执行Nsight Systems，使用PyTorch/Kineto CUDA Activity记录Kernel名称、发射次数与实际GPU活动时间，并导出Chrome trace。Profiling有额外扰动，时间单独列出，不能与无Profiler墙钟直接拼接。','',
        '## 每场景负载与耗时（并发诊断）','',
        '| 场景 | 条件 | 实际K中位数 | bbox/有效实例 | 原入口ms | 控制ms | Warp展开ms | 短列表ms | 展开＋短列表ms | Bitset ms | Bitset/原入口加速 |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in rows:
        lines.append(f'| {v["scene"]} | {v["condition"]} | {v["K_median"]:g} | {v["bbox_instances_median"]:g}/{v["valid_instances_median"]:g} | {v["original"]:.4f} | {v["control"]:.4f} | {v["warp_expand"]:.4f} | {v["short"]:.4f} | {v["warp_short"]:.4f} | {v["bitset"]:.4f} | {v["bitset_vs_original"]:.2f}× |')
    lines+=['','## GPU活动与发射数量（Profiling扰动下）','',
        '只汇总Kineto trace中cat=kernel的事件，排除覆盖整个GPU区间的user annotation，Memcpy与Memset另列，避免重复计时。','',
        '| 条件 | 版本 | Kernel/帧 | Memcpy/帧 | Memset/帧 | 累计Kernel μs/帧 |','|---|---|---:|---:|---:|---:|']
    for v in d['profiles']:lines.append(f'| {v["condition"]} | {v["variant"]} | {v["kernels_per_frame"]:g} | {v["copies_per_frame"]:g} | {v["memsets_per_frame"]:g} | {v["gpu_us"]:.2f} |')
    lines+=['','## Workspace与一致性','', '| 版本 | 实例保留Scratch MiB |','|---|---:|']
    for n,b in d['workspace_bytes'].items():lines.append(f'| {n} | {b/2**20:.3f} |')
    lines+=['','此处为4场景测试后的高水位Scratch，不含输出图像、Torch缓存分配器和原模型。缓冲区扩容按同一实例曾遇到的最大形状保留，不按帧累积。Bitset没有复制可学习Feature或新增网络参数。','',
        '## 失败、限制与下一步','',
        '编译初次因PATH未包含已有conda ninja失败，已仅在本进程PATH修正，没有安装或修改共享环境。首轮验证写报告时发生辅助脚本HERE变量被导入覆盖，已修正；测试输出归档。生命周期测试预计算RGB/协方差时的None格式与原wrapper一致后通过。','',
        'Compute Sanitizer首轮memcheck零错误，initcheck发现Scratch扩容的resize_复制未初始化的旧临时字节。已改为扩容时丢弃旧内容并重新分配；所需状态每帧重建。诊断快照在inspect时预先清零，避免读出无效Gaussian未定义的诊断字段。此修正不改变计算公式或像素，只影响工作区增长行为。','',
        '目前三个实现保持为显式研究选项；最终保留哪项及组合，需要空闲GPU复测。尤其Full Bitset退化不能通过按K人为屏蔽隐藏；后续应根据实际Tile密度、位表扫描成本分析结构限制。短列表是否值得保留应看独立增益，不能仅因组合版本更快就归因。','',
        '下一次空闲时运行同一benchmark.py，不带--allow-concurrent，写入新的独立输出目录；至少重复两轮，比较低点数绝对耗时、同入口控制组、Full及Kernel活动。不能把当前并发数字当作正式端侧性能，也不能将Rasterizer通用收益计入Selector/Compensator网络提速。','']
    sanitizer=r/'SANITIZER_RESULTS.json'
    if sanitizer.exists():
        s=read(sanitizer)
        lines+=['## CUDA内存检查','',f'最终Compute Sanitizer memcheck/initcheck通过状态 {s["passed"]}，错误汇总 {s["error_summaries"]}。检查使用小型真实CUDA入口，覆盖1/33/257点、无效点和全部算法，不代表未测试输入的穷尽证明。','']
    integrity=r/'INTEGRITY.json'
    if integrity.exists():
        v=read(integrity);lines+=['## 完整性核验','',f'保护文件 {v["files"]} 个，变更数 {len(v["changed"])}，通过状态 {v["passed"]}。未修改服务器mycartgs/realLOD、原二进制或Teacher，未commit/push。','']
    report='\n'.join(lines)
    (r/'SPARSE_RASTERIZER_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT!=HERE:(HERE/'SPARSE_RASTERIZER_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
    print('SPARSE_REPORT',len(rows),validation['comparisons'],lifecycle['checks'])


if __name__=='__main__':main()
