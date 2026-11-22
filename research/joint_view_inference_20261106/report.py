"""根据固定测量生成中文交付报告，不修改实验指标。"""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results';OUT=ROOT/'reports'


def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))


def main():
    benchmarks=[read('BENCHMARK_1.json'),read('BENCHMARK_2.json')]
    index=[{(v['condition'],v['backend']):v for v in b['summary']} for b in benchmarks]
    display=read('DISPLAY_VALIDATION.json');nsys=read('INFERENCE_NSYS_analysis.json')
    validation={name:read(name+'.json') for name in ['LAYER_VALIDATION','PIPELINE_VALIDATION','STATE_VALIDATION','TOPOLOGY_VALIDATION','TRAJECTORY_VALIDATION','UNIT_VALIDATION','PURITY_VALIDATION']}
    assert all(v['passed'] for v in validation.values()) and display['passed']
    final=index[-1]
    lines=['# 视角条件Residual48冻结推理CUDA优化报告','',
        '日期：2026-11-06。优化对象为当前本地main的19D Selector＋Residual48视角Compensator，使用已训练的fr1_xyz RGB-D双视角＋LI=0.1模型。只替换冻结推理实现，不训练、不改网络、Loss、选择规则或checkpoint。','',
        '## 结论','',
        '已实现支持joint schema1/2/3/4和旧11D的CUDA／CUDA Graph后端、冻结静态输入缓存、19D输入与K×8条件融合、Anchor视角编码融合、K点属性直接发布、可选关闭计时同步，以及原生FP32显示封包融合。下面以最新一轮实测列出准备和准备＋Rasterizer的收益；不将其外推为HTTP／显示器实际FPS或在线SLAM训练加速。','',
        '|条件|实际K均值|原准备ms|新准备ms|准备加速|原准备＋渲染ms|新准备＋渲染ms|总加速|','|---|---:|---:|---:|---:|---:|---:|---:|']
    conditions=['ratio_0.01','ratio_0.05','ratio_0.1','ratio_0.3','ratio_0.7','points_2000','points_5000']
    for c in conditions:
        a,b=final[c,'pytorch'],final[c,'cuda_graph_no_timing']
        lines.append(f"|{c}|{a['mean_K']:.1f}|{a['prepare_host_ms']:.3f}|{b['prepare_host_ms']:.3f}|{a['prepare_host_ms']/b['prepare_host_ms']:.2f}×|{a['total_host_ms']:.3f}|{b['total_host_ms']:.3f}|{a['total_host_ms']/b['total_host_ms']:.2f}×|")
    lines+=['','两轮独立进程、五相机0/198/396/594/791；每条件48次交错正反后端顺序，舍弃前8次，每后端每条件200个热样本／轮。保持各视角自身候选人口与实际K，不跨场景套用same-K。FP32、TF32关闭、无fast-math，CUDA编译--fmad=false。Host总时间包括相同Event/调度测量成本；Event时间包含CPU提交空隙，纯Kernel时间另由Nsight统计。','',
        '|条件|第1轮总加速|第2轮总加速|','|---|---:|---:|']
    for c in conditions:
        lines.append('|'+c+'|'+'|'.join(f"{v[c,'pytorch']['total_host_ms']/v[c,'cuda_graph_no_timing']['total_host_ms']:.2f}×" for v in index)+'|')
    lines+=['','## 实际保留的实现','',
        '1. FrozenResidualInference兼容11D和19D。19D输入保持原11D物理量＋view7＋mode；全局条件全零，局部按schema路由。Selector首层仍N候选×19，Compensator视角投影仍K×8→64，没有分拆首层、改变GEMM形状、合并四Head或补齐K。','',
        '2. rebuild缓存全图10D Selector静态属性、归一化/asinh/clamp offset3、activated Scale/Opacity、归一化Quaternion、owner、reference及Slot资格。Anchor中心齐次坐标和八角点世界坐标也缓存。相机变化仍重新计算视角数据，禁止近似复用相邻相机。','',
        '3. 视角编码保留原ATen方向归约/sqrt以及四个矩阵乘法的原形状、顺序。CUDA只融合safe_w、投影除法、NDC clamp、八角点范围、asinh/log1p和最终7D写入。Selector条件Gather/拼接和Compensation的K×8条件打包各用单Kernel。encode＋ratio_projection＋view_projection按原顺序逐次FP32舍入，再ReLU。','',
        '4. 只对最终有序K个ID查询Residual48和属性，直接形成Renderer连续Buffer，减少对全部候选的clone、scatter后再compact。Residual按Base和活跃层原顺序累加；未活跃层不加零。XYZ/Scale/Opacity/DC融合解码，Quaternion继续原ATen，四Head独立参数与原调用顺序不变。','',
        '5. Graph按实际K、compensation、decode、全局/局部模式缓存；更新ID、ratio控制量和本帧Anchor视角编码后重放。最多6项／512MiB、形状记录32项；返回属性拥有独立存储，重放和淘汰不会改写旧结果。全局固定ratio完整输出LRU仍8项。动态ratio请求及固定预算候选不足时，未命中Graph则用融合CUDA，不在交互请求中捕获新形状；已有Graph仍可复用。达到请求K的固定点数模式才自动捕获，避免动态K反复建图。','',
        '6. prepare新增measure_time=False，仅去掉分阶段计时的显式device synchronize；原finite检查仍在原消费位置完成。默认True保留既有接口，关闭后分阶段计时标记为不可用，不能把0解释为实际零耗时。Viewer请求可传measure_inference_stages:false。','',
        '7. 原生Viewer新增可选FP32 staging，融合CHW→HWC和原clamp，将图像／参考图合成一次DtoH读回；保持原little-endian FP32协议。单份GPU与pinned Host缓冲，每份最多64MiB，超限回退原封包；没有JPEG、降精度或降低分辨率。Python bytes发布后可独立保存，不受下帧Buffer重写影响。','',
        '## 独立候选收益','',
        '|条件|CUDA准备ms|关闭视角融合ms|CUDA Graph准备ms|Graph关闭计时同步ms|','|---|---:|---:|---:|---:|']
    for c in conditions:
        lines.append('|'+c+'|'+'|'.join(f"{final[c,k]['prepare_host_ms']:.3f}" for k in ['cuda','cuda_no_view_fusion','cuda_graph','cuda_graph_no_timing'])+'|')
    lines+=['','CUDA和Graph保留原GEMM形状，速度收益主要来自静态缓存、融合和调度减少。四Head合成更大GEMM、TensorRT／混合精度及整图推理后再截取均未采用，避免改变累加顺序或算法。历史Residual前缀缓存未获得稳定收益，本轮沿用原序融合查询，没有重新宣称其有效。','',
        '## 数值、像素与状态验证','',
        '|验证|比较数／条数|结果|','|---|---:|---|']
    for n,v in validation.items():lines.append(f"|{n}|{v.get('comparisons',v.get('tests'))}|通过|")
    lines+=['','逐bit用原始字节或int32比较，包含signed zero，不用allclose。分层覆盖训练/未见ratio、节点±2e−8、0/1/7/32/129/1k/13298点；完整路径覆盖五模型、五相机、全局/局部、固定预算、候选不足、补偿开关、Graph与计时开关。','',
        '全轨迹为792帧×34条件=26928组，覆盖Full、全局与局部11ratio、五档matched和六档固定点数，选点SHA匹配既有both_li指标；普通原路径与新路径每组Renderer属性、radii与像素逐bit一致。不重训或更换checkpoint，不保存渲染帧图片，已有PSNR/SSIM/LPIPS数值可沿用；这不是用PSNR接近代替严格等价。','',
        '状态测试覆盖Feature/权重/Master版本变更、Anchor和Gaussian重排序、Clone/Split/Anchor增长、Prune、全局几何修正、106416点合成负载、空拓扑、checkpoint重载、非默认Stream、Graph淘汰与旧输出所有权。大N是合成边界，不称第二个真实场景。模型/Buffer/Master对象、地址、版本、形状、SH阶数、ratio归一化范围或ratio节点变化会拒绝旧缓存，要求rebuild。','']
    lines+=['## Nsight Systems','',
        '|范围|Kernel数|Host发射|同步调用|DtoH|累计Kernel μs|','|---|---:|---:|---:|---:|---:|']
    for v in nsys['summary']:
        if v['label'].startswith(('prepare/pytorch/','prepare/cuda_graph_no_timing/')):
            lines.append('|'+v['label']+'|'+'|'.join(f'{v[k]:.1f}' for k in ['kernels','host_launches','host_syncs','dtoh','kernel_us'])+'|')
    lines+=['','使用CUDA 11.8自带Nsight Systems、NVTX及CUDA Graph node追踪，CUPTI correlation归属同线程Host范围；Graph内部节点计入GPU Kernel，不能把一次Graph launch说成一个Kernel。Profiler墙钟不与无Profiler结果混比。主要剩余开销是原形状GEMM、CUB稳定排序和Rasterizer；本轮未改变像素合成顺序。','',
        '## 连续移动相机的实际延迟','',
        '|条件|真实K范围／种类|Graph捕获次数|原累计秒|CUDA累计秒|Graph累计秒|Graph p99 ms|Graph最大ms|','|---|---|---:|---:|---:|---:|---:|---:|']
    moving=read('MOVING_BENCHMARK.json')
    for v in moving['results']:
        lines.append(f"|{v['condition']}|{v['min_K']}–{v['max_K']}／{v['unique_K']}|{v['captures']}|{v['total_ms']['pytorch']/1000:.3f}|{v['total_ms']['cuda']/1000:.3f}|{v['total_ms']['cuda_graph']/1000:.3f}|{v['p99_ms']['cuda_graph']:.3f}|{max(v['raw_ms']['cuda_graph']):.3f}|")
    lines+=['','每条件从空Graph缓存开始，逐帧相机变化，三后端交错、包括全部792帧和首次捕获成本；与固定相机热态数据分开解释。ratio预算的实际K随候选人口变化，固定请求点数若候选不足也可能改变K。优先根据累计耗时和p99选择cuda或cuda_graph，不能只用热态中位数宣称Graph总是更好。','',
        '## 显存与原生封包','',
        '|后端|额外保留活跃显存MiB|独立进程峰值活跃MiB|Graph项数|Graph预算记账MiB|','|---|---:|---:|---:|---:|']
    for name in ['pytorch','cuda','cuda_graph']:
        v=read('MEMORY_'+name+'.json')
        lines.append(f"|{name}|{v['additional_retained_bytes']/1024**2:.2f}|{v['peak_allocated']/1024**2:.2f}|{v['graphs']}|{v['graph_accounted_bytes']/1024**2:.2f}|")
    lines+=['','同场景各后端独立进程、六种K预热，活跃显存由torch.cuda统计；reserved及Graph记账口径不等同于新增活跃显存。ATen分配次数采用三后端相同CPU allocator Profiler，GPU活动由Nsight记录。原生显示Buffer未使用时不分配；640×480且带参考图时每份约7.03MiB，GPU+pinned Host约14.06MiB。','',
        '|原生封包条件|原ms|仅减少Host复制ms|融合ms|加速|','|---|---:|---:|---:|---:|']
    for v in display['benchmarks']:
        lines.append(f"|{'带参考图' if v['reference'] else '仅图像'}|{v['original_ms']:.3f}|{v['join_only_ms']:.3f}|{v['fused_ms']:.3f}|{v['speedup']:.2f}×|")
    lines+=['','封包比较包含相同FP32 GPU输入、Host读回及Python bytes，不含HTTP和显示器。1×1、17×31、640×480、1200×680和±0/Inf/NaN等边界逐字节验证；首/中/末真实Viewer handler的像素、PSNR、投影点数与参考图结果相同。','',
        '|完整原生处理请求|原ms|新ms|加速|','|---|---:|---:|---:|']
    viewer=read('VIEWER_HANDLER_BENCHMARK.json')
    for v in viewer['results']:
        lines.append(f"|请求{v['budget']}点，Full参考{'开启' if v['reference_full'] else '关闭'}|{v['original_ms']:.3f}|{v['optimized_ms']:.3f}|{v['speedup']:.2f}×|")
    lines+=['','完整处理函数还包含相机、Anchor统计、PSNR、参考图处理和FP32封包，按五相机交错、每条件每后端100样本统计；仍不包含HTTP、客户端和显示器。原图像与参考图字节、PSNR及点数逐值一致。','',
        '## 失败候选与测量修复','',
        '- 首版显示clamp对−0返回−0，而原PyTorch返回＋0；字节检查阻止该版本保留。已改为保留NaN并复现fmax/fmin的signed-zero语义，完整重新编译并验收，失败现场在DISPLAY_SIGNED_ZERO_FAILURE.json。','',
        '- 显示融合初版虽逐字节一致，但带参考图5.22ms慢于原4.56ms。已定位并移除全量bytes生成后再切片的多余复制，同时将最终封包改为一次join；修正版独立复测，原始失败测量保留在DISPLAY_EXTRA_COPY_CANDIDATE.json。','',
        '- 原先无差别对动态ratio捕获Graph，1%连续移动64种实际K触发192次捕获，累计9.77秒慢于原PyTorch4.29秒；热态中位数掩盖了约45ms捕获尖峰。已改为动态ratio miss走融合CUDA、固定实际请求K才捕获，并重跑相同工作量；失败版本的完整记录在before_safe_graph_policy/MOVING_BENCHMARK.json。','',
        '- 当前安装的PyTorch CUDA Profiler在Graph重放采集结束后的postprocessing发生CPU工作线程持续占用，GPU普通推理和Nsight均正常。核实PID/start/命令归属后仅终止本轮卡住的测量进程，保留profiler_cuda_stall现场；改用三后端相同CPU allocator instrumentation＋独立CUDA memory统计，GPU时间继续用Nsight，不修改系统ptrace或共享环境。','',
        '## 使用与回退','',
        '服务器独立源码：/home/prmi/zwj/experiments/joint_view_inference_20261106/source；扩展仅在该目录build内编译。模型使用view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply，只读复用。','',
        '加载BudgetScene时选择inference_backend="cuda_graph"；局部prepare(camera,budget=2000,view_conditioned=True,measure_time=False)，然后render_prepared。全局导出prepare(None,global_ratio=r)，保持零条件。原生Viewer API请求传view_conditioned:true，measure_inference_stages:false；后端cuda_graph需要显式选择，未替换默认配置。','',
        '独立Viewer启动脚本：source/research/joint_view_inference_20261106/launch_viewer.py，使用zwj的Python执行，后接checkpoint、--dataset、--inference-backend cuda_graph等原Viewer参数。脚本只配置本轮build与只读Rasterizer依赖，未修改共享环境。局部视角推理仍须显式开启Viewer视角条件选项，保持原产品语义。','',
        '回退使用inference_backend="pytorch"；只禁用Graph可选"cuda"。视角后处理单独回退为engine.fused_view=False。原生封包可不传staging回到原路径。模型更新和拓扑变更后必须renderer.rebuild()，不能复用旧owner与静态输入。多线程请求继续使用现有scene.lock；不同并行流需要独立可写Workspace，不并发重放同一实例。','',
        '## 完整性与限制','',
        '本轮不修改生产mycartgs／realLOD、不改共享conda、不终止其他Viewer或训练。604个生产文件与三组checkpoint前后SHA核验以FINAL_INTEGRITY.json为准。保留原始源码和独立开关；未commit/push。','',
        '性能来自fr1_xyz的冻结模型，不能宣称其他地图、GPU、工具链或训练也有相同加速。无需重新训练；数值保证限于本轮验证的FP32路径，未覆盖的新工具链须重新验收。实际Viewer还受HTTP传输、参考图评价和显示影响，完整组件时间与原始记录均交付。']
    (OUT/'JOINT_VIEW_INFERENCE_OPTIMIZATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('中文报告生成')


if __name__=='__main__':main()
