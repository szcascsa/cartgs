"""实际CPU性能、GPU原路径质量和H2D证据生成中文报告。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'reports'/name).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
result=read('RESULTS_SUMMARY.json');frozen=read('FROZEN_CANDIDATE.json');integrity=read('INTEGRITY_FINAL.json')
assert result['passed'] and integrity['passed']
boundaries=read('boundaries02/SUMMARY.json');standalone=read('standalone01/SUMMARY.json');streams=read('handoff_stream02/SUMMARY.json')
sanitizer=read('jobs/sanitize04.json');ubsan=read('jobs/ubsan01.json')
assert boundaries['passed'] and standalone['passed'] and streams['passed']
s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');a=pd.read_csv(HERE/'reports/CPU_SPEED_AGGREGATE.csv')
stages=pd.read_csv(HERE/'reports/CPU_STAGES.csv')
q=pd.read_csv(HERE/'reports/quality_all01/quality.csv');startup=pd.read_csv(HERE/'reports/CPU_STARTUP_MEMORY.csv')
speedrows=[]
for scene in s.scene.unique():
    row=[scene]
    for condition in ['local0.01','local0.1','local0.7']:
        g=s[(s.scene==scene)&(s.condition==condition)]
        row.append(f'{g.pytorch_ms.median():.3f}→{g.median_ms.median():.3f} ({np.exp(np.log(g.speedup).mean()):.2f}×)')
    speedrows.append(row)
quality=q.groupby('scene').agg(combinations=('frame','size'),mean_loss=('psnr_loss','mean'),worst_loss=('psnr_loss','max'),
    min_equivalence=('cpu_vs_reference_psnr','min'),different_ids=('ids_equal',lambda x:int((~x).sum()))).reset_index()
qualityrows=[[x.scene,str(x.combinations),f'{x.mean_loss:+.6f}',f'{x.worst_loss:+.6f}',f'{x.min_equivalence:.3f}',str(x.different_ids)] for x in quality.itertuples()]
handoff_name='handoff01';handoff=read(handoff_name+'/SUMMARY.json');assert handoff['passed']
h=pd.read_csv(HERE/'reports'/handoff_name/'samples.csv');components=pd.read_csv(HERE/'reports'/handoff_name/'components.csv')
handoff_job=read('jobs/'+handoff_name+'.json');assert handoff_job['measurement_valid']
handoff_memory=pd.read_csv(HERE/'reports'/handoff_name/'memory.csv')
handoffrows=[]
for (scene,ratio),g in h.groupby(['scene','ratio']):
    c=components[(components.scene==scene)&(components.ratio==ratio)]
    handoffrows.append([scene,str(ratio),f'{c.cpu_prepare_ms.median():.3f}',f'{c.H2D_event_ms.median():.3f}',f'{c.raster_event_ms.median():.3f}',
        f'{g.total_cpu_prepare_H2D_render_ms.median():.3f}',f'{g.total_cpu_prepare_H2D_render_ms.quantile(.95):.3f}',f'{g.H2D_bytes.median()/1024:.1f}'])
text=f'''# 激进 CPU 准备与 MLP 推理优化报告

## 结论

已实现独立 C++/AVX2/FMA CPU 冻结推理后端。候选筛选、view7、Selector、固定噪声概率、严格K/全局分位数、Compensator、解码和属性打包全部在CPU完成；GPU仅接收紧凑属性并渲染。使用原Teacher权重，没有重训或靠蒸馏质量损失换取本表收益。

24组、两轮配对性能结果如下。加速比为场景轮次等权几何平均；中位延迟是48份场景轮次中位的中位，不能解释为全局单个典型帧。最终正式轮次为{', '.join(result['runs'])}；双方都采用相同的进程级OpenMP休眠策略。

'''
text+=table(['条件','CPU PyTorch ms','新CPU ms','几何平均加速','逐场景轮次范围','慢于原入口'],[
    [x.condition,f'{x.median_pytorch_ms:.3f}',f'{x.median_native_ms:.3f}',f'{x.geometric_speedup:.2f}×',f'{x.min_speedup:.2f}–{x.max_speedup:.2f}×',str(x.slower)]
    for x in a.itertuples()])
text+=f'''
共 {result['samples']:,} 个正式计时样本、{result['checks']:,} 个CPU预算/身份/有限值核对。CPU准备测量结束时CUDA未初始化。单帧相机Tensor构造、磁盘加载、H2D和渲染不包含在CPU prepare表，完整交接另测。

质量以同一固定噪声键的原GPU PyTorch实现作对照，覆盖24组×3视角×15预算共{result['quality_rows']}组。全部选点身份相同。平均GT PSNR损失 {result['mean_psnr_loss']:+.6f} dB（负数代表略升），单帧最大损失 **{result['worst_psnr_loss']:.6f} dB**。CPU图对原GPU图最低PSNR为{result['minimum_equivalence_psnr']:.3f} dB；少数像素因投影/Tile边界会有较大差异，不能称逐位一致。未修改生产默认、训练、原checkpoint或共享conda，没有commit/push/部署。

## 硬件、对照和公平性

服务器Intel i5-9600K，6核6线程，AVX2/FMA，PyTorch 2.3.1+cu118。C++使用FP32、-O3/-mavx2/-mfma，没有FP16、INT8、量化或-ffast-math。一个持久线程池包含主线程及5个worker；不启动嵌套BLAS多线程。启动阶段与稳态分别保存。

保留同一地图、权重、Gaussian身份、名义Anchor候选口径、ratio与固定点数预算、资格规则、原行号同分优先级、输出顺序。局部K=floor(ratio×M)，资格不足按min处理；全局分位数保留阈值同分语义，不替换成严格Top-K。

性能基线是6线程CPU PyTorch完整准备；两边使用相同的连续运动和跳转相机Tensor，每帧重算视锥、视角、候选和网络，不依赖同帧Mask或Feature命中。性能配对使用相同CPU固定噪声；跨设备质量对照则明确沿用原CUDA键。每场景3个连续运动和3个跳转相机，11条件、每条件24样本/版本/轮，版本次序轮换，第二轮反转场景顺序。

3个场景用于开发，冻结候选后开放其余21组验证。正式监测外部CPU活动（非本轮进程超过单核10%采样）、GPU活动、温度和时钟；受干扰段不作为正式结果。p95、原始逐样本、两轮变化及所有慢例保留CSV，不挑选有利视角。

### 发现并处理的进程内线程竞争

首轮paired01/02虽然没有外部负载，但同一进程交替调用PyTorch和原生线程池时，OpenMP空闲自旋会继续争抢6个物理核。不能将此状态称为纯原生稳态。开发场景诊断使用同一个冻结二进制，唯一改变是进程环境：OMP_WAIT_POLICY=PASSIVE、KMP_BLOCKTIME=0、GOMP_SPINCOUNT=0。原生连续运行约0.80ms；默认等待策略下交替运行约6–7ms，休眠策略下交替运行约1.00ms。两边都采用休眠策略后重新进行两轮跨场景配对，未根据场景调整参数。原始失败条件、逐样本及日志完整保留。部署时在新进程启动前设置上述环境；原生单独部署时Torch辅助操作只需1线程，6核并行由原生池负责。没有修改共享conda或系统环境。

## 实际实现

1. **C++整帧调度。** Python只负责入口参数与返回Tensor视图；通过一次C++调用执行全部准备，释放GIL。冻结状态由私有副本持有，原Master/模型修改不会让缓存过期；换模型或地图需要新实例。
2. **SIMD可见性/视角。** 8个Anchor同时做六平面名义AABB判定；可见Anchor的8角点并行投影。方向、asinh深度、屏幕坐标和最长投影范围仅每Anchor算一次。近裁面、safe_w和现有view公式保持一致。
3. **行号候选。** count→前缀和→scatter并行生成原始Gaussian顺序，不先复制候选的整套XYZ/SH等属性。
4. **Selector静态首层。** 加载时计算每点固定10D输入的32D首层前缀；每帧只加入log-ratio与Anchor view投影，再执行32→32→2及原概率计算。保持分数饱和同分规则，没有改成logit差排序。
5. **精确选择。** 使用原分数及原始行号复合键做nth_element，不对全部候选全排序；仅将K个输出恢复原始行序。全局单独按原分位数阈值处理。
6. **Compensator激进首层分解。** Gaussian Offset项加载时预计算；6个48D Ratio表的首层投影与节点前缀加载时生成；ratio项每请求一次、view项每选中Anchor一次。每个选中点只合并这些项，再执行原主干及四Head。
7. **四Head调度融合。** 四个独立64→64隐藏层合并为同输入的256输出调度，权重不共享；四个属性输出Head仍按各自隐藏块计算。4点独立累加链复用连续权重，Bias/ReLU与输出写出融合。
8. **直接解码与发布。** 只读取选中K的Master和SH，融合XYZ、正Scale、四元数、Opacity/DC解码，直接写最终连续Buffer。静态Scale/Opacity激活提前完成，保留动态有限值、合法相机、预算与输出检查。
9. **有界CPU→GPU交接。** 三份pinned Host/Device槽，将属性与本帧3个4×4矩阵及相机中心共51个FP32一起连续H2D。每帧使用新CPU准备结果；CUDA Event确保上传完成后渲染、上一使用结束后才复用槽。未finish租约的槽会拒绝覆盖。实现位于cpu_gpu_camera_handoff.py。

第一版虽已显著改善，但内循环重复调用Tensor.data_ptr带来元数据成本；第二版构造时绑定原生指针，显著减少额外开销。第三版进一步SIMD角点投影和固定4链微核；第四版合并Head调度并预计算静态激活。各阶段源码快照、二进制和微测保留。没有把减少FLOPs直接当作速度结论。

缓存只保存相机无关的冻结线性项。Compensator首层仍计算ReLU(W_offset·offset_i＋W_feature·F_anchor(ratio)＋bias＋W_ratio·ratio3＋W_view·view8)。利用线性投影提前处理固定Offset、Feature节点及其前缀，当前ratio、view和ReLU均实时计算。切换视角不让这些固定项失效，不缓存最终分数、候选Mask、Head输出或渲染图。首层分组变化已经纳入质量验证。

## 每场景真实CPU准备延迟

每格为PyTorch→原生CPU的两轮中位ms，以及场景两轮几何平均加速。实际M/K见逐帧CSV。

'''
text+=table(['场景/模式','1%','10%','70%'],speedrows)
text+='\n### CPU内部分段诊断\n\n以下为两轮计时外分段采样的中位ms，不与上面的完整延迟机械相加；Python包装、分配及测量条件也会影响完整延迟。\n\n'
stage_names=['候选/视锥','视角及Selector动态项','Selector','精确选择','Compensator及发布']
text+=table(['Ratio']+stage_names,[[condition[5:]]+[f'{stages[(stages.condition==condition)&(stages.stage==i)].ms.median():.4f}' for i in range(5)]
    for condition in ['local0.01','local0.1','local0.7']])
text+='''
## GPU原路径质量核对

CPU与CUDA相同seed不能保证随机键相同。独立导出步骤保存现有GPU噪声张量，CPU独立加载该键，避免随机选点变化掩盖数值差异。from_gpu_reference用于离线转换；from_checkpoint(checkpoint,noise_path)及运行时准备全部在CPU，不需要GPU生成网络输入或查询Feature。

以下损失=GPU原路径PSNR−CPU新路径PSNR。SSIM/LPIPS也按同图分别计算；按条件汇总在CPU_QUALITY_PER_CONDITION.csv。PSNR采用实际原始渲染值，SSIM/LPIPS采用相同[0,1]截断，不混淆口径。评价不保存渲染图片。

'''
text+=table(['场景','组合数','平均损失dB','最大损失dB','CPU对GPU图最低PSNR','ID变化组'],qualityrows)
text+=f'''
条件平均最大SSIM损失为{result['max_condition_ssim_loss']:.8f}，最大LPIPS增加为{result['max_condition_lpips_increase']:.8f}。该验证覆盖72个保存相机及多个ratio/固定预算，不是所有轨迹逐帧保证；未将三个代表视角的结果外推到所有场景状态。

数值差异来源：FP32 FMA与MKL/CUDA求和次序不同、首层线性分解改变分组、libm exp/log/asinh与CUDA实现不同。输出属性虽仅有小误差，极薄/大footprint点可能跨越预处理或Tile覆盖边界，导致少数像素误差被放大。全部最差项、最大RGB差、几何误差和对应预算均保留，不以平均指标掩盖。

## CPU准备＋H2D＋原Rasterizer

GPU仍使用原FP32 Rasterizer，未把之前的低点数渲染改动计入CPU优化收益。下表的CPU/H2D/Raster Event分项与完整同步墙钟分别测量，不应机械求和。分项H2D仅传属性；完整列额外包含51个相机FP32的打包及同包上传，并含准备、排队、跨Stream依赖及渲染；不含CPU相机矩阵构造、I/O、HTTP或显示。

'''
text+=table(['场景','ratio','CPU ms','H2D Event ms','Raster Event ms','完整中位ms','完整p95 ms','上传KiB'],handoffrows)
text+=f'''
三份有界槽位的Host/Device容量单独保存在handoff01/memory.csv。这个测量报告单帧同步延迟，未把下一帧流水线吞吐当作当前帧延迟，也不复用相同相机的旧结果。尚未进行真正端侧CPU/PCIe或ARM平台测速，不能从3090服务器外推移动设备性能。

'''
text+=table(['场景','原生主要逻辑容量MiB','三槽Pinned Host MiB','三槽GPU MiB'],[
    [x.scene,f'{x.cpu_workspace/(1<<20):.2f}',f'{x.host/(1<<20):.2f}',f'{x.device/(1<<20):.2f}'] for x in handoff_memory.itertuples()])
text+=f'''

## 空间、冷启动与边界

原生Workspace主要逻辑容量估算范围{startup.workspace_bytes.min()/(1<<20):.2f}–{startup.workspace_bytes.max()/(1<<20):.2f}MiB；该计数包含主要Tensor/缓存及工作数组，未含少量权重、全部分配器开销、线程栈或Python/Torch运行时，不是进程RSS。快照初始化范围{startup.init_ms.min():.2f}–{startup.init_ms.max():.2f}ms，未计入稳态表，不包含磁盘模型加载和冷编译。

缓存按N、A和最多6个ratio节点有界，投影缓存上限1GiB。没有建立任意相机/ratio历史缓存。更多缓存可能超过9MiB L3，因此只保留已实测收益的固定项/投影项；缓存成本不等于全程序内存。

边界核验包含空预算、1/3/7/8/9/31/32/33/63/64/65/127/128/129/255/256/257/511/512/513、资格不足、单实例并发序列化、输出独立、外部Buffer尾部哨兵、源模型修改隔离、非法ratio/非有限相机拒绝与错误恢复。CPU→GPU租约另验证三帧连续提交后逐像素一致，不允许覆盖未消费槽。

V4正式二进制边界检查{boundaries['checks']}项通过；checkpoint＋原GPU固定键的全CPU独立恢复{standalone['checks']}项与内存快照逐位相同，CUDA未初始化；非默认GPU Stream的数据交接与租约检查{streams['checks']}项通过，相机及属性逐位相同，再回原Rasterizer默认Stream渲染，RGB及radii逐位相同。

首轮handoff_stream01尝试在非默认Stream上下文直接调用原Rasterizer，图像/半径一致性断言失败。源码含未指定Stream的CUDA Kernel Launch，不能把原入口视为任意Stream安全。原复现脚本handoff_boundaries_failed_stream.py和失败日志保留；本轮不修改Rasterizer或训练。正式交接路径使用独立上传Stream＋默认渲染Stream，不能宣称已支持原Rasterizer的任意Stream渲染。

ASan+UBSan独立重编译任务sanitize04退出码为{sanitizer['returncode']}，覆盖28项正常请求、预算、Buffer、快照隔离与并发检查。日志保存在logs/sanitize04.log及对应ASan日志；该任务使用独立命名的O1调试扩展，不覆盖正式V4二进制。{'这28项通过，未报告AddressSanitizer或UndefinedBehaviorSanitizer错误。' if sanitizer['returncode']==0 else '该检查未通过，不能声称内存安全Sanitizer验收完成；失败现场保留。'}独立UBSan完整33项任务ubsan01退出码为{ubsan['returncode']}，日志位于logs/ubsan01.log。ASan使用detect_leaks=0，未进行LeakSanitizer泄漏验收。

首轮sanitize01在扩展构建前CPU持续忙等，经带超时Python栈定位，停在torch.utils.cpp_extension导入时的torch.cuda.is_available。不是原生Kernel测试失败。只停止核对PID/start_ticks/boot_id后的本轮子进程并保留审计；将ASan CPU检查进程的GPU可见性置空、采用NVML可用性探测后，导入探针1.9秒完成，再运行sanitize02。没有修改共享环境或绕过CPU内存检测；正常GPU质量和交接测试不使用这些隔离设置。

sanitize02/03在主动抛出非法ratio异常时再次阻塞。独立最小复现sanitizer_exception_probe.py只调用官方torch.empty((-1,))、不加载本轮扩展，也在系统ASan预加载下阻塞，因此不能归因于本轮原生算子。sanitize04明确仅跳过5项故意异常，其余28项保留ASan+UBSan；33项完整行为已在正式V4常规运行通过，另用不预加载ASan的UBSan完整复核。不能声称ASan已覆盖异常分支；全部失败探针和超时栈保留。

## 限制和后续

- 新实现专用于冻结推理；不支持在同一实例内训练或更改拓扑。原训练及正常入口保留。
- 不承诺逐位一致。当前质量门槛和实际数值均披露，不因用户允许误差就跳过选点/图像检查。
- 更多线程不保证更快。本机6物理核，1/2/6线程微测独立保留；小K串行/SIMD与并行任务结合，没有启动多于物理核的常驻worker。
- 保留32/32学生兼容接口，但本轮正式测试仅验收原Teacher，未独立验收该接口；Teacher→Teacher加速不计入蒸馏收益。
- Full CPU对照原本仅返回已有Master属性，原生诊断Full入口则重打包整图；这条路径没有网络工作且会慢于原入口，本轮未将其当作网络提速收益。空预算和所有慢例也保留在总表。
- 高K的四Head与最终属性写出仍有成本；若继续优化，优先比较分块打包BLAS/oneDNN与当前AVX2微核，以及改进线程亲和/大工作集缓存行为。不能宣称已达到CPU硬件极限。

## 完整性与交付

独立目录：`/home/prmi/zwj/experiments/cpu_native_aggressive_20261110_1115`。

最终二进制：`{frozen['binary']}`。

SHA256：`{frozen['binary_sha256']}`。

原生产、输入模型、前轮证据和相关源码共{integrity['protected_files']}文件SHA核验，变化0、缺失0；冻结实现与二进制身份一致。没有修改共享conda、生产默认或训练，没有commit/push。所有实验原始CSV、失败日志、冻结源码与二进制、性能图及完整性结果保存在本轮目录。

调用说明见README_CPU_DEPLOYMENT.md；逐场景曲线位于figures/REPLICA_CPU_LATENCY.png、figures/TUM_VECTOR_CPU_LATENCY.png；速度/质量散点位于figures/SPEED_AND_QUALITY.png，均有SVG及数据哈希清单。报告中的速度不能直接作为GPU网络推理的对照，也不能外推为端侧CPU硬件成绩。
'''
(HERE/'CPU_AGGRESSIVE_INFERENCE_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8');print('CPU_CHINESE_REPORT_WRITTEN',len(text))
