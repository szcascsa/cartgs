"""本地从已下载原始结果生成交付；所有百分比可重新计算。"""
from pathlib import Path
import json
import shutil
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def table(a):return a.to_markdown(index=False,floatfmt='.4f')
native=pd.read_csv(HERE/'before_after_benchmarks.csv');launch=pd.read_csv(HERE/'kernel_launch_comparison.csv')
online=pd.read_csv(HERE/'online_performance_comparison.csv');quality=pd.read_csv(HERE/'online_quality_summary.csv')
summary=[];speed=[]
for case in native['case'].unique():
    for stage,metric in [('cycle','synchronized_wall_ms'),('forward_stage_event_sum','cuda_elapsed_ms'),('backward','cuda_elapsed_ms'),('adam','cuda_elapsed_ms')]:
        a=native[(native['case']==case)&(native.stage==stage)&(native.metric==metric)]
        before=a[a.run.isin(['legacy_a','legacy_b'])].p50.mean();after=a[a.run.isin(['fused_a','fused_b'])].p50.mean()
        speed.append(dict(case=case,stage=stage,legacy_ms=before,fused_ms=after,latency_reduction_percent=(1-after/before)*100,speedup=before/after))
speed=pd.DataFrame(speed);speed.to_csv(HERE/'fixed_replay_speedups.csv',index=False)
for phase in [1,2,3,4,5,6]:
    summary.append(dict(phase=phase,module=['Multi-Tensor Dense Adam','Decode14 + VJP','Anchor Input27 + VJP','Anchor Full统计和校验','Selector Sampling/Compaction','Renderer workspace/SSIM'][phase-1],
        implemented=phase<=3,compiled=phase<=3,independent_parity='PASS' if phase<=3 else 'NOT IMPLEMENTED',
        fixed_replay='PASS_WITH_RECORDED_ADAM_ATOMIC_BOUNDARY' if phase<=3 else 'NOT TESTED',
        online_quality_gate='NOT ACCEPTED' if phase<=3 else 'NOT TESTED',default_enabled=False,
        switch={1:'CUDA.fused_dense_adam',2:'CUDA.fused_compensation_decode',3:'CUDA.fused_anchor_input_pack'}.get(phase,'')))
pd.DataFrame(summary).to_csv(ROOT/'cuda_optimization_summary.csv',index=False)
for name in ['before_after_benchmarks.csv','kernel_launch_comparison.csv','online_performance_comparison.csv']:
    shutil.copyfile(HERE/name,ROOT/name)
parity=pd.read_csv(HERE/'correctness_parity.csv')
parity=parity[~parity.scope.isin(['adam','decode'])]
for scope,name in [('adam_final','adam_final_parity.csv'),('decode_final','decode_final_parity.csv')]:
    x=pd.read_csv(HERE/name);x['scope']=scope;parity=pd.concat([parity,x],ignore_index=True)
parity.to_csv(ROOT/'correctness_parity.csv',index=False)
o=online[(online.stage=='total')&online.scope.eq('all_mappings_including_zero_calls')].copy()
o['variant']=o.run.str.contains('fused').map({True:'Fused',False:'Legacy'})
og=o.groupby(['phase','variant'])[['mean','p50','p95','p99']].mean().reset_index()
q=quality[quality.split=='nonbank'].copy();q['variant']=q.run.str.contains('fused').map({True:'Fused',False:'Legacy'})
qg=q.groupby(['policy','ratio','variant'])[['psnr','ssim','lpips','photo']].mean().reset_index()
fixed=pd.read_csv(HERE/'fixed_quality.csv');fixed_gap=fixed.groupby('case').psnr.agg(['min','max']);fixed_gap['range']=fixed_gap['max']-fixed_gap['min']
launch_p=launch[launch.stage.isin(['all','adam','backward','attribute_transform','anchor_input_pack','anchor_lookup','input27'])].pivot_table(index=['case','stage'],columns='run',values='launches_per_iteration').reset_index()
states=json.loads((HERE/'online_model_states.json').read_text());maprows=[]
for x in states:
    maprows.append(dict(run=x['run'],N=x['anchor']['gaussian_count'],Anchor=x['anchor']['anchor_count'],online_steps=x['mapping']['online']['mappings'],tail_steps=x['mapping']['tail']['mappings'],
        online_topology=x['mapping']['online']['topology'],tail_topology=x['mapping']['tail']['topology']))
cg=pd.read_csv(HERE/'continuous_summary.csv');cg=cg[cg.end_cycle==100]
remaining='''# 剩余瓶颈与停止条件

本轮只将 Phase 1～3 接入可选生产路径，三个开关均保持 0。在线质量验收没有通过，不能继续叠加优化后掩盖问题。

1. **P0：可复现的完整在线质量/时钟验收。** 加速使异步SLAM中的Mapping次数、拓扑事件、尾部Full/LoD次数改变。先记录并重放同一真实关键帧/拓扑事件序列，保留完整Adam状态与LR时钟；不能强制修改生产调度或伪造零梯度step。当前200步固定地图重放不代替多视图在线验收。
2. **P0：Renderer/SSIM Backward及Renderer同步。** 优化后总Backward仍91 launches；查看 kernel_execution_detail.csv 按真实kernel总时长排序。Renderer存在num_rendered/bucket_sum的D2H与Forward上下文缓冲保存。SSIM的硬件计数器被ERR_NVGPUCTRPERM阻止，不能据此判定带宽/算力瓶颈。
3. **P1：Anchor Full统计/校验。** 保留FP64 opacity统计、访问定义、slot恢复和完整assert。下一版用一次错误flag读取合并多个scalar同步，再单独验证无效owner不会先触发越界；不降低检查频率。只在独立gate后接入。
4. **P1：Gather→R3 Add→Decode。** 当前保留六组Master gather及R3 gather/add。Decode scope仍3launch，其中2个是raw logit与ST mask gather。下一版可融合gather与decode，但必须验证重复ID scatter、全表Dense Adam与未激活ratio的grad=None。
5. **P1：Stable Compaction。** Sampling/Budget/Compaction仍18launch，dynamic K和Renderer D2H仍同步。只融合比较和mask并不能声称消除K同步；prefix-sum必须保序、精确处理u==p和inactive规则，不改变Softmax或Budget。
6. **P1：Input Pack大N内存布局。** N=100K人工压力测试Forward由0.154ms变为0.184ms，出现回退；不能全规模默认启用。研究coalesced列读取/向量化，在相同真实身份与梯度fixture上重新测。N=10K/50K有效不代表100K有效。
7. **P2：单独R3 Lookup、冻结C的小GEMM。** 当前各2/4launch，收益量较小；优先合并邻接数据流，不单独重写一个高复杂度lookup/MLP。仍使用cuBLAS Encoder，不预设自研27→32 GEMM更快。
8. **暂不并行共享写。** Rasterizer/CUB/SSIM尚未证明current stream正确，保留mutex_render、默认训练顺序。不得多ratio同时Backward或Full/LoD同时更新Master。只读快照推理、pinned预取和keyframe CPU准备可在下一阶段单独测。

Phase 4～6尚未实现，不把其理论收益加入本轮加速数字。无模块被证明完全没有优化空间；R3独立lookup和小MLP当前优先级较低。
'''
(ROOT/'remaining_bottlenecks.md').write_text(remaining,encoding='utf-8')
report=f'''# Anchor / Selector / R3 CUDA 优化实施报告

## 1. 结论与启用决策

已实现、编译并接入**可切换**生产路径的模块为：Multi-Tensor Dense Adam、14D Decode/VJP、Anchor Input27/VJP。Legacy全部保留，三个默认开关均为0。

固定真实checkpoint的LoD完整训练步中位耗时降低 **27.7%～34.8%**（约1.38～1.53倍吞吐）；Full降低约5.1%。LoD的CUDA launch由**418→177**，Full由**143→96**。这不是完整在线SLAM的独占验收结果。

四次真实fr1_xyz运行中，在线Mapping均值由7.408→5.998ms，降低19.0%；Tail由13.846→11.122ms，降低19.7%。完整进程墙钟均值由52.346→45.831秒，降低12.4%（吞吐约1.14倍）。GPU始终存在其他任务，以上是共享负载下观测值，不能承诺稳定独占倍率。

**不默认开启的原因：** 在线最终地图与训练步数变化；non-Bank Full PSNR均值降低0.285dB，same-K 1%降低1.147dB。无法将其因果归于kernel误差，但也不能宣称在线质量未受影响。固定单步质量一致，200步受控连续训练末段均值差小于0.004dB，只支持局部数值/短程证据，不替代在线质量gate。

## 2. 版本、真实架构和环境

- HEAD：`4269d7e3bad6cf5af7e1df1763634b61e98e0b51`，与审计参考一致；本轮新增未提交修改。没有stage、commit或push。
- Variable Anchor 1～10成员、L0/L1/L2、Feature16、Input27、27→32共享Encoder、32→32→2 Selector、32→32→14冻结共享C。
- 生产R3：四张[N,14]独立参数表，`delta=detach(C(h))+b_ratio[ids]`；原Dense Adam，当前ratio表grad defined，其他ratio为None。未见ratio仍按原线性插值与端点规则。
- split继续采用R diag(scale)随机扰动；没有修改clone/split/prune、UID分配、Anchor growing/pruning、Loop closure迁移或Loss。
- 原始真实checkpoint：`/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply`；N=15,891，Anchor=6,121；level分布1686/4179/256。K为156/799/1582/2368。
- 隔离源码与构建：`/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022/source`、`build`。原生产source与原checkpoint保留。
- RTX3090 24GB；CUDA11.8；LibTorch2.3.1 ABI1；g++9；sm86；FP32；`--fmad=false`。Python分析使用单独Torch2.3.1 ABI0环境，未混载ABI。
- 复用原Native Fixture、Gaussian布局和真实相机；完整重放显式恢复六组Master leaf的requires_grad。仅模块压力测试的10K/50K/100K为人工平铺，不能当作独立场景。

## 3. 调用路径与代码

|模块|文件 / 函数|接入位置 / 范围|
|---|---|---|
|Fused Adam host|include/mapping_cuda_optimizer.h；src/mapping_cuda_optimizer.cpp::tryFusedDenseAdam|GaussianMapper::stepWithHistoricalGuard，Guard未启用分支；不支持时整步fallback|
|Multi-Tensor kernel|src/transform_field_cuda_optimizer.cpp::tfCudaAdamDirect；src/transform_field_cuda_optimizer.cu::adamKernel|复用已存在的FP32多Tensor内核，本轮未改TF实现；每批最多32Tensor|
|Decode autograd|include/anchor_selector_cuda.h；src/anchor_selector_cuda.cpp::Decode14|accumulateLodGradients的14维属性decode，opacity ST gate只执行一次|
|Decode kernel|src/anchor_selector_cuda.cu::decodeForward / decodeBackward|各一次kernel，保留外部base属性activation和index_select/scatter|
|Input27 autograd|include/anchor_input_cuda.h；src/anchor_input_cuda.cpp::Input27|GumbelNetworkImpl::anchorInputs，经CARTGS_FUSED_MAPPING_CUDA编译定义接入|
|Input27 kernel|src/anchor_input_cuda.cu::pack / vjp|Feature16、Offset3、Scale3、DC3、Opacity1、logratio1；Feature用atomicAdd累加|
|配置和导出|include/gaussian_mapper.h；src/gaussian_mapper.cpp::readSettings / saveSelectorConfig；cfg/selector/selector_config.yaml|三个独立开关；冻结teacher复制InputPack开关；不改模型参数schema|

Header-only CPU单测不引入CUDA符号依赖；CMake将CARTGS_FUSED_MAPPING_CUDA作为transform_field_cuda的PUBLIC定义传播给生产和相应CUDA测试。原CPU测试仍可使用Legacy。

## 4. Phase 1：Dense Adam设计与正确性

整步预检CUDA FP32、contiguous、gradient/moment形状及device、AMSGrad、重复参数和重叠storage。预检失败在任何参数/state变更前返回false。CPU、非连续、AMSGrad、重叠view保留Legacy。

每个参数使用自身group的lr、beta、eps、weight decay以及AdamParamState.step；生产epsilon从options读取，为1e-15。grad=None不创建state、不递增时钟；梯度为0的未选行仍衰减历史m/v并更新参数。不是SparseAdam，不是selected-K更新，也未改成离线R3_matched。

复用现有32-Tensor descriptor，动态拓扑之后每次从当前optimizer组重新取Tensor，未缓存过期指针。更新保留parameter/m/v version bookkeeping。当前活跃leaf少于32，Full/LoD均只提交1个Adam kernel。

最终固定梯度测试 **13,640项**，参数/m/v的max absolute、relative、RMS误差均为0，bad_elements=0；仍使用预注册atol=2e-7、rtol=2e-6。覆盖五个真实fixture连续30步、100步1e-10量级梯度、零梯度历史moment、grad=None、独立ratio时钟、不同LR，以及67Tensor跨三批、beta(.8,.98)、weight decay、eps1e-15/1e-8切换和fallback。

拓扑测试额外启用Fused，在真实Gaussian append/clone/split/prune、Feature/dual Adam迁移后继续执行step；R3行、UID、optimizer alias、m/v尺寸和clock检查通过。完整六组Master与R3生命周期没有改规则。

最终额外重叠storage保护只增加CPU预检；正式整步/在线计时来自加入此防御检查前的候选，CUDA数值/launch完全相同。最终版模块重新编译并复测；不把早期整步时间说成这次CPU防御分支的精确最终耗时。相关版本源码均保存于evidence。

## 5. Phase 2：Decode14/VJP

输入为选中K行的XYZ、activated Scale、**已由原getRotationActivation正规化的base quaternion**、raw opacity logit、DC、delta14与ST mask。输出为XYZ、Scale、Rotation、gated Opacity、DC，保持[K,3]/[K,4]/[K,1]/[K,1,3]布局。

Forward保留XYZ相加、Scale乘exp、identity+delta quaternion归一化、quaternion乘法和再次归一化、sigmoid(logit+deltaOpacity)、DC相加、ST opacity gate。没有给base rotation额外加一次归一化。Backward保留全部base属性、delta14和mask的VJP；未被使用的输出对应grad=None，不制造零梯度leaf推进Adam。

首版独立算子通过，却在真实Renderer的5%/10%各3个通道出现超容差像素（max分别0.000688/0.001289）。原因定位到ATen归约树：norm4为(a0²+a1²)+(a2²+a3²)，三项点积为(a0b0+a2b2)+a1b1。已按真实ATen顺序修正，旧失败证据保存于reduction_v1_failure，**没有放宽容差**。

最终179项Decode检查全部bad_elements=0。真实属性Forward除rotation外可精确一致；rotation仍有约1e-7量级差异，完整render通过atol2e-5/rtol2e-4。覆盖K=0/1/7、identity、tiny/精确零quaternion、±100饱和logit、极大Scale/XYZ、极小Scale、非连续输入、重复ID的Backward scatter和unused output。精确零quaternion的Legacy NaN VJP也原样保留，不在性能优化中修复算法边界。

## 6. Phase 3：Input27/VJP

单kernel直接读取当前owner、Anchor Feature/center/reference scale与Gaussian属性，生成连续[N,27]；Offset=(xyz-center)/reference scale，不clamp、不缓存跨step变化的数值。原Encoder、Softmax、sampling和budget代码不变。

Backward按owner atomicAdd Feature16梯度，直接回传XYZ/reference scale、activated Scale、DC与activated Opacity梯度，再由原activation回传Master。center/reference scale继续是非训练buffer；没有增加它们的优化路径。支持非连续输入的contiguous拷贝，CPU/不支持Anchor布局走Legacy。

84项真实Variable Anchor检查通过，Input27、Encoder输出、概率在fixture上精确相同；四档Cached ID/顺序/K精确一致。重复owner梯度聚合有FP32 atomic顺序差异，max absolute约1.91e-6，预注册容差内。覆盖1/10成员、owner重排、长尾Offset、空集及非连续输入；实际拓扑单测同时验证变更后continued step。

大规模人工测试发现100K Forward回退（见第10节），故不能把真实15.9K的收益外推到所有地图规模。

## 7. 固定checkpoint完整训练步耗时

每分支10次warmup，60个正式完整step；两次Legacy、两次全候选，另保留Adam-only与Adam+Decode中间版本。表为两次run各自p50的均值。`forward_stage_event_sum`是各Forward scope的CUDA Event span之和，包含CPU提交空隙/外部排队，不是纯kernel时间；cycle为同步墙钟。正式计时与Nsight分开。

{table(speed)}

Profiler实际GPU kernel时间与CPU API/queue分别保存在kernel_execution_detail.csv、cuda_api_detail.csv；不能把microbenchmark耗时占比直接当作生产在线占比。

## 8. CUDA launch变化

{table(launch_p)}

LoD减少241次：Adam -111，Backward整体 -91，Input lookup+pack -8，Decode scope -31。Decode scope的3次包含raw logit gather和ST mask gather各1次；Decode自身为1次。R3 gather/add仍为2次，Selector MLP/Softmax/Sampling没有重写；不将相邻模块收益归功于R3 Lookup。

## 9. 完整在线SLAM、Tail与质量

运行顺序Legacy-A→Fused-A→Fused-B→Legacy-B，同一binary、数据、seed=2026和默认配置，仅改三个实现开关。自然停止，未设Mapping10000，serial_full_lod=false，TF关闭。每Mapping低开销trace保持一致；没有同时跑多个自己的训练进程。

以下是两次运行统计量的均值，单位ms。online/tail采样数不同，不视作同一训练时钟。

{table(og)}

{table(pd.DataFrame(maprows))}

加速改变异步SLAM在相同轨迹时间内执行的Mapping次数，在线topology7→9；Legacy还进行1次Tail topology而Fused没有。因此地图、Full总次数、R3各ratio训练次数和tail预算均不相同。原有调度规则没有修改，但最终训练序列没有保持相同，不能以这些结果证明等价算法质量。

评价复用先前预定64个RGB帧，按各自真实估计pose，四个模型Bank并集排除后有51个non-Bank帧。与先前离线70/44协议不是同一视图划分，不混合相减。四模型统一common-K=129/645/1291/1936；same-K沿原cached-uniform margin排序。保存2304条逐视图记录，PSNR/SSIM/LPIPS/photo均报告。

{table(qg)}

**在线质量gate没有通过。** non-Bank平均Full -0.285dB；same-K四档分别约-1.147/-0.240/-0.451/-0.373dB；1%的一次候选明显较差。不能将不同地图的退化直接定性为CUDA错误，也不能用Cached K更大掩盖common-K退化。两个重复来自同一场景，不是独立场景验证。

## 10. 显存、分配与规模

真实模型未增加训练参数、R3表或Adam状态；三个候选只改变计算/中间分配。人工100K/5K下的allocator峰值额外分配：Dense R3 Adam Legacy约11.20MB→Fused0；Decode Forward约0.786→0.282MB，31→5次allocation；Decode Backward约0.622→0.583MB，60→7次allocation；Input Forward约22.00→10.80MB，8→1次allocation。Input Backward为了显式返回各base梯度，峰值约1.20→4.39MB，2→5次allocation，不能宣称全部内存项下降。

人工Input Pack Forward p50：10K 0.084→0.020ms，50K 0.112→0.073ms，100K **0.154→0.184ms**（回退约20%）。100K Backward 0.177→0.158ms。现有每线程一Gaussian的访存方式仍需按大N优化，不应全规模强开。

在线allocated峰值（项目计数器MB）Legacy-A/B为486.336/471.254，Fused-A/B为466.050/467.997；reserved峰值分别566/530与526/536。不同N/拓扑的进程峰值不能视为kernel本身的显存因果对照。外部负载保存在gpu_samples.json；人工allocator数据也不是整个在线进程峰值。

## 11. 完整Backward、Adam放大与连续训练

固定重放共3,945项比较：除下一段明确记录的parameter_after放大外，Forward、全部六组Master、Anchor Feature、Encoder、Selector、当前ratio R3梯度、其他ratio grad=None、m/v、clock、UID、owner、ID/K均通过原定gate。候选完整梯度max absolute约3.82e-6，RMS最大约1.67e-8，bad_elements=0。不能将这里的gate表述为全部post-step参数逐元素一致。

不将parameter_after差异藏进宽容差：Baseline-repeat已在1% Scale出现约5.54e-4的单元素Adam放大，候选约5.64e-4；R3约6e-5。eps1e-15会放大atomic造成的微小梯度差。固定grad/moment的最终Fused Adam为严格0误差；完整链parameter_after仍逐项记录，未宣称bitwise训练一致。

`worst_error_locations.csv`补充每项Tensor最大误差的flat/多维索引、参考值、候选值，以及可对应时的Gaussian row/UID、Anchor owner、base XYZ/Scale/Rotation/Opacity输入，供复现tiny-gradient放大或渲染边界。

补充四个连续分支：同一真实地图、相机、cached uniform、固定拓扑、fresh且相同Adam初态；每轮Full→轮转一个LoD，共100Full+100LoD、每ratio25次。持续更新参数/moments，不恢复权重，不做伪零梯度step。末5轮汇总如下（LoD混合ratio，不能当各档指标）：

{table(cg[['run','kind','photo_mean','psnr_mean','K_mean']])}

末段PSNR均值差<0.004dB、UID/owner/时钟精确一致；连续参数不满足strict allclose，99步时Baseline-repeat与一个候选均出现选点分歧。此结果支持短程没有明显系统性质量下降，不证明多视图长期在线等价。没有以验证集调LR/步数或选择best checkpoint。

## 12. Phase 4～6与并行化边界

Phase4的Anchor Full FP64统计/完整检查、Phase5 Stable Compaction、Phase6 Renderer workspace均**未实现**。在在线质量gate未通过时不继续叠加生产变更；保留原同步、统计、prune阈值和member规则。

Renderer的num_rendered/bucket_sum D2H、dynamic K同步、六组gather、R3全表Dense Adam读写仍存在。没有删除mutex_render，没有四ratio并行Backward，没有Full/LoD并发更新Master，也没有把只支持默认stream的Renderer强迁多stream。

Nsight Systems成功采集kernel、API和排队；Nsight Compute最终得到ERR_NVGPUCTRPERM，未取得硬件计数器。首次NCU还遇到LibTorch linalg动态库路径问题，补齐后确定权限阻塞。没有据此宣称SSIM受memory bandwidth限制，也未重写SSIM。

Nsight的5%候选重放中，纯kernel时长之和约2.595ms/步，其中SSIM Backward约0.799ms（30.8%，最大单kernel），duplicateWithKeys约0.212ms，SSIM Forward约0.199ms，PerGaussianRenderCUDA约0.187ms。只取3次被Profiler插桩的step且存在外部负载，不能代替正式端到端计时。该数据将下一轮kernel级优先级首先指向SSIM Backward，但其硬件瓶颈类型仍未确定。

## 13. 可复现命令和交付证据

诊断目录：`tests/diagnostics/anchor_selector_r3_cuda_optimization_20261022/`。脚本包括setup_build.py、run_validation.py、verify_replays.py、module_scaling.cpp、controlled_training_replay.cpp、run_continuous.py、evaluate_online.py、analyze_results.py与export_delivery.py。服务器完整证据包delivery_evidence.tar.gz约2.1GiB，保留各分支Tensor快照；本地compact_evidence.tar.gz包含完整CSV/日志/Nsight/source SHA以及一个5%候选完整fixture，旧Native fixture继续复用原审计目录。最终源码SHA见final_source_manifest.json/local_source_manifest.json；服务器保存完整在线checkpoint，不清理原始模型。

服务器构建/单测示例（不要混用Python ABI0库路径）：

```bash
R=/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022
export LD_LIBRARY_PATH="$R/source/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib"
/home/prmi/.local/bin/cmake --build "$R/build" -j2 --target fused_mapping_adam_test fused_compensation_decode_test fused_anchor_input_test optimized_native_benchmark tum_rgbd
"$R/source/bin/fused_mapping_adam_test" /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_adam"
"$R/source/bin/fused_compensation_decode_test" /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_decode"
"$R/source/bin/fused_anchor_input_test" /home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply /home/prmi/zwj/anchor-selector-r3-performance-20261022/native1 "$R/recheck_pack"
```

重放最后一个参数bits：0=Legacy，1=Adam，3=Adam+Decode，7=全部。现有run_validation.py会保留同名目录，不适合直接覆盖旧证据；需要新隔离输出根运行。默认生产开关保持：

```yaml
CUDA.fused_dense_adam: 0
CUDA.fused_compensation_decode: 0
CUDA.fused_anchor_input_pack: 0
```

非支持布局自动Legacy fallback；14D以外的旧补偿路径也保持Legacy。任一实验flag均不改变R3表/ratio插值/checkpoint schema。

## 14. 对八个验收问题的明确回答

1. **整体Mapping提高多少？** 固定地图LoD步延迟降低27.7%～34.8%，Full约5.1%；共享GPU真实在线单Mapping均值降低19.0%，Tail19.7%，完整进程12.4%。不同在线步数/地图和外部负载限制因果解释，尚无生产默认GO。
2. **LoD Forward/Backward/Adam分别？** 四档Forward scope Event span降低10.3%～18.6%，Backward降低28.7%～35.5%，Adam约92.3%～92.5%；对应详细表和run波动保留，纯kernel时间另表。
3. **Anchor/Selector/R3 launch？** Anchor input lookup+pack9→1；Selector MLP/Softmax/sampling未变；R3 lookup/add仍2，decode scope34→3，LoD Adam112→1，总418→177。没有虚报R3 lookup单独加速。
4. **同步和访存空间？** 仍有dynamic K、Renderer D2H、多组gather、Anchor检查scalar同步、大N Input Pack不连续访存；Dense R3全N moment更新属于当前语义，不可通过只更新K行消除。
5. **继续融合哪些？** 优先在独立gate下研究selected gather+R3 add+decode，再研究Anchor检查/Full统计、sampling/稳定compaction；不先重写Renderer全体或自研小GEMM。
6. **哪些不值得继续投入？** 当前单独R3 lookup、小共享C的GEMM不是优先热点；没有证据说这些模块完全无空间。Input Pack大N尚有反例，不能结束优化后全开。
7. **在线SLAM质量受影响吗？** 当前实际在线对照有退化，质量门未通过；固定单步质量一致及短程曲线不能覆盖这个结果。保留Legacy默认。
8. **最大剩余瓶颈？** 技术热点是Renderer/SSIM Backward、Renderer/动态K同步和Anchor Full/Topology处理；最大的接入阻碍是速度改变异步训练序列后的可比较在线质量验收。必须先建立同真实帧/拓扑/优化时钟的记录重放再给生产GO。
'''
(ROOT/'ANCHOR_SELECTOR_R3_CUDA_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
print('PASS generated report and required six machine-readable outputs')
