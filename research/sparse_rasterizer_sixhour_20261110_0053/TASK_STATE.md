# 六小时研究连续状态

开始：2026-11-09 16:53:10 UTC；首45分钟截止17:38:10；候选冻结目标21:38:10；最终截止22:53:10。预算是实际墙钟，不是token。goal已创建且active，不要提前标记complete或暂停。

本轮根目录：服务器 `/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053`，本地为当前checkout的同名research子目录。分支main、HEAD 26c355b32bf373a75877fc4605a4e41b8de9b47f；其他未提交修改不动。不commit/push/部署，不改生产、训练、原模型、共享conda，不启动子Agent。

SSH bridge PTY为74656，JSON行协议op=exec/put/get/quit。桥接凭据只在进程内存；禁止写文件或输出。exec后台任务用本轮launch.py，避免shell的cd&&cmd&持有SSH channel。运行Python `/home/prmi/anaconda3/envs/zwj/bin/python`，CUDA_HOME=/usr/local/cuda-11.8，sm86。服务器没有rg，使用grep/find。

A原入口：`/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so`，sha27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86；与实际运行Viewer的q80副本二进制hash完全相同。B上一版：`/home/prmi/zwj/experiments/sparse_rasterizer_balanced_20261110/build/sparse_rasterizer_balanced_v1.so`，sha349a4c957895bce9c1929d50fbd14878df13a036c2c540d414da351f250a19b8，mode4。本轮直接按.so导入B，禁止触发旧目录JIT重构建。新C为mycartgs_sparse_sixhour_v1，C++类SixHourRasterizer，防止全局pybind冲突。

基线与场景已锁定：reports/PROTOCOL.json、SCENES.json、PROTECTED_BEFORE.json（1037项）。24场景：Replica16，TUM desk RGB-D/room RGB-D/xyz RGB-D/fr2_xyz mono，VECtor corner mono/stereo、corridors stereo、robot mono。调优5组：Replica office1 mono、office2 RGB-D、room0 RGB-D、TUM desk RGB-D、VECtor corner mono；其余19组保留，冻结前不测候选性能。调优与最终的连续轨迹窗口及跳转视角分开。不得逐场景专用参数。

当前首批候选：C0控制；C1_shared使用每Tile共享内存预取属性；C2_warp使用每Warp 32成员寄存器预取+shuffle。所有合成运算及深度次序保留。列表容量默认128，可按独立配置验证32/64/256，溢出完整目录枚举。新源位于source/sparse_inference.cu，预处理在forward_sparse.cu。每实例一个device/Stream，未缓存上一帧结果。

已发现历史B边界缺陷：所有Gaussian都被裁掉、背景含-0.0时，原入口直接复制背景，B做0+1*bg导致符号位改变，最大绝对误差0但bitwise失败。本轮不改B；新C通过sorted_depth_keys[0]==0xffffffff识别真正全不可见，直接复制背景，修复A语义。不可以按单Tile count==0直接复制背景，因为全局存在其他可见点时原入口应保留原表达式。P=0原始CUDA primitive仍返回零，外层产品背景特殊处理须保留。

screen01失败现场保留；screen02已通过792条对比，C0/C1/C2均与A逐位一致，只有B已知负零边界失败。覆盖同深度、P0/1/31/32/33/127/128/129/1023/1024/1025/4097、非整Tile分辨率97x65和641x479、重叠/随机/全不可见、NaN XYZ/Inf Scale，以及5个调优场景新视角。screen02输出reports/screen02/RESULT.json。

正在运行 micro01（launch.py记录runnerPID804375，实际以reports/jobs/micro01.json为准），命令micro_screen.py --output micro01 --validation screen02。仅5个调优场景，local1/5/10/70%及Full，6视角、4重复，每重复24次。A/B/C0/C1/C2轮换次序；CUDA Event另测。看logs/micro01.log和reports/micro01/SUMMARY.json；不要只看JSON就判断GPU正常。编译或测试进行时不要改写同一CUDA源/二进制。

launch.py检查本轮其他运行任务的PID/start_ticks，job_runner.py保存退出码及1Hz外部GPU监控。--measure需放在任务名之前：`python launch.py --measure 名称 python -u 脚本 ...`。不要重用已存在任务名或输出目录；失败保留现场。当前其他驻留GPU进程174824（Viewer）和686016（其他TUM评价），另有其他队列553923，均不操作。最近GPU利用率0%。

后续按真实热点：先读micro01结果，独立profile确认共享/寄存器候选；再研究预处理直接输出depth key/id（去掉depthKeys kernel）、实际footprint的并行Tile展开、避免连续化复制及有界Workspace/计划开销。先正确性再微测，不合格候选记录原因继续下一处。尚未实现actual Alpha参与唯一点数诊断，需在计时外全像素统计并保留原Alpha阈值/早停。正式测试还需高/未见Ratio、固定预算、全24场景、连续/跳转视角、每帧重新准备、p95、两个独立轮次、Sanitizer和完整性。

## 17:41 UTC更新（已用约48分钟）

首45分钟核查已完成，AUDIT_COMPLETE.json在本地和服务器。已多轮独立实现并验证：C1共享属性、C3预处理融合depth keys、C4相机stride在预处理内以shared装入35标量（省2个contiguous kernel）、C5 footprint任务队列（area>256拆成256 tile块，GPU计数，固定SM×4 workers，无K/ratio分派）。C2 Warp寄存器方案被拒绝，Full有回退。

最优已测组合C1345：render_variant=1,fused_keys=True,strided_camera=True,build_variant=1。screen06通过2178条比较；micro05调优集1/5/10%对B加速1.236–1.436×，Full 1.068–1.290×。这只是调优结果，不得当通用性已通过。micro03 stride单独低ratio约1.05–1.13×。micro02被Xorg干扰中止，micro02_retry01通过；其他micro01/03/04/05外部负载监控0。所有完整输入在reports/micro*/samples.csv、SUMMARY等。

Nsight工具实际有：/usr/local/cuda-11.8/bin/nsys(2022.4.2)与ncu(2022.3)。nsight01已成功采集低ratio少量区间，正确report名为gpukernsum,cudaapisum,nvtxkernsum（新命名cuda_gpu_kern_sum不支持，失败日志保留）。nsys-rep和SQLite在evaluation，summary在reports。此Nsight运行有Xorg样本不能用作正式速度。kineto01独立profile无外部活动，证实1% buildBits≈33us，仅20CTAs；C1 pixel kernel≈49us vsB≈58us；C13少一个depthKeys kernel。真实分辨率Replica1200x680。

alphaDiagnostics已加入最新source（screen07编译版），仅debug路径新增inspect[13]=预处理feature，[14]=sorted depth keys；extension().diagnose_alpha提供实际Alpha参与unique flags与交互总数，另输出诊断图要求与渲染逐位一致。diagnostics_check.py已上传但尚未跑，推荐在screen07结束后运行（只调优集）。未进入计时。

正在编译/验证screen07：新增C6自定义整数稳定排序（blockRadix1024+merge path；所有P都走此通用算法，不按K切回CUB），及C13456组合。flags sort_variant=1。测试可能Full较慢，需先正确性再micro。每次改.so后旧micro不重跑除非需要比较新candidate。

当前API顺序最后参数为mode,short,inspect,render_variant,capacity,fused_keys,strided_camera,build_variant,sort_variant；runtime.py已对应。C0与C2等旧对照保留，实际候选配置在micro_screen.py。单独源模式0/1/2使用原路径，只有mode4供最终研究。现在工作区大约<30MB，磁盘15GiB。

下一有依据候选可考虑shared属性的2/4个Alpha提前计算流水线（仍按原顺序更新T和C，逐位验证后才接受），block/微Tile调度，或安全CUDA执行计划；不要用同帧内容缓存。保留场景尚未用于本轮性能调参。

最终硬门限按用户：Full每场景两轮中位不得慢于A，另报相对B最快已验收Full变化；本轮预先追加推荐门限为Full还不能慢于B。低Ratio新增收益必须对B报告，不可复用旧收益。所有失败、干扰轮次保留，不变门限。最后只保留必要二进制/源码/补丁/计时CSV/少量Profile及中文报告曲线；仅可清理本轮可重建中间产物并留清单。

17:45更新：screen08已完成3456条对比，新增alpha2/alpha4流水线和512-block稳定排序均逐位通过。正在micro07(variants C1/C7/C8/C9)，无其他本轮job。micro06自定义1024块整数sort在低K无普遍收益，大Full偶有加速；尚未决定保留。alpha01诊断30输入逐位通过，workloads见reports/alpha01，已含unique实际参与点、交互量、Tile成员分布。当前stage使用约53分钟，仍有5小时7分钟墙钟。

18:06更新：micro07拒绝alpha2/alpha4流水线，无普遍优于C1收益，TUM Full退化；512稳定块排序C9比1024更好。micro08验证cap256比cap128在Full更好且低Ratio基本不变。micro09两像素线程rv5比rv1小幅更好，四像素rv6多个Full退化，拒绝。micro10当前最佳常规配置CbestPair={render_variant:5,capacity:256,fused_keys:true,strided_camera:true,build_variant:1,sort_variant:2}。1%对B约1.46–1.69×；Full对B约1.34–1.57×（调优集）。C12_metadata复用预处理bbox/log阈值通过screen10但低ratio多数更慢1–5us，不推荐。所有source和runtime远端已一致。
正在graph01：新的独立模块mycartgs_sparse_graph_study_v1，build_graph；class CapacityGraphRasterizer。sources由generate_graph_sources.py从常规已验证数学核生成source/graph_forward.cu、graph_kernels.cuh，另source/graph_main.cu和graph_inputs.h。min512 nextPow2容量，padding无效Gaussians只走初始化，稳定排序扩展后不改变实际ID顺序；每帧GPU更新所有输入/相机/输出指针，缓存8个执行计划，内存1GiB上限，bitset单计划256MiB，LRU。所有输入值每帧读取，不复用同帧Mask或Feature。graph_check.py编译+验证尚未看最终，不能宣称成功。若失败检查logs/graph01.log。
Nsight01监控原报警实际为自身派生PID820984，已从日志核对，不是Xorg；NCU01报警同属派生846585，计数器权限ERR_NVGPUCTRPERM真实存在。job_runner已排除实际子孙进程，原日志保留不覆盖，报告不得误称该两次为外部负载。只有micro02是真Xorg干扰。

18:16更新：容量Graph已实际编译并通过graph03的482项检查（随机/同深度/全不可见/非有限、预计算RGB和协方差、形状padding、18份输出存活、8计划淘汰、nondefault Stream）。graph_micro01已完成11200样本（5调优集，16视角，A/B/CbestPair/G两类scope），G相对常规C低ratio再快约2–5%，pipeline约1–3%；但power2填充使Full比C慢1–19%，仍比B快。cold首次调用另记cold_samples.csv，19次miss/400次初始请求。为减少Fullpadding，新增plan_granularity=512（通用512粒度，不按ratio/K绕开算法）；graph04已结束需读结果，再跑graph_micro02 --granularity512。此时独立Graph模块编译source并不跟随后续常规source变化，保持解耦对照。
常规C13_tilefused正在screen11编译验证：render_variant=7,cap256,keys/stride/tasks/sort512，首Warp在renderBlock里按原顺序枚举成员到shared再渲染两个像素，省compact单独核，overflow保留目录完整枚举。global_lists仅inspect写，counts/directory正常写，guard全不可见时也初始化debug计数。尚未测速。之后micro11对CbestPair/C13_tilefused。

## 18:50 UTC连续更新

已恢复既有SSH bridge 74656，goal active，截止仍为22:53:10 UTC，未重置预算。micro11拒绝Tile枚举/渲染融合：低Ratio无稳定收益，多场景Full更慢。图计划最新为共享Workspace+std::map容量索引，常规默认64计划，可配置至512，scratch总1GiB与bitset256MiB上限。graph06通过483项；graph_micro03共享池512粒度重复12s测试通过11200样本/2400项，1%比C再快2–5%，Full快1.5–4.2%，没有外部负载。

graph07新增graph_input_update独立开关，把参数更新核放入图、每帧cudaGraphExecKernelNodeSetParams，483项逐位通过。graph_micro04 14000样本表明H方案无稳定收益，大部分比G慢0–2us，不推荐；最终继续graph_input_update=False。

current_profile01在Kineto采集G时Collection挂起，GPU0%、主线程futex、日志>130s无进展。已核验boot_id/PID/start_ticks后仅SIGTERM本轮914899，PROFILE_INCIDENT.json保留现场。current_profile02跳过Graph的Kineto（Graph改Nsight），30条记录成功、无外部负载。office2 1% C=76.5us kernel累计/6核，pixel46.5us、sort7us，预处理8、schedule5、execute5、compact5。Full C=251us/13核，pixel125.75、compact34.62、merge32.25、schedule19.75、execute15.12、preprocess13.62。全像素Alpha诊断alpha01显示1%几乎覆盖全部Tile，p50成员7–20，不能靠空Tile专项获得大收益。

正在precompile01：Graph新增precompile()，模型加载时仅按地图容量+分辨率预编译全部512粒度计划，不读取视角内容；max512。预期让未见视角/Ratio不发生首次建图。注意实际状态看reports/jobs与日志。

新常规sort01编译失败原因明确：BlockRadixSort radix8的静态shared0x10220>48KiB；日志保留，已把同128×4布局候选改radix7。当前本地source已修改尚待上传与sort02：C14_radix6 sort3，C15_radix7 sort4，C16_blockmerge sort5；所有规模统一算法，完整depth bits+input id稳定顺序。sort_check.py新增直接键/身份核对及RGB/radii逐位；micro_screen.py配置已扩展。不要重复跑旧候选全套。图核仍来自此前稳定快照，不随常规源自动变化。precompile01后依序跑sort02与micro12。

本地git现有11项tracked dirty（selector/mapper/viewer等）均是此前用户修改，本轮未动；只写当前research目录。服务器磁盘14GiB，原生产与24模型保护清单1037项不变。

## 19:42 UTC关键更新：最终验收发现并修复旧Tile coverage误差

截止仍22:53:10 UTC。SSH bridge 74656仍可用。没有子Agent，没有触碰生产。最新rootcase非常重要，不能把此前数千次bitwise通过当普遍证明。

新增sort01 radix8因static shared>48KiB编译失败；改radix7后sort02通过732项，micro12 radix6/7/blockmerge均无普遍稳定收益，不推荐。C17预处理+Block排序融合通过366项，但micro13 Full慢2–5%、5/10%慢5–13us，拒绝。C18/19 cache128/cache64通过612项，micro14各有反例，拒绝。C20一Gaussian一CTA通过408项，micro15低Ratio部分快1–7%但Full部分慢8%，不推荐。最终性能配置仍CbestPair与512粒度Graph。

precompile01通过801项，模型加载只按N与W/H编译全部512容量计划，scratch共享、最多512计划，0.4–8.6ms一次性，4000次变视角请求motion01均0miss。edges01 826项，sanitizer_mem01 memcheck232项0错误。原有全无Alpha但bbox<=256的负零符号位缺陷已加copySparseBackground修正，flag写在bits末尾第二个uint32，inspect新增out[15]。不改变大bbox但全无Alpha时原版0+T*bg语义。Graph与C都已修正。

19:26提前冻结v1并启动formal01（全24×14条件×18视角），仅运行20秒在heldout Replica/office0/rgbd frame1291 local.05、K611处停止：B与A RGB差3个值(1像素447,798)，maxabs0.0009471029。reproduce01/02/03证明B/C/G同一差异，原serial模式与A一致。原A Full forward与inference RGB逐位相同；解析A实际geometry/bin buffers发现C的xy/conic/depth/rgb/bbox和排序均相同，但tile2074多Gaussian459。该Gaussian xy[1227.35144,834.35742], conic[1.49056768,-1.65168738,1.83029652,1], radius523，极薄长footprint导致覆盖判定对舍入敏感。

真实SASS已核对：A duplicateWithKeys和C emitSerial完全同一FMA方向；任务核原C++表达式被nvcc改为另一乘积融合方向。新增source/ordered_tile_coverage.cuh用__fmul_rn/__fadd_rn/__fmaf_rn固定A的计算顺序。尤其tx numerator应FMA(diffy,b*dx,round(diffx*(a*dx)))，不能FMA(diffx,a*dx,round(diffy*(b*dx)))。This is exact baseline semantics, not an epsilon tolerance. 原版二进制SASS在reports/ORIGINAL_TILE_SASS.txt（第一次--function警告其他fatbin无符号，但文件确实含目标函数）；全文另一份ORIGINAL_TILE_SASS_FULL.txt在远端。C的全SASS在CUDA_SASS_DIAGNOSIS.txt（下载本地，最终可保留摘录压缩删除可重建全文并记录）。

独立开关：SparseRenderer(...ordered_coverage=True)是C；GraphRenderer(...ordered_coverage=True)是G，GraphNative构造现在(int max_plans,bool ordered)。新helper仅build_variant1适用。默认False保留失败实现作对照，最终冻结必须True。ordered01重新实际编译(113s)并证明该rootcase C/G RGB、radii、完整Tile列表均与A bitwise，B仍保留历史误差不修改。coverage01直接解析原A sorted tile/depth键和ids，重建原位表；91份输入(5调优组+office0rgbd rootcase)全部Tile位、RGB/radii一致。不是只用PSNR放行。

旧冻结已保存为reports/FROZEN_CANDIDATE_V1_REJECTED.json与V1_REJECTION.json。当前没有有效FROZEN_CANDIDATE.json。正式v1所有数据保留为失败，不作为最终速度。新C/G二进制已改变，需source identity重新冻结与两轮正式测速，不能复用旧timings。旧B只能作为历史速度对照，其数值失败必须逐项披露；新C/G唯一数值oracle是A。

当前正在edges02：final_edges.py --output edges02 --ordered（PID/身份读reports/jobs/edges02.json）。后续依次 sanitizer_mem02 --ordered、sanitizer_init02 --ordered、motion02 --ordered、formal_smoke02 --ordered，完成后freeze.py --revision2（本地最新版尚待上传），再正式两轮建议formal02 round1、formal03 round2（不要重用formal01）。最终all-scenes coverage_check.py --all-scenes需跑。注意本地final_edges、motion_stress、final_benchmark都已更新并上传，freeze.py版本2还没上传。

final_benchmark.py已修正轮换顺序：按四版本一个quartet后才逆序，避免index奇偶与rotation耦合。旧版formal01已因数值错误失败；新两轮均用修正后的次序。对B记录RGB/radii差异、maxabs/maxrel/different_values但不把B当数值oracle；C/G不相同立即assert。每帧prepare+render还比较同版本reference（B自身作为B引用），选点ids都须一致。raw samples可复现，诊断在计时外。正式pipeline每次重新构造并上传相机，不用同帧缓存。

最终脚本已完成但未运行：final_benchmark.py、freeze.py、final_edges.py、coverage_check.py。Nsight最终v1在nsight_final01，节点级CUDA Graph捕获成功191KiB，0外部负载；因coverage代码变更最终v2需另采一次，不能复用v1的profiling当最终。NCU仍ERR_NVGPUCTRPERM，无权全局改驱动。Kineto不能捕获G（会futex挂起），仅C/A/B用Kineto，G用nsys --cuda-graph-trace=node。

后续尚需：新候选验证链、冻结v2、24组两轮严格配对Full gate、p95/波动、19保留组，不逐场景调参；all-scene Tile与Alpha诊断；最终Nsight/事件；输出中文报告、CSV、曲线、1037项保护哈希与二进制身份；仅清理本研究可重建object和冗余Trace（路径归属核查+清单），不删模型。H:/Anaconda3/python.exe有numpy2.1.3/pandas2.2.3/matplotlib3.10可本地画图；默认python没有numpy，不用它画图，不安装共享环境。已读figures-python/environment-setup技能，实际绘图采用本地已有依赖，绘图开始时告知技能名，PNG450dpi+SVG+数据manifest；不需要新conda。
