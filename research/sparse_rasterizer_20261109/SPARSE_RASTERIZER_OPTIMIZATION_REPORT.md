> 2026-11-10补充：两轮空闲GPU正式复测已完成，见[空闲复测报告](SPARSE_RASTERIZER_IDLE_BENCHMARK_REPORT.md)。以下并发诊断保留为历史记录，正式速度以新报告为准。

# 低点数Rasterizer CUDA实现与诊断报告

## 结论与证据范围

已在独立研究目录实现三项候选：稳定Warp展开、短Tile列表Warp混合、全局稳定深度排序与Tile Bitset。没有按Gaussian数量设置新旧路径切换门槛；三项实现对高点数也执行测试。正常训练、生产默认、Selector、Compensator和模型保持不变。

FP32逐像素及radii通过 500 组逐位比较，动态形状、输出独立、预计算RGB/协方差、SH与CUDA Stream通过 45 项检查。Tile成员及同深度顺序独立核验。结果相同意味着匹配输入下PSNR、SSIM、LPIPS相同，不以接近的PSNR代替逐位检查。

用户已明确选择“先完成并发负载下的诊断，正式测速稍后”。服务器其他TUM评价与Viewer进程保留；负载采样记录在LOAD.json。因此下面速度只用于候选筛选，不是空闲GPU的正式加速结论。

并发诊断中，1%/5%/10%条件的Bitset相对原入口加速范围 2.66–3.65×，相对同研究入口控制组 2.66–3.54×。Full结果单独报告，避免用低点数收益掩盖高密度退化。

## 源码与运行边界

服务器 `/home/prmi/zwj/experiments/sparse_rasterizer_20261109/`；本地对应research/sparse_rasterizer_20261109。参考来自实际使用的global_view_joint_lod_20261105/source/cuda_rasterizer及其只读推理二进制，原二进制SHA256为27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86。

`source/sparse_inference.cu`实现独立推理入口和三个算法；`generate_forward.py`从只读参考源生成`source/forward_sparse.cu`和逐字参考展开核`reference_emit.cuh`；`runtime.py`提供显式SparseRenderer；`validate.py`、`test_lifecycle.py`、`sanitizer_smoke.py`及`benchmark.py`负责验证与诊断。

每实例复用有限个Scratch Buffer，所有投影、颜色、深度、Tile成员逐帧重算，输出Tensor独立分配。单个Binning/Bitset工作区上限256MiB，超过时显式报错，不截断结果。实例绑定device与stream；切换stream须另建实例，避免共享Workspace竞争。无训练梯度、无GI/贡献统计路径修改，不缓存上帧可见性或已补偿属性。

## 三项算法

1. 原展开由单线程串行扫描一个Gaussian的包围矩形。Warp候选让32条线程并行计算连续Tile，ballot与popcount稳定压缩有效项。每Gaussian输出仍保持原行优先顺序，末尾填入同样的sentinel，随后使用相同稳定CUB排序。

2. 短列表候选在每个Tile列表长度≤32时，用Warp Shuffle广播Gaussian数据并保持每像素原始混合算式；长列表继续执行同一核中的原循环。这是按Tile实际工作量优化，不是按地图点数屏蔽高点数。

3. Bitset候选对Gaussian中心深度排序一次；无效Gaussian置于末尾。每Tile用按深度rank排列的Bitset表示有效成员，成员判断沿用原包围矩形、conic与Opacity判据。逐位枚举成员，按与原Tile排序相同的Gaussian顺序执行原Alpha合成。消除大规模重复实例排序，以及实例总数的CPU回读；保留FP32运算、Alpha cutoff、透射率早停和同深度输入顺序。

Bitset状态大小为4×Tile数×ceil(K/32)字节，位表扫描随K增长。低K、大投影的工作负载可以避免大量重复排序；高K、密集覆盖下扫描空字和维护位表的成本可能超过节省量，不承诺普遍加速。

## 测量方法

4个Replica场景/模式：office1单目、office2 RGB-D、room0 RGB-D、room2单目。每组8个跳转视角；同一包渲染输入固定，不把Selector/Compensator变化混入。条件包括局部1/5/10/70%、固定1000/5000点和Full；每条件3次重复×40帧，版本交替执行。实际K不足固定预算时如实记录。纯渲染墙钟包含入口调用至CUDA同步，不含模型读取、选点、补偿或图片保存。

逐项消融包含：原已加载二进制、同研究入口的逐字串行展开控制、Warp展开、仅短列表、两者组合、Bitset。控制与候选统一预处理和入口；原入口对照保留真实部署成本。初轮V1控制核并非逐字参考，可能放大算法差距，已修正为V2；V1保留回溯，不用于正式结论。

当前未找到可执行Nsight Systems，使用PyTorch/Kineto CUDA Activity记录Kernel名称、发射次数与实际GPU活动时间，并导出Chrome trace。Profiling有额外扰动，时间单独列出，不能与无Profiler墙钟直接拼接。

## 每场景负载与耗时（并发诊断）

| 场景 | 条件 | 实际K中位数 | bbox/有效实例 | 原入口ms | 控制ms | Warp展开ms | 短列表ms | 展开＋短列表ms | Bitset ms | Bitset/原入口加速 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Replica/office1/mono | full | 16417 | 645592/103583 | 1.1189 | 1.1079 | 0.7232 | 1.1084 | 0.7118 | 0.9040 | 1.24× |
| Replica/office1/mono | local1000 | 1000 | 374738/80563.5 | 1.1086 | 1.1016 | 0.6212 | 1.1205 | 0.6213 | 0.3642 | 3.04× |
| Replica/office1/mono | local10pct | 441.5 | 246736/69262 | 1.1317 | 1.1284 | 0.5545 | 1.1095 | 0.5590 | 0.3267 | 3.46× |
| Replica/office1/mono | local1pct | 43.5 | 69584.5/31139 | 0.9705 | 0.9611 | 0.5051 | 0.9790 | 0.5340 | 0.2877 | 3.37× |
| Replica/office1/mono | local5000 | 3098 | 428815/93588 | 1.0767 | 1.0542 | 0.6515 | 1.0912 | 0.6447 | 0.4485 | 2.40× |
| Replica/office1/mono | local5pct | 220.5 | 170419/55349.5 | 1.1506 | 1.1202 | 0.5586 | 1.1318 | 0.5566 | 0.3356 | 3.43× |
| Replica/office1/mono | local70pct | 3000.5 | 424152/93027.5 | 1.0775 | 1.0765 | 0.6573 | 1.0933 | 0.6613 | 0.4492 | 2.40× |
| Replica/office2/rgbd | full | 43197 | 968839/164361 | 1.1999 | 1.2013 | 0.8596 | 1.2109 | 0.8449 | 1.7429 | 0.69× |
| Replica/office2/rgbd | local1000 | 1000 | 361372/84131.5 | 1.1475 | 1.1310 | 0.6309 | 1.1348 | 0.6402 | 0.3827 | 3.00× |
| Replica/office2/rgbd | local10pct | 1497.5 | 410221/93522.5 | 1.1114 | 1.1115 | 0.6642 | 1.1144 | 0.6713 | 0.4082 | 2.72× |
| Replica/office2/rgbd | local1pct | 149 | 164299/40819.5 | 1.1553 | 1.1159 | 0.5410 | 1.1232 | 0.5600 | 0.3163 | 3.65× |
| Replica/office2/rgbd | local5000 | 5000 | 575498/127492 | 1.1225 | 1.1045 | 0.7259 | 1.1228 | 0.7337 | 0.6112 | 1.84× |
| Replica/office2/rgbd | local5pct | 748.5 | 323389/77001 | 1.1716 | 1.1301 | 0.6085 | 1.1537 | 0.6196 | 0.3633 | 3.22× |
| Replica/office2/rgbd | local70pct | 10487.5 | 711490/150400 | 1.1275 | 1.1173 | 0.7809 | 1.1252 | 0.7752 | 0.7978 | 1.41× |
| Replica/room0/rgbd | full | 79351 | 1.19468e+06/175686 | 1.4569 | 1.4129 | 1.0530 | 1.4301 | 1.0000 | 3.3012 | 0.44× |
| Replica/room0/rgbd | local1000 | 1000 | 414718/94777.5 | 1.1644 | 1.1475 | 0.6487 | 1.1906 | 0.6404 | 0.3833 | 3.04× |
| Replica/room0/rgbd | local10pct | 1854 | 505976/108210 | 1.1671 | 1.1687 | 0.7089 | 1.1841 | 0.6982 | 0.4396 | 2.66× |
| Replica/room0/rgbd | local1pct | 185 | 203374/57313.5 | 1.1582 | 1.1384 | 0.5561 | 1.1332 | 0.5665 | 0.3220 | 3.60× |
| Replica/room0/rgbd | local5000 | 5000 | 667012/137456 | 1.2013 | 1.2035 | 0.7675 | 1.1961 | 0.7966 | 0.6132 | 1.96× |
| Replica/room0/rgbd | local5pct | 927 | 399716/88147.5 | 1.1769 | 1.1665 | 0.6891 | 1.1651 | 0.6700 | 0.3809 | 3.09× |
| Replica/room0/rgbd | local70pct | 12982 | 848680/164724 | 1.1953 | 1.2040 | 0.8834 | 1.2258 | 0.8205 | 0.9241 | 1.29× |
| Replica/room2/mono | full | 22807 | 1.09589e+06/162747 | 1.3259 | 1.2811 | 0.8891 | 1.2553 | 0.8938 | 1.1618 | 1.14× |
| Replica/room2/mono | local1000 | 1000 | 397232/101404 | 1.1672 | 1.1635 | 0.6828 | 1.1762 | 0.6442 | 0.3958 | 2.95× |
| Replica/room2/mono | local10pct | 969.5 | 393243/101434 | 1.1473 | 1.1298 | 0.6632 | 1.1282 | 0.6291 | 0.3807 | 3.01× |
| Replica/room2/mono | local1pct | 96.5 | 117394/42730 | 1.0706 | 1.0536 | 0.5542 | 1.0704 | 0.5685 | 0.3297 | 3.25× |
| Replica/room2/mono | local5000 | 5000 | 759755/149416 | 1.2056 | 1.1946 | 0.7977 | 1.2056 | 0.7906 | 0.6302 | 1.91× |
| Replica/room2/mono | local5pct | 484.5 | 282319/82760 | 1.1658 | 1.1421 | 0.6191 | 1.1501 | 0.6107 | 0.3649 | 3.20× |
| Replica/room2/mono | local70pct | 6789 | 816596/157662 | 1.1951 | 1.1959 | 0.8063 | 1.1965 | 0.8230 | 0.7001 | 1.71× |

## GPU活动与发射数量（Profiling扰动下）

只汇总Kineto trace中cat=kernel的事件，排除覆盖整个GPU区间的user annotation，Memcpy与Memset另列，避免重复计时。

| 条件 | 版本 | Kernel/帧 | Memcpy/帧 | Memset/帧 | 累计Kernel μs/帧 |
|---|---|---:|---:|---:|---:|
| local1pct | original | 18 | 1 | 9 | 772.00 |
| local1pct | control | 18 | 1 | 9 | 773.00 |
| local1pct | warp_expand | 18 | 1 | 9 | 173.80 |
| local1pct | short | 18 | 1 | 9 | 775.70 |
| local1pct | warp_short | 18 | 1 | 9 | 179.70 |
| local1pct | bitset | 9 | 0 | 1 | 131.20 |
| local70pct | original | 18 | 1 | 9 | 749.90 |
| local70pct | control | 18 | 1 | 9 | 756.10 |
| local70pct | warp_expand | 18 | 1 | 9 | 406.10 |
| local70pct | short | 18 | 1 | 9 | 753.30 |
| local70pct | warp_short | 18 | 1 | 9 | 403.90 |
| local70pct | bitset | 14 | 0 | 7 | 789.60 |
| full | original | 18 | 1 | 9 | 853.20 |
| full | control | 18 | 1 | 9 | 863.90 |
| full | warp_expand | 18 | 1 | 9 | 501.60 |
| full | short | 18 | 1 | 9 | 860.50 |
| full | warp_short | 18 | 1 | 9 | 498.70 |
| full | bitset | 14 | 0 | 7 | 2104.00 |

## Workspace与一致性

| 版本 | 实例保留Scratch MiB |
|---|---:|
| control | 98.189 |
| warp_expand | 98.189 |
| short | 98.189 |
| warp_short | 98.189 |
| bitset | 38.323 |

此处为4场景测试后的高水位Scratch，不含输出图像、Torch缓存分配器和原模型。缓冲区扩容按同一实例曾遇到的最大形状保留，不按帧累积。Bitset没有复制可学习Feature或新增网络参数。

## 失败、限制与下一步

编译初次因PATH未包含已有conda ninja失败，已仅在本进程PATH修正，没有安装或修改共享环境。首轮验证写报告时发生辅助脚本HERE变量被导入覆盖，已修正；测试输出归档。生命周期测试预计算RGB/协方差时的None格式与原wrapper一致后通过。

Compute Sanitizer首轮memcheck零错误，initcheck发现Scratch扩容的resize_复制未初始化的旧临时字节。已改为扩容时丢弃旧内容并重新分配；所需状态每帧重建。诊断快照在inspect时预先清零，避免读出无效Gaussian未定义的诊断字段。此修正不改变计算公式或像素，只影响工作区增长行为。

目前三个实现保持为显式研究选项；最终保留哪项及组合，需要空闲GPU复测。尤其Full Bitset退化不能通过按K人为屏蔽隐藏；后续应根据实际Tile密度、位表扫描成本分析结构限制。短列表是否值得保留应看独立增益，不能仅因组合版本更快就归因。

下一次空闲时运行同一benchmark.py，不带--allow-concurrent，写入新的独立输出目录；至少重复两轮，比较低点数绝对耗时、同入口控制组、Full及Kernel活动。不能把当前并发数字当作正式端侧性能，也不能将Rasterizer通用收益计入Selector/Compensator网络提速。

## CUDA内存检查

最终Compute Sanitizer memcheck/initcheck通过状态 True，错误汇总 [0, 0]。检查使用小型真实CUDA入口，覆盖1/33/257点、无效点和全部算法，不代表未测试输入的穷尽证明。

## 完整性核验

保护文件 625 个，变更数 0，通过状态 True。未修改服务器mycartgs/realLOD、原二进制或Teacher，未commit/push。
