# 网络LoD减少点数的显存与访存收益实测

## 结论

同checkpoint测试Replica office2 RGB-D（43,197点）与room0 RGB-D（79,351点）。低Ratio确实显著减少Rasterizer临时显存和Kernel的DRAM访问量；动态网络不会卸载完整Master、Feature和工作区，常驻显存基本不随K下降。若与只加载Full渲染属性的轻量路径比较，当前动态LoD实例的总显存仍可能更高。

本次只增加研究脚本和报告，复用原64/64 Compensator、Selector、v31部署二进制及原Rasterizer，没有修改训练、权重、生产默认或渲染内核，没有保存渲染图片。学生蒸馏不是本轮变量。

## 显存测量方法

18个独立子进程：每场景分别测试Full属性包、完整部署实例Full、六档局部Ratio及连续切换。每进程六个相机×两轮，表取第二轮六视角统计。固定1200×680、FP32。原未补偿Master为Full基线，不把ratio=1补偿当作Full。

使用torch.cuda.memory_allocated/max_memory_allocated记录活跃分配，reserved另列。每次在准备和渲染前分别同步、reset_peak_memory_stats。渲染新增峰值=render阶段峰值减去已准备属性的活跃分配，包含RGB/radii输出、预处理、Tile展开/排序及其他Renderer临时空间。并非仅输出Tensor大小，也不是峰值减去reserved。

加载常驻增量扣除相机/框架初始化基线约8.14MiB；全流程峰值包含该基线。PyTorch allocator统计不含全部CUDA Context、驱动及第三方直接cudaMalloc占用，不能直接当作nvidia-smi整进程数值。其他训练进程显存不会计入本进程allocator。

## 同一动态部署实现：Full与各Ratio

| 场景 | 模式 | 实际K中位数 [最小,最大] | 加载常驻增量 MiB | 渲染新增峰值中位数 MiB | 全流程活跃峰值中位数 MiB | 六视角最大活跃峰值 MiB |
|---|---|---|---:|---:|---:|---:|
| Replica/office2/rgbd | runtime_full | 43197 [43197,43197] | 59.72 | 42.44 | 120.84 | 129.19 |
| Replica/office2/rgbd | 0.01 | 110 [33,178] | 59.72 | 14.77 | 91.27 | 92.30 |
| Replica/office2/rgbd | 0.05 | 554 [167,893] | 59.72 | 18.95 | 95.90 | 98.68 |
| Replica/office2/rgbd | 0.1 | 1108.5 [335,1787] | 59.72 | 22.01 | 99.52 | 101.92 |
| Replica/office2/rgbd | 0.3 | 3326.5 [1005,5363] | 59.72 | 27.70 | 107.38 | 110.18 |
| Replica/office2/rgbd | 0.5 | 5545 [1675,8938] | 59.72 | 28.77 | 110.68 | 115.99 |
| Replica/office2/rgbd | 0.7 | 7763 [2345,12513] | 59.72 | 29.38 | 113.40 | 120.31 |
| Replica/room0/rgbd | runtime_full | 79351 [79351,79351] | 102.53 | 74.52 | 204.56 | 225.69 |
| Replica/room0/rgbd | 0.01 | 266 [99,460] | 102.53 | 16.10 | 126.84 | 129.76 |
| Replica/room0/rgbd | 0.05 | 1332 [499,2300] | 102.53 | 25.14 | 136.17 | 142.90 |
| Replica/room0/rgbd | 0.1 | 2665 [999,4601] | 102.53 | 30.57 | 141.94 | 152.38 |
| Replica/room0/rgbd | 0.3 | 7996 [2997,13803] | 102.53 | 42.40 | 155.33 | 174.69 |
| Replica/room0/rgbd | 0.5 | 13327 [4996,23006] | 102.53 | 48.41 | 162.62 | 187.87 |
| Replica/room0/rgbd | 0.7 | 18488 [6994,32209] | 102.53 | 52.58 | 168.22 | 193.41 |

例如room0在同一个动态部署架构下，从Full到局部1%，渲染新增峰值中位数74.52→16.10MiB；全流程活跃峰值中位数204.56→126.84MiB。完整Master和Feature仍驻留，加载常驻增量102.53MiB保持不变。

## 与轻量Full路径的公平区分

| 场景 | 仅Full属性加载增量 MiB | 动态LoD加载增量 MiB | 轻量Full活跃峰值中位数 MiB | 动态局部1%活跃峰值中位数 MiB |
|---|---:|---:|---:|---:|
| Replica/office2/rgbd | 10.22 | 59.72 | 60.79 | 91.27 |
| Replica/room0/rgbd | 19.15 | 102.53 | 101.92 | 126.84 |

轻量Full只保留同一个Master激活后的最终推理属性，释放加载时原始训练属性副本，不加载Selector、Feature及部署准备工作区；动态路径使用现有DeploymentRenderer完整实例。两者是部署功能不同的实现对照，不把差值全部归于MLP参数。当前完整实例还保留Master原始与激活快照、Full发布包、Anchor Feature、静态前缀、索引/排序工作区等。

## 来回切换和Allocator缓存

| 场景 | 模式 | 热态活跃峰值中位数 MiB | 请求后保留活跃分配 MiB | 当次峰值reserved最大 MiB | Graph数最大 |
|---|---|---:|---:|---:|---:|
| Replica/office2/rgbd | runtime_full | 163.95 | 110.97 | 310.00 | 0 |
| Replica/office2/rgbd | 0.01 | 125.84 | 110.97 | 310.00 | 5 |
| Replica/office2/rgbd | 0.1 | 134.04 | 110.97 | 310.00 | 5 |
| Replica/office2/rgbd | 0.7 | 147.73 | 110.97 | 310.00 | 5 |
| Replica/room0/rgbd | runtime_full | 204.56 | 110.67 | 290.00 | 0 |
| Replica/room0/rgbd | 0.01 | 126.84 | 110.67 | 290.00 | 0 |
| Replica/room0/rgbd | 0.1 | 141.94 | 110.67 | 290.00 | 0 |
| Replica/room0/rgbd | 0.7 | 168.22 | 110.67 | 290.00 | 0 |

office2经历全部Ratio后保留更多Graph及相关存储，allocated仍受加载状态与缓存容量影响；room0大地图走原Host计数路径，未使用同类Graph。reserved是PyTorch复用池，曾经跑过Full后不自动按K缩小。不能用nvidia-smi未下降就否定临时显存节省，也不能把缓存池全称为活跃模型数据。

## Nsight Compute的真实访存计数

先复现普通用户ERR_NVGPUCTRPERM（RmProfilingAdminOnly=1），再使用既有授权以sudo运行Profiler，成功读取计数器；未修改驱动参数。Nsight Compute 2022.3，指标dram__bytes_read.sum、dram__bytes_write.sum、lts__t_bytes.sum，kernel replay、cache-control none、clock-control none。每场景中间一个相机，固定相同checkpoint和原分辨率；五次预热后只捕获一次请求。

render范围仅含已准备属性的渲染；total范围含现有部署Full打包或Selector/Compensator/准备再渲染。下面数据是捕获CUDA Kernel的DRAM读写总量，不是DRAM容量，不是GB/s，也不包含未由Kernel完成的cudaMemcpy/Copy Engine流量。Full发布包的clone使用设备复制，其流量不能从这些Kernel计数补全，因此total列不能称为整卡全部访存。L2 requested与DRAM属于不同层级，不能相加。

捕获期间服务器有其他训练，Nsight会改变执行调度并可能串行化目标Kernel，本报告不以Profiling耗时作速度比较。缓存状态与共享环境可能影响物理DRAM读写；关键条件另起进程独立重复，数值见复测表。

| 场景 | 模式 | 实际K | 范围 | Kernel数 | DRAM读取 MiB | DRAM写入 MiB | 合计 MiB | 相对同范围Full变化 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Replica/office2/rgbd | 0.01 | 96 | render | 18 | 11.90 | 25.14 | 37.04 | -80.3% |
| Replica/office2/rgbd | 0.01 | 96 | total | 29 | 15.53 | 38.10 | 53.62 | -71.4% |
| Replica/office2/rgbd | 0.1 | 965 | render | 18 | 34.64 | 47.31 | 81.95 | -56.4% |
| Replica/office2/rgbd | 0.1 | 965 | total | 29 | 39.21 | 63.71 | 102.92 | -45.1% |
| Replica/office2/rgbd | 0.7 | 6757 | render | 18 | 57.23 | 72.01 | 129.24 | -31.2% |
| Replica/office2/rgbd | 0.7 | 6757 | total | 29 | 75.17 | 97.97 | 173.14 | -7.7% |
| Replica/office2/rgbd | full | 43197 | render | 18 | 87.88 | 100.07 | 187.95 | +0.0% |
| Replica/office2/rgbd | full | 43197 | total | 18 | 87.73 | 99.88 | 187.61 | +0.0% |
| Replica/room0/rgbd | 0.01 | 408 | render | 18 | 21.19 | 33.12 | 54.32 | -88.7% |
| Replica/room0/rgbd | 0.01 | 408 | total | 32 | 33.66 | 44.76 | 78.41 | -83.8% |
| Replica/room0/rgbd | 0.1 | 4081 | render | 18 | 78.34 | 90.93 | 169.26 | -64.9% |
| Replica/room0/rgbd | 0.1 | 4081 | total | 32 | 94.66 | 106.89 | 201.55 | -58.2% |
| Replica/room0/rgbd | 0.7 | 28570 | render | 18 | 181.48 | 190.92 | 372.40 | -22.7% |
| Replica/room0/rgbd | 0.7 | 28570 | total | 32 | 262.87 | 254.36 | 517.22 | +7.2% |
| Replica/room0/rgbd | full | 79351 | render | 18 | 238.37 | 243.31 | 481.68 | +0.0% |
| Replica/room0/rgbd | full | 79351 | total | 18 | 238.39 | 244.25 | 482.64 | +0.0% |

低Ratio渲染Kernel访问量显著下降，加入网络准备后仍保留大部分收益。room0局部70%的total捕获Kernel读写量却比同范围Full高，说明网络准备可能抵消高Ratio的访问节省；由于设备复制流量未覆盖，不能把这一项推广为完整请求总DRAM必然增加。

## 独立重复计数

| 场景 | 模式 | 范围 | 首次 MiB | 复测 MiB | 变化 |
|---|---|---|---:|---:|---:|
| Replica/office2/rgbd | 0.01 | render | 37.04 | 39.17 | +5.73% |
| Replica/office2/rgbd | 0.01 | total | 53.62 | 53.33 | -0.55% |
| Replica/office2/rgbd | 0.7 | total | 173.14 | 170.72 | -1.40% |
| Replica/office2/rgbd | full | render | 187.95 | 188.42 | +0.25% |
| Replica/room0/rgbd | 0.01 | render | 54.32 | 56.12 | +3.32% |
| Replica/room0/rgbd | 0.01 | total | 78.41 | 77.64 | -0.99% |
| Replica/room0/rgbd | 0.7 | total | 517.22 | 515.56 | -0.32% |
| Replica/room0/rgbd | full | render | 481.68 | 481.96 | +0.06% |

## 实际含义与限制

已经证实：在这两张冻结地图上，网络降低点数减少了Renderer临时显存和低Ratio的Kernel访存量；不能因此推出常驻地图显存同比缩小，也不能把访存下降比例当作帧时间加速比例。当前桌面3090未必受DRAM带宽限制；端侧设备可能更受益，但本轮未测试端侧硬件。

Ratio改变同时改变选点、几何补偿和渲染质量，因此本实验是实际部署资源对照，不是仅改变点数的纯因果消融。本轮使用原Teacher网络以隔离网络减少点数的资源收益，不将学生蒸馏质量损失混入。显存覆盖六相机，硬件计数只覆盖每场景一个中间相机，不能作为全轨迹平均。

最直接的后续优化是消除部署实例的冗余Master/Full副本、按需要分配工作区，以及明确Graph缓存上限；它们可减少当前常驻成本。若要让常驻显存随LoD下降，仍需固定LoD导出或按需加载地图块。以上均未在本轮实现。

原始文件：lod_memory_20261109（18个独立进程），lod_traffic_20261109（24个NCU捕获）；汇总为LOD_MEMORY_SUMMARY.csv、LOD_SWITCH_MEMORY_SUMMARY.csv、LOD_DRAM_TRAFFIC_SUMMARY.csv、LOD_DRAM_KERNEL_METRICS.csv。
