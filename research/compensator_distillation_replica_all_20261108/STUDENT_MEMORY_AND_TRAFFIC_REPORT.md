# 蒸馏Compensator显存与访存实测

## 结论

蒸馏学生确实降低MLP计算量，但当前完整部署实例的常驻显存只减少约0.067MiB（68.5KiB）。它不替换或卸载完整Master、Anchor Feature、索引/排序Workspace和Full发布包，因此不能解决此前动态LoD总显存高于轻量Full的问题。

固定相同地图、Selector、Anchor Feature、相机、Ratio和选点，比较原64/64四个两层Head与最终32/32四个直接Head学生。新增32个独立子进程测显存，32个NCU捕获测DRAM字节。所有显存请求逐个核对实际K及选点ID哈希，学生SHA与本轮质量评价一致。没有重训。

DRAM访问收益不普遍：office2的total范围在1%两轮减少约19%–22%、70%减少约8.5%；room0的70%减少约4.3%–5.7%，1%两轮变化约+0.17%和−3.12%，10%单次增加约0.85%。因此不能称所有场景/Ratio都有稳定访存下降，也不能将这些比例当作实际带宽速度提升。

## 常驻和最高总显存

下表是每个条件第二轮六视角中“常驻＋临时”的最高活跃显存，不是逐帧峰值中位数；MiB=2^20字节。加载常驻增量扣除相同初始化基线，总峰值包含该基线。

| 场景 | 模式 | Teacher常驻增量 MiB | Student常驻增量 MiB | 常驻节省 KiB | Teacher最高总显存 MiB | Student最高总显存 MiB | 学生峰值变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| Replica/office2/rgbd | runtime_full | 59.725 | 59.658 | 68.5 | 129.193 | 129.126 | -0.05% |
| Replica/office2/rgbd | 0.01 | 59.725 | 59.658 | 68.5 | 92.303 | 92.265 | -0.04% |
| Replica/office2/rgbd | 0.05 | 59.725 | 59.658 | 68.5 | 98.680 | 98.673 | -0.01% |
| Replica/office2/rgbd | 0.1 | 59.725 | 59.658 | 68.5 | 101.923 | 101.991 | +0.07% |
| Replica/office2/rgbd | 0.3 | 59.725 | 59.658 | 68.5 | 110.177 | 110.323 | +0.13% |
| Replica/office2/rgbd | 0.5 | 59.725 | 59.658 | 68.5 | 115.986 | 116.131 | +0.12% |
| Replica/office2/rgbd | 0.7 | 59.725 | 59.658 | 68.5 | 120.312 | 120.312 | +0.00% |
| Replica/office2/rgbd | runtime_switch | 59.725 | 59.658 | 68.5 | 172.303 | 172.236 | -0.04% |
| Replica/room0/rgbd | runtime_full | 102.529 | 102.462 | 68.5 | 225.687 | 225.620 | -0.03% |
| Replica/room0/rgbd | 0.01 | 102.529 | 102.462 | 68.5 | 129.757 | 129.739 | -0.01% |
| Replica/room0/rgbd | 0.05 | 102.529 | 102.462 | 68.5 | 142.897 | 143.029 | +0.09% |
| Replica/room0/rgbd | 0.1 | 102.529 | 102.462 | 68.5 | 152.383 | 152.503 | +0.08% |
| Replica/room0/rgbd | 0.3 | 102.529 | 102.462 | 68.5 | 174.688 | 174.898 | +0.12% |
| Replica/room0/rgbd | 0.5 | 102.529 | 102.462 | 68.5 | 187.874 | 187.807 | -0.04% |
| Replica/room0/rgbd | 0.7 | 102.529 | 102.462 | 68.5 | 193.411 | 194.433 | +0.53% |
| Replica/room0/rgbd | runtime_switch | 102.529 | 102.462 | 68.5 | 225.688 | 225.621 | -0.03% |

网络原始权重25742→3534参数，FP32参数字节理论节省88832字节；部署实测不等于该值，因为现有实例仍保留Teacher模型对象，并将执行权重替换为转置后的学生参数，还保留学生对象。仅改网络执行宽度，不会按比例缩小大地图数据与准备工作区。

局部渲染临时显存还取决于学生预测的Scale/Rotation/XYZ及Tile覆盖。即使K与ID完全一致，学生和Teacher图像不相同，临时峰值可以略增也可以略减；不能把全部峰值差解释成网络激活显存。

## 与CaRtGS原版和我们的轻量Full比较

| 场景 | CaRtGS原版最高峰值（三次取最大） MiB | 我们的轻量Full最高峰值 MiB | 蒸馏LoD1%最高峰值 MiB |
|---|---:|---:|---:|
| office2 RGB-D | 89.862 | 69.141 | 92.265 |
| room0 RGB-D | 119.799 | 123.045 | 129.739 |

Full历史对照使用同一推理Renderer和相同RGB帧，原版地图使用各自保存SLAM位姿；它不代表原生C++ Viewer完整进程的驱动占用。详情见VANILLA_VS_OURS_MEMORY_REPORT.md。

## DRAM访问量配对

Nsight Compute 2022.3，dram__bytes_read.sum与dram__bytes_write.sum之和。render仅渲染，total含Selector、Compensator、打包和渲染捕获到的CUDA Kernel；不包含Copy Engine设备复制。五次预热后每场景中间一个相机捕获一次，1%/10%/70%，双方实际K和ID一致。

首轮覆盖全部条件，第二轮独立进程复测1%/70%的total范围。小幅差异同时受几何覆盖、缓存和Profiling影响，不能仅凭单帧几个百分点的变化建立机制因果。

| 场景 | Ratio | 范围 | 轮次 | 实际K | Teacher DRAM MiB | Student DRAM MiB | 学生变化 |
|---|---:|---|---:|---:|---:|---:|---:|
| Replica/office2/rgbd | 0.01 | render | 1 | 96 | 35.67 | 37.52 | +5.19% |
| Replica/office2/rgbd | 0.01 | total | 1 | 96 | 53.85 | 41.91 | -22.17% |
| Replica/office2/rgbd | 0.1 | render | 1 | 965 | 80.52 | 80.64 | +0.15% |
| Replica/office2/rgbd | 0.1 | total | 1 | 965 | 100.19 | 94.65 | -5.54% |
| Replica/office2/rgbd | 0.7 | render | 1 | 6757 | 127.76 | 128.25 | +0.38% |
| Replica/office2/rgbd | 0.7 | total | 1 | 6757 | 172.28 | 157.45 | -8.61% |
| Replica/office2/rgbd | 0.01 | total | 2 | 96 | 53.17 | 42.94 | -19.23% |
| Replica/office2/rgbd | 0.7 | total | 2 | 6757 | 167.78 | 153.56 | -8.48% |
| Replica/room0/rgbd | 0.01 | total | 2 | 408 | 76.03 | 73.66 | -3.12% |
| Replica/room0/rgbd | 0.7 | total | 2 | 28570 | 513.47 | 484.13 | -5.72% |
| Replica/room0/rgbd | 0.01 | render | 1 | 408 | 53.96 | 55.59 | +3.03% |
| Replica/room0/rgbd | 0.01 | total | 1 | 408 | 73.13 | 73.25 | +0.17% |
| Replica/room0/rgbd | 0.1 | render | 1 | 4081 | 172.24 | 170.41 | -1.07% |
| Replica/room0/rgbd | 0.1 | total | 1 | 4081 | 197.06 | 198.74 | +0.85% |
| Replica/room0/rgbd | 0.7 | render | 1 | 28570 | 374.46 | 374.04 | -0.11% |
| Replica/room0/rgbd | 0.7 | total | 1 | 28570 | 515.48 | 493.43 | -4.28% |

该表是访存字节量，不是显存容量、GB/s或帧时间。参数与MAC缩小比例不等于DRAM流量缩小比例，Renderer和地图数据访问占据独立成本。学生渲染覆盖变化也会影响流量。

## 口径、质量和完整性

显存采用独立进程的PyTorch CUDA allocated/peak，reserved单独保存；不含全部CUDA Context、驱动及第三方直接分配。两场景各六视角、两轮，另外测完整Ratio切换的Graph/Workspace保留。未使用训练的Adam状态。

NCU使用kernel replay、cache-control none、clock-control none，管理员权限仅运行Profiler，无驱动配置改动。共享环境和缓存状态可能影响DRAM计数，单视角数据不能外推全轨迹均值，Profiler期间耗时不用于测速。

本轮没有新增速度测量。此前空闲3090配对结果中，Compensator组件1.39–2.93×、准备流程1.02–1.46×、准备＋渲染1.006–1.115×。蒸馏主要体现为计算收益，常驻显存不是当前主要收益。

Student非Teacher数值等价替代：Replica16组全局平均PSNR损失约0.154dB、局部约0.276dB，严格质量门限未通过。本轮资源测试不能掩盖这些质量代价，也不修改生产默认。

后续若以常驻显存为目标，应单独研究删除部署快照中的未使用Teacher权重、冗余Master/Full副本及按需工作区；这些改动均未在本轮执行。

文件：student_memory_20261109保存32进程原始记录；student_traffic_20261109保存32个硬件计数捕获；STUDENT_RESOURCE_MEMORY_PAIRED.csv和STUDENT_RESOURCE_DRAM_PAIRED.csv为主要对照。
