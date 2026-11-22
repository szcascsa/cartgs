# CPU 深度优化：小输出微核、顺序压缩与调度长尾

## 结论与适用范围

本轮实际实现、编译和验证了两个精确优化：Selector 32→2及Compensator 64→3/4的小输出微核；保留选点身份和行序的自适应顺序压缩。最终候选没有新增数值误差。线程等待策略和分段累加候选未获得稳定收益，已撤回。

收益集中在较大输出K或全局选点；低Ratio 1%–10%没有稳定的普遍加速。不能将本轮结论写成“所有预算更快”。当前只是独立研究后端，没有替换生产默认，也没有再次计入此前原生化、AVX2或CPU→GPU交接的收益。

本轮真实对照是上一轮`cpu_native_deep_20261110_1222`的冻结二进制，双方均用原生准备与三槽交接。表中为24组×2轮等权几何平均加速，延迟列为48个场景轮次中位的中位；这两种聚合不能直接互相相除。

| 条件 | 上一版ms | 本轮ms | 几何平均加速 | 延迟降低 | 范围 | 慢例/48轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed0 | 0.0703 | 0.0733 | 0.981× | -1.91% | 0.923–1.053 | 36 |
| fixed1000 | 0.5282 | 0.5292 | 1.026× | +2.49% | 0.885–1.137 | 12 |
| fixed5000 | 1.2911 | 1.1055 | 1.137× | +12.06% | 1.043–1.248 | 0 |
| full | 1.0491 | 1.0458 | 0.983× | -1.72% | 0.848–1.086 | 35 |
| global0.01 | 0.5742 | 0.5221 | 1.090× | +8.27% | 1.027–1.152 | 0 |
| global0.1 | 1.0773 | 0.9478 | 1.135× | +11.89% | 1.065–1.190 | 0 |
| local0.01 | 0.3081 | 0.3106 | 1.008× | +0.80% | 0.778–1.218 | 9 |
| local0.03 | 0.3517 | 0.3597 | 0.998× | -0.18% | 0.841–1.063 | 13 |
| local0.05 | 0.3958 | 0.3859 | 1.007× | +0.74% | 0.862–1.059 | 12 |
| local0.08 | 0.4503 | 0.4391 | 1.006× | +0.59% | 0.783–1.096 | 15 |
| local0.1 | 0.4830 | 0.4993 | 0.998× | -0.19% | 0.756–1.085 | 20 |
| local0.2 | 0.6721 | 0.6201 | 1.066× | +6.20% | 0.960–1.140 | 1 |
| local0.3 | 0.8441 | 0.7832 | 1.100× | +9.13% | 0.956–1.159 | 1 |
| local0.4 | 1.0062 | 0.8878 | 1.146× | +12.75% | 1.070–1.451 | 0 |
| local0.5 | 1.1173 | 0.9940 | 1.147× | +12.82% | 0.927–1.434 | 1 |
| local0.6 | 1.3291 | 1.1172 | 1.160× | +13.77% | 1.072–1.561 | 0 |
| local0.7 | 1.3750 | 1.2160 | 1.157× | +13.55% | 0.774–1.264 | 2 |

两轮正式CPU计时样本117,504个，身份/属性检查19,584组，全部逐位相同；CPU测量结束时CUDA未初始化。24组含Replica16、TUM4、VECtor4；开发仅用Replica office2 RGB-D、room0 RGB-D、TUM fr1_xyz RGB-D，其余21组保留验证。候选在正式测试前冻结，未根据保留集调参数。

## 实现及等价性

### 小输出微核

上一轮面向8/16输出的AVX2微核，在2/3/4输出时仍计算无消费者的高通道。本轮以SSE128/FMA仅计算低4通道，Inputs=32/64、Outputs=2/3/4、Rows=1…4编译期展开。每个有效输出仍严格按j=0…D−1执行FMA，再加相同Bias及ReLU；没有改变求和分组、FMA链或激活顺序。只写有效2/3/4个FP32，避免越界。

只涉及Selector最终2输出层与Compensator四个Head的最终3/3/4/4输出层。主干及四Head隐藏层、独立权重、Feature共享、模型结构与物理解码均不变。没有FP16、INT8或fast-math。

### 顺序压缩

- 局部严格K保留原`nth_element`选择。Key仍为Score降序、原Gaussian行号升序，第K+1个Key作为cutoff；从原候选行序扫描，只保留`Better(key,cutoff)`，因此集合、K和最终原行号顺序不变。
- 全局分位数仍使用原阈值及`score >= threshold`，同分保留数量不变，没有把它偷换为严格K。
- 各worker只写自己区间，主线程按区间顺序`memmove`向前压缩；复用既有selected容量，无新增N规模Buffer，不依赖旧帧缓存。
- 低K全候选扫描可能比K项排序更慢。最终`ordered_mode=3`在局部仅当`K*log2(max(2,K)) > M+4096`时使用压缩，否则使用原排序。该常量只在三个开发场景确定，不按场景或最终测试集设参数。
- Full没有调用Selector或Compensator；本轮Full诊断是原整图属性打包，不是Rasterizer速度测试。

### 独立归因

| 条件 | 仅小输出 | 仅自适应压缩 | 合用 |
| --- | --- | --- | --- |
| local0.01 | 1.0138× | 0.9963× | 1.0081× |
| local0.05 | 1.0035× | 0.9880× | 1.0075× |
| local0.1 | 1.0021× | 0.9850× | 0.9981× |
| local0.3 | 0.9972× | 1.0798× | 1.1005× |
| local0.7 | 0.9981× | 1.1409× | 1.1567× |
| fixed1000 | 1.0080× | 1.0146× | 1.0256× |
| fixed5000 | 1.0020× | 1.1089× | 1.1371× |
| global0.1 | 1.0198× | 1.1076× | 1.1349× |
| full | 0.9942× | 0.9901× | 0.9831× |

小输出层在整个网络中占比有限，独立端到端收益较小。顺序压缩主要节约选择后的排序成本，因此高K更有效。低Ratio受候选/视角/Selector等固定成本和OS调度影响，减少几个输出通道不会自动带来毫秒级收益。

## 测量协议

硬件为Intel i5-9600K，6核6线程、AVX2/FMA；每次只驻留一个原生6线程池，Torch辅助1线程。相同checkpoint/权重/固定随机键/背景/相机，双方每帧重新准备。正式CPU测试包含每场景3个连续相机和3个跳转相机，两轮反转场景并轮换版本顺序。每条件每轮每版本3块×12样本，块前6次切换视角预热。包含11种已见/未见local Ratio、fixed0/1000/5000、global1%/10%和Full，共17条件。

CPU时间包含候选、视角、Selector、选择、Compensator、物理解码和输出分配打包；不含磁盘读取、相机矩阵构造、显示或GPU渲染。分段计时另行采集，不放在正式计时区间。监测保留每秒外部CPU/GPU活动，两轮均未检测到外部活跃负载。该监测不能排除OS调度、中断或时钟波动，不等同于每个样本都无扰动。

## 每场景CPU准备

每格为上一版→本轮的两轮中位ms及两轮几何平均加速。全部Ratio、p95、最大延迟、N/M/K见CSV。

| 场景/模式 | 1% | 10% | 70% |
| --- | --- | --- | --- |
| Replica/office0/mono | 0.2671→0.2642 (1.011×) | 0.3726→0.3662 (1.017×) | 0.8792→0.7695 (1.142×) |
| Replica/office0/rgbd | 0.3028→0.3055 (0.991×) | 0.4405→0.5159 (0.864×) | 1.0956→1.1995 (0.930×) |
| Replica/office1/mono | 0.2475→0.2468 (1.003×) | 0.3119→0.3017 (1.033×) | 0.7994→0.6832 (1.170×) |
| Replica/office1/rgbd | 0.5164→0.5073 (1.018×) | 0.7735→0.7635 (1.013×) | 2.3097→1.9142 (1.207×) |
| Replica/office2/mono | 0.2510→0.2440 (1.028×) | 0.2975→0.3016 (0.986×) | 0.8215→0.7020 (1.170×) |
| Replica/office2/rgbd | 0.3927→0.3876 (1.013×) | 0.5057→0.5129 (0.986×) | 1.3411→1.1828 (1.134×) |
| Replica/office3/mono | 0.3568→0.3773 (0.948×) | 0.5312→0.5856 (0.908×) | 1.5572→1.2887 (1.208×) |
| Replica/office3/rgbd | 0.6084→0.5456 (1.110×) | 0.9080→0.8948 (1.015×) | 2.7184→2.2515 (1.207×) |
| Replica/office4/mono | 0.3123→0.3081 (1.013×) | 0.4730→0.4658 (1.015×) | 1.3939→1.2115 (1.151×) |
| Replica/office4/rgbd | 0.6303→0.6197 (1.017×) | 1.0853→1.0566 (1.027×) | 3.1590→2.6934 (1.173×) |
| Replica/room0/mono | 0.4209→0.4162 (1.011×) | 0.6474→0.6326 (1.023×) | 1.9457→1.7813 (1.095×) |
| Replica/room0/rgbd | 1.3904→1.3752 (1.011×) | 2.5095→2.3999 (1.046×) | 7.5214→6.2869 (1.197×) |
| Replica/room1/mono | 0.3695→0.3665 (1.008×) | 0.5937→0.6046 (0.982×) | 1.7326→1.4851 (1.166×) |
| Replica/room1/rgbd | 0.6756→0.6754 (1.000×) | 1.0940→1.0858 (1.008×) | 3.0821→2.5177 (1.224×) |
| Replica/room2/mono | 0.2972→0.2925 (1.016×) | 0.4947→0.4993 (0.991×) | 1.4753→1.2810 (1.152×) |
| Replica/room2/rgbd | 0.6846→0.6699 (1.022×) | 1.1207→1.1810 (0.949×) | 3.0116→2.6491 (1.137×) |
| TUM_RGBD/freiburg1_desk/rgbd | 0.2571→0.2604 (0.988×) | 0.3823→0.3790 (1.009×) | 1.1400→0.9748 (1.169×) |
| TUM_RGBD/freiburg1_room/rgbd | 0.2758→0.3154 (0.881×) | 0.3932→0.3865 (1.018×) | 0.9182→0.9314 (0.992×) |
| TUM_RGBD/freiburg1_xyz/rgbd | 0.2699→0.2705 (0.998×) | 0.4113→0.4019 (1.023×) | 1.2319→1.0258 (1.201×) |
| TUM_RGBD/freiburg2_xyz/mono | 0.4942→0.4886 (1.011×) | 0.7829→0.7710 (1.015×) | 2.3016→1.8478 (1.246×) |
| VECtor/corner-slow/mono | 0.1609→0.1557 (1.033×) | 0.2004→0.1979 (1.013×) | 0.5543→0.4612 (1.202×) |
| VECtor/corner-slow/stereo | 0.1716→0.1696 (1.012×) | 0.2047→0.2034 (1.006×) | 0.4581→0.3880 (1.181×) |
| VECtor/corridors-dolly/stereo | 0.2871→0.2765 (1.038×) | 0.3376→0.3352 (1.007×) | 0.7783→0.6737 (1.155×) |
| VECtor/robot-normal/mono | 0.2032→0.1976 (1.028×) | 0.2429→0.2386 (1.018×) | 0.5947→0.4934 (1.205×) |

### 所有非Full慢例

共122个场景轮次条件的中位更慢。保留每项，不以“原版也有波动”为理由删除。

| 场景 | 轮 | 条件 | 旧ms | 新ms | 加速倍数 |
| --- | --- | --- | --- | --- | --- |
| Replica/office0/mono | 2 | fixed0 | 0.06163 | 0.06580 | 0.9365 |
| Replica/office0/mono | 2 | local0.1 | 0.36760 | 0.36846 | 0.9977 |
| Replica/office0/rgbd | 2 | fixed0 | 0.06862 | 0.07425 | 0.9241 |
| Replica/office0/rgbd | 2 | fixed1000 | 0.52201 | 0.57127 | 0.9138 |
| Replica/office0/rgbd | 1 | local0.01 | 0.30089 | 0.31288 | 0.9617 |
| Replica/office0/rgbd | 1 | local0.03 | 0.31891 | 0.32863 | 0.9704 |
| Replica/office0/rgbd | 2 | local0.03 | 0.31292 | 0.36733 | 0.8519 |
| Replica/office0/rgbd | 2 | local0.05 | 0.35290 | 0.35539 | 0.9930 |
| Replica/office0/rgbd | 1 | local0.08 | 0.39500 | 0.39969 | 0.9883 |
| Replica/office0/rgbd | 2 | local0.08 | 0.39177 | 0.41554 | 0.9428 |
| Replica/office0/rgbd | 1 | local0.1 | 0.44943 | 0.59442 | 0.7561 |
| Replica/office0/rgbd | 2 | local0.1 | 0.43164 | 0.43746 | 0.9867 |
| Replica/office0/rgbd | 2 | local0.5 | 0.93300 | 1.00624 | 0.9272 |
| Replica/office0/rgbd | 1 | local0.7 | 1.10153 | 1.42319 | 0.7740 |
| Replica/office1/mono | 1 | fixed0 | 0.05900 | 0.06311 | 0.9348 |
| Replica/office1/mono | 2 | fixed0 | 0.05558 | 0.05927 | 0.9377 |
| Replica/office1/mono | 2 | local0.03 | 0.23692 | 0.23781 | 0.9963 |
| Replica/office1/mono | 2 | local0.08 | 0.27539 | 0.27548 | 0.9997 |
| Replica/office1/mono | 1 | local0.1 | 0.30210 | 0.30706 | 0.9838 |
| Replica/office1/rgbd | 1 | fixed0 | 0.07739 | 0.07892 | 0.9806 |
| Replica/office1/rgbd | 2 | fixed0 | 0.07595 | 0.07905 | 0.9607 |
| Replica/office1/rgbd | 1 | local0.05 | 0.61875 | 0.62059 | 0.9970 |
| Replica/office1/rgbd | 2 | local0.08 | 0.71963 | 0.72472 | 0.9930 |
| Replica/office2/mono | 2 | fixed0 | 0.06526 | 0.06615 | 0.9866 |
| Replica/office2/mono | 1 | local0.03 | 0.23663 | 0.24793 | 0.9544 |
| Replica/office2/mono | 2 | local0.03 | 0.23357 | 0.23601 | 0.9896 |
| Replica/office2/mono | 1 | local0.08 | 0.28413 | 0.29346 | 0.9682 |
| Replica/office2/mono | 2 | local0.08 | 0.28468 | 0.29251 | 0.9732 |
| Replica/office2/mono | 1 | local0.1 | 0.30018 | 0.30289 | 0.9911 |
| Replica/office2/mono | 2 | local0.1 | 0.29472 | 0.30025 | 0.9816 |
| Replica/office2/rgbd | 1 | fixed0 | 0.08614 | 0.08992 | 0.9580 |
| Replica/office2/rgbd | 2 | fixed0 | 0.08616 | 0.09109 | 0.9459 |
| Replica/office2/rgbd | 1 | fixed1000 | 0.58058 | 0.61147 | 0.9495 |
| Replica/office2/rgbd | 2 | local0.03 | 0.38524 | 0.39121 | 0.9847 |
| Replica/office2/rgbd | 1 | local0.05 | 0.42730 | 0.43206 | 0.9890 |
| Replica/office2/rgbd | 1 | local0.1 | 0.51059 | 0.51147 | 0.9983 |
| Replica/office2/rgbd | 2 | local0.1 | 0.50073 | 0.51440 | 0.9734 |
| Replica/office3/mono | 1 | fixed0 | 0.06478 | 0.06640 | 0.9757 |
| Replica/office3/mono | 1 | fixed1000 | 0.53535 | 0.54344 | 0.9851 |
| Replica/office3/mono | 1 | local0.01 | 0.35677 | 0.40321 | 0.8848 |
| Replica/office3/mono | 1 | local0.05 | 0.42931 | 0.43194 | 0.9939 |
| Replica/office3/mono | 2 | local0.05 | 0.41992 | 0.47809 | 0.8783 |
| Replica/office3/mono | 1 | local0.1 | 0.53023 | 0.56231 | 0.9430 |
| Replica/office3/mono | 2 | local0.1 | 0.53227 | 0.60886 | 0.8742 |
| Replica/office3/mono | 2 | local0.3 | 0.91533 | 0.95789 | 0.9556 |
| Replica/office3/rgbd | 1 | fixed0 | 0.10048 | 0.10221 | 0.9831 |
| Replica/office3/rgbd | 2 | fixed0 | 0.09883 | 0.10121 | 0.9765 |
| Replica/office3/rgbd | 2 | local0.03 | 0.62615 | 0.62940 | 0.9948 |
| Replica/office3/rgbd | 2 | local0.05 | 0.72068 | 0.83595 | 0.8621 |
| Replica/office3/rgbd | 1 | local0.08 | 0.83268 | 0.83962 | 0.9917 |
| Replica/office4/mono | 1 | fixed0 | 0.07002 | 0.07003 | 0.9999 |
| Replica/office4/mono | 1 | local0.1 | 0.46974 | 0.46980 | 0.9999 |
| Replica/office4/rgbd | 1 | fixed0 | 0.10754 | 0.11192 | 0.9609 |
| Replica/office4/rgbd | 2 | fixed0 | 0.10548 | 0.11224 | 0.9398 |
| Replica/office4/rgbd | 1 | fixed1000 | 0.88877 | 0.90405 | 0.9831 |
| Replica/office4/rgbd | 1 | local0.05 | 0.82063 | 0.82714 | 0.9921 |
| Replica/room0/mono | 1 | fixed0 | 0.08744 | 0.09165 | 0.9541 |
| Replica/room0/mono | 2 | fixed0 | 0.08921 | 0.09113 | 0.9789 |
| Replica/room0/mono | 2 | local0.01 | 0.41439 | 0.42263 | 0.9805 |
| Replica/room0/mono | 2 | local0.08 | 0.58723 | 0.74997 | 0.7830 |
| Replica/room0/rgbd | 1 | fixed0 | 0.13814 | 0.14567 | 0.9483 |
| Replica/room0/rgbd | 2 | fixed0 | 0.13807 | 0.14291 | 0.9661 |
| Replica/room0/rgbd | 1 | fixed1000 | 1.53643 | 1.73549 | 0.8853 |
| Replica/room0/rgbd | 1 | local0.01 | 1.37485 | 1.37499 | 0.9999 |
| Replica/room0/rgbd | 1 | local0.03 | 1.66679 | 1.77022 | 0.9416 |
| Replica/room0/rgbd | 1 | local0.05 | 1.89424 | 1.91414 | 0.9896 |
| Replica/room0/rgbd | 2 | local0.05 | 1.90561 | 1.95064 | 0.9769 |
| Replica/room0/rgbd | 1 | local0.08 | 2.27924 | 2.30193 | 0.9901 |
| Replica/room0/rgbd | 2 | local0.08 | 2.26893 | 2.30631 | 0.9838 |
| Replica/room1/mono | 1 | fixed0 | 0.07915 | 0.08308 | 0.9528 |
| Replica/room1/mono | 2 | fixed0 | 0.07885 | 0.07908 | 0.9971 |
| Replica/room1/mono | 1 | fixed1000 | 0.58609 | 0.58960 | 0.9940 |
| Replica/room1/mono | 2 | fixed1000 | 0.58698 | 0.59305 | 0.9898 |
| Replica/room1/mono | 2 | local0.03 | 0.42596 | 0.42739 | 0.9966 |
| Replica/room1/mono | 2 | local0.05 | 0.47129 | 0.47332 | 0.9957 |
| Replica/room1/mono | 1 | local0.08 | 0.54254 | 0.54728 | 0.9913 |
| Replica/room1/mono | 2 | local0.08 | 0.54414 | 0.54543 | 0.9976 |
| Replica/room1/mono | 1 | local0.1 | 0.59527 | 0.60975 | 0.9762 |
| Replica/room1/mono | 2 | local0.1 | 0.59222 | 0.59935 | 0.9881 |
| Replica/room1/rgbd | 1 | fixed0 | 0.09945 | 0.10299 | 0.9657 |
| Replica/room1/rgbd | 2 | fixed0 | 0.09935 | 0.10030 | 0.9905 |
| Replica/room1/rgbd | 1 | local0.01 | 0.67599 | 0.67818 | 0.9968 |
| Replica/room1/rgbd | 1 | local0.03 | 0.77313 | 0.77361 | 0.9994 |
| Replica/room1/rgbd | 2 | local0.05 | 0.85965 | 0.86003 | 0.9996 |
| Replica/room1/rgbd | 2 | local0.1 | 1.08851 | 1.08856 | 1.0000 |
| Replica/room2/mono | 1 | fixed0 | 0.06979 | 0.07050 | 0.9900 |
| Replica/room2/mono | 2 | fixed0 | 0.07063 | 0.07317 | 0.9654 |
| Replica/room2/mono | 2 | fixed1000 | 0.52206 | 0.53421 | 0.9773 |
| Replica/room2/mono | 1 | local0.1 | 0.48977 | 0.49162 | 0.9962 |
| Replica/room2/mono | 2 | local0.1 | 0.49972 | 0.50690 | 0.9858 |
| Replica/room2/rgbd | 1 | fixed0 | 0.10428 | 0.10790 | 0.9665 |
| Replica/room2/rgbd | 2 | fixed0 | 0.10277 | 0.10746 | 0.9564 |
| Replica/room2/rgbd | 2 | fixed1000 | 0.91254 | 0.93424 | 0.9768 |
| Replica/room2/rgbd | 1 | local0.05 | 0.85924 | 0.86085 | 0.9981 |
| Replica/room2/rgbd | 1 | local0.08 | 1.01330 | 1.02752 | 0.9862 |
| Replica/room2/rgbd | 1 | local0.1 | 1.10989 | 1.14962 | 0.9654 |
| Replica/room2/rgbd | 2 | local0.1 | 1.13154 | 1.21245 | 0.9333 |
| TUM_RGBD/freiburg1_desk/rgbd | 1 | fixed0 | 0.06820 | 0.07345 | 0.9285 |
| TUM_RGBD/freiburg1_desk/rgbd | 2 | fixed0 | 0.06948 | 0.07528 | 0.9229 |
| TUM_RGBD/freiburg1_desk/rgbd | 2 | local0.01 | 0.25765 | 0.26697 | 0.9651 |
| TUM_RGBD/freiburg1_desk/rgbd | 1 | local0.03 | 0.27840 | 0.28323 | 0.9830 |
| TUM_RGBD/freiburg1_desk/rgbd | 2 | local0.1 | 0.37631 | 0.37913 | 0.9926 |
| TUM_RGBD/freiburg1_room/rgbd | 1 | fixed1000 | 0.59354 | 0.59366 | 0.9998 |
| TUM_RGBD/freiburg1_room/rgbd | 1 | local0.01 | 0.27598 | 0.35460 | 0.7783 |
| TUM_RGBD/freiburg1_room/rgbd | 2 | local0.01 | 0.27553 | 0.27619 | 0.9976 |
| TUM_RGBD/freiburg1_room/rgbd | 1 | local0.03 | 0.30479 | 0.36224 | 0.8414 |
| TUM_RGBD/freiburg1_room/rgbd | 2 | local0.03 | 0.29393 | 0.33522 | 0.8768 |
| TUM_RGBD/freiburg1_room/rgbd | 1 | local0.08 | 0.35320 | 0.36778 | 0.9604 |
| TUM_RGBD/freiburg1_room/rgbd | 2 | local0.2 | 0.50431 | 0.52517 | 0.9603 |
| TUM_RGBD/freiburg1_room/rgbd | 2 | local0.7 | 0.91827 | 1.03773 | 0.8849 |
| TUM_RGBD/freiburg1_xyz/rgbd | 1 | fixed0 | 0.05840 | 0.05860 | 0.9965 |
| TUM_RGBD/freiburg1_xyz/rgbd | 2 | fixed0 | 0.05871 | 0.05971 | 0.9833 |
| TUM_RGBD/freiburg1_xyz/rgbd | 1 | fixed1000 | 0.43500 | 0.43685 | 0.9958 |
| TUM_RGBD/freiburg1_xyz/rgbd | 1 | local0.01 | 0.26766 | 0.27480 | 0.9740 |
| TUM_RGBD/freiburg2_xyz/mono | 1 | fixed0 | 0.07288 | 0.07628 | 0.9555 |
| TUM_RGBD/freiburg2_xyz/mono | 2 | fixed1000 | 0.65597 | 0.66447 | 0.9872 |
| VECtor/corner-slow/stereo | 2 | fixed0 | 0.04367 | 0.04432 | 0.9853 |
| VECtor/corner-slow/stereo | 2 | local0.08 | 0.19136 | 0.19228 | 0.9953 |
| VECtor/corner-slow/stereo | 2 | local0.1 | 0.20312 | 0.20352 | 0.9981 |
| VECtor/corridors-dolly/stereo | 1 | fixed0 | 0.06320 | 0.06330 | 0.9983 |
| VECtor/corridors-dolly/stereo | 2 | fixed0 | 0.06211 | 0.06215 | 0.9993 |
| VECtor/robot-normal/mono | 2 | fixed0 | 0.05230 | 0.05287 | 0.9892 |

Full诊断48个场景轮次中35项更慢，几何平均0.9831×。Full代码路径没有新增网络计算；实测变化仍需如实记录，不能声称逐场景Full不退化。本轮不改GPU光栅化，不能将其与此前低点数Rasterizer研究的Full验收混同。

## CPU分段诊断

分段是计时外独立采样，各项中位不能机械相加。

| 条件 | 版本 | 候选/视锥 | 视角 | Selector | 选择 | Comp及发布 |
| --- | --- | --- | --- | --- | --- | --- |
| local0.01 | baseline | 0.10865 | 0.00012 | 0.10431 | 0.06306 | 0.02936 |
| local0.01 | combined | 0.11048 | 0.00012 | 0.09797 | 0.06275 | 0.02949 |
| local0.7 | baseline | 0.12094 | 0.00013 | 0.14863 | 0.27638 | 0.85323 |
| local0.7 | combined | 0.12199 | 0.00013 | 0.15450 | 0.08185 | 0.84056 |
| global0.1 | baseline | 0.03849 | 0.00011 | 0.28640 | 0.28349 | 0.40312 |
| global0.1 | combined | 0.03893 | 0.00011 | 0.27507 | 0.16644 | 0.39386 |

## 完整CPU准备＋H2D＋原GPU渲染

双方均使用上一轮已有的原生三槽交接，不拿旧Python交接再次算收益。本帧属性和51个相机FP32同包上传；GPU只渲染。三开发场景、3个跳转相机、11条件、两轮反序，每条件54同步完成样本。包含CPU准备、排队、上传、依赖、原Rasterizer及最终同步；不含相机矩阵构造/I/O/显示，未把流水线吞吐当单帧延迟。两边线程池在该测试中共驻留但依次使用，不能机械地把CPU单池微测相加得到完整路径。

| 场景 | 条件 | 旧完整ms | 新完整ms | 加速 | 旧较大p95 | 新较大p95 |
| --- | --- | --- | --- | --- | --- | --- |
| Replica__office2__rgbd | fixed0 | 0.2066 | 0.2088 | 0.9892× | 0.2148 | 0.2175 |
| Replica__office2__rgbd | fixed1000 | 1.7019 | 1.7070 | 0.9971× | 1.9254 | 4.2880 |
| Replica__office2__rgbd | fixed5000 | 2.7988 | 2.7372 | 1.0234× | 3.3016 | 6.2164 |
| Replica__office2__rgbd | global0.01 | 2.0457 | 1.9772 | 1.0347× | 3.4639 | 5.3728 |
| Replica__office2__rgbd | global0.1 | 3.1400 | 2.9764 | 1.0549× | 3.2518 | 6.5412 |
| Replica__office2__rgbd | local0.01 | 1.5074 | 1.4946 | 1.0086× | 3.1358 | 1.7907 |
| Replica__office2__rgbd | local0.05 | 1.5808 | 1.5742 | 1.0042× | 4.3416 | 3.9334 |
| Replica__office2__rgbd | local0.08 | 1.6550 | 1.6452 | 1.0059× | 4.0291 | 4.0507 |
| Replica__office2__rgbd | local0.1 | 1.6790 | 1.6698 | 1.0055× | 2.5340 | 2.0087 |
| Replica__office2__rgbd | local0.5 | 2.6393 | 2.4707 | 1.0682× | 5.7496 | 3.7128 |
| Replica__office2__rgbd | local0.7 | 3.0828 | 2.8750 | 1.0723× | 7.0770 | 4.9752 |
| Replica__room0__rgbd | fixed0 | 0.2645 | 0.2711 | 0.9758× | 0.2735 | 0.2770 |
| Replica__room0__rgbd | fixed1000 | 2.3873 | 2.3895 | 0.9991× | 4.1176 | 3.1291 |
| Replica__room0__rgbd | fixed5000 | 3.7908 | 3.6660 | 1.0341× | 6.2749 | 4.4592 |
| Replica__room0__rgbd | global0.01 | 3.4628 | 3.3972 | 1.0193× | 4.6389 | 3.5744 |
| Replica__room0__rgbd | global0.1 | 5.9935 | 5.6613 | 1.0587× | 8.4715 | 5.9237 |
| Replica__room0__rgbd | local0.01 | 2.0751 | 2.0898 | 0.9930× | 2.7623 | 5.1654 |
| Replica__room0__rgbd | local0.05 | 2.3628 | 2.3701 | 0.9969× | 3.3078 | 4.8402 |
| Replica__room0__rgbd | local0.08 | 2.6734 | 2.6650 | 1.0032× | 5.9407 | 4.2030 |
| Replica__room0__rgbd | local0.1 | 2.7099 | 2.7323 | 0.9918× | 4.3896 | 4.4747 |
| Replica__room0__rgbd | local0.5 | 5.6864 | 5.2612 | 1.0808× | 9.9301 | 9.2468 |
| Replica__room0__rgbd | local0.7 | 6.6502 | 6.1471 | 1.0818× | 11.9825 | 10.6577 |
| TUM_RGBD__freiburg1_xyz__rgbd | fixed0 | 0.1751 | 0.1767 | 0.9905× | 0.1818 | 0.1869 |
| TUM_RGBD__freiburg1_xyz__rgbd | fixed1000 | 1.0136 | 1.0135 | 1.0001× | 1.0668 | 1.0520 |
| TUM_RGBD__freiburg1_xyz__rgbd | fixed5000 | 1.9678 | 1.8044 | 1.0906× | 3.0815 | 1.8889 |
| TUM_RGBD__freiburg1_xyz__rgbd | global0.01 | 0.9100 | 0.8993 | 1.0119× | 2.8681 | 3.5382 |
| TUM_RGBD__freiburg1_xyz__rgbd | global0.1 | 1.1236 | 1.0854 | 1.0352× | 1.1571 | 3.6608 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.01 | 0.8029 | 0.7945 | 1.0105× | 2.7865 | 0.8387 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.05 | 0.9340 | 0.9310 | 1.0032× | 1.1446 | 0.9712 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.08 | 0.9331 | 0.9355 | 0.9974× | 0.9785 | 0.9802 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.1 | 0.9876 | 0.9909 | 0.9967× | 1.0617 | 4.7224 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.5 | 1.7648 | 1.6343 | 1.0799× | 1.8842 | 1.9493 |
| TUM_RGBD__freiburg1_xyz__rgbd | local0.7 | 2.1872 | 1.9637 | 1.1138× | 5.3839 | 2.1385 |

共7,128个完整路径样本、198组RGB/radii/身份核对；22/66个场景轮次条件更慢。当前证据不支持把CPU组件收益直接宣称为所有实际帧的渲染加速。原始CSV保留host准备排队时间和总延迟。

## 线程长尾及撤回候选

诊断构建记录每次线程分发的worker启动延迟、线程墙钟时间、线程CPU时间及起止CPU编号。trace01覆盖16,200请求，含统计开销，只用于定位，不作正式速度结论。默认策略在一些请求中出现毫秒级分发等待及wall−threadCPU差，说明长尾不全是矩阵运算；起止CPU编号相同不能排除期间迁移，wall−CPU差也不能唯一归因为某个外部进程。

| 策略 | 条件 | 中位ms | p95 ms | 最大ms | 最大启动延迟ms | 最大wall−CPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| control | fixed0 | 0.0926 | 0.0990 | 1.9303 | 0.5237 | 0.4787 |
| control | fixed1000 | 0.7062 | 0.8931 | 5.5876 | 1.3869 | 1.3959 |
| control | local0.01 | 0.4997 | 0.5894 | 6.0271 | 1.4984 | 1.6873 |
| pinned | fixed0 | 0.0957 | 0.1008 | 0.2747 | 0.0688 | 0.0196 |
| pinned | fixed1000 | 0.7551 | 0.9794 | 11.4936 | 9.1779 | 3.9080 |
| pinned | local0.01 | 0.5141 | 0.6192 | 11.3505 | 6.8518 | 0.2265 |
| sleep | fixed0 | 0.0951 | 0.0990 | 0.3072 | 0.0870 | 0.0407 |
| sleep | fixed1000 | 0.7414 | 1.1035 | 4.8378 | 3.8462 | 1.2043 |
| sleep | local0.01 | 0.5126 | 0.5952 | 4.8294 | 2.9698 | 1.1344 |

- 缩短自旋、worker绑核、作用域主线程绑核均实际编译测试，没有稳定跨预算收益；绑核后仍有较大长尾。最终Pool与上一轮源码逐字相同，未保留这些策略。
- 将小输出FP32拆为偶/奇两条累加链会改变求和顺序，开发集最大属性差约0.000488，速度收益不稳定，已撤回；没有因为用户允许小误差就默认接受无收益的误差。
- 始终扫描全部候选的压缩在低K可能比原排序更慢，保留为独立实验开关，最终使用预先冻结的工作量判断。
- 开发版本源码与二进制在`source/development`、`development_binary`保留，tune01/02/03与trace01保留实际结果。没有重复使用已失败的候选充当最终版本。

## 正确性与边界

- 正式CPU 19,584组：身份、K、属性顺序、全部七类属性SHA、finite全部通过，相对上一版最大差0。
- 同分/离散分数、all/striped/one/none资格、空视锥、K跨SIMD/线程边界、局部严格K及全局ties，共1632组通过。
- 常规CPU 33项：输出canary、独立性、源Master修改隔离、非法Ratio/NaN相机拒绝、恢复、同实例并发串行化通过。
- 三种地图CPU checkpoint恢复18组属性/身份逐位通过，CUDA未初始化；原GPU固定噪声快照被保留，未在CPU重新采样替代。
- 原生交接25项：错误恢复、动态K、不同消费者Stream、上传属性和相机、回默认Stream渲染逐位通过。原Rasterizer默认Stream假设未改。
- 24组×3个相机×17条件=1224组配对GPU渲染，RGB及radii逐位相同，GT PSNR/SSIM/LPIPS差均为0。只证明本轮相对上一版无新增差异；上一版相对最初GPU PyTorch的近似另见其报告，不能称整体从最初版本始终逐位相同。该测试不是全轨迹质量证明。
- 独立ASan+UBSan正常路径28项通过，独立UBSan完整33项通过。系统ASan与Torch主动异常不兼容已在前轮独立复现，本轮明确排除5个主动异常路径；不称ASan完整异常覆盖。泄漏检测关闭；CPU Sanitizer不覆盖GPU内存或证明无数据竞争。

## 内存、缓存与部署边界

主要Workspace逻辑估算10.30–177.56MiB。新小输出微核只使用线程栈寄存器/少量局部数组，压缩复用原已reserve的selected容量，不新增每Gaussian缓存。估算不含全部权重、分配器、线程栈和进程共享页，不能当RSS；`CPU_MEMORY.csv`保留逐场景逐版本数据。

三槽位pinned Host/Device容量取决于地图N，与上一版同一实现。每槽属性/相机上传由ready/done Event管理，未完成的ticket不能覆盖；prepare/wait/consume/finish的Stream约定同上一版。缓存只属于加载时冻结快照；参数、Feature、Anchor或拓扑变更必须重建。每帧相机/可见性/选择与补偿仍重新计算，没有用同帧缓存命中或GPU网络运算冒充CPU速度。

本轮正式验证Teacher；Student包装兼容接口未独立验收。结果限定当前x86 AVX2 CPU，不代表ARM/手机或其他核心数。还未测长时间能耗、温控下的持续吞吐，不能声称达到CPU硬件极限。

## 后续优化

1. 低Ratio主成本仍是全候选处理与Selector。继续研究固定快照上的空间数据组织、减少不需要的行访问、让连续候选占用更少cacheline；必须保持名义Anchor候选及严格K，不按历史视角近似筛选。
2. 调度长尾需进一步结合OS调度跟踪、频率和硬件计数器，区分worker唤醒/抢占/功耗状态。当前只得到线程时间证据，未定位每个正式慢样本，不能承诺绑核即可修复。
3. 若尝试更激进INT8/近似激活，应单独验证Selector边界身份及补偿投影长尾；本轮无需引入这些误差。空间换时间仍只允许有界且明确失效的冻结缓存。

## 完整性与复现

- 研究目录：`/home/prmi/zwj/experiments/cpu_native_small_20261110_1318`。
- 基线二进制SHA256：`cc51643a8cd9ab07a340aeb8e3a93f56c8499b0c87568139822678ee310263db`。
- 最终二进制：`/home/prmi/zwj/experiments/cpu_native_small_20261110_1318/build/mycartgs_cpu_small.so`。
- 最终SHA256：`42d4cc3fc34d656297955e420db0b62be8db9a7a6f62cbb1cc4acc91fbb05863`。
- 保护文件2939个：原生产/训练/模型/前轮关键证据变化0、缺失0；本轮冻结源码与二进制哈希一致，无活跃本轮任务，无渲染图片。
- CSV包含逐场景逐Ratio、N/M/K、中位/p95/最大延迟、全部慢例、质量和分段；原始样本与监测日志均保留。曲线与数据哈希见figures。使用方式见`README_CPU_SMALL.md`。
- 未commit、push、部署，未改共享环境。清理仅限本轮可重建Sanitizer对象，清单另存；最终二进制、失败候选及回退版本均保留。
