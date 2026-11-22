# RealLoD Fine：Full＋LoD联合监督实现与验证

日期：2026-11-03。分支real-lod，基准HEAD：857909570760b61337e698af63b725524d164619；本轮为未提交补丁，精确版本见PATCH_MANIFEST.json。

## 结论

已实现在线Full-only→正常Full尾优化→独立Fine。Fine每步使用同一相机和同一更新前Master，先Full反传、再当前ratio LoD反传，累计后只提交一次共享Adam。Master默认可训练；保留LoD-only及冻结Master回退。

原生回归和两组完整TUM fr1_xyz真实在线流程均通过，含全部792个关联RGB-D输入和792帧统一评价。没有非有限Loss/梯度失败；Fine期间Gaussian/Anchor数量不变，Full/LoD/Adam曝光匹配设计。

这些结果确认实现和完整流程可用；不同在线运行的地图及步数不同，本轮未开展同一Full终点和Adam/RNG状态的配对目标函数实验，不能把历史分数差直接当作新增Full Loss的因果收益。

## 代码改动与梯度

- src/real_lod.cpp：新增accumulateRealLodFullGradients，直接渲染未选择/未补偿Master；复用原L1/SSIM和Opacity正则，添加Full Loss finite检查。
- trainRealLodIteration：Full与LoD之间没有清空Master梯度、Adam提交或拓扑操作。Full反向缓存先释放，LoD随后复用现有Renderer/SSIM工作区。
- include/gaussian_mapper.h：新增Full权重及Full渲染/反传计数；因此已重编译全部七个Mapper头文件依赖单元，包括Viewer、三个SLAM主程序及测试，未只替换动态库。
- src/gaussian_mapper.cpp：读取/保存新配置、校验冻结与Full监督组合，并记录目标函数。
- 两份默认Selector YAML：Full权重1、Master detach=0；网络、Feature、Ratio、GI、Budget、学习率及拓扑策略保持原定义。
- 启动器提示Fine每步Full＋一个轮转LoD，只提交一次Adam。

默认目标：

\[L_{\mathrm{Fine}}=\ell(I_{\mathrm{Full}},G)+\ell(I_{\mathrm{LoD}},G)+L_{\mathrm{GI}}+0.01L_{\mathrm{Budget}}+0.001L_{\mathrm{Opacity}},\]

其中 \(\ell=0.8L_1+0.2(1-\mathrm{SSIM})\)，实际系数读取现有配置。这是沿用本项目photo定义的联合目标，不是逐系数复制官方非对称SSIM权重。

| 损失 | Master | Selector | Compensator | Anchor Feature |
|---|---|---|---|---|
| Full photo/Opacity | 有 | 无 | 无 | 无 |
| LoD photo | 有 | 有 | 有 | 有 |
| GI/Budget | 经当前live Selector输入间接传播 | 有 | 无直接路径 | 无直接路径 |

GI标签detach；未激活Feature残差保持undefined gradient及独立Adam时钟。未引入Gaussian私有补偿参数，未修改网络或CUDA Kernel。

## 默认与回退

| 模式 | RealLoD.full_loss_weight | Selector.detach_gaussian_parameters |
|---|---:|---:|
| 新默认Full＋LoD Joint | 1.0 | 0 |
| 历史Joint LoD-only | 0.0 | 0 |
| 历史冻结Master | 0.0 | 1 |

Full权重只乘Full photo；Full开启时Opacity正则每步只计一次。关闭Full时保持历史LoD-only损失。旧显式冻结配置缺少新键时，默认Full权重0；显式正Full权重与detach=1的冲突配置被拒绝。权重要求有限且非负。

默认ratio仍为[0.2,0.4,0.6,0.8]，第二组仅通过独立配置改为[0.01,0.05,0.1,0.3,0.5]。没有加入Full轮转槽，没有设置额外online/总Mapping上限。

## 原生与启动器验证

| 检查 | 结果 |
|---|---|
| Full-only梯度隔离 | 只产生Master梯度，网络和Feature梯度undefined |
| Full＋LoD梯度累计 | 固定fixture的最大绝对差0，融合及原生Adam两路均通过 |
| 每步一次Adam | 8次Fine对应8次Full、8次LoD、Master Adam时钟增加8 |
| Feature更新时钟 | 四张表8/6/4/2，Full不增加Feature曝光 |
| 冻结回退 | 六组Master参数、Adam矩、step及LR逐位不变，网络继续更新 |
| LoD-only回退 | Full训练计数0，Joint Master正常更新 |
| 非有限Full Loss | backward/Adam前拒绝，参数和Adam时钟不变 |
| Checkpoint | 四节点及旧六节点网络归档读取通过 |
| Residual48原回归 | 97项前向/VJP通过 |
| 调度/启动器 | 原生调度及23项启动器测试通过 |

梯度累计测试使用allclose判据，实际观察误差为0；Rasterizer反向包含原子累加，这不是对任意输入和GPU的全局逐bit承诺。新目标函数本身会改变训练，不能称为Zero-Change性能优化。

开发中修复了L1接口的Tensor引用签名，并将测试探针放到GI初始化后；后者初次触发的是测试前置条件，实际生产步骤原本就先刷新GI。最终全部测试通过，保留相关日志，不把失败当作质量结果。

## 两组完整真实流程的实际预算

| ratio组 | online Full | 正常Full tail | Fine | Fine Full/LoD/Adam各次数 | Gaussian | Anchor | ratio曝光 | Fine秒（含GI） |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| r01_05_10_30_50 | 12078 | 400 | 12078 | 12078 | 13893 | 2942 | [2416, 2416, 2416, 2415, 2415] | 99.363 |
| r20_40_60_80 | 12275 | 625 | 12275 | 12275 | 13465 | 2801 | [3069, 3069, 3069, 3068] | 103.410 |

重要性Full扫描另外统计，不属于Full训练曝光。原生与在线断言均检查Full/LoD反传数等于Fine步数；在线Full和正常tail的Selector更新数为0。各组Feature实际Adam时钟等于相应活跃ratio曝光之和。
Fine额外增加N次Full训练渲染和反传；表中耗时是本次实际记录，不是相同工作量性能对照，不声称相对LoD-only有速度提升。

## 792帧渲染质量

| ratio组 | 输出 | PSNR↑ | SSIM↑ | LPIPS↓ | 平均实际K |
|---|---|---:|---:|---:|---:|
| r20_40_60_80 | full | 22.977110 | 0.783081 | 0.247443 | 13465.000 |
| r20_40_60_80 | full_before_lod | 22.637999 | 0.774622 | 0.258770 | 13465.000 |
| r20_40_60_80 | lod_20pct | 22.063034 | 0.753377 | 0.306935 | 2694.000 |
| r20_40_60_80 | lod_40pct | 22.789000 | 0.778826 | 0.253898 | 5315.874 |
| r20_40_60_80 | lod_60pct | 22.858846 | 0.781138 | 0.249008 | 6407.837 |
| r20_40_60_80 | lod_80pct | 22.857546 | 0.781103 | 0.248863 | 6418.823 |
| r01_05_10_30_50 | full | 22.555040 | 0.773388 | 0.268971 | 13893.000 |
| r01_05_10_30_50 | full_before_lod | 22.411120 | 0.770493 | 0.271219 | 13893.000 |
| r01_05_10_30_50 | lod_1pct | 17.595774 | 0.618875 | 0.643698 | 140.000 |
| r01_05_10_30_50 | lod_5pct | 19.593627 | 0.677049 | 0.504112 | 696.000 |
| r01_05_10_30_50 | lod_10pct | 20.382130 | 0.702649 | 0.420750 | 1391.000 |
| r01_05_10_30_50 | lod_30pct | 22.216125 | 0.762453 | 0.291173 | 4159.437 |
| r01_05_10_30_50 | lod_50pct | 22.447672 | 0.770513 | 0.272429 | 6372.484 |

full_before_lod与full使用同一批相机和GT，分别读取Fine前/后Master。全部训练ratio沿用本场景自身N计算预算，没有使用same-K；高ratio实际K仍受原视锥/Slot候选限制。

评价含训练关键帧，不是纯未见视图。没有在本轮选择checkpoint、调学习率或改Loss；没有评价未见ratio、其他场景或多seed，因此不能声称跨场景稳定提升。

Full/LoD联合目标不是Full PSNR不下降的数学保证。若仍出现Full退化，下一项应固定Full终点、Adam、相机和Gumbel序列，隔离Full权重与低ratio相对Budget压力；本轮没有同时调整这些因素。

## 构建、部署与隔离

- 构建根目录：/home/prmi/zwj/experiments/real_lod_full_joint_20261103
- 使用原CMake的GCC9/CUDA11.8/Release/sm86参数，重编译七个受影响CPU单元，复用未变CUDA对象及静态库。
- 原构建目录.ninja_deps、.ninja_log、CMakeCache和build.ninja哈希不变；共享conda及第三方依赖没有修改。
- 旧主项目/home/prmi/zwj/mycartgs的328个受保护文件SHA256复核一致。
- 发布目标为/home/prmi/zwj/realLOD；部署脚本仅在上述测试及完整回归通过后执行，逐文件备份/原子替换并核验ldd。实际发布结果、历史模型核验及回退路径以deployment.json为准。
- 没有commit或push，没有覆盖历史运行；完整模型和评价图片保留在本轮独立目录。

## 模型与证据

- r01_05_10_30_50：/home/prmi/zwj/experiments/real_lod_full_joint_20261103/runs/r01_05_10_30_50/training/24556_shutdown/ply
- r20_40_60_80：/home/prmi/zwj/experiments/real_lod_full_joint_20261103/runs/r20_40_60_80/training/25175_shutdown/ply

online_summary.csv/JSON保存全部质量与实际K，training_verified.json保存阶段/Feature时钟，real_lod_steps.csv保留LoD列并新增full_loss和joint_loss。旧导出不是完整Adam/RNG恢复快照，本轮未把它作为精确配对续训。

原生测试、源码补丁哈希、七单元ABI依赖清单、ldd、运行日志和最终发布核验随证据包归档；大模型、图片和数据集不进入本地证据包。
