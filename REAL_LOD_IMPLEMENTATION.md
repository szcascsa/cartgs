# RealLoD：Full建图后独立训练选择与补偿网络

实现分支：`real-lod`，起点 `857909570760b61337e698af63b725524d164619`。

2026-11-03更新：当前默认Fine每步对同一相机计算Full＋当前ratio LoD损失，累计梯度后提交一次共享Adam，Master可训练。`RealLoD.full_loss_weight: 0`恢复LoD-only；再设`Selector.detach_gaussian_parameters: 1`恢复冻结Master。历史冻结结果保留于`REAL_LOD_FROZEN_MASTER_REPORT.md`；本次联合目标实现与验证见`REAL_LOD_FULL_JOINT_IMPLEMENTATION.md`。

## 训练协议

`Training.real_lod: 1`启用三个互相分开的阶段：

1. 在线SLAM期间只运行Full训练；沿用Anchor Gaussian地图及原Full图像金字塔。
2. SLAM结束后执行原正常Full尾优化；默认不设置额外固定总步数。
3. Full尾部正常结束后，执行独立Fine优化，步骤数等于实际在线Mapping步骤数。LoD轮转`0.2→0.4→0.6→0.8`，每步另计算Full损失；两支同一参数状态、分别backward、只提交一次Adam，Full不占ratio轮转槽。

LoD训练使用原分辨率、保存的全部训练关键帧随机采样。在线步骤、Full尾部步骤、LoD步骤及重要性统计渲染分别计数。`Selector.max_mapping_iterations`的旧总上限不截断新协议；LoD阶段使用独立预算，不受旧优化总迭代上限或在线预热影响。

默认`Selector.anchor_only_pruning: 1`，在线及Full尾部只做整Anchor删除，Clone/Split等Full拓扑行为仍沿用当前实现。Fine阶段固定Gaussian数量和Anchor成员关系，不新增、剪枝或重置Opacity；Master保留Full及LoD梯度，沿用原学习率时钟和Adam历史。

Selector、Compensator及Anchor Feature在在线Full和正常Full尾部没有训练前向/反向，也不推进Adam状态。Fine阶段仍使用原共享Adam及相同网络参数组；Full损失只直接更新Master，LoD损失更新Master、Selector、Compensator和Feature。GI目标detach，GI/Budget预测路径保留当前live Selector输入梯度；不采用原Maturity私有VJP。

Fine默认目标为`photo(Full)+photo(LoD)+GI+0.01 Budget+0.001 Opacity`，photo沿用本项目`0.8 L1+0.2 DSSIM`，实际系数读取当前配置。新Full权重只乘Full photo，Opacity正则每步只计一次；关闭Full分支时保留历史LoD-only损失。该定义采用FlexGS联合监督结构，没有照搬官方非对称L1/SSIM系数。

## 保留的网络与四节点Feature

- Selector：原11D输入，`11→32→32→2`。默认不启用Set/Context变体。
- Compensator：Offset3、Residual Feature48及Ratio3，逻辑54D输入；共享64/64主干、四Head、14D输出。
- Scale保留当前相对Softplus表示；保留现有FP32 CUDA训练输入融合。
- 每Anchor一份48D Base和三份48D残差，节点为20%、40%、60%、80%。

\[
e_a(r)=b_a+\sum_{j=1}^{3}\operatorname{clamp}
\left(\frac{r-r_{j-1}}{r_j-r_{j-1}},0,1\right)d_{a,j}.
\]

连续Ratio3编码范围改为节点最小值和最大值，即默认`[0.2,0.8]`。网络主干宽度及输出定义不变。四张Feature表拥有独立Adam状态，未激活残差保持undefined gradient，不用零梯度代替。

C++及Python累计残差路径支持2～6个节点，兼容旧六节点checkpoint；Bandwidth变体仍要求其原来的六个节点。新模型不把20%以下或80%以上的查询解释为已验证的质量泛化。

## FlexGS重要性：依据实际官方源码

参照本地`reference/FlexGS-official-20260915`，Git版本：

```text
0e9b91f4ddb294c4260b6b83b1dbbb90c1be1982
```

官方`gaussian_renderer/__init__.py::prune_list()`遍历全部训练相机，对未经过选择/补偿的Master执行`count_render`，逐视图相加。其CUDA在满足原Alpha阈值和透射率早停条件的有效贡献处累加：

\[
I_i=\sum_{v\in\mathcal V_{train}}\sum_{p\in\mathrm{valid}(i,v)}o_iT_{i,v,p}.
\]

其中`o_i`为激活Opacity；不是像素有效Alpha `o_i exp(power)`。没有除以像素数，没有EMA，没有按可见次数求平均，也没有显式体积因子。官方源码的统计加法缺少原子保护；本项目复用已有atomicAdd统计，遵循同一统计量，不复现数据竞争，不声称与官方逐bit一致。

实现复用本项目`collect_importance=1`路径，使用Full原分辨率和全部当前训练关键帧，不采用Streaming GI Bank采样、不使用深度、不通过Selector候选或Compensator生成统计。`RasterizeMode::Inference`只避免为统计构造训练反向缓存，保持有效贡献规则。

更新时机对应官方`train.py`：LoD第1步及每1000步更新；刷新读取当前已Joint更新的Master。`FlexGSImportance.refresh_interval`可配置。统计渲染无backward、无Adam更新，另外记录次数和耗时。

伪标签遵循官方代码的分位数及严格大于：

\[
k=\lfloor(1-r)(N-1)\rfloor,\quad
y_i=\mathbf1[I_i>\operatorname{sort}(I)_k],\quad
L_{GI}=\operatorname{mean}|m_i^{ST}-y_i|.
\]

同一份实际可微Gumbel hard mask参与Photo、Budget及GI损失。保留零观测行、分数相等时的官方行为，不将其替换为精确Top-K。默认`Selector.gi_loss_weight: 1.0`，与官方默认Gumbel重要性损失权重相同。

只替换Maturity引导，不声称完整复刻FlexGS训练器：本项目仍保留现有Photo损失、归一化Budget Loss、Anchor视锥候选与Slot规则。尤其全图预算`rN`可能高于视锥内候选数量，必须查看实际K和预算缺口。

## 配置与输出

默认入口仍为`cfg/selector/selector_config.yaml`和`spatial_offset3_residual48.yaml`：

```yaml
Training.real_lod: 1
RealLoD.full_loss_weight: 1.0
FlexGSImportance.enabled: 1
FlexGSImportance.refresh_interval: 1000
Selector.anchor_only_pruning: 1
Selector.target_ratios: [0.2, 0.4, 0.6, 0.8]
Selector.gi_loss_weight: 1.0
Selector.detach_gaussian_parameters: 0
MaturityV2.enabled: 0
MaturityV2.selector_loss_weight: 0.0
MaturityV2.compensation_gate: 0
MaturityV2.contribution_enabled: 0
```

新协议拒绝同时启用Maturity、SGI、TF、Replay或Gradient Guard，避免出现未定义的阶段和梯度组合。`Training.real_lod`缺省0保留旧配置行为；旧配置完整副本在`cfg/selector/spatial_offset3_residual48_legacy.yaml`，不能仅把新配置一个开关设0，就视为恢复所有旧参数。

保存`*_online_end`、`*_full_end`和`*_shutdown`，分别对应在线结束、Fine训练前、Fine训练后。`real_lod_state.json`增加Full训练渲染/反传、LoD训练渲染/反传、Adam提交次数及目标函数权重。`real_lod_steps.csv`保留原LoD损失列，追加`full_loss`和`joint_loss`；`flexgs_importance.pt`保存最近一次重要性统计，与该checkpoint的行顺序对应。现有模型导出并非完整SLAM/Adam/RNG续训快照，状态文件明确记录这一边界。

本轮不增加按距离的局部LoD渲染或新Viewer。网络开始更新只是功能证据，不能据此证明渲染质量优于旧方案；质量收益需要匹配训练对照。

## 首版Joint模式验证记录

### 编译与原生回归

1. 启动器23项测试通过；Python语法及`git diff --check`通过。
2. CUDA11.8、GCC9、Release、sm86独立构建通过，包括`replica_rgbd`、`tum_rgbd`、`tum_mono`及三个回归目标。全新构建耗时713.25秒。
3. `real_lod_schedule_test`通过：在线与Full尾部独立计数、仅ratio轮转、零在线预算及非法阶段切换。
4. 原`residual48_training_test`的97项FP32逐位前向/VJP回归通过。
5. `real_lod_test`通过：8步在线Full＋3步Full尾部期间网络及Feature参数不变、网络Adam状态不推进；8步LoD联合更新Master、Selector、Compensator和Feature。四张Feature表的Adam时钟为`8/6/4/2`，未激活残差不推进时钟。
6. 单Gaussian统计验证`I=有效贡献像素数×0.5`，明确排除了Alpha权重或像素平均。双相机原始重要性求和通过；全零、并列分数及detach标签行为通过。
7. C++和Python均能读取新四节点checkpoint；旧六节点schema读写回归通过。旧六节点专用fixture为空表，原六节点非空训练/VJP另由97项回归覆盖，不把schema测试称为全部历史模型验证。

### 真实在线功能验证

场景为Replica office2 RGB-D前400帧，复用既有只读数据副本，没有设置额外Mapping上限。在线、正常Full尾部和等在线步数LoD一次运行正常结束，退出码0。结果是功能验证，未做旧方法配对质量对照，也未在此轮跑完整office2序列。

| 阶段 | Mapping/训练渲染数 | Full backward | Master实际更新 | Selector更新 |
|---|---:|---:|---:|---:|
| 在线Full | 4,112 | 4,112 | 4,105 | 0 |
| 正常Full尾部 | 788 | 788 | 787 | 0 |
| 独立LoD | 4,112 | 0；LoD backward 4,112 | 4,112 | 4,112 |

在线与Full尾部共8次拓扑事件；现有拓扑流程并非每个Mapping都提交Master Adam，因而Mapping数和Master实际更新数不能混同。本方案按用户要求对齐在线Mapping次数4,112，而非4,105次Master提交。独立LoD不发生拓扑事件，每一步都有实际联合Adam更新。

- 四档ratio分别曝光1,028次，LoD始终1200×680，不插入Full训练槽。
- 重要性于LoD第1、1000、2000、3000、4000步刷新，共5次×42个训练关键帧＝210次额外Full统计渲染；不计入训练曝光，没有backward。统计耗时合计681.89毫秒。
- LoD训练时段计时合计29.89秒，含重要性统计；整个子进程61.16秒。这里是一次功能运行耗时，不是与旧版本的速度对照。
- 四张48D Feature表的实际Adam步数为`4112/3084/2056/1028`，所有表均发生变化且保持有限。LoD非有限Loss/梯度失败数为0。
- LoD前后Gaussian均为18,894个、Anchor均为8,502个。导出模型逐值核对Anchor成员行映射、成员数、UID、下一UID、中心、参考尺度、格子键、层级及分支均保持不变。
- Python实际加载Full结束和LoD结束模型，确认Selector11D、Compensator逻辑54D、全部参数有限，Selector/Compensator/四张Feature表均实际更新。查询20%、25%、30%、40%、50%、60%、70%、80%得到有限的48D Feature；这只证明插值查询正常，不证明未见ratio质量。

独立服务器目录：`/home/prmi/zwj/experiments/real_lod_20261103`。

模型位置：

```text
runs/OFFICE2_RGBD/4112_online_end/ply
runs/OFFICE2_RGBD/4900_full_end/ply
runs/OFFICE2_RGBD/9012_shutdown/ply
```

### 隔离性与证据

`ldd`确认Mapper、Viewer、Rasterizer、simple_knn和imgui来自本轮独立`source/lib`；transform_field_cuda为本轮`build`中的静态库，链接进Mapper。仅只读复用生产项目third_party依赖。服务器主项目328个受保护源码、配置、二进制和动态库的研究前后SHA256一致。没有覆盖生产默认、修改共享环境、提交或push。

本地证据在`research/real_lod_20261103/evidence/`：`native_test.log`、`residual48_test.log`、`online_verification.json`、`online_export_validation.json`、`runtime_identity.json`和`production_integrity.json`。源文件清单及归档在同级`source_manifest.json`、`source.tar.gz`；模型和完整运行结果留在服务器独立目录。

尚未验证：完整序列的Full/逐ratio PSNR、SSIM、LPIPS，旧方案配对质量比较，以及按相机距离混合LoD的渲染实现。下一项质量实验应固定Full结束模型，再配对比较无GI与本FlexGS GI，避免把不同在线地图归因于重要性监督。
