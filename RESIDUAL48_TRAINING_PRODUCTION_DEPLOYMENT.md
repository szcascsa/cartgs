# Residual48 输入训练融合生产部署记录

日期：2026-11-01。服务器 `172.19.163.58`，项目 `/home/prmi/zwj/mycartgs`，环境 `zwj`。

## 部署结果

按用户本次明确要求，将已经完成逐节点诊断的 **Residual48 输入训练融合**设为生产默认。生产代码提交：`2b87284a4e96bac788110a165309b0ee26f21804`。没有 push。

服务器226个源码、配置及回归测试文件与本地部署清单逐文件 SHA256 一致；本次实际同步12个差异文件。8个生产入口与共享库已重新编译，构建耗时227.70秒。`ldd` 确认8个入口加载主项目的新库，无缺失依赖，也没有加载研究目录的库。

部署目录：`/home/prmi/zwj/deployments/residual48_training_20261101`。最终版本身份保存在该目录 `version.json`、`deployment_complete.json` 及主项目 `DEPLOYED_VERSION.json`。

## 实际启用的路径

```yaml
CUDA.fused_residual48_input: 1
CUDA.fused_residual48_training_input: 1
```

两个开关分别控制 no-grad 推理输入融合和训练输入融合。头文件成员缺省、C++配置读取缺省、生产 YAML 及 `spatial_offset3_residual48.yaml` 均已对应设置；省略新键也会启用，显式0可关闭。

- 训练保留每张表的原生 `index_select`，随后融合残差按原序累加、Offset 的归一化/asinh/clamp及51D输入打包；Ratio3仍通过原有网络路径处理，逻辑输入仍为54D。
- 自定义反向融合 XYZ 的局部 VJP；Feature 分支保留原 cat-view/scalar-mul VJP，继续由原生 Scatter 回传到六张独立 Feature 表。
- 前向保存每点6个FP32中间值，不跨步缓存 Master、Feature 或拓扑数据。中心、参考尺度或归一化统计需要梯度时，回退原 ATen 表达式，保留相应梯度。
- 六表参数、Adam一二阶矩、独立时钟、未激活表的 undefined grad 与旧 checkpoint 格式保持原有定义。
- 四Head主体、Selector、Maturity、Loss、学习率、ratio、预算与拓扑规则沿用当前配置。本次没有启用四Head训练融合或强制确定性 Scatter。

生产配置逐键比较只增加上述两个 CUDA 开关；非CUDA训练配置差异为0。1000步Full预热、物理Scale加法及贡献修正40%的Maturity V2仍由原生产配置提供。生产停止预算未改为1500。

## 数值回归

使用本次主项目头文件、新静态库及共享库重新编译独立回归驱动：

| 检查 | 实际结果 |
|---|---|
| 局部前向 / XYZ VJP / gathered Feature VJP | 97组逐bit通过；含空K、动态K、未见ratio、strided上游、极值及几何梯度回退 |
| 真实room0 checkpoint逐节点检查 | 11 ratio ×3 seed，9450条节点记录；前向、Loss、GatherBackward输入、Master XYZ及网络参数梯度逐bit一致 |
| 确定性Scatter下连续更新 | 2 seed ×72步；2592条记录全通过，包括网络/Feature参数、Feature一二阶矩及独立时钟 |
| 正常生产Scatter | 仍出现已诊断的原生原子累加位差，没有将其标为逐bit通过 |

首次连续更新驱动仍引用旧诊断命名空间，发生一次测试驱动编译失败；修正为生产函数后重新编译并通过。失败日志留在 `.attempt1` 文件中，生产构建未因此修改。

## office2 在线部署冒烟

使用真实生产 `replica_rgbd`，Replica office2前400帧，独立测试配置限1500 Mapping；该限制只存在于部署目录的测试配置。故意删除两个新键，验证C++缺省，而非只检查YAML显式值。

| 观测 | 结果 |
|---|---:|
| Online / Tail Mapping | 1500 / 0 |
| Full render / backward | 1071 / 1071 |
| LoD render / Selector更新 | 429 / 429 |
| Master实际更新 | 1500 |
| LoD非有限值 | 0 |
| 六ratio曝光 | 72、72、72、71、71、71 |
| Gaussian / Anchor数 | 9877 / 6793 |
| 六表Adam step | 429、357、285、213、142、71 |
| Feature有限性 | 六表全部通过 |
| Compensation共享网络参数量 | 25230 |

Nsight Systems在真实在线运行中观察到 `trainingPack` **429次**、`trainingXYZVjp` **429次**，证明训练融合前向和反向实际执行。运行导出的有效配置也包含两个开关均为1。

带Nsight的测试进程耗时37.30秒；该值包含Profiling开销，**不用于宣称端到端提速**。本次未新增完整场景质量对照，也未证明完整异步在线训练轨迹或长期PSNR逐bit等价。

## 数值边界

原生共享Anchor Scatter的FP32原子加法仍不确定。前次诊断已将首差定位到原生Scatter，而非融合局部VJP或Adam实现；随机VJP压力轨迹也证明小差异可能经Adam反馈放大。本次依用户明确决定采用该训练路径，没有把这一限制改写为“完整Zero-Change验证通过”。确定性模式只在独立回归进程中启用。

## 备份、回退与完整性

更新前的源码、配置、生产入口和库保存在部署目录的 `before/`，对应SHA在 `before.json`。328个原有受保护文件已核验，本次允许范围以外的变化为0；新文件及实际同步清单见 `deploy_manifest.json`。

只关闭训练输入融合可在完整配置中设置：

```yaml
CUDA.fused_residual48_training_input: 0
```

现成回退配置：`/home/prmi/zwj/deployments/residual48_training_20261101/selector_training_fusion_off.yaml`，通过现有 `--selector-config` 使用；保留推理输入融合。省略新键不会关闭融合。

如需恢复此次部署前的源码、配置和二进制，在没有生产SLAM运行时执行：

```bash
/home/prmi/anaconda3/envs/zwj/bin/python /home/prmi/zwj/deployments/residual48_training_20261101/rollback.py
```

回退仅处理 `before.json` 中列出的文件，不触碰数据集、模型或历史结果。历史独立测试二进制没有全部重编译，运行时应从匹配源码重建；本次8个生产入口及本轮回归驱动已重建。

验收证据位于部署目录及本地 `research/residual48_training_deployment_20261101/evidence/`：`native_validation.json`、`online_validation.json`、`runtime_links.json`、`scope_integrity.json`、`nodes.csv`、`trajectories.csv`。Nsight原始报告及SQLite留在服务器，路径、大小和SHA见 `NSIGHT_ARTIFACTS.json`。没有提交模型、构建产物、大型Tensor快照、Nsight数据或凭据；其他未提交研究工作保持原状。
