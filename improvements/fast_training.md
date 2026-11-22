# Fast 训练配置

启动程序时指定 `--selector-config cfg/selector/fast.yaml`。

`fast.yaml` 基于 `slower.yaml`，仅修改以下四项；TF 启动时间、mapping
预算、ratio、学习率、CD/DSD epochs 和历史检查配置均沿用 `slower.yaml`。

```yaml
Selector.lod_render_scale: 0.5
TransformField.feature_dim: 8
TransformField.multires: [1, 2]
TransformField.hidden_dim: 32
```

默认每次有效 mapping 仍执行完整渲染/反传和 LOD 渲染/反传。
`Training.ultra_light_mode` 默认 `1`，按四次 LoD、一次 Full 轮转，LoD 可联合
更新主 Gaussian；显式设为 `2` 时，每轮 Full 后执行 detached LoD，分别更新
Gaussian 和 Selector/TF。详见 [训练模式](ultra_light_training.md)。

将 `Training.force_tail_optimization_enabled` 设为 `1` 后，忽略
`Selector.max_mapping_iterations`，在线运行到 SLAM 结束，再按
`Training.tail_optimization_steps` 执行精确的尾部 Mapping 步数。
默认 `-1` 表示 SLAM 结束迭代数的一半（向下取整）；非负值表示固定步数。
详见 [强制覆写尾优化](tail_optimization.md)。开关默认关闭。

- SLAM 在线：LOD 的宽高相对于本次 mapping 分辨率缩小，ground truth 同步
  采用 bilinear 缩放，`align_corners=false`。例如 mapping 为 640×480，
  LOD 为 320×240；启用 Gaussian pyramid 时，以当前层分辨率为基准再缩小。
- SLAM 结束：LOD 恢复原图宽高，ground truth 使用原图。即使该关键帧还有
  未完成的 pyramid 训练次数，LOD 也使用原图全分辨率。
- 无 SLAM 的 COLMAP 训练：不应用在线缩放，LOD 沿用本次 mapping 分辨率。

在线状态使用 `pSLAM_->isShutDown()` 判断。相机视场和投影矩阵保持不变。
完整渲染不使用该缩放参数。历史 QUALITY 在线宽高各减半、SLAM 结束恢复
原图；QUALITY 的缩放不依赖 `Selector.lod_render_scale`。

旧配置缺少 `Selector.lod_render_scale` 时，默认值为 `1.0`。
新配置项会随模型一起写入导出的 `selector_config.yaml`。

## 使用次数文件

```yaml
Mapping.record_keyframe_used_times: 0
```

默认关闭 `used_times/keyframe_used_times*.txt` 文件写入，也不创建对应目录。
设置为 `1` 时恢复逐次 mapping 和结束时的使用次数记录。
此开关只影响使用次数文件，不影响控制台的训练、CD/DSD 日志。

## 缓存范围

- GI：Recent Window 的加权均值按 GI version 缓存；append/prune、窗口推进、
  contribution 更新和结构失效都会改变版本。GI Teacher 不再做整图排序，
  而是在当前 eligible 集合上用 `kthvalue` 求 percentile，保持严格 `>`
  阈值、ties 和独立 mature budget 语义。
- Teacher targets：CD/DSD 中分别为每个 Teacher、ratio 和 Gaussian 快照
  建立局部缓存。每个被访问的 Gaussian target 计算一次，训练 batch、epoch
  和 loss 评估复用；新采样的 Selector anchors 按需补齐。
- TF CD 缓存已经完成 geometry transport 的 target。DSD 缓存当前同一份
  Gaussian 输入上的 reference 输出。缓存不保留计算图，Student 梯度照常计算。
- 缓存随该次蒸馏结束释放，不跨 mapping、结构变化或 Teacher 刷新复用。
  Target 存储以快照 population 为索引，增加临时显存占用。

LOD 已接入 [compact rendering](compact_lod_design.md)：只把 protected 和
实际选中的 mature Gaussian 送入 TF/LOD rasterizer；Selector 默认只计算
mature 行，GI/ratio loss 的 population 和共享 optimizer 保持原逻辑。
该优化在线和 SLAM 结束后均启用。
历史 probe 的视角数、ratio 和 interval 未修改；已接入单次 probe 内缓存复用
与按需 QUALITY。历史 QUALITY 在线宽高各减半，SLAM 结束后恢复原图；不做
分散渲染，详见 [历史蒸馏说明](drift_stability.md)。

## 本轮低风险计算优化

- LOD 只生成实际使用的 hard straight-through Gumbel sample；ratio embedding
  只对一个标量前向，再广播到 Gaussian 行。参数布局、梯度路径不变，但减少
  随机数消耗会改变后续随机序列，不能要求整个训练逐步 bitwise 一致。
- `TransformField.mature_only=false` 的兼容模式仍计算全部 Selector 行，
  保留原先 protected TF 的 gating；默认 mature-only 配置使用精简计算。
- 每个关键帧缓存不同源图/分辨率的 bilinear GT，LOD 和 QUALITY 共用。
  原图按不可变输入处理；cull 释放关键帧及其缓存。
- 历史 visibility indices 直接缓存 GPU Tensor，包含空结果；不再往返 CPU
  vector。保留既有 topology/change 失效协议，会相应增加显存占用。
- 单模块 DSD 更新只失效该 Student 的 prepared inputs 和 QUALITY pair，
  不清掉 Teacher、另一个模块或 EMA candidate 的缓存。
- QUALITY 的 PSNR 和 active mature count 一次回读；TF drift 的三个分量
  也合并回读。正常 mapping 删除整设备 synchronize，仍保留自适应 KF
  所需的 full loss 回读，以及离线 render 时间测量的 synchronize。

### Recent Window 的“冗余重建”

满窗口大小为 `W`，窗口淘汰时正确递推为：

```text
S_new = beta * S_old - beta^W * oldest_importance
```

原实现先减 `beta^(W-1)` 项但未衰减其余项，依赖立即全量重建纠正。
本轮先修正递推，再移除每次 push 后的重建；每 256 次窗口推进做一次校准，
限制浮点累积误差。已有 contribution 的更新仍按其当前窗口位置替换。

### 历史坏视角 union

总拼接行数达到整图 Gaussian 数量时，各视角用全局 bool mask 合并，然后
一次 `nonzero`。稀疏集合保留 sort/dedup，避免不必要的整图分配。
两种路径均返回升序 global indices，不改变 distill 的 population。

## CUDA 只读路径

- `Training`：保留现有 rasterization/backward、importance 和 Python 八项
  返回接口。正常完整渲染与 LOD 仍使用此模式。
- `Inference`：QUALITY 使用原 preprocess、tile 排序、alpha 合成逻辑，但
  不保存 autograd、SampleState、bucket 数据、逐像素反传缓存，不做 bucket
  scan/CPU 回读；image 辅助缓冲只含 tile ranges。
- `Visibility`：仍执行原 preprocess，包括 covariance/SH/color 有效性检查，
  直接返回 radii 和原 `radii > 0` indices。不分配图像/binning/sample，
  不排序、不扫描、不做像素渲染。保持历史 native 分辨率的可见性判定。

不以简化 frustum 判断替换原 visibility，也不改变 QUALITY soft opacity。
CUDA 原有 tile instance count 回读在 Inference 中仍保留，未引入新同步机制。

## 完整 mapping cycle 计时

每 1000 个有效 cycle 打印 `[Mapping PERF]`：

- `wall_ms(avg/p95/max)`：从处理 MappingOperation 到训练、optimizer、周期
  AABB、Historical Probe/DSD 返回的 host wall time。尾优化也计入。
- `stage_host_ms`：operations、full_render、gi、full_backward、lod、densify、
  optimizer、aabb、historical 的平均 host 时间。
- `sampled_cuda_elapsed_ms`：每 20 cycle 采 CUDA events，完成后通过
  `cudaEventQuery` 回收，输出各 stage 平均间隔及样本数，不增加
  `cudaDeviceSynchronize/cudaEventSynchronize`。

无已完成 CUDA 样本时显示 `NA(n=0)`；延迟完成的样本进入下一份报告，
不为计时额外等待。最多保留 8 个待完成样本，GPU 积压时跳过新采样。
已完成样本的 CUDA events 池化复用，避免每次采样重新创建、销毁。

Host 时间不能当纯 GPU kernel 时间；CUDA event 区间也包含该 stream 的 host
launch 间隙。该计时针对当前默认 stream 执行链，不是跨 stream profiler。
原 trainingReport 的 `iter_time` 仍是局部训练前半段时间，完整周期以 PERF 为准。

## 目标机验证

本机无 CUDA/LibTorch 开发环境，本轮未构建、未运行这些测试，也未测实测加速比。
在已有 CUDA 11.8 构建目录启用 `CARTGS_BUILD_TESTS=ON` 后运行：

```sh
ctest --test-dir build-cuda118-sm86-gcc9 --output-on-failure \
  -R 'recent_window_importance|gi_teacher|performance_primitives|mapping_profiler|historical_probe_cache|renderer_readonly_cuda|compact_lod'
```

新增/补充测试覆盖：Recent Window 对全量重建参考的 600 次窗口滚动、beta
为 0/1 和中间值、append/prune/失效、非有限输入；GI 严格阈值和 ties；
Selector scalar broadcast/梯度/hard sample；稠密/稀疏 union；GT cache；
模块级缓存失效；CUDA Training/Inference 图像和 Visibility radii/indices
等价性及只读缓冲区，含空图、离屏、非有限点、SH/covariance 分支和分辨率。

Change CD transport、epochs、学习率、full quality 验收和 Replay 策略未修改。
