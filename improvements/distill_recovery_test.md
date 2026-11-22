# 真实渲染恢复能力测试

## 目的与边界

`distill_recovery_test` 使用已训练 checkpoint、真实 room RGB 图像和项目 CUDA rasterizer，
而不是把 tensor loss 或人造质量函数当成 PSNR。测试不启动 SLAM，不更新 master Gaussian、
Teacher、GI、AABB，不修改正常 mapping 的优化器，也不更改生产 DSD 的门控策略。

生产配置中的 `DriftStability.enable` 默认关闭。本独立诊断程序会显式启用恢复，
仍从配置读取学习率、epoch 和门控参数；不要求为运行测试重新打开在线恢复。

每个案例分两层执行：

1. **Strict**：原配置的完整 `maybeProbeAndDistill()`，包括连续确认、function/QUALITY/budget gate、
   post-quality guard 和 rollback。记录实际执行步数及最终接受/拒绝结果。
2. **Optimizer-only**：恢复到相同的退化参数，直接调用生产 `distillModule()`。
   仅绕过触发和接受门控，损失、Teacher target cache、临时 Adam、ratio 交错顺序、学习率、
   batch 和 epoch 均复用生产代码。结果仅供诊断，不代表生产路径已接受更新。

直接优化每个 ratio 的目标集合是 **PSNR 退化历史视角的 visible mature union**，仍受
`max_gaussians` 限制。两张 holdout 相机不参与 union，但其 Gaussian 可能也被训练视角看到；
这是视角级验证，不是 Gaussian 级不相交验证。

## 可复現案例

- `SELECTOR_BIAS`：减少 keep logit，检查选择规模漂移后的恢复。
- `SELECTOR_RANK`：扰动空间/属性对应的 head 权重，保持 ratio embedding 的 head 权重不变；
  校准一次 head bias，使所有 ratio 的平均 expected mature 数量接近原模型。
  校准仅用于制造测试扰动，**不改变生产 Selector，也不声称每个 ratio 数量完全一致**。
- `TF`：扰动 XYZ residual head bias，固定 Selector、hard/cold admission 和 mature Bernoulli uniform。
  Teacher/Student 的选择集合完全相同。

仅当四个训练视角和两个 holdout 视角上，平均 PSNR 均下降超过 0.1 dB，才运行恢复测试。
否则报告 `INCONCLUSIVE_NO_RENDER_DEGRADATION`；不将没有退化的案例算作蒸馏成功或失败。
JSON 保存扰动强度搜索轨迹、逐视角/逐 ratio PSNR、函数误差、预算、实际数量及耗时。
`strict_quality_trace` 另保存生产路径的临时更新质量和预算，revision 1 是回滚之前的参数，
避免只看到回滚后的 `strict_post` 而误判拒绝原因。

测试先断言未扰动 Teacher/Student 的函数和渲染一致，结束后断言 master Gaussian 未改变。
日志使用固定四位小数；JSON 保留原始测量精度，避免微小 TF 误差被四位小数抹去。

## 构建与执行

在已配置 CUDA/LibTorch 的工程环境中：

```bash
cmake -S . -B build-cuda118-sm86-gcc9 -DCARTGS_BUILD_TESTS=ON
cmake --build build-cuda118-sm86-gcc9 --target distill_recovery_test -j2
./bin/distill_recovery_test CHECKPOINT DATASET cfg/selector/fast.yaml OUTPUT_JSON 5 2026 1
```

参数依次为 checkpoint 的 `ply` 目录、RGB dataset 根目录、Selector 配置、报告路径、
epoch 数、测试随机种子、渲染缩放比。最后三个参数可省略，默认原配置 epoch、2026、全分辨率。
缩放比 `0.5` 可复现在线 QUALITY 的尺寸；`1` 检查全分辨率真实质量。

需要 checkpoint 含 `cameras.json`、Selector/TF archive、`transform_state.json`、
对应 iteration 点云和 `selector_metadata.ply`。当前 loader 对应本项目的 TUM RGB-D 输出：
SH degree 3、黑色背景、原始 RGB 图像、相机保存的 Twc。不是任意 dataset 的通用 loader。

## 如何解释结果

- Strict `SKIP / steps=0`：触发链未放行，不能从中判断优化器的恢复能力。
- `RECOVERED`：函数误差下降、训练和 holdout PSNR 改善，且所有测试视角/ratio 的
  剩余质量缺口和新增退化均不超过配置的 PSNR 容差。
- `PARTIAL_RECOVERY`：函数和真实质量一起改善，但未完全恢复。
- `FUNCTION_ONLY_NO_QUALITY_RECOVERY`：函数误差下降不能转换为真实图像恢复。
- `QUALITY_ONLY_FUNCTION_REGRESSION`：图像改善，但总函数误差反而增加，需检查 TF 各分量。
- `OPTIMIZER_FAILED`：未同时形成上述有效恢复证据。

程序退出 0 只表示实验执行及一致性断言成功，**不表示算法恢复成功**。应读取每个案例的
outcome。此测试验证可控扰动下的局部恢复能力，不能替代自然在线遗忘、长序列和多场景评测。
