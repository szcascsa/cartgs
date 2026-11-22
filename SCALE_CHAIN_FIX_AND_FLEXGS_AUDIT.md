# Scale 计算链修复与官方 FlexGS 源码核查

## 结论与当前交付状态

本地生产源码已修正 Scale 的两条不稳定计算链，CUDA、LibTorch fallback、Python 评价路径均已修改。RTX3090 上的隔离生产库编译通过，250 项原生数值对照、真实异常 Gaussian 的 Renderer/Backward 重放及两个既有 Mapping 测试通过。没有使用 Top-K，没有修改 Selector、Budget、Loss、Adam 或训练调度。

**服务器默认生产库尚未替换。** 严格旧/新逐像素回归没有全部通过，失败阈值和证据均保留，不用扩大容差把它标成通过。健康 checkpoint 的 PSNR 变化很小，但已有病态补偿的 checkpoint 出现明显渲染敏感性。因此本轮能确认“指定 Scale 计算错误已修复”，不能宣称“全部异常已解决、全量质量已验收”。服务器候选库和测试均在独立目录中。

官方 FlexGS 的实际训练路径使用 **activated Scale + ds**，而本项目使用 **log-scale residual**。官方没有本项目 `exp(ds)` 的中间溢出节点。这是这条特定问题的明确实现差异；它不是官方永不出现非有限值或 K=0 的证明。

## 1. 版本、工作区与实验边界

- 本地 HEAD：`8e629523e442e6471025edbbb4a53d1a8e9d81f9`，开始时已有未提交的 Gumbel/FlexGS 相关修改，本轮保留。
- 本轮证据：`tests/diagnostics/scale_chain_fix_20261028/`。
- 服务器候选：`/home/prmi/zwj/mycartgs/resultsall/scale_chain_fix_20261028/`。
- 编译复用生产 `build-performance-20261023-121515` 的 g++9、CUDA11.8、LibTorch 编译/链接参数；未另外构造不兼容环境。
- 服务器无 Git，通过文件 SHA 与库 SHA 核对版本。`server_scope_verification.json` 验证十个待改生产源文件和默认 `libgaussian_mapper.so` 仍为修复前版本。
- 修改十个本地生产文件，增加一个原生测试和两行 CMake 测试注册。补丁 `scale_chain_only.patch` 相对本轮开始时的文件生成，排除此前的其他 dirty 修改。
- 未启动场景重训或全量实验，未 stage、commit、push。

当前源码的在线补偿可走 SpatialCompensation，并不只存在旧四表 R3 路径。本轮沿实际 `runSelectorStep` 修改 Scale 输入/解码；R3 表的参数、插值、身份迁移和 Adam 时钟没有改动。

## 2. 已证实的错误及修复

### 2.1 先 exp 残差再相乘会产生不必要的溢出

修复前：

```text
s = exp(master_log_scale) * exp(delta_scale)
```

修复后：

```text
ell_out = master_log_scale + delta_scale
s = exp(ell_out)
dL/d(master_log_scale) = dL/d(delta_scale) = (dL/ds) * s
```

不再从激活后的 Scale 反算 log-scale。保留原 detach 开关，直接读取 Master 的 `scaling_`，选中集合保持不变。

真实反例来自此前保存的 Replica mono office2 Gaussian：base Scale 一维约 `0.0708986595`，残差约 `89.5906982`。`exp(89.59)` 超过 FP32 范围，但相加后的 log-scale 约 `86.9442`，最终 Scale 约 `5.75e37`，仍可表示。旧反向即使收到零 Scale VJP，也会在乘法链中形成 `0 × Inf → NaN`。

修复没有用 clamp、NaN 清零、梯度拒收或强制选择数量掩盖问题。最终 Scale 本身不可表示、或真实梯度乘积超出 FP32 范围时，这个等价重写不提供无条件有限性保证。

### 2.2 Spatial 特征避免 log(exp(ell)) 和先平方再归一化

旧 Scale 特征为 `log(exp(ell))`。Master exp 下溢到零后，反向有 `0/0` 风险。新路径直接读取 `ell`。

旧 Shape 为 `R diag(s²) Rᵀ / max(trace, 1e-12)`，大 Scale 先平方可溢出。令：

```text
epsilon = 1e-12                 # 原 trace floor，不变
a = stopgrad(max(max_j ell_j, 0.5*log(epsilon)))
Cbar = R diag(exp(2*(ell-a))) R^T
Shape = Cbar / max(trace(Cbar), exp(log(epsilon)-2*a))
```

分子和分母同尺度缩放，保留原物理定义及 trace floor，没有给长尾输入增加新截断。ATen/Python 使用 `where(trace >= floor, trace, floor)`，CUDA 使用同样条件，保留原 `clamp_min` 在等值边界的导数约定。

CUDA VJP 直接计算对 log-scale 的导数：Scale 特征部分不再除以 activated Scale，Shape 部分使用 `2*exp(2*(ell-a))`。Quaternion 和其他属性的计算没有改动。

## 3. 真实调用位置

| 文件 / 位置 | 本轮修改 |
|---|---|
| `src/gaussian_mapper.cpp:3396` | 获取带原 detach 语义的 Master raw log-scale |
| `src/gaussian_mapper.cpp:3556` | Spatial 初始化与选中行 forward 使用 raw log-scale |
| `src/gaussian_mapper.cpp:3567` | CUDA Decode 调用 `decode14LogScale` |
| `src/gaussian_mapper.cpp:3573` | fallback 在 log 域应用 residual，再 exp |
| `src/anchor_selector_cuda.cpp:30`、`src/anchor_selector_cuda.cu:29` | 新 log-scale Autograd 入口、Forward 与 VJP；原 activated API 保留作兼容/诊断 |
| `include/anchor_selector_cuda.h` | 声明新 API |
| `improvements/selector/spatial_compensation.h:34` | raw-log Scale 特征及稳定 Shape 的 ATen 实现 |
| `src/spatial_compensation_live.cu:14` | 相同定义的 CUDA Forward/VJP |
| `src/spatial_compensation_cuda.cpp`、`include/spatial_compensation_cuda.h` | 新 log-scale Live Autograd 接口 |
| `python/spatial_compensation.py:20` | 相同特征定义和 Shape 稳定计算 |
| `python/selector.py:489`、`:506` | 评价使用 raw log-scale Spatial 输入及合并指数解码 |

模块参数名称、Tensor 形状、checkpoint 字段和固定归一化 buffers 没有改变。旧 API 默认仍接受 activated Scale，新生产调用显式选择 log-scale API，避免静默重新解释其他调用的输入。

## 4. 原生验收

测试源：`tests/anchor_selector/scale_chain_test.cpp`。候选是真实生产 Mapper/ATen/CUDA 对象的隔离重编译，保留旧 CUDA Decode 入口作为对照。

预注册数值容差没有统一放宽：常规 `atol=2e-5, rtol=2e-4`；真实巨大 Scale 对 FP64 参考为 `atol=0, rtol=1e-5`；FP64 Shape 物理式为 `atol=1e-12, rtol=1e-11`。误差 RMS 在 FP64 中计算，避免误差平方再次溢出。

| 检查 | 结果 |
|---|---|
| K=0/1/17/4096 Decode Forward/VJP，未消费输出 | 通过 |
| 非连续输入、重复 selected ID 的梯度累加 | 通过 |
| 真实反例，旧链 Inf/NaN、新链有限 | 通过 |
| `ell=-200, delta=200` 等抵消边界 | 通过 |
| Spatial CUDA 对 ATen Input55/VJP，含极端 log-scale | 通过 |
| 稳定 Shape 对独立 FP64 直接物理式，含 trace floor | 通过 |
| 非零已学习 Field，五个 ratio 的输出、Master/参数梯度 | 通过 |
| 原生比较总数 | **250 项，容差外元素 0** |

常规 Decode 输出/梯度的最大绝对差约 `1.91e-6`；Spatial 常规对照约 `1.19e-7`。真实巨大 Scale 对 FP64 参考的最大相对误差为 `3.53e-6`。其绝对误差约 `2.03e32`，不能在该尺度下只用绝对误差判断对错；小非零上游 VJP 的误差约 `2.03e-6`，也在其独立阈值内。

### 原生 Renderer 的同样本因果重放

使用真实 `renderer_origin_input.pt`、相机和原 Bank 图像，沿未修改的生产 Renderer、原 photo loss 做 backward。新路径直接调用本轮生产 API，不只用脚本替换一个表达式。

| Decode 路径 | 非有限 Scale 分量 | 非有限 delta 梯度分量 | 非有限 Master log-scale 梯度分量 | loss |
|---|---:|---:|---:|---:|
| 旧 LibTorch | 1 | 1 | 1 | 0.55440563 |
| 旧 CUDA | 1 | 1 | 1 | 0.55440563 |
| 合并 log 的 ATen 参考 | 0 | 0 | 0 | 0.55440563 |
| 新 CUDA `decode14LogScale` | 0 | 0 | 0 | 0.55440563 |
| 将 Scale 残差置零的诊断控制 | 0 | 0 | 0 | 0.55440563 |

所有这些单点重放的 radii 都为 0，Renderer Scale VJP 也为 0。这验证了旧 NaN 在 Decode 链中产生，不能用相同 loss 宣称旧梯度正确。**新 Scale 仍然巨大、Scale² 仍超 FP32，本轮没有修复 Renderer 对超大协方差的处理。**

既有 `spatial_mapping_test` 与 `flexgs_mapping_test` 均链接候选库重新编译并通过，覆盖 photo 训练补偿、LoD Master/Selector 更新、Full 辅助参数/Adam 隔离及原 online/tail 轮转检查。它们是小规模原生 fixture，不是完整 SLAM 或不同场景的质量验证。

## 5. 真实 checkpoint 回归及未通过项

固定旧 checkpoint、Gaussian、相机、ratio、采样 seed，旧/新解码复用同一 mask 和 Selected IDs。两图共十个 ratio cell（10/20/40/60/80%），记录真实 N、K 和 ID SHA。

- 健康 `vanilla_aligned/replica/mono/room2/seed0`：N=20,189。
- 已异常 `natural/replica/mono/office2/seed0`：N=17,383。
- 60 项属性数值对照通过；旧 Scale 可表示的分量单独比较。十个 cell 的新 selected Scale 均为有限值；office2 80% 旧路径有 1 个非有限分量，新路径为 0。
- 旧/新分别在相同三相机渲染：健康 room2 共 15 对；office2 旧 Scale 有限的四档共 12 对。80% 旧 Scale 已 Inf，不做伪装的有效图像 parity。

**逐像素阈值 `atol=2e-4, rtol=2e-4` 未通过，保持失败。** 健康 room2 的 15 对均有局部超阈值像素，最大绝对差 `0.00347281`，最大 RGB RMS 差 `1.25986e-5`；重建 PSNR 差在 `[-0.00002070,+0.00001919] dB`。相同旧输入重复渲染最大差为 0，故这不能归因于本次重复渲染噪声。

独立 `decode_only` 控制固定旧 Spatial residual、仅改 Scale 解码，也出现健康场景的局部像素差。它说明至少部分差异来自指数运算重排的 FP32 舍入，经 Rasterizer 放大；不能把严格像素误差失败删除，也不能把几乎不变的 PSNR 代替梯度验证。

office2 的 12 对中 5 对不通过逐像素阈值，PSNR 差范围 `[-0.30951,+0.43015] dB`，最大 RGB RMS 差约 `0.11256`。例：60% 中间相机，旧/新 Field residual 最大差仅 `7.63e-6`，图像变化却很大；固定旧 residual 的 decode-only 控制在该视图变化很小。这是病态模型/Renderer 对属性微扰敏感的证据，尚未隔离到某个具体协方差运算，不能据此将质量变化归因于错误 VJP，或宣称新路径质量更好。

没有按这些结果选择学习率、checkpoint、补偿结构或训练预算。完整在线连续训练、独立场景 PSNR/SSIM/LPIPS 和 Renderer 病态几何验收尚未完成。**因此不替换服务器默认库，不启动新版全量实验。** 本地修复源码、隔离已编译候选和全部失败证据均保留。

## 6. 官方 FlexGS 为什么没有同样的 exp(ds) 问题

官方来源：[LiuHengyu321/FlexGS](https://github.com/LiuHengyu321/FlexGS)，固定 commit `0e9b91f4ddb294c4260b6b83b1dbbb90c1be1982`。下载的真实源码及 SHA 在 `reference/`、`reference_manifest.json`。以下依据实际训练调用链，不根据同名备用 renderer 推断。

```text
train.py:117 gumbel_select_render
  → gaussian_renderer/__init__.py:551 exp(pc._scaling)
  → :601 pc._deformation(selected activated scales, ratio)
  → scene/deformation.py:200 forward_dynamic / poc_fre
  → :117 scales_emb[:, :3] * mask + ds
  → renderer:619 opacity × ST mask
  → 将全 N 行几何及门控 opacity 交给 Rasterizer
```

`poc_fre` 保留输入的前三列，因此 deformation 中 `scales_emb[:,:3]` 仍是 activated Scale。输出后没有第二个 `exp(ds)`。

| 项目 | 当前 mycartgs | 官方实际训练路径 |
|---|---|---|
| Scale 补偿语义 | log-scale residual，数学上 `exp(ell+ds)` | activated Scale 加法 `s+ds` |
| ds≈89.59 的特定中间量 | 旧 `exp(ds)` 溢出，新链消除该节点 | 不计算 `exp(ds)`，该加法不会触发同一溢出 |
| 未选 Gaussian 的渲染组织 | 生产 compact K 行路径 | 补偿后回填全 N 行，再以 ST opacity 门控 |
| Selector 监督 | 本轮保留当前 Photo/Budget，未引入官方 GI 标签 | fine 且 `lambda_dssim!=0` 时，有 importance 分位数伪标签的逐点 L1，另有比例 L1 |
| K=0 是否被数学排除 | 没有，本轮未修改 | 同样没有 |

官方 `train.py:180–191` 的 importance 伪标签对漏选正例提供不同于单一人口预算的监督；默认 `gumbel_weight=1`、比例项权重 `0.01`。是否因此更稳定，仍需单因素实验，不能只凭静态差异作因果质量结论。该监督不是本轮新增内容。

官方 `train.py:196–198` 还会在 NaN loss 后 `os.execv` 重启。重启不等于从源头消除坏梯度；有限 loss 下的非有限梯度也不由这个判断排除。

`official_operator_tests.json` 使用审阅过的官方 `Deformation.forward_dynamic` 原始 AST 方法及确定性 ds stub：相同大 ds 下，官方加法输出与零 VJP 均有限。另以二分类 logits `[1000,-1000]` 执行其同样的 `F.gumbel_softmax(..., hard=True)`，100 个点可全部不选。这是算子边界反例，不是官方完整训练或官方 checkpoint 重现。

**不建议将当前 log residual 偷换成官方 activated add。** 两者改变了补偿意义、checkpoint 函数和优化尺度；本轮只实现当前定义的稳定等价计算，也不加入 Top-K、强制 K 或最少点数兜底。

## 7. 修复范围与后续源头研究

已确认：原 Decode 的中间 exp 溢出，以及 Spatial 的 exp/log 与先平方归一化数值路径。修复从表达式本身消除这两个不必要节点；没有在坏梯度产生后清零。

仍未解决：

1. 补偿网络为什么产生巨大 log-scale residual。此前零直接 Scale VJP 的点仍会因共享参数和 Adam 历史漂移，这一机制不会被数值重写消除。
2. 最终 Scale 本身可表示但协方差不可表示，及病态 Renderer 对属性舍入的敏感性。应对同一坏 Gaussian 逐段记录 covariance、投影、det/conic、radii 和梯度，定位第一个不可表示或病态运算；本轮没有修改 Renderer。
3. Selector 概率塌缩。官方仍允许 K=0，不能用“换成 FlexGS 选择”或 Top-K 将该优化问题视为已修复。
4. 历史 RGB-D office2 首个非有限梯度尚未完整隔离。本次单点链路证据不能冒充所有历史污染的统一根因。
5. Quaternion 零范数及极端 Adam 二阶 moment 边界是独立问题，本轮没有改动。

下一项最有针对性的验证是第 2 项的固定坏样本 Renderer 分段诊断，并与第 1 项的共享补偿漂移区分。未经这一步和受控短程质量验收，不把当前修复当成新版全量实验的 Go 信号。

## 8. 证据与复现

| 文件 | 内容 |
|---|---|
| `scale_chain_only.patch`、`source/`、`before/` | 本轮专属修改及修改前文件 |
| `candidate_build.json`、`candidate_build_commands.json` | 真实编译、链接参数与库 SHA |
| `native_tests.json/log` | 250 项数值对照 |
| `renderer_tests.json/log` | 同一真实样本的五路 Renderer/Backward 对照 |
| `mapping_regression.json`、两个 Mapping 日志 | 原生小规模训练回归 |
| `python_driver.log` | 保留的严格像素验收失败 |
| `roundoff_diagnostics.json`、`checkpoint_roundoff.csv` | 真实 N/K/ID SHA、属性误差、逐相机 RGB/PSNR 差及 decode-only 控制 |
| `official_operator_tests.json`、`reference/` | 官方路径核查及算子反例，非完整官方训练 |
| `server_scope_verification.json` | 服务器默认生产源码/库未替换的 SHA 证据 |
| `validation_summary.json`、`evidence_sha256.json` | 可读机器摘要、证据包 SHA |

服务器候选目录中，以 `/home/prmi/anaconda3/envs/zwj/bin/python` 依次运行：

```bash
python build_candidate.py
python run_native_tests.py
python run_renderer_test.py
python run_mapping_regression.py
python validate_python.py          # 严格旧/新像素门槛仍会失败，不能标记通过
python inspect_render_roundoff.py  # 分离 Decode 与 Field 舍入，保留全部差异
python verify_server_scope.py
```

候选编译脚本在生产文件 SHA 漂移时拒绝覆盖；它只输出隔离库。开始时因 overlay 相对头文件搜索和两个测试代码拼写/宏使用问题的编译失败日志同样保留，这些是已修正的测试/构建问题，不是运行时训练失败。
