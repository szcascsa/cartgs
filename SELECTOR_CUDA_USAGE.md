# Selector backend 切换与手动编译

`cuda-selector` 分支保留 `legacy | cuda` 两个 backend。用户确认验收通过后，Mapping、Python 评估和随附配置默认改为 `cuda`。CUDA 已通过同 checkpoint、同输入、同随机数/upstream 的直接算法验收：`CUDA_SELECTOR_CORRECTNESS = PASS`。完整 SLAM 的独立运行波动不参与该 correctness gate，最终质量由用户自行测试。

## 编译

在 `/home/prmi/zwj/mycartgs`，使用 `zwj` conda 环境、CUDA 11.8、GCC 9 和 sm86。

已有 `build-deploy-20261015` 的依赖配置可直接复用：

```bash
conda activate zwj
cd /home/prmi/zwj/mycartgs
export PATH=/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH="$PWD/lib:$PWD/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake \
  -S . -B build-deploy-20261015 -DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON
/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake \
  --build build-deploy-20261015 -j2 --target tum_rgbd
```

也可以走完整依赖构建脚本：

```bash
bash build.sh
```

新构建的 CMake 开关和 `build.sh` 均默认启用 CUDA Selector。已有 build 目录会保留 CMake cache，仍建议按上面的命令显式传 `-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON`。编译一次后，同一个 binary 可以通过 YAML 在两种 backend 间切换。显式关闭编译开关时，必须同时配置 `Selector.backend: "legacy"`；配置为 cuda 时会明确报错。

## 运行配置

编辑实际传给 `--selector-config` 的文件，例如 `cfg/selector/fast.yaml`：

```yaml
Selector.enabled: 1
Selector.backend: "cuda"  # 默认；改为 "legacy" 即使用原实现
```

运行参数仍使用：

```bash
... --selector-config /home/prmi/zwj/mycartgs/cfg/selector/fast.yaml
```

缺省 `Selector.backend` 时，Mapping 和 Python 评估也使用 cuda；已有配置显式写 legacy 时仍尊重该值。启动日志应显示 `[Gaussian Mapper] selector_backend=cuda` 或 `selector_backend=legacy`，保存配置会保留实际值。裸 C++/Python 网络模块仍以 legacy 初始化，便于 CPU 测试和独立 parity oracle；产品入口在使用前应用配置。Selector 和 TransformField 的 backend 分别配置，本次只切换 Selector 默认值。

## Python 评估

Python 评估不再需要实验验收环境变量，CUDA backend 会按配置加载。首次编译 extension 时配置：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export CXX=/usr/bin/g++-9
export TORCH_CUDA_ARCH_LIST=8.6
```

Python 首次使用 CUDA Selector 会通过当前 Python Torch 编译自己的 extension；它与 Native LibTorch 的 ABI 独立。

## 本次更新状态

源码包含 ratio GEMM 修复、final backward 分支路由与 ReLU 融合、Selector CUDA Adam 和 Guard preview。最新 backward 优化在 M=219969 的实测：原 CUDA backward 2.27463 → 2.03620 ms，paired F+B 4.28667 → 4.03152 ms，structural caller 5.03120 → 4.80687 ms；这些是新旧 CUDA 的对比，不是完整 Mapping 的加速比。

7 个真实 checkpoint 的直接 forward、selection、输入/18 参数 gradient、各自 1000 步训练及 Adam m/v/step 全部 exact，Guard/Replay/CD/DSD 的直接数学对照通过。完整证据见 `CUDA_SELECTOR_DIRECT_PARITY_REPORT.md` 和 `CUDA_SELECTOR_BACKWARD_OPTIMIZATION_REPORT.md`。两份报告记录各自验收时的状态；默认切换发生在验收之后。本次默认切换不修改 CUDA 计算源码，不替换已有 binary 或模型；源码更新后需重新编译。
