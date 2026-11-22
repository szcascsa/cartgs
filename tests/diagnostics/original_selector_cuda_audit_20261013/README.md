# 原始 Selector CUDA 审计复现

目标：只使用诊断副本和现有 checkpoint，重建 Native C++ 基准、梯度与 profiler 数据。当前结果已完成，不需要重新训练序列。服务器为 `prmi@172.19.163.58`，生产仓库 `/home/prmi/zwj/mycartgs`；认证通过 SSH 交互提供，脚本不保存密码。

## 已执行的环境与输入

- CUDA 11.8、GCC9.4、RTX3090、driver535.230.02、Conda `zwj`（Python3.11、Torch2.3.1+cu118）。
- Native LibTorch 位于 `/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch`，CXX11 ABI=false。
- 现有生产 build：`build-cuda118-sm86-gcc9`。诊断独立 build：本目录的 `build`。
- checkpoint：`results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply`，含point_cloud、Selector、Protection metadata及TF参数。
- 图像/相机具体来源见 `render_fixture.json`。`prepare_render.py`写camera/GT binary，无渲染图片。
- TF 夹具由该 checkpoint 的 transform_state.json 生成 tf_fixture_spec.txt；不依赖此前 TF 审计目录。联合梯度用 legacy LibTorch TF backend。
- 远端无有效仓库HEAD；源码版本以SHA和冻结snapshot为准。`fast.yaml`在诊断后Guard开关变化，不用于本轮基准。

## 服务器命令

下列复测会覆盖**本诊断目录自己的结果**。如需保留本次原始测量，先复制本目录的CSV/JSON/profiler到另一个诊断目录。无需重新生成生产checkpoint，无长序列命令。

```bash
ssh prmi@172.19.163.58
cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_selector_cuda_audit_20261013
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.6
PY=/home/prmi/anaconda3/envs/zwj/bin/python

# 提取真实checkpoint并记录环境/初始生产源码哈希
"$PY" prepare.py > prepare.log 2>&1
"$PY" prepare_render.py > prepare_render.log 2>&1

# 只编译诊断；不重建生产库
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch \
  -DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4 \
  -DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/install/eigen-3.3.7/share/eigen3/cmake
cmake --build build -j2

# 8 sizes x 4 ratios x 3 modes，100warmup+500measurement
./build/native_benchmark . > native.log 2>&1

# 11条真实LibTorch/Renderer/TF梯度路径，独立Adam事务基准
./build/gradient_paths . > gradient_paths.log 2>&1
./build/optimizer_benchmark . > optimizer.log 2>&1
./build/isolated_costs . > isolated_costs.log 2>&1

# 固定噪声Native/Python结构梯度对照与autograd保存量
./build/native_benchmark . gradients > native_gradients.log 2>&1
"$PY" graph_oracle.py > graph_oracle.log 2>&1

# Native Kineto（10 captured iterations，100warmup）
./build/native_benchmark . operators > operators.log 2>&1

# Native Nsight；cudaProfilerApi之后才capture，10iterations
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o native_nsys ./build/native_benchmark . profile > nsight.log 2>&1
nsys export --type sqlite --force-overwrite=true \
  --output native_nsys.sqlite native_nsys.nsys-rep >> nsight.log 2>&1

# 历史日志统计、profile解析、生产源码核对与报告
"$PY" collect_workloads.py > workload.log 2>&1
"$PY" parse_profiles.py > profiles_summary.log 2>&1
"$PY" verify_sources.py
"$PY" build_report.py
"$PY" validate_artifacts.py
```

简化执行入口：`run.sh`为初始基准，`run_extra.sh`为Renderer/optimizer/oracle/profiler，`run_final.sh`为最终修正后梯度、280case独立阶段及profiler重跑。本次三份exit文件均0。各脚本不启动任何生产训练。测量运行时不同时运行重GPU任务；本轮没有锁频，结果包含机器调度噪声。

## 本地只解析与生成报告

在下载CSV/JSON/sqlite/trace后，不需要本地CUDA：

```powershell
Set-Location -LiteralPath 'H:\AIProjects\cartgs优化更细致版\mycartgs'
python tests/diagnostics/original_selector_cuda_audit_20261013/parse_profiles.py
python tests/diagnostics/original_selector_cuda_audit_20261013/build_report.py
python tests/diagnostics/original_selector_cuda_audit_20261013/validate_artifacts.py
```

`remote_bridge.py`只负责通过stdin传入的SSH认证执行命令/SFTP，不内嵌凭据。只有客户端已有可信known_hosts时使用；本轮未修改服务器SSH配置。

## 数据口径

- caller_train是生产forwardHard与真实Protection/masks/6GSgathers，但photometric端仅用opacity VJP和ratio loss；排除Renderer/TF/GI。真实Renderer/TF只在gradient_paths诊断中使用。
- cpp_inference是C++网络复现Python seed0Bernoulli规则，并不是现有生产C++LoD evaluator。
- 每次私有参数更新在计时外恢复checkpoint，Adam moments继续推进；不代表连续训练质量。
- native_benchmark.csv的早期staged_train省略tau1division一个算子；最终Native Nsight/Kineto已显式division。最终性能只用caller_train/cpp_inference。
- 独立阶段表不相加；training_backward只有参数+XYZ梯度，q/scale在这个fixture不requires_grad。selection stage使用重建logits；GI是真实单帧贡献而非全bank。
- event是stream跨度，Nsight active是kernel执行时间，wall是含提交/等待的真实CPU时间。三者不可互换。
- allocator累计requests/bytes不等于cudaMalloc或同时显存；outside18D2D checkpointrestore不在timedstage。
- 完整边界与下一阶段parity设计以仓库根报告为准。
