# 独立 PLY 的 LoD 评估

每个 ratio 执行一次 Selector 和 Transform，将最终选中高斯导出为标准 binary float32 PLY，再重新加载该文件进行渲染。原始全量模型保持完整，FULL 直接加载原始 PLY。

每档输出目录包含：

- `point_cloud.ply`：已完成筛选和补偿的高斯，直接用于渲染。
- `ply_manifest.json`：ratio、来源 PLY、Selector/Transform checkpoint、有效配置和点数。
- `render_time.txt`：本次各帧的轨迹索引与渲染毫秒数。
- `eval.txt`：图像质量指标及实测渲染 FPS。

`summary.json`、`summary.csv` 同时记录每档的 `ply_path`、`render_gaussians` 和 `rasterizer_mode`。每次运行重新导出并覆盖对应档位的产物，避免复用旧模型的结果。

## 构建 Python rasterizer

本次新增专用推理入口，需要在目标 CUDA/Torch 环境重新构建 Python 扩展。在项目根目录执行：

```bash
cd python/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps -e .
cd ../..
```

旧扩展缺少推理入口时会明确报错，不会自动退回训练版前向。原有训练接口和 backward 路径保持不变。

## 导出并评估所有档位

继续使用原来的命令：

```bash
python python/eval_lod.py RESULT_PATH DATASET_PATH
```

例如，0.01 档位保存在 `RESULT_PATH/lod_eval/ratio_0p01/point_cloud.ply`。Selector 和 Transform 仅参与导出，文件重新加载后不再进行筛选或补偿。

## 单独评估导出的 PLY

```bash
python python/run.py RESULT_PATH DATASET_PATH \
  --ply RESULT_PATH/lod_eval/ratio_0p01/point_cloud.ply \
  --shutdown-model 4900_shutdown \
  --output_path RESULT_PATH/ply_eval/ratio_0p01
```

将 `4900_shutdown` 替换为该 PLY 来源的实际 checkpoint。`RESULT_PATH` 提供相机内参和轨迹；PLY 不包含这些信息。此路径不加载 Selector、Transform 或 selector YAML。`--ply` 与 `--selector` 互斥，避免重复补偿。

## 属性与计时口径

PLY 保存补偿后的 XYZ、log-scale 和 quaternion；opacity 保留原始 logit，SH 保留所有系数及原排列。重新加载涉及 scale 的 log/exp 与 quaternion 归一化，数值一致性使用浮点容差验证。

FULL 与 LoD 均调用原有 CUDA `RasterizeMode::Inference` 分支，排除反向传播缓存。渲染前预热 10 次；每帧在相机和高斯输入准备好后同步 CUDA，使用 `perf_counter()` 测量完整 rasterizer 调用，结束后同步 CUDA。该数值包含调用和同步开销，排除模型加载、导出、Selector、Transform、属性压紧、指标计算、CPU 图像传输和图片保存。

`rendering FPS = 1000 / mean(render_time_ms)`。高斯数量减少不保证 FPS 按相同比例增加。

## 验证

```bash
python -B tests/ply_render_python_test.py -v
python -B tests/python_rasterizer_dispatch_test.py -v
python -B tests/render_timing_python_test.py -v
python -B tests/ultrafast_eval_test.py -v
```

前两个脚本包含真实 PLY 往返和 Python 分发检查；分发测试使用模拟 native 入口，不验证 CUDA kernel。两个渲染测试脚本中的 CUDA 用例需要 GPU 和新扩展，检查训练/推理输出及 PLY 重新加载后的图像一致性。CPU 测试通过不能代替 CUDA 构建、图像与性能实测。
