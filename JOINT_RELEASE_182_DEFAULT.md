# 默认优化渲染器集成记录

日期：2026-11-16。用户指定默认使用 `JointReleaseRenderer(..., mode=182)`，并明确无需进行服务器编译。

## 实际改动

- `python/joint_release_renderer.py` 提供默认 Mode182 的冻结推理 API，使用 checkpoint 原 Selector 和指纹匹配的 Student，拒绝其他渲染模式及缺失 Student。
- `python/deployment_infer.py` 默认使用该 API；CPU 当前位姿直接提交给 C++，支持局部 Ratio、固定点预算和 Full，输出数值 CSV。
- `python/joint_release_native` 保存已验收的完整原生源码及哈希，采用项目内独立构建，移除对研究脚本和旧对象文件的运行依赖。
- 纳入配套的 joint19 冻结输入支持、CUDA 打包/发布接口和 Student 加载模块。已有训练与普通 Viewer 的未提交改动不纳入本提交。
- 使用说明见 `python/DEPLOYMENT_RUNTIME.md`。本次默认切换仅影响训练完成后的独立优化推理入口。

## 验证

| 检查 | 结果 |
| --- | --- |
| 新入口 CPU 契约测试 | 12/12：预算、当前位姿路由、Full、梯度拒绝、Student 身份、native 配置与文件身份 |
| 原预算 CLI 回归 | 6/6 |
| 原预算 CPU 数据流回归 | 15/15；仅对缺失的 Rasterizer 模块做导入占位，调用 Rasterizer 会立即报错 |
| 原生源码身份 | 67/67 文件 SHA256 与冻结快照一致，项目内自定义 include 闭包完整 |
| Python 语法、CLI 帮助及错误参数 | 本地检查通过 |
| 预先存在的已跟踪改动 | 与任务开始时逐文件 SHA256 一致，本轮未改写 |

本地 PyTorch 为 CPU 版本，直接执行原预算回归最初因缺少 `diff_gaussian_rasterization` 无法导入；随后明确隔离 CUDA 渲染导入，仅验证实际 CPU 网络/预算数据流，未将其作为 GPU 图像验证。日志保存在 `research/production_joint_release_182_20261116`。

本次没有服务器编译、新打包二进制 GPU 验收、部署或 push。源码身份复用已有 2026-11-15 GPU 验收证据，不将其写作本次重新编译后的结果。

## 精度边界

保守版保留 Mode182 已有 bit-exp 与 `T<0.01` 近似；未启用 Alpha3。已有研究中，新准备调度相对旧 Mode182 的 ID/RGB/radii 通过 3528 次逐位一致检查。相对原 FP32 Student＋F2 的等权平均 PSNR 损失约 0.00849 dB，最差条件均值约 0.20278 dB，最差帧约 0.30253 dB；不能称为相对原 CaRtGS/F2 无损。

原 GPU 验收报告：`research/low_ratio_footprint_tc_9h_20261112_0011/student_resume_20261114_2250/LOW_RATIO_FOOTPRINT_AND_TENSOR_CORE_9H_REPORT.md`。完整使用和容量限制随源码文档保存，不要求该研究目录存在才能运行。
