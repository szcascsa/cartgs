# 本次续算的资料与实现对应

本次计时窗口为 2026-11-15 02:44:06 至 11:44:06（香港时间）。下列资料用于提出候选；外部论文的速度不作为本项目实测证据。

|一手资料|本项目对应|本次判断|
|---|---|---|
|[Kopanas 的 3DGS CUDA 优化记录](https://grgkopanas.github.io/3dgs_cuda_blog/index.html)，Step 5/6 与负结果表|已有 182 的相邻像素映射、T 符号状态；新增提升后的 exp2、直接 half2 编码|已有收益单列；新候选须独立计时，不能用减少浮点算术代替完整延迟证据|
|[Warp-Level Culling for Efficient Blending in 3D Gaussian Splatting](https://plse-splats.github.io/faster-gaussian-splatting/)|Warp 子区域筛除、批量与属性布局|已读历史子块/私有缓存负结果，不重复采用；本次检验误差受控的邻接 Alpha 共享，开发集变慢|
|[NVIDIA：CUDA Graph 启动成本](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)|节点启动成本、相机与请求直接提交、已有容量断点|保持当前 CUDA 11.8；不为测试更换共享环境。较新 Toolkit 的结果不外推为本项目收益|
|[CUDA Graph 动态控制流](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)|GPU 动态 N/K 与工作量传递|当前实现保留设备计数；不依赖当前 Toolkit 不支持的条件节点|

实现与证据必须分清三类：half2 渲染是普通 CUDA 半精度向量指令；Student/Selector 的 WMMA/MMA 是 Tensor Core 执行；相机、索引和位表操作仍使用 CPU C++ 或普通 CUDA。没有 Teacher 前向、训练或重新蒸馏。

本次新增候选包含：弱尾 Alpha 阈值、透射率早停、相邻像素近似、提升后的 exp2、CUB 稳定排序、按 Tile 清零、相机/请求提交融合、View 的 lane 分组、Selector 动态首层按 Anchor 计算，以及冻结投影缓存的 FP16 存储。候选是否保留以本目录控制器认可的测量与画质结果为准。
