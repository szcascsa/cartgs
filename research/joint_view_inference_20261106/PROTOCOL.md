# 视角联合模型的严格等价冻结推理优化

范围：当前本地main的Residual48＋19D Selector，兼容joint schema1/2/3/4与旧11D模型。只优化冻结推理，不训练、不改checkpoint、默认算法或服务器生产目录，不commit/push。

按P0到P1独立验证：静态缓存、19D/8D条件打包、K点直接发布；Anchor视角编码常量缓存与后处理融合；实际K的有界CUDA Graph；去除可选计时同步。四Head及所有GEMM形状、权重、调用顺序保留，视角投影仍K×8。有限值检查和稳定选点不变。

主模型：view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply；单网络视角两个模型及旧11D作兼容检查。相机序列、固定噪声、候选域、ratio与K匹配，不使用近似相机/ratio缓存、padding、TF32或fast-math。

完成标准：FP32输入、logits、14D增量、全部Renderer属性、ID及图像逐bit一致；覆盖训练/未见ratio、空集合、候选不足、同分、动态图形形状、缓存淘汰、参数/Feature/拓扑变动、非默认流和旧输出所有权。任何不等价候选记录首个差异并回退，不用PSNR近似代替。

性能：相同输入交错两轮，分别记录冷/热准备、Rasterizer、总耗时和显存；Nsight Systems对比Kernel、Host发射、同步及DtoH。只有正确且可重复更快的候选保留。共享GPU负载记录，不把组件耗时当Viewer实际FPS。

缓存由模型/参数/Buffer/Master的对象、地址、版本、形状与ratio节点签名保护；变化须rebuild。Graph最多6项/512MiB、形状记录32项，全局输出8项。不并发使用同一可写Workspace，Graph结果发布到独立拥有存储。

显示优化仅在原结果不变前提下处理；不通过JPEG/降低精度缩小传输。LI/GI是训练监督，推理不新增教师Full或反向。
