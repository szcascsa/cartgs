# 剩余瓶颈与停止条件

本轮只将 Phase 1～3 接入可选生产路径，三个开关均保持 0。在线质量验收没有通过，不能继续叠加优化后掩盖问题。

1. **P0：可复现的完整在线质量/时钟验收。** 加速使异步SLAM中的Mapping次数、拓扑事件、尾部Full/LoD次数改变。先记录并重放同一真实关键帧/拓扑事件序列，保留完整Adam状态与LR时钟；不能强制修改生产调度或伪造零梯度step。当前200步固定地图重放不代替多视图在线验收。
2. **P0：Renderer/SSIM Backward及Renderer同步。** 优化后总Backward仍91 launches；查看 kernel_execution_detail.csv 按真实kernel总时长排序。Renderer存在num_rendered/bucket_sum的D2H与Forward上下文缓冲保存。SSIM的硬件计数器被ERR_NVGPUCTRPERM阻止，不能据此判定带宽/算力瓶颈。
3. **P1：Anchor Full统计/校验。** 保留FP64 opacity统计、访问定义、slot恢复和完整assert。下一版用一次错误flag读取合并多个scalar同步，再单独验证无效owner不会先触发越界；不降低检查频率。只在独立gate后接入。
4. **P1：Gather→R3 Add→Decode。** 当前保留六组Master gather及R3 gather/add。Decode scope仍3launch，其中2个是raw logit与ST mask gather。下一版可融合gather与decode，但必须验证重复ID scatter、全表Dense Adam与未激活ratio的grad=None。
5. **P1：Stable Compaction。** Sampling/Budget/Compaction仍18launch，dynamic K和Renderer D2H仍同步。只融合比较和mask并不能声称消除K同步；prefix-sum必须保序、精确处理u==p和inactive规则，不改变Softmax或Budget。
6. **P1：Input Pack大N内存布局。** N=100K人工压力测试Forward由0.154ms变为0.184ms，出现回退；不能全规模默认启用。研究coalesced列读取/向量化，在相同真实身份与梯度fixture上重新测。N=10K/50K有效不代表100K有效。
7. **P2：单独R3 Lookup、冻结C的小GEMM。** 当前各2/4launch，收益量较小；优先合并邻接数据流，不单独重写一个高复杂度lookup/MLP。仍使用cuBLAS Encoder，不预设自研27→32 GEMM更快。
8. **暂不并行共享写。** Rasterizer/CUB/SSIM尚未证明current stream正确，保留mutex_render、默认训练顺序。不得多ratio同时Backward或Full/LoD同时更新Master。只读快照推理、pinned预取和keyframe CPU准备可在下一阶段单独测。

Phase 4～6尚未实现，不把其理论收益加入本轮加速数字。无模块被证明完全没有优化空间；R3独立lookup和小MLP当前优先级较低。
