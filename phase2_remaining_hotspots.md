# 第二阶段之后的剩余热点

以下是实测归属开销，不是承诺可全部回收的加速空间。RTX3090 由其他任务共享；当前全部新开关默认关闭。

1. **P0：SSIM / Renderer Backward。** 5% LoD Nsight 中，Backward 为 1.629 ms，占所计 kernel 时间 52.9%；Renderer Forward 为 0.605 ms。先研究 SSIM 前向统计复用与 Renderer Backward 的真实热点，保留 11×11、原公式和VJP。NCU仍无计数器权限，不能断言属于DRAM瓶颈。
2. **P1：Renderer动态计数与CPU边界。** `num_rendered` 控制CUB sort长度与binning容量，`bucket_sum`控制sample容量；另有radii的nonzero用于监督索引。这些读取有真实消费者。下一步需设备计数接口和容量策略的独立设计；不能删除同步或以padding Gaussian代替真实K。
3. **P1：Topology批量迁移。** 可研究同一survivor permutation同时迁移Master/R3/Adam，必须保留叶子替换、grad=None、别名与clock。emptyCache与驱动释放开销在共享GPU上波动很大，不能以单轮事件结果宣称稳定加速。
4. **P1：MLP输入与Encoder前向融合微基准。** 先比较真实N/K下cuBLAS和自定义融合；不自动重写GEMM、不启用不同精度。Budget继续原ATen归约，不为多省几个launch改变零点附近的符号。
5. **P2：额外Anchor哈希/R3 lookup/workspace复杂化。** Full统计已减少到 8 次launch/1次错误回读，R3仍只gather K行；GPU hash低频且涉及FP64负坐标量化。Workspace只少4个分配请求，steady-state物理设备分配原本就是0，常驻显存反而增加。

边界：人工ULP prune probe连Legacy自身也发生翻转。最终总opacity仍调用原FP64 index_add_；不得以调阈值、扩大容差或取消检查宣称全域exact。长期连续训练存在Renderer atomic和tiny-gradient Adam敏感性，严格差异文件保留，不能改epsilon以通过Gate。
