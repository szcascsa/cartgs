# CPU 原生推理优化设计

目标是在训练结束的冻结模型上，将候选构建、视角编码、Selector、严格预算选择、Compensator、解码及输出打包全部放到 CPU。GPU 只接收准备好的紧凑属性并执行 Rasterizer。正常训练和原生产入口保持原样。本文件是实施设计；原生 CPU 引擎尚未实现，目标延迟不是已测收益。

## 当前实测与优先级

Intel i5-9600K，6核6线程，FP32。office2 RGB-D 地图43,197点，三个真实相机，候选数量中位9,488。原CPU PyTorch准备的两轮中位：1%=9.060ms、10%=10.039ms、70%=15.796ms。

新增轻量分段计时定位到以下结果。各阶段是单独样本中位，不应机械相加；分段包装本身也有少量开销。ATen Profiler数据仅用于归因，不作为正式延迟。

| 阶段 | 1% ms | 10% ms |
| --- | ---: | ---: |
| 状态签名遍历 | 0.172 | 0.174 |
| 名义Anchor视锥 | 0.387 | 0.365 |
| 整批候选属性复制 | 0.598 | 0.573 |
| 视角编码 | 3.417 | 3.430 |
| Selector及输入组织 | 1.357 | 1.307 |
| 严格预算排序 | 0.495 | 0.594 |
| Compensation及解码 | 1.374 | 2.142 |
| 输出打包 | 0.229 | 0.338 |
| 整体准备 | 9.056 | 10.135 |

先处理准备和内存访问，再处理GEMM。CPU上的主要融合收益是减少Python/ATen分派、临时Tensor、内存扫描和线程池往返；不能把CUDA Kernel Launch机制照搬到CPU。

## 1. 一次C++调用执行整帧准备

设计一个独立冻结实例：`CpuPreparedRenderer::prepare(camera, request)`。Python只传相机、ratio或固定预算并接收结果。全流程在C++完成，释放GIL。

实例加载时持有不可变的Master、网络、Feature表和行映射快照；状态变更通过新建实例完成。完整有限值及结构核验在加载时执行；每帧仍检查动态相机、ratio、输出有限值和预算。冻结快照避免原来每帧遍历全部Tensor的Python签名检查，不能直接删除可变模型上的失效保护。

预分配有界候选、分数、排序、隐藏层和输出Workspace。每次调用返回独立所有权或带事件租约的输出，禁止下一帧覆盖尚在使用的结果。初始化、首次内存页触达和权重打包与稳态延迟分开记录。

## 2. 模型加载阶段的空间换时间

安全复用的内容包括：

- Master激活后的Scale、Opacity、归一化Rotation、SH和对应原始log属性。
- Selector使用的固定物理输入，以及Compensator的归一化Offset3。两种offset定义分别沿用各自源码，不能因都是3维而混用。
- Anchor中心、半边长、齐次中心和8角点。
- Gaussian→Anchor映射、每Anchor成员CSR、每Anchor资格点数、原始UID和固定噪声。
- 权重按SIMD微核需要转置/分块打包。
- Residual48的6个节点有序前缀：按原顺序累加完整层；未见ratio只对当前相邻区间的残差插值，不量化ratio。

缓存与相机无关。每帧仍使用新相机、新候选和新ratio；不利用同帧结果命中测速。每份缓存都有明确容量，切换模型/地图时销毁整个实例。

可进一步缓存Selector首层固定输入的32维累加前缀、Compensator Offset3首层64维前缀。若使用固定累加顺序的原生微核，可继续按同一顺序处理动态输入；不能预先宣称与MKL不同矩阵形状的输出逐位一致。

## 3. Anchor视锥和视角编码融合

使用SoA数据布局和AVX2，每批同时处理8个Anchor。完成原六平面名义AABB判定后，只给当前可见Anchor重算view7。近裁面处理、safe_w、最长投影范围和截断公式保持现有语义。

预存8角点，角点投影、屏幕范围、深度范围和方向归一化在原生循环中完成，避免构造Ax8x4临时Tensor及多次max/min/where/cat。一个Anchor的view由其全部成员共享；Compensator仅对选中Anchor使用所需投影。

候选构建优先只输出原Gaussian行号，不先复制候选的XYZ/Rotation/Scale/Opacity/全部SH。以分块count→前缀和→scatter保证原始行序，CSR方案和全图SIMD行号扫描实测选优；不能因为内部按Anchor遍历就改变同分优先级。

## 4. Selector原生SIMD融合

当前网络是19→32→32→2。用AVX2一次计算8个输出神经元，批量同时处理若干Gaussian，输入广播、权重连续读取，融合Bias、ReLU、后续层及Gumbel分数写出。尽量让中间激活留在寄存器/L1，而不是创建多个Tensor。

ratio的log和编码每请求算一次；同Anchor的动态view投影可以复用。静态前缀减少每候选的固定输入计算，但会增加读取字节，因此保留按内存/计算共同评估的消融，不能把减少MAC直接写成加速。

复用现有固定噪声和精确概率分数。概率饱和会影响同分，因此不把softmax分数直接换成logit差排序。

## 5. 严格K选择减少全排序

局部预算保持K=floor(ratio×M)，M为名义候选数量；资格不足仍按原min(K,available)处理。复合比较键为“原分数降序＋原始Gaussian行号升序”。

比较`nth_element`、小K堆和并行Radix选择，优先使用平均O(M)的精确分区得到K个身份，再按原始行号恢复输出顺序。全局FlexGS分位数及阈值同分规则继续单独保持，不能将全局选择悄悄替换为局部严格Top-K。

## 6. Compensator、四Head和解码融合

支持同一Teacher及已有Student；Teacher→Teacher优化与Teacher→Student压缩的收益/质量分别报告。

按K分块，融合Residual48查询、Offset输入、ratio/view投影、主干、四Head、XYZ/Scale/四元数/Opacity/DC解码及最终属性写出。直接从Master和Feature表按选中行读取，不构造整候选补偿副本。共享主干只计算一次；Teacher四个Head保持独立权重，各64维隐藏层使用分组/拼接执行的原生微核，不能共享或删掉权重。与MKL/oneDNN固定权重打包的实现配对，只有实际更快才保留手写核。

进一步激进优化可以利用首层线性分解：Offset项按Gaussian预计算，Feature项按Anchor/ratio计算，view项按Anchor/相机计算，ratio项每请求一次。**这个分解改变FP32求和分组，不能称逐位等价。** 它作为独立候选，先报告残差、选点和Renderer属性误差及实际图像变化；不以代数相等代替数值证明。前缀缓存与Feature投影表只在跨视角实测有收益时保留。

## 7. CPU并行和缓存

仅使用一个固定工作线程池。阶段间最少屏障，按Anchor块、候选块或选中点块并行；MLP小矩阵不再启动嵌套MKL多线程。小任务以SIMD或少线程处理，较大任务由同一池分块。

i5-9600K只有AVX2/FMA，没有AVX-512。实现按其实际能力优化，比较1/2/6线程及可选绑核，记录p95。线程绑定只作用于本研究进程，不调整系统或外部任务。

工作集估算（FP32、N为地图点数、A为Anchor数、J≤6）：

| 可选缓存 | 容量公式 |
| --- | --- |
| Selector首层静态32D | 128N bytes |
| Compensator Offset首层64D | 256N bytes |
| Anchor8角点4D | 128A bytes |
| Residual48节点前缀 | 192JA bytes |
| 首层Feature投影表（激进候选） | 256JA bytes |

N=80,000时前两项合计约29.30MiB，超过本机9MiB L3，必须用真实缓存行为验证，不是缓存越多越快。不默认建立每点×每ratio×64D的大表，不缓存任意相机/ratio查询历史。

## 8. 全CPU准备到GPU渲染的交接

CPU直接写入预分配pinned host buffer，GPU一次连续异步上传后用view解释各属性，无GPU gather、MLP或解码准备。包括实际使用的SH，不能只测14D残差拷贝并遗漏真实输入流量。

使用2–3份有界Host/Device buffer及CUDA Event管理所有权，GPU开始渲染前等待本帧上传完成，CPU写回前等待上一使用结束。可以重叠CPU下一帧准备与GPU当前帧渲染，但必须分别报告单帧延迟和流水线吞吐量；不能用重叠吞吐冒充单帧加速，也不预测相机或复用旧视角。

## 验收与执行顺序

1. C++原生调度、静态物理数据、可见Anchor编码、只传行号：逐阶段对照CPU PyTorch，测完整prepare。
2. AVX2 Selector、稳定选择、Compensator融合和持久线程池：每项独立配对，对比自写微核与打包BLAS。
3. 静态首层/Feature投影空间换时间：独立记录数值差异、缓存成本及跨视角收益，失败则回退候选。
4. Pinned输出与GPU交接：测CPU prepare、H2D、GPU渲染和整个延迟；全部准备仍在CPU。

优先保留FP32、现有网络/权重/严格预算及选择规则。自写FMA/首层分解可能与MKL或CUDA有细微数值差异；沿用用户对部署推理微差的容许，但须单独验证图像误差，记录PSNR/SSIM/LPIPS与选点身份。任何选择变化先定位概率饱和、求和顺序或同分处理，不以总体PSNR相近掩盖错误。未经验证不推广为默认。

至少复用TUM/Replica/VECtor冻结输入，测低/高/未见ratio、固定预算、空集、容量边界、视角连续运动/跳转、冷启动和多轮p50/p95。训练、checkpoint和生产完整性按SHA检查。

第一阶段以office2约1万候选、K≤1000，将完整CPU prepare由9–10ms压到1–2ms作为研究目标，1ms以内为进取目标；不是性能承诺，也不外推至K数万。最终报告是否达标及所有反例，另列CPU→GPU交接后的实际收益。
