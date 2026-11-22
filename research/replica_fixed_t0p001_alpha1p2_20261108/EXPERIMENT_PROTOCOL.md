# Replica固定增生阈值实验：t=0.001，alpha=1.2

范围为Replica的office0–office4、room0–room2，各跑Mono、RGB-D，seed=0各一次，共16组。重新执行完整Full SLAM＋正常Full tail，保存模型和指标，不追加LoD Fine，不保存训练或评价渲染图片。

每层下界为t×alpha^level，上界为下界×alpha。低于下界不产生梯度增生候选；闭区间内提议同层新Anchor；严格超过上界提议下一层新Anchor。原Anchor保留，空间去重等原有条件继续生效，每轮最多向下一层。

|来源层|同层候选区间|下一层候选|
|---|---|---|
|L0|[0.001, 0.0012]|>0.0012，提议L1|
|L1|[0.0012, 0.00144]|>0.00144，提议L2|
|L2|[0.00144, 0.001728]|>0.001728沿用原最大层级规则，跳过|

基础阈值固定，从首次原定Anchor增生事件生效，不做q90校准。g仍是单个成员Gaussian在有效Full统计窗口内的屏幕XY梯度范数均值，不改为Anchor跨成员平均。保留原统计分母、窗口重置、交替拓扑、Gaussian Clone／Split阈值、Anchor空间逐层减半及only anchor prune=1。

独立配置新增：`Selector.anchor_growth_threshold: 0.001`、`Selector.anchor_growth_alpha: 1.2`。阈值0恢复q90模式，alpha默认2；自动模式保留原有>=上界细化的边界，手动模式使用>。不将新的默认写入生产项目。

独立目录为`/home/prmi/zwj/realLOD/checkpoint/replica_fixed_t0p001_alpha1p2_20261108/`。从历史原Full批次源码和配置建立独立副本，重编全部Mapper对象、Mono／RGB-D入口、配置控制测试入口及原生增长测试。数据集和第三方依赖只读复用，RPATH和实际进程maps必须指向本轮独立Mapper。

验收包括本地真实C++分支提取边界测试；CPU／CUDA真实AnchorBank的L0／L1／L2等号、相邻浮点数、非有限和无观测测试；旧边界兼容；16组实际配置到GaussianModel的接线和零Fine测试；在线日志固定阈值及最终checkpoint元数据核对。

完整轨迹评价PSNR、SSIM、LPIPS、ATE、帧数、N及逐帧数值。Mono沿用尺度对齐，RGB-D不作尺度对齐；首个Mono／RGB-D模型先验收评价入口，其余全部训练后补齐评价，与前轮时序一致。只保存指标，不保存渲染图片，逐组检查图片数为0。

结果对照原q90＋alpha2同repeat1及三次最佳、上一轮q90＋1.5／2.5倍、更早q90＋2／3倍及Vanilla三次最佳。各对照的其他指标与N绑定同次。新方案仅一次，异步Mapping步数／曝光和τ₀差异需明确，不将单次对最佳差值解释为稳定因果收益。

OOM仅有明确证据时最多额外重试2次；磁盘不足5GiB停止新任务发射，不清理历史模型；不修改其他队列。保护原源码／配置／二进制／动态库和历史checkpoint，记录前后SHA256；不commit／push。
