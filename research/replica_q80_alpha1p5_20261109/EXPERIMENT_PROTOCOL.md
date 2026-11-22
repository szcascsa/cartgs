# Replica q80基础阈值、alpha=1.5实验

用户要求“t使用0.8百分位，1.5倍”，本轮解释为第80百分位q80，而不是第0.8百分位q0.008。

Replica的office0–office4、room0–room2，各跑Mono／RGB-D，seed=0各一次，共16组。重新运行完整Full SLAM＋正常Full tail，不追加LoD Fine，不保存渲染图片。保留checkpoint、汇总指标、逐帧指标、日志与源码追溯信息。

基础阈值t在首个有效Anchor增长窗口中，由L0成员Gaussian的有限正Full梯度均值计算q80，之后冻结。窗口内没有合格L0值则等待后续窗口，不使用零阈值增长，不增加预热或改变拓扑时机。梯度保持现有屏幕XY范数的成员时间均值，不做Anchor成员间平均，也不改变分母与统计重置。

层级l的下界为t×1.5^l，上界为t×1.5^(l+1)。闭区间内提议同层新Anchor，严格超过上界才提议下一层。L0对应[t,1.5t]，L1对应[1.5t,2.25t]，L2对应[2.25t,3.375t]。L2超过上界继续按原最大层级规则跳过；每轮至多细化一级，原Anchor保留，空间去重等条件不变。

在独立源码副本复用固定阈值配置路径，增加可配置校准分位数及上界边界。默认仍为q90、alpha2、>=上界细化，手动固定阈值能力保留。本轮配置为：

```yaml
Selector.anchor_growth_threshold: 0.0
Selector.anchor_growth_quantile: 0.8
Selector.anchor_growth_alpha: 1.5
Selector.anchor_growth_strict_upper: 1
```

保留only anchor prune=1、Gaussian Clone／Split阈值、空间逐层减半、优化器／Loss／学习率、原Full tail及无视锥设置。数据与第三方预编译依赖只读复用，源码、二进制、动态库、对象文件、日志、模型、配置和评价全部隔离在`/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_20261109/`。

验收包括本地补丁范围与真实分支边界测试；全部Mapper对象和两个入口独立编译；CPU／CUDA真实AnchorBank增生边界；真实GaussianModel的空窗口不校准、仅有限正且有观测L0参与q80／q90、后续窗口不重校准；16组配置接线、零Fine和不保存图片；实际进程maps核对独立库；每次运行校准仅发生一次，日志t恒定，checkpoint分位数与alpha一致。

每组完整轨迹PSNR／SSIM／LPIPS／ATE、N、实际Mapping步数、t数值及各层占比均保存；Mono沿用尺度对齐，RGB-D不作尺度对齐。先做首个Mono／RGB-D模型全轨迹入口验收，再完成其余训练后串行补评，保持此前运行顺序。

主对照为默认q90＋alpha2的每组三次均值，指标和N按各组重复等权平均；另列默认repeat1、固定t0.001＋alpha1.2及上一轮q90＋1.5／2.5。默认最佳可保留溯源，但不作为主判定。新方案每组仅一次，异步SLAM实际训练预算／曝光和初始化仍存在变化，不声称已单独隔离q与alpha的因果作用。

OOM有明确证据才最多额外重试2次；非OOM或数值失败保留现场，不盲目重训。磁盘不足5GiB停止新任务发射，不清理模型。每小时检查实际进程、日志增量及存储；不恢复或改变其他暂停队列，不commit／push，不切换生产默认。
