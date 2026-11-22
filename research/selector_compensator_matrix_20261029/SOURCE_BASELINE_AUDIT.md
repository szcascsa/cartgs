# 当前源码与正式baseline核查

25号R0-G00-B0为正式baseline。Compensator为共享Anchor128、asinh及固定标准化后的局部offset3、连续log-ratio/sin/cos3，共134D；ratio门控、64/64主干、四Head、14D变换及Joint不改。Selector为线性局部offset3、激活Scale3、SH DC3、激活Opacity1、log-ratio1，共11D，11→32→ReLU→32→ReLU→2。

生产默认仍是spatial_localxyz；baseline配置明确选spatial_anchor128_offset_gate。Bound2及历史139D分支不作为25号。E系列移除Selector的Scale/DC/Opacity，使用与Compensator相同预处理的offset3、ratio3、原始共享Feature，区别于历史139D。

快照仅纳入相关11个源码改动、CMake引用的4个小测试源、本研究配置/协议/小报告。模型、数据、构建、密码及大型快照不纳入，其他未跟踪工作保留。研究实现仅修改独立副本。

固定统计时序：25号保持原3000步后首次补偿初始化；其余seed0变体使用25号固定缓冲作为训练来源校准，使E首次LoD可用，两分支共用统计。统计来自训练地图，未按nonBank质量选择。与历史各自估计统计的差异单列。配对seed1同理先跑25号再校准候选。
