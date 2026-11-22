# CaRtGS随机裁剪与Opacity裁剪对照协议

使用32组既有CaRtGS Vanilla的最高Full全轨迹PSNR重复，其余指标及模型绑定同一次。
不重新训练，不改变模型属性，不使用Selector、Compensator、视角候选或新生点保护。

- Ratio：0.01、0.03、0.05、0.08、0.1、0.2、0.3、0.4、0.5、0.6、0.7，共704项。
- 预算：K=floor(r×N_CaRtGS)，Decimal计算避免十进制浮点边界误差。严格保留K个点。
- 随机：每场景seed=0，NumPy PCG64生成一次排列，各Ratio取前K个，属于嵌套随机子集。
- Opacity：复用模型activated opacity，在CPU稳定降序排序；同值按原Gaussian行号。
- 两种方法均将选中行号恢复原顺序再渲染，避免改变像素合成的同深度顺序。
- 评价：逐字复用冻结的CaRtGS run.py尾部，包括Full轨迹、原图、TorchMetrics、逐帧平均。
- 单目保持原correct_scale设置；RGB-D／双目沿用原关闭标记，保留原evo前置对齐逻辑。
- 独立复制评价源码和CUDA扩展；同场景复用冻结Master及有界CPU原图缓存，最多6GiB。
- 每组先核验Full首／中／末帧指标与历史逐帧数值，并核验帧关联和源PLY点数。
- 各自模型总点数不同，因此这是相同比例比较，实际K分别记录，不能称同K对照。
- 随机仅一次seed0，不把单次随机差异解释为统计显著性。
- 我们方法复用原32组已评价结果，其中corner-slow／robot-normal双目采用11月11日最新复跑。
- 不保存渲染／GT图片，不写生产项目或旧实验，保留选点ID、逐帧数值及来源哈希。
- OOM最多额外重试2次；非OOM先保留现场并报错。磁盘不足5GiB暂停，不清理旧数据。
