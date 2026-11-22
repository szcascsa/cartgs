# Compensator32 蒸馏实现与验证报告

已实现独立学生训练、校准初始化、保存恢复、质量评价与冻结 CUDA 部署。本轮只蒸馏 Compensator，Selector、Master、Anchor Feature 及拓扑采用原 checkpoint 的冻结状态。实际速度尚待 GPU 空闲后的配对测量。

当前学生尚未通过沿用的严格质量门槛，所有正式比较均采用同一个教师模型和同一套选点。代码与学生权重作为独立实验交付。

## 学生结构

|项目|Teacher|Student|
|---|---|---|
|物理及 Feature 输入|Offset3＋Residual48，51D|相同|
|连续 ratio|独立 3D 投影|相同|
|局部视角|7D＋模式1D 的独立投影|相同|
|共享主干|64、64|32、32|
|四个 Head|各自 64→64→属性输出|各自 32→属性输出|
|最长 Linear 路径|4层|3层|
|MLP 参数，含视角投影|25,742|3,534|
|每点线性层 MAC，含视角投影|25,344|3,456|

学生网络参数减少约 86.27%，MAC 减少约 86.36%。这些数字不包含冻结 Anchor Feature 表、Selector、Master、Feature 查询、非线性、解码和选点。主地图和 Feature 表的体积仍需独立计算，不能由 MLP 压缩率推断整个 checkpoint 的压缩率或整帧加速。

Residual48 的 Base 与活跃 Residual 按原顺序查询。XYZ 输出继续乘 Anchor 参考尺度，Scale、四元数、Opacity 和 SH DC 沿用对应教师 checkpoint 的物理变换。Teacher 的 Softplus 与物理加法两种 Scale 解码均可读取。

## 蒸馏目标和梯度路径

学生输入、Master 属性、Feature 表、教师目标和参考尺度均 detach。仅学生的 13 个参数 Tensor 及可选视角投影参与 Adam。继承教师前 32 个隐藏神经元用于初始化共享层，再对训练视角的校准样本做 FP64 Ridge 输出层拟合。校准完成后的训练和部署采用 FP32。

监督包括归一化14维残差 MSE、参考尺度归一化 XYZ、物理 Log-Scale、旋转四元数的符号等价距离、激活 Opacity、SH DC，以及同一选点集合下的教师图像 MSE。旋转项为 1−⟨qS,qT⟩²，避免把 q 与 −q 当作不同旋转。加法 Scale 的无效负值在损失中有恢复项，渲染入口继续拒绝非有限属性。

训练同时轮转全局模式和局部模式。全局模式复用当前工程的 FlexGS 阈值及同分规则，局部模式复用名义 Anchor 候选池与严格 K。图像监督使用完整选择集合，属性小批采样只减少每步监督点数。图像的 ratio／模式独立轮转，避免低频间隔与属性周期重合后始终只训练某一种模式。

## 实际训练与质量修正

第一阶段执行 6,000 个 Adam 更新，每次属性样本最多 4,096 点，学习率 0.001，seed=0，训练 ratio 为 0.01、0.05、0.1、0.3、0.5、0.7。每 20 步加入一次权重 1 的完整图像监督，共 300 次。按属性验证分数选择的最佳 checkpoint 位于第 5000 步。

单个匹配验证视角的 12 组梯度诊断中，图像梯度与属性梯度的范数比中位数约为 0.001869。这说明第一阶段图像监督明显偏弱，尚不能把质量差距全部归因于学生容量。诊断只在一个视角执行，方向、范数与频率折算值均保存在 DISTILLATION_GRADIENTS.json。

第二阶段从第一阶段最佳权重初始化，重建 Adam 和采样 RNG，增加 2,000 步图像校准。图像权重改为 1000，每 5 步一次，共 400 次。验证分数加入权重 1000/5 的匹配图像 MSE，使最佳权重的选择与实际训练目标相符。该阶段同时改变图像权重、频率和 checkpoint 选择指标，不能据此拆分三者各自的收益。

最终最佳权重来自校准阶段第 1500 步，沿权重继承路径共有 6500 次更新。两阶段实际执行总预算为 8,000 步。源码保留第一阶段快照 STAGE1_SOURCE.zip，训练配置分别保存在两个模型目录。

最终训练入口默认图像权重为 1000、间隔为 5，全部训练参数均可通过 CLI 配置。--resume 恢复权重、Adam 和采样状态并检查目标一致性，--initialize 只继承权重并开始新训练。旧目标的 checkpoint 用于初始化新目标时需明确选择 --initialize。

## 逐 ratio 质量

评估覆盖 16 个独立测试视角、11 个 ratio、全局与局部两种模式，共 352 组。每五帧留出一帧，训练排除全部留出帧。测试进一步排除用于选择 checkpoint 的三个验证视角。两个阶段使用完全相同的测试视角及教师选点。评价原图全分辨率，图片保存在内存中，磁盘图片数为零。

PSNR 损失表示 Teacher 对 RGB 真值的 PSNR 减去 Student 对同一真值的 PSNR，正值为下降。未见 ratio 为 0.03、0.08、0.2、0.4、0.6。质量门槛预先沿用平均损失不超过 0.01 dB、p99 不超过 0.05 dB、最差不超过 0.1 dB，逐 ratio、逐模式判断。

|Ratio|模式|属性阶段平均损失 dB|图像校准后平均损失 dB|Student 对 Teacher 图像 PSNR dB|门槛|
|---|---|---:|---:|---:|---|
|0.01|全局|0.0818|0.0640|37.659|未通过|
|0.01|局部|0.0638|0.0063|36.685|未通过|
|0.05|全局|0.2056|0.0943|37.620|未通过|
|0.05|局部|0.2491|0.1351|35.891|未通过|
|0.1|全局|0.2560|0.1308|36.955|未通过|
|0.1|局部|0.3526|0.1723|35.536|未通过|
|0.3|全局|0.0823|0.0432|42.246|未通过|
|0.3|局部|0.1947|0.1121|37.726|未通过|
|0.5|全局|0.0572|0.0256|46.060|未通过|
|0.5|局部|0.2056|0.1335|39.494|未通过|
|0.7|全局|0.0603|0.0222|45.868|未通过|
|0.7|局部|0.2068|0.1284|39.511|未通过|
|0.03|全局|0.2434|0.1364|35.412|未通过|
|0.03|局部|0.2890|0.2201|32.469|未通过|
|0.08|全局|0.2559|0.1236|37.217|未通过|
|0.08|局部|0.3347|0.1640|35.574|未通过|
|0.2|全局|0.1026|0.0549|38.170|未通过|
|0.2|局部|0.1681|0.0867|36.097|未通过|
|0.4|全局|0.0460|0.0183|44.641|未通过|
|0.4|局部|0.1910|0.1227|39.107|未通过|
|0.6|全局|0.0572|0.0215|46.274|未通过|
|0.6|局部|0.2031|0.1293|39.671|未通过|

所有逐视角 Teacher／Student PSNR 和属性损失保存在两个 quality.csv。当前结果仅来自 fr1_xyz 的单个教师 checkpoint，尚无跨场景蒸馏结论。

## CUDA 与现有优化兼容

Student CUDA Kernel 在原有冻结部署调度中根据 13 个参数 Tensor 的结构分发。Teacher 的 21 Tensor 分支保持独立。学生专用 4／8／16 点 Tile 在同一次 Kernel 中完成 Feature Gather、顺序 Residual 累加、输入打包、32/32 MLP、四个输出 Head、物理解码与发布。比例三角函数仍按请求计算一次。

继续复用 Selector 静态前缀、每帧可见 Anchor 编码、GPU 动态 N/K、稳定选择、有限值状态、Graph 计划、输出独立存储和大地图 Host 计数路径。Student 作为显式选项传入，正常训练入口与原 Teacher checkpoint 均保留原有身份。

学生 checkpoint 绑定 Teacher、Master、Feature 表、Anchor 映射和 Gaussian 行顺序的 Tensor 内容 SHA256。地图、权重或拓扑变化时需要重新训练／绑定并重建部署实例。加载时检查结构、有限值、视角 schema 和教师签名。

## 测试结果

|验证|结果|
|---|---|
|固定输入 Student CUDA 对 PyTorch|3168组通过，最大残差误差 1.49e-08|
|训练后权重 CUDA 对 PyTorch|33组通过，最大残差误差 1.91e-06，最低图像对比 PSNR 105.584 dB|
|Teacher 原路径|与 v30 的选点和属性逐位一致|
|Graph 与 Native|Student 选点和属性逐位一致|
|五类 checkpoint schema|真实视角、仅模式、单分支视角和旧11D均通过|
|CPU 梯度隔离|Teacher、Master、输入、Feature无梯度，仅 Student 更新|
|Checkpoint／Adam|保存恢复与 CPU 下一步更新逐位一致，错误教师签名拒绝加载|
|真实入口|训练、渲染梯度、保存、恢复、独立测试视角评价通过|
|Compute Sanitizer|memcheck、initcheck 零错误|

CPU 合成样本训练损失从 0.168472 降为 0.036408，合成留出损失从 0.196867 变为 0.232452。该单元测试检验优化器及隔离语义，留出误差上升已保留记录，真实质量结论采用上面的独立场景评价。

Teacher 与 Student 结构不同，因此不要求两者逐位相同。Student 的 CUDA 与自身 PyTorch 使用独立的数值容差与图像对比验收。推理误差与蒸馏误差据此分别评估。

## 服务器空闲后测速

本轮没有推理速度数字。benchmark 入口检查其他 CUDA 进程，GPU 未空闲时拒绝开始，并在每个条件前后复核。采用共同预热、ABBA 顺序、三个重复、跳转相机和实际 N/K，输出 prepare 与 prepare 加原 Rasterizer 的配对数据。MAC 压缩率不作为实测加速。

在本研究目录运行以下命令。所有路径指向独立研究输出。

```bash
cd /home/prmi/zwj/experiments/compensator_distillation_20261108
PY=/home/prmi/anaconda3/envs/zwj/bin/python
ENTRY=source/research/compensator_distillation_20261108/run.py
TEACHER=/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply
DATA=/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz
$PY "$ENTRY" benchmark "$TEACHER" --dataset "$DATA" --student models/fr1_student32_image_calibration/best.pt --output reports/idle_speed
```

重新蒸馏使用 train 子命令；延续同目标使用 --resume；从已有 Student 开始新的目标使用 --initialize。评价使用 evaluate 子命令与 --student。直接部署入口支持 DeploymentRenderer.from_checkpoint(teacher, student_checkpoint=...)，deployment_infer.py 支持 --student。

## 完整性与限制

|对象|核验文件数|变化数|
|---|---:|---:|
|PRODUCTION|604|0|
|INPUTS|42|0|
|LEGACY_INPUTS|28|0|
|LARGE_INPUTS|14|0|

新扩展 /home/prmi/zwj/experiments/compensator_distillation_20261108/build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so，SHA256 为 fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601。

生产源码、已有模型、共享环境与 Rasterizer 采用既有只读输入，Git HEAD 保持本轮起点，未执行 commit／push。学生权重、数值报告和独立二进制保存在本轮目录。

下一步先在空闲 GPU 测真实收益。如果质量仍未达到门槛，优先用同一初始化、预算及测试视角，分别控制图像权重、图像频率和最佳 checkpoint 指标，并检查局部模式的属性残差长尾。当前已验证的压缩与数值兼容性不等同于教师质量可替代性。
