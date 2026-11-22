# 小型视角残差分支实现报告

实现日期：2026-11-05。工作分支：`research/global-view-joint-lod-20261105`。独立服务器目录：`/home/prmi/zwj/experiments/view_budget_residual_20261105`。

## 实际改动

新增独立配置 `ViewBudget.view_residual`，缺省0。新模板 `cfg/selector/spatial_offset3_residual48_global_view_residual_li.yaml` 开启残差并设LI=0.1；原生产默认文件不更改。`ViewBudget.view_residual=0`可回到原局部联合路径；它与旧 `ViewResidual.enabled` / `ViewInput.enabled` 交替调度互斥。

复用已有 `SelectorViewResidual`（39→16→1）和每Anchor7D相机编码；新增657个可训练参数、四个参数Tensor。输入为原Selector第一隐藏层的32D输出（detach）与Anchor7D编码（detach）。局部保留logit加残差，丢弃logit不变；输出层零初始化。全局分支不调用残差。Compensator仍为Residual48，不增加视角输入、私有Gaussian参数或新渲染。

7D编码为Anchor朝向相机的世界坐标单位方向3、以固定场景尺度归一化后的相机深度asinh值1、中心NDC位置2、投影名义Anchor包围盒的归一化最长边log1p值1。使用原始Anchor/相机几何，不通过补偿结果生成；不需要GT或深度。场景尺度首次Fine使用Master包围范围初始化并存入checkpoint；相机编码每次局部前向重新生成，不跨相机/参数或拓扑更新缓存。

`applyViewResidual`增加一个缺省true的detach_global参数，保留旧调用行为；当前新局部分支显式传false。这样局部Photo/Budget仍通过原logit训练原Selector，残差网络的输入detach只隔离其内部旁路，不切断原logit路径。

LI仍为当前未补偿Full的全像素alpha*T/HW标签监督，Loss不改。开启残差时，限定VJP只将LI梯度累加到 `selector_view_residual.*` 四个Tensor；关闭残差时继续原六个Selector Tensor。Master、原Selector、Compensator和Anchor Feature已有梯度必须逐bit保留。

训练仍为同相机同ratio的Full＋全局LoD＋局部LoD，三路反向后一次共享Adam。局部候选仍是名义Anchor视锥、Gumbel-ST＋软预算，未改为Top-K。原checkpoint缺少分支时，独立Fine入口只补上零输出模块；原权重与Anchor人口保持，并沿用fresh Adam/RNG协议，不声称完整SLAM恢复。

## 推理接口

现有Python `FixedBudgetRenderer.prepare(camera, budget=..., view_conditioned=True)` 和 `ratio_budget=...` 可复用新checkpoint；全局导出使用 `global_ratio=...`，不启用view_conditioned。Viewer已有视角条件开关和schema自动识别。当前视角分支使用PyTorch后端；已有CUDA/Graph推理入口对视角条件明确报错，尚未声称新增MLP已CUDA融合。

## 验收范围

本轮只做功能验收：原生CPU/CUDA测试、真实Mapper小步测试、fr1_xyz同Full起点36步三组验收及Python固定预算/ratio推理。未启动11580步完整质量消融，因此不报告PSNR提升或最终训练速度收益。

16项原生入口测试全部通过，含CPU/CUDA的657参数与随机状态检查、零输出评分/选点一致、残差限定VJP精确累加、原梯度逐bit保护、原logit Photo/Budget梯度保留、checkpoint往返、旧detach行为回归、空候选与健康停止回归。

实际CUDA融合Selector测试将原Selector、Master、Compensator和所有Anchor Feature梯度预置为同一非零值，执行限定LI VJP后逐bit检查：只有残差4个Tensor按预期累加，其他梯度不变。另一次反传确认局部Photo/Budget仍能到达原Selector和Master输入。没有用“没有梯度”代替“已有梯度必须保持”的验证。

fr1_xyz以相同Full11980起点分别进行36步：关闭分支/无LI、开启分支/无LI、开启分支/LI=0.1。原61个状态Tensor初始逐bit一致，新增7个状态Tensor（4参数、2归一化buffer、1schema），初始Master文件相同，逐步相机/ratio序列一致。第一步局部Photo、Budget、K与全局K相同。每组Full/global/local各36次、共享Adam36次，六ratio各6步，Feature Adam时钟36/30/24/18/12/6；N=13298、A=2790、拓扑固定。两组开启分支的隐藏层与输出层参数均发生更新。

Python加载两个新checkpoint，在首/中/末三个真实视角完成54项预算渲染验收：0/1k/2k点预算与1%/3%/5%/8%/30%/70%局部ratio。实际K始终min(请求K,合格候选数)，输出有限；固定输入重复选点一致。相机变化确实改变局部logit；局部推理前后重新计算全局导出，选点身份和图像均不变。模型文件哈希保持，不保存渲染图片。

这只验证实现及短程训练健康。36步不支持最终PSNR、收敛质量、跨视角泛化或训练耗时收益结论。新增657个FP32权重约2.57KiB，连同梯度和Adam两组矩约10.27KiB，不包含激活与Tensor元数据；未做新增相机编码/MLP的Nsight测量。

## 尚未覆盖的质量与诊断

完整三组11580步质量消融尚未运行。LI从共享Selector改到视角残差后，Full/global反向不会预先给残差留下梯度，旧日志中的pre_local梯度余弦可能因参考范数为0而返回0；不能把它解释为与局部Photo梯度正交。若下一步测冲突，需要在匹配视图上单独测局部Photo/Budget与LI的VJP，本轮未为诊断增加额外渲染反传。

相机编码不直接表达遮挡、同Anchor成员共用view7，当前无证据证明它足以拟合全部LI教师。全局网络结构未改，但局部Photo仍更新共享Selector，Master和Compensator仍共同更新，因此不能承诺后续完整训练的全局质量完全不变。

旧渲染路径、生产默认和所有此前未提交工作保留。本轮无commit/push。服务器604文件完整性、代码/二进制身份及验收记录见本目录results/reports中的JSON与证据包。
