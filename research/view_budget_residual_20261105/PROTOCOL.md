# 局部小型视角残差：实现与验收协议

本轮实现用户确认的设计，不启动完整质量消融，不切换生产默认，不提交或推送。源码工作树为research/global-view-joint-lod-20261105；服务器独立目录为`/home/prmi/zwj/experiments/view_budget_residual_20261105`，独立源码、对象、二进制、动态库、配置和输出。第三方依赖只读复用，共享conda和其他GPU进程不修改。

新增`ViewBudget.view_residual: 1`；默认0。仅与当前Full＋全局LoD＋局部LoD联合训练兼容，不启动旧`ViewResidual.enabled`或`ViewInput.enabled`的轮转调度。每个Mapping仍同相机同ratio三路反传后一次共享Adam，名义Anchor局部候选、Gumbel-ST软预算和Residual48 Compensation不变。

每Anchor使用已有7D相机编码：朝向相机的世界坐标单位向量3、asinh归一化相机深度1、中心投影NDC位置2、归一化投影包围盒最长边的log1p值1。固定场景尺度在Fine首次使用时由Master范围初始化并保存，编码detach；不依赖深度、GT、LI或额外渲染。每次局部前向按当前相机/Anchor重算，不跨参数或相机变更缓存。

已有Selector第一隐藏层32D detach，与view7拼为39D，经过39→16→1得到保留logit残差，新增657参数。输出层零初始化。局部Photo/Budget通过原全局logit继续更新原Selector及原渲染路径；LI通过限定VJP只更新视角分支四个参数Tensor，不能改变已有Master、原Selector、Compensator或Feature梯度。旧版applyViewResidual默认仍detach全局logit，新局部分支显式采用live logit。

完成标准：CPU/CUDA零初始化、随机状态、选点身份、精确LI梯度累加与隔离、原logit梯度、checkpoint测试；真实Mapper测试Full/global/local曝光、一次Adam、空候选、原实现回归；fr1_xyz原Full checkpoint做36步三组短程验收（关闭分支、开启分支无LI、开启分支LI=0.1），检查61个原状态Tensor逐bit相同、相机序列、人口和Feature Adam时钟；通过Python现有视角推理入口检查三档ratio和固定点数预算，不保存渲染图片。短程结果只证明接线与数值健康，不作为质量收益证据。

生产604个受保护源码、配置及二进制文件前后SHA核对；保存本轮源码和实际加载动态库身份。失败保留现场并定位，不盲目重试。视角推理沿用现有Python/PyTorch后端；旧CUDA/Graph推理接口对视角条件显式拒绝，不声称已融合新增MLP。
