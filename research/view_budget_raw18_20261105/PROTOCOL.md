# fr1_xyz：18D原始输入与39D中间特征的视角残差对照

2026-11-05，用户授权实现18D方案并验证效果。只在独立研究目录编译和实验，不覆盖生产、既有模型或默认配置，不commit/push。

新增raw_no_li、raw_li01两组，与已有view_no_li、view_li01分别匹配比较，原联合无LI作为质量保护基线。已有三组训练和88704条逐帧评价只读复用，不重训或重渲染。

新残差输入严格复用Selector的11D：局部offset3、activated Scale3、SH DC3、activated Opacity1、log(ratio)1；拼接原7D Anchor视角编码，18→16→1，共321参数。旧方案hidden32＋相同view7，39→16→1，共657参数。原始输入和相机编码均detach，keep logit加残差、drop logit不变，输出零初始化。全局导出不调用残差。容量与输入同时变化，因此不能把质量差异完全解释为原始物理信息的作用。

局部Photo/Budget保留live全局logit及原渲染梯度路径，LI=0.1仅对残差四个参数Tensor进行定向VJP。Full＋全局LoD＋局部LoD同相机同ratio反向后一次共享Adam。主Selector、Residual48 Compensator、Loss与优化器参数不变。保留schema1兼容，schema2表示physical11；拒绝混用schema，不新增可训练Gaussian latent。

从同一未训练Full11980起点，fresh Adam/RNG(seed0)，Fine11580步，Master可更新、拓扑固定，N13298、A2790、72训练关键帧。训练ratio[.01,.05,.1,.3,.5,.7]各1930；Full/global/local各11580次，GI刷新额外Full864。Feature六表Adam时钟11580/9650/7720/5790/3860/1930。36步smoke只作验收，完整训练重新从原Full起点开始。核对原61个初始状态Tensor、Master文件及相机/ratio序列；新18D两组分支初态相同，不能宣称与39D新增分支权重相同。

评价相同792帧，Bank72、non-Bank720。主配置34种：Full；global/view各11档ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]；局部固定点数1k/2k/3k/4k/5k/10k；五档使用各自global Alpha贡献数作局部请求预算。并对每个新checkpoint在view1%/5%/10%、points1k/2k关闭残差诊断，共39配置。新增61776行，加已有88704行共150480行。已有匹配预算仅为每模型自身的global贡献数，不称跨模型严格same-K。固定点数仍为min(B,E)，记录实际K和候选不足。

评价FP32、TF32关闭，颜色clamp[0,1]，逐帧PSNR后平均，11×11 SSIM、AlexNet LPIPS(normalize=True)，batch8。记录逐帧配对差异、median/p10/p90、改善视角比例，重点独立报告1%/5%及固定点数。首/中/末×39×2共234次普通/计数Renderer逐bit颜色及radii核对。不保存渲染帧图片，仅保存数据与质量曲线。

延用质量保护门槛：相对原联合无LI，全局六训练ratio平均ΔPSNR≥−0.10dB，每个训练档≥−0.20dB，Full≥−0.10dB。阈值不随结果调整。18D与39D按同LI匹配比较，完整列出全局/局部及SSIM/LPIPS；单seed不证明稳定收益，不以高ratio均值掩盖1%/5%退化，不按non-Bank结果选择checkpoint或终点。

真实OOM最多额外重试2次，其他失败保留现场并定位，不盲重训。共享GPU其他任务不得终止；墙钟时间只作实际记录，不能直接视为网络速度差异。实验前后核验604个受保护生产源码、配置、二进制及动态库SHA，既有输入模型和指标SHA，实际/proc/PID/maps加载本轮独立库。
