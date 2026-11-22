# fr1_xyz：Selector与Compensator共享模式及视角条件

2026-11-06，用户授权：第一轮关闭LI，训练仅模式与真实视角两组，原无视角联合结果只读复用，不重训、不重渲染。生产源码、默认配置、共享环境及历史模型保持不变，不commit/push。

## 实验定义

- baseline：既有无视角联合模型。
- mode：Selector及Compensator均使用条件[零view7,mode1]进行局部训练，全局为零8D。
- view：局部使用相同Anchor的[真实view7,mode1]，全局为零8D。
- Selector原11D扩展到19D，隐藏层32/32不变；Compensator原逻辑54D扩展到62D，保持物理/Feature51投影、ratio3投影、64/64主体和四Head，另加无Bias的8→64条件投影。总新增768参数。
- view7沿用世界方向3、相机深度1、NDC中心2、投影包围盒尺度1；固定场景尺度，detach编码。新增输入权重零初始化，原始权重及Feature表保持。关闭旧Selector视角残差和交替模式。
- 每步同相机同ratio，Full反传→全局LoD1反传→名义Anchor视锥LoD2反传→一次共享Adam。Master可更新，拓扑固定。保持原Gumbel-ST、软预算、Loss及优化器。LoD1预算rN，LoD2预算rNv，采样K不投影为严格Top-K。推理沿用原全局分位数及局部严格预算定义。
- LI为0，不采集额外Full LI，不新增教师Loss；原全局GI保留。

## 固定训练与评价

同一未训练Full11980起点，N13298、Anchor2790、训练关键帧72，seed0，fresh Adam/RNG，Fine11580步。不是完整SLAM/Adam恢复。ratio[.01,.05,.1,.3,.5,.7]各1930；Full/global/local各11580，原GI刷新额外Full864。Feature六表Adam步数11580/9650/7720/5790/3860/1930。36步smoke只作验收，正式训练重读原Full。

与既有无视角模型核对原61状态Tensor（Selector第一层仅比较原11列）、Master初始文件、11580个相机/ratio决策。新两组除模式schema外初态相同。首步前向差异单列，不能把不同GEMM形状默认称逐bit相同。训练采样K和优化曝光分开记录。

评价同792帧，Bank72、non-Bank720。每模型34配置：Full；global/view各11档ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]；局部固定1k/2k/3k/4k/5k/10k；五档按各自global实际Alpha贡献数匹配局部请求预算。该匹配不等于跨模型严格same-K。新两组53856行，复用baseline26928行，总80784行。

FP32、TF32关闭、颜色clamp[0,1]、逐帧PSNR平均、TorchMetrics11×11 SSIM、AlexNet LPIPS(normalize=True)。计数与普通Renderer对首/中/末×34×2=204次颜色/radii逐bit检查。导出XYZ/参考尺度、Scale相对Master倍率、Rotation幅度与逐视角配对变化，不保存渲染帧图片，图表可保存。

## 预声明判断标准

沿用已有全局保护门槛：相对baseline，全局六训练ratio平均ΔPSNR≥−0.10dB，每档≥−0.20dB，Full≥−0.10dB。重点独立报告局部1%/5%、固定点数及未见ratio；不以高ratio均值掩盖低ratio退化。真实视角须与单独训练的mode组比较，不能将推理时置零作为训练对照。单seed、单场景只证明本次实验结果。

所有原生测试、梯度/初始化检查及smoke通过后才能完整训练。保留非有限值、连续零选点、饱和及补偿长尾检查；非OOM失败定位后处理，不盲重训。真实OOM最多额外重试2次。磁盘不足5GiB停止，禁止擅自清理。604个生产文件、输入checkpoint、既有指标和实际动态库加载路径前后核验。
