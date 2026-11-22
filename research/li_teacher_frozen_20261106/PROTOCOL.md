# 固定模型验证LI教师的实际渲染价值

2026-11-06。用户授权验证LI教师是否值得学习。本轮只执行固定模型诊断，不训练，不修改默认配置，不commit/push。

## 固定状态与比较

唯一主模型：view_budget_raw18_20261105/runs/full_raw_li01/23560_shutdown/ply。Master、原Selector、18D残差、Residual48 Compensator、Anchor Feature、Anchor归属、相机、资格Mask、推理Gumbel噪声(seed0)全部冻结。复用相同792帧，Bank72、non-Bank720。候选池沿用Viewer名义Anchor视锥；r=0.01/0.05，每帧K=min(floor(r×候选N),eligibleN)，两网络条件始终为原r，不将K换算成别的ratio。

三种集合：base=关闭视角残差的原推理；view=开启18D视角残差的原推理；teacher=当前未补偿Full的真实alpha×T/HW在同一候选池中按现有LI目标函数排序。base/view保留原有固定Gumbel噪声和稳定同分排序，teacher不添加噪声。教师同分使用checkpoint保存的FlexGS全局GI、再按持久source row升序；固定拓扑下行号与原训练UID口径一致。缺乏有效LI的视角单独标记，不当成有效教师样本。

每种集合分别关闭与开启相同Compensator，且开关前后选点ID必须一致；共3×2×2=12配置×792帧=9504条逐帧指标。补偿按原Residual48及物理Scale加法/softplus、四Head、属性变换执行，不针对teacher重新拟合。Full贡献直接调用已验收CUDA库的anchor_si::CaptureScope(true)和原生Training Forward；不重新实现光栅化、不以可见点数替代贡献。Full教师可使用当前视图信息，是监督质量诊断，不是可部署的低成本推理方案，也不是全局最优点集。

## 正确性与数据

在独立目录复制源快照和原始库来源证据，编译同源码的Python兼容CUDA扩展，调用原始Rasterizer、localImportanceTarget和Photo Loss；模型只读复用。记录当前源码、动态库、模型、全局GI、既有逐帧CSV及604个生产保护文件SHA。检查进程maps加载本轮独立扩展。

构建核查发现原生LibTorch为C++ ABI1、现有Python Torch为ABI0，无法直接链接原生动态库。保留首次失败日志；改为在本轮独立build中将同一份六个CUDA源文件原样编译为ABI0扩展（sm86、FP32、不使用fast-math），不修改Kernel。已复制ABI1库仅作来源证据，不用于Python运行。以普通Renderer逐bit颜色/radii、旧选点SHA和真实Full统计重复验证为验收门槛；不把ABI调整视为算法或数值修改已获验证的替代。

先完成原生teacher排序/同分/空集合/无观测测试，再做首中末与Bank失败视角smoke；颜色和radii与普通Renderer、旧计数Renderer逐bit比较。检查base/view与上一轮相同checkpoint、相机、ratio的选点SHA和PSNR一致；若不一致先定位，不能用PSNR接近替代。检查teacher与Python稳定字典排序一致。对部分固定视角重复原生Full贡献采集，报告数值波动及1%/5%教师ID变化；原生atomic归约差异不称为新算子误差，也不静默忽略。

评价FP32、TF32关闭、相同480×640 RGB、颜色clamp[0,1]、逐帧PSNR后平均、11×11 TorchMetrics SSIM、AlexNet LPIPS(normalize=True)。另记录未clamp图像上的原生训练Photo Loss=0.8×L1+0.2×(1−原生SSIM)，与评价SSIM区分。指标均值、median、p10/p90、改善帧比例、Bank/non-Bank，实际K、投影数、实际Alpha贡献点数、Full LI保留质量、TopK重合率均输出。训练、不补偿、补偿后三种语义分开，不以LI质量保留率升高替代图像质量提升。

## 预先判定口径

主要比较为compensation=on时teacher−base和teacher−view，分别报告1%与5%。若两档teacher−base平均PSNR均≥+0.10dB、median≥0且超过半数视角改善，non-Bank均值也为正，并且SSIM/LPIPS方向不矛盾，视为本checkpoint有进一步学习教师的价值；这是单场景固定地图筛查，不是在线稳定收益证明。若只在某个预算/补偿状态有收益，结论必须限定条件。

若teacher无补偿也差，优先检查教师目标；若teacher无补偿好而加补偿收益消失，指向当前补偿适配；若teacher好但view差，下一步检查学习/监督。不因结果修改权重、门槛、checkpoint或终点。失败视角用于归因，不作为删除评价样本的理由；不按non-Bank结果调参。全轨迹全部保留，不保存渲染帧图片，只保存数值和汇总曲线。
