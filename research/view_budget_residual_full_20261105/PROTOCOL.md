# fr1_xyz视角残差完整训练对照协议

2026-11-05。用户授权在小型视角残差实现完成后进行完整训练和质量验证。复用`view_budget_residual_20261105`中已通过16项原生测试、三组36步验收和54项推理验收的相同二进制。新独立目录`/home/prmi/zwj/experiments/view_budget_residual_full_20261105`，不覆盖此前实现验收、历史模型或生产默认，不commit/push。

三组：baseline=原Full＋全局LoD＋局部LoD，无视角残差、LI=0；view_no_li=同方案增加657参数视角残差、LI=0；view_li01=视角残差、LI=0.1且LI只更新残差四个参数Tensor。用户随后明确要求复用已有baseline：`local_importance_li_20261105/runs/full_no_li`及其已完成的34配置×792帧评价，不重复训练或渲染baseline。仅两个视角组从同一未训练Full11980 checkpoint重新开始，fresh Adam/RNG（seed0），不是接着36步验收模型继续。原61个状态Tensor、Master文件、逐步相机及ratio应一致；两个视角组新增分支初态也应逐bit一致。

每组Fine11580步，训练ratio[0.01,0.05,0.1,0.3,0.5,0.7]各1930。每步Full/global/local三次训练渲染、两LoD继续原Gumbel-ST和软预算，三分支反向后一次共享Adam；全局GI保留，Master可更新且拓扑固定。N13298、Anchor2790、72训练关键帧。六Feature表独立Adam时钟应为11580/9650/7720/5790/3860/1930。记录实际选点累计、预算偏差、有限值及时间；保留失败现场，不盲目重试。

每组评价相同792 RGB帧，Bank72、non-Bank720。主评价与此前一致：Full；global/view各11档ratio[0.01,0.03,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7]；固定预算1k/2k/3k/4k/5k/10k；五档各自全局Alpha贡献数量匹配局部预算，共34配置。0.7仅列训练档。局部推理对视角模型显式开启view_conditioned；全局始终使用原评分。固定点数沿用min(B,E)和网络ratio=B/N，不改变Slot资格或名义Anchor候选，保留不足与外推标记。

预先追加少量同checkpoint诊断：两个视角模型各在1%/5%/10%和1k/2k预算关闭视角残差再评价（每组5配置）。这些数据单独标记，不混入主表或用来挑选checkpoint。主评价80784行，诊断7920行，合计88704行，其中baseline26928行只读复用；新增61776行，新模型首/中/末共234次普通/计数Renderer颜色和radii逐bit核对。

FP32、关闭TF32、颜色clamp[0,1]，逐帧PSNR后平均，TorchMetrics 11×11 SSIM、AlexNet LPIPS(normalize=True)，batch8。输出均值/median/p10/p90、改善视角比例、实际K、投影数和Alpha贡献数，以及Bank/non-Bank；不保存渲染帧图，只保存曲线图和数值。

延用此前非劣筛查门槛，不能看结果后调整：相对本轮baseline，全局六训练ratio平均PSNR下降不超过0.10dB、任何训练档下降不超过0.20dB、Full下降不超过0.10dB。判断局部收益时必须分别列1%/5%、固定预算、未见ratio及SSIM/LPIPS，不能只看均值；该筛查不是单seed统计非劣证明。已有baseline只读复用，新训练顺序固定view_no_li→view_li01，不改变终点或根据non-Bank调权重。

共享GPU可能有其他任务，不修改或终止它们，墙钟差异不能直接当作模型速度差异。每阶段检查真实PID/start_ticks/boot_id、日志/CSV增量和模型身份；训练与评价后核验604个受保护生产文件、二进制与评价输入SHA。源码/配置/模型/日志全部独立。
已有baseline使用上一轮无视角版本二进制，本轮两个视角组使用新二进制；记录二者SHA差异，不能声称三组二进制完全相同。关闭新开关的零初始化/旧路径回归已经通过；仍须说明历史对照和单seed CUDA复跑波动限制。
