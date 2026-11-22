# fr1_xyz：三种视角条件组合加入LI=0.1的匹配实验

2026-11-06。用户要求Selector视角／Comp模式、Selector模式／Comp视角、两网络真实视角三组都加入LI。沿用已验证的LI权重0.1，不做权重搜索。每组从与无LI实验相同的Full11980起点重新Fine11580步，不接着无LI终态继续训练，不重复训练或渲染无LI模型。

## 唯一实验变量

三组分别为selector_li（schema3）、compensator_li（schema4）、both_li（schema2）。各自对应selector_view、compensator_view、view，条件与网络不变，唯一新增训练目标为0.1×局部LI Loss。全局LoD1条件零8D，局部LoD2 mode=1，按原组合读取真实或零view7。原无条件baseline和mode也只读保留作参考，主比较是每组LI对其同结构无LI。

保留原GI、Photo/Budget Loss、学习率、种子、Gumbel-ST采样、软预算、Strict K推理、Softplus Scale和无XYZ限幅，Master可更新，拓扑固定，原Feature48和四Head不改。

## LI真实实现与梯度

复用每步未选择未补偿Master Full：C_i=sum_pixels(T_i*alpha_i)/(H*W)，所有像素，无新增渲染。detach当前步同相机、同拓扑统计，不使用RGB-D深度。该计数是在已有Full渲染内采集，view_budget_li_full_renders表示采集次数，不是额外渲染次数。

局部候选为名义Anchor视锥内所有Gaussian，预算人口Nv；教师排序按当前C、同分按全局GI、再按固定拓扑source row ID，正标签K=min(floor(r*Nv),eligible点数)。候选中贡献全零时跳过LI；若目标K大于正贡献点数，会补零贡献点，须报告教师覆盖和选择贡献质量，不能把教师名义K当实际采样K。

LI Loss=(sum_i |ST_mask_i-label_i|)/max(1,Nv)。保留原可微Gumbel-ST选择权重，教师detach。定向VJP只累加Selector六个私有参数Tensor（含19D首层），不向Master、Compensator、Anchor Feature或相机编码添加LI梯度。已有Full/global及Photo/Budget梯度保留，三路之后一次Adam。LI仍可通过未来选点变化间接改变Master与补偿训练，不能把直接梯度隔离说成终态地图不变。

## 验收、训练和完成标准

先编译、旧开关回归、三schema定向VJP逐bit隔离测试、Full采集颜色/radii相等、空候选、36步每组smoke、跨语言fixture及短程推理。通过才完整训练，正式从原Full重读。核验初始61原Tensor（首层原11列）、Master、11580条相机ratio决策，与无LI匹配；首步Full/global/local Photo/Budget应相同，新增LI及总Loss另列。

seed0；N13298、Anchor2790、72关键帧；Fine11580，ratio[.01,.05,.1,.3,.5,.7]各1930。Full/global/local各11580，原GI额外Full864，LI采集11580、不增加渲染。LI更新次数以实际有效教师为准，fr1_xyz预期11580，缺失需查原因；Adam仍11580，Feature六表时钟11580/9650/7720/5790/3860/1930。fresh Adam/RNG，不称完整SLAM恢复。记录教师K、零贡献填充、Teacher覆盖、选中贡献质量和梯度方向/范数，复用现有低频诊断，不大幅拖慢训练。

## 相同评价与预声明判断

每新组34条件×792帧：Full；global/view各11ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]；局部固定1k/2k/3k/4k/5k/10k；五档按本模型global实际Alpha点数作局部请求预算。这不是跨模型same-K，候选不足时实际K=min(B,E)。只读复用134640行，新增80784行，共215424行。Bank72/nonBank720。FP32、TF32关闭、颜色clamp[0,1]、TorchMetrics11×11 SSIM和AlexNetLPIPS(normalize=True)。306次普通/计数Renderer颜色与radii逐bit检查。

主指标：每结构LI−无LI的1%／5%／10%及固定2k PSNR/SSIM/LPIPS、Full/global质量、配对分布和改善视角比例；XYZ/reference及米制p50/p90/p99/max、Scale/Rotation/opacity、Alpha参与。重点看两网络视角组长尾是否缓解，不能只看教师损失更低。

预声明质量保护沿用：相对原无条件baseline，全局六训练ratio均值≥−0.10dB、任一档≥−0.20dB、Full≥−0.10dB；同时对每个匹配无LI对照应用相同保护并分别报告。低ratio退化不得被高ratio均值掩盖。单seed单场景，不提供跨seed稳定性主张，不据nonBank挑学习率、权重、终点或checkpoint。

必要时仅新模型首/中/末帧检查XYZ>10 reference点的Alpha退出，共36 LoD＋9 Full诊断渲染；历史模型诊断复用，重渲染0，不存渲染帧。10只是诊断分组，不修改训练约束或成功门槛。Master终态差异单列，实际K/曝光不同不能全归为LI结构收益。现有硬故障检查（XYZ256、Scale100且1%越限）、finite、连续零选点、饱和检查保留，未触发不等于几何合理。

所有工作在/home/prmi/zwj/experiments/view_conditioned_li_20261106独立源码/构建/运行目录。保护604个生产文件、历史输入和既有指标的前后SHA，检查实际lib加载、本轮source/binary一致。OOM有证据最多额外2次，非OOM定位，磁盘不足5GiB暂停不清理。不改生产、共享conda或无关进程，不commit/push。输出中文报告、模型路径、完整CSV、PNG/PDF曲线与ZIP。
