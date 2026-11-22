# fr1_xyz：Selector与Compensator视角条件交叉消融

2026-11-06。新增两组，LI全程为0；只读复用原无视角baseline、两网络仅模式、两网络真实视角的现有训练和792帧评价，不重复训练或重渲染。所有构建和输出在独立研究目录，生产默认保持，不commit/push。

## 四格条件

|组|Selector局部条件|Compensator局部条件|来源|
|---|---|---|---|
|mode|零view7＋mode1|零view7＋mode1|历史复用|
|selector_view|真实view7＋mode1|零view7＋mode1|本轮新增|
|compensator_view|零view7＋mode1|真实view7＋mode1|本轮新增|
|view|真实view7＋mode1|真实view7＋mode1|历史复用|

所有组全局LoD1条件均为零8D。Selector19→32→32→2和Compensator Residual48、64/64、四Head及8→64无Bias投影不变；新增两组和旧mode/view注册参数量相同。保持旧schema1/2兼容，新增schema3/4显式存取/克隆/推理。真实编码仅计算一次，按分支屏蔽，不向相机/Anchor几何新增梯度；Master原物理输入与共享Anchor Feature梯度保持。

原Full11980输入，seed0，N13298、Anchor2790、训练关键帧72；fresh Adam/RNG，Fine11580→23560，Master可更新，拓扑固定，不称完整SLAM恢复。Full→全局LoD1→局部LoD2三路按同相机同ratio反传，再一次Adam。ratio[.01,.05,.1,.3,.5,.7]各1930；GI按旧协议额外864 Full，LI0。Feature六表Adam时钟11580/9650/7720/5790/3860/1930。保留Softplus Scale、无XYZ限幅、Gumbel-ST采样和预算Loss。全局人口N、局部名义Anchor视锥人口Nv；训练不严格Top-K，推理局部预算严格计入全部保留点。

## 验收与完成标准

先CPU/CUDA输入路由、768新增参数、零初始化、原始权重/RNG、梯度隔离、空候选、两新schema保存恢复/冻结克隆检查，再两组36步smoke。跨语言fixture逐bit对比，短程实际网络输入hook检查，确认仅模式分支视角列梯度为0，真实分支可学习；全局零条件仍有零梯度，保留Adam动量更新。正式训练从原Full重启，不接smoke。核验与历史61原始状态Tensor和11580相机ratio决策一致，记录K曝光差异，不用最终质量选择终点。

每新组34配置×792帧：Full；global/view各11ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]；局部请求1k/2k/3k/4k/5k/10k；5档按本模型global实际Alpha贡献数请求局部预算（不是跨模型same-K）。新增53856行，历史80784行复用，总134640行。Bank72/non-Bank720，FP32、TF32关闭、clamp[0,1]、11×11 TorchMetrics SSIM、AlexNet LPIPS(normalize=True)。首/中/末204组计数/普通Renderer颜色及radii逐bit检查。记录逐帧PSNR/SSIM/LPIPS/photo、N、K、选点SHA、归一化及米制XYZ、Scale/Rotation长尾、opacity和Alpha参与。

预声明PSNR保护不变：相对无视角baseline，全局六训练ratio均值≥−.10dB，任一档≥−.20dB，Full≥−.10dB；1%/5%、固定预算、LPIPS和几何分别报告，不以均值掩盖退化。四格差分分别计算Selector效应、Compensator效应和交互；同一组Master会被更新，不能把系统差分冒充固定地图的单点因果。单seed单场景，无统计重复稳定性主张。

几何重点比较逐视角XYZ/reference的p50/p90/p99/max和米制统计；跨模型Anchor参考一致但每点层级不同，不能统一乘0.1换算。>10只做预定诊断分组，不是新训练约束或成功阈值。保留旧硬故障阈值（256、Scale100且1%越限），未触发不等于几何合理。必要时仅新模型首/中/末做Alpha退出诊断，旧模型现有诊断只读复用。大Scale或低opacity不单独判定图像损害。不得将推理置零替代单独训练。

OOM有证据最多额外重试2次，非OOM先定位；磁盘不足5GiB暂停，不清理历史结果。不改共享conda、不终止无关进程。保护604个生产文件和所有历史输入的前后SHA，核验实际加载本轮lib、编译源和二进制不变，零渲染帧图片。交付中文报告、完整CSV、PNG/PDF曲线、模型路径、快照和结果ZIP。
