# 默认方案：两网络真实视角＋LI

2026-11-06，按用户最终指示，将“两网络真实视角＋LI=0.1”设为本地默认，并提交到本地main。用户明确要求本轮不更新服务器；未执行SSH、部署或push。

## 配置与行为

- 两个默认入口：`cfg/selector/selector_config.yaml`、`cfg/selector/spatial_offset3_residual48.yaml`。全部180项配置值与已完成实验的`spatial_offset3_residual48_joint_view_li.yaml`一致。
- `ViewBudget.enabled=1`、`conditioning=view`、`li_weight=0.1`；旧独立视角残差、ViewInput交替调度及global_control关闭。
- 在线SLAM与正常尾部仅进行Anchor Full优化；结束后Fine步数等于在线Mapping步数，固定拓扑，允许更新Master。
- Fine每步同相机、同ratio，依次Full、全局LoD、局部LoD反传，最后提交一次共享Adam；三路外层权重1:1:1。
- 全局LoD采用全图候选，两个网络的8D视角／模式条件全零；局部LoD采用名义Anchor视锥候选人口，条件为detach的真实view7＋mode1。
- Selector为物理11D＋条件8D，19→32→32→2；Compensator保留Residual48基座／五级残差、连续ratio3、64/64共享主干及四Head，附加独立8D条件投影。
- 训练ratio为[0.01,0.05,0.1,0.3,0.5,0.7]；训练保留Gumbel-ST与软预算Loss，不将LI教师Top-K替代实际训练采样。
- LI复用当前未选择、未补偿Master Full的全像素alpha×T/HW贡献，不增加Full渲染。教师detach，定向VJP仅向Selector的六个参数Tensor加梯度。
- 全局FlexGS GI仍启用；Maturity与SGI关闭。保留原Softplus Scale解码及无XYZ限幅设置。
- 全局导出使用零条件；局部预算渲染显式使用`view_conditioned=True`，由checkpoint中的条件schema恢复。不能把局部真实视角传入全局导出。

## 兼容与回退

仅修改默认YAML的显式选项，不改变缺失选项时的旧checkpoint加载语义，不覆写已保存的模型配置。

- 两网络真实视角、LI关闭：`spatial_offset3_residual48_joint_view.yaml`。
- 原无视角Full／全局／局部联合：`spatial_offset3_residual48_global_view_joint.yaml`。
- 原Full／全局LoD两路：`spatial_offset3_residual48_global_only.yaml`。
- 单网络视角、仅模式、旧39D／18D残差及LI消融配置继续保留，均为显式选择的研究对照。

## 验证与证据边界

本轮核对466个已验收的实现文件SHA与此前独立编译快照一致；新增默认配置与已完成both_li实验的180个参数完全一致。将已有的视角路由、LI隔离、空候选、保存恢复等测试注册到CTest，同时保留CUDA旧版／新版逐位fixture生成入口。

此前同份实现已完成35项原生检查、三组各36步checkpoint smoke、12组C++／Python fixture逐bit检查、90项短程推理、90项最终输入路由检查与306次普通／计数Renderer逐bit检查。三组均完成11580步，无非有限值。本轮执行本地配置、源码哈希、Python语法、CMake测试注册和Git提交核查；本机未安装LibTorch/CUDA运行环境，未重新编译或重跑GPU测试。

对应服务器研究根目录为`/home/prmi/zwj/experiments/view_conditioned_li_20261106`。默认方案已训练模型位于`runs/full_both_li/23560_shutdown/ply`；本次只切换本地代码默认，不重新训练或自动安装该模型。

fr1_xyz RGB-D、seed0、相同Full起点／11580步／792帧：

|指标|两网络真实视角，无LI|两网络真实视角＋LI|
|---|---:|---:|
|局部1% PSNR|16.9073|16.6471|
|局部5% PSNR|19.5222|19.3987|
|固定2k PSNR|21.4523|21.5753|
|Full PSNR|22.6140|22.6301|
|局部5% XYZ/reference p99视角均值|17.4889|1.2172|
|局部10% XYZ/reference p99视角均值|23.6752|1.3251|
|固定2k XYZ/reference p99视角均值|20.2855|0.3044|

默认选择遵循用户决定。现有证据表明几何长尾和部分固定预算指标改善，低ratio PSNR存在取舍；不是所有质量指标同时提高，也不是多场景／多seed验证。完整数值与完整性核验保留在该轮`LI_VIEW_CONDITIONING_REPORT.md`、`FINAL_INTEGRITY.json`及原结果包，本地实验产物不提交到Git。
