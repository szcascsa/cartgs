# 当前Full贡献的局部LI：配对实验预注册

2026-11-05，研究分支 `research/global-view-joint-lod-20261105`。只在独立服务器目录 `/home/prmi/zwj/experiments/local_importance_li_20261105` 构建、训练和评价。不切换生产默认、不提交、不推送。

## 固定设计

保留当前Full＋全局LoD＋局部LoD，外层权重1:1:1。Selector为11→32→32→2，Compensator为Residual48，六档ratio为0.01、0.05、0.1、0.3、0.5、0.7。两路LoD仍为Gumbel-ST和原预算损失，局部候选为当前名义Anchor AABB视锥的所有成员N，目标rN不取整、不强制Top-K，Slot资格不变。

LI在同一步、同相机、补偿前完整Master的Full渲染上采集所有有效像素的alpha×T，除以HW。严格沿用Alpha阈值和T早停规则；不增加渲染、不跨相机EMA。保持原全局GI累计opacity×T的口径和1000步刷新间隔。只保留当前步LI，并核验步数、相机、拓扑和人口。

从候选池Slot可用成员中，按LI降序、全局GI降序、固定拓扑源行号升序生成前min(floor(rN),E)个正标签。固定Fine中源行号是持久点身份。空池、无可用点或可用点LI全零时跳过LI，不把任意零贡献点强行当作监督。教师标签排序不修改实际采样。LI Loss为mean(abs(ST_mask-label))，分母为整个候选N。LI和标签detach；仅通过限定local_emd.*和soft_net.*六张Tensor的VJP累加参数梯度，不向Master、Compensator或Feature累加LI梯度。原Photo、Budget梯度路径不变。

新增 `ViewBudget.li_weight`，缺省0，主实验0.1。保持原三路反传后一次共享Adam；不合并VJP以冒充数值等价优化，不更改Adam状态或活动Residual表更新时钟。

## 对照和初始化

两组同一独立新二进制：no_li=0、li01=0.1。均从fr1_xyz RGB-D seed0_repeat2的未训练Full模型11980_shutdown初始化，N=13298、A=2790、72训练关键帧；重置Adam/RNG，不声称完整SLAM恢复。Fine固定11580步，各ratio1930步。相机序列、学习率、初始Master及全部网络/Feature Tensor必须配对一致。旧联合训练作为历史额外参照，主结论来自本轮配对。

## 预先固定的评价

完整792 RGB帧，Bank72/非Bank720；训练ratio及未见0.03、0.08、0.2、0.4、0.6。0.7属于训练ratio。评价Full、全局导出和名义Anchor局部严格预算，导出PSNR/SSIM/LPIPS、逐帧差值和实际K/贡献点数。复用原固定seed0推理及原FP32指标协议。额外评价局部固定1k/2k/3k/4k/5k/10k和五档逐帧全局贡献匹配预算，不保存渲染帧图片。高预算不足和ratio外推必须单列。

正式训练前固定质量筛查标准：相对本轮no_li，六训练档全局平均PSNR不下降超过0.10dB，任一训练档不下降超过0.20dB；Full不下降超过0.10dB。局部六训练档均值应提高，同时单列1%/5%和未见档，不能用高ratio掩盖低ratio损失。与上一轮一致，这只是单seed筛查，不是统计非劣证明。不得根据非Bank指标挑终点或事后改权重。

记录LI标签与原全局GI冲突、候选池覆盖Full贡献比例、选择集合捕获LI比例、实际预算误差和有效LI步数。LI显存额外约4N字节，排序、贡献Atomic和额外评分MLP VJP耗时必须计入；共享GPU墙钟只作同负载下的参考。

## 完成标准

1. CPU/CUDA教师排序、空/零/Slot不足、VJP梯度隔离和已有梯度累加测试通过。
2. Full采集前后颜色/radii逐位相同；全局GI仍为原口径，LI不增加Full次数。
3. 小Mapping和真实checkpoint smoke通过，checkpoint参数兼容、六表Adam时钟、一步一次Adam、固定拓扑验证通过。
4. 两组11580步和评价完成；所有失败保留并查明，不能静默重训；生产和输入哈希前后不变。
