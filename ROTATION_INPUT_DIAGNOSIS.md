# Rotation输入退化归因与编码对照

## 结论与建议

输入预处理值得按几何语义验证，直接对Quaternion分量取log不合适。现有证据能确认等价表示与选点/补偿适配问题，但尚不能把正常评价退化唯一归于某一因素。

- rot6相对无Rotation Baseline：两seed -0.015/-0.130 dB，平均-0.072 dB；相对Quaternion平均+0.181 dB。
- cov6相对无Rotation Baseline：两seed -0.223/-0.656 dB，平均-0.439 dB；相对Quaternion平均-0.186 dB。

本轮没有任何处理方式在两个seed上都超过无Rotation Baseline，建议继续保留59维四Head实验Baseline，不自动扩展编码/网络搜索。

Quaternion在本轮固定地图训练中本身冻结，微扰符号跳变是表示诊断，不能描述成训练期间实际发生了这些跳变。其范围小，暂不支持将其认定为主要退化原因。
原始输入尺度有界；最终梯度快照没有显示Rotation使方向冲突更严重。跨模块交换揭示支持集适配，但其分布外性质限制了纯因果分解。


本轮不更改生产算法，先审计当前Quaternion输入，再进行两种预注册6D输入编码控制。所有质量结果为同源fr1_xyz冻结地图；历史评价轨迹反复用于结构研究，不能当作新盲测。

## 1. 当前已证实的事实

- Quaternion已单位归一化、按最大绝对分量为正统一q/-q；分量有界，标准差约0.29～0.35。退化不是未做单位化或明显数值长尾。
- 0.01rad随机小扰动的符号跳变占0.0640%；SO3最近邻中转角<5°而Quaternion输入距离>1的点占0.0731%。边界问题存在，但当前样本占比很小，不能判为主要原因。
- 最大/最小Scale轴比中位数10.69，轴比<1.2的近球形点仅0.97%；不能将Rotation视为大多数Gaussian上的无关信息。
- 同一椭球的局部两轴反号后，Quaternion输入距离约sqrt(2)、Rot6距离2，而归一化协方差最大变化仅约6.6e-7；原Quaternion和Rot6会区分这种等价几何表示。

## 等价椭球输入的实际网络响应

保持真实Master与所有属性不变，仅将输入Rotation换成右乘局部X轴180°的等价表示；对Gaussian协方差没有实质改变。下表记录相同K下网络响应，属于反事实不变性诊断，不是训练收益。

| 方案 | 同K集合保留比例 | 最大概率变化 | 最大补偿分量变化 | non-Bank ΔPSNR |
| --- | --- | --- | --- | --- |
| baseline | 1.0000 | 0 | 0 | +0.000 |
| rotation_all | 0.5322 | 0.633 | 5.96 | -3.647 |
| rot6 | 0.4871 | 0.721 | 3.68 | -4.218 |
| cov6 | 1.0000 | 1.01e-06 | 1.43e-06 | -0.000 |

该诊断能证明模型是否依赖任意的椭球轴符号，不能证明这种反事实变化在原训练轨迹中经常发生，也不能将反事实损失直接归为正常评价的PSNR差距。

## 2. 选点与补偿的交叉渲染

| 范围 | 选点来源 | 补偿来源 | non-Bank PSNR |
| --- | --- | --- | --- |
| all | baseline | baseline | 17.2443 |
| all | baseline | rotation_all | 15.0521 |
| all | baseline | none | 11.4195 |
| all | rotation_all | baseline | 14.0044 |
| all | rotation_all | rotation_all | 16.9909 |
| all | rotation_all | none | 11.3622 |
| 1% | baseline | baseline | 15.3732 |
| 1% | baseline | rotation_all | 13.3178 |
| 1% | baseline | none | 6.9992 |
| 1% | rotation_all | baseline | 12.6347 |
| 1% | rotation_all | rotation_all | 14.0758 |
| 1% | rotation_all | none | 7.4488 |

- 1%无补偿时Rotation选点反而更好，但各自补偿后更差，不能只归因于选择了更差的基础Gaussian。两个模块相互交换后均明显退化，说明对各自训练选点支持集存在适配。
- 交换模块会遇到另一分支较少训练的Gaussian/Anchor，属于分布外模块组合；不能将这些差值相加成纯选择/纯补偿因果贡献，也不是重新优化后的质量上界。

## 3. 共享梯度快照

| 方案 | ratio | Selection梯度范数 | Compensation梯度范数 | Budget梯度范数 | Selection/Comp cosine |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.01 | 0.0585 | 0.0405 | 0.0694 | -0.442 |
| baseline | 0.05 | 0.0374 | 0.0323 | 0.0543 | -0.415 |
| baseline | 0.1 | 0.0230 | 0.0278 | 0.0413 | -0.371 |
| baseline | 0.15 | 0.0198 | 0.0278 | 0.0315 | -0.370 |
| rotation_all | 0.01 | 0.0800 | 0.0989 | 0.0642 | -0.298 |
| rotation_all | 0.05 | 0.0361 | 0.0336 | 0.0546 | -0.307 |
| rotation_all | 0.1 | 0.0307 | 0.0271 | 0.0437 | -0.312 |
| rotation_all | 0.15 | 0.0248 | 0.0290 | 0.0340 | -0.302 |

- 原方案和Rotation方案均有负方向关系；Rotation方案的cosine没有更负，本快照不支持“加入Rotation让方向冲突更严重”的归因。
- 1%补偿对Encoder梯度增大，但仅为最终checkpoint、每ratio固定4个Bank视图的快照；不能代替全训练过程，也不能仅凭范数/cosine认定更新有害。零梯度方向在validated CSV中留空。

## 4. 为什么不直接log Rotation

- ratio是正标量，log改变其尺度与变化分辨率。Quaternion包含负数且受单位约束，对每个分量取log不合法，也不保持旋转语义。
- SO(3)旋转对数是另一种数学变换（轴角向量），不是逐分量log；全局角度接近π时仍有分支问题。本轮未将其作为默认修复。
- 6D旋转采用R的前两列，只修正旋转坐标的连续性；归一化协方差同时考虑椭球Scale和朝向的实际几何。

## 5. 预注册输入编码对照

- Rot6：vec(R[:,0:2])，保持列顺序；Cov6：Sigma=R diag(scale²) Rᵀ，Sigma_bar=Sigma/trace(Sigma)，输入[xx,yy,zz,sqrt2 xy,sqrt2 xz,sqrt2 yz]。
- 两组均原始11维+编码6维，Encoder17→32，四Head输入17+latent16+h32=65。新增输入列零初始化，其余公共权重与原实验同seed复制；14维输出、Loss、Adam、ratio顺序、插值不变。
- 同维数Rot6/Cov6比较可检验表示语义；与Quaternion4比较同时多320参数，不能将全部增量归为纯连续性。相对无Rotation Baseline多960参数。
- 新增4模型，各12000步、每ratio3000，seed3100/3101；复用上一轮相同协议的4个Baseline/Quaternion模型，保留全部结果，不选最好参考。

| 对照 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR | 训练ratio | 未见ratio |
| --- | --- | --- | --- | --- | --- |
| rotation_all − baseline | -0.234 | -0.273 | -0.253 | -0.374 | -0.092 |
| rot6 − baseline | -0.015 | -0.130 | -0.072 | -0.138 | +0.015 |
| cov6 − baseline | -0.223 | -0.656 | -0.439 | -0.505 | -0.352 |
| rot6 − rotation_all | +0.219 | +0.143 | +0.181 | +0.237 | +0.107 |
| cov6 − rotation_all | +0.011 | -0.383 | -0.186 | -0.131 | -0.260 |
| cov6 − rot6 | -0.208 | -0.526 | -0.367 | -0.367 | -0.367 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| rotation_all | 16.99088 | 0.59552 | 0.60062 | 0.15521 |
| rot6 | 17.17202 | 0.59809 | 0.60048 | 0.15231 |
| cov6 | 16.80491 | 0.58866 | 0.61128 | 0.15866 |

## 6. 证据边界

- Master、拓扑、Anchor关联、相机、cached uniform冻结。Selector/Encoder仍训练，集合可变；same-K按logit(p)-logit(cached uniform)排序，只保证预算一致。原Cached实际点数另存。
- 相同54 Bank训练、13内部留出、45历史non-Bank。Bank字段为67视图（包含13留出）；最终12000步，未按non-Bank选LR、步数或checkpoint。
- 高斯椭球局部轴符号对称性被Cov6输入消除；原始Scale向量仍保留，不能宣称整个网络已对所有轴置换不变。
- 两个seed共享一张冻结地图，不能当独立SLAM重复或跨场景稳定结论。GPU有其他任务，耗时不作无干扰性能结论。
- 输入几何问题、优化状态与选点支持集共同适配可以并存；负cosine、交叉渲染损失或表示连续性本身均不能单独证明主因。

## 参考依据

- [Zhou等：旋转表示的连续性](https://arxiv.org/abs/1812.07035)：支持测试6D坐标的数学动机，不证明本项目PSNR必然提高。
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)：Gaussian形状由协方差描述；本项目实际约定另由CUDA源码和预计算协方差渲染gate核对。

![质量曲线](tests/diagnostics/rotation_input_diagnosis_20261026/evidence/analysis/quality_curves.png)

## 本地交付核验

4个新增模型与4个原始参考完整保存，全部12000步、每ratio3000曝光及Adam时钟一致；原型源码与归档SHA一致，逐视图四指标finite，Git HEAD、全部tracked diff和staged diff与实验前一致。没有修改生产默认或执行stage/commit/push。
