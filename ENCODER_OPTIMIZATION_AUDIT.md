# 固定11维输入的共享Encoder优化实验

## 结论

- shallow32相对当前16瓶颈：七档平均+0.485 dB（+0.718/+0.252），两seed均提高；训练ratio+0.528、未见ratio+0.428 dB。
- deep32相对当前16瓶颈：七档平均+0.487 dB（+0.641/+0.332），两seed均提高；训练ratio+0.473、未见ratio+0.505 dB。
- residual32相对当前16瓶颈：七档平均+0.291 dB（+0.530/+0.053），两seed均提高；训练ratio+0.215、未见ratio+0.394 dB。

**本轮优先建议使用11→32 ReLU作为下一版实验Encoder候选。** 浅32与深32的same-K平均PSNR仅差0.002 dB、两个seed方向相反；浅32的LPIPS和photo略好，1%档相对当前16瓶颈提高0.779 dB，而深32在该档平均下降0.057 dB。深32未见ratio相对浅32平均提高0.077 dB，可保留为泛化对照，但不足以据此认定全面更好。
残差版相对浅32平均下降0.194 dB，两个seed同向；函数保持初始化和额外层没有带来本轮总体收益，不推荐替换浅32。
三个候选虽然都改善了当前16瓶颈结构，但仍低于输入含SH DC的早期原Baseline；浅32/深32的剩余差距约0.486/0.485 dB。这个系统参照还混合输入信息差异，不能把剩余差距全算到Encoder。
这些结果只用于本轮探索性比较。无论候选均值高低，不能据此确认跨场景最优网络；若进入生产，仍需独立场景和真实在线Joint验证。当前生产默认保留。


本轮只改变当前11维输入之后的共享Encoder。输入、四Head、Anchor/ratio latent16、Loss、采样、ratio顺序和训练曝光保持一致；生产默认未替换。

## 结构与问题

| 方案 | Encoder | 参数量 | 检验问题 |
| --- | --- | --- | --- |
| bottleneck16 | 11→16 ReLU→32 ReLU→32 ReLU | 1792 | 上一轮当前参照 |
| shallow32 | 11→32 ReLU | 384 | 去掉16瓶颈和额外深度 |
| deep32 | 11→32 ReLU→32 ReLU→32 ReLU | 2496 | 保留深度但去掉16瓶颈 |
| residual32 | h=ReLU(W0u+b0); y=ReLU(h+W2 ReLU(W1h+b1)+b2) | 2496 | 同参数深度的残差连接与函数保持起点 |

- 原始输入固定为全局XYZ3、activated Scale3、sigmoid Opacity1、局部Offset3、logratio1；没有新特征或归一化。补偿输入raw11+Anchor/ratio latent16+h32=59，四Head仍各59→32→32→3/3/4/4。
- 三个候选共同首层取同seed上一组8→32的32行公共权重，新增Offset列0；Selector/Head/latent所有共同权重exact相同。deep32后两层固定seed初始化，residual32末层W2/b2为0，因此step0隐藏函数exact等于shallow32。
- 所有方案初始选择概率和零补偿exact一致；当前16瓶颈与新32首层不保证隐藏函数相同，不能称完全隔离初始化因素。
- 残差末层为0时inner层首个梯度可以为0；真实零补偿初始状态的4次photo+预算反传验证其随后收到非零梯度，不将零初始化的正常延迟误判为失效。

## 相对当前16瓶颈方案的non-Bank same-K结果

| 候选 | seed3100 ΔPSNR | seed3101 ΔPSNR | 七档均值 | 训练ratio | 未见ratio |
| --- | --- | --- | --- | --- | --- |
| shallow32 | +0.718 | +0.252 | +0.485 | +0.528 | +0.428 |
| deep32 | +0.641 | +0.332 | +0.487 | +0.473 | +0.505 |
| residual32 | +0.530 | +0.053 | +0.291 | +0.215 | +0.394 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| original_baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| bottleneck16 | 16.27296 | 0.58122 | 0.63618 | 0.16609 |
| shallow32 | 16.75815 | 0.58916 | 0.60361 | 0.15902 |
| deep32 | 16.75976 | 0.58909 | 0.60555 | 0.15945 |
| residual32 | 16.56445 | 0.59026 | 0.60873 | 0.15988 |

## 候选之间的控制

| 比较 | seed3100 ΔPSNR | seed3101 ΔPSNR | 七档均值 | 训练ratio | 未见ratio |
| --- | --- | --- | --- | --- | --- |
| deep32 − shallow32 | -0.077 | +0.080 | +0.002 | -0.055 | +0.077 |
| residual32 − shallow32 | -0.188 | -0.199 | -0.194 | -0.313 | -0.034 |
| residual32 − deep32 | -0.111 | -0.280 | -0.195 | -0.259 | -0.111 |

## 各ratio相对当前参照

| ratio | K | shallow32 | deep32 | residual32 |
| --- | --- | --- | --- | --- |
| 1% | 68 | +0.779 | -0.057 | +0.367 |
| 2.5% | 170 | +0.779 | +0.477 | +0.938 |
| 5% | 341 | +0.846 | +1.061 | +0.659 |
| 7.5% | 512 | +0.177 | +0.520 | +0.193 |
| 10% | 683 | +0.271 | +0.484 | -0.076 |
| 12.5% | 854 | +0.329 | +0.518 | +0.052 |
| 15% | 1025 | +0.216 | +0.405 | -0.092 |

## Bank、留出及原Cached评价

| 候选 | Bank训练视图ΔPSNR | 内部留出ΔPSNR | non-Bank实际Cached ΔPSNR |
| --- | --- | --- | --- |
| shallow32 | +0.428 | +0.422 | +0.435 |
| deep32 | +0.409 | +0.396 | +0.515 |
| residual32 | +0.259 | +0.268 | +0.164 |

## 协议与证据边界

- 同源fr1_xyz冻结N6838/M2066地图，Master/拓扑/相机/Anchor关联/eligible/cached uniform保持；Encoder/Selector/四Head和当前ratio latent联合训练，因此选择身份可变。same-K保证预算，不等于固定集合的补偿因果。
- 预注册三候选×两seed，共6次完整训练，各12000步、每ratio3000；第二seed反转三臂执行顺序。复用当前16瓶颈与原始Baseline的4个最终模型，不选择最好复跑。
- 固定54 Bank训练320×240、13内部留出和45历史non-Bank评价640×480；训练1/5/10/15%，评价另含2.5/7.5/12.5%；latent插值规则保留。
- fresh fused Adam lr=.001 eps1e-15，原photo=.8L1+.2(1-fast_ssim)加.01相对预算L1；Encoder/Selector各12000更新、每ratio3000次训练曝光，未活跃表grad=None；没有零梯度伪造步数。
- deep32/seed3100在1%档有2次Cached空集合，该两步仅预算分支有实际优化路径：补偿Head实际更新11998次、1% latent表2998次，其他表3000次。其余所有模型无空集合，补偿Head12000次、各表3000次。实际Adam时钟逐参数与空集合记录精确核对；相同训练曝光不等于所有辅助参数的更新次数绝对相同。
- 统一最终12000步评价，保存4k/8k/12k曲线，未依据non-Bank选择LR、step、模型或追加候选。有限步数差距包含优化与初始化，不能称容量极限。
- 原始Baseline使用局部Offset+Scale+SH DC+Opacity+logratio及浅Encoder，输入信息与本轮不同，只是系统效果参照，不能将其差值全归因于Encoder。
- 两seed是同一地图的网络初始化和Bank顺序重复，不是独立SLAM地图。历史non-Bank已多次用于结构设计；本轮仅探索性结果，不作统计显著、跨场景或在线Joint质量保证。
- deep32和residual32参数量相同，但连接方式、末层初始化和优化条件不同；对照检验整套结构，不能将差值只归为单一梯度性质。
- GPU有其他任务，耗时仅作运行记录；新原型不接入生产，不stage/commit/push。
- 训练预检查发现checkpoint继承的encoder_layout描述需更新；该预检查运行中止并保留metadata_preflight，本轮全部6个正式模型从step0重训，未用预检查质量做选择。

![质量曲线](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/quality_curves.png)

## 共享编码的激活情况

| 方案 | 平均零激活比例 | 完全为零通道数范围 | 编码RMS均值 |
| --- | --- | --- | --- |
| bottleneck16 | 0.493 | 4～8 | 0.535 |
| shallow32 | 0.548 | 6～9 | 0.411 |
| deep32 | 0.547 | 5～8 | 0.683 |
| residual32 | 0.534 | 5～11 | 0.673 |

这些是最终checkpoint全部Gaussian上的数值统计，不以激活更稠密或编码幅度更大直接认定渲染更好。

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| bottleneck16 | 3100 | 16.249 | 16.606 | 16.242 |
| bottleneck16 | 3101 | 16.254 | 16.516 | 16.792 |
| shallow32 | 3100 | 16.702 | 17.036 | 16.987 |
| shallow32 | 3101 | 16.329 | 16.604 | 16.892 |
| deep32 | 3100 | 16.215 | 16.778 | 16.863 |
| deep32 | 3101 | 16.433 | 17.130 | 16.964 |
| residual32 | 3100 | 16.214 | 16.801 | 16.869 |
| residual32 | 3101 | 16.541 | 16.484 | 16.701 |

全部方案统一最终12000步，曲线不用于non-Bank选step，也不把有限训练预算下的差距解释为理论容量极限。

## 交付核验

6个新增模型与4个锁定参考完整保存；全部12000步、每ratio3000曝光，实际Adam时钟与非空选择步数精确匹配，deep32/seed3100少2次补偿更新的差异已报告。残差版初始隐藏函数与浅Encoder exact一致。原型源码与服务器归档SHA一致，所有逐视图指标finite。Git HEAD、全部tracked diff及staged diff与实验前相同，已有17个tracked改动保留；未stage、commit或push。

[逐视图质量](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/paired_statistics.csv) · [共享编码统计](tests/diagnostics/s00_encoder_optimization_20261026/evidence/analysis/encoder_output_statistics.csv)
