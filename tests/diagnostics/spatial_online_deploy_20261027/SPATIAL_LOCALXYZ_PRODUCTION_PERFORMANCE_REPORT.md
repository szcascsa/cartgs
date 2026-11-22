# SpatialLocalXYZ生产默认接入与实际提速报告

## 1. 交付结论

生产默认已从S00R00E50D12改为 **SpatialLocalXYZ在线适配版**，本地与服务器`/home/prmi/zwj/mycartgs`一致。服务器复用`build-performance-20261023-121515`，没有隔离构建；4395个清单文件SHA核对通过。已重编tum_rgbd/tum_mono、Replica RGBD/Mono、VECtor Mono/Stereo、view_result和train_colmap，以及针对性的数值与Mapping检查。旧源码与运行产物保存在`/home/prmi/zwj/spatial-online-deploy-20261027/backup`。

当前基于本地commit `74fc1524d8d6345eda8e732a1da2efa9a4934951`加本轮工作区改动；本轮未stage、commit或push。服务器没有可用Git版本记录，具体版本由final_source_manifest.json和二进制SHA定义，不能将它称为单纯的旧commit部署。

相同fr1冻结地图、相同历史非零SpatialLocalXYZ权重、相同K，实时重算属性且对Gaussian输入反传，三块交错实测：

| 计算范围 | 原生参考ms | 完整融合ms | 加速 |
| --- | --- | --- | --- |
| forward | 0.852797 | 0.312369 | 2.7301 |
| forward_backward | 3.75218 | 1.50489 | 2.49332 |

这是**选中K行补偿网络及实时输入构造**的严格对照，F+B不含Renderer、Loss、Adam；不是完整Mapping倍数。原Frozen-only Graph约2.67倍整步骤结果不用于宣称在线收益。

## 2. 真实生产结构与梯度

- Variable Anchor及原增删、pruning、slot eligibility、Cached Bernoulli、ratio轮转和重建Loss保留。
- Selector保持原S00R00输入：Offset3、activated Scale3、SH DC3、activated Opacity1、log ratio1，11→32→32→2。Selector不读取新补偿特征，生产没有新增Anchor/ratio补偿表。
- Compensation独立共享55→64→64；连续log-ratio经1→32→256预测两层FiLM；XYZ/Scale/Rotation/Other四Head64→64→3/3/4/4。
- 19D物理输入顺序为归一化全局XYZ3、asinh局部Offset3、log Scale3、SH DC3、opacity logit1、trace-normalized covariance6；再接36D六频带固定Fourier，55维。Quaternion转协方差的算法和14D属性含义保持原SpatialLocalXYZ。
- XYZ补偿乘所归属Anchor参考尺度；Scale、Quaternion、Opacity和DC继续原14D解码。补偿仅计算实际选中的K点。
- 归一化world median/spread与19D mean/sample-std在首次启用补偿时，从**完整当前地图**计算并保存为buffer，此后固定；N=1的在线边界std设为0.1。新观测、clone/split、prune或闭环后读当前Gaussian/owner/Anchor参考值，不缓存旧属性，不增加身份参数或迁移表。
- LoD保留Master与辅助网络joint梯度，Full只更新Master；TF、SI/GI、ChangeDistill维持默认关闭。没有引入新的Loss、蒸馏、预算或强制10000步。

补偿参数33,806，FP32权重135,224 bytes；若Adam双Moment均建立为270,448 bytes，梯度135,224 bytes。不随N或Anchor数增长。仅保存人口统计buffer，没有Gaussian/Anchor私有训练补偿自由度。

## 3. 基于真实Profile的优化

首轮在线适配保留原生physical构造，仅融合标准化/Fourier与FiLM；真实运行中启用补偿后的LoD约7.08ms，其中补偿前向1.20ms、LoD反向4.08ms、Renderer约0.35ms。该结果表明主要目标应是实时属性构造和反传调度。

据此新增`spatial_compensation_live.cu`：每CUDA线程处理一个选中Gaussian，直接完成world归一化、asinh Offset、log Scale、Quaternion→Rotation→Covariance、trace归一化、19D标准化/clamp及36D Fourier，写K×55。保存K×38 FP32中间量，主要用于反向，约152K bytes。单一Backward核计算XYZ/Scale/Quaternion/DC/Opacity、Anchor中心与参考尺度VJP，再由原Gather/激活Autograd回写Master。所有统计buffer固定，不制造梯度。

FiLM前后向融合；gain/bias仍用LibTorch sum聚合全部K行。cuBLAS主干/四Head权重独立，复用已有decode14、dense Adam、Selector压缩、Renderer workspace与SSIM优化。新CUDA使用FP32、`--fmad=false`、标准sinf/cosf/asinhf/logf，没有fast math/TF32或精度替换。

默认`CUDA.fused_spatial_compensation: 1`。设为0回退相同Spatial数学的LibTorch输入及FiLM，便于受控对照。生产动态Joint不使用Frozen55D缓存，不将Master属性detach，不为Graph固定实际选点K。

## 4. 逐K严格性能

| ratio | K | 范围 | 原生参考ms | 完整融合ms | 倍数 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 68 | forward | 0.827392 | 0.298667 | 2.77029 |
| 0.01 | 68 | forward_backward | 3.76013 | 1.50221 | 2.50307 |
| 0.025 | 170 | forward | 0.834219 | 0.29696 | 2.8092 |
| 0.025 | 170 | forward_backward | 3.77242 | 1.50187 | 2.51182 |
| 0.05 | 341 | forward | 0.844117 | 0.297984 | 2.83276 |
| 0.05 | 341 | forward_backward | 3.76149 | 1.49982 | 2.50796 |
| 0.075 | 512 | forward | 0.862208 | 0.318464 | 2.7074 |
| 0.075 | 512 | forward_backward | 3.74375 | 1.50255 | 2.49159 |
| 0.1 | 683 | forward | 0.860501 | 0.319147 | 2.69626 |
| 0.1 | 683 | forward_backward | 3.71712 | 1.49572 | 2.48517 |
| 0.125 | 854 | forward | 0.863573 | 0.32768 | 2.63542 |
| 0.125 | 854 | forward_backward | 3.75125 | 1.51074 | 2.48306 |
| 0.15 | 1025 | forward | 0.877568 | 0.32768 | 2.67813 |
| 0.15 | 1025 | forward_backward | 3.7591 | 1.52132 | 2.47094 |

Forward包含实时K点physical/Fourier、网络及Anchor单位输出；F+B增加完整Gaussian属性leaf与网络参数反传。原参考与优化从相同非零权重读取，不在计时中更新参数；三块交错、每块35次、10次预热、CUDA Events，保留host提交时间与每样本原始CSV。统计为块内median再跨块mean，七ratio等权。

历史原型在Frozen Master中将19D预先detach，不能直接用其0.7ms/2ms的不同范围来比较当前实时Joint路径。

## 5. 数值、Adam与接口验证

新完整输入融合的随机VJP检查覆盖K0/1/17/68/1025，训练1/5/10/15%与未见2.5/7.5/12.5%，全部网络参数及七种属性输入；单独对55D输入反传，避免只测网络路径时遗漏属性导数。

累计2153项逐张量检查通过。最大绝对误差7.15256e-07，最大张量RMS为6.89037e-08；原阈值Forward atol1e-6/rtol1e-5、Backward atol1e-5/rtol1e-4，未放宽。逐张量数据在live_adam_validation/parity.csv。12步轮转Adam从相同模型和相同新建零状态开始，lr=.001、生产LibTorch默认eps，逐参数、exp_avg、exp_avg_sq与时钟比较在该CSV，使用同一VJP隔离Renderer非确定性。这不是恢复历史在线Adam状态的续训；原生产PLY没有该完整状态。

同一fr1历史非零终点的七档K输出与原PyTorch原型比较，最大绝对误差9.53674e-07，全部通过。该差异来自FP32运算/归约次序，不能宣称bitwise。

生产Mapping检查使用真实Mapper、Renderer和shared Adam：LoD更新Master与Spatial Head，Full不更新辅助参数或其Adam状态，原五槽轮转及online/tail更新数保持。checkpoint中保存新schema、共享权重和归一化buffer，C++/Python读取已验证，Python补偿属性全部有限。新旧E50/Spatial的模型权重结构不同，应由checkpoint schema加载，不能把修改配置当作旧模型权重转换。

## 6. 完整fr1_xyz SLAM观察

两次都使用生产自然停止规则、seed2026、完整原数据、相同Loss/ratio/Joint。第一轮是部分融合在线适配，第二轮是完整实时输入融合。启用非同步sampled CUDA Events及host trace，iteration logging仍为0；没有强制Mapping步数。

| 版本 | Mapping步数 | 补偿LoD步数 | 最终Gaussian | 最终Anchor | 全应用秒 | 全Mapping host均值ms | sampled GPU区间ms | 启用补偿LoD host均值ms | GPU峰值原始记录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_spatial | 6900 | 3120 | 7596 | 2426 | 43.1151 | 4.80906 | 4.8758 | 7.08383 | Peak reserved (MB): 572<br>Peak allocated (MB): 487.697 |
| online_live_fused | 8400 | 4320 | 8345 | 2281 | 42.2368 | 3.85107 | 3.9141 | 4.7552 | Peak reserved (MB): 614<br>Peak allocated (MB): 518.329 |

两个运行都成功保存checkpoint。最终版在相同数据播放期间完成了更多Mapping步，因此总墙钟仅小幅减少。步数6900与8400、地图/相机轨迹不同；4.81→3.85ms、7.08→4.76ms属于**实际在线观察**，不能当严格相同地图与计算曝光的因果倍数，也不能归因为质量提升。严格局部加速来自上一节同图/K/权重对照。两次GPU峰值也受Gaussian/Keyframe数量和生命周期影响，不能把约30.6MiB的allocated差异直接认定为新Kernel额外成本。

当前剩余热点仍是LoD backward（含Renderer与网络反传）、共享小GEMM与Autograd调度；Full和Selector耗时变化小，符合本轮只优化补偿的范围。下一阶段应优先测量**实时55D图外＋共享64网络分桶/静态shape调度**，完整保留K与所有梯度并单独摊销capture；其次才是Renderer动态计数与分配。暂不建议通过降低训练次数、detach Master、缓存过期特征、减少网络或修改loss换速度。

## 7. 范围与回退

**IMPLEMENTED / TESTED**：原生C++/CUDA生产默认、实时K行输入与完整VJP、原Adam与Full/LoD路由、Python导出读取、两次fr1实际SLAM及同图局部性能、服务器源码SHA与常用入口重编。

**PARTIALLY VERIFIED**：无需私有表的append/prune/reorder属性读取；实际SLAM增长与checkpoint可用，但未专门触发所有闭环碰撞事件。固定统计在大范围地图扩张后的质量需要独立实验。

**NOT IMPLEMENTED**：生产训练CUDA Graph、自动Frozen缓存分派、动态K分桶、生产并发Graph、多场景质量/ATE回归、严格匹配完整在线轨迹的性能因果对照。本轮没有评估PSNR/SSIM/LPIPS提升。

回退CUDA：只将CUDA.fused_spatial_compensation设0，保留同一训练结构和checkpoint。回退E50算法需显式选择旧预设与对应checkpoint，或从头训练。服务器备份包含旧一方源码及常用运行产物；需要整体回滚时使用同一备份的源码和二进制，避免ABI混用。
