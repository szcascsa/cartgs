# Quest 3像素负载模拟：office2 CaRtGS与Tensor＋新准备D

## 结论

RTX3090上，每个显示帧独立渲染两张2064×2208图像（共9,114,624像素），原版CaRtGS双眼纯渲染中位8.439ms、双眼准备0.840ms、双眼完整9.234ms。我们D LoD各Ratio的双眼纯渲染0.673–1.351ms，双眼准备0.392–0.422ms，双眼完整1.010–1.723ms。

这里是Quest 3每眼像素尺寸及两次调用的负载模拟，不是Quest 3真机性能。每显示帧两次使用同一当前位姿，分别完整准备和渲染，不设眼间距，不使用OpenXR multiview/单次双眼渲染、注视点降采样或时间扭曲。保留Replica原FoV，不将其称为Quest真实光学视野或立体视觉质量验证。

## 分离计时结果

单位ms；左/右是同位姿两次独立调用编号，不代表真实左右眼视差。所有列分别直接实测，不通过单眼×2外推。准备及完整墙钟和纯渲染Event是独立采样，中位数不能机械相加。

| 方法 | 地图N | 每眼实际K中位数 | 双眼准备 | 第1眼纯渲染 | 第2眼纯渲染 | 双眼纯渲染 | 双眼完整 | 纯渲染倍速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CaRtGS原版Full | 84203 | 84203 | 0.8401 | 4.2486 | 4.1902 | 8.4393 | 9.2342 | 1.000× |
| 我们D Full | 43197 | 43197 | 0.1346 | 0.9206 | 0.8899 | 1.8094 | 1.8836 | 4.664× |
| 我们D 1% | 43197 | 154 | 0.3936 | 0.3446 | 0.3277 | 0.6734 | 1.0097 | 12.533× |
| 我们D 5% | 43197 | 771.5 | 0.3915 | 0.4541 | 0.4383 | 0.8924 | 1.2351 | 9.457× |
| 我们D 10% | 43197 | 1543 | 0.4015 | 0.5133 | 0.4956 | 1.0094 | 1.3669 | 8.361× |
| 我们D 30% | 43197 | 4629.5 | 0.4047 | 0.6151 | 0.5888 | 1.2047 | 1.5668 | 7.005× |
| 我们D 70% | 43197 | 10804 | 0.4223 | 0.6908 | 0.6589 | 1.3506 | 1.7231 | 6.249× |

## 对照身份与口径

- CaRtGS为已有归档Vanilla算法（项目关闭Selector/Compensator等扩展），Replica office2 RGB-D三遍中Full全轨迹PSNR最高的repeat3，原生历史PSNR34.0468dB，地图84,203点。该记录用于确定既有对照，不代表本次高分辨率质量实测。checkpoint为`/home/prmi/zwj/realLOD/vanilla/vanilla_all_datasets_20261031_2011/jobs/replica/rgbd/office2/repeat3/seed0/attempt_1/model/15900_shutdown/ply`。不将其声称为新克隆、未修改上游仓库训练结果。
- 我们仅用最终D：同一3534参数Student＋GPU Selector、FP16 Tensor网络/FP32累加、FP32静态预投影、CUDA视锥、GPU动态N/K、C++相机、新G Renderer。Master地图43,197点，checkpoint为`/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107/runs/Replica/office2/rgbd/attempt_01/model/41340_shutdown/ply`，学生为`/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models/Replica__office2__rgbd/continuous_calibration/best.pt`。不加入A/C或其他网络对照。
- 原版渲染器是此前原始inference路径，扩展SHA256为`27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86`；不是新G或balanced渲染器。两种地图来自同场景的不同训练结果，因此收益同时包含地图规模、选择补偿、渲染器和准备优化，不能全部归于Tensor Core或减少点数。
- 同一组18个RGB时间戳（12连续＋6跳转）；两张地图分别使用自己SLAM保存的对应位姿，避免错误跨地图坐标系套用相机。相同2064×2208、同背景、同SH阶数/视野语义。两图像仍可能因地图、位姿及补偿不同而有不同footprint和渲染复杂度，不是固定同属性的算子对照。
- 双眼准备墙钟：分别构造两次当前相机和属性；原版每眼计算物理属性激活/打包，我们每眼独立重新运行候选、选择、两网络、解码/打包。Full Master本身允许GPU常驻；我们Full包加载时构造一次，LoD不复用前一眼的Mask/网络结果。
- 双眼纯渲染：先在计时外完成两眼各自准备，然后CUDA Event在两个Rasterizer调用前/间/后记录，返回每眼与合计。完全不含Tensor网络、相机准备或属性激活。Event区间可能包含光栅器内部发射与依赖造成的GPU时间线间隙，不是Nsight Kernel时间求和；另存raster_pair_wall。
- 双眼完整墙钟：眼0准备→眼0渲染→眼1准备→眼1渲染→同步，使用同一Stream顺序执行，两个输出同时持有。没有跨眼并发，没有单次multiview优化，也没有用单眼中位数翻倍代替双眼实测。
- 原版CaRtGS的准备时间也包含原MiniCam的当前相机求逆/矩阵与物理属性激活；我们的D相机采用已验收新路径。原版两眼相机未被替换为D来掩盖完整路径差异。纯渲染比较则两侧均在相机/属性准备后开始。
- 不含模型读取、Graph首次构建、图片读取/编码/保存、HTTP、Viewer显示、OpenXR合成、镜片畸变、Tracking/Mapping；因此也不将倒数称为Quest实际显示FPS。
- FoV沿用Replica而非Quest真实镜片，不指定IPD。若需要真实左右眼不同位姿、非对称投影及头显GPU，需要独立协议；本轮只按用户指定的每帧两次像素负载测试。

## 验证与限制

- 三轮×8重复×18视角×7配置，每个配置每种计时432样本，总18144条计时。每眼实际K、候选M、原始p95/最大值、逐视角及逐轮保留，严格预算不改变。
- 252组眼/视角/配置检查输出尺寸3×2208×2064、finite，D各自属性包的新G与原Rasterizer RGB/radii逐位一致。同位姿独立两眼输出和所选ID相同，证明没有漏掉第二次渲染。原版与我们地图不同，未声称两方法图像逐位或PSNR等价。
- 37次低频监控，外部活跃GPU样本0，外部单进程>10% CPU样本0；不能排除所有瞬时调度。所有有效计时保留，无速度阈值筛样本。
- 3520个受保护源码/配置/模型/二进制前后SHA无变化；checkpoint只读，无训练、无JIT/编译、无生产配置修改、无共享conda修改、无commit/push，渲染图片保存0。

## 交付

服务器目录：`/home/prmi/zwj/experiments/quest3_office2_tensor_vs_cartgs_20261111`；分项表reports/QUEST3_OFFICE2_SEPARATED_RESULTS.csv，逐轮/逐视角和全部原始samples.csv，模型与渲染器身份见MODEL_IDENTITIES.json。
