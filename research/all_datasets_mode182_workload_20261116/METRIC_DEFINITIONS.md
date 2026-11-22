# 渲染工作量统计口径

本轮为真实执行路径计数，不是用高斯数量、包围盒面积或采样像素推算工作量。所有原始图像像素均参与统计。发布版与诊断版输出必须逐位一致；诊断耗时不用于报告渲染性能。

## 核心指标

|列名|定义|用途与限制|
|---|---|---|
|`effective_alpha_blends`|有效像素上的非零 `weight = T × alpha` 合成次数|描述有效合成工作；不是贡献权重的和，不含被拒绝候选的成本|
|`active_pixel_tests`|有效像素尚未结束时实际接受测试的 Gaussian–像素对数|包含低 Alpha、形状等条件拒绝的候选，覆盖更多真实计算|
|`scheduled_pixel_pair_slots`|循环实际执行的像素槽位，包含 Warp 尚未共同退出时已结束的像素及图像边界外槽位|描述执行浪费及并行尾部；不同分支的每槽位成本不同，不是指令数|
|`gaussian_tile_instances`|全部 Gaussian–Tile 成员关系数|反映投影 footprint 与成员构建工作；同一 Gaussian 可跨多个 Tile|
|`potential_valid_pixel_pairs`|对每个 Tile 计算成员数 × Tile 内有效像素数，再求和|忽略逐像素提前结束的潜在交互量，通常大于实际测试量|

有效 Alpha 合成量的数学定义为：

\[
W_{\mathrm{blend}}=\sum_{p\in\Omega}\sum_{i\in\mathcal E(p)}\mathbf 1[T_i(p)\alpha_i(p)>0],
\]

其中 \(\mathcal E(p)\) 是当前 mode182 实际执行合成路径中的 Gaussian 顺序。已被提前终止跳过的交互不计为有效合成。各分支的阈值和回退逻辑沿用发布版，因此不能与不同渲染器的计数不加说明地混用。

## 辅助指标

- `K`：实际选出并提交渲染的高斯数。局部 ratio 预算遵循原实验名义 Anchor 候选池，`K=min(floor(ratio×M),eligible)`。
- `projected_gaussians`：`radii>0` 的点数，表示投影有效，不能称为实际参与 Alpha 合成的唯一高斯数。
- `covered_pixels`：最终透射率小于1的有效像素数。
- `terminated_pixels`：按当前执行路径的符号标记已结束的有效像素数。
- `fallback_active_tests`：进入原始安全/溢出回退路径的有效候选测试数。
- `overflow_tiles`：成员数超过当前固定列表容量的 Tile 数。
- `max_tile_gaussians`、`mean_tile_gaussians`、`std_tile_gaussians`：成员负载分布。总量相同也可能因负载不均而耗时不同。
- `*_per_pixel`：工作量除以图像有效像素数，便于比较不同分辨率。
- `ids_sha256`：选点顺序的哈希，与历史三轮测速逐条匹配，防止用不同选点解释速度。

## 后续 ratio 曲线

每帧先计算 `W(ratio)/W(Our Complete Map)`。同场景内对有效全轨迹取平均，再对同数据集、同相机模式的场景等权平均，得到六条曲线。

数据同时保留“逐帧归一化后平均”和“先求和再相除”，二者不能混称。前者与当前逐视角加速比曲线的统计单位一致，建议用作主图。基准名称统一为 **Our Complete Map (1×)**。

Alpha 贡献次数、候选测试数、Tile 展开量分别描述有效合成、测试计算和空间展开工作，不能互相替代。判断指标是否足够解释速度时，使用同场景、同分辨率、同视角的实测毫秒与相关性；不能只凭工作量降低比例推算加速比。
