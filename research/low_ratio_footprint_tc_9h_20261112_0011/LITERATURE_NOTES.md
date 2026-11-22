# 本轮资料与源码核对

检索于2026-11-12。下列作者报告的速度不是当前F2的实测，不作为本轮收益。

|资料|可用机制|本轮应用与边界|
|---|---|---|
|[TC-GS](https://arxiv.org/html/2505.24796v1)；[官方源码](https://github.com/DeepLink-org/3DGSTensorCore)|6项局部多项式映射MMA；Tile局部坐标减少FP16二次项误差|可独立探索Tensor Core幂指数计算，保留FP32颜色/T串行递推；需包含矩阵填充与布局成本|
|[RoofGS](https://arxiv.org/html/2608.15785v1)|双像素增量、多项式系数折叠、范围受限exp近似|本轮15–27模式自行按现有co.y交叉项推导，避免照搬论文b系数定义；Tile局部坐标减少抵消；快速bit/polynomial exp单独开关|
|[HiGS](https://research.nvidia.com/labs/sil/projects/higs/)|分区与渲染采用两种Tile尺度，局部batch共享和负载调度|目标不只是旧宏Tile属性共享；需要重构成员构建边界，不能重复前轮已失败的并集宏Tile实验|
|[TileGS](https://arxiv.org/html/2609.03613v1)|Tile内更短深度范围和前到后遍历；强调减少有效指令而非仅提高occupancy|当前F2已经全局深度稳定排序和成员列表；新收益应来自遮挡后不再建立/装载后方成员，不能只重复分桶|
|[Speedy-Splat](https://speedysplat.github.io/)|SnugBox/AccuTile缩小保守分配范围|F2已有精确Tile覆盖，前轮子块仅筛2.51%/1.21%，本轮不再盲目投入相同小区域筛除|
|[FlashGS](https://maxwellf1.github.io/flashgs_page/)|冗余消除、流水线、访存及调度|前轮Alpha流水和shared/宏Tile已否决；本轮仅在新依赖/近似许可改变条件时重新评估|

源码下载情况：2026-11-12 GitHub Tree API返回403 rate limit；未据此声称已获取或运行官方TC-GS。论文HTML可读。后续如采用其代码需另记文件与哈希。

## 当前自行推导的局部多项式

令当前Tile中心为c，局部像素q=p−c，Gaussian局部中心g=μ−c。原F2二次型为

`power = −.5 a(gx−qx)^2 − b(gx−qx)(gy−qy) −.5 c(gy−qy)^2`。

因此折叠后六项为：`[−.5a,−b,−.5c, a gx+b gy, c gy+b gx, −.5(a gx²+c gy²)−b gx gy+log(opacity)]`，与`[qx²,qx qy,qy²,qx,qy,1]`点积。

第二像素y相差8：`P(x,y+8)=P(x,y)+8(−b x−.5c(2y+8)+c gy+b gx)`。

该代数重排会改变FP32舍入，归类近似，不能称bitwise。并且log-opacity对非正/非有限值的旧语义必须在边界检查中决定回退，不能默认只测正常地图。
