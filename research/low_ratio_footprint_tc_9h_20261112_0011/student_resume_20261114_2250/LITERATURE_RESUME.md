# 本轮资料检索与源码对应

检索时间：2026-11-15，接续已锁定的9小时预算。论文和博客的速度数字不作为本项目证据。

- [Local-GS](https://arxiv.org/html/2606.16566v1)：Tile局部二次式提升、warp一致执行和共享混合权重。当前历史mode38已包含前两类机制的一部分。本轮独立测量去掉冗余极小T判断、`w=αT; T=T-w`及warp像素布局，分别保留开关和同二进制控制。
- [Kopanas的CUDA优化记录](https://grgkopanas.github.io/3dgs_cuda_blog/index.html)：重点阅读两行像素映射、T符号状态和负结果。A100的收益不直接外推到RTX3090。该记录也说明半精度转换、过多独立像素、额外warp缓存可能抵消算术收益，与本轮失败候选一致。
- [TileGS](https://arxiv.org/html/2609.03613v1)：重点是缩短有效遍历，而非单纯追求占用率。本项目已使用深度有序短列表；低Ratio溢出极少，暂不为减少溢出增加分桶及修复阶段。
- [Color-Aware Early Termination](https://www.saiqianzhang.com/gallery/VisArch26_asplos.pdf)：残余颜色误差必须与提前退出成本一起评价。本轮中心尾色扫描、归一化补尾均没有得到稳定且合格的完整收益，保留负结果。
- [MatrixSplat](https://arxiv.org/html/2611.15153v1)：其无序加权累积改变合成定义，本轮未移植。
- [CUDA半精度数学接口](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html)：half2状态候选已实现；Alpha/T打包后的完整渲染更慢，不能用指令理论吞吐替代实测。

本轮新候选保持Gaussian身份、严格K、输入属性、相机、背景与分辨率不变。T递推重组、bit-exp线性变换提升、半精度与早停均按近似候选记录，不称逐位一致优化。
