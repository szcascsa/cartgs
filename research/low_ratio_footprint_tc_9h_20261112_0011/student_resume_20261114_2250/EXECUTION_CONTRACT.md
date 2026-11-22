# 冻结推理执行边界

用户最新明确重置预算：2026-11-15 02:44:06 至 11:44:06（香港时间），10:44:06 起预留最终验收。旧窗口与重置依据保存在 `reports/BUDGET_RESET_20261115_0244.json`，不覆盖历史测量。目录独立于生产，不自动部署。

## 当前可回退实现

- 原FP32 Selector＋同地图FP32 Student＋冻结F2。
- 历史Tensor Core Student与准备调度＋历史mode38。
- 本轮已冻结的182像素核，独立加载，不覆盖上述实现。

## 每帧重算与静态空间

|内容|依赖|每帧处理|失效条件|
|---|---|---|---|
|当前相机View/Projection与Anchor可见性|当前CPU位姿、相机内参|重算|每帧更新，禁止复用旧Mask|
|Selector动态部分|当前Ratio/View、冻结物理输入与权重|重算|Ratio或View变化立即生效|
|Student冻结输入投影|冻结地图、Base/Residual表、权重、拓扑|只缓存线性静态项|任一依赖变化需重建；不跨ReLU|
|Student动态投影|当前Ratio和View|重算|每帧重算|
|严格选择N/K|当前候选与预算|GPU计算|不读取上一帧结果|
|相机35个FP32值的pinned/device缓冲|实例、设备、Stream|覆盖本帧值|同实例跨Stream拒绝；上一帧已完成后才写|
|Tile位表、列表、计数、排序Workspace|容量、分辨率、设备、Stream|每帧清零/重建|扩容或分辨率变化使旧图计划失效|
|Graph计划|容量/分辨率/执行配置|更新当前输入指针后执行|不缓存图像、选中ID或旧预测|
|轻量Host入口|零元素Tensor、绑定函数、相机FoV标量|当前相机矩阵和所有Gaussian属性仍重新提交|FoV变化重算tan；原生入口检查设备与Stream|

共享Workspace仍有1 GiB上限，位表256 MiB上限；具体逐场景分配由计时外的memory CSV报告。182像素核不增加额外全局Workspace，静态投影显存不计作本轮渲染新增空间。

## 数值分类

182保留身份、严格K、排序及成员列表；radii逐位相同。bit-exp、Tile局部多项式和0.01早停是历史近似；本轮提升bit-exp线性部分、变更像素布局、T符号和T-weight递推也按近似记录，不声称相对F2的RGB逐位一致。

相机缓冲、radix收尾、实际K成员处理及Host入口应相对同一网络后端、同一像素核保持ID/RGB/radii逐位一致。最终组合的近似误差必须单独对比FP32 Student＋F2。

Student网络深度/宽度、四个独立Head、Feature表、参数和Selector权重全部不变。不调用Teacher前向、不启动训练或蒸馏。没有匹配Student的14组地图只做Full属性的固定嵌套删点，不替换为Teacher LoD。
