# 工作量数据清单

全部为真实计数，无模拟、插值或平滑。

|文件|内容|
|---|---|
|PER_FRAME_WORKLOAD.csv.gz|32组、有效全轨迹、Complete Map及11档ratio；原始计数、同视角基准、归一化工作量、历史三轮中位耗时、选点ID哈希|
|SCENE_RATIO_WORKLOAD.csv|每场景/相机模式/ratio的原始均值、总量、逐帧归一化均值与分位数|
|SIX_CURVE_DATA.csv|后续六条曲线的数据，同数据集同模式内场景等权|
|CORRELATIONS.csv|逐场景、逐ratio及混合ratio的工作量–耗时描述性相关性|
|CORRELATION_SUMMARY.csv|各相关性口径的汇总，不作因果推断|
|COVERAGE_VALIDATION.csv / VALIDATION.json|逐场景验收、覆盖、保护文件和来源哈希|

计数实现：build_diagnostic.py生成独立CUDA扩展；采集：collect.py；汇总：analyze.py。各次渲染使用冻结Student、名义Anchor局部候选池和同一mode182配置；不训练、不保存渲染图片。基准名称：Our Complete Map (1×)。
