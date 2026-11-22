# 直接核验的一手文献与本轮假设

以下是2026-10-27实际检索并读取的论文页面。论文原任务的收益不能视为当前LoD补偿的实验收益；本轮没有安装或复现完整第三方框架。

1. [Point Transformer，ICCV2021](https://arxiv.org/abs/2012.09164)：点集局部注意力与相对位置编码。启发是让保留Gaussian查询删除邻域，而非仅拼均值。本轮轻量双掩码attention为自有原型，不称原版复现。
2. [Point Transformer V2，NeurIPS2022](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html)：分组向量注意力、位置关系增强。先不引入完整层次分区，避免本轮只测试复杂度。
3. [FiLM，AAAI2018](https://arxiv.org/abs/1709.07871)：以条件输入产生中间特征的共享仿射调制。本轮条件仅log-ratio，所有Gaussian共用调制网络，无ratio查表。
4. [Fourier Features，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)：确定性频率映射可改变低维坐标MLP的频谱偏置。本轮只用固定正弦/余弦与共享Linear，不训练场景网格；这并不保证任意频率都改善图像泛化。

自提假设：局部相似属性可能无法区分不同位置的补偿需求；确定性空间编码增强可分性，而保留/删除交互补充缺失模式。两者通过独立与组合控制区分，不用私有latent替代学习困难。

5. [SIREN，NeurIPS2020作者项目页](https://www.vincentsitzmann.com/siren/)：周期隐式表示和配套初始化。本轮仅作小型自有连续/输出节点原型，1e-3与有依据的1e-4控制；没有声称完整复现或把文献收益当本项目结果。
