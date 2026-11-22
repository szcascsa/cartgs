# 三数据集局部Ratio四项消融图数据清单

| 图 | 数据文件 | 数据性质 | 来源 | 脚本 | 输出 |
|---|---|---|---|---|---|
| 三联图及三个独立图 | data/DATASET_PSNR_DECREASE.csv | 真实实测 | 推理消融、无GI／LI训练消融、Detach Master训练消融 | plot_three_ablations.py | 450 DPI PNG／SVG／PDF |

下降量为最新完整方案减消融。四条线统一使用原图27组共同有效集合，模式内平均后场景等权。Detach Master全部32组另见data/MASTER_DETACH_ALL32_PSNR_DECREASE.csv。
