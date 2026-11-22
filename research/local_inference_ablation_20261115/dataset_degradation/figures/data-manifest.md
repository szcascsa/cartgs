# 局部消融退化图数据清单

| 图 | 数据文件 | 数据性质 | 来源 | 脚本 | 输出 |
|---|---|---|---|---|---|
| TUM三联雷达图 | data/DATASET_DEGRADATION.csv | 真实实测 | LOCAL_ABLATION_COMPARISON.csv | plot_dataset_degradation.py | PNG／SVG／PDF |
| REPLICA三联雷达图 | data/DATASET_DEGRADATION.csv | 真实实测 | LOCAL_ABLATION_COMPARISON.csv | plot_dataset_degradation.py | PNG／SVG／PDF |
| VECTOR三联雷达图 | data/DATASET_DEGRADATION.csv | 真实实测 | LOCAL_ABLATION_COMPARISON.csv | plot_dataset_degradation.py | PNG／SVG／PDF |

PSNR与SSIM的退化量取完整方案减消融，LPIPS取消融减完整方案。先对各场景的两种相机模式等权平均，再对场景等权平均。
