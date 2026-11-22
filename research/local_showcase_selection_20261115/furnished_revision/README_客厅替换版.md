# 明亮且内容丰富的视角替换版

人工检查原图中的家具／物体覆盖，选择明亮且有沙发、茶几、花瓶、靠垫和落地灯的room0客厅；不声明全局质量最优。

Replica RGB-D和Mono均选room0的RGB索引950（从0计数），分别使用各自模型及原估计相机；TUM两组保持。
重新选择展示帧，不修改图像亮度、模型或训练；四档选点身份、K和三项质量指标与历史评价相同。

| 场景／模式 | RGB帧 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR |
|---|---:|---:|---:|---:|---:|
| Replica/room0/rgbd | 950 | 20.75 | 24.40 | 25.74 | 28.06 |
| Replica/room0/mono | 950 | 19.56 | 23.23 | 24.77 | 28.75 |
| TUM_RGBD/freiburg2_xyz/rgbd | 551 | 20.42 | 23.52 | 24.73 | 26.85 |
| TUM_RGBD/freiburg2_xyz/mono | 1622 | 21.33 | 23.79 | 25.05 | 26.98 |

selected/内为16张原尺寸渲染PNG和4张GT。figures/内为单组和四组排版PNG／SVG。指标来自保存8位PNG前的FP32渲染；各自完整选点核验和来源见FURNISHED_SELECTION.json。
