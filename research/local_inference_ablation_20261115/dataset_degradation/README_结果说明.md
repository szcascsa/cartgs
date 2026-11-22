# 三数据集局部Ratio消融退化图

每个数据集一张三联雷达图，展示关闭Compensator与随机选择保留Compensator的质量退化。比较基准为同一Ratio的完整局部方案。PSNR／SSIM下降和LPIPS上升均为正值，向外表示退化增加。

TUM为5场景，Replica为8场景，VECtor为3场景。每场景先平均两种相机模式，再对场景等权平均。每个模式的输入指标已是完整轨迹均值。模式和场景的权重均等。

径向刻度在三个数据集中保持一致，PSNR为0–18 dB，SSIM为0–0.7，LPIPS为0–0.25。Ratio按参考图等角度排列，图下表保留全部11个Ratio的数值。

无补偿使用LoD Fine终点Master属性。随机选择使用固定seed0优先级。数据为冻结checkpoint的推理消融，各视角的候选池和实际K已与完整方案匹配。完整精度与来源保存在data目录。
