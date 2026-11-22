# CaRtGS 未校正原图评价更新

已对当前双网络真实视角＋LI模型的VECtor三个双目场景重做69条全轨迹评价，包含Full、11档全局Ratio及11档局部Ratio。其余29组667条原有评价已使用相同原图口径，复用并核验。

只移除评价命令中的stereo-camera-config及stereo-native-maps，不改变渲染相机、网络、checkpoint、GT时间关联或选点。全部69条评价帧关联、实际K一致，66条LoD选点身份一致。

PSNR／SSIM／LPIPS保留CaRtGS相同的TorchMetrics参数、输入排列和逐帧调用；不作尺度对齐，不保存渲染或GT图片。

| 双目场景 Full | 校正域 PSNR | 原图 PSNR | 校正域 SSIM | 原图 SSIM | 校正域 LPIPS | 原图 LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| VECtor/corner-slow/stereo | 27.188997 | 19.720315 | 0.915025 | 0.839070 | 0.362317 | 0.521885 |
| VECtor/corridors-dolly/stereo | 26.493774 | 19.706734 | 0.929644 | 0.883611 | 0.366388 | 0.454498 |
| VECtor/robot-normal/stereo | 25.783061 | 16.963335 | 0.920924 | 0.848450 | 0.318394 | 0.472568 |

新值已写回服务器原批次LOD_PIPELINE.json、reports/LOD_METRICS.csv；原校正域逐帧评价完整保留，更新前汇总备份在before_original_rgb。训练模型哈希未改变。

这次变化反映评价图像域改变。模型训练仍采用原有训练域，不将原图指标下降解释为模型权重或训练行为退化。Full起点选模记录仍保留历史口径，不重新挑选或训练模型。

全部32图和768条汇总记录见ALL_SCENE_METRICS.csv。逐ratio新旧差异及原路径见EVALUATION_PROTOCOL_CHANGES.csv。
