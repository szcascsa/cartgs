# Replica Full与Vanilla渲染质量图

本次3τ₀ Full＋tail与Vanilla CaRtGS Full比较，8场景，每张图分别展示Mono和RGB-D的PSNR、SSIM、LPIPS和Gaussian总数。
PSNR和SSIM越高越好，LPIPS越低越好；点数是整图存储N，不是单视角可见点或有效Alpha合成点。
本次为seed0单次；Vanilla为seed0三次中PSNR最高重复，其余指标和N绑定同次。两种方法的全轨迹帧覆盖可能不同，未重聚合共同帧。
本轮不绘制ratio，不使用未训练LoD网络，不重新训练或渲染。

|场景|模式|本次PSNR|Vanilla PSNR|本次SSIM|Vanilla SSIM|本次LPIPS|Vanilla LPIPS|本次N|Vanilla N|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|office0|mono|33.729|37.645|0.9330|0.9629|0.2099|0.1215|21,420|40,906|
|office0|rgbd|35.339|37.844|0.9528|0.9654|0.1479|0.1080|34,920|41,934|
|office1|mono|27.523|40.396|0.8769|0.9671|0.2804|0.0949|19,783|52,982|
|office1|rgbd|38.432|40.257|0.9540|0.9687|0.1250|0.0804|31,252|58,187|
|office2|mono|30.860|34.238|0.9145|0.9443|0.2115|0.1331|25,138|68,377|
|office2|rgbd|30.943|34.047|0.9270|0.9492|0.1869|0.1359|44,886|84,203|
|office3|mono|30.717|34.088|0.9061|0.9415|0.1908|0.1184|20,028|59,395|
|office3|rgbd|32.681|34.355|0.9312|0.9465|0.1373|0.1046|51,895|73,531|
|office4|mono|32.662|36.016|0.9334|0.9534|0.1648|0.1051|22,626|56,365|
|office4|rgbd|34.163|36.607|0.9465|0.9612|0.1222|0.0854|53,150|78,451|
|room0|mono|29.867|33.384|0.8888|0.9349|0.1797|0.1083|41,262|100,118|
|room0|rgbd|30.356|33.728|0.8989|0.9430|0.1289|0.0900|82,592|139,948|
|room1|mono|29.962|34.706|0.8967|0.9465|0.2116|0.1128|28,623|89,228|
|room1|rgbd|32.621|35.084|0.9349|0.9519|0.1491|0.0945|54,850|110,504|
|room2|mono|31.821|35.889|0.9305|0.9595|0.1715|0.0965|22,792|66,340|
|room2|rgbd|34.515|37.354|0.9539|0.9691|0.1112|0.0703|52,463|118,040|

## 逐场景图片

- [office0：PNG](office0_full_vs_vanilla.png)／[SVG](office0_full_vs_vanilla.svg)
- [office1：PNG](office1_full_vs_vanilla.png)／[SVG](office1_full_vs_vanilla.svg)
- [office2：PNG](office2_full_vs_vanilla.png)／[SVG](office2_full_vs_vanilla.svg)
- [office3：PNG](office3_full_vs_vanilla.png)／[SVG](office3_full_vs_vanilla.svg)
- [office4：PNG](office4_full_vs_vanilla.png)／[SVG](office4_full_vs_vanilla.svg)
- [room0：PNG](room0_full_vs_vanilla.png)／[SVG](room0_full_vs_vanilla.svg)
- [room1：PNG](room1_full_vs_vanilla.png)／[SVG](room1_full_vs_vanilla.svg)
- [room2：PNG](room2_full_vs_vanilla.png)／[SVG](room2_full_vs_vanilla.svg)
