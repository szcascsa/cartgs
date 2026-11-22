# 双网络真实视角＋LI：批量结果

Full赢家按每个数据集／场景／模式的seed0三次全轨迹PSNR选取，only anchor prune=1。
Full＋全局LoD＋局部LoD，LI=0.1；Master可更新、固定拓扑；未见ratio只评价。
每个工作项保留最终模型和全轨迹数值指标，不保存渲染图片；健康重跑与OOM计数见attempt_history。

|场景／模式|Full起点PSNR|Fine步数|状态|
|---|---:|---:|---|
|Replica/office0/mono|34.749611|18477|complete|
|Replica/office0/rgbd|35.662867|21075|complete|
|Replica/office1/mono|37.580097|18313|complete|
|Replica/office1/rgbd|38.777988|20026|complete|
|Replica/office2/mono|30.929848|18362|complete|
|Replica/office2/rgbd|31.626631|20440|complete|
|Replica/office3/mono|30.834345|17067|complete|
|Replica/office3/rgbd|32.638474|20249|complete|
|Replica/office4/mono|32.872275|18254|complete|
|Replica/office4/rgbd|34.614665|22233|complete|
|Replica/room0/mono|30.331038|16962|complete|
|Replica/room0/rgbd|31.806648|20585|complete|
|Replica/room1/mono|30.269336|17280|complete|
|Replica/room1/rgbd|32.893480|20412|complete|
|Replica/room2/mono|31.646806|18077|complete|
|Replica/room2/rgbd|34.411857|22274|complete|
|TUM_RGBD/freiburg1_desk/mono|19.995212|7709|complete|
|TUM_RGBD/freiburg1_desk/rgbd|19.937788|8700|complete|
|TUM_RGBD/freiburg1_room/mono|15.194895|18351|complete|
|TUM_RGBD/freiburg1_room/rgbd|17.546566|18775|complete|
|TUM_RGBD/freiburg1_xyz/mono|22.851718|8164|complete|
|TUM_RGBD/freiburg1_xyz/rgbd|22.492783|11580|complete|
|TUM_RGBD/freiburg2_xyz/mono|22.111006|43229|complete|
|TUM_RGBD/freiburg2_xyz/rgbd|22.510000|50165|complete|
|TUM_RGBD/freiburg3_long_office_household/mono|20.482028|31452|complete|
|TUM_RGBD/freiburg3_long_office_household/rgbd|22.439143|34442|complete|

逐ratio PSNR/SSIM/LPIPS/ATE/N/实际K见LOD_METRICS.csv；逐帧指标位于evaluation目录。
未因评价质量回选checkpoint或更改训练终点。最高Full结果的三次轨迹覆盖可能不同，不能称共同帧统计检验。
