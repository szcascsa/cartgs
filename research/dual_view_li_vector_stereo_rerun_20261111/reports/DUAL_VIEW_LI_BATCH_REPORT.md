# 双网络真实视角＋LI：批量结果

Full赢家按每个数据集／场景／模式的seed0三次全轨迹PSNR选取，only anchor prune=1。
Full＋全局LoD＋局部LoD，LI=0.1；Master可更新、固定拓扑；未见ratio只评价。
每个工作项保留最终模型和全轨迹数值指标，不保存渲染图片；健康重跑与OOM计数见attempt_history。

|场景／模式|Full起点PSNR|Fine步数|状态|
|---|---:|---:|---|
|VECtor/corner-slow/stereo|26.861866|11470|complete|
|VECtor/robot-normal/stereo|25.271853|13006|complete|

逐ratio PSNR/SSIM/LPIPS/ATE/N/实际K见LOD_METRICS.csv；逐帧指标位于evaluation目录。
未因评价质量回选checkpoint或更改训练终点。最高Full结果的三次轨迹覆盖可能不同，不能称共同帧统计检验。
