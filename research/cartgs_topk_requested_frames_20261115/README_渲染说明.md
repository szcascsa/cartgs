# CaRtGS Top-K指定视角渲染

Replica room0 Mono：输入帧950，对应frame000950.jpg，原评价序号947。TUM fr2_xyz RGB-D：输入帧551，对应1311867188.831350.png，原评价序号551。输入与评价编号从0开始。

分别导出保留1%、5%、10%、30%的结果。Top-K按激活后不透明度稳定降序排序，K=floor(ratio×CaRtGS自身Full地图点数N)，预算使用Decimal计算。直接复用已有selected_ids.npy，同时重新核对排序、ID哈希和对应checkpoint；不训练、不补偿、不修改高斯属性。

复用CaRtGS原评价运行时、原估计相机轨迹和原始RGB图。Replica沿用单目尺度对齐，TUM RGB-D不做尺度校正。每张图的PSNR、SSIM、LPIPS均与此前保存的同帧同Ratio结果核验，最大误差小于2e-5。原源码、checkpoint、参考图与历史评价文件的哈希前后不变。

runs目录包含8张原分辨率渲染PNG和2张GT。图像按原评价方式clamp[0,1]，乘255后截断为uint8，不改变色彩、亮度或相机；以无损PNG保存。figures目录为带GT、K和PSNR的缩略拼图。CSV保存完整精度指标及来源路径；没有重新计算全轨迹指标。
