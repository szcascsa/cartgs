# VECtor双目评价口径核查

结论：corner-slow与robot-normal的当前复跑结果和图中CaRtGS Vanilla使用相同的图像质量评价规则。本次只读检查源码、历史命令、相机、文件与数值，不重训或重渲染。

| 核查项 | 双方实际行为 |
|---|---|
| 评价图像 | 同一批未校正左目原图，逐文件samefile核验 |
| 数据索引与GT | rgb.txt有效数据行一致（Vanilla额外三行注释），groundtruth.txt内容一致 |
| 评价范围 | GT同步后的完整估计轨迹，RGB最近时间戳关联，阈值0.08秒，无帧采样 |
| 相机 | 1224×1024，fx=fy=887.3200073242188；MiniCam与FoV转换函数AST相同 |
| PSNR／SSIM | 相同TorchMetrics构造和逐帧调用，data_range=1.0 |
| LPIPS | 相同TorchMetrics构造与逐帧调用，normalize=True，未另设net_type |
| 颜色与预处理 | BGR转RGB，GT除255，NCHW，预测clamp至[0,1]，不裁剪、不resize、不作有效像素遮罩 |
| 汇总 | 各帧指标算术平均；同一黑背景；SH degree=3 |

| 场景 | 同一原图数量 | 我们核查评价项 | CaRtGS N | 我们N | CaRtGS Full PSNR | 我们Full PSNR |
|---|---:|---:|---:|---:|---:|---:|
| VECtor/corner-slow/stereo | 1238 | 23 | 4995 | 5716 | 19.563524 | 19.737125 |
| VECtor/robot-normal/stereo | 1205 | 23 | 8167 | 6774 | 16.912316 | 16.955900 |

## 比较的限制

- 各自估计轨迹，未用同一组GT相机隔离轨迹误差。
- 统一原图域不等于消除训练校正域与原图之间的几何错位。
- ATE两边均继承traj_ref.align(traj_est, True)，随后correct_scale=False；不可简称完全没有尺度变换。
- 核查源码和历史命令，未重新渲染验证两套CUDA二进制逐位相同。

历史校正图口径下，我们旧模型的Full PSNR为corner-slow 27.188997、robot-normal 25.783061 dB；同一旧模型改用原图评价后分别为19.720315、16.963335 dB。这两套数字不能混用。

两边模型、位姿、Gaussian数量和训练量不同；评价口径相同不代表固定地图／固定相机／同K的受控算法消融。

详细来源与输入哈希见AUDIT_VERIFIED.json；生产配置、原模型和历史指标均未修改。
