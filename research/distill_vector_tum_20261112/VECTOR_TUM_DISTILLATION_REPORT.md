# TUM 与 VECtor Compensator 蒸馏报告

训练核验 16/16，评价核验 16/16，部署验收 13/16，保护文件完整性 True。

每场景使用独立32/32学生，四阶段共17600步；教师、Master、Selector与Anchor Feature冻结。训练六档Ratio并在连续阶段采样中间比例，按每5帧留出学生未见视角，32帧测试。教师和地图已使用同轨迹。

| 场景 | 相机 | 高斯数 | 全局PSNR损失 | 局部PSNR损失 | 通过门限 | 部署 |
|---|---|---:|---:|---:|---:|---|
| freiburg1_desk | mono | 18445 | 0.00952 | 0.02681 | 6/22 | True |
| freiburg1_desk | rgbd | 20601 | 0.00788 | 0.05999 | 4/22 | True |
| freiburg1_room | mono | 27577 | 0.04103 | 0.15643 | 0/22 | True |
| freiburg1_room | rgbd | 72211 | 0.00244 | 0.12542 | 0/22 | True |
| freiburg1_xyz | mono | 13630 | 0.01003 | 0.01669 | 5/22 | True |
| freiburg1_xyz | rgbd | 13298 | 0.01476 | 0.06365 | 1/22 | True |
| freiburg2_xyz | mono | 19910 | 0.07509 | 0.37561 | 0/22 | False |
| freiburg2_xyz | rgbd | 15634 | 0.19240 | 1.14975 | 0/22 | False |
| freiburg3_long_office_household | mono | 14431 | 0.03682 | 0.02946 | 0/22 | True |
| freiburg3_long_office_household | rgbd | 42700 | 0.04332 | 0.12305 | 0/22 | True |
| corner-slow | mono | 4644 | 0.03971 | 0.24821 | 0/22 | True |
| corner-slow | stereo | 5716 | -0.02423 | 0.01060 | 7/22 | True |
| corridors-dolly | mono | 15818 | 0.06003 | 0.05964 | 0/22 | True |
| corridors-dolly | stereo | 26640 | 0.00712 | 0.01181 | 0/22 | False |
| robot-normal | mono | 8189 | 0.08051 | 0.34519 | 0/22 | True |
| robot-normal | stereo | 6774 | 0.00685 | -0.01948 | 1/22 | True |

PSNR损失为同视角、同选点与实际K下的Teacher PSNR减去Student PSNR，正值表示学生退化。质量门限依次为平均损失≤0.01 dB、p99≤0.05 dB、最差≤0.1 dB。每条件32视角，统计范围有限。CUDA部署一致性与蒸馏质量是独立检查。

最终学生在352个场景、选择模式与Ratio条件中有24个通过上述严格门限。全局学生评价使用PyTorch前向，局部学生评价使用CUDA部署路径；因此未通过部署一致性的三组，其局部质量差距同时包含蒸馏与CUDA实现差异，不能直接全部归因于蒸馏。

fr1_room单目的原轨迹有1334条记录，其中356条为时间戳和位姿完全相同的重复记录。删除重复项后使用978个不同视角，模型和原始轨迹均未改写；首次准入失败的日志和后续补跑记录保留在实验目录。不同位姿对应同一图像的冲突仍会被拒绝。

未通过CUDA部署一致性检查的条件：fr2_xyz单目第0帧、全局Ratio 0.7为47.805 dB；fr2_xyz RGB-D第0帧、局部Ratio 0.01为54.393 dB；VECtor corridors-dolly双目第0帧、全局Ratio 0.05为59.686 dB。这些是普通学生输出图与CUDA学生输出图之间的PSNR，不能解释成相对GT的PSNR损失；三个学生的训练与质量评价均已完成。

逐Ratio PSNR、SSIM、LPIPS、实际K与长尾见DISTILLATION_METRICS.csv及各场景QUALITY_PER_VIEW.csv和ATTRIBUTES_PER_VIEW.csv。每组模型位于models/<场景>/continuous_calibration/best.pt；完整身份、错误及校验见RESULTS_VALIDATION.json。训练为两路并发，未做独占GPU速度基准。不保存渲染图像。
