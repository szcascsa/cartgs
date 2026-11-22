# RealLoD 冻结 Master：TUM fr1_xyz 两组完整实验

日期：2026-11-03。实验已完整训练、渲染和评价，不是仅启动任务或短程烟测。

## 结论

LoD 阶段冻结 Master 已生效：两组 Master 更新次数均为0；Full终点与最终Master PLY的SHA256一致，792张Full渲染PNG逐文件一致，逐帧PSNR/SSIM/LPIPS数组一致。Selector、Compensator和活跃Anchor Feature继续训练，未发现非有限值。

冻结保住了Full，但不能据此认定LoD质量提高。低ratio仍存在明显质量损失；历史Joint LoD-only结果仅作参考，地图、在线步数及曝光不匹配，不能作为单因素收益。

## 代码与训练协议

- 本地分支real-lod，HEAD为857909570760b61337e698af63b725524d164619，冻结实现包含未提交修改，精确文件版本见PATCH_MANIFEST.json和部署identity。
- 服务器项目为/home/prmi/zwj/realLOD。旧主项目/home/prmi/zwj/mycartgs未被修改。
- 在线Full建图→正常Full尾优化→独立LoD训练；LoD步数等于实际在线Mapping，不额外设置在线/总步数上限。
- LoD原分辨率、ratio轮转、固定Gaussian/Anchor拓扑，Selector.detach_gaussian_parameters=1。
- 保留Selector11D、Residual48逻辑54D和四Head、现有Scale解码、共享Adam和已启用CUDA路径。
- Master六组梯度保持undefined，Adam值/一阶矩/二阶矩/step和学习率不推进；未激活Feature残差保持undefined gradient。
- FlexGS重要性在首步及每1000步刷新，Full统计扫描无backward；它不等于每步Full图像监督。

## 实际预算

| ratio组 | 在线Full | Full尾部 | LoD | Master LoD更新 | Gaussian | Anchor | ratio曝光 | 子进程训练秒 | LoD秒（含GI） |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| r20_40_60_80 | 6152 | 748 | 6152 | 0 | 13350 | 3387 | [1538, 1538, 1538, 1538] | 99.441 | 54.625 |
| r01_05_10_30_50 | 6091 | 400 | 6091 | 0 | 13801 | 3660 | [1219, 1218, 1218, 1218, 1218] | 95.111 | 52.130 |

每组重要性刷新7次、累计497个Full统计视图。它们另外计数，不包含训练backward。Mapping次数与在线实际Adam提交次数不能混同，完整分阶段计数见training_verified.json中的stage。

## 全部792个共同RGB帧的质量

| ratio组 | 输出 | PSNR↑ | SSIM↑ | LPIPS↓ | 平均实际K | Gaussian总数 |
|---|---|---:|---:|---:|---:|---:|
| r20_40_60_80 | full_before_lod | 22.315772 | 0.767849 | 0.274973 | 13350.000 | 13350 |
| r20_40_60_80 | full | 22.315772 | 0.767849 | 0.274973 | 13350.000 | 13350 |
| r20_40_60_80 | lod_20pct | 21.020382 | 0.724281 | 0.348214 | 2671.000 | 13350 |
| r20_40_60_80 | lod_40pct | 22.151783 | 0.762572 | 0.283028 | 5271.292 | 13350 |
| r20_40_60_80 | lod_60pct | 22.230442 | 0.765277 | 0.278340 | 6372.071 | 13350 |
| r20_40_60_80 | lod_80pct | 22.236286 | 0.765456 | 0.278093 | 6396.330 | 13350 |
| r01_05_10_30_50 | full_before_lod | 22.098815 | 0.765576 | 0.277031 | 13801.000 | 13801 |
| r01_05_10_30_50 | full | 22.098815 | 0.765576 | 0.277031 | 13801.000 | 13801 |
| r01_05_10_30_50 | lod_1pct | 13.924028 | 0.532687 | 0.728796 | 139.000 | 13801 |
| r01_05_10_30_50 | lod_5pct | 17.032520 | 0.607109 | 0.572612 | 691.000 | 13801 |
| r01_05_10_30_50 | lod_10pct | 18.688402 | 0.652636 | 0.483333 | 1381.000 | 13801 |
| r01_05_10_30_50 | lod_30pct | 21.444916 | 0.741355 | 0.313373 | 4137.903 | 13801 |
| r01_05_10_30_50 | lod_50pct | 21.784856 | 0.755443 | 0.290679 | 6535.351 | 13801 |

这是完整序列共同帧评价，含训练关键帧，不是纯未见视图结果。本轮没有训练未见ratio质量评价。表中K是现有视锥候选、Slot及分位数推理后的实际均值，没有使用same-K重采样。高ratio下候选不足会使实际K低于名义rN，不能仅按名义ratio解释质量/速度。

## 历史Joint参考与归因边界

| ratio组 | Joint LoD-only前Full PSNR | Joint后Full PSNR | 变化 | 冻结组前后Full变化 |
|---|---:|---:|---:|---:|
| r20_40_60_80 | 22.347494 | 20.259290 | -2.088204 | 0（逐文件/逐帧一致） |
| r01_05_10_30_50 | 22.342276 | 18.031287 | -4.310989 | 0（逐文件/逐帧一致） |

历史Joint仅计算LoD图像Loss，Master可以改变但没有逐步Full保真目标；冻结组则完全取消Master适应性。两者不构成“联合Full+LoD目标已测试”的证据。下一项有价值的配对是同一Full终点、同一Adam/RNG/视图序列下的Joint LoD-only与Joint Full+LoD。设计见FLEXGS_FINE_TRAINING_DESIGN.md。

## 原生验证与构建

增量构建只重编译两个修改CPU源码及原生测试单元，复用原构建的未变对象和CUDA静态库。原有三个SLAM主程序无需重编译，动态库依赖与RPATH已有部署检查。

通过：native_frozen、native_legacy_adam、native_joint、residual48_training_test（97项前向/VJP）、real_lod_schedule_test。冻结原生测试逐位检查六组Master的参数、Adam状态、step及学习率，同时确认网络/Feature更新；并保留显式Joint模式。

## 完整性与模型

- 旧主项目328个受保护文件SHA256再次核验一致；当前realLOD源文件清单2558项一致。
- 未覆盖历史模型、未切换到新的Full+LoD目标、未commit/push；本轮冻结部署的回退文件在独立构建目录rollback/。
- 导出不是完整SLAM/Adam/RNG断点恢复，不能用PLY/selector.pt声称精确恢复优化历史。
- 耗时是本轮实际运行记录；不是配对速度Benchmark，未将纯渲染FPS当作在线SLAM加速。

- r20_40_60_80最终模型：/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown/ply
- r01_05_10_30_50最终模型：/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r01_05_10_30_50/training/12582_shutdown/ply

实验根目录：/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01

完整证据：combined_summary.csv、training_verified.json、master_bitwise.json、full_image_equality.json、full_metric_equality.json、FINAL_VERIFICATION.json及原生测试日志。证据包排除模型、图片、数据集；完整模型和渲染留在服务器上述目录。
