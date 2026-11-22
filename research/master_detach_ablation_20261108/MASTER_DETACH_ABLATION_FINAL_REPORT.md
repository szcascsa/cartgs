# Master固定消融最终报告（失败组重新运行后）

队列于香港时间2026-11-11T13:02:21.711874+08:00结束；29组成功，3组最终失败，完成667项全轨迹评价。

本次用户授权3个最终失败组各新增第7次尝试。历史失败和原先29组结果保留；健康阈值、训练步数、seed0、网络和LI=0.1保持不变。Master六类参数在Full、全局LoD和局部LoD三路均固定。

|本次重跑组|最终状态|累计健康失败|新增评价|
|---|---|---:|---:|
|TUM_RGBD/freiburg1_room/mono|skipped_health|7|0|
|VECtor/corridors-dolly/mono|skipped_health|7|0|
|VECtor/corridors-dolly/stereo|skipped_health|7|0|

本次健康失败的组仍由原保护停止，没有修改阈值以强行完成；只有训练完成且冻结验证通过的模型才进入评价。

|Ratio|全局LoD PSNR差值(dB)|局部LoD PSNR差值(dB)|
|---|---:|---:|
|1%|-2.0638|-1.7176|
|5%|-1.7960|-1.8739|
|10%|-1.4659|-1.5789|
|30%|-0.8324|-1.2825|
|50%|-0.1389|-0.7581|
|70%|+0.0094|-0.5461|

以上为29个成功场景/相机模式的等权配对平均，差值=冻结Master减去可更新Master。失败场景不混入质量均值；存在失败与重复尝试的幸存样本偏差。

原29组模型文件783项哈希及队列记录均不变；全部成功模型、指标与1866个原Full、baseline和生产受保护文件验证通过。未保存渲染图片。

数据：reports/LOD_METRICS.csv；逐帧：evaluation/；完整审计、逐数据集逐ratio均值：reports/FINAL_AUDIT.json。

旧终态报告与审计：/home/prmi/zwj/realLOD/ablation/reports/final_failures_rerun_1794372888581505567

生产默认未替换，未commit或push。
