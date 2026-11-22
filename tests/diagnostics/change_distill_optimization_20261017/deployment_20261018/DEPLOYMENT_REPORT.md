# ChangeDistill 正式更新记录

已将验证通过的 A（保守 transport）、B（cache-hit）、C（Selector embedding复用）
更新到 `prmi@172.19.163.58:/home/prmi/zwj/mycartgs`，并完成正式 CMake Release 构建。
本地源码已在上一阶段完成优化；本次同步的是同一份实现。

全量构建最初在历史profiling目标上发现过期的生成文件：旧mapper副本的两参数接口与
当前头文件三参数接口不一致。通过现有CMake配置流程重新生成build目录中的诊断副本后，
增量构建成功。未修改生产接口或CMake源码；初次失败和成功重建分别保留在
`build_initial.log` / `build_initial.exit`与`build_refreshed.log` / `build_refreshed.exit`。

- 只同步6个生产文件和1个cache测试，逐文件哈希/标准化文本核对通过。
- 其他26个审计文件及生产YAML保持原值；独立selector_epochs/tf_epochs配置保留。
- 使用现有 `build-cuda118-sm86-gcc9`，CUDA11.8/sm86、LibTorch2.3.1/ABI1、正式`--fmad=false`编译参数。
- 正式共享库 `lib/libgaussian_mapper.so` 已改变哈希，确认导出新的 `selector_cuda::networkMany`。
- `bin/tum_rgbd`、`tum_mono`、`replica_rgbd`、`replica_mono`、`vector_mono`、`vector_stereo`
  的依赖检查无缺失库；主要入口提供必需的--selector-config后，可正确显示Usage。
- ChangeDistill CPU/CUDA、TF、Selector、cache与相关回归均通过；完整CTest结果见下文。
- 没有提交commit，没有重新运行SLAM，也没有改动其他任务的独立runtime。

完整CTest：**61/62通过**。`ultrafast_mapping_cuda`在已有断言
`Mode1 tail advanced clocks without Gaussian/TF optimizer updates`处失败。
同一新测试程序分别加载备份旧库、新库、再次加载新库，均在同一断言失败；
更新前备份的原测试程序+原共享库也复现相同失败。因此这是更新前已存在的问题，
没有把失败隐藏为通过，也没有为让测试变绿修改Mapping、TF loss或原有测试。
四组原始日志与返回码保留在`tail_counterfactual*.log` / `tail_counterfactual.json`。
本次ChangeDistill优化相关测试和数值验收均通过；上述Mapping tail问题保留为独立已知问题。

正式共享库另通过762项跨ratio输出、loss、输入/参数梯度与单步更新比较；全部逐位一致。

部署期间服务器有独立TUM实验，未将并发GPU负载下的耗时作为性能证据。
更新依据为此前同一LC6207、完整分支/事件的空闲GPU复测：完整事件759.142→619.347ms（1.226×）。
本次正式构建侧验证的是编译、数值正确性、回归与实际动态库加载；未把既有隔离runner的数值冒充为正式库新测值。


更新前源码、bin/lib及关键构建产物已备份：

`/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017/deployment_20261018/backup`

当前正式mapper库SHA-256：`b98517fb7665c1871cb162c254731f43b26c3c98e2a78e72a1aecd84f2ecce19`。

运行中的 `resultsall/batch_20261018_tum_aligned/runtime` 使用自己的冻结库，保持其原版本。
新启动 `/home/prmi/zwj/mycartgs/bin/*` 会通过正式库使用本次优化。

本目录保留 `source_before.json`、`source_after.json`、`deployment_manifest.json`、
`build.log`、`ctest.log`、退出码、共享库校验和启动日志；后续可按相同快照复测正式库：

```bash
cd /home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017/deployment_20261018
/home/prmi/anaconda3/envs/zwj/bin/python run_benchmark.py baseline
/home/prmi/anaconda3/envs/zwj/bin/python run_benchmark.py optimized
```

baseline明确从`backup/lib`加载旧库，optimized从正式`mycartgs/lib`加载新库；已有输出不会被覆盖。
