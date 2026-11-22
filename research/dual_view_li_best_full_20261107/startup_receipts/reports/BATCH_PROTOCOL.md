# 两网络真实视角＋LI：批量LoD实验

每个数据集／场景／相机模式选择seed0三次中Full全轨迹PSNR最高一次，only anchor prune=1；VECtor按原生矫正左目口径选择。
当前本地main 26c355b及未提交兼容／推理变更独立快照构建，不覆盖生产源码、bin/lib或共享环境。
Fine采用Full＋全局LoD＋局部LoD，真实view7＋mode1，LI=0.1；Master可更新、拓扑固定；Adam/RNG重置，不称完整SLAM恢复。
训练ratio [.01,.05,.1,.3,.5,.7]；未见ratio [.03,.08,.2,.4,.6]仅评价。全局与局部两种全轨迹评价，另评Full；不保存图片。
沿用原长尾/零选点/饱和/非有限值检查，触发后从原Full重新训练，不改参数或Loss。OOM最多额外重试2次；长尾/零选点重试策略见LOD_PIPELINE.json。
每小时独立监督器核查PID/start_ticks/boot_id、实际进程、日志与CSV增量、磁盘、零图片及健康停止；控制器即时处理健康失败，较每小时更及时。
保留最终模型、逐ratio/逐帧数值结果及必要复现/失败证据；仅清理本轮初始化副本及无用中间导出，不删除原Full或历史实验。

|数据集／场景／模式|Full赢家|Full PSNR|Fine步数|
|---|---|---:|---:|
|Replica/office0/mono|Replica/office0/mono/prune1/seed0_repeat3|34.749611|18477|
|Replica/office0/rgbd|Replica/office0/rgbd/prune1/seed0_repeat3|35.662867|21075|
|Replica/office1/mono|Replica/office1/mono/prune1/seed0_repeat2|37.580097|18313|
|Replica/office1/rgbd|Replica/office1/rgbd/prune1/seed0_repeat2|38.777988|20026|
|Replica/office2/mono|Replica/office2/mono/prune1/seed0_repeat2|30.929848|18362|
|Replica/office2/rgbd|Replica/office2/rgbd/prune1/seed0_repeat3|31.626631|20440|
|Replica/office3/mono|Replica/office3/mono/prune1/seed0_repeat1|30.834345|17067|
|Replica/office3/rgbd|Replica/office3/rgbd/prune1/seed0_repeat3|32.638474|20249|
|Replica/office4/mono|Replica/office4/mono/prune1/seed0_repeat1|32.872275|18254|
|Replica/office4/rgbd|Replica/office4/rgbd/prune1/seed0_repeat1|34.614665|22233|
|Replica/room0/mono|Replica/room0/mono/prune1/seed0_repeat3|30.331038|16962|
|Replica/room0/rgbd|Replica/room0/rgbd/prune1/seed0_repeat1|31.806648|20585|
|Replica/room1/mono|Replica/room1/mono/prune1/seed0_repeat1|30.269336|17280|
|Replica/room1/rgbd|Replica/room1/rgbd/prune1/seed0_repeat1|32.893480|20412|
|Replica/room2/mono|Replica/room2/mono/prune1/seed0_repeat3|31.646806|18077|
|Replica/room2/rgbd|Replica/room2/rgbd/prune1/seed0_repeat3|34.411857|22274|
|TUM_RGBD/freiburg1_desk/mono|freiburg1_desk/mono/prune1/seed0|19.995212|7709|
|TUM_RGBD/freiburg1_desk/rgbd|freiburg1_desk/rgbd/prune1/seed0_repeat2|19.937788|8700|
|TUM_RGBD/freiburg1_room/mono|freiburg1_room/mono/prune1/seed0_repeat3|15.194895|18351|
|TUM_RGBD/freiburg1_room/rgbd|freiburg1_room/rgbd/prune1/seed0_repeat3|17.546566|18775|
|TUM_RGBD/freiburg1_xyz/mono|freiburg1_xyz/mono/prune1/seed0_repeat2|22.851718|8164|
|TUM_RGBD/freiburg1_xyz/rgbd|freiburg1_xyz/rgbd/prune1/seed0_repeat2|22.492783|11580|
|TUM_RGBD/freiburg2_xyz/mono|freiburg2_xyz/mono/prune1/seed0_repeat2|22.111006|43229|
|TUM_RGBD/freiburg2_xyz/rgbd|freiburg2_xyz/rgbd/prune1/seed0_repeat2|22.510000|50165|
|TUM_RGBD/freiburg3_long_office_household/mono|freiburg3_long_office_household/mono/prune1/seed0_repeat2|20.482028|31452|
|TUM_RGBD/freiburg3_long_office_household/rgbd|freiburg3_long_office_household/rgbd/prune1/seed0_repeat3|22.439143|34442|

用户追加要求：VECtor全部6组暂不训练或评价，仅运行Replica16组、TUM10组，共26组。
