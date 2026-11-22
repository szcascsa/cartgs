# 单台AutoDL关机后恢复

已恢复端口43433的cloud_a节点，仍采用2组训练＋2项评价独立队列。端口14086连接被拒绝，按用户要求不启动或迁移该节点任务。

## 恢复前核验

- cloud_a原控制器及其训练／评价进程均已退出；虽然宿主boot_id相同，仍通过PID、start_ticks及实际进程核实，未把旧JSON当作进程存活证据。
- 实验目录、分派数据、环境和checkpoint保留完整，数据盘剩余约28.9GiB。
- 529项源码和10项部署runtime哈希一致，部署脚本哈希一致。未迁移的编译对象不属于云节点运行依赖，不要求在云机存在。
- Python／Torch／CUDA可用；重启后再次通过90个固定CUDA Tensor逐bit核验。
- 4组完成训练的模型全部通过原训练步数、三分支、GI／LI开关、ratio曝光和相机序列核验。
- 50项已完成评价对应的100个summary／逐帧CSV文件记录SHA256，恢复后均保持一致。

## 任务处理

|任务|关机现场|恢复方式|
|---|---|---|
|Replica room1 mono|训练和23项评价完成|全部保留|
|Replica office4 mono|训练和23项评价完成|全部保留|
|Replica office2 mono|训练完成，3项评价完成，global 0.08中断|保留模型与已完成评价，只重做未完成评价|
|Replica room1 RGB-D|训练完成，1项评价完成，global 0.03中断|保留模型与已完成评价，只重做未完成评价|
|Replica office2 RGB-D|训练至4999步，无最终／中途可恢复权重|从同一原Full起点重新执行20440步|
|Replica office4 RGB-D|训练至2808步，无最终／中途可恢复权重|从同一原Full起点重新执行22233步|
|TUM freiburg3 long office household mono／RGB-D|尚未开始|保留队列顺序和训练预算|

中断attempt_01及其日志不覆盖、不删除；四个中断任务新建attempt_02。attempt_history新增interrupted_by_shutdown记录，不记为OOM，不消耗OOM最多额外2次重试额度。未完成训练重跑使用原Full、seed0、fresh Adam／RNG；这不是从中途Adam状态继续。

## 恢复后验收

新控制器已获得唯一文件锁，PID／start_ticks／boot_id与真实进程匹配。两组训练和两项评价均已运行，实测两组训练推进至767／743步，所有日志和训练CSV最近更新时间小于1秒，GPU利用率约98%，显存约3.6GiB，未出现健康失败文件。通过实际maps确认加载独立Mapper。

原服务器负责结果汇总，已部署的30分钟采集与巡检继续使用；本次恢复报告、计划、运行验收和状态快照同步至原服务器cluster/cloud_a。cloud_b保留原任务现场，等待用户启动后另行恢复。

原始训练配置、网络、GI／LI消融条件、ratio、完整步数、原图全轨迹评价口径均未修改；不commit、不push、不修改生产项目。
