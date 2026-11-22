# cloud_b更换机器后原地恢复

原cloud_b端口14086由新端口48709替代，实验目录和8组场景归属保持不变。新机hostname为autodl-container-9d764ea661-70439085。凭据仅进入SSH通道和采集进程内存，不写入本报告、脚本或状态文件。

## 检查与恢复

新机已保留原数据盘及兼容路径，实验目录实际位于/root/autodl-tmp/no_importance_20261115/mirror。未重复传输数据、安装环境或编译。确认旧控制器和子进程均已退出，再取得独占文件锁启动唯一新控制器。

恢复前通过529个源码文件、10项部署runtime和既有调度脚本SHA核验。PyTorch2.3.1+cu118／CUDA11.8正常，新驱动环境下90个固定CUDA Tensor仍与原服务器逐bit一致。4个已训练checkpoint通过原Fine步数、三分支反传、GI／LI关闭、ratio曝光和相机序列核验。

|场景／模式|恢复处理|
|---|---|
|Replica room2 mono|训练及23项评价全部保留|
|Replica office1 mono|训练及23项评价全部保留|
|Replica office0 mono|保留模型和4项评价；global 0.1中断，改为attempt_02重做该项|
|Replica office1 RGB-D|保留模型；2项已登记评价及global 0.05完整未登记结果均保留，继续global 0.08及后续评价|
|VECtor corridors-dolly mono|原训练中断至7456步，无可恢复中途权重；attempt_02从相同Full重新执行20984步|
|Replica office0 RGB-D|原训练中断至4889步，无可恢复中途权重；attempt_02从相同Full重新执行21075步|
|Replica room2 RGB-D|继续原队列等待，22274步|
|VECtor corridors-dolly stereo|继续原队列等待，32777步|

共保留4个已训练模型和53项完整评价。global 0.05通过2000帧覆盖、原图口径、有限值、逐帧均值与summary一致及原参考帧配对核验后登记，未重复渲染，退出码记录为未知而非推断为0。所有53项summary与逐帧CSV共106个文件恢复后SHA未变化。

中断现场及attempt_01全部保留，关机中断不计OOM，不消耗最多额外2次OOM重试额度。未完成训练是从同一Full、seed0、fresh Adam／RNG重跑，不是从中途Adam状态继续。网络、Loss、ratio、预算、Full／全局／局部训练流程与完整步数均未改变。

## 运行与汇总

新节点已同时运行2组训练＋2项评价，首次验收训练分别推进到490／700步，日志／CSV更新时间小于1秒，GPU利用率约99%、显存约3.3GiB，未出现健康失败文件。已核验PID／start_ticks／boot_id和实际加载的独立Mapper。

原服务器的ASSIGNMENTS.json仅更新cloud_b连接端口，32组任务归属不变。旧端口采集进程在身份核对后退出，新采集器已通过48709取得新控制器状态，保留30分钟巡查、模型／指标SHA回收及过滤图片的规则。端点变更和旧错误记录保存在原服务器reports/cloud_b_endpoint_change_*，未删除历史结果。

实验目录继续使用/home/prmi/zwj/experiments/no_importance_all_scenes_20261115。未修改生产项目，未commit／push。
