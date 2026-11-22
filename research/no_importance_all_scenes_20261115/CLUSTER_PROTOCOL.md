# D组无重要性监督：三节点分布式执行协议

算法和评价沿用本轮PROTOCOL.md。关闭GI与LI Loss，保留两网络真实视角、原GI采集、全部训练步数与六ratio轮转。无论节点，均使用各自地图/视角自己的N计算预算。禁止因硬件切换重新挑选Full起点。

## 分派

原服务器172.19.163.58保留已完成及已开始任务，并负责其余8组未开始任务；两台24GB RTX3090各接收8组。ASSIGNMENTS.json记录32组唯一归属，以历史实际训练与全轨迹评价时间平衡。同一场景的两种相机模式共用RGB数据。已完成结果不重训。原服务器保留两个任务线程；AutoDL按用户追加要求使用独立的两个训练线程和两个评价线程，最多同时运行两组训练与两项评价。

节点A端口43433：Replica room1、office2、office4；TUM freiburg3_long_office_household，各两种模式。

节点B端口14086：Replica room2、office0、office1；VECtor corridors-dolly，各两种模式。

原服务器新增8组：Replica room0、office3；TUM freiburg1_room、freiburg2_xyz，各两种模式。

## 独立部署与验证

云节点数据、环境和依赖实际存放于/root/autodl-tmp/no_importance_20261115/mirror。新机专用/home/prmi符号链接保留原始输入路径；CARTGS_CLUSTER_ROOT固定逻辑实验目录，防止resolve后产生不能在原服务器读取的输出路径。生产项目不迁移、不修改。

复制原服务器zwj环境和已验收二进制，保持Python、PyTorch、CUDA运行库、评价依赖版本；不使用云机预装旧Torch。仅复制分派场景RGB、相机与轨迹文件、固定Full checkpoint、小型配对评价参考及所需运行依赖，不复制深度、其他场景或历史LoD模型。源文件逐一SHA256核验；绝对图像路径使用新机内部别名，cameras.json字节不变。

每台新机必须通过：CUDA可用和版本检查；四项原生测试；90个固定CUDA微图Tensor与原服务器逐bit检查；各dataset/mode的6步原生三分支训练与初始权重/PLY核验；Full、全局1%/3%/70%、局部1%各3帧数值评价。验收失败不得启动正式训练。加载Mapper时检查真实解析后的动态库路径。

## 结果、重试与巡查

原服务器是唯一汇总节点。每30分钟核查三节点PID/start_ticks/boot_id、真实进程、CSV/日志增量、磁盘与训练失败；超过10分钟无进展记录CPU等待、IO及日志现场。连续零选点仍停止，几何长尾不终止，非有限值与运行错误保持原检查；明确OOM最多额外重试2次，次数写入原任务状态。

每组完成后流式拉回模型、数值、日志及核验信息。禁止向原服务器传入渲染图片；数据集RGB属于输入，仅部署到需要它的云节点。云机允许保留渲染图片，但本轮自动评价继续采用原数值输出路径，不额外生成全轨迹图片。

每个节点独立持有自己任务的状态；原服务器运行中的JSON不被远程采集器覆盖。中间汇总写CLUSTER_STATUS.json、CLUSTER_PIPELINE.json、CLUSTER_A_VS_D_METRICS.csv。只有原本地调度器退出、所有远程终态任务的模型和数值SHA核验成功，才合并为最终LOD_PIPELINE.json并运行原finish.py。失败任务保留原因，不补造指标。

AutoDL并行调度使用CARTGS_PIPELINE_OVERLAP=1，训练完成并通过核验后才把该模型提交评价池。评价池与后续训练同时工作，两类任务仍在独立进程中，不共享模型或Adam状态。原有显存等待、OOM累计重试、健康检查和结果校验不变。切换时核实旧控制器及其全部子进程，让当前评价自然结束并收割结果，再替换唯一控制器；不重新训练已完成模型、不重复评价已完成ratio。

凭据仅通过不回显输入进入进程内存，不写入代码、命令行参数、报告或状态文件。不commit、不push、不修改生产默认。
