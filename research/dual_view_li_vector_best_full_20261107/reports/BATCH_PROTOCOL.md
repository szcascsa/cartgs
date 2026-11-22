# VECtor双网络真实视角＋LI实验协议

corner-slow、corridors-dolly、robot-normal各运行Mono和双目，共6组。每组使用seed0三次中Full全轨迹PSNR最高的原Full起点，双目赢家按校正后左图域的指标选取。

本轮复制此前双网络真实视角＋LI批次的527个源码文件及16个运行文件，运行文件SHA256相同。配置为LI=0.1、only anchor prune=1、Full＋全局LoD＋局部LoD联合反传后一次共享Adam。Master可更新，拓扑固定，Adam/RNG重新初始化，Fine步数等于原online Mapping步数。

训练ratio为1%、5%、10%、30%、50%、70%。未见ratio为3%、8%、20%、40%、60%，仅评价。每组进行11档全局、11档名义Anchor候选池局部及Full的完整轨迹评价，共23项。输出PSNR、SSIM、LPIPS、ATE、Gaussian数量、实际K及逐帧指标，渲染帧图片保存数为0。

Mono使用尺度对齐。双目使用原生OpenCV左图校正map与保存的相机内参，尺度对齐关闭，采用整个校正图像计算指标。每种入口先执行6步训练及Full、全局1%/3%/70%、局部1%验收。

OOM和健康停止各最多额外重试2次，计数跨控制器重启保留。未知错误阻塞并保存现场。监督器每小时核查真实进程、日志、步数、磁盘和图片数。原Replica/TUM批次保持暂停。

|场景／模式|Full起点PSNR|Fine步数|Gaussian数|评价帧数|
|---|---:|---:|---:|---:|
|VECtor/corner-slow/mono|24.261622|7132|4644|1087|
|VECtor/corner-slow/stereo|26.861866|11470|5716|1238|
|VECtor/corridors-dolly/mono|24.400037|20984|15818|885|
|VECtor/corridors-dolly/stereo|25.322223|32777|26640|920|
|VECtor/robot-normal/mono|22.604794|9355|8189|1183|
|VECtor/robot-normal/stereo|25.271853|13006|6774|1205|
