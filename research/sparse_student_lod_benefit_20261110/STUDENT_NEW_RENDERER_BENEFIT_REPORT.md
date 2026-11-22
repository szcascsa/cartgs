# 新渲染器＋蒸馏Compensator配对速度报告

## 结论

本轮只替换Compensator执行权重，使用同一新G Renderer、最终R相机、Selector、Master、Anchor Feature、候选池及实际K。比较原Teacher64/64四个两层Head与最终Student32/32四个直接Head；Full采用冻结Master属性直接渲染，不每帧复制整图。

4个Replica场景/模式、20个局部Ratio条件中，学生完整流程相对Teacher中位加速范围 1.000–1.152×，配对每帧节省 0.0008–0.1165ms。相对直接Full更快的条件，Teacher为 4/20，Student为 5/20；近1×条件仍需看两轮波动，不能泛称所有LoD都比Full快。

room0的1%档两轮学生相对Teacher约0.997×和1.002×，没有稳定额外加速；该档K很低且大地图候选/Host准备开销仍在。room0的10%档两轮学生均快于直接Full约3.2%–3.3%，Teacher两轮均略慢于Full，这是本轮明确改变盈亏结果的配置。70%虽能节省更多补偿计算，完整学生LoD在四场景仍均慢于直接Full。

## 实验范围与身份

office1单目、office2 RGB-D、room0 RGB-D与room2单目。前三组与上一轮Teacher研究相同；第四组替换了fr1_xyz，因为fr1_xyz现有学生的Teacher指纹与上一轮锁定checkpoint不一致，禁止混用，不为测速重训。两轮共28512条计时，792份逐位Renderer验收和792条计时外GT质量数据。

Student文件为Replica全场景研究中各场景continuous_calibration/best.pt，按完整Teacher/地图/Feature指纹加载，参数3534，Teacher25742。模型路径和SHA、step、fingerprint见MODEL_IDENTITIES.json。

## 完整推理速度

包含当前相机CPU求逆及R准备、候选、Selector、Compensator、属性打包、新G渲染与最后同步。Full没有网络推理，仅当前相机与原Master渲染。单位ms，表为同轮配对样本中位数，未混用旧Renderer或不同测试批次。

| 场景 | Ratio | 实际K中位数 | 直接Full | Teacher LoD | Student LoD | 学生相对Teacher | 学生相对Full | 配对节省Teacher→Student ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Replica/office1/mono | 0.01 | 76 | 0.3900 | 0.4112 | 0.3995 | 1.029× | 0.976× | +0.0105 |
| Replica/office1/mono | 0.05 | 381.5 | 0.3900 | 0.4569 | 0.4335 | 1.054× | 0.900× | +0.0223 |
| Replica/office1/mono | 0.1 | 763 | 0.3900 | 0.4784 | 0.4581 | 1.044× | 0.851× | +0.0196 |
| Replica/office1/mono | 0.3 | 2290 | 0.3900 | 0.5183 | 0.4933 | 1.051× | 0.791× | +0.0231 |
| Replica/office1/mono | 0.7 | 5344 | 0.3900 | 0.5717 | 0.5242 | 1.090× | 0.744× | +0.0423 |
| Replica/office2/rgbd | 0.01 | 154 | 0.4667 | 0.4487 | 0.4371 | 1.027× | 1.068× | +0.0120 |
| Replica/office2/rgbd | 0.05 | 771.5 | 0.4667 | 0.4977 | 0.4753 | 1.047× | 0.982× | +0.0226 |
| Replica/office2/rgbd | 0.1 | 1543 | 0.4667 | 0.5274 | 0.5024 | 1.050× | 0.929× | +0.0239 |
| Replica/office2/rgbd | 0.3 | 4629.5 | 0.4667 | 0.5723 | 0.5513 | 1.038× | 0.847× | +0.0210 |
| Replica/office2/rgbd | 0.7 | 10804 | 0.4667 | 0.6804 | 0.6151 | 1.106× | 0.759× | +0.0650 |
| Replica/room0/rgbd | 0.01 | 218 | 0.6037 | 0.5066 | 0.5064 | 1.000× | 1.192× | +0.0008 |
| Replica/room0/rgbd | 0.05 | 1091.5 | 0.6037 | 0.5863 | 0.5521 | 1.062× | 1.093× | +0.0337 |
| Replica/room0/rgbd | 0.1 | 2184 | 0.6037 | 0.6173 | 0.5844 | 1.056× | 1.033× | +0.0324 |
| Replica/room0/rgbd | 0.3 | 6553.5 | 0.6037 | 0.7380 | 0.6535 | 1.129× | 0.924× | +0.0844 |
| Replica/room0/rgbd | 0.7 | 15080.5 | 0.6037 | 0.8705 | 0.7560 | 1.152× | 0.799× | +0.1165 |
| Replica/room2/mono | 0.01 | 74 | 0.4317 | 0.4299 | 0.4170 | 1.031× | 1.035× | +0.0125 |
| Replica/room2/mono | 0.05 | 374 | 0.4317 | 0.4728 | 0.4592 | 1.030× | 0.940× | +0.0151 |
| Replica/room2/mono | 0.1 | 748 | 0.4317 | 0.4965 | 0.4794 | 1.036× | 0.901× | +0.0171 |
| Replica/room2/mono | 0.3 | 2244.5 | 0.4317 | 0.5429 | 0.5175 | 1.049× | 0.834× | +0.0254 |
| Replica/room2/mono | 0.7 | 5237.5 | 0.4317 | 0.5895 | 0.5530 | 1.066× | 0.781× | +0.0352 |

## 两轮独立重复

| 场景 | Ratio | 第一轮学生/Teacher加速 | 第二轮 | 第一轮学生/Full加速 | 第二轮 |
|---|---:|---:|---:|---:|---:|
| Replica/office1/mono | 0.01 | 1.028× | 1.031× | 0.977× | 0.975× |
| Replica/office1/mono | 0.05 | 1.054× | 1.054× | 0.901× | 0.899× |
| Replica/office1/mono | 0.1 | 1.044× | 1.044× | 0.852× | 0.851× |
| Replica/office1/mono | 0.3 | 1.051× | 1.049× | 0.791× | 0.790× |
| Replica/office1/mono | 0.7 | 1.088× | 1.095× | 0.742× | 0.747× |
| Replica/office2/rgbd | 0.01 | 1.027× | 1.027× | 1.068× | 1.068× |
| Replica/office2/rgbd | 0.05 | 1.048× | 1.047× | 0.982× | 0.982× |
| Replica/office2/rgbd | 0.1 | 1.051× | 1.048× | 0.929× | 0.929× |
| Replica/office2/rgbd | 0.3 | 1.037× | 1.039× | 0.846× | 0.847× |
| Replica/office2/rgbd | 0.7 | 1.104× | 1.111× | 0.756× | 0.763× |
| Replica/room0/rgbd | 0.01 | 0.997× | 1.002× | 1.194× | 1.190× |
| Replica/room0/rgbd | 0.05 | 1.062× | 1.062× | 1.094× | 1.093× |
| Replica/room0/rgbd | 0.1 | 1.057× | 1.056× | 1.033× | 1.032× |
| Replica/room0/rgbd | 0.3 | 1.126× | 1.131× | 0.923× | 0.923× |
| Replica/room0/rgbd | 0.7 | 1.150× | 1.156× | 0.798× | 0.801× |
| Replica/room2/mono | 0.01 | 1.029× | 1.032× | 1.035× | 1.036× |
| Replica/room2/mono | 0.05 | 1.027× | 1.030× | 0.940× | 0.940× |
| Replica/room2/mono | 0.1 | 1.035× | 1.037× | 0.900× | 0.902× |
| Replica/room2/mono | 0.3 | 1.050× | 1.049× | 0.834× | 0.835× |
| Replica/room2/mono | 0.7 | 1.068× | 1.060× | 0.781× | 0.779× |

## 额外收益来自哪里

prepare仅相机＋完整网络准备，raster仅已准备属性的渲染；两者各自单独测量。各列独立中位数不可简单相加替代完整pipeline。

| 场景 | Ratio | Teacher准备 ms | Student准备 ms | Teacher渲染 ms | Student渲染 ms |
|---|---:|---:|---:|---:|---:|
| Replica/office1/mono | 0.01 | 0.3270 | 0.3190 | 0.0908 | 0.0912 |
| Replica/office1/mono | 0.05 | 0.3438 | 0.3236 | 0.1188 | 0.1192 |
| Replica/office1/mono | 0.1 | 0.3523 | 0.3337 | 0.1321 | 0.1334 |
| Replica/office1/mono | 0.3 | 0.3680 | 0.3458 | 0.1566 | 0.1573 |
| Replica/office1/mono | 0.7 | 0.4081 | 0.3622 | 0.1699 | 0.1706 |
| Replica/office2/rgbd | 0.01 | 0.3455 | 0.3344 | 0.1136 | 0.1140 |
| Replica/office2/rgbd | 0.05 | 0.3666 | 0.3430 | 0.1409 | 0.1420 |
| Replica/office2/rgbd | 0.1 | 0.3783 | 0.3542 | 0.1584 | 0.1592 |
| Replica/office2/rgbd | 0.3 | 0.3927 | 0.3712 | 0.1886 | 0.1894 |
| Replica/office2/rgbd | 0.7 | 0.4803 | 0.4149 | 0.2087 | 0.2094 |
| Replica/room0/rgbd | 0.01 | 0.4075 | 0.4083 | 0.1108 | 0.1104 |
| Replica/room0/rgbd | 0.05 | 0.4440 | 0.4099 | 0.1536 | 0.1540 |
| Replica/room0/rgbd | 0.1 | 0.4486 | 0.4169 | 0.1788 | 0.1791 |
| Replica/room0/rgbd | 0.3 | 0.5152 | 0.4339 | 0.2314 | 0.2304 |
| Replica/room0/rgbd | 0.7 | 0.6051 | 0.4914 | 0.2756 | 0.2749 |
| Replica/room2/mono | 0.01 | 0.3359 | 0.3247 | 0.1021 | 0.1022 |
| Replica/room2/mono | 0.05 | 0.3526 | 0.3370 | 0.1291 | 0.1302 |
| Replica/room2/mono | 0.1 | 0.3601 | 0.3431 | 0.1434 | 0.1441 |
| Replica/room2/mono | 0.3 | 0.3743 | 0.3502 | 0.1762 | 0.1769 |
| Replica/room2/mono | 0.7 | 0.4043 | 0.3668 | 0.1949 | 0.1949 |

Teacher和Student选点身份与K一致，但几何输出不同；Student还可能改变Tile footprint和像素合成工作。完整pipeline加速包含这部分实际渲染差异，不能全部归为MLP计算节省。prepare分项更直接反映网络替换后的准备收益。

## 同视角质量代价

本表仅为本轮18个运动/跳转视角的计时外对GT评价，不替代先前32个学生留出视角的质量报告。个别视角可能参与过学生训练或选模，不能称新增留出泛化测试。每个Ratio下Teacher与Student选点相同，Full和LoD质量不保证相等。

| 场景 | Ratio | Full PSNR | Teacher PSNR | Student PSNR | 学生损失 dB |
|---|---:|---:|---:|---:|---:|
| Replica/office1/mono | 0.01 | 37.8093 | 25.1753 | 25.0360 | +0.1393 |
| Replica/office1/mono | 0.05 | 37.8093 | 29.3937 | 29.0871 | +0.3066 |
| Replica/office1/mono | 0.1 | 37.8093 | 31.1389 | 30.5129 | +0.6259 |
| Replica/office1/mono | 0.3 | 37.8093 | 35.6744 | 35.1157 | +0.5588 |
| Replica/office1/mono | 0.7 | 37.8093 | 37.3053 | 36.8388 | +0.4666 |
| Replica/office2/rgbd | 0.01 | 33.2559 | 17.9314 | 17.9675 | -0.0361 |
| Replica/office2/rgbd | 0.05 | 33.2559 | 24.4923 | 24.3701 | +0.1222 |
| Replica/office2/rgbd | 0.1 | 33.2559 | 27.3119 | 27.0933 | +0.2186 |
| Replica/office2/rgbd | 0.3 | 33.2559 | 31.4184 | 31.2855 | +0.1329 |
| Replica/office2/rgbd | 0.7 | 33.2559 | 32.0996 | 31.8714 | +0.2282 |
| Replica/room0/rgbd | 0.01 | 31.9310 | 19.8500 | 19.7632 | +0.0867 |
| Replica/room0/rgbd | 0.05 | 31.9310 | 23.5174 | 23.4251 | +0.0923 |
| Replica/room0/rgbd | 0.1 | 31.9310 | 25.5963 | 25.3075 | +0.2887 |
| Replica/room0/rgbd | 0.3 | 31.9310 | 29.0551 | 28.8507 | +0.2045 |
| Replica/room0/rgbd | 0.7 | 31.9310 | 30.0532 | 29.9710 | +0.0822 |
| Replica/room2/mono | 0.01 | 30.6958 | 20.9896 | 20.9579 | +0.0317 |
| Replica/room2/mono | 0.05 | 30.6958 | 24.3061 | 24.1325 | +0.1736 |
| Replica/room2/mono | 0.1 | 30.6958 | 25.9337 | 25.8054 | +0.1283 |
| Replica/room2/mono | 0.3 | 30.6958 | 28.6125 | 28.6515 | -0.0390 |
| Replica/room2/mono | 0.7 | 30.6958 | 29.0449 | 28.9606 | +0.0843 |

Replica16组先前正式学生留出结果为全局平均PSNR损失约0.154dB、局部约0.276dB，严格质量等价门限未通过。room2单目此前一个普通学生/CUDA图像一致性条件未达70dB，状态继续保留；本轮只证明新Renderer对该学生的既定CUDA输出逐位一致，不改变已有部署门限。

## 测量与完整性

RTX3090、FP32、TF32关闭、CPU线程数2；2轮×6重复×18个连续或跳转视角，第二轮视角逆序，请求顺序逐帧/重复轮转反向。所有请求正式计时前完成18视角预热，Renderer按地图最大容量预编译，未按Ratio调参。模型加载、Graph预编译、GT读取、相机样本构建索引、图片编码/显示、磁盘I/O和在线Tracking/Mapping均不在热态延迟内。

相机每帧按R重新提交、原inverse错误检查立即执行；候选、Selector、Feature查询、Compensator和渲染每帧重算，没有复用同帧Mask/特征输出。直接Full属性是冻结Master的合法静态输入，不缓存任何相机图像。

校验 792 份当前输入的RGB/radii与原A逐位一致；Teacher/Student逐视角K/ID哈希匹配，学生Teacher指纹匹配。保护 1046 项生产/原模型/学生/冻结二进制，SHA变更0项。独立pmon连续采样没有外部GPU活动；未终止或调整驻留进程，未锁GPU频率。采样不能排除更短活动与CPU噪声，不以近1×结果作显著收益承诺。

新增目录只含实验脚本、报告、数值与性能曲线，未训练、未修改网络/Renderer/生产默认、未commit/push、未保存渲染图片。首次加载与预编译成本沿原Renderer报告保留，短会话不能只看热态延迟。

第一轮尝试检测到外部PID1456420短暂2% SM活动，整次判为无效，已保留于rejected_attempt01，未混入正式计时；随后负载恢复后完整重跑。拒绝依据是GPU活动，而非耗时高低，未终止外部进程。

主要文件：STUDENT_VS_TEACHER_SPEED.csv、STUDENT_VS_TEACHER_ROUNDS.csv、MATCHED_VIEWS_QUALITY.csv、samples.csv、correctness.csv、MODEL_IDENTITIES.json、GPU_OBSERVATIONS.json与VALIDATION.json。

![配对速度](figures/STUDENT_VS_TEACHER_NEW_RENDERER.png)
