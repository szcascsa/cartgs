# 三数据集逐场景Ratio结果表

Excel包含Replica、TUM_RGBD、VECtor三个工作表。每张表先按相机模式分区，再按场景分组。Replica为8场景的Mono与RGB-D，TUM为5场景的Mono与RGB-D，VECtor为3场景的Mono与Stereo，共32组。

每个场景包含CaRtGS随机删除、CaRtGS不透明度删除、我们全局Ratio、我们局部Ratio四行。Photo-SLAM和RTGS按本次确认不纳入。

Ratio为1%、3%、5%、8%、10%、20%、30%、40%、50%、60%、70%。每档列出PSNR、SSIM、LPIPS和#GS。PSNR显示两位小数，SSIM和LPIPS显示三位小数，#GS显示整数；Excel与CSV均保留原始数值精度。PSNR、SSIM越高越好，LPIPS越低越好。

## Gaussian数量

我们局部方法的#GS为同一场景、相机模式、Ratio下，全部已评价视角选中Master Gaussian ID的并集基数U。某Gaussian只要被选中过就计入一次，多视角重复出现不重复计数；不同补偿位置仍视为同一母Gaussian。统计对象是提交Renderer的选点，不是Alpha实际贡献点数。

CaRtGS两种删除方法与我们全局Ratio采用固定保留集合大小。此处#GS衡量整段评价轨迹所需的唯一母Gaussian总量，不能作为单帧渲染工作量。逐项数据另保留原逐视角平均K与Full地图N，未用均值K替代U。

## 评价范围

质量结果沿用既有未校正原始RGB评价，不重新训练或渲染。各方法使用各自的估计轨迹和有效关联帧，未重算公共帧或固定相机位姿评价；帧数和评价来源见逐项CSV。

相同Ratio下，各方法的Full地图数量、候选池和实际保留数不同，因此不是同K对照。表格按参考图使用两层表头与场景分组，没有添加数值优胜标记。

## 文件

- 三数据集_各场景Ratio质量与去重点数.xlsx：三个可编辑工作表，冻结前两列及表头。
- REPLICA／TUM／VECTOR_全部场景表.png：每数据集一张完整表格图，包含两种相机模式。
- REPLICA／TUM／VECTOR_全部场景表.csv：每数据集一张宽表，便于复制和再排版。
- data/：各数据集逐项原精度记录，保留评价帧数、Full地图N、原平均K及评价来源。
- QA_RECEIPT.json：覆盖核验、原始数值逐位保留与数据来源SHA256。

局部U复用此前完成的352项统计，其618838条逐帧选点哈希已与历史评价核对。当前表共1408项方法×场景模式×Ratio记录，5632个质量与点数值逐项核验。
