# Replica各场景Anchor Level 2占比

本表读取本次q80校准t、alpha=1.5实验最终Full＋tail checkpoint，未重新训练或渲染。
Anchor2指level=2；占比为L2 Anchor数除以全部Anchor数。成员Gaussian占比按所有存储点统计，包含inactive slot，不是可见点比例。

|场景|Mono L2／全部Anchor|占比|RGB-D L2／全部Anchor|占比|
|---|---:|---:|---:|---:|
|office0|590／6,588|8.96%|523／13,340|3.92%|
|office1|633／6,924|9.14%|650／11,881|5.47%|
|office2|1,100／9,116|12.07%|1,389／18,587|7.47%|
|office3|849／7,035|12.07%|635／22,421|2.83%|
|office4|872／8,245|10.58%|2,331／26,059|8.95%|
|room0|3,056／14,215|21.50%|3,372／32,094|10.51%|
|room1|2,129／10,032|21.22%|2,615／18,752|13.95%|
|room2|1,628／8,242|19.75%|1,928／21,444|8.99%|

|场景|Mono L2成员Gaussian占比|RGB-D L2成员Gaussian占比|
|---|---:|---:|
|office0|4.94%|2.04%|
|office1|5.52%|2.55%|
|office2|7.60%|3.86%|
|office3|6.78%|1.60%|
|office4|7.03%|4.98%|
|room0|13.88%|5.06%|
|room1|12.67%|6.23%|
|room2|11.78%|4.41%|

逐组L0/L1/L2原始计数、模型路径、源文件哈希见ANCHOR_LEVEL_DISTRIBUTION.json。
16组均核验Anchor数量、Gaussian数量、owner范围、member_counts与owner逐行计数一致；统计前后源文件SHA256相同。
