# Replica各场景Anchor Level 2占比

本表读取本次固定t=0.001、alpha=1.2实验最终Full＋tail checkpoint，未重新训练或渲染。
Anchor2指level=2；占比为L2 Anchor数除以全部Anchor数。成员Gaussian占比按所有存储点统计，包含inactive slot，不是可见点比例。

|场景|Mono L2／全部Anchor|占比|RGB-D L2／全部Anchor|占比|
|---|---:|---:|---:|---:|
|office0|374／6,396|5.85%|261／11,528|2.26%|
|office1|317／6,606|4.80%|249／10,946|2.27%|
|office2|735／8,800|8.35%|546／16,944|3.22%|
|office3|597／6,661|8.96%|505／21,072|2.40%|
|office4|515／7,817|6.59%|519／22,985|2.26%|
|room0|2,242／13,348|16.80%|2,649／28,858|9.18%|
|room1|1,551／8,975|17.28%|1,434／16,753|8.56%|
|room2|975／7,424|13.13%|1,085／20,681|5.25%|

|场景|Mono L2成员Gaussian占比|RGB-D L2成员Gaussian占比|
|---|---:|---:|
|office0|2.89%|1.21%|
|office1|2.70%|0.97%|
|office2|6.21%|1.90%|
|office3|5.14%|1.37%|
|office4|4.36%|1.33%|
|room0|11.15%|4.35%|
|room1|9.58%|3.72%|
|room2|8.01%|2.45%|

逐组L0/L1/L2原始计数、模型路径、源文件哈希见ANCHOR_LEVEL_DISTRIBUTION.json。
16组均核验Anchor数量、Gaussian数量、owner范围、member_counts与owner逐行计数一致；统计前后源文件SHA256相同。
