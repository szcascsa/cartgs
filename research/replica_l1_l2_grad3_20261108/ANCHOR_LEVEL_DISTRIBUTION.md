# Replica各场景Anchor Level 2占比

本表读取本次3τ₀阈值实验最终Full＋tail checkpoint，未重新训练或渲染。
Anchor2指level=2；占比为L2 Anchor数除以全部Anchor数。成员Gaussian占比按所有存储点统计，包含inactive slot，不是可见点比例。

|场景|Mono L2／全部Anchor|占比|RGB-D L2／全部Anchor|占比|
|---|---:|---:|---:|---:|
|office0|192／6,501|2.95%|123／12,259|1.00%|
|office1|196／6,882|2.85%|112／11,626|0.96%|
|office2|352／8,804|4.00%|272／17,023|1.60%|
|office3|342／6,035|5.67%|129／20,277|0.64%|
|office4|265／7,799|3.40%|402／22,396|1.79%|
|room0|1,606／12,350|13.00%|1,421／29,229|4.86%|
|room1|828／8,236|10.05%|445／15,835|2.81%|
|room2|719／7,012|10.25%|631／20,151|3.13%|

|场景|Mono L2成员Gaussian占比|RGB-D L2成员Gaussian占比|
|---|---:|---:|
|office0|1.55%|0.62%|
|office1|1.37%|0.39%|
|office2|2.88%|1.11%|
|office3|3.24%|0.48%|
|office4|2.15%|1.05%|
|room0|8.89%|2.49%|
|room1|5.35%|1.25%|
|room2|6.57%|1.60%|

逐组L0/L1/L2原始计数、模型路径、源文件哈希见ANCHOR_LEVEL_DISTRIBUTION.json。
16组均核验Anchor数量、Gaussian数量、owner范围、member_counts与owner逐行计数一致；统计前后源文件SHA256相同。
