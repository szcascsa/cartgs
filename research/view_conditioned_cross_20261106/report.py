"""依据已完成五组匹配结果生成中文报告，不调参、不选择终点。"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'evaluation/reports'
summary=json.loads((OUT/'SUMMARY.json').read_text())
geometry=json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())
factorial=json.loads((OUT/'FACTORIAL_EFFECTS.json').read_text())
S={(v['group'],v['case']):v for v in summary if v['partition']=='all'}
G={(v['group'],v['case']):v for v in geometry}
G.update({(v['group'],v['case']):v for v in json.loads((OUT/'BASELINE_GEOMETRY_SUMMARY.json').read_text())['summary']})
E={(v['case'],v['metric'],v['effect']):v for v in factorial}
groups=['baseline','mode','selector_view','compensator_view','view']
names={'baseline':'原无视角联合','mode':'两网络仅模式','selector_view':'Selector视角／Comp模式',
    'compensator_view':'Selector模式／Comp视角','view':'两网络真实视角'}
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];train=[.01,.05,.1,.3,.5,.7]
gates=json.loads((OUT/'QUALITY_GATES.json').read_text())
tail=json.loads((OUT/'TAIL_DIAGNOSTIC_SUMMARY.json').read_text())
T=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
timing=json.loads((OUT/'TRAINING_TIMING.json').read_text())

lines=['# Selector与Compensator视角输入交叉消融报告','',
    '## 结论与范围','',
    '本轮新增Selector真实视角＋Compensator仅模式、Selector仅模式＋Compensator真实视角两组，LI均关闭。原无视角联合、两网络仅模式、两网络真实视角三组现有结果只读复用，不重复训练或渲染。fr1_xyz RGB-D，seed0，同一Full11980起点，每组Fine11580步，评价同792帧。','',
    '|配置|局部1% PSNR|局部5% PSNR|局部10% PSNR|局部2k PSNR|Full PSNR|',
    '|---|---:|---:|---:|---:|---:|']
for g in groups:
    lines.append('|'+names[g]+'|'+'|'.join(f"{S[g,c]['psnr_mean']:.4f}" for c in ['view_0.01','view_0.05','view_0.1','points_2000','full'])+'|')
lines+=['','|新增组相对原联合|局部1% ΔPSNR|局部5% ΔPSNR|局部六训练档均值|局部未见档均值|全局训练档均值|最差全局训练档|Full ΔPSNR|PSNR保护|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for g in ['selector_view','compensator_view']:
    q=next(v for v in gates if v['reference']=='baseline' and v['group']==g)
    keys=['view_1pct_delta','view_5pct_delta','view_train_mean_delta','view_unseen_mean_delta','global_train_mean_delta','global_train_min_delta','full_delta']
    lines.append('|'+names[g]+'|'+'|'.join(f'{q[k]:+.4f}' for k in keys)+'|'+('通过' if q['quality_guard_passed'] else '未通过')+'|')
lines+=['','预声明保护为相对原无视角联合：全局六训练ratio均值≥−0.10dB、任一档≥−0.20dB、Full≥−0.10dB；没有看结果后调整。质量收益与XYZ、Scale、LPIPS和计算分别判断，不以PSNR保护通过替代几何可靠性。生产默认未更改。','',
    '**核心结论：本轮不能把巨大XYZ长尾简单归因于Compensator直接读取相机。** 仅Compensator真实视角时，局部5%／10%的XYZ p99为2.12／1.90，未复现两网络真实视角时的17.49／23.68；仅Selector真实视角也没有复现该量级。四格结果更支持两分支共同适配中的交互风险，但单seed、不同终态Master与选点曝光尚未排除，不能宣布唯一根因。','',
    '选择收益依预算而异：固定2k仅Selector视角PSNR21.4204dB，接近两者视角21.4523dB；其XYZ p99为0.4662，远低于20.2856。不过其局部1%下降0.2278dB、全局1%下降0.2642dB，未通过预声明全局保护。仅Compensator视角在局部5%改善0.1697dB、10%改善0.2155dB并通过PSNR保护，但局部1%下降0.0771dB，低ratio LPIPS也退化。因此两组都不是各预算、质量指标和几何全面占优的生产替代。','',
    '原两网络仅模式在局部1%本身已有较大XYZ长尾（p99=10.62），说明问题也可能发生于没有真实相机输入的网络，不能把上述5%／10%的交互现象推广为所有ratio的唯一机制。','',
    '## 视角收益与长尾归因','',
    '以下均为单场景单seed的系统差分。四格共享网络宽度、参数量、Full起点和相机ratio决策；Master可在Fine更新，选点及实际优化曝光会变化，因此不冒充固定地图、固定选点的纯因果证明。','',
    '|局部条件|仅Selector视角−仅模式 ΔPSNR|仅Comp视角−仅模式 ΔPSNR|两者视角−仅Comp视角 ΔPSNR|两者视角−仅Selector视角 ΔPSNR|交互ΔPSNR|',
    '|---|---:|---:|---:|---:|---:|']
effects=['selector_at_mode','compensator_at_mode','selector_at_compensator_view','compensator_at_selector_view','interaction']
for case in ['view_0.01','view_0.05','view_0.1','points_1000','points_2000']:
    lines.append('|'+case+'|'+'|'.join(f"{E[case,'psnr',e]['mean']:+.4f}" for e in effects)+'|')
lines+=['','交互定义为view−selector_view−compensator_view＋mode，它衡量本轮组合结果是否可由两个单独效应相加解释；不提供跨seed显著性。','',
    '|局部条件|组别|XYZ/reference p50均值|p90均值|p99均值|跨视角XYZ最大值|Scale倍率p99均值|',
    '|---|---|---:|---:|---:|---:|---:|']
for c in ['view_0.01','view_0.05','view_0.1','points_2000']:
    for g in groups:
        v=G[g,c]
        lines.append('|'+c+'|'+names[g]+'|'+'|'.join(f'{v[k]:.4f}' for k in ['xyz_p50_mean','xyz_p90_mean','xyz_p99_mean','xyz_global_max','scale_p99_mean'])+'|')
lines+=['','这些分位数先按单视角计算，再跨792帧平均，不是把全部点样本合并后的分位数。reference为Anchor当前格子边长2.5／5／10 cm；不同点层级混合，整组p99不能统一乘0.1换算为米。归一化位移不是该点必需移动量的真值。','',
    '|新增组|条件|XYZ米制p99均值|米制最大值|位移>10 reference比例|其中仍有Alpha的点占全部选点|其中radii=0的点占全部选点|',
    '|---|---|---:|---:|---:|---:|---:|']
for g in ['selector_view','compensator_view']:
    for c in ['view_0.01','view_0.05','view_0.1','points_2000']:
        v=G[g,c]
        lines.append('|'+names[g]+'|'+c+'|'+f"{v['xyz_meters_p99_mean']:.4f}|{v['xyz_meters_max_mean']:.4f}|"+
            '|'.join(f"{100*v[k]:.3f}%" for k in ['xyz_over10_fraction_mean','xyz_over10_alpha_fraction_mean','xyz_over10_radii_zero_fraction_mean'])+'|')
lines+=['','>10是训练前声明的诊断分组，不新增训练限幅或Loss。这里两个“其中”列的分母仍为全部选中点，不能误读为条件比例。米制最大值一栏为各视角最大值的均值，逐帧全极值见PER_VIEW.csv。','',
    '### 首／中／末帧Alpha核查','',
    '仅两新增模型各12组LoD＋3组Full，总30次诊断渲染；不保存图片，历史模型重渲染0。24组XYZ与直接按同一Gaussian行调用网络得到的Master＋delta逐bit一致。','',
    '|组别|条件|尾部样本|原Full有Alpha|补偿后有Alpha|补偿后radii=0|补偿后深度≤0.01|opacity<0.01|',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for v in tail:
    lines.append('|'+names[v['group']]+'|'+v['case']+'|'+'|'.join(str(v[k]) for k in ['tail_records','tail_alpha_before','tail_alpha_after','tail_radii_zero','tail_behind_after','tail_opacity_below_001'])+'|')
lines+=['','同一点可跨三帧重复计数。透明度、几何移出和选点变化应分别解释；大Scale或低opacity不能单独当作画质下降原因。m=0只屏蔽前向相机信息，不隔离局部Loss对共享主干、Anchor Feature和Master的更新。少量诊断帧只说明被检查样本，不代表完整792帧所有长尾点。','',
    '## 各ratio与实际预算','',
    '|ratio|类别|原联合局部PSNR|仅模式|仅Selector视角|仅Comp视角|两者视角|',
    '|---:|---|---:|---:|---:|---:|---:|']
for r in ratios:
    lines.append(f"|{r:g}|{'训练' if r in train else '未见'}|"+'|'.join(f"{S[g,f'view_{r:g}']['psnr_mean']:.4f}" for g in groups)+'|')
lines+=['','|ratio|全局原联合PSNR|仅模式|仅Selector视角|仅Comp视角|两者视角|',
    '|---:|---:|---:|---:|---:|---:|']
for r in ratios:
    lines.append(f'|{r:g}|'+'|'.join(f"{S[g,f'global_{r:g}']['psnr_mean']:.4f}" for g in groups)+'|')
lines+=['','|ratio|组别|局部PSNR↑|SSIM↑|LPIPS↓|实际K均值|Alpha点数均值|',
    '|---:|---|---:|---:|---:|---:|---:|']
for r in [.01,.05,.1]:
    for g in groups:
        v=S[g,f'view_{r:g}']
        lines.append(f'|{r:g}|'+names[g]+'|'+'|'.join(f'{v[k]:.6f}' for k in ['psnr_mean','ssim_mean','lpips_mean','K_mean','contributing_K_mean'])+'|')
lines+=['','|请求局部点数|组别|PSNR|平均实际K|候选不足帧数|','|---:|---|---:|---:|---:|']
for b in [1000,2000,3000,4000,5000,10000]:
    for g in groups:
        v=S[g,f'points_{b}'];lines.append(f"|{b}|{names[g]}|{v['psnr_mean']:.4f}|{v['K_mean']:.3f}|{v['budget_shortfall_frames']}|")
lines+=['','实际K=min(请求B,合格候选E)，所有保留点包括不透明度很低或不可见点均计入K。matched五档使用各模型自己的global实际Alpha点数作为该帧局部请求预算，不是跨模型same-K，也不保证最终Alpha点数相同。固定预算使用的network_ratio逐帧记录，部分高预算可能超出训练ratio区间，应与插值结果分开解释。完整global、matched、逐帧median/p10/p90、Bank72/nonBank720及配对改善比例见CSV。','',
    '## 实现、初始化与训练验证','',
    '新增ViewBudget.conditioning=selector_view／compensator_view，checkpoint schema3/4；原off／mode／view和schema1/2保留。Selector19→32→32→2，Compensator Residual48逻辑54＋条件8，保持51D输入投影、ratio3投影、64/64及四Head，条件通过无Bias8→64投影在第一ReLU前相加。相对无条件结构新增768参数（权重3KiB、两Adam矩约6KiB）；本轮相对既有条件两组没有新增参数。仅模式分支的七个视角通道恒零，相关列无数据梯度，因此相同注册参数量不代表相同有效自由度。Selector仍不读取Compensator Feature48表。','',
    'Anchor view7为世界方向3、相机深度1、NDC中心2、投影范围1，固定场景尺度，detach。全局两个网络零8D；局部两个网络mode=1，仅指定网络读取view7，另一网络视角通道和梯度为0。共享代码只计算一次，不引入高斯私有latent。全局训练保留零投影梯度与Adam历史动量更新，冻结global可跳过零投影。','',
    '保留无界XYZ、Softplus Scale（physical_scale_additive=0、softplus_scale=1）、原Master输入梯度和Feature scatter/独立Adam。两网络权重独立。LI=0，不进行LI定向VJP或额外LI渲染；原全局GI保留。Full→global→local逐路反传后一次Adam，Master允许更新，拓扑固定。训练Gumbel-ST及软预算，局部推理严格K；两者不能混为一谈。','',
    '两新增组从相同Full11980重启，seed0，fresh Adam/RNG，正式Fine11580步，不接36步smoke。原61状态Tensor（首层仅原11列）、Master初始文件和11580条相机ratio序列与历史一致；六ratio各1930次。Full/global/local各11580、GI额外Full864；Master/Selector Adam各11580，Feature六表时钟11580/9650/7720/5790/3860/1930。实际K和曝光单独记录TRAINING_EXPOSURE.csv。此入口不是完整SLAM/Adam恢复，轨迹固定，不评价ATE改善。','',
    '25项原生测试及两组36步checkpoint smoke通过，包括CPU/CUDA路由/梯度、零初始化/RNG、空候选、保存恢复和冻结克隆；旧CUDA90个输出/VJP Tensor逐bit比较通过；12组跨语言fixture、60项短程推理及60项最终模型实际输入hook确认分支路由与评价选点SHA一致。204组计数Renderer与普通Renderer颜色/radii逐bit相同。新训练有意改变网络函数，不宣称与旧架构终态逐bit一致。','',
    '两个网络无相机输入的分支在同一Gaussian、同一ratio及同一模型状态下可保持视角无关预测；局部候选集和Gumbel选择会随视角变化，不能因此声称整体局部渲染视角无关。','',
    '## 实际选点曝光','',
    '下表为11580步累计选中Gaussian次数，不是独立点数。四个条件组的全局／局部曝光接近；原无条件baseline局部曝光比条件组更多。步数对齐仍不能把相对baseline的全部变化归为纯结构收益。','',
    '|组别|全局K累计|局部K累计|局部候选人口累计|','|---|---:|---:|---:|']
X=json.loads((OUT/'EXPOSURE_TOTALS.json').read_text())
for g in groups:
    v=next(v for v in X if v['group']==g)
    lines.append(f"|{names[g]}|{v['global_K_exposure_sum']}|{v['local_K_exposure_sum']}|{v['candidate_N_exposure_sum']}|")
lines+=['','仅Selector视角／仅Comp视角局部曝光相对两网络仅模式分别约+0.175%／−0.067%；原无条件baseline的局部曝光约高9.2%。曝光差异是记录到的训练行为，不通过事后改预算去消除；推理1%／5%／10%与2k的实际K在各组相同。','',
    '## Master终态差异','',
    '固定拓扑下按保留的PLY行序比较未补偿Master，CPU完成，不调用Renderer。下表是Master位置变化，不是Compensator输出位移；它量化地图混杂，不能独立判断其质量收益。','',
    '|组别|相对初始Master XYZ p99（mm）|相对原联合终态XYZ p99（mm）|','|---|---:|---:|']
D=json.loads((OUT/'MASTER_DRIFT.json').read_text())['rows']
for g in groups:
    a=next(v for v in D if v['group']==g and v['reference']=='initial')
    b=next(v for v in D if v['group']==g and v['reference']=='baseline')
    lines.append(f"|{names[g]}|{1000*a['xyz_meters']['p99']:.3f}|{1000*b['xyz_meters']['p99']:.3f}|")
lines+=['','完整Scale倍率变化也记录在MASTER_DRIFT.json；部分初始轴尺度很薄，倍率本身不能代替绝对尺度、opacity或可见性解释。','',
    '## 性能、限制与后续','',
    '|组别|训练墙钟秒|Mapping平均ms|median ms|p90 ms|','|---|---:|---:|---:|---:|']
for g in groups:
    v=next(x for x in timing if x['group']==g)
    seconds=f"{T['groups'][g]['seconds']:.2f}" if g in T['groups'] else '历史复用'
    lines.append(f"|{names[g]}|{seconds}|{v['mapping_ms_mean']:.3f}|{v['mapping_ms_median']:.3f}|{v['mapping_ms_p90']:.3f}|")
lines+=['','共享GPU存在其他任务，旧三组来自不同时间。墙钟不能解释为纯编码或CUDA加速，本轮未做Nsight或峰值显存配对微测。两组LI0，无额外训练渲染，评价图片不落盘。','',
    '训练保留finite、连续零选点、概率饱和和原补偿硬故障检查。原阈值为XYZ/reference>256、Scale倍率>100且至少1%越限；未触发不等于几何合理。没有利用nonBank结果挑学习率、终点或checkpoint。','',
    '已证实内容限于本次匹配训练、质量指标和选点/几何记录。四格效应为合理机制证据，未排除Master终态、训练采样K、共享Feature更新及CUDA原子非确定性的影响。下一步最有信息价值的是对mode／compensator_view／view做配对seed复验，判断仅Comp视角的5%收益和两者视角的长尾交互能否重复；固定2k优先保留selector_view作为研究候选，仍需解决其1%退化。若交互可重复，再固定Master和选择集合进行受控补偿诊断，区分选点输入分布与共享Feature更新。当前不直接加大LI权重或用输出限幅掩盖问题。','',
    '## 文件与完整性','',
    '服务器独立目录：/home/prmi/zwj/experiments/view_conditioned_cross_20261106。新模型：runs/full_selector_view/23560_shutdown/ply与runs/full_compensator_view/23560_shutdown/ply。原模型只读保留在上轮目录。','',
    'PER_VIEW.csv共134640行：53856新评价＋80784历史复用；附34配置、完整逐帧、配对统计、四格效应、训练曝光和几何长尾。源快照、独立编译对象/二进制SHA、实际/proc加载映射及输入哈希在reports；最终604个生产文件前后、历史输入、source/binary及零渲染帧图片以FINAL_INTEGRITY.json为依据。生产默认不切换，不commit/push。']
(OUT/'CROSS_VIEW_CONDITIONING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('中文报告已生成')
