"""依据已完成的配对统计生成中文报告，不选择checkpoint或训练终点。"""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'evaluation/reports'
S=json.loads((OUT/'SUMMARY.json').read_text());P=json.loads((OUT/'PAIRED_SUMMARY.json').read_text())
I={(v['group'],v['case'],v['partition']):v for v in S}
J={(v['reference'],v['group'],v['case'],v['partition']):v for v in P}
G=json.loads((OUT/'QUALITY_GATES.json').read_text())
T=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
V=json.loads((ROOT/'reports/VALIDATION_STATUS.json').read_text())
F=json.loads((ROOT/'reports/INFERENCE_VALIDATION.json').read_text())
timing=json.loads((OUT/'TRAINING_TIMING.json').read_text())
names={'baseline':'原无视角联合','mode':'仅模式标记','view':'模式＋真实视角'}
groups=list(names);ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
train=[.01,.05,.1,.3,.5,.7]
lines=['# Selector＋Compensator共享视角条件对照报告','',
    '## 结论与适用范围','',
    '本轮完成fr1_xyz RGB-D的两组新增训练：仅模式标记、模式＋真实视角。两组LI均为0，原无视角联合模型和26,928条逐帧指标只读复用，未重复训练或渲染基线。结论限定为单场景、seed0的一次匹配实验。','',
    '|比较|局部1% ΔPSNR|局部5% ΔPSNR|局部训练档平均|局部未见档平均|全局训练档平均|最差全局训练档|Full ΔPSNR|全局保护|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for v in G:
    vals=[v[k] for k in ['view_1pct_delta','view_5pct_delta','view_train_mean_delta','view_unseen_mean_delta','global_train_mean_delta','global_train_min_delta','full_delta']]
    lines.append('|'+names[v['group']]+'−'+names[v['reference']]+'|'+'|'.join(f'{x:+.4f}' for x in vals)+'|'+('通过' if v['quality_guard_passed'] else '未通过')+'|')
lines+=['','全局保护沿用预声明门槛：六个训练ratio平均ΔPSNR≥−0.10dB，任一训练档≥−0.20dB，Full≥−0.10dB。它约束相对基线的质量；mode/view直接比较也列出同门槛供参考。','']
for g in ['mode','view']:
    gate=next(v for v in G if v['reference']=='baseline' and v['group']==g)
    if gate['quality_guard_passed'] and gate['view_1pct_delta']>0 and gate['view_5pct_delta']>0:
        text='同时满足全局保护及局部1%/5%平均PSNR改善，可作为后续配对seed复验候选。'
    else:text='未同时满足全局保护及局部1%/5%改善，不据此替换原默认方案。'
    lines.append(f'- {names[g]}：{text}')
lines+=['','**综合判断：真实视角组的局部PSNR最好，但尚不建议设为生产默认。** 局部1%/5% LPIPS略有变差，且5%/10%及2k预算出现明显XYZ长尾。仅模式组也在1%出现更大的XYZ长尾。通过预声明PSNR保护不等于通过几何可靠性验证；本轮没有证明质量、位移和开销三者同时改善。','',
    f"真实视角相对原联合：局部1% LPIPS变化{J['baseline','view','view_0.01','all']['delta_lpips_mean']:+.6f}、5%变化{J['baseline','view','view_0.05','all']['delta_lpips_mean']:+.6f}（LPIPS越低越好）；全局5% PSNR变化{J['baseline','view','global_0.05','all']['delta_psnr_mean']:+.4f}dB，不能用全局均值掩盖这档退化。"]
lines+=['','## 实际实现','',
    'Selector与Compensator权重彼此独立；共享是指各自跨全局/局部复用，并不将两网络合并。条件z=[m·view7,m]。LoD1为全零8D；mode组LoD2为[零view7,1]；view组LoD2为[真实view7,1]。复用固定场景尺度及同Anchor相机编码：世界方向3、相机深度1、NDC投影中心2、投影包围盒尺度1，编码detach。Selector仍不读取Compensator的48D Feature表。','',
    'Selector原11D首层扩展为19D，32/32及2D输出保留，新增256参数。Compensator保持原物理/Feature51投影、ratio3投影、Residual48累计特征、64/64主体和四Head，增加无Bias的8→64条件投影，在第一层ReLU前注入，新增512参数。总新增768参数；FP32权重3KiB、两个Adam矩约6KiB，不新增Feature表。两组注册参数量相同，但mode组的七个视角通道恒零，相关输入列没有数据梯度，其有效条件自由度较少。','',
    '旧Selector视角残差关闭，旧ViewInput交替调度关闭；新开关为ViewBudget.conditioning=off/mode/view。全局模式不读取相机；训练仍计算单行零条件投影并广播，保留条件参数的零梯度，使局部空候选时Adam历史动量也正常更新。冻结推理可跳过该零投影。局部两网络读取同一份Anchor视角编码，Compensator仅处理实际选中的K行。Scale、Rotation、Other属性解码沿用输入模型的schema，不新增位移限幅或Loss。','',
    '实际源码/配置核查：本轮基线与两新组均为Selector.physical_scale_additive=0、Selector.softplus_scale=1，即s_out=exp(log_s_master)×softplus(0.5413248546129181+u)，带原FP32最小值保护。历史“物理Scale加法”称谓不等于当前实际解码；本轮没有改变该Scale语义。','',
    '关键修改：improvements/selector/spatial_compensation.h、gumbel_network.h、src/view_budget_training.cpp、src/real_lod.cpp、src/gaussian_mapper.cpp及对应Python加载/推理。保留旧checkpoint加载、冻结克隆及独立开关。新增模型局部推理当前使用PyTorch后端；旧Frozen CUDA推理后端显式拒绝此schema，避免静默使用错误输入。生产默认未变。','',
    '## 训练协议与真实曝光','',
    '同一未训练Full11980起点，N=13,298、Anchor=2,790、72训练关键帧。Fine11580步，最终23560，seed0，fresh Adam/RNG，Master可更新、拓扑固定。该入口不是完整SLAM/Adam恢复。训练ratio[.01,.05,.1,.3,.5,.7]各1930步，Full→全局LoD→局部LoD逐路反传，再一次共享Adam。','',
    '全局预算rN，局部预算rNv（名义Anchor视锥候选数）。训练保持原Gumbel-ST和软预算，不使用严格Top-K；实际K随模型变化。原全局GI保留，LI=0，无LI定向VJP或额外Full LI渲染。完整实际曝光见TRAINING_EXPOSURE.csv；Mapping相同不代表选中点曝光完全相同。','',
    '|组别|Full/全局/局部渲染|Master/Selector Adam|GI额外Full|训练墙钟秒|',
    '|---|---:|---:|---:|---:|']
for g in ['mode','view']:
    v=T['groups'][g];f=v['verified']
    lines.append(f"|{names[g]}|{f['full_training_renders']}/{f['lod_training_renders']}/{f['view_budget_training_renders']}|{f['master_updates']}/{f['selector_updates']}|{f.get('importance_full_renders','见state')}|{v['seconds']:.2f}|")
lines+=['','两新组Feature六表Adam步数均为11580/9650/7720/5790/3860/1930。原61个初始状态Tensor的原始数值逐bit一致，其中Selector首层比较原11列；新增列为零。Master初始PLY及11580个相机/ratio决策与既有基线一致。新两组仅模式schema不同，其余初始化状态一致。','',
    '## 各ratio质量','',
    '同792帧逐帧PSNR平均；Bank72、non-Bank720。训练档为1%/5%/10%/30%/50%/70%，其余为未见ratio。','',
    '|ratio|类别|全局原联合|全局仅模式|全局真实视角|局部原联合|局部仅模式|局部真实视角|',
    '|---:|---|---:|---:|---:|---:|---:|---:|']
for r in ratios:
    vals=[I[g,f'{scope}_{r:g}','all']['psnr_mean'] for scope in ['global','view'] for g in groups]
    lines.append(f'|{r:g}|'+('训练' if r in train else '未见')+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
lines+=['|Full|完整地图|'+'|'.join(f"{I[g,'full','all']['psnr_mean']:.4f}" for g in groups)+'|—|—|—|','',
    '### 低ratio的SSIM与LPIPS','',
    '|范围|ratio|组别|PSNR↑|SSIM↑|LPIPS↓|实际K均值|Alpha贡献点数均值|',
    '|---|---:|---|---:|---:|---:|---:|---:|']
for scope in ['global','view']:
    for r in [.01,.05,.1]:
        for g in groups:
            v=I[g,f'{scope}_{r:g}','all']
            lines.append(f'|{scope}|{r:g}|{names[g]}|'+ '|'.join(f'{v[k]:.5f}' for k in ['psnr_mean','ssim_mean','lpips_mean','K_mean','contributing_K_mean'])+'|')
lines+=['','### 局部配对变化','',
    '|比较|ratio|分区|平均ΔPSNR|median|p10|p90|改善视角比例|',
    '|---|---:|---|---:|---:|---:|---:|---:|']
for ref,g in [('baseline','mode'),('baseline','view'),('mode','view')]:
    for r in [.01,.05]:
        for part in ['all','bank','nonbank']:
            v=J[ref,g,f'view_{r:g}',part]
            lines.append(f'|{names[g]}−{names[ref]}|{r:g}|{part}|'+ '|'.join(f'{v[k]:+.4f}' for k in ['delta_psnr_mean','delta_psnr_median','delta_psnr_p10','delta_psnr_p90'])+f"|{v['psnr_improved_fraction']:.1%}|")
lines+=['','## 局部固定点数与匹配预算','',
    '|请求点数|原联合PSNR|仅模式PSNR|真实视角PSNR|原联合实际K|仅模式实际K|真实视角实际K|',
    '|---:|---:|---:|---:|---:|---:|---:|']
for b in [1000,2000,3000,4000,5000,10000]:
    vals=[I[g,f'points_{b}','all'][key] for key in ['psnr_mean','K_mean'] for g in groups]
    lines.append(f'|{b}|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
lines+=['','局部实际K=min(请求B,合格候选E)。10k可能候选不足，不能称实际10k。matched_0.1至matched_0.5按各模型自己全局导出在该帧实际Alpha贡献数作为局部请求预算；它不等于跨模型严格same-K。每行均记录请求K、实际K、投影点数、Alpha贡献点数、选点SHA及网络实际ratio。','',
    '## 几何与训练健康','',
    'GEOMETRY_SUMMARY.csv记录两新组逐视角XYZ/Anchor参考尺度的p50/p90/p99/max、每点最大轴Scale相对Master倍率、Rotation角度及opacity。汇总为各视角统计量的均值，不冒称全体Gaussian样本的联合分位数。为核查新发现的长尾，原基线仅补充1%/5%/10%与2k四种条件的3168次网络前向属性统计，全部选点SHA与既有评价一致，Renderer调用为0，没有重算图像质量。大Scale、低位移或少量长尾本身不能独立解释质量。','',
    '两组记录的LoD非有限Loss/梯度计数为0；训练正常达到固定终点。连续零选点、概率饱和及补偿异常检查仍启用。原硬故障阈值为XYZ/参考尺度>256、Scale倍率>100，至少1%选点越限才中止；本轮长尾尚未触及它们，未报错不代表几何合理。没有根据non-Bank质量修改学习率、Loss、终点或checkpoint。','']
geometry=json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())
geometry+=json.loads((OUT/'BASELINE_GEOMETRY_SUMMARY.json').read_text())['summary']
geo={(v['group'],v['case']):v for v in geometry}
lines+=['|条件|组别|XYZ p50均值|XYZ p90均值|XYZ p99均值|跨视角XYZ最大值|Scale p99均值|','|---|---|---:|---:|---:|---:|---:|']
for case in ['view_0.01','view_0.05','view_0.1','points_2000']:
    for g in groups:
        v=geo[g,case];lines.append(f'|{case}|{names[g]}|'+'|'.join(f'{v[k]:.4f}' for k in ['xyz_p50_mean','xyz_p90_mean','xyz_p99_mean','xyz_global_max','scale_p99_mean'])+'|')
lines+=['','### 长尾核查：实际网络输出与Alpha合成','',
    '固定首/中/末三帧，检查两新模型的1%/5%/10%与2k，共24组属性；另做30次诊断渲染（24个新模型LoD＋6个新模型Full），不保存图片，不重渲染原基线。这24组XYZ与按同一Gaussian行直接调用网络得到的Master＋delta逐bit一致，排除了统计行错配。','',
    '下表仅统计这三帧中XYZ/参考尺度>10的Gaussian−视角样本；10只是诊断分组值，不是事后修改的训练约束或成功门槛。','',
    '|组别|条件|长尾样本|原Full参与Alpha|补偿后参与Alpha|补偿后radii=0|补偿后位于近裁剪面后|','|---|---|---:|---:|---:|---:|---:|']
tail=json.loads((OUT/'TAIL_DIAGNOSTIC_SUMMARY.json').read_text())
for v in tail:
    lines.append(f"|{names[v['group']]}|{v['case']}|{v['tail_records']}|{v['tail_alpha_before']}|{v['tail_alpha_after']}|{v['tail_radii_zero']}|{v['tail_behind_after']}|")
lines+=['','真实视角组5%长尾31个样本原Full全部有贡献，补偿后全部无Alpha贡献，30个radii变0；10%长尾65个样本中原有64个贡献，补偿后全部无贡献。它们并非大量原本没有Full贡献的点。被检查的视角组长尾没有extent_code>9的近面包围盒哨兵输入，未发现该极端编码直接触发这一批异常。','',
    'Opacity降低与几何移出同时存在：三帧2k长尾128个样本中95个opacity<0.01，不能把它们对图像的影响全归因于XYZ，也不能仅凭Scale或位移大就断言图像更差。另一方面，112个radii=0、58个深度低于近面，证明确有几何退出渲染的情况。严格K始终包含这些点，没有把不可见或透明点从预算中扣除。','',
    '全局mode=0也出现了XYZ长尾，见补偿曲线。m=0只保证前向不读取相机，不隔离局部训练对共享主干、Feature和Master的影响；因此不能把问题简单归为推理时直接加入了视角值。','',
    '已证实：新场在部分选点上预测出很大的、使其退出实际渲染的位移。合理假设：局部Photo目标可能允许补偿通过移出画面来抑制部分点，同时在其他点上改善拟合；不能据此断言全部PSNR收益来自这种行为，也不能把大位移预测当作必要位移真值。共享特征外推、选点变化和视角投影各自的责任尚未隔离。','',
    '## 测试与性能','',
    f"原生测试通过{len(V['tests'])}项，另有两组各36步checkpoint smoke；4套CPU/CUDA C++与Python fixture验证Selector和Compensator输出逐bit一致，{len(F['inference_cases'])}项短程推理覆盖空预算、固定预算及训练/未见ratio。普通Renderer与计数Renderer共204次颜色/radii逐bit检查。",'',
    '测试覆盖原始参数保留、零输入列/投影初始化、RNG隔离、原Master输入/Feature梯度、相机编码detach、条件参数可学习、动态K/空候选、保存恢复及冻结克隆。此处跨语言输出一致不等于宣称新架构训练轨迹与旧架构逐bit一致；二者后续有意学习不同函数。','',
    '开发中修正测试的空Tensor赋值、零初始化Head梯度断言、短程相机CSV前缀比较、旧测试日志目录复用和TorchScript fixture保留字段名；保留失败现场。历史静态库缺少19D，已在独立目录重编译当前CUDA对象，旧11D/21D十组前向/VJP共90个Tensor与历史库逐bit一致。评价helper的旧模块路径覆盖也已修正并保存SOURCE_MODULES.json；最终模型另有60组实际输入钩子检查，两网络条件相同且选点SHA与评价完全一致。正式训练的初始化、Loss和更新策略未因测试修复改变。','',
    '|组别|完整Mapping平均ms|median ms|p90 ms|',
    '|---|---:|---:|---:|']
for v in timing:lines.append(f"|{names[v['group']]}|{v['mapping_ms_mean']:.3f}|{v['mapping_ms_median']:.3f}|{v['mapping_ms_p90']:.3f}|")
lines+=['','以上为实际记录；历史基线和新实验不在同一时刻，共享GPU负载也可能变化，不能直接把墙钟差值解释为网络结构加速/减速。未进行本轮Nsight算子归因或峰值显存配对测量。训练的Residual48输入融合路径保留，没有新增渲染。','',
    '## 局限与下一步','',
    '- 最有信息价值的下一步是分别训练Selector真实视角＋Compensator仅模式、Selector仅模式＋Compensator真实视角，保持本轮起点/调度/LI=0。这能区分选择收益与补偿几何风险；不能用同一训练模型推理时简单置零替代。',
    '- 单场景、单seed，不能证明多场景或重复运行的稳定性。应对值得保留的候选与基线做配对seed复验。',
    '- 两网络同时加入条件，结果反映组合方案；尚未隔离Selector单独加视角、Compensator单独加视角的贡献。',
    '- Master允许在Fine阶段更新，终态地图可能不同；同预算不等于同地图，当前是联合训练系统的对照，不是冻结Master的纯网络收益证明。',
    '- mode与view参数量相同，但视角组获得额外观测信息及可用输入自由度，不能把差异全部表述为容量完全相同下的纯编码收益。',
    '- 固定checkpoint继续Fine、固定拓扑，未验证在线Tracking、Loop Closure或动态拓扑中的长期表现。',
    '- 视角相关几何可能使相邻相机产生补偿跳变；本轮质量评价不等于时间连续性通过，部署前须沿连续轨迹测量同一Gaussian的补偿变化。',
    '- 首轮LI关闭，不能据此声称Full贡献教师已被修复或值得强监督。','',
    '## 可追溯性与交付','',
    '服务器独立目录：`/home/prmi/zwj/experiments/view_conditioned_joint_20261106`。新模型位于`runs/full_mode/23560_shutdown/ply`和`runs/full_view/23560_shutdown/ply`；完整配置位于configs，source hash、构建SHA、实际/proc映射和初始化/曝光检查位于reports。','',
    '80784条指标中53856条为新渲染、26928条为既有基线。输出完整逐帧指标、配对统计、Bank/non-Bank、实际曝光、质量门槛、补偿长尾和四张PNG/PDF图。不保存渲染帧图片。生产604文件、输入模型和既有结果的前后SHA核验以FINAL_INTEGRITY.json为最终依据。生产默认未切换，不commit/push。','']
(OUT/'JOINT_VIEW_CONDITIONING_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(OUT/'JOINT_VIEW_CONDITIONING_REPORT.md')
