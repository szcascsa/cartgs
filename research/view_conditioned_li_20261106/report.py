"""三种视角组合的LI=0.1消融中文报告；只读取固定终点指标。"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'evaluation/reports'
groups=['selector_li','compensator_li','both_li']
controls={'selector_li':'selector_view','compensator_li':'compensator_view','both_li':'view'}
names={'selector_li':'Selector视角／Comp模式＋LI','compensator_li':'Selector模式／Comp视角＋LI','both_li':'两网络视角＋LI',
    'selector_view':'Selector视角／Comp模式','compensator_view':'Selector模式／Comp视角','view':'两网络真实视角',
    'baseline':'原无视角联合','mode':'两网络仅模式'}
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];train=[.01,.05,.1,.3,.5,.7]
S={(v['group'],v['case']):v for v in json.loads((OUT/'SUMMARY.json').read_text()) if v['partition']=='all'}
P={(v['reference'],v['group'],v['case'],v['partition']):v for v in json.loads((OUT/'PAIRED_SUMMARY.json').read_text())}
G={(v['group'],v['case']):v for v in json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())}
gates=json.loads((OUT/'QUALITY_GATES.json').read_text())
training=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
timing=json.loads((OUT/'TRAINING_TIMING.json').read_text())
diag=json.loads((OUT/'LI_DIAGNOSTICS.json').read_text())
tail=json.loads((OUT/'TAIL_DIAGNOSTIC_SUMMARY.json').read_text())
lines=['# 视角条件Selector／Compensator加入LI的匹配消融','',
    '## 结果与判断范围','',
    '本轮新增三组LI=0.1，从原Full11980重新训练Fine11580步；相同结构的无LI结果只读复用，没有在无LI终态追加步数。fr1_xyz RGB-D，seed0、相同初始化、相机与ratio决策、N13298、Anchor2790、固定拓扑、Master可更新。结论限于单场景单seed。','',
    '|结构|局部1%无LI→LI|局部5%无LI→LI|局部10%无LI→LI|固定2k无LI→LI|Full无LI→LI|',
    '|---|---:|---:|---:|---:|---:|']
lines[6:6]=[
    '**结论：LI=0.1没有使三种结构的低ratio画质一致改善，不建议据此开启生产默认。三组局部1%／5%／10% PSNR相对各自无LI均下降；部分固定预算、未见ratio或LPIPS有改善，应作为取舍解释。**','',
    '最值得保留作后续对照的是“两网络真实视角＋LI”：5%／10%／2k的XYZ p99明显下降，2k的PSNR、SSIM、LPIPS同时改善，但1%／5%的PSNR、SSIM下降，且全局1%相对同结构无LI下降0.2460dB，未通过预声明的单档−0.20dB保护。它相对原无视角baseline通过保护，并不抵消相对自身无LI对照的退化。','',
    'Selector模式／Comp视角＋LI的2k三项图像指标同向改善，但其低ratio PSNR退化；Selector视角／Comp模式＋LI的2k PSNR下降而SSIM／LPIPS改善，1% XYZ长尾反而增加。三组均没有出现训练零选点或非有限值，因此这些质量取舍不能直接称为Selector坍缩。','',
    'LI能改变选择集合并缓解某些补偿长尾，这是本次匹配训练的证据；还不能证明LI教师显式学会了可补偿性，或几何收敛就是画质变化的唯一原因。训练累计局部K也增加约3%–4%，Master终态随联合训练变化。','']
for g in groups:
    ref=controls[g]
    lines.append('|'+names[ref]+'|'+'|'.join(f"{S[ref,c]['psnr_mean']:.4f} → {S[g,c]['psnr_mean']:.4f}" for c in ['view_0.01','view_0.05','view_0.1','points_2000','full'])+'|')
lines+=['','|LI−同结构无LI|局部1% ΔPSNR|局部5% ΔPSNR|局部训练档均值|局部未见档均值|全局训练档均值|最差全局训练档|Full ΔPSNR|保护|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for g in groups:
    q=next(v for v in gates if v['group']==g and v['reference']==controls[g])
    fields=['view_1pct_delta','view_5pct_delta','view_train_mean_delta','view_unseen_mean_delta','global_train_mean_delta','global_train_min_delta','full_delta']
    lines.append('|'+names[controls[g]]+'|'+'|'.join(f'{q[k]:+.4f}' for k in fields)+'|'+('通过' if q['quality_guard_passed'] else '未通过')+'|')
lines+=['','保护沿用预声明值：全局六训练ratio平均ΔPSNR≥−0.10dB、任一档≥−0.20dB、Full≥−0.10dB；分别对同结构无LI及原无条件baseline报告。局部1%／5%、SSIM、LPIPS及几何不被这个全局保护替代，不以均值掩盖低ratio退化。','']
lines+=['|相对原无条件baseline|局部1% ΔPSNR|局部5% ΔPSNR|全局训练档均值|最差全局训练档|Full ΔPSNR|保护|',
    '|---|---:|---:|---:|---:|---:|---|']
for g in groups:
    q=next(v for v in gates if v['group']==g and v['reference']=='baseline')
    lines.append('|'+names[g]+'|'+'|'.join(f'{q[k]:+.4f}' for k in ['view_1pct_delta','view_5pct_delta','global_train_mean_delta','global_train_min_delta','full_delta'])+'|'+('通过' if q['quality_guard_passed'] else '未通过')+'|')
lines.append('')
for g in groups:
    ref=controls[g];v1=P[ref,g,'view_0.01','all'];v5=P[ref,g,'view_0.05','all'];v2=P[ref,g,'points_2000','all']
    z5=G[g,'view_0.05']['xyz_p99_mean']-G[ref,'view_0.05']['xyz_p99_mean']
    z1=G[g,'view_0.1']['xyz_p99_mean']-G[ref,'view_0.1']['xyz_p99_mean']
    lines.append(f"- {names[ref]}：LI使局部1%／5% PSNR变化{v1['delta_psnr_mean']:+.4f}／{v5['delta_psnr_mean']:+.4f}dB，2k变化{v2['delta_psnr_mean']:+.4f}dB；5%／10% XYZ p99变化{z5:+.4f}／{z1:+.4f}，低ratio LPIPS变化{v1['delta_lpips_mean']:+.6f}／{v5['delta_lpips_mean']:+.6f}。")
lines+=['','这些是本次训练的实际变化，不能仅凭LI Loss下降或选中贡献质量提高认定方案成功。生产默认未切换，没有权重搜索或按nonBank选择checkpoint。','',
    '## 逐ratio渲染质量','',
    '|ratio|类别|Selector视角无LI|＋LI|Comp视角无LI|＋LI|两者视角无LI|＋LI|',
    '|---:|---|---:|---:|---:|---:|---:|---:|']
for r in ratios:
    lines.append(f"|{r:g}|{'训练' if r in train else '未见'}|"+'|'.join(f"{S[n,f'view_{r:g}']['psnr_mean']:.4f}" for g in groups for n in [controls[g],g])+'|')
lines+=['','|ratio|全局Selector视角无LI|＋LI|Comp视角无LI|＋LI|两者视角无LI|＋LI|',
    '|---:|---:|---:|---:|---:|---:|---:|']
for r in ratios:
    lines.append(f'|{r:g}|'+'|'.join(f"{S[n,f'global_{r:g}']['psnr_mean']:.4f}" for g in groups for n in [controls[g],g])+'|')
lines+=['','|局部ratio|配置|PSNR↑|SSIM↑|LPIPS↓|实际K均值|Alpha点数均值|','|---:|---|---:|---:|---:|---:|---:|']
for r in [.01,.05,.1]:
    for g in groups:
        for n in [controls[g],g]:
            v=S[n,f'view_{r:g}'];lines.append(f'|{r:g}|'+names[n]+'|'+'|'.join(f'{v[k]:.6f}' for k in ['psnr_mean','ssim_mean','lpips_mean','K_mean','contributing_K_mean'])+'|')
lines+=['','## 固定预算与逐视角变化','',
    '|请求B|结构|无LI PSNR|LI PSNR|无LI实际K|LI实际K|','|---:|---|---:|---:|---:|---:|']
for b in [1000,2000,3000,4000,5000,10000]:
    for g in groups:
        a,v=S[controls[g],f'points_{b}'],S[g,f'points_{b}']
        lines.append(f"|{b}|{names[controls[g]]}|{a['psnr_mean']:.4f}|{v['psnr_mean']:.4f}|{a['K_mean']:.3f}|{v['K_mean']:.3f}|")
lines+=['','所有保留点均计入K，实际K=min(B,合格候选E)。5k／10k可能候选不足，不能以请求量冒充实际点数。固定预算的network_ratio逐帧记录，可能超出训练区间，应与ratio插值分开解释。matched五档按每模型自己global实际Alpha点数请求局部预算，不是跨模型same-K，也不保证最终Alpha点数相同。','',
    '|LI−无LI|条件|分区|ΔPSNR mean|median|p10|p90|改善视角比例|相同K比例|','|---|---|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    for case in ['view_0.01','view_0.05','points_2000']:
        for partition in ['all','nonbank']:
            v=P[controls[g],g,case,partition]
            lines.append('|'+names[controls[g]]+'|'+case+'|'+partition+'|'+'|'.join(f'{v[k]:+.4f}' for k in ['delta_psnr_mean','delta_psnr_median','delta_psnr_p10','delta_psnr_p90'])+f"|{100*v['psnr_improved_fraction']:.2f}%|{100*v['same_K_fraction']:.2f}%|")
lines+=['','Bank72、nonBank720，所有模型对应相同792 RGB帧。配对帧分布不等于跨seed置信区间；相同K也不等于相同选点、Master或训练曝光。','',
    '## XYZ、Scale与实际Alpha','',
    '|配置|局部条件|XYZ p50|p90|p99|全视角最大值|Scale p99|','|---|---|---:|---:|---:|---:|---:|']
for case in ['view_0.01','view_0.05','view_0.1','points_2000']:
    for g in groups:
        for n in [controls[g],g]:
            v=G[n,case];lines.append('|'+names[n]+'|'+case+'|'+'|'.join(f'{v[k]:.4f}' for k in ['xyz_p50_mean','xyz_p90_mean','xyz_p99_mean','xyz_global_max','scale_p99_mean'])+'|')
lines+=['','分位数按每视角选点计算后跨792帧取均值，最大值列为全轨迹选点最大值。XYZ单位为所属Anchor边长（2.5／5／10 cm）倍数，不能统一乘0.1换算；Scale为每点最大轴相对Master的倍率。米制分位数、Rotation、opacity及全局XYZ见PER_VIEW.csv／GEOMETRY_SUMMARY.csv。','',
    '|新增模型|条件|XYZ米制p99均值|>10 reference比例|其中有Alpha占全选点|其中radii=0占全选点|','|---|---|---:|---:|---:|---:|']
for g in groups:
    for c in ['view_0.01','view_0.05','view_0.1','points_2000']:
        v=G[g,c];lines.append('|'+names[g]+'|'+c+f"|{v['xyz_meters_p99_mean']:.4f}|"+'|'.join(f'{100*v[k]:.4f}%' for k in ['xyz_over10_fraction_mean','xyz_over10_alpha_fraction_mean','xyz_over10_radii_zero_fraction_mean'])+'|')
lines+=['','|新增组全部33个补偿条件|最大XYZ/reference|出现条件|该条件>10点比例均值|其中有Alpha占全选点|','|---|---:|---|---:|---:|']
for g in groups:
    v=max((v for (n,c),v in G.items() if n==g),key=lambda v:v['xyz_global_max'])
    lines.append('|'+names[g]+f"|{v['xyz_global_max']:.4f}|{v['case']}|{100*v['xyz_over10_fraction_mean']:.5f}%|{100*v['xyz_over10_alpha_fraction_mean']:.5f}%|")
lines+=['','>10仅为事先声明的诊断分组，不是新增Loss、限幅或事后修改的成功门槛。三个比例的分母均为全部选点，不是条件比例。大Scale、低opacity或小位移本身不能代替图像质量。','',
    '|新增模型|三帧条件|尾部样本|原Full有Alpha|补偿后有Alpha|radii=0|深度≤0.01|opacity<0.01|','|---|---|---:|---:|---:|---:|---:|---:|']
for v in tail:
    lines.append('|'+names[v['group']]+'|'+v['case']+'|'+'|'.join(str(v[k]) for k in ['tail_records','tail_alpha_before','tail_alpha_after','tail_radii_zero','tail_behind_after','tail_opacity_below_001'])+'|')
lines+=['','首／中／末三帧仅用于机制核查，同一点可能重复出现。36组XYZ与直接网络Master＋delta逐bit一致；新增36 LoD＋9 Full诊断渲染，不存图片、旧模型重渲染0。几何退出与opacity变化同时存在时分别报告，不把大位移当必需位移真值。','',
    '## LI教师与真实梯度路径','',
    '当前步未补偿Master Full的C_i=Σ_pixels T_iα_i/(HW)，采样全部像素，detach、同相机同拓扑，不使用RGB-D深度。复用已有Full，不额外渲染。局部候选中按C排序，同分按全局GI、再按固定拓扑source row ID；教师K=min(floor(rNv),eligible数)，不足正贡献点时补零贡献点，全部零贡献时跳过。','',
    'LI Loss=Σ|Gumbel-ST mask−label|/max(1,Nv)，权重0.1。教师严格预算不改变训练采样K，训练不投影为Top-K。使用定向VJP只向Selector local_emd和soft_net六个参数Tensor加梯度（包括19D首层），已有Full/global/Photo/Budget梯度保留。LI不直接更新Master、Compensator、Anchor Feature或相机编码；未来选点改变仍可间接改变地图与补偿训练。','',
    '|配置|ratio|LI监督步数|教师K均值|采样K均值|LI Loss均值|贡献诊断样本数|教师补零比例|选中贡献质量占比|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for v in diag['rows']:
    zero=v.get('teacher_zero_fraction_mean');mass=v.get('li_selected_mass_fraction_mean')
    lines.append(f"|{names[v['group']]}|{v['ratio']:g}|{v['supervised']}|{v['li_teacher_K_mean']:.2f}|{v['K_mean']:.2f}|{v['li_loss_mean']:.6f}|{v['diagnostic_samples']}|"+
        ('—' if zero is None else f'{100*zero:.3f}%')+'|'+('—' if mass is None else f'{100*mass:.3f}%')+'|')
lines+=['',diag['diagnostic_scope']+'。教师Loss/K全步记录，贡献/梯度只在首6步及每100步记录；不同ratio采样数明确列出。选中贡献质量占比指未补偿Full贡献质量之和，不是点数，也不等于补偿后实际渲染质量。全局GI使用原opacity×T统计，局部LI使用真实alpha×T，两者不可混称。','',
    '三组5%均有39次贡献诊断，教师补零比例均为0；70%约3.2%教师点为零贡献。现有证据不支持用“大量零贡献教师点”解释5%退化。但贡献为正或覆盖质量高，并不等于该集合在当前Compensator下最适合重建。1%只有首步诊断，不能据此概括该档全训练教师分布。教师K与采样K不同，二者贡献质量占比也不能作为same-K教师优劣比较。','',
    '## 训练曝光、初始化与性能','',
    '每组seed0、相同Full11980重启，Fine11580→23560，fresh Adam/RNG，不称完整SLAM/Adam恢复。72训练关键帧，Master可更新、拓扑固定。Full→global→local同相机同ratio逐路反传后一次Adam；六ratio各1930。Full/global/local各11580，原GI额外Full864，LI只在11580次现有Full内采集。Master/Selector Adam各11580；Feature六表时钟11580/9650/7720/5790/3860/1930。','',
    'Selector19→32→32→2，Compensator Residual48、64/64及四Head保持；输入视角按原schema2/3/4路由。新增LI不增加网络参数；LI当前向量约N×4字节，N13298时约52KiB，Full采集和VJP另有已有Renderer/Autograd临时量，未将该向量大小误称峰值显存增量。无XYZ限幅，Softplus Scale与原配置一致，Maturity及SGI保持关闭。','',
    '35项原生验收、三组36步真实checkpoint smoke、12组跨语言逐bitfixture、90项短程推理及90项最终模型实际输入hook通过。三schema的LI六参数累加正确，所有其他Master/Compensator/Feature已有梯度逐bit保留；Full采集颜色/radii不变，空候选/零贡献跳过、Adam时钟、保存恢复和冻结克隆通过。306组计数/普通Renderer颜色/radii逐bit一致。初始原61状态Tensor、Master文件及11580相机ratio决策与无LI匹配；首步Photo/Budget/Full/global相同，新增LI及总Loss另列。','',
    '|新增组|训练墙钟秒|Mapping mean ms|median ms|p90 ms|累计全局K|累计局部K|','|---|---:|---:|---:|---:|---:|---:|']
exposure=json.loads((OUT/'EXPOSURE_TOTALS.json').read_text())
for g in groups:
    v=next(v for v in timing if v['group']==g);x=next(v for v in exposure if v['group']==g)
    lines.append(f"|{names[g]}|{training['groups'][g]['seconds']:.2f}|{v['mapping_ms_mean']:.3f}|{v['mapping_ms_median']:.3f}|{v['mapping_ms_p90']:.3f}|{x['global_K_exposure_sum']}|{x['local_K_exposure_sum']}|")
lines+=['','|相对同结构无LI|全局累计K变化|局部累计K变化|候选池累计N变化|','|---|---:|---:|---:|']
for g in groups:
    x=next(v for v in exposure if v['group']==g);a=next(v for v in exposure if v['group']==controls[g])
    lines.append('|'+names[controls[g]]+'|'+'|'.join(f"{100*(x[k]/a[k]-1):+.3f}%" for k in ['global_K_exposure_sum','local_K_exposure_sum','candidate_N_exposure_sum'])+'|')
lines+=['','累计K是选中Gaussian次数，不是独立点数；同Mapping步数仍可能有不同曝光。完整各ratio曝光和历史对照耗时见CSV。共享GPU有其他任务，旧实验来自不同时段，本轮没有Nsight/峰值显存匹配测量，不将墙钟差值当LI纯计算开销。','',
    '未补偿Master的终态差异在MASTER_DRIFT.json中仅用CPU量化，不额外渲染；系统质量差异包含后续地图、选点和Feature共同更新。m=0屏蔽前向相机，不隔离共享主干受到的局部训练影响。','',
    '## 局限与后续判断','',
    '本轮没有新增“两网络仅模式＋LI”，因此不能完整分解LI与两网络视角的三因子交互；主证据是三种结构各自LI−无LI的匹配变化。单seed不能排除初始化后CUDA原子次序、训练波动或教师排序近同分的影响，不据此断言唯一根因。','',
    'GI/LI采集统计、教师K、采样K、推理实际K和Alpha贡献点数分别记录。LI教师来自未补偿Full，未直接建模可补偿性；教师监督更强不必然让Compensator更容易局部补偿。若某组指标改善，应优先配对seed复验；若质量或几何退化，依据逐ratio和教师覆盖诊断再设计最小对照，不事后改权重/限幅或直接推广。','',
    '原finite、连续零选点、概率饱和及补偿异常检查保留；原硬故障阈值XYZ/reference>256、Scale倍率>100且1%越限，未触发不等于几何合理。没有根据nonBank挑学习率、权重、终点或checkpoint。','',
    '## 交付与完整性','',
    '服务器独立目录：/home/prmi/zwj/experiments/view_conditioned_li_20261106。新模型位于runs/full_selector_li、runs/full_compensator_li、runs/full_both_li下的23560_shutdown/ply。旧五组训练和评价只读复用。','',
    '评价启动前曾因脚本误指向仅三组的历史CSV而在条数校验处停止；已统一使用common.PREVIOUS指向的五组历史CSV，保留evaluation_recovery_01现场。错误发生在本轮第一张图渲染之前，没有重训或丢弃已完成的新评价，也未改变评价协议。','',
    'PER_VIEW.csv共215424行：80784新评价＋134640历史复用。包括逐帧、Bank/nonBank、逐ratio分布、配对变化、实际K/Alpha、几何和LI诊断；附4张PNG/PDF曲线、完整配置、源码快照、编译/运行身份及输入哈希。604个生产文件与历史输入、编译源和二进制前后核验以FINAL_INTEGRITY.json为依据。不保存渲染帧图片、不切生产默认、不commit/push。']
(OUT/'LI_VIEW_CONDITIONING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('LI中文报告已生成')
