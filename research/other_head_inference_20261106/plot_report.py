"""本地绘制科学曲线并生成中文贡献报告。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/evaluation/reports';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];TRAIN=[.01,.05,.1,.3,.5,.7]
S={(v['variant'],v['case']):v for v in json.loads((OUT/'SUMMARY.json').read_text()) if v['partition']=='all'}
P={(v['case'],v['partition']):v for v in json.loads((OUT/'PAIRED_SUMMARY.json').read_text())}
correct=json.loads((OUT/'CORRECTNESS.json').read_text())
state=json.loads((HERE/'results/evaluation/STATUS.json').read_text())
full=next(v for v in json.loads((OUT/'FULL_REFERENCE.json').read_text())['rows'] if v['partition']=='all')
font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def save_figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(2,2,figsize=(13,9))
for col,scope in enumerate(['global','view']):
    title='全局ratio' if scope=='global' else '名义Anchor局部ratio'
    ax=axes[0,col]
    for variant,label,color,style in [('other_on','完整四Head','#2563EB','-'),('other_off','关闭Other补偿','#D55E00','--')]:
        ax.plot(RATIOS,[S[variant,f'{scope}_{r:g}']['psnr_mean'] for r in RATIOS],label=label,color=color,ls=style,marker='o',ms=3)
    ax.set(title=title,xlabel='Ratio',ylabel='792帧平均PSNR（dB）');ax.legend(frameon=False)
    ax=axes[1,col]
    ax.plot(RATIOS,[P[f'{scope}_{r:g}','all']['delta_psnr_mean'] for r in RATIOS],color='#009E73',marker='o',ms=3)
    ax.axhline(0,color='#64748B',lw=1);ax.set(title=title+'｜Other保留 − 关闭',xlabel='Ratio',ylabel='ΔPSNR（dB，正值代表Other有帮助）')
for ax in axes.flat:
    ax.set_xticks(TRAIN,[f'{r*100:g}%' for r in TRAIN],rotation=30);ax.grid(color='#E2E8F0')
fig.suptitle('fr1_xyz｜Other Head冻结推理消融：Opacity＋SH DC',fontsize=14)
fig.text(.5,.015,'同一两网络真实视角＋LI checkpoint；N=13,298。逐帧选点、K、XYZ／Scale／Rotation完全相同；未重训。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.95));save_figure(fig,'OTHER_HEAD_RATIO_CONTRIBUTION')

budgets=[1000,2000,3000,4000,5000,10000]
fig,axes=plt.subplots(1,3,figsize=(15,5.3))
for ax,(key,label) in zip(axes,[('psnr','PSNR↑（dB）'),('ssim','SSIM↑'),('lpips','LPIPS↓')]):
    for variant,name,color,style in [('other_on','完整四Head','#2563EB','-'),('other_off','关闭Other补偿','#D55E00','--')]:
        ax.plot(budgets,[S[variant,f'points_{n}'][key+'_mean'] for n in budgets],label=name,color=color,ls=style,marker='o')
    ax.set(xlabel='局部请求点数B',ylabel=label);ax.set_xticks(budgets,[f'{n/1000:g}k' for n in budgets],rotation=30)
    ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('Other Head贡献｜同一局部固定预算的三项质量',fontsize=14)
fig.text(.5,.015,'实际K=min(B,合格候选数)；1k至5k实际K及10k候选不足均记录在CSV；Alpha贡献点数允许随Opacity变化。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.94));save_figure(fig,'OTHER_HEAD_FIXED_BUDGET')

lines=['# Other Head冻结推理贡献报告','',
    '本轮选择fr1_xyz RGB-D、seed0、两网络真实视角＋LI=0.1，Fine终点23560，N13298、Anchor2790。两个条件在同一进程中配对渲染相同792 RGB帧，Bank72、nonBank720。仅关闭Other的Opacity logit1＋SH DC3补偿，保留Master原Opacity／DC，几何补偿和选点不变。没有重训。','',
    '## 关键结果','',
    '|局部条件|完整四Head PSNR|关闭Other PSNR|保留Other ΔPSNR|ΔSSIM|ΔLPIPS|改善视角比例|','|---|---:|---:|---:|---:|---:|---:|']
gains=[P[c,'all']['delta_psnr_mean'] for c,part in P if part=='all']
lines[4:4]=[
    f'**结论：Other Head提供小幅、可测的外观修正。全部28种条件的平均PSNR收益为{min(gains):.4f}–{max(gains):.4f}dB，当前没有证据显示它贡献了大幅画质提升。**','',
    '局部1%几乎无平均PSNR收益，且保留Other的LPIPS更差；局部5%／10%则PSNR、SSIM、LPIPS均有小幅改善，84.2%／79.3%的视角PSNR提高。固定2k的PSNR、LPIPS改善，但SSIM略降。故不能仅凭平均PSNR为正宣称所有指标全面提高。','',
    '本消融中XYZ、Scale、Rotation逐值相同，因此Other的即时输出不是两条件间几何长尾差异的来源；这不排除Other在此前联合训练中通过共享主干／Feature梯度间接影响几何。权重保留，只屏蔽4维输出，本轮不测删除该Head后的速度。','']
for case in ['view_0.01','view_0.05','view_0.1','points_1000','points_2000','points_3000','points_5000']:
    a,b=S['other_on',case],S['other_off',case];v=P[case,'all']
    lines.append(f"|{case}|{a['psnr_mean']:.4f}|{b['psnr_mean']:.4f}|{v['delta_psnr_mean']:+.4f}|{v['delta_ssim_mean']:+.6f}|{v['delta_lpips_mean']:+.6f}|{100*v['psnr_improved_fraction']:.2f}%|")
lines+=['','Δ均为Other保留−关闭：PSNR／SSIM正值、LPIPS／L1负值表示Other改善质量。它是当前模型对外观补偿的依赖，不是删Head重训后的能力差，也不是总效果的可加百分比。','']
for scope,label in [('global','全局'),('view','局部')]:
    lines += [f'## {label}逐ratio结果','',
        '|ratio|类别|K均值|保留PSNR|关闭PSNR|保留Other ΔPSNR|保留SSIM|关闭SSIM|保留LPIPS|关闭LPIPS|',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in RATIOS:
        case=f'{scope}_{r:g}';a,b=S['other_on',case],S['other_off',case];v=P[case,'all']
        lines.append(f"|{r:g}|{'训练' if r in TRAIN else '未见'}|{a['K_mean']:.2f}|{a['psnr_mean']:.4f}|{b['psnr_mean']:.4f}|{v['delta_psnr_mean']:+.4f}|{a['ssim_mean']:.6f}|{b['ssim_mean']:.6f}|{a['lpips_mean']:.6f}|{b['lpips_mean']:.6f}|")
    lines.append('')
lines+=['## 固定预算与误差变化','',
    '|局部请求B|实际K均值|候选不足帧|保留PSNR|关闭PSNR|保留Other ΔPSNR|平均MSE降低比例|','|---:|---:|---:|---:|---:|---:|---:|']
for n in budgets:
    case=f'points_{n}';a,b=S['other_on',case],S['other_off',case];v=P[case,'all']
    lines.append(f"|{n}|{a['K_mean']:.2f}|{a['budget_shortfall_frames']}|{a['psnr_mean']:.4f}|{b['psnr_mean']:.4f}|{v['delta_psnr_mean']:+.4f}|{100*v['mean_mse_reduction_fraction']:.2f}%|")
lines+=['','MSE降低比例=1−mean(MSE_on)/mean(MSE_off)，由各帧PSNR反算；这是本消融条件下的图像误差变化，不能称整个方法中Other的贡献百分比。实际K相同，参与Alpha合成的点数可因Opacity改变而不同。固定预算的网络ratio逐帧记录，候选不足与超训练区间应单列。','',
    '## 逐帧配对与Bank划分','',
    '|条件|分区|ΔPSNR mean|median|p10|p90|PSNR改善帧比例|','|---|---|---:|---:|---:|---:|---:|']
for c in ['global_0.01','global_0.05','global_0.1','view_0.01','view_0.05','view_0.1','points_2000']:
    for part in ['all','bank','nonbank']:
        v=P[c,part];lines.append(f'|{c}|{part}|'+'|'.join(f'{v[k]:+.4f}' for k in ['delta_psnr_mean','delta_psnr_median','delta_psnr_p10','delta_psnr_p90'])+f"|{100*v['psnr_improved_fraction']:.2f}%|")
lines+=['','同checkpoint逐帧配对不提供跨seed结论。非Bank仅用于固定终点评价，不用于选模型或调参。','',
    '## 外观幅度与渲染参与','',
    '|条件|平均Opacity概率绝对变化|每视角DC残差norm均值|保留Alpha点数均值|关闭Alpha点数均值|','|---|---:|---:|---:|---:|']
for c in ['global_0.01','global_0.05','global_0.1','view_0.01','view_0.05','view_0.1','points_2000']:
    a,b=S['other_on',c],S['other_off',c]
    lines.append(f"|{c}|{a['opacity_delta_abs_mean_mean']:.6f}|{a['dc_delta_norm_mean_mean']:.6f}|{a['contributing_K_mean']:.2f}|{b['contributing_K_mean']:.2f}|")
lines+=['','Opacity改变单点Alpha与后续透射率，SH DC改变颜色，两者可相互作用。联合关闭不能唯一分解Opacity与DC各自的独立贡献，若需要应另做四格配对。本轮没有改变光栅化几何或以实际Alpha点数重新匹配预算。','',
    '## 实现与正确性','',
    '在既有完整渲染输入上仅替换opacities和dc为同一Master对应行原值。其他Tensor直接复用同一对象，并检查逐bit一致。首／中／末三帧×28条件，直接对Compensator输出最后4维归零，渲染输入与替换方式逐bit一致；168次普通／计数Renderer的颜色、radii逐bit一致。22176组选择SHA与K逐一匹配既有结果，配对两条件的选点、候选人口、实际K和network_ratio逐帧相同。模型state_dict前后逐bit未变。','',
    '原完整四Head历史指标最大绝对差：'+json.dumps(correct['historical_on_max_absolute_difference'],ensure_ascii=False)+'。历史batch顺序可能不同，指标误差按实记录，不将该指标阈值冒充颜色逐bit验证。','',
    f"未补偿Full不调用Other，原指标直接复用：PSNR={full['psnr_mean']:.6f}，SSIM={full['ssim_mean']:.6f}，LPIPS={full['lpips_mean']:.6f}；未增加Full渲染。实际新增44352次配对LoD渲染，另168次Renderer验证；报告渲染阶段墙钟{state['seconds']:.2f}秒，不称速度优化结果。","",
    'FP32、关闭TF32、颜色clamp[0,1]，逐帧PSNR平均、TorchMetrics11×11 SSIM、AlexNet LPIPS(normalize=True)、L1，batch8。源码独立复制自已完成实验，未使用当前其他任务的未提交改动。不保存渲染帧图片，不修改生产默认，不commit／push。','',
    '## 文件与后续','',
    '独立服务器目录：`/home/prmi/zwj/experiments/other_head_inference_20261106`。原模型只读复用：`/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply`。PER_VIEW.csv44352行、PAIRED_PER_VIEW.csv22176行，均包含真实K、Alpha点数、相机帧标识与指标。完整性以FINAL_INTEGRITY.json和输入SHA为依据。','',
    '本轮可以判断Other当前是改善还是损害某些推理条件；不能据此预测删Other后重训的质量或训练速度，也不能单独归因于LI。下一步若需定位Opacity／DC，优先在同checkpoint上分别关闭一个分量；若要决定网络简化，则需另做同Full起点、相同训练曝光的无Other重训对照。','',
    '![ratio贡献](figures/OTHER_HEAD_RATIO_CONTRIBUTION.png)','',
    '![固定预算](figures/OTHER_HEAD_FIXED_BUDGET.png)']
(OUT/'OTHER_HEAD_INFERENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('两张曲线与中文报告已生成')
