"""汇总已完成的矩阵；不根据测试集重新选结构。"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
OUT=ROOT/'analysis'

def read(path):return list(csv.DictReader(Path(path).open()))
def number(x,d=3):return f'{float(x):.{d}f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def aggregate(rows,**filters):return [r for r in rows if all(str(r[k])==str(v) for k,v in filters.items())]

def main():
    summary=json.loads((OUT/'statistical_summary.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    lock=json.loads((ROOT/'winner_lock.json').read_text())
    chosen=lock['winner'];winner=chosen['id']
    screen=read(ROOT/'screen_summary.csv');quality=read(OUT/'quality_summary.csv')
    measured_count=len({r['id'] for r in screen}|{r['config']['id'] for r in lock['scores'] if r['config']['id']!='R3'})
    convergence=read(OUT/'validation_convergence.csv');fair=read(OUT/'fairness_checks.csv')
    sources=json.loads((OUT/'source_topology.json').read_text());full=read(OUT/'source_full_per_view.csv')
    output_checks=read(OUT/'network_output_diagnostics.csv')
    invalid=[r for r in output_checks if int(r['nonfinite_activated_rows']) or r['probabilities_finite']!='True']
    baseline='S16R00E14D11';latest='S16R08E14D11'
    confirmation=summary['confirmation_validation'];independent=summary['independent_fr2_test']
    unresolved=[r['reference'] for r in confirmation['primary'] if r['reference']!='R3' and r['ci95_low']<=0<=r['ci95_high']]
    qsummary=[]
    for scene,split in [('fr1_xyz','validation'),('fr1_xyz','test'),('fr2_xyz','test')]:
        subset=aggregate(quality,scene=scene,split=split,policy='same_k')
        for config in sorted({r['config'] for r in subset}):
            rows=aggregate(subset,config=config)
            groups=[('all',[.01,.025,.05,.075,.1,.125,.15]),('train',[.01,.05,.1,.15]),('unseen',[.025,.075,.125])]
            for label,ratios in groups:
                selected=[r for r in rows if float(r['ratio']) in ratios]
                qsummary.append(dict(scene=scene,split=split,config=config,group=label,
                    **{m:float(np.mean([float(r[m]) for r in selected])) for m in ['psnr','ssim','lpips','photo']}))
    with (OUT/'group_quality_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(qsummary[0]));w.writeheader();w.writerows(qsummary)

    def group_value(scene,split,config,group,metric='psnr'):
        return next(r[metric] for r in qsummary if all(r[k]==v for k,v in dict(scene=scene,split=split,config=config,group=group).items()))
    def primary(reference):return next((r for r in independent['primary'] if r['reference']==reference),None)
    primary_baseline=primary(baseline)
    r3_difference=primary('R3')
    stable=primary_baseline is not None and primary_baseline['ci95_low']>0 and primary_baseline['holm_primary_p']<=.05
    practical=stable and primary_baseline['ci95_low']>.1
    external_message=('另一场景支持七档平均收益，且95%区间下界超过预设0.1 dB实用阈值。' if practical else
        '另一场景支持七档平均收益，但不能确认收益稳定超过0.1 dB。' if stable else
        '另一场景尚未证实相对原16维共享控制具有稳定收益；不将内部验证的获胜配置直接升级为生产默认。')
    if winner==baseline:external_message='内部验证仍选择原16维共享控制；本轮没有证明需要替换该结构。'

    lines=[
        '# Selector 特征维度与 Head 深度矩阵研究',
        '',
        '## 结论与推荐边界',
        '',
        f'内部验证锁定的复验候选最佳神经配置为 **{winner}**：Selector 的 Anchor 特征为 **{chosen["shared"]}维共享 + {chosen["ratio"]}维/ratio**，Compensation 额外输入 **{chosen["extra"]}维/Anchor/ratio**；选择私有头 **{chosen["selector_depth"]}个**、补偿私有头 **{chosen["compensation_depth"]}个**32维隐藏层。Encoder 仍为单层32维。',
        '',external_message,'',
        f'相对同地图、同步数重新训练的R3独立14D，本候选在fr2七档same-K的ΔPSNR为 **{r3_difference["delta_psnr"]:+.3f} dB**，95%CI **[{r3_difference["ci95_low"]:+.3f}, {r3_difference["ci95_high"]:+.3f}]**。最佳MLP配置与是否接近R3是两个独立问题。','',
        f'这只是本轮两张冻结地图、固定初始化、Adam与训练预算下的结果，不是所有场景或充分收敛条件下的全局最优；720种组合中共实测{measured_count}种（72种初筛及补充控制），最终{len(lock["scores"])-1}种神经配置进入统一12000步复验。不能把统计模型预测当作未运行组合的真实质量。性能不参与配置选择。',
        '',
        '## 1. 实验边界与公平性',
        '',
        '- 生产代码、默认配置、Loss、Cached Bernoulli、TF和Joint流程未修改。实验原型仅位于本诊断目录。',
        '- 当前源地图为Variable Anchor、显式Gaussian；使用已有checkpoint身份、Anchor归属、空间参考与停用eligibility。实验中关闭所有拓扑变化并精确冻结Master，不把在线不同地图的历史PSNR直接相减。',
        '- 所有网络从相同初始选择函数 `p=0.5`（eligible行）和零补偿开始。未复用某方案已训练的特征/网络去对照其他方案的随机初始化。',
        '- 初筛每个模型4000次真实backward/Adam step、每ratio1000次；复验和独立场景全部从零训练12000步、每ratio3000次。阶段之间不作不同步数的胜负比较。',
        '- 这里的step是一次LoD优化，不含Full更新。补偿从隔离训练第1步启用，不采用在线3000 Mapping步warmup；所有对照一致。它是固定地图的网络/特征研究，不是完整SLAM Joint训练的替代结论。',
        '- 共享参数每步参与；各ratio独立特征表仅该ratio曝光时获得grad，其他表grad为None。真实Adam时钟单独保存；不使用伪零梯度step补齐。相同步数不代表不同参数化拥有同样的每参数曝光。',
        '- 两个CUDA进程只并发执行独立模型；各模型Loss、优化器、帧序列与步数不合并。早期已完成的串行模型保留，未完成模型从零重跑，见execution_amendment.json。',
        '- 每cycle共用一幅训练相机，依次1%/5%/10%/15%；同seed的所有结构使用相同相机顺序、uniform和预算定义。Half-res训练320×240，Full-res评价640×480。',
        '',
        f'已核验 **{summary["fairness"]["verified_models"]}个**完成模型：步数、每ratio曝光和Master冻结均通过；详见 `analysis/fairness_checks.csv`、每模型 `complete.json`。',
        '',
        f'对最终模型逐ratio、逐选择策略共检查{len(output_checks)}组输出，包含指数尺度与归一化旋转等激活属性；异常组数为 **{len(invalid)}**。原始位移/旋转/尺度幅度见 `analysis/network_output_diagnostics.csv`。训练Loss/梯度有限值检查与激活有效性分别记录；若某候选存在异常，不把其低质量直接归为容量不足。',
        '',
        table(['共同源地图','Gaussian','有成员Anchor / 特征表行','Full non-Bank PSNR（固定）'],[
            [r['scene'],r['N'],f'{r["occupied_anchor_rows"]} / {r["feature_rows"]}',number(np.mean([float(x['psnr']) for x in full if x['scene']==r['scene'] and x['split']=='non-Bank']))] for r in sources]),
        '',
        '## 2. 搜索空间与明确结构',
        '',
        table(['因子','取值','含义'],[
            ['S','0 / 8 / 16 / 32','每Anchor一份共享特征'],
            ['R','0 / 4 / 8 / 16','每Anchor每训练ratio一份特征，插值后送Encoder'],
            ['E','0 / 4 / 8 / 14 / 28','只给补偿头的额外Anchor×ratio特征'],
            ['Ds','0 / 1 / 2','共享Encoder之后，选择私有头的隐藏层数'],
            ['Dc','0 / 1 / 2','共享Encoder之后，补偿私有头的隐藏层数'],
        ]),
        '',
        '设固定属性 `a_i=[offset(3), scale(3), SH_DC(3), opacity(1)]`，则',
        '',
        '```text',
        'x_i(r) = [f_shared[a(i)] (S), f_ratio[a(i),r] (R), a_i (10), log(r) (1)]',
        'h_i(r) = ReLU(Linear_(S+R+11 → 32)(x_i(r)))',
        'p_i(r) = softmax(SelectorHead_Ds(h_i(r)))_1',
        'delta_i(r) = CompensationHead_Dc([h_i(r), extra[a(i),r]]) ∈ R14',
        '```',
        '',
        f'推荐配置的Encoder输入 **{chosen["shared"]+chosen["ratio"]+11}维**，补偿头输入 **{32+chosen["extra"]}维**，输出仍是原14维属性补偿。私有深度0为直接Linear输出，1为Linear→ReLU→Linear，2再增加一层Linear→ReLU。',
        '',
        '“Selector全部共享”仅指R=0；E仍可能非零，不能误称整张网络全部ratio共享。S=R=0时仍有10维属性和log(r)。每Anchor latent总存储维度为 `S+4R+4E`，当前ratio送入Encoder的只是 `S+R`。',
        '',
        '未见2.5%/7.5%/12.5%对相邻训练节点的潜在特征作raw-ratio线性插值，再解码；显式log(r)始终保留。本轮不扫描ratio编码，不扩大网络宽度。',
        '',
        '## 3. 统计设计与数据划分',
        '',
        f'预先用D-optimal选出72配置，含指定控制；分类主效应及5组交互共48列，设计rank={protocol["design_rank"]}、condition={protocol["condition_number"]:.3f}。72×2个优化顺序seed初筛，最终复验使用8个全新seed，避免只选某个seed的最高值。',
        '',
        '初筛后发现效应模型外推不足，在未查看任何non-Bank质量前补充5个Bank内单因子控制：S16R0E14固定时分别改Ds为0/2、Dc为0/2；另以S16R0E8D12对照S0R0E8D12，隔离Selector共享特征的影响。每个补充模型同样12000步、8seed。补充决定及时间写入protocol.json；它是Bank驱动的后续设计，不冒充最初72配置的一部分。最终结构仍只按Bank验证选择。',
        '',
        '- fr1_xyz：同一6838 Gaussian / 2066 Anchor源地图；67幅Bank相机按时间每第五幅留出，54幅训练、13幅内部验证。源Master此前见过全部Bank，留出隔离的是本次辅助网络训练。',
        '- 结构仅按最终步内部验证七档等权PSNR选择。fr1历史45幅non-Bank在结构锁定后才打开；这些图已被此前研究多次使用，不视为全新独立确认。',
        '- 独立fr2_xyz：预先新建一份9275 Gaussian / 2724 Anchor地图后冻结，47幅训练、12幅Bank验证、64幅预先固定non-Bank。只测已锁定配置、原16维共享控制、匹配R3，不按该场景重新选结构。',
        '- 源地图分别来自fr1自然停止的6400与fr2自然停止的30900 Mapping快照；源图生成次数不作为网络对照。每个场景内先固定共同地图，再执行完全相同的LoD训练预算。生产Mapping上限仍为0，未强制10000步。',
        '- 初筛seed2026/2027；内部复验3100–3107；独立场景4100–4107。它们是优化顺序/初始化重复，不是8张独立SLAM地图。',
        '- 最佳频率为对配对seed及连续相机块分别重采样后的获胜频率；所有结构共用相同重采样权重。它只比较进入复验的神经配置，不含R3、未测组合或所有场景，不是“真实最优”的后验概率。PSNR效应报告95%区间；独立场景七档平均用8seed符号置换及Holm校正，置换检验依赖零假设下配对差值可交换/对称。',
        '- 预先规定0.1 dB实用阈值。区间跨0表示未分辨，不表示等价；只有95%区间完全位于±0.1 dB内才称本协议下实用等价。',
        '',
        '## 4. 初筛结果与因子效应',
        '',
        table(['配置','验证PSNR','seed SD','参数量'],[[r['id'],number(r['mean_validation_psnr']),number(r['seed_std']),r['parameters']] for r in screen[:12]]),
        '',
        '![Screen results](analysis/screen_results.png)',
        '',
    ]
    model=summary['effect_model']
    effects=read(OUT/'factor_effects.csv')
    lines += [f'48列交互模型的配置分组leave-one-out R²={model["grouped_loo_r2"]:.3f}，RMSE={model["grouped_loo_rmse_db"]:.3f} dB。'+('预测能力不足，不能据此为未测组合排序或将边际效应解释为普适最优维度。' if model['grouped_loo_r2']<.5 else '可用于解释本轮趋势，但不替代真实渲染复验。'),'',
        table(['因子','模型调整均值最高的取值','该均值PSNR','95%模型区间'],[
            [factor,best['level'],number(best['model_adjusted_validation_psnr']),f'[{float(best["ci_low"]):.3f}, {float(best["ci_high"]):.3f}]']
            for factor in protocol['factors']
            for best in [max([x for x in effects if x['factor']==factor],key=lambda x:float(x['model_adjusted_validation_psnr']))]]),'',
        '上表是平均其他因子后的探索性模型趋势，不能把各行最高取值拼成一个未训练的“最优网络”。同其他四因子完全一致的单因子差值另存 `analysis/matched_screen_contrasts.csv`；它们只有2个初筛seed。完整调整均值与参考水平差值见 `factor_effects.csv`、`factor_level_comparisons.csv`，区间依赖预注册线性效应模型。','',
        '## 5. 统一12000步复验与独立场景','',
        '内部验证的模型均值、重复波动和Bootstrap获胜频率：','',
        table(['配置','七档PSNR','seed SD','Bootstrap获胜频率（仅复验神经候选）'],[
            [r['config']['id'],number(r['mean_validation_psnr']),number(r['seed_std']),number(confirmation['best_frequency'].get(r['config']['id'],0))]
            for r in sorted([x for x in lock['scores'] if x['config']['id']!='R3'],key=lambda x:-x['mean_validation_psnr'])]),'',
        '与首名的配对95%区间仍跨0的候选：'+('、'.join(unresolved) if unresolved else '本次复验中无')+'。这是探索性未分辨集合，不是已证明等价的集合，也不是对720组合的同时置信集。','',
        table(['场景/划分','配置','训练ratio PSNR','未见ratio PSNR','七档平均PSNR'],[
            [scene+'/'+split,c,number(group_value(scene,split,c,'train')),number(group_value(scene,split,c,'unseen')),number(group_value(scene,split,c,'all'))]
            for scene,split in [('fr1_xyz','validation'),('fr1_xyz','test'),('fr2_xyz','test')]
            for c in sorted({r['config'] for r in qsummary if r['scene']==scene and r['split']==split})]),'',
        '![Held-out quality](analysis/quality_curves.png)','',
        '锁定配置和参考在non-Bank上的四指标（七档等权）：','',
        table(['场景','配置','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[
            [r['scene'],r['config'],number(r['psnr']),number(r['ssim'],4),number(r['lpips'],4),number(r['photo'],4)]
            for r in qsummary if r['split']=='test' and r['group']=='all']), '',
        '独立场景相对控制的主要比较（正值有利于锁定配置）：','',
        table(['控制','ΔPSNR','95%CI','P_boot(Δ>0.1dB)','Holm p'],[
            [r['reference'],number(r['delta_psnr']),f'[{r["ci95_low"]:.3f}, {r["ci95_high"]:.3f}]',number(r['bootstrap_over_0p1db']),number(r['holm_primary_p'],4)]
            for r in independent['primary']]),'',
        '以上same-K排序为 `logit(p)-logit(cached_uniform)`，不是直接Top-K(p)，K为同地图 `floor(r*N)`（受共同eligibility上限约束）。训练仍使用原Cached Bernoulli。两种结果均保存，不能用Cached更大的实际K冒充结构质量提升。','',
        table(['场景','ratio','配置','same-K PSNR','Cached PSNR','Cached实际K'],[
            [scene,f'{r*100:g}%',c,
             number(next(float(x['psnr']) for x in quality if x['scene']==scene and x['split']=='test' and x['config']==c and x['policy']=='same_k' and float(x['ratio'])==r)),
             number(next(float(x['psnr']) for x in quality if x['scene']==scene and x['split']=='test' and x['config']==c and x['policy']=='cached' and float(x['ratio'])==r)),
             number(next(float(x['selected']) for x in quality if x['scene']==scene and x['split']=='test' and x['config']==c and x['policy']=='cached' and float(x['ratio'])==r),1)]
            for scene in ['fr1_xyz','fr2_xyz'] for r in [.01,.05,.15] for c in [winner,'R3']]),'',
        '逐ratio四指标与seed标准差见 `analysis/quality_summary.csv`；逐视图mean/median/p10/p90/改善比例见 `paired_view_descriptive.csv`。Visible及opacity阈值proxy不等于精确渲染贡献，未以proxy替代质量。','',
        '## 6. 深度、收敛与“最优数值”的限制','',
    ]
    matched=read(OUT/'matched_confirm_controls.csv')
    lines += ['固定其他设置的12000步、8seed单变量控制（正值有利于候选）：','',
        table(['基线','只改一个因子的候选','ΔPSNR','95%CI','改善seed数','Holm p'],[
            [r['baseline'],r['candidate'],number(r['delta_psnr']),f'[{float(r["ci95_low"]):.3f}, {float(r["ci95_high"]):.3f}]',f'{r["seed_improved"]}/{r["seeds"]}',number(r['holm_p'],4)] for r in matched]),'',
        '这些比较使用Bank内部留出视图，属于当前初始化/训练预算下的单变量证据；区间为未校正95%区间，另列7项比较的Holm p。8seed精确符号置换的最小双侧p为2/256，多重校正后功效有限，不能只按区间不跨0就宣称所有因子均已确证。独立场景只确认锁定配置整体，不据此声称每一个结构因子的跨场景因果效果均已确认。','']
    rows=[]
    for c in sorted({r['config'] for r in convergence if r['scene']=='fr1_xyz'}):
        means={step:np.mean([float(r['psnr']) for r in convergence if r['scene']=='fr1_xyz' and r['config']==c and r['group']=='all' and int(r['step'])==step]) for step in [4000,8000,12000]}
        rows.append([c,*[number(means[x]) for x in [4000,8000,12000]],number(means[12000]-means[8000])])
    lines += [table(['配置','4000步','8000步','12000步','最后4000步增益'],rows),'',
        '![Convergence](analysis/convergence.png)','',
        '上述曲线用于识别固定预算下的优化差异，不用于重新选checkpoint。如果最后4000步仍有明显增益，不能宣称表达能力已到上限；本轮推荐只对等12000步、相同Adam设置有效。特征越多、私有层越深并不自动意味着同样步数下收敛更充分。','',
        '4000步初筛可能漏掉较晚收敛的组合；强制保留深头代表和补充单因子控制可以降低该风险，但不能消除。Hidden width固定32，并未在本轮被证明为最优宽度。','',
        '更大输入/网络通常能通过适当置零或恒等映射包含较小模型，因此负收益不能直接解释为表达能力变差；它也可能来自优化耦合或泛化。删除特征在本轮更好，同样不证明这些特征本身没有信息。本矩阵给出预算内的实测选型，不将它包装成容量上限归因。','',
        'R3参考也在同一冻结地图从相同初始p和零补偿重新训练，保留每Gaussian/ratio独立14D，不是读取此前不同训练状态的Oracle再相减。它与MLP参数共享约束不同，只作为当前预算下的质量参考，不是理论上界。','',
        '## 7. 交付与下一步','',
        f'- 实测推荐：`S={chosen["shared"]}, R={chosen["ratio"]}, E={chosen["extra"]}, Ds={chosen["selector_depth"]}, Dc={chosen["compensation_depth"]}`；是否具备跨场景优势以上述独立场景区间为准。',
        '- 仅观察了两个场景各一张冻结地图；内部seed频率不能代表跨地图统计。没有据此声称在线Joint质量已提高，也没有自动修改生产默认。',
        '- 两张源地图都来自默认R3在线流程。结构排序条件于这类共同Master，不能推断这些网络参与Joint地图增长后仍保持同样排序。',
        '- 不用本次测试集继续搜索维度或checkpoint。若要进入在线对照，应冻结本报告选中的结构，在新SLAM重复中保留相同Mapping预算并报告实际拓扑/优化次数。',
        '- 完整协议、72配置、代码、原始逐视图指标、Adam时钟、选择曝光、source SHA和统计产物随实验保存；生产源文件SHA与本轮开始一致。未stage、commit、push。',
        '',
        '本轮只回答受控网络/特征设置问题；未加入性能择优、新Loss、采样算法或更复杂ratio调制。',''
    ]
    (ROOT/'SELECTOR_CAPACITY_MATRIX_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    plots(screen,quality,convergence,winner)
    print('REPORT_COMPLETE',winner,external_message,flush=True)

def plots(screen,quality,curves,winner):
    plt.rcParams.update({'font.size':10,'figure.dpi':160})
    fig,ax=plt.subplots(figsize=(12,5));top=screen[:20]
    ax.errorbar(range(len(top)),[float(x['mean_validation_psnr']) for x in top],yerr=[float(x['seed_std']) for x in top],fmt='o',capsize=2)
    ax.set_xticks(range(len(top)),[r['id'] for r in top],rotation=65,ha='right');ax.set_ylabel('Held-out Bank PSNR (dB)');ax.set_title('Screen: 4,000 steps, 2 seeds; top 20 measured configurations');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'screen_results.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,['fr1_xyz','fr2_xyz']):
        data=aggregate(quality,scene=scene,split='test',policy='same_k')
        for c in sorted({r['config'] for r in data}):
            x=sorted(aggregate(data,config=c),key=lambda r:float(r['ratio']))
            ax.errorbar([100*float(r['ratio']) for r in x],[float(r['psnr']) for r in x],yerr=[float(r['psnr_seed_std']) for r in x],marker='o',capsize=2,label=c)
        for r in [2.5,7.5,12.5]:ax.axvline(r,color='gray',lw=.5,ls='--')
        ax.set_title(scene+' / non-Bank / exact same-K');ax.set_xlabel('Ratio (%)');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'quality_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,['fr1_xyz','fr2_xyz']):
        data=[r for r in curves if r['scene']==scene and r['group']=='all']
        for c in sorted({r['config'] for r in data}):
            y=[np.mean([float(r['psnr']) for r in data if r['config']==c and int(r['step'])==s]) for s in [4000,8000,12000]]
            ax.plot([4000,8000,12000],y,marker='o',label=c)
        ax.set_title(scene+' / held-out Bank');ax.set_xlabel('Actual optimizer steps');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'convergence.png');plt.close(fig)

if __name__=='__main__':main()
