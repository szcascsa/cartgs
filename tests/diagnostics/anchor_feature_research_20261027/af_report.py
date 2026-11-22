"""从锁定终点与全部重复生成中文报告和可复查统计表。"""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent


def table(data,columns=None,digits=4):
    if columns is not None:data=data[columns]
    def value(v):
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v)
    return '| '+' | '.join(data.columns)+' |\n| '+' | '.join(['---']*len(data.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in data.itertuples(index=False,name=None))


def main():
    exp=pd.read_csv(ROOT/'ANCHOR_FEATURE_EXPERIMENTS.csv');quality=pd.read_csv(ROOT/'quality_macro.csv');ratio=pd.read_csv(ROOT/'quality_per_ratio.csv');pairs=pd.read_csv(ROOT/'paired_deltas.csv')
    assert (exp.status!='NOT COMPLETED').all(),'scheduled experiment missing; report incomplete status explicitly before delivery'
    env=json.loads((ROOT/'environment.json').read_text());decision=json.loads((ROOT/'BANK_LOCKED_CANDIDATE.json').read_text());best=decision['selected']
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(env['started_utc'])).total_seconds()/60
    fixtures={scene:json.loads((ROOT/('fixture_'+scene+'.json')).read_text()) for scene in ['fr1_xyz','fr2_xyz']}
    test=quality[(quality.split=='test')&(quality.ratios=='seven')]
    macro=test.groupby(['tier','scene','config']).agg(seeds=('seed','nunique'),psnr=('psnr','mean'),ssim=('ssim','mean'),lpips=('lpips','mean'),photo=('photo','mean')).reset_index()
    macro.to_csv(ROOT/'nonbank_model_summary.csv',index=False)
    # Scratch and continuation are distinct protocols, never pool them.
    primary=ratio[(ratio.split=='test')&ratio.config.isin(['Baseline','Head8','AnchorFiLM8','FrozenBaseHead8','FrozenBaseContextHead8'])&ratio.tier.isin(['confirm_continue','mechanism_continue','frozen_mechanism'])].groupby(['scene','config','ratio']).agg(psnr=('psnr','mean'),ssim=('ssim','mean'),lpips=('lpips','mean'),photo=('photo','mean'),K=('K','mean'),visible=('visible','mean'),effective=('effective_proxy','mean')).reset_index()
    primary.to_csv(ROOT/'primary_nonbank_per_ratio.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,['fr1_xyz','fr2_xyz']):
        for name in primary.config.unique():
            d=primary[(primary.scene==scene)&(primary.config==name)];ax.plot(d.ratio*100,d.psnr,marker='o',label=name)
        ax.set_title(scene+' | paired seed means');ax.set_xlabel('LoD ratio (%)');ax.set_ylabel('non-Bank PSNR (dB)');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/primary_quality.png',dpi=170);plt.close(fig)
    curves=[]
    for p in ROOT.glob('results/*/*/*/*/validation_curve.csv'):
        meta=json.loads((p.parent/'complete.json').read_text())
        if meta['config']['id'] not in ['Baseline','Head8','AnchorFiLM8','FrozenBaseHead8','FrozenBaseContextHead8']:continue
        if meta['tier'] in ['screen','same_seed_repeat','frozen_same_seed_repeat']:continue
        d=pd.read_csv(p);d['model']=meta['config']['id'];d['scene']=meta['scene'];d['seed']=meta['seed'];d['tier']=meta['tier'];curves.append(d)
    if curves:
        curve=pd.concat(curves);curve.to_csv(ROOT/'all_bank_validation_curves.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,scene in zip(axes,['fr1_xyz','fr2_xyz']):
            for (tier,name),d in curve[(curve.scene==scene)&(curve.tier!='scratch_confirm')].groupby(['tier','model']):
                m=d.groupby('step').psnr.mean();ax.plot(m.index,m.values,marker='o',label=name)
            ax.set_title(scene+' | Bank holdout');ax.set_xlabel('additional real LoD steps');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.2)
        axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/bank_curves.png',dpi=170);plt.close(fig)
    refs=pd.read_csv(ROOT/'references_per_ratio.csv');ref=refs[refs.split=='test'].groupby(['scene','reference'])[['psnr','ssim','lpips','photo']].mean().reset_index()
    performance=[]
    for scene in fixtures:
        d=pd.read_csv(ROOT/f'performance_{scene}.csv');m=d.groupby(['scene','model','stage','ratio','block']).gpu_ms.median().groupby(['scene','model','stage']).mean().reset_index();performance.append(m)
    perf=pd.concat(performance);perf.to_csv(ROOT/'performance_summary.csv',index=False)
    results=pairs[pairs.ratio.isin(['seven','low','unseen'])].copy()
    results.to_csv(ROOT/'paired_macro_deltas.csv',index=False)
    shown=results[(results.ratio=='seven')|((results.model=='FrozenBaseHead8')&results.ratio.isin(['low','unseen']))]
    inventories=[]
    for p in ROOT.glob('results/*/*/*/*/complete.json'):
        d=json.loads(p.read_text())
        for entry in d['parameter_inventory']:inventories.append(dict(tier=d['tier'],scene=d['scene'],model=d['config']['id'],seed=d['seed'],**entry))
    pd.DataFrame(inventories).to_csv(ROOT/'trainable_parameter_inventory.csv',index=False)
    findings=(ROOT/'FINAL_FINDINGS.md').read_text(encoding='utf-8')
    report=f'''# SpatialLocalXYZ + Anchor Feature 研究报告

## 结论范围

本轮基于当前生产默认 **SpatialLocalXYZ**，没有改回59D或E50。生产版本 `{json.loads((ROOT/'TASK_STATE.json').read_text())['local_head']}`；生产源码/配置哈希核对见 `evidence_validation.json`。本轮仅新增独立研究目录，未stage、commit或push。

首次测量起点 `{env['started_utc']}`；报告生成时实际经过 **{elapsed:.1f} 分钟**。完成 {len(exp)} 个训练尝试，总计 {int(exp.steps.sum()):,} 次真实LoD photo backward。预算约180分钟，没有为凑时长扩大维度或网络层数。

**测试前按Bank低ratio配对收益锁定的研究候选：{best}**。该选择记录在 `BANK_LOCKED_CANDIDATE.json`，全部正负对照均评价预定终点。本轮non-Bank视图此前已被历史研究使用，属于独立于本轮训练的留出数据，不能称为全新盲测。continuation两seed仅改变优化帧序；scratch的seed同时决定原网络初始化和帧序。两地图各使用两个seed，并非四次独立SLAM建图。

{findings}

## 1. 当前生产架构与实验边界

生产Compensation为55D物理/空间输入、55→64→64、连续log-ratio FiLM、XYZ/Scale/Rotation/Other四Head、Anchor单位XYZ输出，共33806个共享参数。生产Selector独立使用11D输入，不读取本轮Feature。Variable Anchor真实成员数为1～10；源码split采样读取activated Scale与Rotation；本轮固定地图未运行densification。TF/SI/GI/ChangeDistill默认关闭。

实验使用生产Python同函数的只读快照。`production_input_gate.json`直接调用生产实时physical55并核对缓存55D及最终14D；14组两地图七ratio通过原阈值。`correctness_gates.json`覆盖历史非零网络bitwise输出、函数保持初始化、真实photo Feature梯度及UID/Adam迁移。

本轮冻结Master、Anchor身份/几何、Selector、相机和Cached same-K集合。训练时仅更新Compensation相关参数。实时生产CUDA与Joint Master梯度不在本轮质量实验内；缓存输入只适用于该冻结诊断。归一化沿用历史冻结全地图统计，生产统计则在首次在线补偿时固定，二者建立时点不同。

| 地图 | Gaussian | Anchor | Bank训练视图 | Bank留出 | non-Bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| fr1_xyz | {fixtures['fr1_xyz']['N']} | {fixtures['fr1_xyz']['M']} | {fixtures['fr1_xyz']['train_views']} | {fixtures['fr1_xyz']['bank_holdout_views']} | {fixtures['fr1_xyz']['test_views']} |
| fr2_xyz | {fixtures['fr2_xyz']['N']} | {fixtures['fr2_xyz']['M']} | {fixtures['fr2_xyz']['train_views']} | {fixtures['fr2_xyz']['bank_holdout_views']} | {fixtures['fr2_xyz']['test_views']} |

这些是本轮明确锁定的两张Variable Anchor地图，并非早期13,919 Gaussian/5,177 Anchor地图；没有将不同拓扑的历史质量直接混合。训练ratio1/5/10/15%，未见2.5/7.5/12.5%。选点身份直接复用历史R3终点的固定集合，不重算当前Selector11输出；这是Compensation能力对照，不能称为当前生产Selector11全系统质量。same-K精确排序为 `logit(p)-logit(cached_uniform)`，并非单纯p排序；保留原eligibility和身份。K见逐ratio表。损失 `.8L1+.2(1-fusedSSIM)`，训练320×240，评价640×480。PSNR使用clamp图像，SSIM torchmetrics，LPIPS Alex normalize=True，photo使用原始Renderer输出。

## 2. 发现驱动的实验路线

详见 `RESEARCH_LOG.md`。4k Bank筛选比较Early/Mid/FiLM/Head及Context/Hybrid；Head8与AnchorFiLM8短程约+0.20dB且投影参数都为2048，进入匹配12k验证。fr2反例和同seed重跑反转，促使研究转向优化耦合与信息控制，没有机械扩大Feature维度。

Head8：每Anchor共享8D抽象e，四Head隐藏层分别加共享投影U_t e；e不按ratio存表，ratio仍由原FiLM连续条件化。AnchorFiLM8：在第二层用 `[e,phi(log r)e]` 产生64D gain/bias。Context将成员physical19的mean/max/std与log occupancy（58D）通过共享58→32→8编码，减去冻结初始Encoder输出以保持step0。Hybrid连接learned/context。所有最终14D均由共享神经Head生成，无Gaussian私有物理补偿参数。

继续训练从同一12k终点恢复原base完整Adam，各分支再训练12k；新U与e的Adam从0开始。scratch从头训练12k，同一seed的基线与新模型base初始化完全相同，新分支RNG不扰动base。每ratio均3000次曝光，同seed帧序SHA完全匹配，最终终点预先指定。Dense Adam对M×8整表计时，未访问行梯度为0，但已访问行仍可能因动量更新；访问步数和实际更新数分别保存在feature_diagnostics.pt。

FrozenBaseHead8冻结已经训练为非零的原SpatialLocalXYZ，仅训练U和e；若从生产零输出初始化立刻冻结base，Feature梯度会被零末层永久阻断，不能直接冷启动部署。FrozenBaseContext对应共享Context与U。FreshAdam控制只清空base历史状态，作为优化器诊断，不能与原Adam的差异直接归因为网络。

## 3. non-Bank完整结果：两seed均值

scratch、continuation、冻结base与重复分别列出，不跨协议合并：

{table(macro)}

历史源Spatial12k和R3快照在同地图、同集合、同评价视图重评：

{table(ref)}

R3快照只作质量参照，其训练参数化及历史曝光不同；不能把差距解释为Anchor Feature单独可恢复的理论上限。

## 4. 每ratio质量与实际预算

以下为主要continuation研究臂的两seed均值；scratch和所有控制的完整逐ratio表见 `quality_per_ratio.csv`。实际有效点数为 **radii>0且opacity≥1/255的代理**，不等于经过像素积分测得的真实渲染贡献。

{table(primary)}

![两地图七ratio质量](figures/primary_quality.png)

## 5. 配对差值、波动与训练曲线

每个seed及重复均保留：

{table(shown,['tier','scene','model','seed','ratio','psnr_delta_mean','psnr_delta_median','psnr_delta_p10','psnr_delta_p90','psnr_improved_fraction','ssim_delta_mean','lpips_delta_mean','photo_delta_mean','all_metrics_winner'])}

`paired_deltas.csv`另含逐视图聚类bootstrap区间；相邻时序视图相关，视图区间不能替代独立地图seed不确定性。fr1同seed重跑已出现约0.15dB级别的基线变化及普通Head配对收益反转，因此小幅均值提升不认定稳定收益。

![Bank收敛曲线](figures/bank_curves.png)

1000/6000/12000终点评价均为Bank，未据non-Bank挑步数或学习率。曲线仍变化的模型不能称为表达能力极限。本轮不做学习率搜索。

## 6. Anchor信息机制和occupancy

`feature_statistics.csv`分别记录1成员、2～3、4～10组的访问、非零梯度覆盖、有效秩与实际Adam更新。fr1只有328/2066个Anchor被四个固定训练集合访问，不能将全表平均梯度作为有效学习比例。单成员Anchor的身份记忆作用与多成员共享必须区别。

FakeMatchedHead8在“真实occupancy×四ratio Gaussian选中签名”内打乱Feature归属，精确保留每表行成员数与每档训练访问次数，改变97.7% Gaussian的Feature分组。它保留真实physical55/中心/位移尺度，仅破坏Feature的局部共成员关系。这个控制不等同于仅重命名embedding行；单成员本来就没有共成员关系可破坏。未见ratio访问频次未被强制匹配，解释泛化控制时需保留此限制。

GlobalHead8只有全局一份8D，与原四Head偏置存在冗余参数化；它测优化/全局容量效应，参数量未与完整Anchor表相等。MultiOnly只给2～10成员Anchor启用Feature，避免将收益全部归因于单成员身份。

`mechanism_ablation_summary.csv`给出Feature置零、同occupancy及四ratio Anchor访问签名内打乱、按occupancy禁用、按属性Head禁用、对训练从未选中成员禁用Feature的真实渲染反事实。它们使用已经共同适配的base和U，不能把置零损害等同于新增Feature的因果质量增益，分组贡献也不相加。

每个模型的dense_ratio_continuity.csv对固定1%身份密采141个ratio，检查有限值、相邻输出变化及Anchor单位位移。ratio_gradient.json测一个Bank相机的跨ratio Anchor梯度方向，只作局部诊断；平滑输出及负cosine都不能单独证明泛化或损害。

## 7. 参数、显存与性能

8D表参数8M：fr1为16528、fr2为21792；四Head投影各8×64，总2048。相较33806参数base，learned Head8总量分别52382/57646。Feature权重+梯度+Adam双Moment合计128M bytes，fr1约0.252MiB、fr2约0.333MiB；投影另约0.031MiB。没有ratio乘数。只Gather选中K行，Dense Adam仍扫M行。

以下为三块交错CUDA Events、块内median后平均、七ratio等权：

{table(perf)}

这是冻结55D输入的PyTorch原型成本；Context的58D成员统计也已预计算，聚合成本未计入。生产55D实时CUDA、Master反传和完整Mapping均未计入，不能与生产性能报告中的在线范围直接相除。服务器另有用户GPU进程，未锁时钟或抢占；性能数字是共享负载下的初步测量。精确参数、Adam与显存计数见performance_*.json/csv。FrozenBase减少base梯度，不能把其F+B优势当成同样Joint算法的纯Kernel加速。

## 8. 生命周期与CUDA路径

生产原AnchorBank已有按UID/存活行迁移参数和Adam的机制；本轮独立helper实际验证reorder/prune/append的Feature与moment、优化器alias替换、新行零初始化和旧tensor级step保留；lifecycle_followup.json进一步通过迁移后真实Adam更新及checkpoint恢复后的下一步更新。没有完成生产新Feature接入。

建议独立 `compensation_anchor_features[M,8]`，与Selector旧表分开注册；新Anchor默认e=0、U小非零，初始恢复原Spatial函数。clone/split留在同Anchor时共享原e；新Anchor zero；grow不迁移已有成员。prune/reorder按稳定UID重排行；空间key碰撞不能充当身份。

局部闭环成员迁移可沿用当前成员数加权Feature聚合并清接收行moments，但**抽象latent的线性加权不保证神经补偿连续或质量保持**，仍需事件级验证。整Anchor刚体移动保留UID/e，物理55D和Context按最新几何重算。新行使用共享tensor Adam时钟并不等于独立年龄，需要在后续在线研究明确，而不能伪造零梯度step。

CUDA接入方案：保留现有live55D Kernel，在K行读取owner→e；四Head GEMM前融合U_t e相加，Backward计算U梯度并对重复owner Scatter/Add。Context需在成员/属性变化后重算有版本的聚合，不能读取过期缓存；Dense Adam可先复用，Sparse更新改变moment时钟，不应作为等价优化暗中替换。

## 9. 验证状态与局限

- **IMPLEMENTED / TESTED**：独立Feature原型、函数保持初始化、真实photo训练、两地图两顺序seed、从零对照、访问匹配假分组、冻结base、全局和Context控制、同seed重复、固定K、逐视图PSNR/SSIM/LPIPS/photo、梯度/更新/occupancy统计、输入同函数gate和UID helper。
- **PARTIALLY VERIFIED**：连续ratio在1%～15%七档及固定身份密采；不代表所有ratio质量保证。当前性能是共享GPU冻结输入原型；生命周期helper未覆盖真实SLAM事件。
- **NOT IMPLEMENTED**：生产Feature集成、在线Joint质量、闭环Feature混合质量、Sparse Adam和新CUDA Feature Kernel、独立第三地图盲测。

本轮模型都由共享网络生成14D；允许的Anchor抽象表与禁止的Gaussian直接物理残差严格区分。单成员Anchor可能发挥Gaussian身份记忆，不能隐去这一边界。

## 10. 逐项因果判断

逐项判断已在文首人工核验结论中列出。完整原始证据保留于模型目录，推荐方案仅为下一轮研究候选，不修改生产默认。57个尝试中，7个为4k筛选，其余50个为12k正式/机制对照；关键发现已完成两地图、两seed、从零与同seed重复。因低ratio与LPIPS尚不足以进入生产，停止扩大结构搜索，提前整理交付，没有继续用剩余预算增加维度或选择性重跑。
'''
    (ROOT/'ANCHOR_FEATURE_RESEARCH_REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'report_statistics.json').write_text(json.dumps(dict(elapsed_minutes_at_report=elapsed,runs=len(exp),real_lod_steps=int(exp.steps.sum()),bank_selected=best,production_modified=False),indent=2))
    print('REPORT_DRAFT_COMPLETE',best,elapsed,flush=True)
if __name__=='__main__':main()
