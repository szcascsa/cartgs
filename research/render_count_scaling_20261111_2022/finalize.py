"""核验全部交付证据、相邻反例和完整性，生成最终中文报告。"""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'background_stage'
R=STAGE/'reports'
OUT=R/'delivery'


def read(path):
    return json.loads(path.read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    import formal as first
    first.frozen()
    protected=first.identities()
    spec=importlib.util.spec_from_file_location('final_background_formal',STAGE/'formal.py')
    current=importlib.util.module_from_spec(spec);spec.loader.exec_module(current)
    current.frozen()
    assert read(R/'POST_CHECKS.json')['state']=='complete'
    assert read(R/'INPLACE_BACKGROUND.json')['passed']
    assert read(R/'jobs/causal_balanced01.json')['measurement_valid']
    for tool in ('memcheck','initcheck','synccheck'):
        name='bg_'+tool+'01'
        assert read(ROOT/'reports/jobs'/(name+'.json'))['returncode']==0
        assert 'ERROR SUMMARY: 0 errors' in (ROOT/'logs'/(name+'.log')).read_text()
        assert read(R/(tool+'01')/'BACKGROUND_PASS.json')['passed']
    s=read(OUT/'SUMMARY.json')
    p=pd.read_csv(OUT/'PER_SCENE_RATIO.csv')
    v=pd.read_csv(OUT/'PER_VIEW_VS_FULL.csv')
    w=pd.read_csv(OUT/'WORKLOADS.csv')
    raw=pd.read_csv(OUT/'TIMING_SAMPLES.csv.gz')
    assert len(raw)==337920
    keys=['scene','condition','round','scope','repeat','frame']
    raw['position']=raw.groupby(keys,sort=False).cumcount()
    assert set(raw.position)=={0,1,2,3}
    orders=raw.groupby(['scene','condition','round','scope','variant','position']).size().rename('count').reset_index()
    assert orders['count'].between(14,16).all()
    orders.to_csv(OUT/'ORDER_POSITIONS.csv',index=False)
    offenders=[]
    for (scene,frame,iteration),q in v[(v.variant=='F')&(v.scope=='raster')&v.condition.str.startswith('local')].groupby(['scene','frame','round']):
        old=None
        for _,row in q.sort_values('K').iterrows():
            if old is not None and row.K>old.K and row.ms<old.ms:
                t0=w[(w.scene==scene)&(w.frame==frame)&(w.condition==old.condition)].iloc[0]
                t1=w[(w.scene==scene)&(w.frame==frame)&(w.condition==row.condition)].iloc[0]
                offenders.append(dict(scene=scene,frame=frame,round=iteration,first=old.condition,second=row.condition,
                    K0=old.K,K1=row.K,ms0=old.ms,ms1=row.ms,change_pct=100*(row.ms/old.ms-1),
                    tiles0=t0.tile_instances,tiles1=t1.tile_instances,alpha0=t0.alpha_interactions,alpha1=t1.alpha_interactions))
            old=row
    bad=pd.DataFrame(offenders).sort_values('change_pct')
    bad.to_csv(OUT/'NONMONOTONIC_ADJACENT_CASES.csv',index=False)
    controller=read(R/'FORMAL_CONTROLLER.json')
    excluded=[read(R/'jobs'/(name+'.json')) for name in controller['rejected']]
    (OUT/'EXCLUDED_RUNS.json').write_text(json.dumps(excluded,indent=2))
    gpu=[]
    for name in controller['accepted']:
        for line in (STAGE/'logs'/(name+'_MONITOR.jsonl')).read_text().splitlines():
            gpu.append([float(x.strip().split()[0]) for x in json.loads(line)['gpu'].split(',')])
    environment=dict(clock_sm_mhz=[min(x[2] for x in gpu),max(x[2] for x in gpu)],
                     temperature_c=[min(x[3] for x in gpu),max(x[3] for x in gpu)],
                     observed_gpu_memory_mib_max=max(x[1] for x in gpu))
    (OUT/'ENVIRONMENT.json').write_text(json.dumps(environment,indent=2))
    example=v[(v.scene=='VECtor/corridors-dolly/stereo')&(v.frame==653)&(v.scope=='raster')&
              v.condition.isin(['local0.01','local0.05'])]
    example=example.groupby(['variant','condition']).agg(K=('K','first'),ms=('ms','median')).reset_index()
    example.to_csv(OUT/'EMPTY_CASE_FIX.csv',index=False)
    plot_example(example)
    c=s['correlation']['local/raster']['F']
    primary=s['raster_primary'];pipeline=s['pipeline_primary'];full=s['full']['raster']
    ratios=[]
    for ratio in (.01,.05,.1,.2,.3,.4,.5,.7):
        t=s['conditions']['raster'][f'local{ratio:g}']
        ratios.append([f'{100*ratio:g}%',f"{t['F_p50_ms']:.4f}",f"{t['G_over_F']:.4f}×",
                       f"{t['C_over_F']:.4f}×",f"{(1-1/t['C_over_F'])*100:.2f}%",
                       f"{t['G_slower']}/{t['comparisons']}"])
    scenes=[]
    for scene,q in p[(p.scope=='raster')&p.condition.isin(['local0.1','local0.2','local0.3','local0.4','local0.5'])].groupby('scene'):
        f=p[(p.scene==scene)&(p.scope=='raster')&(p.condition=='full')]
        scenes.append([scene,int(q.N.iloc[0]),f"{np.exp(np.log(q.C_over_F).mean()):.4f}×",
                       f"{np.exp(np.log(q.G_over_F).mean()):.4f}×",f"{max(1/f.G_over_F-1)*100:+.2f}%"])
    exrows=[]
    for name in ('G','C','F1','F'):
        q=example[example.variant==name].set_index('condition')
        exrows.append([name,f"{q.loc['local0.01','ms']:.5f}",f"{q.loc['local0.05','ms']:.5f}"])
    integrity=dict(passed=True,input_files=protected,first_candidate_unchanged=True,previous_G_C_unchanged=True,
                   final_sources_unchanged=True,checkpoint_files_unchanged=True,checked_at=time.time(),
                   production_training_shared_conda_not_modified=True,commit_push_deploy=False,render_images_saved=0)
    (OUT/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
    text=f"""# 渲染耗时与Gaussian数量关系：CUDA优化报告

## 结论

本轮实际实现、编译、逐位验证和双轮测量，最终保留F2作为独立研究入口。主要Ratio 0.1–0.5，纯Rasterizer相对上一轮大footprint版C加速{primary['C_over_F']:.4f}×，耗时降低{(1-1/primary['C_over_F'])*100:.2f}%；相对此前Full安全版G加速{primary['G_over_F']:.4f}×。当前帧完整准备＋渲染相对C降低{(1-1/pipeline['C_over_F'])*100:.2f}%，相对G降低{(1-1/pipeline['G_over_F'])*100:.2f}%。新增节省仍是微秒级，没有达到毫秒级或倍数级普遍提速。

目标解释为“点越少、耗时越低”。固定属性的嵌套删点对照中64/64组严格单调，Spearman均为1；原G也已在该对照单调，不能将其归为本轮新增收益。真实LoD中F2的640组视角/轮次全部正相关，但只有{c['monotonic_views']}/640组逐档单调；4480个相邻K条件仍有{c['reversed_pairs']}个反序，最大{abs(c['worst_reversal_pct']):.2f}%。没有达到任意视角、任意补偿属性都严格单调或线性。

相同测量中G最大反序78.97%，F1为28.75%；F2修复了极小K空渲染的串行扫描反例。三版Spearman中位数均为1，最小值为G 0.333、F1 0.857、F2 {c['min_rho']:.3f}。真实LoD的Gaussian身份、补偿属性和遮挡随Ratio变化，输入K不是像素工作量的唯一决定因素。

Full相对G总体加速{full['G_over_F']:.4f}×，但64个场景×轮次仍有{full['G_slower']}个更慢，最差增加{(1/full['min_G_over_F']-1)*100:.2f}%；相对C有{full['C_slower']}个慢例，最差增加{(1/full['min_C_over_F']-1)*100:.2f}%。未通过原先逐场景Full不退化门限，保留G回退，未替换生产默认。没有人为放慢Full或根据Ratio绕开新路径。

## 对照和实现

- G：前轮已验收的Full安全Graph渲染器。
- C：前轮mode21大footprint预处理任务融合；G→C属于旧收益。
- F1：C＋原顺序32次像素循环展开＋极小请求容量/排序优化。
- F2：F1＋普通背景下跳过冗余全图空成员证明，最终入口background_runtime.Renderer()。

32次循环展开保留每个像素内FP32表达式、Gaussian顺序、透射率早停和颜色累加次序；未并行归约T/C或重排Alpha。Nsight中Replica office2 RGB-D Full像素核从C 134.24µs降至F2 122.82µs。

小请求采用32点容量粒度，大于512点仍按512点容量粒度。32点用稳定整数Warp排序，64/128/256点使用相应规模的稳定BlockRadixSort，之后沿用原512点块排序与归并。全序为原FP32深度位＋原输入行ID，同深度排序和实际K不变，决策只依据通用输入形状。

F2优化copySparseBackground：旧实现以Tile0的单Warp串行扫描全图位表，证明没有实际成员以决定是否直接复制背景。普通有限normal值或+0背景满足FP32的“+0 + 1×bg == bg”，无须该证明。新代码在现有compact核读取本帧背景位模式并跳过证明；负零、denormal、NaN/Inf保留原检查。若原证明为真，全部Tile无成员，正常链为C=+0、T=1，因此普通背景输出逐位相同；若为假，原链本来就不变。没有新增Kernel、Host同步或历史背景缓存。

## 修复的显著反例

VECtor corridors-dolly双目评价索引653，同视角2点时实际Tile成员和Alpha交互均为0；13点时有23,791个Tile实例、5,920,403次Alpha交互，旧版反而2点更慢。

{table(['版本','2点纯渲染ms','13点纯渲染ms'],exrows)}

C→F1含小容量/排序和循环优化；F1→F2才是背景检查的独立增量。Nsight该条件两帧平均compact/背景检查为C 176.86µs、F1 24.47µs、F2 2.53µs。2点图像仍为原背景，RGB/radii逐位一致。

![反例修复](background_stage/reports/delivery/EMPTY_CASE_FIX.png)

## 性能与点数

Replica16组、TUM单目/RGB-D 10组、VECtor单目/双目6组，共32组。每组6个连续运动视角＋4个跳转视角，每条件每版本每轮60次同步墙钟样本，两轮反转场景、条件、视角及版本顺序。局部Ratio为1/5/10/20/30/40/50/70%，另含固定1000/5000点和Full。每个Pipeline请求重新准备当前相机、候选、选择和补偿。所有渲染版本消费同一冻结Teacher网络输出，不混入蒸馏或Tensor Core数值差异。

{table(['Ratio','F2纯渲染p50 ms','G/F2','C/F2','耗时降低vs C','慢于G场景×轮次'],ratios)}

耗时列为场景×Ratio×轮次p50的中位数；加速为配对比值的等权几何均值，两列不能直接相除。10%–50%纯渲染p50，G/C/F2分别{primary['G_p50_ms']:.4f}/{primary['C_p50_ms']:.4f}/{primary['F_p50_ms']:.4f}ms，F2 p95中位{primary['F_p95_ms']:.4f}ms。完整准备＋渲染分别{pipeline['G_p50_ms']:.4f}/{pipeline['C_p50_ms']:.4f}/{pipeline['F_p50_ms']:.4f}ms，F2 p95中位{pipeline['F_p95_ms']:.4f}ms。

主要Ratio纯渲染相对C 320/320个场景×Ratio×轮次更快；完整路径有{pipeline['C_slower']}/320个轻微慢例。27组保留场景纯渲染相对C加速{s['raster_heldout_primary']['C_over_F']:.4f}×。F2相对F1在常规负载的平均收益约0，保留该项为了消除极小K空绘制的大幅反序，没有宣称普遍加速。

![性能和固定属性删点对照](background_stage/reports/delivery/COUNT_SCALING_OVERVIEW.png)

## 逐组明细

加速为10%–50%跨两轮几何平均；Full代价为两轮相对G中最差一轮，正数为更慢。

{table(['场景/模式','地图N','C/F2纯渲染','G/F2纯渲染','Full最差代价vs G'],scenes)}

所有逐视角相邻反序及对应Tile/Alpha工作量见NONMONOTONIC_ADJACENT_CASES.csv。真实LoD非嵌套、补偿footprint变化、遮挡早停和固定像素写出开销仍造成少量反序。完整路径还支付Full旁路的选择/补偿成本，不能只凭K保证快于Full。

剩余最大反例来自TUM fr1_room单目：K从76增到152，Tile实例反而从19,225降至16,610，Alpha交互从3,486,651降至2,899,749，实际像素工作减少。该例耗时下降3.64%符合真实工作量，因此继续要求它仅随K增加而变慢，会与“不增加无效工作、不改变网络属性”的要求冲突。

## 正确性与范围

四版本共28,160份RGB/radii逐位对照通过，其中F2 7,040份；全部Pipeline计时样本结束后还核对当前ID与RGB。生命周期86项＋背景专项452项通过；memcheck、initcheck、synccheck各复验538项均0错误。新增7项同一背景Tensor原位改变普通/敏感值、连续异步提交和历史输出存活检查通过。

保持分辨率、SH阶数、FP32、Gaussian身份、补偿输出、排序、Alpha阈值和顺序，不接受图像容差。训练路径不变，不保存渲染图片。缓存只含执行计划与Workspace，每帧读取当前指针和值；分辨率/容量增长销毁旧地址计划，跨Stream复用拒绝，输出有独立存储。

F2持久数据Workspace最大{s['memory']['F']['max_workspace_mib']:.2f}MiB，与C/F1相同，比G增加{s['memory']['F']['max_workspace_mib']-s['memory']['G']['max_workspace_mib']:.2f}MiB。实际最多170个有界Graph计划，G为155；数据池上限1GiB、位表256MiB、最多512计划。不把Workspace当整个应用显存，驱动Graph状态和代码显存未单独分离。小请求减少填充，未减少有效输入或精度。

第一阶段4组开发、28组保留；第二阶段将发现空背景反例的corridors-dolly双目转入开发，其余27组不调参。最终统计使用337,920条同步墙钟样本、独立CUDA Event和Nsight Graph节点。Profiler时间不代替端到端时延；CPU图片解码、HTTP、显示、存盘和SLAM训练不在统计内。硬件RTX3090，不外推至所有端侧GPU。

四版本每个执行位置每版本覆盖14–16次；早期四版本顺序有偏置的测量完整作废，现场保留于FORMAL_CONTROLLER_UNBALANCED.json。最终另有{len(controller['rejected'])}段外部负载干扰按预先规则整段排除重跑，未按性能筛选。有效轮SM时钟{environment['clock_sm_mhz'][0]:g}–{environment['clock_sm_mhz'][1]:g}MHz、温度{environment['temperature_c'][0]:g}–{environment['temperature_c'][1]:g}℃。没有停止外部进程或锁频。

## 失败候选和交付

- 安全exp早拒绝/opacity分档：逐位通过，但新增判断使多数条件更慢，排除。
- 横向公共项显式复用：RGB出现5.96e-8差异，未通过逐位门限，排除。
- float4属性打包：逐位通过，未优于原顺序展开，排除。
- 2/4/8/16次展开有效，开发集32次更快；保留原Alpha顺序。
- Full仍有慢例；没有通过人为增加无效工作制造单调，未修改门限或隐藏反例。

最终入口background_runtime.Renderer()，原G、C和F1保留回退。未替换生产默认，未改训练、原checkpoint和共享conda，未commit/push/部署。{protected}项模型/输入/既有网络二进制SHA及前轮C源码、G/C冻结二进制核验通过；两份最终冻结源码/二进制保持一致，见INTEGRITY.json。

服务器目录：{ROOT}。归档含最终源码、两阶段加载二进制、配置、计时CSV、正确性/Sanitizer日志和代表性Profile；模型/数据集只记录原路径和SHA，不复制。图以实际输入K为横轴，固定属性嵌套删点只作耗时归因，不作为LoD质量实验。
"""
    (ROOT/'RENDER_COUNT_SCALING_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
    (OUT/'README.md').write_text('最终统计采用均衡双轮；F为F2，F1为第一候选。\nPER_SCENE_RATIO.csv含p50/p95、实际K和G/C/F1对照。\nNONMONOTONIC_ADJACENT_CASES.csv保留相邻反序，ALL_SLOWER_CASES.csv保留慢例。\nWORKLOADS为计时外Alpha/Tile统计，NESTED为固定属性删点对照。\n',encoding='utf-8')
    print('FINAL_REPORT_PASS',protected,len(bad),flush=True)


def plot_example(example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
    for v,color in [('G','#526E98'),('C','#9E7B53'),('F1','#9478A8'),('F','#007F73')]:
        q=example[example.variant==v].sort_values('K')
        ax.plot(q.K,q.ms,'o-',label='F2 final' if v=='F' else v,color=color)
    ax.set(xlabel='Input Gaussian count K',ylabel='Rasterizer latency (ms)',xticks=[2,13],
           title='Same view: removing the empty-draw scan overhead')
    ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    fig.savefig(OUT/'EMPTY_CASE_FIX.png',dpi=200);fig.savefig(OUT/'EMPTY_CASE_FIX.svg');plt.close(fig)


if __name__=='__main__':main()
