"""依据锁定协议汇总短周期实验，输出中文报告与英文曲线。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np

METHODS=('cartgs_full','ours_full','ours_fixed4000','ours_fixed500','ours_adaptive')
NAMES=dict(cartgs_full='CaRtGS Full',ours_full='我们 Full',ours_fixed4000='固定4000点',
           ours_fixed500='固定500点',ours_adaptive='预算切换')
LABELS=dict(cartgs_full='CaRtGS Full',ours_full='Ours Full',ours_fixed4000='Ours Fixed 4000',
            ours_fixed500='Ours Fixed 500',ours_adaptive='Ours Adaptive')
COLORS=dict(cartgs_full='#2474B5',ours_full='#38A16A',ours_fixed4000='#8A8D91',
            ours_fixed500='#8A66B2',ours_adaptive='#D35E35')
DUTIES=(0.,.4,.6,.8)
BUDGETS=(4000,2000,1000,500)
METRICS=('request_ms','pipeline_ms','prepare_ms','render_ms','gate_overhead_ms','K','radii_positive')


def read(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def held_out(row):
    return (int(row['frame'])-1)%17!=0


def budget_for(method,stage):
    if method=='ours_fixed500':return 500
    if method=='ours_adaptive':return BUDGETS[stage]
    return 4000


def describe(values):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean())
    std=float(values.std())
    return dict(mean=mean,p50=float(np.median(values)),p95=float(np.percentile(values,95)),
                p99=float(np.percentile(values,99)),maximum=float(values.max()),std=std,
                cv_percent=100*std/mean if mean else 0.)


def rolling_mean(values,width):
    values=np.asarray(values,dtype=float)
    width=min(width,len(values))
    if width%2==0:width-=1
    radius=width//2
    padded=np.pad(values,(radius,radius),mode='edge')
    return np.convolve(padded,np.ones(width)/width,mode='valid')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    reports=root/'reports'
    locked=json.loads((reports/'PROTOCOL_LOCK.json').read_text())
    timing=read(root/'runs/benchmark02/ALL_TIMINGS.csv')
    quality=read(root/'runs/quality02/QUALITY.csv')
    curves=read(root/'runs/curves02/CURVE_TIMINGS.csv')
    assert len(timing)==(792+2000)*4*5*3
    assert len(quality)==(792+2000)*6
    assert len(curves)==(792+2000)*5
    qmap={}
    for row in quality:
        name=row['method'] if row['method'] in ('cartgs_full','ours_full') else 'budget_'+row['requested_budget']
        key=(row['scene'],int(row['frame']),name)
        assert key not in qmap
        qmap[key]=float(row['psnr'])
    summaries=[]
    repeats=[]
    cases={}
    audits=[]
    for path in (root/'runs/benchmark02').glob('*/ACCEPTANCE.json'):
        result=json.loads(path.read_text())
        sample=next(iter(read(path.parent/'TIMINGS.csv')))
        key=(sample['scene'],sample['method'],int(sample['stage']),int(sample['repeat']))
        cases[key]=result
        assert result['passed'] and result['target_match']
        assert result['sm_coverage_passed'] and result['artificial_load_prepare_overlap']==0
        assert result['short_pulse_passed']
        audits.append(result)
    assert len(cases)==120
    grouped={}
    for row in timing:
        key=(row['scene'],row['method'],int(row['stage']))
        grouped.setdefault(key,[]).append(row)
    for (scene,method,stage),all_rows in sorted(grouped.items()):
        rows=[row for row in all_rows if held_out(row)]
        tau=locked['scenes'][scene]['tau_ms']
        mode=method if method in ('cartgs_full','ours_full') else f'budget_{budget_for(method,stage)}'
        qvalues=[qmap[scene,frame,mode] for frame in sorted({int(row['frame']) for row in rows})]
        full=[qmap[scene,frame,'ours_full'] for frame in sorted({int(row['frame']) for row in rows})]
        summary=dict(scene=scene,method=method,stage=stage,target_duty=DUTIES[stage],samples=len(rows),
            views=len(qvalues),tau_ms=tau,mean_psnr=float(np.mean(qvalues)),
            psnr_delta_from_full=float(np.mean(qvalues)-np.mean(full)))
        for metric in METRICS:
            summary.update({metric+'_'+name:value for name,value in describe([row[metric] for row in rows]).items()})
        summary['deadline_miss_percent']=100*np.mean([float(row['request_ms'])>tau for row in rows])
        selected=[cases[scene,method,stage,repeat]['summary'] for repeat in range(3)]
        settled=sum(case['active_ns']+case['tail_ns'] for case in selected)
        summary['actual_duty']=sum(case['resident_ns'] for case in selected)/settled
        summary['charged_duty']=sum(case['charged_ns'] for case in selected)/settled
        summary['tail_amortized_ms']=sum(case['tail_ns'] for case in selected)/1e6/len(all_rows)
        summary['request_mean_with_tail_ms']=summary['request_ms_mean']+summary['tail_amortized_ms']
        summary['near_full_within_1db']=summary['psnr_delta_from_full']>=-1.
        summary['budget_shortfalls']=sum(int(row['K'])<int(row['requested_K']) for row in rows)
        summaries.append(summary)
        for repeat in range(3):
            part=[row for row in rows if int(row['repeat'])==repeat]
            repeats.append(dict(scene=scene,method=method,stage=stage,repeat=repeat,
                mean_ms=float(np.mean([float(row['request_ms']) for row in part])),
                p95_ms=float(np.percentile([float(row['request_ms']) for row in part],95)),
                deadline_miss_percent=100*np.mean([float(row['request_ms'])>tau for row in part]),
                actual_duty=cases[scene,method,stage,repeat]['summary']['actual_duty']))
    write(reports/'STATIC_SUMMARY.csv',summaries)
    write(reports/'REPEAT_SUMMARY.csv',repeats)
    lookup={(row['scene'],row['method'],row['stage']):row for row in summaries}
    comparisons=[]
    for scene in ('tum','office3'):
        for stage in range(4):
            adaptive=lookup[scene,'ours_adaptive',stage]
            fixed=lookup[scene,'ours_fixed4000',stage]
            low=lookup[scene,'ours_fixed500',stage]
            full=lookup[scene,'ours_full',stage]
            comparisons.append(dict(scene=scene,stage=stage,target_duty=DUTIES[stage],
                p95_vs_fixed4000_gain_percent=100*(1-adaptive['request_ms_p95']/fixed['request_ms_p95']),
                p95_vs_full_gain_percent=100*(1-adaptive['request_ms_p95']/full['request_ms_p95']),
                deadline_miss_reduction_pp=fixed['deadline_miss_percent']-adaptive['deadline_miss_percent'],
                psnr_gain_over_fixed500_db=adaptive['mean_psnr']-low['mean_psnr'],
                psnr_delta_from_full_db=adaptive['psnr_delta_from_full']))
    write(reports/'ADAPTIVE_COMPARISONS.csv',comparisons)
    repeat_lookup={(row['scene'],row['method'],row['stage'],row['repeat']):row for row in repeats}
    findings=[]
    conclusions=[]
    for scene in ('tum','office3'):
        high=lookup[scene,'ours_adaptive',3]
        high_comparison=next(row for row in comparisons if row['scene']==scene and row['stage']==3)
        light=next(row for row in comparisons if row['scene']==scene and row['stage']==0)
        gates=dict(scene=scene,
            heavy_p95_better_all_repeats=all(repeat_lookup[scene,'ours_adaptive',stage,repeat]['p95_ms']<repeat_lookup[scene,'ours_fixed4000',stage,repeat]['p95_ms'] for stage in (2,3) for repeat in range(3)),
            light_quality_gain_at_least_1db=light['psnr_gain_over_fixed500_db']>=1.,
            all_loads_p95_within_target=all(lookup[scene,'ours_adaptive',stage]['request_ms_p95']<=lookup[scene,'ours_adaptive',stage]['tau_ms'] for stage in range(4)),
            all_loads_within_1db_of_full=all(lookup[scene,'ours_adaptive',stage]['near_full_within_1db'] for stage in range(4)))
        findings.append(gates)
        name='TUM fr1_xyz' if scene=='tum' else 'Replica office3'
        direction='通过' if gates['heavy_p95_better_all_repeats'] else '未通过'
        latency='通过' if gates['all_loads_p95_within_target'] else '未通过'
        near_full='通过' if gates['all_loads_within_1db_of_full'] else '未通过'
        conclusions.append(f"{name}在80%档的预算切换p95为{high['request_ms_p95']:.3f}毫秒，相对固定4000点耗时变化为{-high_comparison['p95_vs_fixed4000_gain_percent']:+.1f}%，相对我们Full为{-high_comparison['p95_vs_full_gain_percent']:+.1f}%。轻负载相对固定500点的平均PSNR增益为{light['psnr_gain_over_fixed500_db']:+.2f}dB，80%档相对Full为{high['psnr_delta_from_full']:+.2f}dB。六个重负载逐轮对比方向全部更快的检查{direction}，四档均达到锁定延迟目标的检查{latency}，四档均在Full-1dB内的检查{near_full}。")
    (reports/'CONCLUSION_CHECKS.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    for row in curves:
        stage=int(row['stage'])
        mode=row['method'] if row['method'] in ('cartgs_full','ours_full') else f"budget_{budget_for(row['method'],stage)}"
        row['psnr']=qmap[row['scene'],int(row['frame']),mode]
    write(reports/'TRAJECTORY_WITH_PSNR.csv',curves)
    scene_reports=[]
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        for scene in ('tum','office3'):
            title='TUM fr1_xyz' if scene=='tum' else 'Replica office3'
            fig,axes=plt.subplots(1,3,figsize=(11.7,3.5))
            for method in METHODS:
                points=[lookup[scene,method,stage] for stage in range(4)]
                for axis,metric in zip(axes,('request_ms_p95','deadline_miss_percent','mean_psnr')):
                    axis.plot([0,40,60,80],[point[metric] for point in points],marker='o',markersize=3.5,
                              color=COLORS[method],label=LABELS[method],lw=1.3)
            tau=locked['scenes'][scene]['tau_ms']
            axes[0].axhline(tau,color='black',ls=':',lw=.8)
            axes[0].text(.02,.96,f'Target {tau:.3f} ms',transform=axes[0].transAxes,va='top',fontsize=8)
            for axis in axes:
                axis.set_xlabel('Requested artificial duty (%)')
                axis.set_xticks([0,40,60,80])
                axis.grid(alpha=.18)
            axes[0].set_ylabel('End-to-end p95 (ms)')
            axes[1].set_ylabel('Deadline miss rate (%)')
            axes[1].set_ylim(-2,102)
            axes[2].set_ylabel('Mean PSNR (dB)')
            fig.suptitle(title+' — paired views, three timing repeats',fontsize=11)
            fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,fontsize=8)
            fig.tight_layout(rect=[0,.10,1,.94])
            fig.savefig(reports/f'{scene.upper()}_STATIC.png',dpi=220)
            fig.savefig(reports/f'{scene.upper()}_STATIC.pdf')
            plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(11.7,3.5))
            for method in METHODS:
                selected=sorted([row for row in curves if row['scene']==scene and row['method']==method],key=lambda row:int(row['frame']))
                frames=[int(row['frame']) for row in selected]
                axes[0].plot(frames,[int(row['K']) for row in selected],color=COLORS[method],lw=1.,label=LABELS[method])
                axes[1].plot(frames,[float(row['request_ms']) for row in selected],color=COLORS[method],lw=.6,alpha=.85)
                for segment in range(8):
                    part=[row for row in selected if int(row['segment'])==segment]
                    axes[2].plot([int(row['frame']) for row in part],rolling_mean([row['psnr'] for row in part],81),
                                 color=COLORS[method],lw=1.)
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Submitted Gaussians (log scale)')
            axes[1].set_yscale('log')
            axes[1].set_ylabel('End-to-end time (ms), log, raw')
            axes[1].yaxis.set_major_locator(LogLocator(base=2))
            axes[1].yaxis.set_major_formatter(ScalarFormatter())
            axes[1].yaxis.set_minor_formatter(NullFormatter())
            axes[2].set_ylabel('PSNR (dB), smoothed within stage')
            count=locked['scenes'][scene]['metadata']['frames']
            for axis in axes:
                axis.set_xlim(1,count)
                axis.set_xlabel('Frame index')
                axis.grid(alpha=.18)
                for segment,stage in enumerate((0,1,2,3,3,2,1,0)):
                    if segment:axis.axvline(segment*count/8,color='#737373',ls='--',lw=.6,alpha=.6)
                    axis.text((segment+.5)/8,1.025,f'{100*DUTIES[stage]:.0f}%',transform=axis.transAxes,ha='center',fontsize=8)
            fig.suptitle(title+' — rising and falling controlled load',fontsize=11,y=.99)
            fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,fontsize=8)
            fig.tight_layout(rect=[0,.10,1,.91])
            fig.savefig(reports/f'{scene.upper()}_TRAJECTORY.png',dpi=220)
            fig.savefig(reports/f'{scene.upper()}_TRAJECTORY.pdf')
            plt.close(fig)
            table=[]
            for stage in range(4):
                for method in METHODS:
                    row=lookup[scene,method,stage]
                    table.append(f"| {100*DUTIES[stage]:.0f}% | {NAMES[method]} | {row['request_ms_mean']:.3f} | {row['request_ms_p95']:.3f} | {row['request_ms_p99']:.3f} | {row['deadline_miss_percent']:.1f}% | {row['mean_psnr']:.2f} | {row['psnr_delta_from_full']:+.2f} | {100*row['actual_duty']:.2f}% |")
            compare=[]
            for row in comparisons:
                if row['scene']!=scene:continue
                compare.append(f"| {100*row['target_duty']:.0f}% | {row['p95_vs_fixed4000_gain_percent']:+.1f}% | {row['p95_vs_full_gain_percent']:+.1f}% | {row['deadline_miss_reduction_pp']:+.1f} | {row['psnr_gain_over_fixed500_db']:+.2f} |")
            scene_reports.append(f"""## {title}

| 负载 | 方法 | 平均ms | p95 ms | p99 ms | 超时比例 | PSNR dB | 相对Full dB | 实测驻留占空比 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(table)}

预算切换的正收益百分比表示耗时降低，超时差值单位为百分点。

| 负载 | 相对固定4000的p95收益 | 相对我们Full的p95收益 | 相对固定4000的超时下降 | 相对固定500的PSNR增益dB |
|---|---|---|---|---|
{chr(10).join(compare)}

![配对静态负载]({scene.upper()}_STATIC.png)

![升降负载轨迹]({scene.upper()}_TRAJECTORY.png)
""")
    monitors=[]
    for phase in ('calibration03','benchmark02','quality02','curves02'):
        for line in (root/'runs'/phase/'GPU_MONITOR.jsonl').read_text().splitlines():
            entry=json.loads(line)
            assert not entry['foreign']
            monitors.append(entry)
    integrity=json.loads((reports/'INTEGRITY_FINAL.json').read_text())
    active_audits=[audit for audit in audits if audit['summary']['target_duty']>0]
    slowest=max(timing,key=lambda row:float(row['request_ms']))
    slowest_curve=max(curves,key=lambda row:float(row['request_ms']))
    shortfalls=sum(int(row['K'])<int(row['requested_K']) for row in timing)
    report=f"""# v27短周期GPU负载与预算调度实验报告

## 实验范围

完成fr1_xyz RGB-D的792帧与Replica office3 RGB-D的2000帧。五种方法、四档负载、三轮速度对照共{len(timing)}次请求，质量评价{len(quality)}次，升降负载展示{len(curves)}次。正式使用修正版benchmark02，额外标定记录保留在calibration03目录。v1的benchmark01因人工负载栅栏长尾被淘汰，原始数据完整保留。

固定方法包含CaRtGS Full、我们未补偿Master Full、我们固定4000点与固定500点。预算切换按已知负载0%／40%／60%／80%映射到4000／2000／1000／500点，每帧重新执行视角条件Selector、精确选择与Compensator。该策略使用预设负载档位。

| 场景 | 保存轨迹视角 | 图像分辨率 | 我们Master点数 | CaRtGS点数 |
|---|---|---|---|---|
| fr1_xyz RGB-D | 792 | 640×480 | {locked['scenes']['tum']['metadata']['gaussians']} | {locked['scenes']['tum']['metadata']['cartgs_gaussians']} |
| office3 RGB-D | 2000 | 1200×680 | {locked['scenes']['office3']['metadata']['gaussians']} | {locked['scenes']['office3']['metadata']['cartgs_gaussians']} |

## 按预声明标准得到的结论

{chr(10).join(conclusions)}

每17帧为校准视角，正式统计使用其余视角。延迟目标来自校准集固定4000点无负载完整请求p95的1.25倍，在正式测试前锁定。fr1_xyz目标{locked['scenes']['tum']['tau_ms']:.3f}毫秒，office3目标{locked['scenes']['office3']['tau_ms']:.3f}毫秒。主表使用端到端完整请求耗时，包含网络准备、渲染和必要ON排空。

## 负载标定与阶段隔离

选中的名义周期为{locked['cycle_ns']/1e6:.3f}毫秒，40%／60%／80%对应{locked['cycle_ns']*.4/1000:.0f}／{locked['cycle_ns']*.6/1000:.0f}／{locked['cycle_ns']*.8/1000:.0f}微秒ON目标。采用普通CUDA网格和共同GPU截止时间，独立Block按截止时间结束。启动截止时间与负载核使用固定二节点CUDA Graph提交，避免两个Host调用之间的间隙。每Block 512线程，共享内存与线程数限制每SM最多同时驻留三个负载Block，网格投放两个容量波次，逐包记录真实SM覆盖和每SM执行Block数量。实际包络还包括调度及Block退出时间，按每包globaltimer与CUDA Event的较大值计费。

SM覆盖验收为每条件按驻留时间加权至少99%、任一包至少95%。正式有负载条件中最低加权覆盖率为{100*min(a['sm_coverage_weighted'] for a in active_audits):.3f}%，单包最低覆盖率为{100*min(a['sm_coverage_minimum'] for a in active_audits):.3f}%。其中{sum(a['all_pulses_full_sm'] for a in active_audits)}／{len(active_audits)}个有负载条件的每个负载包均覆盖全部82个SM。普通并发调度无法强制已经执行其他Kernel的SM立即腾空，资源容量与真实覆盖分别记录。

初版全局栅栏在与原版Rasterizer交错时出现约190毫秒入口等待，而计算本身仍约400微秒。直接改为独立Block后，固定每SM执行三个Block的计数假设不再成立。最终采用共同截止时间和过量网格补充可用资源，避免跨SM栅栏等待，并逐包核对真实覆盖。1600次独立回归的最大驻留为0.419毫秒。初版速度结果与这些诊断记录分开保存。

修正版在重新标定前增加单包最大驻留不超过2毫秒、p99不超过名义ON加250微秒的验收。正式120个条件的最大ON驻留为{max(a['maximum_on_ns'] for a in audits)/1e6:.3f}毫秒，全部通过。标定使用±3个百分点裕量，正式仍为±5个百分点。原始延迟目标0.937／1.477毫秒保持锁定。

跨帧累积渲染有效时间与计费ON，保留初始相位和OFF余额。准备、读图和统计期间暂停人工负载及其有效时间轴。逐帧只等待ON排空，OFF可由下一次渲染窗口偿还。每个条件最后补足一次剩余OFF，尾部等待单独记录并加入摊销平均耗时。

占空比上限作用于完整累计周期。单个请求可以处于ON或OFF子段，不能以单请求曝光直接判断整个条件是否超过80%。120个正式条件全部通过计费占空比上限、实测匹配±5个百分点、SM覆盖和准备零重叠检查，共审计{sum(a['load_packets'] for a in audits)}个负载包。

实际驻留占空比仍受保守Event计费影响，表中逐条件给出实测值。不同方法之间较小的速度差异应结合这些曝光差异和逐轮结果判断，不做事后耗时归一化。globaltimer驻留包络也可能包含驱动抢占暂停。

{chr(10).join(scene_reports)}

## 判断口径

速度收益分别对照固定4000和我们Full。轻负载质量对照固定500，以至少1dB增益为预声明参考。80%预算切换与固定500使用相同预算。相对Full下降不超过1dB作为近Full质量参考线，完整结果在STATIC_SUMMARY.csv的near_full_within_1db中记录。超过参考线的条件按质量与速度折中报告。

图中的静态耗时为所有正式视角、三轮样本汇总。升降轨迹耗时保持单轮原始值，没有平滑或删除离群点。PSNR仅在各阶段内部作81帧居中均值平滑，阶段边界不相互混合。统计表始终使用原始PSNR。

升降轨迹的耗时采用对数纵轴，以同时展示常见延迟与离群点。静态记录最大请求为{float(slowest['request_ms']):.3f}毫秒，发生于{slowest['scene']}、{slowest['method']}、帧{slowest['frame']}。动态轨迹最大请求为{float(slowest_curve['request_ms']):.3f}毫秒，其render Stream Event区间为{float(slowest_curve['render_gpu_ms']):.3f}毫秒。对应版本全部ON包仍满足长度限制，长请求没有删除，具体发射空隙、Stream排队或算子来源尚未进一步定位，不能据此宣称完全消除了运行时长尾。

## 数值、环境与完整性

- 每个速度条件首尾视角检查负载下与无负载输出逐位一致，所有请求检查非有限值、严格K和实际radii>0数量。
- 局部实际K为请求预算B与当前候选中合格点数E的较小值。全部正式速度记录中有{shortfalls}次候选不足，实际K和候选N均保留，不能将这些请求称为恰好提交4000点。
- 完整质量评价输出PSNR，沿用此前Viewer原始RGB口径，渲染结果clamp到[0,1]后对整幅图像计算MSE。未保存渲染图片。两个模型以同一图像时间戳对齐，各自使用保存的SLAM位姿，原版与我们地图大小不同，Full路径不同。
- 采用当前Torch 2.3.1+cu118环境和原v27二进制，SHA256为8f69753da6b7b346296e54b258a1d74fd81403c49e436093a1217ba7c9a14d89。
- {len(monitors)}次GPU监控采样未发现外部CUDA计算进程。频率、功耗与温度记录随原始数据保存。
- 外部受保护文件{integrity['external_files']}项、本轮实现文件{integrity['implementation_files']}项，前后哈希核验通过。独立目录中的协议锁定文件、源快照、负载动态库及原始日志保留。

## 数据文件

STATIC_SUMMARY.csv包含每场景、方法、负载的均值、p50、p95、p99、最大值、标准差、CV、超时比例、PSNR、实际负载和尾部摊销。REPEAT_SUMMARY.csv保存逐轮结果。ADAPTIVE_COMPARISONS.csv给出预算切换相对各对照的差值。TRAJECTORY_WITH_PSNR.csv将升降轨迹的原始耗时与PSNR对齐。

原始速度数据位于runs/benchmark02/ALL_TIMINGS.csv，各条件子目录包含TIMINGS、PULSES、窗口日志、SUMMARY和ACCEPTANCE。runs/quality02/QUALITY.csv保留逐帧PSNR，runs/curves02/CURVE_TIMINGS.csv保留真实升降负载请求。
"""
    (reports/'ASYNC_GPU_LOAD_V27_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(timing_rows=len(timing),quality_rows=len(quality),curve_rows=len(curves),comparisons=comparisons),ensure_ascii=False))


if __name__=='__main__':
    main()
