"""从本轮原始请求、负载包及验收记录生成统计和论文图。"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'
REPORTS=DATA/'reports'
RUN=DATA/'runs/formal03'
METHODS=('cartgs_full','ours_full','student_fixed4000','student_fixed1000','student_adaptive')
LABELS={
    'cartgs_full':'CaRtGS Full', 'ours_full':'Ours Full',
    'student_fixed4000':'Distilled, fixed 4k', 'student_fixed1000':'Distilled, fixed 1k',
    'student_adaptive':'Distilled, adaptive 4k → 1k',
}
COLORS=dict(zip(METHODS,('#4477AA','#888888','#228833','#AA3377','#EE7733')))


def save(name,value):
    (REPORTS/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def stats(group):
    elapsed=group.request_ms.to_numpy()
    average=float(elapsed.mean())
    deviation=float(elapsed.std(ddof=0))
    return dict(samples=len(group),request_mean_ms=average,
        request_p50_ms=float(np.percentile(elapsed,50)),request_p95_ms=float(np.percentile(elapsed,95)),
        request_p99_ms=float(np.percentile(elapsed,99)),request_max_ms=float(elapsed.max()),
        request_std_ms=deviation,request_cv=deviation/average,
        prepare_mean_ms=float(group.prepare_ms.mean()),render_mean_ms=float(group.render_ms.mean()),
        gate_mean_ms=float(group.gate_overhead_ms.mean()),prepare_gpu_mean_ms=float(group.prepare_gpu_ms.mean()),
        render_gpu_mean_ms=float(group.render_gpu_ms.mean()),K_min=int(group.K.min()),K_max=int(group.K.max()),
        K_mean=float(group.K.mean()),radii_positive_mean=float(group.radii_positive.mean()),
        candidate_min=int(group.candidate_N.min()),candidate_max=int(group.candidate_N.max()),
        shortfall_requests=int((group.K<group.requested_K).sum()),psnr_mean_db=float(group.psnr.mean()))


def plot(frame,latency_window,name):
    styles={'cartgs_full':'-','ours_full':'--','student_fixed4000':':','student_fixed1000':'-.','student_adaptive':'-'}
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(12,3.9))
        for method in METHODS:
            rows=frame[frame.method==method].sort_values('frame')
            width=2.1 if method=='student_adaptive' else 1.35
            args=dict(color=COLORS[method],lw=width,ls=styles[method])
            axes[0].plot(rows.frame,rows.K,label=LABELS[method],**args)
            latency=rows.request_ms.rolling(latency_window,center=True,min_periods=1).mean()
            axes[1].plot(rows.frame,latency,**args)
            psnr=rows.psnr.rolling(81,center=True,min_periods=1).mean()
            axes[2].plot(rows.frame,psnr,**args)
        axes[0].set_yscale('log')
        axes[0].yaxis.set_major_locator(FixedLocator([1000,2000,4000,10000,49102,73531]))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value/1000:.3g}k'))
        axes[0].minorticks_off()
        axes[0].set_ylabel('Submitted Gaussians (log scale)')
        axes[1].set_ylabel('Total request latency (ms)')
        axes[2].set_ylabel('PSNR (dB)')
        for axis in axes:
            axis.set_xlim(1,2000)
            axis.set_xlabel('Frame index')
            axis.set_xticks([1,500,1000,1500,2000])
            axis.grid(alpha=.18)
            upper=axis.secondary_xaxis('top',functions=(lambda x:20+60*(x-1)/1999,lambda u:1+1999*(u-20)/60))
            upper.set_xticks([20,40,60,80])
            upper.set_xticklabels(['20%','40%','60%','80%'])
            upper.tick_params(labelsize=8)
            upper.set_xlabel('Target GPU load (SM-time duty)',fontsize=8,labelpad=3)
        axes[0].set_title('(a) Rendering budget',fontsize=10,pad=35)
        axes[1].set_title('(b) Response time',fontsize=10,pad=35)
        axes[2].set_title('(c) Rendering quality',fontsize=10,pad=35)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.052),
                   ncol=5,frameon=False,fontsize=8)
        description=(f'Time: {latency_window}-frame centered mean of three-repeat per-frame medians; '
                     'PSNR: 81-frame centered mean. Preparation is included.')
        fig.text(.5,.015,description,ha='center',fontsize=7)
        fig.subplots_adjust(left=.075,right=.985,bottom=.26,top=.77,wspace=.34)
        for extension in ('png','pdf'):
            fig.savefig(REPORTS/f'{name}.{extension}',dpi=240)
        plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True,parents=True)
    source=RUN/'ALL_TIMINGS.csv'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=pd.read_csv(source)
    assert len(rows)==30000 and set(rows.method)==set(METHODS)
    assert rows[['request_ms','prepare_ms','render_ms','K','psnr']].notna().all().all()
    assert np.isfinite(rows[['request_ms','prepare_ms','render_ms','psnr']]).all().all()
    assert not rows.duplicated(['method','repeat','frame']).any()
    for _,group in rows.groupby(['method','repeat']):
        assert len(group)==2000 and set(group.frame)==set(range(1,2001))
    progress=(rows.frame-1)/1999
    assert np.allclose(rows.load_target,.2+.6*progress,rtol=0,atol=1e-14)
    planned=np.floor(4000-3000*progress+.5).astype(int)
    assert np.array_equal(rows.adaptive_budget,planned)
    adaptive=rows[rows.method=='student_adaptive']
    assert np.array_equal(adaptive.K,np.minimum(adaptive.requested_K,adaptive.eligible_N))
    rows['bin']=(rows.frame-1)//250
    rows['shortfall']=rows.K<rows.requested_K
    overall=[]; bins=[]; repeats=[]; accepts=[]
    for method in METHODS:
        group=rows[rows.method==method]
        overall.append(dict(method=method,**stats(group)))
        for index in range(8):
            bins.append(dict(method=method,bin=index,target_low=.2+.075*index,target_high=.275+.075*index,
                             **stats(group[group.bin==index])))
        for repeat in range(3):
            repeats.append(dict(method=method,repeat=repeat,**stats(group[group.repeat==repeat])))
            condition=RUN/f'{repeat}_{method}'
            acceptance=json.loads((condition/'ACCEPTANCE.json').read_text())
            assert acceptance['bin_match'] and acceptance['maximum_on_ns']<=2000000
            assert acceptance['sm_coverage_weighted']>=.99 and acceptance['hardware']['billing_sm_area']
            summary=acceptance['summary']
            assert summary['charged_duty']<=summary['weighted_target_duty']+1e-10
            assert summary['charged_duty']<=.8+1e-10
            row=dict(method=method,repeat=repeat,maximum_on_ms=acceptance['maximum_on_ns']/1e6,
                sm_area_coverage=acceptance['sm_area_coverage'],slot_area_coverage=acceptance['slot_area_coverage'],
                load_packets=acceptance['load_packets'],
                minimum_packet_sm_coverage=acceptance['sm_coverage_minimum'],tail_ms=summary['tail_ns']/1e6,
                tail_per_request_ms=summary['tail_ns']/summary['requests']/1e6,
                charged_duty=summary['charged_duty'],weighted_target=summary['weighted_target_duty'])
            row['maximum_bin_error_pp']=max(abs(x['sm_error'])*100 for x in acceptance['bins'])
            accepts.append(row)
            for values in bins[-8:]:
                item=acceptance['bins'][int(values['bin'])]
                for key in ('resident_envelope_duty','sm_actual_duty','slot_actual_duty','charged_duty','weighted_target'):
                    values[f'{key}_repeat{repeat}']=item[key]
    total=pd.DataFrame(overall)
    per_bin=pd.DataFrame(bins)
    pd.DataFrame(repeats).to_csv(REPORTS/'PER_REPEAT_METRICS.csv',index=False)
    total.to_csv(REPORTS/'OVERALL_METRICS.csv',index=False)
    per_bin.to_csv(REPORTS/'LOAD_BIN_METRICS.csv',index=False)
    pd.DataFrame(accepts).to_csv(REPORTS/'LOAD_ACCEPTANCE.csv',index=False)
    numeric=['request_ms','prepare_ms','render_ms','prepare_gpu_ms','render_gpu_ms','psnr','K','requested_K',
             'radii_positive','load_target','candidate_N','eligible_N','network_ratio']
    frame=rows.groupby(['method','frame'],as_index=False)[numeric].median()
    frame.to_csv(REPORTS/'PER_FRAME_THREE_REPEAT_MEDIAN.csv',index=False)
    plot(frame,51,'DISTILLED_RAMP_SMOOTHED')
    plot(frame,1,'DISTILLED_RAMP_RAW')
    spread=rows.groupby(['method','frame']).psnr.agg(lambda x:x.max()-x.min())
    checks=dict(passed=True,raw_requests=len(rows),methods=5,repeats=3,views=2000,
        source_sha256=before,psnr_max_repeat_spread_db=float(spread.max()),
        adaptive_budget_first=4000,adaptive_budget_last=1000,
        maximum_duty_error_pp=max(x['maximum_bin_error_pp'] for x in accepts),
        maximum_on_ms=max(x['maximum_on_ms'] for x in accepts),
        sm_area_coverage_minimum=min(x['sm_area_coverage'] for x in accepts),
        load_packets=sum(x['load_packets'] for x in accepts),
        shortfalls=int(rows.shortfall.sum()),student_graph_replays=int(rows[rows.source_student].graph_replay.sum()),
        plotted_latency_window=51,plotted_psnr_window=81,
        statistics='All raw requests, population standard deviation (ddof=0); no removed outliers')
    save('ANALYSIS_VALIDATION.json',checks)
    base=total.set_index('method')
    full=base.loc['ours_full']; adapted=base.loc['student_adaptive']
    high=per_bin[per_bin.bin==7].set_index('method')
    speed=float(full.request_mean_ms/adapted.request_mean_ms)
    high_speed=float(high.loc['ours_full','request_mean_ms']/high.loc['student_adaptive','request_mean_ms'])
    fixed=base.loc['student_fixed4000']
    full_reduction=100*(1-adapted.request_mean_ms/full.request_mean_ms)
    fixed_reduction=100*(1-adapted.request_mean_ms/fixed.request_mean_ms)
    fixed_high_reduction=100*(1-high.loc['student_adaptive','request_mean_ms']/high.loc['student_fixed4000','request_mean_ms'])
    low=per_bin[per_bin.bin==0].set_index('method')
    lines=[
        '# Replica蒸馏Compensator连续升载测速报告','',
        f'完成Replica office3 RGB-D全2000帧、五种方法、三轮，共30000次请求。'
        f'使用3534参数的32/32学生与v31部署入口，未重新训练。',
        '',f'Adaptive相对我们Full的完整请求均值加速为{speed:.3f}×；'
        f'最高负载区间72.5%–80%的均值加速为{high_speed:.3f}×。'
        '完整请求包含网络准备、渲染和当前人工负载包排空。',
        '',f'与我们Full相比，Adaptive总均值下降{full_reduction:.2f}%，p95下降'
        f'{100*(1-adapted.request_p95_ms/full.request_p95_ms):.2f}%。'
        f'但与学生固定4000点相比，Adaptive总均值仅再下降{fixed_reduction:.2f}%，'
        f'最高负载区间仅再下降{fixed_high_reduction:.2f}%。'
        f'Adaptive低负载区间均值{low.loc["student_adaptive","request_mean_ms"]:.4f}ms，'
        f'高负载区间仍增至{high.loc["student_adaptive","request_mean_ms"]:.4f}ms。'
        '因此本轮支持“减少点数能改善响应”，不能证明响应完全稳定，也不能把全部Full→Adaptive收益归因于动态降点。',
        '', '## 配对结果','',
        '| 方法 | 提交点数 | 准备均值ms | 渲染均值ms | 总均值ms | p95ms | p99ms | 标准差ms | CV | 平均PSNR dB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for item in overall:
        count=str(item['K_min']) if item['K_min']==item['K_max'] else f"{item['K_min']}–{item['K_max']}"
        lines.append(f"| {LABELS[item['method']]} | {count} | {item['prepare_mean_ms']:.4f} | {item['render_mean_ms']:.4f} | "
                     f"{item['request_mean_ms']:.4f} | {item['request_p95_ms']:.4f} | {item['request_p99_ms']:.4f} | "
                     f"{item['request_std_ms']:.4f} | {item['request_cv']:.4f} | {item['psnr_mean_db']:.4f} |")
    lines+=['','## 负载区间与响应','',
        '| 目标负载区间 | 我们Full均值ms | Adaptive均值ms | Adaptive p95ms | 加速 | Adaptive平均K |',
        '|---|---:|---:|---:|---:|---:|']
    for index in range(8):
        group=per_bin[per_bin.bin==index].set_index('method'); a=group.loc['student_adaptive']; f=group.loc['ours_full']
        lines.append(f'| {20+7.5*index:g}%–{27.5+7.5*index:g}% | {f.request_mean_ms:.4f} | {a.request_mean_ms:.4f} | '
                     f'{a.request_p95_ms:.4f} | {f.request_mean_ms/a.request_mean_ms:.3f}× | {a.K_mean:.1f} |')
    lines+=['','## 协议、验收及限制','',
        '- 帧i从0开始：u=0.2+0.6i/1999；B=floor(4000−3000i/1999+0.5)。'
        '此处Adaptive为已知负载到预算的预设映射，不是延迟闭环控制。',
        '- 各方法使用相同帧序及目标负载。相同RGB时间戳，各自保存的SLAM相机位姿；'
        '我们Full使用49102点未补偿Master，CaRtGS使用73531点原地图。'
        '局部学生从名义Anchor可见候选中严格选择K=min(B,E)。',
        '- 我们Full准备阶段是既有runtime.full()的属性/ID打包与复制，不运行Selector或Compensator；'
        '学生准备阶段包含候选、两网络、严格选择和属性打包。其耗时为实际部署入口请求成本，'
        '不等于纯MLP Kernel时间。Full与CaRtGS的地图和相机估计不同，跨系统PSNR不构成同图同姿消融。',
        '- 最高优先级准备Stream为−5，负载为−4，渲染为0。人工负载仅在渲染窗口运行，'
        '准备、读取图像和统计质量时暂停；优先级不等同跨进程硬件抢占。',
        '- 负载百分比指人工负载占据的归一化SM时间面积，不是整卡排他时间。'
        '同一SM的Block区间先求并集，所有SM求和后除以82得到等效整卡ON时间Q。'
        '计费向上取整Q，按各窗口目标u×active累加许可，直接控制实测SM占空比。'
        'GPU排队、CUDA Event或包络时间不能替代Q；驻留包络占空比可以高于目标，另存。',
        '- 2ms周期、普通网格492Block×512线程、共同GPU截止时间，无跨SM栅栏。'
        '每包记录GPU驻留、SM区间并集、'
        '线程槽位面积和CUDA Event。跨帧累计占空比余额，完整条件尾部仅结清一次OFF。'
        '不能宣称每包完全独占GPU。',
        f"- 15个条件全部通过修订后的负载验收：最大包{checks['maximum_on_ms']:.4f}ms，"
        f"SM时间面积/包络最低{checks['sm_area_coverage_minimum']*100:.3f}%（诊断值），"
        f"250帧区间实际SM占空比相对时间加权目标最大偏差{checks['maximum_duty_error_pp']:.3f}个百分点。"
        '完整轨迹的时间加权目标通常不等于50%；各区间实际值见LOAD_BIN_METRICS.csv。',
        f"- 共{checks['load_packets']}个负载包；条件尾部OFF仅结清一次，"
        'LOAD_ACCEPTANCE.csv单列其摊销成本。图表主耗时为逐请求响应，条件尾部不计入单帧p95。',
        '- formal01的普通网格在4046包中11包低于原单包95%访问覆盖要求，smoke02的Cooperative Launch'
        '最大包3.226ms，均判失败。smoke03新增真实SM面积后，小样本32帧窗出现5.84个百分点偏差。'
        'formal02的我们Full实测SM面积93.21%、最高区间误差12.85个百分点，亦弃用。'
        'smoke05/06的有限启动握手仍出现超时或占空比偏差，弃用。'
        '全部失败数据保留、不并入正式结果。最终改为SM面积反馈计费；完整250帧窗实际SM占空比'
        '相对时间加权目标±5个百分点、整体访问覆盖≥99%、包≤2ms、累计SM占空比≤80%。'
        '不同占空比口径不混合比较，修订来源见PROTOCOL.md。',
        f"- 预算不足请求{checks['shortfalls']}次；学生Graph重放{checks['student_graph_replays']}/18000次。"
        f"同帧三轮PSNR最大差{checks['psnr_max_repeat_spread_db']:.9g}dB。"
        'Graph/Native的40组相机/预算输入与选择身份逐位核对，头尾负载图像与无负载参考逐位核对。',
        '- 图中先取三轮同帧中位数，耗时51帧居中均值、PSNR81帧居中均值，预算保持原始值。'
        '统计表使用全部原始30000次请求，未剔除离群值；标准差为总体标准差。另附未平滑耗时图。',
        '- 各方法三轮顺序轮换。逐秒记录GPU进程与pmon，允许已核验身份的空闲Viewer驻留；'
        '未停止无关进程，采样不能排除间隙的瞬时活动。'
        '纯计时覆盖部署准备与渲染，不包含SLAM训练、图像读取、PSNR计算或显示。',
        '- 本轮对比的是学生v31部署与Full渲染，不单独隔离蒸馏和部署后端的加速来源。'
        '不同帧的视角复杂度与负载同时变化；相同帧同日程的配对可比较方法，'
        '不能把轨迹首尾耗时变化全部归因于负载。固定1k/4k对照用于判断预算变化的收益。',
        '- 耗时图中的阶跃与毫秒级短负载包有关。相同平均占空比并不保证相同单帧相位，'
        '本轮跨帧保存余额、三轮使用不同初相位。高负载时共同1.6ms包和必要排空可能主导响应，'
        '减少几千点的Kernel节省会被等待掩盖；该解释与固定1k/4k接近的实测相符，'
        '并非独立证明了所有波动来源。',
        '', '## 文件与保护','',
        '- 原始数据：runs/formal03/ALL_TIMINGS.csv及每条件PULSES.csv、窗口、TIMINGS.csv、ACCEPTANCE.json。',
        '- 图：DISTILLED_RAMP_SMOOTHED.png/PDF，未平滑耗时：DISTILLED_RAMP_RAW.png/PDF。',
        '- 汇总：OVERALL_METRICS.csv、LOAD_BIN_METRICS.csv、PER_REPEAT_METRICS.csv、LOAD_ACCEPTANCE.csv。',
        '- 模型、生产源码及共享环境的前后SHA核验见FINAL_VALIDATION.json；不保存渲染图像，'
        '不覆盖生产配置，不commit/push。',
    ]
    (REPORTS/'DISTILLED_COMPENSATOR_RAMP_LOAD_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print(total[['method','request_mean_ms','request_p95_ms','request_std_ms','request_cv','psnr_mean_db']].to_string(index=False))
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    main()
