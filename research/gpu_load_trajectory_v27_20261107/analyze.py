"""整理原始计时、逐视角曲线和中文实验报告。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ('cartgs_full', 'ours_full', 'ours_adaptive')
NAMES = dict(cartgs_full='CaRtGS Full', ours_full='我们 Full', ours_adaptive='我们自适应')
DUTIES = (0., .4, .6, .8)
BUDGETS = (4000, 2000, 1000, 500)
METRICS = ('prepare_ms', 'render_ms', 'pipeline_ms', 'gate_overhead_ms', 'request_ms',
           'prepare_gpu_ms', 'render_gpu_ms', 'K', 'candidate_N', 'eligible_N', 'radii_positive')


def read(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(rows, metric):
    values = np.array([float(row[metric]) for row in rows])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
        p95=float(np.percentile(values,95)), maximum=float(values.max()),
        std=float(values.std()), cv_percent=float(values.std()/values.mean()*100) if values.mean() else 0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / 'runs/trajectory01'
    reports = root / 'reports'
    rows = read(folder / 'TIMINGS.csv')
    baseline = read(folder / 'NO_LOAD.csv')
    phases = {row['generation']:row for row in read(folder / 'LOAD_PACKETS.csv.phases.csv')}
    environment = json.loads((folder / 'ENVIRONMENT.json').read_text())
    acceptance = json.loads((folder / 'LOAD_ACCEPTANCE.json').read_text())
    integrity = json.loads((reports / 'INTEGRITY_FINAL.json').read_text())
    assert len(rows) == 792*3*3 and len(baseline) == 792*3
    assert acceptance['passed'] and acceptance['actual_duty_cap']
    assert integrity['passed']
    assert len({(r['repeat'],r['frame'],r['method']) for r in rows}) == len(rows)
    for row in rows:
        phase = phases[row['load_generation']]
        assert float(phase['target_duty']) == float(row['target_duty'])
        row.update(actual_duty=phase['measured_resident_duty'],charged_duty=phase['charged_duty'],
                   load_window_ms=int(phase['window_ns'])/1e6,load_pulses=int(phase['pulses']))
        stage = (int(row['frame'])-1)*4//792
        assert int(row['stage']) == stage and float(row['target_duty']) == DUTIES[stage]
        expected_k = min(BUDGETS[stage],int(row['eligible_N']))
        if row['method'] == 'ours_adaptive':
            assert int(row['K']) == expected_k
    write(reports / 'ALL_FRAMES_WITH_LOAD.csv',rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row['method'],int(row['frame'])),[]).append(row)
    medians = []
    for (method,frame), samples in sorted(grouped.items()):
        assert len(samples) == 3
        entry = dict(method=method,frame=frame,stage=(frame-1)*4//792)
        for metric in METRICS+('actual_duty','charged_duty'):
            entry[metric] = float(np.median([float(r[metric]) for r in samples]))
        medians.append(entry)
    write(reports / 'PER_VIEW_MEDIANS.csv',medians)
    summaries = []
    for stage,duty in enumerate(DUTIES):
        for method in METHODS:
            samples = [r for r in rows if int(r['stage'])==stage and r['method']==method]
            base = [r for r in baseline if int(r['stage'])==stage and r['method']==method]
            item = dict(stage=stage,first_frame=stage*198+1,last_frame=(stage+1)*198,
                        method=method,target_duty=duty,samples=len(samples))
            for metric in METRICS:
                item.update({metric+'_'+key:value for key,value in statistics(samples,metric).items()})
            for metric in ('prepare_ms','render_ms','pipeline_ms','request_ms'):
                item['no_load_'+metric+'_mean'] = statistics(base,metric)['mean']
            item['actual_duty_weighted'] = sum(int(phases[r['load_generation']]['resident_envelope_ns']) for r in samples)/sum(int(phases[r['load_generation']]['window_ns']) for r in samples)
            item['actual_duty_min'] = min(float(r['actual_duty']) for r in samples)
            item['actual_duty_max'] = max(float(r['actual_duty']) for r in samples)
            item['budget_shortfall_cases'] = sum(int(r['K'])<BUDGETS[stage] for r in samples) if method=='ours_adaptive' else 0
            summaries.append(item)
    write(reports / 'STAGE_SUMMARY.csv',summaries)
    for repeat in range(3):
        repeat_rows=[]
        for stage in range(4):
            for method in METHODS:
                selected=[r for r in rows if int(r['repeat'])==repeat and int(r['stage'])==stage and r['method']==method]
                item=dict(repeat=repeat,stage=stage,method=method)
                for metric in ('prepare_ms','render_ms','pipeline_ms','request_ms'):
                    item[metric]=statistics(selected,metric)['mean']
                repeat_rows.append(item)
        write(reports/f'REPEAT_{repeat+1}_SUMMARY.csv',repeat_rows)
    colors = dict(cartgs_full='#2474B5',ours_full='#44A56A',ours_adaptive='#D05C35')
    labels = dict(cartgs_full='CaRtGS Full',ours_full='Ours Full',ours_adaptive='Ours Adaptive')
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(12.2,3.5),layout='constrained')
        for method in METHODS:
            selected=[r for r in medians if r['method']==method]
            for axis,metric in zip(axes,('radii_positive','render_ms','request_ms')):
                axis.plot([r['frame'] for r in selected],[r[metric] for r in selected],
                    color=colors[method],label=labels[method],linewidth=.8,alpha=.95)
        for axis in axes:
            axis.set_xlabel('Frame index')
            axis.set_xlim(1,792)
            axis.grid(alpha=.18,linewidth=.5)
            axis.set_ylim(bottom=0)
            axis.margins(y=.15)
            for stage,duty in enumerate(DUTIES):
                if stage:
                    axis.axvline(stage*198+.5,color='#737373',ls='--',lw=.7,alpha=.6)
                axis.text((stage+.5)/4,1.025,f'{100*duty:.0f}% load',transform=axis.transAxes,
                          ha='center',fontsize=9)
        axes[0].set_ylabel('Rasterizer-visible Gaussians')
        axes[0].set_title('(a) Visible points (radii > 0)',pad=30)
        axes[1].set_ylabel('Rendering time (ms)')
        axes[1].set_title('(b) Rendering wall time',pad=30)
        axes[2].set_ylabel('Whole request time (ms)')
        axes[2].set_title('(c) Preparation + rendering + gate',pad=30)
        axes[0].legend(loc='upper right',frameon=True,framealpha=.9,fontsize=8)
        fig.savefig(reports/'V27_LOAD_TRAJECTORY.png',dpi=220)
        fig.savefig(reports/'V27_LOAD_TRAJECTORY.pdf')
        plt.close(fig)
    table=[]
    no_load=[]
    for item in summaries:
        budget=BUDGETS[item['stage']] if item['method']=='ours_adaptive' else (35301 if item['method']=='cartgs_full' else 13298)
        table.append(f"| {100*item['target_duty']:.0f}% | {NAMES[item['method']]} | {budget} | {item['prepare_ms_mean']:.3f} | {item['render_ms_mean']:.3f} | {item['pipeline_ms_mean']:.3f} | {item['request_ms_mean']:.3f} | {item['render_ms_std']:.3f} | {item['render_ms_cv_percent']:.1f}% |")
        no_load.append(f"| {item['first_frame']}–{item['last_frame']} | {NAMES[item['method']]} | {item['no_load_render_ms_mean']:.3f} | {item['render_ms_mean']:.3f} | {item['render_ms_mean']-item['no_load_render_ms_mean']:.3f} | {100*item['actual_duty_weighted']:.2f}% |")
    summary_by_key={(r['stage'],r['method']):r for r in summaries}
    high = {m:summary_by_key[3,m] for m in METHODS}
    background = read_json_lines(folder/'GPU_MONITOR.jsonl')
    assert all(not entry['foreign'] for entry in background)
    longest = max(rows,key=lambda row:float(row['render_ms']))
    longest_phase = phases[longest['load_generation']]
    report = f"""# v27完整轨迹人工GPU负载测速

## 结果

完成fr1_xyz RGB-D完整792个视角的三轮对比，共7128次分档请求，另有2376次同视角无人工负载对照。0%／40%／60%／80%分别对应帧1–198／199–396／397–594／595–792。局部预算依次为4000／2000／1000／500点。

80%档下CaRtGS Full平均渲染{high['cartgs_full']['render_ms_mean']:.3f}毫秒，我们Full为{high['ours_full']['render_ms_mean']:.3f}毫秒，自适应500点为{high['ours_adaptive']['render_ms_mean']:.3f}毫秒。自适应方案在此负载构造下没有保持接近0%档的渲染延迟。每个渲染请求从ON段开始，80%档的名义ON长度为16毫秒，这一阻塞时间主导了结果。

## 各档平均耗时

单位均为毫秒。pipeline为准备与渲染之和，整次请求还包含人工负载唤醒、排空和OFF结清等待。标准差与CV基于每档198个视角乘三轮的全部原始样本，未去除离群点。

| 目标负载 | 方法 | 提交K | 准备 | 渲染 | pipeline | 整次请求 | 渲染标准差 | 渲染CV |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(table)}

原始记录中，第{int(longest['repeat'])+1}轮帧{longest['frame']}的{NAMES[longest['method']]}出现{float(longest['render_ms']):.3f}毫秒渲染峰值，该请求包含{longest_phase['pulses']}个ON包。两包记录和约{float(longest['render_gpu_ms']):.3f}毫秒的Stream Event区间均保留。现有计时无法进一步区分中间停顿的具体调度来源。此点计入均值、标准差与CV，40%档CaRtGS的较大CV受到它的影响。主图每帧三轮中位数会减弱单轮峰值，原始值见CSV。

## 同视角无负载对照与实际占空比

无负载对照使用相同视角和相同K，覆盖一轮。每帧重新执行网络准备，未缓存上次选择或补偿结果。Full输出属性来自冻结Master。

| 帧范围 | 方法 | 无负载渲染 | 分档渲染 | 差值 | 实测驻留占空比 |
|---|---|---|---|---|---|
{chr(10).join(no_load)}

分档与视角位置同时变化，直接比较不同段也包含视角复杂度差异。上述同视角0%对照用于辅助区分。对照只有一轮，其温度和频率条件与后续三轮不同，逐帧监控保留在原始文件中。

## 图与原始数据

![全轨迹速度与可见点数](V27_LOAD_TRAJECTORY.png)

横轴为帧序号。各条线逐帧取三轮中位数，沿帧序没有平滑。全部三轮原始样本保存在`ALL_FRAMES_WITH_LOAD.csv`，没有删除离群点。左图可见点数采用Rasterizer预处理后的radii>0数量。提交点数K另存于CSV，CaRtGS为35301，我们Full为13298，自适应按四档预算变化。

`PER_VIEW_MEDIANS.csv`包含每个视角的准备、渲染、完整请求、实际K、候选N相关数据的三轮中位数。`STAGE_SUMMARY.csv`包含均值、中位数、p95、最大值、标准差与CV。`REPEAT_1/2/3_SUMMARY.csv`保存逐轮方向。`runs/trajectory01/TIMINGS.csv`为计时原始记录，两个负载CSV记录每包GPU时间和每请求占空比。

## 准备阶段隔离与正确性

准备Stream实际优先级{environment['priorities']['preparation']}，等于设备最高优先级。人工负载Stream为{environment['hardware']['load_stream_priority']}，渲染Stream为0。每次准备前已等待上一负载包完成并结清OFF。

{acceptance['requests']}次请求及{acceptance['load_packets']}个负载包通过阶段审计，人工负载与准备重叠数为{acceptance['artificial_load_prepare_overlap']}。每包覆盖82个SM，每SM三个512线程Block。所有请求的实测驻留占空比与保守计费占空比均未超过目标。最大目标偏差为{100*acceptance['maximum_target_error']:.3f}个百分点，±5个百分点匹配检查结果为{acceptance['target_match_within_five_percentage_points']}。

边界帧1、198、199、396、397、594、595、792的三个方法、三轮分档输出均与同视角无负载参考逐位一致。所有请求检查了非有限值和自适应实际K，候选不足共{sum(r['budget_shortfall_cases'] for r in summaries)}次。Full使用未补偿Master。准备保护针对人工负载的执行重叠，频率、温度和CPU调度仍会影响墙钟计时。

## 环境与完整性

- v27二进制SHA256为`{environment['v27_sha256']}`，与提供的空闲复测报告一致。
- 实际执行环境为Torch {environment['torch']}、CUDA {environment['cuda']}，与本机当前可用环境对应。提供报告记录的Torch 2.0.1与此次实际版本不同，本轮耗时来自此次完整测量。
- 共{len(background)}次GPU进程监控采样，均未发现本轮之外的CUDA计算进程。
- 两个模型按相同RGB时间戳对齐，各自使用其SLAM保存位姿。相机矩阵在计时前构建，显示、图片编码、模型加载与图计划预热均在计时范围之外。
- CaRtGS调用保存的原版Python渲染入口，包含其每帧属性处理。我们Full采用已有冻结部署Full接口并计入属性发布开销。二者的地图大小和入口处理不同，结果表示当前各自实际部署路径。
- 外部受保护文件{integrity['external_files']}项，本轮源代码与动态库{integrity['implementation_files']}项，前后哈希全部一致。模型和生产目录经过完整性核验，实验存于独立目录。保存渲染图片数量为0。

## 负载形态对结果的影响

人工负载通过全SM驻留制造强资源竞争。当前每次请求与负载ON段同步起步，40%／60%／80%的ON目标长度为8／12／16毫秒。这些长度远大于无负载渲染耗时，因此降低Gaussian数量只能减少ON段结束后的渲染工作，难以抵消固定阻塞。

完整请求还要等本周期OFF时间结清，使有负载时的请求耗时接近20毫秒周期。该等待在request_ms和gate_overhead_ms中完整保存。render_ms用于展示调用方实际等待渲染完成的耗时，CUDA Event区间单独记录，它的起止事件可能在首个负载包之后才获得执行机会。

本轮自适应预算由已知人工负载档位直接映射。预算切换在对应视角进入准备网络之前完成，训练好的Selector与Compensator仍按该预算和当前视角重新推理。
"""
    (reports / 'V27_TRAJECTORY_GPU_LOAD_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),baseline=len(baseline),report=str(reports/'V27_TRAJECTORY_GPU_LOAD_REPORT.md')),ensure_ascii=False))


def read_json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__=='__main__':
    main()
