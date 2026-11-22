"""汇总v21准备保护实验，保留原始请求并生成中文报告。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS=('cartgs_full','ours_fixed4000','ours_adaptive')


def read(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run
    assert json.loads((run/'COMPLETION.json').read_text())['state']=='complete'
    env=json.loads((run/'ENVIRONMENT.json').read_text())
    rows=read(run/'TIMINGS.csv')
    quality=read(run/'QUALITY.csv')
    activity=json.loads((run/'pmon_samples.json').read_text())
    pid=json.loads((run/'COMPLETION.json').read_text())['process']['pid']
    out=run/'reports'
    out.mkdir(exist_ok=True)
    summaries=[]
    recoveries=[]
    repeats=sorted({int(v['repeat']) for v in rows})
    for repeat in repeats:
        for method in METHODS:
            part=run/f'{repeat}_{method}'
            audit=json.loads((part/'ACCEPTANCE.json').read_text())
            assert audit['passed'] and audit['artificial_load_prepare_overlap']==0
            selected=[v for v in rows if v['method']==method and int(v['repeat'])==repeat]
            packets=read(part/'LOAD_PACKETS.csv')
            assert len(selected)==audit['requests']
            assert all(int(v['K'])==int(v['requested_K']) for v in selected)
            for stage,duty in enumerate(env['duties']):
                group=[v for v in selected if int(v['stage'])==stage]
                values={key:np.array([float(v[key]) for v in group]) for key in
                    ('K','prepare_ms','render_ms','total_ms','wall_ms','gate_overhead_ms','prepare_gpu_ms')}
                times=values['total_ms']
                phase_ms=sum(1000*(float(v['render_phase_end_s'])-float(v['render_phase_start_s'])) for v in group)
                load_ms=sum(float(v['event_ms']) for v in packets if int(v['stage'])==stage)
                start=audit['stages'][stage]['monotonic_s']
                others=[sum(v['sm'] for v in sample['records'] if v['pid']!=pid)
                    for sample in activity if start+1<=sample['monotonic_s']<start+env['stage_seconds']]
                summaries.append(dict(repeat=repeat,method=method,stage=stage,target_duty_percent=100*duty,
                    samples=len(group),min_K=int(values['K'].min()),max_K=int(values['K'].max()),mean_K=values['K'].mean(),
                    prepare_mean_ms=values['prepare_ms'].mean(),prepare_p95_ms=np.percentile(values['prepare_ms'],95),
                    prepare_gpu_mean_ms=values['prepare_gpu_ms'].mean(),render_mean_ms=values['render_ms'].mean(),
                    render_p95_ms=np.percentile(values['render_ms'],95),total_mean_ms=times.mean(),
                    total_p95_ms=np.percentile(times,95),total_p99_ms=np.percentile(times,99),
                    std_ms=times.std(),cv_percent=100*times.std()/times.mean(),
                    over_target_percent=100*np.mean(times>env['target_ms']),
                    wall_mean_ms=values['wall_ms'].mean(),gate_mean_ms=values['gate_overhead_ms'].mean(),
                    load_event_ms_over_render_phase_percent=100*load_ms/phase_ms,
                    ambient_sm_mean=np.mean(others) if others else None))
            final=[v for v in selected if int(v['stage'])==5]
            first=next((v for v in final if int(v['K'])==4000),None)
            recoveries.append(dict(repeat=repeat,method=method,ever_reduced=any(int(v['K'])<4000 for v in selected),
                final_K=int(final[-1]['K']),
                first_4k_s=None if first is None else float(first['elapsed_s'])-5*env['stage_seconds']))
    write(out/'STAGE_SUMMARY.csv',summaries)
    write(out/'RECOVERY.csv',recoveries)
    report(run,out,env,rows,quality,summaries,recoveries)
    (out/'ANALYSIS.json').write_text(json.dumps(dict(passed=True,rows=len(rows),
        repeats=len(repeats),conditions=len(repeats)*3,stages=len(summaries),
        artificial_load_prepare_overlap=0,dropped_samples=0,timing_smoothing=False),indent=2))
    print('RENDER_ONLY_ANALYSIS_PASS',out)


def report(run,out,env,rows,quality,summary,recovery):
    repeats=sorted({v['repeat'] for v in summary})
    table=['|轮次|方法|K范围|准备均值 ms|渲染均值 ms|总均值 ms|总P95 ms|超目标 %|握手均值 ms|完整墙钟均值 ms|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        if v['stage']==4:
            table.append(f"|{v['repeat']+1}|{v['method']}|{v['min_K']}–{v['max_K']}|{v['prepare_mean_ms']:.4f}|"
                f"{v['render_mean_ms']:.4f}|{v['total_mean_ms']:.4f}|{v['total_p95_ms']:.4f}|{v['over_target_percent']:.1f}|"
                f"{v['gate_mean_ms']:.4f}|{v['wall_mean_ms']:.4f}|")
    quality_table=['|模型／预算|K|PSNR dB|','|---|---:|---:|']
    for v in quality:
        quality_table.append(f"|{v['method']} / {v['budget']}|{v['K']}|{float(v['psnr']):.4f}|")
    selected=[v for v in rows if v['method']=='ours_adaptive' and int(v['stage'])==4]
    heavy=np.array([float(v['total_ms']) for v in selected])
    complete=np.array([float(v['wall_ms']) for v in selected])
    ks=sorted({int(v['K']) for v in selected})
    cpu=env['cpu_priority']
    calibration=json.loads((run/'CALIBRATION.json').read_text())['samples_ms']
    adaptive_rows=[value for value in rows if value['method']=='ours_adaptive']
    adaptive_budgets=sorted({int(value['K']) for value in adaptive_rows})
    graph_rows=[value for value in rows if value['method']!='cartgs_full']
    assert all(value['graph_replay']=='True' for value in graph_rows)
    graph_counts=sorted({int(value['graph_count']) for value in graph_rows})
    recovery_lines=[]
    for value in recovery:
        if value['method']=='ours_adaptive':
            if not value['ever_reduced']:
                detail='全过程保持4000点，未触发预算降低'
            else:
                detail='最后阶段未恢复4000点' if value['first_4k_s'] is None else f"首次恢复4000点用时{value['first_4k_s']:.3f}秒"
            recovery_lines.append(f"- 第{value['repeat']+1}轮结束K={value['final_K']}，{detail}。")
    recovery_text='\n'.join(recovery_lines)
    prep=[]
    for method in METHODS[1:]:
        zero=[v for v in summary if v['method']==method and v['stage']==0]
        high=[v for v in summary if v['method']==method and v['stage']==4]
        prep.append(f"- {method}的准备均值在人工负载0%时为"
            f"{', '.join(format(v['prepare_mean_ms'],'.4f') for v in zero)} ms，80%时为"
            f"{', '.join(format(v['prepare_mean_ms'],'.4f') for v in high)} ms，按轮次排列。")
    load_range=[v['load_event_ms_over_render_phase_percent'] for v in summary if v['stage']==4]
    adaptive_high=[v for v in summary if v['method']=='ours_adaptive' and v['stage']==4]
    below_mean=sum(v['total_mean_ms']<=env['target_ms'] for v in adaptive_high)
    below_p95=sum(v['total_p95_ms']<=env['target_ms'] for v in adaptive_high)
    text=f'''# v21 最高优先级准备与渲染阶段负载实验

## 实测结果

完成{len(repeats)}轮、{len(repeats)*3}组，共{len(rows)}次计时请求。人工负载80%阶段，自适应实际K集合为{ks}，合并总耗时均值{heavy.mean():.4f} ms、P95={np.percentile(heavy,95):.4f} ms，超出预定{env['target_ms']:.4f} ms目标的请求占{100*np.mean(heavy>env['target_ms']):.2f}%。包含负载握手的完整请求墙钟均值为{complete.mean():.4f} ms。

本次自适应运行全过程的K集合为{adaptive_budgets}。标定的200次请求中有{sum(value>5 for value in calibration)}次超过5 ms，中位数为{np.median(calibration):.4f} ms，范围为{min(calibration):.4f}至{max(calibration):.4f} ms。按事先规则得到的目标8.4341 ms受到后台排队影响，控制器全过程保持4000点。本轮可报告v21在该环境下的准备、渲染及阶段隔离结果，自适应降低点数的效果尚未在该轮触发。达到此目标的比例不能作为稳定低延迟的证明。

本轮用户允许在其他服务性能占用较低时开始，使用共享GPU实测口径。现场PyTorch为{env['torch']}、CUDA运行时为{env['cuda']}，原v21报告中的环境字段为PyTorch 2.0.1+cu117，本轮以实际加载、逐位验证和ENVIRONMENT.json记录为准。v21开启prepare_graph与saturated_selection，所有预算在正式计时前预热，逐请求graph_replay和graph_count保存在TIMINGS.csv。

本方法的{len(graph_rows)}次计时请求均核验Graph Replay生效，Graph计划数量集合为{graph_counts}。本轮准备使用最高优先级−5 Stream，Graph的首次捕获及预热已在计时前完成，Replay使用调用者当前Stream。

全量负载包检查通过，人工负载与准备区间的重叠数为0。准备CUDA Stream实际优先级为{env['priorities']['preparation']}，设备最高优先级为{env['priorities']['greatest']}，渲染和人工负载Stream优先级为{env['priorities']['rendering']}／{env['priorities']['load']}。CPU nice请求{cpu['requested']}，实际为{cpu['actual']}，设置成功状态为{cpu['set_success']}。现有权限下GPU最高优先级生效，CPU提升请求的具体状态见ENVIRONMENT.json。

本轮满足人工负载阶段隔离要求。三轮自适应的80%阶段总耗时均值分别为{', '.join(format(v['total_mean_ms'],'.4f') for v in adaptive_high)} ms，其中{below_mean}/3轮的均值和{below_p95}/3轮的P95达到预定目标。其他项目的GPU调度仍影响准备和渲染，背景活动与逐轮数据共同记录。

## 80%人工负载的逐轮结果

{chr(10).join(table)}

各方法分开执行相同日程并循环轮换顺序，其他GPU服务继续运行。既有服务的逐秒活动保存在pmon_samples.json，具体PID和命令行保存在pmon_process_identities.json，逐阶段背景SM均值位于STAGE_SUMMARY.csv。

相同目标人工负载下，固定4000点与自适应的后台负载并非同时发生。逐轮背景活动存在差异，测量差异应结合点数、网络计算和GPU争用共同分析。合并统计按请求数量加权，较慢阶段在固定时长内完成的请求更少，故同时给出逐轮结果。

## 准备阶段检查

{chr(10).join(prep)}

人工负载由低优先级Stream上的原生线程控制，准备前确认上一包完成。最高优先级Stream仍受正在运行的其他项目Kernel和跨进程GPU调度影响。上述准备耗时变化应结合后台活动解释。

## 调度与计时定义

本轮使用v21并采用同进程原生负载线程。准备最高优先级Stream同步完成后，恢复人工负载并确认首包入队，再调用原Rasterizer。Rasterizer同步结束后暂停负载并排空有限尾包。每个负载包的CPU提交／完成区间必须完全位于该请求的渲染阶段内。与上轮v11独立负载进程相比，执行上下文和负载作用阶段均发生变化。

total_ms为prepare_ms与render_ms之和，包含算法所需的CPU调用和对应CUDA Stream同步。gate_overhead_ms为恢复负载、暂停和尾包排空等待。wall_ms覆盖包含握手的完整请求。控制器依据total_ms调整预算，两个口径均保留供读者核查。

人工负载目标为0→20→40→60→80→0%，每阶段{env['stage_seconds']}秒。每1ms周期提交目标时长的有限计算脉冲，每个新渲染窗口重新开始周期，因此短渲染窗口的有效暴露可能高于目标占空比。80%阶段负载事件时间／渲染阶段长度的观测范围为{min(load_range):.1f}%–{max(load_range):.1f}%。该指标包含Kernel调度等待，pmon记录的是整个本轮进程的GPU活动，两者都不等同于人工负载的精确SM资源份额。

预算控制沿用四档4000／3000／2000／1000，每10帧计算平均耗时，超过目标下降一档，连续两窗口低于目标85%上升一档。目标来自当前共享GPU下、人工负载关闭时4000点200次测量P95×1.10。请求周期5ms，跨阶段请求有独立标记。全部样本保留，耗时不平滑。

## 恢复

{recovery_text}

## 渲染质量

{chr(10).join(quality_table)}

使用TUM fr1_xyz RGB-D第396帧，相同RGB图像，两个模型各自SLAM估计位姿。质量在计时外计算，时序中的质量按实际预算绑定。v21与上轮v11的该视角输出像素逐位一致。v21在默认Stream和最高优先级Stream上的11组预算测试覆盖0、1、100、1000、1023、1024、2000、3000、4000、6632、10000，ID、属性和像素逐位一致。保存图片数量为0，图表为统计曲线。

## 完整性及文件

v21二进制SHA256为`{env['binary_sha256']}`。独立运行目录为`{run.parent.parent}`。INTEGRITY.json记录生产、模型和旧实验输入的前后哈希核验。原Rasterizer按原实现执行。

本轮复用已在v13负载实验验收的CUDA Runtime Stream创建方法，并以ExternalStream交给LibTorch。v21短验收和正式测试均核对实际准备优先级为−5。

TIMINGS.csv保留逐请求prepare、render、total、wall和握手耗时、GPU Event时间、实际K和全部阶段时间戳。每组LOAD_PACKETS.csv用于复核人工负载边界，ACCEPTANCE.json保存零重叠检查。STAGE_SUMMARY.csv保留全部阶段的均值、P95、P99、标准差、CV和超目标比例。line_plots中的V21_LINE_REPEAT0／1／2.png和PDF展示三轮时序，耗时采用0.25秒分箱均值折线和P10至P90色带。
'''
    (out/'V21_RENDER_ONLY_LOAD_REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
