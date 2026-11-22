"""汇总v13准备保护实验，保留原始请求并生成中文报告。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS=('cartgs_full','ours_fixed4000','ours_adaptive')
LABELS=('CaRtGS Full','Ours, fixed 4k','Ours, adaptive K')
COLORS=('#36393F','#D55E00','#0072B2')


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
    qmap={int(v['budget']):float(v['psnr']) for v in quality if v['method']=='ours'}
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
            recoveries.append(dict(repeat=repeat,method=method,final_K=int(final[-1]['K']),
                first_4k_s=None if first is None else float(first['elapsed_s'])-5*env['stage_seconds']))
    write(out/'STAGE_SUMMARY.csv',summaries)
    write(out/'RECOVERY.csv',recoveries)
    plot(rows,quality,env,out,repeats)
    report(run,out,env,rows,quality,summaries,recoveries)
    (out/'ANALYSIS.json').write_text(json.dumps(dict(passed=True,rows=len(rows),
        repeats=len(repeats),conditions=len(repeats)*3,stages=len(summaries),
        artificial_load_prepare_overlap=0,dropped_samples=0,timing_smoothing=False),indent=2))
    print('RENDER_ONLY_ANALYSIS_PASS',out)


def plot(rows,quality,env,out,repeats):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    qmap={(v['method'],int(v['budget'])):float(v['psnr']) for v in quality}
    for repeat in repeats:
        fig,axes=plt.subplots(2,3,figsize=(11.3,6.1))
        axes=axes.ravel()
        edge=np.arange(7)*env['stage_seconds']
        axes[0].step(edge,[100*v for v in env['duties']]+[0],where='post',color='#6B7280',lw=1.8)
        for method,color,label in zip(METHODS,COLORS,LABELS):
            selected=[v for v in rows if v['method']==method and int(v['repeat'])==repeat]
            x=[float(v['elapsed_s']) for v in selected]
            axes[1].plot(x,[int(v['K']) for v in selected],color=color,lw=1,label=label)
            for axis,key in ((axes[2],'prepare_ms'),(axes[3],'render_ms'),(axes[4],'total_ms')):
                axis.scatter(x,[float(v[key]) for v in selected],color=color,s=.65,alpha=.5,linewidths=0,label=label)
            psnr=[qmap['cartgs_full',35301] if method=='cartgs_full' else qmap['ours',int(v['K'])] for v in selected]
            axes[5].plot(x,psnr,color=color,lw=.9,label=label)
        axes[0].set(title='Injected load during rendering',ylabel='Target duty (%)',ylim=(0,100))
        axes[1].set(title='Actual input Gaussians',ylabel='K',yscale='log',ylim=(800,43000))
        axes[2].set(title='Protected preparation',ylabel='Latency (ms)',ylim=(0,None))
        axes[3].set(title='Rendering under injected load',ylabel='Latency (ms)',ylim=(0,None))
        axes[4].set(title='Preparation + rendering',ylabel='Latency (ms)',ylim=(0,None))
        axes[4].axhline(env['target_ms'],color='#B91C1C',ls='--',lw=1,label=f"Target {env['target_ms']:.3f} ms")
        axes[5].set(title='Quality at the active budget',ylabel='PSNR (dB)')
        for index,axis in enumerate(axes):
            axis.set(xlabel='Elapsed time (s)',xlim=(0,edge[-1]))
            axis.grid(alpha=.16)
            for boundary in edge[1:-1]:
                axis.axvline(boundary,color='#AAAAAA',lw=.5,ls=':')
            if index in (1,4):
                axis.legend(fontsize=6.2,loc='upper right',frameon=False)
        fig.suptitle(f'v13 | Highest-priority preparation | Render-only injected load | Run {repeat+1}',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.95))
        fig.text(.5,.012,'Shared GPU. All latency samples retained. Gate overhead reported separately. Quality evaluated outside timing.',
            ha='center',fontsize=7,color='#555555')
        for extension in ('png','pdf'):
            fig.savefig(out/f'V13_RENDER_ONLY_REPEAT{repeat}.{extension}',dpi=220)
        plt.close(fig)


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
    recovery_lines=[]
    for value in recovery:
        if value['method']=='ours_adaptive':
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
    text=f'''# v13 最高优先级准备与渲染阶段负载实验

## 实测结果

完成{len(repeats)}轮、{len(repeats)*3}组，共{len(rows)}次计时请求。人工负载80%阶段，自适应实际K集合为{ks}，合并总耗时均值{heavy.mean():.4f} ms、P95={np.percentile(heavy,95):.4f} ms，超出预定{env['target_ms']:.4f} ms目标的请求占{100*np.mean(heavy>env['target_ms']):.2f}%。包含负载握手的完整请求墙钟均值为{complete.mean():.4f} ms。

全量负载包检查通过，人工负载与准备区间的重叠数为0。准备CUDA Stream实际优先级为{env['priorities']['preparation']}，设备最高优先级为{env['priorities']['greatest']}，渲染和人工负载Stream优先级为{env['priorities']['rendering']}／{env['priorities']['load']}。CPU nice请求{cpu['requested']}，实际为{cpu['actual']}，设置成功状态为{cpu['set_success']}。现有权限下GPU最高优先级生效，CPU提升请求的具体状态见ENVIRONMENT.json。

本轮满足人工负载阶段隔离要求，延迟目标仍有超限。三轮自适应的80%阶段总耗时均值分别为{', '.join(format(v['total_mean_ms'],'.4f') for v in summary if v['method']=='ours_adaptive' and v['stage']==4)} ms。仅第二轮均值低于预定目标，逐轮P95均高于目标。其他项目的GPU调度仍影响准备和渲染，因此共享GPU环境下的准备时间会继续波动。

## 80%人工负载的逐轮结果

{chr(10).join(table)}

各方法分开执行相同日程并循环轮换顺序，其他GPU服务继续运行。既有服务的逐秒活动保存在pmon_samples.json，具体PID和命令行保存在pmon_process_identities.json，逐阶段背景SM均值位于STAGE_SUMMARY.csv。

相同目标人工负载下，固定4000点与自适应的后台负载并非同时发生。第一和第三轮自适应准备时间高于固定4000点，第二轮较低。逐轮差异不能全部归因于点数与网络计算。合并统计按请求数量加权，较慢阶段在固定时长内完成的请求更少，故同时给出逐轮结果。

## 准备阶段检查

{chr(10).join(prep)}

人工负载由低优先级Stream上的原生线程控制，准备前确认上一包完成。最高优先级Stream仍受正在运行的其他项目Kernel和跨进程GPU调度影响。上述准备耗时变化应结合后台活动解释。

## 调度与计时定义

本轮使用v13并采用同进程原生负载线程。准备最高优先级Stream同步完成后，恢复人工负载并确认首包入队，再调用原Rasterizer。Rasterizer同步结束后暂停负载并排空有限尾包。每个负载包的CPU提交／完成区间必须完全位于该请求的渲染阶段内。与上轮v11独立负载进程相比，执行上下文和负载作用阶段均发生变化。

total_ms为prepare_ms与render_ms之和，包含算法所需的CPU调用和对应CUDA Stream同步。gate_overhead_ms为恢复负载、暂停和尾包排空等待。wall_ms覆盖包含握手的完整请求。控制器依据total_ms调整预算，两个口径均保留供读者核查。

人工负载目标为0→20→40→60→80→0%，每阶段{env['stage_seconds']}秒。每1ms周期提交目标时长的有限计算脉冲，每个新渲染窗口重新开始周期，因此短渲染窗口的有效暴露可能高于目标占空比。80%阶段负载事件时间／渲染阶段长度的观测范围为{min(load_range):.1f}%–{max(load_range):.1f}%。该指标包含Kernel调度等待，pmon记录的是整个本轮进程的GPU活动，两者都不等同于人工负载的精确SM资源份额。

预算控制沿用四档4000／3000／2000／1000，每10帧计算平均耗时，超过目标下降一档，连续两窗口低于目标85%上升一档。目标来自当前共享GPU下、人工负载关闭时4000点200次测量P95×1.10。请求周期5ms，跨阶段请求有独立标记。全部样本保留，耗时不平滑。

## 恢复

{recovery_text}

## 渲染质量

{chr(10).join(quality_table)}

使用TUM fr1_xyz RGB-D第396帧，相同RGB图像，两个模型各自SLAM估计位姿。质量在计时外计算，时序中的质量按实际预算绑定。v13与上轮v11的该视角输出像素逐位一致。v13在默认Stream和最高优先级Stream上的11组预算测试覆盖0、1、100、1000、1023、1024、2000、3000、4000、6632、10000，ID、属性和像素逐位一致。保存图片数量为0，图表为统计曲线。

## 完整性及文件

v13二进制SHA256为`{env['binary_sha256']}`。独立运行目录为`{run.parent.parent}`。INTEGRITY.json记录生产、模型和旧实验输入的前后哈希核验。原Rasterizer按原实现执行。

初次短验收发现LibTorch Stream池未得到请求的设备最高优先级，随后改用CUDA Runtime创建Stream并以ExternalStream交给LibTorch。该失败现场保存在reports/stream_pool_priority_failure，后续短验收和正式测试均核对实际优先级为−5。

TIMINGS.csv保留逐请求prepare、render、total、wall和握手耗时、GPU Event时间、实际K和全部阶段时间戳。每组LOAD_PACKETS.csv用于复核人工负载边界，ACCEPTANCE.json保存零重叠检查。STAGE_SUMMARY.csv保留全部阶段的均值、P95、P99、标准差、CV和超目标比例。V13_RENDER_ONLY_REPEAT0／1／2.png和PDF展示三轮时序。
'''
    (out/'V13_RENDER_ONLY_LOAD_REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
