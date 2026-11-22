"""原始分辨率PI控制：单次延迟、质量与当前自然GPU负载。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parent
METHODS=['fixed_3000','fixed_256','pid_04']
LABELS=['请求3,000点','固定256点（下限对照）','PI：目标0.4 ms']
COLORS=['#353535','#0077BB','#D55E00']


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def longest(mask):
    best=run=0
    for value in mask:
        run=run+1 if value else 0;best=max(best,run)
    return best


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    source=ROOT/'formal';reports=ROOT/'reports';figures=ROOT/'figures'
    reports.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    validation=json.loads((source/'VALIDATION.json').read_text())
    job=json.loads((ROOT/'logs/formal_job.json').read_text())
    assert validation['passed'] and job['returncode']==0 and validation['requests']==18000
    protocol=json.loads((source/'PROTOCOL.json').read_text())
    assert (protocol['width'],protocol['height'])==(1200,680)
    assert validation['native_resolution'] and not validation['GT_resized']
    rows=read(source/'REQUESTS_WITH_QUALITY.csv')
    assert len(rows)==18000
    data={};summary=[];repeat_summary=[]
    for method in METHODS:
        selected=sorted((row for row in rows if row['method']==method),key=lambda row:(int(row['repeat']),int(row['frame'])))
        assert [(int(row['repeat']),int(row['frame'])) for row in selected]==[(r,i) for r in range(3) for i in range(2000)]
        values={key:np.asarray([float(row[key]) for row in selected]) for key in ['repeat','frame','render_ms','K','requested_K','eligible_N','psnr','controller_host_us','telemetry_host_us','group_late_ms','frame_start_unix']}
        assert all(np.isfinite(array).all() for array in values.values())
        data[method]=values
        def stats(mask):
            t=values['render_ms'][mask];k=values['K'][mask]
            return dict(requests=len(t),mean_ms=float(t.mean()),p50_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),
                p99_ms=float(np.percentile(t,99)),max_ms=float(t.max()),std_ms=float(t.std(ddof=0)),
                in_038_042_percent=float(100*np.mean((t>=.38)&(t<=.42))),below_038_percent=float(100*np.mean(t<.38)),
                over_042_percent=float(100*np.mean(t>.42)),over_target_percent=float(100*np.mean(t>.4)),
                mean_absolute_error_ms=float(np.mean(np.abs(t-.4))),K_mean=float(k.mean()),K_min=int(k.min()),K_max=int(k.max()),
                K_at_256_percent=float(100*np.mean(k==256)),K_at_8192_percent=float(100*np.mean(k==8192)),
                requested_at_8192_percent=float(100*np.mean(values['requested_K'][mask]==8192)),
                requested_clipped_percent=float(100*np.mean(values['requested_K'][mask]>k)),
                under_target_at_available_limit_percent=float(100*np.mean((t<.38)&(k>=np.minimum(8192,values['eligible_N'][mask])))),
                psnr_mean=float(values['psnr'][mask].mean()),
                controller_host_us_p50=float(np.median(values['controller_host_us'][mask])),
                controller_host_us_p95=float(np.percentile(values['controller_host_us'][mask],95)),
                telemetry_host_us_p50=float(np.median(values['telemetry_host_us'][mask])),
                group_late_ms_p95=float(np.percentile(values['group_late_ms'][mask],95)))
        s=dict(method=method,**stats(np.ones(6000,dtype=bool)))
        s['longest_over_042_frames']=max(longest(values['render_ms'][values['repeat']==r]>.42) for r in range(3))
        summary.append(s)
        for r in range(3):
            mask=values['repeat']==r
            repeat_summary.append(dict(method=method,repeat=r,**stats(mask),
                longest_over_042_frames=longest(values['render_ms'][mask]>.42)))
    write(reports/'SUMMARY.csv',summary);write(reports/'PER_REPEAT.csv',repeat_summary)
    pi=data['pid_04'];normal=data['fixed_3000'];minimum=data['fixed_256']
    paired=dict(pi_vs_3000_mean_ms_delta=float(np.mean(pi['render_ms']-normal['render_ms'])),
        pi_vs_3000_faster_percent=float(100*np.mean(pi['render_ms']<normal['render_ms'])),
        pi_vs_3000_psnr_delta=float(np.mean(pi['psnr']-normal['psnr'])),
        min256_and_pi_both_over_042_percent=float(100*np.mean((minimum['render_ms']>.42)&(pi['render_ms']>.42))),
        min256_within_038_042_percent=float(100*np.mean((minimum['render_ms']>=.38)&(minimum['render_ms']<=.42))),
        pi_first_90=dict(requests=270,mean_ms=float(pi['render_ms'][pi['frame']<90].mean())),
        pi_after_90=dict(requests=5730,mean_ms=float(pi['render_ms'][pi['frame']>=90].mean()),
            in_band_percent=float(100*np.mean((pi['render_ms'][pi['frame']>=90]>=.38)&(pi['render_ms'][pi['frame']>=90]<=.42)))))
    save(reports/'PAIRED_ANALYSIS.json',paired)
    monitor=[json.loads(line) for line in (ROOT/'logs/formal_MONITOR.jsonl').read_text().splitlines() if line.strip()]
    windows=[(min(data[method]['frame_start_unix'][data[method]['repeat']==r].min() for method in METHODS),
              max(data[method]['frame_start_unix'][data[method]['repeat']==r].max() for method in METHODS)) for r in range(3)]
    samples=[]
    for item in monitor:
        if not any(lo-1<=item['time']<=hi+1 for lo,hi in windows):continue
        foreign=item['foreign_active'];sm=0.
        for line in foreign:
            fields=line.split()
            if fields[3].isdigit():sm+=int(fields[3])
        parts=item['gpu'].split(',')
        samples.append(dict(unix=item['time'],gpu_util_percent=float(parts[0].split()[0]),
            external_active=bool(foreign),external_SM_sum=sm,external_active_processes=len(foreign),gpu_telemetry=item['gpu']))
    assert samples
    env=dict(samples=len(samples),external_active_samples=sum(row['external_active'] for row in samples),
        external_active_percent=100*sum(row['external_active'] for row in samples)/len(samples),
        gpu_util_mean=float(np.mean([row['gpu_util_percent'] for row in samples])),
        external_SM_sum_mean=float(np.mean([row['external_SM_sum'] for row in samples])),
        same_external_workload_guaranteed=False,paired_methods_interleaved=True)
    write(reports/'TIMING_WINDOW_GPU_LOAD.csv',samples);save(reports/'LOAD_SUMMARY.json',env)

    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    fig,axes=plt.subplots(2,2,figsize=(12.6,7.1))
    for method,label,color in zip(METHODS,LABELS,COLORS):
        values=data[method];mask=values['repeat']==0;x=values['frame'][mask]
        axes[0,0].plot(x,values['K'][mask],color=color,label=label,lw=1.)
        axes[0,1].plot(x,values['render_ms'][mask],color=color,lw=.65,alpha=.85)
        axes[1,0].plot(x,gaussian_filter1d(values['psnr'][mask],35,mode='reflect'),color=color,lw=1.3)
    axes[0,0].set(title='(a) 实时提交预算',ylabel='实际高斯点数 K')
    axes[0,1].set(title='(b) 单次纯渲染时间（不平滑）',ylabel='渲染时间 (ms)')
    axes[0,1].axhspan(.38,.42,color='#D55E00',alpha=.12)
    axes[0,1].axhline(.4,color='#636363',ls='--',lw=.9)
    axes[1,0].set(title='(c) 渲染质量',ylabel='PSNR (dB)')
    first=pi['repeat']==0
    load_frames=np.interp([row['unix'] for row in samples],pi['frame_start_unix'][first],pi['frame'][first],left=-1,right=2001)
    valid=(load_frames>=0)&(load_frames<=1999)
    axes[1,1].step(load_frames[valid],np.asarray([row['gpu_util_percent'] for row in samples])[valid],where='mid',color='#756BB1',label='GPU总利用率')
    axes[1,1].step(load_frames[valid],np.asarray([row['external_SM_sum'] for row in samples])[valid],where='mid',color='#009988',label='外部进程SM指标之和')
    axes[1,1].set(title='(d) 同期自然GPU负载（约1 Hz采样）',ylabel='利用率 / SM指标 (%)',ylim=(0,None))
    axes[1,1].legend(frameon=False,loc='lower left',fontsize=8)
    for axis in axes.flat:
        axis.set_xlim(0,1999);axis.set_xlabel('RGB帧编号');axis.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.51,.965),frameon=False)
    fig.suptitle('Replica office2 RGB-D｜原始1200×680｜Mode40｜PI控制',y=1.005,fontsize=13)
    fig.tight_layout(rect=(0,.06,1,.91))
    fig.text(.51,.025,'图示第一轮完整2000帧；表格统计三轮全部单次请求。仅PSNR平滑，耗时和点数不平滑。',ha='center',fontsize=9)
    for extension in ['png','svg']:fig.savefig(ROOT/'figures'/('PID_NATIVE1200_CONTROL.'+extension),dpi=450,bbox_inches='tight')
    plt.close(fig)
    lines=['# 原始1200×680分辨率：真实场景PI控制复测','',
        '## 结论与数据范围','',
        f'保持PI增益及0.4 ms目标不变，将图像改回原始1200×680。PI平均耗时{summary[-1]["mean_ms"]:.6f} ms，'
        f'p95 {summary[-1]["p95_ms"]:.6f} ms，{summary[-1]["in_038_042_percent"]:.2f}%请求进入0.38–0.42 ms，'
        f'低于0.38 ms为{summary[-1]["below_038_percent"]:.2f}%，高于0.42 ms为{summary[-1]["over_042_percent"]:.2f}%。'
        '是否跟踪成功须结合双侧误差、预算上下限和当前负载判断，不能只用平均耗时评判。','',
        '在当前自然GPU负载下，用同一冻结office2 RGB-D模型、蒸馏Compensator、Tensor Core D、新准备及Mode40测试。'
        '分辨率1200×680，完整2000帧重复三轮；每个视角交错执行请求3000点、最低预算256点和PI三种方法，共18000次单图调用。'
        '目标固定0.4 ms、允许区间0.38–0.42 ms，未根据结果改目标或选择样本。','',
        '| 方法 | 平均ms | p95 ms | p99 ms | 区间命中率 | >0.42 ms | 平均K | PSNR/dB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s,label in zip(summary,LABELS):
        lines.append(f'|{label}|{s["mean_ms"]:.6f}|{s["p95_ms"]:.6f}|{s["p99_ms"]:.6f}|{s["in_038_042_percent"]:.2f}%|{s["over_042_percent"]:.2f}%|{s["K_mean"]:.2f}|{s["psnr_mean"]:.5f}|')
    s=summary[-1]
    lines+=['',f'PI有{s["K_at_256_percent"]:.2f}%的请求达到256点下限；最长连续超过0.42 ms为{s["longest_over_042_frames"]}帧。'
        f'固定256点仍有{summary[1]["over_042_percent"]:.2f}%的请求超过0.42 ms。',
        f'PI请求达到8192上限的比例为{s["requested_at_8192_percent"]:.2f}%；'
        f'实际K达到8192比例为{s["K_at_8192_percent"]:.2f}%；'
        f'已到当前有效预算上限且低于0.38 ms的请求占{s["under_target_at_available_limit_percent"]:.2f}%。'
        '这类帧不能通过本次允许的继续加点达到目标，不延时凑时长。',
        f'PI相对固定3000点平均耗时差为{paired["pi_vs_3000_mean_ms_delta"]:+.6f} ms，平均PSNR差为{paired["pi_vs_3000_psnr_delta"]:+.5f} dB。'
        '这些差异同时受到预算与自然负载变化影响，不把交错测量视为完全相同外部负载的确定性因果对照。','',
        '## 负载与测量','',
        f'- 正式计时窗口共{env["samples"]}个监控样本，外部计算活动出现比例{env["external_active_percent"]:.2f}%，GPU总利用率均值{env["gpu_util_mean"]:.2f}%。',
        '- 监控保留逐进程pmon与GPU频率、温度、显存和功耗；约1 Hz采样不能定位每个亚毫秒请求的精确竞争状态。',
        '- 三方法在同一视角组内轮换顺序，视角组按90 Hz节奏提交；每种方法各一次单图请求，组内三个方法依次执行，不代表单方法端侧FPS。',
        '- 全部单次CUDA Event耗时保留，不取每帧三次中位数、不因负载或耗时剔除样本。',
        f'- 请求3000点的对照保留原候选不足裁剪，实际平均{summary[0]["K_mean"]:.2f}点、最低{summary[0]["K_min"]}点，没有补点。',
        '- PID只用过去完成的Event反馈，不使用当前帧试渲染、GT、离线最优K或耗时曲线。',
        '- 使用原生控制器的同一Event对完成计时；在原prepare已有发布同步后额外做非阻塞Query及日志读取，保证完整收集上一帧反馈。'
        '该遥测放在当前帧计时外，没有新增每帧等待、准备或渲染。',
        '- prepare及网络不计入纯渲染目标；Host控制API耗时与额外遥测分别记录，不能将Host提交时间称作完整GPU完成时延。','',
        '## 控制开销与画质','',
        f'- PI request/start/finish三个API的Host耗时中位数{s["controller_host_us_p50"]:.3f}微秒，p95 {s["controller_host_us_p95"]:.3f}微秒。'
        '这是当前自然负载下API区间实测，不是控制器有无的配对端到端开销差值。',
        f'- 额外非阻塞Query和日志遥测中位数{s["telemetry_host_us_p50"]:.3f}微秒，计时外单独列出。',
        '- 画质在计时结束后按记录的请求预算重放评价，逐项核对K/M/ratio；PSNR从不进入控制器。GT使用原始1200×680 RGB，不缩放。',
        '- 图只展示第一轮原始逐帧耗时；PSNR采用σ=35帧展示平滑。CSV及表格采用原始单次结果。','',
        '## 可达性与限制','',
        '- 最低256点是当前配置的下限，不是数学上的全局最低成本。该对照若仍超时，说明不能靠本配置的继续减点消除所有超时。',
        '- 同一K不保证同一耗时；GPU竞争、频率、视角覆盖面积、遮挡及ratio相关补偿都可影响工作量。',
        '- 若大量请求停在上限仍偏快，或停在下限仍超时，均属于当前允许预算范围内的目标可达性问题，不能仅凭平均值称为稳定控制成功。',
        '- 三轮分别从初始预算3000点启动；稳态和前90帧分别记录，未隐藏启动阶段。','',
        '## 与上一轮高分辨率的关系','',
        '- 上一轮2064×2208期间外部任务活跃；本轮启动时GPU空闲，完整负载监控另列。结果不能直接归因于分辨率一个变量。',
        '- 两轮渲染器、模型、PID参数、预算范围、目标和统计方式相同；两轮质量各用对应分辨率GT，不直接将PSNR差解释为重建训练变化。',
        '- 若要隔离分辨率本身的影响，应在同轮同负载中交错测两分辨率；此次按用户要求只重测原始分辨率。','',
        '## 后续方向','',
        '- 若目标只是延迟上限，应允许低于目标，不通过增加无必要点数或等待追齐时间；若目标是固定耗时，须先检查当前预算是否能达到目标。',
        '- 保持画质下限，并单独记录目标不可达状态；不因瞬时负载尖峰无限减点。',
        '- 本轮不根据结果重新调增益、扩预算或改变目标，保留直接复测的可比性。','',
        '## 完整性','',
        f'- {validation["protected_files"]}个受保护文件SHA未变，原PID源码与动态库未变；无训练、无编译、无生产切换、无commit/push、无图片保存。',
        '- 完整请求及质量：formal/REQUESTS_WITH_QUALITY.csv；三轮汇总：reports/PER_REPEAT.csv；外部负载：logs/formal_MONITOR.jsonl。','']
    (ROOT/'PID_NATIVE1200_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[source/'REQUESTS_WITH_QUALITY.csv',source/'PROTOCOL.json',source/'VALIDATION.json',ROOT/'logs/formal_MONITOR.jsonl',ROOT/'benchmark.py',Path(__file__)]
    save(figures/'DATA_MANIFEST.json',dict(real_data=True,sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        dpi=450,png_svg=True,plotted_repeat=0,all_repeat_metrics=True,raw_latency=True,psnr_sigma_frames=35))
    (figures/'data-manifest.md').write_text('# 数据清单\n\n图表来自本轮原始1200×680分辨率、自然GPU负载真实测量。'
        '输入为formal/REQUESTS_WITH_QUALITY.csv与logs/formal_MONITOR.jsonl，脚本analyze.py；输出PNG/SVG，450 DPI。'
        '图示第一轮全部2000帧；统计覆盖三轮18000次单次请求。仅PSNR展示平滑，其他曲线不平滑。输入SHA见DATA_MANIFEST.json。\n',encoding='utf-8')
    save(reports/'ANALYSIS_VALIDATION.json',dict(passed=True,requests=18000,all_single_samples=True,no_filtering=True,quality_budget_match=True,
        monitor_samples=env['samples'],protected_files=validation['protected_files']))
    print(json.dumps(dict(summary=summary,paired=paired,load=env),ensure_ascii=False))


if __name__=='__main__':main()
