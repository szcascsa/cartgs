"""在本地生成中文图表和报告，保持服务器纯数值评价。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'reports';FIG=OUT/'figures'
NAMES={'pytorch':'原PyTorch','cuda':'精确CUDA v3','deployment':'新部署MLP'}
COLORS={'pytorch':'#64748B','cuda':'#0072B2','deployment':'#D55E00'}


def load(n):return json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def main():
    FIG.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    summary=load('SUMMARY');s={(v['backend'],v['case']):v for v in summary if v['repeat']=='all'}
    qs={v['case']:v for v in load('QUALITY_SUMMARY')};components=load('COMPONENT_SUMMARY');env=load('ENVIRONMENT')
    pairs=load('PAIRED_SPEED');gates=load('QUALITY_GATES');profile=load('PROFILE_SUMMARY')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,case,label in zip(axes,['points500','ratio50'],['局部500点','局部50%']):
        for x,tag in enumerate(NAMES):
            row=s[tag,case];ax.bar(x,row['inference_ms_mean'],color=COLORS[tag],label='推理' if x==0 else None)
            ax.bar(x,row['render_ms_mean'],bottom=row['inference_ms_mean'],color=COLORS[tag],alpha=.35,label='Rasterizer' if x==0 else None)
            ax.text(x,row['total_ms_mean'],f' {row["total_ms_mean"]:.3f} ms',ha='center',va='bottom',fontsize=10)
        ax.set(title=label,ylabel='同步墙钟均值（ms）',xticks=range(3),xticklabels=list(NAMES.values()));ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('相同四Head模型｜两网络真实视角＋LI=0.1｜792帧×3轮',fontsize=13)
    fig.text(.5,.012,'完整prepare包含候选、视角、两网络、严格K及finite检查；不含模型加载、JIT、GT、显示。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));savefig(fig,'LOCAL500_RATIO50_LATENCY')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,case,label in zip(axes,['points500','ratio50'],['局部500点','局部50%']):
        for tag,name in NAMES.items():
            values=[next(v for v in summary if v['backend']==tag and v['case']==case and v['repeat']==str(repeat)) for repeat in range(3)]
            ax.plot([1,2,3],[v['total_ms_mean'] for v in values],marker='o',label=name,color=COLORS[tag])
        ax.set(title=label,xlabel='轮次',ylabel='推理＋渲染均值（ms）',xticks=[1,2,3]);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();savefig(fig,'LOCAL500_RATIO50_ROUNDS')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,case,label in zip(axes,['points500','ratio50'],['局部500点','局部50%']):
        for offset,tag in [(-.18,'cuda'),(.18,'deployment')]:
            values=[next(v for v in components if v['backend']==tag and v['case']==case and v['stage']==stage) for stage in ['selector','compensation_decode']]
            ax.bar(np.arange(2)+offset,[v['stream_ms_mean']*1000 for v in values],.36,color=COLORS[tag],label=NAMES[tag])
        ax.set(title=label,ylabel='CUDA Event均值（μs）',xticks=range(2),xticklabels=['Selector＋score','Compensator＋解码']);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.text(.5,.005,'中心视角固定候选及共同选中点；排除候选构建、view code、排序、finite读回与Rasterizer。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));savefig(fig,'LOCAL500_RATIO50_COMPONENTS')
    lines=['# 新部署MLP：500点及局部50%渲染速度复测\n','日期：2026-11-06。独立目录：`/home/prmi/zwj/experiments/deployment_local500_ratio50_20261106`。\n',
        '## 测试对象与结论\n',
        '使用原四Head“两网络真实视角＋LI=0.1”模型，不使用无Other消融终态。直接采用用户提供的`python/DEPLOYMENT_RUNTIME.md`对应部署源码，五个部署文件SHA与上一轮验收版本一致，两个CUDA扩展在本轮独立目录重新编译。\n']
    for case,label in [('points500','局部500点'),('ratio50','局部50%')]:
        a,b,d=s['pytorch',case],s['cuda',case],s['deployment',case]
        lines.append(f'- {label}：新后端推理 {d["inference_ms_mean"]:.4f} ms、渲染 {d["render_ms_mean"]:.4f} ms、合计 {d["total_ms_mean"]:.4f} ms，均值对应 {d["fps_from_mean"]:.1f} FPS；相对同轮PyTorch总耗时减少 {100*(1-d["total_ms_mean"]/a["total_ms_mean"]):.2f}%，相对精确CUDA减少 {100*(1-d["total_ms_mean"]/b["total_ms_mean"]):.2f}%。')
    lines += ['','以上均为离屏prepare＋Rasterizer时间，不含Viewer传输／显示，不是在线SLAM或训练速度。绝对时延受共享GPU影响；下文全部三轮保留，不挑最快轮。\n',
        '## 匹配条件\n',table(['项目','设置'],[
        ['Checkpoint',env['checkpoint']],['模型','N=13298、Anchor2790、Residual48、64/64主干、四Head、joint schema2、LI=0.1'],
        ['局部候选','Viewer名义可见Anchor的全部成员；相同slot资格、seed0固定Gumbel噪声、稳定严格K'],
        ['500点','K=min(500,E)，网络ratio=min(1,500/Nv)'],['50%','K=min(floor(.5Nv),E)，网络ratio=.5'],
        ['视角','fr1_xyz RGB-D全792帧，相同640×480标定／SLAM位姿'],['预热及轮次','64视角预热；三轮分别顺序、逆序、seed0乱序，每帧轮换后端顺序'],
        ['范围','16632次墙钟、672次CUDA Event；1584次全帧质量配对、132组SSIM/LPIPS；不保存渲染帧'],
        ['精度','FP32，TF32关闭；新MLP改变FMA、view code和Quaternion运算顺序，允许小误差，未改权重／训练'],
        ['后端','PyTorch与精确CUDA prepare均measure_time=False，保留finite检查；无CUDA Graph、无上一帧选点复用']]),
        '## 完整推理与渲染速度\n',table(['预算','后端','推理均值ms','渲染均值ms','合计均值ms','合计median','p90','p99'],[
        [case,NAMES[tag],*[f'{s[tag,case][key]:.4f}' for key in ['inference_ms_mean','render_ms_mean','total_ms_mean','total_ms_median','total_ms_p90','total_ms_p99']]]
        for case in ['points500','ratio50'] for tag in NAMES]),
        '推理从prepare调用前到返回，包含候选、视锥、view、两MLP、Residual48、严格K和finite检查；这些路径在返回前都有必要的结果完成依赖。渲染从prepare返回到Rasterizer及最后同步完成。CUDA Event辅助测量另存CUDA_EVENT_SUMMARY.csv，不混入无Event的主计时。相机构建和模型／JIT初始化不计入。\n',
        table(['预算','轮次','PyTorch总ms','精确CUDA总ms','部署总ms'],[
        [case,repeat+1,*[f'{next(v for v in summary if v["backend"]==tag and v["case"]==case and v["repeat"]==str(repeat))["total_ms_mean"]:.4f}' for tag in NAMES]]
        for case in ['points500','ratio50'] for repeat in range(3)]),
        f'当前Master未补偿Full的本轮均值为 {s["full","full"]["total_ms_mean"]:.4f} ms（13298点）。此行是当前方法地图，不能当作原CaRtGS的35301点Full速度。没有重新测原CaRtGS，不能与历史不同GPU负载的2ms／8ms／34ms直接相减。\n',
        '## 实际预算与MLP组件\n',table(['预算','候选N均值','范围','实际K均值','范围','网络ratio范围'],[
        [case,f'{s["deployment",case]["candidate_N_mean"]:.3f}',f'{s["deployment",case]["candidate_N_min"]:.0f}–{s["deployment",case]["candidate_N_max"]:.0f}',
         f'{s["deployment",case]["K_mean"]:.3f}',f'{s["deployment",case]["K_min"]:.0f}–{s["deployment",case]["K_max"]:.0f}',
         f'{s["deployment",case]["network_ratio_min"]:.5f}–{s["deployment",case]["network_ratio_max"]:.5f}'] for case in ['points500','ratio50']]),
        '所有配对计时的候选N和实际K一致。Selector处理全部候选Nv，Compensator只处理选中的K；500点预算不会令Selector仅处理500点。\n',
        table(['预算','组件','后端','候选N','K','Event均值μs','Eventmedianμs','Host均值μs'],[
        [v['case'],v['stage'],NAMES[v['backend']],v['N'],v['K'],f'{v["stream_ms_mean"]*1000:.2f}',f'{v["stream_ms_median"]*1000:.2f}',f'{v["wall_ms_mean"]*1000:.2f}'] for v in components]),
        '组件微测固定中心视角输入、共同选中ID、每条件32次预热后200对交错重复；Selector含score，Compensation含Residual读取、MLP和属性发布／解码，排除候选、view code、选择排序、finite读回及Rasterizer。CUDA Event还可能包含CPU提交空隙和GPU争用，不能当作全部为纯Kernel执行。\n',
        '## 输出及质量误差\n',table(['预算','精确CUDA PSNR','部署PSNR','平均ΔPSNR','最差单帧Δ','选点ID全同帧比例','最大像素绝对差'],[
        [case,f'{qs[case]["base_psnr_mean"]:.8f}',f'{qs[case]["deployment_psnr_mean"]:.8f}',f'{qs[case]["delta_psnr_mean"]:+.3e}',
         f'{qs[case]["delta_psnr_min"]:+.3e}',f'{qs[case]["ids_same_fraction"]*100:.2f}%',f'{qs[case]["image_max_abs_max"]:.6g}'] for case in qs]),
        table(['预算','66视角平均ΔSSIM','平均ΔLPIPS','XYZ最大差','Scale最大差','Quaternion最大差'],[
        [case,f'{qs[case]["delta_ssim_mean"]:+.3e}',f'{qs[case]["delta_lpips_mean"]:+.3e}',
         *[f'{qs[case][key+"_max"]:.6g}' for key in ['means3D_max_abs','scales_max_abs','rotations_max_abs']]] for case in qs]),
        f'预声明质量门槛：平均ΔPSNR≥−0.01dB、最差1%视角分位≥−0.05dB、单帧≥−0.1dB，结果：**{"通过" if gates["passed"] else "未通过"}**。原PyTorch与精确CUDA在smoke和66帧×两预算的packet／图像逐bit通过；新部署后端不逐bit相同。最大像素差、属性误差和radii差异完整记录，PSNR接近不代表每像素相同。\n',
        '## Kernel、内存与运行状态\n',table(['预算','后端','每次Kernel均数','Kernel累计μs','同步调用均数'],[
        [case,NAMES[tag],f'{np.mean([v["kernel_launches"] for v in profile if v["scope"]=="METHOD/"+tag+"/"+case]):.2f}',
         f'{np.mean([v["kernel_us"] for v in profile if v["scope"]=="METHOD/"+tag+"/"+case]):.2f}',
         f'{np.mean([v["synchronizations"] for v in profile if v["scope"]=="METHOD/"+tag+"/"+case]):.2f}']
        for case in ['points500','ratio50'] for tag in NAMES]),
        'Profiler仅4个代表视角，定位调用数量；主速度统计不含Profiler。新Selector每Gaussian一个Warp，Compensator每Gaussian一个Block且四Warp并行四Head。候选构建和严格K排序保留原实现，两个有前后依赖的网络没有强行并行。\n',
        f'独立部署实例常驻显存增量约 {env["deployment_persistent_increment_bytes"]/2**20:.2f} MiB，包含私有冻结模型、Master、预处理和Workspace。构造及JIT耗时 {env["deployment_load_compile_seconds"]:.2f} s，在计时外。更多峰值数据在PEAK_MEMORY.json。\n',
        '服务器存在其他GPU任务，GPU_OBSERVATIONS.json记录每轮边界的进程、利用率、频率、功耗与温度。未停止无关任务。采用同一进程交错匹配减少负载偏差，仍不能视为独占3090或端侧设备的结果。\n',
        '## 完整性与交付\n',
        '- 独立源码、独立build、原checkpoint只读。604个生产关键文件与checkpoint前后SHA一致；模型未训练，生产默认未切换，未commit/push。',
        '- 两个扩展的实际加载路径与SHA在ENVIRONMENT.json；原始逐帧时间TIMINGS.csv、Event、固定输入组件、逐帧质量及Profiler均保留。',
        '- 所有对外可读报告为中文；只生成三张分析图及PDF，不保存渲染帧。\n',
        '![总耗时](figures/LOCAL500_RATIO50_LATENCY.png)\n\n![三轮结果](figures/LOCAL500_RATIO50_ROUNDS.png)\n\n![组件时间](figures/LOCAL500_RATIO50_COMPONENTS.png)\n']
    (OUT/'DEPLOYMENT_LOCAL500_RATIO50_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
