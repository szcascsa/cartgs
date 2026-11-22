"""合并八方法完整轨迹，校验独立活动筛选并生成中文三栏图。"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'reports'
METHODS=[f'ours_{b}' for b in (100,200,300,400,500)]+['ratio_50','ratio_80','cartgs_full']
NAMES={**{f'ours_{b}':f'{b}点预算' for b in (100,200,300,400,500)},'ratio_50':'局部ratio 0.5','ratio_80':'局部ratio 0.8','cartgs_full':'CaRtGS Full'}
COLORS=dict(zip(METHODS,['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#21A6C7','#6D3A9C','#202124']))
STYLES={m:('-.' if m=='ratio_50' else ':' if m=='ratio_80' else '-') for m in METHODS}


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def stats(values,key):
    a=np.asarray([float(v[key]) for v in values]);assert np.isfinite(a).all()
    return {f'{key}_{name}':float(v) for name,v in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90)),('min',a.min()),('max',a.max())]}


def merge():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    acceptance=json.loads((OUT/'TIMING_ACCEPTANCE.json').read_text());assert acceptance['complete'] and acceptance['accepted_rows']==19008
    attempts=json.loads((OUT/'TIMING_ATTEMPTS.json').read_text());accepted=[a for a in attempts if a['accepted']]
    assert len(accepted)==3 and {a['repeat'] for a in accepted}=={0,1,2}
    for a in accepted:
        assert a['coverage'] and not any(s['external_active'] for s in a['samples'])
        assert max(b['receipt_unix']-c['receipt_unix'] for c,b in zip(a['samples'],a['samples'][1:]))<1.5
    fixed=read(ROOT/'reference/FIXED_QUALITY.csv');ratios=read(ROOT/'reference/RATIO_PER_VIEW.csv');cart=read(ROOT/'reference/CARTGS_PER_VIEW.csv')
    assert len(fixed)==3960 and len(ratios)==1584 and len(cart)==792
    q={(f'ours_{v["budget"]}',int(v['frame'])):v for v in fixed};q.update({(v['method'],int(v['frame'])):v for v in ratios})
    q.update({('cartgs_full',int(v['frame'])):v for v in cart});assert len(q)==6336
    raw=read(OUT/'TIMINGS.csv');assert len(raw)==19008
    times=defaultdict(list)
    for v in raw:
        assert v['accepted']=='True' and np.isclose(float(v['prepare_ms'])+float(v['render_ms']),float(v['total_ms']),atol=1e-9,rtol=0)
        assert int(v['attempt']) in {a['attempt'] for a in accepted}
        assert int(v['K'])==int(q[v['method'],int(v['frame'])]['K'])
        times[v['method'],int(v['frame'])].append(v)
    rows=[];summary=[];rounds=[];regions=[]
    for method in METHODS:
        values=[]
        for frame in range(792):
            v=q[method,frame];t=times[method,frame];ref=q['cartgs_full',frame]
            assert len(t)==3 and {int(x['repeat']) for x in t}=={0,1,2}
            assert abs(float(v['timestamp'])-float(ref['timestamp']))<1e-6
            x=dict(method=method,frame=frame,rgb_index=int(ref['rgb_index']),timestamp=float(v['timestamp']),map_N=35301 if method=='cartgs_full' else 13298,
                K=int(v['K']),projected_K=int(v['projected_K']),contributing_K=int(v['contributing_K']),psnr=float(v['psnr']),
                candidate_N=int(v.get('candidate_N',35301)),eligible_N=int(v.get('eligible_N',35301)),network_ratio=float(v.get('network_ratio',0)),
                shortfall=int(v.get('shortfall',0)),quality_source='previous_same_checkpoint_native_v4' if method!='cartgs_full' else 'previous_vanilla_full',timing_source='fresh_quiet_rounds')
            for key in ('prepare_ms','render_ms','total_ms'):x[key]=float(np.mean([float(a[key]) for a in t]))
            assert 0<=x['contributing_K']<=x['projected_K']<=x['K'];values.append(x);rows.append(x)
        actual=[v for v in raw if v['method']==method];s=dict(method=method,name=NAMES[method],frames=792,timing_samples=2376,shortfall_frames=sum(v['shortfall']>0 for v in values))
        for key in ('K','projected_K','contributing_K','psnr','candidate_N','network_ratio'):s.update(stats(values,key))
        for key in ('prepare_ms','render_ms','total_ms'):s.update(stats(actual,key))
        summary.append(s)
        for repeat in range(3):
            r=dict(method=method,repeat=repeat)
            for key in ('prepare_ms','render_ms','total_ms'):r.update(stats([v for v in actual if int(v['repeat'])==repeat],key))
            rounds.append(r)
            for name,pred in [('frame_lt_520',lambda f:f<520),('frame_ge_520',lambda f:f>=520)]:
                region=dict(method=method,repeat=repeat,region=name)
                for key in ('prepare_ms','render_ms','total_ms'):region.update(stats([v for v in actual if int(v['repeat'])==repeat and pred(int(v['frame']))],key))
                regions.append(region)
    full=summary[-1]['total_ms_mean']
    for s in summary:s.update(total_speedup_vs_cartgs=full/s['total_ms_mean'],total_reduction_vs_cartgs_percent=100*(1-s['total_ms_mean']/full))
    write(OUT/'PER_VIEW_COMPARISON.csv',rows);write(OUT/'SUMMARY.csv',summary);save(OUT/'SUMMARY.json',summary)
    write(OUT/'ROUND_SUMMARY.csv',rounds);save(OUT/'ROUND_SUMMARY.json',rounds);save(OUT/'TIMING_REGION_DIAGNOSTIC.json',regions)
    save(OUT/'PLOT_PROTOCOL.json',dict(frames=792,columns=3,count='actual nonzero alpha-compositing contributors',count_smoothing=False,
        psnr=dict(sigma=35,window=211,truncate=3,mode='reflect'),timing=dict(smoothing=False,reduction='per-view mean of three externally validated whole rounds; no cross-frame filtering'),
        averages='raw values; not smoothed values',cartgs='fresh standard Full timing; historical quality and alpha counts',saved_render_images=0))
    save(OUT/'ANALYSIS_VALIDATION.json',dict(passed=True,methods=8,frames=792,combined_rows=len(rows),timing_rows=len(raw),
        accepted_attempts=[a['attempt'] for a in accepted],rejected_attempts=[a['attempt'] for a in attempts if not a['accepted']],
        active_external_samples_in_accepted_rounds=0,pmon_resolution_seconds=1,latency_based_filtering=False))
    return rows,summary


def plots(rows,summary):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    folder=OUT/'figures';folder.mkdir(exist_ok=True);data={};smooth=[];s={v['method']:v for v in summary};x=np.arange(792)
    for method in METHODS:
        values=sorted([v for v in rows if v['method']==method],key=lambda v:v['frame'])
        d={key:np.asarray([v[key] for v in values]) for key in ('K','projected_K','contributing_K','psnr','prepare_ms','render_ms','total_ms')}
        d['psnr_smooth']=gaussian_filter1d(d['psnr'],35,truncate=3,mode='reflect')
        data[method]=d
        for i in range(792):smooth.append(dict(method=method,frame=i,**{k:float(v[i]) for k,v in d.items()}))
    write(OUT/'PLOT_DATA_RAW_AND_SMOOTHED.csv',smooth)
    for raw in (False,True):
        fig=plt.figure(figsize=(24,10));gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.05,1.15],hspace=.45,wspace=.24)
        ac=fig.add_subplot(gs[0,0]);az=fig.add_subplot(gs[1,0],sharex=ac);ap=fig.add_subplot(gs[:,1]);at=fig.add_subplot(gs[0,2]);ar=fig.add_subplot(gs[1,2],sharex=at)
        suffix='' if raw else '_smooth';cart=data['cartgs_full'];black=COLORS['cartgs_full']
        ac.plot(x,cart['contributing_K']/1000,color=black,lw=1.8,label=f'CaRtGS贡献｜均值{s["cartgs_full"]["contributing_K_mean"]:,.0f}点')
        ac.axhline(35.301,color='#AAAAAA',ls='--',lw=1.1,label='CaRtGS整图 N=35,301');ac.axhline(13.298,color='#777777',ls='--',lw=1.1,label='我们整图 N=13,298')
        ac.axhspan(0,.55,color='#E2E8F0',alpha=.65)
        for method in METHODS[:-1]:
            d=data[method];v=s[method];color=COLORS[method];ls=STYLES[method];label=NAMES[method]
            ac.plot(x,d['contributing_K']/1000,color=color,ls=ls,lw=1.2,
                label=f'{label}｜贡献{v["contributing_K_mean"]:,.0f}点' if method.startswith('ratio') else None)
            if method.startswith('ours'):az.plot(x,d['contributing_K'],color=color,lw=1.3,label=f'{label}｜贡献{v["contributing_K_mean"]:.1f}点')
            ap.plot(x,d['psnr'+suffix],color=color,ls=ls,lw=1.0 if raw else 2.0,label=f'{label}｜{v["psnr_mean"]:.3f} dB')
            at.plot(x,d['prepare_ms'],color=color,ls=ls,lw=1.0)
            ar.plot(x,d['render_ms'],color=color,ls=ls,lw=1.0,
                label=f'{label}｜处理{v["prepare_ms_mean"]:.3f}／渲染{v["render_ms_mean"]:.3f} ms')
        ap.plot(x,cart['psnr'+suffix],color=black,lw=2.3,label=f'CaRtGS原版｜{s["cartgs_full"]["psnr_mean"]:.3f} dB')
        ar.plot(x,cart['render_ms'],color=black,lw=1.3,label=f'CaRtGS Full（本轮）｜{s["cartgs_full"]["render_ms_mean"]:.3f} ms')
        peak_method=max(METHODS[:-1],key=lambda m:data[m]['prepare_ms'].max())
        peak_frame=int(data[peak_method]['prepare_ms'].argmax());peak=float(data[peak_method]['prepare_ms'][peak_frame])
        if peak>2*np.percentile(data[peak_method]['prepare_ms'],99):
            at.scatter([peak_frame],[peak],color=COLORS[peak_method],s=18,zorder=5,clip_on=False)
            at.annotate(f'保留尖峰：第{peak_frame}帧 {peak:.3f} ms',xy=(peak_frame,peak),xytext=(.20,.85),textcoords='axes fraction',
                fontsize=8.5,arrowprops=dict(arrowstyle='->',color=COLORS[peak_method]))
        ac.set(title='① 每个视角的实际Alpha贡献点数',ylabel='Gaussian数（千）',ylim=(0,37.5))
        az.set(title='100–500点区域放大',ylabel='Gaussian数（点）',ylim=(0,525))
        ap.set(title='② PSNR原始逐帧值' if raw else '② PSNR：强高斯平滑',ylabel='PSNR（dB）')
        at.set(title='③ 处理／准备：逐视角耗时，不平滑',ylabel='处理（ms）',ylim=(0,None))
        ar.set(title='③ 渲染：逐视角耗时，不平滑',ylabel='渲染（ms）',ylim=(0,None))
        for ax in (ac,az,ap,at,ar):ax.set_xlim(0,791);ax.set_xticks(np.arange(0,801,100));ax.grid(alpha=.22);ax.set_xlabel('共同RGB帧序号')
        for ax in (ac,at):ax.tick_params(labelbottom=False);ax.set_xlabel('')
        ac.legend(loc='upper center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=8.5,columnspacing=.8)
        az.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=9,columnspacing=.8)
        ap.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2,frameon=False,fontsize=9,columnspacing=.7)
        ar.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=1,frameon=False,fontsize=8.8,labelspacing=.4)
        fig.suptitle('fr1_xyz RGB-D｜最新native v4推理MLP｜固定点数＋局部ratio 0.5／0.8',fontsize=16,y=.98)
        fig.subplots_adjust(left=.045,right=.995,top=.90,bottom=.28)
        note=('PSNR原始值；' if raw else 'PSNR：σ=35帧（211帧窗口）；')+'耗时为每个视角三轮测量均值，不做跨帧平滑。点数不平滑，图例均值均取原始样本。'
        fig.text(.5,.055,note,ha='center',fontsize=10)
        fig.text(.5,.032,'本轮三轮计时通过独立pmon外部GPU活动检查（1秒粒度），受干扰整轮已排除；CaRtGS也在同轮重新测速。',ha='center',fontsize=9.6,color='#475569')
        fig.text(.5,.010,'ratio 0.8超过训练上限0.7，740／792帧受到slot合格点数限制；处理＋渲染才是本方法总开销。',ha='center',fontsize=9.6,color='#475569')
        name='NATIVE_FIXED_AND_RATIO_RAW' if raw else 'NATIVE_FIXED_AND_RATIO_THREE_COLUMNS'
        for ext in (['png'] if raw else ['png','pdf','svg']):fig.savefig(folder/(name+'.'+ext),dpi=190,facecolor='white')
        plt.close(fig)


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def report(summary):
    s={v['method']:v for v in summary};env=json.loads((OUT/'ENVIRONMENT.json').read_text());validation=json.loads((OUT/'ANALYSIS_VALIDATION.json').read_text())
    lines=['# fr1_xyz局部预算：空闲GPU重新测速及ratio补充\n',
        '日期：2026-11-07。服务器独立目录：`/home/prmi/zwj/experiments/native_budget_ratio_idle_20261107`。\n',
        '## 结果\n',
        '八种方法均覆盖792个共同RGB帧。下表时间取通过GPU活动检查的三轮全部原始样本均值，PSNR取792帧原始均值。新计时替换此前受并发任务影响的速度曲线；质量和点数仍使用相同模型已完成的结果。\n',
        table(['方法','输入K均值','Alpha贡献点数均值','PSNR dB','处理ms','渲染ms','合计ms','相对原版总耗时倍速'],[
            [v['name'],f'{v["K_mean"]:.2f}',f'{v["contributing_K_mean"]:.2f}',f'{v["psnr_mean"]:.5f}',f'{v["prepare_ms_mean"]:.4f}',
             f'{v["render_ms_mean"]:.4f}',f'{v["total_ms_mean"]:.4f}',f'{v["total_speedup_vs_cartgs"]:.3f}×'] for v in summary]),
        'ratio 0.5／0.8不是固定点数：按当前可见名义Anchor的成员数Nv计算floor(r×Nv)，再受原slot资格数E限制。0.8在740／792帧达不到请求点数；实际K均值6366.58，真实Alpha贡献点数均值5492.76。它超过训练最大ratio 0.7，属于上界外推，不能作为训练0.8的结果。两ratio平均PSNR仅差约0.00354dB，不代表0.8有明确质量收益。\n',
        '此前约520帧后的巨大延迟抬升没有在本轮复现。500点的前520帧／后272帧总延迟均值约0.6962／0.6996ms；局部0.5约0.8105／0.7967ms。支持此前抬升主要受测量时段干扰影响，不支持把该现象解释为后段视角本身需要数倍渲染计算。\n',
        '## 使用的模型和实现\n',
        f'模型：`{env["checkpoint"]}`。四个Head、两网络真实视角＋LI=0.1，Master13298点、2790 Anchor。使用用户指定NATIVE_SELECTION_OPTIMIZATION_REPORT对应的native v4：C++整条prepare、精确Radix阈值／稳定压缩、合并finite状态，不使用旧Python准备路径替代。',
        f'实际扩展：`{env["binary"]}`，SHA256 `{env["binary_sha256"]}`。复用已验证二进制的独立副本，修改的只是研究脚本中的扩展加载位置，没有修改MLP、候选、严格K或补偿数学。推理专用MLP沿用已有FP32 FMA实现，本轮不宣称它与训练cuBLAS路径逐bit等同。',
        f'原版CaRtGS：`{env["vanilla"]["checkpoint"]}`，35301点。新计时调用原始renderer.py和原始Rasterizer扩展，标准Full每次执行属性激活；没有以静态属性缓存对照替换原版。扩展SHA256 `{env["original_extension_sha256"]}`。\n',
        '## 无外部GPU活动的计时验收\n',
        '- 独立nvidia-smi pmon以1秒粒度记录各PID的SM、显存带宽及编解码活动。连续两个完整周期未检测到外部活动才开始整轮；整轮前2秒、测量区间和结束后3秒存在非零外部活动则整轮拒绝。',
        f'- 通过整轮编号：{validation["accepted_attempts"]}；拒绝编号：{validation["rejected_attempts"]}。拒绝原因包含外部GPU活动和采样覆盖不足；全部保留在ALL_TIMINGS.csv及TIMING_ATTEMPTS.json，不按延迟快慢筛选。',
        '- 通过三轮：顺序、逆序、seed0乱序。逐帧轮转八方法顺序，64视角×八方法预热。每方法2376条，总19008条。没有复用旧第三轮，也没有截取最快时段或丢弃慢帧。',
        '- 本次“空闲”表示pmon采样未检测到外部GPU活动；它不是GPU独占锁，不能排除低于采样粒度的短活动或所有CPU调度噪声。PMON_RAW.log、PMON_SAMPLES.json及GPU_OBSERVATIONS.json保留可核查证据。',
        '- RTX3090，FP32，TF32关闭，两CPU线程，640×480。处理从prepare前至返回，包含Selector、严格K、Residual48／Compensator、打包及finite依赖；渲染从正常render调用至cuda同步。加载、指标、GT、相机、传输／显示在计时外。',
        '- CaRtGS的标准Full属性准备属于其render调用内部，独立处理列为0只表示没有Selector／Compensator独立阶段。判断总速度应比较本方法处理＋渲染与原版标准Full。\n',
        table(['方法','处理median ms','渲染median ms','合计median ms','合计p90 ms'],[
            [v['name'],f'{v["prepare_ms_median"]:.4f}',f'{v["render_ms_median"]:.4f}',f'{v["total_ms_median"]:.4f}',f'{v["total_ms_p90"]:.4f}'] for v in summary]),
        '## 质量、数量和平滑口径\n',
        '- 五档固定预算复用上一轮native v4同checkpoint、同二进制的3960条完整质量／计数结果；0.5／0.8使用新增1584条结果；原版复用792帧真实Alpha计数和PSNR。首／中／末帧×八方法共24个PSNR核对通过，阈值1e-5dB。',
        '- 新两ratio和五固定预算的计数Renderer在三个代表帧、共21组与正常Renderer颜色和radii逐bit一致；计数Renderer仅计算质量／点数，不参与测速。真实Alpha点数以合成权重质量>0定义，不把输入K或radii>0当作实际贡献点数。',
        '- 三栏图：左栏真实贡献点数及100–500点放大；中栏PSNR；右栏上下分别处理和渲染。PSNR使用σ=35、truncate=3、reflect，窗口211帧；耗时为每个视角三轮测量均值，不做跨帧平滑，不过滤尖峰。原始点数不平滑。图例和表格均值均取原始数据。',
        '- 所有方法按同一RGB帧对齐；两模型使用各自保存SLAM位姿和地图，不能将质量差全部归因于选择／补偿。0.5和0.8使用同一模型同一相机。训练、Master和网络权重本轮均未更改。',
        '- TIMING_REGION_DIAGNOSTIC.json按每轮、每方法分开记录520帧前后延迟，避免把测量时间段的干扰当成视角固有复杂度。\n',
        '## 完整性与交付\n',
        '- 604个生产关键文件、两个checkpoint、相机轨迹、研究源码／二进制、历史输入均进行SHA前后核验；生产默认不变，不修改共享conda，没有commit／push，没有保存渲染帧图片。',
        '- TIMINGS.csv：19008条验收通过样本；ALL_TIMINGS.csv：全部尝试；PER_VIEW_COMPARISON.csv：6336条八方法对齐结果；PLOT_DATA_RAW_AND_SMOOTHED.csv：原始及平滑曲线。',
        '- 主图PNG／PDF／SVG以及原始PSNR图在figures；配置、源码身份、环境、活动记录和验证结果一并归档。\n',
        '![三栏图](figures/NATIVE_FIXED_AND_RATIO_THREE_COLUMNS.png)\n']
    (OUT/'NATIVE_FIXED_AND_RATIO_IDLE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    rows,summary=merge();plots(rows,summary);report(summary)
    print('ANALYSIS_PLOTS_PASS',len(summary),'methods')
