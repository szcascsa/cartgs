"""论文引言图：点数、ratio、PSNR、总耗时，中央下格为统一图例。"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'reports'
METHODS=['cartgs_full','ours_full','ratio_50','ratio_80','fixed_2000','fixed_4000']
COLORS=dict(zip(METHODS,['#202124','#747B84','#17A6C4','#7041A4','#0072B2','#D55E00']))
STYLES=dict(zip(METHODS,['-','--','-.',':','-','-']))
CN={'cartgs_full':'CaRtGS原版 · Full','ours_full':'本方法 · Full（不使用LoD）','ratio_50':'本方法 · 局部ratio 0.5',
    'ratio_80':'本方法 · 局部ratio 0.8','fixed_2000':'本方法 · 目标预算2,000点','fixed_4000':'本方法 · 目标预算4,000点'}
EN={'cartgs_full':'Original CaRtGS (Full)','ours_full':'Ours (Full, no LoD)','ratio_50':r'Ours (local $r=0.5$)',
    'ratio_80':r'Ours (local $r=0.8$)','fixed_2000':r'Ours (target $K=2{,}000$)','fixed_4000':r'Ours (target $K=4{,}000$)'}


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)


def summary_stats(values,key):
    a=np.asarray([float(v[key]) for v in values]);assert np.isfinite(a).all()
    return {key+'_'+name:float(value) for name,value in [('mean',a.mean()),('std',a.std()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90)),('min',a.min()),('max',a.max())]}


def merge():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    acceptance=json.loads((OUT/'TIMING_ACCEPTANCE.json').read_text());assert acceptance['complete'] and acceptance['accepted_rows']==14256
    attempts=json.loads((OUT/'TIMING_ATTEMPTS.json').read_text());valid=[v for v in attempts if v['accepted']]
    assert len(valid)==3 and {v['repeat'] for v in valid}=={0,1,2}
    for v in valid:assert v['coverage'] and not any(s['external_active'] for s in v['samples'])
    quality=read(ROOT/'evaluation/OURS_PER_VIEW.csv');timing=read(OUT/'TIMINGS.csv');previous=read(ROOT/'reference/PREVIOUS_PER_VIEW.csv')
    assert len(quality)==3960 and len(timing)==14256
    q={(v['method'],int(v['frame'])):v for v in quality};old={(v['method'],int(v['frame'])):v for v in previous}
    groups=defaultdict(list)
    for v in timing:
        assert v['accepted']=='True';assert abs(float(v['prepare_ms'])+float(v['render_ms'])-float(v['total_ms']))<1e-9
        groups[v['method'],int(v['frame'])].append(v)
    rows=[];summary=[]
    for method in METHODS:
        values=[]
        for frame in range(792):
            v=old['cartgs_full',frame] if method=='cartgs_full' else q[method,frame]
            ref=old['cartgs_full',frame];assert abs(float(v['timestamp'])-float(ref['timestamp']))<1e-6
            if method=='cartgs_full':total=float(v['total_ms'])
            else:
                t=groups[method,frame];assert len(t)==3 and {int(x['repeat']) for x in t}=={0,1,2}
                assert all(int(x['K'])==int(v['K']) for x in t);total=float(np.mean([float(x['total_ms']) for x in t]))
            full=method.endswith('full')
            x=dict(method=method,frame=frame,timestamp=float(v['timestamp']),map_N=35301 if method=='cartgs_full' else 13298,
                K=int(v['K']),requested_K=int(v.get('requested_K',v['K'])),projected_K=int(v['projected_K']),contributing_K=int(v['contributing_K']),
                candidate_N=int(v['candidate_N']),eligible_N=int(v['eligible_N']),network_ratio=None if full else float(v['network_ratio']),
                actual_retention_ratio=None if full else int(v['K'])/int(v['candidate_N']),shortfall=int(v['shortfall']),psnr=float(v['psnr']),total_ms=total,
                timing_source='unchanged_previous_quiet_run' if method=='cartgs_full' else 'fresh_three_quiet_rounds')
            assert 0<=x['contributing_K']<=x['projected_K']<=x['K'];values.append(x)
        smoothed=gaussian_filter1d(np.asarray([v['psnr'] for v in values]),35,truncate=3,mode='reflect')
        for x,psnr in zip(values,smoothed):x['psnr_smooth']=float(psnr)
        rows.extend(values);item=dict(method=method,name=CN[method],frames=792,shortfall_frames=sum(v['shortfall']>0 for v in values))
        for key in ('K','contributing_K','psnr','total_ms'):item.update(summary_stats(values,key))
        if not method.endswith('full'):item.update(summary_stats(values,'network_ratio'))
        item['per_view_latency_cv_percent']=100*item['total_ms_std']/item['total_ms_mean'];item['contribution_count_cv_percent']=100*item['contributing_K_std']/item['contributing_K_mean']
        summary.append(item)
    assert len(rows)==4752
    oldplot={(v['method'],int(v['frame'])):v for v in read(ROOT/'reference/PREVIOUS_PLOT_DATA.csv')}
    for v in rows:
        if v['method']=='cartgs_full':
            ref=old['cartgs_full',v['frame']];assert all(v[k]==float(ref[k]) for k in ('K','contributing_K','psnr','total_ms'))
            assert v['psnr_smooth']==float(oldplot['cartgs_full',v['frame']]['psnr_smooth'])
        elif v['method'].startswith('ratio_'):
            ref=old[v['method'],v['frame']];assert int(ref['K'])==v['K'] and int(ref['contributing_K'])==v['contributing_K']
            assert abs(float(ref['psnr'])-v['psnr'])<1e-5
        elif v['method'].startswith('fixed_'):
            target=int(v['method'].split('_')[1]);assert v['requested_K']==target and v['K']==min(target,v['eligible_N'])
            assert v['network_ratio']==min(1.,target/v['candidate_N'])
    control=[v for v in timing if v['method']=='cartgs_control'];assert len(control)==2376
    baseline=next(v for v in summary if v['method']=='cartgs_full')['total_ms_mean'];fresh=float(np.mean([float(v['total_ms']) for v in control]));drift=100*(fresh/baseline-1)
    save(OUT/'BASELINE_DRIFT_CHECK.json',dict(plotted_cartgs_mean_ms=baseline,new_control_mean_ms=fresh,drift_percent=drift,threshold_percent=5,within_threshold=abs(drift)<=5,plotted_cartgs_replaced=False))
    write(OUT/'PAPER_INTRO_PER_VIEW.csv',rows);write(OUT/'PAPER_INTRO_SUMMARY.csv',summary);save(OUT/'PAPER_INTRO_SUMMARY.json',summary)
    save(OUT/'PLOT_PROTOCOL.json',dict(grid=[2,3],legend_cell=[2,2],panels={'counts':[1,1],'ratio':[2,1],'psnr':[1,2],'latency':'right column spans both rows'},
        methods=METHODS,count='actual Alpha-compositing contributors after rendering',ratio='actual network input condition; not K/Nv after eligibility cap',full_ratio=None,
        psnr=dict(sigma=35,truncate=3,mode='reflect',window=211),timing=dict(smoothing=False,reduction='per-view arithmetic mean of three whole accepted repeats',scope='prepare + render + final synchronization'),
        cartgs_data_unchanged=True,frame_count=792,render_images_saved=0))
    save(OUT/'ANALYSIS_VALIDATION.json',dict(passed=True,methods=6,frames=792,combined_rows=4752,fresh_quality_rows=3960,accepted_timing_rows=14256,
        cartgs_data_unchanged=True,psnr_smoothing_unchanged=True,timing_smoothing=False,baseline_control_within_5_percent=abs(drift)<=5))
    return rows,summary


def draw(rows,language,latency_summary=None):
    cn=language=='CN';names=CN if cn else EN
    if cn:
        font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    else:font='DejaVu Sans'
    scale=1. if cn else .64
    plt.rcParams.update({'font.family':font,'font.size':10.8*scale,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'path','axes.linewidth':.7*scale,'xtick.major.width':.6*scale,'ytick.major.width':.6*scale})
    data={m:sorted([v for v in rows if v['method']==m],key=lambda v:v['frame']) for m in METHODS};x=np.arange(792)
    fig=plt.figure(figsize=(18.2,8.1) if cn else (7.3,4.25));gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.05,1.0],wspace=.32,hspace=.38 if cn else .55)
    ac=fig.add_subplot(gs[0,0]);ar=fig.add_subplot(gs[1,0],sharex=ac);ap=fig.add_subplot(gs[0,1],sharex=ac);legend=fig.add_subplot(gs[1,1]);at=fig.add_subplot(gs[:,2],sharex=ac)
    handles=[]
    for m in METHODS:
        v=data[m];color=COLORS[m];ls=STYLES[m]
        ac.plot(x,[a['contributing_K']/1000 for a in v],color=color,ls=ls,lw=(1.4 if m=='cartgs_full' else 1.15)*scale)
        ap.plot(x,[a['psnr_smooth'] for a in v],color=color,ls=ls,lw=(1.9 if m=='cartgs_full' else 1.8)*scale)
        line=at.plot(x,[a['total_ms'] for a in v],color=color,ls=ls,lw=(1.15 if m=='cartgs_full' else .9)*scale)[0]
        assert np.array_equal(line.get_ydata(),np.asarray([a['total_ms'] for a in v]))
        if not m.endswith('full'):ar.plot(x,[a['network_ratio'] for a in v],color=color,ls=ls,lw=1.5*scale)
        handles.append(Line2D([],[],color=color,ls=ls,lw=1.9*scale,label=names[m]))
    ac.axhline(35.301,color='#AAAAAA',ls='--',lw=1.*scale);ac.axhline(13.298,color='#888888',ls='--',lw=1.*scale)
    ac.text(785,35.301+.65,'CaRtGS N=35,301',ha='right',va='bottom',fontsize=8.5*scale,color='#707070')
    ac.text(785,13.298+.65,'Ours N=13,298',ha='right',va='bottom',fontsize=8.5*scale,color='#707070')
    ac.set(title='(a) 实际贡献Gaussian点数' if cn else '(a) Contributing Gaussians',ylabel='Gaussian数（千）' if cn else 'Gaussians (k)',ylim=(0,38))
    ar.set(title='(d) 固定预算所需的自适应ratio' if cn else '(d) Adaptive ratio for point budgets',ylabel='网络ratio条件' if cn else 'Network ratio condition',ylim=(0,1.))
    ap.set(title='(b) 渲染质量' if cn else '(b) Rendering quality',ylabel='PSNR (dB)')
    at.set(title='(c) 处理＋渲染总耗时' if cn else '(c) Total per-view latency',ylabel='总耗时（ms）' if cn else 'Total latency (ms)',ylim=(0,None))
    if latency_summary is not None:
        by_method={item['method']:item for item in latency_summary}
        short_names=({'cartgs_full':'CaRtGS原版','ratio_50':'局部ratio 0.5','fixed_2000':'固定2,000点','fixed_4000':'固定4,000点*'} if cn else
            {'cartgs_full':'CaRtGS','ratio_50':r'Local $r=0.5$','fixed_2000':r'Fixed $K=2{,}000$','fixed_4000':r'Fixed $K=4{,}000$*'})
        cells=[[short_names[method],f'{by_method[method]["total_ms_std"]:.5f}',f'{by_method[method]["per_view_latency_cv_percent"]:.4f}%'] for method in METHODS]
        statistics=at.table(cellText=cells,colLabels=['方法' if cn else 'Method',r'$\sigma$ (ms)','CV'],
            colWidths=[.46,.28,.26],cellLoc='center',bbox=[.035,.54,.93,.23])
        statistics.auto_set_font_size(False)
        statistics.set_fontsize(10.*scale)
        for (row,column),cell in statistics.get_celld().items():
            cell.set_edgecolor('#D5DAE0')
            cell.set_linewidth(.55*scale)
            cell.set_facecolor('#F1F4F8' if row==0 else 'white')
            if row==0:cell.get_text().set_fontweight('bold')
            elif column==0:cell.get_text().set_color(COLORS[METHODS[row-1]])
    for ax in (ac,ar,ap,at):
        ax.set_xlim(0,791);ax.set_xticks(np.arange(0,801,100) if cn else [0,250,500,750]);ax.set_xlabel('RGB帧序号' if cn else 'RGB frame index');ax.grid(alpha=.20,lw=.55*scale)
    legend.set_axis_off()
    legend.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.),frameon=False,fontsize=10.3*scale,handlelength=2.7,labelspacing=1.0,borderaxespad=0)
    note=('Full无网络ratio条件。\n固定点数为目标预算，资格不足按实际K。\nPSNR强平滑；耗时逐视角均值、不平滑。' if cn else
        'Full bypasses the ratio condition.\nFixed K denotes the target point budget.\nPSNR is smoothed; latency is unsmoothed.')
    if latency_summary is not None:
        note=('固定预算ratio按B/Nv自适应调整。\nPSNR强平滑；总耗时不平滑。\n*4,000点第0帧剔除一次异常测量。\n仍保留全部792个视角。' if cn else
            'Fixed-budget ratio adapts as B/Nv.\nPSNR smoothed; latency unsmoothed.\n*One timing repeat removed at frame 0.\nAll 792 views retained.')
    if cn or latency_summary is not None:legend.text(0,.015,note,transform=legend.transAxes,fontsize=9.3*scale,va='bottom',linespacing=1.55,color='#4B5563')
    fig.subplots_adjust(left=.057 if cn else .068,right=.985,top=.93,bottom=.12 if cn else .14)
    foot=('同一fr1_xyz RGB-D的792帧；点数为渲染后实际Alpha贡献点。总耗时包含LoD推理；Full关闭选择与补偿。' if cn else
        'fr1_xyz RGB-D, 792 views. Counts are post-render Alpha contributors. LoD latency includes selection and compensation.')
    if latency_summary is not None:
        foot=('fr1_xyz RGB-D，共792帧。σ与CV基于逐视角总耗时，包含选择、补偿、打包及渲染；σ采用ddof=0。' if cn else
            'fr1_xyz RGB-D, 792 views. Statistics use per-view total latency, including LoD inference.')
    if cn:fig.text(.5,.025,foot,ha='center',fontsize=9.4*scale,color='#374151')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'PAPER_INTRO_{language}.{ext}',dpi=250 if cn else 600,facecolor='white')
    plt.close(fig)


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def report(summary):
    s={v['method']:v for v in summary};drift=json.loads((OUT/'BASELINE_DRIFT_CHECK.json').read_text(encoding='utf-8'))
    lines=['# fr1_xyz引言图：局部ratio、固定预算及Full\n',
        '日期：2026-11-07。独立目录：`/home/prmi/zwj/experiments/paper_intro_budget_20261107`。\n',
        '## 实际结果\n',
        table(['方法','输入K均值','Alpha贡献点数均值','PSNR原始均值dB','总耗时ms','逐视角耗时CV'],[
            [CN[v['method']],f'{v["K_mean"]:.2f}',f'{v["contributing_K_mean"]:.2f}',f'{v["psnr_mean"]:.5f}',f'{v["total_ms_mean"]:.4f}',f'{v["per_view_latency_cv_percent"]:.2f}%'] for v in summary]),
        f'当前小场景中，本方法Full总耗时{s["ours_full"]["total_ms_mean"]:.4f}ms，快于四种LoD。这是关闭Selector／Compensator后的实测结果，图中完整保留。可以展示相对原版的紧凑地图和点数／质量／延迟权衡，不能声称该场景中LoD比本方法Full更快。',
        '固定2000／4000预算的实际贡献点数CV分别为1.25%／4.54%，低于原版13.84%；逐视角总时间标准差约0.0243／0.0217ms，低于原版0.0376ms。但相对时间CV为3.30%／2.70%，高于原版2.18%。因此可描述更稳定的点数规模、更低绝对耗时及更小绝对抖动，不应泛称所有LoD策略的相对延迟都更稳定。',
        '固定2000点全部达到预算；固定4000点在11／792帧受到原资格规则限制，实际K均值3993.04；局部0.8在740／792帧不足请求预算，实际K均值6366.58。0.8超过训练最大ratio 0.7。图例使用“目标预算”，不将实际贡献点数当成输入K。\n',
        '## 2×3版式\n',
        '- (1,1)：渲染后真实Alpha贡献点数；保留原版曲线和两条地图N参考线。',
        '- (2,1)：网络实际输入ratio条件。固定B使用min(1,B/Nv)，局部0.5／0.8为常数。Full不经过LoD网络，无ratio条件，不虚构为r=1。输入ratio与资格截断后的K/Nv是不同量，CSV均保存。',
        '- (1,2)：PSNR对帧序号；sigma=35、truncate=3、reflect，211帧强平滑。图例／表格均值用原始值。',
        '- (2,2)：六方法统一图例，颜色与线型跨面板一致。',
        '- 右列合并上下格：(c)总墙钟延迟。包含准备、Selector／Compensator、输入打包、光栅化与最终同步；各视角三轮均值，不进行跨帧平滑或剔除尖峰。\n',
        '## Full与比较口径\n',
        '本方法Full为当前同一个23560_shutdown checkpoint的13298点Master。调用部署runtime.full()，每次输出复制／打包在计时内，接同一正常render_prepared；不调用选择、补偿或Anchor视锥筛选。冻结Master属性在模型加载时预处理，不缓存相机、可见性或图像。这是部署Full路径，不是重新训练的无LoD消融。',
        'LoD为最新native v4，四Head、两网络真实视角＋LI=0.1。可见名义Anchor成员构成Nv，原slot资格数E，K受目标与E共同限制。贡献点在渲染之后统计，未用于提前筛候选。',
        '原版CaRtGS点数、PSNR及总时间逐值复用native_budget_ratio_idle_20261107，主图未替换其任何数据。新测原版仅用于环境漂移核查。',
        f'原版图中均值{drift["plotted_cartgs_mean_ms"]:.6f}ms，新控制均值{drift["new_control_mean_ms"]:.6f}ms，变化{drift["drift_percent"]:+.3f}%，预声明5%阈值核查结果：{drift["within_threshold"]}。原版使用标准renderer及原始Rasterizer，本方法使用部署Renderer；这是完整实现路径比较，不能将全部速度差归于LoD算法。',
        '两地图分别使用各自保存SLAM位姿，比较共同792个RGB帧和640×480尺寸。质量差含地图、位姿、训练与补偿的共同影响。\n',
        '## 验证与限制\n',
        '- 五种本方法新评价3960行。带计数Renderer只用于点数和PSNR；三代表帧×五方法15组颜色／radii与正常Renderer逐bit一致。原版三帧质量核对通过。',
        '- 新计时六条件（含不绘图的原版控制）×792帧×三轮=14256次，64视角预热，帧顺序／逆序／seed0乱序，方法顺序逐帧轮转。',
        '- 独立pmon按1秒粒度验收整轮，无延迟数值筛选。当前“无外部GPU活动”不代表GPU独占锁或完全无CPU调度噪声。原始记录保留。',
        '- PAPER_INTRO_PER_VIEW.csv保存每个曲线点的原始数量、原始／平滑PSNR、未平滑总时间、网络ratio、实际K/Nv和shortfall；BASELINE_DRIFT_CHECK.json保存独立控制。',
        '- 当前场景可用于展示固定预算的点数稳定性及ratio随候选数变化，但有限帧结果不能外推为所有数据集、端侧GPU或端到端SLAM加速。',
        '- 604个生产关键文件、checkpoint、原始曲线、源码和二进制前后SHA一致；没有训练、渲染帧图片、生产更新、commit／push。\n',
        '## 图注建议\n',
        '在fr1_xyz RGB-D的792个视角上比较原版CaRtGS、本方法Full及四种LoD预算策略。(a)实际参与Alpha合成的Gaussian数量，虚线为地图总点数；(d)网络输入ratio条件，固定点数策略按可见名义Anchor候选数自适应调整；(b)强平滑PSNR；(c)包含选择、补偿和光栅化的逐视角总延迟，耗时不平滑。固定点数为目标预算，候选资格不足时采用实际K。完整Master路径不经过LoD网络。\n',
        '![引言图](figures/PAPER_INTRO_CN.png)\n']
    (OUT/'PAPER_INTRO_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    rows,summary=merge();draw(rows,'CN');draw(rows,'EN');report(summary)
    print('PAPER_INTRO_PLOTS_PASS',len(rows),'rows')
