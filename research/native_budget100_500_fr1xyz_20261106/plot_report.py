"""三栏图：真贡献点数、强平滑PSNR、分开的处理与渲染延迟。"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'reports';BUDGETS=[100,200,300,400,500]
COLORS=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']


def main():
    with (OUT/'PER_VIEW_COMPARISON.csv').open() as f:rows=list(csv.DictReader(f))
    summary=json.loads((OUT/'SUMMARY.json').read_text());s={v['method']:v for v in summary}
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    methods=['ours_'+str(b) for b in BUDGETS]+['cartgs_full'];data={}
    smooth=[]
    for method in methods:
        values=sorted([v for v in rows if v['method']==method],key=lambda v:int(v['frame']));assert len(values)==792
        d={k:np.asarray([float(v[k]) for v in values]) for k in ['psnr','contributing_K','projected_K','K','prepare_ms','render_ms','total_ms']}
        for k in ['psnr','prepare_ms','render_ms','total_ms']:d[k+'_smooth']=gaussian_filter1d(d[k],35 if k=='psnr' else 12,truncate=3,mode='reflect')
        data[method]=d
        for i in range(792):smooth.append(dict(method=method,frame=i,**{k:float(v[i]) for k,v in d.items()}))
    with (OUT/'PLOT_DATA_RAW_AND_SMOOTHED.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(smooth[0]));w.writeheader();w.writerows(smooth)
    x=np.arange(792)
    for raw in [False,True]:
        fig=plt.figure(figsize=(23,9));gs=fig.add_gridspec(2,3,width_ratios=[1,1.05,1.05],height_ratios=[1,1],hspace=.40,wspace=.24)
        ac=fig.add_subplot(gs[0,0]);az=fig.add_subplot(gs[1,0],sharex=ac);ap=fig.add_subplot(gs[:,1]);at=fig.add_subplot(gs[0,2]);ar=fig.add_subplot(gs[1,2],sharex=at)
        black='#202124';suffix='' if raw else '_smooth';cart=data['cartgs_full']
        ac.plot(x,cart['contributing_K']/1000,color=black,lw=1.8,label=f'CaRtGS实际贡献｜均值{s["cartgs_full"]["contributing_K_mean"]:,.0f}点')
        ac.axhline(35.301,color='#A0A0A0',ls='--',lw=1.3,label='CaRtGS整图 N=35,301')
        ac.axhline(13.298,color='#777777',ls='--',lw=1.3,label='我们整图 N=13,298')
        ac.axhspan(0,.55,color='#E2E8F0',alpha=.6)
        ac.set(title='① 每个视角的实际Alpha贡献点数',ylabel='Gaussian数（千）',ylim=(0,37.5))
        ac.legend(loc='lower left',bbox_to_anchor=(0,.08),frameon=False,fontsize=9)
        ap.plot(x,cart['psnr'+suffix],color=black,lw=2.3,label=f'CaRtGS原版｜{s["cartgs_full"]["psnr_mean"]:.3f} dB')
        ar.plot(x,cart['render_ms'+suffix],color=black,lw=1.8,ls='--',label=f'CaRtGS Full（历史）｜{s["cartgs_full"]["render_ms_mean"]:.3f} ms')
        for b,color in zip(BUDGETS,COLORS):
            key='ours_'+str(b);d=data[key];v=s[key]
            ac.plot(x,d['contributing_K']/1000,color=color,lw=.9)
            az.plot(x,d['contributing_K'],color=color,lw=1.2,label=f'{b}点预算｜贡献均值{v["contributing_K_mean"]:.1f}点')
            ap.plot(x,d['psnr'+suffix],color=color,lw=1.0 if raw else 2.1,label=f'{b}点预算｜{v["psnr_mean"]:.3f} dB')
            at.plot(x,d['prepare_ms'+suffix],color=color,lw=1.0 if raw else 1.8)
            ar.plot(x,d['render_ms'+suffix],color=color,lw=1.0 if raw else 1.8,
                label=f'{b}点｜处理{v["prepare_ms_mean"]:.3f}／渲染{v["render_ms_mean"]:.3f} ms')
        az.set(title='低点数区域放大：实际贡献点数',ylabel='Gaussian数（点）',ylim=(0,525))
        ap.set(title='② PSNR原始逐帧值' if raw else '② PSNR：强高斯平滑',ylabel='PSNR（dB）')
        at.set(title='③ 处理／准备：选点＋补偿等',ylabel='处理（ms）')
        ar.set(title='③ 渲染调用：CaRtGS为原版Full入口',ylabel='渲染（ms）')
        for a in [ac,az,ap,at,ar]:a.set_xlim(0,791);a.grid(alpha=.22);a.set_xlabel('共同RGB帧序号');a.set_xticks(np.arange(0,801,100))
        for a in [ac,at]:a.tick_params(labelbottom=False);a.set_xlabel('')
        az.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False,fontsize=9,columnspacing=.8)
        handles,labels=ap.get_legend_handles_labels();ap.legend(handles[1:]+handles[:1],labels[1:]+labels[:1],loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False,fontsize=9,columnspacing=.8)
        handles,labels=ar.get_legend_handles_labels();ar.legend(handles[1:]+handles[:1],labels[1:]+labels[:1],loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1,frameon=False,fontsize=8.7,labelspacing=.35)
        fig.suptitle('fr1_xyz RGB-D｜最新native v4推理专用MLP｜两网络真实视角＋LI=0.1',fontsize=15,y=.98)
        fig.subplots_adjust(left=.045,right=.995,top=.90,bottom=.24)
        note='原始PSNR；点数不平滑。' if raw else 'PSNR：σ=35帧（211帧窗口）；耗时：三次逐帧均值后σ=12帧；点数不平滑。'
        fig.text(.5,.035,note+'图例均值均由原始数据计算。',ha='center',fontsize=10)
        fig.text(.5,.012,'CaRtGS复用历史结果，耗时并非同轮GPU负载；处理含选点、补偿与finite检查，不含加载、指标、传输显示。',ha='center',fontsize=9.5,color='#475569')
        name='NATIVE_FIXED100_500_RAW' if raw else 'NATIVE_FIXED100_500_THREE_COLUMNS'
        for ext in (['png'] if raw else ['png','pdf','svg']):fig.savefig(figdir/(name+'.'+ext),dpi=190,facecolor='white')
        plt.close(fig)
    report(summary)


def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,v))+' |' for v in values])+'\n'


def report(summary):
    s={v['method']:v for v in summary};env=json.loads((OUT/'ENVIRONMENT.json').read_text())
    lines=['# 最新推理MLP：fr1_xyz每视角100–500点三栏图\n',
        '日期2026-11-06，服务器目录`/home/prmi/zwj/experiments/native_budget100_500_fr1xyz_20261106`。\n',
        '## 图和数值\n',
        '使用NATIVE_SELECTION_OPTIMIZATION_REPORT对应的native v4源码：C++整条准备、精确Radix Kth阈值／稳定压缩、合并finite状态。模型为四Head、两网络真实视角＋LI=0.1的13298点地图，2790 Anchor。原版CaRtGS为35301点RGB-D repeat1地图。\n',
        table(['方法','输入K','实际贡献点数均值','PSNR dB','处理均值ms','渲染均值ms','总均值ms'],[
        [('CaRtGS Full（历史）' if v['method']=='cartgs_full' else f'本方法预算{v["budget"]}'),f'{v["K_mean"]:.0f}',f'{v["contributing_K_mean"]:.2f}',
         f'{v["psnr_mean"]:.5f}',f'{v["prepare_ms_mean"]:.4f}',f'{v["render_ms_mean"]:.4f}',f'{v["total_ms_mean"]:.4f}'] for v in summary]),
        '## 三栏布局与平滑\n',
        '- 左栏上图展示原版实际Alpha贡献点数和两地图总N；下图放大本方法100–500点，颜色固定。实线贡献点数可能小于预算K；没有把输入点数或radii>0投影数冒充Alpha贡献数。',
        '- 中栏显示每个共同RGB帧的PSNR，σ=35帧、truncate=3、reflect边界，等效211帧强高斯平滑。图例PSNR均值仍取原始792帧，不以平滑值重新评价。',
        '- 右栏上下分别是处理／准备和Rasterizer，先对同帧三次测量取均值，再σ=12帧、truncate=3、reflect平滑（73帧窗口）。两类开销分开标注，同预算在各栏使用同颜色。',
        '- 原始图NATIVE_FIXED100_500_RAW.png、原始与平滑数值PLOT_DATA_RAW_AND_SMOOTHED.csv一并保留。曲线接近时不人为错位、不根据结果挑平滑窗口或剔除时间异常。\n',
        '## 预算、计算和评价\n',
        '预算固定100/200/300/400/500点；Viewer名义可见Anchor的全部Gaussian成员构成候选Nv，K=min(B,E)，E遵循原slot资格；网络ratio=min(1,B/Nv)。Selector处理Nv，Compensator只处理K。',
        '处理均值为外层墙钟，从prepare调用前到结果返回，包含请求检查、C++准备、候选／view、Selector、严格K、Residual48／Compensator、属性打包及finite等待。渲染为正常Rasterizer调用至最后同步；不使用带计数版Renderer测速度。加载、JIT、相机、GT、指标、Viewer显示在计时外。',
        '64个视角预热；792帧×三轮×五预算=11880次新测速。帧序顺序／逆序／seed0乱序，逐帧轮转预算顺序；保留全部样本。质量与Alpha点数在同一次带计数渲染结果上取得，共3960条；计数版与普通Renderer在首／中／末三帧×五预算共15组颜色和radii逐bit通过。没有保存渲染帧图片。\n',
        table(['预算','处理median ms','总median ms','总p90 ms','总p99 ms','低于训练最小ratio的帧数'],[
        [b,f'{s["ours_"+str(b)]["prepare_ms_median"]:.4f}',f'{s["ours_"+str(b)]["total_ms_median"]:.4f}',f'{s["ours_"+str(b)]["total_ms_p90"]:.4f}',
         f'{s["ours_"+str(b)]["total_ms_p99"]:.4f}',s['ours_'+str(b)]['below_training_min_frames']] for b in BUDGETS]),
        '训练最低ratio为.01，100点在候选Nv>10000时属于轻微下界外推，报告单独记录，没有重训。原训练模型不变；本轮不把小预算质量和上界外推结果混称训练效果。\n',
        '## 原版复用及比较限制\n',
        '遵循用户此前复用原版结果的要求：原版点数来自vanilla_fr1xyz_render_counts_20261105/reports/rgbd_repeat1_per_view.csv，PSNR及三轮标准Full耗时来自fr1xyz_local50_latency_20261106/reports/QUALITY_COUNTS.csv和TIMINGS.csv。只取cartgs_full，不取静态属性缓存对照。逐帧检查RGB时间戳、文件路径、输入N、投影数一致；所有792帧对齐。',
        '两方法地图和各自保存SLAM位姿不同，相同的是RGB帧及640×480评价尺寸。原版历史时段和本轮GPU负载不同，因此时间曲线仅并列参考，不据此宣称严格配对加速。原版Full每次属性激活／输入准备仍计在其标准render调用内；其处理栏为0表示没有独立Selector/Compensator阶段，不代表其Python渲染没有任何准备开销。',
        '右栏约520帧后的均值抬升受测量时段影响：前两轮后段较慢，第三轮乱序的前段／后段总耗时约0.998／0.999ms，没有复现该差异。故不把平滑时间曲线直接解释为视角固有计算复杂度。全部三轮原始样本保留，GPU快照及TIMING_REGION_DIAGNOSTIC.json可复核。',
        '本轮新后端相对上一轮C++部署v2的逐bit验证见用户指定报告；相对正常PyTorch仍允许既有FP32 FMA／视角／Quaternion次序微误差，不能称与训练路径逐bit相同。\n',
        '## 完整性与交付\n',
        f'- checkpoint：`{env["checkpoint"]}`，未使用无Other消融模型。',
        f'- 实际native扩展：`{env["binary"]["path"]}`，SHA256 `{env["binary"]["sha256"]}`；源码逐文件核对用户报告的最终版本，独立构建。',
        '- SOURCE_FINAL_HASHES.json、ENVIRONMENT.json记录源码与二进制身份；604个生产关键文件、原checkpoint及历史CSV前后SHA一致。没有生产默认切换、训练、commit或push。',
        '- PER_VIEW_COMPARISON.csv包含六方法×792帧=4752条对齐结果；TIMINGS.csv保存11880条新时间样本；OURS_PER_VIEW.csv保存3960条新质量／贡献计数；历史原始数据保留在reference。',
        '- PNG／PDF／SVG主图和原始PSNR图在figures目录。\n',
        '![三栏图](figures/NATIVE_FIXED100_500_THREE_COLUMNS.png)\n']
    (OUT/'NATIVE_FIXED100_500_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
