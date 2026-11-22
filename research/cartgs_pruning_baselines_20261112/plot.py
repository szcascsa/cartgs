"""每场景展示随机／Opacity裁剪及我们全局LoD；Ratio与实际K分别展示。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
COLORS={'random':'#777777','opacity':'#EE7733','ours':'#0077BB'}


def main():
    data=json.loads((ROOT/'reports/COMPARISON_DATA.json').read_text(encoding='utf-8'))
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font));family='Microsoft YaHei'
    else:
        family='Noto Sans CJK SC'
    plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,'svg.fonttype':'none'})
    destination=ROOT/'figures';destination.mkdir(exist_ok=True)
    manifest=[]
    for group in data:
        ours=sorted([x for x in group['ours'] if x['scope']=='global'],key=lambda x:x['ratio'])
        full=next(x for x in group['ours'] if x['scope']=='full')
        vanilla=group['vanilla']
        fig,axes=plt.subplots(2,3,figsize=(14,8.4))
        fig.subplots_adjust(left=.065,right=.985,top=.81,bottom=.15,hspace=.43,wspace=.28)
        for column,(metric,label) in enumerate([('psnr','PSNR ↑'),('ssim','SSIM ↑'),('lpips','LPIPS ↓')]):
            for method,title in [('random','随机裁剪'),('opacity','Opacity裁剪')]:
                points=sorted([x for x in group['pruning'] if x['method']==method],key=lambda x:x['ratio'])
                axes[0,column].plot([x['ratio'] for x in points],[x[metric] for x in points],
                     color=COLORS[method],marker='o',ms=3.2,lw=1.7)
                axes[1,column].plot([x['actual_K'] for x in points],[x[metric] for x in points],
                     color=COLORS[method],marker='o',ms=3.2,lw=1.7)
            axes[0,column].plot([x['ratio'] for x in ours],[x[metric] for x in ours],
                color=COLORS['ours'],marker='o',ms=3.2,lw=1.9)
            axes[1,column].plot([x['actual_K_mean'] for x in ours],[x[metric] for x in ours],
                color=COLORS['ours'],marker='o',ms=3.2,lw=1.9)
            for row in range(2):
                ax=axes[row,column]
                ax.axhline(vanilla[metric],color='#444444',ls='--',lw=1)
                ax.axhline(full[metric],color='#0077BB',ls=':',lw=1)
                ax.set_title(label,loc='left',fontsize=12,fontweight='bold')
                ax.grid(axis='y',alpha=.25);ax.set_facecolor('#FAFBFC')
                ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8.5)
                if row==0:
                    ax.set_xlabel('保留Ratio（各自模型N）',fontsize=10)
                    ticks=[.01,.05,.1,.2,.3,.4,.5,.6,.7]
                    ax.set_xticks(ticks,[f'{x:g}' for x in ticks],rotation=50);ax.set_xlim(0,.72)
                else:
                    ax.set_xlabel('实际送入Renderer的Gaussian数量K',fontsize=10)
                    ax.ticklabel_format(axis='x',style='sci',scilimits=(4,4))
                if metric=='psnr':ax.set_ylabel('dB')
        fig.suptitle(group['group']+'：全局裁剪对照',y=.985,fontsize=16,fontweight='bold')
        fig.text(.5,.94,f"CaRtGS N={vanilla['gaussians']:,}，评价{vanilla['frames']:,}帧；"
                 f"我们 N={full['gaussian_count']:,}，评价{full.get('frames',full.get('evaluated_frames')):,}帧",ha='center',fontsize=10)
        handles=[Line2D([],[],color=COLORS[k],lw=1.8,label=v) for k,v in
                 [('random','CaRtGS 随机裁剪（seed=0）'),('opacity','CaRtGS Opacity裁剪'),('ours','我们：全局LoD')]]
        handles += [Line2D([],[],color='#444444',ls='--',label='CaRtGS Full'),
                    Line2D([],[],color='#0077BB',ls=':',label='我们 Full')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False,fontsize=9.5)
        fig.text(.5,.035,'原图全轨迹评价；裁剪不训练、不补偿。K=floor(r×N_CaRtGS)，各Ratio嵌套；上排相同比例，非同K实验。',
                 ha='center',fontsize=9,color='#56636D')
        name=group['group'].replace('/','__')
        for suffix in ['png','svg']:fig.savefig(destination/(name+'.'+suffix),dpi=450)
        plt.close(fig);manifest.append(name)
    (destination/'data-manifest.md').write_text('# 绘图数据清单\n\n'
        '全部数据为真实实验结果，无模拟值、无平滑；原始逐帧值保存在服务器各output目录。\n\n'
        '|图|数据|来源|脚本|输出|\n|---|---|---|---|---|\n'
        '|每场景裁剪对照，共32图|reports/COMPARISON_DATA.json|本轮704项评价及已有方法结果|plot.py|figures/*.png、*.svg|\n',encoding='utf-8')
    print('FIGURES_COMPLETE',len(manifest))


if __name__=='__main__':main()
