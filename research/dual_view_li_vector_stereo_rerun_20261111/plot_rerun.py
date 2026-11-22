"""对照旧／新两次双目LoD结果，真实Ratio坐标，不改原指标。"""
from pathlib import Path
import csv
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parent
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]


def main():
    rows=list(csv.DictReader((R/'reports/RERUN_COMPARISON.csv').open(encoding='utf-8-sig')))
    assert len(rows)==46
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    figures=R/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,9.2))
    fig.subplots_adjust(left=.065,right=.985,top=.85,bottom=.14,hspace=.42,wspace=.26)
    for line,scene in enumerate(['corner-slow','robot-normal']):
        selected=[r for r in rows if r['id']==f'VECtor/{scene}/stereo']
        full,=[r for r in selected if r['scope']=='full']
        for column,(metric,title) in enumerate([('psnr','PSNR ↑'),('ssim','SSIM ↑'),('lpips','LPIPS ↓')]):
            ax=axes[line,column]
            for scope,color in [('global','#0077BB'),('view_ratio','#EE7733')]:
                data=sorted([r for r in selected if r['scope']==scope],key=lambda r:float(r['ratio']))
                assert [float(r['ratio']) for r in data]==RATIOS
                for prefix,style,width in [('old_','--',1.4),('new_','-',1.9)]:
                    ax.plot(RATIOS,[float(r[prefix+metric]) for r in data],ls=style,lw=width,
                            marker='o' if prefix=='new_' else None,markersize=3.8,color=color)
            ax.axhline(float(full['old_'+metric]),color='#666666',ls='--',lw=1.2)
            ax.axhline(float(full['new_'+metric]),color='#009988',ls='-.',lw=1.3)
            ax.set_title(f'{scene} 双目 · {title}',loc='left',fontsize=12,fontweight='bold')
            ax.set_xticks(RATIOS,[f'{r:g}' for r in RATIOS],rotation=55)
            ax.set_xlim(0,.72);ax.set_xlabel('Ratio（实际数值线性轴）',fontsize=9)
            ax.tick_params(labelsize=8.5);ax.set_facecolor('#FAFBFC');ax.grid(axis='y',alpha=.25)
            ax.spines[['top','right']].set_visible(False)
            if metric=='psnr':ax.set_ylabel('dB')
    handles=[Line2D([],[],color='#0077BB',ls='--',label='旧全局'),Line2D([],[],color='#0077BB',label='本次全局'),
             Line2D([],[],color='#EE7733',ls='--',label='旧局部'),Line2D([],[],color='#EE7733',label='本次局部'),
             Line2D([],[],color='#666666',ls='--',label='旧 Full'),Line2D([],[],color='#009988',ls='-.',label='本次 Full')]
    fig.suptitle('VECtor 双目：相同Full起点、seed=0、相同配置的LoD重跑对照',y=.982,fontsize=18,fontweight='bold')
    fig.text(.5,.94,'LI=0.1 · 双网络真实视角 · corner-slow 11,470步 / robot-normal 13,006步 · CaRtGS未校正原图全轨迹评价',
             ha='center',fontsize=10,color='#56636D')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.915),ncol=6,frameon=False,fontsize=10)
    fig.text(.5,.025,'本次为LoD Fine阶段重新训练；模型／配置来源相同，不重新运行在线SLAM。所有曲线均为实测均值，不作平滑。',
             ha='center',fontsize=9,color='#56636D')
    for extension in ['png','svg']:fig.savefig(figures/('STEREO_RERUN_COMPARISON.'+extension),dpi=450)
    plt.close(fig)
    (figures/'data-manifest.md').write_text('# 数据来源\n\n真实实验数据，未使用模拟值。\n\n'
         '| 图 | 数据 | 来源 | 脚本 | 输出 |\n|---|---|---|---|---|\n'
         '| 两组双目同配置重跑 | reports/RERUN_COMPARISON.csv | 旧批次及本次各23项原图全轨迹评价 | plot_rerun.py | STEREO_RERUN_COMPARISON.png / .svg |\n',encoding='utf-8')
    archive_path=R/'RERUN_RESULTS.zip'
    replace={str(p.relative_to(R)).replace('\\','/'):p for p in figures.iterdir()}
    replace['plot_rerun.py']=R/'plot_rerun.py'
    temporary=archive_path.with_suffix('.tmp.zip')
    with zipfile.ZipFile(archive_path) as source,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for item in source.infolist():
            if item.filename not in replace:archive.writestr(item,source.read(item.filename))
        for name,path in replace.items():archive.write(path,name)
    temporary.replace(archive_path)
    print('PLOT_COMPLETE',figures/'STEREO_RERUN_COMPARISON.png')


if __name__=='__main__':main()
