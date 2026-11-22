"""用已核验真实结果绘制逐场景Full质量图，不重新渲染，不绘制LoD。"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT if (ROOT/'data/FULL_VS_VANILLA.json').exists() else ROOT/'figures/full_vs_vanilla'
INPUT=OUT/'data/FULL_VS_VANILLA.json'
SCENES=['office0','office1','office2','office3','office4','room0','room1','room2']
MODES=['mono','rgbd']
COLORS={'ours':'#009988','vanilla':'#7A8190'}
METRICS=[('psnr','PSNR ↑','dB',3),('ssim','SSIM ↑','',4),
         ('lpips','LPIPS ↓','',4),('gaussian_count','Gaussian 数量 N','点',0)]


def fmt(key,value,precision):
    return f'{value:,.0f}' if key=='gaussian_count' else f'{value:.{precision}f}'


def style():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    assert font.is_file(),'缺少核验过的中文字体'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'svg.fonttype':'none','font.size':10,'axes.spines.right':False,
        'axes.spines.top':False,'savefig.facecolor':'white','axes.titleweight':'bold'})
    warnings.filterwarnings('error',message='Glyph .* missing from font')


def decorate(ax,key,values):
    maximum=max(values)
    ax.set_ylim(0,1.12 if key=='ssim' else maximum*1.25)
    if key=='ssim':ax.set_yticks([0,.25,.5,.75,1.])
    if key=='gaussian_count':ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v/1000:g}k'))
    ax.grid(axis='y',color='#D8E0E6',linewidth=.6,alpha=.75)
    ax.set_axisbelow(True);ax.set_facecolor('#FAFBFC')
    ax.spines[['bottom','left']].set_color('#B8C2CA')
    ax.tick_params(axis='both',labelsize=9,length=3,color='#B8C2CA')


def plot_scene(scene,lookup):
    fig,axes=plt.subplots(2,4,figsize=(13.8,7.4),facecolor='white')
    fig.subplots_adjust(left=.075,right=.98,bottom=.14,top=.775,hspace=.65,wspace=.30)
    fig.suptitle(f'Replica · {scene}｜Full 渲染质量与 Gaussian 数量',
                 x=.5,y=.985,fontsize=19,fontweight='bold',color='#243746')
    fig.text(.5,.920,'本次 L1→L2 阈值 3τ0 · Full SLAM＋tail · 完整轨迹评价',
             ha='center',fontsize=11,color='#52616D')
    handles=[Patch(facecolor=COLORS['ours'],label='本次 Full（3τ0）'),
             Patch(facecolor=COLORS['vanilla'],label='Vanilla CaRtGS Full')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.903),ncol=2,
               frameon=False,fontsize=11,handlelength=1.6,columnspacing=3)
    for row,mode in enumerate(MODES):
        group=lookup[scene,mode];ours,vanilla=group['ours'],group['vanilla']
        mode_name='Mono 单目' if mode=='mono' else 'RGB-D'
        for col,(key,title,unit,precision) in enumerate(METRICS):
            ax=axes[row,col];values=[ours[key],vanilla[key]]
            bars=ax.bar([0,1],values,color=[COLORS['ours'],COLORS['vanilla']],width=.58)
            decorate(ax,key,values)
            ax.set_title(title,loc='left',pad=12,fontsize=12,color='#273D4A')
            ax.set_xticks([0,1],['本次 Full','Vanilla']);ax.set_ylabel(unit,fontsize=9)
            for bar,value in zip(bars,values):
                ax.annotate(fmt(key,value,precision),(bar.get_x()+bar.get_width()/2,value),
                            xytext=(0,6),textcoords='offset points',ha='center',va='bottom',
                            fontsize=11,fontweight='bold',color='#243746')
        y=axes[row,0].get_position().y1+.048
        fig.text(.075,y,f'{mode_name}  ·  本次 {ours["frames"]:,} 帧 / Vanilla {vanilla["frames"]:,} 帧',
                 fontsize=11.5,fontweight='bold',color='#243746')
    fig.text(.075,.073,'点数为 Full 地图全部存储 Gaussian 数 N，不是单视角可见点数；图中未使用 Selector 或 Compensation。',
             fontsize=9,color='#55636D')
    fig.text(.075,.043,'本次为 seed=0 单次；Vanilla 取 seed=0 三次中 PSNR 最高一次，SSIM／LPIPS／N 绑定同次。',
             fontsize=9,color='#55636D')
    fig.text(.075,.017,'各自完整轨迹均值，未作共同 RGB 帧重聚合；不将两者差值解释为阈值改动的单因素收益。',
             fontsize=8.5,color='#697783')
    png=OUT/(scene+'_full_vs_vanilla.png');svg=png.with_suffix('.svg')
    fig.canvas.draw()
    heading,subtitle=fig.texts[:2]
    assert not heading.get_window_extent().overlaps(subtitle.get_window_extent()),scene+'标题与副标题重叠'
    fig.savefig(png,dpi=450);fig.savefig(svg);plt.close(fig)
    with Image.open(png) as img:dimensions=list(img.size);assert min(dimensions)>2500
    return dict(scene=scene,png=png.name,svg=svg.name,dimensions=dimensions)


def plot_overview(mode,lookup):
    fig,axes=plt.subplots(2,2,figsize=(15,10),facecolor='white')
    fig.subplots_adjust(left=.075,right=.98,bottom=.145,top=.84,hspace=.40,wspace=.22)
    name='Mono 单目' if mode=='mono' else 'RGB-D'
    fig.suptitle(f'Replica · {name}｜各场景 Full 渲染结果',y=.975,fontsize=21,fontweight='bold',color='#243746')
    fig.legend(handles=[Patch(facecolor=COLORS['ours'],label='本次 Full（3τ0）'),
                        Patch(facecolor=COLORS['vanilla'],label='Vanilla CaRtGS Full')],
               loc='upper center',bbox_to_anchor=(.5,.938),ncol=2,frameon=False,fontsize=12)
    x=np.arange(len(SCENES));width=.35
    for ax,(key,title,unit,precision) in zip(axes.flat,METRICS):
        all_values=[]
        for method,shift in [('ours',-width/2),('vanilla',width/2)]:
            values=[lookup[s,mode][method][key] for s in SCENES];all_values+=values
            bars=ax.bar(x+shift,values,width=width,color=COLORS[method])
            for bar,value in zip(bars,values):
                label=f'{value/1000:.1f}k' if key=='gaussian_count' else f'{value:.{2 if key=="psnr" else 3}f}'
                ax.annotate(label,(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),
                            textcoords='offset points',ha='center',va='bottom',fontsize=7.3,rotation=90)
        decorate(ax,key,all_values)
        ax.set_ylim(0,1.28 if key=='ssim' else max(all_values)*1.38)
        ax.set_xticks(x,SCENES,rotation=25,ha='right');ax.set_title(title,loc='left',fontsize=13,pad=11)
        ax.set_ylabel(unit)
    fig.text(.075,.077,'N为地图全部存储Gaussian数；每个场景的精确点数及完整精度指标见逐场景图和data/FULL_VS_VANILLA.json。',fontsize=10,color='#55636D')
    fig.text(.075,.047,'本次seed0单次；Vanilla按同场景同模式三次最高PSNR选一次，其余指标与点数绑定同次；各自全轨迹评价。',fontsize=10,color='#55636D')
    fig.text(.075,.017,'本次只绘图，未重新训练或渲染。两种系统的训练预算、拓扑和轨迹可能不同，图中差值不是阈值改动的单因素收益。',fontsize=9,color='#697783')
    path=OUT/('overview_'+mode+'.png');fig.savefig(path,dpi=450);fig.savefig(path.with_suffix('.svg'));plt.close(fig)
    return path


def main():
    style();data=json.loads(INPUT.read_text(encoding='utf-8'));assert data['verification']['groups']==16
    lookup={(g['scene'],g['mode']):g for g in data['groups']}
    assert set(lookup)=={(s,m) for s in SCENES for m in MODES}
    manifest=['# 真实数据清单','',
              '|图|数据文件|数据性质|来源|脚本|输出|','|---|---|---|---|---|---|',
              '|8场景Mono/RGB-D Full比较|data/FULL_VS_VANILLA.json|真实实验|本次16个Full+tail checkpoint及48个Vanilla候选的已存全轨迹结果|plot_full_vs_vanilla.py（ZIP中附带）|每场景PNG/SVG；两张模式总览PNG/SVG|','',
              '只读取已存数值，不重新渲染。每个比较项的checkpoint路径、评价目录、seed、repeat、帧数和来源SHA256均保存在JSON。',
              'Full地图N由PLY header核验；PSNR/SSIM/LPIPS由原始逐帧数值复算。']
    (OUT/'data-manifest.md').write_text('\n\n'.join(manifest)+'\n',encoding='utf-8')
    outputs=[]
    for scene in SCENES:outputs.append(plot_scene(scene,lookup));print('PLOTTED',scene,flush=True)
    overviews=[plot_overview(m,lookup) for m in MODES]
    thumb_width=1200;thumb_height=660
    contact=Image.new('RGB',(thumb_width*2,thumb_height*4),'#E8EDF1')
    for index,item in enumerate(outputs):
        with Image.open(OUT/item['png']) as original:
            thumb=ImageOps.contain(original.convert('RGB'),(thumb_width-12,thumb_height-12))
            x=(index%2)*thumb_width+(thumb_width-thumb.width)//2;y=(index//2)*thumb_height+(thumb_height-thumb.height)//2
            contact.paste(thumb,(x,y))
    contact.save(OUT/'all_scenes_preview.png')
    lines=['# Replica Full与Vanilla渲染质量图','',
           '本次3τ₀ Full＋tail与Vanilla CaRtGS Full比较，8场景，每张图分别展示Mono和RGB-D的PSNR、SSIM、LPIPS和Gaussian总数。',
           'PSNR和SSIM越高越好，LPIPS越低越好；点数是整图存储N，不是单视角可见点或有效Alpha合成点。',
           '本次为seed0单次；Vanilla为seed0三次中PSNR最高重复，其余指标和N绑定同次。两种方法的全轨迹帧覆盖可能不同，未重聚合共同帧。',
           '本轮不绘制ratio，不使用未训练LoD网络，不重新训练或渲染。','',
           '|场景|模式|本次PSNR|Vanilla PSNR|本次SSIM|Vanilla SSIM|本次LPIPS|Vanilla LPIPS|本次N|Vanilla N|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        for mode in MODES:
            g=lookup[scene,mode];a,b=g['ours'],g['vanilla']
            lines.append(f'|{scene}|{mode}|{a["psnr"]:.3f}|{b["psnr"]:.3f}|{a["ssim"]:.4f}|{b["ssim"]:.4f}|{a["lpips"]:.4f}|{b["lpips"]:.4f}|{a["gaussian_count"]:,}|{b["gaussian_count"]:,}|')
    lines+=['','## 逐场景图片','']
    lines.extend(f'- [{s}：PNG]({s}_full_vs_vanilla.png)／[SVG]({s}_full_vs_vanilla.svg)' for s in SCENES)
    (OUT/'README_结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(passed=True,scene_figures=len(outputs),mode_overviews=2,png_dpi=450,
                      metric_records=32,data_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                      outputs=outputs,real_data=True,rerendered=False,title_overlap_checked=True)
    (OUT/'PLOT_VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'Replica_Full_vs_Vanilla_3tau0.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(OUT))
        z.write(Path(__file__),'plot_full_vs_vanilla.py')
    print('COMPLETE',archive,flush=True)


if __name__=='__main__':main()
