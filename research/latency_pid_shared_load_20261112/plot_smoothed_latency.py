"""只平滑耗时展示，不改变原始请求、汇总指标或未平滑图。"""
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
SIGMA=10
NAME='PID_SHARED_LOAD_CONTROL_SMOOTHED'


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    source=ROOT/'formal/REQUESTS_WITH_QUALITY.csv'
    load_path=ROOT/'reports/TIMING_WINDOW_GPU_LOAD.csv'
    protected=[source,ROOT/'reports/SUMMARY.csv',ROOT/'figures/PID_SHARED_LOAD_CONTROL.png']
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    records=read(source);load=read(load_path)
    figures=ROOT/'figures';data={};plotted=[]
    for method in METHODS:
        rows=sorted((row for row in records if row['method']==method and row['repeat']=='0'),key=lambda row:int(row['frame']))
        assert [int(row['frame']) for row in rows]==list(range(2000))
        values={key:np.asarray([float(row[key]) for row in rows]) for key in ['frame','K','render_ms','psnr','frame_start_unix']}
        values['render_ms_smooth']=gaussian_filter1d(values['render_ms'],SIGMA,mode='reflect')
        values['psnr_smooth']=gaussian_filter1d(values['psnr'],35,mode='reflect')
        data[method]=values
        plotted.extend(dict(method=method,**{key:float(array[i]) for key,array in values.items()}) for i in range(2000))
    data_path=figures/'PID_SHARED_LOAD_SMOOTHED_DATA.csv'
    with data_path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    fig,axes=plt.subplots(2,2,figsize=(12.6,7.1))
    for method,label,color in zip(METHODS,LABELS,COLORS):
        values=data[method];x=values['frame']
        axes[0,0].plot(x,values['K'],color=color,label=label,lw=1.)
        line,=axes[0,1].plot(x,values['render_ms_smooth'],color=color,lw=1.3)
        assert np.array_equal(line.get_ydata(),values['render_ms_smooth'])
        axes[1,0].plot(x,values['psnr_smooth'],color=color,lw=1.3)
    axes[0,0].set(title='(a) 实时提交预算',ylabel='实际高斯点数 K')
    axes[0,1].set(title='(b) 纯渲染时间（高斯平滑σ＝10帧）',ylabel='渲染时间 (ms)')
    axes[0,1].axhspan(.38,.42,color='#D55E00',alpha=.12)
    axes[0,1].axhline(.4,color='#636363',ls='--',lw=.9)
    axes[1,0].set(title='(c) 渲染质量',ylabel='PSNR (dB)')
    pi=data['pid_04']
    load_frames=np.interp([float(row['unix']) for row in load],pi['frame_start_unix'],pi['frame'],left=-1,right=2001)
    valid=(load_frames>=0)&(load_frames<=1999)
    axes[1,1].step(load_frames[valid],np.asarray([float(row['gpu_util_percent']) for row in load])[valid],where='mid',color='#756BB1',label='GPU总利用率')
    axes[1,1].step(load_frames[valid],np.asarray([float(row['external_SM_sum']) for row in load])[valid],where='mid',color='#009988',label='外部进程SM指标之和')
    axes[1,1].set(title='(d) 同期自然GPU负载（约1 Hz采样）',ylabel='利用率 / SM指标 (%)',ylim=(0,None))
    axes[1,1].legend(frameon=False,loc='lower left',fontsize=8)
    for axis in axes.flat:
        axis.set_xlim(0,1999);axis.set_xlabel('RGB帧编号');axis.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.51,.965),frameon=False)
    fig.suptitle('Replica office2 RGB-D｜2064×2208｜Mode40｜共享GPU下的PI控制',y=1.005,fontsize=13)
    fig.tight_layout(rect=(0,.075,1,.91))
    fig.text(.51,.04,'图示第一轮2000帧：耗时σ＝10帧、PSNR σ＝35帧高斯平滑，仅用于展示。',ha='center',fontsize=9)
    fig.text(.51,.008,'原始单次计时、p95及目标命中率保持不变；点数和负载曲线未平滑。',ha='center',fontsize=9)
    for extension in ['png','svg']:fig.savefig(figures/f'{NAME}.{extension}',dpi=450,bbox_inches='tight')
    plt.close(fig)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in before.items())
    manifest=dict(real_data=True,frames=2000,repeat=0,methods=METHODS,latency_sigma_frames=SIGMA,psnr_sigma_frames=35,
        smoothing='scipy.ndimage.gaussian_filter1d, reflect; presentation only',raw_metrics_unchanged=True,
        sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,load_path,Path(__file__),data_path]},
        dpi=450,png_svg=True,protected_hashes_unchanged=True)
    (figures/f'{NAME}_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,latency_sigma_frames=SIGMA,frames_per_method=2000,original_files_unchanged=True)))


if __name__=='__main__':main()
