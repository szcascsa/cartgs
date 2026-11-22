"""最终推理收益和移动相机成本的中文科学图。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent/'results/reports';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.right':False,'axes.spines.top':False})


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


def main():
    data=json.loads((OUT/'BENCHMARK_2.json').read_text());index={(v['condition'],v['backend']):v for v in data['summary']}
    conditions=['ratio_0.01','ratio_0.05','ratio_0.1','ratio_0.3','ratio_0.7','points_2000','points_5000']
    labels=['1%','5%','10%','30%','70%','固定2k','请求5k'];x=np.arange(len(labels))
    colors=['#64748B','#009E73','#D55E00'];backends=['pytorch','cuda','cuda_graph_no_timing']
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    for ax,key,title in zip(axes,['prepare_host_ms','total_host_ms'],['选点与补偿准备','准备＋Rasterizer']):
        for j,(name,color,label) in enumerate(zip(backends,colors,['原PyTorch','融合CUDA','安全Graph策略＋关闭计时同步'])):
            ax.bar(x+(j-1)*.24,[index[c,name][key] for c in conditions],width=.24,color=color,label=label)
        ax.set(xticks=x,xticklabels=labels,ylabel='热态中位数（ms）',title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('fr1_xyz RGB-D｜冻结19D Selector＋Residual48视角Compensation',fontsize=13)
    fig.text(.5,.015,'五相机、相同K与输入、每条件每后端200热样本／轮；不包含HTTP和显示，不代表在线SLAM训练加速。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.83));save(fig,'INFERENCE_LATENCY')

    moving=json.loads((OUT/'MOVING_BENCHMARK.json').read_text())['results'];x=np.arange(len(moving))
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    for ax,field,title in zip(axes,['total_ms','p99_ms'],['连续792帧累计时间（秒，包含捕获）','逐帧p99延迟（ms）']):
        for j,(name,color,label) in enumerate(zip(['pytorch','cuda','cuda_graph'],colors,['原PyTorch','融合CUDA','安全Graph策略'])):
            values=[v[field][name]/(1000 if field=='total_ms' else 1) for v in moving]
            ax.bar(x+(j-1)*.24,values,width=.24,color=color,label=label)
        ax.set(xticks=x,xticklabels=['1%','10%','70%','固定2k','请求5k'],title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('相机持续移动的实际开销｜不以热态结果掩盖捕获尖峰',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.83));save(fig,'MOVING_CAMERA_LATENCY')


if __name__=='__main__':main()
