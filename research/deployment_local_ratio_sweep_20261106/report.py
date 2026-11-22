"""本地中文报告和可导出ratio–质量／延迟曲线。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'reports';RATIOS=[.1,.3,.5,.8]


def load(n):return json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))


def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,v))+' |' for v in values])+'\n'


def main():
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':10,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    speed=load('SPEED_SUMMARY');s={v['case']:v for v in speed if v['repeat']=='all'}
    quality=load('QUALITY_SUMMARY');q={(v['backend'],v['case']):v for v in quality if v['partition']=='all'}
    delta={v['case']:v for v in load('PAIRED_QUALITY_SUMMARY') if v['partition']=='all'}
    env=load('ENVIRONMENT');gates=load('QUALITY_GATES')
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figures/(name+'.'+ext),dpi=180,facecolor='white',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,(key,label) in zip(axes,[('psnr','PSNR↑（dB）'),('ssim','SSIM↑'),('lpips','LPIPS↓')]):
        values=[q['deployment',f'local_{r:g}'][key+'_mean'] for r in RATIOS]
        ax.plot(RATIOS,values,color='#0072B2',lw=2,marker='o',label='新部署MLP')
        ax.scatter([.8],[values[-1]],s=75,facecolors='none',edgecolors='#D55E00',linewidths=1.5,label='0.8：外推')
        ax.axhline(q['full','full'][key+'_mean'],color='#64748B',ls='--',lw=1.5,label='当前Master Full')
        ax.set(xlabel='当前视角Ratio',ylabel=label,xticks=RATIOS,xticklabels=['10%','30%','50%','80%']);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('fr1_xyz RGB-D｜两网络真实视角＋LI=0.1｜792帧平均质量',fontsize=13)
    fig.text(.5,.005,'训练最高ratio为0.7；0.8超过训练范围。实际K按名义可见Anchor成员数计算，受slot资格限制。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));save(fig,'LOCAL_RATIO_QUALITY')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(4);prep=[s[f'local_{r:g}']['prepare_ms_mean'] for r in RATIOS];render=[s[f'local_{r:g}']['render_ms_mean'] for r in RATIOS]
    axes[0].bar(x,prep,label='准备：选点＋补偿等',color='#0072B2');axes[0].bar(x,render,bottom=prep,label='Rasterizer',color='#009E73')
    for i,r in enumerate(RATIOS):axes[0].text(i,prep[i]+render[i],f'{prep[i]+render[i]:.3f} ms',ha='center',va='bottom',fontsize=10)
    axes[0].set_ylim(0,max(np.asarray(prep)+np.asarray(render))*1.3)
    axes[0].set(xlabel='局部Ratio',ylabel='同步墙钟均值（ms）',xticks=x,xticklabels=['10%','30%','50%','80%（外推）']);axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis='y',alpha=.2)
    for r in RATIOS:
        values=[next(v for v in speed if v['case']==f'local_{r:g}' and v['repeat']==str(i)) for i in range(3)]
        axes[1].plot([1,2,3],[v['total_ms_mean'] for v in values],marker='o',label=f'ratio={r:g}')
    axes[1].set(xlabel='轮次',ylabel='准备＋渲染均值（ms）',xticks=[1,2,3]);axes[1].legend(frameon=False);axes[1].grid(alpha=.2)
    fig.text(.5,.005,'64视角预热，792帧×3轮；包括finite检查；模型加载、JIT、相机构建、GT／指标和显示在计时外。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'LOCAL_RATIO_LATENCY')
    lines=['# 新部署MLP：局部0.1／0.3／0.5／0.8的质量与速度\n',
        '日期2026-11-06，服务器目录`/home/prmi/zwj/experiments/deployment_local_ratio_sweep_20261106`。\n',
        '## 主要结果\n',
        '四Head、两网络真实视角＋LI=0.1的同一冻结checkpoint，N=13298、Anchor2790。下表为全部792帧、三轮速度样本均值；质量在同样792帧上计算，包含Bank72及nonBank720。\n',
        table(['局部ratio','实际K均值','PSNR dB↑','SSIM↑','LPIPS↓','准备ms','渲染ms','总计ms'],[
        [f'{r:g}'+('（外推）' if r==.8 else ''),f'{q["deployment",f"local_{r:g}"]["K_mean"]:.2f}',
         *[f'{q["deployment",f"local_{r:g}"][key+"_mean"]:.5f}' for key in ['psnr','ssim','lpips']],
         *[f'{s[f"local_{r:g}"][key+"_mean"]:.4f}' for key in ['prepare_ms','render_ms','total_ms']]] for r in RATIOS]),
        f'当前Master Full：PSNR {q["full","full"]["psnr_mean"]:.5f} dB，SSIM {q["full","full"]["ssim_mean"]:.5f}，LPIPS {q["full","full"]["lpips_mean"]:.5f}；仅Full渲染均值 {s["full"]["total_ms_mean"]:.4f} ms（13298点）。这是当前方法地图，不能当作原版CaRtGS的35301点Full。\n',
        '准备时间包含视锥与候选构建、view code、Selector与score、严格K选择、Residual48查询、Compensator、解码、属性打包及finite检查。渲染时间为Rasterizer调用直到最后同步完成。没有单独把纯MLP耗时充当整个准备时间。\n',
        f'0.8档的 {q["deployment","local_0.8"]["budget_shortfall_frames"]}/792 帧受现有slot资格限制，实际保留比例K/Nv的逐帧均值为 {q["deployment","local_0.8"]["actual_fraction_mean"]*100:.2f}%，不是每帧完整80%。资格规则只排除“slot标为inactive且Master opacity<1/255”的点，未更改、未补点；网络条件保持0.8。\n',
        '## Ratio及计时口径\n',
        '- 采用Viewer名义可见Anchor成员数Nv：K=min(floor(r×Nv),E)，E为slot合格成员数；所有保留点计入K。Selector处理Nv，Compensator和Rasterizer处理实际K。',
        '- 0.1／0.3／0.5为训练ratio。训练最高ratio为0.7，0.8为外推：Residual Feature六表全部活跃，连续ratio编码仍使用0.8；没有补训或改成0.7。',
        '- RTX3090、640×480、FP32、TF32关闭、seed0固定Gumbel噪声。64视角预热，792帧三轮顺序／逆序／seed0乱序，每帧轮换方法顺序，全部11880次主测速保留。另480次CUDA Event。',
        '- 相机／GT／指标／编译／模型读取和Viewer传输显示均不计入；无CUDA Graph、无上帧Mask或最终输出缓存。准备返回前有finite完成依赖。\n',
        table(['ratio','准备median ms','总median ms','总p90 ms','总p99 ms','点数范围','候选不足帧'],[
        [r,f'{s[f"local_{r:g}"]["prepare_ms_median"]:.4f}',f'{s[f"local_{r:g}"]["total_ms_median"]:.4f}',f'{s[f"local_{r:g}"]["total_ms_p90"]:.4f}',
         f'{s[f"local_{r:g}"]["total_ms_p99"]:.4f}',f'{s[f"local_{r:g}"]["K_min"]:.0f}–{s[f"local_{r:g}"]["K_max"]:.0f}',q['deployment',f'local_{r:g}']['budget_shortfall_frames']] for r in RATIOS]),
        table(['ratio','第一轮总ms','第二轮总ms','第三轮总ms'],[[r,*[f'{next(v for v in speed if v["case"]==f"local_{r:g}" and v["repeat"]==str(i))["total_ms_mean"]:.4f}' for i in range(3)]] for r in RATIOS]),
        '共享GPU上的其他任务、时钟、温度等按轮次记录在GPU_OBSERVATIONS.json。所有轮次保留，未筛掉长尾；不能拿本轮与历史不同负载的1ms／2ms／34ms直接相减。FPS若按1000/均值计算仅代表离屏准备＋渲染，不是Viewer或在线SLAM帧率。\n',
        '## 全轨迹质量与部署数值误差\n',
        '同792帧评价新部署、原精确CUDA四ratio及未补偿Full，共7128条。PSNR为逐帧PSNR均值，SSIM使用11×11窗口，LPIPS-AlexNet normalize=True，输入RGB clamp到[0,1]；photo-L1另存CSV。\n',
        table(['ratio','精确CUDA PSNR','部署PSNR','平均ΔPSNR','最差单帧ΔPSNR','平均ΔSSIM','平均ΔLPIPS','选点ID相同帧比例'],[
        [r,f'{q["cuda",f"local_{r:g}"]["psnr_mean"]:.7f}',f'{q["deployment",f"local_{r:g}"]["psnr_mean"]:.7f}',
         *[f'{delta[f"local_{r:g}"][key]:+.3e}' for key in ['delta_psnr_mean','delta_psnr_min','delta_ssim_mean','delta_lpips_mean']],
         f'{delta[f"local_{r:g}"]["ids_same_fraction"]*100:.2f}%'] for r in RATIOS]),
        f'相对精确CUDA的预声明门槛：平均ΔPSNR≥−0.01dB，最差1%视角分位≥−0.05dB，单帧最低≥−0.1dB。四档结果：**{"全部通过" if gates["passed"] else "存在失败"}**。',
        '原PyTorch与精确CUDA在五视角×四ratio的packet逐bit通过。新部署改变FP32 FMA、视角和Quaternion归约次序，不逐bit相同；不将PSNR接近等同于逐像素一致。\n',
        table(['ratio','最大像素绝对差','平均像素绝对差','XYZ最大差','Scale最大差','radii一致帧比例'],[
        [r,*[f'{delta[f"local_{r:g}"][key]:.6g}' for key in ['image_max_abs_max','image_mean_abs_mean','means3D_max_abs_max','scales_max_abs_max']],f'{delta[f"local_{r:g}"]["radii_same_fraction"]*100:.2f}%'] for r in RATIOS]),
        table(['ratio','Bank PSNR（72帧）','nonBank PSNR（720帧）','全轨迹投影点数均值'],[
        [r,*[f'{next(v for v in quality if v["backend"]=="deployment" and v["case"]==f"local_{r:g}" and v["partition"]==part)["psnr_mean"]:.5f}' for part in ['bank','nonbank']],
         f'{q["deployment",f"local_{r:g}"]["projected_K_mean"]:.2f}'] for r in RATIOS]),
        'Bank/nonBank只用于报告，未据此选择权重、ratio、训练终点或checkpoint。0.8是本checkpoint上的冻结外推结果，不能外推到其他场景或称为0.8训练效果。\n',
        '## 版本、显存与交付\n',
        f'- Checkpoint：`{env["checkpoint"]}`。两网络真实视角＋LI=0.1，四Head，未使用无Other消融模型。',
        f'- 新部署实例持久显存增量约 {env["deployment_persistent_bytes"]/2**20:.2f} MiB；加载／独立编译约 {env["deployment_load_compile_seconds"]:.2f} s，在计时之外。峰值在PEAK_MEMORY.json。',
        '- 沿用上一轮已验收的运行时源码，逐文件SHA核对；独立build及实际so路径／SHA见ENVIRONMENT.json。没有修改网络、模型权重或生产默认。',
        '- 604个主项目关键文件、checkpoint和源码前后SHA一致。没有训练、commit、push，也未修改共享环境或终止无关进程。',
        '- reports/TIMINGS.csv及CUDA_EVENTS.csv为原始性能；evaluation/PER_FRAME.csv为逐帧质量／点数；PAIRED_PER_FRAME.csv为数值误差。三张汇总CSV与Bank/nonBank统计均保留。只保存两张分析曲线及PDF，不保存渲染帧。\n',
        '![Ratio质量](figures/LOCAL_RATIO_QUALITY.png)\n\n![Ratio延迟](figures/LOCAL_RATIO_LATENCY.png)\n']
    (OUT/'DEPLOYMENT_LOCAL_RATIO_SWEEP_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
