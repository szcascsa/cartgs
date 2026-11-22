from common import *
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=ROOT/'reports'


def read(n):return json.loads((R/(n+'.json')).read_text())


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])


def main():
    rounds=[read('BENCHMARK_'+str(i)) for i in [1,2]]
    labels={'r01':'局部1%','r70':'局部70%','changing_ratio':'变化ratio','changing_points':'变化点预算','jumping_view':'跳转视角＋变化预算'}
    speed=[];tails=[];csvrows=[]
    for i,c in enumerate(rounds[0]['results']):
        values=[r['results'][i] for r in rounds]
        med=lambda tag,stage:statistics.mean(v['summary'][tag][stage]['median_ms'] for v in values)
        a,b=med('previous_cpp','prepare'),med('device_counts','prepare')
        x,y=med('previous_cpp','total'),med('device_counts','total')
        speed.append([labels[c['case']],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%',f'{x:.3f}',f'{y:.3f}',f'{100*(1-y/x):.1f}%'])
        for repeat,v in enumerate(values,1):
            a,b=v['summary']['previous_cpp']['prepare'],v['summary']['device_counts']['prepare']
            tails.append([repeat,labels[c['case']],f"{a['mean_ms']:.3f} → {b['mean_ms']:.3f}",
                f"{a['p90_ms']:.3f} → {b['p90_ms']:.3f}",f"{a['p99_ms']:.3f} → {b['p99_ms']:.3f}",f"{100*b['under2ms_fraction']:.1f}%"])
            for tag,stages in v['summary'].items():
                for stage,stats in stages.items():csvrows.append(dict(run=repeat,case=c['case'],backend=tag,stage=stage,**stats))
    with (R/'LATENCY_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    profile={v['label']:v for v in read('PROFILE_ANALYSIS')['summary']};ns=[]
    for ratio in ['0.01','0.1','0.7']:
        a,b=[profile[f'prepare/{tag}/{ratio}'] for tag in ['previous_cpp','cpp']]
        ns.append([ratio,f"{a['kernels']:g} → {b['kernels']:g}",f"{a['syncs']:g} → {b['syncs']:g}",
            f"{a['dtoh']:g} → {b['dtoh']:g}",f"{a['kernel_us']:.2f} → {b['kernel_us']:.2f}",f"{a['wall_us']:.2f} → {b['wall_us']:.2f}"])
    traj=read('TRAJECTORY_VALIDATION');val=read('VALIDATION');large=read('LARGE_PIPELINE');sel=read('DEVICE_SELECTION');audit=read('CONTROL_AUDIT')
    oldmem,newmem=read('MEMORY_previous_cpp'),read('MEMORY_device_counts');identity=read('BINARY_IDENTITY')
    loops=[]
    for row in read('LOOP_BENCHMARK')['results']:
        a,b=[row['summary'][tag]['median_ms'] for tag in ['capacity_grid','bounded_grid']]
        loops.append([row['repeat'],row['ratio'],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%'])
    big_fail=[];big_final=[]
    for destination,filename in [(big_fail,'stage_v5/LARGE_BENCHMARK'),(big_final,'LARGE_BENCHMARK')]:
        for row in read(filename)['results']:
            a,b=[row['summary'][tag]['median_ms'] for tag in ['previous_cpp','device_counts']]
            destination.append([row['capacity'],row['repeat'],row['ratio'],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%'])
    sections=['''# 部署推理：GPU动态N/K与准备同步优化报告

日期：2026-11-07。独立实验目录：/home/prmi/zwj/experiments/deployment_device_counts_20261107。

本轮只处理训练结束后的冻结部署入口，不修改训练、正常训练前向、训练评价、常规推理入口、生产默认或原模型。比较基线是上一轮已经完成的C++部署v4，实际只读加载其原.so，未用早期PyTorch时间放大增量收益。

## 设备端连续准备

候选压缩得到计数后，GPU直接计算N、合资格数量、目标预算、实际K与ratio条件，再连续执行视角编码、Selector、精确选点及Compensator。准备末尾一次读回96字节控制记录，包含统计与finite状态；随后构造精确K的Tensor视图，对接现有Rasterizer。原来的中途候选计数读回已移除。

比例模式K=min(floor(r×N),eligible)，网络条件使用请求r；固定预算模式K=min(B,eligible)，网络条件使用min(1,B/N)。资格不足时不把网络条件改成K/N。连续ratio的sin/cos及Residual权重每个请求只计算一次，未重复到每个Gaussian。

加载时建立有界GPU Scratch；输出ID与属性按用户预算和整图规模给出的上界分配，实际只计算有效N/K。GPU按实际K密集写入属性，末尾读回后创建视图，容量尾部不进入渲染。返回的存储独立于下一请求，没有通过覆盖上一帧结果提速。

最终仅在冻结地图容量≤16384时启用设备计数和单CTA精确Radix阈值选择；更大容量保留原C++计数及排序路径。此前分块设备选择虽然通过正确性检查，但大地图实测出现退化，已从最终部署路径移除并独立归档。同分优先级、完整FP32分数比较、固定噪声和最终点序保持原语义。stats.device_counts明确标记本请求使用的路径。

Selector、设备选点、Compensator通过同一CUDA Stream保证依赖。异常状态使后续阶段退出，Host统一检查后才返回给Rasterizer，保留错误来源和恢复行为。正常路径没有Python中间调度，也没有scheduler='python'回退。

## 两轮动态测量

RTX3090，fr1_xyz RGB-D，Master13298点、2790个Anchor；每条件每轮792帧、20次预热，两种后端逐帧交错顺序。下表为两轮中位耗时的平均，完整原始分布及mean/p90/p99见JSON与LATENCY_SUMMARY.csv。

准备时间使用Python调用外层墙钟，包含入参检查、C++/CUDA调度、必要等待和返回封包。准备加渲染等待GPU完成，包含Rasterizer，排除相机构建、HTTP和显示。没有复用同帧视锥、ratio→Anchor48结果缓存或CUDA Graph。
''',table(['条件','原准备ms','新准备ms','下降','原准备＋渲染ms','新准备＋渲染ms','下降'],speed),'''
变化ratio=.01/.03/.05/.08/.1/.2/.3/.4/.5/.6/.7；变化点预算=1000/2000/3000/4000/5000/10000。跳转视角按(frame×137)%792访问保存轨迹，同时轮换ratio与固定预算。每次仍根据新相机重算视锥、候选、视角条件、Residual查询及两网络。

初查时GPU空闲，后续出现其他LoD训练进程，未停止或修改无关任务。交错配对用于减少负载变化偏差；共享GPU排队仍会影响均值和尾部，不能把当前毫秒数外推到未知端侧硬件或承诺每帧低于2ms。
''',table(['轮次','条件','准备mean ms 原→新','准备p90 ms 原→新','准备p99 ms 原→新','新准备<2ms'],tails),'''
## Nsight Systems

按NVTX Host范围和CUPTI correlation归属12个动态相机×3种ratio的实际工作。纯Kernel累计时长和完整墙钟分别报告，不将少一次读回直接等同于同样比例的GPU计算加速。
''',table(['ratio','Kernel数','同步API数','DtoH数','累计Kernel μs 原→新','Host范围μs 原→新'],ns),'''
原v4准备有效补偿路径传回24B计数和4B状态；新路径一次传回96B，字节量略增但减少一次Host往返。Rasterizer内部Gaussian→Tile展开仍有独立动态计数读取，本轮未修改Rasterizer二进制，不称整个渲染链无同步。

## 有限Block循环候选

第一版v5采用容量上界发射，在高ratio下观察到GPU累计Kernel时长增加。随后测试Selector与Compensator最多1024个Block的grid-stride循环：每个Warp/Block读取实际N/K处理多项，点内运算顺序保留；Selector增加Warp同步、Compensator在下一点覆写共享内存前增加Block同步。

以下为v5容量发射与有限Block候选的独立交错对照，200个跳转视角×两轮。低ratio两轮均略慢，高ratio方向不一致，均值也更高，故循环候选未保留。候选源码与二进制哈希在reports/loop_candidate，v5结果在reports/stage_v5。
''',table(['轮次','ratio','v5准备ms','循环候选准备ms','下降'],loops),f'''
循环候选恢复为v5的逐点发射方式后，再加入静态容量路由，最终形成v7。实际扩展：{identity['path']}，SHA256：{identity['sha256']}。最终保留方式以该二进制及源码快照为准。

## 正确性与数值范围
''',table(['验证','覆盖','结果'],[
    ['五类checkpoint',f"{val['cases']}组／{val['comparisons']}次比较；旧11D与joint schema1/2/3/4",'选点、属性、像素、radii逐bit一致'],
    ['全轨迹',f"{traj['frames']}帧／{traj['cases']}组／{traj['comparisons']}次比较",'全部通过逐bit检查'],
    ['大容量保留路径',f"{large['cases']}组；{large['capacities']}点合成重复地图",'确认device_counts=False；ID/属性及抽查渲染逐bit一致'],
    ['最终设备选择',f"{sel['checks']}组；最大容量{sel['capacity_max']}点",'严格K、带毒尾部、状态保护通过'],
    ['边界与生命周期','零预算、空候选、资格不足、独立快照、非默认Stream、输出所有权、NaN载荷及错误恢复','通过'],
    ['常规入口',f"{read('NORMAL_TESTS')['tests']}项预算/Viewer/CLI测试，6帧独立部署CLI",'通过'],
    ['Compute Sanitizer','memcheck对最终真实准备、设备选择、错误状态与尾部越界', '0 errors']]),f'''
连续ratio控制审计另有{audit['cases']}组：FP32控制记录有{audit['control_mismatches']}组与CPU数学库逐位不同，最大绝对差{audit['max_control_abs']:.9g}；其中有效K>0的差异为{audit.get('active_control_mismatches','另见CONTROL_AUDIT.json')}组。ID差异{audit['ids_mismatches']}组，输出属性差异{audit['attribute_mismatches']}组。该审计保留独立数值差异，不能将GPU/CPU的log/sin/cos宣称为数学上逐位等价。

完整轨迹对照像素一致，因此本次已验场景的PSNR/SSIM/LPIPS无新增变化，无需重复跑感知网络代替像素等价证明。这里的基线是部署v4；部署网络相对正常PyTorch的既有微误差仍见上一阶段报告，不能扩大为与训练实现完全等价。

## 内存与保护

旧部署快照及持久状态保留{oldmem['retained_bytes']/1024**2:.2f} MiB，新版{newmem['retained_bytes']/1024**2:.2f} MiB；100次动态输入测得峰值增量分别{oldmem['peak_extra_bytes']/1024**2:.2f}/{newmem['peak_extra_bytes']/1024**2:.2f} MiB。新显存用于控制记录、分数/视角Scratch和容量上界输出，不随请求数累计。应用自行保留返回Tensor所占的存储不属于隐式缓存。

正常网络结构、训练Loss、Ratio、优化器、步数、拓扑与默认配置未改；不更新Master或Anchor Feature。生产604文件与70个模型文件前后哈希逐项一致，见FINAL_INTEGRITY.json及完整manifest。没有修改共享conda、终止无关进程、训练、保存渲染图片、commit、push或部署生产。

## 大地图失败候选与保留路径

以下为分块设备计数候选的独立配对准备微测：两种合成地图、100个跳转视角、两轮，不包含Rasterizer。部分条件退化明显，因此该候选不进入最终默认。
''',table(['容量','轮次','ratio','原C++准备ms','分块候选ms','下降'],big_fail),'''
最终大地图恢复原C++计数与排序路径；下面是恢复后与原v4的独立配对。两批GPU负载不同，只比较各批内部配对，不能跨表相减计算收益。微测调用同一C++接口，不含Python入参校验和渲染。
''',table(['容量','轮次','ratio','原C++准备ms','最终保留路径ms','下降'],big_final),'''
## 当前边界与下一步

最终小地图加速路径在准备末尾仍读回一次N/K与状态，以兼容现有精确K的Torch/Rasterizer接口；大地图保持原来的中途计数依赖。若继续移除它，需要独立部署Rasterizer直接消费GPU上的K，并同步处理Tile实例数、排序、容量扩展和错误屏障。单纯预分配内存或删除同步不足以完成这一点。

容量不足时必须显式报错或扩容重算，禁止截断点数。后续优先对大地图分块Histogram/归约和容量早退成本单独归因，再研究设备端Tile列表与计数排序。1024 Block循环已因收益不足撤回；百万点分块选择只属于已撤回候选的正确性数据，不能作为最终默认覆盖范围。合成重复地图不代表新场景渲染质量或在线训练结果。
''']
    (R/'GPU_DEVICE_COUNTS_OPTIMIZATION_REPORT.md').write_text('\n\n'.join(sections))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    folder=R/'figures';folder.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(13.5,4.7))
    for ax,stage,title in zip(axes,['prepare','total'],['完整准备','准备＋渲染']):
        for j,tag in enumerate(['previous_cpp','device_counts']):
            y=[statistics.mean(v['results'][i]['summary'][tag][stage]['median_ms'] for v in rounds) for i in range(5)]
            bars=ax.bar([i+(j-.5)*.36 for i in range(5)],y,.36,label='原C++部署v4' if j==0 else 'GPU动态计数')
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.set_xticks(range(5),[labels[v['case']] for v in rounds[0]['results']],rotation=14,fontsize=9)
        ax.set_ylabel('中位耗时（ms）');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('动态视角与预算：GPU计数连续准备（RTX3090，共享GPU）');fig.tight_layout()
    fig.savefig(folder/'device_counts_speed.png',dpi=180);fig.savefig(folder/'device_counts_speed.pdf');plt.close(fig)
    print('REPORT_CREATED',flush=True)


if __name__=='__main__':main()
