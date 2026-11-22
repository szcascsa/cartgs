from common import *
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=ROOT/'reports'


def read(name):return json.loads((R/(name+'.json')).read_text())


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])


def main():
    rounds=[read('BENCHMARK_'+str(i)) for i in [1,2]]
    labels={'r01':'局部1%','r70':'局部70%','changing_ratio':'逐帧变化ratio','changing_points':'逐帧变化点预算'}
    speed=[];detail=[];csvrows=[]
    for i,c in enumerate(rounds[0]['results']):
        values=[r['results'][i] for r in rounds]
        med=lambda tag,stage:statistics.mean(v['summary'][tag][stage]['median_ms'] for v in values)
        a,b=med('previous_cpp','prepare'),med('native_select','prepare')
        x,y=med('previous_cpp','total'),med('native_select','total')
        speed.append([labels[c['case']],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%',
            f'{x:.3f}',f'{y:.3f}',f'{100*(1-y/x):.1f}%'])
        for run,v in enumerate(values,1):
            a,b=v['summary']['previous_cpp']['prepare'],v['summary']['native_select']['prepare']
            detail.append([run,labels[c['case']],f"{a['mean_ms']:.3f} → {b['mean_ms']:.3f}",
                f"{a['p90_ms']:.3f} → {b['p90_ms']:.3f}",f"{a['p99_ms']:.3f} → {b['p99_ms']:.3f}",
                f"{100*b['under2ms_fraction']:.1f}%"])
            for tag,stages in v['summary'].items():
                for stage,stats in stages.items():csvrows.append(dict(run=run,case=c['case'],backend=tag,stage=stage,**stats))
    with (R/'LATENCY_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    prof={v['label']:v for v in read('PROFILE_ANALYSIS')['summary']};ns=[]
    for ratio in ['0.01','0.1','0.7']:
        a,b=[prof[f'prepare/{tag}/{ratio}'] for tag in ['previous_cpp','cpp']]
        ns.append([ratio,f"{a['wall_us']:.1f} → {b['wall_us']:.1f}",f"{a['kernel_us']:.1f} → {b['kernel_us']:.1f}",
            f"{a['kernels']:g} → {b['kernels']:g}",f"{a['syncs']:g} → {b['syncs']:g}",f"{a['dtoh']:g} → {b['dtoh']:g}"])
    valid=read('VALIDATION');traj=read('TRAJECTORY_VALIDATION');extra=read('EXTRA_VALIDATION')
    oldmem,newmem=read('MEMORY_previous_cpp'),read('MEMORY_native_select')
    micro=read('SELECTION_BENCHMARK')['results'];microrows=[]
    for row in micro:
        a,b=row['median_ms']['cub'],row['median_ms']['radix']
        microrows.append([row['N'],row['ratio'],f'{1000*a:.1f}',f'{1000*b:.1f}',f'{a/b:.2f}×',
            '阈值＋压缩' if row['N']<=16384 else 'C++ CUB'])
    sections=['''# 冻结部署：精确选点与状态同步优化报告

本轮仅修改训练结束后的部署实现。正常训练、训练前向、训练评价以及现有普通推理入口不接入部署版。独立目录：/home/prmi/zwj/experiments/deployment_native_select_20261106。

## 实现

1. 删除部署接口的scheduler参数、Python调度分支、旧私有辅助函数以及CLI的--scheduler；部署准备统一由C++执行。历史v2源码和二进制只读保留作实验对照，不作为产品回退接口。
2. 对不超过16384个候选的请求，以三轮11bit Radix精确查找Kth score，再按“高于阈值的点＋原顺序前若干同分点”稳定压缩。score完整32bit参与比较，正负零按原CUB的同分语义处理。输出严格K个点并恢复原候选顺序；没有近似Top-K、量化分数或改变同分优先级。
3. 一次将候选Key读入寄存器；Histogram和阈值查找在CTA内并行执行。分块计数与稳定压缩直接读取全局eligibility并发布全局Gaussian ID，消除两次索引Gather和两次全量排序。
4. 大于16384候选继续在C++使用原CUB算法，避免单CTA阈值算法在大地图退化。新Scratch按地图规模有界分配，不按相机或ratio积累缓存。
5. Selector和Compensator共享本请求finite状态。CUDA Stream依赖保证Selector完成后启动补偿；状态1表示Selector失败，此时Compensator在读取选点前统一退出。状态2表示补偿失败，不用于CTA中途退出，避免线程在__syncthreads前分歧。Host统一读回并优先报告Selector错误；异常结果不进入Rasterizer。有效补偿路径减少一次状态清零、一次4B读回和一次Stream同步。关闭补偿时仍在选点前检查Selector。

网络计算公式、四Head、Feature共享和训练配置未改。没有复用同帧视锥Mask、按ratio缓存Anchor48结果、CUDA Graph或减少渲染点数。

## 两轮动态端到端实测

对照是上一轮已优化的C++部署版v2，不是早期PyTorch版本。RTX3090，fr1_xyz RGB-D，13298个Gaussian、2790个Anchor；每轮每条件792帧、20次预热，后端顺序逐帧交错。相机、候选N、预算和ratio真实更新。

下表取两轮中位耗时的平均。准备使用Python调用外层墙钟，包含请求检查、C++/CUDA调度、必要同步及返回封包；准备＋渲染等待GPU完成，包含原Rasterizer，不含HTTP、显示、相机构建和训练。
''',table(['条件','原准备ms','新准备ms','准备下降','原准备＋渲染ms','新准备＋渲染ms','完整下降'],speed),'''
ratio轮转为.01/.03/.05/.08/.1/.2/.3/.4/.5/.6/.7；点预算轮转为1000/2000/3000/4000/5000/10000。实际K由每帧自身N和原资格规则决定，保留预算不足shortfall。原始样本见BENCHMARK_1/2.json。

GPU同时有其他用户任务，未终止无关进程。均值、p90/p99会受GPU排队干扰，不把中位数下降解释为所有帧稳定满足某个毫秒上限。
''',table(['轮次','条件','准备mean ms 原→新','准备p90 ms 原→新','准备p99 ms 原→新','新准备<2ms'],detail),'''
## Nsight Systems

12个动态相机×3个ratio，按NVTX范围和CUPTI correlation归属实际发射的工作，不把不同Profiler条件下Host和GPU时间相减当作精确CPU开销。
''',table(['ratio','Host范围μs 原→新','累计GPU Kernel μs 原→新','Kernel数','同步API数','DtoH数'],ns),'''
## 选择算子CUDA Event微测

相同score、候选ID、eligibility和K，交错9轮，每轮100次，包含调用发射可能形成的GPU空隙；它不是纯Kernel时间，纯GPU归因以上面的Nsight为准。大于16384行是CUB路由结果，不能计作阈值算法加速。
''',table(['候选N','ratio','原链μs','新链μs','倍率','实际方法'],microrows),'''
## 正确性与接口
''',table(['验证','覆盖','结果'],[
    ['精确选点',f"{read('SELECTION_VALIDATION')['checks']}组，N=0–131073，分数同值/饱和/正负零、资格Mask、非平凡全局ID、阈值分流边界",'ID逐bit一致'],
    ['五类模型完整路径',f"{valid['cases']}组／{valid['comparisons']}次比较；旧11D与joint schema1/2/3/4",'通过'],
    ['全轨迹',f"{traj['frames']}帧／{traj['cases']}组／{traj['comparisons']}次比较",'ID/属性/radii/像素逐bit一致'],
    ['生命周期与错误','独立快照、源模型更新、inference_mode、非默认Stream、输出所有权、空候选、12次Selector异常与补偿异常恢复','通过'],
    ['常规测试',f"{read('NORMAL_TESTS')['tests']}项预算/Viewer/CLI测试",'通过'],
    ['独立部署CLI','6帧真实推理，旧scheduler选项明确拒绝','通过']]),'''
以上逐bit结论针对上一轮部署v2。本轮没有额外渲染数值误差，因此PSNR/SSIM/LPIPS不变；不重复运行感知网络代替像素等价验证。部署网络相对正常PyTorch的既有微误差仍见deployment_point_runtime_20261106报告，不能将本轮结论扩大为与训练网络逐bit一致。

接口现在为DeploymentRenderer.from_checkpoint(checkpoint)，无需scheduler参数；使用runtime.render(camera,budget=2000)或ratio_budget=.1。普通训练和原Viewer默认后端保持原实现；没有把部署入口用于训练。
''',f'''
## 显存与受保护文件

旧C++部署私有快照保留{oldmem['retained_bytes']/1024**2:.2f} MiB，新版{newmem['retained_bytes']/1024**2:.2f} MiB；100次变化输入测得峰值增量分别{oldmem['peak_extra_bytes']/1024**2:.2f}/{newmem['peak_extra_bytes']/1024**2:.2f} MiB。当前小地图不再为新C++管线分配全量排序Scratch；仍保留初始正常推理对象的兼容状态，未为节省显存重构它。

SOURCE_FINAL_HASHES.json与BINARY_IDENTITY.json记录最终源码、实际加载.so和编译参数；CUDA_RESOURCES.txt记录资源用量，阈值算子的寄存器、共享内存和Local/Stack资源用量。生产604文件及70个checkpoint文件前后哈希逐项一致，FINAL_INTEGRITY.json可核验。共享conda未修改，没有训练、渲染图片、自动commit、push或生产部署。

## 失败候选和修正

最初4轮8bit、重复读取score的单CTA阈值方案在32768/65536候选上退化到约0.14/0.27ms，高于原CUB约0.08–0.10ms，未无条件保留。最终只在≤16384候选使用寄存器和并行Histogram查找；更大候选走原CUB。初版数值与性能证据保存在SELECTION_BENCHMARK_v3.json、SELECTION_VALIDATION_v3.json和selection_only_v3_source.tar.gz。

配对加载新旧扩展时，首次发现pybind全局Pipeline类型重复注册；已将新版类绑定设为module_local，重新编译并完成全部回归。不是渲染算法差异，错误日志保留在runner_pybind_collision.log。

## 后续空间

目前仍有一次动态候选计数读回和一次最终错误状态读回。继续减少同步需要让Selector、选点、Compensator及Rasterizer共同接收device-side N/K，并重新设计输出容量与错误屏障，属于下一项独立工程；不能仅删除同步。高K下补偿网络和Rasterizer的Gaussian→Tile展开将限制整体收益。

Selector和Compensator存在真实数据依赖，不能并行计算同一个请求的二者。Gaussian间和四Head之间已经CUDA并行；下一步可针对目标端侧GPU研究小矩阵批处理/权重分块与多帧流水，但不能把预缓存固定输入或并发任务排队差异当成加速。实际端侧型号未知，本报告的毫秒数仅代表RTX3090。
''']
    (R/'NATIVE_SELECTION_OPTIMIZATION_REPORT.md').write_text('\n\n'.join(sections))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    folder=R/'figures';folder.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,stage,title in zip(axes,['prepare','total'],['完整准备','准备＋渲染']):
        for j,tag in enumerate(['previous_cpp','native_select']):
            y=[statistics.mean(v['results'][i]['summary'][tag][stage]['median_ms'] for v in rounds) for i in range(4)]
            bars=ax.bar([i+(j-.5)*.36 for i in range(4)],y,.36,label='上一版C++部署' if j==0 else '精确选择＋状态融合')
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
        ax.set_xticks(range(4),[labels[v['case']] for v in rounds[0]['results']],rotation=12)
        ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.set_ylabel('中位耗时（ms）');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('动态输入增量优化（RTX3090；有其他GPU任务）');fig.tight_layout()
    fig.savefig(folder/'native_prepare_speed.png',dpi=180);fig.savefig(folder/'native_prepare_speed.pdf');plt.close(fig)
    print('REPORT_CREATED',flush=True)


if __name__=='__main__':main()
