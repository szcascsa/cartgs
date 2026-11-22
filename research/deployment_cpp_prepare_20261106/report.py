from common import *
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=ROOT/'reports'
def read(name):return json.loads((R/(name+'.json')).read_text())
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in row)+'|' for row in rows])
def main():
    rounds=[read('BENCHMARK_'+str(i)) for i in [1,2]]
    labels={'r01':'局部1%','r70':'局部70%','changing_ratio':'逐帧变化ratio','changing_points':'逐帧变化点预算'}
    speed=[];detail=[];csvrows=[]
    for i,c in enumerate(rounds[0]['results']):
        values=[r['results'][i] for r in rounds]
        med=lambda tag,stage:statistics.mean(v['summary'][tag][stage]['median_ms'] for v in values)
        a,b=med('previous_python','prepare'),med('cpp','prepare');x,y=med('previous_python','total'),med('cpp','total')
        speed.append([labels[c['case']],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%',f'{x:.3f}',f'{y:.3f}',f'{100*(1-y/x):.1f}%'])
        for run,v in enumerate(values,1):
            a,b=v['summary']['previous_python']['prepare'],v['summary']['cpp']['prepare']
            detail.append([run,labels[c['case']],f"{a['mean_ms']:.3f} → {b['mean_ms']:.3f}",
                f"{a['p90_ms']:.3f} → {b['p90_ms']:.3f}",f"{a['p99_ms']:.3f} → {b['p99_ms']:.3f}",f"{100*b['under2ms_fraction']:.1f}%"])
            for tag,stages in v['summary'].items():
                for stage,stats in stages.items():csvrows.append(dict(run=run,case=c['case'],backend=tag,stage=stage,**stats))
    with (R/'LATENCY_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    prof={v['label']:v for v in read('PROFILE_ANALYSIS')['summary']};ns=[]
    for ratio in ['0.01','0.1','0.7']:
        a,b=[prof[f'prepare/{tag}/{ratio}'] for tag in ['previous_python','cpp']]
        ns.append([ratio,f"{a['wall_us']:.1f} → {b['wall_us']:.1f}",f"{a['kernel_us']:.1f} → {b['kernel_us']:.1f}",
            f"{a['kernels']:g} → {b['kernels']:g}",f"{a['syncs']:g} → {b['syncs']:g}",f"{a['dtoh']:g} → {b['dtoh']:g}"])
    valid=read('VALIDATION');traj=read('TRAJECTORY_VALIDATION');extra=read('EXTRA_VALIDATION');identity=read('BINARY_IDENTITY')
    pm,cm=read('MEMORY_python'),read('MEMORY_cpp');unchanged=read('UNCHANGED_SOURCES')
    retained=(cm['retained_bytes']-pm['retained_bytes'])/1024**2
    sections=['''# 冻结部署完整准备的C++调度优化报告

日期：2026-11-06。独立目录：/home/prmi/zwj/experiments/deployment_cpp_prepare_20261106。

## 结果

已将完整准备调度迁入C++：每帧从Python只调用一次DeploymentPipeline.prepare。候选构建、计数读回、ratio编码、Selector、稳定严格K排序、Compensator、finite状态检查和Tensor打包全部在deployment_prepare.cpp执行。Python保留应用输入、相机对象及调用接口。原Rasterizer调用仍沿用原入口，没有宣称整个应用已不依赖Python／LibTorch。

CUDA网络源码与上一版部署完全相同，本轮不增加数值近似。训练及训练期间评价、常规pytorch/cuda/cuda_graph入口保持原实现；只有显式调用的训练后部署版默认使用cpp调度。scheduler='python'保留回退对照。

## 两轮动态测量

RTX3090、fr1_xyz，Master13298点、2790个Anchor；每轮每条件792帧，20次预热，后端顺序逐帧轮转。全部相机、候选N及变化预算真实更新。未增加同帧Mask、ratio特征结果缓存或CUDA Graph。

以下是两轮中位耗时的平均。基线是上一轮已融合CUDA网络的Python部署调度，原版v1.so只读加载；同时测当前v2的python调度，确认新增编译并非收益来源。prepare外层墙钟包括Python参数检查、C++调用、必要同步及结果返回，既不减去等待，也不使用native内部prepare_ms代替用户实际时间。相机对象构建在后端计时外共用。
''',table(['条件','原Python准备ms','C++准备ms','准备下降','原准备＋渲染ms','新准备＋渲染ms','完整下降'],speed),'''
变化ratio=.01/.03/.05/.08/.1/.2/.3/.4/.5/.6/.7；变化点预算=1000/2000/3000/4000/5000/10000。每帧按自己的候选N算ratio和K，严格K和shortfall规则不变。全部原始样本见BENCHMARK_1/2.json；LATENCY_SUMMARY.csv保留三个后端、mean、median、p90和p99。

GPU上存在其他用户任务，未终止无关进程。繁忙阶段的排队等待会主导均值与长尾；中位数改善不能理解为所有帧固定快同样百分比。下表完整保留受干扰的结果：
''',table(['轮次','条件','准备mean ms 原→新','准备p90 ms 原→新','准备p99 ms 原→新','新准备低于2ms'],detail),'''
## Nsight Systems证据

12个真实相机／ratio，原Python版与C++版逐帧交错；以NVTX Host发射范围与CUPTI correlation归属。完整trace保存在服务器reports/PROFILE.nsys-rep及PROFILE.sqlite。
''',table(['ratio','Host范围μs 原→新','累计Kernel μs 原→新','Kernel数','同步API数','DtoH数'],ns),'''
Kernel数、同步次数及DtoH次数保持一致，GPU Kernel累计时间接近，Host发射范围缩短，支持收益主要来自调度迁移。不能简单用Host范围减Kernel时间作为精确Python耗时：CPU提交与GPU执行重叠，还存在拷贝和驱动等待。

## 实际迁移内容

1. Python参数验证后只跨一次pybind边界；C++直接调度原CUDA算子，不再反复回到Python生成中间Tensor和控制数据。
2. C++持有固定容量候选／排序工作区、形状信息及Pinned Host计数／错误缓冲。计数Host有效载荷24字节，错误状态4字节，不随帧增长；每次仍读取最新值。
3. 原Python的math.log/sin/cos与Residual权重控制在C++使用同等double运算。GPU仍使用相同FP32网络，未换精度、网络形状或Head。
4. C++生成最终七个Tensor视图、选点ID和统计字段；返回的属性与ID独立于下一帧结果，未通过覆盖上一请求输出换速度。
5. 执行native核心及GPU等待期间释放GIL；单个实例仍按文档要求由应用串行调用，不把多线程并发操作同一可写Workspace当作并行加速。

候选数N决定动态K和Tensor形状，因此保留一次候选计数读回；Selector错误状态在排序前检查，Compensator错误状态在渲染前检查。三次DtoH／同步均保留，没有以延后必要检查或改变预算换取收益。

## 正确性
''',table(['验证','覆盖','结果'],[
    ['首批逐bit检查',read('SMOKE')['comparisons'],'通过'],
    ['五checkpoint完整路径',f"{valid['cases']}组／{valid['comparisons']}项比较",'通过'],
    ['全轨迹',f"{traj['frames']}帧／{traj['cases']}组／{traj['comparisons']}项比较",'通过'],
    ['边界与只读性','inference_mode、非默认Stream、输出所有权、源模型更新、空候选、非有限值','通过'],
    ['实际独立CLI','python和cpp各6帧','通过'],
    ['常规预算／Viewer／CLI测试','24项','通过']]),'''
逐bit检查覆盖选点ID、候选N、实际K、ratio、shortfall、全部渲染属性、radii和像素。包含旧11D与joint schema1/2/3/4、ratio0–1、未见ratio、预算0／1／7／2000／5000／超额预算、关闭补偿和全局模式输入。全轨迹18条件×792帧共14256组，像素全部一致，因而本轮调度迁移没有额外PSNR／SSIM／LPIPS变化；不重复运行感知网络来代替等价检查。

上一轮部署FP32网络相对原PyTorch仍存在已报告的微小数值差异；本报告的逐bit结论指C++调度与上一轮部署调度，不把它扩大为部署网络与训练版逐bit相等。
''',f'''
## 显存、接入与限制

Python调度私有快照保留{pm['retained_bytes']/1024**2:.2f} MiB，C++调度保留{cm['retained_bytes']/1024**2:.2f} MiB，增加{retained:.2f} MiB。100次变化输入的峰值增量分别为{pm['peak_extra_bytes']/1024**2:.2f}／{cm['peak_extra_bytes']/1024**2:.2f} MiB。新增内存主要是native有界工作区，不缓存相机／ratio输出，不会按请求数量增长。

构造方式：DeploymentRenderer.from_checkpoint(checkpoint, scheduler='cpp')；回退为scheduler='python'。独立CLI可传--scheduler cpp或--scheduler python。两种模式均只适用于训练结束后的冻结部署；常规训练和评价不导入该模块。native返回的prepare_ms只计核心，速度判断使用外层墙钟。

新增deployment_prepare.cpp；修改仅限deployment_runtime.py、deployment_runtime_bindings.cpp、deployment_infer.py及DEPLOYMENT_RUNTIME.md。CUDA网络文件和其余{len(unchanged['files'])}个既有文件哈希保持一致。实际扩展：{identity['path']}，SHA256 {identity['sha256']}。源码与编译参数见BINARY_IDENTITY.json；RUNTIME_MAPS.txt证明实际加载位置。

本次测量仍是RTX3090。未知端侧GPU、功耗和更大地图的耗时尚未测量，不能直接按本表承诺端侧毫秒数。Python应用相机输入和Rasterizer外壳仍存在；C++native准备核心本身没有Python回调。

## 剩余瓶颈与下一步

目前排序、动态计数／错误检查的GPU→CPU依赖、以及高K的Compensator CUDA计算已占主要剩余时间。继续换Host语言无法消除这些GPU工作和必要等待。下一步最有价值的是在目标设备核查稳定严格K排序所占比例，研究与当前同分规则完全一致的radix阈值选择＋稳定压缩；保留原排序作对照。若需要进一步减少形状读回，应把动态K接口延伸到Rasterizer，需要单独验证，不能用固定容量padding改变语义。

没有使用复用同帧Mask、预缓存各ratio特征、跳过finite检查、改变严格K或修改训练的方式提速。生产目录mycartgs／realLOD和共享conda未修改；604个生产文件及70个模型文件前后哈希见FINAL_INTEGRITY.json及完整manifest。未commit、push或切换生产默认。
''']
    (R/'CPP_PREPARE_OPTIMIZATION_REPORT.md').write_text('\n\n'.join(sections))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    folder=R/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,stage,title in zip(axes,['prepare','total'],['完整准备','准备＋渲染']):
        for j,tag in enumerate(['previous_python','cpp']):
            y=[statistics.mean(v['results'][i]['summary'][tag][stage]['median_ms'] for v in rounds) for i in range(4)]
            ax.bar([i+(j-.5)*.36 for i in range(4)],y,.36,label='部署Python调度' if j==0 else '部署C++调度')
        ax.set_xticks(range(4),[labels[v['case']] for v in rounds[0]['results']],rotation=12);ax.set_ylabel('中位耗时（ms）');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('相同CUDA网络：仅迁移完整准备调度（RTX3090，有并发任务）');fig.tight_layout();fig.savefig(folder/'cpp_prepare_speed.png',dpi=180);fig.savefig(folder/'cpp_prepare_speed.pdf');plt.close(fig)
    print('REPORT_CREATED',flush=True)


if __name__=='__main__':main()
