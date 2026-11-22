from common import *
from collections import defaultdict
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=ROOT/'reports'
def read(name):return json.loads((R/(name+'.json')).read_text())
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in row)+'|' for row in rows])
def main():
    perf=[read('BENCHMARK_'+str(j)) for j in [1,2]]
    names={'r01':'局部1%','r70':'局部70%','changing_ratio':'逐帧变化ratio','changing_points':'逐帧变化点预算'}
    speed=[];tail=[];csvrows=[]
    for i,c in enumerate(perf[0]['results']):
        trials=[p['results'][i] for p in perf]
        med=lambda tag,stage:statistics.mean(v['summary'][tag][stage]['median_ms'] for v in trials)
        a,b=med('v3','prepare'),med('deployment','prepare');x,y=med('v3','total'),med('deployment','total')
        speed.append([names[c['case']],f'{a:.3f}',f'{b:.3f}',f'{100*(1-b/a):.1f}%',f'{x:.3f}',f'{y:.3f}',f'{100*(1-y/x):.1f}%'])
        for run,v in enumerate(trials,1):
            p=v['summary']['deployment']['prepare'];t=v['summary']['deployment']['total']
            tail.append([run,names[c['case']],f"{p['p90_ms']:.3f}",f"{p['p99_ms']:.3f}",f"{100*p['under_2ms_fraction']:.2f}%",f"{t['p99_ms']:.3f}"])
            for tag,stages in v['summary'].items():
                for stage,stats in stages.items():csvrows.append(dict(run=run,case=c['case'],backend=tag,stage=stage,**stats))
    with (R/'LATENCY_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    q=read('QUALITY');quality=[]
    for v in q['summaries']:
        quality.append([v['condition'],f"{v['base_psnr']:.6f}",f"{v['fast_psnr']:.6f}",f"{v['mean_delta']:+.2e}",f"{v['worst_delta']:+.2e}",f"{v['ids_same_fraction']*100:.1f}%"])
    comp=[]
    for v in read('COMPONENTS')['results']:
        if v['stage']!='two_networks_decode':continue
        a,b=v['summary']['v3'],v['summary']['deployment']
        comp.append([v['ratio'],v['N'],v['K'],f"{a['event']['median_ms']:.4f}",f"{b['event']['median_ms']:.4f}",f"{b['event']['p99_ms']:.4f}",f"{b['host']['median_ms']:.4f}"])
    profile={v['label']:v for v in read('PROFILE_ANALYSIS')['summary']};kernel=[]
    for ratio in ['0.01','0.1','0.7']:
        a,b=[profile[f'prepare/{tag}/{ratio}'] for tag in ['v3','deployment']]
        kernel.append([ratio,int(a['kernels']),int(b['kernels']),f"{100*(1-b['kernels']/a['kernels']):.1f}%",f"{a['kernel_us']:.1f} → {b['kernel_us']:.1f}",f"{a['syncs']:g} → {b['syncs']:g}",f"{a['dtoh']:g} → {b['dtoh']:g}"])
    p=read('PERCEPTUAL');m=read('MEMORY');identity=read('BINARY_IDENTITY');g=read('GEOMETRY')
    api=read('API_VALIDATION');isolation=read('DEPLOYMENT_ISOLATION')
    normal=read('NORMAL_UNIT_VALIDATION');cli=read('CLI_VALIDATION')
    sections=['''# 训练结束后的极致部署推理优化报告

日期：2026-11-06。服务器实验目录：`/home/prmi/zwj/experiments/deployment_point_runtime_20261106`。

## 结论与适用范围

已实现独立部署后端：Selector每Gaussian一个Warp，Compensator每Gaussian一个Block，Block内四个Warp并行运行四Head。输入读取、Residual48累加、MLP、视角／ratio投影、物理属性解码与发布融合为少数CUDA Kernel。模型权重、网络结构、训练算法保持原样；部署FP32允许FMA与归约顺序引入细小误差。

当前训练和常规推理入口没有切换；只新增`deployment_runtime.py`、两个CUDA扩展源码、`deployment_infer.py`和中文使用文档。之前试验的v4 epilogue／多Stream修改保存在独立研究目录，常规后端恢复为本轮开始时的v3版本。部署构造私有冻结快照，拒绝带梯度调用；训练期间评价仍走原路径。

本次质量主对照为已证明与原PyTorch逐bit一致的v3；另直接用未优化PyTorch检查66帧×6ratio的SSIM、LPIPS和选点。没有用训练补救部署误差，也没有重训模型。

**2 ms目标：本次已训练模型和RTX3090上的网络及完整准备达到目标；不能把RTX3090结果称为未知端侧GPU的测量。** 实际模型13298点、2790个Anchor；视角候选约4467–10822点，预算最高10000但受slot可选数限制。更大地图与其他端侧设备需要相同入口复测。

## 每帧动态输入的完整耗时

两个独立进程轮次，每条件792帧，20次预热不计入；后端顺序逐帧交错。相机、候选点数及变化ratio／点预算真实变化，无同帧Mask复用、无ratio→Anchor48结果缓存、无全局输出缓存、无CUDA Graph。部署只在加载时冻结并转置网络权重、预处理不随视角变化的Master输入。

下表为两轮中位耗时的平均值；同轮对照v3，没有把早期PyTorch→CUDA收益重复计入。准备包含候选、view、Selector、严格K排序、Compensator、解码和finite检查；准备＋渲染再包含原Rasterizer。计时从prepare调用前到完成依赖，GPU同步成本保留，相机对象构建在两后端计时外共享。
''',table(['条件','v3准备ms','部署准备ms','准备下降','v3准备＋渲染ms','部署准备＋渲染ms','完整下降'],speed),'''
变化ratio轮转.01/.03/.05/.08/.1/.2/.3/.4/.5/.6/.7；变化预算轮转1000/2000/3000/4000/5000/10000。严格K按当前视角自身候选人口计算，不把其他场景的N代入。全部原始时延和mean／median／p90／p99保存在BENCHMARK_1/2.json、LATENCY_SUMMARY.csv。

部署尾延迟如下；2 ms比例按全部792个测量帧计算，没有剔除长尾。
''',table(['轮次','条件','准备p90 ms','准备p99 ms','准备低于2ms比例','准备＋渲染p99 ms'],tail),'''
测量期间服务器有其他用户任务，负载会随时间变化，未停止无关进程。此前较重负载的同算术版本数据保存在pre_final/，与最终两轮分开报告；最终统计不与早期不同负载样本混合。Kernel减少是结构事实，纯时间变化仍含GPU争用和Host调度影响。HTTP、图像封包、客户端显示不在本轮速度范围。

## 两网络＋补偿解码分项

使用真实候选行及实际选点，视角编码、候选和排序在该微测外预备；两后端执行相同形状的两网络。包括Selector输入／score、Compensator输入和属性解码，省去finite的Host读回，仅用于定位，不等于完整准备耗时。80组交错采样，CUDA Event包含CPU提交间隙与并发GPU等待；纯Kernel时间见Nsight。
''',table(['ratio','候选N','实际K','v3 Event中位ms','部署Event中位ms','部署Event p99 ms','部署Host中位ms'],comp),'''
## Nsight Systems归因

每ratio十二个真实相机，逐帧交错；NVTX范围以CUPTI correlation归属，嵌套node不相加。原trace保存在服务器reports/PROFILE.nsys-rep与PROFILE.sqlite。
''',table(['ratio','v3准备Kernel','部署准备Kernel','减少','累计Kernel μs 原→新','同步API 原→新','DtoH 原→新'],kernel),'''
每Anchor视角编码从多次GEMM／逐元素处理融合为一次Kernel。Selector网络、输入读取和score主计算一次Kernel，另外一次Kernel清零错误状态；Compensator主干、四Head、解码发布一次Kernel，另外清零错误状态一次。不能把整套流程称为一个Kernel：视锥压缩、严格K排序以及Selector→Compensator的数据依赖仍需分阶段执行。

## 渲染质量与选点

验收门限在全轨迹结果前声明：逐条件平均PSNR损失≤0.01 dB；99%视角损失≤0.05 dB；最差视角损失≤0.1 dB。所有条件都通过。训练ratio和未见ratio逐项列出，固定预算另外比较；Full未补偿数据保留原版且通过逐bit检查。
''',table(['条件','原PSNR dB','部署PSNR dB','平均Δ dB','最差Δ dB','相同选点视角'],quality),f'''
792帧×16条件共{q['cases']}组。全部选点ID一致，严格K与候选N一致，未发现非有限值。PSNR计算采用原FP32图像和同一真值，微小Δ受到FP32指标分辨率影响，不能称为数学上零误差。

直接与未优化PyTorch比较的66帧、396组：SSIM最大绝对变化{p['max_abs_ssim_delta']:.3e}，LPIPS最大绝对变化{p['max_abs_lpips_delta']:.3e}；平均ΔSSIM={p['ssim_mean_delta']:+.3e}，平均ΔLPIPS={p['lpips_mean_delta']:+.3e}。该样本的选点一致比例{100*p['ids_same_fraction']:.1f}%。完整逐帧指标见PER_FRAME.csv，感知指标样本见PERCEPTUAL.csv。

部分像素存在接近0.015的最大绝对差值；平均图像误差远小于该孤立极值。原因可包括FP32几何微差触发Alpha阈值、tile／depth排序边界变化；本轮没有逐个定位每一个像素差异，因此只认为质量近似等价，绝不称逐像素一致或声称这些原因已全部证实。未保存渲染图像。

几何诊断另取66帧×6ratio，对照XYZ参考尺度单位、Scale倍率和旋转角长尾。如下为所有诊断中原／部署统计的最大绝对变化；原模型已有的补偿长尾保留，不把位移变小当作本轮优化收益。
''',table(['几何统计','最大配对绝对变化'],[[k,f'{v:.8g}'] for k,v in g['max_paired_absolute_change'].items()]),'''
## 部署实现与训练隔离

- Selector：每Gaussian一个Warp，连续读取转置后的共享权重；融合11D／19D输入打包、32/32主干、2类输出与固定Gumbel score。固定噪声仍按原全局Gaussian行ID读取。同分排序、eligible和严格K沿用v3稳定排序。
- Compensator：每Gaussian一个128线程Block；64/64主干共享中间激活，四个Warp分别执行XYZ、Scale、Rotation、Other。融合当前Base＋五级Residual48查询、offset3输入、ratio3和模式／view8投影、原14D输出、Scale／Quaternion／Opacity／SH DC解码和最终属性写入。
- per-Anchor视角：中心／八角点变换、方向、屏幕范围和深度编码每帧一次Kernel，无RGB-D依赖。名义Anchor视锥仍保留原GEMM和精确判断，不扩张或缩小候选池。
- FP32：没有使用FP16、BF16、TF32或fast-math；显式FMA与自定义累加顺序允许细小差异。Residual层累加保持原顺序。未改网络容量、权重、Head语义、ratio和预算规则。
- 生命周期：部署对象复制并冻结模型与Master，加载时预检非有限值；源模型后续训练／更新不会改变该快照。切换地图或checkpoint时创建新对象；同一对象串行使用。没有训练Adam或训练Autograd实现。
- 诊断／异常：每帧在Selector和补偿输出产生处汇总非有限错误并在相应依赖点读回；保留候选计数、严格K和有限值检查，未为速度跳过异常。

接口示例与实际命令行在python/DEPLOYMENT_RUNTIME.md。正常入口未导入部署模块；启动必须显式使用DeploymentRenderer或deployment_infer.py。
''',f'''
## 边界、独立性与开销

五类checkpoint（旧11D及joint schema1/2/3/4）共{api['cases']}组接口检查通过；覆盖0／1／7／1000／超额预算、ratio0–1、空候选、关闭补偿、源模型后续更新、输出所有权、非默认CUDA Stream、inference_mode、保存后重载、非法请求、非有限值拒绝。Full未补偿路径逐bit通过。部署接口在启用梯度时明确报错；模型、Master、RNG和grad未被部署调用改写。

真实独立CLI使用inference_mode完成{cli['frames']}帧、每帧2000点，未保存图片；原预算／Viewer／CLI的{normal['tests']}项既有测试通过。常规源码{isolation['normal_files']}个文件前后相同，无常规模块自动导入deployment。另有本地11个训练和正常推理关键文件前后SHA记录。

私有快照额外保留显存{m['retained_snapshot_bytes']/1024**2:.2f} MiB；100次变化输入请求的峰值增量{m['peak_extra_bytes']/1024**2:.2f} MiB。额外内存包括冻结模型、转置权重、输入预处理和CUB工作区。没有按ratio或相机增长的缓存，没有跨请求保存网络激活；底层Torch内存池的reserved不等同于活跃快照大小。
''','''## 并行候选的实际取舍

此前已编译测过两种方法，证据保存在reports/network_trials：

1. 原GEMM的Bias＋ReLU epilogue融合：224项随机层检查和8568项真实网络检查逐bit通过，网络本体有小幅收益，完整动态路径增益较小。它没有被写入训练或正常默认路径。
2. 四Head独立CUDA Stream：数学输出通过同样检查，但Event中位数通常比单Stream epilogue更慢，动态端到端收益不稳定。小Head不足以抵消Stream Event等待和调度成本。最终部署采用同Block四Warp并行，使四Head共享主干和权重访问，减少Launch，而非继续增加Stream。
3. 早期每Gaussian一个Warp的完整MLP在大批量有逐bit一致样本，小批量出现1e-6量级差异。用户随后明确允许部署误差；最终以真实渲染和既定质量门限验收，没有把该原型的bitwise失败隐藏。
4. 感知指标脚本曾与标准库profile模块重名，导入失败；已重命名脚本并完成重新测试，失败日志保留。这不是训练／CUDA数值故障。

## 接入身份、限制和下一步
''',f'''部署扩展：`{identity['path']}`，SHA256 `{identity['sha256']}`。Torch {identity['torch']}、CUDA {identity['cuda']}、{identity['device']}。BINARY_IDENTITY.json记录实际编译参数、源码SHA和v3扩展身份；RUNTIME_MAPS.txt证明实际加载位置。

服务器独立实验目录及构建目录互相隔离，不修改/home/prmi/zwj/mycartgs、realLOD或共享conda。完整性清单覆盖604个生产文件及70个模型文件，逐文件前后记录见FINAL_INTEGRITY.json与PRODUCTION／INPUTS／LEGACY_INPUTS manifest。没有commit、push或切生产默认。

本轮验收集中在fr1_xyz的已训练13k点模型。目标端侧设备型号、功耗、时钟、地图规模及驱动尚未实测，不保证所有平台达到相同毫秒数。当前构建仍由Python调用独立C++／CUDA扩展与原Rasterizer；没有宣称已生成无LibTorch的嵌入式二进制。目标设备应按自己的CUDA架构重新编译，并用deployment_infer.py和真实移动相机复测。

下一步优先在目标设备测Kernel占用和寄存器／共享内存瓶颈；然后评估大K下每Block多个Gaussian及共享权重Tile，避免小网络被重复权重读取限制。当前完整路径中稳定排序、候选计数Host依赖和Rasterizer已占较大比例；若需继续降低端到端耗时，应测这些模块，而不是仅追求把已有0.1ms量级MLP合成形式上的一个算子。任何新路径仍需相同质量门限和固定checkpoint配对验证。
''']
    (R/'DEPLOYMENT_RUNTIME_OPTIMIZATION_REPORT.md').write_text('\n\n'.join(sections))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    figs=R/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,stage,title in zip(axes,['prepare','total'],['完整准备','准备＋渲染']):
        for j,tag in enumerate(['v3','deployment']):
            vals=[statistics.mean(v['results'][i]['summary'][tag][stage]['median_ms'] for v in perf) for i in range(4)]
            ax.bar([i+(j-.5)*.36 for i in range(4)],vals,.36,label='上一轮CUDA v3' if tag=='v3' else '冻结部署版')
        ax.set_xticks(range(4),[names[v['case']] for v in perf[0]['results']],rotation=12)
        ax.axhline(2,color='#dc2626',linestyle='--',linewidth=1,label='2 ms');ax.set_title(title);ax.set_ylabel('中位耗时（ms）');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('真实动态输入：两轮中位耗时平均（RTX3090）');fig.tight_layout();fig.savefig(figs/'deployment_speed.png',dpi=180);fig.savefig(figs/'deployment_speed.pdf');plt.close(fig)
    subset=[v for v in q['summaries'] if v['condition'].startswith('r')];ratios=[float(v['condition'][1:]) for v in subset]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));axes[0].plot(ratios,[v['base_psnr'] for v in subset],'-o',label='原CUDA v3');axes[0].plot(ratios,[v['fast_psnr'] for v in subset],'--x',label='冻结部署版');axes[0].set_xlabel('当前视角预算ratio');axes[0].set_ylabel('全轨迹平均PSNR（dB）');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(ratios,[1e6*v['mean_delta'] for v in subset],'-o',label='平均ΔPSNR');axes[1].plot(ratios,[1e6*v['worst_delta'] for v in subset],'-x',label='最差视角ΔPSNR');axes[1].axhline(0,color='gray',linewidth=1);axes[1].set_xlabel('当前视角预算ratio');axes[1].set_ylabel('PSNR差异（微dB）');axes[1].legend();axes[1].grid(alpha=.2);fig.suptitle('792帧逐ratio配对质量：曲线近乎重合');fig.tight_layout();fig.savefig(figs/'deployment_quality.png',dpi=180);fig.savefig(figs/'deployment_quality.pdf');plt.close(fig)
    print('CHINESE_REPORT_CREATED',flush=True)


if __name__=='__main__':main()
