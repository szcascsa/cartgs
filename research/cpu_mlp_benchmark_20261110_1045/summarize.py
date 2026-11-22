"""生成CPU实测表及中文口径说明；不把MLP时间当作完整渲染。"""
from pathlib import Path
import json,hashlib
import pandas as pd

HERE=Path(__file__).resolve().parent;root=HERE/'reports/cpu_formal01'
summary=json.loads((root/'SUMMARY.json').read_text());assert summary['passed'] and not summary['cuda_initialized']
environment=json.loads((root/'ENVIRONMENT.json').read_text())
data=pd.read_csv(root/'mlp_samples.csv');prepare=pd.read_csv(root/'cpu_prepare_samples.csv')
s=data.groupby(['threads','points','variant']).agg(median_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),samples=('ms','size')).reset_index()
r=data.groupby(['threads','points','variant','round']).ms.median().reset_index()
span=r.groupby(['threads','points','variant']).ms.agg(round_min_ms='min',round_max_ms='max').reset_index();s=s.merge(span,on=['threads','points','variant'])
s.to_csv(HERE/'reports/CPU_MLP_RESULTS.csv',index=False)
p=prepare.groupby(['threads','ratio']).agg(median_ms=('ms','median'),p95_ms=('ms',lambda x:x.quantile(.95)),
    candidate_N_median=('candidate_N','median'),candidate_N_min=('candidate_N','min'),candidate_N_max=('candidate_N','max'),
    K_median=('K','median'),K_min=('K','min'),K_max=('K','max'),samples=('ms','size')).reset_index()
p.to_csv(HERE/'reports/CPU_PREPARE_RESULTS.csv',index=False)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
text='''# 当前 MLP 的 CPU 推理速度实测

实测设备为服务器 Intel Core i5-9600K，6个物理核、6线程；PyTorch 2.3.1+cu118 的CPU FP32/MKL路径。CUDA_VISIBLE_DEVICES为空，加载、输入和网络均在CPU，结束时CUDA未初始化。没有修改网络、权重或生产配置。

使用Replica office2 RGB-D真实checkpoint及其最终蒸馏学生，地图43,197点。Selector为19→32→32→2，1,762参数；原Compensator为Residual48、64/64共享主干与四个两层Head，25,742参数；学生为32/32主干与四个直接Head，3,534参数。后两者均保留ratio3和view8投影。参数量不包含Master、Anchor Feature表和索引。

## 六线程稳态MLP前向

每条件两轮、共120样本、3份来自真实Gaussian与相机的输入，轮换网络次序；第二轮反转点数顺序。下表为中位/p95，单位ms。点数B是各组件的实际批量大小，不是一次单点调用乘以B。

'''
rows=[]
for B in [1,100,500,1000,5000,10000,40000]:
    row=[str(B)]
    for name in ['selector','teacher','student']:
        value=s[(s.threads==6)&(s.points==B)&(s.variant==name)].iloc[0]
        row.append(f'{value.median_ms:.4f} / {value.p95_ms:.4f}')
    rows.append(row)
text+=table(['输入点数B','Selector','原Compensator','蒸馏Compensator'],rows)
text+='''
Selector在部署时处理所有候选点M，Compensator只处理选中点K。因此同一行的Selector和Compensator时间不能直接当成一次实际推理总时间相加。

这项计时包含MLP线性层、ReLU、ratio/view投影、输出Head及cat，Selector含softmax，补偿含XYZ参考尺度乘法。输入已经组装完成；不包含Feature查询、候选筛选、排序选点、相机构造、磁盘I/O、渲染、CPU/GPU数据搬运。没有将GPU专用融合Kernel当成CPU实现。输入组装前后的Teacher/Selector在单线程与原CPU函数逐值核对通过，学生采用其原有CPU forward；这里未做CPU对GPU图像质量验证。

## 线程数影响

以下为中位ms，全部线程数和全部规模的p95及两轮范围在CPU_MLP_RESULTS.csv。多线程收益取决于批量大小，不能用理论核心数线性换算。

'''
rows=[]
for threads in [1,2,6]:
    for B in [1000,10000,40000]:
        row=[str(threads),str(B)]
        for name in ['selector','teacher','student']:
            row.append(f'{s[(s.threads==threads)&(s.points==B)&(s.variant==name)].iloc[0].median_ms:.4f}')
        rows.append(row)
text+=table(['线程','B','Selector','原Compensator','学生Compensator'],rows)
text+='''
## 真实CPU准备流程

额外使用现有FixedBudgetRenderer的CPU PyTorch路径，测名义Anchor筛选、真实视角编码、候选Gather、Selector、严格K、Residual48 Feature查询、Compensator、解码及渲染输入打包。每次重算，没有复用同帧Mask/Feature；使用3个真实保存相机、两轮、每条件72次。仅原Teacher，包含状态/有限值检查，不包括相机构造、Rasterizer或CPU/GPU拷贝。

'''
rows=[]
for value in p.itertuples(index=False):
    rows.append([str(value.threads),str(value.ratio),f'{int(value.candidate_N_median)} [{int(value.candidate_N_min)},{int(value.candidate_N_max)}]',
        f'{int(value.K_median)} [{int(value.K_min)},{int(value.K_max)}]',f'{value.median_ms:.3f}',f'{value.p95_ms:.3f}'])
text+=table(['线程','Ratio','候选M中位[最小,最大]','选中K中位[最小,最大]','准备中位ms','准备p95 ms'],rows)
text+='''
CPU准备流程明显慢于单独MLP。上述数据不等于完整CPU渲染帧率，也不能直接证明把当前GPU网络移到CPU会更快；数据位置与跨设备同步需要单独测量。

## 复现与完整性

真实权重、输入PLY及所用冻结Python源码的SHA在PROTECTED_BEFORE/AFTER.json，全部一致。共7,560个MLP样本和648个CPU准备样本。没有训练、保存新权重、修改共享环境、commit或push。

原始记录、CPU/MKL配置和模型路径见reports/cpu_formal01；CPU_MLP_RESULTS.csv和CPU_PREPARE_RESULTS.csv为汇总。未锁频、未绑核；开始时检查CPU后台负载，保留p95和两轮范围，不将本机结果外推到其他CPU或移动设备。
'''
(HERE/'CPU_MLP_INFERENCE_REPORT.md').write_text(text,encoding='utf-8')
print(s[s.threads==6].pivot(index='points',columns='variant',values='median_ms').round(5).to_string())
print(p.round(4).to_string(index=False));print('CPU_REPORT_COMPLETE',len(text))
