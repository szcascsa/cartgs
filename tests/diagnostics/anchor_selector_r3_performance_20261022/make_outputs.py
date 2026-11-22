"""由当前审计原始证据生成中文报告中的数表和机器可读热点。"""
import csv
import hashlib
import json
from pathlib import Path
import statistics

D=Path(__file__).resolve().parent
REPO=D.parents[2]
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):
    a=sorted(v);return dict(mean=statistics.mean(a),median=statistics.median(a),p95=a[min(len(a)-1,int(.95*len(a)))],p99=a[min(len(a)-1,int(.99*len(a)))])
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

native=[]
for run in ['native1','native2']:native+=read(D/'evidence'/run/'native_stages.csv')
stage=read(D/'nsight_stage_statistics.csv');online=read(D/'online_statistics.csv')
meta=json.loads((D/'evidence/native1/native_metadata.json').read_text())
detail=json.loads((D/'evidence/fixture_details.json').read_text())
cases=['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']
labels=['Full','1%','5%','10%','15%']
native_rows=[]
for case,label in zip(cases,labels):
    r=[x for x in native if x['case']==case and x['stage']=='adam'];s=stats([float(x['cycle_wall_ms']) for x in r])
    kernels=sum(float(x['kernel_launches_per_iteration']) for x in stage if x['case']==case)
    native_rows.append([label,r[0]['K'],len(r),*[f'{s[k]:.3f}' for k in ['mean','median','p95','p99']],int(kernels)])

locations={
 'activations':('src/gaussian_model.cpp:64','getScalingActivation/getRotationActivation/getOpacityActivation','N','P1','多个elementwise和normalize；Master梯度'),
 'anchor_lookup':('improvements/selector/anchor_bank.h:323','inputs','N','P0','owner gather/offset；Feature梯度index_add'),
 'input27':('improvements/selector/gumbel_network.h:329','anchorInputs/gaussianAttributes','N','P0','cat/full/log和临时Tensor'),
 'encoder':('improvements/selector/gumbel_network.h:87','forwardSelectionFeatures shared27→32','N','P1','cuBLAS GEMM+ReLU；不可预设自写更快'),
 'selector_head':('improvements/selector/gumbel_network.h:87','forwardSelectionFeatures32→32→2','N','P1','GEMM/温度/softmax；Budget及ST梯度'),
 'sample_budget_compact':('include/selector_handoff.h:16;src/gaussian_mapper.cpp:3402','sampleSelectorST/selectorRatioLoss/nonzero','N','P0','动态K D2H同步；保持row顺序与ST'),
 'selected_gather':('include/gaussian_renderer.h:54','GaussianRenderInput::indexSelect','K','P0','六组属性gather及反向scatter'),
 'frozen_compensation_head':('improvements/selector/gumbel_network.h:109','compensationResidual baseline','K','P1','冻结C仍Forward并建立随后detach的图'),
 'r3_gather_add':('improvements/selector/gaussian_compensation_bank.h:34','forward','K','P2','读取K行；grad是N×14；不可直接换SparseAdam'),
 'attribute_transform':('improvements/selector/gumbel_network.h:121;src/gaussian_mapper.cpp:3419','applyCompensationResidual/applyAppearanceResidual','K','P0','34次launch；双normalize/quaternion及exp/sigmoid'),
 'renderer_forward':('src/gaussian_renderer.cpp:76;cuda_rasterizer/rasterizer_impl.cu:390','GaussianRenderer::render/Rasterizer::forward','K或N及tile实例','P0','已有CUDA；计数D2H与workspace分配/scan/sort'),
 'photo_budget_loss':('include/loss_utils.h:168;src/gaussian_mapper.cpp:3514','FusedSSIMMap/l1_loss','H×W','P1','已有CUDA；完整图像成本不随K线性下降'),
 'backward':('src/gaussian_rasterizer.cpp:64;cuda_rasterizer/conv.cu:1002','Renderer/SSIM/ATen autograd','N,K,H×W','P0','合并scope；包含SSIM和Renderer及大量elementwise/scatter'),
 'adam':('src/gaussian_mapper.cpp:2313;src/transform_field_cuda_optimizer.cpp:97','ensureSharedMappingOptimizer/base Adam::step','活跃叶的全部元素','P0','默认dense Adam；14活跃参数×8 kernels=112'),
}
hot=[];stage_rows=[]
for name in sorted({x['stage'] for x in native if x['case']=='lod_0.050000'}):
    r=[x for x in native if x['case']=='lod_0.050000' and x['stage']==name]
    wall=stats([float(x['host_submit_ms']) for x in r]);event=stats([float(x['cuda_elapsed_ms']) for x in r])
    ns=next(x for x in stage if x['case']=='lod_0.050000' and x['stage']==name)
    source,fn,size,priority,note=locations[name]
    stage_rows.append([name,f"{wall['mean']:.4f}",f"{event['mean']:.4f}",f"{event['median']:.4f}",f"{event['p95']:.4f}",f"{float(ns['kernel_ms_per_iteration']):.4f}",int(float(ns['kernel_launches_per_iteration']))])
    hot.append(dict(module=name,source=source,function=fn,measurement_scope='原生冻结checkpoint5%重放；不是完整Mapping占比',frequency='每LoD；Full仅对应子集',
        N=15891,K=799,size=size,execution='C++ LibTorch/ATen/CUDA',samples=len(r),host_mean_ms=wall['mean'],event_mean_ms=event['mean'],
        event_median_ms=event['median'],event_p95_ms=event['p95'],event_p99_ms=event['p99'],kernel_ms=float(ns['kernel_ms_per_iteration']),
        launches=float(ns['kernel_launches_per_iteration']),mapping_host_share_percent='NA',allocation_sync=note,
        forward_backward='Backward和Adam独立列；其余为Forward',cuda_feasible='是或已有CUDA',fusion_feasible='见opportunities',parallel_feasible='受共享写与default stream限制',
        complexity='见候选接口',correctness_risk='Adam/非确定性/集合边界需独立门控',priority=priority,environment='RTX3090共享GPU；无独占验收'))
total=next(float(x['mean']) for x in online if x['run']=='trace1' and x['phase']=='all' and x['stage']=='total' and x['scope']=='per_mapping_including_zero_calls')
for name in ['operations','lod','selector_forward','lod_render','lod_backward','full_render','full_backward','densify','optimizer','gaussian_optimizer','lod_optimizer']:
    r=next(x for x in online if x['run']=='trace1' and x['phase']=='all' and x['stage']==name and x['scope']=='per_mapping_including_zero_calls')
    hot.append(dict(module='online_'+name,source='src/gaussian_mapper.cpp:1087/1524/3241',function='MappingProfiler '+name,
        measurement_scope='真实SLAM每Mapping CPU wall；lod含selector/render/backward，禁止重叠相加',frequency='实际4410 Mapping',N='动态',K='动态Cached',size='实时地图',execution='生产C++线程及GPU',samples=r['samples'],
        host_mean_ms=r['mean'],event_mean_ms='NA',event_median_ms='NA',event_p95_ms='NA',event_p99_ms='NA',kernel_ms='NA',launches='NA',
        mapping_host_share_percent=100*float(r['mean'])/total,allocation_sync='API wall可能归属之前排队GPU工作；不是kernel纯时间',forward_backward=name,
        cuda_feasible='按细分节点',fusion_feasible='按细分节点',parallel_feasible='按DAG',complexity='按细分节点',correctness_risk='实际训练步数与拓扑随异步SLAM变化',priority='P0' if name in ['lod','lod_backward','optimizer'] else 'P1',environment='同一共享GPU负载'))
write(REPO/'performance_hotspots.csv',hot)

opportunities=[
 ('K1','P0','多Tensor dense Adam融合','112 launches/LoD；Full48','保留epsilon/bias correction/gradNone/每叶step及全人口moment','高','参数/m/v/step→原位更新','先固定梯度Adam fixture；再短程质量','减少elementwise launch和中间Tensor；无实测加速承诺'),
 ('K2','P0','选后属性变换Forward与VJP融合','34 launches Forward；Backward182的部分','保持两次normalize、raw quaternion、scale双路径及opacity ST','中','K行base+14D→紧凑属性及梯度','属性/梯度fixture；q符号和极小范数','减少临时Tensor与反向填零/加法'),
 ('K3','P0','Anchor validation与Full窗口统计归并','assert实测约0.5ms；每Full调用','所有检查仍执行；FP64 opacity统计不能改FP32','中','owner/levels/features/visible/opacity→状态与错误flag','分母/访问次数/owner容量与错误fixture','减少多次item同步及重复gather'),
 ('K4','P0','Anchor Feature+Offset+属性输入pack','lookup5+pack4 launches','XYZ与激活属性仍接收原梯度；Feature按owner求和','中','N行attrs+owner+A16→N27','非连续DC和重复owner梯度','减少9次launch和N行临时张量'),
 ('K5','P0','概率/ST/预算与稳定compaction','18 launches并有D2H','uniform<p；p-p.detach；Budget分母和abs零点；stable rows','中高','N概率/uniform→mask/Budget/indices/K','选集/K精确，临界值、K0、全选','减少扫描临时量；当前Renderer仍需CPU K'),
 ('F1','P1','Selected gather+R3+属性变换拼接','6+2+34 launches','R3训练单表；插值推理两表；dense语义不变','中高','Gaussian SoA/ids/残差→Renderer SoA','含Master/表梯度及插值fixture','可合并K行载入并降低临时分配'),
 ('F2','P1','Linear+bias+ReLU epilogue','Encoder2；Selector5；现有cuBLAS','容差与ReLU临界；不得丢bias/改维度','中','N27,W→N32或N2','cuBLAS/Lt/custom按N比较','优先试现有GEMM epilogue，不能预设custom胜出'),
 ('F3','P1','冻结C使用NoGrad及版本化常零快路','4 launches；真实本checkpointC精确0','导入非零C不能跳过；保留原输出','低中','selected h→delta0','零C与非零C checkpoint','消除无用图；只有经确认常零时才能跳过GEMM'),
 ('R1','P0','Renderer workspace容量复用及计数路径','每LoD26 kernels；多次D2H','不得改排序/可见性/桶/透明度早停；必须stream正确','高','K属性+camera→render/context','Renderer原生重复与退化场景fixture','降低分配及同步；尚无候选实现收益'),
 ('R2','P1','SSIM backward本地profile与等价优化','5% native Nsight该单kernel约0.80ms','相同SSIM常数/窗口/边界/归一化','高','image/gt/dmap→dimage','图像/梯度fixture','已CUDA；需NCU后再判断带宽/占用限制'),
 ('T1','P1','Topology一次索引重排多Tensor搬运','事件densify均值33至35ms','同步迁移六组参数/R3四表/m/v/UID/Anchor和元数据','高','keep/parent maps→新状态','append/clone/split/prune/reorder生命周期','降低每事件多次全量cat/gather'),
 ('T2','P1','增量CPU空间桶与批量量化','noop prune rebuild约2ms','同层key桶/稳定UID/碰撞顺序与容量必须一致','中高','CPU keys/变更集→查询桶','迁移碰撞fixture','先避免无变化重建；GPU hash优先级更低'),
 ('T3','P1','按owner分段排序/容量分配','CPU stable_sort(N)及D2H','梯度tie按原row排序；split净增1；最大10','高','gradient/owner/counts→parent indices','边界NaN/tie及split父子fixture','需实际事件子段计时后再CUDA化'),
 ('A1','P1','Keyframe预取与Pinned H2D','源码存在CPU clone/blocking to和CV upload','保持入队/处理顺序、buffer寿命与相机版本','中','CPU图像→GPU图像及event','same-frame顺序与snapshot一致','重叠CPU解码/传输与当前Mapping'),
 ('A2','P1','CPU Anchor只读计划与GPU重放重叠','拓扑事件33至35ms；未分解CPU可重叠部分','版本化几何snapshot，应用前校验；禁止共享写并行','高','snapshot/version→待提交计划','stale-plan拒绝；相同拓扑结果','只有足够CPU计算量且snapshot成本低才获益'),
 ('A3','P1','固定snapshot多ratio推理批处理','N/owner/基础属性可复用','不能跨Master训练step缓存；共享头含log ratio','中','snapshot+ratios→各集合与渲染属性','每ratio同选集/K与属性','可缓存ratio无关input或first GEMM子项'),
 ('A4','P2','多ratio训练多Stream','没有安全生产重叠证据','Master/Feature/Encoder/Adam共享写；default-stream kernels','高','目前禁止直接并行step','先stream-aware Renderer与显式reduce','可能SM争用/显存增加；不是推荐首项'),
 ('G1','P1','固定N子图CUDA Graph','418 launch/LoD但当前动态K与D2H阻止整图','按拓扑epoch和容量bucket重新捕获；不改集合','高','固定地址和shape子图','图重放/拓扑失效/随机数fixture','先优化可捕获MLP或dense Adam子图'),
 ('S1','P2','R3改稀疏Adam','gather仅2 launch不是热点','改变未选中行moment衰减和参数动量，不等价','高','不作为本轮语义保持方案','必须另立算法实验','拒绝无论证直接替换SparseAdam'),
 ('S2','P2','全量Anchor GPU hash重写','每步Selector不查hash；只事件CPU lookup','保持量化floor/层级/碰撞桶','高','事件候选→空间桶','负坐标/边界/闭环碰撞','低频且更大热点已确认，暂不做'),
]
write(REPO/'optimization_opportunities.csv',[dict(id=i,priority=p,module=m,evidence=e,semantic_constraint=s,complexity=c,interface=io,verification=v,potential=b,implemented=False) for i,p,m,e,s,c,io,v,b in opportunities])

online_rows=[]
for run in ['trace1','untraced','trace2']:
    runtime=next(x for x in json.loads((D/'evidence/online_runs.json').read_text()) if x['name']==run)
    saved=json.loads(next((D/'evidence'/run).glob('*_shutdown/ply/mapping_stage_state.json')).read_text())
    on=saved['online'];tail=saved['tail'];
    row=[run,f"{runtime['wall_s']:.2f}",on['mappings'],tail['mappings']]
    if run!='untraced':
        for phase in ['all','0','1']:
            x=next(x for x in online if x['run']==run and x['phase']==phase and x['stage']=='total' and x['scope']=='per_mapping_including_zero_calls')
            row.append(f"{float(x['mean']):.3f} / {float(x['p95']):.3f}")
    else:row+=['日志均值见原始training.log','同左','同左']
    online_rows.append(row)
shares=[]
for name in ['operations','full_render','full_loss','full_backward','lod','optimizer','gaussian_optimizer','lod_optimizer','densify']:
    x=next(x for x in online if x['run']=='trace1' and x['phase']=='all' and x['stage']==name and x['scope']=='per_mapping_including_zero_calls')
    shares.append([name,f"{float(x['mean']):.4f}",f"{100*float(x['mean'])/total:.2f}%"])
sub=[]
for name in ['selector_forward','lod_render','lod_loss','lod_backward']:
    x=next(x for x in online if x['run']=='trace1' and x['phase']=='all' and x['stage']==name and x['scope']=='per_mapping_including_zero_calls')
    sub.append([name,f"{float(x['mean']):.4f}",f"{100*float(x['mean'])/total:.2f}%"])
events=[]
for run in ['trace1','trace2']:
    for phase in ['topology_event','full_stats_without_topology']:
        x=next(x for x in online if x['run']==run and x['phase']==phase and x['stage']=='densify' and x['scope']=='actual_source_event')
        events.append([run,phase,x['samples'],*[f"{float(x[k]):.3f}" for k in ['mean','median','p95','p99']]])
topo=[]
for run in ['native1','native2']:
    r=read(D/'evidence'/run/'topology_stages.csv')
    for name in sorted({x['operation'] for x in r}):
        s=stats([float(x['completed_ms']) for x in r if x['operation']==name]);topo.append([run,name,*[f'{s[k]:.4f}' for k in ['mean','median','p95']]])

replacements={
 '@@ONLINE_TABLE@@':table(['运行','进程总wall秒','Online步','Tail步','整体mean/p95 ms','Online mean/p95 ms','Tail mean/p95 ms'],online_rows),
 '@@SHARES@@':table(['生产scope','每Mapping host ms','整体host占比'],shares),
 '@@LOD_SUB@@':table(['LOD内嵌scope','每Mapping host ms','整体host占比'],sub),
 '@@NATIVE_TABLE@@':table(['固定状态分支','实际K','测量次数','wall mean ms','median','p95','p99','Nsight launches/步'],native_rows),
 '@@STAGES@@':table(['5% scope','host提交mean','Event mean','Event median','Event p95','Nsight kernel纯时间/步','launches/步'],stage_rows),
 '@@EVENTS@@':table(['运行','事件scope','次数','host mean ms','median','p95','p99'],events),
 '@@TOPO@@':table(['重复','原语','完成mean ms','median','p95'],topo),
 '@@MEMBERS@@':table(['成员数','Anchor数'],[[i,detail['members'][str(i)]] for i in range(1,11)]),
 '@@LAYERS@@':table(['层级','Anchor数'],[[i,detail['levels'][str(i)]] for i in range(3)]),
 '@@STRESS@@':table(['N','K','ratio','mean ms','median','p95','峰值allocated MiB'],[[x['N'],x['K'],x['ratio'],*[f'{float(x[k]):.3f}' for k in ['mean_ms','median_ms','p95_ms']],f"{int(x['peak_allocated_bytes'])/2**20:.2f}"] for x in read(D/'evidence/scale_stress.csv') if x['ratio']=='0.05']),
}
online_profile=json.loads((D/'evidence_online/online_profile_summary.json').read_text())
ms=online_profile['mapping_state'];anchor=online_profile['anchor_state']
replacements['@@ONLINE_PROFILE@@']=table(['项目','实际记录'],[
    ['Online / Tail Mapping步数',str(ms['online']['mappings'])+' / '+str(ms['tail']['mappings'])],
    ['R3补偿Forward次数',anchor['compensation_forwards']],
    ['捕获的整进程CUDA kernel次数',online_profile['kernel_launches']],
    ['kernel duration总和ms（可重叠，不是进程wall）',f"{online_profile['kernel_ms_sum']:.3f}"],
    ['实际出现的GPU stream',', '.join(str(x['streamId']) for x in online_profile['streams'])],
    ['最终Gaussian / Anchor',str(anchor['gaussian_count'])+' / '+str(anchor['anchor_count'])],
    ['是否可作同轨迹无Profiler性能对照','否：更新次数、拓扑、SLAM时序与GPU负载不同']])
text=(D/'report_template.md').read_text(encoding='utf-8')
for k,v in replacements.items():text=text.replace(k,v)
assert '@@' not in text
(REPO/'ANCHOR_SELECTOR_R3_PERFORMANCE_AUDIT.md').write_text(text,encoding='utf-8')
print('GENERATED report and machine readable hotspots/opportunities')
