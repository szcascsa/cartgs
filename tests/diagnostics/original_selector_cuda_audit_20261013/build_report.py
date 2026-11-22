"""把已完成的测量表写入中文报告；不运行训练、不修改生产源码。"""
import csv
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

D = Path(__file__).resolve().parent
def js(name):
    return json.loads((D/name).read_text(encoding='utf-8'))
def rows(name):
    return list(csv.DictReader((D/name).open(encoding='utf-8')))
def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(v).replace('|', '\\|') for v in row)+' |' for row in data])
def f(value):
    return f'{float(value):.3f}'
def q(value):
    if not value:
        return '缺失'
    return ' / '.join(f'{value[k]:.1f}'.removesuffix('.0') for k in ['min','median','p90','p95','max'])

inventory=js('inventory.json'); verification=js('source_verification.json')
parameters=js('parameter_shapes.json'); ns=js('native_nsys_summary.json')
ops=js('operator_breakdown.json')['operators']; alloc=js('tensor_allocations.json')
baseline=rows('native_benchmark.csv'); isolated=rows('native_isolated_costs.csv')
optimizer=rows('native_optimizer.csv'); gradients=rows('gradient_paths.csv')
workload=js('workload_quantiles.json')

sections={}
sections['ENVIRONMENT']=table(['项目','实测值'],[
    ['GPU/driver','RTX 3090 24GB / 535.230.02'],
    ['CUDA/nvcc','11.8 / 11.8.89'],
    ['Python/Torch','3.11.16 / 2.3.1+cu118；CXX11 ABI=false'],
    ['Native LibTorch','third_party/libtorch-cu118/libtorch；动态链接该目录的 libtorch/libc10'],
    ['GCC / Nsight','9.4.0 / Nsight Systems 2022.4.2.1'],
    ['生产构建目录','build-cuda118-sm86-gcc9；CUDA_ARCHITECTURES=86'],
    ['诊断构建目录','D/build；Release/-O3/C++17；四个独立可执行程序'],
    ['运行环境变量','CUDA_LAUNCH_BLOCKING、CUDA_VISIBLE_DEVICES 初始均未设置'],
    ['GPU状态','初始 nvidia-smi: 687MiB used，SM1755MHz；没有锁频或并发GPU负载全程监测'],
    ['存储','最后 df: / 分区868G，约11G可用；诊断约100MB，不删除已有结果'],
])
provenance=[['源码清单',f"{len(verification['source_hashes'])} 文件；source_hashes_before.json / source_verification.json"]]
before=js('source_hashes_before.json')
for name in ['improvements/selector/gumbel_network.h','src/gaussian_mapper.cpp','CMakeLists.txt']:
    provenance.append([name, '`'+before[name]+'`'])
for name,item in verification['binary_hashes'].items():
    if name.startswith('lib/'):
        t=datetime.fromtimestamp(item['mtime'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +0800')
        provenance.append([name, f"SHA256 `{item['sha256']}`；mtime {t}"])
provenance.append(['selector.pt','`'+verification['checkpoint_hashes']['selector.pt']+'`'])
provenance.append(['point_cloud / metadata','完整 PLY SHA256 与其余checkpoint/diagnostic binary hashes见 source_verification.json'])
provenance.append(['本地工作区','branch/HEAD/status单列于 local_workspace.json；原有TF/config改动与build zip保留'])
sections['PROVENANCE']=table(['证据','值'],provenance)
sections['PARAMETERS']=table(['Parameter','shape','stride','numel'],[
    [p['name'],str(p['shape']),str(p['stride']),p['numel']] for p in parameters])

gtable=[]
for name in dict.fromkeys(r['test'] for r in gradients):
    subset=[r for r in gradients if r['test']==name]
    counts=[]
    for module in ['selector','gaussian','tf']:
        block=[r for r in subset if r['module']==module]
        counts.append(str(sum(r['grad_defined']=='1' and float(r['norm'])>0 for r in block))+'/'+str(len(block)))
    finite=all(r['finite']=='1' for r in subset if r['grad_defined']=='1')
    gtable.append([name,*counts,'通过' if finite else '失败'])
sections['GRADIENTS']=table(['fresh graph test','Selector 非零张量','GS 非零张量','TF 非零张量','finite'],gtable)

sections['CHECKPOINT_COUNTS']=table(['ratio','M/network','Protection actual','Mature selected','K/selected','T/target','实际/目标'],[
    [c['ratio'],inventory['M'],c['protection_actual'],c['mature_actual'],c['active_gaussians'],f(c['target_total_gaussians']),f(c['actual_vs_target'])] for c in inventory['counts']])
wt=[]
for path, ratios in workload['per_source'].items():
    if 'protection_recheck' not in path and 'chcl-validation' not in path:
        continue
    label='正常 protection_recheck' if 'protection_recheck' in path else 'Rotation '+path.split('/')[-2]
    for ratio, data in ratios.items():
        wt.append([label,ratio,data.get('active_gaussians',{}).get('samples',0),q(data.get('total_gaussians')),q(data.get('active_gaussians')),q(data.get('tf'))])
sections['WORKLOADS']='每格顺序 **min / median / p90 / p95 / max**；count=日志行数。\n\n'+table(
    ['source','ratio','count','Ntotal','K/selected','TF J'],wt)
sections['WORKLOADS']+='\n\n合并来源仅用于规模覆盖（同一顺序），M缺失不插值：\n\n'+table(
    ['ratio','Ntotal','network M（样本数）','K/selected','protected population（样本数）','TF J'],[
        [r,q(v.get('total_gaussians')),q(v.get('N_network'))+f" ({v.get('N_network',{}).get('samples',0)})",
         q(v.get('active_gaussians')),q(v.get('protected_population'))+f" ({v.get('protected_population',{}).get('samples',0)})",q(v.get('tf'))]
        for r,v in workload['ratios'].items()])

bt=[]
for n in sorted({int(r['N_total']) for r in baseline}):
    for mode in ['cpp_inference','caller_train']:
        selected=[r for r in baseline if int(r['N_total'])==n and r['mode']==mode and r['ratio']=='0.05' and r['stage']=='total']
        ev=next(r for r in selected if r['metric']=='cuda_event_ms'); wall=next(r for r in selected if r['metric']=='wall_ms')
        bt.append([mode,n,ev['N_network'],f(ev['mean']),f(ev['p50']),f(ev['p95']),f(wall['mean']),f(wall['p50']),f(wall['p95'])])
sections['BASELINE']='r=.05，单位ms：\n\n'+table(['mode','Ntotal','M','event mean','p50','p95','wall mean','p50','p95'],bt)
it=[]
for n in [1000,10000,52181,108688,219969]:
    for stage in ['inference_network','training_forward','training_backward','feature_activations','selection_mask_index','gi_teacher','protection_budget']:
        selected=[r for r in isolated if int(r['N'])==n and r['ratio']=='0.05' and r['stage']==stage]
        ev=next(r for r in selected if r['metric']=='cuda_event_ms'); wall=next(r for r in selected if r['metric']=='wall_ms')
        it.append([n,stage,f(ev['mean']),f(ev['p50']),f(ev['p95']),f(wall['mean']),f(wall['p50']),f(wall['p95'])])
sections['ISOLATED']=table(['N/fixture rows','stage','event mean','p50','p95','wall mean','p50','p95'],it)
ot=[]
for mode in dict.fromkeys(r['mode'] for r in optimizer):
    ev=next(r for r in optimizer if r['mode']==mode and r['metric']=='cuda_event_ms')
    wall=next(r for r in optimizer if r['mode']==mode and r['metric']=='wall_ms')
    ot.append([mode,f(ev['mean']),f(ev['p50']),f(ev['p95']),f(wall['mean']),f(wall['p50']),f(wall['p95'])])
sections['OPTIMIZER']=table(['optimizer stage','event mean','p50','p95','wall mean','p50','p95'],ot)

active=sum(v['kernel_active_ms'] for k,v in ns['stages'].items() if k!='outside')
nt=[]
for stage,v in ns['stages'].items():
    if stage=='outside': continue
    spans=v['per_iteration_spans']; mean=lambda key:sum(s[key] for s in spans)/len(spans) if spans else 0
    nt.append([stage,int(v['kernels_per_iteration']),f(v['kernel_active_ms']),f(v['kernel_active_ms']/active*100)+'%',f(v['nvtx_host_ms']),f(mean('span_ms')),f(mean('gap_ms'))])
nt.append(['合计',int(ns['total_iteration_kernel_count']/ns['captured_iterations']),f(active),'100%','不相加','不相加','不相加'])
sections['NSIGHT']=table(['stage','kernels/iter','kernel active ms','active占比','NVTX host ms','device span ms','span gap ms'],nt)
sections['SYNCHRONIZATION']=table(['API（整个capture）','次数','累计CPU ms'],[
    [name,v['count'],f(v['cpu_ms'])] for name,v in ns['runtime_apis'].items()])
sections['SYNCHRONIZATION']+=f"\n\n相邻launch gap mean={ns['launch_gap_mean_ms']:.6f}ms，max={ns['launch_gap_max_ms']:.6f}ms。D2H=3/iter，timed D2D=2/iter，outside restore D2D=18/iter；memset=11/iter。API次数是capture总计，不都属于timed stage。"
names=['aten::addmm','aten::mm','aten::relu','aten::threshold_backward','aten::softmax','aten::_softmax','aten::exponential_',
       'aten::log','aten::max','aten::scatter_','aten::nonzero','aten::where','aten::index_select','aten::index_copy',
       'aten::index','aten::masked_select','aten::cat','aten::stack','aten::clone','aten::contiguous','aten::expand','aten::repeat',
       'aten::sum','aten::_to_copy','aten::addcdiv_','aten::topk','aten::sort','aten::kthvalue']
byname={v['name']:v for v in ops}
sections['OPERATORS']=table(['operator','calls/iter','kernels/iter','active ms/iter','CPU inclusive ms/iter'],[
    [name,byname.get(name,{}).get('count_per_iteration',0),byname.get(name,{}).get('kernels_per_iteration',0),
     f(byname.get(name,{}).get('kernel_active_ms_per_iteration',0)),f(byname.get(name,{}).get('cpu_inclusive_ms_per_iteration',0))] for name in names])
sections['OPERATORS']+='\n\n表中kthvalue=0，因为该profiler fixture排除GI；真实GI teacher源码/独立基准确实调用kthvalue。没有调用不等于全Mapping不存在该operator。'
aliases=['ampere_sgemm_64x64_tn','ampere_sgemm_32x32_sliced1x4_nt','vectorized clamp/ReLU',
         'reduce sum','ampere_sgemm_64x64_nn','vectorized threshold backward','elementwise threshold backward',
         'ampere_sgemm_32x128_nn','CatArrayBatchedCopy','ampere_sgemm_32x128_tn',
         'cutlass_80_simt_sgemm_128x32_8x5_nn','cutlass_80_simt_sgemm_128x32_8x5_nt',
         'scalar mul AUnary','CUDAFunctor_add','indexSelectLargeIndex','scalar mul BUnary','splitKreduce',
         'binary mul','binary div','addcdiv']
sections['TOP_KERNELS']='前20个kernel按同名累计执行时间排序；模板kernel简称与完整名称对照保存在`native_nsys_summary.json`：\n\n'+table(
    ['#','kernel简称','ms/iter'],[[i+1,aliases[i],f(v['per_iteration_ms'])] for i,v in enumerate(ns['top20_kernels'])])
sections['MEMORY']=table(['allocation metric','实测'],[
    ['Native allocator requests/iter',f(alloc['allocation_requests_per_iteration'])],
    ['累计requested bytes/iter',f"{alloc['requested_bytes_per_iteration']:,.0f} B"],
    ['process peak allocated',f"{alloc['peak_allocated_bytes']:,} B / {alloc['peak_allocated_bytes']/2**20:.2f} MiB"],
    ['process peak reserved',f"{alloc['peak_reserved_bytes']:,} B / {alloc['peak_reserved_bytes']/2**20:.2f} MiB"],
])
sections['MEMORY']+='\n\n按 CUDA allocator instant event 的同 thread 最内 CPU operator 归属（排除 empty API），前12项如下；全部归属与未落入 scope 的事件保存在 tensor_allocations.json。这里的 requests 可复用 cache，不等于 cudaMalloc。\n\n'+table(
    ['operator scope','requests/iter','累计申请 MiB/iter'],[
        [v['name'],f(v['requests_per_iteration']),f(v['requested_bytes_per_iteration']/2**20)] for v in alloc['by_operator'][:12]])
fractions={'仅network forward':ns['stages']['network']['kernel_active_ms']/active,
           'network forward + backward':(ns['stages']['network']['kernel_active_ms']+ns['stages']['backward']['kernel_active_ms'])/active,
           'row selection/index/gather':sum(ns['stages'][n]['kernel_active_ms'] for n in ['gumbel','probability','row_argmax','mask','compact_indices','gather'])/active,
           'optimizer':ns['stages']['optimizer']['kernel_active_ms']/active}
sections['PREDICTIONS']=table(['理想化改良范围','实测active占比 f','该范围2×','该范围4×','该范围归零上限'],[
    [name,f(frac*100)+'%',f(1/(1-frac+frac/2))+'×',f(1/(1-frac+frac/4))+'×',f(1/(1-frac))+'×'] for name,frac in fractions.items()])
sections['PREDICTIONS']+='\n\nforward+backward行包含非纯MLP的backward，属于宽范围上限；full hot path无法全部归零，必须保留计算和caller外阶段。'
sections['MAPPING_LIMITS']=table(['历史normal窗口','Mapping wall mean ms','GI host ms','LoD host ms','整个LoD归零条件上限'],[
    [it,f(w),f(gi),f(lod),f(w/(w-lod))+'×'] for it,w,gi,lod in [(1000,28.716,17.0131,6.5231),(2000,54.0656,35.6418,10.7158),(3000,65.3385,44.7698,14.0427)]])
sections['ARTIFACTS']=table(['文件（均在D，报告在仓库根）','内容'],[
    ['ORIGINAL_SELECTOR_CUDA_AUDIT.md','本报告，本地与服务器镜像'],
    ['native_benchmark.csv / .json / native_samples.csv','无profiler基准与全部48,000样本'],
    ['native_isolated_costs.csv / .json','280独立阶段case的mean/p50/p95'],
    ['native_optimizer.csv / native_preview_validity.json','Adam/preview/zero_grad及事务validity'],
    ['native_nsys.nsys-rep / .sqlite / _summary.json','Native Nsight原始trace、stage、API、copy、top20'],
    ['native_operator_trace.json / operator_breakdown.json','Native Kineto原始operator/shape/kernel归属'],
    ['parameter_shapes.json / tensor_allocations.json','真实参数shape/stride、allocator统计'],
    ['workload_rows.json / workload_quantiles.json / mapping_gi_logs.json','源path+line、13日志的真实规模/ratio分布与Mapping阶段'],
    ['gradient_paths.csv / gradient_losses.csv / gradient_parity_and_graph.json','11条Native图路径、各tensor梯度与固定noise对照'],
    ['native_gradient_oracle.pt / native_parity.json / native_validity.csv','结构oracle、Python probability parity、baseline finite'],
    ['inventory.json / checkpoint_selector.yaml / current_fast.yaml','环境/Git真实限制、checkpoint和变化后的配置'],
    ['source_hashes_before.json / source_verification.json / local_workspace.json','生产源码及checkpoint/binarySHA与本地独立版本记录'],
    ['source_snapshot/ / complete_source_snapshot/ / *.tar.gz','冻结远端审计源码'],
    ['README.md / run*.sh / CMakeLists.txt / *.cpp / *.py','完整复现命令、诊断源码、报告生成与验证'],
    ['*.log / *.exit / build_CMakeCache.txt','真实构建、测量、检查完成证据；run/extra/final均exit0'],
])

template=(D/'report_template.md').read_text(encoding='utf-8')
for key,value in sections.items():
    template=template.replace('{{'+key+'}}',value)
assert not re.search(r'\{\{[A-Z_]+\}\}',template)
(D.parents[2]/'ORIGINAL_SELECTOR_CUDA_AUDIT.md').write_text(template,encoding='utf-8')
(D/'native_isolated_costs.json').write_text(json.dumps(dict(protocol=dict(warmup=100,measurement=500,stream='current default CUDA stream',
    limitation='Independent fixtures; backward only parameters and XYZ; do not sum stages as production caller'),rows=isolated),indent=2),encoding='utf-8')
print(json.dumps(dict(report=str(D.parents[2]/'ORIGINAL_SELECTOR_CUDA_AUDIT.md'),bytes=len(template.encode('utf-8')),active_ms=active,baseline_rows=len(baseline),isolated_rows=len(isolated))))
