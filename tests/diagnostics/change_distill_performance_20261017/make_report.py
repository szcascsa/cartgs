"""将静态审计与已验证实测数据合成中文交付报告。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics as st

D = Path(__file__).resolve().parent
RAW = D / 'raw'
REPO = D.parents[2]


def read(name):
    return json.loads((RAW / name).read_text(encoding='utf-8'))


def csv_rows(name):
    with (RAW / name).open(encoding='utf-8') as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in rows]]) + '\n\n'


def fmt(x, digits=3):
    return f'{float(x):.{digits}f}'


rows = read('benchmark_summary.json')
fits = read('epoch_fit.json')
components = csv_rows('component_summary.csv')
training_stages = csv_rows('training_stage_summary.csv')
environment = read('environment.json')
status = read('kernel_profile_status.json')
parity = read('instrumentation_parity.json')
traces = {branch: read(branch + '_trace_summary.json') for branch in ['selector', 'tf']}


def case(branch, epoch=5, batch=8192, ratios=4, mode=0):
    return next(r for r in rows if (r['branch'], r['epochs'], r['batch_size'], r['ratio_count'], r['instrumentation_mode']) ==
                (branch, epoch, batch, ratios, mode))


def component(branch, name, epoch=5, batch=8192, ratios=4):
    key = f'{branch}_e{epoch}_b{batch}_r{ratios}_m1'
    return [r for r in components if r['case'] == key and r['component'] == branch + '/' + name]


s5, t5 = case('selector'), case('tf')
lines = ['# ChangeDistill 性能源码审计与 Profiling\n\n',
         '审计日期：2026-10-17（Asia/Hong_Kong）。对象：当前本地 `mycartgs` 冻结实现；实测机器：`prmi@172.19.163.58`，RTX 3090。\n\n',
         '**本轮没有优化生产代码。** 复用真实 Local LC 6207，分别执行 Selector／TF 的1、2、5、10、20 epochs、5种batch、1/2/4 ratios实验；每个正式性能case包含cold、2次独立warmup、10次从相同输入恢复的重复测量。所有训练loss与正式诊断评估保留。\n\n',
         f'主要结论：TF 的次线性 epoch scaling 有明确的固定开销来源：**冻结Teacher结果的几何 transport**。当前模型的TF总时间拟合为 `T(E)≈{fits["tf"]["fixed_ms"]:.2f}+E×{fits["tf"]["ms_per_epoch"]:.2f} ms`；'
         f'Selector为 `T(E)≈{fits["selector"]["fixed_ms"]:.2f}+E×{fits["selector"]["ms_per_epoch"]:.2f} ms`。固定项是回归估计，组件实测和kernel证据见后文。\n\n',
         '![Epoch、batch与ratio scaling](tests/diagnostics/change_distill_performance_20261017/plots/scaling_summary.png)\n\n',
         table(['主benchmark，B8192/R4/E5', 'CPU wall中位数/ms', 'CUDA span中位数/ms', '实际steps', '10次std/ms'],
               [[r['branch'], fmt(r['cpu_wall_ms_median']), fmt(r['cuda_elapsed_ms_median']),
                 r['selector_steps'] + r['tf_steps'], fmt(r['cpu_wall_ms_std'])] for r in [s5, t5]]),
         'CUDA span 包含stream空闲，不能解读为SM计算时间。绝对时间不同于上一轮报告不代表本轮做了优化；本轮重跑旧runner用于检查测量边界，历史机器状态的剩余差异明确保留。\n\n',
         (D / 'audit_static.md').read_text(encoding='utf-8') + '\n\n']

lines += ['## 10. Torch／LibTorch／CUDA／GPU 环境\n\n',
          table(['项', '本次记录'], [
              ['GPU', environment['gpu']], ['CUDA runtime / compiler', environment['cuda'] + ' / nvcc 11.8'],
              ['Python PyTorch', environment['torch']], ['原生benchmark LibTorch', '2.3.1，项目 third_party/libtorch-cu118/libtorch，C++11 ABI=1'],
              ['Python wheel ABI', 'ABI=0；未用此wheel链接原生benchmark'],
              ['编译', 'g++，-O3 -DNDEBUG -std=c++17，CUDA compute_86/sm_86'],
              ['驱动 / 完整信息', '535.230.02；raw/environment.json'],
              ['精度', 'FP32；原生run逐次记录allow_tf32_cublas，未更改'],
              ['Profiler', 'Nsight Systems 2022.4.2；Nsight Compute 2022.3.0；LibTorch Kineto'],
          ]),
          '完整compiler、nvcc、nvidia-smi -q、PyTorch build配置、原生编译/链接参数分别保留在 `raw/environment.json`、`raw/build_commands_all.json`，实际动态链接见 `raw/native_ldd.txt`。Python wheel与项目LibTorch版本号相同但ABI不同；最终可执行文件链接项目LibTorch，避免以Python环境信息代替实际native库。\n\n']

lines += ['## 11. Cold-start、steady-state 与 epoch scaling\n\n']
for branch in ['selector', 'tf']:
    rlist = [case(branch, epoch=e) for e in [1, 2, 5, 10, 20]]
    lines += [f'### {branch}\n\n',
              table(['epochs', 'cold/ms', 'warm mean', 'median', 'std', 'min', 'max', 'CUDA median', 'steps', 'ms/step含固定项'],
                    [[r['epochs'], fmt(r['cold_cpu_ms']), fmt(r['cpu_wall_ms_mean']), fmt(r['cpu_wall_ms_median']),
                      fmt(r['cpu_wall_ms_std']), fmt(r['cpu_wall_ms_min']), fmt(r['cpu_wall_ms_max']),
                      fmt(r['cuda_elapsed_ms_median']), r['selector_steps'] + r['tf_steps'],
                      fmt(r['ms_per_optimizer_step_including_fixed'])] for r in rlist]),
              f'最小二乘 `T(E)=fixed+E×per_epoch`：fixed={fits[branch]["fixed_ms"]:.3f} ms，'
              f'per_epoch={fits[branch]["ms_per_epoch"]:.3f} ms，R²={fits[branch]["r_squared"]:.6f}。'
              '所有重复原始值与残差保留在 all_runs.csv、epoch_fit.json；不将两个端点的拟合当作独立因果证据。\n\n']

lines += ['## 12. Why does TF timing scale sub-linearly with epochs?\n\n',
          '上一轮5→20 epochs的TF为611.684→1091.39 ms，两点直线等价于约451.782 ms固定项＋31.980 ms/epoch；Selector为503.643→1914.26 ms，等价于33.437 ms固定项＋94.041 ms/epoch。此计算只展示现象，本轮用独立组件计时和完整epoch曲线验证其来源。\n\n']
for branch in ['selector', 'tf']:
    r1, r20 = case(branch, epoch=1), case(branch, epoch=20)
    names = ['geometry_transport', 'teacher_forward', 'primary_evaluation', 'untouched_evaluation',
             'target_cache', 'backward', 'adam_step'] if branch == 'tf' else ['teacher_forward', 'primary_evaluation', 'anchor_evaluation', 'backward', 'adam_step']
    lines += [f'**{branch} 固定与变化成本（mode1，inclusive，不可直接相加）：**\n\n',
              table(['组件', 'E1 CPU/ms', 'E1 calls', 'E20 CPU/ms', 'E20 calls'],
                    [[name, fmt(component(branch, name, epoch=1)[0]['cpu_inclusive_ms']) if component(branch,name,epoch=1) else 'NA',
                      component(branch,name,epoch=1)[0]['calls'] if component(branch,name,epoch=1) else 'NA',
                      fmt(component(branch,name,epoch=20)[0]['cpu_inclusive_ms']) if component(branch,name,epoch=20) else 'NA',
                      component(branch,name,epoch=20)[0]['calls'] if component(branch,name,epoch=20) else 'NA'] for name in names])]
lines += ['TF Teacher evaluation与geometry transport的调用数随E保持不变，训练backward/Adam随E增加。144个correction groups ×20个Teacher batch/ratio miss造成大量nonzero、索引和index_copy。后续epoch复用已transport的targets，摊薄了固定项。该现象不需要用“20 epoch每步更少样本”或“跳过loss”解释。\n\n',
          'warmup能够移除首次GEMM/allocator等启动影响，但不能消除每事件都必须重新生成的Teacher cache与transport；每轮都恢复模型和新Adam，不能通过复用训练后的状态把固定项人为移出。生产timer不是纯CUDA计时，但本轮同步native total与CUDA spans都保留，固定transport也由kernel/CPU事件证实。\n\n',
          '旧runner复跑数据见 `raw/previous_runner_replication/`，历史日志见 `raw/previous_timing_raw.csv`。历史运行的CPU/GPU频率、其它任务、系统调度没有完整同时期counter；若绝对时间不同，不能把剩余差异归因为某一项warmup/allocator/JIT而无证据。本路径没有TorchScript JIT训练编译步骤。\n\n']
previous=[]
for epochs in [5,20]:
    samples={'SELECTOR':[],'TF':[]}
    for path in (RAW/'previous_runner_replication').glob(f'e{epochs}_rep*/stdout.log'):
        for module,value in re.findall(r'\[(SELECTOR|TF) CD\].*?time_ms=([0-9.]+)',path.read_text(encoding='utf-8'))[-2:]:
            samples[module].append(float(value))
    previous.append([epochs,fmt(st.median(samples['SELECTOR'])),fmt(st.median(samples['TF'])),len(samples['TF'])])
lines += ['**同机重跑上一轮原始可执行文件（分支内部timer，三次中位数）：**\n\n',
          table(['epochs','Selector/ms','TF/ms','repeats'],previous),
          '旧runner现在也显示更低的Selector绝对时间，与本轮独立runner扣除coordinator前置开销后的量级一致。因此不能把历史503/1914 ms到本轮226/825 ms的差值称为代码加速。TF固定transport与约31–32 ms/epoch的规律在两套runner中一致。\n\n']

lines += ['## 13. 细粒度组件时间：频率、调用数与扰动\n\n',
          '下表为E5/B8192/R4的独立event。CPU和CUDA均为inclusive；网络forward嵌套在evaluation/cache中，不能跨层求和。CUDA Event span可能比CPU enqueue时间长，也可能包含CPU空闲；实际kernel GPU time另见operator表。`raw/component_summary.csv`同时保存exclusive时间与全部scaling case。\n\n']
partition_rows=csv_rows('cpu_partition.csv')
for branch in ['selector','tf']:
    selected=[r for r in partition_rows if r['case']==f'{branch}_e5_b8192_r4_m1']
    lines += [f'**{branch} CPU wall互斥分解（mode1，含队列等待和计时扰动）：**\n\n',
              table(['阶段','CPU wall/ms','占比/%'],[[r['phase'],fmt(r['cpu_exclusive_ms']),fmt(r['percentage_partition'])] for r in selected]),
              f'互斥范围合计{sum(float(r["cpu_exclusive_ms"]) for r in selected):.3f} ms。'
              '该表可求和；后面的inclusive组件表用于看调用粒度，不能重复求和。\n\n']
frequency = {
    'teacher_forward': 'EVENT-ONCE population，分batch×ratio',
    'geometry_transport': 'EVENT-ONCE target生成；内部144组',
    'target_cache': '每BATCH×RATIO；含命中与miss',
    'cache_missing_indices': '每BATCH×RATIO', 'cache_miss_evaluate': 'EVENT-ONCE population',
    'cache_storage_allocate': 'EVENT-ONCE×R', 'cache_target_gather': '每BATCH×RATIO',
    'anchor_sampling': 'EPOCH-ONCE', 'shuffle_cpu_h2d_gather': 'EPOCH-ONCE，anchors/primary/audit各有调用',
    'permutation_h2d': '每shuffle', 'shuffle_index_gather': '每shuffle',
    'primary_evaluation': 'EVENT-ONCE baseline + EPOCH-ONCE', 'anchor_evaluation': '同primary evaluation',
    'untouched_evaluation': 'EVENT-ONCE pre/post两次', 'student_forward': '每BATCH×RATIO，含正式评估',
    'gaussian_gather': '每BATCH×RATIO', 'loss_construction': '每BATCH×RATIO',
    'backward': 'OPTIMIZER-STEP', 'adam_step': 'OPTIMIZER-STEP', 'zero_grad': 'BATCH-ONCE+结束一次',
    'finite_check_d2h': 'BATCH-ONCE', 'positive_check_d2h': 'BATCH-ONCE',
    'outside_identity_filter': 'EVENT-ONCE外部过滤', 'network_inside_filter': '每次TF forward',
    'hexplane_query': '每次TF forward', 'tf_mlp': '每次TF forward',
    'hexplane_backward': 'RATIO-ONCE/每step', 'tf_mlp_backward': 'RATIO-ONCE/每step',
    'output_backward': 'RATIO-ONCE/每step', 'selector_network_backward': 'RATIO×strata/每step',
    'adam_construction': 'EVENT-ONCE', 'aabb_diagonal_d2h': 'EVENT-ONCE',
    'primary_batch_view': 'BATCH-ONCE', 'anchor_batch_view': 'BATCH-ONCE',
}
for branch in ['selector', 'tf']:
    names = ['teacher_forward', 'geometry_transport', 'target_cache', 'cache_missing_indices',
             'cache_miss_evaluate', 'cache_storage_allocate', 'cache_target_gather', 'anchor_sampling',
             'shuffle_cpu_h2d_gather', 'permutation_h2d', 'shuffle_index_gather', 'primary_batch_view',
             'anchor_batch_view', 'primary_evaluation', 'anchor_evaluation', 'untouched_evaluation',
             'outside_identity_filter', 'network_inside_filter', 'student_forward', 'gaussian_gather',
             'hexplane_query', 'tf_mlp', 'loss_construction', 'backward', 'selector_network_backward',
             'tf_mlp_backward', 'hexplane_backward', 'output_backward', 'adam_construction', 'adam_step',
             'zero_grad', 'finite_check_d2h', 'positive_check_d2h', 'aabb_diagonal_d2h']
    selected = [(name, component(branch, name)[0]) for name in names if component(branch, name)]
    lines += [f'### {branch}\n\n', table(['Component', 'Frequency', 'CPU总/ms', 'CUDA span总/ms', 'CPU/Call µs', 'Calls'],
        [[name, frequency.get(name, '见源码表'), fmt(row['cpu_inclusive_ms']), fmt(row['cuda_span_inclusive_ms']),
          fmt(row['cpu_us_per_call']), row['calls']] for name, row in selected])]
    native, observed = case(branch), case(branch, mode=1)
    lines += [f'mode1/mode0 CPU总时间比：{observed["cpu_wall_ms_median"]/native["cpu_wall_ms_median"]:.3f}。'
              '细粒度event创建、记录与metadata读取会扰动host调度；不能把其绝对组件值直接从native总时间中相减。\n\n']
prep_rows=[]
for branch in ['selector','tf']:
    key=f'{branch}_e5_b8192_r4_m1'
    for item in components:
        if item['case']==key and item['component'].startswith('event/'):
            prep_rows.append([branch,item['component'],'EVENT-ONCE',fmt(item['cpu_inclusive_ms']),fmt(item['cuda_span_inclusive_ms']),item['calls']])
lines += ['**事件快照与Teacher准备（在finalize总时间之外单列）：**\n\n',
          table(['branch','Component','Frequency','CPU总/ms','CUDA span总/ms','Calls'],prep_rows),
          '临时Adam construction属于固定事件成本，Adam state实际分配发生在首个optimizer step。batch构造的slice view、H2D permutation和gather已分别列出；miscellaneous可由raw组件exclusive时间和coordinator总范围核对，不将所有nested inclusive时间求和。\n\n']

lines += ['## 14. Batch scaling 与尾 batch\n\n',
          '保持所有训练点、四个ratio、相同网络与loss。batch改变实际step数，这是实验变量。完整事件时间包括已有审计；TF audit大小随batch变化，故另给训练batch范围内的samples/s、forward/backward/Adam。训练范围来自mode1，受插桩扰动；native总时间用于稳定吞吐比较。\n\n']
batch_details = []
for branch in ['selector', 'tf']:
    values = []
    for batch in [2048, 4096, 8192, 16384, 32768]:
        row = case(branch, batch=batch)
        train = component(branch, 'training_batch', batch=batch)
        forward = component(branch, 'student_forward', batch=batch)
        backward = component(branch, 'backward', batch=batch)
        adam = component(branch, 'adam_step', batch=batch)
        training_forward=next(r for r in training_stages if r['case']==f'{branch}_e5_b{batch}_r4_m1' and r['stage']=='student_forward')
        train_ms = float(train[0]['cpu_inclusive_ms']) if train else None
        # Forward includes evaluation; keep this distinction explicit.
        values.append([batch, fmt(row['cpu_wall_ms_median']), row['selector_steps']+row['tf_steps'],
                       fmt(row['samples_per_second_including_audit'], 0),
                       fmt(row['peak_allocated_bytes']/2**20), fmt(train_ms) if train_ms else 'NA',
                       fmt(training_forward['cpu_ms']),
                       fmt(float(backward[0]['cpu_inclusive_ms'])) if backward else 'NA',
                       fmt(float(adam[0]['cpu_inclusive_ms'])) if adam else 'NA'])
        batch_details.append({'branch': branch, 'batch': batch, 'native_cpu_median_ms': row['cpu_wall_ms_median'],
                              'training_cpu_ms_instrumented': train_ms,
                              'training_ratio_samples_per_second_instrumented': row['training_ratio_sample_visits']/(train_ms/1000) if train_ms else None,
                              'all_student_forward_cpu_ms_instrumented': float(forward[0]['cpu_inclusive_ms']) if forward else None})
    lines += [f'### {branch}\n\n', table(['batch', 'native事件median/ms', 'steps', 'ratio样本/s含固定审计', 'peak allocated MiB',
                                       '训练batch总/ms*', '训练forward/ms*', 'backward总/ms*', 'Adam总/ms*'], values)]
(RAW / 'batch_training_summary.json').write_text(json.dumps(batch_details, indent=2))
lines += ['`*`为插桩范围。`raw/batch_training_summary.json`保留训练阶段吞吐和全部Student forward总耗时，后者含正式评估；不要把含评估forward除以训练steps当成每step forward。\n\n',
          'Selector B8192并没有一个5100左右的普通tail：生产stratumBatch将两个群体均匀切成4份。primary为7421/7422，anchor为3800/3801；这些小变化不构成极小tail。TF才是3个8192加5107，tail仍为full的62.34%，并非tiny batch。\n\n']
tails = []
for branch in ['selector', 'tf']:
    key = f'{branch}_e5_b8192_r4_m1'
    for row in components:
        if row['case'] == key and '/batch_shape_' in row['component']:
            tails.append([branch, row['component'].split('/')[-1], row['calls'], fmt(row['cpu_us_per_call']), fmt(row['cuda_span_us_per_call'])])
lines += [table(['branch', '实际shape范围', 'calls', 'CPU µs/call', 'CUDA span µs/call'], tails),
          'tail范围包括该batch的ratio forwards、loss、backward与Adam，不能解释为单一kernel效率；是否出现更高每样本成本应按实际样本数归一化，最终优化仍须保持训练语义。\n\n']
full=component('tf','batch_shape_8192')[0]
tail=component('tf','batch_shape_5107')[0]
penalty=(float(tail['cuda_span_us_per_call'])/5107)/(float(full['cuda_span_us_per_call'])/8192)
lines += [f'TF tail/full的CUDA span每样本成本比约{penalty:.3f}。本事件tail耗时接近full，但样本少37.66%，说明该batch范围存在显著非按样本线性缩放的成本；这不等同于硬件counter证明的memory-bound。\n\n']

lines += ['## 15. Ratio scaling\n\n']
for branch in ['selector', 'tf']:
    lines += [f'### {branch}\n\n', table(['ratios', '取值', 'native median/ms', 'steps', 'ratio样本/s'],
       [[r, '[0.01,0.05,0.10,0.15]' if r==4 else '[0.01,0.05]' if r==2 else '[0.01]',
         fmt(case(branch,ratios=r)['cpu_wall_ms_median']),
         case(branch,ratios=r)['selector_steps']+case(branch,ratios=r)['tf_steps'],
         fmt(case(branch,ratios=r)['samples_per_second_including_audit'],0)] for r in [1,2,4]])]
lines += ['每个ratio都会增加forward、loss与custom backward节点，但不增加optimizer.step次数；串行ratio loop的可共享数据依赖已在第4节逐项列出。TF fixed Teacher transport也按ratio重复，所以其ratio scaling同时改变固定开销和训练开销。\n\n']

lines += ['## 16. Operator／kernel／indexing 与 backward\n\n',
          '以下为单个代表epoch的LibTorch Kineto trace。其CPU/GPU运行时间受profiler影响，适合定位operator和kernel，不替代native median。已按CUDA correlation → launch所在CPU线程scope归因，避免将该build写出的CUDA External id误当CPU operator id；全部kernel均完成归因。\n\n']
for branch in ['selector', 'tf']:
    trace = traces[branch]
    lines += [f'### {branch}\n\n',
              table(['指标', '结果'], [
                  ['kernel calls', trace['kernel_calls']], ['total kernel GPU ms', fmt(trace['total_kernel_gpu_ms'])],
                  ['backward kernel GPU ms', fmt(trace['backward_kernel_gpu_ms'])],
                  ['Student forward kernel GPU ms', fmt(trace['student_forward_kernel_gpu_ms'])],
                  ['index/gather/nonzero/scatter/where 直接operator GPU ms', fmt(trace['index_operator_kernel_gpu_ms'])],
                  ['上述index占全部kernel时间', fmt(100*trace['index_operator_kernel_gpu_ms']/trace['total_kernel_gpu_ms'])+'%'],
                  ['duration<10µs kernels', f'{trace["tiny_kernel_calls_lt_10us"]} / {trace["kernel_calls"]}'],
                  ['tiny kernel累计GPU ms', fmt(trace['tiny_kernel_time_lt_10us_ms'])],
              ]),
              table(['互斥阶段', 'kernel GPU ms'], [[k,fmt(v)] for k,v in trace['exclusive_phase_kernel_gpu_ms'].items()])]
    operators = [r for r in csv_rows(branch + '_operators.csv') if r['backward'] == 'True'][:12]
    lines += ['**backward top operators / custom scopes**\n\n',
              table(['operator', 'CUDA/ms', 'kernel calls', 'backward GPU %'],
                    [[r['operator'],fmt(float(r['gpu_us'])/1000),r['kernel_calls'],fmt(r['percentage_backward_kernel_time'])] for r in operators])]
    kernels = [r for r in csv_rows(branch + '_kernels.csv') if r['backward'] == 'True'][:8]
    lines += ['**backward top kernels**\n\n', table(['kernel', 'calls', 'GPU总/ms', 'average/µs'],
                  [[r['kernel'],r['calls'],fmt(float(r['total_gpu_us'])/1000),fmt(r['average_gpu_us'])] for r in kernels])]
lines += ['专用CUDA路径没有aten grid_sampler_backward；其对应工作为queryGradKernel。MLP GEMM、bias reduction、ReLU、Output quaternion backward和ATen损失链分别保留在原始trace及top表。custom Function下的直接CUDA调用归到相应scope，因此scope与ATen operator均可出现于同一排行榜；百分比分母统一为已归因backward kernel总时间，不使用CPU耗时冒充CUDA。\n\n',
          '本模型的grid query并非主要forward热点：代表epoch中全部76次Teacher/Student queryKernel合计1.256 ms，对应76次Warp32 MLP为15.927 ms。backward中MLP占43.59%，HexPlane query backward占25.02%。因此“TF主要慢在grid_sample”不符合该CUDA模型的实测；legacy默认模型没有在此事件中做相同runtime结论。\n\n',
          'Selector参数虽只有4962，但backward仍有可测的GEMM与reduction热点；不能把全部成本称为launch overhead。其大量短kernel、cache hit动态索引与串行多ratio编排值得优化调查，须同时保留真正矩阵计算的成本。两分支Adam/zero_grad在E5总事件中的占比很小，未显示为首要瓶颈。\n\n',
          'TF完整事件的索引开销很大，且多数来自Teacher transport固定项；仅比较“backward vs event total”会严重低估训练kernel的优化价值。后续应按固定项与steady training分别优化，不把两类瓶颈混为一谈。\n\n']

lines += ['## 17. Nsight Systems／Compute 与利用率结论\n\n',
          table(['branch', 'Nsight Systems', 'Nsight Compute', '选中kernel'],
                [[branch, status.get(branch,{}).get('nsys',status.get(branch,{}).get('nsys_error','unknown')),
                  status.get(branch,{}).get('ncu',status.get(branch,{}).get('ncu_error','unknown')),
                  status.get(branch,{}).get('ncu_selected_hotspot','unknown')] for branch in ['selector','tf']]),
          table(['Nsight独立采集','kernel calls','kernel GPU总/ms','<10µs calls'],
                [[branch,read('nsys_summary.json')[branch]['kernel_calls'],fmt(read('nsys_summary.json')[branch]['total_kernel_gpu_ms']),
                  read('nsys_summary.json')[branch]['tiny_kernel_calls_lt_10us']] for branch in ['selector','tf']]),
          '原始 `.nsys-rep`、SQLite、stats CSV、NCU CSV/stdout全部保留。此版nsys使用gpukernsum/cudaapisum/nvtxsum报告名；不匹配的尝试另保存在profiler_attempt1/2。NCU只有实际捕获硬件counter才算完成；退出码0但“No kernels were profiled”不算counter证据。若驱动权限拒绝则记录原文，不改驱动、不安装工具。\n\n',
          '本次NCU实际阻塞为 **ERR_NVGPUCTRPERM**：当前用户无GPU Performance Counter访问权限。已记录原始错误，未修改系统权限。因此SM active、achieved occupancy、DRAM吞吐和SM throughput的硬件counter结论均为 **inconclusive**；Kineto给出的估算occupancy不替代硬件实测。\n\n',
          '利用率结论：两个分支均存在大量微小kernel及频繁launch；TF尤其包含与144组transport有关的host/synchronization/索引编排成本。trace中的kernel间空隙和实际kernel时间可支持“launch与编排值得优先调查”，但不能据此声称SM在全部训练中都闲置。完整CD事件的compute-bound／memory-bound分类为 **inconclusive**；单kernel counter即使可用，也不能推广到混合的整个事件。batch8192不是通过本轮证明的最优点；较大batch的吞吐变化还混入step数、Teacher分块、diagnostic audit大小和TF Warp32/cuBLAS dispatch变化。\n\n']

lines += ['## 18. 复现、验证与交付文件\n\n',
          table(['验证', '结果'], [
              ['相同event输入', 'raw/input_sha256.json；原始event.json保存；snapshot未重建'],
              ['重复初始化', '每run参数逐tensor torch::equal检查；新Manager、新临时Adam、同种子'],
              ['实际step数', f'{len(read("run_contracts.json"))} 条run记录通过公式/实际计数验证'],
              ['Gaussian / normal Adam', '每run断言未修改'],
              ['插桩参数一致性 Selector', str(parity.get('selector',{}).get('all_bitwise_equal')) + ' bitwise; allclose=' + str(parity.get('selector',{}).get('allclose'))],
              ['插桩参数一致性 TF', str(parity.get('tf',{}).get('all_bitwise_equal')) + ' bitwise; allclose=' + str(parity.get('tf',{}).get('allclose'))],
              ['TF parity容差', 'atol=2e-5、rtol=2e-4，逐参数max/mean误差见JSON；atomic非逐bit确定性'],
              ['正式配置', '未修改；全部scaling设置仅存在于独立benchmark'],
          ]),
          '本地主要证据目录：`tests/diagnostics/change_distill_performance_20261017/`。\n\n',
          '此报告中的 `raw/`、`source/`、`instrumented/`、`plots/` 均相对于上述证据目录；根目录报告中的图像链接则使用仓库相对路径。\n\n',
          table(['路径', '内容'], [
              ['source/ + raw/source_manifest.json', '当前本地冻结源码与逐文件SHA-256'],
              ['instrumented/ + instrumentation.diff', '隔离插桩副本及完整差异'],
              ['raw/cases/*/{cold,warmup_*,warm_*}/runtime.json', '全部逐run CPU/CUDA/显存/step数据'],
              ['raw/cases/*_m1/warm_0/components.csv', '逐次组件CPU与CUDA Event spans'],
              ['raw/cases/*_m2/profile/trace.json', '原始LibTorch Kineto Chrome trace'],
              ['raw/*_operators.csv / *_kernels.csv / *_transfers.csv', '归因后的operator/kernel/设备传输统计'],
              ['raw/*_allocation_events.csv', '原始allocation/deallocation事件'],
              ['raw/nsys_*.nsys-rep / *.sqlite / *_stats.csv', 'Nsight Systems原始与导出数据'],
              ['raw/nsys_*_kernels.csv / *_api.csv', '从原始SQLite导出的单表CSV，可直接程序读取'],
              ['raw/ncu_*.csv / *.log', '有限热点硬件counter或不可用原因'],
              ['raw/benchmark_summary.csv/json + all_runs.csv', '统计汇总，含mean/median/std/min/max'],
              ['raw/epoch_fit.json + batch_training_summary.json', 'epoch拟合与训练阶段batch吞吐'],
              ['raw/previous_runner_replication/', '本轮同机重跑上一轮runner'],
              ['raw/run_contracts.json + instrumentation_parity.json', 'step、隔离与插桩一致性验证'],
              ['raw/*tensor_metadata.json及各case的tensor_metadata.json', 'shape、stride、contiguous、dtype、device、requires_grad、grad_fn'],
          ]),
          '服务器完整工作目录：`/home/prmi/zwj/mycartgs/results/change_distill_performance_audit/20261017`。原始完整snapshot继续保存在上一轮目录，SHA-256记录用于复用。原始profiling文件已完整复制本地，并打包为 `raw_profiling_outputs.tar.gz`；`audit_source.tar.gz`为冻结源码、插桩、独立runner和分析脚本。\n\n',
          '复现顺序：冻结/核对source → `build_native.py` → `run_matrix.py --environment` → `run_matrix.py` → `run_kernel_profiles.py` → `replicate_previous_runner.py` → `validate_results.py` → `analyze.py` → `make_report.py`。远程脚本包含本次明确主机的绝对路径，不把它们描述为通用安装工具。\n\n',
          '**可直接进入下一轮设计的信息：** TF event固定transport是已证实瓶颈；Selector跨ratio重算三个属性embedding、两分支cache-hit动态索引和custom backward多算输入梯度具有明确源码依据；具体fusion、vectorization、Graph或混合精度方案仍需独立数值/性能验收，本轮不实现、不据此删减任何算法工作。\n']
text = ''.join(lines)
(REPO / 'CHANGE_DISTILL_PERFORMANCE_AUDIT.md').write_text(text, encoding='utf-8', newline='\n')
(D / 'CHANGE_DISTILL_PERFORMANCE_AUDIT.md').write_text(text.replace('tests/diagnostics/change_distill_performance_20261017/plots/scaling_summary.png','plots/scaling_summary.png'), encoding='utf-8', newline='\n')
print('report lines:', len(text.splitlines()))
