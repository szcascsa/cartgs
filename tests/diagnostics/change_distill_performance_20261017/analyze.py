"""汇总原始数据；CUDA span 与实际 kernel busy time 使用不同列。"""
from pathlib import Path
from collections import defaultdict, Counter
import bisect
import csv
import json
import math
import statistics as stat

D = Path(__file__).resolve().parent
RAW = D / 'raw'


def save_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    return {'mean': stat.mean(values), 'median': stat.median(values),
            'std': stat.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values), 'n': len(values)}


def cases():
    all_rows, aggregate = [], []
    for path in sorted((RAW / 'cases').glob('*/runs.json')):
        if (path.parent / 'CONTAMINATED.json').exists():
            continue
        rows = json.loads(path.read_text())
        for row in rows:
            all_rows.append({'case': path.parent.name, **row})
        warm = [r for r in rows if r['label'].startswith('warm_')]
        if not warm:
            continue
        row = warm[0]
        entry = {k: row[k] for k in ['branch', 'epochs', 'batch_size', 'ratio_count', 'instrumentation_mode']}
        entry['case'] = path.parent.name
        entry['cold_cpu_ms'] = rows[0]['cpu_wall_ms']
        entry['cold_cuda_ms'] = rows[0]['cuda_elapsed_ms']
        for key in ['cpu_wall_ms', 'cuda_elapsed_ms', 'event_preparation_ms']:
            for name, value in summary([r[key] for r in warm]).items():
                entry[key + '_' + name] = value
        entry['selector_steps'] = row['selector_steps']
        entry['tf_steps'] = row['tf_steps']
        entry['peak_allocated_bytes'] = max(r['peak_allocated_bytes'] for r in warm)
        entry['additional_allocated_bytes'] = max(r['additional_allocated_bytes'] for r in warm)
        total_samples = row['selector_count'] + row['anchor_count'] if row['branch'] == 'selector' else row['tf_count']
        entry['training_gaussian_visits'] = total_samples * row['epochs']
        entry['training_ratio_sample_visits'] = total_samples * row['epochs'] * row['ratio_count']
        entry['samples_per_second_including_audit'] = entry['training_ratio_sample_visits'] / (entry['cpu_wall_ms_median'] / 1000)
        steps = row['selector_steps'] + row['tf_steps']
        entry['ms_per_epoch_including_fixed'] = entry['cpu_wall_ms_median'] / row['epochs']
        entry['ms_per_optimizer_step_including_fixed'] = entry['cpu_wall_ms_median'] / steps if steps else None
        aggregate.append(entry)
    save_csv(RAW / 'all_runs.csv', all_rows)
    save_csv(RAW / 'benchmark_summary.csv', aggregate)
    (RAW / 'benchmark_summary.json').write_text(json.dumps(aggregate, indent=2))
    fits = {}
    for branch in ['selector', 'tf']:
        rows = [r for r in aggregate if r['branch'] == branch and r['batch_size'] == 8192 and
                r['ratio_count'] == 4 and r['instrumentation_mode'] == 0]
        if len(rows) < 5:
            continue
        xs = [r['epochs'] for r in rows]
        ys = [r['cpu_wall_ms_median'] for r in rows]
        xbar, ybar = stat.mean(xs), stat.mean(ys)
        slope = sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)
        intercept = ybar-slope*xbar
        residuals = [y-(intercept+slope*x) for x, y in zip(xs, ys)]
        fits[branch] = {'fixed_ms': intercept, 'ms_per_epoch': slope,
                        'r_squared': 1-sum(r*r for r in residuals)/sum((y-ybar)**2 for y in ys),
                        'epochs': xs, 'median_ms': ys, 'residual_ms': residuals,
                        'warning': 'regression intercept is an estimate; measured components are separate'}
    (RAW / 'epoch_fit.json').write_text(json.dumps(fits, indent=2))
    return aggregate


def components():
    output = []
    partitions = []
    training_rows=[]
    for path in (RAW / 'cases').glob('*/warm_*/components.csv'):
        if '_m1' not in str(path):
            continue
        with path.open() as file:
            rows = list(csv.DictReader(file))
        by_id = {int(r['id']): r for r in rows}
        children_cpu, children_gpu = defaultdict(float), defaultdict(float)
        for r in rows:
            children_cpu[int(r['parent'])] += float(r['cpu_us'])
            children_gpu[int(r['parent'])] += float(r['cuda_span_us'])
        partition = defaultdict(float)
        training=defaultdict(list)
        for r in rows:
            path_names, cursor = [], r
            while cursor is not None:
                path_names.append(cursor['component'].split('/')[-1])
                cursor = by_id.get(int(cursor['parent']))
            if 'coordinator_finalize' not in path_names:
                continue
            if 'training_batch' in path_names and path_names[0] in [
                    'student_forward','loss_construction','backward','adam_step','zero_grad','target_cache']:
                training[path_names[0]].append(r)
            own = max(0, float(r['cpu_us'])-children_cpu[int(r['id'])])
            if 'geometry_transport' in path_names:
                phase = 'teacher_geometry_transport'
            elif 'teacher_forward' in path_names:
                phase = 'teacher_forward'
            elif 'backward' in path_names:
                phase = 'backward'
            elif 'adam_step' in path_names:
                phase = 'optimizer_step'
            elif 'zero_grad' in path_names:
                phase = 'zero_grad'
            elif 'student_forward' in path_names:
                phase = 'student_forward'
            elif 'target_cache' in path_names:
                phase = 'target_cache_other'
            elif 'loss_construction' in path_names:
                phase = 'loss_construction'
            elif 'permutation_h2d' in path_names:
                phase = 'permutation_h2d'
            elif 'shuffle_index_gather' in path_names or 'gaussian_gather' in path_names:
                phase = 'gaussian_or_order_gather'
            elif any(n in path_names for n in ['anchor_sampling','shuffle_cpu_h2d_gather']):
                phase = 'anchor_and_order_cpu_preparation'
            elif any(n in path_names for n in ['scope_indices','outside_identity_filter','filtered_indices']):
                phase = 'mask_and_filter_preparation'
            elif any(n in path_names for n in ['finite_check_d2h','positive_check_d2h','aabb_diagonal_d2h']):
                phase = 'explicit_scalar_sync'
            elif any(n in path_names for n in ['primary_evaluation','anchor_evaluation','untouched_evaluation']):
                phase = 'evaluation_scalar_and_loop_other'
            else:
                phase = 'coordination_batch_logging_and_instrumentation_misc'
            partition[phase] += own
        total = sum(partition.values())
        for phase, value in partition.items():
            partitions.append({'case': path.parents[1].name, 'phase': phase,
                               'cpu_exclusive_ms': value/1000, 'percentage_partition': 100*value/total if total else 0})
        for stage, values in training.items():
            training_rows.append({'case':path.parents[1].name,'stage':stage,'calls':len(values),
                                  'cpu_ms':sum(float(v['cpu_us']) for v in values)/1000,
                                  'cuda_span_ms':sum(float(v['cuda_span_us']) for v in values)/1000})
        groups = defaultdict(list)
        for r in rows:
            groups[r['component']].append(r)
        for name, values in groups.items():
            cpu = sum(float(r['cpu_us']) for r in values)
            gpu = sum(float(r['cuda_span_us']) for r in values)
            cpu_self = sum(max(0, float(r['cpu_us'])-children_cpu[int(r['id'])]) for r in values)
            gpu_self = sum(max(0, float(r['cuda_span_us'])-children_gpu[int(r['id'])]) for r in values)
            output.append({'case': path.parents[1].name, 'component': name, 'calls': len(values),
                           'cpu_inclusive_ms': cpu/1000, 'cuda_span_inclusive_ms': gpu/1000,
                           'cpu_exclusive_ms': cpu_self/1000, 'cuda_span_exclusive_ms': gpu_self/1000,
                           'cpu_us_per_call': cpu/len(values), 'cuda_span_us_per_call': gpu/len(values)})
    output.sort(key=lambda r: (r['case'], -r['cpu_inclusive_ms']))
    save_csv(RAW / 'component_summary.csv', output)
    save_csv(RAW / 'cpu_partition.csv', partitions)
    save_csv(RAW / 'training_stage_summary.csv', training_rows)


def parse_trace(path, branch):
    events = json.loads(path.read_text())['traceEvents']
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation')]
    kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    by_thread = defaultdict(list)
    external = {}
    for event in cpu:
        by_thread[event['tid']].append(event)
        key = event.get('args', {}).get('External id')
        if key is not None:
            external[key] = event
    ancestry = {}
    for tid, rows in by_thread.items():
        stack = []
        rows.sort(key=lambda e: (e['ts'], -e.get('dur', 0)))
        for event in rows:
            while stack and stack[-1]['ts'] + stack[-1].get('dur', 0) < event['ts'] + event.get('dur', 0) - 0.01:
                stack.pop()
            ancestry[id(event)] = list(stack)
            stack.append(event)
    correlation = {}
    launches_by_thread = defaultdict(list)
    for event in events:
        if event.get('cat') == 'cuda_runtime':
            args = event.get('args', {})
            if 'correlation' in args:
                correlation[args['correlation']] = event
                launches_by_thread[event['tid']].append(event)
    launch_parent = {}
    # This LibTorch Kineto build writes CUDA correlation IDs as External id;
    # map the correlated launch to its containing CPU scope instead of equating
    # that ID to a CPU operator's unrelated External id.
    for tid, launches in launches_by_thread.items():
        combined = [(e, False) for e in by_thread[tid]] + [(e, True) for e in launches]
        combined.sort(key=lambda pair: (pair[0]['ts'], pair[1], -pair[0].get('dur', 0)))
        stack = []
        for event, is_launch in combined:
            end = event['ts'] + event.get('dur', 0)
            while stack and stack[-1]['ts'] + stack[-1].get('dur', 0) < end - 0.01:
                stack.pop()
            if is_launch:
                if stack:
                    launch_parent[id(event)] = stack[-1]
            else:
                stack.append(event)
    kernel_groups = defaultdict(list)
    op_groups = defaultdict(list)
    allocations = []
    attributed, backward_us, forward_us, indexing_us = 0, 0.0, 0.0, 0.0
    phase_gpu = defaultdict(float)
    for kernel in kernels:
        args = kernel.get('args', {})
        launch = correlation.get(args.get('correlation'))
        operator = launch_parent.get(id(launch)) if launch else None
        ancestors = [] if operator is None else ancestry.get(id(operator), []) + [operator]
        scopes = [e['name'] for e in ancestors if e.get('cat') == 'user_annotation']
        backward = any('backward' in e['name'].lower() for e in ancestors)
        forward = any(name.endswith('/student_forward') for name in scopes)
        # Device kernels launched directly by custom functions have user-scope IDs.
        opname = operator['name'] if operator else 'unattributed'
        if operator:
            attributed += 1
        duration = float(kernel['dur'])
        if any(name.endswith('/geometry_transport') for name in scopes):
            phase = 'teacher_geometry_transport'
        elif any(name.endswith('/teacher_forward') for name in scopes):
            phase = 'teacher_forward'
        elif backward:
            phase = 'backward'
        elif any(name.endswith('/student_forward') for name in scopes):
            phase = 'student_forward'
        elif any(name.endswith('/adam_step') for name in scopes):
            phase = 'adam'
        elif any(name.endswith('/loss_construction') for name in scopes):
            phase = 'loss_construction'
        else:
            phase = 'index_cache_event_and_misc'
        phase_gpu[phase] += duration
        if backward:
            backward_us += duration
        if forward:
            forward_us += duration
        if any(word in opname.lower() for word in ['index', 'gather', 'nonzero', 'scatter', 'where']):
            indexing_us += duration
        kernel_groups[(kernel['name'], backward)].append(duration)
        op_groups[(opname, backward)].append(duration)
    total_us = sum(float(e['dur']) for e in kernels)
    transfer_summary = defaultdict(lambda: {'calls': 0, 'gpu_us': 0.0, 'bytes': 0})
    for event in events:
        if event.get('cat') not in ('gpu_memcpy', 'gpu_memset') or event.get('ph') != 'X':
            continue
        row = transfer_summary[event['name']]
        row['calls'] += 1
        row['gpu_us'] += event.get('dur', 0)
        row['bytes'] += event.get('args', {}).get('bytes', 0)
    save_csv(RAW / (branch + '_transfers.csv'), [{'operation': name, **row} for name, row in transfer_summary.items()])
    kernel_rows = [{'branch': branch, 'kernel': name, 'backward': backward, 'calls': len(values),
                    'total_gpu_us': sum(values), 'average_gpu_us': stat.mean(values),
                    'percentage_all_kernel_time': 100*sum(values)/total_us if total_us else 0}
                   for (name, backward), values in kernel_groups.items()]
    op_rows = [{'branch': branch, 'operator': name, 'backward': backward, 'kernel_calls': len(values),
                'gpu_us': sum(values), 'percentage_all_kernel_time': 100*sum(values)/total_us if total_us else 0,
                'percentage_backward_kernel_time': 100*sum(values)/backward_us if backward and backward_us else 0}
               for (name, backward), values in op_groups.items()]
    kernel_rows.sort(key=lambda r: -r['total_gpu_us'])
    op_rows.sort(key=lambda r: -r['gpu_us'])
    save_csv(RAW / (branch + '_kernels.csv'), kernel_rows)
    save_csv(RAW / (branch + '_operators.csv'), op_rows)
    for event in events:
        if event.get('name') == '[memory]':
            allocations.append({'timestamp': event['ts'], **event.get('args', {})})
    if allocations:
        fields = sorted(set().union(*(r.keys() for r in allocations)))
        with (RAW / (branch + '_allocation_events.csv')).open('w', newline='') as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader(); writer.writerows(allocations)
    report = {'kernel_calls': len(kernels), 'total_kernel_gpu_ms': total_us/1000,
              'attributed_kernel_calls': attributed, 'backward_kernel_gpu_ms': backward_us/1000,
              'student_forward_kernel_gpu_ms': forward_us/1000, 'index_operator_kernel_gpu_ms': indexing_us/1000,
              'tiny_kernel_calls_lt_10us': sum(float(e['dur']) < 10 for e in kernels),
              'tiny_kernel_time_lt_10us_ms': sum(float(e['dur']) for e in kernels if float(e['dur']) < 10)/1000,
              'categories': dict(Counter(e.get('cat', '') for e in events)),
              'allocation_events': len(allocations), 'exclusive_phase_kernel_gpu_ms': {k:v/1000 for k,v in phase_gpu.items()}}
    (RAW / (branch + '_trace_summary.json')).write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    cases()
    components()
    for branch in ['selector', 'tf']:
        paths = list((RAW / 'cases').glob(branch + '_e1_b8192_r4_m2/profile/trace.json'))
        if paths:
            print(branch, parse_trace(paths[0], branch))
