"""离线重分类已有Profile Kernel；不重复采样GPU，不改变原始Trace或时长。"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def classify(row):
    name = row.kernel
    if 'compactActivePendingBits' in name:
        return 'member_compact'
    if 'clearActivePendingBits' in name:
        return 'member_clear'
    if 'uploadRequest' in name:
        return 'request_upload'
    if any(value in name for value in ['makePlanes', 'visibleCounts', 'DeviceCompactInitKernel', 'sgemm']):
        return 'candidate_view_prepare'
    return row.module


def main():
    parts, calls = [], []
    for folder in R.glob('*_RELEASE_ANALYSIS'):
        frame = pd.read_csv(folder / 'KERNELS.csv.gz')
        frame['module'] = frame.apply(classify, axis=1)
        frame['profile'] = folder.name
        parts.append(frame)
        c = pd.read_csv(folder / 'CALLS.csv.gz')
        c['profile'] = folder.name
        calls.append(c)
    kernels, samples = pd.concat(parts), pd.concat(calls)
    keys = ['profile', 'scene', 'workload', 'condition', 'variant', 'scope']
    denominator = samples.groupby(keys).agg(calls=('call', 'size')).reset_index()
    modules = kernels.groupby(keys + ['module']).agg(gpu_us=('gpu_us', 'sum'),
        kernel_count=('gpu_us', 'size')).reset_index().merge(denominator, on=keys, validate='many_to_one')
    modules['gpu_us_per_call'] = modules.gpu_us / modules.calls
    modules['kernels_per_call'] = modules.kernel_count / modules.calls
    modules.to_csv(R / 'FINAL_PROFILE_COMPONENTS.csv', index=False)
    summary = samples.groupby(keys).agg(kernel_gpu_us=('kernel_gpu_us', 'mean'), kernels=('kernels', 'mean'),
        profiled_host_us=('range_host_us', 'mean'), synchronize_calls=('synchronize_calls', 'mean'),
        graph_launch_calls=('graph_launch_calls', 'mean'), memcpy_calls=('memcpy_calls', 'mean'),
        measured_calls=('call', 'size')).reset_index()
    summary.to_csv(R / 'FINAL_PROFILE_CALLS.csv', index=False)
    apis = pd.concat([pd.read_csv(folder / 'APIS.csv.gz').assign(profile=folder.name) for folder in R.glob('*_RELEASE_ANALYSIS')])
    api = apis.groupby(keys + ['api']).agg(host_us=('host_us', 'sum'), count=('host_us', 'size')).reset_index().merge(denominator, on=keys, validate='many_to_one')
    api['count_per_call'] = api['count'] / api.calls
    api['host_us_per_call'] = api.host_us / api.calls
    api.to_csv(R / 'FINAL_PROFILE_APIS.csv', index=False)
    print('PROFILE_COMPONENTS_COMPLETE', len(summary), len(modules))


if __name__ == '__main__':
    main()
