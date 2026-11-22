"""从Nsight原始CSV生成组件级归因；干扰Profile明确隔离。"""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'reports/profile_suite02';summary=json.loads((root/'SUMMARY.json').read_text())
    job=json.loads((HERE/'reports/jobs/profile_suite02.json').read_text())
    assert summary['passed'] and job['measurement_valid'] and job['foreign_active_samples']==0
    rows=[];kernels=[]
    cases=summary['rows']+[dict(variant='G',condition='local0.01',scope='raster',calls=20,label='final02')]
    if (HERE/'reports/profile_q01/SUMMARY.json').exists():
        q_job=json.loads((HERE/'reports/jobs/profile_q01.json').read_text())
        assert q_job['measurement_valid'] and q_job['foreign_active_samples']==0
        cases.append(dict(variant='Q',condition='local0.01',scope='pipeline',calls=20,label='final_Q01'))
    if (HERE/'reports/profile_r01/SUMMARY.json').exists():
        r_job=json.loads((HERE/'reports/jobs/profile_r01.json').read_text())
        assert r_job['measurement_valid'] and r_job['foreign_active_samples']==0
        cases.append(dict(variant='R',condition='local0.01',scope='pipeline',calls=20,label='final_R01'))
    for case in cases:
        label=case['label'];prefix=HERE/'reports'/label if label in ['final02','final_Q01','final_R01'] else root/label
        kernel=pd.read_csv(str(prefix)+'_gpukernsum.csv');api=pd.read_csv(str(prefix)+'_cudaapisum.csv')
        count=case['calls']
        row=dict(variant=case['variant'],condition=case['condition'],scope=case['scope'],calls=count,
            kernel_count_per_frame=kernel.Instances.sum()/count,kernel_us_per_frame=kernel['Total Time (ns)'].sum()/count/1000,
            api_time_us_per_frame=api['Total Time (ns)'].sum()/count/1000,
            host_kernel_launches_per_frame=api[api.Name.str.startswith('cudaLaunchKernel')]['Num Calls'].sum()/count,
            graph_launches_per_frame=api[api.Name.str.startswith('cudaGraphLaunch')]['Num Calls'].sum()/count,
            device_syncs_per_frame=api[api.Name.str.startswith('cudaDeviceSynchronize')]['Num Calls'].sum()/count,
            stream_syncs_per_frame=api[api.Name.str.startswith('cudaStreamSynchronize')]['Num Calls'].sum()/count,
            memcpy_calls_per_frame=api[api.Name.str.startswith('cudaMemcpy')]['Num Calls'].sum()/count,
            cuda_memsets_per_frame=api[api.Name.str.startswith('cudaMemset')]['Num Calls'].sum()/count,
            source=label,profiling_only=True)
        rows.append(row)
        for _,item in kernel.iterrows():
            kernels.append(dict(variant=case['variant'],condition=case['condition'],scope=case['scope'],
                kernel_name=item['Name'],count_per_frame=item['Instances']/count,
                total_us_per_frame=item['Total Time (ns)']/count/1000,source=label))
    pd.DataFrame(rows).to_csv(HERE/'GPU_PROFILE.csv',index=False)
    pd.DataFrame(kernels).to_csv(HERE/'GPU_KERNEL_DETAILS.csv',index=False)
    observations=[]
    for name in ['formal02','formal03','formal04','preparation01','preparation02','camera_final01','camera_final02',
        'async_final01','async_final02','async_final03','count_scaling01','profile_suite01','profile_suite02','profile_q01','profile_r01','microtile01']:
        identity=HERE/'reports/jobs'/(name+'.json');monitor=HERE/'logs'/(name+'_MONITOR.jsonl')
        if not identity.exists() or not monitor.exists():continue
        job=json.loads(identity.read_text());gpu=[]
        if job['state'] not in ['complete','failed']:continue
        for line in monitor.read_text().splitlines():
            value=json.loads(line);fields=value['gpu'].replace('MiB','').replace('MHz','').replace('W','').replace('%','').split(',')
            gpu.append([float(s.strip()) for s in fields])
        d=pd.DataFrame(gpu,columns=['utilization_pct','memory_MiB','clock_MHz','temperature_C','power_W'])
        observations.append(dict(job=name,state=job['state'],measurement_valid=job['measurement_valid'],
            foreign_active_samples=job['foreign_active_samples'],monitor_samples=len(d),
            clock_min_MHz=d.clock_MHz.min(),clock_median_MHz=d.clock_MHz.median(),clock_max_MHz=d.clock_MHz.max(),
            temperature_min_C=d.temperature_C.min(),temperature_max_C=d.temperature_C.max(),
            memory_max_MiB=d.memory_MiB.max(),power_max_W=d.power_W.max()))
    pd.DataFrame(observations).to_csv(HERE/'GPU_ENVIRONMENT.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
