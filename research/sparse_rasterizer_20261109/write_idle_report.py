"""两轮空闲复测、同入口消融和真实准备＋渲染汇总。"""
from pathlib import Path
import csv
import json
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results/reports').exists() else HERE
OUT=ROOT/'reports/idle_retest_20261110'


def read(p):return json.loads(p.read_text())


def csv_rows(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    done=read(OUT/'COMPLETE.json');assert done['passed']
    identity=read(OUT/'PROTOCOL.json');integrity=read(OUT/'INTEGRITY.json')
    variants=['original','control','warp_expand','short','warp_short','bitset']
    inputs=[];workloads=[];profiles=[]
    for round_id,name in enumerate(done['accepted_rounds'],1):
        folder=OUT/name
        summary=read(folder/'SUMMARY.json')
        assert summary['passed'] and summary['formal_idle_benchmark'] and not summary['concurrent_diagnostic']
        assert all(not v['active'] for v in summary['observations'])
        process=read(OUT/(name+'_PROCESS.json')) if (OUT/(name+'_PROCESS.json')).exists() else None
        monitors=read(OUT/(name+'_MONITOR.json'))
        if process:
            for observation in monitors:
                for line in observation['pmon'].splitlines():
                    fields=line.split()
                    if len(fields)>=7 and fields[0].isdigit() and fields[1].isdigit() and int(fields[1])!=process['pid']:
                        assert not any(v.isdigit() and int(v)>0 for v in fields[3:7]),('External activity in raw monitor',line)
        else:assert all(not v['foreign_active'] for v in monitors)
        rows=csv_rows(folder/'samples.csv');assert len(rows)==20160
        assert {v['variant'] for v in rows}==set(variants)
        groups={}
        for row in rows:
            groups.setdefault((row['scene'],row['condition'],row['frame'],row['repeat'],row['iteration']),[]).append(row)
            row['run']=round_id
        assert len(groups)==3360 and all(len(g)==6 and len({v['K'] for v in g})==1 for g in groups.values())
        inputs+=rows
        w=csv_rows(folder/'workloads.csv');assert len(w)==224
        workloads.append(w)
        profiles+=summary['profiles']
    assert workloads[0]==workloads[1],'Workloads changed between rounds'
    profile_scope='接受轮次内独立Profiling'
    if not profiles:
        profiles=read(OUT/'round1/SUMMARY.json')['profiles']
        profile_scope='与正式墙钟分开的探索Profiling（该轮原始监控发现Xorg活动，不用于空闲GPU性能结论）'
    rows=[]
    for scene,condition in sorted({(v['scene'],v['condition']) for v in inputs}):
        g=[v for v in inputs if (v['scene'],v['condition'])==(scene,condition)]
        w=[v for v in workloads[0] if (v['scene'],v['condition'])==(scene,condition)]
        item=dict(scene=scene,condition=condition,K_median=st.median(int(v['K']) for v in g),K_min=min(int(v['K']) for v in g),
            K_max=max(int(v['K']) for v in g),bbox_instances_median=st.median(int(v['tile_bbox_instances']) for v in w),
            valid_instances_median=st.median(int(v['valid_tile_instances']) for v in w))
        times={n:st.median(float(v['wall_ms']) for v in g if v['variant']==n) for n in variants}
        item.update(times)
        for n in variants[1:]:
            item[n+'_speedup']=times['original']/times[n]
            per_round=[];per_repeat=[]
            for run in [1,2]:
                rr=[v for v in g if v['run']==run]
                per_round.append(st.median(float(v['wall_ms']) for v in rr if v['variant']=='original')/
                    st.median(float(v['wall_ms']) for v in rr if v['variant']==n))
                for repeat in range(3):
                    per_repeat.append(st.median(float(v['wall_ms']) for v in rr if v['variant']=='original' and int(v['repeat'])==repeat)/
                        st.median(float(v['wall_ms']) for v in rr if v['variant']==n and int(v['repeat'])==repeat))
            item.update({n+'_run1_speedup':per_round[0],n+'_run2_speedup':per_round[1],
                n+'_repeat_min':min(per_repeat),n+'_repeat_max':max(per_repeat)})
        item['bitset_vs_control']=times['control']/times['bitset']
        item['short_vs_control']=times['control']/times['short']
        item['warp_short_vs_warp']=times['warp_expand']/times['warp_short']
        rows.append(item)
    write_csv(OUT/'RASTERIZER_IDLE_RESULTS.csv',rows)
    pipeline=read(OUT/'pipeline/SUMMARY.json');assert pipeline['passed'] and pipeline['bitwise_images'] and pipeline['identical_selection']
    p_rows=csv_rows(OUT/'pipeline/samples.csv');assert len(p_rows)==8640
    assert not any(v['active'] for v in pipeline['observations'])
    pipe=[]
    for scene,condition in sorted({(v['scene'],v['condition']) for v in p_rows}):
        g=[v for v in pipeline['results'] if (v['scene'],v['condition'])==(scene,condition)]
        idx={v['variant']:v for v in g}
        pipe.append(dict(scene=scene,condition=condition,K_median=idx['original']['K_median'],
            original_total_ms=idx['original']['total_ms'],bitset_total_ms=idx['bitset']['total_ms'],
            bitset_prepare_ms=idx['bitset']['prepare_ms'],bitset_render_ms=idx['bitset']['render_ms'],
            total_speedup=idx['bitset']['speedup'],warp_total_ms=idx['warp_expand']['total_ms'],warp_total_speedup=idx['warp_expand']['speedup']))
    write_csv(OUT/'PIPELINE_IDLE_RESULTS.csv',pipe)
    low=[v for v in rows if v['condition'] in ['local1pct','local5pct','local10pct']]
    lowpipe=[v for v in pipe if v['condition'] in ['local1pct','local5pct','local10pct']]
    report=['# 低点数Rasterizer空闲GPU复测报告','','## 结果','',
        f'两轮正式纯渲染复测已完成，采用同一最终CUDA二进制、原checkpoint、相机和渲染输入。1%/5%/10%条件的Bitset相对原入口加速 {min(v["bitset_speedup"] for v in low):.2f}–{max(v["bitset_speedup"] for v in low):.2f}×，纯渲染绝对节省 {min(v["original"]-v["bitset"] for v in low):.3f}–{max(v["original"]-v["bitset"] for v in low):.3f}ms。','',
        f'另做每次重新准备当前视角的完整推理测试，低Ratio准备＋渲染加速 {min(v["total_speedup"] for v in lowpipe):.2f}–{max(v["total_speedup"] for v in lowpipe):.2f}×。这是准备与渲染调用的墙钟，不包含图像编码、传输、Viewer显示或SLAM训练。','',
        '没有按K人为屏蔽高点数；Full与高Ratio结果完整保留。高密度下Bitset可能退化，仍按实测呈现；短列表候选单独比较，不把组合收益归给短列表。所有候选是显式研究选项，生产默认未修改。','',
        '## GPU状态、运行与验证','',
        'RTX 3090，CUDA11.8，FP32、TF32关闭，CPU线程数2。初始利用率0%，两项计算进程仅驻留显存，三次pmon未检测到实际工作；原任务未被终止或调整。两轮正式测试的逐条件检查与独立监控未检测到外部工作；监控为采样，不能排除采样间瞬时波动。未锁定GPU频率。','',
        f'原始监控复核发现最初首轮存在Xorg图形活动：初版外部独立监控仅将计算进程活动计为干扰，遗漏了该图形进程，已依据原始pmon记录剔除整个首轮。最初第二轮启动也被Xorg活动拦截，未产生计时samples.csv。带Profiler的首轮重测又在检查时遇到Xorg活动而中断；随后将Profiling与正式墙钟分开，正式接受轮次为{done["accepted_rounds"]}，重新监控同时检查图形和计算活动。所有剔除轮次和启动均保留，没有根据测速结果挑选轮次。','',
        '每轮4场景×7种工作负载×3次重复×40帧×6版本，共20160条计时；两轮40320条，对应6720组同输入配对。每条件8个跳转视角，预热每个视角每版3次。版本次序交替，统一在调用前后CUDA同步。相同条件逐帧K和工作量一致；全部候选在每轮每个输入上与原图逐位一致。此前500组RGB/radii逐位检查、45项生命周期检查及memcheck/initcheck零错误仍有效，本轮CUDA源码及二进制保持相同。','',
        '纯渲染复用已准备的同一输入包以隔离渲染器成本；后面的完整推理测试每次重新调用同一Teacher DeploymentRenderer.prepare，保持Selector、Compensator和严格K不变，然后渲染，逐次核对ID与图像逐位一致。该测试8640条计时，不复用同帧视锥结果、不引入蒸馏差异。','',
        f'候选二进制SHA256：`{identity["candidate_sha256"]}`。参考二进制SHA256：`{identity["reference_sha256"]}`。','',
        '## 纯渲染耗时（两轮样本中位数）','',
        '| 场景/模式 | 条件 | 实际K中位数 [最小,最大] | 原入口ms | 同入口控制ms | Warp展开ms | Bitset ms | Bitset加速 | 两轮加速 | 六次重复范围 |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---|']
    for v in rows:
        report.append(f'| {v["scene"]} | {v["condition"]} | {v["K_median"]:g} [{v["K_min"]},{v["K_max"]}] | {v["original"]:.4f} | {v["control"]:.4f} | {v["warp_expand"]:.4f} | {v["bitset"]:.4f} | {v["bitset_speedup"]:.2f}× | {v["bitset_run1_speedup"]:.2f}/{v["bitset_run2_speedup"]:.2f} | {v["bitset_repeat_min"]:.2f}–{v["bitset_repeat_max"]:.2f} |')
    report+=['','## 完整准备＋渲染（每次视角重新准备）','',
        '| 场景/模式 | 条件 | K中位数 | 原总耗时ms | Bitset总耗时ms | 加速 | Bitset版准备/渲染ms | Warp版总耗时ms |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for v in pipe:
        report.append(f'| {v["scene"]} | {v["condition"]} | {v["K_median"]:g} | {v["original_total_ms"]:.4f} | {v["bitset_total_ms"]:.4f} | {v["total_speedup"]:.2f}× | {v["bitset_prepare_ms"]:.4f}/{v["bitset_render_ms"]:.4f} | {v["warp_total_ms"]:.4f} |')
    report+=['','各列分别取中位数，因此准备中位数与渲染中位数之和不保证等于总耗时中位数。完整推理计时不包括计时后的选点/图像比较开销。','',
        '## 逐项归因','',
        'Bitset的收益来自对K个Gaussian共享稳定深度排序，使用每Tile成员位表避免对大量展开实例反复排序，并省去实例总数的CPU回读。低K但大投影的情况最有优势。Full时位表清零/构建和扫描成本随K增长，可能抵消排序收益；这是本算法的自然适用边界，未使用按K绕开Full的人工分派。','',
        '| 场景/模式 | 条件 | bbox/有效Tile实例中位数 | Bitset/同入口控制加速 | 仅短列表/控制加速 | 组合/仅Warp加速 |',
        '|---|---|---|---:|---:|---:|']
    for v in rows:
        report.append(f'| {v["scene"]} | {v["condition"]} | {v["bbox_instances_median"]:g}/{v["valid_instances_median"]:g} | {v["bitset_vs_control"]:.2f}× | {v["short_vs_control"]:.3f}× | {v["warp_short_vs_warp"]:.3f}× |')
    report+=['','## Kernel与内存活动（独立Profiling）','',
        f'{profile_scope}。仅统计cat=kernel，排除GPU user annotation，Memcpy与Memset另列；Kernel累计活动不等于端到端延迟，当前此表Kernel耗时仅用于定位，不作为正式加速倍数依据。','',
        '| 条件 | 版本 | Kernel/帧 | Memcpy/帧 | Memset/帧 | Kernel μs/帧 |','|---|---|---:|---:|---:|---:|']
    for v in profiles:
        report.append(f'| {v["condition"]} | {v["variant"]} | {v["kernels_per_frame"]:g} | {v["copies_per_frame"]:g} | {v["memsets_per_frame"]:g} | {v["gpu_us"]:.2f} |')
    report+=['','## 完整性与交付','',
        f'本轮保护生产/参考文件及4个checkpoint输入共 {integrity["files"]} 项，变更数 {len(integrity["changed"])}；候选CUDA二进制和原SOURCE_MANIFEST均未变化。实际只增加测速、监控、分析脚本及报告，不修改生产模型、默认配置、共享conda或其他进程，未commit/push。','',
        '此前并发诊断的2.66–3.65×保留为历史证据。本次空闲复测是正式性能证据，两个环境的绝对时间差不应解释为代码再次优化，均使用同一候选二进制。','',
        'RASTERIZER_IDLE_RESULTS.csv记录所有版本和六次重复范围；PIPELINE_IDLE_RESULTS.csv记录实际推理结果。round*/samples.csv、workloads.csv、LOAD.json、外部MONITOR.json及trace文件保留原始证据。','']
    text='\n'.join(report)
    (OUT/'SPARSE_RASTERIZER_IDLE_BENCHMARK_REPORT.md').write_text(text,encoding='utf-8')
    (HERE/'SPARSE_RASTERIZER_IDLE_BENCHMARK_REPORT.md').write_text(text,encoding='utf-8')
    result=dict(passed=True,render_samples=len(inputs),paired_workloads=6720,pipeline_samples=len(p_rows),
        low_raster_speedup=[min(v['bitset_speedup'] for v in low),max(v['bitset_speedup'] for v in low)],
        low_pipeline_speedup=[min(v['total_speedup'] for v in lowpipe),max(v['total_speedup'] for v in lowpipe)],
        identical_workloads=True,integrity_passed=integrity['passed'],formal_idle_benchmark=True)
    (OUT/'RESULTS_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
