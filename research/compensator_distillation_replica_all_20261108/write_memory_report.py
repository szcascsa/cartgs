"""显存与硬件访存计数汇总；不将访存字节量当作带宽速度。"""
from pathlib import Path
import csv
import io
import json
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS=ROOT/'reports'
MiB=2**20


def read(path):return json.loads(path.read_text())


def csv_out(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    memory=REPORTS/'lod_memory_20261109';traffic=REPORTS/'lod_traffic_20261109'
    assert read(memory/'COMPLETE.json')['passed']
    memory_rows=[];switch_rows=[]
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd']:
        for mode in ['full_map_only','runtime_full','0.01','0.05','0.1','0.3','0.5','0.7','runtime_switch']:
            d=read(memory/f'{slug}__{mode}.json');rows=[r for r in d['rows'] if r['cycle']==1]
            assert len(rows)==(42 if mode=='runtime_switch' else 6)
            assert d['saved_images']==0
            memory_rows.append(dict(scene=d['scene'],worker_mode=mode,
                initialized_baseline_MiB=d['initial']['allocated']/MiB,
                loaded_allocated_MiB=d['loaded']['allocated']/MiB,
                added_resident_MiB=d['resident_after_load']/MiB,
                K_median=st.median(r['K'] for r in rows),K_min=min(r['K'] for r in rows),K_max=max(r['K'] for r in rows),
                render_peak_delta_median_MiB=st.median(r['render_peak_delta'] for r in rows)/MiB,
                render_peak_delta_max_MiB=max(r['render_peak_delta'] for r in rows)/MiB,
                whole_peak_median_MiB=st.median(r['whole_step_peak_allocated'] for r in rows)/MiB,
                whole_peak_max_MiB=max(r['whole_step_peak_allocated'] for r in rows)/MiB,
                final_retained_MiB=d['final']['allocated']/MiB,final_reserved_MiB=d['final']['reserved']/MiB,
                logical_packet_median_MiB=st.median(r['logical_input_bytes'] for r in rows)/MiB,
                storage_packet_median_MiB=st.median(r['storage_input_bytes'] for r in rows)/MiB,
                graphs_max=max(r['graph_count'] for r in rows)))
            if mode=='runtime_switch':
                for m in ['runtime_full','0.01','0.05','0.1','0.3','0.5','0.7']:
                    g=[r for r in rows if r['mode']==m]
                    switch_rows.append(dict(scene=d['scene'],mode=m,
                        whole_peak_median_MiB=st.median(r['whole_step_peak_allocated'] for r in g)/MiB,
                        retained_median_MiB=st.median(r['retained_after'] for r in g)/MiB,
                        reserved_max_MiB=max(r['render_peak_reserved'] for r in g)/MiB,
                        graphs_max=max(r['graph_count'] for r in g)))
    csv_out(REPORTS/'LOD_MEMORY_SUMMARY.csv',memory_rows)
    csv_out(REPORTS/'LOD_SWITCH_MEMORY_SUMMARY.csv',switch_rows)
    traffic_rows=[];kernels=[]
    for p in sorted(traffic.glob('*.csv')):
        metadata=read(p.with_suffix('.json'))
        text=p.read_text();assert '==ERROR==' not in text,(p,text[-1000:])
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[i:]))))
        sums={k:0. for k in ['dram__bytes_read.sum','dram__bytes_write.sum','lts__t_bytes.sum']}
        seen=set()
        for r in rows:
            assert r['Metric Unit']=='byte',r
            key=(r['ID'],r['Metric Name']);assert key not in seen,key;seen.add(key)
            value=float(r['Metric Value'].replace(',',''));assert value>=0
            sums[r['Metric Name']]+=value
            kernels.append(dict(capture=p.stem,scene=metadata['scene'],mode=metadata['mode'],scope=metadata['scope'],
                kernel_id=r['ID'],kernel=r['Kernel Name'],metric=r['Metric Name'],bytes=value))
        assert all(len([k for k in seen if k[0]==ident])==3 for ident in {r['ID'] for r in rows})
        traffic_rows.append(dict(capture=p.stem,scene=metadata['scene'],mode=metadata['mode'],scope=metadata['scope'],
            repeat=2 if p.stem.startswith('repeat2_') else 1,frame=metadata['frame'],K=metadata['K'],
            candidate_N=metadata['candidate_N'],kernel_launches=len({r['ID'] for r in rows}),
            dram_read_MiB=sums['dram__bytes_read.sum']/MiB,dram_write_MiB=sums['dram__bytes_write.sum']/MiB,
            dram_total_MiB=(sums['dram__bytes_read.sum']+sums['dram__bytes_write.sum'])/MiB,
            l2_requested_MiB=sums['lts__t_bytes.sum']/MiB))
    assert len(traffic_rows)==24,len(traffic_rows)
    csv_out(REPORTS/'LOD_DRAM_TRAFFIC_SUMMARY.csv',traffic_rows)
    csv_out(REPORTS/'LOD_DRAM_KERNEL_METRICS.csv',kernels)
    mi={(r['scene'],r['worker_mode']):r for r in memory_rows}
    lines=['# 网络LoD减少点数的显存与访存收益实测','','## 结论','',
        '同checkpoint测试Replica office2 RGB-D（43,197点）与room0 RGB-D（79,351点）。低Ratio确实显著减少Rasterizer临时显存和Kernel的DRAM访问量；动态网络不会卸载完整Master、Feature和工作区，常驻显存基本不随K下降。若与只加载Full渲染属性的轻量路径比较，当前动态LoD实例的总显存仍可能更高。','',
        '本次只增加研究脚本和报告，复用原64/64 Compensator、Selector、v31部署二进制及原Rasterizer，没有修改训练、权重、生产默认或渲染内核，没有保存渲染图片。学生蒸馏不是本轮变量。','',
        '## 显存测量方法','',
        '18个独立子进程：每场景分别测试Full属性包、完整部署实例Full、六档局部Ratio及连续切换。每进程六个相机×两轮，表取第二轮六视角统计。固定1200×680、FP32。原未补偿Master为Full基线，不把ratio=1补偿当作Full。','',
        '使用torch.cuda.memory_allocated/max_memory_allocated记录活跃分配，reserved另列。每次在准备和渲染前分别同步、reset_peak_memory_stats。渲染新增峰值=render阶段峰值减去已准备属性的活跃分配，包含RGB/radii输出、预处理、Tile展开/排序及其他Renderer临时空间。并非仅输出Tensor大小，也不是峰值减去reserved。','',
        '加载常驻增量扣除相机/框架初始化基线约8.14MiB；全流程峰值包含该基线。PyTorch allocator统计不含全部CUDA Context、驱动及第三方直接cudaMalloc占用，不能直接当作nvidia-smi整进程数值。其他训练进程显存不会计入本进程allocator。','',
        '## 同一动态部署实现：Full与各Ratio','',
        '| 场景 | 模式 | 实际K中位数 [最小,最大] | 加载常驻增量 MiB | 渲染新增峰值中位数 MiB | 全流程活跃峰值中位数 MiB | 六视角最大活跃峰值 MiB |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in memory_rows:
        if r['worker_mode'] in ['full_map_only','runtime_switch']:continue
        lines.append(f'| {r["scene"]} | {r["worker_mode"]} | {r["K_median"]:g} [{r["K_min"]},{r["K_max"]}] | {r["added_resident_MiB"]:.2f} | {r["render_peak_delta_median_MiB"]:.2f} | {r["whole_peak_median_MiB"]:.2f} | {r["whole_peak_max_MiB"]:.2f} |')
    lines+=['','例如room0在同一个动态部署架构下，从Full到局部1%，渲染新增峰值中位数74.52→16.10MiB；全流程活跃峰值中位数204.56→126.84MiB。完整Master和Feature仍驻留，加载常驻增量102.53MiB保持不变。','',
        '## 与轻量Full路径的公平区分','',
        '| 场景 | 仅Full属性加载增量 MiB | 动态LoD加载增量 MiB | 轻量Full活跃峰值中位数 MiB | 动态局部1%活跃峰值中位数 MiB |',
        '|---|---:|---:|---:|---:|']
    for scene in ['Replica/office2/rgbd','Replica/room0/rgbd']:
        a=mi[(scene,'full_map_only')];b=mi[(scene,'0.01')]
        lines.append(f'| {scene} | {a["added_resident_MiB"]:.2f} | {b["added_resident_MiB"]:.2f} | {a["whole_peak_median_MiB"]:.2f} | {b["whole_peak_median_MiB"]:.2f} |')
    lines+=['','轻量Full只保留同一个Master激活后的最终推理属性，释放加载时原始训练属性副本，不加载Selector、Feature及部署准备工作区；动态路径使用现有DeploymentRenderer完整实例。两者是部署功能不同的实现对照，不把差值全部归于MLP参数。当前完整实例还保留Master原始与激活快照、Full发布包、Anchor Feature、静态前缀、索引/排序工作区等。','',
        '## 来回切换和Allocator缓存','',
        '| 场景 | 模式 | 热态活跃峰值中位数 MiB | 请求后保留活跃分配 MiB | 当次峰值reserved最大 MiB | Graph数最大 |',
        '|---|---|---:|---:|---:|---:|']
    for r in switch_rows:
        if r['mode'] not in ['runtime_full','0.01','0.1','0.7']:continue
        lines.append(f'| {r["scene"]} | {r["mode"]} | {r["whole_peak_median_MiB"]:.2f} | {r["retained_median_MiB"]:.2f} | {r["reserved_max_MiB"]:.2f} | {r["graphs_max"]} |')
    lines+=['','office2经历全部Ratio后保留更多Graph及相关存储，allocated仍受加载状态与缓存容量影响；room0大地图走原Host计数路径，未使用同类Graph。reserved是PyTorch复用池，曾经跑过Full后不自动按K缩小。不能用nvidia-smi未下降就否定临时显存节省，也不能把缓存池全称为活跃模型数据。','',
        '## Nsight Compute的真实访存计数','',
        '先复现普通用户ERR_NVGPUCTRPERM（RmProfilingAdminOnly=1），再使用既有授权以sudo运行Profiler，成功读取计数器；未修改驱动参数。Nsight Compute 2022.3，指标dram__bytes_read.sum、dram__bytes_write.sum、lts__t_bytes.sum，kernel replay、cache-control none、clock-control none。每场景中间一个相机，固定相同checkpoint和原分辨率；五次预热后只捕获一次请求。','',
        'render范围仅含已准备属性的渲染；total范围含现有部署Full打包或Selector/Compensator/准备再渲染。下面数据是捕获CUDA Kernel的DRAM读写总量，不是DRAM容量，不是GB/s，也不包含未由Kernel完成的cudaMemcpy/Copy Engine流量。Full发布包的clone使用设备复制，其流量不能从这些Kernel计数补全，因此total列不能称为整卡全部访存。L2 requested与DRAM属于不同层级，不能相加。','',
        '捕获期间服务器有其他训练，Nsight会改变执行调度并可能串行化目标Kernel，本报告不以Profiling耗时作速度比较。缓存状态与共享环境可能影响物理DRAM读写；关键条件另起进程独立重复，数值见复测表。','',
        '| 场景 | 模式 | 实际K | 范围 | Kernel数 | DRAM读取 MiB | DRAM写入 MiB | 合计 MiB | 相对同范围Full变化 |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    primary=[r for r in traffic_rows if r['repeat']==1]
    for r in primary:
        base=next(v for v in primary if v['scene']==r['scene'] and v['mode']=='full' and v['scope']==r['scope'])
        change=100*(r['dram_total_MiB']/base['dram_total_MiB']-1)
        lines.append(f'| {r["scene"]} | {r["mode"]} | {r["K"]} | {r["scope"]} | {r["kernel_launches"]} | {r["dram_read_MiB"]:.2f} | {r["dram_write_MiB"]:.2f} | {r["dram_total_MiB"]:.2f} | {change:+.1f}% |')
    lines+=['','低Ratio渲染Kernel访问量显著下降，加入网络准备后仍保留大部分收益。room0局部70%的total捕获Kernel读写量却比同范围Full高，说明网络准备可能抵消高Ratio的访问节省；由于设备复制流量未覆盖，不能把这一项推广为完整请求总DRAM必然增加。','',
        '## 独立重复计数','',
        '| 场景 | 模式 | 范围 | 首次 MiB | 复测 MiB | 变化 |',
        '|---|---|---|---:|---:|---:|']
    for r in traffic_rows:
        if r['repeat']!=2:continue
        base=next(v for v in primary if all(v[k]==r[k] for k in ['scene','mode','scope','K','frame']))
        lines.append(f'| {r["scene"]} | {r["mode"]} | {r["scope"]} | {base["dram_total_MiB"]:.2f} | {r["dram_total_MiB"]:.2f} | {100*(r["dram_total_MiB"]/base["dram_total_MiB"]-1):+.2f}% |')
    lines+=['','## 实际含义与限制','',
        '已经证实：在这两张冻结地图上，网络降低点数减少了Renderer临时显存和低Ratio的Kernel访存量；不能因此推出常驻地图显存同比缩小，也不能把访存下降比例当作帧时间加速比例。当前桌面3090未必受DRAM带宽限制；端侧设备可能更受益，但本轮未测试端侧硬件。','',
        'Ratio改变同时改变选点、几何补偿和渲染质量，因此本实验是实际部署资源对照，不是仅改变点数的纯因果消融。本轮使用原Teacher网络以隔离网络减少点数的资源收益，不将学生蒸馏质量损失混入。显存覆盖六相机，硬件计数只覆盖每场景一个中间相机，不能作为全轨迹平均。','',
        '最直接的后续优化是消除部署实例的冗余Master/Full副本、按需要分配工作区，以及明确Graph缓存上限；它们可减少当前常驻成本。若要让常驻显存随LoD下降，仍需固定LoD导出或按需加载地图块。以上均未在本轮实现。','',
        '原始文件：lod_memory_20261109（18个独立进程），lod_traffic_20261109（24个NCU捕获）；汇总为LOD_MEMORY_SUMMARY.csv、LOD_SWITCH_MEMORY_SUMMARY.csv、LOD_DRAM_TRAFFIC_SUMMARY.csv、LOD_DRAM_KERNEL_METRICS.csv。','']
    report='\n'.join(lines)
    (REPORTS/'LOD_MEMORY_AND_TRAFFIC_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':(HERE/'LOD_MEMORY_AND_TRAFFIC_REPORT.md').write_text(report,encoding='utf-8')
    save_result=dict(memory_workers=len(memory_rows),memory_view_requests=sum(84 if r['worker_mode']=='runtime_switch' else 12 for r in memory_rows),
        hardware_captures=len(traffic_rows),hardware_metric_rows=len(kernels),passed=True,
        units='MiB=2^20 bytes',cuda_copy_engine_included=False,models_or_renderer_changed=False,
        speed_claimed=False,full_trajectory_traffic_claimed=False)
    (REPORTS/'LOD_MEMORY_TRAFFIC_VALIDATION.json').write_text(json.dumps(save_result,indent=2))
    print(json.dumps(save_result,indent=2))


if __name__=='__main__':main()
