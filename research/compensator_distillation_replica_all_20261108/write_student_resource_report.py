"""蒸馏学生与原Compensator的总显存、DRAM访存配对报告。"""
from pathlib import Path
import csv
import io
import json
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS=ROOT/'reports'
MiB=2**20


def read(p):return json.loads(p.read_text())


def csv_out(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    mem=REPORTS/'student_memory_20261109';traffic=REPORTS/'student_traffic_20261109'
    complete=read(mem/'COMPLETE.json');assert complete['passed'] and complete['selection_ids_identical']
    memory_rows=[];paired=[];switch=[];checked=0
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd']:
        for mode in ['runtime_full','0.01','0.05','0.1','0.3','0.5','0.7','runtime_switch']:
            pair={}
            for version in ['teacher','student']:
                d=read(mem/f'{slug}__{version}__{mode}.json')
                rows=[r for r in d['rows'] if r['cycle']==1]
                assert len(rows)==(42 if mode=='runtime_switch' else 6)
                if version=='student':
                    expected=read(REPORTS/slug/'QUALITY_SUMMARY.json')['identities']['continuous_calibration']['sha256']
                    assert d['student_sha256']==expected
                else:assert d['student_checkpoint'] is None
                summary=dict(scene=d['scene'],mode=mode,version=version,
                    added_resident_MiB=d['resident_after_load']/MiB,loaded_active_MiB=d['loaded']['allocated']/MiB,
                    K_median=st.median(r['K'] for r in rows),
                    peak_total_max_MiB=max(r['whole_step_peak_allocated'] for r in rows)/MiB,
                    peak_total_median_MiB=st.median(r['whole_step_peak_allocated'] for r in rows)/MiB,
                    render_peak_delta_median_MiB=st.median(r['render_peak_delta'] for r in rows)/MiB,
                    retained_MiB=d['final']['allocated']/MiB,reserved_MiB=d['final']['reserved']/MiB,
                    student_sha256=d['student_sha256'])
                memory_rows.append(summary);pair[version]=(summary,d)
                if mode=='runtime_switch':
                    for m in ['runtime_full','0.01','0.1','0.7']:
                        g=[r for r in rows if r['mode']==m]
                        switch.append(dict(scene=d['scene'],version=version,mode=m,
                            total_peak_max_MiB=max(r['whole_step_peak_allocated'] for r in g)/MiB,
                            retained_median_MiB=st.median(r['retained_after'] for r in g)/MiB,
                            reserved_max_MiB=max(r['render_peak_reserved'] for r in g)/MiB))
            a,b=[pair[k][1]['rows'] for k in ['teacher','student']]
            assert len(a)==len(b)
            assert all(all(x[k]==y[k] for k in ['mode','cycle','frame','K','selection_sha256']) for x,y in zip(a,b))
            checked+=len(a)
            a,b=[pair[k][0] for k in ['teacher','student']]
            paired.append(dict(scene=a['scene'],mode=mode,teacher_resident_MiB=a['added_resident_MiB'],
                student_resident_MiB=b['added_resident_MiB'],resident_saved_KiB=(a['added_resident_MiB']-b['added_resident_MiB'])*1024,
                teacher_peak_MiB=a['peak_total_max_MiB'],student_peak_MiB=b['peak_total_max_MiB'],
                student_peak_change_pct=100*(b['peak_total_max_MiB']/a['peak_total_max_MiB']-1),
                teacher_render_peak_MiB=a['render_peak_delta_median_MiB'],student_render_peak_MiB=b['render_peak_delta_median_MiB']))
    csv_out(REPORTS/'STUDENT_RESOURCE_MEMORY.csv',memory_rows)
    csv_out(REPORTS/'STUDENT_RESOURCE_MEMORY_PAIRED.csv',paired)
    csv_out(REPORTS/'STUDENT_RESOURCE_SWITCH.csv',switch)
    traffic_rows=[]
    for p in sorted(traffic.glob('*.csv')):
        d=read(p.with_suffix('.json'));text=p.read_text();assert '==ERROR==' not in text,p
        lines=text.splitlines();idx=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[idx:]))))
        sums={k:0. for k in ['dram__bytes_read.sum','dram__bytes_write.sum','lts__t_bytes.sum']};seen=set()
        for r in rows:
            assert r['Metric Unit']=='byte'
            key=(r['ID'],r['Metric Name']);assert key not in seen;seen.add(key)
            value=float(r['Metric Value'].replace(',',''));assert value>=0;sums[r['Metric Name']]+=value
        assert len(seen)==3*len({r['ID'] for r in rows})
        version='student' if d['student_checkpoint'] else 'teacher'
        traffic_rows.append(dict(scene=d['scene'],version=version,ratio=d['mode'],scope=d['scope'],
            repeat=2 if p.stem.startswith('repeat2_') else 1,
            frame=d['frame'],K=d['K'],selection_sha256=d['selection_sha256'],
            kernel_launches=len({r['ID'] for r in rows}),dram_read_MiB=sums['dram__bytes_read.sum']/MiB,
            dram_write_MiB=sums['dram__bytes_write.sum']/MiB,
            dram_total_MiB=(sums['dram__bytes_read.sum']+sums['dram__bytes_write.sum'])/MiB,
            l2_requested_MiB=sums['lts__t_bytes.sum']/MiB))
    assert len(traffic_rows)==32,len(traffic_rows)
    traffic_pairs=[]
    for a in traffic_rows:
        if a['version']!='teacher':continue
        b=next(r for r in traffic_rows if r['version']=='student' and all(r[k]==a[k] for k in ['scene','ratio','scope','frame','K','selection_sha256','repeat']))
        traffic_pairs.append(dict(scene=a['scene'],ratio=a['ratio'],scope=a['scope'],repeat=a['repeat'],K=a['K'],
            teacher_dram_MiB=a['dram_total_MiB'],student_dram_MiB=b['dram_total_MiB'],
            student_change_pct=100*(b['dram_total_MiB']/a['dram_total_MiB']-1)))
    csv_out(REPORTS/'STUDENT_RESOURCE_DRAM.csv',traffic_rows)
    csv_out(REPORTS/'STUDENT_RESOURCE_DRAM_PAIRED.csv',traffic_pairs)
    lines=['# 蒸馏Compensator显存与访存实测','','## 结论','',
        '蒸馏学生确实降低MLP计算量，但当前完整部署实例的常驻显存只减少约0.067MiB（68.5KiB）。它不替换或卸载完整Master、Anchor Feature、索引/排序Workspace和Full发布包，因此不能解决此前动态LoD总显存高于轻量Full的问题。','',
        '固定相同地图、Selector、Anchor Feature、相机、Ratio和选点，比较原64/64四个两层Head与最终32/32四个直接Head学生。新增32个独立子进程测显存，32个NCU捕获测DRAM字节。所有显存请求逐个核对实际K及选点ID哈希，学生SHA与本轮质量评价一致。没有重训。','',
        'DRAM访问收益不普遍：office2的total范围在1%两轮减少约19%–22%、70%减少约8.5%；room0的70%减少约4.3%–5.7%，1%两轮变化约+0.17%和−3.12%，10%单次增加约0.85%。因此不能称所有场景/Ratio都有稳定访存下降，也不能将这些比例当作实际带宽速度提升。','',
        '## 常驻和最高总显存','',
        '下表是每个条件第二轮六视角中“常驻＋临时”的最高活跃显存，不是逐帧峰值中位数；MiB=2^20字节。加载常驻增量扣除相同初始化基线，总峰值包含该基线。','',
        '| 场景 | 模式 | Teacher常驻增量 MiB | Student常驻增量 MiB | 常驻节省 KiB | Teacher最高总显存 MiB | Student最高总显存 MiB | 学生峰值变化 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f'| {r["scene"]} | {r["mode"]} | {r["teacher_resident_MiB"]:.3f} | {r["student_resident_MiB"]:.3f} | {r["resident_saved_KiB"]:.1f} | {r["teacher_peak_MiB"]:.3f} | {r["student_peak_MiB"]:.3f} | {r["student_peak_change_pct"]:+.2f}% |')
    lines+=['','网络原始权重25742→3534参数，FP32参数字节理论节省88832字节；部署实测不等于该值，因为现有实例仍保留Teacher模型对象，并将执行权重替换为转置后的学生参数，还保留学生对象。仅改网络执行宽度，不会按比例缩小大地图数据与准备工作区。','',
        '局部渲染临时显存还取决于学生预测的Scale/Rotation/XYZ及Tile覆盖。即使K与ID完全一致，学生和Teacher图像不相同，临时峰值可以略增也可以略减；不能把全部峰值差解释成网络激活显存。','',
        '## 与CaRtGS原版和我们的轻量Full比较','',
        '| 场景 | CaRtGS原版最高峰值（三次取最大） MiB | 我们的轻量Full最高峰值 MiB | 蒸馏LoD1%最高峰值 MiB |',
        '|---|---:|---:|---:|']
    old=list(csv.DictReader((REPORTS/'VANILLA_VS_OURS_MEMORY.csv').open(encoding='utf-8-sig')))
    for name in ['office2','room0']:
        original=max(float(r['whole_peak_max_MiB']) for r in old if r['version']=='cartgs_full' and '/'+name+'/' in r['scene'])
        full=float(next(r for r in old if r['version']=='full_map_only' and '/'+name+'/' in r['scene'])['whole_peak_max_MiB'])
        student=next(r for r in paired if '/'+name+'/' in r['scene'] and r['mode']=='0.01')['student_peak_MiB']
        lines.append(f'| {name} RGB-D | {original:.3f} | {full:.3f} | {student:.3f} |')
    lines+=['','Full历史对照使用同一推理Renderer和相同RGB帧，原版地图使用各自保存SLAM位姿；它不代表原生C++ Viewer完整进程的驱动占用。详情见VANILLA_VS_OURS_MEMORY_REPORT.md。','',
        '## DRAM访问量配对','',
        'Nsight Compute 2022.3，dram__bytes_read.sum与dram__bytes_write.sum之和。render仅渲染，total含Selector、Compensator、打包和渲染捕获到的CUDA Kernel；不包含Copy Engine设备复制。五次预热后每场景中间一个相机捕获一次，1%/10%/70%，双方实际K和ID一致。','',
        '首轮覆盖全部条件，第二轮独立进程复测1%/70%的total范围。小幅差异同时受几何覆盖、缓存和Profiling影响，不能仅凭单帧几个百分点的变化建立机制因果。','',
        '| 场景 | Ratio | 范围 | 轮次 | 实际K | Teacher DRAM MiB | Student DRAM MiB | 学生变化 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in traffic_pairs:
        lines.append(f'| {r["scene"]} | {r["ratio"]} | {r["scope"]} | {r["repeat"]} | {r["K"]} | {r["teacher_dram_MiB"]:.2f} | {r["student_dram_MiB"]:.2f} | {r["student_change_pct"]:+.2f}% |')
    lines+=['','该表是访存字节量，不是显存容量、GB/s或帧时间。参数与MAC缩小比例不等于DRAM流量缩小比例，Renderer和地图数据访问占据独立成本。学生渲染覆盖变化也会影响流量。','',
        '## 口径、质量和完整性','',
        '显存采用独立进程的PyTorch CUDA allocated/peak，reserved单独保存；不含全部CUDA Context、驱动及第三方直接分配。两场景各六视角、两轮，另外测完整Ratio切换的Graph/Workspace保留。未使用训练的Adam状态。','',
        'NCU使用kernel replay、cache-control none、clock-control none，管理员权限仅运行Profiler，无驱动配置改动。共享环境和缓存状态可能影响DRAM计数，单视角数据不能外推全轨迹均值，Profiler期间耗时不用于测速。','',
        '本轮没有新增速度测量。此前空闲3090配对结果中，Compensator组件1.39–2.93×、准备流程1.02–1.46×、准备＋渲染1.006–1.115×。蒸馏主要体现为计算收益，常驻显存不是当前主要收益。','',
        'Student非Teacher数值等价替代：Replica16组全局平均PSNR损失约0.154dB、局部约0.276dB，严格质量门限未通过。本轮资源测试不能掩盖这些质量代价，也不修改生产默认。','',
        '后续若以常驻显存为目标，应单独研究删除部署快照中的未使用Teacher权重、冗余Master/Full副本及按需工作区；这些改动均未在本轮执行。','',
        '文件：student_memory_20261109保存32进程原始记录；student_traffic_20261109保存32个硬件计数捕获；STUDENT_RESOURCE_MEMORY_PAIRED.csv和STUDENT_RESOURCE_DRAM_PAIRED.csv为主要对照。','']
    report='\n'.join(lines)
    (REPORTS/'STUDENT_MEMORY_AND_TRAFFIC_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':(HERE/'STUDENT_MEMORY_AND_TRAFFIC_REPORT.md').write_text(report,encoding='utf-8')
    validation=dict(passed=True,memory_workers=32,matched_memory_request_pairs=checked,
        hardware_captures=len(traffic_rows),hardware_pairs=len(traffic_pairs),selection_ids_identical=True,
        student_checkpoint_sha_verified=True,models_changed=False,speed_measured=False,
        copy_engine_included=False)
    (REPORTS/'STUDENT_RESOURCE_VALIDATION.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
