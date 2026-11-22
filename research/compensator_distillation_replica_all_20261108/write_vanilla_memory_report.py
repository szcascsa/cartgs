"""原版三次地图、我们的Full和完整动态LoD的显存对照。"""
from pathlib import Path
import csv
import json
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS=ROOT/'reports'
MiB=2**20


def read(path):return json.loads(path.read_text())


def stats(d):
    rows=[r for r in d['rows'] if r['cycle']==1]
    assert len(rows)==6
    return dict(scene=d['scene'],map_N=rows[0]['map_N'],loaded_allocated_MiB=d['loaded']['allocated']/MiB,
        added_resident_MiB=d['resident_after_load']/MiB,
        render_peak_delta_median_MiB=st.median(r['render_peak_delta'] for r in rows)/MiB,
        whole_peak_median_MiB=st.median(r['whole_step_peak_allocated'] for r in rows)/MiB,
        whole_peak_max_MiB=max(r['whole_step_peak_allocated'] for r in rows)/MiB,
        retained_MiB=d['final']['allocated']/MiB,reserved_MiB=d['final']['reserved']/MiB,
        K_median=st.median(r['K'] for r in rows))


def main():
    folder=REPORTS/'vanilla_memory_20261109'
    assert read(folder/'COMPLETE.json')['passed']
    allrows=[];summary=[]
    lines=['# CaRtGS原版、我们的Full与动态LoD显存比较','','## 结论','',
        '在相同冻结推理Renderer口径下，当前动态LoD的全流程活跃显存峰值没有低于CaRtGS原版Full，也没有低于我们自己的轻量Full；它只是在已加载完整动态网络的实例内部，相比调用该实例的Full降低Renderer临时峰值。','',
        '不能把网络实例内Full的较高显存当作原版CaRtGS的成本，也不能用仅Rasterizer临时峰值代表完整部署显存。以下全部MiB为2^20字节，均已包含相同相机/框架初始化的PyTorch活跃分配基线。','',
        '## 相同口径汇总','',
        '| 场景 | 版本 | 地图N | 加载后常驻活跃分配 MiB | 每帧活跃总峰值中位数 MiB |',
        '|---|---|---|---:|---:|']
    for scene in ['office2','room0']:
        original=[]
        for p in sorted(folder.glob(f'Vanilla__{scene}__rgbd__*.json')):
            d=read(p);assert d['matched_rgb_frames']
            row=stats(d);row['version']='cartgs_full';row['checkpoint']=d['case']['checkpoint']
            original.append(row);allrows.append(row)
        assert len(original)==3
        mean_resident=st.mean(r['loaded_allocated_MiB'] for r in original)
        mean_peak=st.mean(r['whole_peak_median_MiB'] for r in original)
        lines.append(f'| {scene} RGB-D | CaRtGS原版Full（三次） | {min(r["map_N"] for r in original):,}–{max(r["map_N"] for r in original):,} | {mean_resident:.2f} [{min(r["loaded_allocated_MiB"] for r in original):.2f}–{max(r["loaded_allocated_MiB"] for r in original):.2f}] | {mean_peak:.2f} [{min(r["whole_peak_median_MiB"] for r in original):.2f}–{max(r["whole_peak_median_MiB"] for r in original):.2f}] |')
        slug=f'Replica__{scene}__rgbd'
        own={}
        for mode,label in [('full_map_only','我们的轻量Full'),('runtime_full','加载完整网络后调用Full'),
                ('0.01','完整网络＋局部1%'),('0.1','完整网络＋局部10%'),('0.7','完整网络＋局部70%')]:
            d=read(REPORTS/'lod_memory_20261109'/f'{slug}__{mode}.json');row=stats(d)
            row.update(version=mode,checkpoint=next(s['checkpoint'] for s in read(REPORTS/'SCENES.json') if s['slug']==slug))
            allrows.append(row);own[mode]=row
            lines.append(f'| {scene} RGB-D | {label} | {row["map_N"]:,} | {row["loaded_allocated_MiB"]:.2f} | {row["whole_peak_median_MiB"]:.2f} |')
        low=own['0.01'];full=own['full_map_only'];runtime=own['runtime_full']
        summary.append(dict(scene=scene,vanilla_mean_peak_MiB=mean_peak,vanilla_mean_resident_MiB=mean_resident,
            own_full_peak_MiB=full['whole_peak_median_MiB'],own_lod1_peak_MiB=low['whole_peak_median_MiB'],
            lod1_peak_change_vs_vanilla_pct=100*(low['whole_peak_median_MiB']/mean_peak-1),
            lod1_peak_change_vs_own_full_pct=100*(low['whole_peak_median_MiB']/full['whole_peak_median_MiB']-1),
            lod1_peak_change_vs_loaded_runtime_full_pct=100*(low['whole_peak_median_MiB']/runtime['whole_peak_median_MiB']-1)))
    lines+=['','原版汇总以三次已完成seed0运行分别测量，每次先取六视角总峰值中位数，再对三次取平均，方括号为三次范围；不是按结果挑选最有利的地图。我们的模型沿用本轮已锁定Teacher，不重新选择或训练。','',
        '## 三种比较的含义','',
        '| 场景 | 动态1%相对原版Full峰值变化 | 相对我们的轻量Full | 相对已加载网络实例的Full |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["scene"]} | {r["lod1_peak_change_vs_vanilla_pct"]:+.1f}% | {r["lod1_peak_change_vs_own_full_pct"]:+.1f}% | {r["lod1_peak_change_vs_loaded_runtime_full_pct"]:+.1f}% |')
    lines+=['','原版与我们的地图数量不同，渲染覆盖、Gaussian Scale和各自SLAM位姿也不同；不能从该表推断相同点数的Renderer性能，或把全部差异归于选择网络。它回答的是当前已保存模型在统一推理口径下的实际资源需求。','',
        '低Ratio减小了Gaussian属性输出包、预处理/排序临时空间；但当前部署实例还保留原始与激活Master、Full发布副本、Anchor Feature、索引/排序Workspace、Selector静态前缀等。网络MLP权重本身很小，不能把数十MiB额外占用称为MLP参数量。','',
        '只比较常驻分配时，当前动态实例更高；只比较同一动态实例的临时渲染峰值时，低Ratio更低。这两项可以同时成立。若目标是降低整个部署显存，需要先消除冗余地图副本/按需工作区，或使用固定LoD导出及分块加载；本轮未修改这些实现。','',
        '## 测量与边界','',
        '沿用LOD_MEMORY_AND_TRAFFIC_REPORT.md的显存工具：每条件独立进程、六相机、两轮，统计热态第二轮；1200×680、FP32、原Rasterizer推理入口，无训练、无图像落盘。原版六个checkpoint补测共72次渲染，加上此前18进程360次请求。','',
        '原版使用各自CameraTrajectory_TUM保存的SLAM位姿，与我们六个测试视角逐个匹配相同RGB帧编号；不把我们的相机外参直接套到原版地图坐标系。原版三次PLY、相机和轨迹SHA前后保持一致。','',
        '轻量Full路径只加载并保留最终Gaussian推理属性，不加载网络、不保留训练态原始属性副本。它对原版和我们的Full完全相同，属于公平的冻结推理实现对照；并未实测原生CaRtGS C++ Viewer整个进程的nvidia-smi占用。','',
        '统计使用PyTorch CUDA allocator活跃分配，不包含CUDA Context、驱动及所有第三方直接分配；reserved缓存池另存CSV。服务器存在其他训练，本进程显存统计可独立读取，本轮不报告新的速度结果。质量没有重新匹配，显存收益不能代替PSNR/SSIM/LPIPS验证。','',
        '本次仍只覆盖两个Replica RGB-D场景，不能推广到全部场景。三次原版地图和原始结果全部保留，不覆盖生产默认、不改网络或渲染内核、不commit/push。','',
        '文件：VANILLA_VS_OURS_MEMORY.csv保存所有原版重复和我们的配置；VANILLA_VS_OURS_MEMORY_SUMMARY.json保存计算口径；vanilla_memory_20261109保存原始记录、共同RGB帧及输入SHA。','']
    report='\n'.join(lines)
    (REPORTS/'VANILLA_VS_OURS_MEMORY_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':(HERE/'VANILLA_VS_OURS_MEMORY_REPORT.md').write_text(report,encoding='utf-8')
    with (REPORTS/'VANILLA_VS_OURS_MEMORY.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (REPORTS/'VANILLA_VS_OURS_MEMORY_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
