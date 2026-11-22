from pathlib import Path
import json
import csv
import statistics


STUDY=Path(__file__).resolve().parent
REPORTS=STUDY/'results/reports'
SHARED=STUDY.parent/'deployment_compact_kernels_20261107/results/reports'


def read(name,root=REPORTS):
    return json.loads((root/name).read_text(encoding='utf-8'))


def main():
    assert read('ACCEPTANCE_COMPLETE.json')['passed']
    results={};records=[]
    for repeat in [1,2,3]:
        for row in read(f'BENCHMARK_{repeat}.json')['results']:
            results.setdefault((row['group'],row['condition']),[]).append(row)
            a,b=[row['summary'][n] for n in ['v21','v27']]
            diff=[x-y for x,y in zip(row['samples']['v21']['prepare_ms'],row['samples']['v27']['prepare_ms'])]
            records.append(dict(group=row['group'],condition=row['condition'],repeat=repeat,
                N_min=min(x['N'] for x in row['actual']),N_max=max(x['N'] for x in row['actual']),
                K_min=min(x['K'] for x in row['actual']),K_max=max(x['K'] for x in row['actual']),
                v21_prepare_ms=a['prepare_ms']['median'],v27_prepare_ms=b['prepare_ms']['median'],
                prepare_reduction_percent=100*(1-b['prepare_ms']['median']/a['prepare_ms']['median']),
                v21_total_ms=a['total_ms']['median'],v27_total_ms=b['total_ms']['median'],
                total_reduction_percent=100*(1-b['total_ms']['median']/a['total_ms']['median']),
                v21_p90_ms=a['prepare_ms']['p90'],v27_p90_ms=b['prepare_ms']['p90'],
                v21_p99_ms=a['prepare_ms']['p99'],v27_p99_ms=b['prepare_ms']['p99'],
                paired_saved_us=1000*statistics.median(diff),faster_fraction=sum(x>0 for x in diff)/len(diff),
                gpu_start_sm_mhz=row['gpu_before']['gpu']['clocks.sm'],gpu_end_sm_mhz=row['gpu_after']['gpu']['clocks.sm']))
    with (STUDY/'IDLE_RETEST_PERFORMANCE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    table=[]
    def values(rows,metric):
        return [statistics.median(x['summary'][tag][metric]['median'] for x in rows) for tag in ['v21','v27']]
    for (group,condition),rows in results.items():
        assert len(rows)==3
        a,b=values(rows,'prepare_ms');x,y=values(rows,'total_ms')
        table.append(f'| {group} | {condition} | {a:.4f} | {b:.4f} | {100*(1-b/a):+.1f}% | {x:.4f} | {y:.4f} | {100*(1-y/x):+.1f}% |')
    profile={x['label']:x for x in read('PREPARE_PROFILE_ANALYSIS.json')['summary']}
    kernels=[];nodes=[]
    for group in ['tum','replica']:
        for condition in ['1000','5000','70pct']:
            a,b=[profile[f'prepare/{n}/{group}/{condition}'] for n in ['v21','v27']]
            component=lambda row:sum(x['gpu_us'] for x in row['top'] if 'compensation' in x['name'])
            ca,cb=component(a),component(b)
            kernels.append(f"| {group}/{condition} | {a['kernel_us']:.2f} | {b['kernel_us']:.2f} | {ca:.2f} | {cb:.2f} | {100*(1-cb/ca):+.1f}% |")
            nodes.append(f"| {group}/{condition} | {a['kernel_nodes']:g}→{b['kernel_nodes']:g} | {a['cpu_launches']:g}→{b['cpu_launches']:g} | {a['syncs']:g}→{b['syncs']:g} | {a['dtoh_bytes']:g}→{b['dtoh_bytes']:g} |")
    old={}
    for i in [1,2,3]:
        for row in read(f'BENCHMARK_{i}.json',SHARED)['results']:old.setdefault((row['group'],row['condition']),[]).append(row)
    compared=[]
    for group in ['tum','replica']:
        for condition in ['1000','5000','70pct']:
            shared=old[(group,condition)];idle=results[(group,condition)]
            for tag in ['v21','v27']:
                oldtag='optimized' if tag=='v27' else tag
                a=statistics.median(x['summary'][oldtag]['prepare_ms']['median'] for x in shared)
                b=statistics.median(x['summary'][tag]['prepare_ms']['median'] for x in idle)
                compared.append(f'| {group}/{condition} | {tag} | {a:.4f} | {b:.4f} |')
    audit=read('EXCLUSIVE_GPU_AUDIT.json');validation=read('VALIDATION.json');identity=read('BINARY_IDENTITY.json');integrity=read('FINAL_INTEGRITY.json')
    gpu=[r[k]['gpu'] for rows in results.values() for r in rows for k in ['gpu_before','gpu_after']]
    frequency=[x['clocks.sm'] for x in gpu];temperature=[x['temperature.gpu'] for x in gpu]
    protection=[]
    for name in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        row=integrity[name];protection.append(f"| {name} | {row['files']} | {len(row['changed'])} |")
    text='''# 空闲 GPU 下 v21 与 v27 部署推理复测

复测日期为 2026-11-07。用户清空其他作业后，启动前实测 GPU 利用率 0%、显存 24 MiB、计算进程列表为空。本轮对 v21 与 v27 采用相同输入做三轮配对测试，并独立执行 Nsight Systems。

复测支持保留 v27。70% 请求下，fr1_xyz 与 Replica 的准备耗时分别降低 16.9% 和 16.8%，完整推理分别降低 5.9% 和 5.6%。固定 5000 点的准备耗时分别降低 5.0% 和 6.5%。fr1_xyz 的 1% 请求三轮方向有波动，中位数约慢 2 μs，该条件尚无稳定收益。

## 范围与身份

| 项目 | 内容 |
|---|---|
| GPU | RTX 3090 |
| 软件 | PyTorch 2.0.1+cu117、CUDA 11.8 编译与分析工具、驱动 535.230.02 |
| 原版 | 已验收部署 v21，SHA256 `{BASELINE}` |
| 优化版 | 已验收部署 v27，SHA256 `{CANDIDATE}` |
| fr1_xyz | 13298 个 Gaussian、2790 个 Anchor，使用完整 792 帧轨迹中的跳转视角 |
| Replica office3 RGB-D | 49102 个 Gaussian、19936 个 Anchor，使用保存的 137 个视角 |
| 独立目录 | `/home/prmi/zwj/experiments/deployment_idle_retest_20261107_2052` |

直接加载前次已验收的两份二进制，源码、模型与二进制前后哈希匹配。测试脚本和输出单独保存，算法代码的完整性另行核验。

## 空闲条件核验

每两秒记录 GPU 利用率、显存、频率、功耗、温度和全部计算进程。每个测试条件开始、结束时再次核验，出现其他计算进程会报错停止。

本轮监控得到 {SAMPLES} 个采样记录，全部计算进程均属于本轮测试。监控与条件边界检查均未发现外部计算进程。测速条件边界记录的 SM 频率为 {FREQUENCY_MIN}–{FREQUENCY_MAX} MHz，温度为 {TEMP_MIN}–{TEMP_MAX}°C。

## 方法

- 使用同一 checkpoint、相机、固定选择键和实际预算。依次测试固定 1000/2000/5000/10000 点，以及 1%/5%/10%/30%/50%/70% 和混合请求。
- 所有执行计划先预热，再进行两秒渲染预热。每条件执行 20 个预热请求和 160 个测量请求，连续三轮共 480 对样本。两版执行先后交替，第二轮交换场景顺序。
- 准备时间包括已有相机 Tensor 进入部署入口后的视锥、候选、Selector、精确选择、Compensator、属性发布和必要同步。完整推理继续执行原 Rasterizer。相机矩阵构造、显示和编码另计。
- 每个条件保存原始样本、p90/p99、CUDA Event、实际 N/K 和 GPU 状态。下面的汇总对三轮中位数再取中位数。正百分比表示耗时减少。

## 三轮未插桩墙钟结果

所有时间单位为 ms。完整推理包含相同的原 Rasterizer。

| 场景 | 请求 | v21 准备 | v27 准备 | 准备降低 | v21 完整 | v27 完整 | 完整降低 |
|---|---|---|---|---|---|---|---|
{PERFORMANCE}

逐轮结果、p90/p99 和实际 N/K 见 IDLE_RETEST_PERFORMANCE.csv。小幅差异应结合逐轮一致性判断。mix 请求轮转固定点数和 ratio，每次跳转视角，全部相机与预算准备重新执行。

## Nsight Systems 独立核验

每条件采集 30 个预热后的 NVTX 范围。插桩时间独立报告，端到端结论使用前表的未插桩结果。GPU 子节点按实际执行时间归属请求。

| 条件 | v21 累计 GPU μs | v27 累计 GPU μs | v21 Compensator μs | v27 Compensator μs | 补偿耗时降低 |
|---|---|---|---|---|---|
{KERNELS}

| 条件 | GPU 节点前→后 | CPU 发射前→后 | 同步前→后 | DtoH 字节前→后 |
|---|---|---|---|---|
{NODES}

原始 nsys-rep、SQLite 和逐 Kernel 明细位于服务器 reports。此次只复测已验收的部署版本，低点数与高点数分别报告。

## 与上次共享负载的绝对耗时对照

两轮测试采用同一算法二进制和模型。下表的绝对值还受到频率、温度、样本数和测试时间的影响，不能仅凭差值把所有变化归因于某一个后台进程。

| 场景／请求 | 版本 | 前次共享 GPU 准备 ms | 本次空闲 GPU 准备 ms |
|---|---|---|---|
{SHARED}

## 数值与完整性

本轮在两场景各选三个相隔视角，检查固定预算、训练与未见 ratio、0/1、超额预算。共 {CASES} 组、{COMPARISONS} 项比较，ID、属性、像素和 radii 全部逐位一致。此前完整轨迹 16722 组逐位验证与 Sanitizer 0 错误结果继续对应这两份相同二进制。

| 核验集合 | 文件数 | 变化数 |
|---|---|---|
{PROTECTION}

PRODUCTION 覆盖 mycartgs 与 realLOD 的关键源码、配置、二进制和动态库，其余三组覆盖 84 个模型输入文件。Rasterizer 前后 SHA256 为 `{RASTERIZER}`。本轮测试报告与源码包保存在独立目录。
'''
    replacements=dict(BASELINE=identity['baseline']['sha256'],CANDIDATE=identity['binary']['sha256'],
        SAMPLES=audit['samples'],FREQUENCY_MIN=int(min(frequency)),FREQUENCY_MAX=int(max(frequency)),TEMP_MIN=int(min(temperature)),TEMP_MAX=int(max(temperature)),
        PERFORMANCE='\n'.join(table),KERNELS='\n'.join(kernels),NODES='\n'.join(nodes),SHARED='\n'.join(compared),
        CASES=validation['cases'],COMPARISONS=validation['comparisons'],PROTECTION='\n'.join(protection),RASTERIZER=integrity['RASTERIZER']['sha256'])
    for k,v in replacements.items():text=text.replace('{'+k+'}',str(v))
    (STUDY/'IDLE_GPU_INFERENCE_RETEST_REPORT.md').write_text(text,encoding='utf-8')
    print('REPORT_READY')


if __name__=='__main__':main()
