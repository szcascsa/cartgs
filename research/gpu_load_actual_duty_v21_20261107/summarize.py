"""从最终验收原始记录生成中文报告，不启动GPU计算。"""
from pathlib import Path
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/validation04'


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    result = json.loads((RUN / 'VALIDATION.json').read_text())
    integrity = json.loads((ROOT / 'reports/INTEGRITY_FINAL.json').read_text())
    environment = json.loads((RUN / 'ENVIRONMENT.json').read_text())
    windows = read(RUN / 'WINDOW_PULSES.csv.phases.csv')
    requests = read(RUN / 'V21_PULSES.csv.phases.csv')
    hardware = result['hardware']
    assert result['passed'] and result['long_window_target_match']['passed']
    assert result['short_window_target_match']['passed']
    rows = []
    for duty in (.2, .4, .6, .8):
        long_values = [float(row['measured_resident_duty']) * 100
                       for row in windows if float(row['target_duty']) == duty]
        short_values = [float(row['measured_resident_duty']) * 100
                        for row in requests if float(row['target_duty']) == duty]
        rows.append(dict(target_percent=100*duty, long_cases=len(long_values),
            long_min_percent=min(long_values), long_max_percent=max(long_values),
            request_cases=len(short_values), request_min_percent=min(short_values),
            request_max_percent=max(short_values)))
    with (ROOT / 'reports/DUTY_SUMMARY.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = '\n'.join(
        f"| {row['target_percent']:.0f}% | {row['long_min_percent']:.2f}%–{row['long_max_percent']:.2f}% "
        f"| {row['request_min_percent']:.2f}%–{row['request_max_percent']:.2f}% |"
        for row in rows)
    report = f"""# GPU人工负载实测占空比与v21验收报告

## 结论

已实现有界、全SM覆盖的人工负载，按GPU实测驻留包络时间闭环补足OFF时段，输入上限80%。最终validation04的三轮长窗口和60次v21请求通过上限、目标匹配、资源覆盖与阶段隔离检查。原生产代码、模型和既有实验未改写。

普通CUDA无法保证其他进程完全无法运行。当前ON段占满同一CUDA上下文内全部SM的线程驻留槽，并保留大量共享内存；跨进程仍受驱动调度和抢占控制。本实现不修改其他进程、MPS、设备独占模式或系统调度配置。

## 实测占空比

长窗口按0%、20%、40%、60%、80%、0%顺序执行三轮，每档主动保持150毫秒，pause可能为结清OFF时间而延长窗口。短请求为1000和4000点两种预算、每档每预算5次，共60次。

| 目标占空比 | 长窗口三轮范围 | v21请求范围，每档10次 |
|---|---|---|
{table}

0%档没有发射负载Kernel。长窗口最大目标偏差为{100*result['long_window_target_match']['maximum_absolute_error']:.3f}个百分点，短请求最大偏差为{100*result['short_window_target_match']['maximum_absolute_error']:.3f}个百分点，均满足预声明的±5个百分点标准。

所有长窗口、短请求和主动析构测试中，驻留占空比最高{100*result['maximum_measured_duty']:.4f}%，保守计费占空比最高{100*result['maximum_charged_duty']:.4f}%，均小于80%。

## 测量与控制定义

每个Block读取GPU globaltimer和SM编号。单包驻留包络ON时间为全部Block最早入口至最晚出口的时间；CUDA Event可能额外包含调度间隙，计费ON采用两者较大值。单独保留平均Block驻留时长。

```text
ON_resident = max(block_end) - min(block_begin)
ON_charged = max(ON_resident, CUDA_Event_interval)
required_elapsed = ceil(sum(ON_charged) / requested_duty)
```

完成每包后，仅当从本窗口resume起累计墙钟达到required_elapsed，才发射下一包或确认pause。pause和析构都必须等待当前包完成并结清OFF时间。负值、NaN、Inf和超过0.8的输入在C ABI层直接拒绝。

80%约束针对完整渲染窗口及每个结清OFF时间后的累计周期，不针对任意截取的瞬时区间。ON子段允许占满GPU；不应将nvidia-smi整卡利用率当作本程序占空比。准备和请求间隔不在渲染窗口分母内，因此整个应用墙钟上的人工负载占比通常更低。

入口至出口globaltimer区间可能包含驱动抢占暂停。它是驻留包络测量，不能证明人工负载实际独占执行了同等时间，也不能区分外部进程在驱动分时期间使用GPU的份额。这里的保守计费用于上限控制。

## ON段资源与强度

- GPU：RTX 3090，{hardware['sm_count']}个SM。
- cooperative grid：{hardware['total_blocks']}个Block，每Block {hardware['threads_per_block']}线程，每线程8条独立FP32 FMA链。
- 每SM实测{hardware['resident_blocks_per_sm']}个Block、{hardware['resident_threads_per_sm']}个线程，等于设备最大可驻留线程数；每包核对全部SM及每SM Block数。
- 每Block动态共享内存{hardware['shared_bytes_per_block']}字节，每SM动态预留{hardware['shared_bytes_per_block']*hardware['resident_blocks_per_sm']}字节，另有编译器静态共享内存；设备每SM容量{hardware['shared_bytes_per_sm']}字节。
- 名义周期20毫秒，20%/40%/60%/80%对应4/8/12/16毫秒计算ON目标。实际包络可能因网格同步或驱动调度延长，OFF仍按实测补足。没有无限运行的Kernel。
- 网络准备Stream优先级{hardware['preparation_highest_priority']}，负载Stream优先级{hardware['load_stream_priority']}，渲染默认优先级0。优先级是CUDA调度提示，不构成跨进程抢占保证。

## v21正确性与阶段边界

- 二进制SHA256：`{environment['binary_sha256']}`。
- 实际Torch {environment['torch']} / CUDA {environment['cuda']}，保持现有FP32推理，未修改Selector、Compensator或Rasterizer。
- 同一固定视角、同一checkpoint，1000/4000点各预算的全部负载档输出，与无人工负载参考图像SHA256逐位一致。
- 60次请求、{result['pulse_count']}个负载包全部处于对应渲染阶段内；人工负载与准备阶段重叠数为0。
- 负载排空及OFF结清后才进入下次准备；主动析构同样通过OFF结清检查。
- CPU变长ON累计占空比测试、非法输入测试和CUDA编译通过。不保存渲染图片，不训练模型。

原始证据位于`runs/validation04/`：`WINDOW_PULSES.csv`、`V21_PULSES.csv`及各自`.phases.csv`、`V21_TIMINGS.csv`、`DESTRUCTOR_PULSES.csv`、`VALIDATION.json`、`ENVIRONMENT.json`和`GPU_AFTER.json`。

## 修正过程与失败候选

1. validation01：固定100微秒短脉冲，上限通过，但80%目标仅达到约29%–31%。CPU提交及跨进程排队占比过大，未通过目标匹配。
2. validation02：5毫秒名义周期，80%档约65%，仍未通过目标匹配。
3. validation03：20毫秒名义周期、每SM一个1024线程Block，长窗口通过；短请求最大目标偏差5.81个百分点，尚未满足全部请求的±5个百分点观察标准。
4. validation04：保留20毫秒周期，改成每SM三个512线程Block，覆盖全部线程槽；长窗口和短请求全部通过。前三版源代码与二进制已归档，不覆盖失败证据。

旧版本VALIDATION中单一`passed`字段只代表当时执行的检查，不能代替后来增加的目标匹配判定。最终以validation04及其分项字段为准。

## 完整性与运行边界

验收前后{integrity['files']}项受保护文件哈希一致。相对初始清单，本轮目录之外{integrity['initial_external_files']}项受保护文件完全未变；变化只有本轮负载源码、验证入口、协议和负载动态库四项。原始checkpoint、生产代码、既有推理动态库及旧实验受保护文件均未改写。未commit/push，未修改共享conda，未停止其他任务。最后一个验证进程已退出，人工负载不在后台持续运行。

验证期间服务器仍有其他计算进程，因此本轮不宣称无干扰的CaRtGS/自适应速度对比，不从这些记录推断降点稳定负载的收益。这里只验收负载强度、占空比和阶段隔离。

20毫秒周期会引入毫秒级阻塞下限，尤其是短渲染请求。后续完整速度比较必须同时报告纯渲染时间、负载启停等待和请求墙钟，不能把gate等待隐藏后称为端到端耗时。当前复用的参考测速入口保留历史计时口径，本轮未执行其完整比较主程序。

服务器独立目录：`{ROOT}`。复验入口：`source/research/gpu_load_actual_duty_v21_20261107/validate_load.py --label <新的不重复标签>`。首次运行前应只读核对GPU占用，避免将外部任务变化当作模型收益。

## 官方调度语义参考

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html
"""
    (ROOT / 'reports/GPU_ACTUAL_DUTY_REPORT.md').write_text(report, encoding='utf-8')
    manifest = {}
    for path in sorted((ROOT / 'source/research/gpu_load_actual_duty_v21_20261107').iterdir()):
        if path.is_file():
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = ROOT / 'build/librender_load.so'
    manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'reports/FINAL_IMPLEMENTATION_HASHES.json').write_text(json.dumps(manifest, indent=2))
    print('FINAL_DUTY_REPORT_READY', json.dumps(rows))


if __name__ == '__main__':
    main()
