"""保留原交付，在独立子目录生成耗时不平滑版本。"""
import hashlib
import json
import shutil
from pathlib import Path
import analyze_plot as plot

original=plot.OUT
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [original/'TIMINGS.csv',original/'PER_VIEW_COMPARISON.csv',original/'SUMMARY.json']}
rows=plot.read(original/'PER_VIEW_COMPARISON.csv')
for row in rows:
    row['frame']=int(row['frame'])
    for key in ('K','projected_K','contributing_K','psnr','prepare_ms','render_ms','total_ms'):row[key]=float(row[key])
summary=json.loads((original/'SUMMARY.json').read_text(encoding='utf-8'))
plot.OUT=original/'timing_unsmoothed';plot.OUT.mkdir(exist_ok=True)
for name in ('ENVIRONMENT.json','ANALYSIS_VALIDATION.json'):shutil.copy2(original/name,plot.OUT/name)
plot.plots(rows,summary);plot.report(summary)
report_path=plot.OUT/'NATIVE_FIXED_AND_RATIO_IDLE_REPORT.md'
report_path.write_text('> 本目录为耗时不平滑的重绘版本；原始TIMINGS.csv、PER_VIEW_COMPARISON.csv和实验验证记录位于上级reports目录。原始数据与PSNR平滑保持不变。\n\n'+report_path.read_text(encoding='utf-8'),encoding='utf-8')
data=plot.read(plot.OUT/'PLOT_DATA_RAW_AND_SMOOTHED.csv')
assert len(data)==6336 and not any(k in data[0] for k in ('prepare_ms_smooth','render_ms_smooth','total_ms_smooth'))
before={(v['method'],int(v['frame'])):v for v in plot.read(original/'PLOT_DATA_RAW_AND_SMOOTHED.csv')}
for row in data:
    old=before[row['method'],int(row['frame'])]
    for key in ('prepare_ms','render_ms','total_ms','psnr','psnr_smooth'):assert float(row[key])==float(old[key])
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in protected.items())
protocol=json.loads((original/'PLOT_PROTOCOL.json').read_text(encoding='utf-8'))
protocol['timing']=dict(smoothing=False,reduction='per-view mean of three repeats',cross_frame_filtering=False,outliers_removed=False)
plot.save(plot.OUT/'PLOT_PROTOCOL.json',protocol)
plot.save(plot.OUT/'UNSMOOTHED_VALIDATION.json',dict(passed=True,methods=8,views=792,rows=6336,raw_data_unchanged=True,
    timing_smoothing=False,psnr_smoothing_unchanged=True,protected_inputs=protected))
print('UNSMOOTHED_TIMING_PLOTS_PASS',plot.OUT)
