"""汇总同 binary 顺序 A/B；保留原始 mean/p50/p95/event/wall，禁止用 profiling 时间代替 wall。"""
import csv
import json
from pathlib import Path

D=Path(__file__).resolve().parent
grid=[1000,5000,10000,20000,50000,100000,200000,219969]
def timing(path):
    return {(int(r.get('M',0)),r['stage'],r['metric']):r for r in csv.DictReader(path.open())}
def values(path,n,stage,metric='wall'):
    r=timing(path)[n,stage,metric]
    return {k:float(r[k]) for k in ['mean_ms','p50_ms','p95_ms']}
stages=['network_forward','network_backward','network_forward_backward']
result={'M_grid':[], 'repeat_219969':[], 'source_sha256':{}}
for n in grid:
    entry={'M':n,'network':{},'caller':{}}
    for label,mode in [('legacy',-1),('baseline',1),('final',6)]:
        entry['network'][label]={s:{metric:values(D/f'final_measure/strategy_{mode}.csv',n,s,metric)
            for metric in ['event','wall']} for s in stages}
    for label,mode in [('baseline',1),('final',6)]:
        rows=list(csv.DictReader((D/f'caller_{mode}_final/grid_caller_{n}.csv').open()))
        entry['caller'][label]={backend:{r['metric']:{k:float(r[k]) for k in ['mean_ms','p50_ms','p95_ms']}
            for r in rows if r['backend']==backend and r['stage']=='current_caller_structural'} for backend in ['legacy','cuda']}
    result['M_grid'].append(entry)
for number in [2,3]:
    result['repeat_219969'].append({label:{s:values(D/f'final_measure/repeat_{number}/strategy_{mode}.csv',219969,s)
        for s in stages} for label,mode in [('baseline',1),('final',6)]})
import hashlib
root=D.parents[2]
for name in ['src/selector_cuda.cu','src/selector_cuda.cpp','include/selector_cuda.h']:
    result['source_sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
summary=json.loads((D/'candidate_6_all_final/summary.json').read_text())
result['correctness']={'complete':summary['complete'],'failures':len(summary['failures']),
    'checks':sum(v['checks'] for v in summary['categories'].values()),
    'all_exact':all(v['all_exact'] for v in summary['categories'].values()),
    'maximum_error':max(v['max_abs'] for v in summary['categories'].values()),
    'legacy_repeat_maximum_error':max(v['legacy_repeat_max_abs'] for v in summary['categories'].values())}
assert result['correctness']['complete'] and result['correctness']['all_exact'] and not result['correctness']['failures']
(D/'final_summary.json').write_text(json.dumps(result,indent=2)+'\n')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
rows=[]
for e in result['M_grid']:
    b=e['network']['baseline'];f=e['network']['final'];l=e['network']['legacy']
    rows.append([e['M'],f"{l['network_backward']['wall']['mean_ms']:.4f}",f"{b['network_backward']['wall']['mean_ms']:.4f}",
        '/'.join(f"{f['network_backward']['wall'][k]:.4f}" for k in ['mean_ms','p50_ms','p95_ms']),
        f"{b['network_forward_backward']['wall']['mean_ms']:.4f} → {f['network_forward_backward']['wall']['mean_ms']:.4f}",
        f"{b['network_forward']['wall']['mean_ms']:.4f} → {f['network_forward']['wall']['mean_ms']:.4f}"])
text=table(['M','legacy B','原 CUDA B','新 CUDA B mean/p50/p95','CUDA F+B 前→后','CUDA forward 前→后'],rows)
rows=[]
for e in result['M_grid']:
    b=e['caller']['baseline']['cuda']['wall'];f=e['caller']['final']['cuda']['wall'];l=e['caller']['final']['legacy']['wall']
    rows.append([e['M'],f"{l['mean_ms']:.4f}",f"{b['mean_ms']:.4f}",'/'.join(f"{f[k]:.4f}" for k in ['mean_ms','p50_ms','p95_ms']),
        f"{b['mean_ms']/f['mean_ms']:.3f}×",f"{l['mean_ms']/f['mean_ms']:.3f}×"])
text+='\n'+table(['M','legacy caller','原 CUDA caller','新 CUDA caller mean/p50/p95','相对原 CUDA','相对 legacy'],rows)
(D/'performance_tables.md').write_text(text)
print(json.dumps(result['correctness'],indent=2));print(text)
