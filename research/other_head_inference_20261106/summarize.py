"""汇总Other保留减去关闭的逐帧配对贡献，不进行额外渲染。"""
import csv
import json
from collections import defaultdict
import numpy as np
from common import ROOT,PREVIOUS,CASES,save

OUT=ROOT/'evaluation/reports'
METRICS=['psnr','ssim','lpips','photo_l1']


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)))
        writer.writeheader();writer.writerows(rows)


def stats(values,prefix,result):
    a=np.asarray(values,dtype=np.float64);assert np.isfinite(a).all()
    for name,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:
        result[prefix+'_'+name]=float(value)


def partition(values):
    for name in ['all','bank','nonbank']:
        selected=[v for v in values if name=='all' or (str(v['bank'])=='True')==(name=='bank')]
        assert len(selected)==dict(all=792,bank=72,nonbank=720)[name]
        yield name,selected


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    assert json.loads((OUT/'CORRECTNESS.json').read_text())['passed']
    with (OUT/'PER_VIEW.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==44352
    lookup={(v['variant'],v['case'],int(v['frame'])):v for v in rows};assert len(lookup)==len(rows)
    grouped=defaultdict(list)
    for row in rows:grouped[row['variant'],row['case']].append(row)
    summary=[];paired=[]
    for (variant,case),values in grouped.items():
        for name,chosen in partition(values):
            item={k:values[0][k] for k in ['variant','case','scope','ratio','ratio_split']}
            item.update(partition=name,frames=len(chosen))
            for key in METRICS+['K','projected_K','contributing_K','requested_K','candidate_N','network_ratio',
                'opacity_delta_abs_mean','opacity_delta_abs_max','dc_delta_norm_mean','dc_delta_norm_p99']:
                stats([float(v[key]) for v in chosen],key,item)
            item['budget_shortfall_frames']=sum(int(v['budget_shortfall'])>0 for v in chosen)
            summary.append(item)
    for name,scope,value in CASES:
        for frame in range(792):
            a,b=lookup['other_on',name,frame],lookup['other_off',name,frame]
            assert all(a[k]==b[k] for k in ['image','timestamp','K','requested_K','candidate_N','eligible_N','selection_sha256','network_ratio'])
            item={k:a[k] for k in ['case','scope','ratio','ratio_split','frame','bank','K']}
            for key in METRICS+['projected_K','contributing_K']:item['delta_'+key]=float(a[key])-float(b[key])
            item['mse_on']=10**(-float(a['psnr'])/10);item['mse_off']=10**(-float(b['psnr'])/10)
            paired.append(item)
    paired_grouped=defaultdict(list)
    for v in paired:paired_grouped[v['case']].append(v)
    pair_summary=[]
    for case,values in paired_grouped.items():
        for name,chosen in partition(values):
            item={k:values[0][k] for k in ['case','scope','ratio','ratio_split']};item.update(partition=name,frames=len(chosen))
            for key in METRICS+['projected_K','contributing_K']:stats([v['delta_'+key] for v in chosen],'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([v['delta_psnr']>0 for v in chosen]))
            item['ssim_improved_fraction']=float(np.mean([v['delta_ssim']>0 for v in chosen]))
            item['lpips_improved_fraction']=float(np.mean([v['delta_lpips']<0 for v in chosen]))
            item['mean_mse_reduction_fraction']=float(1-np.mean([v['mse_on'] for v in chosen])/np.mean([v['mse_off'] for v in chosen]))
            pair_summary.append(item)
    write('SUMMARY.csv',summary);save(OUT/'SUMMARY.json',summary)
    write('PAIRED_PER_VIEW.csv',paired);write('PAIRED_SUMMARY.csv',pair_summary);save(OUT/'PAIRED_SUMMARY.json',pair_summary)
    legacy=json.loads((PREVIOUS/'evaluation/reports/SUMMARY.json').read_text())
    full=[v for v in legacy if v['group']=='both_li' and v['case']=='full']
    assert len(full)==3
    save(OUT/'FULL_REFERENCE.json',dict(rows=full,reused=True,renderer_calls=0,reason='未补偿Full不调用Other Head'))
    save(ROOT/'reports/SUMMARIZATION.json',dict(passed=True,rows=len(rows),paired_rows=len(paired),cases=28,
        all_frame_same_selection_and_K=True,summary_rows=len(summary),paired_summary_rows=len(pair_summary)))
    for v in pair_summary:
        if v['partition']=='all' and v['case'] in ['view_0.01','view_0.05','view_0.1','points_2000']:
            print(json.dumps(v,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
