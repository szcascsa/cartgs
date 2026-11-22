"""正式训练后独立核对cell-field/ordering并锁定ranking；仅posthoc诊断。"""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tensors(p):return {k:v.detach().cpu().numpy()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()}
def corr(a,b):return float(np.corrcoef(a,b)[0,1])if np.std(a)>0 and np.std(b)>0 else None
def stats(a):return dict(mean=float(a.mean()),std=float(a.std()),p1=float(np.quantile(a,.01)),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max_abs=float(np.abs(a).max()))

def run():
    assert read(ROOT/'training_complete.json')['all_passed']
    torch.set_num_threads(2);curves=[];hard=[];views=[];coeff=[];budget=[];grad=[];hist=[];overlap=[];promotion=[];occupancy_corr=[];sets={};rankings={};checks=[]
    models=read(ROOT/'training_complete.json')['model_checks']
    for meta in models:
        scene=meta['scene'];ratio=meta['ratio'];seed=meta['seed'];arm='Grid-MLP'if meta['arm']=='mlp'else'Independent-Cell';base_meta=dict(scene=scene,ratio=ratio,seed=seed,arm=arm)
        out=ROOT/scene/'models'/f'{arm}_{int(100*ratio)}_seed_{seed}'
        mapping=np.load(ROOT/scene/'cell_mapping.npz');ids=mapping['gaussian_source_row'];cell=mapping['cell_id'];oc=mapping['occupancy'];centers=mapping['normalized_cell_center'];n=len(ids);m=len(oc);k=int(n*ratio+.5)
        data=np.load(OLD/scene/'ranking_inputs.npz');base=np.log(data['raw_I'].astype(float)+1e-12);np.testing.assert_array_equal(ids,data['row_ids'])
        raw_order=np.lexsort((ids,-base));raw_set=np.sort(ids[raw_order[:k]]);raw_count=np.bincount(cell[raw_order[:k]],minlength=m)
        raw_rank=np.empty(n,np.int64);raw_rank[raw_order]=np.arange(1,n+1);group_order=np.lexsort((ids,-base,cell))
        rec=[json.loads(x[3:])for x in(out/'training.log').read_text().splitlines()if x.startswith('PG ')]
        for z in rec:
            if z['kind']not in ['training','hard_summary','hard_view']:continue
            zz={key:value for key,value in z.items()if key not in ['kind','selected_source_rows']}
            if z['kind']=='training':curves.append(dict(**base_meta,**zz))
            elif z['kind']=='hard_summary':
                hard.append(dict(**base_meta,**zz));sets[f'{scene}_{arm}_{ratio}_{seed}_step_{z["step"]}']=np.array(z['selected_source_rows'],np.int64)
            else:views.append(dict(**base_meta,**zz))
        max_weight_error=0.;max_prediction_error=0.
        for step in [0,250,1000,2000]:
            state=tensors(out/f'step_{step}.pt');a=state['cell_coefficient'];r=state['cell_raw_output'];ga=state['gaussian_correction'];score=base+ga
            np.testing.assert_allclose(state['score'],score,atol=5e-14,rtol=1e-14)
            np.testing.assert_array_equal(state['cell_id'],cell);np.testing.assert_array_equal(state['occupancy'],oc);np.testing.assert_array_equal(state['normalized_cell_centers'],centers)
            np.testing.assert_array_equal(ga,a[cell]);np.testing.assert_allclose(a,r-np.sum(r*oc/n),atol=1e-10,rtol=1e-11)
            weight_error=abs(np.dot(oc,a)/n);max_weight_error=max(max_weight_error,weight_error);assert weight_error<1e-12*max(1,float(abs(a).max()))
            if arm=='Grid-MLP':
                model=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1,bias=False)).double()
                model.load_state_dict(dict(torch.jit.load(str(out/f'step_{step}_weights.pt'),map_location='cpu').named_parameters()))
                with torch.no_grad():prediction=model(torch.from_numpy(centers)).flatten().numpy()
                max_prediction_error=max(max_prediction_error,float(abs(prediction-r).max()));np.testing.assert_allclose(prediction,r,atol=1e-9,rtol=1e-10)
            np.testing.assert_array_equal(group_order,np.lexsort((ids,-score,cell)))
            if step==0:assert np.all(a==0);np.testing.assert_array_equal(np.lexsort((ids,-score)),raw_order)
            order=np.lexsort((ids,-score));selected=np.sort(ids[order[:k]]);np.testing.assert_array_equal(selected,sets[f'{scene}_{arm}_{ratio}_{seed}_step_{step}'])
            assert len(selected)==len(np.unique(selected))==k
            rankings[f'{scene}_{arm}_{ratio}_{seed}_step_{step}']=ids[order]
            for quantity,value in [('coefficient',a),('exp_coefficient',np.exp(a))]:
                assert np.isfinite(value).all();coeff.append(dict(**base_meta,step=step,quantity=quantity,M=m,weighted_mean_coefficient=float(np.dot(oc,a)/n),**stats(value)))
            if step==2000:
                sets[f'{scene}_{arm}_{ratio}_{seed}']=selected;sets[f'{scene}_Raw_{ratio}']=raw_set
                final_count=np.bincount(cell[order[:k]],minlength=m);delta=final_count-raw_count;direct=state['direct_selected_count'];direct_count=np.bincount(cell,weights=direct,minlength=m);ever=np.bincount(cell,weights=(direct>0).astype(float),minlength=m)
                assert delta.sum()==0
                added=np.setdiff1d(selected,raw_set);assert int(delta.clip(min=0).sum())==len(added)
                for c in range(m):
                    budget.append(dict(**base_meta,cell_id=c,occupancy=int(oc[c]),coefficient=float(a[c]),raw_selected_count=int(raw_count[c]),final_selected_count=int(final_count[c]),delta_k=int(delta[c]),direct_selected_events=int(direct_count[c]),ever_direct_selected_Gaussians=int(ever[c]),final_abs_gradient=float(abs(state['last_raw_output_gradient'][c])),gradient_EMA=float(state['gradient_ema'][c])))
                for value in np.unique(delta):hist.append(dict(**base_meta,delta_k=int(value),cell_count=int((delta==value).sum())))
                occupancy_corr.append(dict(**base_meta,correlation_occupancy_coefficient=corr(oc,a),correlation_occupancy_abs_coefficient=corr(oc,abs(a)),correlation_occupancy_gradient_EMA=corr(oc,state['gradient_ema']),correlation_coefficient_delta_k=corr(a,delta)))
                inter=len(np.intersect1d(selected,raw_set));overlap.append(dict(**base_meta,reference='Raw',K=k,intersection=inter,Jaccard=inter/(2*k-inter),added=k-inter,removed=k-inter))
                idx=np.searchsorted(ids,added);rr=raw_rank[idx];percent=(rr-1)/(n-1)
                promotion.append(dict(**base_meta,K=k,added=len(added),mean_Raw_rank_percentile=float(percent.mean())if len(added)else None,p90_Raw_rank_percentile=float(np.quantile(percent,.9))if len(added)else None,max_Raw_rank=int(rr.max())if len(added)else None,count_Raw_rank_gt_2K=int((rr>2*k).sum())))
                np.savez_compressed(out/'final_field.npz',gaussian_source_row=ids,cell_id=cell,cell_raw_output=r,cell_coefficient=a,gaussian_correction=ga,final_score=score,final_ranking=ids[order],selected_set=selected,occupancy=oc,normalized_cell_centers=centers,gradient_EMA=state['gradient_ema'],direct_selected_count=direct)
        for p in sorted(out.glob('diagnostic_*.pt')):
            step=int(p.stem.split('_')[-1]);state=tensors(p);g=np.abs(state['raw_output_gradient']);ema=state['gradient_ema'];top=max(1,int(np.ceil(.1*m)))
            np.testing.assert_array_equal(state['raw_count'],raw_count)
            grad.append(dict(**base_meta,step=step,M=m,gradient_mean=float(g.mean()),gradient_std=float(g.std()),gradient_p50=float(np.quantile(g,.5)),gradient_p90=float(np.quantile(g,.9)),gradient_p99=float(np.quantile(g,.99)),gradient_max=float(g.max()),top10_gradient_mass_share=float(np.sort(g)[-top:].sum()/g.sum())if g.sum()>0 else None,EMA_mean=float(ema.mean()),EMA_std=float(ema.std()),EMA_p50=float(np.quantile(ema,.5)),EMA_p90=float(np.quantile(ema,.9)),EMA_p99=float(np.quantile(ema,.99)),top10_EMA_mass_share=float(np.sort(ema)[-top:].sum()/ema.sum())if ema.sum()>0 else None,correlation_gradient_occupancy=corr(g,oc),correlation_EMA_occupancy=corr(ema,oc)))
        checks.append(dict(**base_meta,all_passed=True,weighted_mean_max_abs=max_weight_error,independent_MLP_forward_max_abs=max_prediction_error,intra_cell_violations=0,forward_count=m if arm=='Grid-MLP'else 0,N=n,M=m,all_corrections_shared_exact=True,all_selected_same_K_unique=True))
    for name,rows in [('training_curves.csv',curves),('hard_training_dynamics.csv',hard),('hard_training_per_view.csv',views),('cell_coefficient_statistics.csv',coeff),('cell_budget_redistribution.csv',budget),('cell_budget_histogram.csv',hist),('cell_gradient_statistics.csv',grad),('occupancy_correction_correlations.csv',occupancy_corr),('selected_set_overlap.csv',overlap),('raw_rank_promotion.csv',promotion)]:write(name,rows)
    np.savez_compressed(ROOT/'selected_sets.npz',**sets);np.savez_compressed(ROOT/'rankings.npz',**rankings)
    save(ROOT/'selected_before_quality.json',dict(all_passed=True,models=checks,selected_SHA=sha(ROOT/'selected_sets.npz'),rankings_SHA=sha(ROOT/'rankings.npz'),formal_updates=len(curves)))
    print('COLLECTED',len(checks),len(curves),flush=True)

if __name__=='__main__':run()
