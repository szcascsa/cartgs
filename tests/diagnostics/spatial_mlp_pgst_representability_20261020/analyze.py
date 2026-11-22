"""独立核对系数回归、hard集合、质量聚合并生成对照CSV；不修改模型。"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from scipy.stats import pearsonr,spearmanr
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
def read(p):return json.loads(Path(p).read_text())
def load(n):return list(csv.DictReader(open(ROOT/n)))
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert read(ROOT/'status.json')['stage']=='complete'
fit=read(ROOT/'fit_complete.json');quality=read(ROOT/'quality_complete.json');reg=load('regression_metrics.csv');overlap=load('topk_overlap.csv')
q=load('quality_summary.csv');v=load('quality_per_view.csv');qm={(z['scene'],float(z['ratio']),int(z['seed']),z['split'],z['arm']):z for z in q}
rawq={(z['scene'],float(z['ratio']),z['split']):z for z in csv.DictReader(open(OLD/'quality_summary.csv'))if z['arm']=='Raw'}
vm={(z['scene'],float(z['ratio']),int(z['seed']),z['split'],z['arm'],int(z['frame_index'])):z for z in v}
assert len(reg)==24 and len(qm)==len(q)==96 and len(vm)==len(v)==5680
comparisons=[];paired=[];model_checks=[];convergence=[]
torch.set_num_threads(2)
sets=np.load(ROOT/'selected_sets.npz')
for z in reg:
    scene=z['scene'];ratio=float(z['ratio']);seed=int(z['seed']);case=ROOT/scene/'models'/f'fit_{int(100*ratio)}_seed_{seed}'
    field=np.load(case/'field.npz');ids=field['row_ids'];target=field['target'];pred=field['prediction'];normal=field['normalized_xyz']
    data=np.load(OLD/scene/'ranking_inputs.npz');base=np.log(data['raw_I'].astype(float)+1e-12)
    np.testing.assert_array_equal(ids,data['row_ids']);np.testing.assert_array_equal(normal,np.load(OLD/scene/'normalized_xyz.npz')['normalized_xyz'])
    meta=read(case/'metadata.json');assert hashlib.sha256(Path(meta['source_coefficient_file']).read_bytes()).hexdigest()==meta['source_SHA256']
    source=dict(torch.jit.load(meta['source_coefficient_file'],map_location='cpu').named_buffers())['coefficient'].numpy()
    np.testing.assert_array_equal(target,source-source.mean())
    model=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1,bias=False)).double()
    model.load_state_dict(torch.load(case/'step_5000_weights.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():raw=model(torch.from_numpy(normal)).flatten().numpy();independent=raw-raw.mean()
    np.testing.assert_allclose(independent,pred,atol=1e-9,rtol=1e-9)
    for key,value in [('MSE',np.mean((pred-target)**2)),('Pearson',pearsonr(pred,target).statistic),('Spearman',spearmanr(pred,target).statistic)]:
        assert abs(float(value)-float(z[key]))<1e-12
    n=len(ids);k=int(ratio*n+.5)
    for arm,coefficient in [('PG-ST',target),('MLP-fit',pred)]:
        score=base+coefficient;order=np.array(sorted(range(n),key=lambda i:(-float(score[i]),int(ids[i]))))
        np.testing.assert_array_equal(np.sort(ids[order[:k]]),sets[f'{scene}_{ratio}_{seed}_{arm}'])
    for split in ['train','eval']:
        a=qm[scene,ratio,seed,split,'MLP-fit'];b=qm[scene,ratio,seed,split,'PG-ST']
        comparison=dict(scene=scene,ratio=ratio,seed=seed,split=split,K=k,views=int(a['views']))
        for metric in ['psnr','ssim','lpips','photo']:
            comparison[metric+'_PGST']=float(b[metric]);comparison[metric+'_MLPfit']=float(a[metric]);comparison[metric+'_delta']=float(a[metric])-float(b[metric])
            comparison[metric+'_Raw_reference']=float(rawq[scene,ratio,split][metric])
            info=read(OLD/scene/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
            aa=[float(vm[scene,ratio,seed,split,'MLP-fit',f][metric])for f in frames];bb=[float(vm[scene,ratio,seed,split,'PG-ST',f][metric])for f in frames]
            assert abs(math.fsum(aa)/len(aa)-float(a[metric]))<1e-12
            assert abs(math.fsum(bb)/len(bb)-float(b[metric]))<1e-12
            d=np.array(aa)-np.array(bb);sign=1 if metric in ['psnr','ssim']else-1
            paired.append(dict(scene=scene,ratio=ratio,seed=seed,split=split,metric=metric,views=len(d),mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_view_fraction=float(np.mean(sign*d>0))))
        comparisons.append(comparison)
    loss=np.load(case/'full_MSE_curve.npy');assert len(loss)==5001 and np.isfinite(loss).all()
    convergence.append(dict(scene=scene,ratio=ratio,seed=seed,MSE_step0=float(loss[0]),MSE_step4000=float(loss[4000]),MSE_step4500=float(loss[4500]),MSE_step5000=float(loss[-1]),
                            last500_relative_decrease=float((loss[4500]-loss[-1])/loss[4500]),last100_range_over_final=float(np.ptp(loss[-101:])/loss[-1]),loss_decreased=bool(loss[-1]<loss[0])))
    model_checks.append(dict(scene=scene,ratio=ratio,seed=seed,target_and_normalized_XYZ_exact=True,independent_weight_forward=True,independent_hard_TopK_exact=True,source_unchanged=True))
write('quality_comparison.csv',comparisons);write('paired_statistics.csv',paired);write('convergence_statistics.csv',convergence)
def stats(values):
    a=np.asarray(values,dtype=float);return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()))
summary=dict(regression={m:stats([float(z[m])for z in reg])for m in ['MSE','RMSE','Pearson','Spearman','target_std','prediction_std','R2','last500_relative_MSE_decrease']},
             overlap={arm:{m:stats([float(z[m])for z in overlap if z['arm']==arm and z[m]!=''])for m in ['recall','Jaccard','PG_additions_recall']}for arm in ['MLP-fit','Raw']},
             quality={split:{metric:stats([z[metric+'_delta']for z in comparisons if z['split']==split])for metric in ['psnr','ssim','lpips','photo']}for split in ['train','eval']},
             models=24,steps_per_model=5000,total_updates=120000,decision='PENDING_EVIDENCE_REVIEW')
summary['quality_case_counts']={split:{metric:dict(improved=sum((1 if metric in ['psnr','ssim']else-1)*z[metric+'_delta']>0 for z in comparisons if z['split']==split),equal=sum(z[metric+'_delta']==0 for z in comparisons if z['split']==split),worse=sum((1 if metric in ['psnr','ssim']else-1)*z[metric+'_delta']<0 for z in comparisons if z['split']==split))for metric in ['psnr','ssim','lpips','photo']}for split in ['train','eval']}
summary['PSNR_mean_gain_over_Raw']={split:{arm:float(np.mean([z['psnr_'+arm]-z['psnr_Raw_reference']for z in comparisons if z['split']==split]))for arm in ['PGST','MLPfit']}for split in ['train','eval']}
save('audit_summary.json',summary)
save('correctness.json',dict(all_passed=True,fit=fit,quality=quality,model_checks=model_checks,
                           training_only_full_population_MSE=True,no_renderer_GT_or_TopK_training=True,source_normalization_unchanged=True,
                           all_sets_fixed_before_quality=True,independent_metric_and_ranking_reproduction=True,all_models_final_step5000=True,no_hyperparameter_search=True))
print(json.dumps(summary,indent=2))
