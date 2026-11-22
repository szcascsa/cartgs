"""纯 full-population MSE 回归；训练过程不导入renderer、GT或Top-K损失。"""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import torch
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027]
STEPS=5000;LR=.001
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(name,rows):
    with open(ROOT/name,'w',newline='')as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def buffers(p):return {k:v.detach()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()}
def model():
    net=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1,bias=False)).double().cuda()
    with torch.no_grad():net[-1].weight.zero_()
    return net
def run():
    ROOT.mkdir(exist_ok=True)
    protocol=dict(scenes=SCENES,ratios=RATIOS,seeds=SEEDS,target='PG-ST step2000 per-Gaussian coefficient, full population centered',
                  prediction='XYZ tiny MLP, full population centered',architecture=[3,32,32,1],activation='SiLU',parameters=1216,
                  final_layer_bias=False,initialization='default hidden Linear; zero final weight',normalization='exact archived frozen TF AABB formula, no refit/clamp',
                  dtype='FP64',loss='mean((pred-target)^2), full valid AND observed population',optimizer='Adam',lr=LR,
                  betas=[.9,.999],eps=1e-8,weight_decay=0,steps=STEPS,checkpoint_steps=[0,250,1000,2500,5000],
                  curve_interval=50,final_export='step5000, no ranking/render based selection',lr_search=False,
                  render_or_sampling_in_training=False,training_inputs=['normalized XYZ','centered frozen PG-ST coefficient'],
                  no_other_feature_or_normalization=True,code_sha=sha(__file__))
    save(ROOT/'preregistration.json',protocol)
    oldsets=np.load(OLD/'selected_sets.npz');checks=read(OLD/'input_checks.json')['scenes']
    regression=[];overlaps=[];curve=[];rankings={};sets={};metadata=[];count=0
    for scene in SCENES:
        data=np.load(OLD/scene/'ranking_inputs.npz');ids=data['row_ids'];base=np.log(data['raw_I'].astype(np.float64)+1e-12)
        normalized=np.load(OLD/scene/'normalized_xyz.npz')['normalized_xyz']
        lo=np.array(read(OLD/scene/'normalization.json')['aabb_min']);hi=np.array(read(OLD/scene/'normalization.json')['aabb_max'])
        np.testing.assert_array_equal(normalized,2*(data['xyz'].astype(np.float64)-lo)/(hi-lo)-1)
        with open(OLD/scene/'training_input.bin','rb')as f:
            n=np.fromfile(f,dtype=np.int64,count=1)[0];rows=np.fromfile(f,dtype=np.int64,count=n);bb=np.fromfile(f,dtype=np.float64,count=n)
        np.testing.assert_array_equal(ids,rows);np.testing.assert_array_equal(base,bb)
        x=torch.from_numpy(normalized).cuda();xcopy=x.clone()
        raw_order=np.lexsort((ids,-base));n=len(ids)
        for ratio in RATIOS:
            k=int(n*ratio+.5)
            for seed in SEEDS:
                case=f'fit_{int(100*ratio)}_seed_{seed}';out=ROOT/scene/'models'/case;out.mkdir(parents=True,exist_ok=False)
                source=OLD/scene/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'step_2000.pt'
                before=sha(source);assert before==checks[scene]['legacy_model_sha'][str(source)]
                a=buffers(source)['coefficient'].numpy();target=a-a.mean();y=torch.from_numpy(target).cuda();ycopy=y.clone()
                # 这是target/source mapping gate，不是训练信号。
                pg_order=np.lexsort((ids,-(base+target)));pg_selected=np.sort(ids[pg_order[:k]])
                np.testing.assert_array_equal(pg_selected,oldsets[f'{scene}_PG-ST_{ratio}_{seed}'])
                torch.manual_seed(seed);net=model();optimizer=torch.optim.Adam(net.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=0)
                assert sum(p.numel()for p in net.parameters())==1216
                with torch.no_grad():assert torch.all(net(x)==0)
                losses=[];start=time.time()
                for step in range(STEPS+1):
                    raw=net(x).flatten();pred=raw-raw.mean();loss=(pred-y).square().mean();value=float(loss.detach());losses.append(value)
                    assert np.isfinite(value)
                    if step%50==0:
                        curve.append(dict(scene=scene,ratio=ratio,seed=seed,step=step,MSE=value,prediction_std=float(pred.detach().std(unbiased=False)),target_std=float(y.std(unbiased=False))))
                        save(ROOT/'status.json',dict(stage='regression',scene=scene,ratio=ratio,seed=seed,step=step,completed_models=count,total_models=24))
                    if step in protocol['checkpoint_steps']:
                        torch.save({k:v.detach().cpu()for k,v in net.state_dict().items()},out/f'step_{step}_weights.pt')
                    if step==STEPS:break
                    optimizer.zero_grad();loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in net.parameters())
                    optimizer.step()
                pred=pred.detach().cpu().numpy();assert sha(source)==before and torch.equal(x,xcopy)and torch.equal(y,ycopy)
                assert np.isfinite(pred).all()
                order=np.lexsort((ids,-(base+pred)));selected=np.sort(ids[order[:k]])
                key=f'{scene}_{ratio}_{seed}';rankings[key+'_PG-ST']=ids[pg_order];rankings[key+'_MLP-fit']=ids[order]
                sets[key+'_PG-ST']=pg_selected;sets[key+'_MLP-fit']=selected
                rawset=np.sort(ids[raw_order[:k]]);pg_added=np.setdiff1d(pg_selected,rawset)
                for arm,selection in [('MLP-fit',selected),('Raw',rawset)]:
                    inter=len(np.intersect1d(selection,pg_selected));hit=len(np.intersect1d(selection,pg_added))
                    overlaps.append(dict(scene=scene,ratio=ratio,seed=seed,arm=arm,N=n,K=k,intersection=inter,recall=inter/k,Jaccard=inter/(2*k-inter),missed=k-inter,
                                         PG_additions_vs_Raw=len(pg_added),PG_additions_recovered=hit,PG_additions_recall=hit/len(pg_added)if len(pg_added)else'',added_vs_PG=len(np.setdiff1d(selection,pg_selected))))
                mse=float(np.mean((pred-target)**2));variance=float(np.var(target))
                reg=dict(scene=scene,ratio=ratio,seed=seed,N=n,K=k,steps=STEPS,lr=LR,MSE=mse,RMSE=float(np.sqrt(mse)),Pearson=float(pearsonr(pred,target).statistic),Spearman=float(spearmanr(pred,target).statistic),
                         target_std=float(target.std()),prediction_std=float(pred.std()),target_mean=float(target.mean()),prediction_mean=float(pred.mean()),zero_predictor_MSE=variance,R2=1-mse/variance,
                         initial_MSE=losses[0],MSE_step4000=losses[4000],MSE_step4500=losses[4500],last500_relative_MSE_decrease=(losses[4500]-losses[-1])/losses[4500],
                         target_exact_zero_fraction_before_centering=float((a==0).mean()),runtime_seconds=time.time()-start)
                regression.append(reg);write('regression_metrics.csv',regression);write('topk_overlap.csv',overlaps);write('training_curve.csv',curve)
                np.savez_compressed(out/'field.npz',row_ids=ids,normalized_xyz=normalized,target=target,prediction=pred,score_pg=base+target,score_mlp=base+pred,PG_selected=pg_selected,MLP_selected=selected)
                np.save(out/'full_MSE_curve.npy',np.asarray(losses))
                meta=dict(scene=scene,ratio=ratio,seed=seed,source_coefficient_file=str(source),source_SHA256=before,normalized_xyz_SHA256=tsha(normalized),target_SHA256=tsha(target),row_ids_SHA256=tsha(ids),target_selected_exact_PGST=True,input_target_unchanged=True,loss='full_population_MSE_only')
                save(out/'metadata.json',meta);metadata.append(meta);count+=1
                print('FITTED',count,scene,ratio,seed,json.dumps(reg),flush=True)
    np.savez_compressed(ROOT/'rankings.npz',**rankings);np.savez_compressed(ROOT/'selected_sets.npz',**sets)
    save(ROOT/'fit_complete.json',dict(all_passed=True,models=24,updates=24*STEPS,rankings_SHA=sha(ROOT/'rankings.npz'),selected_sets_SHA=sha(ROOT/'selected_sets.npz'),models_metadata=metadata))
    save(ROOT/'status.json',dict(stage='regression_complete',models=24))

if __name__=='__main__':
    try:run()
    except Exception as error:save(ROOT/'status.json',dict(stage='failed',error=str(error)));raise
