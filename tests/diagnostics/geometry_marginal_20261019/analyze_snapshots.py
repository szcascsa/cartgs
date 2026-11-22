"""只读每个full-refresh snapshot：Raw vs Geo Marginal，同候选同K机制统计。"""
import csv,json,struct,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
results=[]
for folder in sorted((ROOT/'online_runs').glob('*')):
    if folder.name.endswith('_shadow'):continue
    scene,seed,arm=folder.name.rsplit('_',2)
    for snap in sorted((folder/'model').glob('marginal_snapshot_*.bin'),key=lambda p:int(p.stem.split('_')[-1])):
        iteration=int(snap.stem.split('_')[-1]);dtype=np.dtype([('xyz','<f8',(3,)),('radius','<f8'),('I','<f8'),('id','<i8')])
        with snap.open('rb')as f:n,h=struct.unpack('<qd',f.read(16));x=np.frombuffer(f.read(),dtype=dtype).copy()
        assert len(x)==n
        out=folder/('mechanism_'+str(iteration));subprocess.run([str(ROOT/'rank_cli'),str(snap),str(out),'optimized',str(int(.05*n+.5))],check=True)
        with open(str(out)+'.bin','rb')as f:size=struct.unpack('<q',f.read(8))[0];order_ids=np.frombuffer(f.read(size*8),dtype='<i8')
        idx={int(row):i for i,row in enumerate(x['id'])};marginal=np.array([idx[int(row)]for row in order_ids]);raw=np.lexsort((x['id'],-x['I']));keys,cell=np.unique(np.floor(x['xyz']/h).astype(np.int64),axis=0,return_inverse=True);occupancy=np.bincount(cell);mass=np.bincount(cell,weights=x['I']);important=np.lexsort((np.arange(len(mass)),-mass));stop=np.searchsorted(np.cumsum(mass[important]),.8*mass.sum())+1;important=important[:stop];dense=np.lexsort((np.arange(len(mass)),-occupancy))[:max(1,int(np.ceil(.1*len(mass))))]
        for ratio in [.01,.05]:
            K=int(ratio*n+.5);a=set(raw[:K]);b=set(marginal[:K]);jaccard=len(a&b)/len(a|b)
            for method,order in [('Raw',raw),('Marginal',marginal)]:
                selected=order[:K];selected_counts=np.bincount(cell[selected],minlength=len(keys));red=[];duplicates=0
                for c in np.flatnonzero(selected_counts):
                    g=selected[cell[selected]==c];xyz=x['xyz'][g];radius=x['radius'][g];dist=((xyz[:,None]-xyz[None,:])**2).sum(2);sigma=.5*(radius[:,None]+radius[None,:]);k=np.exp(-dist/(2*sigma**2+1e-12));np.fill_diagonal(k,0);red.extend(k.max(1));duplicates+=int((k[np.triu_indices(len(g),1)]>.9).sum())
                results.append(dict(scene=scene,seed=int(seed),training_arm=arm,iteration=iteration,ranking=method,ratio=ratio,candidates=n,cells=len(keys),occupancy_mean=float(occupancy.mean()),occupancy_median=float(np.median(occupancy)),occupancy_max=int(occupancy.max()),fixed_h=h,current_d50=2*float(np.median(x['radius'])),h_over_current_h8=float(h/(16*np.median(x['radius']))),K=K,selected_count=len(selected),redundancy_mean=float(np.mean(red)),duplicate_pairs_gt09=duplicates,important_mass_coverage=float(mass[important][selected_counts[important]>0].sum()/mass[important].sum()),all_mass_coverage=float(mass[selected_counts>0].sum()/mass.sum()),dense_cell_budget_share=float(selected_counts[dense].sum()/K),raw_marginal_jaccard=jaccard))
    print('MECHANISM',folder.name,len(results),flush=True)
write('online_mechanism.csv',results)
