"""冻结2000步score与cached uniform；先锁集合与预算，再读取quality。"""
from pathlib import Path
import csv,hashlib,json,os,subprocess
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;GRID=ROOT.parent/'grid-mlp-implicit-si-20261020';OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,z):(ROOT/name).write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tensors(p):return {k:v.detach()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()}

def run():
    torch.set_num_threads(2)
    save('status.json',dict(stage='preparing'))
    save('preregistration.json',dict(scenes=SCENES,ratios=RATIOS,seeds=SEEDS,checkpoint=2000,
        arms=['Grid-MLP','Independent-Cell'],training=False,selection='original calibrated sigmoid + original saved cached uniform; no refill/truncate',
        renderer='selected rows only; original opacity, mask value1; nominal ratio passed to frozen TF unchanged',
        controls=['initial Raw-Cached same uniform','same learned score hard Top-L','Raw hard Top-L','historical hard Top-K'],
        point_count='Cached count L is observed; controls matched L; original hard K is nominal-budget context',
        no_grid_or_model_change=True,no_production_change=True))
    assert read(GRID/'correctness.json')['all_passed']
    jobs=[];cases=[];inputs={}
    for scene in SCENES:
        (ROOT/scene/'inference').mkdir(parents=True,exist_ok=True)
        for arm in ['Grid-MLP','Independent-Cell']:
            for ratio in RATIOS:
                for seed in SEEDS:
                    for step in [0,2000]:
                        model=GRID/scene/'models'/f'{arm}_{int(100*ratio)}_seed_{seed}'
                        path=model/f'step_{step}.pt';meta=read(model/'completed.json');k=int(meta['N']*ratio+.5)
                        out=ROOT/scene/'inference'/f'{arm}_{int(100*ratio)}_{seed}_step{step}.json'
                        jobs.append(f'{path} {OLD/scene/"training_input.bin"} {k} {out}')
                        cases.append((scene,arm,ratio,seed,step,path,out));inputs[str(path)]=sha(path)
    (ROOT/'jobs.txt').write_text('\n'.join(jobs)+'\n')
    cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
    env=dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDACXX='/usr/local/cuda-11.8/bin/nvcc',CUDA_HOME='/usr/local/cuda-11.8')
    for name,command in [('configure',[cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_ARCHITECTURES=86','-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),('compile',[cmake,'--build',str(ROOT/'build'),'-j1'])]:
        with open(ROOT/(name+'.log'),'w')as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env['LD_LIBRARY_PATH']=str(OLD/'fr1_room/runtime/lib')+':'+env.get('LD_LIBRARY_PATH','')
    with open(ROOT/'native.log','w')as log:subprocess.run([str(ROOT/'build/cached_inference'),str(ROOT/'jobs.txt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    sets={};counts=[];checks=[];arrays={};case_rows=[]
    for scene,arm,ratio,seed,step,path,out in cases:
        state=tensors(path);score=state['score'].cuda();u=state['cached_uniform'].cuda();n=len(score);k=int(n*ratio+.5)
        low=float(score.min())-40;high=float(score.max())+40
        for _ in range(42):
            mid=.5*(low+high)
            if float((score-mid).sigmoid().sum())>k:low=mid
            else:high=mid
        p=(score-.5*(low+high)).sigmoid();local=torch.nonzero(u<p).flatten().cpu().numpy()
        data=np.load(OLD/scene/'ranking_inputs.npz');ids=data['row_ids'];assert (np.diff(ids)>0).all()
        native=read(out);selected=ids[local];np.testing.assert_array_equal(selected,native['selected_source_rows'])
        assert native['binary_forward_exact']and native['repeated_exact']and len(selected)==len(np.unique(selected))
        if step==0:
            original=[json.loads(line[3:])for line in(path.parent/'training.log').read_text().splitlines()if line.startswith('PG ')]
            first=next(z for z in original if z['kind']=='training');assert first['sampled_N']==len(selected)
            key=f'{scene}_Raw-Cached_{ratio}_{seed}'
            if key in sets:np.testing.assert_array_equal(selected,sets[key])
            else:
                sets[key]=selected;case_rows.append(dict(scene=scene,variant='Raw-Cached',ratio=ratio,seed=seed,K=k,L=len(selected),selected_key=key))
        else:
            prefix=f'{scene}_{arm}_{ratio}_{seed}';arrays[prefix+'_score']=score.cpu().numpy();arrays[prefix+'_cached_uniform']=u.cpu().numpy();arrays[prefix+'_probability']=p.cpu().numpy();arrays[prefix+'_source_rows']=ids
            order=np.lexsort((ids,-score.cpu().numpy()));base=np.log(data['raw_I'].astype(float)+1e-12);raw_order=np.lexsort((ids,-base));L=len(selected)
            for variant,rows in [(arm+'-Cached',selected),(arm+'-Hard-L',np.sort(ids[order[:L]])),('Raw-Hard-L-'+arm,np.sort(ids[raw_order[:L]]))]:
                key=f'{scene}_{variant}_{ratio}_{seed}';sets[key]=rows;case_rows.append(dict(scene=scene,variant=variant,ratio=ratio,seed=seed,K=k,L=L,selected_key=key))
        counts.append(dict(scene=scene,arm=arm,ratio=ratio,seed=seed,step=step,N=n,K=k,L=len(selected),L_over_K=len(selected)/k,expected_count=native['expected_count'],min_sampling_margin=native['min_sampling_margin']))
        checks.append(dict(scene=scene,arm=arm,ratio=ratio,seed=seed,step=step,native_Python_selected_exact=True,repeated_exact=True,forward_binary_exact=True))
    assert len(case_rows)==168 and len(checks)==96
    np.savez_compressed(ROOT/'selected_sets.npz',**sets);np.savez_compressed(ROOT/'inference_arrays.npz',**arrays)
    write('evaluation_cases.csv',case_rows);write('selection_counts.csv',counts)
    save('selected_before_quality.json',dict(all_passed=True,checks=checks,selected_SHA=sha(ROOT/'selected_sets.npz'),arrays_SHA=sha(ROOT/'inference_arrays.npz'),model_SHA=inputs,checkpoints_unchanged=all(sha(Path(p))==h for p,h in inputs.items())))
    save('status.json',dict(stage='prepared',cases=len(case_rows)))
    print('PREPARED',len(case_rows),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as error:save('status.json',dict(stage='failed',error=str(error)));raise
