"""固定grid的correctness+Bank-only LR gate，成功后才启动主/控制矩阵。"""
from pathlib import Path
import json,os,subprocess,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020';FIX=ROOT.parent/'spatial-mlp-optimization-fix-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027]
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def status(stage,**kw):save(ROOT/'status.json',dict(stage=stage,**kw))
def execute(scene,ratio,seed,steps,lr,arm,out):
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    env=dict(os.environ,LD_LIBRARY_PATH=str(OLD/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    cmd=[str(ROOT/'build/grid_worker'),str(OLD/scene/'training_manifest.json'),str(OLD/scene/'training_input.bin'),str(ROOT/scene/'grid.bin'),str(out),str(ratio),str(seed),str(steps),str(lr),arm]
    with open(out/'training.log','w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(x[3:])for x in(out/'training.log').read_text().splitlines()if x.startswith('PG ')];assert records[-1]['completed']and records[-1]['frozen_exact']and records[-1]['grid_frozen_exact']
    tr=[z for z in records if z['kind']=='training'];hard=[z for z in records if z['kind']=='hard_summary'];assert len(tr)==steps
    # 仅比对历史frame schedule与uniform，不读取PG coefficient、C/Fixed或eval quality。
    original=[json.loads(x[3:])for x in(OLD/scene/'models'/f'MLP_{int(100*ratio)}_seed_{seed}'/'training.log').read_text().splitlines()if x.startswith('PG ')]
    assert [z['frame']for z in tr]==[z['frame']for z in original if z['kind']=='training'][:steps]
    state=dict(torch.jit.load(str(out/'step_0.pt'),map_location='cpu').named_buffers())
    oldstate=dict(torch.jit.load(str(OLD/scene/'models'/f'MLP_{int(100*ratio)}_seed_{seed}'/'step_0.pt'),map_location='cpu').named_buffers())
    assert torch.equal(state['cached_uniform'],oldstate['cached_uniform'])
    mapping=np.load(ROOT/scene/'cell_mapping.npz');np.testing.assert_array_equal(state['cell_id'].numpy(),mapping['cell_id']);np.testing.assert_array_equal(state['normalized_cell_centers'].numpy(),mapping['normalized_cell_center'])
    assert torch.all(state['cell_coefficient']==0)
    data=np.load(OLD/scene/'ranking_inputs.npz');ids=data['row_ids'];base=np.log(data['raw_I'].astype(float)+1e-12);raw=np.lexsort((ids,-base));rawI=np.lexsort((ids,-data['raw_I']))
    np.testing.assert_array_equal(raw,rawI)
    np.testing.assert_array_equal(np.sort(ids[raw[:int(len(ids)*ratio+.5)]]),hard[0]['selected_source_rows'])
    assert all(z['MLP_forward_count']==len(mapping['cell_key'])if arm=='mlp'else z['MLP_forward_count']==0 for z in tr)
    for ratio_check in RATIOS:
        k=int(len(ids)*ratio_check+.5);np.testing.assert_array_equal(ids[raw[:k]],ids[np.lexsort((ids,-(base+state['gaussian_correction'].numpy())))[:k]])
    check=dict(all_passed=True,scene=scene,ratio=ratio,seed=seed,steps=steps,lr=lr,arm=arm,seconds=time.time()-start,
               frame_schedule_uniform_exact=True,initial_all_four_Raw_sets_exact=True,frozen_exact=True,grid_frozen_exact=True,intra_cell_violations=0,
               M=int(len(mapping['cell_key'])),N=int(len(ids)),max_normalized_gradient_sum=max(z['normalized_gradient_sum']for z in tr))
    save(out/'completed.json',check);return hard,check

def main():
    protocol=dict(scenes=SCENES,ratios=RATIOS,seeds=SEEDS,steps=2000,architecture='3-32-32-1 SiLU, final bias=false',parameters=1216,
                  MLP_input='occupied-cell centers ONLY, M rows',normalization='frozen occupied-center min/max box',grid='fixed world-origin XYZ-only target2048 sparse occupancy',
                  gauge='differentiable occupancy-weighted mean over Gaussian population',calibration='unchanged implicit expected-K VJP',sampler='legacy cached-uniform ST Bernoulli',
                  loss='.8 L1+.2(1-fast_ssim)',optimizer='Adam .9/.999 eps1e-8 wd0',EMA_beta=.95,
                  LR_sanity=dict(scene='fr1_xyz',ratio=.05,seed=2026,lrs=[3e-5,1e-4,3e-4],steps=250,
                                 select='finite lowest step250 full-Bank hard photo; ties: higher PSNR then lower LR',
                                 stop='all three final Bank hard photo strictly worse than initial => STOP FULL MATRIX'),
                  formal_arms=['Grid-MLP','Independent-Cell'],control='same chosen LR, gauge/calibration/sampling/loss/schedule; one independent scalar per occupied cell',
                  no_eval_training_or_LR_selection=True,no_regularization=True,no_grid_changes=True,no_production_changes=True,commit=False,stage=False,push=False)
    save(ROOT/'preregistration.json',protocol)
    status('grid_preparation');subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True)
    status('build')
    cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
    env=dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDACXX='/usr/local/cuda-11.8/bin/nvcc',CUDA_HOME='/usr/local/cuda-11.8')
    for name,cmd in [('configure',[cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),('compile',[cmake,'--build',str(ROOT/'build'),'-j1'])]:
        with open(ROOT/(name+'.log'),'w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    runtime=dict(env,LD_LIBRARY_PATH=str(OLD/'fr1_room/runtime/lib')+':'+env.get('LD_LIBRARY_PATH',''))
    for exe,path in [(ROOT/'build/grid_gate',ROOT/'weighted_gauge_gate.json'),(FIX/'build/selection_regression',ROOT/'calibration_gate.json')]:
        with open(path,'w')as f:subprocess.run([str(exe)],env=runtime,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(ROOT/'weighted_gauge_gate.json')['all_passed']and read(ROOT/'calibration_gate.json')['all_passed']
    results=[]
    for lr in protocol['LR_sanity']['lrs']:
        status('LR_sanity',lr=lr);hard,check=execute('fr1_xyz',.05,2026,250,lr,'mlp',ROOT/'lr_sanity'/f'lr_{lr}')
        results.append(dict(lr=lr,initial_photo=hard[0]['photo'],final_photo=hard[-1]['photo'],initial_PSNR=hard[0]['psnr'],final_PSNR=hard[-1]['psnr'],photo_delta=hard[-1]['photo']-hard[0]['photo'],PSNR_delta=hard[-1]['psnr']-hard[0]['psnr'],correctness=check))
        print('LR_SANITY',json.dumps(results[-1]),flush=True)
    failed=all(z['photo_delta']>0 for z in results);selected=None if failed else min(results,key=lambda z:(z['final_photo'],-z['final_PSNR'],z['lr']))['lr']
    save(ROOT/'lr_sanity.json',dict(all_correctness_passed=True,optimization_gate_passed=not failed,results=results,selected_lr=selected,selection_before_any_quality_baseline_reads=True))
    if failed:
        status('optimization_gate_failed',classification='GRID MLP OPTIMIZATION GATE FAILED',formal_models=0);return
    completed=[]
    for arm,label in [('mlp','Grid-MLP'),('independent','Independent-Cell')]:
        for scene in SCENES:
            for ratio in RATIOS:
                for seed in SEEDS:
                    status('formal_training',arm=label,scene=scene,ratio=ratio,seed=seed,completed=len(completed),total=48)
                    _,check=execute(scene,ratio,seed,2000,selected,arm,ROOT/scene/'models'/f'{label}_{int(100*ratio)}_seed_{seed}')
                    completed.append(check);print('TRAINED',label,scene,ratio,seed,len(completed),flush=True)
    save(ROOT/'training_complete.json',dict(all_passed=True,models=48,formal_updates=96000,lr=selected,model_checks=completed))
    status('training_complete',models=48)

if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=str(error));raise
