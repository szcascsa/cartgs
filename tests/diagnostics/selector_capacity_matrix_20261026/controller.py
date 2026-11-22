"""固定预算分阶段矩阵；选择仅用fr1 Bank验证，不读测试质量选结构。"""
import gc
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import json
from pathlib import Path
import time
import numpy as np
import torch

from runtime import ROOT,Scene,Network,train,evaluate,save_json,write_csv,RATIOS
from design import identity

PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
DESIGN=json.loads((ROOT/'screen_design.json').read_text())
R3=dict(shared=16,ratio=0,extra=14,selector_depth=1,compensation_depth=1,id='R3',direct=True)
BASE=dict(shared=16,ratio=0,extra=14,selector_depth=1,compensation_depth=1)
LATEST=dict(shared=16,ratio=8,extra=14,selector_depth=1,compensation_depth=1)
BASE['id']=identity(BASE);LATEST['id']=identity(LATEST)

def status(stage,**values):
    save_json(ROOT/'controller_status.json',dict(stage=stage,**values));print(stage,values,flush=True)
def model(scene,info):
    net=Network(scene,info['config'],info['seed']);payload=torch.load(info['checkpoint'],map_location='cuda')
    net.load_state_dict(payload['state_dict']);return net.eval()
def full_eval(scene,info,directory,split):
    path=Path(directory)/(split+'_full.csv')
    if path.exists():return
    net=model(scene,info)
    rows=evaluate(scene,net,split,full_metrics=True,policies=('same_k','cached'))
    for row in rows:row.update(config=info['config']['id'],seed=info['seed'])
    write_csv(path,rows);del net;gc.collect()

WORKER_SCENE=None
def worker_init(name):
    global WORKER_SCENE
    WORKER_SCENE=Scene(name)
def worker(job):
    cfg,seed,steps,directory,checkpoints,splits=job
    info=train(WORKER_SCENE,cfg,seed,steps,directory,checkpoints=checkpoints)
    for split in splits:full_eval(WORKER_SCENE,info,directory,split)
    gc.collect();return info
def run_jobs(scene,phase,configs,seeds,steps,checkpoints,splits=()):
    tasks=[]
    for seed in seeds:
        order=np.random.default_rng(seed+60000).permutation(len(configs))
        for i in order:
            cfg=configs[int(i)];directory=ROOT/'results'/scene.name/phase/cfg['id']/str(seed)
            tasks.append((cfg,seed,steps,str(directory),checkpoints,splits))
    results=[]
    with ProcessPoolExecutor(max_workers=PROTOCOL.get('execution_workers',2),mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(scene.name,)) as pool:
        futures={pool.submit(worker,task):task for task in tasks}
        for future in as_completed(futures):
            info=future.result();results.append(info)
            status(phase.upper(),completed=len(results),total=len(tasks),config=info['config']['id'],seed=info['seed'])
            save_json(ROOT/(phase+'_runs.json'),results)
    return results

def screen(scene):
    total=len(DESIGN)*len(PROTOCOL['screen']['seeds'])
    results=run_jobs(scene,'screen',DESIGN,PROTOCOL['screen']['seeds'],4000,[1000,2000,4000])
    summary=[]
    for cfg in DESIGN:
        selected=[x for x in results if x['config']['id']==cfg['id']]
        vals=[x['validation_score'] for x in selected]
        summary.append(dict(**cfg,mean_validation_psnr=float(np.mean(vals)),seed_std=float(np.std(vals,ddof=1)),
            parameters=selected[0]['parameters'],steps=4000,seeds=len(vals)))
    summary.sort(key=lambda x:(-x['mean_validation_psnr'],x['id']))
    write_csv(ROOT/'screen_summary.csv',summary)
    selected=[]
    def add(c):
        cfg={k:c[k] for k in ['shared','ratio','extra','selector_depth','compensation_depth','id']}
        if cfg['id'] not in [x['id'] for x in selected]:selected.append(cfg)
    for c in summary[:3]:add(c)
    add(next(x for x in summary if x['ratio']==0))
    add(BASE);add(LATEST)
    add(next(x for x in summary if x['selector_depth']==2))
    add(next(x for x in summary if x['compensation_depth']==2))
    selected.append(R3)
    lock=dict(configs=selected,selection='registered rule, final4000 validation mean only; no nonbank quality accessed',
        screen_best=summary[0],screen_completed=total)
    save_json(ROOT/'confirmation_design.json',lock)
    status('SCREEN_COMPLETE',best=summary[0],finalists=len(selected))
    return selected

def confirm(scene,configs):
    seeds=PROTOCOL['confirm']['seeds']
    results=run_jobs(scene,'confirm',configs,seeds,12000,[4000,8000,12000],('validation',))
    save_json(ROOT/'confirmation_runs.json',results)
    scores=[]
    for cfg in configs:
        vals=[x['validation_score'] for x in results if x['config']['id']==cfg['id']]
        scores.append(dict(config=cfg,mean_validation_psnr=float(np.mean(vals)),seed_std=float(np.std(vals,ddof=1))))
    neural=sorted([x for x in scores if x['config']['id']!='R3'],key=lambda x:(-x['mean_validation_psnr'],x['config']['id']))
    lock=dict(winner=neural[0]['config'],scores=scores,selection='final12000 fr1 validation, eight new seeds; no nonbank quality used',
        test_scene_architecture_locked=True)
    save_json(ROOT/'winner_lock.json',lock)
    status('WINNER_LOCKED',winner=lock['winner'],score=neural[0]['mean_validation_psnr'])
    # Only now open historical fr1 non-Bank. It is a reference, not an untouched new-scene claim.
    targets={lock['winner']['id'],BASE['id'],LATEST['id'],'R3'}
    for info in results:
        if info['config']['id'] in targets:
            full_eval(scene,info,ROOT/'results'/scene.name/'confirm'/info['config']['id']/str(info['seed']),'test')
    return lock['winner']

def external(winner):
    while not (ROOT/'prepare.exit').exists():time.sleep(2)
    assert (ROOT/'prepare.exit').read_text().strip()=='0','Independent map preparation failed'
    scene=Scene('fr2_xyz');configs=[]
    for cfg in [winner,BASE,R3]:
        if cfg['id'] not in [x['id'] for x in configs]:configs.append(cfg)
    seeds=PROTOCOL['final_scene']['seeds']
    results=run_jobs(scene,'external',configs,seeds,12000,[4000,8000,12000],('validation','test'))
    assert scene.unchanged()
    status('ALL_EXPERIMENTS_COMPLETE',winner=winner,external_models=len(results))

if __name__=='__main__':
    scene=Scene()
    # Startup gate only. It never selects a learning rate or structure.
    for cfg in DESIGN:
        net=Network(scene,cfg,2026)
        with torch.no_grad():
            p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten();res=net.compensate(h,ids,.05)
            assert torch.equal(p,.5*scene.eligible) and torch.count_nonzero(res)==0
        del net
    save_json(ROOT/'startup_correctness.json',dict(initial_function_matched=True,initial_compensation_zero=True,
        dimensions_checked=len(DESIGN),production_modified=False))
    configs=screen(scene)
    winner=confirm(scene,configs)
    assert scene.unchanged()
    del scene;gc.collect();torch.cuda.empty_cache()
    external(winner)
