"""Bank内效应模型外推不足时，补足预先指定的单因子Head控制。"""
import gc
import json
import time
from pathlib import Path
import torch
from controller import ROOT,Scene,BASE,LATEST,R3,run_jobs,full_eval,status,identity,save_json

CONTROLS=[dict(shared=s,ratio=0,extra=e,selector_depth=ds,compensation_depth=dc)
          for s,e,ds,dc in [(16,14,0,1),(16,14,2,1),(16,14,1,0),(16,14,1,2),(16,8,1,2)]]
for c in CONTROLS:c['id']=identity(c)

def main():
    while not (ROOT/'matrix.exit').exists():time.sleep(20)
    assert (ROOT/'matrix.exit').read_text().strip()=='0'
    scene=Scene('fr1_xyz')
    original=json.loads((ROOT/'confirmation_runs.json').read_text())
    old_lock=json.loads((ROOT/'winner_lock.json').read_text())
    if not (ROOT/'initial_winner_lock.json').exists():save_json(ROOT/'initial_winner_lock.json',old_lock)
    if not (ROOT/'initial_external_runs.json').exists():
        save_json(ROOT/'initial_external_runs.json',json.loads((ROOT/'external_runs.json').read_text()))
    more=run_jobs(scene,'confirm',CONTROLS,list(range(3100,3108)),12000,[4000,8000,12000],('validation',))
    combined={(r['config']['id'],r['seed']):r for r in original+more}
    combined=list(combined.values());save_json(ROOT/'confirmation_runs.json',combined);save_json(ROOT/'confirm_runs.json',combined)
    configs={r['config']['id']:r['config'] for r in combined};scores=[]
    import numpy as np
    for key,c in configs.items():
        values=[r['validation_score'] for r in combined if r['config']['id']==key]
        assert len(values)==8
        scores.append(dict(config=c,mean_validation_psnr=float(np.mean(values)),seed_std=float(np.std(values,ddof=1))))
    ranked=sorted([r for r in scores if r['config']['id']!='R3'],key=lambda r:(-r['mean_validation_psnr'],r['config']['id']))
    winner=ranked[0]['config']
    lock=dict(winner=winner,scores=scores,selection='final12000 fr1 Bank validation only, same eight seeds; registered five matched controls added before any non-Bank quality inspection',
        initial_winner=old_lock['winner'],test_scene_architecture_locked=True,nonbank_used_for_selection=False)
    save_json(ROOT/'winner_lock.json',lock)
    status('SUPPLEMENT_WINNER_LOCKED',winner=winner)
    targets={winner['id'],BASE['id'],LATEST['id'],'R3'}
    for info in combined:
        if info['config']['id'] in targets:
            full_eval(scene,info,ROOT/'results/fr1_xyz/confirm'/info['config']['id']/str(info['seed']),'test')
    assert scene.unchanged()
    del scene;gc.collect();torch.cuda.empty_cache()
    # Prior primary-run tests are stored but never read by this selection function.
    external=Scene('fr2_xyz')
    final=run_jobs(external,'external',[winner],list(range(4100,4108)),12000,[4000,8000,12000],('validation','test'))
    assert external.unchanged()
    status('ALL_EXPERIMENTS_COMPLETE',winner=winner,matched_control_models=len(more),source='Bank-only supplemented architecture search')
    save_json(ROOT/'supplement_completion.json',dict(controls=CONTROLS,models=len(more),winner=winner,external_models=len(final),nonbank_used_for_selection=False))

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'supplement.exit').write_text('1');raise
    (ROOT/'supplement.exit').write_text('0')
