"""三项预指定Bank-only长预算诊断；复用原binary，0..2000重放以恢复Adam历史。"""
from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np

ROOT=Path(__file__).resolve().parent
GRID=ROOT.parent/'grid-mlp-implicit-si-20261020'
OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz']

def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def records(path):return [json.loads(line[3:])for line in path.read_text().splitlines()if line.startswith('PG ')]

def main():
    binary=GRID/'build/grid_worker'
    protocol=dict(scenes=SCENES,ratio=.05,seed=2026,steps=5000,lr=.0003,
        reason='fixed scene list; same case as prior LR sanity extended to all three scenes; no evaluation-based selection',
        comparison='same original executable; rerun 0..2000 with original Adam history, check selected rows exact at2000, then inspect2000..5000',
        binary_SHA=hashlib.sha256(binary.read_bytes()).hexdigest(),grid_changed=False,loss_changed=False,architecture_changed=False,
        no_early_stop=True,no_nonBank_read=True,no_lr_search=True)
    save('continuation_preregistration.json',protocol);summary=[]
    for scene in SCENES:
        save('continuation_status.json',dict(stage='running',scene=scene,completed=len(summary),total=3))
        out=ROOT/'continuation'/scene;out.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,LD_LIBRARY_PATH=str(OLD/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
        command=[str(binary),str(OLD/scene/'training_manifest.json'),str(OLD/scene/'training_input.bin'),str(GRID/scene/'grid.bin'),str(out),'.05','2026','5000','.0003','mlp']
        start=time.time()
        with open(out/'training.log','w')as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=records(out/'training.log');old=records(GRID/scene/'models/Grid-MLP_5_seed_2026/training.log')
        assert rows[-1]['completed']and rows[-1]['frozen_exact']and rows[-1]['grid_frozen_exact']
        tr=[r for r in rows if r['kind']=='training'];otr=[r for r in old if r['kind']=='training']
        assert len(tr)==5000 and [r['frame']for r in tr[:2000]]==[r['frame']for r in otr]
        hard={r['step']:r for r in rows if r['kind']=='hard_summary'};original={r['step']:r for r in old if r['kind']=='hard_summary'}
        assert hard[0]['selected_source_rows']==original[0]['selected_source_rows']
        replay_all_sets=all(hard[step]['selected_source_rows']==r['selected_source_rows']for step,r in original.items())
        assert replay_all_sets,'STOP_REPLAY_HARD_SETS_DIFFER'
        replay_metric_error=max(abs(hard[step][metric]-r[metric])for step,r in original.items()for metric in ['photo','psnr'])
        assert replay_metric_error==0,'STOP_REPLAY_HARD_METRICS_DIFFER'
        z=dict(scene=scene,ratio=.05,seed=2026,steps=5000,seconds=time.time()-start,
               replay_all_hard_sets_exact=True,replay_hard_metrics_max_abs=replay_metric_error,
               replay_training_photo_max_abs=max(abs(a['sampled_photo']-b['sampled_photo'])for a,b in zip(tr,otr)),
               checkpoints=[{k:hard[step][k]for k in ['step','photo','psnr','added','changed_cells','budget_transferred']}for step in [0,250,1000,2000,3000,4000,5000]],
               photo_final_minus_2000=hard[5000]['photo']-hard[2000]['photo'],photo_final_minus_Raw=hard[5000]['photo']-hard[0]['photo'],
               PSNR_final_minus_2000=hard[5000]['psnr']-hard[2000]['psnr'],PSNR_final_minus_Raw=hard[5000]['psnr']-hard[0]['psnr'],
               frozen_exact=True,grid_frozen_exact=True)
        summary.append(z);save('continuation_results.json',dict(all_passed=True,models=summary,completed=len(summary)));print(json.dumps(z),flush=True)
    save('continuation_status.json',dict(stage='complete',models=3,updates=15000))

if __name__=='__main__':
    try:main()
    except Exception as error:save('continuation_status.json',dict(stage='failed',error=str(error)));raise
