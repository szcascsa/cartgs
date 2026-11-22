"""仅评价预先锁定的终点，完整保存Bank与non-Bank逐视图数据。"""
import argparse,gc,json,runpy
from pathlib import Path
import torch
from af_runtime import ROOT,Study,evaluate,save,csv,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    assert plan['frozen_before_nonbank'];torch.backends.cuda.matmul.allow_tf32=False
    if not (ROOT/'BANK_LOCKED_CANDIDATE.json').exists():runpy.run_path(str(ROOT/'af_bank_decision.py'),run_name='__main__')
    study=Study(plan.get('scene','fr1_xyz'))
    for item in plan['models']:
        directory=Path(item['directory']);info=json.loads((directory/'complete.json').read_text());payload=torch.load(info['checkpoint'],map_location='cuda')
        if info['config'].get('grouping')=='signature':study.fake_owners=torch.load(ROOT/f'signature_fake_{study.scene.name}.pt',map_location='cuda')
        net=study.make(info['config'],info['seed'],info['route']);net.load_state_dict(payload['state_dict']);net.eval()
        assert torch.equal(payload['uids'],study.uids) and torch.equal(payload['owners'],study.owners)
        target=directory/'formal';target.mkdir(exist_ok=True)
        for split in ['bank_train','validation','test']:
            path=target/(split+'_per_view.csv')
            if path.exists():continue
            rows=evaluate(study,net,split,True)
            for row in rows:row.update(config=info['config']['id'],seed=info['seed'],steps=info['steps'],prior_steps=info['prior_steps'],tier=info['tier'],route=info['route'])
            csv(path,rows)
        save(target/'manifest.json',dict(checkpoint=info['checkpoint'],sha256=sha(info['checkpoint']),no_selection_on_nonbank=True,plan=plan,
            metrics='PSNR clipped image; SSIM torchmetrics; LPIPS alex normalized; raw renderer photo .8L1+.2DSSIM',effective_count='radii>0 and alpha>=1/255 proxy; not pixel-integrated contribution'))
        print('FORMAL',info['config']['id'],info['seed'],info['route'],study.scene.name,flush=True)
        del net,payload;gc.collect();torch.cuda.empty_cache()
    ref=ROOT/'references'/study.scene.name;ref.mkdir(parents=True,exist_ok=True)
    for split in ['bank_train','validation','test']:
        path=ref/(split+'_per_view.csv')
        if not path.exists():csv(path,evaluate(study,None,split,True))
    initial=study.make(dict(feature='none'),7200)
    for split in ['bank_train','validation','test']:
        path=ref/('spatial_initial_'+split+'_per_view.csv')
        if not path.exists():csv(path,evaluate(study,initial,split,True))
    save(ROOT/(Path(a.plan).stem+'_complete.json'),dict(models=len(plan['models']),scene=study.scene.name))
if __name__=='__main__':main()
