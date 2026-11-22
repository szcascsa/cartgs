"""预登记设计后评价；Bank/nonBank逐视图证据与模型选择分开。"""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
import run as r
from neural import audit_parameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args();plan=json.loads(Path(args.plan).read_text())
    assert plan['frozen_before_nonbank'];scene=r.rt.Scene(plan.get('scene','fr1_xyz'))
    fixture=r.Fixture(scene,plan.get('source_seed',3100),plan.get('source_family','R3'));done=[]
    for item in plan['models']:
        directory=Path(item['directory'])
        while not(directory/'complete.json').exists():time.sleep(10)
        info=json.loads((directory/'complete.json').read_text());cfg=info['config'];net=r.make(fixture,cfg,info['seed'])
        payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict']);net.eval();output=directory/'formal';output.mkdir(exist_ok=True)
        if cfg['kind']!='reference59':r.rt.save_json(output/'trainable_parameters.json',dict(pure_shared=True,parameters=audit_parameters(net),identity_trainable=False))
        for split in ['bank_train','validation','test']:
            target=output/(split+'_per_view.csv')
            if target.exists():continue
            rows=r.evaluate(fixture,net,split,full=True,policies=('same_k','cached'))
            for row in rows:row.update(config=cfg['id'],seed=info['seed'],scene=scene.name,tier=info['tier'],steps=info['steps'],total_steps=info['steps']+info.get('prior_steps',0))
            r.rt.write_csv(target,rows)
        with torch.no_grad():
            values={f'{ratio}_{policy}':net(ratio,ids,policy).cpu().numpy() for (ratio,policy),ids in fixture.sets.items()}
            r.np.savez_compressed(output/'compensation.npz',**values)
        assert scene.unchanged()
        r.rt.save_json(output/'manifest.json',dict(model=cfg,seed=info['seed'],source=fixture.manifest,checkpoint=info['checkpoint'],
            checkpoint_sha256=r.rt.sha(info['checkpoint']),code_sha256=info['code_sha256'],
            evaluation_code_sha256={n:r.rt.sha(r.ROOT/n) for n in ['evaluate.py','run.py','neural.py']},
            no_model_selection_on_test=True,counts='radii and alpha threshold proxy; not pixel-integrated contribution'))
        done.append(str(directory));print('EVALUATED',cfg['id'],info['seed'],scene.name,flush=True)
        del net,payload;gc.collect();torch.cuda.empty_cache()
    for split in ['bank_train','validation','test']:
        ref=r.ROOT/'references'/scene.name/fixture.source_family/str(fixture.source_seed);ref.mkdir(parents=True,exist_ok=True)
        target=ref/(split+'_per_view.csv')
        if not target.exists():r.rt.write_csv(target,r.evaluate(fixture,None,split,full=True,policies=('same_k','cached')))
    r.rt.save_json(r.ROOT/(Path(args.plan).stem+'_complete.json'),dict(models=done,fixture=fixture.manifest))


if __name__=='__main__':main()
