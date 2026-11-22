"""后验归因：Gaussian被选过，不等于在目标ratio获得过两侧训练支持。"""
import json,types
import torch
from af_runtime import ROOT,Study,TRAIN,RATIOS,evaluate,csv,save


def main():
    torch.backends.cuda.matmul.allow_tf32=False;counts=[];rows=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        s=Study(scene);lo=torch.full((s.scene.N,),float('inf'),device='cuda');hi=torch.zeros_like(lo)
        for r in TRAIN:
            ids=s.fixture.sets[r,'same_k'];lo[ids]=torch.minimum(lo[ids],lo.new_full((len(ids),),r));hi[ids]=torch.maximum(hi[ids],hi.new_full((len(ids),),r))
        for r in RATIOS:
            ids=s.fixture.sets[r,'same_k'];counts.append(dict(scene=scene,ratio=r,K=len(ids),below_min_train_ratio=int((r<lo[ids]-1e-7).sum()),above_max_train_ratio=int((r>hi[ids]+1e-7).sum()),never_selected=int((hi[ids]==0).sum())))
        tier='mechanism_continue' if scene=='fr1_xyz' else 'frozen_mechanism'
        for seed in [7200,7201]:
            directory=ROOT/'results'/tier/scene/'FrozenBaseHead8'/str(seed);meta=json.loads((directory/'complete.json').read_text());net=s.make(meta['config'],seed);net.load_state_dict(torch.load(meta['checkpoint'],map_location='cuda')['state_dict']);original=net.latent
            for mode in ['normal','no_ratio_extrapolated','only_ratio_extrapolated']:
                def filtered(self,ids,r):
                    outside=(r<lo[ids]-1e-7)|(r>hi[ids]+1e-7)
                    keep=~outside if mode=='no_ratio_extrapolated' else outside
                    return original(ids,r)*keep[:,None]
                if mode!='normal':net.latent=types.MethodType(filtered,net)
                for split in ['validation','test']:
                    data=evaluate(s,net,split,False)
                    for row in data:row.update(seed=seed,ablation=mode)
                    rows+=data;csv(ROOT/'ratio_support_counterfactual_per_view.csv',rows)
                net.latent=original
    csv(ROOT/'per_gaussian_ratio_support.csv',counts)
    save(ROOT/'ratio_support_manifest.json',dict(posthoc=True,training_or_model_selection=False,fixed_K=True,definition='min/max ratio where Gaussian identity is selected for Bank optimization; not a guarantee of visible photo gradient',limit='conditional final function ablation, effects not additive; no new model or train configuration'))
    print('RATIO_SUPPORT_DIAG_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
