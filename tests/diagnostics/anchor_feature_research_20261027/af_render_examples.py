"""预定相机示例：首个non-Bank视图及中间视图，不按质量挑图。"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from af_runtime import ROOT,Study,save
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
s=Study(a.scene);nets={}
for name in ['Baseline','FrozenBaseHead8','FrozenBaseContextHead8']:
    tier='confirm_continue' if name=='Baseline' else 'mechanism_continue' if name=='FrozenBaseHead8' and a.scene=='fr1_xyz' else 'frozen_mechanism'
    d=ROOT/'results'/tier/a.scene/name/'7200';meta=json.loads((d/'complete.json').read_text());net=s.make(meta['config'],7200);net.load_state_dict(torch.load(meta['checkpoint'],map_location='cuda')['state_dict']);nets[name]=net
out=ROOT/'figures';out.mkdir(exist_ok=True);views=s.scene.load_test();indices=[0,len(views)//2];records=[]
with torch.no_grad():
    for vi in indices:
        v=views[vi];fig,axes=plt.subplots(3,5,figsize=(17,9))
        for row,r in enumerate([.01,.05,.075]):
            ids=s.fixture.sets[r,'same_k'];images=[v['gt']]
            for name,net in nets.items():images.append(s.scene.render(ids,net(r,ids),v)[0].clamp(0,1))
            images.append(s.scene.render(ids,s.fixture.teacher[r,'same_k'],v)[0].clamp(0,1))
            for col,(name,image) in enumerate(zip(['GT']+list(nets)+['R3 snapshot'],images)):
                axes[row,col].imshow(image.permute(1,2,0).cpu().numpy());axes[row,col].axis('off');score=float(-10*torch.log10((image-v['gt']).square().mean())) if col else 0.
                axes[row,col].set_title(f'{name} | r={r:g}' + (f' | {score:.2f}dB' if col else ''),fontsize=9)
        fig.suptitle(a.scene+' | fixed non-Bank view '+str(vi)+' | same-K | seed7200',fontsize=11);fig.tight_layout();fig.savefig(out/f'{a.scene}_view_{vi}.png',dpi=140);plt.close(fig);records.append(dict(view=vi,path=v['path']))
save(out/(a.scene+'_examples_manifest.json'),dict(views=records,selection='first and middle non-Bank cameras fixed before image inspection',not_evidence_of_mean_quality=True))
print('RENDER_EXAMPLES_COMPLETE',a.scene,flush=True)
