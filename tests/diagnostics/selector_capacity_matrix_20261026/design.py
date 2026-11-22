"""预注册混合因子D-optimal矩阵；不读取任何渲染质量。"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
LEVELS=dict(shared=[0,8,16,32],ratio=[0,4,8,16],extra=[0,4,8,14,28],selector_depth=[0,1,2],compensation_depth=[0,1,2])
INTERACTIONS=[('shared','ratio'),('shared','selector_depth'),('ratio','selector_depth'),('extra','compensation_depth'),('selector_depth','compensation_depth')]
def identity(c):
    return f"S{c['shared']:02d}R{c['ratio']:02d}E{c['extra']:02d}D{c['selector_depth']}{c['compensation_depth']}"
def contrasts(v,levels):
    x=np.zeros(len(levels)-1);j=levels.index(v)
    if j<len(x):x[j]=1
    else:x[:]=-1
    return x
def features(c):
    blocks={k:contrasts(c[k],v) for k,v in LEVELS.items()}
    return np.concatenate([[1.],*blocks.values(),*[np.outer(blocks[a],blocks[b]).flatten() for a,b in INTERACTIONS]])

def create():
    configs=[dict(zip(LEVELS,v)) for v in itertools.product(*LEVELS.values())]
    for c in configs:c['id']=identity(c)
    x=np.stack([features(c) for c in configs]);scale=np.sqrt(np.mean(x*x,axis=0));x=x/scale
    required=[dict(shared=s,ratio=r,extra=e,selector_depth=ds,compensation_depth=dc) for s,r,e,ds,dc in [
        (16,0,14,1,1),(8,8,14,1,1),(16,8,14,1,1),(32,0,14,1,1),
        (16,0,0,1,1),(0,16,14,1,1),(0,0,0,1,1),(16,8,28,2,2)]]
    fixed=[next(i for i,c in enumerate(configs) if c['id']==identity(v)) for v in required]
    chosen=list(fixed)
    ridge=np.eye(x.shape[1])*1e-7
    while len(chosen)<72:
        inv=np.linalg.inv(ridge+x[chosen].T@x[chosen]);lev=np.sum((x@inv)*x,axis=1);lev[chosen]=-np.inf
        chosen.append(int(np.argmax(lev)))
    for _ in range(12):
        exchanges=0
        for slot in range(len(fixed),len(chosen)):
            inv=np.linalg.inv(ridge+x[chosen].T@x[chosen]);old=x[chosen[slot]]
            lev=np.sum((x@inv)*x,axis=1);h=float(old@inv@old);cross=x@inv@old
            gains=(1-h)*(1+lev)+cross*cross;gains[chosen]=-np.inf;j=int(np.argmax(gains))
            if gains[j]>1+1e-7:chosen[slot]=j;exchanges+=1
        if not exchanges:break
    selected=[configs[i] for i in chosen]
    rank=int(np.linalg.matrix_rank(x[chosen]));assert rank==x.shape[1],(rank,x.shape)
    protocol=dict(factors=LEVELS,private_head_width=32,encoder_width=32,encoder_hidden_layers=1,
        full_grid_size=len(configs),screen_configs=72,design='categorical effect-coded D-optimal; mandatory controls plus coordinate exchange',
        interactions=INTERACTIONS,design_rank=rank,design_columns=x.shape[1],condition_number=float(np.linalg.cond(x[chosen])),
        source_scene='fr1_xyz',source_checkpoint='/home/prmi/zwj/anchor-shared16-ratio8-20261026/r3_independent14_seed2026/6400_shutdown/ply',
        frozen=['Master all attributes','Gaussian population/IDs','Anchor owners/centers/scales','eligibility','camera poses','cached uniform'],
        trained=['new shared/ratio Anchor features','new encoder','new Selector head','new Compensation head','new per-Anchor per-ratio extra features'],
        initializer='all features zero; common attribute/logratio encoder block; common nonzero latent input columns; Selector final zero; Compensation zero output, nonzero extra columns for linear head to avoid dead bilinear branch',
        scalar_ratio='log(ratio), fixed in every configuration',feature_interpolation='linear raw ratio; endpoint clamp; no extra modulation network',
        training_ratios=[.01,.05,.1,.15],unseen_ratios=[.025,.075,.125],
        loss='0.8L1+0.2(1-fast_ssim) + 0.01*abs(sum(p)-r*N)/max(1,r*N)',
        selection='original cached-uniform Bernoulli, selected-only opacity ST; original compact renderer; hard same-K margin diagnostic',
        optimizer=dict(name='Adam',implementation='PyTorch fused CUDA, same implementation in all formal arms',lr=.001,eps=1e-15,betas=[.9,.999],weight_decay=0),
        schedule='one shuffled Bank camera per cycle, ratio1/5/10/15 in order; no early stop; common seed schedules',
        screen=dict(steps=4000,exposures_per_ratio=1000,seeds=[2026,2027],ranking='mean PSNR on held-out Bank cameras, equal weights seven ratios',
            diagnostic_checkpoints=[1000,2000,4000],nonbank_read_for_selection=False),
        validation='each fifth Bank camera held out; training only remaining cameras; no map/camera optimization; source Master itself previously saw Bank',
        confirm=dict(steps=12000,exposures_per_ratio=3000,seeds=list(range(3100,3108)),
            candidates='top3 distinct screen configs + best all-shared + original16 shared baseline + latest16+8 baseline + best selector-depth2 + best compensation-depth2, deduplicate; plus matched R3 reference',
            checkpoint_selection='final12000 only; 4000/8000/12000 curves for convergence, not test-driven step choice'),
        final_scene=dict(name='fr2_xyz',source='one fresh default R3 SLAM map, then freeze same map for all finalists',
            configs='locked fr1 validation winner + original16 shared baseline + R3 reference',steps=12000,seeds=list(range(4100,4108)),
            test='non-Bank views fixed before quality; no architecture reselection using this scene'),
        statistics=dict(primary='equal mean PSNR across seven ratios on held-out Bank for architecture selection',
            practical_margin_db=.1,interval='paired seed plus contiguous camera-block hierarchical bootstrap, 95%',
            confirm_test='exact paired seed sign permutation with Holm family correction; report raw view effects separately',
            winner_probability='bootstrap probability of best, not a guarantee of global optimum',
            equivalence='if differences not resolved, report statistically indistinguishable candidate set rather than a unique optimum',
            model_effects='main factors plus registered interactions; grouped leave-one-configuration-out validation; effect-model assumptions explicit'),
        final_evaluation='PSNR SSIM LPIPS photo, Cached actualK + common deterministic sameK; train/validation/test kept separate',
        execution_workers=2,execution='independent CUDA processes; no loss or optimizer merging; completed jobs resumable',
        no_production_changes=True,no_stage_commit_push=True)
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    (BASE/'screen_design.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    np.savez(BASE/'design_matrix.npz',full=x,selected=np.array(chosen),scales=scale)
    print(json.dumps(dict(configs=len(selected),full=len(configs),rank=rank,condition=protocol['condition_number'])))
if __name__=='__main__':create()
