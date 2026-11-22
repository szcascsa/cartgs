"""保持每Anchor成员数及各训练ratio访问次数的随机分组控制。"""
import argparse,gc,json
from pathlib import Path
import torch
from af_runtime import ROOT,Study,TRAIN,train,save,sha


def matched_fake(study):
    signatures=torch.zeros(study.scene.N,dtype=torch.long,device='cuda')
    for j,ratio in enumerate(TRAIN):signatures[study.fixture.sets[ratio,'same_k']]+=1<<j
    key=study.counts[study.owners]*16+signatures
    generator=torch.Generator(device='cuda').manual_seed(8712);permutation=torch.arange(study.scene.N,device='cuda')
    for value in key.unique():
        ids=(key==value).nonzero().flatten();permutation[ids]=ids[torch.randperm(len(ids),device='cuda',generator=generator)]
    fake=study.owners[permutation]
    assert torch.equal(torch.bincount(fake,minlength=study.M),study.counts)
    for ratio in TRAIN:
        ids=study.fixture.sets[ratio,'same_k']
        assert torch.equal(torch.bincount(fake[ids],minlength=study.M),torch.bincount(study.owners[ids],minlength=study.M))
    path=ROOT/f'signature_fake_{study.scene.name}.pt';torch.save(fake,path)
    save(ROOT/f'signature_fake_{study.scene.name}.json',dict(sha256=sha(path),changed_gaussian_fraction=float((fake!=study.owners).float().mean()),
        exact_population_counts=True,exact_each_training_ratio_counts=True,construction='shuffle Gaussian ownership within occupancy x four-ratio membership signature; physical inputs and real Anchor scales unchanged'))
    return fake


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);a=p.parse_args();q=json.loads(Path(a.queue).read_text())
    torch.backends.cuda.matmul.allow_tf32=False;study=Study(q.get('scene','fr1_xyz'));study.fake_owners=matched_fake(study);runs=[]
    for job in q['jobs']:
        runs.append(train(study,job,job.get('tier',q['tier'])));save(ROOT/(Path(a.queue).stem+'_results.json'),runs);gc.collect();torch.cuda.empty_cache()
    save(ROOT/(Path(a.queue).stem+'_complete.json'),dict(runs=len(runs)))
if __name__=='__main__':main()
