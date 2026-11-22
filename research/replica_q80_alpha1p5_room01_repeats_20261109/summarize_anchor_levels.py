"""逐次核验额外重复checkpoint的Anchor层级与owner计数。"""
from pathlib import Path
import sys
_HERE=Path(__file__).resolve().parent
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=_HERE]
import torch
sys.path.insert(0,str(_HERE))
from prepare_experiment import R,read,write,sha


def main():
    jobs=read(R/'QUEUE.json');assert len(jobs)==8
    rows=[]
    for job in jobs:
        assert job['state']=='complete'
        model=Path(job['result']['checkpoint']);archive,=model.rglob('anchor_pruning_state.pt')
        metadata,=model.rglob('anchor_selector_state.json');before={str(p):sha(p) for p in [archive,metadata]}
        native=torch.jit.load(str(archive),map_location='cpu');buffers=dict(native.named_buffers())
        levels=buffers['levels'].reshape(-1).to(torch.int64)
        owner=buffers['gaussian_owner'].reshape(-1).to(torch.int64)
        members=buffers['member_counts'].reshape(-1).to(torch.int64);meta=read(metadata)
        assert len(levels)==len(members)==meta['anchor_count']==job['result']['anchor_count']
        assert len(owner)==meta['gaussian_count']==job['result']['gaussian_count']
        assert torch.all((levels>=0)&(levels<=2)).item() and torch.all((owner>=0)&(owner<len(levels))).item()
        assert torch.equal(torch.bincount(owner,minlength=len(levels)),members)
        ac=torch.bincount(levels,minlength=3).tolist();gc=torch.bincount(levels[owner],minlength=3).tolist()
        assert sum(ac)==len(levels) and sum(gc)==len(owner)
        assert all(sha(Path(p))==value for p,value in before.items())
        rows.append(dict(id=job['id'],scene=job['scene'],mode=job['mode'],repeat=job['repeat'],
            checkpoint=str(model),anchor_count=len(levels),gaussian_count=len(owner),anchor_level_counts=ac,
            gaussian_level_counts=gc,l2_anchor_fraction=ac[2]/len(levels),l2_gaussian_fraction=gc[2]/len(owner),input_sha256=before))
    write(R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json',dict(rows=rows,
        definition='L2 Anchor数/全部Anchor数；L2成员数/全部存储Gaussian数，包含inactive slot'))
    print('VERIFIED_8_REPEAT_ANCHOR_CHECKPOINTS',flush=True)


if __name__=='__main__':main()
