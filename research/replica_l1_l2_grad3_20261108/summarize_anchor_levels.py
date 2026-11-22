"""只读最终Full checkpoint，统计L0/L1/L2 Anchor及其成员Gaussian占比。"""
from pathlib import Path
import hashlib
import json
import sys
import time
# 避免批次queue.py遮蔽PyTorch使用的Python标准库queue。
_HERE=Path(__file__).resolve().parent
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=_HERE]
import torch

R=Path('/home/prmi/zwj/realLOD/checkpoint/replica_l1_l2_grad3_20261108')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    jobs=json.loads((R/'QUEUE.json').read_text());assert len(jobs)==16
    rows=[]
    for job in jobs:
        assert job['state']=='complete',job['id']
        model=Path(job['result']['checkpoint'])
        archive,=model.rglob('anchor_pruning_state.pt')
        metadata,=model.rglob('anchor_selector_state.json')
        before={str(p):sha(p) for p in [archive,metadata]}
        native=torch.jit.load(str(archive),map_location='cpu')
        buffers=dict(native.named_buffers())
        levels=buffers['levels'].reshape(-1).to(torch.int64)
        owner=buffers['gaussian_owner'].reshape(-1).to(torch.int64)
        members=buffers['member_counts'].reshape(-1).to(torch.int64)
        state=json.loads(metadata.read_text())
        assert len(levels)==len(members)==state['anchor_count']==job['result']['anchor_count']
        assert len(owner)==state['gaussian_count']==job['result']['gaussian_count']
        assert torch.all((levels>=0)&(levels<=2)).item()
        assert torch.all((owner>=0)&(owner<len(levels))).item()
        assert torch.equal(torch.bincount(owner,minlength=len(levels)),members)
        counts=torch.bincount(levels,minlength=3).tolist()
        gaussian_counts=torch.bincount(levels[owner],minlength=3).tolist()
        assert sum(counts)==len(levels) and sum(gaussian_counts)==len(owner)
        assert all(sha(Path(p))==h for p,h in before.items())
        rows.append(dict(scene=job['scene'],mode=job['mode'],checkpoint=str(model),
                    anchor_count=len(levels),gaussian_count=len(owner),
                    anchor_level_counts=counts,gaussian_level_counts=gaussian_counts,
                    l2_anchor_fraction=counts[2]/len(levels),
                    l2_gaussian_fraction=gaussian_counts[2]/len(owner),input_sha256=before))
    output=dict(checked_unix=time.time(),experiment='L1-to-L2 3*tau0 Full+tail',
                definition='L2 Anchor数/全部Anchor数；成员占比=L2 Anchor拥有的Gaussian数/全部Gaussian数',
                source='最终checkpoint anchor_pruning_state.pt中的levels、gaussian_owner、member_counts',rows=rows)
    (R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# Replica各场景Anchor Level 2占比','',
           '本表读取本次3τ₀阈值实验最终Full＋tail checkpoint，未重新训练或渲染。',
           'Anchor2指level=2；占比为L2 Anchor数除以全部Anchor数。成员Gaussian占比按所有存储点统计，包含inactive slot，不是可见点比例。','',
           '|场景|Mono L2／全部Anchor|占比|RGB-D L2／全部Anchor|占比|',
           '|---|---:|---:|---:|---:|']
    scenes=list(dict.fromkeys(r['scene'] for r in rows));lookup={(r['scene'],r['mode']):r for r in rows}
    for scene in scenes:
        m,d=lookup[scene,'mono'],lookup[scene,'rgbd']
        lines.append(f"|{scene}|{m['anchor_level_counts'][2]:,}／{m['anchor_count']:,}|{m['l2_anchor_fraction']:.2%}|{d['anchor_level_counts'][2]:,}／{d['anchor_count']:,}|{d['l2_anchor_fraction']:.2%}|")
    lines+=['','|场景|Mono L2成员Gaussian占比|RGB-D L2成员Gaussian占比|','|---|---:|---:|']
    for scene in scenes:
        m,d=lookup[scene,'mono'],lookup[scene,'rgbd']
        lines.append(f"|{scene}|{m['l2_gaussian_fraction']:.2%}|{d['l2_gaussian_fraction']:.2%}|")
    lines+=['','逐组L0/L1/L2原始计数、模型路径、源文件哈希见ANCHOR_LEVEL_DISTRIBUTION.json。',
            '16组均核验Anchor数量、Gaussian数量、owner范围、member_counts与owner逐行计数一致；统计前后源文件SHA256相同。']
    text='\n'.join(lines)+'\n'
    (R/'reports/ANCHOR_LEVEL_DISTRIBUTION.md').write_text(text)
    print(text)


if __name__=='__main__':main()
