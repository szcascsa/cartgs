"""交付检查：预算/身份/源代码未变、完整逐视图指标及无Gaussian私有参数。"""
import hashlib,json
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/prmi/zwj/mycartgs')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows=[];orders={};sources={};counters={}
    for p in ROOT.glob('results/*/*/*/*/complete.json'):
        d=json.loads(p.read_text());key=(d['scene'],d['seed'],d['steps']);order=d['frame_order_sha256']
        if key in orders:assert orders[key]==order,('frame order',p)
        orders[key]=order;source=d['source_checkpoint_sha256']
        if d['scene'] in sources:assert sources[d['scene']]==source,('source',p)
        sources[d['scene']]=source;assert d['step0_max_abs']==0 and all(n==d['steps']//4 for n in d['ratio_exposures'].values())
        assert not any('gaussian' in v['name'].lower() for v in d['parameter_inventory'])
        clocks=d['optimizer_steps'];base=[v for k,v in clocks.items() if k.startswith('base.')];extra=[v for k,v in clocks.items() if not k.startswith('base.')]
        expected=d['steps']+(12000 if d['route']=='continue' and not d['config'].get('fresh_base_adam') else 0)
        assert all(v==expected for v in base),(p,base,expected)
        assert all(v==d['steps'] for v in extra),(p,extra)
        for path in (p.parent/'formal').glob('*_per_view.csv'):
            data=pd.read_csv(path);assert not data[['psnr','ssim','lpips','photo']].isna().any().any()
            fixture=json.loads((ROOT/('fixture_'+d['scene']+'.json')).read_text())
            assert all(int(row.selected)==fixture['K'][str(row.ratio)] for row in data.itertuples())
        counters[d['tier']]=counters.get(d['tier'],0)+1
    env=json.loads((ROOT/'environment.json').read_text());production={name:digest(REPO/name)==value for name,value in env['production_sha256'].items()};assert all(production.values())
    for p in ROOT.glob('formal_plan_*.json'):
        if p.name.endswith('_complete.json'):continue
        plan=json.loads(p.read_text());assert plan['frozen_before_nonbank']
        for item in plan['models']:
            directory=Path(item['directory']);assert (directory/'complete.json').exists() and (directory/'formal/test_per_view.csv').exists(),directory
    locked=(ROOT/'BANK_LOCKED_CANDIDATE.json').stat().st_mtime
    assert locked<=min(p.stat().st_mtime for p in ROOT.glob('results/*/*/*/*/formal/*_per_view.csv'))
    record=dict(passed=True,production_sha_unchanged=production,runs_by_tier=counters,source_checkpoints=sources,frame_schedule_groups=len(orders),
        limits='source/K/optimizer/exposure/file completeness check; not proof of online lifecycle or reconstruction generalization')
    (ROOT/'evidence_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
if __name__=='__main__':main()
