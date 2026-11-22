"""核验独立快照及历史输入，生成两组配置。"""
import json
from pathlib import Path
import shutil
from common import ROOT,HERE,BASE,OLD,sha,save


def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/view_conditioned_joint_20261106')
    assert shutil.disk_usage(ROOT).free>5*1024**3
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    assert len(before)==604 and all(sha(n)==v['sha256'] for n,v in before.items())
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    assert json.loads((OLD/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    for name in ['fr1xyz_camera_config.yaml','fr1xyz_mapping_config.yaml']:
        shutil.copyfile(OLD/'configs'/name,ROOT/'configs'/name)
    template=json.loads((OLD/'configs/raw_no_li.json').read_text())
    text=(OLD/'configs/off.yaml').read_text()
    for group in ['mode','view']:
        lines=[]
        for line in text.splitlines():
            if line.startswith('ViewBudget.view_residual:'):line='ViewBudget.view_residual: 0'
            if line.startswith('ViewBudget.li_weight:'):line='ViewBudget.li_weight: 0.0'
            lines.append(line)
        selector=ROOT/'configs'/f'{group}.yaml'
        selector.write_text('\n'.join(lines)+f'\nViewBudget.conditioning: {group}\n')
        value=dict(template)
        for key in ['camera_config','mapping_config']:value[key]=str(ROOT/'configs'/Path(template[key]).name)
        value['selector_config']=str(selector);value['output']=str(ROOT/'runs'/f'smoke_{group}')
        value['source_snapshot']='reports/SOURCE_HASHES.json'
        value['input_sha256']={key:sha(value[key]) for key in template['input_sha256']}
        save(ROOT/'configs'/f'{group}.json',value)
    protected={str(p):sha(p) for p in [BASE/'evaluation/reports/PER_VIEW.csv',BASE/'runs/full_no_li/step0_selector.pt',
        BASE/'runs/full_no_li/step0_master.ply',BASE/'runs/full_no_li/view_budget_steps.jsonl']}
    for p in Path(template['full_ply']).rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    save(ROOT/'reports/HISTORICAL_INPUTS.json',protected)
    shutil.copyfile(HERE/'PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    print(json.dumps(dict(protected_files=len(before),source_files=len(source),input_files=len(protected),free_GiB=shutil.disk_usage(ROOT).free/1024**3)),flush=True)


if __name__=='__main__':main()
