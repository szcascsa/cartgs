"""独立研究预检，复用旧三组指标而不重训或重渲染。"""
import json
from pathlib import Path
import shutil
from common import ROOT,HERE,BASE,PREVIOUS,GROUPS,sha,save


def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/view_conditioned_cross_20261106')
    assert shutil.disk_usage(ROOT).free>5*1024**3
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    assert len(before)==604 and all(sha(n)==v['sha256'] for n,v in before.items())
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    assert json.loads((PREVIOUS/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    for name in ['fr1xyz_camera_config.yaml','fr1xyz_mapping_config.yaml']:
        shutil.copyfile(PREVIOUS/'configs'/name,ROOT/'configs'/name)
    template=json.loads((PREVIOUS/'configs/full_view.json').read_text())
    text=(PREVIOUS/'configs/view.yaml').read_text()
    for group in GROUPS:
        lines=[('ViewBudget.conditioning: '+group) if line.startswith('ViewBudget.conditioning:') else line for line in text.splitlines()]
        selector=ROOT/'configs'/f'{group}.yaml';selector.write_text('\n'.join(lines)+'\n')
        value=dict(template)
        for key in ['camera_config','mapping_config']:value[key]=str(ROOT/'configs'/Path(template[key]).name)
        value['selector_config']=str(selector);value['output']=str(ROOT/'runs'/f'smoke_{group}')
        value['source_snapshot']='reports/SOURCE_HASHES.json'
        value['input_sha256']={key:sha(value[key]) for key in template['input_sha256']}
        save(ROOT/'configs'/f'{group}.json',value)
    protected={str(p):sha(p) for p in [BASE/'runs/full_no_li/step0_selector.pt',BASE/'runs/full_no_li/step0_master.ply',
        BASE/'runs/full_no_li/view_budget_steps.jsonl',PREVIOUS/'evaluation/reports/PER_VIEW.csv',
        PREVIOUS/'evaluation/reports/BASELINE_GEOMETRY_PER_VIEW.csv',PREVIOUS/'evaluation/reports/BASELINE_GEOMETRY_SUMMARY.json']}
    for group in ['mode','view']:
        for p in (PREVIOUS/'runs'/f'full_{group}'/'23560_shutdown/ply').rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
        for name in ['real_lod_steps.csv','view_budget_steps.jsonl']:p=PREVIOUS/'runs'/f'full_{group}'/name;protected[str(p)]=sha(p)
    for p in Path(template['full_ply']).rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    save(ROOT/'reports/HISTORICAL_INPUTS.json',protected)
    shutil.copyfile(HERE/'PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    print(json.dumps(dict(protected_files=len(before),source_files=len(source),input_files=len(protected),free_GiB=shutil.disk_usage(ROOT).free/1024**3)),flush=True)


if __name__=='__main__':main()
