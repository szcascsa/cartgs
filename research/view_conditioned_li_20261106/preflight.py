"""独立快照与只读历史输入核验，生成LI=0.1的三组配置。"""
import json
from pathlib import Path
import shutil
from common import ROOT,HERE,BASE,PREVIOUS,GROUPS,MODES,HISTORY_GROUPS,history_run,sha,save


def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106')
    assert shutil.disk_usage(ROOT).free>5*1024**3
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    assert len(before)==604 and all(sha(n)==v['sha256'] for n,v in before.items())
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    assert json.loads((PREVIOUS/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    for name in ['fr1xyz_camera_config.yaml','fr1xyz_mapping_config.yaml']:
        shutil.copyfile(PREVIOUS/'configs'/name,ROOT/'configs'/name)
    template=json.loads((PREVIOUS/'configs/full_selector_view.json').read_text())
    text=(PREVIOUS/'configs/selector_view.yaml').read_text().replace('本轮LI关闭','本轮LI=0.1')
    for group in GROUPS:
        lines=[]
        for line in text.splitlines():
            if line.startswith('ViewBudget.conditioning:'):line='ViewBudget.conditioning: '+MODES[group]
            if line.startswith('ViewBudget.li_weight:'):line='ViewBudget.li_weight: 0.1'
            lines.append(line)
        selector=ROOT/'configs'/f'{group}.yaml';selector.write_text('\n'.join(lines)+'\n')
        value=dict(template)
        for key in ['camera_config','mapping_config']:value[key]=str(ROOT/'configs'/Path(template[key]).name)
        value['selector_config']=str(selector);value['output']=str(ROOT/'runs'/f'smoke_{group}')
        value['source_snapshot']='reports/SOURCE_HASHES.json'
        value['input_sha256']={key:sha(value[key]) for key in template['input_sha256']}
        save(ROOT/'configs'/f'{group}.json',value)
    protected={str(p):sha(p) for p in [BASE/'runs/full_no_li/step0_selector.pt',BASE/'runs/full_no_li/step0_master.ply',
        BASE/'runs/full_no_li/view_budget_steps.jsonl',PREVIOUS/'evaluation/reports/PER_VIEW.csv',PREVIOUS/'evaluation/reports/SUMMARY.json',
        PREVIOUS/'evaluation/reports/BASELINE_GEOMETRY_PER_VIEW.csv',PREVIOUS/'evaluation/reports/BASELINE_GEOMETRY_SUMMARY.json']}
    for g in HISTORY_GROUPS:
        folder=history_run(g)
        for p in (folder/'23560_shutdown/ply').rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
        for name in ['real_lod_steps.csv','view_budget_steps.jsonl']:
            p=folder/name;protected[str(p)]=sha(p)
    for p in Path(template['full_ply']).rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    save(ROOT/'reports/HISTORICAL_INPUTS.json',protected)
    shutil.copyfile(HERE/'PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    print(json.dumps(dict(protected_files=len(before),source_files=len(source),input_files=len(protected),free_GiB=shutil.disk_usage(ROOT).free/1024**3)),flush=True)


if __name__=='__main__':main()
