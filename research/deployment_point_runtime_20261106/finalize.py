from common import *
import csv
import shutil
import subprocess
import sys
import tarfile
import time
import unittest
import zipfile


def main():
    reports=ROOT/'reports'
    if 'RELEASE_CHECKS_PASS' not in (ROOT/'logs/release_runner.log').read_text():
        raise RuntimeError('Release GPU checks are incomplete')
    assert json.loads((reports/'QUALITY.json').read_text())['passed']
    assert json.loads((reports/'PERCEPTUAL.json').read_text())['passed']
    for name in ['API_VALIDATION','CHECKPOINT_LOAD','GEOMETRY']:
        assert json.loads((reports/(name+'.json')).read_text())['passed'],name
    # Exercise the actual independent command line under inference_mode.
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'source/python'),str(OLD/'evaluation/python'),
        str(OLD/'evaluation/python/diff-gaussian-rasterization')])
    output=ROOT/'evaluation/CLI_SMOKE.csv'
    subprocess.check_call([sys.executable,str(ROOT/'source/python/deployment_infer.py'),str(CHECKPOINTS['both_li']),
        '--dataset',DATASET,'--frames','6','--budget','2000','--output',str(output)],env=env)
    with output.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==6 and all(int(v['K'])==2000 and int(v['candidate_N'])>=2000 for v in rows)
    save(reports/'CLI_VALIDATION.json',dict(passed=True,frames=6,entry='python/deployment_infer.py',
        images_saved=0,uses_inference_mode=True))
    # Existing CPU tests prove the normal entry points retain their original contracts.
    tests=Path('/home/prmi/zwj/experiments/dynamic_inference_fusion_20261106/source/tests')
    dest=ROOT/'source/tests';dest.mkdir(exist_ok=True)
    for name in ['anchor_budget_test.py','anchor_budget_cli_test.py','budget_viewer_test.py','anchor_lod_test.py']:
        shutil.copy2(tests/name,dest/name)
    sys.path.insert(0,str(dest))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['anchor_budget_test','anchor_budget_cli_test','budget_viewer_test'])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and result.testsRun==24
    save(reports/'NORMAL_UNIT_VALIDATION.json',dict(passed=True,tests=result.testsRun))
    baseline=Path('/home/prmi/zwj/experiments/dynamic_inference_fusion_20261106/source/python')
    unchanged={}
    for p in baseline.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.cu','.html']:
            relative=p.relative_to(baseline);actual=ROOT/'source/python'/relative
            assert sha(actual)==sha(p),relative
            unchanged[str(relative)]=sha(actual)
    save(reports/'NORMAL_SOURCE_UNCHANGED.json',dict(passed=True,files=unchanged))
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS']:
        before=json.loads((reports/(prefix+'_BEFORE.json')).read_text());after={}
        for name,old in before.items():
            p=Path(name);after[name]=dict(bytes=p.stat().st_size,sha256=sha(p)) if isinstance(old,dict) else sha(p)
        assert before==after,prefix
        save(reports/(prefix+'_AFTER.json'),after)
    # No normal module may automatically import the deployment implementation.
    import ast
    imports=[]
    for name in unchanged:
        p=ROOT/'source/python'/name
        if p.suffix!='.py':continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            modules=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                     [item.name for item in node.names] if isinstance(node,ast.Import) else [])
            imports.extend((name,m) for m in modules if m.startswith('deployment_'))
    assert not imports
    save(reports/'DEPLOYMENT_ISOLATION.json',dict(passed=True,normal_files=len(unchanged),normal_imports_deployment=[],
        training_changes=False,normal_inference_changes=False,independent_entry=True))
    subprocess.check_call([sys.executable,str(HERE/'report.py')])
    inputs=set(json.loads((reports/'INPUTS_AFTER.json').read_text()))|set(json.loads((reports/'LEGACY_INPUTS_AFTER.json').read_text()))
    assert len(inputs)==70
    images=[str(p) for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    assert not images
    identity=json.loads((reports/'BINARY_IDENTITY.json').read_text())
    assert all(sha(ROOT/'source/python'/n)==h for n,h in identity['sources'].items())
    source=[p for p in (ROOT/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cu','.cpp','.md','.html','.json']]
    save(reports/'SOURCE_FINAL_HASHES.json',{str(p.relative_to(ROOT/'source')):sha(p) for p in source})
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as archive:
        for p in sorted(source):archive.add(p,arcname=str(p.relative_to(ROOT/'source')),recursive=False)
    verdict=dict(passed=True,protected_production_files=604,changed_production_files=[],protected_checkpoint_files=70,
        normal_source_files_unchanged=len(unchanged),quality_cases=12672,quality_frames=792,perceptual_cases=396,
        api_cases=330,cli_frames=6,normal_unit_tests=24,training_runs=0,render_images_saved=0,
        production_deployed=False,production_defaults_changed=False,commit=False,push=False,
        source_archive_sha256=sha(ROOT/'source_final.tar.gz'),completed_unix=time.time(),free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    save(reports/'FINAL_INTEGRITY.json',verdict)
    save(reports/'COMPLETION.json',dict(state='complete',passed=True,report='DEPLOYMENT_RUNTIME_OPTIMIZATION_REPORT.md'))
    destination=ROOT/'DEPLOYMENT_RUNTIME_RESULTS.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in ['reports','logs','evaluation']:
            for p in (ROOT/directory).rglob('*'):
                if p.is_file() and p.name!='ARCHIVE_SHA256.json' and p.suffix not in ['.sqlite','.nsys-rep']:
                    z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz')
    save(reports/'ARCHIVE_SHA256.json',dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size))
    print('FINAL_COMPLETE',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
