from common import *
from deployment_runtime import DeploymentRenderer
import ast
import csv
import shutil
import subprocess
import tarfile
import unittest
import zipfile


def main():
    reports=ROOT/'reports'
    assert 'VALIDATION_BENCHMARK_COMPLETE' in (ROOT/'logs/runner.log').read_text()
    for name in ['DEVICE_SELECTION','VALIDATION','LARGE_PIPELINE','EXTRA_VALIDATION','TRAJECTORY_VALIDATION','CONTROL_AUDIT']:
        assert json.loads((reports/(name+'.json')).read_text())['passed'],name
    assert 'SANITIZER_CASES_PASS' in (ROOT/'logs/final_diagnostics.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in (ROOT/'logs/final_diagnostics.log').read_text()
    assert not json.loads((reports/'LOOP_DECISION.json').read_text())['kept']
    for j in [1,2]:
        rows=json.loads((reports/f'BENCHMARK_{j}.json').read_text())['results'];assert len(rows)==5
        for row in rows:
            assert row['summary']['device_counts']['prepare']['median_ms']<row['summary']['previous_cpp']['prepare']['median_ms']
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')])
    output=ROOT/'evaluation/CLI.csv'
    if not output.exists():
        subprocess.check_call([sys.executable,str(ROOT/'source/python/deployment_infer.py'),str(CHECKPOINTS['both_li']),
            '--dataset',DATASET,'--frames','6','--budget','2000','--output',str(output)],env=env)
    with output.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==6 and all(int(v['K'])==2000 for v in rows)
    save(reports/'CLI_VALIDATION.json',dict(passed=True,frames=6,images_saved=0))
    changed={'deployment_runtime.py','deployment_runtime_bindings.cpp','deployment_runtime_kernels.cu',
        'deployment_prepare.cpp','deployment_selection.cu','DEPLOYMENT_RUNTIME.md'}
    untouched={};reference=ROOT/'reference/python'
    for p in reference.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ['.py','.cu','.cpp','.h','.md','.html']:continue
        name=str(p.relative_to(reference))
        if name in changed:continue
        assert sha(ROOT/'source/python'/name)==sha(p),name
        untouched[name]=sha(p)
    save(reports/'UNCHANGED_SOURCES.json',dict(passed=True,files=untouched,
        changed_deployment_only=sorted(changed),new_deployment_file='deployment_device_control.h'))
    for name in untouched:
        if not name.endswith('.py') or name.startswith('deployment_'):continue
        for node in ast.walk(ast.parse((ROOT/'source/python'/name).read_text(encoding='utf-8-sig'))):
            modules=[node.module or ''] if isinstance(node,ast.ImportFrom) else [v.name for v in node.names] if isinstance(node,ast.Import) else []
            assert not any(m.startswith('deployment_') for m in modules),name
    dest=ROOT/'source/tests';dest.mkdir(exist_ok=True)
    origin=Path('/home/prmi/zwj/experiments/deployment_point_runtime_20261106/source/tests')
    for name in ['anchor_budget_test.py','anchor_budget_cli_test.py','budget_viewer_test.py','anchor_lod_test.py']:shutil.copy2(origin/name,dest/name)
    sys.path.insert(0,str(dest));suite=unittest.defaultTestLoader.loadTestsFromNames(['anchor_budget_test','anchor_budget_cli_test','budget_viewer_test'])
    result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful() and result.testsRun==24
    save(reports/'NORMAL_TESTS.json',dict(passed=True,tests=24,no_automatic_deployment_imports=True))
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS']:
        before=json.loads((reports/(prefix+'_BEFORE.json')).read_text());after={}
        for name,info in before.items():
            p=Path(name);after[name]=dict(bytes=p.stat().st_size,sha256=sha(p)) if isinstance(info,dict) else sha(p)
        assert after==before,prefix;save(reports/(prefix+'_AFTER.json'),after)
    s=scene();runtime=DeploymentRenderer(s.model,s.gaussians,s.config)
    binary=Path(runtime._ext.__file__).resolve();assert binary==ROOT/'build/mycartgs_deployment_fp32_v7/mycartgs_deployment_fp32_v7.so'
    names=['deployment_runtime.py','deployment_runtime_bindings.cpp','deployment_runtime_kernels.cu',
        'deployment_prepare.cpp','deployment_selection.cu','deployment_device_control.h','deployment_infer.py','DEPLOYMENT_RUNTIME.md']
    save(reports/'BINARY_IDENTITY.json',dict(path=str(binary),sha256=sha(binary),sources={n:sha(ROOT/'source/python'/n) for n in names},
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),tf32=torch.backends.cuda.matmul.allow_tf32,
        build_ninja=(binary.parent/'build.ninja').read_text()))
    maps=Path('/proc/self/maps').read_text();assert str(binary) in maps;(reports/'RUNTIME_MAPS.txt').write_text(maps)
    subprocess.check_call([sys.executable,str(HERE/'report.py')])
    source=[p for p in (ROOT/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cu','.cpp','.h','.md','.json','.html']]
    save(reports/'SOURCE_FINAL_HASHES.json',{str(p.relative_to(ROOT/'source')):sha(p) for p in source})
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as a:
        for p in source:a.add(p,arcname=str(p.relative_to(ROOT/'source')),recursive=False)
    assert not [p for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    final=dict(passed=True,protected_production_files=604,changed_production_files=[],protected_checkpoint_files=70,
        unchanged_previous_source_files=len(untouched),training_runs=0,images_saved=0,production_deployed=False,
        production_default_changed=False,commit=False,push=False,device_count_map_capacity_max=16384,
        large_maps_use_previous_cpp=True,loop_candidate_kept=False,general_device_candidate_kept=False,
        source_archive_sha256=sha(ROOT/'source_final.tar.gz'),free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    save(reports/'FINAL_INTEGRITY.json',final);save(reports/'COMPLETION.json',dict(state='complete',passed=True))
    output=ROOT/'GPU_DEVICE_COUNTS_RESULTS.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['reports','logs','evaluation','reference']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.name!='ARCHIVE_SHA256.json' and p.suffix not in ['.nsys-rep','.sqlite','.so']:
                    z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz')
    save(reports/'ARCHIVE_SHA256.json',dict(path=str(output),bytes=output.stat().st_size,sha256=sha(output)))
    print('FINAL_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':main()
