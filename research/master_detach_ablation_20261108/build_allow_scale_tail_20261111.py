"""独立编译Scale长尾放行开关；保留旧二进制与构建清单。"""
from pathlib import Path
import json
import shutil
import subprocess
import time
import run_batch as b
import supervisor

R=b.R;p=b.p
B=R.parent/'baseline/dual_view_li_best_full_20261107'


def execute(name,command):
    output=Path(command[command.index('-o')+1]);output.resolve().relative_to(R.resolve())
    assert not output.is_symlink()
    started=time.monotonic()
    with (R/'logs'/('allow_scale_tail_'+name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=R,env=b.environment(),stdout=log,stderr=subprocess.STDOUT)
    p.write(R/'reports'/('allow_scale_tail_'+name+'.json'),dict(command=command,
        returncode=result.returncode,seconds=time.monotonic()-started))
    print(name,result.returncode,round(time.monotonic()-started,2),flush=True)
    assert result.returncode==0,name


def from_baseline(name):
    command=p.read(B/'reports'/name)['command']
    for root in [str(B),str(R.parent/'saved_experiments/dual_view_li_best_full_20261107')]:
        command=[arg.replace(root,str(R)) for arg in command]
    return command


def main():
    assert p.read(R/'LOD_PIPELINE.json')['phase']=='finished'
    assert not (R/'HOLD').exists()
    for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']:assert not supervisor.alive(p.read(R/name))
    assert shutil.disk_usage(R).free>5*2**30
    backup=R/'reports/allow_scale_tail_build_backup_20261111';backup.mkdir()
    for name in ['reports/BUILD_HASHES.json','reports/READ_ONLY_BUILD_INPUTS.json',
                 'reports/link_mapper.json','runtime/lib/libgaussian_mapper.so',
                 'runtime/bin/lod_health_guard_test','objects/gaussian_mapper.o']:
        file=R/name;shutil.copy2(file,backup/file.name)
    execute('compile_gaussian_mapper',p.read(R/'reports/compile_gaussian_mapper.o.json')['command'])
    execute('compile_view_budget_training',from_baseline('compile_view_budget_training.cpp.o.json'))
    execute('compile_health_test',from_baseline('compile_lod_health_guard_test.json'))
    execute('link_health_test',from_baseline('link_lod_health_guard_test.json'))
    link=p.read(R/'reports/link_mapper.json')['command']
    link=[str(R/'objects/view_budget_training.cpp.o') if arg.endswith('/objects/view_budget_training.cpp.o')
          else arg for arg in link]
    execute('link_mapper',link)
    with (R/'logs/allow_scale_tail_health_test.log').open('w') as log:
        result=subprocess.run([str(R/'runtime/bin/lod_health_guard_test')],env=b.environment(),
                              stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'allow_scale_tail_health_test'
    p.write(R/'reports/link_mapper.json',p.read(R/'reports/allow_scale_tail_link_mapper.json'))
    reused={arg:p.sha(Path(arg)) for arg in link if arg.endswith(('.o','.a'))
            and not Path(arg).is_relative_to(R)}
    p.write(R/'reports/READ_ONLY_BUILD_INPUTS.json',reused)
    p.write(R/'reports/BUILD_HASHES.json',{str(f.relative_to(R)):p.sha(f)
        for folder in ['runtime','objects'] for f in (R/folder).rglob('*') if f.is_file()})
    p.write(R/'reports/ALLOW_SCALE_TAIL_BUILD.json',dict(passed=True,backup=str(backup),
        configuration='LoDHealth.scale_factor_limit=0 disables stopping, diagnostic limit remains 100',
        default_unchanged=True,master_detach_unchanged=True,health_test_returncode=0,
        finished_unix=time.time(),source_hashes={name:p.sha(R/'source'/name) for name in
        ['include/lod_health_guard.h','src/gaussian_mapper.cpp','tests/lod_health_guard_test.cpp']}))
    print('ALLOW_SCALE_TAIL_BUILD_AND_HEALTH_TEST_PASS',flush=True)


if __name__=='__main__':main()
