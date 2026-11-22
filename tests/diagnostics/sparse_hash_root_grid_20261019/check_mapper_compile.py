"""使用现有 production 编译参数，对隔离 snapshot 的 mapper TU 作编译检查。"""
from pathlib import Path
import json,shlex,subprocess
ROOT=Path(__file__).resolve().parent
baseline=Path('/home/prmi/zwj/mycartgs/build-cuda118-sm86-gcc9')
commands=subprocess.check_output(['/home/prmi/.local/bin/ninja','-C',str(baseline),'-t','commands','gaussian_mapper'],text=True)
line=next(x for x in commands.splitlines() if '/src/gaussian_mapper.cpp' in x and ' -c ' in x)
original=shlex.split(line);args=[original[0],'-I'+str(ROOT/'source'),'-I'+str(ROOT/'source/include')]
skip=False
for value in original[1:]:
    if skip:skip=False;continue
    if value in ['-o','-MF','-MT','-MQ']:skip=True;continue
    if value in ['-c','-MD','-MMD','-MP']:continue
    if value.endswith('/src/gaussian_mapper.cpp'):value=str(ROOT/'source/src/gaussian_mapper.cpp')
    args.append(value)
args.extend(['-O0','-fsyntax-only'])
with open(ROOT/'mapper_compile.log','w')as log:subprocess.run(args,cwd=baseline,check=True,stdout=log,stderr=subprocess.STDOUT)
(ROOT/'mapper_compile_check.json').write_text(json.dumps({'all_passed':True,'validation':'production mapper translation-unit compile','link_full_SLAM_binary':False,'command':args},indent=2)+'\n')
print('MAPPER_COMPILE_PASS',flush=True)
