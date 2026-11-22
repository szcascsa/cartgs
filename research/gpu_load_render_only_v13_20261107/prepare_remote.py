"""建立独立v13运行树并核查已有输入。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess


ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/gpu_load_adaptive_v11_20261107')
V13=Path('/home/prmi/zwj/experiments/deployment_network_gemm_20261107')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (ROOT/'STATUS.json').exists()
    previous=json.loads((OLD/'reports/PROTECTED_BEFORE.json').read_text())
    assert all(sha(name)==value for name,value in previous.items())
    for name in ('python','original_raster','evaluation'):
        shutil.copytree(OLD/'source'/name,ROOT/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(OLD/'source/original_renderer.py',ROOT/'source/original_renderer.py')
    shutil.copytree(V13/'source/python',ROOT/'source/python',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__'))
    for name,source in [('mycartgs_deployment_fp32_v13',V13),('mycartgs_frozen_inference_v3',OLD)]:
        target=ROOT/'build'/name
        target.mkdir(parents=True)
        shutil.copy2(source/'build'/name/(name+'.so'),target/(name+'.so'))
    library=ROOT/'build/mycartgs_deployment_fp32_v13/mycartgs_deployment_fp32_v13.so'
    assert sha(library)=='eb0195413f9b44a79fe93f4f1e313437469a6c367adf289b69006e190339e15e'
    identity=json.loads((V13/'reports/BINARY_IDENTITY.json').read_text())
    assert all(sha(ROOT/'source'/name)==value for name,value in identity['source_sha256'].items())
    (ROOT/'reports/V13_BINARY_IDENTITY.json').write_text(json.dumps(identity,indent=2))
    shutil.copy2(OLD/'runs/formal_02/QUALITY.csv',ROOT/'reports/V11_QUALITY.csv')
    study=Path(__file__).resolve().parent
    command=['/usr/local/cuda-11.8/bin/nvcc','-O3','-std=c++17','-arch=sm_86','--shared',
        '-Xcompiler','-fPIC','-Xcompiler','-pthread',str(study/'render_load.cu'),
        '-o',str(ROOT/'build/librender_load.so'),'-Xlinker','-rpath','-Xlinker','/usr/local/cuda-11.8/lib64']
    build=subprocess.run(command,capture_output=True,text=True)
    (ROOT/'logs/build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    test=subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python','-B','-m','unittest','discover','-v'],
        cwd=study,capture_output=True,text=True)
    (ROOT/'logs/cpu_tests.log').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stderr
    files=[p for name in ('source','build') for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected=dict(previous)
    protected.update({str(p):sha(p) for p in files})
    (ROOT/'reports/PROTECTED_BEFORE.json').write_text(json.dumps(protected,indent=2))
    (ROOT/'reports/BUILD_VALIDATION.json').write_text(json.dumps(dict(passed=True,tests=11,
        command=command,load_binary_sha256=sha(ROOT/'build/librender_load.so'),
        protected_files=len(protected)),indent=2))
    print('PREPARE_V13_PASS',len(protected))


if __name__=='__main__':
    main()
