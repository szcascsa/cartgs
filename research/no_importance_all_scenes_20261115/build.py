"""使用完整方案的已验证编译命令，独立重建全部Mapper C++对象。"""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import time
from prepare import R,B,read,write,sha


def execute(name,cmd):
    assert str(B) not in ' '.join(cmd),'构建命令仍包含历史批次写入路径'
    output=Path(cmd[cmd.index('-o')+1]);assert R in output.parents
    start=time.monotonic()
    with (R/'logs'/(name+'.log')).open('w') as log:
        p=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    write(R/'reports'/(name+'.json'),dict(command=cmd,returncode=p.returncode,seconds=time.monotonic()-start))
    assert p.returncode==0,name+'失败，见对应编译日志'
    print(name,round(time.monotonic()-start,2),flush=True)


def command(name):return [s.replace(str(B),str(R)) for s in read(B/'reports'/(name+'.json'))['command']]


def main():
    assert read(R/'PREPARED.json')['passed']
    link=command('link_mapper');tasks=[]
    flags=command('compile_gaussian_mapper.o')
    for obj in [Path(s) for s in link if s.endswith('.o')]:
        name=obj.name
        source=R/'source'/('third_party/tinyply/tinyply.cpp' if name=='tinyply.cpp.o' else 'src/'+name[:-2]+('' if name.endswith('.cpp.o') else '.cpp'))
        cmd=list(flags);cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(source);tasks.append((name,cmd))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for f in concurrent.futures.as_completed([pool.submit(execute,'compile_'+n,c) for n,c in tasks]):f.result()
    execute('link_mapper',link)
    for name in ['lod_from_full','lod_health_guard_test','joint_view_conditioning_test','joint_view_cuda_parity_test']:
        execute('compile_'+name,command('compile_'+name));execute('link_'+name,command('link_'+name))
    hashes={str(p.relative_to(R)):sha(p) for folder in ['objects','runtime'] for p in (R/folder).rglob('*') if p.is_file()}
    write(R/'reports/BUILD_HASHES.json',hashes);write(R/'BUILD_READY.json',dict(passed=True,objects=len(tasks),files=len(hashes)))


if __name__=='__main__':main()
