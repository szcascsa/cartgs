"""独立重编译Mapper及Replica入口，避免头文件内联实现混用。"""
from pathlib import Path
import sys
# 批次自带queue.py，加载线程池时先确保使用Python标准库queue。
_HERE=Path(__file__).resolve().parent
sys.path[:]=[x for x in sys.path if Path(x or '.').resolve()!=_HERE]
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(_HERE))
import os
import subprocess
import time
from prepare_experiment import R,F,read,write,sha

T=Path('/home/prmi/zwj/experiments/tum_full_tail_prune_20261103')
J=Path('/home/prmi/zwj/experiments/real_lod_full_joint_20261103')


def replace_roots(arg):
    for root in [F,T,J,Path('/home/prmi/zwj/experiments/multi_dataset_full_tail_prune1_seed0_20261103')]:
        for folder in ['source','runtime']:
            arg=arg.replace(str(root/folder),str(R/folder))
    if arg in ['libtransform_field_cuda.a','libselector_cuda.a']:
        arg=str(Path('/home/prmi/zwj/experiments/real_lod_20261103/build')/arg)
    return arg


def execute(name,command):
    output=Path(command[command.index('-o')+1])
    assert output.is_relative_to(R) and not output.is_symlink()
    report=R/'reports'/(name+'.json')
    if '-c' in command and report.exists() and output.exists():
        prior=read(report)
        newest=max(Path(command[-1]).stat().st_mtime,
                   max(p.stat().st_mtime for folder in ['include','improvements']
                       for p in (R/'source'/folder).rglob('*.h')))
        if prior['returncode']==0 and prior['command']==command and output.stat().st_mtime>=newest:
            print(name,'unchanged',flush=True);return
    start=time.monotonic()
    with (R/'logs'/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    write(R/'reports'/(name+'.json'),dict(command=command,returncode=result.returncode,seconds=time.monotonic()-start))
    print(name,result.returncode,round(time.monotonic()-start,2),flush=True)
    assert result.returncode==0,name


def compile_command(template,source,output):
    command=[replace_roots(x) for x in template]
    for flag in ['-MT','-MF','-o','-c']:
        if flag in command:
            pos=command.index(flag);del command[pos:pos+2]
    command=[x for x in command if x!='-MD']
    return command+['-o',str(output),'-c',str(source)]


def main():
    assert Path(__file__).resolve().parent==R and (R/'HOLD').exists()
    assert read(R/'PREPARED.json')['jobs']==16 and not (R/'BUILD_READY.json').exists()
    # 使用原Full批次的链接参数；所有含Anchor头文件的Mapper对象从同一副本重编译。
    compile_template=read(T/'compile_real_lod.json')['command']
    link_template=read(T/'link_gaussian_mapper.json')['command']
    replacements={};tasks=[]
    for old in [x for x in link_template if x.endswith('.o')]:
        basename=Path(old).name
        short=basename.removeprefix('CMakeFiles_gaussian_mapper.dir_src_')
        relative='third_party/tinyply/tinyply.cpp' if short=='tinyply.cpp.o' else 'src/'+short[:-2]+('' if short.endswith('.cpp.o') else '.cpp')
        source=R/'source'/relative;assert source.is_file(),source
        output=R/'objects'/basename
        tasks.append(('compile_'+basename,compile_command(compile_template,source,output)))
        replacements[old]=str(output)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(execute,*task) for task in tasks]
        for future in futures:future.result()
    command=[replacements.get(x,replace_roots(x)) for x in link_template]
    command[command.index('-o')+1]=str(R/'runtime/lib/libgaussian_mapper.so')
    execute('link_mapper',command)
    for entry in ['replica_mono','replica_rgbd']:
        template=read(F/('compile_'+entry+'.json'))['command']
        output=R/'objects'/(entry+'.o')
        execute('compile_'+entry,compile_command(template,R/'source/examples'/(entry+'.cpp'),output))
        command=read(F/('link_'+entry+'.json'))['command']
        command=[str(output) if x.endswith('/objects/'+entry+'.o') else replace_roots(x) for x in command]
        command[command.index('-o')+1]=str(R/'runtime/bin'/entry)
        execute('link_'+entry,command)
    # 实际growBlocks原生验收使用相同编译参数，不依赖Python镜像的规则。
    output=R/'objects/growth_threshold_test.o'
    execute('compile_growth_test',compile_command(read(F/'compile_replica_mono.json')['command'],R/'growth_threshold_test.cpp',output))
    command=read(F/'link_replica_mono.json')['command']
    command=[str(output) if x.endswith('/objects/replica_mono.o') else replace_roots(x) for x in command]
    command[command.index('-o')+1]=str(R/'runtime/bin/growth_threshold_test')
    execute('link_growth_test',command)
    env=dict(os.environ,LD_LIBRARY_PATH=str(R/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    links={}
    for entry in ['replica_mono','replica_rgbd','growth_threshold_test']:
        binary=R/'runtime/bin'/entry
        elf='/home/prmi/anaconda3/bin/patchelf'
        current=subprocess.check_output([elf,'--print-rpath',str(binary)],text=True).strip()
        subprocess.run([elf,'--set-rpath',replace_roots(current),str(binary)],check=True)
        text=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
        assert str(R/'runtime/lib/libgaussian_mapper.so') in text and 'not found' not in text
        links[entry]=text
    write(R/'BUILD_READY.json',dict(passed=True,compiled=True,ldd=links,
          binaries={str(p.relative_to(R)):sha(p) for folder in ['runtime/bin','runtime/lib']
                    for p in (R/folder).iterdir() if p.is_file()}))
    print('BUILD_COMPLETE_NATIVE_TESTS_AND_NO_TRAIN_GATE_REQUIRED',flush=True)


if __name__=='__main__':main()
