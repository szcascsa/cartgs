"""隔离构建 Anchor SI；不覆盖服务器生产源码、模型或二进制。"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile,time
root=Path('/home/prmi/zwj/anchor-si-20261024')
old=Path('/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023')
status={'started':time.time()}
try:
    if not (root/'source').exists():
        shutil.copytree(old/'source',root/'source',symlinks=True)
        shutil.copytree(old/'build',root/'build',symlinks=True)
        for p in (root/'build').rglob('*'):
            if not p.is_file() or p.is_symlink() or p.suffix not in {'.txt','.cmake','.make','.ninja','.internal','.ts','.d'}:continue
            try:text=p.read_text()
            except UnicodeDecodeError:continue
            if str(old) in text:
                stat=p.stat();p.write_text(text.replace(str(old),str(root)));os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    manifest=json.loads((root/'source_manifest.json').read_text())
    with zipfile.ZipFile(root/'source.zip') as archive:
        for name,want in manifest['files'].items():
            data=archive.read(name);assert hashlib.sha256(data).hexdigest()==want
            p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists() or p.read_bytes()!=data:p.write_bytes(data)
    commands=[['/home/prmi/.local/bin/cmake','-S',str(root/'source'),'-B',str(root/'build')],
              ['/home/prmi/.local/bin/cmake','--build',str(root/'build'),'-j2','--target','anchor_si_test','anchor_si_frozen_train','anchor_si_microbenchmark','tum_rgbd']]
    for name,command in zip(['configure','compile'],commands):
        status['phase']=name;(root/'build_status.json').write_text(json.dumps(status))
        with (root/(name+'.log')).open('w') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
    status.update(phase='complete',finished=time.time(),passed=True)
except Exception as error:
    status.update(phase='failed',error=str(error));raise
finally:(root/'build_status.json').write_text(json.dumps(status,indent=2))
