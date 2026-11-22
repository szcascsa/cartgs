"""核验本轮冻结文件和输入，不重写模型、生产或共享环境。"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import argparse

HERE=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;assert not output.exists()
    rows=[]
    manifests=[HERE.parent/'reports/INPUT_SHA256.json',HERE.parent/'reports/PROTECTED_SHA256.json',HERE/'reports/PROTECTED_SHA256.json']
    for path in manifests:
        data=json.loads(path.read_text());bad=[name for name,value in data.items() if sha(name)!=value]
        rows.append(dict(manifest=str(path),files=len(data),mismatches=bad));assert not bad,(path,bad)
    models=json.loads((HERE/'reports/EXISTING_STUDENTS.json').read_text())
    for name,model in models.items():assert sha(model['student'])==model['student_sha256'],name
    freezes=[]
    for filename in ['FROZEN_COHERENT.json','FROZEN_RENDER_FINAL.json','FROZEN_STUDENT.json']:
        data=json.loads((HERE/'reports'/filename).read_text());bad=[]
        for name,value in data.get('sources',{}).items():
            path=Path(name) if Path(name).is_absolute() else HERE/name
            if sha(path)!=value:bad.append(str(path))
        if 'binary_sha256' in data:
            name=data.get('binary','build_coherent/mycartgs_student_coherent_20261114_v1.so' if 'COHERENT' in filename else '')
            if name:assert sha(HERE/name)==data['binary_sha256']
        assert not bad,(filename,bad);freezes.append(dict(manifest=filename,sources=len(data.get('sources',{})),passed=True))
    images=[]
    for p in HERE.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.webp','.bmp'] and 'figures' not in p.relative_to(HERE).parts:images.append(str(p))
    assert not images,images
    result=dict(passed=True,time=time.time(),manifests=rows,students=len(models),frozen=freezes,
        render_images=images,free_bytes=shutil.disk_usage(HERE).free,no_training=True,no_teacher_benchmark=True,
        no_commit_push_deploy=True,scope='所列输入、生产源码、原研究、共享conda-meta与Student文件哈希逐个验证')
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False));print('INTEGRITY_PASS',sum(x['files'] for x in rows),len(models))


if __name__=='__main__':main()
