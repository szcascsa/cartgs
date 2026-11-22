"""通过应用显式导出的日志提取CPU CSV；校验长度和SHA，不访问模型。"""
from pathlib import Path
import base64
import hashlib
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
e=root/'evidence/left_eye_shared'
data=subprocess.check_output(['Q:/tools/android-sdk/platform-tools/adb.exe','-s','2G97C5ZJ2G00PX',
    'logcat','-d','-v','raw','-s','SplatKitCpuCsv:I','*:S'])
output=Path(sys.argv[1]) if len(sys.argv)>1 else e/'apk_cpu_benchmarks'
output.mkdir(parents=True,exist_ok=True)
(output.parent/'APK_CPU_CSV_EXPORT.log').write_bytes(data)
header=None
count=0
for line in data.decode('utf-8',errors='replace').splitlines():
    if line.startswith('BEGIN '):
        _,name,size,digest=line.split()
        assert name.startswith('cpu_inference_') and name.endswith('.csv') and Path(name).name==name
        header=(name,int(size),digest)
        chunks=[]
    elif line.startswith('DATA ') and header:
        chunks.append(line[5:])
    elif line.startswith('END ') and header:
        name,size,digest=header
        assert line[4:]==name
        raw=base64.b64decode(''.join(chunks),validate=True)
        assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest
        (output/name).write_bytes(raw)
        print(f'VERIFIED {name} bytes={size} sha256={digest}')
        count+=1
        header=None
assert count>=2, 'Expected both completed foreground CPU reports'
