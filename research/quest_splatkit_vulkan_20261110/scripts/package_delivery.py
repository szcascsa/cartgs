from pathlib import Path
import os
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
files=[]
for folder in ['host','source/splatkit','deps','cmake','tests','scripts','report','evidence']:
    for base,dirs,names in os.walk(root/folder):
        dirs[:]=[d for d in dirs if d not in {'.git','.gradle','.cxx','build','.gradle-user','.idea','__pycache__','media'}]
        for name in names:
            path=Path(base)/name
            if path.suffix.lower() in {'.mcgs','.ply','.spz','.apk','.mp4','.gif','.jpg','.png','.bin','.so','.a','.zip'}:
                if not (folder=='report' and path.suffix=='.png'):continue
            if 'shaders' in path.parts and path.suffix=='.spv':continue
            files.append((path,path.relative_to(root).as_posix()))
files.append((root/'README_CN.md','README_CN.md'))
archive=root/'delivery/mycartgs-splatkit-baseline-source.zip'
entries=[]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path,name in sorted(files,key=lambda value:value[1]):
        z.write(path,name)
        entries.append(dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for entry in entries:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
(root/'delivery/SOURCE_CONTENTS.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
outputs=[archive,root/'delivery/mycartgs-splatkit-baseline.apk',root/'report/SPLATKIT_VULKAN_BASELINE_CN.md',
    root/'report/SPLATKIT_POINT_COUNT.png',root/'evidence/POINT_COUNT_SUMMARY.csv',root/'evidence/POINT_COUNT_SAMPLES.csv',root/'evidence/FINAL_VALIDATION.json',
    root/'report/LEFT_EYE_SHARED_CPU_BENCHMARK_CN.md',root/'evidence/left_eye_shared/VALIDATION.json',
    root/'evidence/left_eye_shared/CPU_BENCHMARK_SUMMARY.csv',root/'evidence/left_eye_shared/CPU_APK_BENCHMARK_SUMMARY.csv']
outputs += [root/'report/QUEST_CPU_OPTIMIZATION_CN.md',root/'report/QUEST_CPU_LATENCY.png',
    root/'evidence/cpu_optimization/VALIDATION.json',root/'evidence/cpu_optimization/APK_CPU_SUMMARY.csv',
    root/'evidence/cpu_optimization/PAIRED_RESULTS.csv',root/'evidence/cpu_optimization/AGGREGATE_RESULTS.csv',
    root/'evidence/cpu_optimization/CPU_OPTIMIZATION.patch',root/'evidence/cpu_optimization/baseline/mycartgs-splatkit-baseline.apk',
    root/'evidence/cpu_optimization/baseline/source_0.2.2.zip']
manifest={path.relative_to(root).as_posix():dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in outputs}
(root/'delivery/DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',source_files=len(entries),source_zip_bytes=archive.stat().st_size),indent=2))
