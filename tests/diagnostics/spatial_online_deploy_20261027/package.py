"""只同步本地最新版一方源码/配置，保留服务器数据、环境和历史输出。"""
import hashlib,json,pathlib,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[3];STAGE=pathlib.Path(__file__).resolve().parent
files=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
files.update(['improvements/selector/spatial_compensation.h','include/spatial_compensation_cuda.h',
    'src/spatial_compensation_cuda.cpp','src/spatial_compensation_cuda.cu','python/spatial_compensation.py',
    'src/spatial_compensation_live.cu',
    'tests/anchor_selector/spatial_compensation_test.cpp','tests/anchor_selector/spatial_mapping_test.cpp',
    'tests/anchor_selector/spatial_live_benchmark.cpp'])
suffixes={'.cpp','.h','.hpp','.cu','.cuh','.py','.sh','.yaml','.yml','.cmake','.json'}
files=sorted(p for p in files if not p.startswith('third_party/') and (pathlib.Path(p).suffix in suffixes or pathlib.Path(p).name=='CMakeLists.txt')
    and not any(x in pathlib.Path(p).parts for x in ['evidence','source_evidence','code_versions','dependencies','results']))
manifest=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_production_update=True,files={})
with zipfile.ZipFile(STAGE/'payload.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files:
        value=(ROOT/p).read_bytes();manifest['files'][p]=hashlib.sha256(value).hexdigest();z.writestr(p,value)
(STAGE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('PACKAGED',len(files),(STAGE/'payload.zip').stat().st_size)
