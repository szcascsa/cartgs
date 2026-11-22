"""验证本地/服务器源码、默认关闭状态，封装可复核的小体积证据。"""
from pathlib import Path
import hashlib,json,platform,tarfile
import cv2,torch
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=json.loads((ROOT/'source_expected.json').read_text())
for name,digest in expected.items():
    assert sha(ROOT/'source'/name)==digest,name
assert json.loads((ROOT/'status.json').read_text())['all_passed']
assert json.loads((ROOT/'final_status.json').read_text())['all_passed']
result=json.loads((ROOT/'sparse_root_results.json').read_text())
assert result['legacy_reference_bitwise_equal'] and all(result['gates'].values())
configs={}
for name in ['cfg/selector/fast.yaml','cfg/selector/selector_config.yaml']:
    fs=cv2.FileStorage(str(ROOT/'source'/name),cv2.FILE_STORAGE_READ)
    assert fs.isOpened(),name
    values={key:int(fs.getNode('GlobalGI.HierarchicalRepresentative.'+key).real())for key in ['enable','diagnostics']}
    fs.release();assert values=={'enable':0,'diagnostics':0},(name,values)
    configs[name]=values
parity=json.loads((ROOT/'mathematical_parity.json').read_text())
text=(ROOT/'source/src/hierarchical_representative_si.cpp').read_text()
blocks={
    'geometry':text[text.index('torch::Tensor geometry('):text.index('double quantile(')],
    'representative_greedy':text[text.index('CellSelection selectCell('):text.index('Result build(')],
    'redistribution':text[text.index('    auto local_output ='):text.index('  int64_t root_ordinal =')],
}
# Expected block digests were checked against source_before before the first build.
for key,value in blocks.items():
    assert hashlib.sha256(value.encode()).hexdigest()==parity['unchanged_blocks_sha256'][key],key
metadata={'all_passed':True,'source_sha256':expected,'default_off_opencv':configs,
          'unchanged_math_sha256':parity['unchanged_blocks_sha256'],
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                         'GPU':torch.cuda.get_device_name(0),'platform':platform.platform()},
          'scope':{'training':False,'online_SLAM':False,'quality_evaluation':False,'production_deployment':False,
                   'full_SLAM_binary_link':False,'commit':False,'push':False}}
(ROOT/'final_verification.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
files=[]
for pattern in ['*.json','*.log','*.py','CMakeLists.txt','*.csv','*_output.npz','fr1_room_input.bin','fr1_room_inputs.npz','fixture/*','old_reference/*']:
    files.extend(p for p in ROOT.glob(pattern)if p.is_file())
files.extend(ROOT/'source'/name for name in expected)
files=sorted(set(files)-{ROOT/'artifact_manifest.json'})
manifest={str(p.relative_to(ROOT)):sha(p)for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(ROOT/'sparse_root_evidence.tar.gz','w:gz')as archive:
    for p in files+[ROOT/'artifact_manifest.json']:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps({'all_passed':True,'files':len(files),'archive_bytes':(ROOT/'sparse_root_evidence.tar.gz').stat().st_size,
                  'archive_sha256':sha(ROOT/'sparse_root_evidence.tar.gz')},indent=2),flush=True)
