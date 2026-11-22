"""验证本轮相对未提交 hierarchy 基线的边界并保存单独 diff。"""
from pathlib import Path
import hashlib,json,difflib
TASK=Path(__file__).resolve().parent
REPO=TASK.parent/'mycartgs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['CMakeLists.txt','cfg/selector/fast.yaml','cfg/selector/selector_config.yaml',
       'include/hierarchical_representative_si.h','include/sparse_hash_root_grid.h',
       'src/hierarchical_representative_si.cpp','src/sparse_hash_root_grid.cpp',
       'tests/hierarchical_representative_si_test.cpp','tests/sparse_hash_root_grid_test.cpp',
       'include/gaussian_mapper.h','src/gaussian_mapper.cpp','include/global_gi_bank.h',
       'src/global_gi.cpp','src/gi_teacher.cpp','include/selector_handoff.h',
       'improvements/selector/gumbel_network.h','cuda_rasterizer/forward.cu',
       'tests/global_gi_test.cpp','include/global_gi.h','include/gi_teacher.h',
       'include/elastic_entry.h','include/model_snapshot.h','include/selector_cuda.h']
(TASK/'source_expected.json').write_text(json.dumps({n:sha(REPO/n)for n in files},indent=2)+'\n')
changed=['include/hierarchical_representative_si.h','src/hierarchical_representative_si.cpp','tests/hierarchical_representative_si_test.cpp']
before=json.loads((TASK/'audit_before.json').read_text())
unchanged={k:sha(REPO/k)==v for k,v in before['source_before'].items() if k not in changed}
assert all(unchanged.values()),unchanged
old=(TASK/'source_before/src/hierarchical_representative_si.cpp').read_text()
new=(REPO/'src/hierarchical_representative_si.cpp').read_text()
markers={'geometry':('torch::Tensor geometry(','double quantile('),
         'representative_greedy':('CellSelection selectCell(','Result build('),
         'redistribution':('    auto local_output =','  for (const auto& item : roots)')}
results={}
for key,(lo,hi) in markers.items():
    oldblock=old[old.index(lo):old.index(hi)]
    newblock=new[new.index(lo):new.index('  int64_t root_ordinal ='if key=='redistribution'else hi)]
    assert oldblock==newblock,key
    results[key]=hashlib.sha256(newblock.encode()).hexdigest()
assert results==json.loads((TASK/'mathematical_parity.json').read_text())['unchanged_blocks_sha256']
diff=[]
for key in changed:
    diff.extend(difflib.unified_diff((TASK/'source_before'/key).read_text().splitlines(True),
        (REPO/key).read_text().splitlines(True),fromfile='before/'+key,tofile='after/'+key))
for key in ['include/sparse_hash_root_grid.h','src/sparse_hash_root_grid.cpp','tests/sparse_hash_root_grid_test.cpp']:
    diff.extend(difflib.unified_diff([],(REPO/key).read_text().splitlines(True),fromfile='/dev/null',tofile='after/'+key))
(TASK/'sparse_root_partition.diff').write_text(''.join(diff),encoding='utf-8')
result={'all_passed':True,'untouched_modules':unchanged,'unchanged_math':results}
(TASK/'local_unchanged_modules.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
