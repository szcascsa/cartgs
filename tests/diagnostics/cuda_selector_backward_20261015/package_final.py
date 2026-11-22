"""完成全部 GPU 测量后再压缩证据，避免 CPU 压缩干扰 host wall。"""
from pathlib import Path
import json
import tarfile

D=Path(__file__).resolve().parent
assert (D/'final.exit').read_text().strip()=='0'
assert (D/'paired.exit').read_text().strip()=='0'
s=json.loads((D/'candidate_6_all_final/summary.json').read_text())
assert s['complete'] and not s['failures'] and all(v['all_exact'] for v in s['categories'].values())
names=['candidate_6_all_final','final_measure','caller_1_final','caller_6_final',
       'profile_final_1','profile_final_6','profile_final_row','boundary_final.csv','boundary_final.log',
       'boundary_before_inclusive16.csv','boundary6_vs_baseline1.csv','native_tests.log','final_guard.csv',
       'final_guard.log','final_direct.log','final_build.log','native_build.log','guard_build.log',
       'final_configure.log','final_pipeline.log','final.exit','final_summary.json','performance_tables.md',
       'paired.csv','paired.log','paired_fb.nsys-rep','paired_fb.sqlite','paired_profile.csv',
       'paired_profile_summary.json','paired_build.log','paired_nsys.log','paired.exit']
with tarfile.open(D/'final_evidence.tar.gz','w:gz') as t:
    for name in names:t.add(D/name,arcname=name)
print('final evidence bytes',(D/'final_evidence.tar.gz').stat().st_size)
