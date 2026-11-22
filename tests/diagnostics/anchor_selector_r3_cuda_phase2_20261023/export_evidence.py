"""Allowlist证据导出；不清理模型/原始Tensor，也不覆盖上一轮目录。"""
from pathlib import Path
import hashlib,json,tarfile
B=Path('/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023');R=B/'final_legacy_reduction'
manifest=json.loads((B/'final_source_manifest.json').read_text());verification=[]
for name,expected in manifest['files'].items():
    path=B/'source'/name;actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'MISSING';verification.append(dict(file=name,expected=expected,actual=actual,matched=actual==expected))
(R/'source_verification.json').write_text(json.dumps(verification,indent=2));assert all(x['matched'] for x in verification),'Source differs from local delivery'
selected=[]
for path in R.rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(R)
    if path.suffix in ['.csv','.json','.log','.yaml'] and not any(x in rel.parts for x in ['point_cloud','point_cloud_before','images','render','renders']):selected.append((path,Path('results')/rel))
# Full native Forward/VJP/Adam fixtures at every trained ratio, plus continuous
# and topology snapshots; all other raw files stay on server unchanged.
for folder in ['bench_1.0_0_B','bench_1.0_0_E']:
    for path in (R/folder).glob('*_parity.pt'):selected.append((path,Path('results')/path.relative_to(R)))
for folder in ['continuous_B_a','continuous_B_b','continuous_E_a','continuous_E_b']:
    for pattern in ['0_*.pt','49_*.pt']:
        for path in (R/folder).glob(pattern):selected.append((path,Path('results')/path.relative_to(R)))
for path in (R/'topology_checkpoint').glob('*.pt'):selected.append((path,Path('results')/path.relative_to(R)))
for name in ['final_source_manifest.json','final_driver.log','build_driver.log','compile.log','configure.log','ncu_ssim_metrics.log','phase6_stress_initial_failure.log','phase6_stress_initial_parity.csv','phase6_stress_atomic_failure.log','phase6_stress_atomic_parity.csv','legacy_reduction_gate.log','deterministic_threshold_driver.log']:
    path=B/name
    if path.exists():selected.append((path,Path('build_and_failed_gates')/name))
for folder in ['phase8','deterministic_threshold_evidence']:
    for path in (B/folder).glob('*.csv'):selected.append((path,Path('build_and_failed_gates')/folder/path.name))
for target in ['cuda_rasterizer','transform_field_cuda','gaussian_mapper','phase2_staged_mapping_benchmark']:
    for file in ['flags.make','link.txt']:
        path=B/'build/CMakeFiles'/(target+'.dir')/file
        if path.exists():selected.append((path,Path('build_flags')/target/file))
records=[]
with tarfile.open(B/'phase2_final_evidence.tar.gz','w:gz',compresslevel=1) as archive:
    for path,relative in selected:
        archive.add(path,arcname=relative.as_posix(),recursive=False);records.append(dict(path=relative.as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(B/'evidence_archive_manifest.json').write_text(json.dumps(records,indent=2));print('EXPORT_PASS',len(records),'files', (B/'phase2_final_evidence.tar.gz').stat().st_size,'archive bytes',flush=True)
