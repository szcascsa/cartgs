"""源码SHA、实际构建目标、生产配置及两次在线完成状态的最终审计。"""
import hashlib,json,pathlib,time
ROOT=pathlib.Path('/home/prmi/zwj/mycartgs');STAGE=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((STAGE/'final_source_manifest.json').read_text());mismatches=[]
for name,digest in m['files'].items():
    p=ROOT/name
    if not p.is_file() or sha(p)!=digest:mismatches.append(name)
targets=['spatial_compensation_test','spatial_mapping_test','tum_rgbd','tum_mono','replica_rgbd','replica_mono','vector_mono','vector_stereo','view_result','train_colmap']
log=(STAGE/'build_live_default.log').read_text()
assert all(('Built target '+name) in log for name in targets)
assert not mismatches,mismatches
assert 'Selector.compensation_method: "spatial_localxyz"' in (ROOT/'cfg/selector/selector_config.yaml').read_text()
for name in ['online_spatial','online_live_fused']:assert json.loads((STAGE/name/'completion.json').read_text())['returncode']==0
for path,marker in [('live_adam_unit.log','SPATIAL_PARITY_PASS'),('live_mapping_gate.log','PASS: compensation photo gradient')]:assert marker in (STAGE/path).read_text()
result=dict(phase='complete',pass_=True,finished=time.time(),source_files_verified=len(m['files']),base_commit=m['base_commit'],source_is_uncommitted_working_tree=True,
    production_default='spatial_localxyz',production_cuda='live K-row input+Fourier+FiLM and existing decode14; Master gradients preserved',
    targets=targets,binary_sha256={name:sha(ROOT/'bin'/name) for name in targets},library_sha256={p.name:sha(p) for p in (ROOT/'lib').glob('libgaussian_*.so')},
    backup=str(STAGE/'backup'),online_runs=['online_spatial','online_live_fused'],commit=False,stage=False,push=False)
(STAGE/'status.json').write_text(json.dumps(result,indent=2));(STAGE/'deployment_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
