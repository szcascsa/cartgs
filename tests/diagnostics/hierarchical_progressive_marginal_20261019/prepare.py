"""只核对/复制冻结输入和旧runtime；任何实际输入SHA不符即停止。"""
from pathlib import Path
import json,hashlib,shutil,subprocess,os
ROOT=Path(__file__).resolve().parent
S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019')
G=Path('/home/prmi/zwj/geometry-marginal-online-20261019')
H=Path('/home/prmi/zwj/sparse-hash-root-grid-20261019')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
expected=read(H/'input_manifest.json');checks={};mismatch=[]
for path,value in expected['frozen_sources'].items():
    actual=sha(path);checks[path]={'expected':value['sha256'],'actual':actual}
    if actual!=value['sha256']:mismatch.append(path)
old=read(S/'audit_manifest.json')
extra={S/'contributions.pt':old['contribution_matrix']['sha256'],S/'ranking_inputs.npz':expected['ranking_arrays_sha'],
       S/'audit_worker':old['worker_sha'],H/'fr1_room_input.bin':expected['input_sha']}
for p,digest in extra.items():
    actual=sha(p);checks[str(p)]={'expected':digest,'actual':actual}
    if actual!=digest:mismatch.append(str(p))
# Keep production untouched. The historical runtime backup supplies the original consumed bytes.
backup=G/'runtime_before/lib';resolved={}
for name in ['libgaussian_mapper.so','libcuda_rasterizer.so']:
    original='/home/prmi/zwj/mycartgs/lib/'+name;digest=old['frozen_sources'][original]['sha256']
    actual=sha(backup/name);resolved[original]={'resolved':str(backup/name),'expected':digest,'actual':actual,'current_production_sha':sha(original)}
    if actual!=digest:mismatch.append(str(backup/name))
save('frozen_sha_checks.json',{'all_passed':not mismatch,'checked':checks,'runtime_resolution':resolved,'mismatch':mismatch})
if mismatch:raise RuntimeError('STOP_FROZEN_INPUT_SHA_MISMATCH '+repr(mismatch))
info=read(S/'input_manifest.json');assert len(info['bank_frames'])==69 and len(info['evaluation_frames'])==161
assert set(info['bank_frames']).isdisjoint(info['evaluation_frames'])
assert sorted(info['bank_frames']+info['evaluation_frames'])==list(range(len(info['camera_ids'])))
for p in ['input_manifest.json','population.json','ranking_inputs.npz','audit_worker','common.py','native.cpp']:
    if p=='common.py':continue  # Locally frozen copy is already uploaded.
    shutil.copy2(S/p,ROOT/p)
shutil.copytree(backup,ROOT/'runtime/lib',dirs_exist_ok=True)
for p in ['fr1_room_input.bin','fr1_room_inputs.npz']:shutil.copy2(H/p,ROOT/p)
shutil.copy2(H/'sparse_root_results.json',ROOT/'prior_structure.json')
for p in ['input_manifest.json','mathematical_parity.json','source_expected.json']:
    shutil.copy2(H/p,ROOT/('prior_'+p))
env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
ldd=subprocess.check_output(['ldd',str(ROOT/'audit_worker')],env=env,text=True)
assert str(ROOT/'runtime/lib/libgaussian_mapper.so')in ldd
assert str(ROOT/'runtime/lib/libcuda_rasterizer.so')in ldd
(ROOT/'worker_ldd.txt').write_text(ldd)
save('preparation.json',{'all_passed':True,'data_SHA_exact':True,'historical_runtime_SHA_exact':True,
    'bank_views':69,'evaluation_views':161,'evaluation_label':'fixed non-Bank archived evaluation views',
    'production_unchanged':True,'quality_started':False,'preregistration_sha256':sha(ROOT/'preregistration.json')})
print('FROZEN_INPUTS_PASS 69_Bank 161_non_Bank runtime_exact',flush=True)
