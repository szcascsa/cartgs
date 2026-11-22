from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/prmi/zwj/mycartgs');OLD=Path('/home/prmi/zwj/hierarchical-representative-si-20261019')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=json.loads((ROOT/'audit_before.json').read_text())
assert sha(OLD/'fr1_room_input.bin')==before['old_fixture_sha256']
prior=json.loads((OLD/'input_manifest.json').read_text())
for p,v in prior['frozen_sources'].items():assert sha(p)==v['sha256'],p
assert sha(Path('/home/prmi/zwj/geometry-marginal-online-20261019/ranking_inputs.npz'))==prior['ranking_arrays_sha']
source=ROOT/'source'
for n in ['include','improvements/selector']:shutil.copytree(REPO/n,source/n,dirs_exist_ok=True)
for n in ['src/global_gi.cpp','src/gi_teacher.cpp','tests/global_gi_test.cpp']:
    p=source/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
for n in ['fr1_room_input.bin','fr1_room_inputs.npz']:shutil.copy2(OLD/n,ROOT/n)
for n in ['fr1_room_output.bin','fr1_room_tree.csv','fr1_room_telemetry.json']:
    d=ROOT/'old_reference'/n;d.parent.mkdir(exist_ok=True);shutil.copy2(OLD/'smoke'/n,d)
shutil.copy2(OLD/'input_manifest.json',ROOT/'input_manifest.json')
print('FROZEN_DATA_EXACT',prior['N'],prior['candidate_N'],sha(ROOT/'fr1_room_input.bin')), 
