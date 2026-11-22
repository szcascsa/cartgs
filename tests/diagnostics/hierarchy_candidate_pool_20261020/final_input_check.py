from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
checks=[]
for s in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/s;prior=json.loads((p/'prior_checks.json').read_text());info=json.loads((p/'input_manifest.json').read_text())
    for source,a in prior['source_sha'].items():assert sha(source)==a['sha256'],('STOP_END_INPUT_CHANGED',source)
    assert sha(info['cameras'])==prior['camera_sha256']
    for source,expected in prior['artifact_sha'].items():assert sha(source)==expected
    tensor=json.loads((p/'frozen_tensor_checks.json').read_text());assert tensor['ok']and tensor['maturity_all_one']
    checks.append(dict(scene=s,original_source_count=len(prior['source_sha']),original_artifacts=len(prior['artifact_sha']),camera_archive_unchanged=True,Master_TF_tensor_verify=tensor,all_passed=True))
(ROOT/'final_input_checks.json').write_text(json.dumps(dict(all_passed=True,scenes=checks),indent=2)+'\n')
print('END_FROZEN_INPUT_PASS',checks,flush=True)
