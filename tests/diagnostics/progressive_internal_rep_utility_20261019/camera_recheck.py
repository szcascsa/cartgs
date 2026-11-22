from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
checks=[]
for s in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/s;prior=json.loads((p/'prior_checks.json').read_text());info=json.loads((p/'input_manifest.json').read_text());actual=hashlib.sha256(Path(info['cameras']).read_bytes()).hexdigest()
    assert actual==prior['camera_sha256']
    if s=='fr1_room':expected=prior['source_sha'][info['cameras']]['sha256']
    else:
        old=Path('/home/prmi/zwj/hierarchical-xyz-f1f2-mechanism-20261019');cam=old/s/'camera_checks.json';manifest=json.loads((old/'bundle_manifest.json').read_text())['files'];assert hashlib.sha256(cam.read_bytes()).hexdigest()==manifest[s+'/camera_checks.json']['sha256'];expected=json.loads(cam.read_text())['camera_archive_sha256'];shutil.copy2(cam,p/'prior_camera_checks.json')
    assert expected==actual
    checks.append(dict(scene=s,camera=info['cameras'],sha256=actual,historical_expected=expected,exact=True))
(ROOT/'camera_checks.json').write_text(json.dumps(dict(all_passed=True,checks=checks),indent=2)+'\n')
print('CAMERA_HISTORICAL_EXACT_PASS',flush=True)
