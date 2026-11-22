"""锁定第一地图与测试帧；创建独立fr2地图，不读取任何评价质量。"""
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
REPO=Path('/home/prmi/zwj/mycartgs')
scenes=dict(fr1_xyz=dict(checkpoint='/home/prmi/zwj/anchor-shared16-ratio8-20261026/r3_independent14_seed2026/6400_shutdown/ply',
    dataset='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
    test_indices=json.loads(Path('/home/prmi/zwj/anchor-shared16-ratio8-20261026/validation/evaluation_plan.json').read_text())['nonbank_frames']))
path=ROOT/'scenes.json'
if path.exists():scenes.update(json.loads(path.read_text()))
path.write_text(json.dumps(scenes,indent=2))
if 'fr2_xyz' not in scenes:
    run=ROOT/'independent_map_fr2';run.mkdir(exist_ok=True)
    config=run/'input_selector.yaml'
    config.write_text((REPO/'cfg/selector/selector_config.yaml').read_text())
    cp=list(run.glob('*_shutdown/ply'))
    if not cp:
        env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',CARTGS_AUDIT_SEED='5200',
            LD_LIBRARY_PATH=str(REPO/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
        cmd=[REPO/'bin/tum_rgbd',REPO/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
            REPO/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml',REPO/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
            '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg2_xyz',REPO/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_xyz.txt',
            run,'no_viewer','--selector-config',config]
        with (run/'training.log').open('w') as log:subprocess.run(list(map(str,cmd)),cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cp=list(run.glob('*_shutdown/ply'))
    assert len(cp)==1
    data=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg2_xyz')
    rgb=np.loadtxt(data/'rgb.txt',dtype=str,comments='#');bank={c['img_name'] for c in json.loads((cp[0]/'cameras.json').read_text())}
    trajectory=np.loadtxt(run/'CameraTrajectory_TUM.txt',comments='#')
    available=[i for i,(stamp,name) in enumerate(rgb) if name not in bank and abs(trajectory[:,0]-float(stamp)).min()<.08]
    assert len(available)>=64
    chosen=[available[int(i)] for i in np.rint(np.linspace(0,len(available)-1,64))]
    scenes['fr2_xyz']=dict(checkpoint=str(cp[0]),dataset=str(data),test_indices=chosen)
    path.write_text(json.dumps(scenes,indent=2))
print('SCENES_PREPARED',json.dumps(scenes),flush=True)
