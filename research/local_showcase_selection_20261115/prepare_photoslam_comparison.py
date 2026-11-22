"""提取用户提供的Photo-SLAM独立图片，核对视角与原始GT并打包统一评价输入。"""
from pathlib import Path
import csv
import hashlib
import io
import json
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
PHOTO=WORKSPACE/'outputs/pruned_requested_views_20261115'
OUT=WORKSPACE/'outputs/local_photoslam_comparison_20261115'
PAIRS=[('Replica/room0/mono','photo__Replica__room0__mono',950,947),
       ('TUM_RGBD/freiburg2_xyz/rgbd','photo__TUM__freiburg2_xyz__rgbd',551,551)]
RATIOS=[.01,.05,.1,.3]


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=load(ROOT/'furnished_revision/FURNISHED_SELECTION.json')
    photo=load(PHOTO/'VIEW_MANIFEST.json')
    reported=load(PHOTO/'VIEW_METRICS_WITH_FULL.json')
    protected={str(p):sha(p) for p in [PHOTO/'VIEW_MANIFEST.json',PHOTO/'VIEW_METRICS_WITH_FULL.json',
                                     PHOTO/'Photo-SLAM_指定视角剪枝指标.csv',PHOTO/'Photo-SLAM_requested_views.zip',
                                     ROOT/'furnished_revision/FURNISHED_SELECTION.json']}
    evaluations=[];images=[];groups=[];original=[]
    with zipfile.ZipFile(PHOTO/'Photo-SLAM_requested_views.zip') as archive:
        assert archive.testzip() is None
        for group,key,rgb_index,trajectory_index in PAIRS:
            ours=next(x for x in source['groups'] if x['id']==group)
            pmeta=next(x for x in photo['records'] if x['group']==key)
            assert ours['frame']['rgb_index']==pmeta['input_frame_zero_based']==rgb_index
            assert ours['frame']['trajectory_index']==pmeta['trajectory_index_zero_based']==trajectory_index
            assert Path(ours['frame']['rgb_path']).name==Path(pmeta['rgb_file']).name
            alias='replica_room0_mono' if group.startswith('Replica') else 'tum_fr2_xyz_rgbd'
            folder=OUT/'images'/alias;folder.mkdir(parents=True,exist_ok=True)
            ours_source=ROOT/'furnished_revision/selected'/group
            raw=np.array(Image.open(ours_source/'GT.png').convert('RGB'))
            raw_photo=np.array(Image.open(io.BytesIO(archive.read(key+'/gt_raw.png'))).convert('RGB'))
            assert np.array_equal(raw,raw_photo)
            shutil.copyfile(ours_source/'GT.png',folder/'GT_raw.png')
            (folder/'GT_photo_native.png').write_bytes(archive.read(key+'/gt_evaluation.png'))
            native=np.array(Image.open(folder/'GT_photo_native.png').convert('RGB'))
            groups.append(dict(id=group,alias=alias,rgb_index=rgb_index,trajectory_index=trajectory_index,
                               raw_gt_identical=True,raw_native_pixel_mae=float(np.abs(raw.astype(float)-native.astype(float)).mean()),
                               photo_N=pmeta['N0'],ours_N=ours['frame']['N'],
                               candidate_N=ours['frame']['candidates'],photo_source_id=pmeta['source_id']))
            for method in ['Photo-SLAM','Ours']:
                for ratio in RATIOS+([1.] if method=='Photo-SLAM' else []):
                    pct=int(ratio*100);file=folder/(f'photo_{pct:03d}.png' if method=='Photo-SLAM' else f'ours_{pct:03d}.png')
                    if method=='Photo-SLAM':
                        name='full.png' if ratio==1 else 'ratio_'+str(ratio).replace('.','p')+'.png'
                        file.write_bytes(archive.read(key+'/'+name))
                        record=next(x for x in reported if x['group']==key and float(x['ratio'])==ratio)
                        K=int(record['K']);N=pmeta['N0'];pool=N
                        original.append(dict(group=group,method='Photo-SLAM (opacity Top-K)',ratio=ratio,
                                             K=K,N=N,pool_N=pool,psnr=float(record['psnr']),ssim=float(record['ssim']),
                                             lpips=float(record['lpips']),gt_domain='native_undistorted',
                                             precision='FP32原报告',source='Photo-SLAM/VIEW_METRICS_WITH_FULL.json'))
                    else:
                        source_png=ours_source/f'ratio_{pct:02d}.png';protected[str(source_png)]=sha(source_png)
                        shutil.copyfile(source_png,file)
                        record=next(x for x in ours['records'] if x['ratio']==ratio)
                        K=record['K'];N=record['N'];pool=record['candidates']
                        original.append(dict(group=group,method=method,ratio=ratio,K=K,N=N,pool_N=pool,
                                             **record['scores'],gt_domain='original_rgb',precision='FP32原报告',
                                             source='Ours/FURNISHED_SELECTION.json'))
                    assert Image.open(file).size==(raw.shape[1],raw.shape[0])
                    image=dict(group=group,alias=alias,method=method,ratio=ratio,K=K,N=N,pool_N=pool,
                               render=str(file.relative_to(OUT)).replace('\\','/'),
                               gt=str((folder/'GT_raw.png').relative_to(OUT)).replace('\\','/'),
                               sha256=sha(file),rgb_index=rgb_index,trajectory_index=trajectory_index)
                    images.append(image);evaluations.append(dict(image,gt_domain='original_rgb',precision='uint8_PNG_to_FP32'))
    (OUT/'INPUT_MANIFEST.json').write_text(json.dumps(dict(groups=groups,images=images,evaluations=evaluations,
        original=original,protected_sha256=protected),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'EVALUATION_REQUEST.json').write_text(json.dumps(evaluations,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(OUT/'EVALUATION_INPUTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUT/'EVALUATION_REQUEST.json','EVALUATION_REQUEST.json')
        for path in (OUT/'images').rglob('*.png'):archive.write(path,path.relative_to(OUT))
    assert all(sha(Path(name))==digest for name,digest in protected.items())
    print('PREPARED',OUT,'images',len(images),'groups',json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':main()
