"""固定随机抽取10个位姿，导出Full/全局0.3/局部0.3/GT与成对核验。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parent
batch=Path('/home/prmi/zwj/realLOD/baseline/dual_view_li_best_full_20261107')
seed=20261112


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    job=next(j for j in json.loads((batch/'LOD_PIPELINE.json').read_text())['jobs']
             if j['id']=='Replica/office1/mono')
    evaluations={e['scope']:e for e in job['evaluations'] if e['scope']=='full' or abs(e['ratio']-.3)<1e-6}
    full_output=Path(evaluations['full']['result']['output'])
    available=list(csv.DictReader((full_output/'per_frame_metrics.csv').open()))
    chosen=sorted(random.Random(seed).sample(available,10),key=lambda r:int(r['rgb_index']))
    indices=[int(r['rgb_index']) for r in chosen]
    model=Path(job['trained_model']).resolve();shutdown='37213_shutdown';ply=model/shutdown/'ply'
    config=batch/'configs/dual_view_li.yaml'
    protected=[model/'CameraTrajectory_TUM.txt',ply/'selector.pt',ply/'cameras.json',
        ply/'point_cloud/iteration_37213/point_cloud.ply',config,batch/'LOD_PIPELINE.json']
    before={str(f):sha(f) for f in protected}
    global_output=Path(evaluations['global']['result']['output'])
    metadata=dict(dataset='Replica',scene='office1',mode='mono',model_variant='双网络真实视角＋LI baseline，Master可更新',
        batch=str(batch),model=str(model),data=job['full_evaluation']['data'],shutdown=shutdown,
        selector=str(ply/'selector.pt'),selector_config=str(config),random_seed=seed,
        sampling='Python random.Random(seed).sample，1994个原评价关联帧中均匀无放回抽10帧，不按质量筛选',
        available_views=len(available),gt_indices=indices,ratio=.3,
        global_sampling_rng=str(global_output/'selector_sampling_rng.pt'),
        original_evaluations={scope:e['result']['output'] for scope,e in evaluations.items()},
        input_sha256=before,image_protocol='original_rgb，无缩放、裁剪或额外位姿优化',
        local_budget='名义Anchor视锥候选池，K=min(floor(0.3*N_candidate),eligible)，双网络真实视角',
        global_budget='沿用原全局ratio导出和零视角条件，复用原评价seed0及相同随机数调用顺序')
    if (root/'EXPERIMENT.json').exists():
        prior=json.loads((root/'EXPERIMENT.json').read_text())
        assert prior['input_sha256']==before and prior['gt_indices']==indices
    (root/'EXPERIMENT.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (root/'FRAME_SET.json').write_text(json.dumps(dict(gt_indices=indices,seed=seed),indent=2)+'\n')
    print('SAMPLED_FRAMES',indices,flush=True)
    sys.path.insert(0,str(batch));import run_batch
    env=run_batch.environment()
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for scope in ['full','global','view_ratio']:
        folder={'full':'full_0','global':'global_0.3','view_ratio':'local_0.3'}[scope]
        if (root/folder/'SELECTED_FRAME_RESULTS.json').exists():continue
        with (logs/(scope+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,'-u',str(root/'render_one_scope.py'),str(root),scope],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert process.returncode==0,scope+'失败，见'+str(logs/(scope+'.log'))
        print('SCOPE_FINISHED',scope,flush=True)
    outputs={scope:root/folder for scope,folder in [('full','full_0'),('global','global_0.3'),('view_ratio','local_0.3')]}
    results={scope:json.loads((out/'SELECTED_FRAME_RESULTS.json').read_text()) for scope,out in outputs.items()}
    images=root/'images'
    for name in ['GT','Full','Global_0.3','Local_0.3']:(images/name).mkdir(parents=True,exist_ok=True)
    rows=[];parity={};poses=[]
    for scope in outputs:
        original=list(csv.DictReader((Path(metadata['original_evaluations'][scope])/'per_frame_metrics.csv').open()))
        reference={int(r['rgb_index']):r for r in original}
        parity[scope]=max(abs(r['psnr']-float(reference[r['gt_index']]['psnr'])) for r in results[scope])
    import numpy as np
    reference_global=np.load(global_output/'selector_selected_ids.npy')
    actual_global=np.load(outputs['global']/'selector_selected_ids.npy')
    assert np.array_equal(reference_global,actual_global),'全局选点身份发生变化'
    original_local=json.loads((Path(metadata['original_evaluations']['view_ratio'])/'selection_hashes.json').read_text())
    actual_local=json.loads((outputs['view_ratio']/'selection_hashes.json').read_text())
    frame_positions={int(r['rgb_index']):i for i,r in enumerate(available)}
    assert actual_local==[original_local[frame_positions[i]] for i in indices],'局部选点身份发生变化'
    assert max(parity.values())<.001,parity
    for i,index in enumerate(indices):
        full,glob,local=[results[scope][i] for scope in ['full','global','view_ratio']]
        assert full['c2w']==glob['c2w']==local['c2w']
        stem='view_%02d_frame_%06d'%(i+1,index)
        gt=outputs['full']/'gt'/Path(full['image']).name
        shutil.copy2(gt,images/'GT'/(stem+'.png'))
        for row,name in [(full,'Full'),(glob,'Global_0.3'),(local,'Local_0.3')]:
            shutil.copy2(row['image'],images/name/(stem+'.png'))
        xyz=np.asarray(full['c2w'])[:3,3].tolist()
        rows.append(dict(view=i+1,gt_index=index,trajectory_index=full['trajectory_index'],timestamp=full['timestamp'],
            camera_x=xyz[0],camera_y=xyz[1],camera_z=xyz[2],full_psnr=full['psnr'],global_psnr=glob['psnr'],
            local_psnr=local['psnr'],full_K=full['K'],global_K=glob['K'],local_K=local['K'],
            local_candidate_N=local['candidate_N'],image_stem=stem))
        poses.append(dict(view=i+1,gt_index=index,timestamp=full['timestamp'],c2w=full['c2w'],w2c=full['w2c']))
    with (root/'CAMERAS_AND_METRICS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    camera=json.loads((ply/'cameras.json').read_text())[0]
    (root/'CAMERA_POSES.json').write_text(json.dumps(dict(coordinate_system='原Master地图坐标系，与实际渲染一致',
        width=camera['width'],height=camera['height'],fx=camera['fx'],fy=camera['fy'],cameras=poses),ensure_ascii=False,indent=2)+'\n')
    assert all(sha(Path(name))==digest for name,digest in before.items())
    receipt=dict(passed=True,views=10,original_images=40,input_files_unchanged=len(before),
        pose_equality_verified=True,global_selected_ids_match_original=True,local_selected_ids_match_original=True,
        maximum_psnr_difference_vs_original=parity,random_seed=seed,
        means={scope:float(np.mean([r['psnr'] for r in results[scope]])) for scope in results})
    (root/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (root/'说明.md').write_text('# Replica office1单目：随机10个位姿渲染对比\n\n'
        '使用双网络真实视角＋LI baseline最终checkpoint，非Master detach消融。'
        '固定随机种子20261112，从1994个原评价关联位姿均匀无放回抽取10个，不按PSNR筛选。'
        'Full、全局0.3、局部0.3在完全相同估计相机位姿上渲染；GT为对应原始RGB，1200×680。\n\n'
        '全局0.3沿用原全图选点与零视角条件；局部0.3使用名义Anchor视锥候选池的严格预算及双网络真实视角。'
        '实际K不是Alpha合成可见点数，两种0.3预算分母不同。PSNR基于原始FP32渲染计算；PNG为8位无损落盘。\n\n'
        'CAMERAS_AND_METRICS.csv包含位姿位置、时间戳、逐帧PSNR与K；CAMERA_POSES.json包含完整c2w/w2c矩阵。'
        'images/含40张原分辨率PNG。10帧均值仅描述本随机样本，不等于全轨迹指标。'
        '全局与局部选点身份和原评价一致，源模型及配置哈希未变化，复验见VERIFIED.json。\n')
    print('ALL_VERIFIED',json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
