"""核验同配置重跑并导出旧／新逐ratio指标、训练序列与源码身份对照。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile

R=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare_sequence(a,b,fields):
    left=list(csv.DictReader(a.open()));right=list(csv.DictReader(b.open()))
    assert len(left)==len(right)
    different=[]
    for index,(x,y) in enumerate(zip(left,right)):
        if any(x[name]!=y[name] for name in fields):different.append(index+1)
    return dict(rows=len(left),identical=not different,different_rows=len(different),first_different=different[:10])


def compare_manifest(old,new):
    for key in ['full_ply','point_cloud','anchor_state','initial_iteration','mode','dataset','dataset_root','optimizer_protocol']:
        assert old[key]==new[key],key
    for key in ['camera_config','mapping_config','selector_config']:
        assert sha(Path(old[key]))==sha(Path(new[key])),key
    assert sorted(old['input_sha256'].values())==sorted(new['input_sha256'].values())


def summarize_health(path):
    result={}
    with path.open() as stream:
        for line in stream:
            record=json.loads(line);branch=record.get('branch','global');summary=result.setdefault(branch,dict(records=0))
            summary['records']+=1
            for key in ['scale_factor_max','xyz_anchor_units_max','zero_streak_global',
                        'zero_streak_ratio','saturation_streak_ratio','scale_tail_fraction','xyz_tail_fraction']:
                value=record[key];assert math.isfinite(value)
                summary[key]=max(summary.get(key,0),value)
    return result


def main():
    state=read(R/'LOD_PIPELINE.json');old={job['id']:job for job in read(R/'REFERENCE_JOBS.json')}
    assert state['phase']=='finished' and all(job['state']=='complete' for job in state['jobs'])
    before=read(R/'PROTECTED_BEFORE.json');after=read(R/'PROTECTED_AFTER.json');assert before==after
    assert all(sha(Path(name))==digest for names in before.values() for name,digest in names.items())
    source=read(R/'reports/SOURCE_HASHES.json');build=read(R/'reports/BUILD_HASHES.json')
    assert all(sha(R/'source'/name)==digest for name,digest in source.items())
    assert all(sha(R/name)==digest for name,digest in build.items())
    rows=[];details=[]
    for job in state['jobs']:
        reference=old[job['id']];out=Path(job['attempt']);old_out=Path(reference['attempt'])
        compare_manifest(read(old_out/'manifest.json'),read(out/'manifest.json'))
        initialization=read(out/'model/initialization_verified.json');assert initialization['passed']
        verified=read(out/'model/training_verified.json');assert verified['passed'] and verified['steps']==job['target_steps']
        keyframes=compare_sequence(old_out/'model/training_keyframes.csv',out/'model/training_keyframes.csv',['step','keyframe_id'])
        training_trace=compare_sequence(old_out/'model/real_lod_steps.csv',out/'model/real_lod_steps.csv',['step','ratio'])
        for name in ['CameraTrajectory_TUM.txt','TrackingTime.txt']:
            assert sha(old_out/'model'/name)==sha(out/'model'/name)
        previous={(item['scope'],item['ratio']):item for item in reference['evaluations']}
        assert len(job['evaluations'])==len(previous)==23
        for item in job['evaluations']:
            assert item['state']=='complete'
            prior=previous[(item['scope'],item['ratio'])]
            new_result=item['result'];old_result=prior['result']
            new_path=Path(new_result['output']);old_path=Path(old_result['output'])
            assert read(new_path/'image_preprocessing.json')==read(old_path/'image_preprocessing.json')=={'protocol':'original_rgb'}
            association=compare_sequence(old_path/'per_frame_metrics.csv',new_path/'per_frame_metrics.csv',
                 ['synchronized_trajectory_index','rgb_index','trajectory_timestamp','rgb_timestamp','rgb_path'])
            assert association['identical']
            actual=read(new_path/'metric_summary.json')
            frame_rows=list(csv.DictReader((new_path/'per_frame_metrics.csv').open()))
            for key in ['psnr','ssim','lpips']:
                assert math.isfinite(new_result[key]) and new_result[key]==actual[key]
                assert abs(sum(float(row[key]) for row in frame_rows)/len(frame_rows)-new_result[key])<1e-9
            row=dict(id=job['id'],scope=item['scope'],ratio=item['ratio'],ratio_kind=item['kind'],
                     seed=0,fine_steps=job['target_steps'],frames=new_result['evaluated_frames'],
                     old_N=old_result['gaussian_count'],new_N=new_result['gaussian_count'],
                     old_K_mean=old_result['actual_K_mean'],new_K_mean=new_result['actual_K_mean'],
                     old_K_min=old_result['actual_K_min'],new_K_min=new_result['actual_K_min'],
                     old_K_max=old_result['actual_K_max'],new_K_max=new_result['actual_K_max'],
                     evaluation_protocol='cartgs_original_rgb_full_trajectory',
                     frame_associations_equal=True,old_output=str(old_path),new_output=str(new_path))
            for key in ['psnr','ssim','lpips']:
                row['old_'+key]=old_result[key];row['new_'+key]=new_result[key]
                row['delta_'+key]=new_result[key]-old_result[key]
            if item['scope']=='global':
                import numpy as np
                a=np.load(old_path/'selector_selected_ids.npy');b=np.load(new_path/'selector_selected_ids.npy')
                inter=len(set(a.tolist())&set(b.tolist()));row['selected_id_jaccard']=inter/max(1,len(set(a.tolist())|set(b.tolist())))
                row['selected_identities_equal']=bool(np.array_equal(a,b))
            elif item['scope']=='view_ratio':
                a=read(old_path/'selection_hashes.json');b=read(new_path/'selection_hashes.json')
                row['selected_identities_equal']=a==b
                row['equal_selection_frame_fraction']=sum(x==y for x,y in zip(a,b))/len(a)
            rows.append(row)
        details.append(dict(id=job['id'],full_checkpoint_unchanged=True,configs_identical=True,
                  initialization_bitwise_verified=True,keyframe_sequence=keyframes,ratio_sequence=training_trace,
                  steps=job['target_steps'],attempts=len(job['attempt_history']),
                  health_failures=job['health_failures'],oom_failures=job['oom_failures'],
                  training_seconds=job['attempt_history'][-1]['seconds'],trained_model=job['trained_model'],
                  gaussian_count=job['evaluations'][0]['result']['gaussian_count'],
                  evaluated_frames=job['evaluations'][0]['result']['evaluated_frames'],
                  old_health=summarize_health(old_out/'model/lod_health.jsonl'),
                  new_health=summarize_health(out/'model/lod_health.jsonl')))
    assert len(rows)==46
    csv_path=R/'reports/RERUN_COMPARISON.csv'
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
    images=[str(p) for folder in ['runs','evaluation','validation'] for p in (R/folder).rglob('*')
            if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']]
    assert not images
    receipt=dict(complete=True,jobs=details,comparison_rows=46,source_files_verified=len(source),
          runtime_files_verified=len(build),protected_files=sum(len(x) for x in before.values()),
          protected_hashes_unchanged=True,old_results_replaced=False,render_images_saved=0,
          reference_batch=state['reference_batch'])
    (R/'reports/RERUN_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    lines=['# corner-slow与robot-normal双目：同配置配对重跑','',
      '两组均从上次相同Full checkpoint重新执行LoD Fine，随后完成每组23项全轨迹数值评价。'
      '原模型及历史汇总不替换，新结果独立保存。', '',
      '## 一致性核验','',
      f'原批次{len(source)}个源码文件、{len(build)}个运行文件SHA256相同，三个配置文件逐字相同。'
      f'{receipt["protected_files"]}个受保护文件研究前后哈希未变化。', '',
      'seed=0，Full＋全局LoD＋局部LoD、双网络真实视角＋LI=0.1、Master可更新、固定拓扑。'
      '训练Ratio为[0.01,0.05,0.1,0.3,0.5,0.7]；未见Ratio为[0.03,0.08,0.2,0.4,0.6]，仅评价。', '',
      '训练沿用原双目相机校正；评价统一CaRtGS未校正原图，关闭尺度对齐。'
      '每项新旧评价的轨迹／RGB关联逐帧相同，PSNR／SSIM／LPIPS均核对逐帧均值，未保存渲染图片。', '',
      '| 场景 | Fine步数 | Gaussian数量N | 评价帧数 | 训练秒数 | 初始化逐bit | 关键帧序列相同 | Ratio序列相同 | 健康失败／OOM |',
      '|---|---:|---:|---:|---:|---|---|---|---|']
    for d in details:
        lines.append(f"| {d['id']} | {d['steps']} | {d['gaussian_count']} | {d['evaluated_frames']} | {d['training_seconds']:.1f} | 通过 | {d['keyframe_sequence']['identical']} | {d['ratio_sequence']['identical']} | {d['health_failures']}／{d['oom_failures']} |")
    lines+=['','健康日志按global／view_budget分别汇总到RERUN_VERIFIED.json，包含全程最大Scale倍数、XYZ Anchor单位位移、零选点及概率饱和连续次数；不能仅用单点概率达到1判定整体坍缩。']
    lines+=['','## 复跑差异摘要','']
    for d in details:
        selected=[row for row in rows if row['id']==d['id']]
        full,=[row for row in selected if row['scope']=='full']
        lines.append(f"{d['id']}：Full PSNR由{full['old_psnr']:.4f}变为{full['new_psnr']:.4f} dB（{full['delta_psnr']:+.4f} dB）。")
        for scope,label in [('global','全局'),('view_ratio','局部')]:
            group=[row for row in selected if row['scope']==scope]
            delta=[row['delta_psnr'] for row in group]
            low=next(row for row in group if row['ratio']==.01)
            lines.append(f"{label}11个Ratio的PSNR差值范围为{min(delta):+.4f}至{max(delta):+.4f} dB；1%为{low['old_psnr']:.4f}→{low['new_psnr']:.4f} dB。")
        lines+=['',f"新模型：`{d['trained_model']}`。",'']
    lines+=['','## Full终点与逐Ratio结果','','差值均为本次−旧结果；PSNR／SSIM越高越好，LPIPS越低越好。', '',
      '| 场景 | 模式 | Ratio | 旧PSNR | 本次PSNR | ΔPSNR | 旧SSIM | 本次SSIM | 旧LPIPS | 本次LPIPS | 本次平均K |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['scope']} | {row['ratio']:g} | {row['old_psnr']:.6f} | {row['new_psnr']:.6f} | {row['delta_psnr']:+.6f} | {row['old_ssim']:.6f} | {row['new_ssim']:.6f} | {row['old_lpips']:.6f} | {row['new_lpips']:.6f} | {row['new_K_mean']:.2f} |")
    lines+=['','## 解释边界','',
      '此次是相同Full起点的LoD阶段复跑，未重新执行在线SLAM。'
      '相同seed和配置不保证训练中GPU原子梯度累加完全确定；本报告分别记录初始化、相机序列、Ratio序列和最终结果。'
      '单次复跑差异不能独自证明旧数据损坏，也不能把双目校正训练域与未校正原图评价域之间的差异当作训练故障。', '',
      '完整精度、逐视角、选点重合及来源见RERUN_COMPARISON.csv及各评价目录。']
    (R/'reports/RERUN_REPORT.md').write_text('\n'.join(lines)+'\n')
    archive_path=R/'RERUN_RESULTS.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in (R/'reports').rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(R))
        for name in ['LOD_PIPELINE.json','PROTECTED_BEFORE.json','PROTECTED_AFTER.json','REFERENCE_JOBS.json','PREPARED.json']:
            archive.write(R/name,name)
    print(json.dumps(receipt,ensure_ascii=False))
    print('REPORT',R/'reports/RERUN_REPORT.md',flush=True)


if __name__=='__main__':main()
