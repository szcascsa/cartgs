"""原批次两组双目LoD配对重跑；复制冻结源码/运行库，不改历史结果。"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time

R=Path(__file__).resolve().parent
B=R.parent/'dual_view_li_vector_best_full_20261107'
IDS={'VECtor/corner-slow/stereo','VECtor/robot-normal/stereo'}


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert R.name=='dual_view_li_vector_stereo_rerun_20261111'
    assert not (R/'LOD_PIPELINE.json').exists()
    assert shutil.disk_usage(R).free>5*2**30
    original=read(B/'LOD_PIPELINE.json')
    assert original['phase']=='finished'
    reference=[job for job in original['jobs'] if job['id'] in IDS]
    assert len(reference)==2 and all(job['state']=='complete' for job in reference)
    source_hashes=read(B/'reports/SOURCE_HASHES.json')
    runtime_hashes=read(B/'reports/BUILD_HASHES.json')
    assert all(sha(B/'source'/name)==digest for name,digest in source_hashes.items())
    assert all(sha(B/name)==digest for name,digest in runtime_hashes.items())
    protected={B/'LOD_PIPELINE.json',B/'reports/LOD_METRICS.csv',B/'configs/dual_view_li.yaml'}
    for folder in ['source','runtime','evaluation_runtime']:
        protected.update(p for p in (B/folder).rglob('*') if p.is_file() and p.suffix not in ['.pyc'])
    for job in reference:
        full=Path(job['full_job']['result']['checkpoint'])
        old=Path(job['trained_model'])
        protected.update(p for p in full.rglob('*') if p.is_file())
        protected.update(p for p in old.rglob('*') if p.is_file())
        protected.update([Path(job['camera_config']),Path(job['mapping_config'])])
        for item in job['evaluations']:
            output=Path(item['result']['output'])
            assert read(output/'image_preprocessing.json')=={'protocol':'original_rgb'}
            protected.update(output/name for name in ['metric_summary.json','per_frame_metrics.csv'])
        manifest=read(Path(job['attempt'])/'manifest.json')
        assert all(sha(Path(name))==digest for name,digest in manifest['input_sha256'].items())
    for base in [Path('/home/prmi/zwj/realLOD'),Path('/home/prmi/zwj/mycartgs')]:
        for relative in ['src/gaussian_mapper.cpp','include/gaussian_mapper.h',
                         'improvements/selector/spatial_compensation.h',
                         'cfg/selector/spatial_offset3_residual48.yaml',
                         'bin/tum_rgbd','bin/lod_from_full','lib/libgaussian_mapper.so',
                         'python/run.py','python/selector.py']:
            if (base/relative).is_file():protected.add(base/relative)
    write(R/'PROTECTED_BEFORE.json',{'historical_and_production':{str(p):sha(p) for p in sorted(protected)}})
    for name in ['source','runtime','evaluation_runtime']:
        shutil.copytree(B/name,R/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['configs','reports','logs','runs','evaluation','validation']:(R/name).mkdir(exist_ok=True)
    for name in ['pipeline.py','oom_retry.py','lod_eval_entry.py','eval_joint_entry.py',
                 'verify_initialization.py','validate_startup.py','test_queue.py','supervisor.py']:
        shutil.copy2(B/name,R/name)
    text=(B/'run_batch.py').read_text()
    old="assert len(state['jobs'])==6 and all(j['dataset']=='VECtor' for j in state['jobs'])"
    new="assert len(state['jobs'])==2 and {j['id'] for j in state['jobs']}=={'VECtor/corner-slow/stereo','VECtor/robot-normal/stereo'}"
    assert text.count(old)==1
    text=text.replace(old,new)
    stereo="        if job['mode']=='stereo':cmd+=['--stereo-camera-config',job['camera_config'],'--stereo-native-maps',str(R/'configs/vector_stereo_native_maps.yml.gz')]"
    assert text.count(stereo)==1
    text=text.replace(stereo,"        # 用户已统一口径：评价使用CaRtGS未校正原图，不加载双目校正map。")
    marker="result.update(scope=item['scope'],output=str(out),saved_images=0)"
    assert text.count(marker)==1
    text=text.replace(marker,"result.update(scope=item['scope'],output=str(out),saved_images=0,evaluation_protocol='cartgs_original_rgb_full_trajectory')")
    (R/'run_batch.py').write_text(text)
    shutil.copy2(B/'configs/dual_view_li.yaml',R/'configs/dual_view_li.yaml')
    jobs=copy.deepcopy(reference)
    for job in jobs:
        for key in ['process','attempt','observed_at','log_bytes','trace_bytes','trained_model','error',
                    'retry_round_history','retry_round','round_first_attempt']:
            job.pop(key,None)
        job.update(state='pending',evaluations=[],attempt_history=[],health_failures=0,oom_failures=0)
        for key in ['camera_config','mapping_config']:
            source=Path(job[key]);target=R/source.relative_to(B);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);assert sha(source)==sha(target);job[key]=str(target)
        full=Path(job['full_job']['result']['checkpoint'])
        assert read(full/'real_lod_state.json')['lod_steps']==0
        assert read(full/'real_lod_state.json')['online_full_steps']==job['target_steps']
    assert all(sha(R/'source'/name)==digest for name,digest in source_hashes.items())
    assert all(sha(R/name)==digest for name,digest in runtime_hashes.items())
    for name in ['SOURCE_HASHES.json','SOURCE_PROVENANCE.json','BUILD_HASHES.json']:
        shutil.copy2(B/'reports'/name,R/'reports'/name)
    protocol=copy.deepcopy(original['protocol']);protocol.pop('excluded_dataset',None)
    protocol.update(evaluation_protocol='cartgs_original_rgb_full_trajectory',repeated_seed=0,
                    full_selection='保留原选中Full，不重新选择赢家',scope='LoD Fine重训＋完整轨迹评价')
    write(R/'REFERENCE_JOBS.json',reference)
    write(R/'LOD_PIPELINE.json',dict(phase='prepared',jobs=jobs,validated_modes=[],protocol=protocol,
          counts={'pending':2},reference_batch=str(B)))
    write(R/'reports/BOOTSTRAP.json',dict(phase='build_complete',prepared_unix=time.time(),
          runtime_reused_from=str(B),runtime_hashes_verified=len(runtime_hashes),
          source_hashes_verified=len(source_hashes),independent_source_runtime=True,compiled=False))
    receipt=dict(prepared=True,seed=0,jobs=[dict(id=j['id'],fine_steps=j['target_steps'],
                    full_checkpoint=j['full_job']['result']['checkpoint']) for j in jobs],
          source_files=len(source_hashes),runtime_files=len(runtime_hashes),
          config_sha256=sha(R/'configs/dual_view_li.yaml'),protected_files=len(protected),
          quality_comparison='与旧模型相同完整轨迹、未校正原图、11档全局＋11档局部＋Full',
          render_images_saved=0)
    write(R/'PREPARED.json',receipt)
    lines=['# VECtor两组双目配对重跑协议','',
      '按用户要求重跑corner-slow双目及robot-normal双目的LoD Fine阶段，旧结果和Full起点只读保留。', '',
      '使用原批次冻结的527个源码文件及16个运行文件，SHA256逐项一致，不使用工作树后续改动，不重新编译或修改生产配置。', '',
      '- 相同seed=0；corner-slow 11,470步，robot-normal 13,006步。',
      '- 双网络真实视角＋LI=0.1；Full＋全局LoD＋局部LoD，每步联合反传后一次Adam。',
      '- Master可更新，拓扑固定；每次从原Full起点重新初始化Adam/RNG，不是完整SLAM恢复。',
      '- 训练Ratio：[0.01,0.05,0.1,0.3,0.5,0.7]轮转；未见Ratio：[0.03,0.08,0.2,0.4,0.6]仅评价。',
      '- 全局训练全图候选；局部使用原名义Anchor视锥候选预算。训练相机校正与原批次完全相同。',
      '- 评价以CaRtGS未校正原图为准，双目无尺度对齐；每组23项完整轨迹数值评价，不保存渲染图片。',
      '- 健康守卫、零选点和XYZ／Scale长尾阈值均保留；健康停止或明确OOM各最多额外重试2次，历史现场不覆盖。',
      '- 初始化Master／网络逐bit验证；原训练及新训练的keyframe序列、更新预算、身份与数值结果分别比较。',
      '- 只在新独立目录写配置、模型、二进制、日志与评价，不commit／push。', '',
      'Full起点PSNR及选择赢家记录仍保留当时校正左图口径，最终对照以此次及11月8日更新的未校正原图结果为准。', '',
      '| 场景 | Fine步数 | Full起点 |','|---|---:|---|']
    for job in jobs:lines.append(f"| {job['id']} | {job['target_steps']} | {job['full_job']['result']['checkpoint']} |")
    (R/'reports/RERUN_PROTOCOL.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
