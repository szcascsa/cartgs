"""从完整方案的只读快照准备D组32场景；不修改生产或历史结果。"""
from pathlib import Path
import collections
import copy
import difflib
import hashlib
import json
import os
import shutil
import time

R=Path(os.environ.get('CARTGS_CLUSTER_ROOT',Path(__file__).resolve().parent))
B=Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
V=B.parent/'dual_view_li_vector_best_full_20261107'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def replace(text,old,new,count=1):
    assert text.count(old)==count,old
    return text.replace(old,new)


def patch_sources():
    patches={
      'include/lod_health_guard.h':[
        ('bool enabled=true;','bool enabled=true, zero_selection_only=false;'),
        ('if(saturation_streak[ratio_index]>=saturation_patience)',
         'if(!zero_selection_only&&saturation_streak[ratio_index]>=saturation_patience)'),
        ('if(a[0]>=limit)return "compensation_scale_tail";',
         'if(!zero_selection_only&&a[0]>=limit)return "compensation_scale_tail";'),
        ('if(a[2]>=limit)return "compensation_xyz_tail";',
         'if(!zero_selection_only&&a[2]>=limit)return "compensation_xyz_tail";')],
      'src/gaussian_mapper.cpp':[
        ('readInt("LoDHealth.zero_patience",lod_health_guard_.zero_patience);',
         'int zero_only=0;readInt("LoDHealth.zero_selection_only",zero_only);\n'
         '  lod_health_guard_.zero_selection_only=zero_only!=0;\n'
         '  readInt("LoDHealth.zero_patience",lod_health_guard_.zero_patience);'),
        ('settings_file << "LoDHealth.zero_patience: " << lod_health_guard_.zero_patience',
         'settings_file << "LoDHealth.zero_selection_only: " << int(lod_health_guard_.zero_selection_only) << \'\\n\';\n'
         '  settings_file << "LoDHealth.zero_patience: " << lod_health_guard_.zero_patience')],
      'tests/lod_health_guard_test.cpp':[
        ('  auto differentiable=torch::ones({100},options.requires_grad(true));',
         '  LodHealthGuard only;only.zero_selection_only=true;only.zero_patience=2;only.saturation_patience=1;\n'
         '  TORCH_CHECK(only.compensation(xyz,large,op,xyz,scale,ref).empty());\n'
         '  TORCH_CHECK(only.compensation(displaced,scale,op,xyz,scale,ref).empty());\n'
         '  TORCH_CHECK(only.compensation(xyz,scale,op,xyz,torch::zeros_like(scale),ref)=="nonfinite_compensation");\n'
         '  TORCH_CHECK(only.selection(0,1,.01,100,10,high,eligible).empty());\n'
         '  TORCH_CHECK(only.selection(0,2,.01,100,1,low,eligible).empty());\n'
         '  TORCH_CHECK(only.selection(0,3,.01,100,0,low,eligible).empty());\n'
         '  TORCH_CHECK(only.selection(0,4,.01,100,0,low,eligible)=="consecutive_zero_selection_with_candidates");\n'
         '  TORCH_CHECK(only.selection(0,5,.01,100,1,bad,eligible)=="nonfinite_selector_probability");\n'
         '  TORCH_CHECK(!only.sample["scale_factor_max"].isNull()||!only.sample["selected"].isNull());\n'
         '  auto differentiable=torch::ones({100},options.requires_grad(true));')],
      'tests/anchor_selector/joint_view_conditioning_test.cpp':[
        ('  void verify(bool empty,bool isolation=false) {',
         '  void verifyD() {\n'
         '    TORCH_CHECK(selector_gi_loss_weight_==0&&view_budget_li_weight_==0&&flexgs_importance_enabled_,"D supervision switches");\n'
         '    TORCH_CHECK(lod_health_guard_.zero_selection_only&&view_budget_health_.zero_selection_only,"D health switches");\n'
         '    verify(false);\n'
         '  }\n'
         '  void verify(bool empty,bool isolation=false) {'),
        ('  JointViewMapperTest mapper(out,empty?mode.substr(0,mode.size()-6):mode);mapper.verify(empty,isolation);',
         '  JointViewMapperTest mapper(out,empty?mode.substr(0,mode.size()-6):mode);\n'
         '  if(mode=="view_d")mapper.verifyD();else mapper.verify(empty,isolation);')]
    }
    changes={};diff=[]
    for name,edits in patches.items():
        p=R/'source'/name;before=p.read_text();after=before
        for a,b in edits:after=replace(after,a,b)
        p.write_text(after);changes[name]=dict(before=sha(B/'source'/name),after=sha(p))
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),'A/'+name,'D/'+name))
    (R/'reports/D_SOURCE_CHANGES.diff').write_text(''.join(diff));write(R/'reports/SOURCE_CHANGES.json',changes)


def main():
    assert R==Path('/home/prmi/zwj/experiments/no_importance_all_scenes_20261115')
    assert not (R/'PREPARED.json').exists() and shutil.disk_usage(R).free>10*2**30
    for n in ['configs','runs','validation','objects','evaluation','reports','logs']:(R/n).mkdir(exist_ok=True)
    source=read(B/'reports/SOURCE_HASHES.json');assert source==read(V/'reports/SOURCE_HASHES.json')
    protected={}
    # Protect current production source/config/runtime and all historical batch inputs.
    for base in [Path('/home/prmi/zwj/mycartgs'),Path('/home/prmi/zwj/realLOD')]:
        for folder in ['src','include','improvements','cfg','python','bin','lib']:
            for p in (base/folder).rglob('*'):
                if p.is_file() and p.suffix not in ['.pyc'] and '__pycache__' not in p.parts:
                    protected[str(p)]=sha(p)
        for name in ['CMakeLists.txt','DEPLOYED_VERSION.json']:
            p=base/name
            if p.is_file():protected[str(p)]=sha(p)
    for n,h in source.items():
        p=B/'source'/n;assert sha(p)==h,n
        dest=R/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        protected[str(p)]=h
    for name in ['ORB-SLAM3','tinyply','simple-knn']:
        p=R/'source/third_party'/name;p.parent.mkdir(exist_ok=True)
        if not p.exists():p.symlink_to((B/'source/third_party'/name).resolve(),target_is_directory=True)
    shutil.copytree(B/'evaluation_runtime',R/'evaluation_runtime',symlinks=True)
    shutil.copytree(B/'runtime/lib',R/'runtime/lib',symlinks=True)
    (R/'runtime/bin').mkdir()
    for p in (B/'runtime').rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    # Keep original runner as a library for the existing manifest/renderer protocol.
    for n in ['pipeline.py','oom_retry.py','lod_eval_entry.py','verify_initialization.py']:
        shutil.copy2(B/n,R/n);protected[str(B/n)]=sha(B/n)
    text=(B/'eval_joint_entry.py').read_text()
    text=replace(text,"li_weight=.1)","li_weight=0.,gi_loss_weight=0.)")
    (R/'eval_joint_entry.py').write_text(text)
    text=(B/'configs/dual_view_li.yaml').read_text()
    text=replace(text,'Selector.gi_loss_weight: 1.0','Selector.gi_loss_weight: 0.0')
    text=replace(text,'ViewBudget.li_weight: 0.1','ViewBudget.li_weight: 0.0')
    text+='\n# D组：仅关闭重要性监督；长尾与饱和只记录，连续零选点和非有限值仍停止。\nLoDHealth.zero_selection_only: 1\n'
    (R/'configs/no_importance.yaml').write_text(text)
    (R/'source/cfg/selector/spatial_offset3_residual48_joint_view_d.yaml').write_text(text)
    jobs=[]
    for batch in [B,V]:
        state=read(batch/'LOD_PIPELINE.json');assert state['phase']=='finished'
        for n in ['LOD_PIPELINE.json','configs/dual_view_li.yaml','reports/LOD_METRICS.csv']:
            protected[str(batch/n)]=sha(batch/n)
        for old in state['jobs']:
            assert old['state']=='complete' and len(old['evaluations'])==23
            keys=['id','dataset','scene','mode','winner','full_psnr','full_job','full_evaluation','target_steps','camera_config','mapping_config']
            j={k:copy.deepcopy(old[k]) for k in keys};j.update(state='pending',attempt_history=[],evaluations=[],oom_failures=0,
                reference_model=old['trained_model'],reference_batch=str(batch),reference_evaluations=copy.deepcopy(old['evaluations']))
            for key in ['camera_config','mapping_config']:
                src=Path(j[key]);dest=R/'configs'/j['id']/(key+'.yaml');dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dest);j[key]=str(dest);protected[str(src)]=sha(src)
            full=Path(j['full_job']['result']['checkpoint']);assert read(full/'real_lod_state.json')['lod_steps']==0
            for folder in [full,Path(j['reference_model'])]:
                for p in folder.rglob('*'):
                    if p.is_file():protected[str(p)]=sha(p)
            for n in ['CameraTrajectory_TUM.txt','TrackingTime.txt']:
                p=Path(j['full_evaluation']['model'])/n;protected[str(p)]=sha(p)
            assert j['target_steps']==j['full_job']['result']['online_steps']
            jobs.append(j)
    assert len(jobs)==32 and len({j['id'] for j in jobs})==32
    assert collections.Counter(j['dataset'] for j in jobs)==dict(Replica=16,TUM_RGBD=10,VECtor=6)
    # Start with small TUM cases; each job retains its original Full winner and step budget.
    jobs.sort(key=lambda j:(0 if j['id']=='TUM_RGBD/freiburg1_xyz/rgbd' else 1,j['target_steps']))
    patch_sources()
    # Source hash inventory from the original allowlist, plus the one D test config.
    current={n:sha(R/'source'/n) for n in source};current['cfg/selector/spatial_offset3_residual48_joint_view_d.yaml']=sha(R/'source/cfg/selector/spatial_offset3_residual48_joint_view_d.yaml')
    write(R/'reports/SOURCE_HASHES.json',current);write(R/'PROTECTED_BEFORE.json',protected)
    write(R/'LOD_PIPELINE.json',dict(phase='prepared',jobs=jobs,validated_modes=[],active=None,counts=dict(pending=32),protocol=dict(
        seed=0,gi_loss_weight=0.,li_weight=0.,importance_collection_retained=True,conditioning='both_real_view',
        master_updates=True,topology_frozen=True,train_ratios=[.01,.05,.1,.3,.5,.7],unseen_ratios=[.03,.08,.2,.4,.6],
        health='only consecutive zero with eligible candidates; finite checks retained; tails/saturation logged only',
        zero_patience=8,max_oom_retries=2,health_retries=0,frames='same complete estimated trajectory as A',
        evaluation_protocol='cartgs_original_rgb_full_trajectory',saved_images=0)))
    write(R/'PREPARED.json',dict(passed=True,jobs=32,total_fine_steps=sum(j['target_steps'] for j in jobs),
        protected_files=len(protected),source_files=len(current),source_origin=str(B),created_unix=time.time()))
    print(json.dumps(read(R/'PREPARED.json')),flush=True)


if __name__=='__main__':main()
