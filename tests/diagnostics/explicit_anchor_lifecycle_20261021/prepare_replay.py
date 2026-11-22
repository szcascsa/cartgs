"""仅在服务器隔离副本插入生命周期记录；不修改生产或原实验源码。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

original = Path('/home/prmi/zwj/octree-explicit-20261021')
root = Path('/home/prmi/zwj/octree-explicit-diagnostics-20261021')
root.mkdir(exist_ok=False)
repo = root/'source'
shutil.copytree(original/'source', repo, symlinks=True)
out = root/'evidence'
out.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((original/'validation/source_manifest.json').read_text())
assert all(sha(original/'source'/p) == h for p,h in manifest.items())
plan = json.loads((original/'validation/evaluation_plan.json').read_text())
model_hashes = {p:h for p,h in plan['model_sha256'].items() if '/explicit_seed' in p}
assert all(sha(p)==h for p,h in model_hashes.items())
(out/'original_source_manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'original_model_manifest.json').write_text(json.dumps(model_hashes,indent=2))
(out/'protocol.json').write_text(json.dumps(dict(
    scene='fr1_xyz',seeds=[2026,2027],mapping_steps=10000,
    diagnostic_replay=True,algorithm_changes=False,
    fast_death_mapping_thresholds=[1,10,50,100,250,500,1000],
    contribution_views='64 uniformly spaced indices from original 792 common views',
    real_observations='distinct XYZ in birth observation batch; growth sources are NOT real observations',
    lifecycle_clock='GaussianModel protection.mapping_iteration; initialization=0',
    contribution='d(sum first-channel unclamped render)/d(precomputed first-channel color)=sum_pixels(alpha*T)',
    pruning='original per-member opacity<0.02; whole block only when all ten fail'),indent=2))

def replace(path, old, new):
    p=repo/path
    s=p.read_text()
    assert s.count(old)==1,(path,s.count(old),old[:90])
    p.write_text(s.replace(old,new))

replace('improvements/selector/anchor_bank.h','  void enableExplicitBlocks();',
        '  int64_t diagnostic_step = 0;\n  bool diagnostic_initial = false;\n  void enableExplicitBlocks();')
replace('improvements/selector/explicit_anchor_blocks.h','#include <set>',
'''#include <set>
#include <fstream>
#include <cstdlib>

inline std::ofstream& anchorDiagnosticTrace() {
  static std::ofstream stream(std::getenv("CARTGS_ANCHOR_TRACE") ? std::getenv("CARTGS_ANCHOR_TRACE") : "/dev/null");
  return stream;
}''')
replace('improvements/selector/explicit_anchor_blocks.h','    std::vector<int64_t> chosen;',
'''    if(std::getenv("CARTGS_ANCHOR_TRACE")) {
      bool has_coarse=false,has_same=false;
      if(inherit) for(auto i:entry.second) {
        if(old_levels.accessor<int64_t,1>()[i/kGaussiansPerBlock]==key[0]) has_same=true;
        else has_coarse=true;
      }
      const char* source=!inherit ? (diagnostic_initial?"initial_observation":"online_observation") :
          (has_coarse?(has_same?"growth_mixed":"growth_fine"):"growth_same");
      anchorDiagnosticTrace()<<"B,"<<diagnostic_step<<','<<(next_uid_.item<int64_t>()+j)<<','<<key[0]<<','
          <<key[1]<<','<<key[2]<<','<<key[3]<<','<<source<<','<<entry.second.size()<<','<<distinct.size()<<'\\n';
    }
    std::vector<int64_t> chosen;''')
replace('improvements/selector/explicit_anchor_blocks.h','  const auto removed=size()-ids.numel();',
'''  const auto removed=size()-ids.numel();
  if(std::getenv("CARTGS_ANCHOR_TRACE")) {
    auto dead=(~keep.select(1,0)).cpu().contiguous();auto uid=block_uids_.cpu().contiguous();
    for(int64_t j=0;j<size();++j) if(dead.accessor<bool,1>()[j])
      anchorDiagnosticTrace()<<"D,"<<diagnostic_step<<','<<uid.accessor<int64_t,1>()[j]<<'\\n';
    anchorDiagnosticTrace().flush();
  }''')
replace('src/gaussian_model.cpp','    auto seeds=explicit_anchors_->observeBlocks(fused_point_cloud,color,',
'''    explicit_anchors_->diagnostic_initial=true;explicit_anchors_->diagnostic_step=0;
    auto seeds=explicit_anchors_->observeBlocks(fused_point_cloud,color,''')
replace('src/gaussian_model.cpp','    fused_point_cloud=seeds.xyz;color=seeds.color;block_sigma=seeds.sigma;',
'''    explicit_anchors_->diagnostic_initial=false;
    fused_point_cloud=seeds.xyz;color=seeds.color;block_sigma=seeds.sigma;''')
replace('src/gaussian_model.cpp','    appendAnchorSeeds(explicit_anchors_->observeBlocks(new_point_cloud,new_colors,',
'''    explicit_anchors_->diagnostic_step=protection_.mapping_iteration;
    appendAnchorSeeds(explicit_anchors_->observeBlocks(new_point_cloud,new_colors,''')
replace('src/gaussian_model.cpp','      appendAnchorSeeds(explicit_anchors_->growBlocks(xyz_,color,grads,denom_,anchor_growth_threshold_,2,',
'''      explicit_anchors_->diagnostic_step=protection_.mapping_iteration;
      appendAnchorSeeds(explicit_anchors_->growBlocks(xyz_,color,grads,denom_,anchor_growth_threshold_,2,''')
replace('src/gaussian_model.cpp','  if(explicit_anchors_) mask=explicit_anchors_->wholeBlockPrune(mask);',
'''  if(explicit_anchors_) {
    explicit_anchors_->diagnostic_step=protection_.mapping_iteration;
    if(std::getenv("CARTGS_ANCHOR_TRACE")) {
      auto uid=explicit_anchors_->blockUIDs().cpu().contiguous();
      auto flags=mask.cpu().contiguous();auto opacity=getOpacityActivation().detach().cpu().contiguous().reshape({-1});
      for(int64_t j=0;j<uid.numel();++j) {
        int failures=0;for(int k=0;k<10;++k) failures+=flags.accessor<bool,1>()[j*10+k];
        anchorDiagnosticTrace()<<"P,"<<protection_.mapping_iteration<<','<<uid.accessor<int64_t,1>()[j]<<','<<failures;
        for(int k=0;k<10;++k) anchorDiagnosticTrace()<<','<<opacity.accessor<float,1>()[j*10+k];
        anchorDiagnosticTrace()<<'\\n';
      }
    }
    mask=explicit_anchors_->wholeBlockPrune(mask);
  }''')

changed=['improvements/selector/anchor_bank.h','improvements/selector/explicit_anchor_blocks.h','src/gaussian_model.cpp']
(out/'instrumented_source_manifest.json').write_text(json.dumps({p:sha(repo/p) for p in manifest},indent=2))
for p in changed:
    target=out/'instrumented_source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/p,target)
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
         OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg')
def state(stage,**kw): (out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
state('BUILDING')
cmake='/home/prmi/.local/bin/cmake'
cmd=[cmake,'-S',str(repo),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86',
     '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
     '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
     '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
     '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
     '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake',
     '-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON']
with (out/'build.log').open('w') as f:
    for command in [cmd,[cmake,'--build',str(root/'build'),'--target','tum_rgbd','explicit_anchor_test','-j2']]:
        subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for device in ['cpu','cuda']:
    d=out/device;d.mkdir()
    with (out/('correctness_'+device+'.log')).open('w') as f:
        subprocess.run([str(repo/'bin/explicit_anchor_test'),device],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
jobs=[]
for seed in (2026,2027):
    run=root/f'diagnostic_seed{seed}';run.mkdir()
    cfg=original/f'explicit_seed{seed}/input_selector.yaml'
    shutil.copy2(cfg,run/'input_selector.yaml')
    cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
         repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
         '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
         repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',run,'no_viewer','--selector-config',run/'input_selector.yaml']
    state('DIAGNOSTIC_REPLAY',seed=seed)
    start=time.monotonic()
    with (run/'training.log').open('w') as f:
        subprocess.run(list(map(str,cmd)),cwd=repo,env=dict(env,CARTGS_AUDIT_SEED=str(seed),
            CARTGS_ANCHOR_TRACE=str(run/'lifecycle_trace.csv')),stdout=f,stderr=subprocess.STDOUT,check=True)
    assert '10000 iterations completed.' in (run/'training.log').read_text()
    jobs.append(dict(seed=seed,wall_seconds=time.monotonic()-start,output=str(run)))
    (out/'training_jobs.json').write_text(json.dumps(jobs,indent=2))
assert all(sha(p)==h for p,h in model_hashes.items())
assert all(sha(original/'source'/p)==h for p,h in manifest.items())
state('COMPLETED',original_models_unchanged=True,original_source_unchanged=True)
