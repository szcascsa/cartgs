"""fr1_xyz 零调参冻结输入；复用上一轮相机转换、worker、renderer与ranking数学。"""
from pathlib import Path
import hashlib,json,shutil,subprocess,importlib.util
ROOT=Path(__file__).resolve().parent
PREV=Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019')
SPARSE=Path('/home/prmi/zwj/sparse-hash-root-grid-20261019')
REPO=Path('/home/prmi/zwj/mycartgs')
CHECKPOINT=REPO/'resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_xyz/full/attempt_1/model/9900_shutdown/ply'
DATASET=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
ROOT.mkdir(exist_ok=True)
for name in ['common.py','evaluate.py','progressive_ranker.py','experiment_data.py','fixed_marginal_reference.py','test_progressive.py','export_tree.py','native.cpp','audit_worker']:
    shutil.copy2(PREV/name,ROOT/name)
shutil.copytree(PREV/'runtime',ROOT/'runtime',dirs_exist_ok=True)
shutil.copy2('/home/prmi/zwj/second-scene-strong-go-20261018/camera_prepare_source.py',ROOT/'camera_prepare_source.py')
prior=read(PREV/'preregistration.json')
save('preregistration.json',dict(dataset=str(DATASET),checkpoint=str(CHECKPOINT),checkpoint_rule='existing full final shutdown, highest final iteration; no quality selection',
    arms=['A_Raw','B_FixedMarginal','C_Hierarchical','F3_ProgressiveMarginal'],ratios=prior['ratios'],K=prior['K'],candidate=prior['candidate'],maturity=1,
    frozen_tree=prior['frozen_tree'],progressive_semantics=prior['progressive_semantics'],heap_tie=prior['heap_tie'],local_tie=prior['local_tie'],
    geometry_diagnostic=prior['geometry_diagnostic'],F3_gate=prior['F3_gate'],F3_strong=prior['F3_strong'],Q5_Fixed_comparison=prior['Q5_Fixed_comparison'],
    evaluation='all archived KF excluding union of actual GlobalGI anchor/representative Bank; fixed before quality; original SLAM views, not unseen',
    C_vs_A_primary='at least3 ratios deltaPSNR>=-.02; low mean deltaPSNR>.10 OR one>.20 and other>=-.02; systematic SSIM/LPIPS tradeoff downgrades interpretation',
    runtime_gate=False,parameter_search=False,optimizer_updates=0,production_integration=False,commit=False,push=False))
(ROOT/'protocol.json').write_bytes((ROOT/'preregistration.json').read_bytes())
files=[CHECKPOINT/'cameras.json',CHECKPOINT/'view_bank_orb.json',CHECKPOINT/'transform_field.pt',CHECKPOINT/'selector_config.yaml']
files+=list((CHECKPOINT/'point_cloud').glob('*/point_cloud.ply'))
assert len(files)==5
cams=sorted(read(CHECKPOINT/'cameras.json'),key=lambda a:a['id']);ids=[a['id']for a in cams]
files += [DATASET/a['img_name']for a in cams]
files += [PREV/'runtime/lib/libgaussian_mapper.so',PREV/'runtime/lib/libcuda_rasterizer.so',PREV/'audit_worker',SPARSE/'build/libhierarchy.a']
files += [SPARSE/'source'/n for n in ['include/hierarchical_representative_si.h','src/hierarchical_representative_si.cpp','include/sparse_hash_root_grid.h','src/sparse_hash_root_grid.cpp']]
files += [PREV/n for n in ['progressive_ranker.py','fixed_marginal_reference.py','common.py','evaluate.py']]
save('frozen_sources.json',{str(p):dict(sha256=sha(p),bytes=p.stat().st_size)for p in files})
# 原相机转换函数逐字复用，只有 dataset/checkpoint/output 路径不同。
spec=importlib.util.spec_from_file_location('camera_source',ROOT/'camera_prepare_source.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=ROOT
info=module.scene('fr1_xyz',CHECKPOINT,DATASET)
bank=read(CHECKPOINT/'view_bank_orb.json');bank_ids=sorted(set(sum([a['keyframes']for a in bank['banks']if a['name']in ['GlobalGI anchor','GlobalGI representative']],[])))
assert bank_ids and set(bank_ids).issubset(ids)
manifest={k:info[k]for k in ['master','tf','config','cameras','dataset','camera_ids']}
manifest.update(bank_ids=bank_ids,bank_frames=[ids.index(i)for i in bank_ids],evaluation_frames=[i for i,k in enumerate(ids)if k not in set(bank_ids)])
assert set(manifest['bank_frames']).isdisjoint(manifest['evaluation_frames'])and len(manifest['evaluation_frames'])>=8
save('input_manifest.json',manifest)
save('camera_checks.json',dict(cameras=len(ids),Bank=len(bank_ids),nonBank=len(manifest['evaluation_frames']),Bank_nonBank_disjoint=True,
    all_archived_KF_used=True,requested_dataset_used=True,image_hashes=info['image_hashes'],camera_archive_sha256=sha(info['cameras']),
    source_scene_label=bank.get('scene'),source_scene_label_is_stale_metadata_only=True,dataset_identity='camera img_name files loaded from explicit requested dataset; Bank uses only keyframe IDs'))
save('reused_source_checks.json',{n:dict(previous_sha256=sha(PREV/n),current_sha256=sha(ROOT/n),exact=(ROOT/n).read_bytes()==(PREV/n).read_bytes())for n in ['progressive_ranker.py','fixed_marginal_reference.py','common.py','evaluate.py','audit_worker']})
print('FROZEN',len(ids),'CAMERAS',len(bank_ids),'BANK',len(manifest['evaluation_frames']),'EVAL',flush=True)
