"""建立独立评价快照，冻结32组CaRtGS源模型与已有方法结果。"""
from pathlib import Path
import json
import shutil

from common import ROOT, SAVED, RATIOS, METHODS, load, write, sha


def main():
    assert not (ROOT/'EXPERIMENTS.json').exists()
    source = SAVED/'vanilla_all_datasets_20261031_2011/runtime/python'
    destination = ROOT/'runtime/python'
    destination.mkdir(parents=True,exist_ok=True)
    copied = {}
    for path in source.rglob('*'):
        if not path.is_file() or path.suffix not in ['.py','.so']:
            continue
        if any(part in ['__pycache__','build','.git'] for part in path.relative_to(source).parts):
            continue
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        assert sha(path) == sha(target)
        copied[str(path)] = dict(copy=str(target),sha256=sha(path))
    group_path = SAVED/'all_scene_quality_figures_20261108/data/ALL_SCENE_RESULTS.json'
    bundle = load(group_path)
    rerun_path = SAVED/'dual_view_li_vector_stereo_rerun_20261111/LOD_PIPELINE.json'
    rerun = {job['id']:job for job in load(rerun_path)['jobs']}
    groups = []
    protected = {str(group_path):sha(group_path),str(rerun_path):sha(rerun_path)}
    for group in bundle['groups']:
        vanilla = group['vanilla']
        model = Path(vanilla['checkpoint'])
        evaluation = Path(vanilla['evaluation'])
        request_path = evaluation.parents[1]/'process/request.json'
        request = load(request_path)['command']
        assert '--no-save-images' in request and '--selector' not in request
        data = request[5]
        cloud, = (model/'ply/point_cloud').glob('iteration_*/point_cloud.ply')
        references = group['ours']
        ours_source = 'original_32_group_batch'
        if group['id'] in rerun:
            references = [dict(item['result'],scope=item['scope'],ratio=item['ratio'],
                              ratio_kind=item['kind']) for item in rerun[group['id']]['evaluations']]
            ours_source = 'stereo_rerun_20261111'
        group_config = dict(id=group['id'],dataset=group['dataset'],scene=group['scene'],mode=group['mode'],
                            vanilla=vanilla,data=data,ply=str(cloud),
                            correct_scale='--correct_scale' in request,
                            ours=references,ours_source=ours_source,
                            evaluations=22,state='pending',attempts=[])
        assert sorted(str(float(item['ratio'])) for item in references if item['scope']=='global') == sorted(str(float(r)) for r in RATIOS)
        data_inputs=[Path(data)/'pose_TUM.txt'] if group['dataset']=='Replica' else [Path(data)/'rgb.txt',Path(data)/'groundtruth.txt']
        for path in [model/'ply/cameras.json',model/'ply/cfg_args',cloud,
                     model.parent/'CameraTrajectory_TUM.txt',model.parent/'TrackingTime.txt',
                     evaluation/'evaluation_frames.json',request_path,
                     evaluation/'psnr.txt',evaluation/'ssim.txt',evaluation/'lpips.txt']+data_inputs:
            assert path.is_file(), path
            protected[str(path)] = sha(path)
        assert (Path(data)/'groundtruth.txt').is_file() or (Path(data)/'pose_TUM.txt').is_file()
        groups.append(group_config)
    # 先完成最常用的fr1_xyz，然后双目复跑场景，再按场景顺序执行其余组。
    groups.sort(key=lambda g:(0 if g['scene']=='freiburg1_xyz' else
                              1 if g['dataset']=='VECtor' else 2,g['id']))
    assert len(groups)==32 and len(RATIOS)*len(METHODS)*len(groups)==704
    write(ROOT/'EXPERIMENTS.json',dict(phase='prepared',ratios=RATIOS,methods=METHODS,seed=0,groups=groups))
    write(ROOT/'PROTECTED_BEFORE.json',protected)
    write(ROOT/'SOURCE_MANIFEST.json',copied)
    text = '''# CaRtGS随机裁剪与Opacity裁剪对照协议

使用32组既有CaRtGS Vanilla的最高Full全轨迹PSNR重复，其余指标及模型绑定同一次。
不重新训练，不改变模型属性，不使用Selector、Compensator、视角候选或新生点保护。

- Ratio：0.01、0.03、0.05、0.08、0.1、0.2、0.3、0.4、0.5、0.6、0.7，共704项。
- 预算：K=floor(r×N_CaRtGS)，Decimal计算避免十进制浮点边界误差。严格保留K个点。
- 随机：每场景seed=0，NumPy PCG64生成一次排列，各Ratio取前K个，属于嵌套随机子集。
- Opacity：复用模型activated opacity，在CPU稳定降序排序；同值按原Gaussian行号。
- 两种方法均将选中行号恢复原顺序再渲染，避免改变像素合成的同深度顺序。
- 评价：逐字复用冻结的CaRtGS run.py尾部，包括Full轨迹、原图、TorchMetrics、逐帧平均。
- 单目保持原correct_scale设置；RGB-D／双目沿用原关闭标记，保留原evo前置对齐逻辑。
- 独立复制评价源码和CUDA扩展；同场景复用冻结Master及有界CPU原图缓存，最多6GiB。
- 每组先核验Full首／中／末帧指标与历史逐帧数值，并核验帧关联和源PLY点数。
- 各自模型总点数不同，因此这是相同比例比较，实际K分别记录，不能称同K对照。
- 随机仅一次seed0，不把单次随机差异解释为统计显著性。
- 我们方法复用原32组已评价结果，其中corner-slow／robot-normal双目采用11月11日最新复跑。
- 不保存渲染／GT图片，不写生产项目或旧实验，保留选点ID、逐帧数值及来源哈希。
- OOM最多额外重试2次；非OOM先保留现场并报错。磁盘不足5GiB暂停，不清理旧数据。
'''
    (ROOT/'PROTOCOL.md').write_text(text)
    print(json.dumps(dict(prepared=True,groups=32,evaluations=704,source_files=len(copied),
                         protected_files=len(protected),free_gib=shutil.disk_usage(ROOT).free/2**30)))


if __name__=='__main__':main()
