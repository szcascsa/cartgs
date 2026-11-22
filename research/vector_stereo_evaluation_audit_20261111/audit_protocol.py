"""只读核查两组VECtor双目的CaRtGS与本次复跑评价口径。"""
from pathlib import Path
import ast
import hashlib
import json

SAVED = Path('/home/prmi/zwj/realLOD/saved_experiments')
VANILLA = SAVED/'vanilla_all_datasets_20261031_2011'
RERUN = SAVED/'dual_view_li_vector_stereo_rerun_20261111'
OUT = Path(__file__).resolve().parent
INPUTS = {}


def read(path):
    data = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8')


def load(path):
    return json.loads(read(path))


def fingerprint(node):
    return ast.dump(node, include_attributes=False)


def definitions(tree):
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def assignments(tree, names):
    return {name: [fingerprint(node) for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == name for target in node.targets)]
            for name in names}


def main():
    sources = [VANILLA/'runtime/python/run.py', RERUN/'source/python/run_metrics_only.py']
    texts = [read(path) for path in sources]
    trees = [ast.parse(text) for text in texts]
    shared = {}
    for name in ['associate_frames', 'loadTUM']:
        shared[name] = fingerprint(definitions(trees[0])[name]) == fingerprint(definitions(trees[1])[name])
    names = ['calc_psnr', 'calc_ssim', 'calc_lpips', 'gt_image_torch', 'render_image_torch',
             'val_psnr', 'val_ssim', 'val_lpips', 'bg_color', 'background']
    for name in names:
        a, b = [assignments(tree, [name])[name] for tree in trees]
        shared[name] = bool(a) and a == b
    for file, names in [('utils.py', ['MiniCam', 'focal2fov']),
                        ('renderer.py', ['_make_rasterizer', 'prepare_render_data'])]:
        a, b = [definitions(ast.parse(read(base/file)))
                for base in [VANILLA/'runtime/python', RERUN/'source/python']]
        for name in names:
            shared[name] = fingerprint(a[name]) == fingerprint(b[name])
    assert all(shared.values()), shared
    assert all('traj_ref.align(traj_est, True)' in text for text in texts)
    assert all('np.mean(psnr_list)' in text and 'np.mean(ssim_list)' in text
               and 'np.mean(lpips_list)' in text for text in texts)
    groups = load(SAVED/'all_scene_quality_figures_20261108/data/ALL_SCENE_RESULTS.json')['groups']
    jobs = load(RERUN/'LOD_PIPELINE.json')['jobs']
    results = []
    for job in jobs:
        baseline = next(group for group in groups if group['id'] == job['id'])['vanilla']
        old_output = Path(baseline['evaluation'])
        old_request = load(old_output.parents[1]/'process/request.json')
        old_frames = load(old_output/'evaluation_frames.json')
        assert '--correct_scale' not in old_request['command']
        old_camera = load(Path(baseline['checkpoint'])/'ply/cameras.json')[0]
        expected = {key: old_camera[key] for key in ['width', 'height', 'fx', 'fy']}
        full_result = next(item['result'] for item in job['evaluations'] if item['scope'] == 'full')
        full_command = load(Path(full_result['output'])/'launch.json')['command']
        shutdown = full_command[full_command.index('--shutdown-model')+1]
        new_camera = load(Path(job['trained_model'])/shutdown/'ply/cameras.json')[0]
        actual = {key: new_camera[key] for key in expected}
        assert expected == actual
        same_files = None
        for item in job['evaluations']:
            output = Path(item['result']['output'])
            command = load(output/'launch.json')['command']
            assert not any(flag in command for flag in
                           ['--correct_scale', '--stereo-camera-config', '--stereo-native-maps',
                            '--evaluation-frame-set'])
            assert load(output/'image_preprocessing.json') == {'protocol': 'original_rgb'}
            frames = load(output/'evaluation_frames.json')
            assert old_frames['gt_indices'] == frames['gt_indices']
            if same_files is None:
                same_files = sum(Path(a).samefile(b) for a, b in zip(old_frames['rgb_paths'], frames['rgb_paths']))
                assert same_files == len(frames['rgb_paths']) == len(old_frames['rgb_paths'])
                reference_paths = frames['rgb_paths']
            else:
                assert frames['rgb_paths'] == reference_paths
        for name in ['rgb.txt', 'groundtruth.txt']:
            old_data = Path(old_request['command'][5])/name
            new_data = Path(full_command[4])/name
            old_rows, new_rows = [
                [line.split() for line in read(path).splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
                for path in [old_data, new_data]]
            assert old_rows == new_rows, name
        values = {}
        for name in ['psnr', 'ssim', 'lpips']:
            raw = [float(value) for value in read(old_output/(name+'.txt')).splitlines() if value.strip()]
            assert len(raw) == same_files
            assert abs(sum(raw)/len(raw)-baseline[name]) < 1e-9
            values[name] = {'cartgs': baseline[name], 'ours_full': full_result[name]}
        results.append(dict(id=job['id'],same_rgb_files=same_files,frames_equal=True,
                            evaluation_records_verified=len(job['evaluations']),intrinsics=actual,
                            vanilla_evaluation=str(old_output),ours_full_evaluation=full_result['output'],
                            vanilla_gaussians=baseline['gaussians'],ours_gaussians=full_result['gaussian_count'],
                            values=values))
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest for name,digest in INPUTS.items())
    receipt = dict(protocol_equal=True,scenes=results,source_checks=shared,
                   protected_inputs_unchanged=True,input_sha256=INPUTS,
                   caveats=['各自估计轨迹，未用同一组GT相机隔离轨迹误差',
                            '统一原图域不等于消除训练校正域与原图之间的几何错位',
                            'ATE两边均继承traj_ref.align(traj_est, True)，随后correct_scale=False；不可简称完全没有尺度变换',
                            '核查源码和历史命令，未重新渲染验证两套CUDA二进制逐位相同'])
    (OUT/'AUDIT_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    lines = ['# VECtor双目评价口径核查','',
             '结论：corner-slow与robot-normal的当前复跑结果和图中CaRtGS Vanilla使用相同的图像质量评价规则。'
             '本次只读检查源码、历史命令、相机、文件与数值，不重训或重渲染。','',
             '| 核查项 | 双方实际行为 |','|---|---|',
             '| 评价图像 | 同一批未校正左目原图，逐文件samefile核验 |',
             '| 数据索引与GT | rgb.txt有效数据行一致（Vanilla额外三行注释），groundtruth.txt内容一致 |',
             '| 评价范围 | GT同步后的完整估计轨迹，RGB最近时间戳关联，阈值0.08秒，无帧采样 |',
             '| 相机 | 1224×1024，fx=fy=887.3200073242188；MiniCam与FoV转换函数AST相同 |',
             '| PSNR／SSIM | 相同TorchMetrics构造和逐帧调用，data_range=1.0 |',
             '| LPIPS | 相同TorchMetrics构造与逐帧调用，normalize=True，未另设net_type |',
             '| 颜色与预处理 | BGR转RGB，GT除255，NCHW，预测clamp至[0,1]，不裁剪、不resize、不作有效像素遮罩 |',
             '| 汇总 | 各帧指标算术平均；同一黑背景；SH degree=3 |','',
             '| 场景 | 同一原图数量 | 我们核查评价项 | CaRtGS N | 我们N | CaRtGS Full PSNR | 我们Full PSNR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        lines.append(f"| {result['id']} | {result['same_rgb_files']} | {result['evaluation_records_verified']} | "
                     f"{result['vanilla_gaussians']} | {result['ours_gaussians']} | "
                     f"{result['values']['psnr']['cartgs']:.6f} | {result['values']['psnr']['ours_full']:.6f} |")
    lines += ['', '## 比较的限制',''] + [f'- {text}。' for text in receipt['caveats']]
    lines += ['', '历史校正图口径下，我们旧模型的Full PSNR为corner-slow 27.188997、robot-normal 25.783061 dB；'
              '同一旧模型改用原图评价后分别为19.720315、16.963335 dB。这两套数字不能混用。', '',
              '两边模型、位姿、Gaussian数量和训练量不同；评价口径相同不代表固定地图／固定相机／同K的受控算法消融。', '',
              '详细来源与输入哈希见AUDIT_VERIFIED.json；生产配置、原模型和历史指标均未修改。']
    (OUT/'EVALUATION_PROTOCOL_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(protocol_equal=True,scenes=results,source_checks=shared),ensure_ascii=False))


if __name__ == '__main__':
    main()
