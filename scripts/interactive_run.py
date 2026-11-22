#!/usr/bin/env python3
"""交互选择训练/渲染任务；只调用现有入口，不更改算法或默认配置。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DATASET_DEFAULTS = {
    'TUM_RGBD': ('tum_rgbd', 'RGB-D/TUM', 'tum_rgbd.yaml'),
    'Replica': ('replica_rgbd', 'RGB-D/Replica', 'replica_rgbd.yaml'),
    '3DLite': ('tum_rgbd', 'RGB-D/TUM', 'tum_rgbd.yaml'),
    'VECtor': ('vector_mono', 'Monocular/VECtor', 'VECtor.yaml'),
}


class Cancelled(Exception):
    pass


def ask(title, default=None):
    suffix = f' [{default}]' if default is not None else ''
    while True:
        value = input(title + suffix + '：').strip()
        if value.lower() in {'q', 'quit'}:
            raise Cancelled()
        if value or default is not None:
            return value or str(default)
        print('请输入内容；输入 q 返回主菜单。')


def menu(title, labels, default=1):
    print('\n' + title)
    for i, label in enumerate(labels, 1):
        print(f'  {i}. {label}')
    while True:
        value = ask('选择编号（q 返回主菜单）', default)
        if value.isdigit() and 1 <= int(value) <= len(labels):
            return int(value) - 1
        print('编号无效。')


def path_prompt(title, default=None, kind='file'):
    while True:
        path = Path(ask(title, default).strip('"\'')).expanduser()
        path = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
        if kind == 'new' and path.exists():
            print('目录已存在，请换一个新目录，避免覆盖旧结果。')
        elif kind == 'file' and not path.is_file():
            print('文件不存在。')
        elif kind == 'dir' and not path.is_dir():
            print('目录不存在。')
        else:
            return path


def choose_file(title, candidates, preferred=None):
    paths = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    paths.sort(key=lambda p: (p != preferred, str(p)))
    labels = [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in paths]
    index = menu(title, labels + ['手动输入文件路径'])
    return paths[index] if index < len(paths) else path_prompt(title)


def default_file(title, candidates):
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None:
        print(f'{title}没有匹配文件，请输入该数据集的正确文件。')
        path = path_prompt(title)
    print(f'  {title}：{path}')
    return path


def output_prompt(title, default):
    while True:
        base = path_prompt(title, default, kind='path')
        suffix = ask('实验说明后缀（可留空，例如 no_tail_anchor_prune；自动加在目录末尾）', '')
        if any(c in suffix for c in '/\\<>:"|?*') or any(ord(c) < 32 for c in suffix):
            print('说明不能包含路径分隔符或文件名特殊字符。')
            continue
        output = base.with_name(base.name + ('_' + suffix if suffix else ''))
        if output.exists():
            print('添加说明后的目录已存在，请重新选择。')
            continue
        print(f'最终输出目录：{output}')
        return output


def config_values(text):
    return dict(re.findall(r'^([A-Za-z_][\w.]*)\s*:\s*([^\n]*)', text, re.M))


def merge_config(text, overrides):
    for key, value in overrides.items():
        pattern = re.compile(r'^' + re.escape(key) + r'\s*:[^\n]*', re.M)
        replacement = f'{key}: {value}'
        if pattern.search(text):
            text = pattern.sub(lambda _: replacement, text)
        else:
            text = text.rstrip() + '\n' + replacement + '\n'
    return text


def budget_values(text):
    values = config_values(text)
    result = {}
    for key, default, minimum, maximum in [
        ('Training.force_tail_optimization_enabled', 0, 0, 1),
        ('Training.tail_optimization_steps', -1, -1, 2147483647),
        ('Selector.max_mapping_iterations', 0, 0, 2147483647),
        ('Selector.anchor_only_pruning', 0, 0, 4),
    ]:
        raw = values.get(key, str(default)).split('#', 1)[0].strip()
        if (not re.fullmatch(r'[+-]?\d+', raw) or not minimum <= int(raw) <= maximum or
                (key == 'Selector.anchor_only_pruning' and int(raw) not in (0, 1, 3, 4))):
            if key == 'Selector.anchor_only_pruning':
                raise ValueError(f'{key}: {raw} 无效；只能为 0/1/3/4，3 固定十成员且关闭 slot，4 固定十成员且开启 slot。')
            hint = ('这是开关，只能为 0/1；步数应填 Training.tail_optimization_steps（尾部）'
                    '或 Selector.max_mapping_iterations（总 Mapping）' if maximum == 1 else
                    f'必须为 {minimum} 至 {maximum} 的整数')
            raise ValueError(f'{key}: {raw} 无效；{hint}。')
        result[key] = int(raw)
    return result


def integer_override(title, key, values, overrides, minimum=0, maximum=1, allowed=None):
    current = values.get(key, '').split('#', 1)[0].strip()
    while True:
        raw = ask(f'{title}（回车保持基底 {current or "默认值"}）', '')
        if not raw:
            candidate = current
        else:
            candidate = raw
        if (re.fullmatch(r'[+-]?\d+', candidate) and minimum <= int(candidate) <= maximum and
                (allowed is None or int(candidate) in allowed)):
            if raw:
                overrides[key] = str(int(raw))
            return int(candidate)
        if allowed is not None:
            print('请输入 ' + '/'.join(map(str, allowed)) + '。此项不是菜单编号。')
        else:
            print(f'请输入 {minimum} 至 {maximum} 的整数。此项不是菜单编号。')


def print_budget(text):
    values = budget_values(text)
    enabled, steps, total = [values[k] for k in ['Training.force_tail_optimization_enabled',
                                               'Training.tail_optimization_steps', 'Selector.max_mapping_iterations']]
    if config_values(text).get('Training.real_lod', '0').split('#', 1)[0].strip() == '1':
        tail = ('floor(SLAM结束时迭代数 / 2)' if steps == -1 else str(steps)) if enabled else '原正常尾部停止规则'
        print(f'训练步数规则：在线仅 Full → Full 尾优化（{tail}）→ LoD 优化，LoD 步数等于实际在线 Mapping。')
        config = config_values(text)
        frozen = config.get('Selector.detach_gaussian_parameters', '0').split('#', 1)[0].strip() == '1'
        full_weight = float(config.get('RealLoD.full_loss_weight', '0' if frozen else '1').split('#', 1)[0])
        if full_weight > 0:
            print('  Fine 每步 Full＋一个轮转 LoD，累计梯度后只提交一次 Adam。')
        else:
            print('  LoD 每步只轮转一个 ratio。')
        print('  Full 尾部和重要性统计渲染不计入在线步数。')
        if total:
            print(f'  RealLoD 不受 Selector.max_mapping_iterations={total} 的旧总步数上限限制。')
        return
    if enabled:
        length = 'floor(SLAM结束时迭代数 / 2)' if steps == -1 else str(steps)
        print(f'训练步数规则：在线 SLAM + 尾部 {length} 个 Mapping；总计不是尾部输入值。')
        if total:
            print(f'  强制尾部已开启：Selector.max_mapping_iterations={total} 被忽略。')
    elif total:
        print(f'训练步数规则：目标总 Mapping={total}（包含在线与尾部），不按每档 ratio 重复计算。')
        print('  若 Mapping 配置另设有限 Optimization.max_num_iterations，仍受该优化上限限制。')
    else:
        print('训练步数规则：原停止策略；force_tail=0 不等于禁用全部尾优化，最终总步数不是固定值。')
    if not enabled and steps != -1:
        print(f'  强制尾部已关闭：Training.tail_optimization_steps={steps} 不生效。')
    print('  要在 SLAM 结束后立即停止：force_tail=1 且 tail_optimization_steps=0。')


def config_overrides(config):
    if menu('基底配置的使用方式', ['直接使用基底设置', '调整本次实验（原文件不变）']) == 0:
        return {}
    values = config_values(config.read_text(encoding='utf-8-sig'))
    overrides = {}
    values.setdefault('Training.force_tail_optimization_enabled', '0')
    values.setdefault('Training.tail_optimization_steps', '-1')
    values.setdefault('Selector.max_mapping_iterations', '0')
    values.setdefault('Selector.anchor_only_pruning', '0')
    enabled = integer_override('Training.force_tail_optimization_enabled：直接输入 0/1 开关，不是步数',
                               'Training.force_tail_optimization_enabled', values, overrides)
    if enabled:
        integer_override('Training.tail_optimization_steps：尾部额外 Mapping 步数，-1 自动取半，0 不跑尾部',
                         'Training.tail_optimization_steps', values, overrides, -1, 2147483647)
    integer_override('Selector.anchor_only_pruning：0/1 原模式；3 固定十点关闭 slot；4 固定十点开启 slot',
                     'Selector.anchor_only_pruning', values, overrides, 0, 4, (0, 1, 3, 4))
    print('其他覆盖：逐行输入已有配置键和值，例如 Training.tail_optimization_steps: 2000。')
    print('输入值沿用 OpenCV YAML 写法（字符串加引号，数组用 []）；空行结束。')
    while True:
        line = ask('覆盖 key: value（也支持 key=value）', '')
        if not line:
            break
        pair = re.split(r'\s*[:=]\s*', line, maxsplit=1)
        if len(pair) != 2 or pair[0] not in values or not pair[1].strip():
            print('请输入基底中已有的键及非空值。')
            continue
        key, value = pair[0], pair[1].strip()
        try:
            budget_values(merge_config(config.read_text(encoding='utf-8-sig'), {**overrides, key: value}))
        except ValueError as error:
            print(error)
            continue
        overrides[key] = value
    for key, value in overrides.items():
        print(f'  本次覆盖：{key}: {value}')
    return overrides


def total_mapping_override(config, overrides):
    values = config_values(merge_config(config.read_text(encoding='utf-8-sig'), overrides))
    key = 'Selector.max_mapping_iterations'
    # 强制尾部开启时，基底中的总目标不生效，不能将它当作默认启用值。
    enabled = values.get('Training.force_tail_optimization_enabled', '0').split('#', 1)[0].strip()
    current = values.get(key, '0') if enabled == '0' else '0'
    print('\n整体步数控制：包含在线与尾部 Mapping；正整数优先于强制尾部设置，0 保留尾部设置。')
    total = integer_override('整体 Mapping 总步数：0 不限制总步数', key,
                             {key: current}, overrides, 0, 2147483647)
    if total > 0:
        overrides[key] = str(total)
        overrides['Training.force_tail_optimization_enabled'] = '0'
        print(f'  本次总 Mapping 目标={total}；强制尾部关闭，不再额外追加指定尾步。')
    return overrides


def datasets():
    found = []
    for root in [Path.home() / 'datasets', REPO / 'data']:
        for family in ['TUM_RGBD', 'Replica', '3DLite', 'VECtor']:
            directory = root / family
            if directory.is_dir():
                found.extend((family, p.resolve()) for p in sorted(directory.iterdir())
                             if p.is_dir() and not p.name.startswith('.'))
    return list(dict.fromkeys(found))


def choose_dataset():
    available = datasets()
    labels = [f'[{family}] {path.name}  ({path})' for family, path in available]
    index = menu('选择数据集', labels + ['手动输入数据集目录'])
    if index < len(available):
        return available[index]
    path = path_prompt('数据集目录', kind='dir')
    family = next((p.name for p in path.parents if p.name in DATASET_DEFAULTS),
                  'Replica' if (path / 'results').is_dir() else 'TUM_RGBD')
    return family, path


def require_files(*paths):
    for path in paths:
        if not path.is_file():
            raise ValueError(f'缺少文件：{path}')


def parse_ratios(text):
    ratios = []
    for token in re.split(r'[,，\s]+', text.strip()):
        if not token:
            continue
        value = None if token.lower() == 'full' else float(token.rstrip('%')) / 100.
        if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError('ratio 必须为大于 0、至多 100 的百分数，或 Full。')
        if value not in ratios:
            ratios.append(value)
    if not ratios:
        raise ValueError('至少选择一个 ratio 或 Full。')
    return ratios


def config_summary(config):
    text = config.read_text(encoding='utf-8-sig')
    for key in ['Selector.architecture', 'Selector.compensation_method',
                'Training.lod_start_iteration', 'Selector.physical_scale_additive',
                'MaturityV2.enabled', 'MaturityV2.contribution_enabled',
                'MaturityV2.contribution_weight', 'MaturityV2.compensation_gate',
                'Selector.target_ratios', 'Selector.max_mapping_iterations',
                'Training.force_tail_optimization_enabled', 'Selector.anchor_only_pruning',
                'Training.serial_full_lod', 'TransformField.enable', 'GlobalGI.mode']:
        match = re.search(r'^' + re.escape(key) + r'\s*:\s*([^#\n]+)', text, re.M)
        if match:
            print(f'  {key}: {match.group(1).strip()}')


def train_command(mode, dataset, camera, mapping, selector, output, index=None):
    command = [REPO / 'bin' / mode, REPO / 'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
               camera, mapping, dataset]
    if mode in {'tum_rgbd', 'vector_mono', 'vector_stereo'}:
        if index is None:
            raise ValueError('该训练入口需要关联文件或时间戳文件。')
        command.append(index)
    return list(map(str, command + [output, 'no_viewer', '--selector-config', selector]))


def training_plan(family, dataset):
    mode, folder, mapping_name = DATASET_DEFAULTS[family]
    scene = dataset.name.replace('rgbd_dataset_', 'tum_')
    if family == 'VECtor':
        scene = 'VECtor'
    elif family == '3DLite':
        scene = {'offices_aligned': 'offices_640', 'offices_tum': 'offices'}.get(scene, scene)
    print(f'\n数据集固定设置（自动匹配）：训练入口 {mode}')
    camera_dir = REPO / 'cfg/ORB_SLAM3' / folder
    camera = default_file('ORB 相机配置', [camera_dir / (scene + '.yaml')])
    mapping_dir = REPO / 'cfg/gaussian_mapper' / folder
    mapping = default_file('Gaussian Mapping 配置', [mapping_dir / mapping_name])
    index = None
    if mode == 'tum_rgbd':
        directory = camera_dir / 'associations'
        aliases = [scene + '.txt', dataset.name.replace('rgbd_dataset_freiburg', 'fr') + '.txt']
        candidates = [directory / name for name in aliases] + [dataset / 'associations.txt']
        index = default_file('RGB / Depth 关联文件', candidates)
        rows = [line.split('#', 1)[0].split() for line in index.read_text().splitlines()]
        rows = [row for row in rows if row]
        if not rows or any(len(row) != 4 for row in rows):
            raise ValueError('关联文件应为四列：RGB时间戳 RGB路径 Depth时间戳 Depth路径。')
        require_files(dataset / rows[0][1], dataset / rows[0][3])
    elif mode.startswith('vector_'):
        if not (dataset / 'rgb').is_dir() or (mode == 'vector_stereo' and not (dataset / 'rgb1').is_dir()):
            raise ValueError('VECtor 需要 rgb/；双目还需要 rgb1/。')
        index = default_file('图像时间戳文件', [dataset / 'rgb/timestamp.txt', dataset / 'timestamp.txt'])
    elif mode.startswith('replica_'):
        if not list((dataset / 'results').glob('frame*')):
            raise ValueError('Replica 数据目录缺少 results/frame* 图像。')
        if mode == 'replica_rgbd' and not list((dataset / 'results').glob('depth*')):
            raise ValueError('Replica RGB-D 数据目录缺少 results/depth*。')
    else:
        require_files(dataset / 'rgb.txt')
    selector = choose_file('Selector / Compensation 基底配置', (REPO / 'cfg/selector').glob('*.yaml'),
                           REPO / 'cfg/selector/selector_config.yaml')
    config_summary(selector)
    overrides = total_mapping_override(selector, config_overrides(selector))
    output = output_prompt('训练输出目录', REPO / 'results/interactive' /
                           f'{dataset.name}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    command = train_command(mode, dataset, camera, mapping, selector, output, index)
    require_files(Path(command[0]), Path(command[1]))
    return dict(task='train', dataset=str(dataset), output=str(output), commands=[command],
                training_mode=mode,
                configs=dict(camera=str(camera), mapping=str(mapping), selector=str(selector)),
                overrides={'selector': overrides})


def find_checkpoints(root):
    if root.name == 'ply':
        root = root.parent
    found = []
    for current, directories, _ in os.walk(root):
        path = Path(current)
        if path.name.endswith('_shutdown') and path.name.split('_', 1)[0].isdigit():
            iteration = path.name.split('_', 1)[0]
            if (path / 'ply/cameras.json').is_file() and (path / f'ply/point_cloud/iteration_{iteration}/point_cloud.ply').is_file():
                found.append(path)
            directories[:] = []
        else:
            directories[:] = [name for name in directories if not name.startswith('.') and
                              name not in {'image', 'gt', 'rgb', 'depth', 'point_cloud', '__pycache__', 'backup'}]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def render_commands(checkpoint, dataset, config, output, ratios, ply=None, correct_scale=False, save_images=True,
                    stereo_camera_config=None):
    commands = []
    for ratio in ratios:
        label = 'ply' if ply is not None else ('full' if ratio is None else
                'lod_' + format(ratio * 100, '.8g').replace('.', 'p') + 'pct')
        command = [sys.executable, '-u', str(REPO / 'python/run.py'), str(checkpoint.parent), str(dataset),
                   '--shutdown-model', checkpoint.name, '--output_path', str(output / label),
                   '--no-selector-anchor-frustum']
        if ply is not None:
            command += ['--ply', str(ply)]
        elif ratio is not None:
            command += ['--selector', str(checkpoint / 'ply/selector.pt'), '--selector-ratio', str(ratio),
                        '--selector-config', str(config)]
        if correct_scale:
            command.append('--correct_scale')
        if not save_images:
            command.append('--no-save-images')
        if stereo_camera_config:
            command += ['--stereo-camera-config', str(stereo_camera_config)]
        commands.append(command)
    return commands


def rendering_plan(dataset, family='TUM_RGBD'):
    root = path_prompt('模型结果目录（可输入具体 *_shutdown 或其 ply 目录）', REPO / 'results', kind='dir')
    checkpoints = find_checkpoints(root)
    if not checkpoints:
        raise ValueError('此目录下没有包含 cameras.json 和 point_cloud.ply 的完整 shutdown 模型。')
    checkpoint = checkpoints[menu('选择模型 checkpoint', [str(p) for p in checkpoints])]
    source = checkpoint.parent
    require_files(source / 'CameraTrajectory_TUM.txt', source / 'TrackingTime.txt')
    if 'replica' in str(dataset).lower():
        require_files(dataset / 'pose_TUM.txt')
    else:
        require_files(dataset / 'groundtruth.txt')
    exported = menu('渲染方式', ['Full / Selector LoD（含模型补偿）', '渲染已导出的 PLY']) == 1
    ply, config, overrides = None, None, {}
    if exported:
        ply = path_prompt('已导出的 PLY 文件')
        ratios = [None]
    else:
        print('Full = 原 Master；100 = Selector 的 100% LoD。百分数示例：Full,1,2.5,5,10,15')
        ratios = parse_ratios(ask('需要渲染的档位', 'Full,1,5,10,15'))
        if any(r is not None for r in ratios):
            require_files(checkpoint / 'ply/selector.pt')
            iteration = checkpoint.name.split('_', 1)[0]
            require_files(checkpoint / f'ply/point_cloud/iteration_{iteration}/selector_metadata.ply')
            saved = checkpoint / 'ply/selector_config.yaml'
            config = choose_file('渲染配置（优先使用该模型保存的配置）',
                                 [saved] + list((REPO / 'cfg/selector').glob('*.yaml')), saved)
            config_summary(config)
            overrides = config_overrides(config)
    launch = source / 'launch.json'
    previous = json.loads(launch.read_text(encoding='utf-8')) if launch.is_file() else {}
    mode = previous.get('training_mode', DATASET_DEFAULTS[family][0])
    mapping = previous.get('configs', {}).get('mapping', '').replace('\\', '/')
    correct_scale = mode.endswith('_mono') or '/Monocular/' in mapping
    stereo_camera_config = None
    if mode.endswith('_stereo') or '/Stereo/' in mapping:
        saved_camera = previous.get('configs', {}).get('camera')
        stereo_camera_config = Path(saved_camera) if saved_camera else None
        if not stereo_camera_config or not stereo_camera_config.is_file():
            stereo_camera_config = choose_file('双目评价使用训练时的相机标定',
                list((REPO/'cfg/ORB_SLAM3/Stereo'/family).glob('*.yaml')))
    print(f'固定渲染设置：保存图片和指标；尺度对齐：{"开启（单目）" if correct_scale else "关闭（RGB-D / 双目）"}')
    output = output_prompt('渲染输出目录', REPO / 'results/interactive_renders' / source.name)
    commands = render_commands(checkpoint, dataset, config, output, ratios, ply, correct_scale,
                               stereo_camera_config=stereo_camera_config)
    return dict(task='render', dataset=str(dataset), checkpoint=str(checkpoint), output=str(output),
                commands=commands, configs={'selector': str(config)} if config else {},
                overrides={'selector': overrides} if config else {})


def write_render_summary(output, commands):
    summary = []
    for command in commands:
        directory = Path(command[command.index('--output_path') + 1])
        phase = 'LOD' if '--selector' in command else ('PLY' if '--ply' in command else 'FULL')
        metrics = {}
        for line in (directory / 'eval.txt').read_text(encoding='utf-8').splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    continue
        row = dict(ratio=float(command[command.index('--selector-ratio') + 1]) if phase == 'LOD' else 1.,
                   phase=phase, output_path=str(directory), **metrics)
        stats = directory / 'selector_stats.json'
        if stats.is_file():
            row.update(json.loads(stats.read_text(encoding='utf-8')))
        summary.append(row)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader()
        writer.writerows(summary)
    print(f'渲染汇总：{output / "summary.json"}')


def execute_plan(plan, dry_run=False):
    output = Path(plan['output'])
    texts = {label: merge_config(Path(name).read_text(encoding='utf-8-sig'), plan.get('overrides', {}).get(label, {}))
             for label, name in plan['configs'].items()}
    if 'selector' in texts:
        budget_values(texts['selector'])
    effective = {name: str(output / (label + '_effective.yaml')) for label, name in plan['configs'].items()}
    commands = [[effective.get(token, token) for token in command] for command in plan['commands']]
    print(f"\n任务：{plan['task']}\n数据：{plan['dataset']}\n输出：{output}")
    if plan['task'] == 'train' and 'selector' in texts:
        print_budget(texts['selector'])
    for label, values in plan.get('overrides', {}).items():
        for key, value in values.items():
            print(f'覆盖 {label}：{key}: {value}')
    for command in commands:
        print('\n' + shlex.join(command))
    if dry_run:
        print('\n试运行结束：未启动任务，也未创建输出目录。')
        return 0
    if ask('开始执行？输入 y 开始，其他内容返回主菜单', 'n').lower() != 'y':
        return 0
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(REPO / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    for key, value in {'OMP_NUM_THREADS': '2', 'QT_QPA_PLATFORM': 'offscreen',
                       'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'}.items():
        env.setdefault(key, value)
    for key in ['CARTGS_COMPENSATION_SPLIT4', 'CARTGS_BUDGET_METHOD', 'CARTGS_BUDGET_AUDIT', 'CARTGS_MLP_GRAD_AUDIT']:
        env.pop(key, None)
    plan['config_sha256'], plan['effective_config_sha256'] = {}, {}
    for label, name in plan['configs'].items():
        path = Path(name)
        plan['config_sha256'][label] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, output / (label + '_input.yaml'))
        text = texts[label]
        target = Path(effective[name])
        target.write_text(text, encoding='utf-8')
        plan['effective_config_sha256'][label] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / 'config_overrides.json').write_text(json.dumps(plan.get('overrides', {}), ensure_ascii=False, indent=2), encoding='utf-8')
    plan['commands'] = commands
    (output / 'launch.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    status = dict(completed=0, total=len(plan['commands']), exit_code=None)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        for command in commands:
            log.write('\n$ ' + shlex.join(command) + '\n')
            log.flush()
            with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    code = 130
            status['exit_code'] = code
            if code == 0:
                status['completed'] += 1
            (output / 'launch_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            if code != 0:
                print(f'任务退出，返回码 {code}。已保留日志：{output / "run.log"}')
                return code
    if plan['task'] == 'render':
        write_render_summary(output, commands)
    print(f'任务完成，结果位于：{output}')
    return 0


def main():
    parser = argparse.ArgumentParser(description='mycartgs 交互式训练 / 渲染')
    parser.add_argument('--dry-run', action='store_true', help='仅交互生成命令，不执行或创建输出')
    args = parser.parse_args()
    print('mycartgs 交互式启动器；路径相对项目根目录，q 返回主菜单。')
    while True:
        try:
            action = menu('选择任务', ['训练', '渲染 / 质量评价', '退出'])
            if action == 2:
                return 0
            family, dataset = choose_dataset()
            plan = training_plan(family, dataset) if action == 0 else rendering_plan(dataset, family)
            code = execute_plan(plan, args.dry_run)
            if args.dry_run or code == 130:
                return code
        except Cancelled:
            continue
        except (ValueError, OSError) as error:
            print(f'无法启动：{error}')
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            return 0


if __name__ == '__main__':
    sys.exit(main())
