"""验证启动入口、checkpoint绑定及不覆盖结果；不启动SLAM或CUDA。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'scripts/interactive_run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cartgs launcher ')
        self.repo = Path(self.temp.name)
        self.patch = mock.patch.object(launcher, 'REPO', self.repo)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temp.cleanup)

    def file(self, name, content='fixture'):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def checkpoint(self, name):
        self.file(f'{name}/ply/cameras.json', '[]')
        iteration = Path(name).name.split('_', 1)[0]
        self.file(f'{name}/ply/point_cloud/iteration_{iteration}/point_cloud.ply')
        return self.repo / name

    def test_existing_native_entry_signatures_and_paths_with_spaces(self):
        for mode in ['tum_rgbd', 'replica_rgbd', 'tum_mono', 'replica_mono', 'vector_mono', 'vector_stereo']:
            paths = [self.repo / name for name in ['data with spaces', 'camera.yaml', 'mapping.yaml', 'selector.yaml', 'new output']]
            command = launcher.train_command(mode, *paths, index=self.repo / 'associations.txt')
            extra = int(mode in {'tum_rgbd', 'vector_mono', 'vector_stereo'})
            self.assertEqual(len(command), 9 + extra)
            self.assertEqual(command[4], str(paths[0]))
            self.assertEqual(command[5 + extra], str(paths[4]))
            self.assertEqual(command[-3:], ['no_viewer', '--selector-config', str(paths[3])])

    def test_full_is_distinct_from_selector_100_and_checkpoint_is_explicit(self):
        checkpoint = self.checkpoint('run/100_shutdown')
        commands = launcher.render_commands(checkpoint, self.repo / 'dataset', self.repo / 'saved.yaml',
                                             self.repo / 'new output', [None, 1., .025])
        self.assertNotIn('--selector', commands[0])
        self.assertIn('--selector', commands[1])
        self.assertEqual(commands[1][commands[1].index('--selector-ratio') + 1], '1.0')
        self.assertEqual(commands[2][commands[2].index('--selector-ratio') + 1], '0.025')
        for command in commands:
            self.assertEqual(command[command.index('--shutdown-model') + 1], checkpoint.name)
            self.assertIn(str(checkpoint.parent), command)
            self.assertIn('--no-selector-anchor-frustum', command)

    def test_checkpoint_search_preserves_choice_and_accepts_direct_ply(self):
        a = self.checkpoint('a/100_shutdown')
        b = self.checkpoint('b/200_shutdown')
        self.file('incomplete/300_shutdown/ply/cameras.json')
        self.assertEqual(set(launcher.find_checkpoints(self.repo)), {a, b})
        self.assertEqual(launcher.find_checkpoints(a / 'ply'), [a])

    def test_stereo_render_carries_training_calibration(self):
        checkpoint = self.checkpoint('stereo/100_shutdown')
        calibration = self.file('stereo_camera.yaml')
        commands = launcher.render_commands(checkpoint,self.repo/'dataset',self.repo/'saved.yaml',
            self.repo/'render',[None,.1],stereo_camera_config=calibration)
        for command in commands:
            self.assertEqual(command[command.index('--stereo-camera-config')+1],str(calibration))

    def test_stereo_render_plan_reuses_effective_training_camera(self):
        checkpoint=self.checkpoint('stereo_run/200_shutdown')
        camera=self.file('stereo_run/camera_effective.yaml')
        self.file('stereo_run/CameraTrajectory_TUM.txt')
        self.file('stereo_run/TrackingTime.txt')
        self.file('stereo_run/launch.json',json.dumps(dict(training_mode='vector_stereo',configs=dict(camera=str(camera)))))
        self.file('dataset/groundtruth.txt')
        inputs=[str(checkpoint),'1','1','Full',str(self.repo/'rectified render'),'']
        with mock.patch('builtins.input',side_effect=inputs), \
             mock.patch.object(launcher,'choose_file') as choose,contextlib.redirect_stdout(io.StringIO()):
            plan=launcher.rendering_plan(self.repo/'dataset','VECtor')
        choose.assert_not_called()
        command=plan['commands'][0]
        self.assertEqual(command[command.index('--stereo-camera-config')+1],str(camera))
        self.assertNotIn('--correct_scale',command)

    def test_ratio_percent_units_and_invalid_values(self):
        self.assertEqual(launcher.parse_ratios('Full,1,2.5%,5，10 15 100 full'), [None, .01, .025, .05, .1, .15, 1.])
        for text in ['nan', 'inf', '0', '-1', '101', '', 'not-a-ratio']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                launcher.parse_ratios(text)

    def test_existing_output_and_dry_run_cannot_overwrite_or_launch(self):
        keep = self.file('old output/keep.txt', 'unchanged')
        plan = dict(task='train', dataset='fixture', output=str(keep.parent), commands=[['never-run']], configs={})
        with mock.patch('builtins.input', return_value='y'), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            launcher.execute_plan(plan)
        self.assertEqual(keep.read_text(), 'unchanged')
        plan['output'] = str(self.repo / 'new output')
        with mock.patch.object(subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan, dry_run=True), 0)
            popen.assert_not_called()
        self.assertFalse(Path(plan['output']).exists())

    def test_training_menu_end_to_end_dry_run(self):
        dataset = self.repo / 'data/TUM_RGBD/rgbd_dataset_freiburg1_room'
        self.file(str(dataset.relative_to(self.repo) / 'rgb/one.png'))
        self.file(str(dataset.relative_to(self.repo) / 'depth/one.png'))
        self.file('cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_room.yaml')
        self.file('cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_room.txt',
                  '1 rgb/one.png 1 depth/one.png\n')
        self.file('cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml')
        self.file('cfg/selector/selector_config.yaml', 'Selector.compensation_method: "r3_independent14"')
        self.file('bin/tum_rgbd')
        self.file('third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt')
        output = self.repo / 'never-created'
        inputs = ['1', '1', '1', '1', '16400', str(output), '']
        with mock.patch.object(sys, 'argv', ['interactive_run.py', '--dry-run']), \
             mock.patch.object(launcher, 'datasets', return_value=[('TUM_RGBD', dataset)]), \
             mock.patch('builtins.input', side_effect=inputs), \
             mock.patch.object(subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(launcher.main(), 0)
            popen.assert_not_called()
        self.assertIn('r3_independent14', capture.getvalue())
        self.assertIn('目标总 Mapping=16400', capture.getvalue())
        self.assertIn('Training.force_tail_optimization_enabled: 0', capture.getvalue())
        self.assertIn('tum_freiburg1_room.txt', capture.getvalue())
        self.assertNotIn('选择训练模式', capture.getvalue())
        self.assertNotIn('手动输入文件路径', capture.getvalue().split('Selector / Compensation')[0])
        self.assertFalse(output.exists())

    def test_render_menu_prefers_matching_saved_config(self):
        checkpoint = self.checkpoint('run/200_shutdown')
        self.file('run/CameraTrajectory_TUM.txt')
        self.file('run/TrackingTime.txt')
        saved = self.file('run/200_shutdown/ply/selector_config.yaml')
        self.file('run/200_shutdown/ply/selector.pt')
        self.file('run/200_shutdown/ply/point_cloud/iteration_200/selector_metadata.ply')
        self.file('cfg/selector/selector_config.yaml')
        self.file('dataset/groundtruth.txt')
        output = self.repo / 'results/interactive_renders/run'
        inputs = [str(checkpoint / 'ply'), '1', '1', 'Full,100,2.5', '1', '1', '', '']
        with mock.patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
            plan = launcher.rendering_plan(self.repo / 'dataset')
        self.assertEqual(plan['configs']['selector'], str(saved))
        self.assertEqual(plan['output'], str(output))
        self.assertEqual(len(plan['commands']), 3)
        for command in plan['commands']:
            self.assertNotIn('--no-save-images', command)
            self.assertNotIn('--correct_scale', command)
            if '--selector-config' in command:
                self.assertEqual(command[command.index('--selector-config') + 1], str(saved))
        self.assertFalse(output.exists())

    def test_dataset_defaults_only_prompt_for_selector(self):
        selector = self.file('cfg/selector/selector_config.yaml')
        self.file('third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt')
        scenes = [
            ('TUM_RGBD', 'rgbd_dataset_freiburg1_room', 'tum_rgbd', 'RGB-D/TUM', 'tum_freiburg1_room', 'tum_rgbd.yaml'),
            ('Replica', 'office0', 'replica_rgbd', 'RGB-D/Replica', 'office0', 'replica_rgbd.yaml'),
            ('3DLite', 'offices_aligned', 'tum_rgbd', 'RGB-D/TUM', 'offices_640', 'tum_rgbd.yaml'),
            ('3DLite', 'offices_tum', 'tum_rgbd', 'RGB-D/TUM', 'offices', 'tum_rgbd.yaml'),
            ('VECtor', 'corner-slow', 'vector_mono', 'Monocular/VECtor', 'VECtor', 'VECtor.yaml'),
        ]
        for family, scene, mode, folder, camera_name, mapping_name in scenes:
            with self.subTest(scene=scene):
                dataset = self.repo / 'data' / family / scene
                self.file(str(dataset.relative_to(self.repo) / 'rgb/one.png'))
                self.file(str(dataset.relative_to(self.repo) / 'depth/one.png'))
                self.file(str(dataset.relative_to(self.repo) / 'associations.txt'), '1 rgb/one.png 1 depth/one.png\n')
                self.file(str(dataset.relative_to(self.repo) / 'rgb/timestamp.txt'), '1\n')
                self.file(str(dataset.relative_to(self.repo) / 'results/frame0.jpg'))
                self.file(str(dataset.relative_to(self.repo) / 'results/depth0.png'))
                camera = self.file(f'cfg/ORB_SLAM3/{folder}/{camera_name}.yaml')
                mapping = self.file(f'cfg/gaussian_mapper/{folder}/{mapping_name}')
                self.file(f'bin/{mode}')
                with mock.patch.object(launcher, 'choose_file', return_value=selector) as choose, \
                     mock.patch.object(launcher, 'config_overrides', return_value={}), \
                     mock.patch.object(launcher, 'total_mapping_override', return_value={}), \
                     mock.patch.object(launcher, 'output_prompt', return_value=self.repo / ('out_' + scene)), \
                     mock.patch('builtins.input', side_effect=AssertionError('不应询问固定参数')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    plan = launcher.training_plan(family, dataset)
                choose.assert_called_once()
                self.assertIn('Selector', choose.call_args.args[0])
                self.assertEqual(plan['training_mode'], mode)
                self.assertEqual(plan['configs']['camera'], str(camera))
                self.assertEqual(plan['configs']['mapping'], str(mapping))

    def test_missing_camera_never_silently_uses_another_scene(self):
        unrelated = self.file('cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml')
        correct = self.file('external_room.yaml')
        missing = self.repo / 'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_room.yaml'
        with mock.patch.object(launcher, 'path_prompt', return_value=correct) as prompt, contextlib.redirect_stdout(io.StringIO()):
            result = launcher.default_file('ORB 相机配置', [missing])
        self.assertEqual(result, correct)
        self.assertNotEqual(result, unrelated)
        prompt.assert_called_once()

    def test_render_scale_uses_saved_training_mode(self):
        checkpoint = self.checkpoint('mono_run/200_shutdown')
        self.file('mono_run/CameraTrajectory_TUM.txt')
        self.file('mono_run/TrackingTime.txt')
        self.file('mono_run/launch.json', json.dumps({'training_mode': 'tum_mono'}))
        self.file('dataset/groundtruth.txt')
        output = self.repo / 'render mono'
        inputs = [str(checkpoint), '1', '1', 'Full', str(output), '']
        with mock.patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
            plan = launcher.rendering_plan(self.repo / 'dataset')
        self.assertIn('--correct_scale', plan['commands'][0])
        self.assertNotIn('--no-save-images', plan['commands'][0])

    def test_output_suffix_is_appended_and_existing_suffix_cannot_be_overwritten(self):
        self.file('experiment_existing/keep.txt')
        inputs = [str(self.repo / 'experiment'), 'existing', str(self.repo / 'experiment'), '关闭尾优化_仅Anchor剪枝']
        with mock.patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
            output = launcher.output_prompt('输出', self.repo / 'default')
        self.assertEqual(output.name, 'experiment_关闭尾优化_仅Anchor剪枝')
        self.assertFalse(output.exists())
        self.file('existing base/keep.txt')
        with mock.patch('builtins.input', side_effect=[str(self.repo / 'existing base'), 'new variant']), contextlib.redirect_stdout(io.StringIO()):
            output = launcher.output_prompt('输出', self.repo / 'default')
        self.assertEqual(output.name, 'existing base_new variant')
        self.assertFalse(output.exists())

    def test_overrides_menu_keeps_base_and_manual_value_wins(self):
        config = self.file('base.yaml', '%YAML:1.0\nTraining.force_tail_optimization_enabled: 1\n'
                           'Selector.anchor_only_pruning: 0\nTraining.tail_optimization_steps: -1\n')
        before = config.read_bytes()
        inputs = ['2', '0', '1', 'Missing.key: 1', 'Selector.anchor_only_pruning: 2',
                  'Training.tail_optimization_steps=2000', 'Selector.anchor_only_pruning: 0', '']
        with mock.patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
            overrides = launcher.config_overrides(config)
        self.assertEqual(overrides, {'Training.force_tail_optimization_enabled': '0',
                                    'Selector.anchor_only_pruning': '0', 'Training.tail_optimization_steps': '2000'})
        self.assertEqual(config.read_bytes(), before)
        merged = launcher.merge_config(config.read_text(), overrides)
        self.assertEqual(merged.count('Selector.anchor_only_pruning:'), 1)
        self.assertEqual(launcher.config_values(merged)['Training.tail_optimization_steps'], '2000')

    def test_tail_switch_and_tail_length_are_direct_inputs_not_menu_numbers(self):
        config = self.file('base.yaml', '%YAML:1.0\nTraining.force_tail_optimization_enabled: 0\n'
                           'Training.tail_optimization_steps: -1\nSelector.max_mapping_iterations: 0\n'
                           'Selector.anchor_only_pruning: 1\n')
        inputs = ['2', '16400', '1', '-2', '16400', '', '']
        with mock.patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
            overrides = launcher.config_overrides(config)
        self.assertEqual(overrides, {'Training.force_tail_optimization_enabled': '1',
                                    'Training.tail_optimization_steps': '16400'})
        merged = launcher.merge_config(config.read_text(), overrides)
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            launcher.print_budget(merged)
        self.assertIn('尾部 16400', capture.getvalue())
        self.assertIn('总计不是尾部输入值', capture.getvalue())
        self.assertIn('Training.force_tail_optimization_enabled: 0', config.read_text())

    def test_total_mapping_input_is_distinct_from_tail_length(self):
        config = self.file('base.yaml', '%YAML:1.0\nTraining.force_tail_optimization_enabled: 1\n'
                           'Training.tail_optimization_steps: 2000\nSelector.max_mapping_iterations: 0\n'
                           'Selector.anchor_only_pruning: 1\n')
        before = config.read_bytes()
        with mock.patch('builtins.input', side_effect=['-1', '1.5', '16400']), contextlib.redirect_stdout(io.StringIO()):
            overrides = launcher.total_mapping_override(config, {})
        self.assertEqual(overrides['Selector.max_mapping_iterations'], '16400')
        self.assertEqual(overrides['Training.force_tail_optimization_enabled'], '0')
        self.assertNotIn('Training.tail_optimization_steps', overrides)
        self.assertEqual(config.read_bytes(), before)

    def test_zero_total_keeps_tail_settings_and_clears_existing_total(self):
        for enabled in [0, 1]:
            with self.subTest(enabled=enabled):
                config = self.file('base.yaml', f'Training.force_tail_optimization_enabled: {enabled}\n'
                                   'Training.tail_optimization_steps: 2000\nSelector.max_mapping_iterations: 16400\n')
                with mock.patch('builtins.input', return_value='0'), contextlib.redirect_stdout(io.StringIO()):
                    overrides = launcher.total_mapping_override(config, {})
                self.assertEqual(overrides, {'Selector.max_mapping_iterations': '0'})
                merged = launcher.budget_values(launcher.merge_config(config.read_text(), overrides))
                self.assertEqual(merged['Training.force_tail_optimization_enabled'], enabled)
                self.assertEqual(merged['Training.tail_optimization_steps'], 2000)

    def test_blank_total_preserves_base_stop_policy_and_existing_overrides(self):
        for enabled, total in [(0, 0), (0, 16400), (1, 16400)]:
            with self.subTest(enabled=enabled, total=total):
                config = self.file('base.yaml', f'Training.force_tail_optimization_enabled: {enabled}\n'
                                   f'Selector.max_mapping_iterations: {total}\n')
                prior = {'Selector.anchor_only_pruning': '1'}
                with mock.patch('builtins.input', return_value=''), contextlib.redirect_stdout(io.StringIO()):
                    overrides = launcher.total_mapping_override(config, dict(prior))
                merged = launcher.budget_values(launcher.merge_config(config.read_text(), overrides))
                self.assertEqual(merged['Selector.max_mapping_iterations'], total)
                self.assertEqual(merged['Training.force_tail_optimization_enabled'], enabled)
                self.assertEqual(merged['Selector.anchor_only_pruning'], 1)

    def test_invalid_boolean_in_base_is_blocked_before_launch_or_output_creation(self):
        config = self.file('room16400.yaml', '%YAML:1.0\nTraining.force_tail_optimization_enabled: 16400\n')
        output = self.repo / 'not_created'
        plan = dict(task='train', dataset='fixture', output=str(output), configs={'selector': str(config)},
                    commands=[['never-run', str(config)]])
        with mock.patch.object(subprocess, 'Popen') as popen, self.assertRaisesRegex(ValueError, '这是开关'):
            launcher.execute_plan(plan, dry_run=True)
        popen.assert_not_called()
        self.assertFalse(output.exists())
        for key, value in [('Training.tail_optimization_steps', '-2'),
                           ('Selector.max_mapping_iterations', '-1'),
                           ('Training.force_tail_optimization_enabled', '1.5')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.budget_values(f'{key}: {value}\n')

    def test_real_lod_reports_three_phases_and_online_sized_lod_budget(self):
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            launcher.print_budget('Training.real_lod: 1\nTraining.force_tail_optimization_enabled: 1\nTraining.tail_optimization_steps: 0\nSelector.max_mapping_iterations: 8000\n')
        output=capture.getvalue()
        self.assertIn('LoD 步数等于实际在线 Mapping',output)
        self.assertIn('Full＋一个轮转 LoD',output)
        self.assertIn('只提交一次 Adam',output)
        self.assertIn('不受 Selector.max_mapping_iterations=8000',output)
        self.assertNotIn('SLAM 结束后立即停止',output)

    def test_fixed_ten_modes_are_preserved_and_mode_two_is_rejected(self):
        for mode in (0, 1, 3, 4):
            with self.subTest(mode=mode):
                text=f'Selector.anchor_only_pruning: {mode}\n'
                self.assertEqual(launcher.budget_values(text)['Selector.anchor_only_pruning'], mode)
        for mode in (-1, 2, 5, 16400):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                launcher.budget_values(f'Selector.anchor_only_pruning: {mode}\n')
        config=self.file('fixed.yaml', 'Training.force_tail_optimization_enabled: 0\nSelector.anchor_only_pruning: 0\n')
        before=config.read_bytes()
        with mock.patch('builtins.input', side_effect=['2', '', '2', '3', 'Selector.anchor_only_pruning: 4', '']), contextlib.redirect_stdout(io.StringIO()):
            overrides=launcher.config_overrides(config)
        self.assertEqual(overrides, {'Selector.anchor_only_pruning': '4'})
        self.assertEqual(config.read_bytes(), before)

    def test_zero_switch_does_not_claim_zero_tail_and_ignored_total_is_reported(self):
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            launcher.print_budget('Training.force_tail_optimization_enabled: 0\nTraining.tail_optimization_steps: 0\n')
        self.assertIn('不等于禁用全部尾优化', capture.getvalue())
        self.assertIn('tail_optimization_steps=0 不生效', capture.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            launcher.print_budget('Training.force_tail_optimization_enabled: 1\nTraining.tail_optimization_steps: 0\n'
                                  'Selector.max_mapping_iterations: 16400\n')
        self.assertIn('尾部 0', capture.getvalue())
        self.assertIn('16400 被忽略', capture.getvalue())

    def test_effective_config_is_used_by_child_without_changing_source(self):
        config = self.file('base.yaml', '%YAML:1.0\nTraining.force_tail_optimization_enabled: 1 # old\n'
                           'Selector.anchor_only_pruning: 0\nSelector.compensation_method: "r3_independent14"\n')
        before = config.read_bytes()
        output = self.repo / 'new config run'
        child = ('import sys; from pathlib import Path; '
                 'p=Path(sys.argv[1]); s=p.read_text(); '
                 'assert p.name=="selector_effective.yaml"; '
                 'assert "Training.force_tail_optimization_enabled: 0" in s; '
                 'assert "Selector.max_mapping_iterations: 16400" in s; '
                 'assert "Selector.anchor_only_pruning: 1" in s; '
                 'assert "r3_independent14" in s; print("effective config verified")')
        with mock.patch('builtins.input', return_value='16400'), contextlib.redirect_stdout(io.StringIO()):
            overrides = launcher.total_mapping_override(config, {'Selector.anchor_only_pruning': '1'})
        plan = dict(task='train', dataset='fixture', output=str(output), configs={'selector': str(config)},
                    overrides={'selector': overrides},
                    commands=[[sys.executable, '-c', child, str(config)]])
        with mock.patch('builtins.input', return_value='y'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan), 0)
        self.assertEqual(config.read_bytes(), before)
        self.assertEqual((output / 'selector_input.yaml').read_bytes(), before)
        saved = json.loads((output / 'launch.json').read_text())
        self.assertEqual(saved['commands'][0][-1], str(output / 'selector_effective.yaml'))
        self.assertNotEqual(saved['config_sha256']['selector'], saved['effective_config_sha256']['selector'])
        self.assertEqual(json.loads((output / 'config_overrides.json').read_text()), plan['overrides'])

    def test_render_completion_writes_eval_lod_compatible_summary(self):
        output = self.repo / 'render summary'
        child = ('import sys,json; from pathlib import Path; '
                 'p=Path(sys.argv[sys.argv.index("--output_path")+1]); p.mkdir(parents=True); '
                 '(p/"eval.txt").write_text("psnr: 25.5\\nssim: 0.9\\nlpips: 0.1\\nFPS: 1.2e3\\n"); '
                 '(p/"selector_stats.json").write_text(json.dumps(dict(active_gaussians=123,total_gaussians=456)))')
        commands = [[sys.executable, '-c', child, '--output_path', str(output / 'full')],
                    [sys.executable, '-c', child, '--output_path', str(output / 'lod_100pct'),
                     '--selector', 'fixture.pt', '--selector-ratio', '1.0'],
                    [sys.executable, '-c', child, '--output_path', str(output / 'lod_2p5pct'),
                     '--selector', 'fixture.pt', '--selector-ratio', '.025']]
        plan = dict(task='render', dataset='fixture', output=str(output), configs={}, commands=commands)
        with mock.patch('builtins.input', return_value='y'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan), 0)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['phase'] for row in summary], ['FULL', 'LOD', 'LOD'])
        self.assertEqual([row['ratio'] for row in summary], [1., 1., .025])
        for row in summary:
            self.assertEqual(row['psnr'], 25.5)
            self.assertEqual(row['FPS'], 1200.)
            self.assertEqual(row['active_gaussians'], 123)
        self.assertTrue((output / 'summary.csv').is_file())

    def test_child_failure_is_logged_and_stops_remaining_commands(self):
        output = self.repo / 'new output'
        config = self.file('source.yaml', 'original-config')
        plan = dict(task='fixture', dataset='fixture', output=str(output), configs={'selector': str(config)},
                    commands=[[sys.executable, '-c', 'print("intentional failure"); raise SystemExit(7)'],
                              [sys.executable, '-c', 'raise SystemExit(99)']])
        with mock.patch('builtins.input', return_value='y'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan), 7)
        self.assertIn('intentional failure', (output / 'run.log').read_text())
        status = json.loads((output / 'launch_status.json').read_text())
        self.assertEqual(status['exit_code'], 7)
        self.assertEqual(status['completed'], 0)
        self.assertEqual((output / 'selector_input.yaml').read_text(), config.read_text())
        self.assertEqual(config.read_text(), 'original-config')
        self.assertFalse((output / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
