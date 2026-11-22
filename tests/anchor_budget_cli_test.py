"""执行真实入口的参数解析段；不初始化CUDA指标或启动渲染。"""
import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

script = Path(__file__).resolve().parents[1] / 'python/run.py'
tree = ast.parse(script.read_text(encoding='utf-8'))
main = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")
stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'sh_degree' for t in n.targets))
code = compile(ast.Module(body=main.body[:stop], type_ignores=[]), str(script), 'exec')


class AnchorBudgetCLITest(unittest.TestCase):
    def parse(self, options):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'argv',
                ['run.py', 'model', 'dataset', '--output_path', directory] + options):
            scope = dict(ArgumentParser=argparse.ArgumentParser, argparse=argparse, os=os)
            with contextlib.redirect_stderr(io.StringIO()):exec(code, scope)
            return scope['args']

    def test_default_remains_disabled(self):
        args = self.parse([])
        self.assertIsNone(args.inference_budget)
        self.assertIsNone(args.selector_anchor_frustum)

    def test_existing_single_ratio_still_parses(self):
        args = self.parse(['--selector', 'selector.pt', '--selector-ratio', '.4'])
        self.assertEqual(args.selector_ratio, .4)
        self.assertIsNone(args.inference_budget)

    def test_explicit_budget_uses_nominal_frustum_without_global_ratio(self):
        args = self.parse(['--inference-budget', '5000', '--selector', 'selector.pt'])
        self.assertTrue(args.selector_anchor_frustum)
        self.assertIsNone(args.selector_ratio)
        self.assertEqual(args.inference_budget, 5000)

    def test_budget_requires_checkpoint_and_nonnegative_count(self):
        for options in [['--inference-budget', '10'], ['--inference-budget', '-1', '--selector', 's.pt']]:
            with self.assertRaises(SystemExit):self.parse(options)

    def test_rejects_conflicting_single_ratio_or_frustum_options(self):
        for options in [['--selector-ratio', '.4'], ['--selector-sampling-rng', 'rng.pt'],
                        ['--no-selector-anchor-frustum']]:
            with self.assertRaises(SystemExit):
                self.parse(['--inference-budget', '5000', '--selector', 'selector.pt'] + options)

    def test_policy_options_require_explicit_enable(self):
        for options in [['--budget-sampling-seed', '0'], ['--anchor-distance-lod']]:
            with self.assertRaises(SystemExit):self.parse(options)


if __name__ == '__main__':unittest.main()
