"""不加载CUDA模型，验证普通渲染默认全图候选及显式旧视锥开关。"""
from pathlib import Path
import argparse
import ast
import unittest


ROOT=Path(__file__).resolve().parents[1]


class RenderDefaultsTest(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((ROOT/'python/run.py').read_text(encoding='utf-8'))
        registration=next(node for node in ast.walk(tree) if isinstance(node,ast.Expr)
            and isinstance(node.value,ast.Call) and any(isinstance(arg,ast.Constant)
            and arg.value=='--selector-anchor-frustum' for arg in node.value.args))
        fallback=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
            and ast.unparse(node.test)=='args.selector_anchor_frustum is None')
        self.registration=compile(ast.fix_missing_locations(ast.Module(body=[registration],type_ignores=[])),str(ROOT/'python/run.py'),'exec')
        self.fallback=compile(ast.fix_missing_locations(ast.Module(body=[fallback],type_ignores=[])),str(ROOT/'python/run.py'),'exec')

    def parse(self,flags,stored):
        parser=argparse.ArgumentParser();exec(self.registration,dict(parser=parser,argparse=argparse))
        args=parser.parse_args(flags)
        exec(self.fallback,dict(args=args,selector_config={'anchor_frustum_culling':stored}))
        return args.selector_anchor_frustum

    def test_saved_checkpoint_cannot_enable_frustum_by_default(self):
        for stored in [0,1]:self.assertFalse(self.parse([],stored))

    def test_explicit_legacy_and_global_paths(self):
        self.assertTrue(self.parse(['--selector-anchor-frustum'],0))
        self.assertFalse(self.parse(['--no-selector-anchor-frustum'],1))

    def test_production_training_presets(self):
        import re
        for name in ['selector_config.yaml','spatial_offset3_residual48.yaml']:
            text=(ROOT/'cfg/selector'/name).read_text(encoding='utf-8')
            self.assertEqual(re.findall(r'^Selector.anchor_only_pruning:\s*(\d+)\s*$',text,re.M),['1'])
            self.assertEqual(re.findall(r'^Selector.anchor_frustum_culling:\s*(\d+)\s*$',text,re.M),['0'])


if __name__=='__main__':unittest.main()
