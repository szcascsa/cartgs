"""现有预算、Viewer协议及命令行回归，不启动用户Viewer。"""
from common import *
import unittest

sys.path.insert(0,str(ROOT/'source/tests'))
names=['anchor_budget_test','budget_viewer_test','anchor_budget_cli_test']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
save(ROOT/'reports/UNIT_VALIDATION.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
    failures=len(result.failures),errors=len(result.errors)))
assert result.wasSuccessful()
