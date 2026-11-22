from common import *
import unittest
sys.path.insert(0,str(ROOT/'source/tests'))
suite=unittest.defaultTestLoader.loadTestsFromNames(['anchor_budget_test','budget_viewer_test','anchor_budget_cli_test'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
save(ROOT/'reports/UNIT_VALIDATION.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
if not result.wasSuccessful():raise SystemExit(1)
