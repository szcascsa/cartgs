"""独立评估进程使用Agg；不修改用户的~/.evo/settings.json。"""
import runpy
import sys
from pathlib import Path

import matplotlib
from evo.tools import settings

matplotlib.use("Agg")
settings.SETTINGS.plot_backend = "Agg"
source = Path(__file__).resolve().parents[3] / "python" / "run.py"
sys.path.insert(0, str(source.parent))
sys.argv[0] = str(source)
runpy.run_path(str(source), run_name="__main__")
