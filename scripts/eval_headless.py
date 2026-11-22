"""Run the existing evaluator headlessly without changing the user's evo settings."""
import runpy
import sys
from pathlib import Path
from evo.tools import settings

settings.SETTINGS.plot_backend = "Agg"
script = Path(__file__).resolve().parents[1] / "python" / "run.py"
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
