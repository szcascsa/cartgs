from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B','gpu_validate.py'],cwd=root,check=True)
subprocess.run([sys.executable,'-B','benchmark.py'],cwd=root,check=True)
