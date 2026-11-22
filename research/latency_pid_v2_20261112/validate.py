from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B','test_gpu.py'],cwd=root,check=True)
subprocess.run([sys.executable,'-B','benchmark.py','--smoke'],cwd=root,check=True)
