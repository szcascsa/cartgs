"""第二阶段沿用相同统计标准，增加原冻结F1的独立配对对照。"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
stage=ROOT/'background_stage'
source=(ROOT/'analyze.py').read_text()
source=source.replace("V=('G','C','F')","V=('G','C','F1','F')")
source=source.replace("R/'causal01/","R/'causal_balanced01/")
source=source.replace('len(d)==253440 and len(checks)==21120','len(d)==337920 and len(checks)==28160')
source=source.replace("p['C_over_F']=p.p50_ms_C/p.p50_ms_F","p['C_over_F']=p.p50_ms_C/p.p50_ms_F;p['F1_over_F']=p.p50_ms_F1/p.p50_ms_F")
source=source.replace("C_over_F=geometric(data.C_over_F),","C_over_F=geometric(data.C_over_F),F1_over_F=geometric(data.F1_over_F),")
source=source.replace("'F':'#007F73'","'F1':'#9478A8','F':'#007F73'")
source=source.replace("'F':'New F'","'F1':'First candidate F1','F':'Final F2'")
path=stage/'analyze.py'
assert not path.exists()
path.write_text(source)
subprocess.run([sys.executable,'-B',str(path)],cwd=stage,check=True)
