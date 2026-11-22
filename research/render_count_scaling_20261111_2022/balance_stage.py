"""保留已完成现场，仅修正四版本计时次序；CUDA候选保持冻结。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
stage=ROOT/'background_stage'
controller=stage/'reports/FORMAL_CONTROLLER.json'
assert json.loads(controller.read_text())['state']=='complete'
shutil.copy2(controller,stage/'reports/FORMAL_CONTROLLER_UNBALANCED.json')
frozen=stage/'reports/FROZEN.json'
shutil.copy2(frozen,stage/'reports/FROZEN_PRE_BALANCE.json')
p=stage/'formal.py'
text=p.read_text().replace('from render_study import *','from render_study import *\nfrom balanced_order import order')
old='p.rotate(list(runs),rep+j+ci+args.round)'
assert old in text
text=text.replace(old,'order(list(runs),rep+j+ci+args.round,reverse=args.round==2)')
p.write_text(text)
p=stage/'causal_count.py'
text=p.read_text().replace('from render_study import *','from render_study import *\nfrom balanced_order import order')
text=text.replace('p.rotate(list(runs),rep+j)','order(list(runs),rep+j,reverse=(rep//4)%2==1)')
p.write_text(text)
p=stage/'run_formal.py'
text=p.read_text().replace("name=f'formal_r{iteration}_g{group}_a{attempt}'",
                           "name=f'balanced_r{iteration}_g{group}_a{attempt}'")
p.write_text(text)
f=json.loads(frozen.read_text())
for rel in ('formal.py','../balanced_order.py'):
    f['sources'][rel]=hashlib.sha256((stage/rel).read_bytes()).hexdigest()
f['measurement_order']='四版本每个计时单元60次，循环轮换，第二轮整体反序；无奇偶耦合'
frozen.write_text(json.dumps(f,ensure_ascii=False,indent=2))
print('BALANCED_MEASUREMENT_READY',flush=True)
