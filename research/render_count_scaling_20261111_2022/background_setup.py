"""生成第二阶段配对驱动，第一阶段源码和结果保持只读。"""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'background_stage'
assert STAGE.is_dir() and not (STAGE/'formal.py').exists()
prefix='from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent.parent))\n'
formal=(ROOT/'formal.py').read_text()
formal=prefix+formal.replace('from render_study import *',
    'from render_study import *\nimport formal as first_formal\nfrom background_runtime import Renderer,extension\nfrom scaling_runtime import Renderer as FirstRenderer\nHERE=HERE/"background_stage"')
formal=formal.replace("'C':old.Renderer(21),'F':Renderer(f['mode'])",
    "'C':old.Renderer(21),'F1':FirstRenderer(18),'F':Renderer(f['mode'])")
formal=formal.replace("out.mkdir(exist_ok=False)","out.mkdir(exist_ok=False)\n    first_formal.frozen()")
formal=formal.replace('HERE=HERE/"background_stage"',
    'HERE=HERE/"background_stage"\nTUNE=TUNE+["VECtor__corridors-dolly__stereo"]')
(STAGE/'formal.py').write_text(formal)
(STAGE/'run_formal.py').write_text((ROOT/'run_formal.py').read_text())
causal=(ROOT/'causal_count.py').read_text()
causal=prefix+causal.replace('from render_study import *',
    'from render_study import *\nfrom background_runtime import Renderer\nfrom scaling_runtime import Renderer as FirstRenderer\nHERE=HERE/"background_stage"')
causal=causal.replace("'C':old.Renderer(21),'F':Renderer(f['mode'])",
    "'C':old.Renderer(21),'F1':FirstRenderer(18),'F':Renderer(f['mode'])")
# prefix将root放在sys.path[0]，显式加载本阶段frozen，避免Python模块同名碰撞。
causal=causal.replace('from formal import frozen',
    'import importlib.util\n_spec=importlib.util.spec_from_file_location("background_formal",HERE/"formal.py")\n_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)\nfrozen=_mod.frozen')
(STAGE/'causal_count.py').write_text(causal)
print('BACKGROUND_DRIVERS_READY',flush=True)
