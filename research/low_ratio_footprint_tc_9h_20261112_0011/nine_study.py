"""复用冻结的32组输入与F2；不回写上轮研究模块。"""
from pathlib import Path
import sys
import cProfile  # 在加入旧研究目录前固定标准库profile，避免旧profile.py遮蔽。
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'large_footprint_subtiles_20261111_2210'
sys.path.insert(0,str(OLD))
from footprint_study import *
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from render_runtime import Renderer,MODES,extension


def verify_protected():
    for name in ('PROTECTED_SHA256.json','INPUT_SHA256.json'):
        manifest=json.loads((HERE/'reports'/name).read_text())
        for path,value in manifest.items():assert sha(path)==value,path


def nested_packet(packet,ratio,seed=3407):
    """固定排列取前缀，再按原ID顺序发布；相机及各ID属性始终固定。"""
    n=len(packet['inputs']['means3D']);k=min(n,max(0,int(n*ratio)))
    generator=torch.Generator(device='cpu').manual_seed(seed)
    ids=torch.randperm(n,generator=generator)[:k].sort().values.cuda()
    return dict(sh_degree=packet['sh_degree'],inputs={name:(value.index_select(0,ids) if isinstance(value,torch.Tensor) and value.ndim>0 and len(value)==n else value)
        for name,value in packet['inputs'].items()}),ids
